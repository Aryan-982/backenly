/**
 * THE AI GATEWAY — the only code that talks to a model provider.
 * =============================================================
 *
 *   const result = await generate({ purpose: 'brain.classify', system, messages })
 *
 * A call names its purpose; lib/ai/llm/config.ts decides the provider, model,
 * thinking, effort, output ceiling and timeout. On Backenly Cloud that is
 * Claude Opus 4.6 on Amazon Bedrock for every purpose. Self-hosted installs can
 * use the Claude API or OpenAI instead.
 *
 * What it guarantees:
 *   - every request carries an explicit output ceiling and only parameters the
 *     routed model accepts (lib/ai/llm/params.ts);
 *   - its worst-case cost is reserved against the deployment's monthly limits
 *     before it is sent, and settled from the provider's own usage after
 *     (lib/ai/llm/spend.ts); a reservation that cannot be written stops the call;
 *   - every failure is an AiError with a kind, user-facing copy and whether
 *     inference may have run (lib/ai/llm/errors.ts);
 *   - a failure is retried only when the provider rejected it before inference,
 *     so a retry can never run, and bill, a request twice;
 *   - usage comes back in one shape with cache reads and writes kept apart, and
 *     priced at the model's own cache rates (lib/ai/cost-tracker.ts).
 */

import type Anthropic from '@anthropic-ai/sdk'
import { resolveProvider, routeFor, type ProviderConfig } from './config'
import { AiError, classifyProviderError } from './errors'
import { buildParams, shouldStream } from './params'
import { anthropicProvider, bedrockProvider } from './providers/anthropic'
import { openaiProvider } from './providers/openai'
import { defaultPayer, reserveSpend, settleSpend } from './spend'
import type { GenerateRequest, GenerateResult, LlmProvider, ProviderId, PurposeRoute } from './types'
import { costOf, maxCostOf, normalizeUsage } from './usage'

export type { AiPurpose, GenerateRequest, GenerateResult, NormalizedUsage, ProviderId } from './types'
export { AiError } from './errors'
export type { AiErrorKind, Consumption } from './errors'
export { aiAvailability } from './config'
export { normalizeUsage, costOf } from './usage'

const MAX_ATTEMPTS = 3
/** Margin past the last attempt's timeout before an unsettled reservation is booked as unknown. */
const SETTLE_GRACE_MS = 5 * 60_000

let override: LlmProvider | null = null
let cached: { key: string; provider: LlmProvider } | null = null
let retryDelayMs = (attempt: number) => 500 * 2 ** (attempt - 1)

/** Route every call to `provider` (a scripted one in tests). `null` restores configuration. */
export function setProviderForTests(provider: LlmProvider | null): void {
  override = provider
}

export function setRetryDelayForTests(fn: ((attempt: number) => number) | null): void {
  retryDelayMs = fn ?? ((attempt: number) => 500 * 2 ** (attempt - 1))
}

function configKey(c: ProviderConfig): string {
  return c.id === 'bedrock' ? `bedrock:${c.region}` : `${c.id}:${c.apiKey.length}:${c.apiKey.slice(-4)}`
}

function currentProvider(): LlmProvider {
  if (override) return override
  const resolved = resolveProvider()
  if ('reason' in resolved) throw new AiError('not_configured', resolved.reason, { consumption: 'none' })
  const key = configKey(resolved.config)
  if (cached?.key === key) return cached.provider
  const c = resolved.config
  const provider =
    c.id === 'bedrock' ? bedrockProvider(c.region) : c.id === 'anthropic' ? anthropicProvider(c.apiKey) : openaiProvider(c.apiKey)
  cached = { key, provider }
  return provider
}

function finish(
  message: Anthropic.Message,
  provider: ProviderId,
  route: PurposeRoute,
  req: GenerateRequest,
  attempts: number,
): GenerateResult {
  const usage = normalizeUsage(message.usage)
  const ran = { provider, consumption: 'ran' as const, usage }

  if (message.stop_reason === 'refusal') {
    const e = new AiError('refused', 'The model declined the request.', ran)
    e.attempts = attempts
    throw e
  }

  const text = message.content
    .filter((b): b is Anthropic.TextBlock => b.type === 'text')
    .map((b) => b.text)
    .join('')

  let json: unknown
  if (req.jsonSchema) {
    const cutOff = message.stop_reason === 'max_tokens'
    try {
      if (cutOff) throw new Error('cut off')
      json = JSON.parse(text)
    } catch {
      const e = new AiError(
        'output_invalid',
        cutOff ? 'The JSON answer was cut off at the output ceiling.' : 'The answer was not valid JSON.',
        ran,
      )
      e.attempts = attempts
      throw e
    }
  }

  return {
    provider,
    model: route.model,
    message,
    text,
    ...(req.jsonSchema ? { json } : {}),
    usage,
    costUsd: costOf(route.model, usage),
    attempts,
  }
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

export async function generate(req: GenerateRequest): Promise<GenerateResult> {
  const provider = currentProvider()
  // A scripted provider stands in for Claude, so it is routed as Claude.
  const route = routeFor(req.purpose, provider.id === 'scripted' ? 'anthropic' : provider.id)
  const params = buildParams(route, req)
  const stream = shouldStream(route, params.max_tokens)

  const reservation = await reserveSpend({
    purpose: req.purpose,
    model: route.model,
    payer: req.payer ?? defaultPayer(req.purpose),
    amountUsd: maxCostOf(route.model, params),
    expiresInMs: route.timeoutMs * MAX_ATTEMPTS + SETTLE_GRACE_MS,
    billingAccountId: req.billing?.billingAccountId,
    projectId: req.billing?.projectId,
  })

  for (let attempt = 1; ; attempt++) {
    let message: Anthropic.Message
    try {
      message = await provider.send(params, { timeoutMs: route.timeoutMs, signal: req.signal, stream })
    } catch (err) {
      const e = classifyProviderError(err, provider.id)
      e.attempts = attempt
      if (e.retryable && e.consumption === 'none' && attempt < MAX_ATTEMPTS && !req.signal?.aborted) {
        await sleep(retryDelayMs(attempt))
        continue
      }
      await settleFailure(reservation, e, route.model)
      throw e
    }

    try {
      const result = finish(message, provider.id, route, req, attempt)
      await settleSpend(reservation, { consumption: 'ran', costUsd: result.costUsd, usage: result.usage })
      return result
    } catch (err) {
      const e = err instanceof AiError ? err : classifyProviderError(err, provider.id)
      await settleFailure(reservation, e, route.model)
      throw e
    }
  }
}

async function settleFailure(reservation: Awaited<ReturnType<typeof reserveSpend>>, e: AiError, model: string): Promise<void> {
  if (e.consumption === 'ran' && e.usage) {
    await settleSpend(reservation, { consumption: 'ran', costUsd: costOf(model, e.usage), usage: e.usage })
  } else if (e.consumption === 'none') {
    await settleSpend(reservation, { consumption: 'none' })
  } else {
    // Inference may have run and its usage is not known: book the reservation.
    await settleSpend(reservation, { consumption: 'unknown' })
  }
}
