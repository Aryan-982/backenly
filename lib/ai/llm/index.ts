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
 *   - every failure is an AiError with a kind, user-facing copy and whether
 *     inference may have run (lib/ai/llm/errors.ts);
 *   - a failure is retried only when the provider rejected it before inference,
 *     so a retry can never run, and bill, a request twice;
 *   - usage comes back in one shape with cache reads and writes kept apart, and
 *     priced at the model's own cache rates (lib/ai/cost-tracker.ts).
 *
 * It does not charge anyone. Metering and spend limits sit above it.
 */

import type Anthropic from '@anthropic-ai/sdk'
import { calculateCost } from '../cost-tracker'
import { resolveProvider, routeFor, type ProviderConfig } from './config'
import { AiError, classifyProviderError } from './errors'
import { buildParams, shouldStream } from './params'
import { anthropicProvider, bedrockProvider } from './providers/anthropic'
import { openaiProvider } from './providers/openai'
import type { GenerateRequest, GenerateResult, LlmProvider, NormalizedUsage, ProviderId, PurposeRoute } from './types'

export type { AiPurpose, GenerateRequest, GenerateResult, NormalizedUsage, ProviderId } from './types'
export { AiError } from './errors'
export type { AiErrorKind, Consumption } from './errors'
export { aiAvailability } from './config'

const MAX_ATTEMPTS = 3

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

export function normalizeUsage(u: Anthropic.Usage | null | undefined): NormalizedUsage {
  const read = u?.cache_read_input_tokens ?? 0
  const write = u?.cache_creation_input_tokens ?? 0
  const write1h = u?.cache_creation?.ephemeral_1h_input_tokens ?? 0
  const write5m = u?.cache_creation ? (u.cache_creation.ephemeral_5m_input_tokens ?? 0) : Math.max(0, write - write1h)
  const input = u?.input_tokens ?? 0
  return {
    promptTokens: input + read + write,
    inputTokens: input,
    cacheReadTokens: read,
    cacheWrite5mTokens: write5m,
    cacheWrite1hTokens: write1h,
    outputTokens: u?.output_tokens ?? 0,
  }
}

export function costOf(model: string, usage: NormalizedUsage): number {
  return calculateCost(model, usage.promptTokens, usage.outputTokens, usage.cacheReadTokens, {
    tokens5m: usage.cacheWrite5mTokens,
    tokens1h: usage.cacheWrite1hTokens,
  })
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
      throw e
    }
    return finish(message, provider.id, route, req, attempt)
  }
}
