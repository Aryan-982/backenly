/**
 * AI CREDIT HOLDS
 * ===============
 * Before a customer-paid model call is sent, its worst case is held against the
 * account's AI credits in one conditional update on the account's monthly usage
 * row. Parallel turns used to each pass the read-only gate (enforceAiCredits)
 * and overshoot together; now a call whose worst case does not fit is refused
 * before it is sent.
 *
 * Holding is policy, not commercial recording, so it is public (see
 * lib/entitlements/provider.ts for that split). Charging is unchanged:
 * chargeAiCredits after the turn, from the tokens the provider reported, and
 * only for a turn that changed something. A hold is released when its call
 * settles, whatever happened, so a hold never charges and nothing is billed
 * twice. A call whose answer was lost reports no tokens, so the customer pays
 * nothing for it; the gateway's own ledger books it against the platform.
 *
 * Known window: between a call settling and its turn being charged, its tokens
 * are neither held nor charged, so a parallel turn can start in that gap. That
 * bounds an overshoot by one turn's usage per concurrent turn, where it was
 * unbounded before.
 *
 * A process that dies mid-call never releases its hold. Holds are treated as
 * released once nothing has been held on the row for HOLD_TTL_MINUTES, so a
 * crash costs an account at most that long of reduced headroom.
 *
 * Fail-open where it always was: no account, no entitlements, unlimited credits
 * or a failed entitlement read mean no hold. A hold that cannot be WRITTEN
 * refuses the call, because sending it then would spend credits nobody checked.
 */

import { prisma } from '@/lib/db/prisma'
import { AiError } from '@/lib/ai/llm/errors'
import type { CustomerSpendHook } from '@/lib/ai/llm/spend'
import { creditAccountOf } from '@/lib/usage/account'
import { aiCreditCap, creditsFromTokens, thisMonth, TOKENS_PER_CREDIT } from './policy'
import { invalidateUsageCache } from './usage-cache'

export const HOLD_TTL_MINUTES = 15

/** A call nobody needs to hold credits for. */
const NO_HOLD = { id: 'none' }

const HOLD_ID = /^hold:(.+):(\d{4}-\d{2}):(\d+)$/

export const creditHoldHook: CustomerSpendHook = {
  async reserve({ maxTokens, billingAccountId, projectId }) {
    const account = billingAccountId ?? (projectId ? await creditAccountOf(projectId) : null)
    if (!account) return NO_HOLD
    const cap = await aiCreditCap(account).catch(() => null)
    if (cap === null) return NO_HOLD

    const month = thisMonth()
    const tokens = Math.max(0, Math.ceil(maxTokens))
    let held: number
    try {
      await prisma.accountAiUsage.upsert({
        where: { billingAccountId_date: { billingAccountId: account, date: month } },
        update: {},
        create: { billingAccountId: account, date: month },
      })
      // One statement, so two calls racing for the last credits cannot both
      // fit. A stale hold counts as zero, in the check and in what is kept.
      held = await prisma.$executeRaw`
        UPDATE "account_ai_usage"
           SET "heldTokens" = (CASE WHEN "heldAt" IS NULL OR "heldAt" < now() - make_interval(mins => ${HOLD_TTL_MINUTES}::int)
                                    THEN 0 ELSE "heldTokens" END) + ${tokens},
               "heldAt" = now()
         WHERE "billingAccountId" = ${account}
           AND "date" = ${month}
           AND ceil(("tokenCount"
                     + (CASE WHEN "heldAt" IS NULL OR "heldAt" < now() - make_interval(mins => ${HOLD_TTL_MINUTES}::int)
                             THEN 0 ELSE "heldTokens" END)
                     + ${tokens})::numeric / ${TOKENS_PER_CREDIT}) <= ${cap}`
    } catch (err) {
      throw new AiError(
        'spend_unavailable',
        `Your AI credits could not be checked for this call, so it was not sent (${err instanceof Error ? err.message : String(err)}).`,
        { consumption: 'none' },
      )
    }
    if (held === 0) {
      throw new AiError(
        'credits_exhausted',
        `This call could use up to ${creditsFromTokens(tokens).toLocaleString()} AI credits, more than this account has ` +
          'left this month, so it was not sent. Credits reset on the 1st, or upgrade for more.',
        { consumption: 'none' },
      )
    }
    invalidateUsageCache(account)
    return { id: `hold:${account}:${month}:${tokens}` }
  },

  async settle(id) {
    const m = HOLD_ID.exec(id)
    if (!m) return
    const [, account, month, tokens] = m
    await prisma.$executeRaw`
      UPDATE "account_ai_usage"
         SET "heldTokens" = GREATEST("heldTokens" - ${Number(tokens)}, 0)
       WHERE "billingAccountId" = ${account} AND "date" = ${month}`
    invalidateUsageCache(account)
  },
}
