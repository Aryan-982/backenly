# How AI calls are reserved and settled

Every model call goes through the AI gateway (`lib/ai/llm`). Its cost is reserved before the request is sent and settled after the answer comes back (`lib/ai/llm/spend.ts`).

## Limits

A deployment can set two monthly limits, in US dollars, for the calendar month in UTC.

| Variable | Counts |
|---|---|
| `AI_TOTAL_MONTHLY_HARD_LIMIT_USD` | Every call, whoever pays for it. On Backenly Cloud it protects the AWS credits that pay for Bedrock. |
| `AI_PLATFORM_MONTHLY_BUDGET_USD` | Calls the deployment pays for itself, as opposed to calls billed to a customer. Today those are the `autonomy.*` purposes. |

- **Unset** means no limit. Every call is still recorded.
- **Not a number** (a typo) is treated as zero, so it blocks every call instead of quietly allowing unlimited spend.

When a limit is reached, the call is not sent. It fails with `spend_limit`, and the caller shows the error's user message. Deterministic work, such as the autonomy loop's checks and repairs, does not depend on these calls and carries on.

## Reserving

Before a request is sent, its worst case is reserved against each limit that applies:

- every prompt token is estimated generously and priced as a cache write;
- the full output ceiling is priced as output.

The check and the increment are one conditional `UPDATE` on `ai_spend_periods`. Two concurrent calls cannot both pass the check on the same remaining room: the row lock makes the second one wait and check again. `tests/integration/ai-spend-ledger.spec.ts` runs 20 concurrent reservations against a limit with room for 10, and exactly 10 are granted.

**If the reservation cannot be written, the call is not sent.** It fails with `spend_unavailable`. A call nothing has accounted for could pass a limit, so unrecorded spend is never allowed.

Each reservation is also a row in `ai_spend_holds`, with an expiry past the call's last possible attempt.

## Settling: three outcomes

Every call ends in exactly one of these. `AiError.consumption` says which.

| Outcome | When | The deployment's ledger | A customer |
|---|---|---|---|
| **Ran** | The provider returned an answer with its usage. This includes a refusal or an answer cut off at the output ceiling. | Charged the real cost, from the provider's own token counts. | Charged the same real cost. |
| **None** | The request was never sent, or the provider refused it before running inference: invalid request, bad credentials, no model access, throttling, overload. | Charged nothing; the reservation is released. | Charged nothing. |
| **Unknown** | The request was sent and the answer was lost: a timeout, a dropped connection, a server error, or the process dying mid-call. | The full reservation is booked as unknown usage, which counts against both limits. | Charged nothing. |

A timeout is not proof that nothing ran. The provider may have done the work and billed for it. So:

- the deployment books an unknown outcome at the conservative upper bound;
- a customer is never charged for an answer they did not receive.

Every unknown booking is logged.

**Retries.** Only failures that were refused before inference (throttling, overload) are retried. A failure whose outcome is unknown is never retried, because a retry could run, and be billed, a second time.

**Abandoned reservations.** If the process dies after reserving, the reservation stays open until its expiry. It is then booked as unknown. The expiry check runs at most once a minute per process, before new reservations.

## Customer billing

Customer-paid calls go through a `CustomerSpendHook`. The default is the credit hold in `lib/entitlements/credit-holds.ts`:

- `reserve` holds the call's worst case, in effective tokens, on the account's monthly `account_ai_usage` row, in one conditional update. A call whose worst case does not fit in the credits left is refused with `credits_exhausted` before it is sent, and two calls racing for the last credits cannot both go out. A hold that cannot be written refuses the call with `spend_unavailable`.
- `settle` releases the hold, whatever happened. It never charges.

Charging stays where it was: `chargeAiCredits` after the turn, from the tokens the provider reported, and only for a turn that changed something. So nothing is billed twice, questions stay free, and a call whose answer was lost costs the customer nothing.

Two limits of the design, stated rather than hidden:

- Between a call settling and its turn being charged, its tokens are neither held nor charged, so a parallel turn can start in that gap. An overshoot is bounded by one turn's usage per concurrent turn.
- A process that dies mid-call leaves its hold. Holds count as released once nothing has been held on the row for 15 minutes.

Where there are no finite credits (a self-hosted install, an unlimited plan, or entitlements that cannot be read), nothing is held. The deployment's own limits apply either way. `setCustomerSpendHook(null)` turns holds off.

## Where to look

| Question | Answer |
|---|---|
| This month's spend | `ai_spend_periods`: one row per month and scope (`total`, `platform`), with reserved, settled and unknown dollars. |
| One call | `ai_spend_holds`: its reservation, status, charge and expiry. |
| Token detail | `ai_usage`, written by the cost tracker. |

On Bedrock, the AWS invoice is the record of what AWS actually charged. An AWS Budgets alert on that line is the outer limit.
