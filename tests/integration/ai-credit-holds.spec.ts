/**
 * AI credit holds on a real Postgres: a customer-paid call's worst case is held
 * against the account's credits before it is sent, in one conditional update,
 * so two calls racing for the last credits cannot both go out. Holds never
 * charge, are released on settle, and go stale after a crash.
 *
 * The credit cap comes from the entitlements provider, which the OSS edition
 * these tests run as answers "unlimited", so only that read is stubbed. The
 * usage row, the update and the race are real.
 */

import crypto from 'crypto'
import { prisma } from '@/lib/db/prisma'

let cap: number | null = 10
jest.mock('@/lib/entitlements/policy', () => {
  const actual = jest.requireActual('@/lib/entitlements/policy')
  return { ...actual, aiCreditCap: jest.fn(async () => cap) }
})

import { generate, setProviderForTests } from '@/lib/ai/llm'
import { scriptedMessage, scriptedProvider } from '@/lib/ai/llm/providers/scripted'
import { memorySpendLedger, setCustomerSpendHook, setSpendLedgerForTests } from '@/lib/ai/llm/spend'
import { creditHoldHook } from '@/lib/entitlements/credit-holds'
import { thisMonth } from '@/lib/entitlements/policy'

const account = `holds-${crypto.randomBytes(6).toString('hex')}`
const month = thisMonth()
const reserve = (maxTokens: number) =>
  creditHoldHook.reserve({ purpose: 'brain.agent', model: 'test', maxUsd: 0, maxTokens, billingAccountId: account })
const row = () => prisma.accountAiUsage.findUniqueOrThrow({ where: { billingAccountId_date: { billingAccountId: account, date: month } } })

beforeEach(async () => {
  cap = 10 // 10 credits = 10,000 tokens
  await prisma.accountAiUsage.deleteMany({ where: { billingAccountId: account } })
})

afterEach(() => {
  setProviderForTests(null)
  setSpendLedgerForTests(null)
  setCustomerSpendHook(undefined)
})

afterAll(async () => {
  await prisma.accountAiUsage.deleteMany({ where: { billingAccountId: account } })
})

describe('AI credit holds', () => {
  it('holds a call that fits, refuses one that does not, and frees the hold on settle', async () => {
    const first = await reserve(6_000)
    expect(first.id).toMatch(/^hold:/)
    expect((await row()).heldTokens).toBe(6_000)

    await expect(reserve(6_000)).rejects.toMatchObject({ kind: 'credits_exhausted', consumption: 'none' })
    expect((await row()).heldTokens).toBe(6_000)

    await creditHoldHook.settle(first.id, { chargeUsd: 0.01, outcome: 'ran' })
    const after = await row()
    expect(after.heldTokens).toBe(0)
    // A hold never charges: the turn's charge is chargeAiCredits, after the turn.
    expect(after.tokenCount).toBe(0)
    await expect(reserve(6_000)).resolves.toMatchObject({ id: expect.stringMatching(/^hold:/) })
  })

  it('lets exactly one of two racing calls take the last credits', async () => {
    const results = await Promise.allSettled([reserve(6_000), reserve(6_000)])
    expect(results.filter((r) => r.status === 'fulfilled')).toHaveLength(1)
    expect((await row()).heldTokens).toBe(6_000)
  })

  it('counts what was already used, and refuses a worst case past the cap', async () => {
    await prisma.accountAiUsage.create({ data: { billingAccountId: account, date: month, tokenCount: 9_500 } })
    await expect(reserve(1_000)).rejects.toMatchObject({ kind: 'credits_exhausted' })
    await expect(reserve(500)).resolves.toMatchObject({ id: expect.stringMatching(/^hold:/) })
  })

  it('treats a hold a crash left behind as released once it is stale', async () => {
    await prisma.accountAiUsage.create({
      data: { billingAccountId: account, date: month, heldTokens: 9_000, heldAt: new Date(Date.now() - 20 * 60_000) },
    })
    await expect(reserve(6_000)).resolves.toMatchObject({ id: expect.stringMatching(/^hold:/) })
    expect((await row()).heldTokens).toBe(6_000)
  })

  it('holds nothing where there is nothing to enforce', async () => {
    cap = null
    await expect(reserve(1_000_000)).resolves.toEqual({ id: 'none' })
    expect(await prisma.accountAiUsage.count({ where: { billingAccountId: account } })).toBe(0)
  })

  it('is the gateway’s default for customer-paid calls, and is released after the call', async () => {
    setSpendLedgerForTests(memorySpendLedger())
    const provider = scriptedProvider([
      async () => {
        // In flight: held.
        expect((await row()).heldTokens).toBeGreaterThan(0)
        return scriptedMessage({ text: 'ok', usage: { input_tokens: 10, output_tokens: 5 } })
      },
    ])
    setProviderForTests(provider)
    await generate({ purpose: 'brain.answer', messages: [{ role: 'user', content: 'hi' }], billing: { billingAccountId: account } })
    expect(provider.calls).toHaveLength(1)
    expect((await row()).heldTokens).toBe(0)
  })

  it('refuses a customer-paid call before it is sent when its worst case does not fit', async () => {
    cap = 1 // 1,000 tokens, less than brain.answer's output ceiling
    setSpendLedgerForTests(memorySpendLedger())
    const provider = scriptedProvider([scriptedMessage({ text: 'never sent' })])
    setProviderForTests(provider)
    await expect(
      generate({ purpose: 'brain.answer', messages: [{ role: 'user', content: 'hi' }], billing: { billingAccountId: account } }),
    ).rejects.toMatchObject({ kind: 'credits_exhausted' })
    expect(provider.calls).toHaveLength(0)
  })
})
