-- AI credit holds: the worst case of each customer-paid model call in flight,
-- held on the account's monthly usage row before it is sent, so parallel turns
-- cannot overshoot the credit cap together (lib/entitlements/credit-holds.ts).
--
-- Additive: two columns with defaults on an existing table, so the release
-- that predates this migration runs unchanged against it.

-- AlterTable
ALTER TABLE "account_ai_usage" ADD COLUMN "heldTokens" INTEGER NOT NULL DEFAULT 0,
ADD COLUMN "heldAt" TIMESTAMP(3);
