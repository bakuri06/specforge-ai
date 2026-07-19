# Feature: Account Balance Transfer with Daily Limit

## Input Validation

- Transfer amount must be a positive decimal with at most 2 decimal places;
  reject zero, negative, or non-numeric input with a clear validation error.
- Recipient account ID must reference an existing, active account; reject
  transfers to closed or non-existent accounts.
- Sender must have sufficient available balance (balance minus any
  held/pending amounts) to cover the transfer amount plus any applicable fee
  before submission is accepted.

## Core Calculation Framework

- Daily transfer limit is $5,000 total per sender account, reset at midnight
  UTC; the limit is calculated as the sum of all transfers with status
  `completed` or `processing` initiated by that account since the last reset.
- A 1% fee applies to any single transfer with an amount over $1,000; the fee
  is calculated as `round(amount * 0.01, 2)` and is deducted from the
  sender's balance in addition to the transfer amount.
- Attempting a transfer that would exceed the remaining daily limit (amount +
  fee > remaining limit) must be rejected before any ledger call is made.

## Network Architecture

- Balance movement is performed via a call to the external Ledger Service's
  `/transfers` endpoint.
- If the Ledger Service call times out (no response within 10 seconds), the
  transfer is marked `failed` and the sender's held balance is released; the
  user is shown a retryable error rather than an ambiguous success/failure
  state.
- If the Ledger Service returns a 5xx error, the system retries the call once
  after a 2-second backoff before marking the transfer `failed`.

## State Lifecycles

- `pending` -> `processing` -> `completed`: normal successful path once the
  Ledger Service confirms the transfer.
- `pending` -> `processing` -> `failed`: Ledger Service call times out, errors
  after retry, or explicitly rejects the transfer.
- `failed` transfers release any held sender balance immediately and do not
  count against the daily limit.
- A `completed` transfer may transition to `reversed` only via a separate,
  explicitly-authorized reversal process (out of scope for this feature).

## Out of Scope

- The reversal/refund process for completed transfers.
- Multi-currency transfers (this feature assumes both accounts are in the
  same currency).
