Feature: Money Transfer

Users should be able to transfer money between their accounts. There's a daily
limit of $5000 total transfers per user. Each transfer over $1000 has a 1%
fee. The system talks to an external ledger service to actually move the
money - if that call times out we need to handle it somehow. Transfers go
through some kind of pending/processing/done state. Need to also think about
what happens if a transfer fails partway through.
