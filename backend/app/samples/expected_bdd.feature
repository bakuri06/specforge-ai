Feature: Account Balance Transfer with Daily Limit

  Scenario: Transfer within daily limit succeeds
    Given a user with $10,000 available balance and no prior transfers today
    When the user enters recipient account and amount $500 and submits
    Then the transfer is created with status "pending" and transitions to "processing"
    And when the Ledger Service confirms the transfer
    Then the transfer status becomes "completed" and the sender balance decreases by $500

  Scenario Outline: Transfer exceeding daily limit is rejected
    Given a user who has already transferred <already_transferred> today
    When the user submits a transfer of <amount>
    Then the system rejects the transfer with a "daily limit exceeded" error before any ledger call is made
    And the sender balance is unchanged

    Examples:
      | already_transferred | amount |
      | $4,800               | $300   |

  Scenario: Ledger Service timeout marks transfer as failed
    Given a user submits a valid transfer of $200 while the Ledger Service does not respond
    When the Ledger Service call exceeds the 10 second timeout
    Then the transfer status becomes "failed"
    And the held sender balance is released
    And the user sees a retryable error message
