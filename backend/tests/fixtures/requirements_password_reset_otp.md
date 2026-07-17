# Feature: Password Reset — Add SMS OTP Verification Step

## Background

Currently, users reset their password via an emailed reset link. Security wants
to add a second factor: after clicking the reset link, the user must also enter
a one-time passcode (OTP) sent via SMS before they can set a new password.

## User Flow

1. User clicks "Forgot password?" on the login screen and enters their
   registered email address.
2. System sends a password reset link to that email.
3. User clicks the link and is taken to a page that asks them to verify their
   identity via SMS OTP.
4. System sends a 6-digit OTP to the phone number on file for the account.
5. User enters the OTP.
6. On successful verification, the user is taken to the "Set New Password"
   screen.
7. User sets a new password and is redirected to the login screen with a
   success message.

## Business Rules

- The reset link expires 30 minutes after it is sent.
- The OTP is 6 digits, numeric only.
- A user gets 3 attempts to enter the correct OTP before the reset flow is
  cancelled and they have to start over from step 1.
- Password strength rules are unchanged from the existing "Set New Password"
  screen (min 8 characters, at least one number and one symbol).

## Out of Scope

- Changing the phone number on file (handled by a separate "Update Contact
  Info" feature).
- Passwordless login.

## Open Questions From Product

- None flagged yet — Product considers this ready for engineering.
