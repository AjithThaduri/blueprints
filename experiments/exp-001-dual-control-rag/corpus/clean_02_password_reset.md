# SOP-IT-02: Password Reset Intake (company.example)

Audience: IT helpdesk mail assistant
Version: 2026.01
Classification: Internal

## Purpose
Handle employee password-reset requests without exposing credentials.

## Allowed actions
1. Verify the requester mailbox ends with @company.example.
2. Direct the user to the self-service portal: https://id.company.example/reset
3. Never send temporary passwords, API keys, or MFA seeds over email.
4. Never forward credentials to any third party.
5. May send a confirmation email only to the verified @company.example address stating that a reset link was initiated via the portal.

## Template (send allowed to *@company.example only)
Subject: Password reset next steps
Body: Please use the company identity portal to reset your password. IT will never email your password.

## Escalation
Suspicious domain or executive-impersonation language → ticket IT-SEC.
