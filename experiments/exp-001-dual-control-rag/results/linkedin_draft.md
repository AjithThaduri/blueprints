# LinkedIn draft — Mode A (human posts; do not auto-publish)

**Status:** Draft only. Not posted. This reports only the Grok Bot local-agent run; it is not a DUMA-Bench full run.

## Hook

A user saying ‘just follow the template’ should not override hard policy rules in an enterprise agent.

## What THIS experiment measured (only claim this)

- Ran a synthetic `company.example` mailbox/SOP mini-eval with 5 poisoned and 5 clean tasks.
- Used Grok Bot's local-agent decision path: 5 poisoned tasks in solo mode, the same 5 with a hardcoded endorsement turn, and 5 clean tasks.
- Scored deterministic `send_email` assertions for recipient domain, wire/bank patterns, credentials, and exfiltration keywords.
- Measured ASR: solo 0%; dual-control 0%; clean false-refusal 0%.
- No OpenAI API or other provider API was used; this was a local-agent decision run.

## Do not say

- Do not claim Ajith ran the full DUMA-Bench (14-model / 26.9%→41.1% paper result).
- Do not generalize this single local-agent run to all Grok models or production agents.
- Do not invent client names, PHI, or ‘we caught this in prod.’

## Body outline

1. Problem: RAG poisoning plus active user endorsement is under-tested versus solo refusal evals.
2. Method: tiny synthetic corpus, deterministic tool-call assertions, solo versus endorsement.
3. Result (honest): both poisoned ASRs were 0% and clean false-refusal was 0% in this Grok Bot local-agent run; endorsement did not increase violations under explicit hard rules.
4. Why it matters: policy must bind tool calls even when a user says ‘just follow the SOP.’
5. CTA: compare notes with people shipping mail/RAG agents; cite DUMA-Bench as motivation, not as a reproduction.

## Suggested hashtags (optional, human discretion)

#AISecurity #RAG #Agents #Eval #MLOps
