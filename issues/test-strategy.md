# Discuss and revise the test strategy

Status: Discussion required. No test policy is fixed.

## Goal

Agree on a test strategy that detects framework failures and limits unnecessary tests. Then revise the suite.

## Requirements

- Review the current suite. Identify duplicate coverage, missing failure cases, and test maintenance costs.
- Discuss focused tests, integration tests, and training tests. Agree on which groups the framework needs.
- Discuss test data, runtime, hardware, and network requirements.
- Agree on rules for adding, changing, and removing tests before revising the suite.
- Apply the agreed strategy to the current tests.
- Put the agreed policy in `docs/about/contributing.md`. Reference it from `AGENTS.md` for coding agents.

## Completion criteria

- The agreed policy states what to test and which checks to run for each change.
- The revised suite passes under the agreed conditions. Record its runtime and coverage changes.

## Input needed

Discuss the review findings and agree on the policy before changing tests.
