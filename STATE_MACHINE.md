# Job Agent – Formal State Machine Specification

This document defines the lifecycle states and allowed transitions
for job processing inside the agent system.

The `status` field represents lifecycle state.
The `decision` field represents classification output.

---

# 1. State List

## Processing States (Agent-Driven)

- discovered
- profile_extracted
- scored

## Human Gate States

- awaiting_human_decision
- awaiting_final_review

## Workflow States

- draft_generated

## Terminal States

- skipped
- rejected
- submitted

Terminal states have no outgoing transitions.

---

# 2. Transition Table

## Agent Transitions

### discovered → profile_extracted
- Trigger: agent
- Condition: profile missing
- Action: extract_job_profile
- Next: profile_extracted

### profile_extracted → scored
- Trigger: agent
- Condition: score missing
- Action: score_job
- Next: scored

### scored → skipped
- Trigger: agent
- Condition: decision == "skip"
- Action: none
- Next: skipped

### scored → awaiting_human_decision
- Trigger: agent
- Condition: decision == "candidate"
- Action: none
- Next: awaiting_human_decision

---

## Human Transitions

### awaiting_human_decision → rejected
- Trigger: human
- Action: none
- Next: rejected

### awaiting_human_decision → draft_generated
- Trigger: human (approve)
- Action:
  - generate resume adaptation
  - generate cover letter
- Next: draft_generated

### draft_generated → awaiting_final_review
- Trigger: agent (same UI request)
- Action: persist draft
- Next: awaiting_final_review

### awaiting_final_review → submitted
- Trigger: human
- Action: submit application
- Next: submitted

### awaiting_final_review → rejected
- Trigger: human
- Action: none
- Next: rejected

### skipped → awaiting_human_decision (manual override)
- Trigger: human
- Action:
  - set decision = "candidate"
- Next: awaiting_human_decision

---

# 3. Execution Strategy

The agent should:

- Advance jobs automatically through agent transitions
- Stop at human gate states
- Never override human decisions
- Never allow illegal transitions

The UI is responsible for human-triggered transitions.

---

# 4. Illegal Transitions (Examples)

- skipped → draft_generated
- rejected → any other state
- submitted → any other state
- discovered → submitted

---

# 5. Design Principles

- status = lifecycle state
- decision = classification result
- Terminal states are immutable
- State transitions must be explicit
- No implicit jumps allowed
