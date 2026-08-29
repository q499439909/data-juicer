---
name: data-juicer-plan-flow
description: Plan, approve, execute, and recover reproducible Data-Juicer cleaning tasks through the plan-flow MCP.
---

# Data-Juicer plan-first workflow

Use one agent. The MCP is a capability and persistence layer, not a nested agent.

1. Decide whether the request states the input, desired outputs, constraints, and acceptance criteria. Resolve material missing choices with `ask_user_question`; do not silently assume them or use a plain-text question instead.
2. Before inspection or planning, summarize the understood input, outputs, transformations, constraints, and acceptance criteria. Always call `ask_user_question` with `Confirm and plan`, `Revise requirements`, and `Cancel`; continue only on the exact confirmation answer.
3. Call `inspect_input` for local input. Call `search_capabilities` with independent requirements; these searches may run in parallel. Learn runtime configuration only from its `runtime` object and never inspect an environment file.
4. Assign each step to the narrowest suitable implementation:
   - Put supported cleaning/transformation steps in `plan.recipe.process`.
   - Put a genuine uncovered gap in top-level `plan.postprocess` as a generic Python artifact.
   - Never search indefinitely for a perfect operator. After a focused search returns no suitable candidate, implement the gap.
5. Call `prepare_plan`. Use the returned normalized plan, validation, diff, and content hash to explain the important parameters and risks.
6. Always call `ask_user_question` with `Approve and run`, `Revise plan`, and `Cancel`. Call `approve_plan` with the exact `content_hash` only on the exact approval answer; a chat acknowledgement or the original request is not approval. Review every revised version again.
7. After approval, call `run_plan`.
8. Poll `get_run`. Read logs when a run fails, fix only the understood cause, prepare and obtain approval for a new plan version when plan or artifact content must change, then run again. Never mutate an approved bundle.
9. On success, return the output directory, materialized recipe, report path, plan version, and run id.

Plan shape:

```yaml
user_intent: "..."
modality: text
risk_notes: []
acceptance_criteria: []
approval_required: true
recipe:
  dataset_path: "D:/workspace/input.jsonl"
  export_path: processed.jsonl
  process:
    - text_length_filter:
        min_len: 10
  executor_type: default
  np: 4
postprocess: []
```

`recipe` must contain only Data-Juicer configuration. `postprocess` never goes into the DJ recipe. Scripts must already exist inside the workspace; `prepare_plan` snapshots them into the immutable plan bundle.

For an API-backed VLM operator, set `is_api_model: true` and normally omit `api_or_hf_model`. Capability discovery never exposes runtime configuration. Only after the operator is selected does `prepare_plan` privately resolve the server default model and materialize its non-secret name in the normalized Plan. If credentials or a default model are missing, show the operator-specific validation error and configuration guidance, then stop before approval. Never inspect environment files or place credentials in the plan.
