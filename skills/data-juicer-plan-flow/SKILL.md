---
name: data-juicer-plan-flow
description: Plan, approve, execute, and recover reproducible Data-Juicer cleaning tasks through the plan-flow MCP.
---

# Data-Juicer plan-first workflow

Use one agent. The MCP is a capability and persistence layer, not a nested agent.

1. Decide whether the request states the input, desired outputs, constraints, and acceptance criteria. Ask only for material missing choices. Do not prepare a plan until those choices are answered.
2. Call `inspect_input` for local input. Call `search_capabilities` with independent requirements; these searches may run in parallel.
3. Assign each step to the narrowest suitable implementation:
   - Put supported cleaning/transformation steps in `plan.recipe.process`.
   - Put a genuine uncovered gap in top-level `plan.postprocess` as a generic Python artifact.
   - Never search indefinitely for a perfect operator. After a focused search returns no suitable candidate, implement the gap.
4. Call `prepare_plan`. Use the returned normalized plan, validation, diff, and content hash to explain the important parameters and risks. Do not approve on the user's behalf.
5. For revisions, call `prepare_plan` again with the same `task_id` and the previous `plan_version` as `base_plan_version`. Every revision creates a new `plan_vNNN`; never edit or overwrite an existing version.
6. After explicit user approval, call `approve_plan` with the exact `content_hash`, then call `run_plan`.
7. Poll `get_run`. Read logs when a run fails, fix only the understood cause, prepare and obtain approval for a new plan version when plan or artifact content must change, then run again. Never mutate an approved bundle.
8. On success, return the output directory, materialized recipe, report path, plan version, and run id.

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
