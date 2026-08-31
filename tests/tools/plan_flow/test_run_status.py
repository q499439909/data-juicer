import json
from pathlib import Path

from data_juicer.tools.plan_flow.run_status import read_run_steps


def _write_events(run_path: Path, events: list[dict]) -> None:
    target = run_path / "work" / "job-1"
    target.mkdir(parents=True)
    (target / "events_test.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )


def test_operation_events_map_by_index_even_when_operator_names_repeat(tmp_path):
    process = [
        {"text_length_filter": {"min_len": 2}},
        {"text_length_filter": {"min_len": 5}},
    ]
    _write_events(
        tmp_path,
        [
            {
                "event_type": "op_start",
                "operation_idx": 1,
                "operation_name": "text_length_filter",
                "timestamp": 10.0,
            },
            {
                "event_type": "op_complete",
                "operation_idx": 1,
                "operation_name": "text_length_filter",
                "timestamp": 11.5,
                "metadata": {"duration_seconds": 1.5, "input_rows": 10, "output_rows": 8},
            },
        ],
    )

    result = read_run_steps(tmp_path, process, "running")

    assert result["steps"][0]["status"] == "pending"
    assert result["steps"][1]["status"] == "succeeded"
    assert result["steps"][1]["duration_ms"] == 1500
    assert result["steps"][1]["output_rows"] == 8


def test_missing_or_mismatched_index_is_diagnostic_and_never_name_guessed(tmp_path):
    process = [{"clean_html_mapper": {}}]
    _write_events(
        tmp_path,
        [
            {"event_type": "op_complete", "operation_name": "clean_html_mapper"},
            {"event_type": "op_complete", "operation_idx": 0, "operation_name": "another_mapper"},
        ],
    )

    result = read_run_steps(tmp_path, process, "running")

    assert result["steps"] == [{"process_index": 0, "operator_name": "clean_html_mapper", "status": "pending"}]
    assert len(result["diagnostics"]) == 2


def test_cancelled_run_gives_every_unfinished_step_an_explicit_terminal_state(tmp_path):
    result = read_run_steps(tmp_path, [{"clean_html_mapper": {}}, {"text_length_filter": {}}], "cancelled")

    assert [step["status"] for step in result["steps"]] == ["cancelled", "cancelled"]
    assert result["mapping_complete"] is True
