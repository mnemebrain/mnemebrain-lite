"""Load and validate scenarios from JSON."""
from __future__ import annotations

import json
from pathlib import Path

from mnemebrain_core.benchmark.scenarios.schema import (
    Action,
    Expectation,
    Scenario,
    VALID_ACTION_TYPES,
)


def validate_scenario(scenario: Scenario) -> None:
    """Validate a scenario's internal consistency.

    Checks that all action types are valid, action labels are unique,
    and all expectation action_labels reference defined actions.

    Raises:
        ValueError: if any validation rule is violated.
    """
    labels: set[str] = set()
    for action in scenario.actions:
        if action.type not in VALID_ACTION_TYPES:
            raise ValueError(
                f"Invalid action type '{action.type}' in scenario '{scenario.name}'. "
                f"Valid types: {VALID_ACTION_TYPES}"
            )
        if action.label in labels:
            raise ValueError(
                f"Duplicate action label '{action.label}' in scenario '{scenario.name}'"
            )
        labels.add(action.label)

    for exp in scenario.expectations:
        if exp.action_label not in labels:
            raise ValueError(
                f"Expectation references unknown action '{exp.action_label}' "
                f"in scenario '{scenario.name}'"
            )


def load_scenarios(path: Path | str) -> list[Scenario]:
    """Load and validate scenarios from a JSON file.

    Each entry in the JSON array is parsed into a Scenario dataclass.
    All scenarios are validated via validate_scenario before being returned.

    Args:
        path: Path to the JSON file containing scenario definitions.

    Returns:
        List of validated Scenario objects.

    Raises:
        ValueError: if any scenario fails validation.
        FileNotFoundError: if the path does not exist.
        json.JSONDecodeError: if the file is not valid JSON.
    """
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)

    scenarios: list[Scenario] = []
    for entry in raw:
        actions = [Action(**{k: v for k, v in a.items()}) for a in entry.get("actions", [])]
        expectations = [Expectation(**{k: v for k, v in e.items()}) for e in entry.get("expectations", [])]
        scenario = Scenario(
            name=entry["name"],
            description=entry["description"],
            category=entry["category"],
            requires=entry.get("requires", []),
            actions=actions,
            expectations=expectations,
        )
        validate_scenario(scenario)
        scenarios.append(scenario)

    return scenarios
