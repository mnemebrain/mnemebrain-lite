"""System benchmark runner -- executes scenarios against memory system adapters."""
from __future__ import annotations

from mnemebrain_core.benchmark.interface import MemorySystem
from mnemebrain_core.benchmark.scenarios.schema import Scenario
from mnemebrain_core.benchmark.scoring import (
    ScenarioScore,
    evaluate_expectations,
)


class SystemBenchmarkRunner:
    """Runs benchmark scenarios against MemorySystem adapters.

    For each scenario, capabilities are checked first; scenarios requiring
    capabilities the system lacks are recorded as skipped.  For eligible
    scenarios every action is executed in order and the action results are
    collected, then expectations are evaluated by the scoring layer.
    """

    def run_scenario(self, system: MemorySystem, scenario: Scenario) -> ScenarioScore:
        """Execute a single scenario against *system* and return a ScenarioScore.

        Returns a skipped ScenarioScore when the system is missing any
        capability listed in scenario.requires.
        """
        system_caps = {c.value for c in system.capabilities()}
        for req in scenario.requires:
            if req not in system_caps:
                return ScenarioScore(
                    scenario_name=scenario.name,
                    category=scenario.category,
                    checks=[],
                    skipped=True,
                )

        system.reset()
        action_results: dict[str, object] = {}

        for action in scenario.actions:
            if action.type == "store":
                result = system.store(claim=action.claim or "", evidence=action.evidence or [])
                action_results[action.label] = result
            elif action.type == "query":
                result = system.query(claim=action.claim or "")
                action_results[action.label] = result
            elif action.type == "retract":
                target_result = action_results.get(action.target_label or "")
                if target_result is not None and hasattr(target_result, "belief_id"):
                    result = system.retract(target_result.belief_id)  # type: ignore[union-attr]
                    action_results[action.label] = result
            elif action.type == "explain":
                result = system.explain(claim=action.claim or "")
                action_results[action.label] = result
            elif action.type == "revise":
                target_result = action_results.get(action.target_label or "")
                if target_result is not None and hasattr(target_result, "belief_id"):
                    result = system.revise(
                        belief_id=target_result.belief_id,  # type: ignore[union-attr]
                        evidence=action.evidence or [],
                    )
                    action_results[action.label] = result
            elif action.type == "sandbox_fork":
                result = system.sandbox_fork(
                    scenario_label=action.scenario_label or "",
                )
                action_results[action.label] = result
            elif action.type == "sandbox_assume":
                sandbox_result = action_results.get(action.sandbox_label or "")
                belief_result = action_results.get(action.belief_label or "")
                if (
                    sandbox_result is not None
                    and hasattr(sandbox_result, "sandbox_id")
                    and belief_result is not None
                    and hasattr(belief_result, "belief_id")
                ):
                    result = system.sandbox_assume(
                        sandbox_id=sandbox_result.sandbox_id,  # type: ignore[union-attr]
                        belief_id=belief_result.belief_id,  # type: ignore[union-attr]
                        truth_state=action.truth_state_override or "true",
                    )
                    action_results[action.label] = result
            elif action.type == "sandbox_resolve":
                sandbox_result = action_results.get(action.sandbox_label or "")
                belief_result = action_results.get(action.belief_label or "")
                if (
                    sandbox_result is not None
                    and hasattr(sandbox_result, "sandbox_id")
                    and belief_result is not None
                    and hasattr(belief_result, "belief_id")
                ):
                    result = system.sandbox_resolve(
                        sandbox_id=sandbox_result.sandbox_id,  # type: ignore[union-attr]
                        belief_id=belief_result.belief_id,  # type: ignore[union-attr]
                    )
                    action_results[action.label] = result
            elif action.type == "sandbox_discard":
                sandbox_result = action_results.get(action.sandbox_label or "")
                if sandbox_result is not None and hasattr(sandbox_result, "sandbox_id"):
                    system.sandbox_discard(sandbox_result.sandbox_id)  # type: ignore[union-attr]
                    action_results[action.label] = sandbox_result
            elif action.type == "add_attack":
                attacker_result = action_results.get(action.attacker_label or "")
                target_result = action_results.get(action.target_label or "")
                if (
                    attacker_result is not None
                    and hasattr(attacker_result, "belief_id")
                    and target_result is not None
                    and hasattr(target_result, "belief_id")
                ):
                    result = system.add_attack(
                        attacker_id=attacker_result.belief_id,  # type: ignore[union-attr]
                        target_id=target_result.belief_id,  # type: ignore[union-attr]
                        attack_type=action.attack_type or "contradicts",
                        weight=action.weight or 0.5,
                    )
                    action_results[action.label] = result
            elif action.type == "wait_days":
                if action.wait_days:
                    try:
                        system.set_time_offset_days(action.wait_days)
                    except NotImplementedError:
                        pass

        checks = evaluate_expectations(scenario.expectations, action_results)
        return ScenarioScore(
            scenario_name=scenario.name,
            category=scenario.category,
            checks=checks,
            skipped=False,
        )

    def run_all(
        self,
        systems: list[MemorySystem],
        scenarios: list[Scenario],
    ) -> dict[str, list[ScenarioScore]]:
        """Run all scenarios against every system in *systems*.

        Returns a mapping of system name -> list of ScenarioScore, one entry
        per scenario in the order they were provided.
        """
        results: dict[str, list[ScenarioScore]] = {}
        for system in systems:
            scores: list[ScenarioScore] = []
            for scenario in scenarios:
                scores.append(self.run_scenario(system, scenario))
            results[system.name()] = scores
        return results
