from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hmt.core.memory_tree import ActionNode, IntentNode, MemoryTree, StageNode


@dataclass
class AWMWorkflowUnit:
    workflow_id: str
    instruction: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        step_text = " ".join(str(step.get("description", step.get("action", ""))) for step in self.steps)
        return f"{self.instruction} {step_text}".strip()


def memory_tree_to_awm_units(tree: MemoryTree) -> list[AWMWorkflowUnit]:
    units: list[AWMWorkflowUnit] = []
    for intent in tree.intents.values():
        stages = [stage for stage in tree.stages.values() if stage.parent_intent_id == intent.intent_id]
        steps = []
        for stage in sorted(stages, key=lambda s: s.span.start):
            actions = [action for action in tree.actions.values() if action.parent_stage_id == stage.stage_id]
            for action in sorted(actions, key=lambda a: a.source.step_index if a.source and a.source.step_index is not None else 0):
                steps.append(
                    {
                        "stage": stage.name,
                        "action": action.operation,
                        "description": action.semantic_description.text(),
                        "argument_template": action.argument_template,
                    }
                )
        units.append(
            AWMWorkflowUnit(
                workflow_id=f"awm_{intent.intent_id}",
                instruction=intent.canonical_intent,
                steps=steps,
                source=intent.source.to_dict() if intent.source else {},
            )
        )
    return units


def awm_unit_to_memory_tree(unit: AWMWorkflowUnit) -> MemoryTree:
    from hmt.core.construction import stable_id
    from hmt.core.memory_tree import SemanticDescription, SourceMetadata, Span

    tree = MemoryTree()
    source = SourceMetadata.from_dict(unit.source) if unit.source else None
    intent = IntentNode(
        intent_id=stable_id("intent", unit.workflow_id, unit.instruction),
        canonical_intent=unit.instruction,
        domain_hint="other",
        source=source,
    )
    tree.add_intent(intent)
    stage = StageNode(
        stage_id=stable_id("stage", unit.workflow_id, "linear"),
        parent_intent_id=intent.intent_id,
        name="linear awm workflow",
        span=Span(1, max(1, len(unit.steps))),
        pre_conditions=["task-relevant page is visible"],
        post_conditions=["workflow has been replayed"],
        source=source,
    )
    tree.add_stage(stage)
    for index, step in enumerate(unit.steps, start=1):
        action = ActionNode(
            action_id=stable_id("act", unit.workflow_id, index, step),
            parent_stage_id=stage.stage_id,
            operation=str(step.get("action", "click")),
            argument_template=step.get("argument_template"),
            semantic_description=SemanticDescription(label_or_text=str(step.get("description", ""))),
            source=SourceMetadata(dataset=source.dataset if source else "", episode_id=source.episode_id if source else "", step_index=index),
        )
        tree.add_action(action)
    return tree
