"""One-step constrained planner, with no model calls in the estimator."""
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class Action(str, Enum):
    REUSE = 'reuse'
    RETRIEVE = 'retrieve'
    PAGE_IN = 'page_in'
    SEARCH = 'search'
    CALL_MODEL = 'call_model'
    VERIFY = 'verify'
    SYNTHESIZE = 'synthesize'
    STOP = 'stop'


@dataclass(frozen=True)
class QualityContract:
    required_keys: Tuple[str, ...]
    require_citations: bool = True
    evaluator_version: str = 'structured-ground-truth-v1'


@dataclass(frozen=True)
class ExecutionPlan:
    action: Action
    reason: str
    # Units stay separate: this planner never adds dollars, tokens and scores.
    estimated_input_bytes: int


@dataclass(frozen=True)
class BreakEvenDetector:
    bypass_below_bytes: int = 2048
    minimum_reduction_bytes: int = 256

    def should_bypass(self, original_bytes: int, selected_bytes: int) -> bool:
        return (original_bytes < self.bypass_below_bytes
                or original_bytes - selected_bytes < self.minimum_reduction_bytes)


@dataclass
class PlannerState:
    original_bytes: int
    selected_bytes: int
    selection_ready: bool = False
    response_ready: bool = False
    verification_done: bool = False
    failed: bool = False


class HeuristicPlanner:
    def next_action(self, state: PlannerState) -> ExecutionPlan:
        if state.failed:
            return ExecutionPlan(Action.STOP, 'failure retained; no automatic retry', 0)
        if state.verification_done:
            return ExecutionPlan(Action.STOP, 'quality check finished', 0)
        if state.response_ready:
            return ExecutionPlan(Action.VERIFY, 'deterministic contract check required', 0)
        if not state.selection_ready:
            return ExecutionPlan(Action.RETRIEVE, 'select context without consulting ground truth',
                                 state.original_bytes)
        return ExecutionPlan(Action.CALL_MODEL, 'context selected; one bounded generation',
                             state.selected_bytes)
