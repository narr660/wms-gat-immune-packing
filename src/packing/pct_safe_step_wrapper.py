from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SafeStepResult:
    observation: Any
    reward: float
    done: bool
    info: dict
    executed: bool
    placement_succeeded: bool
    safety_penalty: float


class PCTSafeStepWrapper:
    """
    Wraps original PCT env.step().

    The adapter must evaluate a leaf before this method.
    Immune memory changes only after confirmed placement.
    """

    def __init__(
        self,
        env,
        adapter,
        risk_reward_weight=1.0,
        enforce_policy_block=False,
    ):
        self.env = env
        self.adapter = adapter
        self.risk_reward_weight = float(
            risk_reward_weight
        )
        self.enforce_policy_block = bool(
            enforce_policy_block
        )

    def step(
        self,
        leaf,
        incoming_uid,
        incoming_size,
        packed_items,
        observed_exposure,
        leaf_index=0,
    ):
        evaluation = self.adapter.evaluate_leaf(
            leaf_index=leaf_index,
            leaf=leaf,
            incoming_uid=incoming_uid,
            incoming_size=incoming_size,
            packed_items=packed_items,
        )

        if not evaluation.valid:
            raise ValueError(
                "WMS 직접 금지 또는 설정된 정책 차단 후보입니다."
            )

        if (
            self.enforce_policy_block
            and evaluation.policy_blocked
        ):
            raise ValueError(
                "예측 위험이 정책 차단 임계값을 초과했습니다."
            )

        before_count = len(self.env.space.boxes)

        observation, base_reward, done, info = (
            self.env.step(leaf)
        )

        after_count = len(self.env.space.boxes)

        placement_succeeded = (
            after_count == before_count + 1
        )

        penalty = (
            self.risk_reward_weight
            * evaluation.risk_penalty
            if placement_succeeded else 0.0
        )

        reward = float(base_reward) - penalty

        if placement_succeeded:
            self.adapter.commit_success(
                incoming_uid=incoming_uid,
                evaluation=evaluation,
                observed_exposure=observed_exposure,
            )

        info = dict(info)
        info["base_reward"] = float(base_reward)
        info["safety_penalty"] = penalty
        info["placement_succeeded"] = placement_succeeded
        info["hard_forbidden"] = evaluation.hard_forbidden
        info["policy_blocked"] = evaluation.policy_blocked

        return SafeStepResult(
            observation=observation,
            reward=reward,
            done=bool(done),
            info=info,
            executed=True,
            placement_succeeded=placement_succeeded,
            safety_penalty=penalty,
        )
