from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ApprovalPolicy, ChangeImpact, PreparedChange

RISK_LEVEL_MAP = {"low": 1, "medium": 2, "high": 3}


class ApprovalPolicyEngine:
    def __init__(self, policies: list[ApprovalPolicy]):
        self.policies = policies

    def is_manual_approval_required(
        self, changes: list[PreparedChange], impact: ChangeImpact | None
    ) -> bool:
        """
        Evaluates policies against a set of changes.
        Returns True if manual approval is required, False if auto-approval is granted.
        """
        if not self.policies:
            return True

        total_lines_changed = sum(len(change.diff.splitlines()) for change in changes)
        changed_paths = [change.path for change in changes]

        risk_level = "low"
        if impact and impact.targets:
            target_risks = [RISK_LEVEL_MAP.get(t.risk, 1) for t in impact.targets]
            max_risk_val = max(target_risks) if target_risks else 1
            risk_level = next(
                (k for k, v in RISK_LEVEL_MAP.items() if v == max_risk_val), "low"
            )

        auto_approved_by_policy = False

        for policy in self.policies:
            if self._policy_matches(
                policy, changed_paths, total_lines_changed, risk_level
            ):
                if policy.action == "require_approval":
                    return True
                if policy.action == "auto_approve":
                    auto_approved_by_policy = True

        return not auto_approved_by_policy

    def _policy_matches(
        self,
        policy: ApprovalPolicy,
        changed_paths: list[str],
        lines_changed: int,
        risk_level: str,
    ) -> bool:
        if policy.if_risk_is_at_most:
            if RISK_LEVEL_MAP.get(risk_level, 99) > RISK_LEVEL_MAP.get(
                policy.if_risk_is_at_most, 0
            ):
                return False
        if policy.if_max_lines_changed is not None:
            if lines_changed > policy.if_max_lines_changed:
                return False
        if policy.if_path_matches:
            if not all(any(fnmatch.fnmatch(path, pattern) for pattern in policy.if_path_matches) for path in changed_paths):
                return False
        if policy.if_path_does_not_match:
            if any(any(fnmatch.fnmatch(path, pattern) for pattern in policy.if_path_does_not_match) for path in changed_paths):
                return False
        return True