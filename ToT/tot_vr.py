import os
import re
import json
import time
import argparse
import threading
from dataclasses import dataclass, asdict, replace, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai


@dataclass
class Branch:

    branch_id: str
    vulnerability_type: str = ""
    trigger_point: str = ""
    causal_explanation: str = ""
    status: str = "candidate"
    score: Optional[float] = None
    reason: str = ""
    pruned_at: Optional[str] = None

    claims: List[Dict[str, Any]] = field(default_factory=list)
    claim_checks: List[Dict[str, Any]] = field(default_factory=list)

    code_support_score: Optional[float] = None
    cve_alignment_score: Optional[float] = None
    repair_score: Optional[float] = None
    final_score: Optional[float] = None
    code_verdict: str = ""
    cve_verdict: str = ""
    cve_identity_ok: Optional[bool] = None
    cve_mismatched_dimensions: List[str] = field(default_factory=list)
    cve_unestablished_dimensions: List[str] = field(default_factory=list)
    cve_type_only_identity: bool = False
    repair_consistent: Optional[bool] = None
    repair_patch: str = ""
    verification_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self, include_status: bool = False) -> str:
        parts = [
            f"{self.branch_id}:",
            f"  vulnerability_type: {self.vulnerability_type or '(unspecified)'}",
            f"  trigger_point: {self.trigger_point or '(unspecified)'}",
            f"  causal_explanation: {self.causal_explanation or '(unspecified)'}",
        ]
        if self.claims:
            claim_lines = []
            for claim in self.claims:
                if not isinstance(claim, dict):
                    continue
                text = str(claim.get("claim") or "").strip()
                if not text:
                    continue
                lines = claim.get("evidence_lines") or []
                evidence = str(claim.get("evidence") or "").strip()
                suffix = []
                if lines:
                    suffix.append("lines=" + ",".join(str(x) for x in lines))
                if evidence:
                    suffix.append("evidence=" + evidence)
                claim_lines.append("    - " + text + (" (" + "; ".join(suffix) + ")" if suffix else ""))
            if claim_lines:
                parts.append("  claims:\n" + "\n".join(claim_lines))
        if include_status:
            parts.append(f"  status: {self.status}")
            if self.score is not None:
                parts.append(f"  score: {self.score:.3f}")
            if self.code_support_score is not None:
                parts.append(f"  code_support_score: {self.code_support_score:.3f}")
            if self.cve_alignment_score is not None:
                parts.append(f"  cve_alignment_score: {self.cve_alignment_score:.3f}")
            if self.repair_score is not None:
                parts.append(f"  repair_score: {self.repair_score:.3f}")
            if self.reason:
                parts.append(f"  reason: {self.reason}")
        return "\n".join(parts)


def _coerce_score(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "yes", "1", "supported", "consistent"):
        return True
    if text in ("false", "no", "0", "unsupported", "contradicted", "inconsistent"):
        return False
    return None


def _coerce_dimension_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = re.split(r"[,;]+", value)
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    out: List[str] = []
    for item in value:
        for part in re.split(r"[,;]+", str(item)):
            text = part.strip().lower().replace(" ", "_")
            if text and text not in out:
                out.append(text)
    return out


def _coerce_line_numbers(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        try:
            return [int(value)]
        except (TypeError, ValueError):
            return []
    if isinstance(value, str):
        return [int(x) for x in re.findall(r"\bL?(\d+)\b", value)]
    if isinstance(value, (list, tuple, set)):
        out: List[int] = []
        for item in value:
            out.extend(_coerce_line_numbers(item))
        return sorted(set(out))
    return []


def _is_meta_claim(claim: str) -> bool:
    text = claim.strip().lower()
    if not text:
        return False
    patterns = (
        r"^(the\s+)?(official\s+)?cve\b",
        r"^(the\s+)?(security\s+)?advisory\b",
        r"^(the\s+)?commit\s+(message|description)\b",
        r"\bcve\s+(description|says|states|reports|identifies)\b",
        r"\bofficial\s+description\b",
    )
    return any(re.search(p, text) for p in patterns)


def _coerce_claims(value: Any) -> List[Dict[str, Any]]:
    items: List[Any] = []
    if isinstance(value, list):
        items = value
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict):
                item = dict(item)
                item.setdefault("claim_type", str(key))
                items.append(item)
            else:
                items.append({"claim": str(item), "claim_type": str(key)})
    elif isinstance(value, str):
        items = [{"claim": value, "claim_type": "other"}]

    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            item = {"claim": item}
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or item.get("text") or item.get("statement") or "").strip()
        if not claim or _is_meta_claim(claim):
            continue
        evidence_lines = _coerce_line_numbers(
            item.get("evidence_lines")
            or item.get("lines")
            or item.get("line_numbers")
            or item.get("line")
        )
        out.append(
            {
                "claim": claim,
                "claim_type": str(item.get("claim_type") or item.get("type") or "other").strip().lower(),
                "evidence_lines": evidence_lines,
                "evidence": str(item.get("evidence") or item.get("code_evidence") or "").strip(),
                "confidence": _coerce_score(item.get("confidence")),
            }
        )
    return out


def _branch_from_mapping(item: Dict[str, Any], fallback_id: str) -> Branch:
    bid = item.get("branch_id") or item.get("id") or item.get("branch") or item.get("label") or fallback_id
    bid = str(bid).strip()
    m = re.search(r"(\d+)\s*$", bid)
    if m:
        bid = f"Branch{int(m.group(1))}"
    elif not bid:
        bid = fallback_id
    return Branch(
        branch_id=bid,
        vulnerability_type=str(item.get("vulnerability_type") or item.get("type") or item.get("vulnerability") or "").strip(),
        trigger_point=str(item.get("trigger_point") or item.get("trigger") or item.get("trigger_condition") or "").strip(),
        causal_explanation=str(item.get("causal_explanation") or item.get("explanation") or item.get("causal_mechanism") or item.get("mechanism") or "").strip(),
        status=str(item.get("status") or "candidate").strip() or "candidate",
        score=_coerce_score(item.get("score")),
        reason=str(item.get("reason") or "").strip(),
        claims=_coerce_claims(item.get("claims") or item.get("claim_ledger") or item.get("evidence_claims")),
        code_support_score=_coerce_score(item.get("code_support_score")),
        cve_alignment_score=_coerce_score(item.get("cve_alignment_score")),
        repair_score=_coerce_score(item.get("repair_score")),
        final_score=_coerce_score(item.get("final_score")),
        code_verdict=str(item.get("code_verdict") or "").strip(),
        cve_verdict=str(item.get("cve_verdict") or "").strip(),
        cve_identity_ok=_coerce_bool(item.get("cve_identity_ok")),
        cve_mismatched_dimensions=_coerce_dimension_list(item.get("cve_mismatched_dimensions")),
        cve_unestablished_dimensions=_coerce_dimension_list(item.get("cve_unestablished_dimensions")),
        cve_type_only_identity=bool(_coerce_bool(item.get("cve_type_only_identity"))),
        repair_consistent=_coerce_bool(item.get("repair_consistent")),
        repair_patch=str(item.get("repair_patch") or item.get("patch") or "").strip(),
        verification_notes=str(item.get("verification_notes") or "").strip(),
    )


def _dedupe_branches(branches: Sequence[Branch]) -> List[Branch]:
    seen = set()
    out: List[Branch] = []
    for b in branches:
        key = (b.vulnerability_type.strip().lower(), b.trigger_point.strip().lower(), b.causal_explanation.strip().lower())
        if not any(key) or key in seen:
            continue
        seen.add(key)
        out.append(b)
    return out


def extract_json(text: str) -> Optional[Any]:
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S | re.I)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for token in ("{", "["):
        start = 0
        while True:
            idx = cleaned.find(token, start)
            if idx < 0:
                break
            try:
                obj, _ = decoder.raw_decode(cleaned[idx:])
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                pass
            start = idx + 1
    return None


def _json_only_prompt(prompt: str) -> str:
    return prompt + "\n\nIMPORTANT: Return valid JSON only. Do not use Markdown fences, comments, or any text before or after the JSON object."


class ToTVRGenerator:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        base_url: Optional[str] = None,
        min_branches: int = 2,
        max_branches: int = 6,
        max_reconsideration_rounds: int = 1,
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        timeout: float = 120.0,
        client: Any = None,
        reasoning_effort: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        verification_mode: str = "dual",
        min_verification_branches: int = 2,
        enable_repair_check: bool = True,
        enable_claim_reflection: bool = True,
        reflection_rounds: int = 1,
        enable_description_audit: bool = True,
        code_support_threshold: float = 0.65,
        cve_alignment_threshold: float = 0.60,
        type_only_cve_threshold: float = 0.35,
        repair_threshold: float = 0.30,
        final_threshold: float = 0.60,
        audit_threshold: float = 0.70,
        max_repair_candidates: int = 2,
    ):
        if min_branches < 1:
            raise ValueError("min_branches must be >= 1")
        if max_branches < min_branches:
            raise ValueError("max_branches must be >= min_branches")
        if verification_mode not in ("dual", "legacy"):
            raise ValueError("verification_mode must be 'dual' or 'legacy'")

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if client is None and not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in the OPENAI_API_KEY environment variable.")
        if client is None:
            client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = openai.OpenAI(**client_kwargs)

        self.client = client
        self.model = model
        self.min_branches = min_branches
        self.max_branches = max_branches
        self.max_reconsideration_rounds = max(0, max_reconsideration_rounds)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.extra_body = dict(extra_body or {})
        self.verbose = verbose

        self.verification_mode = verification_mode
        self.min_verification_branches = max(1, min_verification_branches)
        self.enable_repair_check = enable_repair_check
        self.enable_claim_reflection = enable_claim_reflection
        self.reflection_rounds = max(0, reflection_rounds)
        self.enable_description_audit = enable_description_audit
        self.code_support_threshold = code_support_threshold
        self.cve_alignment_threshold = cve_alignment_threshold
        self.type_only_cve_threshold = type_only_cve_threshold
        self.repair_threshold = repair_threshold
        self.final_threshold = final_threshold
        self.audit_threshold = audit_threshold
        self.max_repair_candidates = max(1, max_repair_candidates)

        self.system_prompt = (
            "You are a security analysis assistant specialized in source-code vulnerability reasoning. "
            "Follow the instructions for each step and use only the information provided in the current prompt. "
            "When multiple candidate branches are required, label them Branch1 through BranchN, where N is "
            "determined by the evidence and may vary across samples; do not force a fixed number of branches. "
            "When refining, pruning, or verifying branches, base the decision on the provided code and evidence. "
            "Prefer explicit, line-grounded claims over broad prose. Follow the required output format and do not "
            "include unrelated content."
        )

    def _is_reasoning_model(self) -> bool:
        name = (self.model or "").lower()
        return name.startswith(("gpt-5", "o1", "o3", "o4"))

    def _call_llm(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        use_temperature = not self._is_reasoning_model()
        use_completion_tokens = self._is_reasoning_model()
        last_error: Optional[Exception] = None

        for attempt in range(4):
            kwargs: Dict[str, Any] = {"model": self.model, "messages": messages, "timeout": self.timeout}
            if use_temperature:
                kwargs["temperature"] = self.temperature
            if use_completion_tokens:
                kwargs["max_completion_tokens"] = self.max_output_tokens
            else:
                kwargs["max_tokens"] = self.max_output_tokens
            if self.reasoning_effort and self._is_reasoning_model():
                kwargs["reasoning_effort"] = self.reasoning_effort
            if self.extra_body:
                kwargs["extra_body"] = self.extra_body
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                return content.strip()
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if "temperature" in message and "temperature" in kwargs:
                    use_temperature = False
                    continue
                if "max_completion_tokens" in message and use_completion_tokens:
                    use_completion_tokens = False
                    continue
                if "max_tokens" in message and not use_completion_tokens:
                    use_completion_tokens = True
                    continue
                if "reasoning_effort" in message and "reasoning_effort" in kwargs:
                    self.reasoning_effort = None
                    continue
                if attempt < 3:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                break
        raise RuntimeError(f"LLM call failed after retries: {last_error}")

    def _call_json(self, prompt: str, stage: str) -> Tuple[Any, str]:
        text = self._call_llm(_json_only_prompt(prompt))
        data = extract_json(text)
        if data is not None:
            return data, text
        repair = (
            "Your previous response was not valid JSON and could not be parsed. "
            "Return the same analysis again as one valid JSON object only. "
            "Do not add Markdown fences or prose.\n\nPrevious response:\n" + text[:6000]
        )
        text2 = self._call_llm(_json_only_prompt(prompt + "\n\n" + repair))
        data = extract_json(text2)
        if data is None:
            raise ValueError(f"Could not parse JSON returned by stage: {stage}")
        return data, text2

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _next_branch_id(self, branches: Sequence[Branch]) -> str:
        used = set()
        for b in branches:
            m = re.search(r"(\d+)$", b.branch_id)
            if m:
                used.add(int(m.group(1)))
        n = 1
        while n in used:
            n += 1
        return f"Branch{n}"

    def _fit_branch_count(self, branches: List[Branch], allow_empty: bool = False) -> List[Branch]:
        branches = _dedupe_branches(branches)
        if len(branches) > self.max_branches:
            branches = branches[: self.max_branches]
        if not branches and not allow_empty:
            raise ValueError("The model returned no candidate branches.")
        return branches

    @staticmethod
    def _format_branches(branches: Sequence[Branch], include_status: bool = False) -> str:
        if not branches:
            return "(none)"
        return "\n\n".join(b.describe(include_status=include_status) for b in branches)

    @staticmethod
    def _format_line_evidence(function_code: str, line_numbers: Sequence[Any], limit: int = 40) -> str:
        code_lines = function_code.splitlines()
        out: List[str] = []
        seen = set()
        for raw in list(line_numbers)[:limit]:
            try:
                n = int(raw)
            except (TypeError, ValueError):
                out.append(str(raw))
                continue
            if n in seen:
                continue
            seen.add(n)
            if 1 <= n <= len(code_lines):
                out.append(f"L{n}: {code_lines[n - 1].strip()}")
            else:
                out.append(f"L{n}: <line unavailable>")
        return "\n".join(out) if out else "(none)"

    def _ensure_verification_diversity(
        self,
        retained: Sequence[Branch],
        archived: Sequence[Branch],
    ) -> Tuple[List[Branch], List[Branch]]:
        pool = list(retained)
        if len(pool) >= self.min_verification_branches:
            return pool, list(archived)

        def mechanism_key(branch: Branch) -> str:
            text = f"{branch.vulnerability_type} {branch.trigger_point}".lower()
            text = re.sub(r"\bbranch\s*\d+\b", " ", text)
            text = re.sub(r"[^a-z0-9]+", " ", text)
            return " ".join(text.split())

        existing = {mechanism_key(b) for b in pool}
        candidates: List[Branch] = []
        for branch in archived:
            if branch.status == "pruned":
                key = mechanism_key(branch)
                if key and key not in existing:
                    candidates.append(branch)
        candidates.sort(
            key=lambda b: (
                len(b.causal_explanation or ""),
                b.code_support_score if b.code_support_score is not None else -1.0,
            ),
            reverse=True,
        )
        revived: List[Branch] = []
        remaining = [b for b in archived if b not in candidates]
        for branch in candidates:
            if len(pool) >= self.min_verification_branches:
                break
            copy = replace(
                branch,
                status="revived_for_verification",
                pruned_at=None,
                reason=(branch.reason + " | revived to preserve branch diversity").strip(" |"),
            )
            pool.append(copy)
            revived.append(copy)
        return pool, remaining

    def _parse_branch_list(self, data: Any, fallback: Sequence[Branch] = ()) -> List[Branch]:
        items: List[Dict[str, Any]] = []
        if isinstance(data, list):
            items = [x for x in data if isinstance(x, dict)]
        elif isinstance(data, dict):
            for key in ("branches", "candidate_branches", "revised_branches", "retained_branches", "refined_branches", "new_branches"):
                value = data.get(key)
                if isinstance(value, list):
                    items.extend(x for x in value if isinstance(x, dict))
            if not items and any(k in data for k in ("vulnerability_type", "type")):
                items = [data]
        existing_ids = {b.branch_id for b in fallback}
        out: List[Branch] = []
        for idx, item in enumerate(items, start=1):
            fallback_id = f"Branch{idx}" if f"Branch{idx}" not in existing_ids else self._next_branch_id(list(out) + list(fallback))
            out.append(_branch_from_mapping(item, fallback_id))
        return out

    @staticmethod
    def _merge_branch_updates(current: Sequence[Branch], updates: Sequence[Branch]) -> List[Branch]:
        by_id = {b.branch_id: b for b in current}
        merged: List[Branch] = []
        updated_ids = set()
        for upd in updates:
            updated_ids.add(upd.branch_id)
            base = by_id.get(upd.branch_id)
            if base is None:
                merged.append(upd)
                continue
            if upd.vulnerability_type:
                base.vulnerability_type = upd.vulnerability_type
            if upd.trigger_point:
                base.trigger_point = upd.trigger_point
            if upd.causal_explanation:
                base.causal_explanation = upd.causal_explanation
            if upd.claims:
                base.claims = upd.claims
            if upd.score is not None:
                base.score = upd.score
            if upd.reason:
                base.reason = upd.reason
            merged.append(base)
        for branch in current:
            if branch.branch_id not in updated_ids:
                merged.append(branch)
        return merged

    def _initialization_prompt(self, function_code: str, critical_lines: str) -> str:
        return (
            "Analyze the following source code and construct a variable number of candidate reasoning branches. "
            f"The number of branches must be between {self.min_branches} and {self.max_branches}, inclusive, "
            "and must be determined by the evidence. Do not pad the list to a fixed count and do not force exactly "
            "three branches. If the code supports only a few plausible interpretations, return fewer branches.\n\n"
            "Each branch must describe a distinct candidate vulnerability interpretation and contain: (1) a possible "
            "vulnerability type, (2) a concrete trigger point in the code, and (3) a detailed causal explanation of how "
            "the trigger leads to the security impact, including the relevant data flow and the missing or "
            "insufficient check.\n\n"
            f"Code:\n{function_code}\n\nCritical Lines:\n{critical_lines}\n\n"
            "Return a JSON object with this schema:\n"
            '{"branches": [{"id": "Branch1", "vulnerability_type": "...", "trigger_point": "...", "causal_explanation": "..."}, {"id": "Branch2", "...": "..."}]}\n'
            "Use consecutive branch labels Branch1, Branch2, ... and return only branches that are genuinely distinct."
        )

    def _expand_prompt(self, function_code: str, critical_lines: str, existing: Sequence[Branch], target_min: int) -> str:
        return (
            "The current candidate pool contains too few distinct interpretations. Propose additional candidate "
            f"reasoning branches until the pool has at least {target_min} branches, without exceeding {self.max_branches}. "
            "Each new branch must be distinct from the existing branches and must contain a vulnerability type, a "
            "concrete trigger point, and a causal explanation.\n\n"
            f"Code:\n{function_code}\n\nCritical Lines:\n{critical_lines}\n\n"
            f"Existing Branches:\n{self._format_branches(existing)}\n\n"
            "Return a JSON object with this schema:\n"
            '{"branches": [{"id": "BranchN", "vulnerability_type": "...", "trigger_point": "...", "causal_explanation": "..."}]}'
        )

    def step_1_branch_initialization(self, function_code: str, critical_lines: str) -> Tuple[List[Branch], str, Dict[str, Any]]:
        data, raw = self._call_json(self._initialization_prompt(function_code, critical_lines), stage="branch initialization")
        branches = self._fit_branch_count(self._parse_branch_list(data), allow_empty=True)
        expansion_info: Dict[str, Any] = {"expanded": False, "raw": None, "added": 0}

        if len(branches) < self.min_branches:
            self._log(f"  Expanding candidate pool from {len(branches)} to at least {self.min_branches} branches.")
            try:
                exp_data, exp_raw = self._call_json(self._expand_prompt(function_code, critical_lines, branches, self.min_branches), stage="branch expansion")
                new_branches = self._parse_branch_list(exp_data, fallback=branches)
                before = len(branches)
                branches = self._fit_branch_count(list(branches) + list(new_branches), allow_empty=True)
                expansion_info.update(expanded=True, raw=exp_raw, added=max(0, len(branches) - before))
            except Exception as exc:
                expansion_info["error"] = str(exc)
                self._log(f"  Branch expansion failed: {exc}")

        if not branches:
            raise ValueError("Branch initialization produced no branches.")
        for idx, branch in enumerate(branches, start=1):
            branch.branch_id = f"Branch{idx}"
            branch.status = "candidate"
        return branches, raw, expansion_info

    def _reflection_prompt(self, function_code: str, branches: Sequence[Branch]) -> str:
        return (
            "Review the candidate branches against the source code. For each branch, check whether it is internally "
            "consistent and supported by the code semantics. Revise unclear or weak branches when a minor correction "
            "makes them plausible, and prune only branches that are clearly unsupported or contradictory. Retain "
            "several plausible branches when the evidence is inconclusive. Do not force a fixed number of retained "
            "branches. You may add a new branch only if the review reveals a genuinely distinct interpretation that "
            "was missed.\n\n"
            f"Code:\n{function_code}\n\nCandidate Branches:\n{self._format_branches(branches)}\n\n"
            "Return a JSON object with this schema:\n"
            '{"branches": [{"id": "Branch1", "decision": "retain|revise|prune", "vulnerability_type": "...", "trigger_point": "...", "causal_explanation": "...", "reason": "..."}], "reflection": "..."}\n'
            "Every input branch must appear exactly once in the output. For a revised branch, provide the corrected "
            "fields. For a pruned branch, explain the contradiction. New branches, if any, should use a new BranchN id "
            "and decision retain."
        )

    def _apply_decisions(self, data: Any, current: Sequence[Branch]) -> Tuple[List[Branch], List[Branch], Dict[str, str], Dict[str, str]]:
        decisions: Dict[str, str] = {}
        reasons: Dict[str, str] = {}
        if isinstance(data, dict):
            for item in data.get("branches", []) or []:
                if not isinstance(item, dict):
                    continue
                bid = str(item.get("id") or item.get("branch_id") or "").strip()
                m = re.search(r"(\d+)$", bid)
                if m:
                    bid = f"Branch{int(m.group(1))}"
                decisions[bid] = str(item.get("decision") or "").strip().lower()
                reasons[bid] = str(item.get("reason") or "").strip()
        kept: List[Branch] = []
        pruned: List[Branch] = []
        for branch in current:
            decision = decisions.get(branch.branch_id, "retain")
            if decision.startswith("prune"):
                branch.status = "pruned"
                branch.reason = reasons.get(branch.branch_id, branch.reason)
                pruned.append(branch)
            else:
                kept.append(branch)
        return kept, pruned, decisions, reasons

    def step_2_branch_reflection_and_pruning(self, function_code: str, branches: Sequence[Branch]) -> Tuple[List[Branch], List[Branch], str, Dict[str, Any]]:
        data, raw = self._call_json(self._reflection_prompt(function_code, branches), stage="branch reflection and pruning")
        updates = self._parse_branch_list(data, fallback=branches)
        if not updates:
            for b in branches:
                b.status = "retained"
            return list(branches), [], raw, {"fallback": True}

        updates = self._fit_branch_count(updates, allow_empty=True)
        merged = self._merge_branch_updates(branches, updates)
        retained, pruned, decisions, reasons = self._apply_decisions(data, merged)
        for b in pruned:
            b.pruned_at = "reflection"
            b.status = "pruned"
        for b in retained:
            b.status = "retained"
            b.pruned_at = None

        if not retained:
            best = max(merged, key=lambda b: len(b.causal_explanation))
            best.status = "retained"
            best.pruned_at = None
            retained = [best]
            pruned = [b for b in pruned if b.branch_id != best.branch_id]

        retained = self._fit_branch_count(retained)
        info = {"reflection": data.get("reflection", "") if isinstance(data, dict) else "", "decisions": decisions}
        return retained, pruned, raw, info

    def _refinement_prompt(
        self,
        branches: Sequence[Branch],
        vulnerable_lines: str,
        line_relations: str,
        critical_lines: str = "",
        function_code: str = "",
    ) -> str:
        return (
            "Refine the retained branches using the actual source text of the critical lines and the typed line "
            "relations below. Do not rely only on the branch prose. For every kept branch, produce an explicit "
            "claim ledger. Each claim must be one atomic, checkable statement and must cite the line number(s) that "
            "support it. Split compound statements: keep the visible code fact (call arguments, assignments, guards, "
            "returns) as its own claim, and put any inference about unshown callee or caller behavior in a separate "
            "claim. If a claim is an inference rather than a direct code fact, say so in evidence and lower its "
            "confidence. Every branch must contain at least one claim that is directly checkable against the shown "
            "function; do not let the entire ledger depend on code that is not provided. Label at least one claim as "
            "root_cause and at least one claim as impact. If the root cause lies in an unshown callee or caller, still "
            "include a root_cause claim that cites the visible call site that reaches it, and state the unshown part "
            "in its evidence. Do not omit the root cause from the ledger merely because it is not fully visible.\n\n"
            "Claim types: root_cause, trigger, data_flow, missing_check, impact, other.\n"
            "Do not restate the CVE or an advisory as a claim. Prune a branch only when the source text or relation "
            "clearly contradicts it. Lack of evidence about an unshown callee, caller, or framework is NOT a "
            "contradiction: if the shown call site is consistent with the branch, retain it, keep the visible "
            "call-site facts as supported claims, and put the behavior of unshown code in a separate claim whose "
            "evidence says 'not shown (callee/caller)' and whose confidence is lower. Do not prune a branch merely "
            "because its root cause is located in a callee that is not part of the supplied function. Do not force a "
            "fixed number of branches.\n\n"
            f"Source Code (authoritative line numbering):\n{function_code or '(not provided)'}\n\n"
            f"Critical Lines (source text):\n{critical_lines or '(none)'}\n\n"
            f"Selected Vulnerable Lines (source text):\n{vulnerable_lines}\n\n"
            f"Typed Line Relations:\n{line_relations}\n\n"
            f"Retained Branches:\n{self._format_branches(branches)}\n\n"
            "Return a JSON object with this schema:\n"
            '{"branches": [{"id": "Branch1", "decision": "refine|retain|prune", '
            '"vulnerability_type": "...", "trigger_point": "...", "causal_explanation": "...", '
            '"claims": [{"claim": "...", "claim_type": "root_cause|trigger|data_flow|missing_check|impact|other", '
            '"evidence_lines": [12, 15], "evidence": "...", "confidence": 0.0}], "reason": "..."}], '
            '"refinement": "..."}\n'
            "Every input branch must appear exactly once. For kept branches, provide the updated trigger point, "
            "causal explanation, and claim ledger grounded in the source text."
        )

    def step_3_branch_refinement(
        self,
        branches: Sequence[Branch],
        vulnerable_lines: str,
        line_relations: str,
        critical_lines: str = "",
        function_code: str = "",
    ) -> Tuple[List[Branch], List[Branch], str, Dict[str, Any]]:
        data, raw = self._call_json(
            self._refinement_prompt(branches, vulnerable_lines, line_relations, critical_lines, function_code),
            stage="branch refinement",
        )
        updates = self._parse_branch_list(data, fallback=branches)
        if not updates:
            for b in branches:
                b.status = "refined"
            return list(branches), [], raw, {"fallback": True}

        updates = self._fit_branch_count(updates, allow_empty=True)
        merged = self._merge_branch_updates(branches, updates)
        refined, pruned, decisions, reasons = self._apply_decisions(data, merged)
        for b in pruned:
            b.pruned_at = "refinement"
            b.status = "pruned"
        for b in refined:
            b.status = "refined"
            b.pruned_at = None

        if not refined:
            best = max(merged, key=lambda b: len(b.causal_explanation))
            best.status = "refined"
            best.pruned_at = None
            refined = [best]
            pruned = [b for b in pruned if b.branch_id != best.branch_id]

        refined = self._fit_branch_count(refined)
        info = {"refinement": data.get("refinement", "") if isinstance(data, dict) else "", "decisions": decisions}
        return refined, pruned, raw, info

    def _verification_prompt(self, branches: Sequence[Branch], cve_description: str) -> str:
        if cve_description.strip():
            evidence_block = (
                "Compare every refined branch with the official CVE description. Judge consistency in vulnerability "
                "type, root cause, trigger condition, and impact. Rank the branches by how well they are supported. "
                "Select the best-supported branch as the verified branch. If no branch is sufficiently supported, set "
                "verification_result to reconsider and verified_branch_id to null; the caller will then revisit "
                "alternative or previously pruned interpretations.\n\n"
                f"Official CVE Description:\n{cve_description}\n"
            )
        else:
            evidence_block = (
                "No reliable official CVE description is available for this sample. Rank the refined branches by "
                "internal consistency with the code and structural evidence, and select the best-supported branch. Set "
                "verification_result to consistent when a branch is clearly supported; otherwise set it to reconsider "
                "and verified_branch_id to null.\n"
            )
        return (
            f"{evidence_block}\nRefined Branches:\n{self._format_branches(branches)}\n\n"
            "Return a JSON object with this schema:\n"
            '{"ranked_branches": [{"id": "Branch1", "score": 0.0, "consistent": true, "reason": "..."}], "verified_branch_id": "Branch1" or null, "verification_result": "consistent|reconsider", "reason": "..."}\n'
            "Scores must be between 0 and 1. Use the same branch ids as the input."
        )

    @staticmethod
    def _lookup_claim_type(branch: Branch, claim_text: str) -> str:
        if not claim_text:
            return ""
        needle = re.sub(r"\s+", " ", claim_text).strip().lower()
        for claim in branch.claims or []:
            if not isinstance(claim, dict):
                continue
            text = re.sub(r"\s+", " ", str(claim.get("claim") or "")).strip().lower()
            if text and (text == needle or text in needle or needle in text):
                return str(claim.get("claim_type") or "").strip().lower()
        return ""

    def _structural_verification_prompt(
        self,
        branches: Sequence[Branch],
        function_code: str,
        critical_lines: str,
    ) -> str:
        return (
            "You are an adversarial code verifier. Do NOT use a CVE description in this step. Check every branch "
            "against the actual source code and its claim ledger. For each claim, try to falsify it: does the cited "
            "line actually support the claim? Is the claimed data flow present? Is the claimed missing check really "
            "missing? Mark a claim as supported only when the code text justifies it.\n\n"
            "Verdict values:\n"
            "- supported: the shown source text directly justifies the claim.\n"
            "- uncertain: the shown code is compatible with the claim but does not fully establish it.\n"
            "- out_of_scope: the claim depends on code that is not shown (callee body, caller, framework, or runtime "
            "state), and the shown code neither confirms nor contradicts it. Treat this as conditional rather than "
            "failed; do not use it to punish a branch.\n"
            "- unsupported: the claim asserts a fact that should be visible in the shown code but no shown line "
            "supports it, or the cited lines do not contain the claimed fact.\n"
            "- contradicted: the shown source text contradicts the claim.\n\n"
            "A contradicted claim is a hard failure. Unsupported claims reduce support; out_of_scope claims do not. "
            "If a claim mixes a visible fact with an inference about unshown code, split it into separate checks: "
            "judge the visible part against the source and mark only the external part out_of_scope. Always report "
            "the concrete call-site, assignment, guard, and return facts visible in the shown function as their own "
            "supported checks when they are true. Copy the claim_type (root_cause, trigger, data_flow, missing_check, "
            "impact, other) from the claim ledger when available.\n\n"
            f"Source Code:\n{function_code}\n\nCritical Lines (source text):\n{critical_lines}\n\n"
            f"Branches and Claim Ledgers:\n{self._format_branches(branches)}\n\n"
            "Return a JSON object with this schema:\n"
            '{"branches": [{"id": "Branch1", '
            '"claim_checks": [{"claim": "...", "claim_type": "root_cause|trigger|data_flow|missing_check|impact|other", '
            '"verdict": "supported|uncertain|out_of_scope|unsupported|contradicted", '
            '"evidence_lines": [12], "reason": "..."}], '
            '"code_support_score": 0.0, "contradictions": ["..."], '
            '"verdict": "supported|weak|contradicted", "notes": "..."}], "reason": "..."}\n'
            "Use the same branch ids as the input. Scores must be between 0 and 1."
        )

    @staticmethod
    def _normalize_code_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text).strip())

    @staticmethod
    def _strip_line_number_annotations(text: str) -> str:
        cleaned = str(text)
        cleaned = re.sub(
            r"\s*\((?:see\s+)?(?:lines?|L)\s*\d+"
            r"(?:\s*(?:,|and|-|–)\s*(?:(?:lines?|L)\s*)?\d+)*\)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s+(?:at|on)\s+(?:lines?|L)\s*\d+"
            r"(?:\s*(?:,|and|-|–)\s*(?:(?:lines?|L)\s*)?\d+)*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b(?:lines?|L)\s*\d+"
            r"(?:\s*(?:,|and|-|–)\s*(?:(?:lines?|L)\s*)?\d+)*",
            "the cited statement",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
        return cleaned.strip()

    @classmethod
    def _extract_quoted_snippets(cls, text: str) -> List[str]:
        snippets: List[str] = []
        for match in re.finditer(r"(['\"`])(.{4,240}?)\1", str(text)):
            snippet = match.group(2).strip()
            if not snippet:
                continue
            if not re.search(r"[A-Za-z0-9_]", snippet):
                continue
            identifier = re.search(
                r"[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]*"
                r"|[A-Za-z_][A-Za-z0-9_]*\s*(?:->|\.)\s*[A-Za-z_]"
                r"|[A-Za-z_][A-Za-z0-9_]*\s*\(",
                snippet,
            )
            structure = re.search(
                r"[(){}\[\];=<>+*/&|!,]|\b(if|for|while|return|goto|sizeof)\b",
                snippet,
            )
            if not identifier or not structure:
                continue
            lower = snippet.lower()
            if lower in ("not shown", "callee", "caller", "no shown line", "verbatim", "not shown (callee/caller)"):
                continue
            snippets.append(snippet)
        return snippets

    def _verify_check_citations(
        self,
        check: Dict[str, Any],
        code_lines: Sequence[str],
    ) -> None:
        snippets = self._extract_quoted_snippets(check.get("reason") or "")
        if not snippets:
            check["citation_status"] = "not_provided"
            return
        norm_lines = [self._normalize_code_text(line) for line in code_lines]
        full = " ".join(norm_lines)
        line_starts: List[int] = []
        cursor = 0
        for norm in norm_lines:
            line_starts.append(cursor)
            cursor += len(norm) + 1
        found: set = set()
        for snippet in snippets:
            needle = self._normalize_code_text(snippet)
            if not needle:
                continue
            search_at = 0
            while True:
                hit = full.find(needle, search_at)
                if hit < 0:
                    break
                line_no = 1
                for idx, start in enumerate(line_starts, start=1):
                    if hit < start:
                        break
                    line_no = idx
                found.add(line_no)
                search_at = hit + 1
        old = set(check.get("evidence_lines") or [])
        if not found:
            check["citation_status"] = "unverified"
            if check.get("verdict") == "supported":
                check["verdict"] = "uncertain"
                note = "citation not found in the shown source; verdict downgraded to uncertain"
                reason = str(check.get("reason") or "").strip()
                check["reason"] = f"{reason} [{note}]" if reason else note
            return
        if old and old.issubset(found):
            check["citation_status"] = "verified"
            check["evidence_lines"] = sorted(old)
            return
        relocated = found if not old else (found - old)
        if not relocated:
            relocated = found
        check["citation_status"] = "relocated"
        note = "cited line(s) relocated from the quoted code: " + ", ".join(
            f"L{idx}" for idx in sorted(relocated)
        )
        reason = str(check.get("reason") or "").strip()
        check["reason"] = f"{reason} [{note}]" if reason else note
        check["evidence_lines"] = sorted(found if not old else (old & found) | relocated)

    def _apply_code_verification_data(
        self,
        data: Any,
        branches: Sequence[Branch],
        function_code: str = "",
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            raw_items = data.get("branches") or data.get("branch_verification") or []
            if isinstance(raw_items, list):
                items = [x for x in raw_items if isinstance(x, dict)]

        by_id = {b.branch_id: b for b in branches}
        results: List[Dict[str, Any]] = []
        for item in items:
            bid = str(item.get("id") or item.get("branch_id") or "").strip()
            m = re.search(r"(\d+)", bid)
            if m:
                bid = f"Branch{int(m.group(1))}"
            branch = by_id.get(bid)
            if branch is None:
                continue
            checks_raw = item.get("claim_checks") or item.get("claims") or []
            checks: List[Dict[str, Any]] = []
            if isinstance(checks_raw, list):
                for check in checks_raw:
                    if not isinstance(check, dict):
                        continue
                    verdict = str(check.get("verdict") or check.get("status") or "").strip().lower()
                    if verdict not in ("supported", "uncertain", "out_of_scope", "unsupported", "contradicted"):
                        verdict = "uncertain"
                    claim_text = str(check.get("claim") or check.get("text") or "").strip()
                    claim_type = str(check.get("claim_type") or check.get("type") or "").strip().lower()
                    original_type = self._lookup_claim_type(branch, claim_text)
                    if original_type:
                        claim_type = original_type
                    elif not claim_type:
                        claim_type = "other"
                    checks.append(
                        {
                            "claim": claim_text,
                            "claim_type": claim_type,
                            "verdict": verdict,
                            "evidence_lines": _coerce_line_numbers(
                                check.get("evidence_lines") or check.get("lines")
                            ),
                            "reason": str(check.get("reason") or "").strip(),
                        }
                    )
            score = _coerce_score(item.get("code_support_score"))
            if function_code:
                code_lines = function_code.splitlines()
                for check in checks:
                    self._verify_check_citations(check, code_lines)
            if checks:
                weights = {
                    "supported": 1.0,
                    "uncertain": 0.6,
                    "out_of_scope": 0.6,
                    "unsupported": 0.0,
                    "contradicted": 0.0,
                }
                score = sum(weights.get(c["verdict"], 0.0) for c in checks) / len(checks)
                core_types = {"root_cause", "trigger", "data_flow", "missing_check"}
                supported_core = any(
                    c["verdict"] == "supported" and c.get("claim_type") in core_types
                    for c in checks
                )
                if not supported_core:
                    score = min(score, 0.40)
            elif score is None:
                score = 0.0
            contradictions = item.get("contradictions") or []
            if isinstance(contradictions, str):
                contradictions = [contradictions]
            contradictions = [str(c) for c in contradictions if str(c).strip()]
            hard_failures = [c for c in checks if c["verdict"] == "contradicted"]
            if hard_failures:
                score = min(score, 0.25)
            elif contradictions:
                score = min(score, 0.35)
            verdict = str(item.get("verdict") or "").strip().lower()
            if verdict not in ("supported", "weak", "contradicted"):
                if hard_failures or contradictions:
                    verdict = "contradicted"
                elif score >= self.code_support_threshold:
                    verdict = "supported"
                else:
                    verdict = "weak"
            branch.claim_checks = checks
            branch.code_support_score = score
            branch.code_verdict = verdict
            branch.verification_notes = str(item.get("notes") or "").strip()
            results.append(
                {
                    "id": bid,
                    "claim_checks": checks,
                    "code_support_score": score,
                    "contradictions": contradictions,
                    "verdict": verdict,
                    "notes": branch.verification_notes,
                }
            )
        return results

    @staticmethod
    def _verification_changed(
        before: Sequence[Dict[str, Any]],
        after: Sequence[Dict[str, Any]],
    ) -> bool:
        def key(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
            return {
                str(r.get("id")): [
                    (
                        str(c.get("claim") or "").strip().lower(),
                        str(c.get("verdict") or ""),
                        tuple(c.get("evidence_lines") or []),
                    )
                    for c in (r.get("claim_checks") or [])
                ]
                for r in rows
            }
        return key(before) != key(after)

    def _claim_reflection_prompt(
        self,
        branches: Sequence[Branch],
        previous: Sequence[Dict[str, Any]],
        function_code: str,
        critical_lines: str,
    ) -> str:
        return (
            "You are your own adversarial reviewer. You previously produced the verification below. Your task is "
            "not to defend it and not to confirm it. Your task is to try to falsify each claim, then keep only the "
            "verdicts that survive. A skeptical reviewer will open the source file and check your evidence, so an "
            "answer that merely repeats the previous verdicts has failed this step.\n\n"
            "Follow these steps in order. Do not skip a step, and do not merge steps.\n\n"
            "STEP 1 - Make each claim falsifiable.\n"
            "For every claim, state silently what observable code fact must be true for it to hold. If the claim is "
            "really an interpretation ('this allows an attacker to...'), separate the observable fact from the "
            "interpretation. The interpretation is judged only after the observable fact is checked.\n\n"
            "STEP 2 - Locate verbatim evidence.\n"
            "For every claim previously marked supported or uncertain, find the exact source line that shows the "
            "claimed fact. Quote that line verbatim and give its line number. A line number without a verbatim quote "
            "is not evidence. If the only matching text is in the claim itself, or in a CVE-style restatement, there "
            "is no evidence.\n\n"
            "STEP 3 - Attack the claim.\n"
            "For each claim, test these failure modes explicitly:\n"
            "- Call site vs callee body: a call such as foo(x) does not prove what foo does with x. If the claimed "
            "effect happens inside foo and foo is not shown, the claim is out_of_scope, not supported.\n"
            "- Nearby vs exact line: the quoted line must contain the claimed variable, field, operation, or missing "
            "check. A nearby assignment or a similar-looking line does not count.\n"
            "- Missing check: the check must be absent from the shown path that reaches the sink. Absence from the "
            "critical-line subset alone does not prove a missing check.\n"
            "- Data-flow direction: the tainted value must actually flow from the source to the sink in the shown "
            "assignments, arguments, or returns. A plausible path is not a shown path.\n"
            "- Strength: 'may', 'can', 'if', and 'under attacker control' must not become categorical statements.\n"
            "- Anchor echo: if a claim restates the CVE, that is identity information, not independent code evidence. "
            "It cannot by itself make a visible code fact supported.\n\n"
            "STEP 4 - Re-decide each verdict.\n"
            "- supported: steps 2 and 3 both pass, with a verbatim line that directly shows the fact.\n"
            "- uncertain: the shown code is compatible with the claim but does not establish it.\n"
            "- out_of_scope: the claim depends on a callee, caller, framework, or runtime state that is not shown. "
            "Treat this as conditional rather than failed.\n"
            "- unsupported: the claim asserts a fact that should be visible in the shown function, but no shown line "
            "establishes it.\n"
            "- contradicted: a verbatim shown line contradicts the claim.\n"
            "Downgrade any supported claim that fails step 3. Do not upgrade a claim just because it sounds plausible. "
            "Do not add new claims, do not delete claims, and do not rewrite claim text; the ledger must stay aligned "
            "with the branch.\n\n"
            "STEP 5 - Report the weakest claim.\n"
            "Identify the single claim that came closest to failing and explain why it survived. If any verdict "
            "changed, list each change as old -> new with the exact reason. If no verdict changed, say so only after "
            "this weakest-claim analysis. 'All claims look fine' without naming a weakest claim is not an acceptable "
            "reflection.\n\n"
            f"Source Code:\n{function_code}\n\nCritical Lines (source text):\n{critical_lines or '(none)'}\n\n"
            f"Previous Verification:\n{json.dumps(list(previous), ensure_ascii=False, indent=2)}\n\n"
            "Return the corrected full verification in the same JSON shape, plus the reflection fields:\n"
            '{"branches": [{"id": "Branch1", '
            '"claim_checks": [{"claim": "...", "claim_type": "root_cause|trigger|data_flow|missing_check|impact|other", '
            '"verdict": "supported|uncertain|out_of_scope|unsupported|contradicted", '
            '"evidence_lines": [12], "reason": "..."}], '
            '"code_support_score": 0.0, "contradictions": ["..."], '
            '"verdict": "supported|weak|contradicted", "notes": "..."}], '
            '"weakest_claim": "the claim that came closest to failing and why it survived", '
            '"changed_verdicts": [{"claim": "...", "from": "...", "to": "...", "reason": "..."}], '
            '"reflection": "one short paragraph on what you corrected and why"}'
        )

    def step_4a_code_verification(
        self,
        branches: Sequence[Branch],
        function_code: str,
        critical_lines: str,
    ) -> Dict[str, Any]:
        data, raw = self._call_json(
            self._structural_verification_prompt(branches, function_code, critical_lines),
            stage="structural code verification",
        )
        results = self._apply_code_verification_data(data, branches, function_code)
        reflection_raw = ""
        reflection_notes: List[str] = []
        reflection_weakest: List[str] = []
        reflection_changes: List[Dict[str, Any]] = []
        if self.enable_claim_reflection and results:
            for round_idx in range(self.reflection_rounds):
                ref_data, ref_raw = self._call_json(
                    self._claim_reflection_prompt(branches, results, function_code, critical_lines),
                    stage=f"claim self-reflection {round_idx + 1}",
                )
                reflection_raw += ("\n" if reflection_raw else "") + ref_raw
                if isinstance(ref_data, dict):
                    note = str(ref_data.get("reflection") or "").strip()
                    if note:
                        reflection_notes.append(note)
                    weakest = str(ref_data.get("weakest_claim") or "").strip()
                    if weakest:
                        reflection_weakest.append(weakest)
                    changes = ref_data.get("changed_verdicts") or []
                    if isinstance(changes, dict):
                        changes = [changes]
                    if isinstance(changes, list):
                        for change in changes:
                            if isinstance(change, dict):
                                reflection_changes.append(change)
                new_results = self._apply_code_verification_data(ref_data, branches, function_code)
                if not new_results:
                    break
                changed = self._verification_changed(results, new_results)
                results = new_results
                if not changed:
                    break
        return {
            "raw": raw,
            "branches": results,
            "reason": str(data.get("reason") or "").strip() if isinstance(data, dict) else "",
            "reflection_raw": reflection_raw,
            "reflection_notes": reflection_notes,
            "reflection_weakest_claims": reflection_weakest,
            "reflection_changed_verdicts": reflection_changes,
            "reflected": bool(reflection_raw),
        }

    def _cve_alignment_prompt(self, branches: Sequence[Branch], cve_description: str) -> str:
        return (
            "Compare each branch with the official CVE description. Judge the vulnerability identity on four "
            "independent dimensions: vulnerability_type, root_cause, trigger, and impact. Classify every "
            "dimension into exactly one of three states: matched, conflicting, or unestablished.\n\n"
            "- matched: the branch states something compatible with the CVE on this dimension. Wording does "
            "not need to be identical.\n"
            "- conflicting: the branch states something incompatible with the CVE on this dimension.\n"
            "- unestablished: the branch does not state enough about this dimension, or the CVE is too vague "
            "to decide. Do not mark a dimension conflicting merely because the CVE gives 'unspecified "
            "vectors', names only a component, or because the mechanism lies in a callee that is not shown.\n\n"
            "Impact compatibility: a specific mechanism such as a NULL pointer dereference, out-of-bounds "
            "access, or heap corruption matches a broader CVE impact such as a crash, denial of service, or "
            "code execution when that mechanism is a recognized cause of the stated impact. Only mark impact "
            "conflicting when the branch's impact cannot lead to the CVE's impact.\n\n"
            "Root cause and trigger: when the CVE gives only a component or 'unspecified vectors', a "
            "code-grounded root cause or trigger must be marked unestablished, not conflicting, unless it "
            "contradicts a fact the CVE actually states.\n\n"
            "Set consistent=true only when vulnerability_type is matched and impact is matched or "
            "unestablished, and no critical dimension (vulnerability_type, root_cause, impact) is conflicting. "
            "A branch that only shares a trigger or a code location is NOT the same vulnerability. Do not "
            "reward a branch merely for adding more code-level detail; that detail is checked separately "
            "against the source code.\n\n"
            f"Official CVE Description:\n{cve_description}\n\n"
            f"Branches:\n{self._format_branches(branches)}\n\n"
            "Return a JSON object with this schema:\n"
            '{"branches": [{"id": "Branch1", "cve_alignment_score": 0.0, "consistent": true, '
            '"matched_dimensions": ["vulnerability_type", "root_cause"], '
            '"conflicting_dimensions": [], "unestablished_dimensions": ["impact"], "reason": "..."}], '
            '"reason": "..."}\n'
            "Each dimension must appear in at most one of the three lists. Use the same branch ids as the "
            "input. Scores must be between 0 and 1. All dimension names must be lowercase and selected from "
            "vulnerability_type, root_cause, trigger, impact."
        )

    def step_4b_cve_alignment(self, branches: Sequence[Branch], cve_description: str) -> Dict[str, Any]:
        if not cve_description.strip():
            for branch in branches:
                branch.cve_alignment_score = None
                branch.cve_verdict = "not_available"
            return {"raw": "", "branches": [], "reason": "CVE description unavailable", "available": False}

        data, raw = self._call_json(
            self._cve_alignment_prompt(branches, cve_description),
            stage="CVE alignment verification",
        )
        items: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            raw_items = data.get("branches") or data.get("ranked_branches") or []
            if isinstance(raw_items, list):
                items = [x for x in raw_items if isinstance(x, dict)]

        by_id = {b.branch_id: b for b in branches}
        results: List[Dict[str, Any]] = []
        for item in items:
            bid = str(item.get("id") or item.get("branch_id") or "").strip()
            m = re.search(r"(\d+)", bid)
            if m:
                bid = f"Branch{int(m.group(1))}"
            branch = by_id.get(bid)
            if branch is None:
                continue
            score = _coerce_score(item.get("cve_alignment_score") or item.get("score"))
            consistent = _coerce_bool(item.get("consistent"))
            if score is None and consistent is not None:
                score = 0.8 if consistent else 0.2
            if score is None:
                score = 0.0

            def _dimensions(raw: Any) -> List[str]:
                if isinstance(raw, str):
                    raw = re.split(r"[,;]+", raw)
                if not isinstance(raw, list):
                    return []
                valid = {"vulnerability_type", "root_cause", "trigger", "impact"}
                out: List[str] = []
                for value in raw:
                    text = str(value).strip().lower().replace(" ", "_")
                    if text in valid and text not in out:
                        out.append(text)
                return out

            matched = _dimensions(item.get("matched_dimensions"))
            structured = (
                "conflicting_dimensions" in item
                or "unestablished_dimensions" in item
            )
            if structured:
                conflicting = _dimensions(item.get("conflicting_dimensions"))
                unestablished = _dimensions(item.get("unestablished_dimensions"))
            else:
                conflicting = _dimensions(
                    item.get("mismatched_dimensions") or item.get("conflicting_dimensions")
                )
                unestablished = []
            matched_set = set(matched)
            conflicting_set = set(conflicting)
            unestablished_set = set(unestablished)
            type_ok = "vulnerability_type" in matched_set
            type_only_identity = type_ok and not (
                {"root_cause", "trigger", "impact"} & matched_set
            )
            required_match = type_ok and (
                bool({"root_cause", "impact"} & matched_set) or type_only_identity
            )
            impact_ok = (
                "impact" in matched_set or "impact" in unestablished_set
            )
            critical_conflict = bool(
                {"vulnerability_type", "root_cause", "impact"} & conflicting_set
            )
            if not matched and not conflicting and not unestablished:
                identity_ok = bool(consistent) and score >= max(self.cve_alignment_threshold, 0.70)
            else:
                required_score = self.cve_alignment_threshold
                if type_only_identity:
                    required_score = self.type_only_cve_threshold
                identity_ok = (
                    bool(consistent)
                    and score >= required_score
                    and required_match
                    and type_ok
                    and impact_ok
                    and not critical_conflict
                )
            branch.cve_alignment_score = score
            branch.cve_identity_ok = identity_ok
            branch.cve_mismatched_dimensions = conflicting
            branch.cve_unestablished_dimensions = unestablished
            branch.cve_type_only_identity = type_only_identity
            branch.cve_verdict = "consistent" if identity_ok else "inconsistent"
            results.append(
                {
                    "id": bid,
                    "cve_alignment_score": score,
                    "consistent": identity_ok,
                    "verifier_consistent": bool(consistent) if consistent is not None else None,
                    "matched_dimensions": matched,
                    "conflicting_dimensions": conflicting,
                    "unestablished_dimensions": unestablished,
                    "mismatched_dimensions": conflicting,
                    "identity_ok": identity_ok,
                    "required_match": required_match,
                    "type_only_identity": type_only_identity,
                    "impact_ok": impact_ok,
                    "critical_conflict": critical_conflict,
                    "reason": str(item.get("reason") or "").strip(),
                }
            )
        return {
            "raw": raw,
            "branches": results,
            "reason": str(data.get("reason") or "").strip() if isinstance(data, dict) else "",
            "available": True,
        }

    def _repair_prompt(
        self,
        branch: Branch,
        function_code: str,
        critical_lines: str,
    ) -> str:
        supported = [
            c for c in branch.claim_checks if c.get("verdict") in ("supported", "uncertain")
        ]
        return (
            "Perform a counterfactual repair check for this candidate vulnerability interpretation. Propose the "
            "smallest plausible source-code change that would remove the claimed root cause while preserving normal "
            "functionality. If the interpretation is too vague, unsupported, or would require an implausible rewrite, "
            "say so and mark repair_consistent=false. A correct root cause should usually admit a small, local fix at "
            "or near the claimed trigger.\n\n"
            f"Source Code:\n{function_code}\n\nCritical Lines (source text):\n{critical_lines}\n\n"
            f"Candidate Branch:\n{branch.describe(include_status=True)}\n\n"
            f"Claim Checks:\n{json.dumps(supported, ensure_ascii=False)}\n\n"
            "Return a JSON object with this schema:\n"
            '{"patch": "...", "repair_consistent": true, "repair_score": 0.0, '
            '"reason": "...", "patched_claims": ["..."]}\n'
            "repair_score must be between 0 and 1. If no small plausible patch exists, use repair_consistent=false "
            "and explain why."
        )

    def step_4c_repair_check(
        self,
        branches: Sequence[Branch],
        function_code: str,
        critical_lines: str,
    ) -> Dict[str, Any]:
        ranked = sorted(
            [b for b in branches if b.code_support_score is not None],
            key=lambda b: b.code_support_score or 0.0,
            reverse=True,
        )
        candidates = ranked[: self.max_repair_candidates]
        results: List[Dict[str, Any]] = []
        raws: List[str] = []
        for branch in candidates:
            try:
                data, raw = self._call_json(
                    self._repair_prompt(branch, function_code, critical_lines),
                    stage="counterfactual repair check",
                )
            except Exception as exc:
                results.append({"id": branch.branch_id, "error": str(exc)})
                continue
            raws.append(raw)
            score = _coerce_score(data.get("repair_score")) if isinstance(data, dict) else None
            consistent = _coerce_bool(data.get("repair_consistent")) if isinstance(data, dict) else None
            if score is None:
                score = 0.7 if consistent else 0.2 if consistent is False else 0.0
            branch.repair_score = score
            branch.repair_consistent = consistent
            branch.repair_patch = str(data.get("patch") or "").strip() if isinstance(data, dict) else ""
            results.append(
                {
                    "id": branch.branch_id,
                    "repair_score": score,
                    "repair_consistent": consistent,
                    "patch": branch.repair_patch,
                    "reason": str(data.get("reason") or "").strip() if isinstance(data, dict) else "",
                }
            )
        return {"raw": "\n\n".join(raws), "branches": results}

    def _score_and_select(
        self,
        branches: Sequence[Branch],
        cve_available: bool,
    ) -> Dict[str, Any]:
        ranked: List[Dict[str, Any]] = []
        for branch in branches:
            code = branch.code_support_score
            cve = branch.cve_alignment_score if cve_available else None
            repair = branch.repair_score if branch.repair_consistent is not None else None
            weights: List[Tuple[float, Optional[float]]] = [(0.55, code)]
            if cve_available:
                weights.append((0.30, cve))
            if repair is not None:
                weights.append((0.15, repair))
            total_weight = sum(w for w, value in weights if value is not None)
            if total_weight <= 0:
                final_score = 0.0
            else:
                final_score = sum(w * value for w, value in weights if value is not None) / total_weight
            branch.final_score = final_score
            hard_fail = any(c.get("verdict") == "contradicted" for c in branch.claim_checks)
            code_ok = code is not None and code >= self.code_support_threshold and not hard_fail
            cve_ok = (not cve_available) or (
                cve is not None
                and branch.cve_verdict == "consistent"
                and branch.cve_identity_ok is not False
                and cve >= (
                    self.type_only_cve_threshold
                    if branch.cve_type_only_identity
                    else self.cve_alignment_threshold
                )
            )
            repair_ok = (
                not self.enable_repair_check
                or branch.repair_consistent is None
                or bool(branch.repair_consistent)
                or (code_ok and cve_ok)
            )
            if (
                self.enable_repair_check
                and bool(branch.repair_consistent)
                and repair is not None
                and repair < self.repair_threshold
                and not (code_ok and cve_ok)
            ):
                repair_ok = False
            verified = (
                code_ok
                and cve_ok
                and repair_ok
                and final_score >= self.final_threshold
            )
            ranked.append(
                {
                    "id": branch.branch_id,
                    "final_score": round(final_score, 4),
                    "code_support_score": code,
                    "cve_alignment_score": cve,
                    "cve_identity_ok": branch.cve_identity_ok,
                    "cve_mismatched_dimensions": branch.cve_mismatched_dimensions,
                    "cve_unestablished_dimensions": branch.cve_unestablished_dimensions,
                    "cve_type_only_identity": branch.cve_type_only_identity,
                    "repair_score": repair,
                    "repair_consistent": branch.repair_consistent,
                    "code_verdict": branch.code_verdict,
                    "cve_verdict": branch.cve_verdict,
                    "root_cause_scope": self._root_cause_scope(branch),
                    "verified": verified,
                }
            )
        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        verified_items = [x for x in ranked if x["verified"]]
        best_verified = verified_items[0] if verified_items else None
        best_fallback = ranked[0] if ranked else None
        by_id = {b.branch_id: b for b in branches}
        verified_branch = by_id.get(best_verified["id"]) if best_verified else None
        fallback_branch = by_id.get(best_fallback["id"]) if best_fallback else None
        if verified_branch is not None:
            verified_branch.status = "verified"
        return {
            "ranked_branches": ranked,
            "verified_branch_id": verified_branch.branch_id if verified_branch else None,
            "verified_branch": verified_branch,
            "verified_branches": [
                by_id[item["id"]]
                for item in verified_items
                if item["id"] in by_id
            ],
            "verification_result": "consistent" if verified_branch is not None else "reconsider",
            "fallback_candidate": best_fallback,
            "fallback_branch": fallback_branch,
            "basis": "dual" if cve_available else "structural",
        }

    def step_4_branch_verification(self, branches: Sequence[Branch], cve_description: str) -> Dict[str, Any]:
        data, raw = self._call_json(self._verification_prompt(branches, cve_description), stage="branch verification")
        ranked: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            raw_ranked = data.get("ranked_branches") or []
            if isinstance(raw_ranked, list):
                for item in raw_ranked:
                    if not isinstance(item, dict):
                        continue
                    bid = str(item.get("id") or item.get("branch_id") or "").strip()
                    m = re.search(r"(\d+)$", bid)
                    if m:
                        bid = f"Branch{int(m.group(1))}"
                    ranked.append({"id": bid, "score": _coerce_score(item.get("score")), "consistent": bool(item.get("consistent", False)), "reason": str(item.get("reason") or "").strip()})

        branch_by_id = {b.branch_id: b for b in branches}
        for item in ranked:
            branch = branch_by_id.get(item["id"])
            if branch is not None:
                if item["score"] is not None:
                    branch.score = item["score"]
                if item["reason"]:
                    branch.reason = item["reason"]

        verified_id = None
        if isinstance(data, dict):
            value = data.get("verified_branch_id") or data.get("verified_branch")
            if value and str(value).strip().lower() not in ("none", "null", "n/a", "no", ""):
                m = re.search(r"(\d+)", str(value))
                if m:
                    verified_id = f"Branch{int(m.group(1))}"
        result = str(data.get("verification_result") or "").strip().lower() if isinstance(data, dict) else ""
        consistent = [x for x in ranked if x["consistent"]]
        consistent.sort(key=lambda x: (x["score"] if x["score"] is not None else -1), reverse=True)
        ranked_sorted = sorted(
            ranked,
            key=lambda x: (x["score"] if x["score"] is not None else -1),
            reverse=True,
        )
        fallback_candidate = consistent[0] if consistent else (ranked_sorted[0] if ranked_sorted else None)

        if result in ("", "consistent"):
            if verified_id not in branch_by_id:
                if consistent:
                    verified_id = consistent[0]["id"]
                elif ranked_sorted:
                    verified_id = ranked_sorted[0]["id"]
                else:
                    verified_id = None
            if verified_id is None:
                result = "reconsider"
            else:
                result = "consistent"
        else:
            verified_id = None
            result = "reconsider"

        verified_branch: Optional[Branch] = None
        if verified_id and verified_id in branch_by_id:
            verified_branch = branch_by_id[verified_id]
            verified_branch.status = "verified"
        fallback_branch: Optional[Branch] = None
        if fallback_candidate is not None:
            fallback_branch = branch_by_id.get(fallback_candidate["id"])
        return {
            "raw": raw,
            "ranked_branches": ranked,
            "verified_branch_id": verified_id,
            "verified_branch": verified_branch,
            "verification_result": result,
            "fallback_candidate": fallback_candidate,
            "fallback_branch": fallback_branch,
            "reason": str(data.get("reason") or "").strip() if isinstance(data, dict) else "",
            "basis": "cve" if cve_description.strip() else "structural",
        }

    def _reconsideration_prompt(self, function_code: str, critical_lines: str, refined: Sequence[Branch], archived: Sequence[Branch], cve_description: str, previous: Dict[str, Any]) -> str:
        pruned_text = self._format_branches([b for b in archived if b.status == "pruned"], include_status=True)
        previous_ids = [b.branch_id for b in refined]
        return (
            "The current retained branches were not sufficiently supported during verification. Return to the complete "
            "candidate pool and perform identity-preserving reconciliation, not free-form guessing. You may revive a "
            "previously pruned branch with a corrected explanation, or propose a genuinely new interpretation.\n\n"
            "Use the CVE only as an identity anchor: it tells you the vulnerability type, the affected routine, the "
            "trigger class, and the impact. It does not tell you the code-level facts of this function. Build the "
            "reconciled branch from two explicitly separated evidence sets:\n"
            "1) visible facts: assignments, guards, calls, arguments, and returns that are directly present in the "
            "supplied function; each must cite a line number;\n"
            "2) unshown mechanism: behavior of a callee, caller, or framework that is not in the supplied function. "
            "Include this only when the CVE or the shown call site implies it, and mark its evidence as 'not shown'. "
            "Do not invent visible code facts.\n\n"
            "A near-miss branch that has the right call site but the wrong root-cause wording must be repaired, not "
            "discarded: keep its visible facts, add the CVE's identity for the unshown part, and state the root cause "
            "so that it covers both. Do not simply paraphrase the CVE; the branch must still be anchored in a "
            "concrete code path.\n\n"
            f"Code:\n{function_code}\n\nCritical Lines (source text):\n{critical_lines}\n\n"
            f"Currently Retained Branches:\n{self._format_branches(refined, include_status=True)}\n\n"
            f"Previously Pruned Branches:\n{pruned_text}\n\n"
            f"Official CVE Description:\n{cve_description or '(unavailable)'}\n\n"
            f"Previous Verification Result:\n{previous.get('verification_result')} -- {previous.get('reason', '')}\n\n"
            "Return a JSON object with this schema:\n"
            '{"branches": [{"id": "Branch1", "vulnerability_type": "...", "trigger_point": "...", '
            '"causal_explanation": "...", "claims": [{"claim": "...", "claim_type": "...", '
            '"evidence_lines": [12], "evidence": "...", "confidence": 0.0}], "reason": "..."}], '
            '"reconsideration": "..."}\n'
            f"Return at least one branch and never exceed {self.max_branches} branches in total. Current retained ids: "
            f"{', '.join(previous_ids)}."
        )

    def _reconsider(self, function_code: str, critical_lines: str, refined: Sequence[Branch], archived: Sequence[Branch], cve_description: str, previous: Dict[str, Any]) -> Tuple[List[Branch], str, Dict[str, Any]]:
        data, raw = self._call_json(self._reconsideration_prompt(function_code, critical_lines, refined, archived, cve_description, previous), stage="branch reconsideration")
        proposals = self._parse_branch_list(data, fallback=list(refined) + list(archived))
        if not proposals:
            raise ValueError("Reconsideration produced no alternative branches.")

        pool = list(refined) + [b for b in archived if b.status == "pruned"]
        by_id = {b.branch_id: b for b in pool}
        revived: List[Branch] = []
        for prop in proposals:
            base = by_id.get(prop.branch_id)
            if base is None:
                revived.append(prop)
                continue
            revived.append(
                replace(
                    base,
                    vulnerability_type=prop.vulnerability_type or base.vulnerability_type,
                    trigger_point=prop.trigger_point or base.trigger_point,
                    causal_explanation=prop.causal_explanation or base.causal_explanation,
                    reason=prop.reason or base.reason,
                    claims=prop.claims or base.claims,
                    claim_checks=[],
                    code_support_score=None,
                    cve_alignment_score=None,
                    repair_score=None,
                    final_score=None,
                    code_verdict="",
                    cve_verdict="",
                    cve_identity_ok=None,
                    cve_mismatched_dimensions=[],
                    cve_unestablished_dimensions=[],
                    cve_type_only_identity=False,
                    repair_consistent=None,
                    repair_patch="",
                    verification_notes="",
                    status="reconsidered",
                    pruned_at=None,
                    score=None,
                )
            )

        mentioned = {b.branch_id for b in revived}
        for branch in refined:
            if branch.branch_id not in mentioned:
                branch.status = "refined"
                revived.append(branch)
        revived = _dedupe_branches(revived)
        if len(revived) > self.max_branches:
            revived = revived[: self.max_branches]
        info = {"raw": raw, "reconsideration": data.get("reconsideration", "") if isinstance(data, dict) else "", "proposed_ids": [b.branch_id for b in proposals]}
        return revived, raw, info

    @staticmethod
    def _root_cause_scope(branch: Optional[Branch]) -> str:
        if branch is None:
            return "unestablished"
        checks = branch.claim_checks or []
        root_checks = [
            c for c in checks
            if str(c.get("claim_type") or "").strip().lower() == "root_cause"
        ]
        if any(str(c.get("verdict") or "") == "supported" for c in root_checks):
            return "visible"
        if any(str(c.get("verdict") or "") == "out_of_scope" for c in root_checks):
            return "callee"
        return "unestablished"

    def _coverage_claims(
        self,
        branches: Sequence[Branch],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen = set()
        for branch in branches:
            if branch is None:
                continue
            for claim in self._supported_claims(branch):
                text = str(claim.get("claim") or "").strip()
                if not text:
                    continue
                key = (
                    re.sub(r"\s+", " ", text).strip().lower(),
                    str(claim.get("claim_type") or "").strip().lower(),
                    str(claim.get("verdict") or "").strip().lower(),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "branch": branch.branch_id,
                        "claim": text,
                        "claim_type": claim.get("claim_type"),
                        "verdict": claim.get("verdict"),
                        "evidence_lines": claim.get("evidence_lines") or [],
                    }
                )
        return rows

    @staticmethod
    def _supported_claims(branch: Optional[Branch]) -> List[Dict[str, Any]]:
        if branch is None:
            return []
        checks = branch.claim_checks or []
        usable = [
            c for c in checks
            if c.get("verdict") in ("supported", "uncertain", "out_of_scope")
        ]
        if usable:
            return usable
        return [
            c for c in (branch.claims or [])
            if (c.get("confidence") is None or (c.get("confidence") or 0.0) >= 0.5)
        ]

    def step_5_generate_description(
        self,
        verified_branch: Optional[Branch],
        function_code: str = "",
        fallback_branches: Sequence[Branch] = (),
        critical_lines: str = "",
        cve_description: str = "",
        supporting_branches: Sequence[Branch] = (),
    ) -> str:
        if verified_branch is not None:
            basis_branch = verified_branch
            basis_label = "Selected branch"
        elif fallback_branches:
            basis_branch = max(
                fallback_branches,
                key=lambda b: (
                    1 if b.cve_identity_ok is True else 0,
                    b.final_score if b.final_score is not None else (b.code_support_score or 0.0),
                ),
            )
            basis_label = "Best-supported branch"
        else:
            basis_branch = None
            basis_label = "No branch"

        identity_anchored = (
            verified_branch is None and bool(cve_description.strip())
        )

        coverage_pool: List[Branch] = []
        for candidate in ([verified_branch] if verified_branch is not None else []) + list(supporting_branches):
            if candidate is not None and candidate.branch_id not in {b.branch_id for b in coverage_pool}:
                coverage_pool.append(candidate)
        if not coverage_pool and basis_branch is not None:
            coverage_pool = [basis_branch]
        coverage_rows = self._coverage_claims(coverage_pool)
        claim_text = json.dumps(coverage_rows, ensure_ascii=False, indent=2) if coverage_rows else "(none)"
        basis = (
            f"{basis_label}:\n{basis_branch.describe(include_status=True) if basis_branch else '(none)'}\n\n"
            f"Accepted Claim Ledger from {len(coverage_pool)} branch(es) "
            f"(supported/uncertain = code facts; out_of_scope = conditional mechanism from code not shown):\n"
            f"{claim_text}\n\n"
        )
        coverage_block = (
            "Coverage and detail requirements. Write a detailed vulnerability report, not a short summary, "
            "and cover all three dimensions:\n"
            "(1) Identity and root cause: the vulnerability type, the root cause, and the affected code region.\n"
            "(2) Trigger and path: the source of attacker-controlled data, each visible transformation or check, "
            "the vulnerable operation, and the path from input to that operation; include visible call-site facts "
            "and, when necessary, the unshown callee path stated conditionally.\n"
            "(3) Impact: the security consequence.\n"
            "The accepted claim ledger is aggregated from every branch that passed verification. Treat supported and "
            "uncertain claims as code facts, and out_of_scope claims as conditional mechanisms. Do not omit an "
            "accepted claim that is compatible with the identity merely because it comes from another branch or is "
            "out_of_scope. Do not compress several accepted claims into one vague sentence: name the concrete "
            "functions, variables, fields, call sites, data flows, and missing or insufficient checks that the "
            "ledger supports. If two claims conflict, keep the one with the stronger evidence and silently drop the "
            "weaker one. Before returning, check that all three dimensions are present and that every accepted claim "
            "is either used in the report or deliberately excluded as irrelevant to this vulnerability. "
            "Prefer 250-400 words and two to four paragraphs when the ledger supports that much detail; do not "
            "invent facts merely to reach a length, and do not shorten a justified report for brevity. "
            "Do not write absolute line numbers such as 'L12' or 'line 17' in the final report; name the "
            "statement, call site, or code region instead.\n\n"
        )
        rsi_block = (
            "Before returning, silently run one self-improvement cycle. Do not print the draft or the critique.\n"
            "(1) Draft the most detailed report that the ledger and source support.\n"
            "(2) Critique it sentence by sentence: mark every sentence that adds a function, variable, field, line, "
            "data flow, missing check, or impact that is not in the ledger or directly visible in the source; mark "
            "every causal step stated more strongly than the ledger allows; mark every required component (root cause, "
            "affected region, trigger, path, impact) that is missing; and mark every accepted claim that was omitted.\n"
            "(3) Rewrite the report: delete or weaken unsupported sentences, restore missing components and accepted "
            "claims only when the ledger supports them, and keep the vulnerability identity unchanged. Do not compress "
            "the report; a longer justified report is preferred over a short summary. Do not include a CVE ID or "
            "advisory; state only the supported vulnerability facts.\n"
            "Return only the rewritten report.\n\n"
        )
        if identity_anchored:
            prompt = (
                coverage_block
                + rsi_block
                + "The selected branch did not pass the vulnerability-identity check. Write a detailed, "
                "self-contained vulnerability report whose vulnerability identity is the one stated in the "
                "official CVE description below. Use the claim ledger and source code for all compatible "
                "code-level facts; do not present an unrelated mechanism "
                "as the root cause. Name the visible function and the shown call sites, argument flows, "
                "data flows, and missing checks that lead into the vulnerable component. When the visible code "
                "does not expose the exact CVE mechanism, describe "
                "the affected region and the unshown callee or caller path at that level instead of inventing "
                "details. Include every compatible visible code fact (function name, call-site sequence, argument "
                "flow, missing comparison) rather than only the first one. Report only vulnerability facts; omit CVE IDs, branch "
                "labels, or verification scores. The report must state the "
                "root cause, the affected code region, the trigger condition and path, and the security impact at "
                "the level supported by the CVE and the visible code. If the code and CVE together support no "
                "vulnerability, output exactly: There is no vulnerability.\n\n"
                f"Official CVE Description:\n{cve_description}\n\n"
                f"{basis}Critical Lines (source text):\n{critical_lines or '(none)'}\n\n"
                f"Source Code:\n{function_code}\n\nReturn only the final vulnerability report (no headings or bullets)."
            )
        else:
            prompt = (
                coverage_block
                + rsi_block
                + "Write a detailed, self-contained vulnerability report using the claim ledger and the source code. "
                "Base every statement on the claim ledger and the source code; when the visible code does not establish "
                "a detail, describe only what is supported. Report only vulnerability facts; omit branch labels, verification scores, "
                "candidate selection, and CVE wording.\n\n"
                "Do not collapse the mechanism into a short summary. Explain the causal chain step by step: where the "
                "attacker-controlled data enters, how it is transformed or checked, which operation consumes it, what "
                "check is missing or insufficient, and why that leads to the impact. Name the concrete functions, "
                "variables, fields, call sites, and comparisons that the ledger supports.\n\n"
                "Use claims with verdict supported or uncertain as concrete code facts. Claims marked out_of_scope "
                "describe behavior in callee, caller, or framework code that is not in this function; use them only to "
                "state the vulnerability mechanism and impact, and do not invent additional details beyond them. "
                "Preserve the evidential strength of each claim: if the ledger says a fact is conditional, inferred, or "
                "depends on the callee ('can', 'may', 'if'), do not turn it into a categorical statement. Do not "
                "add a root cause, function name, variable name, or trigger that is absent from the claim ledger and not "
                "visible in the code. Every concrete statement must be traceable to a claim.\n\n"
                "The report must state the root cause, the affected code region, the trigger condition and path, and the "
                "security impact. If the evidence supports no vulnerability, output exactly: There is no vulnerability.\n\n"
                f"{basis}Critical Lines (source text):\n{critical_lines or '(none)'}\n\n"
                f"Source Code:\n{function_code}\n\nReturn only the final vulnerability report (no headings or bullets)."
            )
        return self._call_llm(prompt).strip()

    def _audit_prompt(
        self,
        description: str,
        function_code: str,
        critical_lines: str,
        supported_claims: Sequence[Dict[str, Any]],
        cve_description: str = "",
        identity_anchored: bool = False,
    ) -> str:
        anchor_rule = ""
        anchor_block = ""
        if identity_anchored:
            anchor_rule = (
                "This description is identity-anchored: its vulnerability identity comes from the reference "
                "description. Accept statements that restate that identity, and check that any added code-level "
                "detail is supported by the source or claim ledger. Evaluate mechanism statements outside the shown "
                "function by their stated condition rather than rejecting the description. "
            )
            anchor_block = f"Official CVE Description:\n{cve_description}\n\n"
        return (
            "Perform a sentence-level fact audit and repair of the generated vulnerability description. Follow "
            "these steps in order; do not skip a step and do not merge the audit with the repair.\n\n"
            "STEP 1 - Segment. Split the description into atomic factual statements. A statement that contains two "
            "facts is split into two. If a statement mixes a visible fact with an unshown consequence, split it as "
            "well: judge the visible part against the exact lines (for example, two calls with different arguments "
            "are visible) and judge the consequence separately as out_of_scope. Ignore rhetoric and keep only "
            "checkable content.\n\n"
            "STEP 2 - Trace. For every statement, find either a claim in the ledger or an exact source line that "
            "justifies it. Quote the line verbatim and give its number. If the only support is the CVE identity, say "
            "so explicitly; do not pretend it is a code fact.\n\n"
            "STEP 3 - Classify. Use supported only when the quote directly shows the statement. Use uncertain when "
            "the code is compatible but does not establish it. Use out_of_scope when it depends on a callee, caller, "
            "framework, or runtime state that is not shown; accept this as scope-limited support but check that the "
            "description preserves its conditional strength. Use unsupported when the statement asserts a visible "
            "code fact that no shown line establishes. Use contradicted only with a verbatim contradictory line.\n\n"
            "STEP 4 - Find the weakest sentences. Identify every unsupported or overstated statement, and also every "
            "invented variable, field, line, data flow, or impact. Be strict.\n\n"
            "STEP 5 - Repair. Rewrite the description using only justified statements. Remove unsupported details "
            "instead of guessing replacements. Preserve supported sentences, the vulnerability identity, and the "
            "security impact. Keep any component that is supported only at out_of_scope strength as a conditional "
            "statement ('can', 'may', 'if') rather than deleting it. Preserve every justified code-level detail and "
            "do not compress a detailed report into a short summary; trim only unsupported or redundant material. "
            "The revised description must still state the "
            "root cause, affected code region, trigger and path, and security impact; do not reduce it to a neutral code "
            "walkthrough. Delete any sentence about what the source does not show or what remains unverifiable, "
            "and rewrite or delete any reference to a CVE, CVE ID, or advisory. Keep the result a direct vulnerability "
            "report: no meta-commentary about missing source or unverifiable details, and no mention of the audit or "
            "branch labels. Do not put absolute line numbers such as 'L12' or 'line 17' in the revised "
            "description; name the statement, call site, or code region instead.\n\n"
            + anchor_rule
            + f"Generated Description:\n{description}\n\n"
            f"{anchor_block}"
            f"Claim Ledger:\n{json.dumps(list(supported_claims), ensure_ascii=False, indent=2)}\n\n"
            f"Critical Lines (source text):\n{critical_lines or '(none)'}\n\n"
            f"Source Code:\n{function_code}\n\n"
            "Return a JSON object with this schema:\n"
            '{"claims": [{"claim": "...", "verdict": "supported|uncertain|out_of_scope|unsupported|contradicted", '
            '"evidence_lines": [12], "reason": "..."}], '
            '"unsupported_claims": ["..."], "contradictions": ["..."], '
            '"audit_score": 0.0, "revised_description": "..."}\n'
            "audit_score must be between 0 and 1. revised_description must be a detailed vulnerability report of one "
            "to four paragraphs and must not mention the audit or branch labels."
        )

    def step_5b_audit_description(
        self,
        description: str,
        function_code: str,
        critical_lines: str,
        supported_claims: Sequence[Dict[str, Any]],
        cve_description: str = "",
        identity_anchored: bool = False,
    ) -> Dict[str, Any]:
        data, raw = self._call_json(
            self._audit_prompt(
                description,
                function_code,
                critical_lines,
                supported_claims,
                cve_description=cve_description,
                identity_anchored=identity_anchored,
            ),
            stage="description audit",
        )
        if not isinstance(data, dict):
            return {
                "raw": raw,
                "audit_score": None,
                "unsupported_claims": [],
                "contradictions": [],
                "revised_description": "",
                "claims": [],
            }
        claims: List[Dict[str, Any]] = []
        raw_claims = data.get("claims") or []
        if isinstance(raw_claims, list):
            for item in raw_claims:
                if not isinstance(item, dict):
                    continue
                verdict = str(item.get("verdict") or item.get("status") or "").strip().lower()
                if verdict not in ("supported", "uncertain", "out_of_scope", "unsupported", "contradicted"):
                    verdict = "uncertain"
                claims.append(
                    {
                        "claim": str(item.get("claim") or item.get("text") or "").strip(),
                        "verdict": verdict,
                        "evidence_lines": _coerce_line_numbers(item.get("evidence_lines") or item.get("lines")),
                        "reason": str(item.get("reason") or "").strip(),
                    }
                )
        unsupported = data.get("unsupported_claims") or []
        if isinstance(unsupported, str):
            unsupported = [unsupported]
        contradictions = data.get("contradictions") or []
        if isinstance(contradictions, str):
            contradictions = [contradictions]
        return {
            "raw": raw,
            "audit_score": _coerce_score(data.get("audit_score")),
            "unsupported_claims": [str(x) for x in unsupported if str(x).strip()],
            "contradictions": [str(x) for x in contradictions if str(x).strip()],
            "revised_description": str(data.get("revised_description") or "").strip(),
            "claims": claims,
        }

    def _verify_pool(
        self,
        branches: Sequence[Branch],
        function_code: str,
        critical_lines: str,
        cve_description: str,
    ) -> Dict[str, Any]:
        cve_available = bool(cve_description.strip())
        if self.verification_mode == "legacy":
            return self.step_4_branch_verification(branches, cve_description)

        code_result = self.step_4a_code_verification(branches, function_code, critical_lines)
        cve_result = self.step_4b_cve_alignment(branches, cve_description)
        repair_result: Dict[str, Any] = {"raw": "", "branches": []}
        if self.enable_repair_check:
            repair_result = self.step_4c_repair_check(branches, function_code, critical_lines)
        selection = self._score_and_select(branches, cve_available=cve_available)
        selection["code_verification"] = code_result
        selection["cve_verification"] = cve_result
        selection["repair_verification"] = repair_result
        return selection

    def generate_vulnerability_description(self, function_code: str, critical_lines: str, vulnerable_lines: str, line_relations: str, cve_description: str, sample_id: Optional[Any] = None) -> Dict[str, Any]:
        start = time.time()
        log_prefix = f"[sample {sample_id}] " if sample_id is not None else ""

        self._log(f"{log_prefix}Step 1: branch initialization")
        branches, step1_raw, expansion_info = self.step_1_branch_initialization(function_code, critical_lines)
        initial_count = len(branches)

        self._log(f"{log_prefix}Step 2: reflection and pruning ({len(branches)} candidate branches)")
        retained, pruned_2, step2_raw, step2_info = self.step_2_branch_reflection_and_pruning(function_code, branches)
        retained_before_diversity = len(retained)
        retained, pruned_after_diversity = self._ensure_verification_diversity(retained, pruned_2)
        diversity_revived_2 = [b.to_dict() for b in retained if b.status == "revived_for_verification"]

        self._log(f"{log_prefix}Step 3: structural refinement with source lines")
        refined, pruned_3, step3_raw, step3_info = self.step_3_branch_refinement(
            retained,
            vulnerable_lines,
            line_relations,
            critical_lines=critical_lines,
            function_code=function_code,
        )
        archive: List[Branch] = list(pruned_after_diversity) + list(pruned_3)
        refined_before_diversity = len(refined)
        refined, archive = self._ensure_verification_diversity(refined, archive)
        diversity_revived_3 = [b.to_dict() for b in refined if b.status == "revived_for_verification"]
        diversity_revived = diversity_revived_2 + diversity_revived_3

        self._log(f"{log_prefix}Step 4: dual-channel verification")
        verification = self._verify_pool(refined, function_code, critical_lines, cve_description)
        reconsideration_trace: List[Dict[str, Any]] = []
        verified_branch = verification.get("verified_branch")

        if verified_branch is None and self.max_reconsideration_rounds > 0:
            for round_idx in range(1, self.max_reconsideration_rounds + 1):
                self._log(f"{log_prefix}Step 4d: reconsideration round {round_idx}")
                try:
                    pool, rec_raw, rec_info = self._reconsider(
                        function_code, critical_lines, refined, archive, cve_description, verification
                    )
                except Exception as exc:
                    reconsideration_trace.append({"round": round_idx, "error": str(exc)})
                    break
                try:
                    refined, newly_pruned, ref_raw, ref_info = self.step_3_branch_refinement(
                        pool,
                        vulnerable_lines,
                        line_relations,
                        critical_lines=critical_lines,
                        function_code=function_code,
                    )
                except Exception as exc:
                    reconsideration_trace.append({"round": round_idx, "error": f"refinement: {exc}"})
                    break
                archive.extend(newly_pruned)
                refined, archive = self._ensure_verification_diversity(refined, archive)
                verification = self._verify_pool(refined, function_code, critical_lines, cve_description)
                reconsideration_trace.append({
                    "round": round_idx,
                    "reconsideration": rec_info,
                    "refinement": ref_info,
                    "verification": {
                        "verified_branch_id": verification.get("verified_branch_id"),
                        "verification_result": verification.get("verification_result"),
                        "ranked_branches": verification.get("ranked_branches"),
                        "fallback_candidate": verification.get("fallback_candidate"),
                        "code_verification": verification.get("code_verification"),
                        "cve_verification": verification.get("cve_verification"),
                        "repair_verification": verification.get("repair_verification"),
                    },
                    "raw_reconsideration": rec_raw,
                    "raw_refinement": ref_raw,
                })
                verified_branch = verification.get("verified_branch")
                if verified_branch is not None:
                    break

        self._log(f"{log_prefix}Step 5: description synthesis")
        fallback_branch = verification.get("fallback_branch")
        fallback_pool: List[Branch] = []
        if fallback_branch is not None:
            fallback_pool.append(fallback_branch)
        fallback_pool.extend(b for b in refined if b.branch_id not in {x.branch_id for x in fallback_pool})
        if verified_branch is None and not fallback_pool:
            fallback_pool = list(retained)
        fallback_identity_anchored = verified_branch is None and bool(cve_description.strip())
        supporting_branches = verification.get("verified_branches") or []
        final_description = self.step_5_generate_description(
            verified_branch,
            function_code=function_code,
            fallback_branches=fallback_pool,
            critical_lines=critical_lines,
            cve_description=cve_description,
            supporting_branches=supporting_branches,
        )
        if not final_description:
            if verified_branch is not None:
                final_description = verified_branch.causal_explanation
            elif fallback_pool:
                final_description = fallback_pool[0].causal_explanation
            else:
                final_description = "There is no vulnerability."

        description_status = (
            "verified"
            if verified_branch is not None
            else ("cve_anchored_fallback" if fallback_identity_anchored else "fallback")
        )
        audit_result: Dict[str, Any] = {}
        selected_branch = verified_branch or fallback_branch
        if (
            self.enable_description_audit
            and final_description.strip()
            and final_description.strip() != "There is no vulnerability."
        ):
            self._log(f"{log_prefix}Step 5b: description audit")
            try:
                audit_result = self.step_5b_audit_description(
                    final_description,
                    function_code,
                    critical_lines,
                    self._supported_claims(selected_branch),
                    cve_description=cve_description,
                    identity_anchored=fallback_identity_anchored,
                )
                audit_score = audit_result.get("audit_score")
                has_problem = bool(
                    audit_result.get("unsupported_claims")
                    or audit_result.get("contradictions")
                    or (audit_score is not None and audit_score < self.audit_threshold)
                )
                if has_problem and audit_result.get("revised_description"):
                    final_description = audit_result["revised_description"]
                    description_status = (
                        "audited_verified"
                        if verified_branch is not None
                        else (
                            "audited_cve_anchored_fallback"
                            if fallback_identity_anchored
                            else "audited_fallback"
                        )
                    )
                elif has_problem:
                    description_status = (
                        "audit_failed_verified"
                        if verified_branch is not None
                        else (
                            "audit_failed_cve_anchored_fallback"
                            if fallback_identity_anchored
                            else "audit_failed_fallback"
                        )
                    )
            except Exception as exc:
                audit_result = {"error": str(exc)}
                description_status = "audit_error"

        final_description = self._strip_line_number_annotations(final_description)
        selected_for_scores = verified_branch or fallback_branch
        root_cause_scope = self._root_cause_scope(selected_for_scores)
        return {
            "sample_id": sample_id,
            "num_branches_initial": initial_count,
            "num_branches_retained": retained_before_diversity,
            "num_branches_after_diversity": len(retained),
            "num_branches_refined": len(refined),
            "num_branches_refined_before_diversity": refined_before_diversity,
            "branches_initial": [b.to_dict() for b in branches],
            "branches_after_reflection": [b.to_dict() for b in retained],
            "branches_after_refinement": [b.to_dict() for b in refined],
            "branches_pruned": [b.to_dict() for b in archive],
            "branches_revived_for_diversity": diversity_revived,
            "verified_branch_id": verification.get("verified_branch_id"),
            "verified_branch_ids": [b.branch_id for b in supporting_branches],
            "step_5_coverage_claim_count": len(
                self._coverage_claims(
                    supporting_branches
                    or ([selected_for_scores] if selected_for_scores is not None else [])
                )
            ),
            "verification_result": verification.get("verification_result"),
            "verification_basis": verification.get("basis"),
            "verification_mode": self.verification_mode,
            "verification_fallback_candidate": verification.get("fallback_candidate"),
            "verified_branch": verified_branch.to_dict() if verified_branch else None,
            "selected_branch": selected_for_scores.to_dict() if selected_for_scores else None,
            "code_support_score": selected_for_scores.code_support_score if selected_for_scores else None,
            "cve_alignment_score": selected_for_scores.cve_alignment_score if selected_for_scores else None,
            "repair_score": selected_for_scores.repair_score if selected_for_scores else None,
            "final_score": selected_for_scores.final_score if selected_for_scores else None,
            "root_cause_scope": root_cause_scope,
            "description_status": description_status,
            "final_description": final_description,
            "description_audit": audit_result,
            "elapsed_seconds": round(time.time() - start, 3),
            "trace": {
                "step_1_initialization_raw": step1_raw,
                "step_1_expansion": expansion_info,
                "step_2_reflection_raw": step2_raw,
                "step_2_info": step2_info,
                "step_2_diversity": {
                    "before": retained_before_diversity,
                    "after": len(retained),
                    "revived": diversity_revived_2,
                },
                "step_3_refinement_raw": step3_raw,
                "step_3_info": step3_info,
                "step_3_diversity": {
                    "before": refined_before_diversity,
                    "after": len(refined),
                    "revived": diversity_revived,
                },
                "step_4_verification_raw": verification.get("raw"),
                "step_4_code_verification": verification.get("code_verification"),
                "step_4_cve_verification": verification.get("cve_verification"),
                "step_4_repair_verification": verification.get("repair_verification"),
                "step_5_description_audit": audit_result,
                "step_5_identity_anchored": fallback_identity_anchored,
                "reconsideration": reconsideration_trace,
            },
        }


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}")
    return records


def _format_critical_lines(critical_lines: Any, function_code: str, explicit_records: Optional[str] = None) -> str:
    if explicit_records:
        return explicit_records
    if isinstance(critical_lines, str):
        return critical_lines
    if not critical_lines:
        return "(none)"
    critical_lines = list(critical_lines)[:20]
    code_lines = function_code.splitlines()
    out: List[str] = []
    for value in critical_lines:
        try:
            n = int(value)
        except (TypeError, ValueError):
            out.append(str(value))
            continue
        if 1 <= n <= len(code_lines):
            out.append(f"L{n}: {code_lines[n - 1].strip()}")
        else:
            out.append(f"L{n}: <line unavailable>")
    return "\n".join(out)


def _format_selected_lines(selected: Any, function_code: str) -> str:
    if isinstance(selected, str):
        return selected
    if not selected:
        return "(none)"
    code_lines = function_code.splitlines()
    out: List[str] = []
    for value in selected:
        try:
            n = int(value)
        except (TypeError, ValueError):
            out.append(str(value))
            continue
        if 1 <= n <= len(code_lines):
            out.append(f"L{n}: {code_lines[n - 1].strip()}")
        else:
            out.append(f"L{n}: <line unavailable>")
    return "\n".join(out) if out else "(none)"


def _format_relations(relations: Any) -> str:
    if relations is None:
        return "(none)"
    if isinstance(relations, str):
        return relations
    out: List[str] = []
    for rel in relations:
        if not isinstance(rel, dict):
            out.append(str(rel))
            continue
        src = rel.get("src", rel.get("source"))
        dst = rel.get("dst", rel.get("target"))
        weight = rel.get("weight")
        rtype = rel.get("type") or rel.get("relation") or rel.get("label")
        parts = [f"L{src} -> L{dst}"]
        if rtype:
            parts.append(f"({rtype})")
        if weight is not None:
            try:
                parts.append(f"weight={float(weight):.3f}")
            except (TypeError, ValueError):
                parts.append(f"weight={weight}")
        out.append(" ".join(parts))
    return "\n".join(out) if out else "(none)"


def _read_existing_ids(output_path: str, overwrite: bool) -> set:
    if overwrite or not os.path.exists(output_path):
        return set()
    done = set()
    with open(output_path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = record.get("sample_id", record.get("sample_idx"))
            if sid is not None and record.get("final_description"):
                done.add(str(sid))
    return done


def _run_one(generator: ToTVRGenerator, record: Dict[str, Any]) -> Dict[str, Any]:
    sample_id = record.get("sample_id", record.get("sample_idx", record.get("index")))
    function_code = record.get("function_code") or record.get("code") or record.get("func") or ""
    if not function_code.strip():
        raise ValueError(f"Sample {sample_id} has no function_code")
    critical_text = _format_critical_lines(record.get("critical_lines", []), function_code, record.get("critical_line_records"))
    selected = record.get("selected_vulnerable_lines")
    if selected is None:
        vulnerable_text = record.get("selected_vulnerable_lines_text") or record.get("vulnerable_lines", "")
    else:
        vulnerable_text = _format_selected_lines(selected, function_code)
    if isinstance(vulnerable_text, list):
        vulnerable_text = _format_selected_lines(vulnerable_text, function_code)
    elif not isinstance(vulnerable_text, str):
        vulnerable_text = str(vulnerable_text)
    relations_text = _format_relations(record.get("line_relations"))
    result = generator.generate_vulnerability_description(
        function_code=function_code,
        critical_lines=critical_text,
        vulnerable_lines=vulnerable_text or "(none)",
        line_relations=relations_text,
        cve_description=str(record.get("cve_description") or ""),
        sample_id=sample_id,
    )
    return {
        "sample_id": sample_id,
        "sample_idx": record.get("sample_idx", sample_id),
        "cve_description": record.get("cve_description", ""),
        "final_description": result["final_description"],
        "verified": result["verified_branch"] is not None,
        "verified_branch_id": result["verified_branch_id"],
        "verified_branch_ids": result.get("verified_branch_ids"),
        "step_5_coverage_claim_count": result.get("step_5_coverage_claim_count"),
        "verification_result": result["verification_result"],
        "verification_basis": result["verification_basis"],
        "verification_mode": result.get("verification_mode"),
        "verification_fallback_candidate": result["verification_fallback_candidate"],
        "code_support_score": result.get("code_support_score"),
        "cve_alignment_score": result.get("cve_alignment_score"),
        "repair_score": result.get("repair_score"),
        "final_score": result.get("final_score"),
        "root_cause_scope": result.get("root_cause_scope"),
        "description_status": result.get("description_status"),
        "description_audit": result.get("description_audit"),
        "selected_branch": result.get("selected_branch"),
        "num_branches_initial": result["num_branches_initial"],
        "num_branches_retained": result["num_branches_retained"],
        "num_branches_after_diversity": result.get("num_branches_after_diversity"),
        "num_branches_refined": result["num_branches_refined"],
        "num_branches_refined_before_diversity": result.get("num_branches_refined_before_diversity"),
        "branches_initial": result["branches_initial"],
        "branches_after_reflection": result["branches_after_reflection"],
        "branches_after_refinement": result["branches_after_refinement"],
        "branches_pruned": result["branches_pruned"],
        "branches_revived_for_diversity": result.get("branches_revived_for_diversity"),
        "verified_branch": result["verified_branch"],
        "elapsed_seconds": result["elapsed_seconds"],
        "trace": result["trace"],
    }


def _summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    ok = sum(1 for r in records if r.get("final_description") and not r.get("error"))
    errors = total - ok
    verified = sum(1 for r in records if r.get("verified"))
    counts = [r.get("num_branches_initial", 0) for r in records if not r.get("error")]
    status_counts: Dict[str, int] = {}
    for record in records:
        status = str(record.get("description_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    scope_counts: Dict[str, int] = {}
    verified_scope_counts: Dict[str, int] = {}
    citation_counts: Dict[str, int] = {}
    for record in records:
        scope = str(record.get("root_cause_scope") or "unknown")
        scope_counts[scope] = scope_counts.get(scope, 0) + 1
        if record.get("verified"):
            verified_scope_counts[scope] = verified_scope_counts.get(scope, 0) + 1
        selected = record.get("selected_branch") or {}
        for check in selected.get("claim_checks") or []:
            status = str(check.get("citation_status") or "not_checked")
            citation_counts[status] = citation_counts.get(status, 0) + 1

    def numeric_mean(key: str) -> Optional[float]:
        values = [
            float(r[key]) for r in records
            if isinstance(r.get(key), (int, float))
        ]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "total": total,
        "completed": ok,
        "errors": errors,
        "verified": verified,
        "verified_rate": round(verified / ok, 4) if ok else 0.0,
        "branch_count_min": min(counts) if counts else 0,
        "branch_count_max": max(counts) if counts else 0,
        "branch_count_mean": round(sum(counts) / len(counts), 3) if counts else 0.0,
        "description_status_counts": status_counts,
        "root_cause_scope_counts": scope_counts,
        "verified_root_cause_scope_counts": verified_scope_counts,
        "citation_status_counts": citation_counts,
        "code_support_mean": numeric_mean("code_support_score"),
        "cve_alignment_mean": numeric_mean("cve_alignment_score"),
        "repair_score_mean": numeric_mean("repair_score"),
        "final_score_mean": numeric_mean("final_score"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tree-of-Thought-guided Vulnerability Reasoning (ToT-VR)")
    parser.add_argument("--input", help="JSONL file with ToT inputs")
    parser.add_argument("--output", help="JSONL file for generated descriptions")
    parser.add_argument("--summary", help="Optional JSON file for run statistics")
    parser.add_argument("--model", default=os.environ.get("TOT_MODEL", "gpt-5"))
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--min_branches", type=int, default=2)
    parser.add_argument("--max_branches", type=int, default=6)
    parser.add_argument("--max_reconsideration_rounds", type=int, default=1)
    parser.add_argument(
        "--verification_mode",
        choices=["dual", "legacy"],
        default="dual",
        help="dual: code-support + CVE-alignment + repair check; legacy: original CVE-only verifier",
    )
    parser.add_argument(
        "--min_verification_branches",
        type=int,
        default=2,
        help="Minimum distinct mechanisms kept for verification (revives archived branches if needed)",
    )
    repair_group = parser.add_mutually_exclusive_group()
    repair_group.add_argument(
        "--enable_repair_check",
        dest="enable_repair_check",
        action="store_true",
        help="Enable the counterfactual repair check (default)",
    )
    repair_group.add_argument(
        "--disable_repair_check",
        dest="enable_repair_check",
        action="store_false",
        help="Disable the counterfactual repair check",
    )
    parser.set_defaults(enable_repair_check=True)
    reflection_group = parser.add_mutually_exclusive_group()
    reflection_group.add_argument(
        "--enable_claim_reflection",
        dest="enable_claim_reflection",
        action="store_true",
        help="Enable the self-critique pass over claim verdicts (default)",
    )
    reflection_group.add_argument(
        "--disable_claim_reflection",
        dest="enable_claim_reflection",
        action="store_false",
        help="Disable the self-critique pass over claim verdicts",
    )
    parser.set_defaults(enable_claim_reflection=True)
    parser.add_argument(
        "--reflection_rounds",
        type=int,
        default=1,
        help="Maximum self-critique rounds after the first code verification (default 1)",
    )
    audit_group = parser.add_mutually_exclusive_group()
    audit_group.add_argument(
        "--enable_description_audit",
        dest="enable_description_audit",
        action="store_true",
        help="Enable the post-synthesis fact audit (default)",
    )
    audit_group.add_argument(
        "--disable_description_audit",
        dest="enable_description_audit",
        action="store_false",
        help="Disable the post-synthesis fact audit",
    )
    parser.set_defaults(enable_description_audit=True)
    parser.add_argument("--code_support_threshold", type=float, default=0.65)
    parser.add_argument("--cve_alignment_threshold", type=float, default=0.60)
    parser.add_argument("--type_only_cve_threshold", type=float, default=0.35)
    parser.add_argument("--repair_threshold", type=float, default=0.30)
    parser.add_argument("--final_threshold", type=float, default=0.60)
    parser.add_argument("--audit_threshold", type=float, default=0.70)
    parser.add_argument("--max_repair_candidates", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_output_tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--sample_retries", type=int, default=2)
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Send the provider-specific thinking-disabled request field when supported",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    extra_body = {"thinking": {"type": "disabled"}} if args.disable_thinking else None

    if not args.input:
        api_key = os.environ.get(args.api_key_env)
        try:
            generator = ToTVRGenerator(
                api_key=api_key,
                model=args.model,
                base_url=args.base_url,
                min_branches=args.min_branches,
                max_branches=args.max_branches,
                max_reconsideration_rounds=args.max_reconsideration_rounds,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
                extra_body=extra_body,
                verbose=args.verbose,
                verification_mode=args.verification_mode,
                min_verification_branches=args.min_verification_branches,
                enable_repair_check=args.enable_repair_check,
                enable_claim_reflection=args.enable_claim_reflection,
                reflection_rounds=args.reflection_rounds,
                enable_description_audit=args.enable_description_audit,
                code_support_threshold=args.code_support_threshold,
                cve_alignment_threshold=args.cve_alignment_threshold,
                type_only_cve_threshold=args.type_only_cve_threshold,
                repair_threshold=args.repair_threshold,
                final_threshold=args.final_threshold,
                audit_threshold=args.audit_threshold,
                max_repair_candidates=args.max_repair_candidates,
            )
            sample_code = 'char tmp[64];\nstrcpy(tmp, buf);\nprintf("%s", tmp);\n'
            result = generator.generate_vulnerability_description(
                function_code=sample_code,
                critical_lines="L1: char tmp[64];\nL2: strcpy(tmp, buf);",
                vulnerable_lines="L2: strcpy(tmp, buf);",
                line_relations="L2 -> L1 (writes to fixed-size buffer tmp without bounds checking)",
                cve_description="Stack-based buffer overflow caused by unsafe use of strcpy without bounds checking.",
                sample_id="builtin-example",
            )
            print("\n--- Final Generated Description ---")
            print(result["final_description"])
            return 0
        except Exception as exc:
            print(f"Error: {exc}")
            return 1

    if not args.output:
        parser.error("--output is required when --input is provided")
    records = _load_jsonl(args.input)
    records = records[args.start:]
    if args.limit is not None:
        records = records[: args.limit]

    if args.dry_run:
        print(json.dumps({
            "input": args.input,
            "records": len(records),
            "first_sample_id": records[0].get("sample_id") if records else None,
            "model": args.model,
            "min_branches": args.min_branches,
            "max_branches": args.max_branches,
            "verification_mode": args.verification_mode,
            "repair_check": args.enable_repair_check,
            "claim_reflection": args.enable_claim_reflection,
            "reflection_rounds": args.reflection_rounds,
            "description_audit": args.enable_description_audit,
        }, ensure_ascii=False, indent=2))
        return 0

    done = _read_existing_ids(args.output, args.overwrite) if args.resume else set()
    if done:
        records = [r for r in records if str(r.get("sample_id", r.get("sample_idx", r.get("index")))) not in done]
    if not records:
        print("No records left to process.")
        return 0

    api_key = os.environ.get(args.api_key_env)
    generator = ToTVRGenerator(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        min_branches=args.min_branches,
        max_branches=args.max_branches,
        max_reconsideration_rounds=args.max_reconsideration_rounds,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        extra_body=extra_body,
        verbose=args.verbose,
        verification_mode=args.verification_mode,
        min_verification_branches=args.min_verification_branches,
        enable_repair_check=args.enable_repair_check,
        enable_claim_reflection=args.enable_claim_reflection,
        reflection_rounds=args.reflection_rounds,
        enable_description_audit=args.enable_description_audit,
        code_support_threshold=args.code_support_threshold,
        cve_alignment_threshold=args.cve_alignment_threshold,
        type_only_cve_threshold=args.type_only_cve_threshold,
        repair_threshold=args.repair_threshold,
        final_threshold=args.final_threshold,
        audit_threshold=args.audit_threshold,
        max_repair_candidates=args.max_repair_candidates,
    )
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    mode = "a" if args.resume and os.path.exists(args.output) else "w"
    lock = threading.Lock()
    completed: List[Dict[str, Any]] = []

    with open(args.output, mode, encoding="utf-8") as out:
        def process(record: Dict[str, Any]) -> Dict[str, Any]:
            last_error: Optional[Exception] = None
            for attempt in range(args.sample_retries + 1):
                try:
                    return _run_one(generator, record)
                except Exception as exc:
                    last_error = exc
                    if attempt < args.sample_retries:
                        time.sleep(min(2 ** attempt, 8))
            sample_id = record.get("sample_id", record.get("sample_idx", record.get("index")))
            return {
                "sample_id": sample_id,
                "sample_idx": record.get("sample_idx", sample_id),
                "final_description": "",
                "verified": False,
                "error": str(last_error),
            }

        if args.workers <= 1:
            for record in records:
                result = process(record)
                completed.append(result)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                status = "ok" if result.get("final_description") else "ERROR"
                print(f"[{len(completed)}/{len(records)}] sample {result.get('sample_id')}: {status}")
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process, r): r for r in records}
                for future in as_completed(futures):
                    result = future.result()
                    with lock:
                        completed.append(result)
                        out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out.flush()
                        status = "ok" if result.get("final_description") else "ERROR"
                        print(f"[{len(completed)}/{len(records)}] sample {result.get('sample_id')}: {status}")

    stats = _summary(completed)
    print("\nSummary:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)
    return 0 if stats["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
