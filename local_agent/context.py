from __future__ import annotations

import datetime
import re
from collections import defaultdict
from pathlib import Path

from .filesystem import ProjectFilesystem, ProtectedPathError, SandboxViolation
from .models import ProjectContext, ProjectMemory, SemanticIndex


_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_STOP_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of", "on", "or", "the", "to", "with", "add", "fix", "page", "feature", "application", "simple", "existing", "system", "register", "implement", "make"}
_SYMBOL_NAME_RE = re.compile(r"`([^`]+)`|([A-Za-z_][A-Za-z0-9_.]+)") # Matches `symbol` or plain_symbol.with.dots
_UI_TASK_TERMS = {"about", "component", "frontend", "layout", "navigation", "nav", "page", "react", "route", "router", "screen", "ui", "view"}
_RELATION_BONUS = {
    "imports": 0.10,
    "router_route": 0.24,
    "router_uses": 0.18,
    "layout_uses": 0.20,
    "navigation_uses": 0.20,
    "navigation_routes": 0.18,
    "page_uses": 0.12,
    "tested_by": 0.16,
}
_SYNONYM_MAP = {
    "login": {"auth", "authentication", "signin", "user", "session"},
    "auth": {"login", "authentication", "signin", "user", "session"},
    "authentication": {"login", "auth", "signin", "user", "session"},
    "signin": {"login", "auth", "authentication"},
    "signup": {"register", "registration", "user"},
    "register": {"signup", "registration", "user"},
}


class ContextSelector:
    """Deterministically selects task-relevant files before provider calls."""

    def __init__(self, root: str | Path, max_files: int = 24, max_chars: int = 30000, max_file_chars: int = 5000, max_tokens: int = 7500, dependency_depth: int = 1, project_memory: ProjectMemory | None = None):
        self.filesystem = ProjectFilesystem(root)
        self.max_files = max_files
        self.max_chars = max_chars
        self.max_file_chars = max_file_chars
        self.max_tokens = max_tokens
        self.dependency_depth = dependency_depth
        self.project_memory = project_memory or ProjectMemory()

    def select(self, task: str, context: ProjectContext) -> ProjectContext:
        keywords = self._keywords(task)
        raw_terms = {word.lower() for word in _WORD_RE.findall(task)}
        ui_task = bool(raw_terms & _UI_TASK_TERMS)
        repository_map = context.repository_map
        records = {item.path: item for item in repository_map.files} if repository_map else {}
        candidates = sorted(set(records) or (set(context.source_files + context.test_files + context.config_files + context.dependency_files + context.documentation_files)))
        modified = self._modified_paths(context.git_status)
        package_dependencies = {
            token.lower()
            for dependency in context.metadata.get("package_dependencies", [])
            for token in self._keywords(str(dependency).replace("/", " ").replace("-", " "))
        }
        scored: dict[str, dict[str, object]] = {}

        # Phase 3.15.6: Use a dictionary to store boosts to handle different priorities.
        semantic_boosts: dict[str, tuple[float, str]] = {}  # path -> (boost_score, reason)
        semantic_index = context.metadata.get("semantic_index")
        if semantic_index and isinstance(semantic_index, SemanticIndex):
            candidate_symbols = self._extract_candidate_symbols(task)

            # Separate qualified and unqualified symbols
            qualified_candidates = {s for s in candidate_symbols if '.' in s}
            unqualified_candidates = candidate_symbols - qualified_candidates

            # 1. Process exact qualified matches (highest priority)
            for q_symbol in qualified_candidates:
                parts = q_symbol.rsplit('.', 1)
                if len(parts) != 2: continue
                parent_name, child_name = parts

                for path, file_index in semantic_index.files.items():
                    for symbol in file_index.symbols:
                        if symbol.name == child_name and symbol.parent == parent_name:
                            reason = f"semantic qualified symbol match: {q_symbol}"
                            # Use a higher boost for qualified matches
                            current_boost, _ = semantic_boosts.get(path, (0.0, ""))
                            # Qualified match boost: 0.75
                            if 0.75 > current_boost:
                                semantic_boosts[path] = (0.75, reason)

            # 2. Process exact unqualified matches
            exact_matches = semantic_index.find_symbols(unqualified_candidates)
            for path, symbol in exact_matches:
                reason = f"semantic symbol definition match: {symbol.name}"
                current_boost, _ = semantic_boosts.get(path, (0.0, ""))
                # Unqualified exact match boost: 0.45
                if 0.45 > current_boost:
                    semantic_boosts[path] = (0.45, reason)

            # 3. Substring symbol matches for broader discovery (lowest priority)
            for query in unqualified_candidates:
                if len(query) > 4:
                    substring_matches = semantic_index.search_symbols(query)
                    for path, symbol in substring_matches:
                        reason = f"semantic symbol substring match: {symbol.name} (for '{query}')"
                        current_boost, _ = semantic_boosts.get(path, (0.0, ""))
                        # Substring match boost: 0.25
                        if 0.25 > current_boost:
                            semantic_boosts[path] = (0.25, reason)

        # Phase 3.15.7: Dependency-aware expansion from semantic seeds
        if repository_map and repository_map.relationships:
            semantic_seed_paths = sorted(list(semantic_boosts.keys()))  # Deterministic

            # Build a quick lookup for relationships for performance
            forward_deps = defaultdict(list)
            reverse_deps = defaultdict(list)
            for rel in repository_map.relationships:
                forward_deps[rel.source].append(rel.target)
                reverse_deps[rel.target].append(rel.source)

            for seed_path in semantic_seed_paths:
                # A file is related if it imports the seed, or is imported by the seed.
                related_paths = set(forward_deps.get(seed_path, [])) | set(reverse_deps.get(seed_path, []))

                for related_path in sorted(list(related_paths)):  # Deterministic
                    if related_path == seed_path:
                        continue

                    reason = f"related to semantic match in {seed_path}"
                    current_boost, _ = semantic_boosts.get(related_path, (0.0, ""))
                    if 0.12 > current_boost:
                        semantic_boosts[related_path] = (0.12, reason)

        # Phase 3.22: Add boosts from Project Memory
        memory_boosts: dict[str, tuple[float, str]] = {}  # path -> (boost, reason)
        if self.project_memory:
            task_keywords = self._keywords(task)
            expanded_task_keywords = set(task_keywords)
            for kw in task_keywords:
                expanded_task_keywords.update(_SYNONYM_MAP.get(kw, set()))
            for memory in self.project_memory.memories:
                # Relevance: memory content has task keywords or synonyms, and it's related to a file
                mem_keywords = self._keywords(memory.content)
                if memory.related_path and (expanded_task_keywords & mem_keywords):
                    # Stale memory check: decay score based on age.
                    age_days = (datetime.datetime.now(datetime.timezone.utc) - memory.timestamp).days
                    decay_factor = max(0, 1 - (age_days / 90))  # Memory influence decays over 90 days
                    if decay_factor > 0:
                        boost = 0.20 * memory.confidence * decay_factor
                        reason = f"project memory: {memory.category.value}"

                        current_boost, _ = memory_boosts.get(memory.related_path, (0.0, ""))
                        if boost > current_boost:
                            memory_boosts[memory.related_path] = (boost, reason)

        for relative in candidates:
            try:
                with self.filesystem.resolve(relative).open("r", encoding="utf-8") as handle:
                    content = handle.read(max(12000, self.max_file_chars))
            except (OSError, UnicodeDecodeError, ProtectedPathError, SandboxViolation):
                continue
            record = records.get(relative)
            path_words = set(self._keywords(relative.replace("/", " ")))
            content_words = set(self._keywords(content[:12000]))
            path_matches = len(keywords & path_words)
            content_matches = len(keywords & content_words)
            score = 0.02 + min(0.42, path_matches * 0.20) + min(0.10, content_matches * 0.012)
            reasons: list[str] = []
            if path_matches:
                reasons.append("task keyword match in path")
            if content_matches:
                reasons.append("task keyword match in content")

            if relative in semantic_boosts:
                boost, reason = semantic_boosts[relative]
                score += boost
                reasons.append(reason)

            if relative in memory_boosts:
                boost, reason = memory_boosts[relative]
                score += boost
                reasons.append(reason)

            if relative in modified:
                score += 0.14
                reasons.append("recently modified")
            if relative in context.config_files or relative in context.dependency_files:
                score += 0.06
                reasons.append("configuration or dependency file")
            if relative in context.test_files:
                score += 0.03
                reasons.append("test file")
            if relative in context.source_files and not (record and record.is_test):
                score += 0.08
                reasons.append("source implementation")
            if record and record.is_entry_point:
                score += 0.10
                reasons.append("application entry point")
            if record and record.language and any(record.language.lower() in keyword for keyword in keywords):
                score += 0.03
                reasons.append("task language relevance")
            lower_path = relative.lower()
            if ui_task:
                if "router" in lower_path or "route" in lower_path:
                    score += 0.34
                    reasons.append("router/navigation architecture")
                if "layout" in lower_path or "<outlet" in content.lower():
                    score += 0.28
                    reasons.append("layout relationship")
                if "navigation" in lower_path or "navbar" in lower_path or "sidebar" in lower_path or any(token in content for token in ("<NavLink", "navItems")):
                    score += 0.26
                    reasons.append("navigation relationship")
                if record and ("/pages/" in f"/{lower_path}/" or lower_path.endswith(("page.tsx", "page.ts"))):
                    score += 0.12
                    reasons.append("feature/page relationship")
                if lower_path == "package.json":
                    score += 0.30
                    reasons.append("framework configuration")
                    if keywords & package_dependencies:
                        score += 0.18
                        reasons.append("package dependency relevance")
                elif relative in context.dependency_files:
                    score += 0.04
                    reasons.append("framework configuration")
            scored[relative] = {"score": min(score, 0.99), "reasons": reasons or ["repository candidate"] , "content": content}

        relationships: dict[str, list[tuple[str, str]]] = defaultdict(list)
        incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
        if repository_map:
            for relationship in repository_map.relationships:
                relationships[relationship.source].append((relationship.target, relationship.kind))
                incoming[relationship.target].append((relationship.source, relationship.kind))
        base_scores = {path: float(item["score"]) for path, item in scored.items()}
        for source, targets in relationships.items():
            if source not in scored:
                continue
            for target, kind in targets:
                if target not in scored:
                    continue
                bonus = _RELATION_BONUS.get(kind, 0.08)
                if (base_scores[source] - semantic_boosts.get(source, (0.0, ""))[0]) >= 0.20:
                    scored[target]["score"] = min(0.99, float(scored[target]["score"]) + bonus * 0.65)
                    if kind == "router_route":
                        explanation = "direct router relationship"
                    elif kind == "tested_by":
                        explanation = f"tested by {source}"
                    else:
                        explanation = f"direct {kind} relationship from {source}"
                    scored[target]["reasons"] = list(scored[target]["reasons"]) + [explanation]
        for target, sources in incoming.items():
            if target not in scored or (base_scores[target] - semantic_boosts.get(target, (0.0, ""))[0]) < 0.20:
                continue
            for source, kind in sources:
                if source not in scored:
                    continue
                bonus = _RELATION_BONUS.get(kind, 0.08)
                scored[source]["score"] = min(0.99, float(scored[source]["score"]) + bonus)
                if kind == "imports":
                    explanation = f"imported by {target}"
                elif kind == "tested_by":
                    explanation = f"test relationship with {target}"
                else:
                    explanation = f"reverse {kind} relationship from {target}"
                scored[source]["reasons"] = list(scored[source]["reasons"]) + [explanation]
        ranked = sorted(scored, key=lambda path: (-float(scored[path]["score"]), path))
        selected: list[str] = []
        previews: dict[str, str] = {}
        selected_items: list[dict[str, object]] = []
        excluded: list[dict[str, str]] = []
        total = 0
        estimated_tokens = 0
        depths: dict[str, int] = {}
        truncated_files: list[str] = []
        seed_paths = [
            path for path in ranked
            if float(scored[path]["score"]) >= 0.4 and not (records.get(path) and records[path].is_test)
        ]
        if not seed_paths:
            seed_paths = [path for path in ranked if not (records.get(path) and records[path].is_test)][:1]
        pending = [(path, 0) for path in seed_paths]
        scheduled = set(seed_paths)
        pre_expansion_scores = {path: float(scored[path]["score"]) for path in scored}
        while pending and len(selected) < self.max_files:
            pending.sort(key=lambda item: (-float(scored[item[0]]["score"]), item[0]))
            relative, depth = pending.pop(0)
            if relative in selected:
                continue
            content = str(scored[relative]["content"])
            preview = content[:self.max_file_chars]
            allowed_bytes = min(self.max_chars - total, self.max_tokens * 4 - total)
            while preview and len(preview.encode("utf-8")) > allowed_bytes:
                preview = preview[:-1]
            preview_bytes = len(preview.encode("utf-8"))
            next_tokens = (total + preview_bytes + 3) // 4
            if not preview and content:
                excluded.append({"path": relative, "reason": "context budget reached"})
                continue
            if len(preview) < len(content):
                truncated_files.append(relative)
            selected.append(relative)
            previews[relative] = preview
            total += preview_bytes
            estimated_tokens = next_tokens
            depths[relative] = depth
            reasons = list(scored[relative]["reasons"])
            if depth:
                parent = next((source for source, targets in relationships.items() if any(target == relative for target, _ in targets)), None)
                reasons.append(f"dependency of {parent}" if parent else "dependency expansion")
            selected_items.append({"path": relative, "score": round(float(pre_expansion_scores[relative]), 3), "reason": reasons, "dependency_depth": depth})
            if depth < self.dependency_depth:
                for target, kind in relationships.get(relative, []):
                    if target in scored and target not in selected and target not in {item[0] for item in pending}:
                        scored[target]["score"] = max(float(scored[target]["score"]), float(scored[relative]["score"]) * (0.78 if kind == "imports" else 0.84))
                        scored[target]["reasons"] = list(scored[target]["reasons"]) + [f"{kind} relationship from {relative}"]
                        pending.append((target, depth + 1))
                        scheduled.add(target)
            if not pending and len(selected) < self.max_files:
                fallback = next((path for path in ranked if path not in scheduled and path not in selected and float(scored[path]["score"]) >= 0.20), None)
                if fallback:
                    pending.append((fallback, 0))
                    scheduled.add(fallback)
        selected_set = set(selected)
        for relative in scored:
            if relative not in selected_set and not any(item["path"] == relative for item in excluded):
                excluded.append({"path": relative, "reason": "lower relevance than selected context"})
        if repository_map:
            excluded.extend(item for item in repository_map.ignored_paths + repository_map.protected_paths if item["path"] not in selected_set)
        context.metadata["selected_files"] = selected
        context.metadata["selected_file_previews"] = previews
        context.metadata["context_selection"] = {
            "keyword_count": len(keywords), "candidate_count": len(candidates), "selected_count": len(selected),
            "max_chars": self.max_chars, "max_file_chars": self.max_file_chars, "max_tokens": self.max_tokens,
            "estimated_tokens": estimated_tokens, "dependency_depth": self.dependency_depth,
            "truncated_files": truncated_files,
            "selected_items": selected_items, "excluded_count": len(excluded),
        }
        context.metadata["context_excluded"] = excluded
        context.metadata["selected_relationships"] = [
            {"source": item.source, "target": item.target, "kind": item.kind}
            for item in (repository_map.relationships if repository_map else [])
            if item.source in selected_set or item.target in selected_set
        ]
        return context

    @staticmethod
    def _keywords(value: str) -> set[str]:
        cleaned = re.sub(r"`[^`]+`", " ", value)
        camel_case = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", cleaned)
        return {word.lower() for word in _WORD_RE.findall(camel_case) if len(word) > 2 and word.lower() not in _STOP_WORDS}

    def _extract_candidate_symbols(self, task: str) -> set[str]:
        candidates: set[str] = set()
        for match in _SYMBOL_NAME_RE.finditer(task):
            symbol = match.group(1) or match.group(2)
            is_backtick_name = match.group(1) is not None
            is_qualified_name = "." in symbol if symbol else False
            is_identifier_style = bool(symbol and ("_" in symbol or re.search(r"[a-z0-9][A-Z]", symbol)))
            if symbol and (is_backtick_name or is_qualified_name or is_identifier_style):
                candidates.add(symbol)
                # Add last component for qualified names
                if is_qualified_name:
                    candidates.add(symbol.split('.')[-1])
        return {s for s in candidates if s and s.lower() not in _STOP_WORDS and len(s) > 2}

    @staticmethod
    def _modified_paths(status: str) -> set[str]:
        result: set[str] = set()
        for line in status.splitlines():
            if line.startswith("##") or len(line) < 4:
                continue
            value = line[3:].strip()
            if " -> " in value:
                value = value.rsplit(" -> ", 1)[-1]
            result.add(Path(value.strip('"')).as_posix())
        return result
