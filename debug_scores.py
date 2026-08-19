import tempfile
from pathlib import Path
from local_agent.context import ContextSelector
from local_agent.models import (
    FileIndex, ProjectContext, RepositoryFile, RepositoryMap,
    SemanticIndex, SymbolDefinition, SymbolLocation,
)


def build(records):
    root = Path(tempfile.mkdtemp()) / "repo"
    root.mkdir(parents=True)
    (root / "user_service.py").write_text("class UserService:\n  def save(self, user):\n    pass\n")
    (root / "auth_service.py").write_text("class AuthService:\n  def save(self, user):\n    pass\n")
    (root / "unrelated_saver.py").write_text("def save():\n  pass\n")
    (root / "analytics_service.js").write_text("class AnalyticsService {\n  track() {}\n}\n")
    (root / "other.py").write_text("class Other:\n  def other_method(self):\n    pass\n")

    def sym(name, kind, line, parent=None):
        return SymbolDefinition(name=name, kind=kind, location=SymbolLocation(start_line=line, end_line=line + 1), parent=parent)
    si = SemanticIndex(files={
        "user_service.py": FileIndex("user_service.py", "Python", "h1", [sym("UserService", "class", 1), sym("save", "method", 2, "UserService")]),
        "auth_service.py": FileIndex("auth_service.py", "Python", "h2", [sym("AuthService", "class", 1), sym("save", "method", 2, "AuthService")]),
        "unrelated_saver.py": FileIndex("unrelated_saver.py", "Python", "h3", [sym("save", "function", 1)]),
        "analytics_service.js": FileIndex("analytics_service.js", "JavaScript", "h4", [sym("AnalyticsService", "class", 1), sym("track", "method", 2, "AnalyticsService")]),
        "other.py": FileIndex("other.py", "Python", "h5", [sym("Other", "class", 1), sym("other_method", "method", 2, "Other")]),
    })
    ctx = ProjectContext(root=str(root), source_files=list(si.files.keys()), metadata={"semantic_index": si}, repository_map=records)
    sel = ContextSelector(root)
    return sel, ctx, root


def score(sel, ctx, task, path):
    sel.select(task, ctx)
    items = ctx.metadata["context_selection"]["selected_items"]
    for it in items:
        if it["path"] == path:
            return it["score"]
    return 0.0


def run(label, records):
    print(f"=== {label} ===")
    sel, ctx, root = build(records)
    import io
    from local_agent.context import ContextSelector as _CS  # noqa
    # unqualified save
    for p in ["user_service.py", "auth_service.py", "unrelated_saver.py"]:
        ctx2 = ProjectContext(root=str(root), source_files=list(_to_si(root).files.keys()) if False else ctx.source_files, metadata={"semantic_index": ctx.metadata["semantic_index"]}, repository_map=records)
        s = score(sel, ctx2, "Fix the `save` function", p)
        print(f"  unq save {p}: {s:.3f}")
    # qualified UserService.save
    for p in ["user_service.py", "auth_service.py"]:
        s = score(sel, ctx, "Fix `UserService.save` method", p)
        print(f"  qual UserService.save {p}: {s:.3f}")
    # parent container mismatch
    for p in ["user_service.py", "auth_service.py"]:
        s = score(sel, ctx, "Fix `AdminService.save`", p)
        print(f"  mismatch AdminService.save {p}: {s:.3f}")
    # class only
    s = score(sel, ctx, "Update the `UserService`", "user_service.py")
    print(f"  class UserService user_service: {s:.3f}")
    s = score(sel, ctx, "Update the `UserService`", "auth_service.py")
    print(f"  class UserService auth_service: {s:.3f}")


def _to_si(root):
    return None


if __name__ == "__main__":
    # records = None (empty repository_map)
    run("records empty", None)
