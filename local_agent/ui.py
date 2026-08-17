"""Small local Tkinter UI for configuring and exercising the coding agent."""

from __future__ import annotations

import queue
import re
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
import tkinter as tk

from .config import AgentConfig
from .models import PreparedChange, RunReport
from .orchestrator import Orchestrator
from .providers import ProviderError, build_provider


PLACEHOLDER_RE = re.compile(r"^(\.\.\.|YOUR_|your_|CHANGEME|placeholder|<.*>)$", re.IGNORECASE)


def is_placeholder_key(value: str) -> bool:
    candidate = value.strip()
    return bool(candidate and (PLACEHOLDER_RE.fullmatch(candidate) or candidate.lower().startswith("your_")))


class AgentUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DungX AI Coding Agent")
        self.root.geometry("760x760")
        self.root.minsize(640, 620)
        self.events: queue.Queue[tuple] = queue.Queue()
        self.busy = False

        self.provider_var = tk.StringVar(value="gemini")
        self.key_var = tk.StringVar(value="")
        self.model_var = tk.StringVar(value="gemini-2.5-flash")
        self.project_var = tk.StringVar(value=str(Path.cwd()))
        self.dry_run_var = tk.BooleanVar(value=False)
        self.approval_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="● Not connected")

        self._build()
        self.root.after(100, self._drain_events)

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(8, weight=1)

        ttk.Label(outer, text="DungX AI Coding Agent", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))

        ttk.Label(outer, text="Provider").grid(row=1, column=0, sticky="w", pady=4)
        self.provider_box = ttk.Combobox(outer, textvariable=self.provider_var, values=("gemini", "openai", "mock", "antigravity"), state="readonly")
        self.provider_box.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        self.provider_box.bind("<<ComboboxSelected>>", self._provider_changed)

        ttk.Label(outer, text="Gemini API Key").grid(row=2, column=0, sticky="w", pady=4)
        self.key_entry = ttk.Entry(outer, textvariable=self.key_var, show="*")
        self.key_entry.grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(outer, text="Model").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(outer, textvariable=self.model_var).grid(row=3, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(outer, text="Project").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(outer, textvariable=self.project_var).grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Button(outer, text="Browse...", command=self._browse_project).grid(row=4, column=2, padx=(8, 0), pady=4)

        self.connection_button = ttk.Button(outer, text="Test Connection", command=self.test_connection)
        self.connection_button.grid(row=5, column=0, columnspan=3, sticky="w", pady=(10, 4))

        options = ttk.Frame(outer)
        options.grid(row=6, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Checkbutton(options, text="Dry Run", variable=self.dry_run_var).pack(side="left", padx=(0, 18))
        ttk.Checkbutton(options, text="Require Approval", variable=self.approval_var).pack(side="left")

        ttk.Label(outer, text="Task").grid(row=7, column=0, columnspan=3, sticky="w", pady=(12, 4))
        self.task_text = scrolledtext.ScrolledText(outer, height=7, wrap="word")
        self.task_text.grid(row=8, column=0, columnspan=3, sticky="nsew")

        self.run_button = ttk.Button(outer, text="Run Agent", command=self.run_agent)
        self.run_button.grid(row=9, column=0, columnspan=3, sticky="w", pady=(10, 4))
        ttk.Label(outer, textvariable=self.status_var).grid(row=10, column=0, columnspan=3, sticky="w", pady=4)

        ttk.Label(outer, text="Output").grid(row=11, column=0, columnspan=3, sticky="w", pady=(12, 4))
        self.output = scrolledtext.ScrolledText(outer, height=12, wrap="word", state="disabled")
        self.output.grid(row=12, column=0, columnspan=3, sticky="nsew")
        outer.rowconfigure(12, weight=2)

    def _provider_changed(self, _event=None) -> None:
        current = self.model_var.get().strip()
        if not current or current in {"gemini-2.5-flash", "gemini-3.7-flash"}:
            self.model_var.set("gemini-3.7-flash" if self.provider_var.get() == "antigravity" else "gemini-2.5-flash")

    def _browse_project(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.project_var.get() or str(Path.cwd()))
        if selected:
            self.project_var.set(selected)

    def _append(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.insert("end", text.rstrip() + "\n")
        self.output.see("end")
        self.output.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.connection_button.configure(state=state)
        self.run_button.configure(state=state)
        self.provider_box.configure(state="disabled" if busy else "readonly")

    def _credentials(self) -> str | None:
        """Return the UI credential for runtime config without global persistence."""
        value = self.key_var.get().strip()
        if value and is_placeholder_key(value):
            raise ValueError("replace the API-key placeholder before connecting")
        return value or None

    def _config(self) -> AgentConfig:
        provider = self.provider_var.get().lower()
        api_key = self._credentials()
        model = self.model_var.get().strip()
        if not model:
            model = "gemini-3.7-flash" if provider == "antigravity" else ("gemini-2.5-flash" if provider == "gemini" else "gpt-4.1-mini")
        config = AgentConfig.from_environment(
            self.project_var.get().strip() or ".",
            provider=provider,
            model=model,
            api_key=api_key,
            dry_run=self.dry_run_var.get(),
            approval="always" if self.approval_var.get() else "never",
        )
        config.validate()
        return config

    def test_connection(self) -> None:
        if self.busy:
            return
        try:
            config = self._config()
            self.status_var.set("● Testing connection...")
            self._set_busy(True)
        except (ValueError, ProviderError) as exc:
            messagebox.showerror("Connection", str(exc))
            return
        threading.Thread(target=self._connection_worker, args=(config,), daemon=True).start()

    def _connection_worker(self, config: AgentConfig) -> None:
        try:
            provider = build_provider(config)
            if config.provider in {"gemini", "antigravity"}:
                response = provider.test_connection()  # type: ignore[attr-defined]
                if response != "GEMINI_LIVE_OK":
                    raise ProviderError(f"{config.provider} responded with {response!r}, expected GEMINI_LIVE_OK")
                self.events.put(("connection", True, f"Connection successful\nResponse: GEMINI_LIVE_OK\nModel: {provider.model}", provider.model))
            elif config.provider == "mock":
                self.events.put(("connection", True, "Mock provider is available offline.", None))
            else:
                self.events.put(("connection", True, "Provider credentials loaded.", None))
        except (ProviderError, OSError, ValueError) as exc:
            self.events.put(("connection", False, str(exc)))

    def run_agent(self) -> None:
        if self.busy:
            return
        task = self.task_text.get("1.0", "end").strip()
        if not task:
            messagebox.showwarning("Task required", "Enter a task before running the agent.")
            return
        try:
            config = self._config()
        except (ValueError, ProviderError, OSError) as exc:
            messagebox.showerror("Configuration", str(exc))
            return
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")
        self.status_var.set("● Running...")
        self._set_busy(True)
        threading.Thread(target=self._run_worker, args=(config, task), daemon=True).start()

    def _run_worker(self, config: AgentConfig, task: str) -> None:
        try:
            provider = build_provider(config)
            orchestrator = Orchestrator(config, provider)
            report = orchestrator.run(task, progress=lambda message: self.events.put(("output", message)), approval_callback=self._approval_callback)
            self.events.put(("finished", report))
        except (ProviderError, ValueError, OSError) as exc:
            self.events.put(("error", str(exc)))

    def _approval_callback(self, changes: list[PreparedChange]) -> bool:
        event = threading.Event()
        result: dict[str, bool] = {}
        self.events.put(("approval", changes, event, result))
        event.wait()
        return result.get("approved", False)

    def _drain_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "output":
                    self._append(event[1])
                elif kind == "connection":
                    self._set_busy(False)
                    self.status_var.set("● Connected" if event[1] else "● Connection failed")
                    selected_model = event[3] if len(event) > 3 else None
                    if selected_model:
                        self.model_var.set(selected_model)
                    self._append(event[2])
                    if not event[1]:
                        messagebox.showerror("Connection", event[2])
                elif kind == "approval":
                    changes, wait_event, result = event[1], event[2], event[3]
                    added = sum(line.startswith("+") and not line.startswith("+++") for change in changes for line in change.diff.splitlines())
                    removed = sum(line.startswith("-") and not line.startswith("---") for change in changes for line in change.diff.splitlines())
                    approved = messagebox.askyesno("Apply AI changes?", f"{len(changes)} files\n{added} lines added\n{removed} lines removed\n\nApply these changes?")
                    result["approved"] = approved
                    wait_event.set()
                elif kind == "finished":
                    self._show_report(event[1])
                    self._set_busy(False)
                    self.status_var.set("● Complete" if event[1].completed else "● Incomplete")
                elif kind == "error":
                    self._set_busy(False)
                    self.status_var.set("● Error")
                    self._append("Error: " + event[1])
                    messagebox.showerror("Agent", event[1])
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _show_report(self, report: RunReport) -> None:
        self._append(f"Outcome: {report.outcome}")
        self._append(f"Result: {'COMPLETE' if report.completed else 'INCOMPLETE'}")
        self._append(f"Changed files: {', '.join(sorted(set(report.changed_files))) or 'none'}")
        if report.dry_run or report.approval_required:
            self._append(report.proposed_diff or "(no changes proposed)")
        for execution in report.executions:
            self._append(f"{'PASS' if execution.succeeded else 'FAIL'}: {execution.command} (exit {execution.exit_code})")
        if report.review:
            self._append(f"Review: {report.review.verdict} - {report.review.summary}")
        for metric in report.provider_metrics:
            self._append(f"Metric: {metric.request_type} ~{metric.approximate_input_tokens} in / ~{metric.approximate_output_tokens} out ({metric.duration_seconds:.3f}s)")


def main() -> int:
    root = tk.Tk()
    AgentUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
