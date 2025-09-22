# unified_logger.py (final patch)
from __future__ import annotations
import os, sys, tempfile, logging, inspect
from typing import Any, Dict, Optional, List, Protocol
from dataclasses import dataclass, field

# ---------------------------
# Context (label stack)
# ---------------------------

@dataclass
class LogContext:
    labels: List[str] = field(default_factory=list)

    def push(self, label: str) -> None:
        self.labels.append(label)

    def pop(self) -> None:
        if self.labels:
            self.labels.pop()

    def as_str(self) -> str:
        return " | ".join(self.labels)


# ---------------------------
# Progress protocol
# ---------------------------

class Progress(Protocol):
    def advance(self, n: int = 1) -> None: ...
    def set_description(self, desc: str) -> None: ...
    def get_description(self) -> str: ...
    def status(self, message: str, level: str = "INFO") -> None: ...
    def close(self) -> None: ...


class NullProgress:
    def advance(self, n: int = 1) -> None: pass
    def close(self) -> None: pass
    def set_description(self, desc: str) -> None: pass
    def get_description(self) -> str: return ""
    def status(self, message: str, level: str = "INFO") -> None: pass


class TqdmProgress:
    def __init__(self, total: int, desc: str, leave: bool, formatter: logging.Formatter):
        import tqdm
        self._pbar = tqdm.tqdm(total=total, desc=desc, leave=leave)
        self._formatter = formatter

    def _format_with_fmt(self, msg: str, level: str = "INFO") -> str:
        frame = inspect.currentframe().f_back.f_back
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno
        record = logging.LogRecord(
            name="UnifiedLogger.progress",
            level=getattr(logging, level.upper(), logging.INFO),
            pathname=filename,
            lineno=lineno,
            msg=msg,
            args=(),
            exc_info=None,
        )
        return self._formatter.format(record)

    def get_description(self) -> str:
        return self._pbar.desc

    def set_description(self, desc: str) -> None:
        formatted = self._format_with_fmt(desc)
        self._pbar.set_description_str(formatted)
        self._pbar.refresh()

    def status(self, message: str, level: str = "INFO") -> None:
        formatted = self._format_with_fmt(message, level)
        self._pbar.set_description_str(formatted)
        self._pbar.refresh()

    def advance(self, n: int = 1) -> None:
        self._pbar.update(n)

    def close(self) -> None:
        self._pbar.close()


# ---------------------------
# Sink interface
# ---------------------------

class Sink(Protocol):
    def start(self, experiment: Optional[str], run_name: Optional[str],
              params: Optional[Dict[str, Any]], tags: Optional[Dict[str, Any]]) -> None: ...
    def end(self, status: str = "FINISHED") -> None: ...
    def event(self, level: str, message: str) -> None: ...
    def metric(self, name: str, value: float, context: str) -> None: ...
    def figure(self, name: str, figure: Any) -> None: ...
    def artifact(self, path: str, dest: Optional[str]) -> None: ...
    def progress(self, total: int, desc: str, leave: bool, formatter: logging.Formatter) -> Progress: ...


# ---------------------------
# StdLogSink (console + file)
# ---------------------------

class ConsoleTqdmHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            try:
                from tqdm import tqdm
                tqdm.write(msg)
            except Exception:
                sys.stderr.write(msg + "\n")
        except Exception:
            self.handleError(record)


class StdLogSink:
    def __init__(self, path: str, fmt: str):
        self._logger = logging.getLogger("UnifiedLogger.stdout")
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        formatter = logging.Formatter(fmt)

        ch = ConsoleTqdmHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

        fh = logging.FileHandler(path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

    def start(self, experiment, run_name, params, tags): pass
    def end(self, status="FINISHED"): pass

    def event(self, level: str, message: str) -> None:
        frame = inspect.currentframe().f_back.f_back
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno
        lr = self._logger.makeRecord(
            name=self._logger.name,
            level=getattr(logging, level.upper(), logging.INFO),
            fn=filename,
            lno=lineno,
            msg=message,
            args=(),
            exc_info=None
        )
        self._logger.handle(lr)

    def metric(self, name: str, value: float, context: str): pass
    def figure(self, name: str, figure: Any): pass
    def artifact(self, path: str, dest: Optional[str]): pass
    def progress(self, total: int, desc: str, leave: bool, formatter: logging.Formatter) -> Progress:
        return NullProgress()


# ---------------------------
# TqdmSink
# ---------------------------

class TqdmSink:
    def __init__(self, default_leave: bool = False, formatter: Optional[logging.Formatter] = None):
        self._default_leave = default_leave
        self._formatter = formatter or logging.Formatter("%(message)s")

    def start(self, experiment, run_name, params, tags): pass
    def end(self, status="FINISHED"): pass
    def event(self, level, message): pass
    def metric(self, name, value, context): pass
    def figure(self, name, figure): pass
    def artifact(self, path, dest): pass

    def progress(self, total: int, desc: str, leave: bool, formatter: logging.Formatter) -> Progress:
        return TqdmProgress(total, desc, leave if leave is not None else self._default_leave, self._formatter)


# ---------------------------
# MlflowSink (metrics only shown)
# ---------------------------

class MlflowSink:
    def __init__(self, tracking_uri: Optional[str] = None, experiment: Optional[str] = None):
        import mlflow
        self.mlflow = mlflow
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        self.experiment = experiment
        self._run_active = False

    def start(self, experiment, run_name, params, tags): ...
    def end(self, status="FINISHED"): ...
    def event(self, level, message): pass
    def metric(self, name, value, context):
        self.mlflow.log_metric(name, float(value))
        if context:
            self.mlflow.set_tag("context", context)
    def figure(self, name, figure): ...
    def artifact(self, path, dest): ...
    def progress(self, total, desc, leave, formatter): return NullProgress()


# ---------------------------
# UnifiedLogger
# ---------------------------

class UnifiedLogger:
    def __init__(self, fmt: str = "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)3d | %(message)s"):
        self._fmt = fmt
        self._formatter = logging.Formatter(fmt)
        self._sinks: List[Sink] = []
        self._context = LogContext()
        self._pbar: Progress = NullProgress()

    def add_sink(self, sink: Sink): self._sinks.append(sink)

    # ---- Context management ----
    def context(self, label: str): self._context.push(label)
    def uncontext(self): self._context.pop()
    def _ctx_str(self) -> str: return self._context.as_str()

    # ---- Events ----
    def _event(self, level: str, msg: str):
        prefix = self._ctx_str()
        full_msg = f"{prefix} | {msg}" if prefix else msg
        for s in self._sinks:
            try: s.event(level, full_msg)
            except Exception: pass
    def info(self, msg): self._event("INFO", msg)
    def warning(self, msg): self._event("WARNING", msg)
    def error(self, msg): self._event("ERROR", msg)

    # ---- Metrics ----
    def metric(self, name: str, value: float):
        ctx = self._ctx_str()
        for s in self._sinks:
            try: s.metric(name, float(value), context=ctx)
            except Exception: pass

    # ---- Figures / Artifacts ----
    def figure(self, name: str, figure: Any):
        for s in self._sinks:
            try: s.figure(name, figure)
            except Exception: pass
    def artifact(self, path: str, dest: Optional[str] = None):
        for s in self._sinks:
            try: s.artifact(path, dest)
            except Exception: pass

    # ---- Progress ----
    def progress_start(self, total: int, *, desc: Optional[str] = None, leave: bool = False):
        prefix = self._ctx_str()
        combined = f"{prefix} | {desc}" if desc and prefix else (desc or prefix)
        frame = inspect.currentframe().f_back
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno
        record = logging.LogRecord(
            name="UnifiedLogger.progress",
            level=logging.INFO,
            pathname=filename,
            lineno=lineno,
            msg=combined,
            args=(),
            exc_info=None,
        )
        formatted = self._formatter.format(record)
        for s in self._sinks:
            try:
                self._pbar = s.progress(total, formatted, leave, self._formatter)
                if not isinstance(self._pbar, NullProgress):
                    return
            except Exception: continue
        self._pbar = NullProgress()

    def progress_status(self, msg: str, level: str = "INFO"):
        desc = self._ctx_str()
        full_msg = f"{desc} | {msg}" if desc else msg
        self._pbar.status(full_msg, level)

    def progress_advance(self, n: int = 1):
        self._pbar.advance(n)

    def progress_close(self): self._pbar.close()


# ---------------------------
# Singleton
# ---------------------------

_singleton: Optional[UnifiedLogger] = None
def get_logger() -> UnifiedLogger:
    global _singleton
    if _singleton is None:
        FMT = "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)3d | %(message)s"
        _singleton = UnifiedLogger(fmt=FMT)
        _singleton.add_sink(StdLogSink(path="sleepwalker.log", fmt=FMT))
        _singleton.add_sink(TqdmSink(formatter=_singleton._formatter))
    return _singleton

logger: UnifiedLogger = get_logger()
