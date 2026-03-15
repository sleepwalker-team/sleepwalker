# unified_logger.py
from __future__ import annotations
import os, sys, logging, inspect
from typing import Any, Dict, Optional, List, Protocol
from dataclasses import dataclass, field
from contextlib import contextmanager

# ---------------------------
# Level-aware formatter
# ---------------------------

class LevelAwareFormatter(logging.Formatter):
    def __init__(self, fmt_info: str, fmt_other: str, datefmt: Optional[str] = None):
        super().__init__(datefmt=datefmt)
        self._fmt_info = logging.Formatter(fmt_info, datefmt=datefmt)
        self._fmt_other = logging.Formatter(fmt_other, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return self._fmt_info.format(record)
        else:
            return self._fmt_other.format(record)

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
        self.total = total
        self.consumed = 0

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
        n = min(n, self.total - self.consumed)
        self._pbar.update(n)
        self.consumed += n

    def close(self) -> None:
        self._pbar.close()


# ---------------------------
# Sink interface
# ---------------------------

class Sink(Protocol):
    def start(self, run_name: Optional[str],
              params: Optional[Dict[str, Any]], tags: Optional[Dict[str, Any]]) -> None: ...
    def end(self, status: str = "FINISHED") -> None: ...
    def event(self, level: str, message: str, context: str) -> None: ...
    def metric(self, name: str, value: float, step:int, context: str) -> None: ...
    def figure(self, name: str, figure: Any, context: str) -> None: ...
    def artifact(self, path: str, dest: Optional[str], context: str) -> None: ...
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
    def __init__(self, path: str, formatter: logging.Formatter):
        self._logger = logging.getLogger(f"UnifiedLogger.stdout.{id(self)}")
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)

        # Console
        ch = ConsoleTqdmHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

        # File
        fh = logging.FileHandler(path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

    def start(self, run_name, params, tags): pass
    def end(self, status="FINISHED"): pass

    def event(self, level: str, message: str, context: str) -> None:
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

    def metric(self, name: str, value: float, step:int, context: str): pass
    def figure(self, name: str, figure: Any, context: str): pass
    def artifact(self, path: str, dest: Optional[str], context: str): pass
    def progress(self, total: int, desc: str, leave: bool, formatter: logging.Formatter) -> Progress:
        return NullProgress()


# ---------------------------
# TqdmSink
# ---------------------------

class TqdmSink:
    def __init__(self, default_leave: bool = False, formatter: Optional[logging.Formatter] = None):
        self._default_leave = default_leave
        self._formatter = formatter or logging.Formatter("%(message)s")

    def start(self, run_name, params, tags): pass
    def end(self, status="FINISHED"): pass
    def event(self, level, message, context: str): pass
    def metric(self, name, value, step, context: str): pass
    def figure(self, name, figure, context: str): pass
    def artifact(self, path, dest, context: str): pass

    def progress(self, total: int, desc: str, leave: bool, formatter: logging.Formatter) -> Progress:
        return TqdmProgress(total, desc, leave if leave is not None else self._default_leave, self._formatter)


# ---------------------------
# MlflowSink
# ---------------------------

class MlflowSink:
    def __init__(self, tracking_uri: Optional[str] = None, experiment: Optional[str] = None, artifact_uri:Optional[str]= None):
        import mlflow
        self.mlflow = mlflow
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        
        self.artifact_uri = artifact_uri
        self.experiment = experiment
        self._run_active = False
        self._run_id = None
        self._experiment_id = None

    def start(self, run_name: Optional[str],
              params: Optional[Dict[str, Any]], tags: Optional[Dict[str, Any]]):
        exp_name = self.experiment
        if exp_name:
            exp = self.mlflow.get_experiment_by_name(exp_name)
            exp_id = exp.experiment_id if exp else self.mlflow.create_experiment(exp_name, artifact_location=self.artifact_uri)
        else:
            exp_id = None

        active = self.mlflow.active_run()
        if active is None:
            self.mlflow.start_run(
                experiment_id=exp_id,
                run_name=run_name,
                tags=tags or {}
            )
        #     self._run_id = run.info.run_id
        #     self._experiment_id = run.info.experiment_id
        # else:
        #     self._run_id = active.info.run_id
        #     self._experiment_id = active.info.experiment_id
        if params:
            try:
                self.mlflow.log_params(params)
            except Exception:
                pass
        self._run_active = True

    # @property
    # def run_id(self):
    #     return self._run_id

    # @property
    # def experiment_id(self):
    #     return self._experiment_id            

    def end(self, status: str = "FINISHED"):
        if self._run_active:
            self.mlflow.end_run(status=status)
            self._run_active = False

    def event(self, level: str, message: str, context: str):
        # MLflow doesn't handle free-form log events; ignore
        pass

    def metric(self, name: str, value: float, step:int, context: str):
        # Context is ignored → metric name must be explicit
        self.mlflow.log_metric(name, float(value), step=step)

    def figure(self, name: str, figure: Any, context: str):
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.close()
        try:
            if hasattr(figure, "savefig"):
                figure.savefig(tmp.name, bbox_inches="tight")
            elif callable(figure):
                figure(tmp.name)
            else:
                raise TypeError("figure must be matplotlib.Figure or callable(path)")
            artifact_path = f"figures/{name}"
            self.mlflow.log_artifact(tmp.name, artifact_path=artifact_path)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def artifact(self, path: str, dest: Optional[str], context: str):
        # Context is ignored → artifact path only depends on dest
        artifact_path = dest or ""
        if os.path.isdir(path):
            self.mlflow.log_artifacts(path, artifact_path=artifact_path)
        else:
            self.mlflow.log_artifact(path, artifact_path=artifact_path)

    def progress(self, total: int, desc: str, leave: bool, formatter: logging.Formatter) -> Progress:
        # MLflow has no progress bar concept
        return NullProgress()


# ---------------------------
# UnifiedLogger
# ---------------------------

class UnifiedLogger:
    def __init__(self, formatter: logging.Formatter, level: int = logging.INFO):
        self._formatter = formatter
        self._sinks: List[Sink] = []
        self._context = LogContext()
        self._pbar: Progress = NullProgress()
        self._level = level  

    def set_level(self, level: str | int) -> None:
        """Change the active logging level (e.g. 'DEBUG', 'WARNING')."""
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self._level = level

    def get_level(self) -> int:
        return self._level

    def add_sink(self, sink: Sink): self._sinks.append(sink)

    # ---- Context management ----
    def context(self, label: str): self._context.push(label)
    def uncontext(self): self._context.pop()
    def _ctx_str(self) -> str: return self._context.as_str()

    # ---- Run lifecycle ----
    def start_run(self, run_name: Optional[str] = None,
                  params: Optional[Dict[str, Any]] = None, tags: Optional[Dict[str, Any]] = None):
        for s in self._sinks:
            try: s.start(run_name, params, tags)
            except Exception: pass
        if run_name:
            self.context(run_name)

    def end_run(self, status: str = "FINISHED"):
        for s in self._sinks:
            try: s.end(status)
            except Exception: pass
        self.uncontext()

    # ---- Events ----
    def _event(self, level: str, msg: str):
        lvl = getattr(logging, level.upper(), logging.INFO)
        if lvl < self._level:
            return  # below threshold → skip

        prefix = self._ctx_str()
        full_msg = f"{prefix} | {msg}" if prefix else msg
        ctx = self._ctx_str()
        for s in self._sinks:
            try:
                s.event(level, full_msg, context=ctx)
            except Exception:
                pass
    def info(self, msg): self._event("INFO", msg)
    def warning(self, msg): self._event("WARNING", msg)
    def error(self, msg): self._event("ERROR", msg)
    def debug(self, msg): self._event("DEBUG", msg)
    
    # ---- Metrics ----
    def metric(self, name: str, value: float, step:int = 0):
        ctx = self._ctx_str()
        for s in self._sinks:
            try: s.metric(name, float(value), step=step, context=ctx)
            except Exception: pass

    # ---- Figures / Artifacts ----
    def figure(self, name: str, figure: Any):
        ctx = self._ctx_str()
        for s in self._sinks:
            try: s.figure(name, figure, context=ctx)
            except Exception: pass

    def artifact(self, path: str, dest: Optional[str] = None):
        ctx = self._ctx_str()
        for s in self._sinks:
            try: s.artifact(path, dest, context=ctx)
            except Exception: pass

    # ---- Progress ----
    def progress_start(self, total: int, *, desc: Optional[str] = None, leave: bool = True):
        # skip entirely if below level
        if logging.INFO < self._level:
            self._pbar = NullProgress()
            return self._pbar

        try:
            self._pbar.close()
        except Exception:
            pass

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
                    return self._pbar
            except Exception: continue
        self._pbar = NullProgress()
        return self._pbar

    def progress_status(self, msg: str, level: str = "INFO"):
        lvl = getattr(logging, level.upper(), logging.INFO)
        if lvl < self._level:
            return 
    
        desc = self._ctx_str()
        full_msg = f"{desc} | {msg}" if desc else msg
        self._pbar.status(full_msg, level)

    def progress_advance(self, n: int = 1):
        self._pbar.advance(n)

    def progress_close(self): self._pbar.close()

@contextmanager
def suppress_stdout_logging(logger: UnifiedLogger):
    for sink in logger._sinks:
        if isinstance(sink, StdLogSink):
            for handler in sink._logger.handlers:
                handler._old_level = handler.level
                handler.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        for sink in logger._sinks:
            if isinstance(sink, StdLogSink):
                for handler in sink._logger.handlers:
                    handler.setLevel(handler._old_level)
                    del handler._old_level

# ---------------------------
# Singleton
# ---------------------------

_singleton: Optional[UnifiedLogger] = None
def get_logger(level: str | int = logging.INFO) -> UnifiedLogger:
    global _singleton
    if _singleton is None:
        FMT_INFO  = "%(asctime)s | %(levelname)s | %(message)s"
        FMT_OTHER = "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)3d | %(message)s"
        formatter = LevelAwareFormatter(FMT_INFO, FMT_OTHER)

        _singleton = UnifiedLogger(formatter=formatter, level=level)
        _singleton.add_sink(StdLogSink(path="sleepwalker.log", formatter=formatter))
        _singleton.add_sink(TqdmSink(formatter=formatter))
    else:
        _singleton.set_level(level)
    return _singleton

logger: UnifiedLogger = get_logger()