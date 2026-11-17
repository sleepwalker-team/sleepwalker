"""Optional CPU, GPU, memory and disk monitoring through a logger sink.

Disk counters describe individual devices, not job-specific I/O.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading
import time

from sleepwalker.utils import NullProgress, Sink, logger


@contextmanager
def telemetry_run(options, *, output, run_name, tags=None, overwrite=False):
    """Record an evaluation lifecycle with optional telemetry sink settings.

    An overwrite of evaluation results records telemetry in a fresh ``rerun-*``
    subdirectory when the telemetry directory already exists.
    """
    if options is None:
        yield
        return
    settings = dict(options)
    directory = Path(settings.pop("output", output))
    if overwrite and directory.exists():
        directory = Path(tempfile.mkdtemp(prefix="rerun-", dir=directory))
    sink = TelemetrySink(directory, **settings)
    logger.add_sink(sink)
    started = False
    status = "FAILED"
    try:
        logger.start_run(run_name=run_name, tags=tags)
        started = True
        yield
        status = "FINISHED"
    finally:
        try:
            if started:
                logger.end_run(status)
        finally:
            logger.remove_sink(sink)


def process_stats():
    records = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            pid = int(path.parent.name)
            records[pid] = (int(fields[1]), int(fields[11]) + int(fields[12]), int(fields[21]), int(fields[19]))
        except (OSError, ValueError, IndexError):
            continue
    return records


def tree_stats(root_pid):
    records = process_stats()
    selected = {root_pid}
    while True:
        children = {pid for pid, record in records.items() if record[0] in selected}
        expanded = selected | children
        if expanded == selected:
            break
        selected = expanded
    result = {}
    for pid in selected:
        if pid not in records:
            continue
        try:
            io = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/io').read_text().splitlines())
            result[pid] = {'ticks': records[pid][1], 'start_ticks': records[pid][3], 'rss_bytes': records[pid][2] * os.sysconf('SC_PAGE_SIZE'),
                           'read_bytes': int(io['read_bytes']), 'write_bytes': int(io['write_bytes'])}
        except (OSError, ValueError):
            continue
    return result


def disk_stats(devices):
    result = {}
    for line in Path('/proc/diskstats').read_text().splitlines():
        fields = line.split()
        name = fields[2]
        if devices and name not in devices:
            continue
        if not devices and (name.startswith('loop') or name.startswith('ram')):
            continue
        values = list(map(int, fields[3:]))
        result[name] = {'reads': values[0], 'read_bytes': values[2] * 512, 'read_ms': values[3],
                        'writes': values[4], 'write_bytes': values[6] * 512, 'write_ms': values[7],
                        'busy_ms': values[9], 'in_flight': values[8]}
    return result


def gpu_stats(gpu):
    if gpu is None or shutil.which('nvidia-smi') is None:
        return None
    command = ['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total', '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=2, check=True)
        fields = [value.strip() for value in result.stdout.strip().split(',')]
        return {'uuid': fields[0], 'name': fields[1], 'utilization_pct': float(fields[2]),
                'memory_activity_pct': float(fields[3]), 'memory_used_mib': float(fields[4]), 'memory_total_mib': float(fields[5])}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def inventory():
    commands = {
        'cpu': ['lscpu'],
        'storage': ['lsblk', '-o', 'NAME,TYPE,SIZE,ROTA,MODEL,FSTYPE,MOUNTPOINTS'],
        'gpu': ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version', '--format=csv'],
        'raid': ['cat', '/proc/mdstat'],
    }
    result = {'hostname': platform.node(), 'platform': platform.platform(), 'python': platform.python_version(),
              'cpu_count': os.cpu_count(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'thread_caps': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    for name, command in commands.items():
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=3)
            result[name] = completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()
        except (OSError, subprocess.SubprocessError) as error:
            result[name] = str(error)
    return result


class TelemetrySink(Sink):
    """Sample resource utilization between the logger's run start and end.

    Args:
        output: Directory for ``machine.json``, ``telemetry.jsonl`` and
            ``summary.json``. Existing artifacts are not overwritten.
        interval: Sampling interval in seconds; must be positive.
        resources: Enable Linux resource probes. If false, record only run
            duration and status.
        gpu: Physical NVIDIA index or UUID. Defaults to the first
            ``CUDA_VISIBLE_DEVICES`` entry, if set.
        io_devices: Disk device names to monitor. An empty sequence selects all
            devices except loop and RAM devices; counters remain per device.

    This sink does not inspect batches or compute training metrics. Register
    it alongside other sinks before ``start_run`` and always call ``end_run``.
    """

    def __init__(self, output, *, interval=1.0, resources=True, gpu=None, io_devices=()):
        if interval <= 0:
            raise ValueError("Telemetry interval must be positive.")
        self.output = Path(output)
        self.interval = float(interval)
        self.resources = bool(resources)
        self.gpu = gpu if gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0] or None
        self.devices = set(io_devices)
        self.pid = os.getpid()
        self.run_name = None
        self.tags = {}
        self.started = None
        self.previous = None
        self.stopped = threading.Event()
        self.thread = None
        self.error = None
        self.handles = {}

    def start(self, run_name, params, tags):
        if self.started is not None:
            raise RuntimeError("Telemetry sink has already started.")
        self.output.mkdir(parents=True, exist_ok=True)
        names = ["summary.json"]
        if self.resources:
            if not Path("/proc/stat").is_file():
                raise RuntimeError("Resource telemetry currently requires Linux /proc; use resources: false elsewhere.")
            names += ["telemetry.jsonl", "machine.json"]
        for name in names:
            if (self.output / name).exists():
                raise FileExistsError(self.output / name)
        self.started = time.monotonic()
        self.run_name = run_name
        self.tags = dict(tags or {})
        if self.resources:
            (self.output / "machine.json").write_text(json.dumps(inventory(), indent=2) + "\n")
            self.handles["resources"] = (self.output / "telemetry.jsonl").open("x")
            self.thread = threading.Thread(target=self.record_resources, daemon=True, name="sleepwalker-telemetry")
            self.thread.start()

    def check_error(self):
        if self.error is not None:
            raise RuntimeError("Telemetry recording failed.") from self.error

    def write(self, name, record):
        try:
            self.handles[name].write(json.dumps(record) + "\n")
            self.handles[name].flush()
        except Exception as error:
            self.error = error
            raise

    def end(self, status="FINISHED"):
        if self.started is None:
            return
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
        try:
            (self.output / "summary.json").write_text(json.dumps({"run_name": self.run_name, "tags": self.tags, "status": "FAILED" if self.error else status, "elapsed_s": time.monotonic() - self.started, "telemetry_error": str(self.error) if self.error else None}, indent=2) + "\n")
        finally:
            for handle in self.handles.values():
                handle.close()
        self.check_error()

    def progress(self, total, desc, leave, formatter):
        return NullProgress()

    def snapshot(self):
        cpu = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            fields = line.split()
            if fields[0].rstrip(":") in {"MemTotal", "MemAvailable", "Cached", "Shmem"}:
                memory[fields[0].rstrip(":") + "_bytes"] = int(fields[1]) * 1024
        return {"timestamp": time.time(), "elapsed_s": time.monotonic() - self.started, "host_cpu_ticks": cpu, "memory": memory, "processes": tree_stats(self.pid), "disks": disk_stats(self.devices), "gpu": gpu_stats(self.gpu)}

    def rates(self, sample):
        if self.previous is None:
            return
        previous = self.previous
        dt = sample["elapsed_s"] - previous["elapsed_s"]
        ticks = [a - b for a, b in zip(sample["host_cpu_ticks"], previous["host_cpu_ticks"])]
        total = sum(ticks)
        sample["host_cpu_utilization_pct"] = 100 * (total - ticks[3] - ticks[4]) / max(total, 1)
        sample["host_iowait_pct"] = 100 * ticks[4] / max(total, 1)
        tick_delta = sum(max(0, process["ticks"] - previous["processes"][pid]["ticks"]) for pid, process in sample["processes"].items() if pid in previous["processes"] and process["start_ticks"] == previous["processes"][pid]["start_ticks"])
        sample["process_tree_cpu_core_pct"] = 100 * tick_delta / os.sysconf("SC_CLK_TCK") / dt
        sample["process_tree_rss_sum_bytes"] = sum(process["rss_bytes"] for process in sample["processes"].values())
        for device, disk in sample["disks"].items():
            if device not in previous["disks"]:
                continue
            old = previous["disks"][device]
            disk["read_bytes_s"] = max(0, disk["read_bytes"] - old["read_bytes"]) / dt
            disk["write_bytes_s"] = max(0, disk["write_bytes"] - old["write_bytes"]) / dt
            disk["busy_pct"] = max(0, disk["busy_ms"] - old["busy_ms"]) / (dt * 10)

    def record_resources(self):
        try:
            while not self.stopped.is_set():
                sample = self.snapshot()
                self.rates(sample)
                self.write("resources", sample)
                self.previous = sample
                self.stopped.wait(self.interval)
        except Exception as error:
            self.error = error
