#!/usr/bin/env python3
"""Print one compact CPU, memory, disk, GPU, and process snapshot."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""The command samples silently for --interval seconds, prints once, and exits.
Use watch for continuous updates, for example:

  watch -n 2 tools/perf_monitor.py --interval 1

CPU% in the process table uses 100% per fully occupied logical CPU. GPU PCIe RX
and TX are MB/s from the GPU's point of view. CPU/N is the last logical CPU and
its NUMA node. Disk TOTAL sums rates and shows average/maximum UTIL% across the
selected disks. CPU-RAM bandwidth needs hardware counters and is not estimated.
LOAD is the average number of runnable or uninterruptible tasks over 1/5/15 min.
""",
    )
    parser.add_argument("-i", "--interval", type=float, default=1.0, help="measurement window in seconds (default: 1)")
    parser.add_argument("--top", type=int, default=6, help="number of processes to show (default: 6)")
    parser.add_argument("--sort", choices=("cpu", "io", "memory"), default="cpu", help="process sort order (default: cpu)")
    parser.add_argument("--disk", action="append", default=[], help="block device to monitor; repeat for several (default: physical whole disks)")
    parser.add_argument("--no-gpu", action="store_true", help="do not call nvidia-smi")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.top < 0:
        parser.error("--top cannot be negative")
    return args


def read_cpu() -> tuple[int, int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    return sum(values[:8]), values[3], values[4]


def read_memory() -> dict[str, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, value = line.split(":", 1)
        values[name] = int(value.split()[0]) / 1024**2
    used = values["MemTotal"] - values["MemAvailable"]
    return {
        "used": used,
        "total": values["MemTotal"],
        "available": values["MemAvailable"],
        "percent": 100.0 * used / values["MemTotal"],
        "swap_used": values["SwapTotal"] - values["SwapFree"],
        "swap_total": values["SwapTotal"],
    }


def read_cpu_frequencies() -> dict[int, list[float]]:
    packages = {}
    cpu_paths = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"), key=lambda path: int(path.name[3:]))
    for cpu_path in cpu_paths:
        frequency_path = cpu_path / "cpufreq" / "scaling_cur_freq"
        package_path = cpu_path / "topology" / "physical_package_id"
        try:
            frequency_ghz = int(frequency_path.read_text(encoding="utf-8")) / 1_000_000
            package = int(package_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        packages.setdefault(package, []).append(frequency_ghz)
    return packages


def read_temperatures(disk_names: list[str]) -> tuple[dict[str, float], float | None, list[tuple[str, float]]]:
    disk_devices = {Path("/sys/class/block", name, "device").resolve(): name for name in disk_names}
    disk_temperatures = {}
    cpu_temperatures = []
    other_temperatures = []
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            name = (hwmon / "name").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        sensors = []
        for input_path in sorted(hwmon.glob("temp*_input")):
            label_path = input_path.with_name(input_path.name.replace("_input", "_label"))
            try:
                temperature = int(input_path.read_text(encoding="utf-8")) / 1000
                label = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else input_path.stem
                sensors.append((label, temperature))
            except (FileNotFoundError, ValueError):
                continue
        if not sensors:
            continue
        device = (hwmon / "device").resolve()
        if name == "nvme" and device in disk_devices:
            composite = next((temperature for label, temperature in sensors if label.lower() == "composite"), sensors[0][1])
            disk_temperatures[disk_devices[device]] = composite
        elif name in {"k10temp", "coretemp", "zenpower"}:
            cpu_temperatures.extend(temperature for label, temperature in sensors if not label.lower().startswith("ccd"))
        else:
            label, temperature = max(sensors, key=lambda sensor: sensor[1])
            other_temperatures.append((f"{name}@{device.name}/{label}", temperature))
    return disk_temperatures, max(cpu_temperatures, default=None), other_temperatures


def read_cpu_nodes() -> dict[int, int]:
    nodes = {}
    for cpu_path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        node_path = next(cpu_path.glob("node[0-9]*"), None)
        if node_path is not None:
            nodes[int(cpu_path.name[3:])] = int(node_path.name[4:])
    return nodes


def device_numa_node(device: Path) -> int | None:
    resolved = device.resolve()
    for path in (resolved, *resolved.parents):
        numa_path = path / "numa_node"
        if numa_path.exists():
            try:
                node = int(numa_path.read_text(encoding="utf-8"))
                return None if node < 0 else node
            except (FileNotFoundError, ValueError):
                return None
    return None


def read_numa_topology(disk_names: list[str], gpus: list[dict[str, float | str | None]]) -> list[dict[str, float | int | str]]:
    disks_by_node = {}
    for disk in disk_names:
        node = device_numa_node(Path("/sys/class/block", disk, "device"))
        disks_by_node.setdefault(node, []).append(disk)
    gpus_by_node = {}
    for gpu in gpus:
        pci_bus = gpu.get("pci_bus")
        if pci_bus is None:
            continue
        node = device_numa_node(Path("/sys/bus/pci/devices", str(pci_bus)))
        gpus_by_node.setdefault(node, []).append(str(gpu["id"]))

    rows = []
    for node_path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*"), key=lambda path: int(path.name[4:])):
        node = int(node_path.name[4:])
        cpu_list = (node_path / "cpulist").read_text(encoding="utf-8").strip()
        sockets = set()
        for cpu_path in node_path.glob("cpu[0-9]*"):
            package_path = cpu_path / "topology" / "physical_package_id"
            if package_path.exists():
                sockets.add(int(package_path.read_text(encoding="utf-8")))
        memory_line = next(line for line in (node_path / "meminfo").read_text(encoding="utf-8").splitlines() if "MemTotal" in line)
        memory_gib = int(memory_line.split()[-2]) / 1024**2
        rows.append({"node": node, "sockets": ",".join(str(socket) for socket in sorted(sockets)) or "-", "cpus": cpu_list, "memory": memory_gib, "gpus": ",".join(gpus_by_node.get(node, [])) or "-", "disks": ",".join(disks_by_node.get(node, [])) or "-"})
    return rows


def select_disks(requested: list[str]) -> list[str]:
    if requested:
        missing = [name for name in requested if not Path("/sys/class/block", name).exists()]
        if missing:
            raise ValueError(f"Unknown block device(s): {', '.join(missing)}")
        return requested

    block_devices = sorted(Path("/sys/class/block").iterdir())
    physical = [path.name for path in block_devices if (path / "device").exists() and not (path / "partition").exists()]
    if physical:
        return physical
    return [path.name for path in block_devices if not (path / "partition").exists() and not path.name.startswith(("loop", "ram", "zram", "fd", "sr"))]


def read_disks(names: list[str]) -> dict[str, tuple[int, int, int]]:
    wanted = set(names)
    counters = {}
    for line in Path("/proc/diskstats").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[2] in wanted:
            counters[fields[2]] = (int(fields[5]), int(fields[9]), int(fields[12]))
    missing = wanted - counters.keys()
    if missing:
        raise RuntimeError(f"No /proc/diskstats counters for: {', '.join(sorted(missing))}")
    return counters


def calculate_disks(previous: dict[str, tuple[int, int, int]], current: dict[str, tuple[int, int, int]], elapsed: float) -> list[dict[str, float | str]]:
    rows = []
    for name in sorted(current):
        read_sectors, written_sectors, busy_ms = current[name]
        old_read, old_written, old_busy = previous[name]
        rows.append({
            "name": name,
            "util": min(100.0, 100.0 * max(0, busy_ms - old_busy) / (elapsed * 1000)),
            "read": max(0, read_sectors - old_read) * 512 / elapsed / 1024**2,
            "write": max(0, written_sectors - old_written) * 512 / elapsed / 1024**2,
        })
    return rows


def read_processes(cpu_nodes: dict[int, int]) -> dict[int, dict[str, int | str | None]]:
    processes = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat = (directory / "stat").read_text(encoding="utf-8")
            close_paren = stat.rfind(")")
            fields = stat[close_paren + 2 :].split()
            pid = int(directory.name)
            command_bytes = (directory / "cmdline").read_bytes().replace(b"\0", b" ").strip()
            command = command_bytes.decode(errors="replace") or f"[{stat[stat.find('(') + 1 : close_paren]}]"
            command = " ".join(command.split())
            read_bytes = None
            write_bytes = None
            try:
                io_values = {}
                for line in (directory / "io").read_text(encoding="utf-8").splitlines():
                    name, value = line.split(":", 1)
                    io_values[name] = int(value)
                read_bytes = io_values["read_bytes"]
                write_bytes = io_values["write_bytes"]
            except PermissionError:
                pass
            processes[pid] = {
                "ticks": int(fields[11]) + int(fields[12]),
                "rss": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
                "processor": int(fields[36]),
                "node": cpu_nodes.get(int(fields[36])),
                "read": read_bytes,
                "write": write_bytes,
                "command": command,
            }
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return processes


def calculate_processes(previous: dict[int, dict[str, int | str | None]], current: dict[int, dict[str, int | str | None]], elapsed: float, memory_bytes: float) -> list[dict[str, float | int | str | None]]:
    rows = []
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    for pid, values in current.items():
        old = previous.get(pid)
        if old is None:
            continue
        read_rate = None if values["read"] is None or old["read"] is None else max(0, int(values["read"]) - int(old["read"])) / elapsed / 1024**2
        write_rate = None if values["write"] is None or old["write"] is None else max(0, int(values["write"]) - int(old["write"])) / elapsed / 1024**2
        rows.append({
            "pid": pid,
            "cpu": 100.0 * max(0, int(values["ticks"]) - int(old["ticks"])) / ticks_per_second / elapsed,
            "memory": 100.0 * int(values["rss"]) / memory_bytes,
            "rss": int(values["rss"]) / 1024**3,
            "processor": int(values["processor"]),
            "node": values["node"],
            "read": read_rate,
            "write": write_rate,
            "io": (read_rate or 0.0) + (write_rate or 0.0),
            "command": str(values["command"]),
        })
    return rows


def number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def read_gpu_query() -> list[dict[str, float | str | None]] | None:
    fields = "index,pci.bus_id,utilization.gpu,utilization.memory,memory.used"
    try:
        result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 5:
            pci_parts = values[1].lower().split(":")
            pci_bus = f"{pci_parts[0][-4:]}:{pci_parts[1]}:{pci_parts[2]}"
            rows.append({"id": values[0], "pci_bus": pci_bus, "gpu": number(values[2]), "mem": number(values[3]), "fb": number(values[4]), "rx": None, "tx": None})
    return rows or None


def read_gpu_dmon() -> list[dict[str, float | str | None]] | None:
    try:
        result = subprocess.run(["nvidia-smi", "dmon", "-s", "pucmt", "-c", "1"], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    header = next((line.lstrip()[1:].split() for line in result.stdout.splitlines() if line.lstrip().startswith("# gpu")), None)
    if header is None:
        return None
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != len(header):
            continue
        values = dict(zip(header, fields))
        rows.append({"id": values["gpu"], "gpu": number(values.get("sm", "-")), "mem": number(values.get("mem", "-")), "fb": number(values.get("fb", "-")), "rx": number(values.get("rxpci", "-")), "tx": number(values.get("txpci", "-")), "temperature": number(values.get("gtemp", "-")), "memory_temperature": number(values.get("mtemp", "-")), "core_clock": number(values.get("pclk", "-")), "memory_clock": number(values.get("mclk", "-"))})
    return rows or None


def gpu_snapshot(disabled: bool) -> tuple[list[dict[str, float | str | None]], str | None]:
    if disabled:
        return [], "disabled"
    if shutil.which("nvidia-smi") is None:
        return [], "nvidia-smi not found"
    samples = read_gpu_dmon()
    gpu_info = read_gpu_query()
    if samples is not None:
        info_by_id = {str(gpu["id"]): gpu for gpu in gpu_info or []}
        for sample in samples:
            sample.update({"pci_bus": info_by_id.get(str(sample["id"]), {}).get("pci_bus")})
        return samples, None
    if gpu_info is not None:
        return gpu_info, "PCIe counters unavailable"
    return [], "nvidia-smi cannot access the NVIDIA driver"


def format_rate(value: float | None, width: int) -> str:
    return "-".rjust(width) if value is None else f"{value:.1f}".rjust(width)


def print_snapshot(cpu_percent: float, wait_percent: float, memory: dict[str, float], cpu_frequencies: dict[int, list[float]], cpu_temperature: float | None, numa_topology: list[dict[str, float | int | str]], disk_temperatures: dict[str, float], other_temperatures: list[tuple[str, float]], disks: list[dict[str, float | str]], gpus: list[dict[str, float | str | None]], gpu_note: str | None, processes: list[dict[str, float | int | str | None]], args: argparse.Namespace) -> None:
    cpu_count = os.cpu_count() or 1
    load = os.getloadavg()
    terminal_width = shutil.get_terminal_size((90, 24)).columns
    print(f"PERFORMANCE SNAPSHOT  {datetime.now():%Y-%m-%d %H:%M:%S}  ({args.interval:g}s window)")
    print(f"CPU  {cpu_percent:5.1f}% busy ({cpu_percent * cpu_count / 100:5.1f}/{cpu_count} logical CPUs)  I/O wait {wait_percent:4.1f}%")
    print(f"LOAD avg tasks {load[0]:.1f}/{load[1]:.1f}/{load[2]:.1f} (1m/5m/15m)")
    if cpu_frequencies:
        all_frequencies = [frequency for frequencies in cpu_frequencies.values() for frequency in frequencies]
        packages = "  ".join(f"socket{package} {sum(frequencies) / len(frequencies):.2f}" for package, frequencies in sorted(cpu_frequencies.items()))
        temperature = "n/a" if cpu_temperature is None else f"{cpu_temperature:.1f} C"
        print(f"CLK  CPU avg {sum(all_frequencies) / len(all_frequencies):.2f} GHz ({min(all_frequencies):.2f}-{max(all_frequencies):.2f})  {packages}  temp {temperature}")
    print(f"MEM  {memory['used']:.1f}/{memory['total']:.1f} GiB used ({memory['percent']:.1f}%)  avail {memory['available']:.1f} GiB  swap {memory['swap_used']:.1f}/{memory['swap_total']:.1f} GiB")

    print(f"\n{'NUMA':>4} {'SOCKET':>6} {'LOGICAL CPUS':<14} {'RAM GiB':>7} {'GPUS':<6} DISKS")
    for node in numa_topology:
        print(f"{int(node['node']):4d} {str(node['sockets']):>6} {str(node['cpus']):<14} {float(node['memory']):7.1f} {str(node['gpus']):<6} {node['disks']}")

    print(f"\n{'DISK':<13} {'UTIL%':>11} {'READ MiB/s':>12} {'WRITE MiB/s':>12} {'TEMP C':>7}")
    for disk in disks:
        temperature = disk_temperatures.get(str(disk["name"]))
        print(f"{str(disk['name'])[:13]:<13} {float(disk['util']):11.1f} {float(disk['read']):12.1f} {float(disk['write']):12.1f} {format_rate(temperature, 7)}")
    if not disks:
        print("No block devices selected")
    else:
        average_util = sum(float(disk["util"]) for disk in disks) / len(disks)
        maximum_util = max(float(disk["util"]) for disk in disks)
        print(f"{'TOTAL avg/max':<13} {f'{average_util:.1f}/{maximum_util:.1f}':>11} {sum(float(disk['read']) for disk in disks):12.1f} {sum(float(disk['write']) for disk in disks):12.1f} {'-':>7}")

    print("\nGPU   SM%  MEM%  FB GiB  RX MB/s  TX MB/s  TEMP G/M C  CORE/MEM MHz")
    for gpu in gpus:
        fb = "-" if gpu.get("fb") is None else f"{float(gpu['fb']) / 1024:.1f}"
        gpu_temperature = "-" if gpu.get("temperature") is None else f"{float(gpu['temperature']):.0f}"
        memory_temperature = "-" if gpu.get("memory_temperature") is None else f"{float(gpu['memory_temperature']):.0f}"
        core_clock = "-" if gpu.get("core_clock") is None else f"{float(gpu['core_clock']):.0f}"
        memory_clock = "-" if gpu.get("memory_clock") is None else f"{float(gpu['memory_clock']):.0f}"
        print(f"{str(gpu['id']):>3} {format_rate(gpu.get('gpu'), 5)} {format_rate(gpu.get('mem'), 5)} {fb:>7} {format_rate(gpu.get('rx'), 8)} {format_rate(gpu.get('tx'), 8)} {f'{gpu_temperature}/{memory_temperature}':>11} {f'{core_clock}/{memory_clock}':>13}")
    if not gpus:
        print(f"GPU metrics unavailable: {gpu_note}")
    elif gpu_note:
        print(f"Note: {gpu_note}")

    if other_temperatures:
        print("\nOTHER DEVICE TEMPERATURES (kernel hwmon)")
        print("DEVICE@PCI/SENSOR                    TEMP C")
        for name, temperature in other_temperatures:
            print(f"{name[:35]:<35} {temperature:6.1f}")

    sort_key = {"cpu": "cpu", "io": "io", "memory": "rss"}[args.sort]
    processes.sort(key=lambda row: float(row[sort_key]), reverse=True)
    command_width = max(12, terminal_width - 60)
    print(f"\nPROCESSES  top {args.top} by {args.sort}  (CPU%=one logical CPU)")
    print(f"{'PID':>7} {'CPU/N':>5} {'CPU%':>6} {'MEM%':>5} {'RSS GiB':>7} {'READ M/s':>9} {'WRITE M/s':>10}  COMMAND")
    for process in processes[: args.top]:
        command = str(process["command"])
        if len(command) > command_width:
            command = command[: max(1, command_width - 1)] + "…"
        cpu_node = f"{int(process['processor'])}/{process['node'] if process['node'] is not None else '-'}"
        print(f"{int(process['pid']):7d} {cpu_node:>5} {float(process['cpu']):6.1f} {float(process['memory']):5.1f} {float(process['rss']):7.2f} {format_rate(process.get('read'), 9)} {format_rate(process.get('write'), 10)}  {command}")


def main() -> None:
    args = parse_args()
    disk_names = select_disks(args.disk)
    cpu_nodes = read_cpu_nodes()
    previous_processes = read_processes(cpu_nodes)
    previous_disks = read_disks(disk_names)
    previous_cpu = read_cpu()
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=1) as executor:
        gpu_future = executor.submit(gpu_snapshot, args.no_gpu)
        time.sleep(args.interval)
        current_cpu = read_cpu()
        elapsed = time.monotonic() - started
        current_disks = read_disks(disk_names)
        current_processes = read_processes(cpu_nodes)
        memory = read_memory()
        cpu_frequencies = read_cpu_frequencies()
        disk_temperatures, cpu_temperature, other_temperatures = read_temperatures(disk_names)
        gpus, gpu_note = gpu_future.result()
        numa_topology = read_numa_topology(disk_names, gpus)

    total_delta = current_cpu[0] - previous_cpu[0]
    idle_delta = current_cpu[1] - previous_cpu[1]
    wait_delta = current_cpu[2] - previous_cpu[2]
    cpu_percent = 100.0 * (total_delta - idle_delta - wait_delta) / total_delta if total_delta else 0.0
    wait_percent = 100.0 * wait_delta / total_delta if total_delta else 0.0
    disks = calculate_disks(previous_disks, current_disks, elapsed)
    processes = calculate_processes(previous_processes, current_processes, elapsed, memory["total"] * 1024**3)
    print_snapshot(cpu_percent, wait_percent, memory, cpu_frequencies, cpu_temperature, numa_topology, disk_temperatures, other_temperatures, disks, gpus, gpu_note, processes, args)


if __name__ == "__main__":
    main()
