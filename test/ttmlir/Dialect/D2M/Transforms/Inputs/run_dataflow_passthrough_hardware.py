# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in TTIR-to-device check. Uses the caller's matched ttrt/Metal environment."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

SEED = 20260915
SHAPE = (64, 64)
FIXTURE = Path(__file__).resolve().parents[1] / "dataflow_passthrough_e2e.mlir"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_runtime(metal_home):
    # Select roots before package initialization loads any extension or DSO.
    root_keys = ("TT_METAL_RUNTIME_ROOT", "TT_METAL_RUNTIME_ROOT_EXTERNAL")
    roots = {key: os.environ.get(key) for key in root_keys}
    requested_root = (
        roots["TT_METAL_RUNTIME_ROOT_EXTERNAL"] or roots["TT_METAL_RUNTIME_ROOT"]
    )
    if not requested_root:
        raise RuntimeError("Set TT_METAL_RUNTIME_ROOT to the matched installed runtime")
    os.environ["TT_METAL_HOME"] = str(metal_home)
    library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var(
        "LDLIBRARY"
    )
    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    from ttrt.runtime import _ttmlir_runtime as runtime

    # ttrt package initialization may rewrite these variables for its bundled
    # wheel. Restore the caller's roots, not the source checkout, and reject
    # a DSO loaded from another installation rather than pretending to switch it.
    for key, value in roots.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    os.environ["TT_METAL_HOME"] = str(metal_home)
    metal_libraries = [
        path for path in loaded_libraries() if "libtt_metal" in Path(path).name
    ]
    if not metal_libraries or any(
        not Path(path).resolve().is_relative_to(Path(requested_root).resolve())
        for path in metal_libraries
    ):
        raise RuntimeError("Loaded Metal library is outside the requested runtime root")
    runtime.runtime.set_metal_home(str(metal_home))
    runtime.runtime.set_current_device_runtime(runtime.runtime.DeviceRuntime.TTMetal)
    return runtime


def loaded_libraries():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if path.startswith("/") and any(
            name in path
            for name in (
                "_ttmlir_runtime",
                "libTTMLIRRuntime",
                "libtt_metal",
                "libtt-umd",
            )
        ):
            paths.add(path)
    return {path: digest(path) for path in sorted(paths)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def query(args):
    runtime = load_runtime(args.metal_home)
    descriptor = runtime.runtime.get_current_system_desc()
    descriptor.store(str(args.output / "device.ttsys"))
    data = json.loads(descriptor.as_json())
    chips = data["system_desc"]["chip_descs"]
    if len(data["system_desc"]["chip_desc_indices"]) != 1:
        raise RuntimeError("Expected one visible device")
    for chip in chips:
        if chip["arch"] != "Wormhole_b0" or chip["l1_unreserved_base"] <= 1024:
            raise RuntimeError("Expected a live Wormhole descriptor with reserved L1")
    write_json(
        args.output / "descriptor.json",
        {
            "system_desc": data["system_desc"],
            "libraries": loaded_libraries(),
            "runtime_revision": descriptor.ttmlir_git_hash,
        },
    )


def execute(args):
    import torch

    runtime = load_runtime(args.metal_home)
    from ttrt.common.util import Binary, FileManager, Logger

    binary = Binary(
        Logger(), FileManager(Logger()), str(args.output / f"{args.mode}.ttm")
    )
    descriptor_info = json.loads((args.output / "descriptor.json").read_text())
    if binary.fbb.ttmlir_git_hash != descriptor_info["runtime_revision"]:
        raise RuntimeError(
            "Compiler and runtime revisions differ; rebuild the matched stack"
        )
    torch.manual_seed(SEED)
    inputs = [
        (torch.randn(SHAPE, dtype=torch.bfloat16) * 0.125).contiguous()
        for _ in range(2)
    ]
    reference = -torch.relu(torch.matmul(*inputs))
    options = runtime.runtime.MeshDeviceOptions()
    options.mesh_shape = [1, 1]
    options.mesh_offset = [0, 0]
    options.enable_program_cache = True
    device = runtime.runtime.open_mesh_device(options)
    borrowed, converted, checks = [], [], []
    try:
        for index, tensor in enumerate(inputs):
            host = runtime.runtime.create_borrowed_host_tensor(
                tensor.data_ptr(),
                list(tensor.shape),
                list(tensor.stride()),
                tensor.element_size(),
                Binary.Program.to_data_type(tensor.dtype),
            )
            borrowed.append(host)
            layout = runtime.runtime.get_layout(binary.fbb, 0, index)
            converted.append(runtime.runtime.to_layout(host, device, layout, True))
        for _ in range(args.runs):
            result = runtime.runtime.submit(device, binary.fbb, 0, converted)[0]
            try:
                host = runtime.runtime.to_host(result, untilize=True, blocking=True)[0]
                output = torch.empty(SHAPE, dtype=torch.bfloat16)
                runtime.runtime.memcpy(output.data_ptr(), host)
            finally:
                runtime.runtime.deallocate_tensor(result, force=True)
            actual, expected = output.float(), reference.float()
            pcc = torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[
                0, 1
            ].item()
            close = torch.allclose(actual, expected, rtol=0.02, atol=0.005)
            checks.append(
                {
                    "pcc": pcc,
                    "allclose": close,
                    "max_abs_error": (actual - expected).abs().max().item(),
                    "output_sha256": hashlib.sha256(
                        output.view(torch.uint16).numpy().tobytes()
                    ).hexdigest(),
                    "passed": close and pcc >= 0.999,
                }
            )
            write_json(
                args.output / f"{args.mode}.results.json",
                {
                    "checks": checks,
                    "libraries": loaded_libraries(),
                    "rtol": 0.02,
                    "atol": 0.005,
                    "min_pcc": 0.999,
                },
            )
            if not checks[-1]["passed"]:
                raise RuntimeError(f"Numerical mismatch: {checks[-1]}")
    finally:
        runtime.runtime.close_mesh_device(device)


def git_identity(path):
    def git(*command):
        return subprocess.check_output(
            ["git", "-C", str(path), *command], text=True
        ).strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "status": git("status", "--short"),
        "diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--translate", type=Path, required=True)
    parser.add_argument("--metal-home", type=Path, required=True)
    parser.add_argument(
        "--device", required=True, help="Exact PCI BDF, e.g. 0000:c1:00.0"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New artifact directory"
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--worker", choices=("query", "run"), help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("off", "on"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("compiler", "translate", "metal_home", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.runs < 2:
        parser.error("--runs must be at least 2 to check repeated execution")
    os.environ["TT_VISIBLE_DEVICES"] = args.device
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)
    if args.worker:
        (query if args.worker == "query" else execute)(args)
        return

    devices = [
        p
        for p in Path("/sys/class/tenstorrent").glob("tenstorrent!*")
        if (p / "device").resolve().name == args.device
    ]
    if len(devices) != 1:
        parser.error("--device must identify exactly one local Tenstorrent device")
    kmd = devices[0].name.split("!")[-1]
    busy = subprocess.run(["fuser", "-v", f"/dev/tenstorrent/{kmd}"])
    if busy.returncode != 1:
        raise RuntimeError("Device is busy, or fuser could not confirm it is unused")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "input_source": "checked_in_ttir",
        "fixture_sha256": digest(FIXTURE),
        "runner_sha256": digest(__file__),
        "seed": SEED,
        "runs": args.runs,
        "device": {"kmd": int(kmd), "bdf": args.device},
        "compiler": {"path": str(args.compiler), "sha256": digest(args.compiler)},
        "translate": {"path": str(args.translate), "sha256": digest(args.translate)},
        "source": git_identity(FIXTURE.parents[5]),
        "metal": git_identity(args.metal_home),
        "umd": git_identity(args.metal_home / "tt_metal/third_party/umd"),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
                "TT_VISIBLE_DEVICES",
                "TT_METAL_LOCAL_ONLY",
                "TT_METAL_RUNTIME_ROOT",
                "TT_METAL_RUNTIME_ROOT_EXTERNAL",
            )
        },
        "passed": False,
    }
    write_json(args.output / "manifest.json", manifest)

    def run(name, command, timeout=180):
        with (args.output / f"{name}.log").open("w") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
            )

    worker = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--compiler",
        str(args.compiler),
        "--translate",
        str(args.translate),
        "--metal-home",
        str(args.metal_home),
        "--device",
        args.device,
        "--output",
        str(args.output),
        "--runs",
        str(args.runs),
    ]
    run("query", [*worker, "--worker", "query"])
    descriptor = args.output / "device.ttsys"
    manifest["descriptor_sha256"] = digest(descriptor)
    descriptor_info = json.loads((args.output / "descriptor.json").read_text())
    for mode in ("off", "on"):
        pipeline = f"system-desc-path={descriptor}"
        if mode == "on":
            pipeline += " enable-dataflow-planning=true dump-dataflow-plan=true"
        ir = args.output / f"{mode}.mlir"
        run(
            f"compile-{mode}",
            [
                str(args.compiler),
                str(FIXTURE),
                f"--ttir-to-ttmetal-pipeline={pipeline}",
                "--mlir-print-op-generic",
                "-o",
                str(ir),
            ],
        )
        run(
            f"serialize-{mode}",
            [
                str(args.translate),
                str(ir),
                "--ttmetal-to-flatbuffer",
                "-o",
                str(args.output / f"{mode}.ttm"),
            ],
        )
    if (args.output / "off.mlir").read_bytes() != (
        args.output / "on.mlir"
    ).read_bytes():
        raise RuntimeError("Planner changed final TTMetal IR")
    trace = (args.output / "compile-on.log").read_text()
    for stage in (
        "d2m-dataflow-plan",
        "d2m-dataflow-blocking",
        "d2m-dataflow-allocation",
    ):
        if stage not in trace:
            raise RuntimeError(f"Missing planner stage: {stage}")
    results = {}
    for mode in ("off", "on"):
        run(f"run-{mode}", [*worker, "--worker", "run", "--mode", mode])
        results[mode] = json.loads((args.output / f"{mode}.results.json").read_text())
        if results[mode]["libraries"] != descriptor_info["libraries"]:
            raise RuntimeError("Query and execution used different runtime libraries")
    checks = [check for result in results.values() for check in result["checks"]]
    if len(checks) != 2 * args.runs or not all(check["passed"] for check in checks):
        raise RuntimeError("Missing or failed execution checks")
    if len({check["output_sha256"] for check in checks}) != 1:
        raise RuntimeError("Outputs differ between modes or repetitions")
    manifest.update(
        {
            "passed": True,
            "identical_ir": True,
            "identical_outputs": True,
            "results": results,
        }
    )
    write_json(args.output / "manifest.json", manifest)
    print(f"PASS: planner off/on, {args.runs} runs each; artifacts: {args.output}")


if __name__ == "__main__":
    main()
