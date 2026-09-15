// RUN: %python %S/Inputs/run_dataflow_passthrough_hardware.py --help | FileCheck %s --check-prefix=HELP
// RUN: not %python %S/Inputs/run_dataflow_passthrough_hardware.py --compiler unused --translate unused --metal-home unused --device invalid --output %t --runs 1 2>&1 | FileCheck %s --check-prefix=RUNS
// RUN: not %python %S/Inputs/run_dataflow_passthrough_hardware.py --compiler unused --translate unused --metal-home unused --device invalid --output %t 2>&1 | FileCheck %s --check-prefix=DEVICE
// HELP: Opt-in TTIR-to-device check
// RUNS: --runs must be at least 2
// DEVICE: --device must identify exactly one local Tenstorrent device

// Invalid requests must be rejected before importing the runtime or querying a card.
