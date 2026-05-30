# LatencyServe-Edge

A high-performance, ultra-low-latency inference engine written in native Rust. Designed for hardware-constrained edge deployment environments, LatencyServe-Edge addresses memory-bandwidth bottlenecks and cache starvation by eliminating redundant allocations and localizing memory operations directly within hardware registers.

The architecture pairs OS-level virtual memory mapping with a scenario-aware Mixture-of-Experts (MoE) routing engine to dynamically scale computational workloads between power-efficient single-threaded execution and multi-threaded parallel work-stealing loops.

## Core Architectural Pillars

- **Zero-Copy Tensor Ingestion:** Leverages virtual memory maps via `memmap2` to project raw binary weight files straight into the application's virtual address space. Tensors are exposed as zero-copy array views (`&[f32]`), achieving O(1) load times and bypassing physical RAM reallocation overhead.
- **Inline Graph Fusion:** Eliminates intermediate memory write-backs by fusing adjacent layers (Linear + ReLU) into a single localized runtime loop `Y = max(0, XW + B)`. Accumulation occurs entirely within CPU registers to maintain maximum L1/L2 cache locality.
- **Scenario-Aware MoE Routing:** Inspects incoming feature characteristics (e.g., spatial variance metrics) to intelligently assign inference tasks across execution paths:
  - *Lightweight Expert:* Single-threaded execution track for minimal power draw on simple payloads.
  - *Dense Expert:* Multi-threaded execution track driven by a `rayon` work-stealing parallel engine for massive parameter blocks.

## Directory Layout

```text
latency-serve-edge/
├── Cargo.toml           # Dependency manifests and aggressive release compiler profiles
├── src/
│   ├── lib.rs           # Root library entry point exposing core submodules
│   ├── memory.rs        # Zero-copy memory management and region window views
│   ├── fusion.rs        # Operator fusion structures and Rayon parallel fusers
│   └── routing.rs       # Scenario-aware MoE routing engine mechanics
├── examples/
│   └── benchmark.rs     # Performance execution suite validating processing times
└── tests/
    └── test_server.rs   # Precision arithmetic and boundary validation integration tests
```

## Compilation Profile

To achieve deterministic sub-millisecond execution, the release pipeline relies on strict Link-Time Optimization and aggressive single-codegen-unit loop optimizations:

```toml
[dependencies]
memmap2 = "0.9.10"
rayon  = "1.12.0"
thiserror = "1.0.69"

[profile.release]
opt-level     = 3
lto           = true
codegen-units = 1
panic         = "abort"
```

## Performance Benchmark

Verified on local hardware simulating a **2.09 Million Parameter Matrix (2048 × 1024):**

### Debug Build (`cargo run --example benchmark`)

Unoptimized instruction loops illustrating raw hardware variance before compiler optimizations:

| Expert | Execution Mode | Time |
|---|---|---|
| Lightweight Expert | Single-Thread | ~2.01ms |
| Dense Expert | Multi-Thread | ~1.33ms |

### Release Build (`cargo run --example benchmark --release`)

With `opt-level=3`, `lto=true`, and `codegen-units=1`, the compiler applies function inlining, register tracking, and loop unrolling:

| Expert | Execution Mode | Time |
|---|---|---|
| Lightweight Expert | Single-Thread | **352.8µs** |
| Dense Expert | Multi-Thread Parallel | **1.39ms** |

> Lightweight Expert achieves sub-400µs execution via register-localized fused kernels. Dense Expert distributes multi-million parameter matrices across all available CPU cores via Rayon work-stealing.

## Verification and Testing

The codebase maintains a zero-panic safety record validated by automated integration tests covering arithmetic logic, virtual memory slicing, and routing boundaries.

```bash
cargo test
```

```
running 3 tests
test test_scenario_aware_routing_boundaries ... ok
test test_register_fused_linear_relu_math   ... ok
test test_zero_copy_alignment_and_slicing   ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
```

## Getting Started

**Verify compilation integrity:**
```bash
cargo check
```

**Run optimized performance benchmark:**
```bash
cargo run --example benchmark --release
```

**Run integration test suite:**
```bash
cargo test
```

## License

This project is open-source and available under the MIT License.