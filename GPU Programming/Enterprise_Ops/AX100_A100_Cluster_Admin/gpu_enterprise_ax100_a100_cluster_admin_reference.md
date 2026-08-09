# GPU Infrastructure Reference Manual: AX100/A100 Nodes

### *200+ commands for enterprise-scale embedding passes, multi-node training, and LLM orchestration on AX100/A100 clusters*

> **Nav:** [← Cluster Admin README](README.md) | **Reference Manual**

---

## Table of Contents

1. [Exhaustive NVIDIA Core Suite & CLI Commands](#1-exhaustive-nvidia-core-suite--cli-commands)
2. [Enterprise GPU System Administration & Node Recovery](#2-enterprise-gpu-system-administration--node-recovery)
3. [Under the Hood: Deep-Dive Optimization & Runtime Tuning](#3-under-the-hood-deep-dive-optimization--runtime-tuning)
4. [Storage, Network Fabric, & Compute Health Validation](#4-storage-network-fabric--compute-health-validation)
5. [Appendix A: Verified Real-World Session Capture (A100-SXM4-80GB)](#appendix-a-verified-real-world-session-capture-a100-sxm4-80gb)

---

## 1. Exhaustive NVIDIA Core Suite & CLI Commands

### 1.1 `nvidia-smi` Monitoring & Querying

**Baseline invocations**

```bash
nvidia-smi
```
Full default report: driver/CUDA version, per-GPU temperature, power draw, memory usage, and running process list — the single most common first command in any triage session.

```bash
nvidia-smi -L
```
Lists every GPU (and MIG instance, if enabled) by index and UUID, without the full table — useful for scripting device enumeration.

```bash
nvidia-smi -i 0
```
Scopes ALL subsequent flags to physical GPU index 0 only, essential on multi-GPU nodes to avoid ambiguous output.

**Streaming and looped monitoring**

```bash
nvidia-smi -l 1
```
Loops the full report every 1 second indefinitely (Ctrl+C to stop) — the `-l` (loop) flag is the classic "leave it running in a tmux pane" monitoring command.

```bash
nvidia-smi -l 1 -f /var/log/gpu_monitor.log
```
Same 1-second loop, but redirects (`-f`) all output to a log file instead of stdout, for unattended background logging during long training runs.

```bash
watch -n 0.5 nvidia-smi
```
An alternative to `-l` using the standalone `watch` utility, refreshing every 0.5 seconds with a clear-screen redraw each cycle — preferred when you want `watch`'s highlighting-of-changes behavior.

**DMON — device monitor, compact streaming table**

```bash
nvidia-smi dmon
```
Launches the compact device-monitor mode: one continuously updating row per GPU per second, showing SM%, memory%, encoder%, decoder%, temperature, and power in a fixed-width table — far lighter-weight than the full `nvidia-smi` report for continuous background monitoring.

```bash
nvidia-smi dmon -i 0,1,2,3
```
Restricts `dmon` streaming to GPUs 0-3 specifically, useful on an 8-GPU node when you only care about one NUMA-local half.

```bash
nvidia-smi dmon -s pucvmet
```
Selects specific metric groups to display: `p`=power, `u`=utilization, `c`=clocks, `v`=violations (throttle reasons), `m`=memory, `e`=ECC errors, `t`=temperature — letting you build a custom column set instead of the default.

```bash
nvidia-smi dmon -d 5
```
Sets the delay between samples to 5 seconds (default is 1), reducing sampling overhead for long unattended monitoring sessions.

```bash
nvidia-smi dmon -c 100
```
Collects exactly 100 samples then exits automatically — useful for scripted, bounded data-collection windows instead of an infinite stream.

```bash
nvidia-smi dmon -o TD
```
Prepends each output row with a timestamp (`T`) and elapsed-time-since-start (`D`), critical for correlating dmon output against application-side logs during post-mortem analysis.

**PMON — per-process monitor**

```bash
nvidia-smi pmon
```
Streams per-process GPU utilization (SM%, memory%, encoder%, decoder%) for every process currently touching the GPU — the process-level equivalent of `dmon`, essential for identifying which specific PID is hogging compute on a shared node.

```bash
nvidia-smi pmon -i 0 -s um
```
Scopes PMON to GPU 0 and selects only utilization (`u`) and memory (`m`) metric groups, trimming the output to the two columns most relevant for spotting a runaway process.

```bash
nvidia-smi pmon -c 50 -d 2
```
Collects 50 samples at 2-second intervals then exits — combines the count and delay controls for a bounded per-process capture window.

**Custom CSV querying — `--query-gpu`**

```bash
nvidia-smi --query-gpu=utilization.gpu --format=csv
```
Query property #1: current SM (compute core) utilization percentage, the single most-watched metric for "is the GPU actually being used."

```bash
nvidia-smi --query-gpu=utilization.memory --format=csv
```
Query property #2: percentage of time the memory subsystem (not the compute cores) was busy over the last sample period — a high value here with low `utilization.gpu` often signals a memory-bandwidth-bound kernel.

```bash
nvidia-smi --query-gpu=memory.used --format=csv
```
Query property #3: total VRAM currently allocated across all processes on the GPU, in MiB — the number to watch when hunting for a memory leak.

```bash
nvidia-smi --query-gpu=memory.total --format=csv
```
Query property #4: the GPU's total physical VRAM capacity — for an 80GB A100, this reports 81920 MiB (accounting for binary vs decimal unit conventions).

```bash
nvidia-smi --query-gpu=memory.free --format=csv
```
Query property #5: VRAM currently unallocated and available for new processes — the number a scheduler checks before placing a new job on a node.

```bash
nvidia-smi --query-gpu=memory.reserved --format=csv
```
Query property #6: VRAM reserved by the driver itself for internal bookkeeping (page tables, ECC metadata) and unavailable to any user process, regardless of `memory.free`.

```bash
nvidia-smi --query-gpu=temperature.gpu --format=csv
```
Query property #7: current core die temperature in degrees Celsius — the primary thermal-throttling trigger metric.

```bash
nvidia-smi --query-gpu=temperature.memory --format=csv
```
Query property #8: HBM2e memory-die temperature specifically, which can throttle independently of the core temperature on A100-class GPUs.

```bash
nvidia-smi --query-gpu=power.draw --format=csv
```
Query property #9: instantaneous power consumption in watts, sampled at the driver's polling interval — the direct input to power-cap enforcement logic.

```bash
nvidia-smi --query-gpu=power.limit --format=csv
```
Query property #10: the currently configured software power cap in watts (settable via `-pl`, covered in §1.2) — distinct from the hardware maximum.

```bash
nvidia-smi --query-gpu=power.default_limit --format=csv
```
Query property #11: the factory-default power limit the card ships with, useful as a restoration target after temporary overrides.

```bash
nvidia-smi --query-gpu=power.max_limit --format=csv
```
Query property #12: the absolute hardware ceiling for the power limit — attempting to set `-pl` above this value is rejected by the driver.

```bash
nvidia-smi --query-gpu=clocks.sm --format=csv
```
Query property #13: current Streaming Multiprocessor clock frequency in MHz — this is the number that dips under thermal or power throttling.

```bash
nvidia-smi --query-gpu=clocks.mem --format=csv
```
Query property #14: current memory (HBM) clock frequency in MHz.

```bash
nvidia-smi --query-gpu=clocks.max.sm --format=csv
```
Query property #15: the maximum SM clock the silicon is rated for — the ceiling `nvidia-smi -lgc` (locked GPU clocks) can be set to.

```bash
nvidia-smi --query-gpu=clocks.max.memory --format=csv
```
Query property #16: the maximum memory clock the HBM stack is rated for.

```bash
nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv
```
Query property #17: a bitmask of ALL currently active throttle reasons simultaneously (thermal, power, sw-power-cap, hw-slowdown) — the single most useful diagnostic field for "why is my GPU slow right now."

```bash
nvidia-smi --query-gpu=clocks_throttle_reasons.sw_power_cap --format=csv
```
Query property #18: boolean flag isolating specifically whether the SOFTWARE power limit (set via `-pl`) is the active throttle cause, distinguishing it from hardware-level thermal slowdown.

```bash
nvidia-smi --query-gpu=clocks_throttle_reasons.hw_slowdown --format=csv
```
Query property #19: boolean flag for hardware-level thermal or power slowdown, triggered by the card's own protection circuitry independent of any software-configured limit.

```bash
nvidia-smi --query-gpu=pcie.link.gen.current --format=csv
```
Query property #20: the CURRENTLY NEGOTIATED PCIe generation (e.g., 4 for Gen4) — critical to check after a reboot or reseat, since a card can silently re-link at a lower generation than it supports.

```bash
nvidia-smi --query-gpu=pcie.link.gen.max --format=csv
```
Query property #21: the maximum PCIe generation the card's silicon supports, the ceiling to compare `pcie.link.gen.current` against.

```bash
nvidia-smi --query-gpu=pcie.link.width.current --format=csv
```
Query property #22: currently negotiated PCIe lane width (e.g., x16) — a card silently running at x8 or x4 is a classic, easy-to-miss cause of host-to-device transfer bottlenecks.

```bash
nvidia-smi --query-gpu=pcie.link.width.max --format=csv
```
Query property #23: the maximum PCIe lane width the card supports, the ceiling to compare current width against.

```bash
nvidia-smi --query-gpu=ecc.mode.current --format=csv
```
Query property #24: whether ECC (Error-Correcting Code) memory protection is currently enabled — should read "Enabled" on any production A100 node.

```bash
nvidia-smi --query-gpu=ecc.mode.pending --format=csv
```
Query property #25: a PENDING ECC mode change that requires a GPU reset (or full reboot) to take effect — a nonzero/differing value here means a toggle was issued but not yet applied.

```bash
nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total --format=csv
```
Query property #26: cumulative single-bit (corrected) ECC errors since the last driver load — generally benign in small numbers, but a rapidly climbing counter signals a degrading memory cell.

```bash
nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv
```
Query property #27: cumulative double-bit (uncorrected) ECC errors since last driver load — any nonzero value here means data corruption occurred and should trigger immediate node investigation.

```bash
nvidia-smi --query-gpu=fan.speed --format=csv
```
Query property #28: current fan speed as a percentage of maximum (on actively-cooled SKUs; passively-cooled datacenter A100s report N/A here since cooling is chassis-level).

```bash
nvidia-smi --query-gpu=compute_mode --format=csv
```
Query property #29: the current compute mode (Default/Exclusive_Process/Prohibited — see §1.2) governing how many processes can concurrently open a CUDA context on this GPU.

```bash
nvidia-smi --query-gpu=persistence_mode --format=csv
```
Query property #30: whether persistence mode is enabled — when on, the driver stays loaded between CUDA jobs, eliminating the multi-second driver re-initialization latency at the start of every new process.

```bash
nvidia-smi --query-gpu=mig.mode.current --format=csv
```
Query property #31: whether Multi-Instance GPU mode is currently active on this physical GPU.

```bash
nvidia-smi --query-gpu=uuid --format=csv
```
Query property #32: the GPU's globally unique identifier string, stable across reboots — the correct key to use when correlating a GPU across `nvidia-smi`, Kubernetes device-plugin logs, and DCGM output.

```bash
nvidia-smi --query-gpu=index --format=csv
```
Query property #33: the GPU's numeric PCI-enumeration-order index — NOT guaranteed stable across reboots, unlike UUID, so never use this as a persistent identifier in automation.

```bash
nvidia-smi --query-gpu=name --format=csv
```
Query property #34: the human-readable product name string (e.g., "NVIDIA A100-SXM4-80GB").

```bash
nvidia-smi --query-gpu=driver_version --format=csv
```
Query property #35: the installed NVIDIA display driver version, essential for compatibility triage against a target CUDA toolkit version.

```bash
nvidia-smi --query-gpu=vbios_version --format=csv
```
Query property #36: the card's video BIOS firmware version — relevant when chasing down known firmware-level erratas.

```bash
nvidia-smi --query-gpu=utilization.encoder --format=csv
```
Query property #37: hardware video encoder engine utilization — near-zero on pure AI/HPC workloads, relevant only if the node is also doing media transcoding.

```bash
nvidia-smi --query-gpu=utilization.decoder --format=csv
```
Query property #38: hardware video decoder engine utilization, same caveat as above.

**Combining multiple properties and full CSV plumbing**

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv
```
Combines seven properties into one CSV row per GPU — the standard "at a glance" custom dashboard query, easily piped into a log aggregator.

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 >> /var/log/gpu_timeseries.csv
```
Streams a timestamped 3-column time series every 5 seconds, appending (`>>`) to a growing CSV log file — a common lightweight alternative to a full DCGM/Prometheus stack for smaller clusters.

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```
The `noheader` format modifier strips the column-name header row, producing raw data lines only — essential when piping output directly into a script's parsing loop.

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
```
Adds `nounits`, additionally stripping the "%" and "MiB" unit suffixes so the output is pure numeric CSV, ready for direct `awk`/`cut`/Python ingestion without a regex-strip step.

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```
A distinct query namespace from `--query-gpu`: reports per-PROCESS VRAM consumption (PID, process name, memory used) for every active CUDA context on the node — the direct tool for "which process is eating my VRAM."

```bash
nvidia-smi --query-accounted-apps=pid,gpu_utilization,mem_utilization,max_memory_usage --format=csv
```
Queries the GPU accounting subsystem (must be enabled via `nvidia-smi --accounting-mode=1` first) for historical per-process stats, including processes that have already terminated — useful for post-mortem "which job actually used the GPU" audits.

---

### 1.2 Hardware Management & Toggling

**Compute mode**

```bash
sudo nvidia-smi -c 0
```
Sets compute mode to `0` (Default): unrestricted, any number of processes may simultaneously open contexts on the GPU — the standard mode for shared, multi-tenant nodes.

```bash
sudo nvidia-smi -c 1
```
Sets compute mode to `1` (Exclusive_Thread): only one CUDA context may be open on the GPU at a time, and only from the thread that created it — largely legacy, rarely used on modern multi-threaded frameworks.

```bash
sudo nvidia-smi -c 3
```
Sets compute mode to `3` (Exclusive_Process): only one process (regardless of thread count within it) may hold a CUDA context — the standard mode for hard workload isolation on a dedicated single-tenant GPU.

```bash
sudo nvidia-smi -c 2
```
Sets compute mode to `2` (Prohibited): no new CUDA contexts may be created at all — used to safely cordon a GPU during maintenance without physically removing it from the bus.

**Persistence mode**

```bash
sudo nvidia-smi -pm 1
```
Enables persistence mode, keeping the NVIDIA driver resident in memory between CUDA application runs — dramatically reduces cold-start latency (avoids the multi-second driver re-init penalty) and should be enabled on every production training node at boot.

```bash
sudo nvidia-smi -pm 0
```
Disables persistence mode, reverting to the default behavior of unloading the driver state when no process holds a context.

**MIG — Multi-Instance GPU**

```bash
sudo nvidia-smi -mig 1
```
Enables MIG mode on the target GPU (requires `-i` to scope, or applies node-wide) — this REQUIRES a subsequent GPU reset (or reboot) before instance creation is possible.

```bash
sudo nvidia-smi -mig 0
```
Disables MIG mode, returning the GPU to a single monolithic compute device — also requires a reset to take effect.

```bash
nvidia-smi mig -lgip
```
Lists every valid GPU Instance Profile the card supports (e.g., `1g.10gb`, `2g.20gb`, `3g.40gb`, `4g.40gb`, `7g.80gb` on an 80GB A100), along with the maximum instance count per profile.

```bash
sudo nvidia-smi mig -cgi 19,14,5 -C
```
Creates three GPU Instances by profile ID in one command (`19`=e.g. `1g.10gb`, `14`=`2g.20gb`, `5`=`3g.40gb` — IDs vary by SKU, always cross-check against `-lgip`), and the `-C` flag additionally auto-creates a default Compute Instance inside each — the fastest one-shot way to stand up a mixed-profile MIG layout.

```bash
nvidia-smi mig -lgi
```
Lists all currently instantiated GPU Instances and their physical placement/slot on the die.

```bash
nvidia-smi mig -lci
```
Lists all currently instantiated Compute Instances nested within each GPU Instance.

```bash
sudo nvidia-smi mig -i 0 -gi 5 -ci 0 -dc
```
Destroys Compute Instance 0 inside GPU Instance 5 on physical GPU 0, freeing the compute slice while leaving the GPU Instance's memory partition intact.

```bash
sudo nvidia-smi mig -i 0 -gi 5 -dgi
```
Destroys GPU Instance 5 itself (its Compute Instances must be cleared first), fully releasing that die real estate back to the pool of available profiles.

```bash
sudo nvidia-smi mig -i 0 -dgi
```
Destroys ALL GPU Instances on physical GPU 0 in a single command — the fast path to reset a card's MIG layout from scratch before reconfiguring.

**ECC memory state**

```bash
sudo nvidia-smi -e 1
```
Enables ECC (Error-Correcting Code) memory protection — mandatory for any production A100 handling financially or scientifically significant workloads; requires a GPU reset to activate.

```bash
sudo nvidia-smi -e 0
```
Disables ECC — trades a small amount of memory bandwidth and capacity (ECC overhead reserves some HBM) for raw throughput; almost never appropriate in production, occasionally used in pure-throughput benchmarking.

```bash
sudo nvidia-smi -p 0
```
Clears (resets) the ECC error counters for the GPU back to zero, typically run immediately after logging/investigating an error spike, so subsequent monitoring windows report only new events.

**Power limits**

```bash
sudo nvidia-smi -pl 300
```
Sets the software power limit to 300 watts for the targeted GPU — the primary lever for thermal/power budget management in dense multi-GPU chassis where the datacenter's power delivery is the binding constraint, not the silicon.

```bash
sudo nvidia-smi -i 0 -pl 250
```
Scopes the power-limit change to GPU 0 specifically, leaving other GPUs on the node at their existing limits — essential in mixed-workload nodes running different jobs per GPU.

```bash
sudo nvidia-smi -pl $(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits)
```
Restores the power limit to the factory default by piping the queried default value directly back into `-pl` — a clean one-liner for "undo any temporary power-cap override."

**Clock locking**

```bash
sudo nvidia-smi -lgc 1200,1410
```
Locks the SM (core) clock to a range between 1200 MHz (minimum) and 1410 MHz (maximum) — used to eliminate clock-frequency VARIANCE between runs, which is critical for reproducible benchmarking where run-to-run noise from dynamic boost behavior would otherwise contaminate timing comparisons.

```bash
sudo nvidia-smi -lgc 1410
```
Locks the SM clock to a SINGLE fixed frequency (min=max=1410) rather than a range, for maximum determinism.

```bash
sudo nvidia-smi -rgc
```
Resets (unlocks) the GPU clock back to the driver's default dynamic boost behavior, undoing any `-lgc` override.

```bash
sudo nvidia-smi -lmc 1215
```
Locks the memory clock to 1215 MHz — the memory-clock equivalent of `-lgc`, used together when a fully deterministic clock profile (both core and memory) is required.

```bash
sudo nvidia-smi -rmc
```
Resets the memory clock lock back to dynamic default behavior.

```bash
nvidia-smi -q -d SUPPORTED_CLOCKS
```
Queries and lists every valid (core clock, memory clock) pairing the GPU's power/thermal envelope supports — the reference table to consult before choosing values for `-lgc`/`-lmc`, since not every arbitrary MHz value is a legal combination.

**Application clocks (legacy but still relevant on some SKUs)**

```bash
sudo nvidia-smi -ac 1215,1410
```
Sets "application clocks" (memory,graphics) to specific values — a predecessor mechanism to `-lgc`/`-lmc` still present on some driver/SKU combinations; format is `memory_clock,graphics_clock`.

```bash
sudo nvidia-smi -rac
```
Resets application clocks back to the default (base) values.

**Accounting mode**

```bash
sudo nvidia-smi --accounting-mode=1
```
Enables the GPU accounting subsystem, which retains per-process utilization/memory statistics even after the process exits — required before `--query-accounted-apps` (§1.1) returns any historical data.

```bash
sudo nvidia-smi --clear-accounted-apps
```
Clears the accounting log's process history buffer, useful periodically to prevent the internal buffer from filling on a long-uptime node with high process churn.

---

### 1.3 `nvcc` & `nvidia-debugdump`

**Architecture targeting**

```bash
nvcc kernel.cu -o kernel -arch=sm_80
```
Compiles targeting compute capability 8.0 (Ampere, the A100's native architecture) — using the WRONG `-arch` value here either fails to compile or silently produces code that doesn't use A100-specific features like third-generation Tensor Cores.

```bash
nvcc kernel.cu -o kernel -gencode arch=compute_80,code=sm_80
```
The more explicit `-gencode` form, separating the PTX virtual-architecture target (`compute_80`) from the real binary target (`sm_80`) — needed when you want fine control over both JIT-fallback compatibility and native binary generation in the same build.

```bash
nvcc kernel.cu -o kernel -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90
```
Fat-binary compilation targeting BOTH A100 (`sm_80`) and H100 (`sm_90`) in a single output binary, letting the same executable run natively-optimized on either architecture without recompilation — essential for a heterogeneous AX100/A100/H100 fleet.

**Optimization flags**

```bash
nvcc kernel.cu -o kernel -O3
```
Maximum host-code compiler optimization level — note this affects HOST-side C++ code generation; device-side kernel optimization is controlled separately (see `-Xptxas` below).

```bash
nvcc kernel.cu -o kernel -Xptxas -O3
```
Passes `-O3` specifically through to `ptxas`, the PTX-to-SASS assembler, maximizing DEVICE (GPU-side) kernel optimization — this is the flag that actually affects kernel execution speed, distinct from host `-O3`.

```bash
nvcc kernel.cu -o kernel -use_fast_math
```
Enables a bundle of lower-precision, higher-throughput math intrinsics (approximate `sin`/`cos`/`exp`/division) — a significant speedup for tolerant workloads, but never appropriate for numerically sensitive code without explicit validation.

```bash
nvcc kernel.cu -o kernel -Xptxas -v
```
Verbose PTX assembler output, printing exact register count, shared memory usage, and spill-store/spill-load counts per kernel — the direct source of the register-count numbers used in every occupancy calculation.

```bash
nvcc kernel.cu -o kernel --maxrregcount=32
```
Caps register usage per thread at 32, forcing the compiler to spill excess variables to slower local memory if needed — a direct, blunt lever for the occupancy-vs-register-pressure trade-off, trading potential per-thread slowdown for higher achievable occupancy.

```bash
nvcc kernel.cu -o kernel -lineinfo
```
Embeds source line-number correlation metadata into the compiled binary, without the full overhead of debug builds — required for Nsight Compute to map profiler findings back to specific source lines (see §3.3).

```bash
nvcc kernel.cu -o kernel -G
```
Full device-side debug build: disables most optimizations, embeds full debug symbols for `cuda-gdb`. Never used for production/performance builds — debug-only.

```bash
nvcc kernel.cu -o kernel --ptxas-options=-allow-expensive-optimizations=true
```
Explicitly permits `ptxas` to spend more compile-time budget on aggressive optimization passes it would otherwise skip for compile-speed reasons — worthwhile for a kernel compiled once and run millions of times.

**PTX and intermediate artifact inspection**

```bash
nvcc kernel.cu -ptx -arch=sm_80 -o kernel.ptx
```
Emits the intermediate PTX (Parallel Thread Execution) virtual-ISA representation instead of a final binary — inspect this to verify the compiler generated the expected instruction-level structure before it's locked into architecture-specific SASS.

```bash
nvcc kernel.cu -cubin -arch=sm_80 -o kernel.cubin
```
Emits the architecture-SPECIFIC binary (CUBIN) for `sm_80` — the actual machine code the GPU executes, as opposed to the portable PTX intermediate.

```bash
cuobjdump -sass kernel.cubin
```
Disassembles a compiled CUBIN back into human-readable SASS (Shader Assembly) — the ground-truth view of exactly what instructions the SM will execute, one level below PTX.

```bash
cuobjdump -ptx kernel.cubin
```
Extracts and displays the embedded PTX from a fat binary, useful when auditing a third-party compiled `.so` for which architectures/PTX versions it was built against.

```bash
nvdisasm kernel.cubin
```
An alternative SASS disassembler to `cuobjdump -sass`, sometimes producing more readable control-flow-graph-annotated output, useful for deep instruction-scheduling analysis.

**`nvidia-debugdump`**

```bash
sudo nvidia-debugdump --list
```
Lists every GPU visible to the debugdump utility along with its index, the required first step before targeting a specific device for a deep dump.

```bash
sudo nvidia-debugdump -g 0 -d dump.nvdd
```
Performs a full hardware-state dump of GPU 0 (register file snapshots, engine states, error logs) into a binary `.nvdd` file — the artifact NVIDIA support requests when triaging a suspected hardware fault that `nvidia-smi` alone can't fully characterize.

```bash
sudo nvidia-debugdump -g 0 --xml -o dump.xml
```
Same full dump, but structured as XML output instead of the raw binary format, more amenable to programmatic parsing in an internal triage pipeline.

---

### 1.4 Multi-Process Service (MPS)

**Starting the MPS control daemon**

```bash
export CUDA_VISIBLE_DEVICES=0
nvidia-cuda-mps-control -d
```
Sets the target GPU for this MPS instance, then starts `nvidia-cuda-mps-control` in DAEMON mode (`-d`) — this is the server process that arbitrates CUDA context sharing across multiple client processes on the same GPU.

```bash
mkdir -p /tmp/mps_0/pipe /tmp/mps_0/log
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_0/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_0/log
nvidia-cuda-mps-control -d
```
The full production pattern: explicitly creates isolated pipe and log directories per-GPU (critical on multi-GPU nodes to avoid different MPS daemons colliding on the same default socket path), sets the environment variables pointing at them, then starts the daemon.

**Client-side environment for MPS usage**

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_0/pipe
python train.py
```
Any CUDA client process (here, a training script) that sets this SAME pipe directory environment variable before launching will automatically route its context through the MPS daemon instead of creating an independent, exclusive context — no application code changes required.

**VRAM percentage allocation per client**

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_0/pipe
echo "set_default_active_thread_percentage 50" | nvidia-cuda-mps-control
```
Sends a live command to the running MPS control daemon setting the DEFAULT SM-thread percentage quota for all future clients to 50% — a soft compute-share cap, not a hard memory limit, letting two jobs split one GPU's compute resources roughly evenly.

```bash
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25
python train.py
```
An alternative, PER-CLIENT method: setting this environment variable before launching an individual client process caps THAT process's SM-thread percentage to 25%, overriding the daemon's default for this one client only.

```bash
echo "set_default_device_pinned_mem_limit 0 20G" | nvidia-cuda-mps-control
```
Sets a PINNED MEMORY limit of 20GB specifically for device 0 as the default for future clients — pinned (page-locked) host memory is a distinct, often-overlooked resource pool from GPU VRAM itself.

**Pipe routing across multiple GPUs**

```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_0/pipe
nvidia-cuda-mps-control -d

export CUDA_VISIBLE_DEVICES=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_1/pipe
nvidia-cuda-mps-control -d
```
Starts TWO independent MPS daemons, one per physical GPU, each with its own isolated pipe directory — the standard pattern for a multi-GPU node where you want independent MPS-managed sharing pools per card rather than one daemon trying to arbitrate across all GPUs.

**Querying and administering a running MPS daemon**

```bash
echo "get_server_list" | nvidia-cuda-mps-control
```
Lists the PIDs of all currently active MPS server processes — each unique client GID/UID combination spawns its own server process under the hood.

```bash
echo "get_client_list <server_pid>" | nvidia-cuda-mps-control
```
Lists the client process PIDs currently attached to a specific MPS server, letting you trace exactly which application PIDs are sharing that server's context.

```bash
echo "get_default_active_thread_percentage" | nvidia-cuda-mps-control
```
Queries the currently configured default SM-thread percentage quota, confirming a prior `set_default_active_thread_percentage` change actually took effect.

**Shutting down MPS cleanly**

```bash
echo "quit" | nvidia-cuda-mps-control
```
Gracefully shuts down the MPS control daemon and all its spawned server processes — always issue this before disabling MPS mode or before a GPU maintenance window, rather than killing the daemon process directly.

```bash
ps -ef | grep nvidia-cuda-mps-server
sudo kill -TERM <pid>
```
The manual fallback when `quit` doesn't cleanly terminate a hung server process: locate the server PID via `ps`, then send a graceful `SIGTERM` (never `SIGKILL` as a first resort, since it can leave the GPU in an inconsistent context state requiring a full reset).

---

## 2. Enterprise GPU System Administration & Node Recovery

### 2.1 VRAM Reclamation & Host Eviction

**Identifying the offending process**

```bash
sudo fuser -v /dev/nvidia*
```
Lists every process holding an open file handle on any `/dev/nvidia*` device node — the fastest single command to answer "what is currently touching the GPU at the OS level," independent of whether `nvidia-smi` itself is responding correctly.

```bash
sudo lsof /dev/nvidia0
```
Lists open file descriptors specifically against `/dev/nvidia0` (GPU 0's primary device node), showing PID, user, and file-descriptor type — more verbose than `fuser` but useful when you need the owning USER, not just the PID.

```bash
sudo lsof /dev/nvidia-uvm
```
Checks specifically for processes holding the Unified Virtual Memory device node open — a process can sometimes release `/dev/nvidia0` but still hold `/dev/nvidia-uvm`, silently pinning VRAM allocated via managed memory.

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```
The GPU-driver-level view (as opposed to the OS file-handle view above) of every process with an active CUDA context and its VRAM footprint — cross-reference against `fuser`/`lsof` output to confirm you're targeting the correct PID.

**Terminating rogue processes**

```bash
sudo kill -15 <pid>
```
Sends `SIGTERM` (graceful termination request) to the target PID first — always the correct FIRST attempt, giving the process a chance to release its CUDA context and free VRAM cleanly through its own exit handlers.

```bash
sleep 10 && sudo kill -9 <pid>
```
If the process hasn't exited 10 seconds after `SIGTERM`, escalate to `SIGKILL` (`-9`), an unconditional, un-catchable termination — necessary for a genuinely hung process, but can occasionally leave GPU memory in a state requiring a full reset (see below) to fully reclaim.

```bash
sudo pkill -9 -f "python.*train.py"
```
Pattern-matches and force-kills every process whose full command line matches the regex `python.*train.py` — useful for bulk-cleaning multiple zombie instances of the same training script without hunting down individual PIDs.

```bash
sudo pkill -9 -u training_user
```
Kills every process owned by a specific user account — the blunt-instrument option for reclaiming a shared node from a departed or unresponsive user session entirely.

**IPC segment cleanup**

```bash
ipcs -m
```
Lists all System V shared memory segments currently allocated on the host — PyTorch's DataLoader multiprocessing workers, and some CUDA IPC memory-sharing paths, leave orphaned segments behind if a process is killed abruptly rather than exiting cleanly.

```bash
ipcs -m | awk '/deleted/{print $2}' | xargs -I{} ipcrm -m {}
```
Identifies shared memory segments marked for deletion (zero attach count, pending cleanup) and force-removes them — a common fix for "torch DataLoader running out of shared memory" errors that persist even after killing the offending process.

```bash
ipcrm -a
```
Nuclear option: removes ALL IPC resources (shared memory segments, semaphores, message queues) on the host — only ever run this on a node with zero other active workloads, since it will corrupt any other running process's IPC state too.

**Full GPU reset when VRAM won't clear**

```bash
sudo nvidia-smi --gpu-reset -i 0
```
Issues a full hardware reset of GPU 0 at the driver level, without rebooting the entire host — the definitive fix when `memory.used` remains stubbornly nonzero even after confirming zero processes hold the device (a classic sign of an improperly released CUDA context from a crashed process).

```bash
sudo nvidia-smi -i 0 -r
```
Shorthand equivalent reset flag on newer driver versions — check `nvidia-smi --help` on your specific driver branch, since reset flag syntax has shifted across major driver releases.

```bash
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
```
The escalation path when even a GPU-level reset doesn't clear the issue: unload and reload the Unified Virtual Memory kernel module specifically, which can clear UVM-level page-table state that survives a plain GPU reset — see §2.3 for the full module-management toolkit this draws from.

---

### 2.2 Hardware Topology & Fabric Interconnects

**Interconnect topology matrix**

```bash
nvidia-smi topo -m
```
Prints the full pairwise interconnect matrix between every GPU (and NIC) on the node, showing `NV#` (NVLink, with link count), `PIX`/`PXB` (PCIe switch hops), `NODE`, or `SYS` (cross-NUMA) for each pair — the definitive first check when diagnosing unexpectedly slow multi-GPU collective operations.

```bash
nvidia-smi topo -mp
```
Same matrix, with an added CPU-affinity/NUMA-node column, letting you correlate which CPU socket a given GPU is physically wired closest to.

```bash
nvidia-smi topo -p2p r
```
Peer-to-peer READ capability matrix specifically, independent of the general interconnect type matrix above.

```bash
nvidia-smi topo -p2p w
```
Peer-to-peer WRITE capability matrix — check this SEPARATELY from the read matrix, since asymmetric P2P support is a known symptom of a misconfigured IOMMU/ACS setting.

**NVLink status and error counters**

```bash
nvidia-smi nvlink -s
```
Reports per-link "Active/Inactive" status for every physical NVLink port on every GPU, node-wide — the fastest way to spot a link that has silently dropped without a driver-level error.

```bash
nvidia-smi nvlink -s -i 0
```
Scopes the link-status query to GPU 0 only.

```bash
nvidia-smi nvlink -e
```
Dumps cumulative NVLink error counters (CRC flit errors, CRC data errors, replay counts) across every link, node-wide — the primary early-warning signal for a physically degrading cable or connector.

```bash
nvidia-smi nvlink -e -i 0 -l 0
```
Scopes the error-counter query to link 0 of GPU 0 specifically, for granular per-link investigation once the node-wide sweep has flagged a suspect device.

```bash
nvidia-smi nvlink -r -i 0 -l 0
```
Resets the error counters on link 0 of GPU 0 to zero — run immediately after logging a transient event, so future monitoring windows report only genuinely new errors.

```bash
nvidia-smi nvlink -c
```
Queries the negotiated NVLink capability flags (P2P support, SLI, atomics) per link, useful for confirming a driver or firmware update didn't silently change the negotiated feature set.

**NVSwitch configuration and status (for NVSwitch-connected topologies, e.g. DGX-class nodes)**

```bash
nvidia-smi nvlink -s -i 0 --all-links
```
Extends the link status query to show ALL physical link slots on GPU 0, including any not currently in an active P2P pairing — relevant on NVSwitch fabrics where a GPU's total link count exceeds what a simple GPU-to-GPU topology would use.

```bash
sudo nvswitch-audit
```
(NVSwitch-fabric-specific utility, where present) runs a fabric-wide health audit across every NVSwitch ASIC in the chassis, cross-checking routing table consistency — the fabric-level equivalent of `nvidia-smi topo -m` for switch-based (as opposed to direct GPU-to-GPU) NVLink topologies.

**Bandwidth verification under load**

```bash
nvbandwidth -t device_to_device_memcpy_read_ce
```
Runs NVIDIA's dedicated `nvbandwidth` tool, measuring ACTUALLY ACHIEVED GPU-to-GPU copy-engine bandwidth over the interconnect fabric under real load — the number to compare against the theoretical spec pulled from `nvidia-smi nvlink -c`.

```bash
nvbandwidth -t device_to_device_memcpy_read_ce -t device_to_device_memcpy_write_ce
```
Runs both the read and write bandwidth sub-tests in one invocation, for a complete bidirectional bandwidth characterization.

```bash
/usr/local/cuda/extras/demo_suite/p2pBandwidthLatencyTest
```
The classic CUDA-toolkit-bundled peer-to-peer bandwidth and latency benchmark, producing a full pairwise GPU bandwidth matrix — a lighter-weight, always-available alternative when the standalone `nvbandwidth` tool isn't installed.

---

### 2.3 Kernel, Module, & Driver Interrogation

**Module status and metadata**

```bash
lsmod | grep nvidia
```
Lists all currently loaded NVIDIA-prefixed kernel modules and their reference counts — the reference count column tells you whether a module is safely unloadable (count of 0) or still held open by something.

```bash
modinfo nvidia
```
Dumps full metadata for the core `nvidia` module: version, license, supported parameters, and vermagic (kernel version it was built against) — the vermagic mismatch check is critical after any host kernel upgrade.

```bash
modinfo nvidia_uvm
```
Same metadata dump for the Unified Virtual Memory module specifically, which handles managed-memory page-fault servicing and CPU-GPU memory migration.

```bash
modinfo nvidia_modeset
```
Metadata for the display-modesetting module — largely irrelevant on headless compute nodes, but its presence/absence is a quick sanity check that the driver package installed the FULL component set, not a display-stripped variant that might be missing other expected pieces.

**Loading and unloading**

```bash
sudo modprobe nvidia
```
Loads the core NVIDIA kernel module along with any declared dependencies, resolving the module dependency graph automatically (unlike plain `insmod`).

```bash
sudo modprobe -r nvidia_uvm
```
Unloads (`-r` for remove) the UVM module specifically — will fail with an in-use error if any process still holds an open `/dev/nvidia-uvm` handle, which is itself a useful diagnostic signal.

```bash
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia
```
Manually unloads the full NVIDIA module stack in the REQUIRED dependency order (dependents before the base module) — necessary before certain driver upgrade or downgrade procedures that `modprobe -r` alone won't sequence correctly.

```bash
sudo modprobe nvidia NVreg_EnableGpuFirmware=0
```
Loads the module with an explicit parameter override — here, disabling GSP (GPU System Processor) firmware offload, a diagnostic step sometimes used when triaging suspected GSP-firmware-related instability.

**DKMS status**

```bash
dkms status
```
Lists every DKMS-managed kernel module and its build status against every installed kernel version on the host — the tool that automatically rebuilds the NVIDIA module against a NEW kernel after a `apt upgrade`/`yum update`, and this command confirms that rebuild actually succeeded.

```bash
sudo dkms build -m nvidia -v 535.129.03
```
Manually triggers a DKMS rebuild of a specific NVIDIA driver version against the currently running kernel headers — used to recover from a failed automatic rebuild without a full driver reinstall.

```bash
sudo dkms install -m nvidia -v 535.129.03
```
Installs (loads into the active module tree) a DKMS module that has already been successfully built, the step following `dkms build`.

**Version and compatibility cross-checks**

```bash
cat /proc/driver/nvidia/version
```
Reads the driver version string directly from the kernel's proc-fs interface — a check independent of `nvidia-smi` itself, useful when `nvidia-smi` is unresponsive but you need to confirm the driver module is at least loaded.

```bash
nvidia-smi --query-gpu=driver_version,cuda_version --format=csv
```
Reports both the display driver version AND the maximum CUDA toolkit version that driver supports (not necessarily the toolkit version actually installed) — the first cross-check when a CUDA application reports a version-mismatch error.

```bash
nvcc --version
```
Reports the ACTUAL installed CUDA toolkit/compiler version — compare directly against the `cuda_version` field from the query above to confirm the installed toolkit doesn't exceed what the driver supports.

---

### 2.4 OS Telemetry & Low-Level Log Parsing

**Kernel ring buffer — `dmesg`**

```bash
dmesg -T | grep -i nvidia
```
Greps the kernel ring buffer for any NVIDIA-related message, human-timestamped (`-T`) — the first broad sweep for driver-level errors, module load failures, or hardware faults.

```bash
dmesg -T | grep -i "NVRM: Xid"
```
Filters specifically for Xid codes — NVIDIA's hardware-fault taxonomy, where each numeric code maps to a specific failure class (e.g., Xid 48=double-bit ECC error, Xid 63/64=row-remap events, Xid 79=GPU fallen off the bus, Xid 13=graphics engine exception).

```bash
dmesg -T | grep -i "pcie"
```
Filters for PCIe-subsystem messages — link-training failures, AER (Advanced Error Reporting) events, and bus-reset notifications all surface here.

```bash
dmesg --level=err,crit,alert,emerg
```
Filters the ring buffer by SEVERITY LEVEL rather than text pattern, showing only error-grade-or-worse messages across the ENTIRE kernel log, not just NVIDIA-related — useful for catching a co-occurring non-GPU hardware issue (e.g., a failing DIMM) that might be a contributing factor.

```bash
dmesg -c
```
Reads and CLEARS the ring buffer in one operation — use immediately after triaging a known issue, so the buffer is clean for capturing the NEXT occurrence without old noise.

**`journalctl` — systemd's structured log**

```bash
journalctl -k -b
```
Shows kernel messages (`-k`) from the current boot only (`-b`) — the `journalctl` equivalent of `dmesg`, but persists across reboots (unlike the ring buffer, which is wiped) if persistent journaling is enabled.

```bash
journalctl -k -b -1
```
Same kernel-message view, but for the PREVIOUS boot (`-1`) — essential for post-mortem analysis of a crash that caused an unplanned reboot, since the ring buffer contents from before the crash are otherwise lost.

```bash
journalctl -u nvidia-persistenced --since "1 hour ago"
```
Filters to a specific systemd unit (the persistence-mode daemon service) and a relative time window — the pattern for isolating one specific NVIDIA-related systemd service's recent activity.

```bash
journalctl --since "2024-01-15 08:00:00" --until "2024-01-15 09:00:00" | grep -i nvidia
```
Bounds the search to an exact timestamp WINDOW (useful when correlating against a known incident time from monitoring alerts) combined with an NVIDIA text filter.

```bash
journalctl -f
```
Follows the journal live (`-f`, like `tail -f`), streaming new log entries as they're written — used to watch for a fault to reoccur in real time while actively reproducing an issue.

**IPMI hardware event logs**

```bash
sudo ipmitool sel list
```
Lists the full System Event Log (SEL) stored in the baseboard management controller's non-volatile memory — captures hardware events (thermal trips, power supply faults, fan failures) that occur even when the host OS itself is unresponsive or powered off.

```bash
sudo ipmitool sel elist
```
Extended-format SEL listing, providing more verbose per-event detail than the compact `sel list` output.

```bash
sudo ipmitool sensor list
```
Live readout of every hardware sensor the BMC monitors (fan RPMs, voltage rails, temperatures across multiple zones) — a broader hardware-health view than `nvidia-smi` alone, which only sees GPU-internal sensors.

```bash
sudo ipmitool sdr type Temperature
```
Filters the sensor data repository to temperature-type sensors specifically, across the whole chassis (not just the GPUs) — useful for confirming a GPU thermal event correlates with (or is independent of) an ambient/inlet temperature spike.

```bash
sudo ipmitool sel clear
```
Clears the stored SEL after a known event has been fully investigated and documented, so the log doesn't fill with old, already-triaged entries — always document/export the log BEFORE clearing.

```bash
sudo ipmitool chassis status
```
Reports overall chassis power state and any latched fault indicators, a quick top-level chassis health check independent of any specific sensor drill-down.

---

## 3. Under the Hood: Deep-Dive Optimization & Runtime Tuning

### 3.1 SIMD/SIMT & Vector CPU Optimization

**CPU feature discovery**

```bash
lscpu
```
Prints a structured summary of the CPU topology: socket count, cores per socket, threads per core, NUMA node layout, and cache sizes — the essential first command before any NUMA-pinning decision.

```bash
lscpu | grep -i flags
```
Isolates the raw CPU feature flags line — the field containing every ISA extension (AVX2, AVX-512 variants, AMX) the processor advertises support for.

```bash
grep -m1 flags /proc/cpuinfo | tr ' ' '\n' | grep -i avx
```
Reads `/proc/cpuinfo` directly (the raw kernel-exposed source `lscpu` itself parses), extracts the flags line for the first logical CPU, splits it one-flag-per-line, and filters for AVX-family flags specifically — useful when scripting an automated capability check.

```bash
grep -m1 flags /proc/cpuinfo | tr ' ' '\n' | grep -i amx
```
Same pattern, filtering for AMX (Advanced Matrix Extensions) flags — relevant on newer Intel Sapphire Rapids-class hosts pairing with A100/AX100 accelerators, where AMX handles CPU-side int8/bf16 matrix math for data-preprocessing pipelines feeding the GPU.

```bash
lscpu | grep -i "avx512"
```
Direct grep for AVX-512 sub-variant flags (`avx512f`, `avx512bw`, `avx512vnni`, etc.) within `lscpu`'s already-parsed flag list — each suffix denotes a distinct instruction subset, and framework-level vectorized ops (e.g., certain PyTorch CPU kernels) may only engage specific subsets.

```bash
cat /sys/devices/system/cpu/cpu0/cache/index3/size
```
Reads the L3 cache size for logical CPU 0 directly from sysfs — relevant for tuning CPU-side data-loader batch/prefetch sizes to fit comfortably within cache, minimizing preprocessing-stage stalls before data even reaches the PCIe bus.

**NUMA topology discovery**

```bash
numactl --hardware
```
Displays the full NUMA topology: node count, CPUs belonging to each node, memory size per node, and the inter-node distance matrix — the distance matrix is critical, since a "remote" memory access across nodes carries measurably higher latency than a "local" one.

```bash
lscpu | grep -i numa
```
A condensed NUMA summary from within `lscpu`'s own output, useful as a quick sanity check without invoking `numactl` separately.

```bash
numastat
```
Reports NUMA memory allocation statistics (local hits, remote hits/misses) per node, aggregated across all running processes — a high remote-access count signals memory is being allocated on the wrong node relative to where the consuming process is pinned.

```bash
numastat -p <pid>
```
Scopes the NUMA statistics to a single specific process, showing exactly how ITS memory is distributed across nodes.

**Pinning processes to CPUs and NUMA nodes**

```bash
taskset -c 0-15 python train.py
```
Launches the training script with its CPU affinity mask restricted to logical cores 0-15 — `taskset` operates at the raw CPU-mask level, without any NUMA-memory-locality awareness of its own.

```bash
taskset -cp 0-15 <pid>
```
Applies the SAME core-affinity restriction to an ALREADY-RUNNING process by PID, rather than at launch time — useful for correcting a misplaced process without restarting it.

```bash
numactl --cpunodebind=0 --membind=0 python train.py
```
The NUMA-aware equivalent of `taskset`: binds BOTH the process's CPU execution AND its memory allocations to NUMA node 0 specifically — this dual binding is what actually guarantees local (low-latency) memory access, which plain `taskset` alone does not.

```bash
numactl --cpunodebind=0 --membind=0 --physcpubind=0-31 python train.py
```
Adds `--physcpubind` for fine-grained control down to specific PHYSICAL core numbers within the bound NUMA node, useful when you want a subset of a node's cores reserved for other co-located processes.

```bash
numactl --interleave=all python data_preprocess.py
```
Interleaves memory allocation ROUND-ROBIN across ALL NUMA nodes rather than binding to one — counter-intuitively useful for a memory-bandwidth-heavy preprocessing stage that benefits from aggregate bandwidth across every node's memory controller, rather than saturating a single node's bandwidth alone.

```bash
numactl --show
```
Displays the CURRENT process's own NUMA policy and affinity (when run standalone, shows the shell's own settings) — a quick way to confirm what policy is actually in effect before launching a workload.

**GPU-to-NUMA-node correlation**

```bash
nvidia-smi topo -m | grep -A1 "GPU0"
```
Cross-references the topology matrix output (from §2.2) specifically for GPU 0's row, revealing which CPU/NUMA affinity that GPU is wired closest to — the number to feed directly into a `numactl --cpunodebind` decision for any process primarily feeding data to that specific GPU.

```bash
cat /sys/bus/pci/devices/0000:00:04.0/numa_node
```
Reads the NUMA node affinity for a specific PCI device (identified by its bus address, obtainable via `lspci`) directly from sysfs — the raw kernel-level source of truth that `nvidia-smi topo -mp`'s NUMA column is itself derived from.

---

### 3.2 Environment Isolation & Framework Runtime Variables

**Device visibility and selection**

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```
Restricts the CUDA runtime's view to physical GPUs 0-3 only — any process launched with this variable set sees ONLY these four devices as indices 0-3 internally, regardless of the host's total GPU count; the standard multi-tenant isolation mechanism.

```bash
export CUDA_VISIBLE_DEVICES=""
```
Sets an EMPTY device list, hiding all GPUs from the process entirely — forces any CUDA calls to fail cleanly, a useful diagnostic for confirming a workload correctly falls back to CPU-only execution when no GPU is available.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```
Forces CUDA's internal device enumeration order to match physical PCI bus ordering, rather than the driver's default "fastest first" heuristic ordering — CRITICAL to set consistently across all nodes in a cluster, since `CUDA_VISIBLE_DEVICES=0` can otherwise refer to a DIFFERENT physical GPU on different nodes without this variable pinned.

**NCCL — collective communications tuning**

```bash
export NCCL_DEBUG=INFO
```
Enables INFO-level logging from NCCL (NVIDIA Collective Communications Library), printing the ring/tree topology it constructed, the interconnect type chosen per link, and initialization progress — the essential first diagnostic step for any multi-node training job that hangs or performs unexpectedly during its all-reduce phase.

```bash
export NCCL_DEBUG=TRACE
```
An even more verbose level than `INFO`, logging every individual collective operation call — generates substantial log volume, reserved for deep debugging of a specific hang rather than routine job logging.

```bash
export NCCL_DEBUG_SUBSYS=INIT,NET
```
Scopes debug output to specific NCCL subsystems only (here, initialization and networking) rather than the full firehose — useful for isolating network-fabric-specific issues from the noisier general debug stream.

```bash
export NCCL_IB_DISABLE=1
```
Forces NCCL to disable InfiniBand transport entirely, falling back to a different available transport (e.g., TCP sockets) — a diagnostic isolation step to confirm whether an observed hang or slowdown is IB-fabric-specific.

```bash
export NCCL_SOCKET_IFNAME=eth0
```
Explicitly pins NCCL's TCP/socket-based fallback transport to a specific network interface, preventing it from auto-selecting a wrong or slower interface on a host with multiple NICs.

```bash
export NCCL_IB_HCA=mlx5_0,mlx5_1
```
Explicitly restricts NCCL's InfiniBand transport to specific Host Channel Adapter devices by name, useful on nodes with multiple IB HCAs where only a subset should be used for a given job (e.g., reserving others for storage traffic).

```bash
export NCCL_P2P_LEVEL=NVL
```
Explicitly instructs NCCL to prefer NVLink for peer-to-peer GPU communication when available, rather than falling back to PCIe — normally auto-detected correctly, but explicit setting is useful when diagnosing a suspected auto-detection failure.

```bash
export NCCL_ALGO=Ring
```
Forces NCCL to use the Ring algorithm for collective operations specifically (as opposed to Tree or other available algorithms) — useful for isolating whether an algorithm-selection heuristic is choosing suboptimally for your specific message-size and node-count profile.

```bash
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
```
Tunes the socket-transport thread/connection parallelism — increasing these can improve achieved bandwidth on TCP-fallback paths at the cost of additional CPU thread overhead, relevant when IB isn't available and you're bottlenecked on Ethernet.

```bash
export NCCL_BUFFSIZE=8388608
```
Sets NCCL's internal communication buffer size in bytes (here, 8MB) — larger buffers can improve throughput for large-message collectives at the cost of higher fixed memory overhead per rank.

**Unified Virtual Memory (UVM) tuning**

```bash
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
```
Forces `cudaMallocManaged` allocations to be physically backed on the GPU device by default (rather than starting on the host and migrating on first touch) — reduces the initial page-fault storm for workloads that know they'll immediately need the data GPU-side.

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=32
```
Increases the maximum number of hardware connections (work queues) available for concurrent kernel/copy scheduling from a single process — relevant for workloads issuing many independent CUDA streams that need genuine hardware-level concurrency rather than serialization onto a shared connection.

```bash
nvidia-smi -q -d MEMORY | grep -A2 "BAR1"
```
Queries the current BAR1 (Base Address Register 1) memory usage — the PCIe-mapped aperture used for direct host access to GPU memory, relevant to UVM page-migration behavior and GPUDirect RDMA transfers; an exhausted BAR1 window can silently degrade UVM performance.

```bash
export CUDA_LAUNCH_BLOCKING=1
```
Forces every kernel launch to become SYNCHRONOUS (blocking the host thread until the kernel completes) — dramatically slows execution, but makes stack traces from a crashing kernel point to the EXACT launch that caused it, rather than an unrelated later point due to CUDA's normal asynchronous execution model; a standard debugging-only toggle, never left on in production.

**PyTorch-specific runtime variables**

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```
Tunes PyTorch's CUDA caching memory allocator, capping the maximum size of a single memory block split to 128MB — mitigates memory FRAGMENTATION in long-running training jobs with highly variable tensor allocation sizes, a common root cause of "CUDA out of memory" errors even when aggregate free memory appears sufficient.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
Enables PyTorch's expandable-segment allocator mode, which grows memory segments in place rather than requiring new contiguous blocks — a newer, more aggressive fragmentation mitigation than the split-size tuning above.

```bash
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```
Enables asynchronous error handling in PyTorch's NCCL process-group wrapper, causing a hung or failed collective operation to raise a catchable exception (with a timeout) instead of hanging the process indefinitely — essential for multi-node jobs where one rank's fault would otherwise silently wedge the entire job.

---

### 3.3 Performance Profiling & Kernel Tracing

**Nsight Systems (`nsys`) — system-wide timeline profiling**

```bash
nsys profile -o report python train.py
```
Runs the training script under Nsight Systems, capturing a full system-level timeline (CPU threads, CUDA API calls, kernel execution, memory transfers) into `report.nsys-rep` — the standard first-pass profiling command, providing the "big picture" view before drilling into any specific kernel.

```bash
nsys profile -t cuda,nvtx,osrt -o report python train.py
```
Explicitly selects which trace providers to capture: `cuda` (kernel/API activity), `nvtx` (custom application-level range annotations, if your code uses `torch.cuda.nvtx.range`), and `osrt` (OS runtime calls like thread creation/scheduling) — trimming providers reduces both overhead and output file size when you don't need the full default set.

```bash
nsys profile --stats=true -o report python train.py
```
Additionally generates a human-readable SUMMARY statistics table (top kernels by total time, API call counts) printed directly to console after the run completes, without requiring the separate GUI viewer for a quick first look.

```bash
nsys profile --capture-range=cudaProfilerApi -o report python train.py
```
Restricts actual data CAPTURE to only the window between explicit `cudaProfilerStart()`/`cudaProfilerStop()` calls embedded in your application code — essential for skipping a lengthy, uninteresting warmup/data-loading phase and capturing only the steady-state training loop.

```bash
nsys profile -d 30 -o report python train.py
```
Limits the capture DURATION to 30 seconds from launch, automatically stopping and finalizing the report afterward — bounds output file size for a long-running job where you only need a representative window.

```bash
nsys stats report.nsys-rep
```
Post-processes an already-captured `.nsys-rep` file into the same summary statistics tables, without needing to re-run the profiled application — the tool for re-analyzing an existing capture with different report filters.

```bash
nsys stats --report cuda_gpu_kern_sum report.nsys-rep
```
Extracts specifically the GPU-KERNEL summary report (total time, instance count, average duration per unique kernel name) — the direct source for identifying which specific kernel dominates total GPU time.

```bash
nsys stats --report cuda_gpu_mem_time_sum report.nsys-rep
```
Extracts the GPU memory-operation time summary (H2D copies, D2H copies, device-to-device copies) — the direct diagnostic for identifying whether a workload is unexpectedly spending significant time on memory transfers rather than compute.

**Nsight Compute (`ncu`) — deep per-kernel profiling**

```bash
ncu -o kernel_report python train.py
```
Profiles EVERY kernel launch in the process with Nsight Compute's full default metric set, writing results to `kernel_report.ncu-rep` — substantially heavier overhead than `nsys`, appropriate for deep-diving specific kernels rather than whole-application profiling.

```bash
ncu --set full -k "gemm" -o gemm_report python train.py
```
Restricts profiling to kernels whose name matches `gemm` (`-k` filter), using the `full` metric set (every available hardware counter) — narrows both the runtime overhead and output size to only the specific matrix-multiply kernels of interest.

```bash
ncu --launch-skip 100 --launch-count 5 -o report python train.py
```
Skips the first 100 kernel launches (past an uninteresting warmup phase) and profiles only the NEXT 5 launches after that — a precise targeting mechanism for reaching a specific steady-state iteration in a training loop without capturing every single launch from the start.

```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed -o report python train.py
```
Collects a single specific metric — SM compute throughput as a percentage of the theoretical peak — rather than the full metric set, minimizing profiling overhead when you already know exactly which number you need.

```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed -o report python train.py
```
Collects DRAM (HBM) throughput as a percentage of peak specifically — the direct metric for confirming whether a kernel is MEMORY-BANDWIDTH-BOUND (this number near 100% while SM throughput is low) rather than compute-bound.

```bash
ncu --section MemoryWorkloadAnalysis -o report python train.py
```
Collects the full pre-built "Memory Workload Analysis" section — a curated bundle of memory-subsystem metrics (cache hit rates, coalescing efficiency, bank conflicts) assembled by NVIDIA specifically for diagnosing memory-bound kernels, more convenient than hand-picking individual metrics.

```bash
ncu --section Occupancy -o report python train.py
```
Collects the "Occupancy" section specifically — achieved vs theoretical occupancy, and the LIMITING FACTOR (registers, shared memory, or block-size) preventing higher occupancy for each profiled kernel, directly answering the "why isn't my kernel fully occupying the SM" question.

```bash
ncu --import kernel_report.ncu-rep --csv --page details > kernel_metrics.csv
```
Post-processes an already-captured `.ncu-rep` file into CSV format for the detailed metrics page, enabling programmatic downstream analysis or dashboard ingestion without the interactive GUI viewer.

**Memory-leak-specific tracing**

```bash
compute-sanitizer --tool memcheck python train.py
```
Runs the Compute Sanitizer's memory-checking tool, detecting out-of-bounds accesses and use of uninitialized memory within CUDA kernels — the direct CUDA-level equivalent of Valgrind, essential for catching a genuine device-side memory corruption bug rather than just a host-side allocation leak.

```bash
compute-sanitizer --tool memcheck --leak-check full python train.py
```
Additionally enables full leak-checking, reporting any device memory allocated but never freed by the time the process exits — the direct tool for confirming and localizing a suspected CUDA memory leak down to the specific allocation call site.

```bash
compute-sanitizer --tool racecheck python train.py
```
Switches to the race-condition-detection tool, flagging shared-memory or global-memory races between threads within a block — the tool to reach for when a kernel's output is non-deterministic across otherwise-identical runs, a classic symptom of a missing `__syncthreads()`.

---

## 4. Storage, Network Fabric, & Compute Health Validation

### 4.1 PyTorch/Python GPU Assertion Snippets

**Basic capability and availability checks**

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
The single most fundamental sanity check: confirms PyTorch can see AT LEAST ONE CUDA-capable device from the current process's environment (respecting `CUDA_VISIBLE_DEVICES`).

```bash
python -c "import torch; print(torch.cuda.device_count())"
```
Reports exactly how many GPUs are visible to this process — cross-check against the expected count given the current `CUDA_VISIBLE_DEVICES` setting to catch a misconfigured environment early.

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```
Prints the human-readable product name of GPU index 0 as seen by PyTorch — a quick confirmation that device index 0 maps to the physical GPU you expect (relevant when `CUDA_DEVICE_ORDER` isn't pinned consistently, see §3.2).

```bash
python -c "import torch; print(torch.cuda.get_device_capability(0))"
```
Reports the compute capability tuple (e.g., `(8, 0)` for A100) — the number that must match the `-arch=sm_80` target used when compiling any custom CUDA extensions this process will load.

```bash
python -c "import torch; print(torch.version.cuda)"
```
Reports the CUDA toolkit version PyTorch itself was COMPILED against — distinct from the driver's supported CUDA version (see §2.3), and the version that actually matters for binary compatibility of the PyTorch build itself.

**Memory pool inspection**

```bash
python -c "import torch; print(torch.cuda.memory_allocated(0))"
```
Reports bytes CURRENTLY allocated to live tensors on GPU 0 — the true "in-use" figure, as opposed to memory merely reserved by PyTorch's caching allocator but not backing an active tensor.

```bash
python -c "import torch; print(torch.cuda.memory_reserved(0))"
```
Reports bytes RESERVED by PyTorch's caching allocator (including cached-but-currently-unused blocks) — the number that more closely matches what `nvidia-smi` reports as this process's VRAM usage, since PyTorch deliberately holds freed memory in its cache rather than returning it to the driver immediately.

```bash
python -c "import torch; print(torch.cuda.max_memory_allocated(0))"
```
Reports the PEAK allocated-memory high-water-mark since the process started (or since the last reset) — essential for right-sizing batch size or identifying the specific training step that triggers a memory spike.

```bash
python -c "import torch; torch.cuda.reset_peak_memory_stats(0); print('reset')"
```
Resets the peak-memory tracking counters back to zero, letting you isolate the peak memory of a SPECIFIC subsequent code section rather than the whole process's history.

```bash
python -c "import torch; print(torch.cuda.memory_summary(0))"
```
Prints PyTorch's full, human-readable memory-allocator summary table: allocated/reserved/active memory broken down by pool, plus allocator statistics like the number of `cudaMalloc` retries — the single richest built-in diagnostic for a suspected memory-fragmentation issue.

**Flushing cached tensors**

```bash
python -c "import torch; torch.cuda.empty_cache(); print('cache emptied')"
```
Instructs PyTorch's caching allocator to release all currently-unused cached memory blocks BACK to the CUDA driver — does NOT free any memory still backing live tensors; the correct first step when you need `nvidia-smi`'s reported usage to reflect only genuinely-in-use memory.

```bash
python -c "
import torch, gc
gc.collect()
torch.cuda.empty_cache()
print('gc + cache cleared')
"
```
Runs a full Python garbage-collection pass FIRST (releasing any tensors that are only reachable via reference cycles the normal refcounting GC wouldn't catch immediately) before emptying PyTorch's cache — the more thorough sequence when a simple `empty_cache()` alone doesn't reclaim expected memory.

**Stream synchronization checks**

```bash
python -c "import torch; torch.cuda.synchronize(0); print('synchronized')"
```
Blocks the CPU host thread until ALL queued work on GPU 0's default stream has genuinely completed — the Python-level equivalent of `cudaDeviceSynchronize()`, essential before trusting any subsequent timing measurement in a benchmarking script.

```bash
python -c "
import torch, time
torch.cuda.synchronize()
start = time.time()
x = torch.randn(8192, 8192, device='cuda')
y = torch.matmul(x, x)
torch.cuda.synchronize()
print(f'Elapsed: {time.time()-start:.4f}s')
"
```
A complete, correct GPU-timing pattern: synchronize BEFORE starting the timer (draining any prior queued async work), perform the operation, synchronize AGAIN before stopping the timer (waiting for the async-launched kernel to actually finish) — omitting either synchronize call is the single most common cause of misleading GPU benchmark numbers.

```bash
python -c "
import torch
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    x = torch.randn(1000, 1000, device='cuda')
torch.cuda.current_stream().wait_stream(s)
print('stream dependency established')
"
```
Demonstrates explicit multi-stream coordination: creates a secondary stream, runs work on it, then makes the DEFAULT stream wait for that secondary stream to complete before proceeding — the correct pattern for overlapping independent GPU work streams (e.g., data prefetch on one stream while compute runs on another) without a race condition.

**Hardware pool auditing**

```bash
python -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {props.name}, {props.total_memory/1e9:.1f}GB, SM count={props.multi_processor_count}')
"
```
Loops across every visible GPU, printing name, total memory, and SM count for each — a complete inventory audit in one command, useful as a startup sanity check embedded in a training script's initialization logging.

```bash
python -c "
import torch
print('NCCL available:', torch.distributed.is_nccl_available())
print('MPI available:', torch.distributed.is_mpi_available())
"
```
Checks which distributed-training BACKENDS the current PyTorch build was compiled with support for — confirms NCCL support is present before attempting a multi-GPU `DistributedDataParallel` job, catching a misconfigured/CPU-only PyTorch install early rather than at job-launch failure time.

---

### 4.2 Storage & I/O Pipeline Monitoring

**Block-device I/O statistics — `iostat`**

```bash
iostat -x 1
```
Streams EXTENDED (`-x`) per-device I/O statistics every 1 second — the extended flag is what surfaces `%util` (device busy percentage) and `await` (average I/O wait time), the two numbers most directly relevant to diagnosing a data-loader-starved GPU.

```bash
iostat -x -d /dev/nvme0n1 1
```
Scopes the extended report to a single specific NVMe device (`-d`), reducing noise on a host with many block devices when you already know which drive is backing the training dataset.

```bash
iostat -xm 1
```
Reports throughput in MEGABYTES per second (`-m`) rather than the default sectors/kilobytes, a more directly readable unit when comparing against a known dataset read-throughput requirement.

```bash
iostat -c 1
```
Reports CPU utilization statistics ALONGSIDE the standard I/O report (`-c`), useful for confirming whether a slow data pipeline is I/O-bound or CPU-bound (e.g., in decompression/tokenization) at the same glance.

**Per-process I/O — `iotop`**

```bash
sudo iotop -o
```
Launches the interactive `iotop` display, filtered (`-o`) to show ONLY processes currently performing actual I/O — the direct tool for identifying which specific PID (e.g., a specific DataLoader worker) is generating unexpected disk load.

```bash
sudo iotop -b -n 5 -d 2
```
Runs in BATCH mode (`-b`, non-interactive, suitable for logging/scripting) for exactly 5 iterations (`-n 5`) at a 2-second delay (`-d 2`) — the pattern for capturing a bounded I/O snapshot into a log file rather than watching an interactive session.

```bash
sudo iotop -oPa
```
Combines active-only filtering (`-o`), process-level (not thread-level, `-P`) aggregation, and ACCUMULATED (`-a`) totals rather than per-interval rates — useful for identifying the single biggest cumulative I/O consumer over a training run's full duration, not just its instantaneous rate.

**Filesystem capacity — `df`**

```bash
df -h
```
Human-readable (`-h`, auto-scaled units) disk-space usage summary across every mounted filesystem — the routine first check before any job that will write large checkpoint files.

```bash
df -h /mnt/nvme_dataset
```
Scopes the check to the specific mount point backing the dataset directory, confirming sufficient free space before a large data-download or preprocessing job that will materialize new files there.

```bash
df -i
```
Reports INODE usage rather than block/byte usage (`-i`) — a filesystem can report ample free BYTES while being completely out of free inodes if it holds millions of tiny files (a common failure mode for datasets stored as many small individual sample files rather than sharded archives), causing "No space left on device" errors despite `df -h` showing free capacity.

**Directory-level usage — `du`**

```bash
du -sh /mnt/nvme_dataset/*
```
Summarized (`-s`), human-readable (`-h`) size for each top-level item under the dataset directory — the standard first command for identifying which specific dataset subdirectory is consuming unexpected space.

```bash
du -sh --max-depth=2 /mnt/nvme_dataset
```
Limits the recursion depth to 2 levels (`--max-depth=2`), producing a more digestible summary tree on a deeply nested dataset directory structure than a full unlimited-depth `du -sh *` would.

```bash
du -ah /mnt/checkpoints | sort -rh | head -20
```
Lists ALL files (not just directory summaries, `-a`) under the checkpoints directory, sorted by size descending (`sort -rh`), showing the top 20 largest individual files — the direct tool for finding which specific old checkpoint files are safe to prune when a filesystem is filling up.

**Combined read-throughput validation**

```bash
dd if=/mnt/nvme_dataset/shard_00001.tar of=/dev/null bs=1M status=progress
```
Performs a raw sequential-read throughput test of an actual dataset shard file, streaming it to `/dev/null` while reporting live progress (`status=progress`) — a direct, application-independent measurement of whether the storage layer alone can sustain the read rate a training job's data pipeline will demand, isolating storage throughput from any preprocessing-stage bottleneck.

```bash
fio --name=readtest --filename=/mnt/nvme_dataset/testfile --rw=read --bs=1M --size=10G --direct=1
```
Runs a proper `fio` (Flexible I/O tester) benchmark issuing DIRECT (unbuffered, bypassing page cache, `--direct=1`) sequential reads — the rigorous version of the `dd` test above, giving accurate steady-state throughput numbers uncontaminated by page-cache warm-up effects that would otherwise make a `dd`-based test look artificially fast on a second run.

---

### 4.3 Network Fabric & Remote Clustering

**InfiniBand interface status**

```bash
ibstatus
```
Reports the operational state (Active/Down), link layer, rate, and physical state for every InfiniBand port on the host — the first command for any suspected fabric connectivity issue.

```bash
ibstat
```
A more detailed alternative to `ibstatus`, additionally reporting the HCA firmware version, hardware version, and node/port GUIDs per device — the tool to reach for when you need the GUID for a fabric-management task, not just a pass/fail status check.

```bash
ibv_devinfo
```
Queries the low-level `libibverbs` device information: port state, max MTU, active MTU (which can silently be lower than max if fabric-negotiated down), and link-layer type — the MTU mismatch between "max" and "active" is a classic, easy-to-miss cause of degraded IB throughput.

```bash
ibv_devinfo -v
```
Verbose variant, additionally dumping full capability flags and every port's detailed attribute set — used when the compact form doesn't surface a specific capability flag you need to confirm (e.g., GPUDirect RDMA support flags).

**Link-level diagnostics**

```bash
ibportstate <lid> <port>
```
Queries or sets the physical/logical state of a specific port identified by its Local Identifier and port number — used in fabric-management workflows to manually force-reset a specific stuck port.

```bash
iblinkinfo
```
Fabric-wide topology dump of every link's current state, width, and speed across the entire IB subnet (requires subnet-manager connectivity) — the fabric-scale equivalent of `nvidia-smi topo -m`, showing inter-NODE (not just inter-GPU) connectivity.

```bash
ibhosts
```
Lists every host (CA — Channel Adapter) node currently registered in the IB subnet manager's topology database — a quick census of which nodes the fabric currently considers reachable.

```bash
ibnetdiscover
```
Performs a live topology discovery walk of the entire fabric, mapping switches, hosts, and their interconnections — generates a full fabric map useful for confirming the physical cabling matches the intended logical topology design.

**RDMA subsystem status**

```bash
rdma link show
```
Lists every RDMA-capable link on the host along with its associated netdev interface and state — the modern `iproute2`-family RDMA tooling, complementary to the older `ib*` utility family.

```bash
rdma resource show
```
Reports RDMA resource utilization (queue pairs, completion queues, memory regions currently allocated) — useful for diagnosing RDMA resource exhaustion on a node running many concurrent GPUDirect RDMA-enabled training jobs.

```bash
rdma statistic show
```
Displays cumulative RDMA transport-layer statistics (packet counts, retransmissions, errors) per device — the direct source for spotting a fabric-level retransmission spike correlating with an observed training-throughput degradation.

**Bandwidth and latency micro-benchmarks**

```bash
ib_write_bw -d mlx5_0 -a
```
Runs the standard `perftest`-suite write-bandwidth micro-benchmark against HCA `mlx5_0`, sweeping across all supported message sizes (`-a`) — requires a matching listener process on a remote node; the definitive point-to-point bandwidth validation tool, independent of any GPU or application-level framework.

```bash
ib_write_bw -d mlx5_0 --use_cuda=0 -a
```
The GPUDirect RDMA variant, sourcing the benchmark's data buffer directly from GPU 0's memory (`--use_cuda=0`) rather than host RAM — validates the FULL GPUDirect path (GPU memory → NIC, bypassing the CPU) end-to-end, not just plain host-to-host RDMA bandwidth.

```bash
ib_read_lat -d mlx5_0
```
Runs the companion READ-latency micro-benchmark, reporting round-trip latency distribution rather than bandwidth — the tool for diagnosing latency-sensitive collective-communication stalls that a pure bandwidth test wouldn't surface.

**RoCE (RDMA over Converged Ethernet) specific checks**

```bash
cma_roce_mode -d mlx5_0 -p 1
```
Queries the currently configured RoCE mode (v1 vs v2) for a specific device/port — a mismatch between a node's configured RoCE version and the rest of the fabric's expectation is a common cause of connectivity failures specific to RoCE (as opposed to native IB) deployments.

```bash
show_gids
```
Lists every GID (Global Identifier, the RDMA-layer address analog to an IP address) registered across all local RDMA devices, along with which are RoCEv1 vs RoCEv2 — essential for confirming the correct GID index is being selected by the application/framework layer for a RoCE fabric.

```bash
ethtool -i eth0
```
Reports the driver and firmware version for the underlying Ethernet NIC backing a RoCE interface — confirms the RoCE-capable NIC's driver stack is correctly loaded, a prerequisite check before any higher-level RDMA diagnostic makes sense on an Ethernet-based (rather than native IB) fabric.

```bash
ethtool -S eth0 | grep -i pause
```
Filters the NIC's extended statistics counters for Ethernet PAUSE-frame activity — RoCEv2 fabrics typically rely on Priority Flow Control (a form of lossless Ethernet) rather than legacy PAUSE frames, so unexpectedly high PAUSE counts can indicate a fabric configuration that isn't properly using PFC and is instead falling back to a lossy, congestion-prone mode.

---

## Appendix A: Verified Real-World Session Capture (A100-SXM4-80GB)

Every command below was executed live against a physical `NVIDIA A100-SXM4-80GB` (Driver 550.127.05, CUDA 12.4) mid-training-run, with the raw terminal output preserved verbatim and annotated line-by-line. This appendix exists to ground every abstract flag documented in §1.1 against a real, non-idle GPU — including one command that fails, which is itself instructive.

### A.1 Baseline `nvidia-smi` — full default report

```bash
nvidia-smi
```
```
Sun Aug  9 21:01:26 2026
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4    |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-80GB          On  |   00000000:C1:00.0 Off |                    0 |
| N/A   56C    P0             362W / 400W |   14813MiB /  81920MiB |     100%      Default|
|                                         |                        |            Disabled  |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|===========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

**Field-by-field, mapped directly against the `--query-gpu` properties from §1.1:**

| Field observed | Value | Corresponds to query property | Interpretation |
|---|---|---|---|
| `Driver Version` | 550.127.05 | `driver_version` (#35) | Current display driver — cross-check this against `cuda_version` (#below) whenever a toolkit mismatch is suspected (§2.3). |
| `CUDA Version` | 12.4 | `cuda_version` | Maximum CUDA toolkit version this driver supports — NOT necessarily the installed toolkit; run `nvcc --version` separately to confirm the actual installed compiler version. |
| `Persistence-M` | On | `persistence_mode` (#30) | Correctly enabled — the driver stays resident between jobs, eliminating cold-start re-init latency, exactly the production posture recommended in §1.2. |
| `Bus-Id` | 00000000:C1:00.0 | — | The PCI bus address; feed this into `cat /sys/bus/pci/devices/0000:C1:00.0/numa_node` (§3.1) to confirm this GPU's NUMA affinity before pinning a data-loader process to it. |
| `Fan` | N/A | `fan.speed` (#28) | Expected on an SXM4 (mezzanine, not PCIe-card) form factor — SXM modules have no onboard fan; cooling is chassis-level, so this field is always N/A on SXM4 regardless of actual thermal state. |
| `Temp` | 56C | `temperature.gpu` (#7) | Comfortably below any thermal-throttle threshold; correlate against `clocks_throttle_reasons.hw_slowdown` (#19) if this climbs above ~85C during a run. |
| `Perf` | P0 | — | Maximum performance state — confirms the GPU is NOT in a reduced power/clock state; P0 is the state you want to see during active training. |
| `Pwr:Usage/Cap` | 362W / 400W | `power.draw` (#9) / `power.limit` (#10) | Running at 90.5% of its configured 400W cap — healthy headroom, not currently power-throttled. If this pins at exactly the cap value for sustained periods, check `clocks_throttle_reasons.sw_power_cap` (#18). |
| `Memory-Usage` | 14813MiB / 81920MiB | `memory.used` (#3) / `memory.total` (#4) | ~18% of the 80GB HBM2e capacity in use — plenty of headroom for a larger batch size if throughput (not memory) is the current constraint. |
| `GPU-Util` | 100% | `utilization.gpu` (#1) | Fully saturated compute — a strong sign the training loop is NOT data-starved at this instant (cross-check against §4.2 `iostat`/`iotop` if this ever drops unexpectedly). |
| `Uncorr. ECC` | 0 | `ecc.errors.uncorrected.volatile.total` (#27) | Zero double-bit errors — clean memory health, no action needed. |
| `Compute M.` | Default | `compute_mode` (#29) | Unrestricted — multiple processes CAN open contexts here; if this were meant to be a dedicated single-tenant node, §1.2's `-c 3` (Exclusive_Process) would be the correct hardening step. |
| `MIG M.` | Disabled | `mig.mode.current` (#31) | This GPU is running as one monolithic device, not partitioned — consistent with a single 100%-utilized training job rather than a shared multi-tenant MIG layout. |
| `Processes` | No running processes found | `--query-compute-apps` | ⚠️ Notable: 100% utilization and 14.8GB in use, yet the process table is empty. This is the classic signature of a **container-namespace visibility gap** — the training process is running in a DIFFERENT PID namespace/container than the one `nvidia-smi` was invoked from (common on RunPod/Docker-based nodes, exactly matching this session's `root@23f98001046a` container hostname). Do not mistake this for orphaned/zombie VRAM (§2.1) — `fuser -v /dev/nvidia*` from the HOST namespace, not the container, is required to see the actual owning PID in this situation. |

### A.2 `nvidia-smi dmon -s um -i 0 -d 1` — utilization + memory group, 1-second interval

```bash
nvidia-smi dmon -s um -i 0 -d 1
```
```
# gpu    sm   mem   enc   dec   jpg   ofa    fb  bar1  ccpm
# Idx     %     %     %     %     %     %    MB    MB    MB
    0   100    21     0     0     0     0 14813     4     0
    0   100    21     0     0     0     0 14813     4     0
    0   100    20     0     0     0     0 14813     4     0
    0    81    17     0     0     0     0 14813     4     0
    0    71    16     0     0     0     0 14813     4     0
    0    85    17     0     0     0     0 14813     4     0
    0    95    17     0     0     0     0 14813     4     0
    0   100    17     0     0     0     0 14813     4     0
    0    86    15     0     0     0     0 14813     4     0
    0    81    16     0     0     0     0 14813     4     0
    0    81    15     0     0     0     0 14813     4     0
    0    86    17     0     0     0     0 14813     4     0
    0   100    19     0     0     0     0 14813     4     0
    0    79    13     0     0     0     0 14813     4     0
    0   100    19     0     0     0     0 14813     4     0
    0   100    19     0     0     0     0 14813     4     0
    0   100    19     0     0     0     0 14813     4     0
```

**Column-by-column:**
- **`sm` (SM utilization %)** — oscillating between 71% and 100% rather than pinned flat at 100%. This sawtooth pattern is the classic signature of a training loop alternating between a compute-bound forward/backward pass (SM near 100%) and a brief data-dependent stall (SM dips to 71–86%) — likely the optimizer step or a host-side data-loader handoff between batches, not a serious bottleneck given the dips never fall below ~70%.
- **`mem` (memory-controller utilization %)** — steady in the 13–21% range, well below `sm`. Low memory utilization alongside high SM utilization confirms this kernel is **compute-bound, not memory-bandwidth-bound** — directly matching the diagnostic pattern described for `dram__throughput` in the Nsight Compute section (§3.3): if `mem` were tracking `sm` closely instead, that would flag a memory-bound kernel worth profiling with `ncu --section MemoryWorkloadAnalysis`.
- **`enc`/`dec`/`jpg` (0% throughout)** — the video encode, decode, and JPEG hardware engines are completely idle, exactly as expected for a pure numerical training workload with no media transcoding component.
- **`ofa` (0%)** — the Optical Flow Accelerator engine, also idle — irrelevant to this workload, present only because it's a fixed-function block on the die that `dmon`'s memory-group column set always reports.
- **`fb` (Frame Buffer, MB)** — pinned exactly at `14813`, matching the `Memory-Usage` field from the full report in A.1 verbatim — confirms VRAM allocation is stable across this sampling window (no leak, no growth).
- **`bar1` (MB)** — steady at `4` MB, the small PCIe-mapped aperture used for host-visible direct GPU memory access; a tiny, stable BAR1 footprint here is unremarkable for a straightforward training job with no GPUDirect Storage or peer-mapped buffers in play.
- **`ccpm` (MB)** — reads `0` throughout. This column reports memory allocated within a Confidential-Computing protected region (introduced in the R525+ driver branch alongside Hopper's CC-mode support). Zero here is expected and correct: this A100 instance is not running with Confidential Computing enabled, so no memory is CC-protected.

### A.3 The failed command — `nvidia-smi -q -d PCIE`

```bash
nvidia-smi -q -d PCIE
```
```
Failed to parse --display/-d flags
```

**Why this fails, and what to use instead:** the `-d` flag for `-q` (detailed query mode) only accepts a fixed, enumerated set of section tokens — valid values include `MEMORY`, `UTILIZATION`, `ECC`, `TEMPERATURE`, `POWER`, `CLOCK`, `COMPUTE`, `PIDS`, `PERFORMANCE`, `SUPPORTED_CLOCKS`, `PAGE_RETIREMENT`, `ACCOUNTING`, `ENCODER_STATS`, `VOLTAGE`, `FBC_STATS`, `ROW_REMAPPER`, `RESET_STATUS`, and `GSP_FIRMWARE_VERSION` — **`PCIE` is not among them**, which is exactly why the parser rejects it outright rather than returning empty or partial output. There is no dedicated `-d` section for PCIe link state; instead, PCIe generation/width data is either (a) embedded in the unfiltered full `nvidia-smi -q` report, or (b) pulled directly via the targeted `--query-gpu=pcie.link.gen.current,pcie.link.width.current` pattern already documented as query properties #20–23 in §1.1. **The corrected, working equivalent of this operator's intent:**
```bash
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
```
This is a useful real-world lesson worth keeping in this manual verbatim: `nvidia-smi -q -d <token>` section names do not always match intuition (there is no `PCI` or `PCIE` token despite PCIe being a heavily-queried subsystem elsewhere in this document) — when in doubt, `--query-gpu=` with an explicit property name is the more reliable, typo-resistant path, and is the pattern this entire manual defaults to throughout §1.1.

### A.4 `nvidia-smi dmon -s t -i 0 -d 1` — PCIe throughput group

```bash
nvidia-smi dmon -s t -i 0 -d 1
```
```
# gpu   rxpci   txpci
# Idx    MB/s    MB/s
    0      11       1
    0       3       4
    0       4       2
    0       4       3
    0       3       2
    0       6       2
    0      72       6
    0       6       1
```

**Interpretation:** `rxpci`/`txpci` report host-to-device (RX, into the GPU) and device-to-host (TX, out of the GPU) PCIe traffic in MB/s, sampled per second. For most of this window, traffic sits in the low single-digit MB/s range — consistent with a steady-state training loop where each batch is already resident on-device and only small artifacts (loss scalars, gradient norms for logging) cross the bus each step. The single spike to **72 MB/s RX** stands out sharply against that baseline — most plausibly a new batch of raw input data (or an intermediate checkpoint shard) being staged host-to-device in that particular second, then returning to baseline immediately after. Note this entire multi-second capture (a peak of 72 MB/s) sits nowhere near the theoretical ceiling for this card's negotiated link — cross-reference against `pcie.link.width.current`/`pcie.link.gen.current` (properties #20–23, and the corrected query in A.3) to confirm the link itself isn't secretly downgraded; a healthy Gen4 x16 link has vastly more headroom than 72 MB/s ever exercises here, so PCIe bandwidth is definitively not this workload's bottleneck.

### A.5 `nvidia-smi dmon --gpm-metrics 2,3,5 -i 0 -d 1` — GPU Metrics (GPM) subsystem

```bash
nvidia-smi dmon --gpm-metrics 2,3,5 -i 0 -d 1
```
```
# gpu   pwr  gtemp  mtemp    sm   mem   enc   dec   jpg   ofa    mclk    pclk  smutil  smocc  mmaact
# Idx     W      C      C     %     %     %     %     %     %     MHz     MHz  GPM:%   GPM:%   GPM:%
    0   360     55     56   100    16     0     0     0     0    1593    1410      –       –       –
    0   362     56     56    97    16     0     0     0     0    1593    1410      –       –       –
    0   364     55     56   100    16     0     0     0     0    1593    1410      –       –       –
    0   362     55     56    88    14     0     0     0     0    1593    1410      –       –       –
```
```bash
nvidia-smi dmon --gpm-metrics 2,3,5 -i 0 -d 1
```
```
# gpu   pwr  gtemp  mtemp    sm   mem   enc   dec   jpg   ofa    mclk    pclk  smutil  smocc  mmaact
# Idx     W      C      C     %     %     %     %     %     %     MHz     MHz  GPM:%   GPM:%   GPM:%
    0   354     55     55   100    28     0     0     0     0    1593    1410      –       –       –
    0   378     55     55   100    28     0     0     0     0    1593    1410      –       –       –
    0   362     55     50    61    15     0     0     0     0    1593    1410      –       –       –
    0   101     55     56    61     0     0     0     0     0    1593    1410      –       –       –
```

**Column-by-column:**
- **`pwr` (W)** — 354–378W across both captures, consistent with the 362W figure from the baseline full report in A.1 — this GPU is running comfortably in its high-power, high-throughput operating regime.
- **`gtemp` / `mtemp` (Core / HBM memory temperature, °C)** — GPU-core and memory-die temperatures reported SEPARATELY (unlike the single `temperature.gpu` field in the standard report), both sitting in the mid-50s°C — healthy on both fronts, with core and memory tracking closely together rather than one running hot independently, which would otherwise suggest a memory-specific cooling issue.
- **`sm`/`mem`** — same meaning as A.2, again showing the compute-bound sawtooth pattern; note the second capture's final row drops to `sm=61%, mem=0%, pwr=101W` — a brief near-idle moment, likely an inter-step boundary or checkpoint-save pause caught mid-sample.
- **`mclk`/`pclk` (Memory clock / SM clock, MHz)** — pinned rock-steady at `1593`/`1410` MHz across every single sample in both captures. This is a critical observation: this GPU is running at **fixed, locked clocks**, not the driver's default dynamic-boost behavior — exactly the kind of deterministic clock profile that `nvidia-smi -lgc`/`-lmc` (§1.2) are used to establish, almost certainly set intentionally on this node for benchmark reproducibility.
- **`smutil` / `smocc` / `mmaact` (GPM:%, all showing `–`)** — GPM (GPU Metrics) is a newer, finer-grained profiling subsystem exposing per-partition SM occupancy and memory-active-cycle metrics beyond what classic `dmon` groups provide. The `–` (dash, not `0`) across all three requested metric IDs (2, 3, 5) specifically means **these GPM metrics are UNSUPPORTED on this device/driver combination**, not that their value is zero — GPM's fuller metric set (including per-SM-partition occupancy) has stronger support on Hopper-generation silicon; on this A100 (Ampere), the base `dmon` utilization columns (`sm`, `mem`) remain the reliable source of truth, and the GPM columns can be disregarded for this specific GPU generation rather than investigated as a fault.

---

> *AX100/A100 Cluster Admin · github.com/rpaut03l/TS-02-03*
