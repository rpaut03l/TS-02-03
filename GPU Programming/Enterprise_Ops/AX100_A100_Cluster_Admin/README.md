# 🧩 AX100 / A100 Cluster Administration

### *Enterprise Linux GPU node administration — monitoring, recovery, tuning, fabric diagnostics*

> **Nav:** [← Enterprise Ops](../README.md) | [← GPU Programming](../../README.md)

---

## 👶 What this folder is

[`A100_MIG_DCGM_GKE/`](../A100_MIG_DCGM_GKE/) covers Kubernetes/GKE-specific MIG and DCGM diagnostics. This folder is the broader, platform-agnostic companion: bare-metal and general enterprise Linux cluster administration for AX100/A100 nodes — the commands you'd reach for whether the node is running Kubernetes, Slurm, or nothing at all.

---

## 📁 Files in this folder

| File | What it is |
|---|---|
| [gpu_enterprise_ax100_a100_cluster_admin_reference.md](gpu_enterprise_ax100_a100_cluster_admin_reference.md) | 200+ commands across `nvidia-smi` monitoring, MIG/ECC/power/clock management, `nvcc`/`nvidia-debugdump`, MPS, VRAM reclamation, NVLink/fabric topology, kernel module interrogation, OS/IPMI log parsing, NUMA/AVX-512 tuning, NCCL/UVM environment variables, Nsight Systems/Compute profiling, PyTorch assertion snippets, NVMe I/O monitoring, and InfiniBand/RoCE diagnostics — **plus Appendix A: a verified real-world terminal session captured live on an A100-SXM4-80GB node, annotating actual `dmon`, `--gpm-metrics`, and full-report output line by line, including a real failed command and why it fails** |

---

> *GPU Programming · Enterprise Ops · github.com/rpaut03l/TS-02-03*
