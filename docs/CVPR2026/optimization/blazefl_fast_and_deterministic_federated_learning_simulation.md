---
title: >-
  [Paper Note] BlazeFL: Fast and Deterministic Federated Learning Simulation
description: >-
  [CVPR 2026 (Workshop: FedVision)][Optimization][Federated learning simulation] BlazeFL is a lightweight single-machine federated learning simulation framework built on Python free-threading. By combining shared-memory execution with per-client isolated RNG streams, it achieves up to 3.1× speedup and bit-level reproducibility.
tags:
  - "CVPR 2026 (Workshop: FedVision)"
  - Optimization
  - Federated learning simulation
  - deterministic reproducibility
  - free-threading
  - shared memory
  - FedAvg
date: 2026-05-08
content_hash: b80110817b94308e
---

# BlazeFL: Fast and Deterministic Federated Learning Simulation

**Conference**: CVPR 2026 (Workshop: FedVision)
**arXiv**: [2604.03606](https://arxiv.org/abs/2604.03606)
**Code**: [GitHub](https://github.com/kitsuyaazuma/blazefl)
**Area**: Federated Learning / System Optimization
**Keywords**: Federated learning simulation, deterministic reproducibility, free-threading, shared memory, FedAvg

## TL;DR

BlazeFL is a lightweight single-machine federated learning simulation framework built on Python free-threading. By combining shared-memory execution with per-client isolated RNG streams, it achieves up to 3.1× speedup and bit-level reproducibility.

## Background & Motivation

**State of the Field**: FL research commonly relies on single-machine simulation, emulating hundreds to thousands of virtual clients within a single training-aggregation loop. Mainstream frameworks such as Flower (Ray backend), FedML, and pfl-research adopt multiprocessing or external distributed runtimes (Ray, MPI, NCCL, Horovod) for parallelism.

**Limitations of Prior Work**: Single-machine simulation faces two core challenges. (1) **Efficiency bottleneck**: In multiprocess architectures, each communication round requires cross-process serialization and transfer of model parameters. For computer vision tasks involving large models (deep ResNets, ViTs, etc.), serialization and IPC overhead severely limit simulation throughput. (2) **Lack of reproducibility**: FL simulation involves multiple sources of randomness (client sampling, data partitioning, mini-batch ordering, data augmentation, regularization). Parallel execution introduces non-determinism through shared random state and completion-order-dependent aggregation. Even with manually set global seeds in Flower, repeated runs yield different model weights and final accuracy.

**Root Cause**: There is a fundamental trade-off between throughput and reproducibility — increasing parallelism accelerates simulation but amplifies non-determinism, forcing researchers to choose between speed and reproducibility, or to write complex control logic.

**Paper Goals**: Design a single-machine FL simulation framework that simultaneously addresses efficiency and reproducibility without requiring an external distributed runtime.

**Starting Point**: Python 3.14's free-threading (PEP 703 / PEP 779) bypasses GIL constraints, enabling true parallelism with threads instead of processes. Threads sharing an address space naturally eliminate serialization overhead; combined with per-client independent RNG streams, determinism is guaranteed.

**Core Idea**: Single-process thread-level shared-memory execution + per-client isolated RNG streams + fixed-order result collection = fast and bit-level reproducible FL simulation.

## Method

### Overall Architecture

BlazeFL's architecture is organized around a single-process execution model. The main thread coordinates the entire FL loop: client sampling → parameter distribution → parallel local training → result collection → server aggregation → global evaluation. Worker threads execute client training tasks within a shared address space; parameter exchange between server and clients is performed via shared memory, eliminating serialization, IPC, and external object stores. The framework also provides a process-based shared-memory mode alongside the free-threaded mode, enabling ablation experiments that isolate the contributions of the execution model and RNG control respectively.

### Key Designs

1. **Shared-Memory Execution (Free-Threading Execution)**:

    - **Function**: Executes virtual clients as worker threads in parallel within a single process, enabling high-concurrency FL simulation.
    - **Mechanism**: Leverages Python 3.14's free-threading support (GIL removal) so that client threads directly read and write model parameters in a shared address space. Downlink packages prepared by the server are consumed directly from shared memory by worker threads, and client training outputs are returned to the server through the same shared space. This completely eliminates cross-process serialization, pipe communication, and external object store overhead.
    - **Design Motivation**: Traditional Python threads are constrained by the GIL and cannot achieve true CPU parallelism, which is why existing FL frameworks universally adopt multiprocessing. However, multiprocessing introduces process boundaries, runtime orchestration, and redundant parameter transfer overheads that are particularly pronounced in communication-intensive single-machine simulation. Free-threading removes this constraint, enabling threads to achieve true parallelism while sharing memory.

2. **Controlled Deterministic Execution**:

    - **Function**: Guarantees bit-level identical training trajectories across repeated runs with varying degrees of parallelism on a fixed software/hardware stack.
    - **Mechanism**: Each client is bound to an independent RNG suite initialized from a deterministic seed schedule. All local random operations within a client (data sampling, shuffling, augmentation, dropout) consume only that client's own RNG stream, fully decoupled from scheduling order. Crucially, results are collected in the fixed order of the sampled client list (rather than completion order), ensuring consistent floating-point accumulation order during FedAvg aggregation.
    - **Design Motivation**: Non-determinism in FL simulation has two independent sources: (a) shared or inconsistent random state causes RNG consumption to depend on scheduling order; (b) completion-order-dependent aggregation causes floating-point addition order to vary (floating-point addition is non-associative), producing minute differences even with fixed seeds that amplify across rounds. Both sources must be addressed simultaneously.

3. **Protocol-Based Interfaces**:

    - **Function**: Reduces framework lock-in, allowing users to integrate existing PyTorch training code with minimal modification.
    - **Mechanism**: Uses Python `typing.Protocol` (PEP 544) to define interfaces, accepting any object that implements the required methods as a valid server handler or client trainer without requiring inheritance from a specific base class. Standard PyTorch training components can be used directly.
    - **Design Motivation**: Many FL frameworks require researchers to adapt to framework-specific base classes, lifecycle hooks, or runtime object hierarchies, making small experimental changes unnecessarily invasive. Protocol-based interfaces preserve static type checking while keeping user code close to a standard PyTorch training loop.

### Loss & Training

BlazeFL is a system framework rather than an algorithmic contribution. Experiments adopt standard FedAvg aggregation. Specific settings: 100 clients, Non-IID partitioning (each client holds only 2 classes), a subset of clients is selected per round for parallel training, each client trains locally for 5 epochs (500 samples), the server takes a weighted average of collected model updates, and the global model is evaluated on 10,000 test samples. The full FL loop runs for 5 communication rounds. Runtime dependencies are minimal — only the Python standard library (threading, multiprocessing) and PyTorch, with no external schedulers or RPC stacks.

## Key Experimental Results

### Main Results: Throughput Comparison (Wall-Clock Time over 5 Communication Rounds)

**High-Performance Server** (48-core CPU + NVIDIA H100 + 192 GB RAM):

| Model | Optimal Parallelism | BlazeFL (FT) Speedup vs. Flower | Notes |
|-------|--------------------|---------------------------------|-------|
| CNN | P=32~64 | **Up to 3.1×** | Communication-intensive; shared memory advantage largest |
| ResNet-18 | P=16~32 | **Up to 1.4×** | Increased compute fraction reduces advantage |
| ResNet-50 | P=8~16 | **Up to 1.1×** | Compute-dominated; advantage narrows |
| ResNet-101 | P=8 | ~1.0× | Fully compute-dominated; near parity |

**Workstation** (32-core CPU + NVIDIA Quadro RTX 6000 + 256 GB RAM):

| Model | BlazeFL (FT) vs. Flower | Notes |
|-------|------------------------|-------|
| CNN | BlazeFL clearly faster | Lightweight workload advantage consistent |
| ResNet-18 | Comparable | Trend weakens |
| ResNet-50 | Flower slightly faster | VRAM constraints cause CUDA allocator lock contention |
| ResNet-101 | Flower slightly faster | Same; process isolation helps avoid single-process lock bottleneck |

The degradation of BlazeFL on large models under workstation conditions is attributed to the RTX 6000's 24 GB VRAM limit: under high concurrency, PyTorch's CUDA caching allocator global mutex becomes a bottleneck, and lock contention is more severe in single-process multithreaded mode than in multiprocess mode, where each process has an independent CUDA context.

### Ablation Study: Reproducibility Analysis

**Experiment 1: 10 Repeated Runs at Fixed Parallelism P=32**

| Configuration | Final Accuracy Std Dev (pp) | Round-level SHA-256 Hash Consistent |
|--------------|----------------------------|-------------------------------------|
| Flower (no seed control) | 1.24 | ❌ |
| Flower (manual global seed) | 0.18 | ❌ |
| BlazeFL (process mode) | **0.00** | ✅ |
| BlazeFL (free-threaded) | **0.00** | ✅ |

**Experiment 2: Determinism Across Parallelism Levels (BlazeFL free-threaded, same seed)**

| Parallelism P | Accuracy Difference vs. P=1 (pp) | Hash Consistent vs. P=1 |
|--------------|----------------------------------|------------------------|
| 1 | — | — |
| 2 | 0.0 | ✅ |
| 4 | 0.0 | ✅ |
| 8 | 0.0 | ✅ |
| 16 | 0.0 | ✅ |
| 32 | 0.0 | ✅ |
| 64 | 0.0 | ✅ |

Final accuracy is 20.53% across all parallelism levels, and SHA-256 hashes for all 5 rounds are identical to those at P=1.

### Key Findings

- In communication-intensive scenarios (lightweight CNN), shared-memory execution yields the largest speedup (3.1×), as serialization and IPC overhead is eliminated.
- As model size increases (ResNet-50/101), computation dominates and the marginal benefit of communication optimization diminishes.
- BlazeFL's determinism covers not only final accuracy but the **entire training trajectory** (SHA-256 hashes of the global model at each round are fully identical).
- Diagnosis of Flower's non-determinism reveals the root cause: floating-point accumulation order during aggregation varies with scheduling, and initial $10^{-6}$-level errors are amplified across rounds by local training and aggregation (logit $L_2$ distances diverge in a fan-like pattern).
- In VRAM-constrained environments, BlazeFL's single-process multithreaded mode may underperform multiprocess mode due to CUDA allocator global lock contention; falling back to BlazeFL's process-based mode is recommended in such cases.

## Highlights & Insights

- **First FL simulation framework based on Python free-threading**, demonstrating the practical value of PEP 703/779 in real ML systems and serving as a reference for other Python applications requiring high-concurrency shared memory.
- **Advances reproducibility from "consistent final metrics" to "bit-level identical training trajectories"**, which is critical for fine-grained behavioral analysis of FL algorithms (e.g., round-wise model weight evolution, client contribution tracking).
- The diagnostic analysis of Flower's non-determinism is instructive: manually setting global seeds (random/numpy/torch) is insufficient to guarantee reproducibility, as the non-associativity of floating-point addition accumulates and amplifies across communication rounds.
- The minimalist dependency design (Python standard library + PyTorch only) lowers the installation barrier and reduces experimental fragility caused by version changes in external runtimes.

## Limitations & Future Work

- **Single-machine simulation only**: Multi-node distributed deployment is not supported; extending to multi-machine would require network communication and distributed synchronization, fundamentally altering the performance and reproducibility model.
- **Determinism requires a fixed software/hardware stack**: Bit-level consistency across machines or platforms is not guaranteed; different library versions, kernels, or hardware may alter floating-point behavior.
- **RNG management challenges in vision pipelines**: Some torchvision transforms (e.g., random crop, random flip) internally rely on global RNG state rather than explicit generators; users must ensure that data augmentation pipelines are compatible with BlazeFL's explicit generator management.
- **Ecosystem maturity**: Python free-threading is still in early stages; some third-party libraries (especially those with complex native extensions) have not yet adapted, limiting fair baseline comparisons (Flower cannot run under Python 3.14).
- **Suboptimal performance in VRAM-constrained settings**: Single-process multithreaded mode degrades under VRAM pressure due to CUDA allocator lock contention, requiring a fallback to process-based mode.
- Multi-GPU scenarios are not evaluated, and non-FedAvg aggregation algorithms are not tested.

## Related Work & Insights

- **Flower**: The most mainstream open-source FL framework, using a Ray backend for distributed simulation. Feature-rich but incurs serialization overhead in single-machine communication-intensive scenarios.
- **FedML**: A comprehensive FL library supporting multiple backends (MPI, NCCL, etc.), more oriented toward production deployment.
- **pfl-research**: Apple's privacy-focused federated learning simulation framework, specializing in differential privacy scenarios.
- **PyTorch shared memory tensors**: Placing tensors in shared memory via `torch.multiprocessing` reduces serialization overhead but still requires explicit process management.
- **Insights**: This work prompts reflection on the fact that in many ML systems engineering problems, new Python language features (such as free-threading) may offer cleaner solutions than external runtimes. Moreover, "reproducibility" should not be limited to approximate consistency of final metrics, but should pursue full controllability of the training trajectory.

## Rating

- **Novelty**: ⭐⭐⭐ First application of Python free-threading to FL simulation with a novel angle of attack, though the core idea (shared memory + isolated RNG) is not complex.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematic evaluation across two hardware platforms, four model architectures, and seven parallelism levels; rigorous reproducibility verification at SHA-256 hash level; includes root-cause diagnosis of Flower's non-determinism.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, honest and detailed limitations discussion (five dedicated subsections), well-matched figures and tables.
- **Value**: ⭐⭐⭐ A reasonable contribution as a workshop paper with direct practical value for FL systems researchers, though the scope is limited to single-machine simulation.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions](../../AAAI2026/optimization/ecpv2_fast_efficient_and_scalable_global_optimization_of_lipschitz_functions.md)
- [\[ICLR 2026\] DeepAFL: Deep Analytic Federated Learning](../../ICLR2026/optimization/deepafl_deep_analytic_federated_learning.md)
- [\[CVPR 2026\] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated learning](scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)
- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](../../AAAI2026/optimization/data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[CVPR 2026\] Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift](fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_.md)

<!-- RELATED:END -->
