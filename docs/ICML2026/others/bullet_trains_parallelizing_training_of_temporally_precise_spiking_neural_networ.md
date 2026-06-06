---
title: >-
  [Paper Note] Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks
description: >-
  [ICML 2026][Spiking Neural Networks] The authors propose a parallel training method for Spiking Neural Networks (SNNs) based on parallel associative scan…
tags:
  - "ICML 2026"
  - "Spiking Neural Networks"
  - "Parallel Associative Scan"
  - "Precise Spike Timing"
  - "Event-driven"
  - "Neuromorphic Computing"
date: 2026-05-08
content_hash: 9914b6ab3e99c74c
---

# Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2603.13283](https://arxiv.org/abs/2603.13283)  
**Code**: https://github.com/ToddMorrill/snn-bullet-trains  
**Area**: Spiking Neural Networks / Parallel Training  
**Keywords**: Spiking Neural Networks, Parallel Associative Scan, Precise Spike Timing, Event-driven, Neuromorphic Computing

## TL;DR

The authors propose a parallel training method for Spiking Neural Networks (SNNs) based on parallel associative scan, achieving up to 44x speedup while maintaining exact hard-reset dynamics and utilizing a differentiable numerical root solver for machine-precision spike timing calculations.

## Background & Motivation

**Background**: Spiking Neural Networks (SNNs) process information in an event-driven manner, performing computations only when spikes occur, which naturally aligns with biological neural computation and neuromorphic hardware. However, current SNN research primarily relies on GPU training, which faces severe parallelization bottlenecks.

**Limitations of Prior Work**: The "charge–fire–reset" dynamics of SNNs are inherently sequential—a neuron must determine whether it generates an output spike after consuming each input spike before processing the next. This leads to training times that grow linearly $O(N)$ with the number of spikes, proving highly inefficient on GPUs. Existing parallelization methods either completely remove the reset mechanism (PSN), use soft-reset approximations (SPikE-SSM), or relax discontinuous spike generation into continuous sigmoid proxies (FPT), all of which deviate from exact hard-reset semantics.

**Key Challenge**: A fundamental contradiction exists between parallelization and exact hard-reset dynamics—the non-linear dependence introduced by hard-reset blocks full parallelization, while abandoning it reduces the neuron's non-linear expression and biological fidelity. Furthermore, nearly almost all existing implementations rely on discrete time grids, limiting spike time precision to the time step size and failing to distinguish the order of spikes within the same window.

**Goal**: (1) Achieve parallel processing of SNN spike events while maintaining exact hard-reset; (2) Implement machine-precision spike time solving independent of discrete-time approximations.

**Key Insight**: The authors observe that the subthreshold state transition of a Leaky Integrate-and-Fire (LIF) neuron can be represented as an affine map. Since the composition of affine maps remains an affine map, it naturally satisfies the associativity required for parallel associative scans. Using speculative chunked execution allows for parallel processing of spikes within chunks, while analytical checks quickly locate output spikes.

**Core Idea**: Use parallel associative scans to consume multiple input spikes simultaneously, combined with a Newton-Raphson root solver to precisely locate spike times, achieving significant GPU speedup while fully maintaining hard-reset semantics.

## Method

### Overall Architecture

The system operates in an event-driven manner: each LIF neuron maintains an input spike queue, partitioning input spikes into chunks of a fixed size $K$. Within each chunk, a parallel associative scan is used to compute all future states at once. An analytical check determines if an output spike exists within the chunk; if so, a Newton-Raphson solver locates the exact spike time, performs a hard-reset, and proceeds to the next chunk. The final layer uses weighted leaky integrators to transform spike trains into classification logits. The computational depth of the entire process is $O(C \log K)$, where $C$ is the number of chunks and $K$ is the chunk size.

### Key Designs

1.  **Parallel Associative Scan**:
    *   **Function**: Reduces the $O(N)$ operation of sequential processing for $N$ input spikes to a parallel depth of $O(C \log K)$.
    *   **Mechanism**: The subthreshold transition of an LIF neuron from state $\mathbf{s}_0 = [V_0, I_0]^\top$ to $\mathbf{s}_1 = [V_1, I_1]^\top$ can be expressed as an affine map $\mathbf{s}_1 = M_1 \mathbf{s}_0 + \mathbf{b}_1$. Here, $M_1$ is a decay matrix determined by membrane and synaptic time constants, and $\mathbf{b}_1$ encodes synaptic weight injection. The composition of two consecutive affine maps $\text{Combine}((M_2, \mathbf{b}_2), (M_1, \mathbf{b}_1)) = (M_2 M_1, M_2 \mathbf{b}_1 + \mathbf{b}_2)$ is still an affine map and satisfies associativity. Thus, JAX's `associative_scan` can compute all intermediate states for a chunk of $K$ spikes in $O(\log K)$ depth.
    *   **Design Motivation**: State Space Models (SSMs) face similar temporal parallelization issues, and associative scans are widely used there. The key contribution here is extending this to SNN scenarios with non-linear hard-resets, bypassing sequential dependencies via speculative execution and analytical spike detection.

2.  **Differentiable Spike Time Solver**:
    *   **Function**: Solves for continuous spike time $t^\star$ with machine precision without discrete-time approximations or restrictive assumptions on the neuron model.
    *   **Mechanism**: A root function is defined as $R(\mathbf{p}, t) = V(V_0, I_0, t) - V_{\text{th}} = 0$. Within the interval between each pair of consecutive input spikes, the system first analytically calculates the time of maximum voltage $t_{V_{\max}}$ and the corresponding voltage $V(t_{V_{\max}})$ to determine if a spike occurs. If it does, Newton-Raphson iteration solves for the exact spike time. The unimodality of the voltage function within the interval ensures uniqueness of the spike time and convergence of the solver. Gradients are computed directly using the Implicit Function Theorem $\partial t^\star / \partial \mathbf{p}$ rather than backpropagating through iteration steps.
    *   **Design Motivation**: Analytical spike time solutions require model constraints (e.g., $\tau_m = 2\tau_s$), limiting flexibility. A numerical root solver applies to any neuron model while avoiding precision loss from discretization and preventing computation/memory from scaling linearly with the number of time steps.

3.  **Speculative Chunked Execution**:
    *   **Function**: "Speculatively" processes an entire chunk of inputs in parallel despite not knowing if an output spike will occur, balancing parallel efficiency with hard-reset correctness.
    *   **Mechanism**: A fixed chunk size $K$ (e.g., 128) is set. After performing an associative scan on the chunk, the system checks in parallel whether any interval between adjacent input spikes triggers an output spike. If output spikes exist, computations after the first output spike are discarded; the system resets at the spike time and restarts from the next chunk. A maximum number of output spikes per neuron $S_{\max}$ is defined, paired with a spike count regularizer to maintain sparse firing.
    *   **Design Motivation**: Since neuron firing is typically sparse relative to input spikes, most chunks do not generate output spikes, leaving very little discarded computation. This "speculate-then-verify" strategy leverages the massive parallelism of GPUs; even with occasional wasted computation, the overall throughput significantly outperforms sequential processing.

### Loss & Training

The output layer employs $N_{\text{cls}}$ weighted leaky integrators, converting spike trains into logits via an exponential decay weighted integral $\int_0^{\tau_{\max}} e^{-t/\tau_{\text{LI}}} V(t) dt$, where earlier spikes receive higher weights. Cross-entropy loss is used for training, supplemented by a spike count regularizer to control firing sparsity. Synaptic weights $w_{ij}$ and learnable synaptic delays $d_{ij}$ are optimized end-to-end via exact gradients (rather than surrogate gradients).

## Key Experimental Results

### Main Results

| Dataset | Method | Exact Gradient | Continuous Spike Time | Parallelized | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MNIST | Göltz et al. (1F350H, $\tau_m=2\tau_s$) | ✓ | ✓ | ✗ | 97.20% |
| MNIST | Wunderlich & Pehle (1F350H) | ✓ | ✓ | ✗ | 97.60% |
| MNIST | **Ours** (1F350H) | ✓ | ✓ | ✓ | **98.04%** |
| SHD | Hammouamri et al. (2F256HD) | ✗ | ✗ | ✗ | **95.07%** |
| SHD | Mészáros et al. (2F512HD) | ✓ | ✗ | ✗ | 93.10% |
| SHD | **Ours** (2F512HD) | ✓ | ✓ | ✓ | 94.96% |
| SSC | Hammouamri et al. (2F512HD) | ✗ | ✗ | ✗ | **80.69%** |
| SSC | Mészáros et al. (2F512HD) | ✓ | ✗ | ✗ | 76.10% |
| SSC | **Ours** (2F512HD) | ✓ | ✓ | ✓ | 77.79% |

### Ablation Study

| Configuration | Speedup | Description |
| :--- | :--- | :--- |
| Max Speedup (SHD) | 44× | Relative to sequential event-driven baseline |
| chunk size = 128 | Optimal | Stable performance across various batch sizes and hidden dims |
| Yin-Yang, $\Delta t \to 0$ (Continuous) | Highest Accuracy | Full temporal resolution |
| Yin-Yang, $\Delta t = 1$ ms | Significant drop | Discretization leads to accuracy loss |
| Yin-Yang, $\Delta t \geq 2$ ms | ~33% (Random) | Complete loss of temporal encoding capability |

### Key Findings

*   **Parallel associative scan achieves up to 44x speedup while maintaining exact hard-resets**, showing significant advantages particularly with large batch sizes and hidden dimensions where sequential methods suffer from sharply increasing training times.
*   **A chunk size of 128 is a robust choice**; while larger chunks increase parallelism, they also increase memory bandwidth pressure and discarded computation. In practice, due to sparse firing, wasted computation is minimal.
*   **Continuous spike timing is crucial for temporal encoding tasks**: In the Yin-Yang ITD task, discretization at $\Delta t \geq 2$ ms reduces accuracy to random levels, while the continuous method maintains peak precision.
*   On SHD/SSC, the proposed method slightly trails surrogate gradient methods (Hammouamri et al.). The authors attribute this to current benchmarks relying primarily on rate-coding, where the smoothness of surrogate gradients provides an advantage.

## Highlights & Insights

*   **Transfer of Associative Scan from SSM to SNN**: Successfully migrating the mature parallel associative scan technique from State Space Models to SNNs with non-linear hard-resets. The speculative execution strategy is the core adaptation—a "parallelize first, correct later" paradigm with potential applications in other parallel problems involving conditional branching.
*   **Gradients via Implicit Function Theorem**: Instead of backpropagating through solver iterations, the gradient of spike time with respect to parameters is obtained directly from $R(\mathbf{p}, t^\star) = 0$ using the Implicit Function Theorem. This is both elegant and efficient, applicable to any scenario involving differentiable root solving.
*   **Numerical Solvers Liberating Neuron Models**: Traditional analytical methods require constraints like $\tau_m = 2\tau_s$. Numerical methods break these restrictions, enabling heterogeneous time constants and more complex neuron models.

## Limitations & Future Work

*   Currently validated only on fully connected feedforward architectures; **not yet extended to convolutional or recurrent SNNs**, where input queues for each neuron are dynamic and parallelization is more challenging.
*   Current benchmarks (SHD, SSC) **primarily rely on rate-coding**, failing to fully exploit the advantages of continuous spike timing. The community lacks large-scale, strictly temporal encoding benchmarks.
*   Fixed computational budgets (chunk count $C$ and output spike limit $S_{\max}$) might result in some input spikes remaining unprocessed in extreme high-firing-rate scenarios.
*   The impact of continuous spike time training on deployment has not yet been verified on actual neuromorphic hardware.

## Related Work & Insights

*   **PSN** (Fang et al., 2023): Removes the reset mechanism, reducing it to a linear filter solvable by convolution; efficient but loses non-linear expression.
*   **SPikE-SSM** (Zhong et al., 2024): Decouples reset from integration, using soft-resets (linear subtraction) instead of hard-resets.
*   **FPT** (Feng et al., 2025): Models hard-resets via fixed-point iteration scanning, but forward propagation must be relaxed to continuous sigmoid to ensure convergence.
*   **EventProp** (Wunderlich & Pehle, 2021): Exact gradient SNN training in continuous time, but processed sequentially. The proposed method can be viewed as an accelerated version of this.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition](../../ICLR2026/others/training_deep_normalization-free_spiking_neural_networks_with_lateral_inhibition.md)
- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](../../AAAI2026/others/pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[ICML 2026\] Singular Bayesian Neural Networks](singular_bayesian_neural_networks.md)
- [\[AAAI 2026\] TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling](../../AAAI2026/others/tdsnns_competitive_topographic_deep_spiking_neural_networks_for_visual_cortex_mo.md)
- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](../../AAAI2026/others/i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)

</div>

<!-- RELATED:END -->
