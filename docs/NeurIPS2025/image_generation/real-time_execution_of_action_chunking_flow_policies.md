---
title: >-
  [Paper Note] Real-Time Execution of Action Chunking Flow Policies
description: >-
  [NeurIPS 2025][Image Generation][Real-time inference] This paper proposes Real-Time Chunking (RTC), which frames asynchronous action chunk execution as an inpainting problem. By freezing already-executed actions and "inp…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Real-time inference"
  - "action chunking"
  - "flow matching"
  - "inpainting guidance"
  - "VLA"
date: 2026-05-08
content_hash: a8afd63e7f370ece
---

# Real-Time Execution of Action Chunking Flow Policies

**Conference**: NeurIPS 2025
**arXiv**: [2506.07339](https://arxiv.org/abs/2506.07339)
**Code**: [Project Page](https://pi.website/research/real_time_chunking)
**Area**: Diffusion/Flow Models / Robot Control
**Keywords**: Real-time inference, action chunking, flow matching, inpainting guidance, VLA

## TL;DR

This paper proposes Real-Time Chunking (RTC), which frames asynchronous action chunk execution as an inpainting problem. By freezing already-executed actions and "inpainting" the remainder to be consistent with the prefix, RTC enables smooth real-time execution of diffusion/flow policies without any retraining.

## Background & Motivation

Modern VLAs (Vision-Language-Action models) are increasingly powerful for robot control, yet face a fundamental tension: **larger models perform better but incur higher latency**. For instance, a 3B-parameter $\pi_0$ VLA requires 46ms for KV cache prefilling alone, while control loops demand a 20ms frame interval. Remote inference further compounds latency with network overhead.

**Action chunking**—where the model outputs multiple steps at once—partially alleviates the latency issue but introduces new core challenges:

**Chunk boundary discontinuity**: Adjacent action chunks may jump to different modes of the distribution, producing abrupt, out-of-distribution motions.

**Synchronous inference pauses**: The default approach halts execution after each chunk and waits for the next one, introducing visible pauses that alter robot dynamics.

**Naive asynchronous strategies fail**: Simply switching to a new chunk produces extremely high accelerations; Temporal Ensembling averages multiple predictions, but the mean of a multimodal distribution may not correspond to any valid action.

The authors illustrate the problem clearly: if the current chunk plans to navigate above an obstacle while the new chunk plans to go below, switching after a 7-step delay produces violent, out-of-distribution accelerations.

## Method

### Overall Architecture

RTC frames real-time execution as an **inpainting problem**: when generating a new action chunk, the action prefix that will inevitably have been executed by the time inference completes is "frozen," and the remainder is "inpainted" to be consistent with that prefix. The algorithm runs an inference loop continuously in a background thread, while the main thread consumes one action every $\Delta t$.

### Key Designs

1. **Flow matching inpainting via $\Pi$GDM**: At each denoising step, a gradient-based guidance term is added to encourage the generated output to match the known target values (the frozen actions). The corrected velocity field is:
    $\mathbf{v}_{\Pi\text{GDM}}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau) = \mathbf{v}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau) + \min\left(\beta, \frac{1-\tau}{\tau \cdot r_\tau^2}\right)(\mathbf{Y} - \widehat{\mathbf{A}_t^1})^\top \text{diag}(\mathbf{W}) \frac{\partial \widehat{\mathbf{A}_t^1}}{\partial \mathbf{A}_t^\tau}$
   where $\widehat{\mathbf{A}_t^1} = \mathbf{A}_t^\tau + (1-\tau)\mathbf{v}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau)$ is an estimate of the final denoised output. The guidance weight clipping $\beta$ is the authors' contribution, preventing instability under few-step denoising.

2. **Soft Masking**: This is the key innovation ensuring cross-chunk continuity. Using only the first $d$ actions as a hard mask provides too weak an inpainting signal, making policy switching likely. Soft masking leverages all $H-s$ overlapping actions, with weights decaying exponentially from 1 to 0:
    $\mathbf{W}_i = \begin{cases} 1 & \text{if } i < d \\ c_i \frac{e^{c_i}-1}{e-1} & \text{if } d \leq i < H-s \\ 0 & \text{if } i \geq H-s \end{cases}$
   where $c_i = \frac{H-s-i}{H-s-d+1}$. Intuitively, actions further in the future should receive less attention.

3. **Asynchronous execution system**: Thread safety is ensured via mutex locks and condition variables:

    - `GetAction`: Called by the controller every $\Delta t$; returns the next action from the current chunk.
    - `InferenceLoop`: A background thread runs inference continuously, using a sliding window of past latencies to conservatively estimate the next delay.
    - New chunks are atomically swapped in as soon as they are ready, with execution stride $s = \max(d, s_{\min})$.

### Loss & Training

RTC is a pure inference-time algorithm that **requires no training or retraining**. It is applicable to any action chunking policy based on diffusion or flow matching. The guidance term is computed via backpropagation through a vector-Jacobian product, which constitutes the only additional computational overhead.

## Key Experimental Results

### Simulation: 12 Dynamic Tasks in Kinetix

| Method | d=0 Solve Rate | d=2 Solve Rate | d=4 Solve Rate | Robustness to Latency |
|--------|---------------|---------------|---------------|----------------------|
| Naive Async | ~48% | ~42% | ~33% | Poor |
| TE (Temporal Ensembling) | ~35% | ~33% | ~30% | Worst |
| BID (Bidirectional Decoding) | ~51% | ~46% | ~38% | Moderate |
| RTC (Hard Mask) | ~52% | ~48% | ~42% | Good |
| **RTC (Soft Mask)** | **~54%** | **~50%** | **~43%** | **Best** |

### Real-World: 6 Bimanual Manipulation Tasks ($\pi_{0.5}$ VLA)

| Method | Avg. Throughput (No Delay) ↑ | Avg. Throughput (+100ms) ↑ | Avg. Throughput (+200ms) ↑ |
|--------|------------------------------|---------------------------|---------------------------|
| Synchronous Inference | ~0.35 | ~0.28 | ~0.22 |
| TE (Sparse) | ~0.36 | N/A (triggered protective stop) | N/A |
| TE (Dense) | ~0.33 | N/A (triggered protective stop) | N/A |
| **RTC** | **~0.40** | **~0.40** | **~0.40** |

### Key Findings

- **RTC is fully robust to latency**: Injecting +200ms of additional delay produces no performance degradation, whereas synchronous inference degrades linearly and TE methods trigger robot protective stops due to excessive jitter.
- **Simultaneous gains in speed and quality**: RTC not only executes faster (achieving a 20% improvement over synchronous inference even after accounting for the removal of inference pauses), but also completes tasks earlier by reducing errors and retries.
- **Lighting match task**: In the task demanding the highest precision—where no retry is possible—RTC achieves substantially higher success rates.
- Soft masking outperforms hard masking at low latency; the gap narrows at high latency.

## Highlights & Insights

- **Elegant problem reformulation**: Recasting the asynchronous chunk-splicing problem in real-time control as an inpainting problem provides a theoretically grounded solution with established methods.
- **Pure inference-time approach**: No modification to the training pipeline is required; the method is applicable to all diffusion/flow policies, including large deployed models such as $\pi_{0.5}$.
- **Analogy for soft masking**: The exponentially decaying weights mirror the increasing discounting of future uncertainty, analogous to receding-horizon discounting in control theory.
- **Thoroughness of real-world validation**: 480 episodes, 28 hours of pure execution time, 6 task types including mobile manipulation, evaluated under varying injected latencies.

## Limitations & Future Work

- The RTC guidance term requires backpropagation to compute VJPs, increasing inference latency by approximately 28% (97ms vs. 76ms).
- The method is only applicable to diffusion and flow matching policies, not to autoregressive or VQ-based policies.
- Real-world experiments do not cover highly dynamic scenarios such as legged locomotion (tested only in simulation).
- The ablation over soft mask decay functions (exponential vs. linear vs. cosine) is discussed only in the appendix.

## Related Work & Insights

- **Diffuser** first applied diffusion inpainting for reinforcement learning constraints, but without guidance-based formulation and without consideration of real-time control.
- **BID** maintains chunk continuity via rejection sampling but requires 32 parallel batch samples, making its computational cost far greater than RTC.
- **Consistency policies and streaming diffusion policies** reduce denoising steps via distillation, but cannot eliminate the latency of a single forward pass.
- Hierarchical VLA designs (System 1/2 architectures) are orthogonal to RTC and may be combined with it.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Introduces inpainting guidance into real-time robot control with precise problem formulation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Simulation + real world, multiple latency conditions, 480 episodes.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem exposition is exceptionally clear; figures are intuitive; algorithm pseudocode is complete.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses a core deployment bottleneck for large robot models; plug-and-play.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](../../ICLR2026/image_generation/compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [\[NeurIPS 2025\] Failure Prediction at Runtime for Generative Robot Policies](failure_prediction_at_runtime_for_generative_robot_policies.md)
- [\[NeurIPS 2025\] Flow Matching Neural Processes](flow_matching_neural_processes.md)
- [\[ICCV 2025\] StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation](../../ICCV2025/image_generation/streamdiffusion_a_pipeline-level_solution_for_real-time_interactive_generation.md)
- [\[NeurIPS 2025\] Value Gradient Guidance for Flow Matching Alignment](value_gradient_guidance_for_flow_matching_alignment.md)

</div>

<!-- RELATED:END -->
