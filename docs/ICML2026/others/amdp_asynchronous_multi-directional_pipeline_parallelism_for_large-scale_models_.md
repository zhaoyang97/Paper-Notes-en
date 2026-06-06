---
title: >-
  [Paper Note] AMDP: Asynchronous Multi-Directional Pipeline Parallelism for Large-Scale Models Training
description: >-
  [ICML 2026][pipeline parallelism] AMDP utilizes multi-directional asynchronous pipelines, a one-step parameter mismatch upper bound, gradient accumulation…
tags:
  - "ICML 2026"
  - "pipeline parallelism"
  - "asynchronous training"
  - "parameter mismatch"
  - "gradient accumulation"
  - "ZeRO"
date: 2026-05-08
content_hash: 84b6ff26c5e1eccf
---

# AMDP: Asynchronous Multi-Directional Pipeline Parallelism for Large-Scale Models Training

**Conference**: ICML 2026  
**arXiv**: [2605.29664](https://arxiv.org/abs/2605.29664)  
**Code**: https://github.com/Vinsmoke86/AMDP  
**Area**: LLM Efficiency / Distributed Training  
**Keywords**: pipeline parallelism, asynchronous training, parameter mismatch, gradient accumulation, ZeRO  

## TL;DR
AMDP utilizes multi-directional asynchronous pipelines, a one-step parameter mismatch upper bound, gradient accumulation, and ZeRO state sharding to enhance the throughput of large-scale model pipeline parallel training while maintaining near-synchronous convergence. In 8-GPU GPT/BERT experiments, it achieves a maximum improvement of approximately 17% over the strongest asynchronous baselines.

## Background & Motivation
**Background**: Large-scale model training typically requires pipeline parallelism to partition network layers across multiple GPUs. Synchronous pipelines such as GPipe, DAPPLE, and Inter-1F1B exhibit stable convergence but suffer from pipeline bubbles due to forward-backward dependencies. Asynchronous pipelines like PipeDream improve utilization, but the inconsistency of parameter versions between forward and backward passes can harm convergence.

**Limitations of Prior Work**: Traditional asynchronous 1F1B continuously feeds minibatches into the pipeline to eliminate bubbles. As pipeline depth increases, early stages may undergo several parameter updates between the forward and backward passes of a specific minibatch, leading to stale gradients or parameter mismatch. While parameter caching ensures consistency, it introduces delayed gradients and memory overhead; parameter prediction methods rely on approximating future weights, making errors difficult to control.

**Key Challenge**: Training systems require both the high throughput of asynchronous pipelines and the stable convergence of synchronous training. The fundamental issue is not whether asynchronous training is viable, but how to structurally limit the parameter mismatch between forward and backward passes while filling the bubbles caused by restricted feeding rates.

**Goal**: AMDP aims to restrict the parameter mismatch of each stage to within a single step and use multiple pipelines with complementary directions to fill idle time, while controlling communication and memory costs through gradient accumulation and ZeRO.

**Key Insight**: The authors analyze the structural source of parameter mismatch: the more minibatches stage 0 reads before its first backward pass, the larger the maximum mismatch becomes. If stage 0 is forced to read a maximum of two minibatches, the mismatch for all stages will not exceed one step.

**Core Idea**: By using a "feed less to control mismatch, use multi-directional concurrency to supplement utilization" approach, the convergence risk of asynchronous training is compressed from depth-dependent to a constant level.

## Method
AMDP can be viewed as a rescheduling of asynchronous pipelines. Rather than pursuing full load for a single pipeline, it ensures each pipeline maintains a controlled mismatch and uses multiple pipelines with different directions to fill idle slots. In this way, system-level utilization remains high, while the optimization process observes a delay close to synchronous one-step optimization.

### Overall Architecture
The model is partitioned into $d$ pipeline stages. In standard asynchronous 1F1B, to eliminate bubbles, stage 0 is allowed to read nearly $d$ minibatches before the first backward pass, resulting in a parameter mismatch of approximately $d-1$ for stage 0. AMDP fixes the input of stage 0 to two minibatches; thus, the mismatch for any stage $i$ is $\min(n,d-i)-1$, which has an upper bound of 1 when $n=2$.

Restricting the number of inputs introduces idle time; therefore, AMDP launches multiple pipelines simultaneously. For a depth $d$, the active ratio of a single controlled pipeline is approximately $2/d$, so launching $d/2$ pipelines with complementary directions can saturate the devices. Different pipelines are mapped to GPUs in a Chimera-style fashion, but AMDP executes asynchronously and resolves operation conflicts on the same GPU using FIFO rules.

To reduce the frequency of all-reduce communication after each backward pass, AMDP does not update parameters immediately. Instead, gradients from multiple minibatches are accumulated until a threshold is reached for a unified reduce and update. Finally, AMDP employs ZeRO concepts so that the optimizer state for each stage is held by only one GPU, while other replicas send gradients and receive updated parameters, avoiding optimizer state duplication across multiple pipelines.

### Key Designs
1. **One-step parameter mismatch upper bound**:

	- **Function**: Controls the version gap between parameters used in forward and backward passes within the asynchronous pipeline.
	- **Mechanism**: The parameter mismatch satisfies $\mathrm{mismatch}(i)=\min(n,d-i)-1$; AMDP sets $n=2$ so that the mismatch for all stages does not exceed 1.
	- **Design Motivation**: In previous asynchronous methods, mismatch grows linearly with pipeline depth, leading to convergence instability in deep models; constant mismatch limits asynchronous perturbations to second-order terms.

2. **Multi-directional concurrent pipeline scheduling**:

	- **Function**: Compensates for the leading and trailing bubbles generated after reducing minibatch inputs.
	- **Mechanism**: Approximately $d/2$ pipelines with different directions are initiated for depth $d$. Even/odd pipelines use different GPU mapping directions to allow idle periods to overlap; conflicts are handled via FIFO delays.
	- **Design Motivation**: Relying on a single controlled pipeline provides stable convergence but low utilization; multi-directional concurrency decouples "mismatch control" from "high throughput."

3. **Gradient accumulation and ZeRO state sharding**:

	- **Function**: Reduces communication frequency and lowers the optimizer state memory overhead caused by multiple pipeline replicas.
	- **Mechanism**: Gradients are updated collectively during bubbles after reaching a threshold, ensuring only the first $d$ minibatches in each accumulation window experience a one-step mismatch; ZeRO ensures stage $i$ optimizers reside only on GPU $i$, with other replicas synchronizing via reduce and broadcast.
	- **Design Motivation**: Multiple pipelines increase pressure on parameter and optimizer states; without accumulation and sharding, system gains would be offset by communication and memory costs.

### Loss & Training
AMDP does not change the model training objective, only the semantics of pipeline execution and updates. The theoretical section proves that under the assumptions of $L$-smooth non-convex objectives and unbiased stochastic gradients with bounded variance, the average gradient norm upper bound with one-step mismatch only introduces $O(\eta^2)$ perturbation compared to synchronous SGD. Experiments utilize AdamW, mixed precision, and a microbatch size of 4, comparing throughput, memory, and convergence on GPT-style and BERT-style models.

## Key Experimental Results

### Main Results
The experimental hardware consists of 8 NVIDIA A800 80GB GPUs with NVLink 3.0 interconnects. Models include a ~1.56B parameter GPT-style model and a ~1.04B parameter BERT-style model. The table below excerpts the 8-GPU throughput results in ktokens/s.

| Model | $d$ | $b$ | PipeDream-2BW | XPipe | Inter-1F1B | AMDP | Gain vs. Best Baseline |
|------|-----|-----|---------------|-------|------------|------|--------------|
| GPT-style | 4 | 16 | 38.6 | 38.5 | 35.4 | 39.1 | +1.3% |
| GPT-style | 4 | 64 | 41.0 | 40.7 | 39.8 | 42.1 | +2.7% |
| GPT-style | 8 | 32 | 70.3 | 66.0 | 57.0 | 75.5 | +7.4% |
| GPT-style | 8 | 128 | 71.6 | 69.7 | 67.5 | 83.7 | +16.9% |
| BERT-style | 8 | 32 | 74.3 | 73.6 | 37.5 | 78.5 | +5.7% |
| BERT-style | 8 | 128 | 75.8 | 75.6 | 58.8 | 86.1 | +13.6% |

### Ablation Study
The authors further investigated the effects of gradient accumulation thresholds and ZeRO, reporting training quality metrics.

| Configuration | Key Metric | Description |
|------|---------|------|
| GPT, AMDP, $d=8,b=128$ | 40k iter train loss 2.90 | Close to Inter-1F1B's 2.88 |
| BERT, AMDP, $d=8,b=128$ | 40k iter train loss 2.36 | Comparable to DAPPLE |
| GPT, reaching loss 2.9 | 23% faster than Inter-1F1B | High throughput without significant convergence sacrifice |
| BERT, reaching loss 2.4 | 22% faster than DAPPLE | Significant wall-clock convergence advantage |
| Accumulation Threshold 1/2/4/8, GPT | 75.5 / 78.9 / 83.7 / 83.3 | Moderate threshold is optimal |
| Accumulation Threshold 1/2/4/8, BERT | 78.5 / 81.0 / 86.1 / 84.6 | Diminishing returns with larger thresholds |
| w/o ZeRO, GPT/BERT | 80.3 / 82.7 | Throughput ~4% lower |
| with ZeRO, GPT/BERT | 83.7 / 86.1 | Reduces redundant optimizer state and boosts throughput |

### Key Findings
- AMDP's throughput advantage becomes more pronounced as pipeline depth and update batch size increase, as deep pipelines and large accumulation windows more easily expose stage imbalance and bubbles in baselines.
- Convergence curves are close to synchronous methods, indicating that the one-step mismatch upper bound is more reliable than "complete asynchrony with post-hoc remediation."
- AMDP's peak memory is slightly higher than XPipe and PipeDream-2BW but lower than the high activation peaks of Inter-1F1B, with a more balanced memory distribution.
- In 16-GPU two-node experiments, AMDP maintains the highest throughput across pure pipeline and hybrid pipeline+data parallel configurations, reaching 159.8 ktokens/s for $d=8,b=128$.

## Highlights & Insights
- The strongest aspect of the paper is the derivation of a structural formula for mismatch, around which the system is designed. It is not an empirical schedule but addresses the root cause: "how many minibatches are read before the first backward pass."
- Multi-directional scheduling transfers the intuition of Chimera to asynchronous training, shifting the goal from "filling bubbles synchronously" to "filling bubbles under controlled mismatch," which better fits asynchronous stability.
- ZeRO is not an optional optimization here but a necessary condition for making multiple pipeline replicas scalable. Otherwise, the throughput gains from multiple directions would be consumed by optimizer state duplication.

## Limitations & Future Work
- The effectiveness of AMDP depends on pipeline stage partitioning and the ratio of forward to backward pass times; more validation is needed for extreme imbalance, strong communication bottlenecks, or non-Transformer architectures.
- Theoretical analysis is based on smooth objectives and SGD-style assumptions. While the appendix extends this to Adam-like optimizers, it remains an approximate explanation.
- The implementation complexity of multi-directional scheduling is higher than standard 1F1B, and costs related to integration with existing frameworks, debugging, and fault tolerance must be considered.
- Future work could combine mismatch-aware learning rates, automatic stage partitioning, and dynamic selection of pipeline counts to make scheduling more adaptive.

## Related Work & Insights
- **vs DAPPLE / Inter-1F1B**: Synchronous methods provide stable convergence but significant bubbles; AMDP exchanges higher throughput via controlled asynchrony and multi-directional scheduling.
- **vs PipeDream / PipeDream-2BW**: The PipeDream series eliminates bubbles but requires handling delayed gradients or parameter versions; AMDP directly limits mismatch to one step, reducing convergence risk at the source.
- **vs XPipe / vNAG**: Parameter prediction methods attempt to estimate future weights; AMDP does not predict parameters but reduces the degree of inconsistency through structural scheduling.
- **vs Chimera**: Chimera is a synchronous bidirectional pipeline; AMDP adopts the multi-directional idea for asynchronous scenarios and incorporates gradient accumulation and ZeRO.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ The combination of multi-directional asynchrony and a one-step mismatch upper bound has significant system design value and builds upon mature pipeline parallelism concepts.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers multiple models, depths, batch sizes, memory, convergence, and 16-GPU scalability, though validation on true ultra-large-scale models remains to be seen.
- **Writing Quality**: ⭐⭐⭐⭐☆ Problem decomposition is clear, with theory, scheduling diagrams, and system experiments providing mutual support.
- **Value**: ⭐⭐⭐⭐☆ Highly practical for large-scale model training systems, especially in scenarios where high throughput is desired without risking asynchronous convergence failure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Torus Graphs for Large-Scale Neural Phase Analysis](torus_graphs_for_large_scale_neural_phase_analysis.md)
- [\[ICML 2026\] HASTE: Hardware-Aware Dynamic Sparse Training for Large Output Spaces](haste_hardware-aware_dynamic_sparse_training_for_large_output_spaces.md)
- [\[CVPR 2026\] Next-Scale Autoregressive Models for Text-to-Motion Generation](../../CVPR2026/others/next-scale_autoregressive_models_for_text-to-motion_generation.md)
- [\[ICCV 2025\] Kaputt: A Large-Scale Dataset for Visual Defect Detection](../../ICCV2025/others/kaputt_a_large-scale_dataset_for_visual_defect_detection.md)
- [\[CVPR 2026\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](../../CVPR2026/others/sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)

</div>

<!-- RELATED:END -->
