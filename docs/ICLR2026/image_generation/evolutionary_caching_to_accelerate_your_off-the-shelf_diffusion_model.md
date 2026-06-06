---
title: >-
  [Paper Note] Evolutionary Caching to Accelerate Your Off-the-Shelf Diffusion Model
description: >-
  [ICLR 2026][Image Generation][Diffusion model acceleration] This paper proposes ECAD (Evolutionary Caching to Accelerate Diffusion models), which employs a genetic algorithm to automatically search for optimal caching sc…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Diffusion model acceleration"
  - "caching schedule"
  - "genetic algorithm"
  - "Pareto optimization"
  - "training-free"
date: 2026-05-08
content_hash: 688cd41e8dfb74e7
---

# Evolutionary Caching to Accelerate Your Off-the-Shelf Diffusion Model

**Conference**: ICLR 2026
**arXiv**: [2506.15682](https://arxiv.org/abs/2506.15682)  
**Code**: Available ([Project Page](https://research.aniaggarwal.com/ecad))  
**Area**: Image Generation
**Keywords**: Diffusion model acceleration, caching schedule, genetic algorithm, Pareto optimization, training-free

## TL;DR

This paper proposes ECAD (Evolutionary Caching to Accelerate Diffusion models), which employs a genetic algorithm to automatically search for optimal caching schedules along the speed–quality Pareto frontier. Without modifying model parameters and using only 100 calibration prompts, ECAD achieves 2–3× inference speedup while maintaining or even improving generation quality.

## Background & Motivation

Diffusion models dominate image generation, yet their inference requires 20–50 iterative denoising steps, incurring substantial computational cost. Existing acceleration methods fall into two categories:

**Training-based methods** (distillation, pruning, etc.): require significant training cost and may degrade quality.

**Training-free caching methods**: reuse intermediate features to reduce computation, but rely heavily on hand-crafted heuristics.

Core limitations of existing caching methods:
- **FORA**: provides only discrete speedup levels (e.g., 2×, 3×), lacking intermediate flexibility.
- **ToCa**: requires manual hyperparameter tuning per model; parameters optimized for PixArt-α do not transfer to PixArt-Σ.
- **TaylorSeer**: incurs large memory overhead, reducing batch size by 66%.
- All methods depend on human-designed heuristics and extensive hyperparameter tuning.

## Method

### Overall Architecture

ECAD reformulates diffusion model caching as a **multi-objective Pareto optimization problem**:

$$\min_S (C(S), Q(S))$$

where $C(S)$ denotes computational cost (MACs) and $Q(S)$ denotes generation quality (Image Reward). The caching schedule is represented as a binary tensor $S \in \{0,1\}^{N \times B \times C}$, where $N$ is the number of diffusion steps, $B$ is the number of transformer blocks, and $C$ is the number of cacheable components.

The framework comprises four customizable components:
1. **Binary caching tensor**: defines the granularity of the search space.
2. **Calibration prompts**: 100 prompts from the Image Reward Benchmark.
3. **Quality metrics**: Image Reward (quality) + MACs (speed).
4. **Initial population**: randomly initialized or seeded from prior schedules such as FORA or TGATE.

### Key Designs

#### 1. Component-Level Caching

Selective caching is applied to functional components within each transformer block of the DiT:

- **PixArt-α/Σ** (28 blocks): self-attention $f_{\text{SA}}$, cross-attention $f_{\text{CA}}$, feed-forward network $f_{\text{FFN}}$
- **FLUX.1-dev** (19 full + 38 single blocks): attention, feed-forward network, MLP, etc.

$$f_{\text{comp}}^b(z'_t, t, c) = \begin{cases} \text{compute}(z'_t, c, t) & \text{recompute} \\ \text{cache}[f_{\text{comp}}^b, t+1] & \text{use cache} \end{cases}$$

#### 2. NSGA-II Genetic Algorithm

ECAD adopts NSGA-II for multi-objective optimization. Core operations are as follows:

| Operation | Implementation |
|-----------|----------------|
| Selection | Tournament selection + non-dominated sorting |
| Crossover | 4-point crossover: recombines two schedules |
| Mutation | Bit-flip mutation: randomly toggles cache/recompute decisions |
| Fitness | Bi-objective: Image Reward↑ + MACs↓ |

**Algorithm procedure**:
1. Initialize population (random + heuristic schedules).
2. Per generation: generate images for each schedule → compute Image Reward and MACs.
3. NSGA-II selection → crossover → mutation → next generation.
4. Aggregate Pareto frontier across all generations.

#### 3. Gradient-Free, Weight-Unmodified Optimization

- **No gradient computation**: zero memory overhead; runs on a single low-end GPU.
- **No model weight modification**: original model parameters remain fully intact.
- **Asynchronous execution**: evaluation of candidate schedules can be parallelized.
- **No batch size constraints**: unlike distillation-based methods, imposes no high VRAM requirements.

### Loss & Training

ECAD involves no training loss. The optimization objective is Pareto frontier discovery:

- **Quality objective**: Image Reward (100 prompts × 10 seeds).
- **Speed objective**: MACs (hardware-agnostic).
- **PixArt-α**: 550 generations × 72 candidates/generation × 1,000 images/candidate.
- **FLUX.1-dev**: 250 generations × 24 candidates/generation.

## Key Experimental Results

### Main Results

**Table 1: PixArt-α 256×256 Main Results**

| Method | Speedup | Image Reward↑ | COCO FID↓ | MJHQ FID↓ |
|--------|---------|--------------|-----------|-----------|
| No cache | 1.00× | 0.97 | 24.84 | 9.75 |
| FORA (N=3) | 2.01× | 0.83 | 24.50 | 11.11 |
| ToCa (N=3, R=90%) | 2.35× | 0.68 | 24.01 | 11.80 |
| **ECAD fast** | **1.97×** | **0.99** | **20.58** | **8.02** |
| **ECAD fastest** | **2.58×** | 0.77 | **19.54** | **8.67** |

At 2.58× speedup, ECAD "fastest" achieves a COCO FID of 19.54, which is 4.47 points lower than ToCa at 2.35× speedup (24.01).

**Table 1: FLUX.1-dev 256×256 Main Results**

| Method | Speedup | Image Reward↑ | COCO FID↓ |
|--------|---------|--------------|-----------|
| No cache | 1.00× | 1.04 | 25.76 |
| FORA (N=3) | 2.44× | 0.93 | 23.51 |
| TaylorSeer (N=5, O=2) | 2.55× | 0.54 | 29.66 |
| **ECAD fast** | **2.58×** | **1.04** | **21.61** |
| **ECAD fastest** | **3.37×** | 0.89 | 26.66 |

### Ablation Study

**Genetic Scalability (Table 2)**:

| Generations | Speedup | Image Reward↑ | MJHQ FID↓ |
|-------------|---------|--------------|-----------|
| 1 | 1.14× | 1.00 | 9.40 |
| 50 | 1.79× | 0.98 | 7.97 |
| 150 | 1.90× | 1.00 | 8.11 |
| 500 | 2.17× | 0.96 | 8.49 |

Only 50 generations are sufficient to surpass the no-acceleration baseline, with steady improvement as optimization continues.

**Acceleration Strategy Ablation**:
- Reducing population size (72→24): equivalent effect to reducing the number of generations.
- Reducing images per prompt (10→3): marginal impact.
- Reducing number of prompts (100→33): significantly degrades quality.

### Key Findings

1. **Pareto frontier paradigm**: provides continuously adjustable speed–quality trade-offs rather than discrete operating points.
2. **Cross-model transfer**: schedules optimized for PixArt-α transfer to PixArt-Σ; with only 50 generations of fine-tuning, performance surpasses optimization from scratch.
3. **Cross-resolution transfer**: schedules optimized at 256×256 remain competitive when directly applied at 1024×1024.
4. **Quality beyond baseline**: ECAD "fast" achieves 2× speedup while FID improves over the no-cache baseline.

## Highlights & Insights

1. **Paradigm shift**: transitioning from "hand-crafted heuristics" to "automated search for optimal caching schedules" fundamentally changes the methodology of diffusion caching.
2. **Minimal resource requirements**: 100 text prompts + single GPU + gradient-free computation = deployable under severely constrained conditions.
3. **Framework generality**: both the search space (caching tensor shape) and fitness functions (quality/speed metrics) are fully customizable.
4. **Counterintuitive finding**: FID improves after caching-based acceleration — suggesting that certain recomputed steps introduce "noise," and skipping them is actually beneficial.
5. **Extensibility to video**: the framework is modality-agnostic and naturally extends to text-to-video generation.

## Limitations & Future Work

1. Optimization relies on automatic metrics (Image Reward); substituting human evaluation may yield different outcomes.
2. The computational overhead of the genetic algorithm (550 generations × 72 candidates × 1,000 images) remains non-trivial.
3. Integration with training-based methods (e.g., distillation) has not been explored.
4. Validation is limited to DiT architectures; U-Net architectures have not been tested.
5. Domain bias in calibration prompts may affect performance in specific application scenarios.

## Related Work & Insights

- **FORA**: the first caching method for DiTs; ECAD can use its schedule to initialize the population.
- **ToCa**: fine-grained caching requiring manual tuning; ECAD automates this process.
- **DiCache**: allows the diffusion model to determine the caching strategy, but still relies on heuristics.
- **TaylorSeer**: predicts features via Taylor expansion, but incurs large memory overhead.
- Insight: the use of genetic algorithms for neural architecture search proves equally effective in the domain of inference acceleration.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Reformulating caching as Pareto optimization represents a paradigm-level innovation.
- Technical Contribution: ⭐⭐⭐⭐ — The method is elegant and effective, though the core technique (NSGA-II) is not novel in itself.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three models × multiple datasets × multiple metrics × transfer experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear exposition supported by extensive tables and figures.
- Overall Recommendation: ⭐⭐⭐⭐⭐ — A highly practical method that changes the practice of diffusion model acceleration.

## Background & Motivation

## Core Problem

## Method

## Key Experimental Results

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Inspiration & Connections

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning](image_can_bring_your_memory_back_a_novel_multi-modal_guided_attack_against_image.md)
- [\[ICLR 2026\] Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control](follow-your-shape_shape-aware_image_editing_via_trajectory-guided_region_control.md)
- [\[ICML 2026\] Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces](../../ICML2026/image_generation/discrete_diffusion_samplers_and_bridges_off-policy_algorithms_and_applications_i.md)
- [\[ICML 2026\] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](../../ICML2026/image_generation/evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)
- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)

</div>

<!-- RELATED:END -->
