---
title: >-
  [Paper Note] CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach
description: >-
  [CVPR 2026][LLM Efficiency][Continual Learning] Proposes the CHEEM framework that leverages hierarchical exploration-exploitation (HEE) NAS to automatically learn task-aware dynamic ViT backbones—selecting Reuse/New/Adap…
tags:
  - "CVPR 2026"
  - "LLM Efficiency"
  - "Continual Learning"
  - "Exemplar-Free Class-Incremental Learning"
  - "Neural Architecture Search"
  - "Layer Operations"
  - "Vision Transformer"
date: 2026-05-08
content_hash: 6d8d48d19b044f75
---

# CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach

**Conference**: CVPR 2026  
**arXiv**: [2303.08250](https://arxiv.org/abs/2303.08250)  
**Code**: [GitHub](https://github.com/savadikarc/cheem)  
**Area**: LLM Efficiency  
**Keywords**: Continual Learning, Exemplar-Free Class-Incremental Learning, Neural Architecture Search, Layer Operations, Vision Transformer

## TL;DR

Proposes the CHEEM framework that leverages hierarchical exploration-exploitation (HEE) NAS to automatically learn task-aware dynamic ViT backbones—selecting Reuse/New/Adapt/Skip operations at each layer—significantly outperforming prompt-based methods on MTIL and VDD continual learning benchmarks, approaching the full fine-tuning upper bound.

## Background & Motivation

**Background**: ViT-based continual learning has made progress, primarily divided into prompt-based methods (frozen backbone + learned prompts) and parameter-adjustment methods (adapters/masks).

**Limitations of Prior Work**: (a) Prompt-based methods maintain stability but lack plasticity—frozen backbones cannot adapt to tasks significantly different from pretraining; (b) Parameter-adjustment methods apply uniform operations per layer, ignoring variations in task difficulty/similarity—simple tasks do not need all layers, while difficult tasks may need entirely new layers.

**Key Challenge**: Existing methods fix or incrementally increase computation regardless of task difficulty—they lack Skip (simplifying the model for easy tasks) and New (introducing entirely new layers for highly dissimilar tasks) operations.

**Goal**: Learn task-aware dynamic model architectures that automatically allocate Reuse/Adapt/New/Skip operations across layers.

**Key Insight**: Formulate continual learning as a continual memory learning problem, composed of internal parametric memory (NAS-driven dynamic backbone) and external centroid memory (task ID inference).

**Core Idea**: Hierarchical exploration-exploitation (HEE) NAS sampling that automatically selects layer operations based on task similarity.

## Method

### Overall Architecture

Starting from a pretrained ViT, MLPDown layers and MHSA projection layers are selected as plastic components. For each new task: (1) construct a supernet (search space); (2) HEE-NAS trains the supernet; (3) compute-aware evolutionary search selects the target network; (4) retrain the target network.

### Key Designs

1. **Four Layer Operations**:

    - **Reuse**: Directly reuse layer parameters from a previous task for knowledge transfer
    - **Adapt**: Add LoRA low-rank adaptation on top of previous parameters, $\theta_l^{(3,)} = \theta_l^{(2,)} + B_l \cdot A_l$
    - **New**: Introduce entirely new layers (randomly initialized) for cases highly dissimilar to previous tasks
    - **Skip**: Skip the entire FFN/MHSA block to reduce computation for simple tasks
   
   Design Motivation: Reuse (knowledge transfer) and Adapt (fine-tuning) are standard options in existing methods, but New (handling large domain gaps) and Skip (handling simple tasks) are overlooked key operations. For example, transitioning from ImageNet to MNIST allows skipping many layers.

2. **Hierarchical Exploration-Exploitation (HEE) Sampling**:

    - **Exploitation**: Sampling based on task similarity—computing normalized cosine similarity $S_l^{i,t} = \text{NormCosine}(\mu_l^i, \mu_l^{i \to t})$ between old experts and the new task to form a categorical distribution
    - Two-level hierarchical sampling: Level 1 samples between old tasks and Aux (joint New/Skip); Level 2 uses Bernoulli to decide Reuse (probability $\rho_i$) vs. Adapt (probability $1-\rho_i$)
    - **Exploration**: Uniform random sampling, used with probability $\epsilon_1=0.3$
    - During supernet training, exploitation is favored ($\epsilon_1=0.3$); during evolutionary search, exploration is favored ($\epsilon_2=0.5$)
   
   Design Motivation: Pure uniform sampling ignores task similarity information, while pure exploitation gets trapped in local optima. HEE balances both through epoch-wise scheduling.

3. **Compute-Aware Target Network Selection**: A predefined performance tolerance threshold $\tau=2\%$ is set; within each candidate group, candidates are sorted by compute in ascending order, with crossover and mutation evolutionary search. After selection, learnable parameters are retrained from scratch.
   Design Motivation: Pursuing only the highest accuracy ignores computational efficiency; group-wise sorting ensures selecting the lightest architecture at comparable performance.

### Evaluation Metric

The paper proposes Figure of Merit (FoM):
$$\text{FoM}(m,n) = \frac{A\mathbb{A}^{UB} - A\mathbb{A}^n}{A\mathbb{A}^{UB} - A\mathbb{A}^m} \cdot \frac{\text{FLOPs}^n}{\text{FLOPs}^m}$$
This jointly considers the accuracy gap and computational cost; FoM > 1 indicates method m is superior to n.

## Key Experimental Results

### Main Results — CHEEM vs. Upper Bound

| Method | MTIL (ViT-B) | MTIL (DEiT-T) | VDD (ViT-B) | VDD (DEiT-T) |
|--------|-------------|---------------|-------------|---------------|
| Upper Bound (Full-FT) | 88.1 | 75.3 | 88.7 | 76.21 |
| CHEEM | 85.9 | 74.5 | 86.7 | 76.18 |
| Gain | -2.2 | -0.8 | -2.0 | **-0.03** |

### Ablation Study — FoM Comparison

| Method | MTIL ViT-B | MTIL DEiT-T | VDD ViT-B | VDD DEiT-T |
|--------|-----------|-------------|-----------|------------|
| CODA-P | 24.2 | 84.6 | 35.9 | 2811.3 |
| S-Prompts | 3.2 | 10.5 | 5.5 | 338.8 |
| LoRA-CL | 1.7 | 5.7 | 1.5 | 73.6 |
| Tuna | 50.6 | 242.4 | 58.6 | 4538.8 |

All FoM > 1 indicates CHEEM is superior to all baselines on the combined accuracy-compute metric.

### Key Findings

1. **Task-aware architectures are semantically reasonable**: Caltech101 (simple task) skips 5 MLP blocks; FGVC Aircraft (fine-grained difficult task) uses a combination of New + Adapt + Skip.
2. **Near-lossless on DEiT-Tiny VDD**: 76.18 vs. 76.21 (Full-FT), a gap of only 0.03%, showing CHEEM can fully exploit capacity even on small models.
3. **LoRA-CL (a special case of CHEEM with all layers set to Adapt)** approaches CHEEM on ViT-Base (FoM 1.7), but the gap is large on DEiT-Tiny (FoM 73.6), indicating small models require New/Skip operations.
4. Prompt-based methods nearly fail on DEiT-Tiny (CODA-P achieves only 5.6% Acc), as they are limited by backbone capacity.

## Highlights & Insights

- The first complete implementation of Reuse/New/Adapt/Skip four operations in continual learning, forming a full spectrum from "learning to reuse" to "learning to grow" to "learning to prune."
- The learned task architectures are highly interpretable—simple tasks are lightweight, difficult tasks are complex, consistent with human intuition.
- HEE sampling cleverly uses CLS token similarity as a proxy for task similarity.
- The FoM metric provides a comprehensive evaluation tool for the continual learning community.

## Limitations & Future Work

- NAS search adds training time; while the per-task overhead is manageable, multi-task accumulation is non-negligible.
- The external task centroid memory (for task ID inference) follows prior work; joint optimization with internal memory is not explored in depth.
- Validated only on classification tasks; not extended to more complex vision tasks such as detection/segmentation.
- The New operation introduces entirely new parameters, which may lead to excessive parameter growth when the number of tasks is very large.

## Related Work & Insights

- Conceptually similar to L2G (DARTS-driven ConvNet continual learning), but CHEEM adds Skip operations and HEE sampling, designed specifically for ViTs.
- Provides a viable solution for the open problem of "adaptive compute allocation in continual learning."
- HEE sampling can be generalized to other NAS scenarios requiring exploration-exploitation balance.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of four operations + HEE sampling is novel, though individual technical components (NAS/LoRA/Skip) are existing
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two benchmarks × two backbones × multiple baselines + interpretable architecture analysis
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagrams, but the paper is lengthy
- Value: ⭐⭐⭐⭐ Significantly outperforms prompt-based methods on challenging benchmarks, approaching full fine-tuning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning](../../ICLR2026/llm_efficiency/one-prompt_strikes_back_sparse_mixture_of_experts_for_prompt-based_continual_lea.md)
- [\[ICLR 2026\] Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](../../ICLR2026/llm_efficiency/understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)
- [\[ICLR 2026\] Expert Divergence Learning for MoE-based Language Models](../../ICLR2026/llm_efficiency/expert_divergence_learning_for_moe-based_language_models.md)
- [\[AAAI 2026\] Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning](../../AAAI2026/llm_efficiency/resource_efficient_sleep_staging_via_multi-level_masking_and_prompt_learning.md)
- [\[ICLR 2026\] Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective](../../ICLR2026/llm_efficiency/randomization_boosts_kv_caching_learning_balances_query_load_a_joint_perspective.md)

</div>

<!-- RELATED:END -->
