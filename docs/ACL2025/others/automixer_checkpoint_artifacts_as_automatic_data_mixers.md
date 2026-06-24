---
title: >-
  [Paper Note] AutoMixer: Checkpoint Artifacts as Automatic Data Mixers
description: >-
  [ACL 2025][Data Mixing] Proposed the AutoMixer framework, which leverages checkpoint models saved during training as "data mixers" to regroup and reweight training data by aggregating first-order influence function approximations across multiple checkpoints, achieving a performance improvement of up to 1.93% across eight reasoning benchmarks.
tags:
  - "ACL 2025"
  - "Data Mixing"
  - "Checkpoint Utilization"
  - "Influence Functions"
  - "Pre-training"
  - "Reasoning Benchmarks"
date: 2026-05-08
content_hash: fac550dd63d32750
---

# AutoMixer: Checkpoint Artifacts as Automatic Data Mixers

**Conference**: ACL 2025  
**arXiv**: [2506.21910](https://arxiv.org/abs/2506.21910)  
**Code**: None  
**Area**: Others  
**Keywords**: Data Mixing, Checkpoint Utilization, Influence Functions, Pre-training, Reasoning Benchmarks

## TL;DR

Proposed the AutoMixer framework, which leverages checkpoint models saved during training as "data mixers" to regroup and reweight training data by aggregating first-order influence function approximations across multiple checkpoints, achieving a performance improvement of up to 1.93% across eight reasoning benchmarks.

## Background & Motivation

- **Background**: The efficacy of language model pre-training heavily depends on the composition of training data. Determining the optimal data mixing ratio for specific target skills remains a core challenge in pre-training. Existing methods typically determine data ratios through manually set domain weights or heuristic rules.
- **Limitations of Prior Work**: Direct modeling of the relationship between data and tasks is highly challenging, leading to a "chicken-and-egg" dilemma: to identify which data facilitates the acquisition of a specific skill, the model must first exhibit that skill; yet to exhibit the skill, it requires appropriate training data. Brute-force searching through all possible data combinations is computationally intractable.
- **Key Challenge**: While influence functions can estimate the first-order contribution of training samples to model performance, computing influence scores based solely on a single checkpoint (usually the final model) ignores the non-monotonic emergence of skills during training—a skill in which a specific intermediate checkpoint excels may fade in subsequent training.
- **Goal**: To design a framework that automatically identifies task-relevant training data and determines optimal sampling weights, fully utilizing intermediate checkpoints generated during training as an undervalued resource.
- **Key Insight**: It is observed that different checkpoints exhibit varying peak capabilities across different tasks (as shown in Table 1, where a 25M-parameter model reaches peak performance for different tasks at different step counts), allowing these checkpoints to serve as task-specific data samplers.
- **Core Idea**: Leverage a simulated training run of a proxy model to select the best-performing checkpoints for each task, compute the influence scores of training samples on each checkpoint, and aggregate these scores to guide data regrouping and sampling weight allocation.

## Method

### Overall Architecture

AutoMixer optimizes pre-training data through a two-step process: (1) **Data Regrouping**—obtaining task-optimal checkpoints via a simulated training run, computing sample influence scores, and reallocating raw data into task-aligned data groups; (2) **Datamix Reweighting**—determining sampling probabilities based on the aggregated influence density of each group to guide the loading proportions of each data group during pre-training.

### Key Designs

1. **Checkpoint-Based Task Identification**: During the simulated training of a proxy model (75M or 350M parameters), checkpoints are saved at set intervals, and the performance of all checkpoints is evaluated on target benchmarks. The best-performing checkpoint for each task is selected as the "sampler" for that task. The key insight is that optimal checkpoints for different tasks emerge at different stages of training (e.g., HellaSwag typically peaks late, whereas other tasks converge early), demonstrating that skill acquisition is non-monotonic.
2. **Multi-Checkpoint Influence Aggregation**: Influence scores of training samples are calculated individually for the $k$ selected checkpoints. The DataInf method is adopted to efficiently approximate the inverse Hessian, computing gradients only at the embedding and final layers to reduce computational overhead and enhance discriminability. Influence scores are aggregated using weighted "blending factors." The blending factor $\alpha_j$ is normalized by checkpoint steps, giving higher weights to tasks that optimize at later stages (slower-to-learn skills). The joint influence score is defined as $\mathcal{I}_{\text{joint}}(x_i) = \sum_{j=1}^{k} \alpha_j \cdot \mathcal{I}(x_i; \theta_j)$.
3. **Influence Density-Driven Sampling Weights**: For each data group, the influence density is calculated as $\rho_g = \frac{1}{T_g} \sum_{x_i \in g} \mathcal{I}_{\text{joint}}(x_i) \cdot s_i$, where $T_g$ is the total token count within the group, and $s_i$ is the token count of sample $h_i$. The sampling weight is defined as $w_g = \rho_g / \sum_{g'} \rho_{g'}$, ensuring that data groups with high influence densities are sampled more heavily during pre-training.

### Loss & Training

- **Pre-training Objective**: Standard causal language modeling (CLM) using cross-entropy loss.
- **Influence Function Approximation**: The DataInf method is utilized to bypass explicit Hessian inversion using a layer-wise regularization parameter $\lambda_l = 0.1 \times (n \cdot d_l)^{-1} \sum_{i=1}^{n} \|\nabla_{\theta_l} \ell\|_2^2$.
- **Discriminative Layer Selection**: Influence scores are computed only at the embedding and output layers, avoiding cancellation effects in intermediate layers while improving computational efficiency and score distinctiveness.
- **Training Configuration**: A decoder-only model based on the Llama-3 architecture is trained on 32 GPUs (4 nodes × 8 H100s). The simulated run uses 6.4B tokens (100K steps × batch size 8 × 4 nodes × seq length 2048).

## Key Experimental Results

### Main Results

Trained on the FineWeb-Edu dataset, zero-shot evaluation is conducted on eight common-sense reasoning benchmarks. Results denote the percentage accuracy improvement relative to the uniform sampling baseline.

| Method | ARC-E | ARC-H | BoolQ | PIQA | SIQA | HellaSwag | OBQA | WinoGrande | Average |
|------|-------|-------|-------|------|------|-----------|------|-----------|------|
| **350M Parameters** | | | | | | | | | |
| PPL Sampling | +0.35 | +0.60 | +0.44 | +0.70 | -0.10 | +0.55 | +0.40 | +0.90 | +0.66 |
| N-gram Sampling | +0.74 | +1.22 | +0.79 | +1.03 | +1.09 | +0.62 | +1.16 | +0.85 | +0.60 |
| AutoMixer-75M | -0.15 | +0.12 | -0.14 | +0.01 | -0.10 | +0.05 | -0.03 | -0.05 | -0.04 |
| **AutoMixer-350M** | **+2.23** | **+0.55** | **+2.16** | **+2.05** | **+2.12** | **+2.33** | **+2.01** | **+2.14** | **+1.93** |
| **1.5B Parameters** | | | | | | | | | |
| PPL Sampling | +0.20 | +0.52 | +0.32 | +0.40 | +0.07 | +0.18 | +0.75 | +0.68 | +0.48 |
| N-gram Sampling | +0.88 | +0.82 | +1.02 | +0.58 | +0.45 | +1.22 | +0.54 | +0.90 | +0.79 |
| **AutoMixer-350M** | **+1.26** | +0.39 | **+1.35** | **+1.22** | **+1.38** | **+1.45** | **+1.33** | **+1.41** | **+1.22** |

### Ablation Study

| Configuration | Average Gain (%) | Description |
|------|-----------|------|
| Final checkpoint only | +0.7 | Insufficient information from a single checkpoint |
| All 10 checkpoints | +0.8 | Indiscriminate aggregation yields limited effects |
| AutoMixer-350M (Selective checkpoints) | +1.22 | Task-aligned checkpoint selection is most effective |
| AutoMixer-75M Proxy | -0.04~-0.01 | An overly small proxy model fails to provide effective signals |
| AutoMixer-350M Proxy | +1.05~+1.93 | Proxy sizes matching the target model's scale are most effective |

### Key Findings

- **Proxy Model Scale is Crucial**: The 75M proxy model provides virtually no effective data selection signals (negative average gain), whereas the 350M proxy model leads to significant performance gains. Aligning the proxy model scale with the target model scale is a critical factor for performance.
- **Selective Checkpoints Outperform Full Aggregation**: Utilizing only the final checkpoint (+0.7%) or aggregating all checkpoints (+0.8%) is far inferior to AutoMixer's task-aligned selection strategy (+1.22% to +1.93%).
- **Gains Diminish as the Target Model Scale Increases**: The 350M target model achieves +1.93%, the 1.5B model achieves +1.22%, and the 3B model achieves +1.05%. This is likely because the proxy model is fixed at 350M, resulting in a widening discrepancy with larger target models.
- **AutoMixer Maintains an Advantage Throughout Training**: The performance trajectory reveals that AutoMixer-350M consistently outperforms uniform sampling from the early stages of training, ultimately reaching an accuracy of 56.45% vs 51.82%.
- **Small Proxy Models Tend to Favor Long Sentences**: The 75M proxy model favors long-text samples with high influence scores, whereas the 350M model better discriminates between high- and low-influence samples.

## Highlights & Insights

- **A New Perspective of Checkpoints as Signal Sources**: Repurposing intermediate checkpoints, which are typically discarded or only used for resumes, into data-quality signals is a highly elegant and practical observation.
- **Multi-Checkpoint Aggregation Captures Skill Emergence Dynamics**: Sampling checkpoints at different training steps and aggregating their influence scores effectively models the non-monotonic nature of skill acquisition.
- **Efficiency-Oriented Engineering Design**: Calculating gradients only at the embedding and final layers, alongside using the DataInf approximation to avoid Hessian inversion, are highly practical designs that drastically reduce computational overhead without sacrificing efficacy.
- **Intuitive Design of Blending Factors**: Allocating weights based on checkpoint step counts assigns a higher data priority to tasks that take longer to learn, aligning with the intuition that harder-to-learn skills require more targeted data support.

## Limitations & Future Work

- The computational overhead of computing influence scores for the proxy model remains substantial (approximately 120 hours/100 GPUs for checkpoint evaluation plus 48 hours for the simulated run), which limits its scalability to larger pre-training runs.
- Evaluated only on reasoning benchmarks, without exploring other capability dimensions such as generation or coding tasks.
- The failure of the 75M proxy model implies that the framework is highly sensitive to the scale of the proxy model, and theoretical guidance on selecting the optimal proxy scale is currently lacking.
- The current data regrouping is a one-time process; dynamic adjustments (such as iteratively updating the data mixture during training) have not been explored.
- Potential data bias issues are not discussed—optimizing for specific benchmarks might lead to the degradation of other capabilities.

## Related Work & Insights

- **vs Data Mixing Laws (Ye et al., 2024)**: While Data Mixing Laws fit data mixing ratios via predictor models, AutoMixer directly leverages the influence functions of checkpoints, removing the need for auxiliary predictor models.
- **vs TAGET (Chang et al., 2024)**: A prior work by the same authors utilized n-gram sampling for target-aware data selection. AutoMixer replaces n-gram matching with influence functions, providing a stronger theoretical foundation at the cost of higher computational overhead.
- **vs Data Selection Methods like DSIR/D4**: These methods are typically based on domain matching or perplexity, whereas AutoMixer provides finer, sample-level signals through multi-checkpoint influence functions.

## Rating

- Novelty: ⭐⭐⭐⭐ Repurposing checkpoint artifacts as data-mixing signals offers a novel and practical perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Spans three model scales, four baselines for comparison, and detailed ablations and analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear framework description, complete mathematical derivations, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Establishes a new paradigm for pre-training data optimization, with generalizable ideas regarding intermediate checkpoint utilization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FACE: Faithful Automatic Concept Extraction](../../NeurIPS2025/others/face_faithful_automatic_concept_extraction.md)
- [\[ICML 2025\] Efficient Network Automatic Relevance Determination](../../ICML2025/others/efficient_network_automatic_relevance_determination.md)
- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)
- [\[ACL 2025\] AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)
- [\[ACL 2025\] FastMCTS: A Simple Sampling Strategy for Data Synthesis](fastmcts_a_simple_sampling_strategy_for_data_synthesis.md)

</div>

<!-- RELATED:END -->
