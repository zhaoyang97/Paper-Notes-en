---
title: >-
  [Paper Note] Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging
description: >-
  [NeurIPS 2025][Model Compression][Continual Learning] This paper proposes a new paradigm called Test-Time Continual Model Merging (TTCMM) and the Mingle framework, which employs a low-rank mixture-of-experts architecture with an adaptive null-space constrained gating mechanism to dynamically merge models at test time using a small number of unlabeled samples. Mingle outperforms state-of-the-art methods by 7–9% across multiple benchmarks while reducing forgetting to near zero.
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Continual Learning"
  - "Model Merging"
  - "Test-Time Adaptation"
  - "Mixture of Experts"
  - "Null-Space Constraint"
  - "Low-Rank Decomposition"
date: 2026-05-08
content_hash: f3bbc0915ffd76cf
---

# Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging

**Conference**: NeurIPS 2025
**arXiv**: [2505.11883](https://arxiv.org/abs/2505.11883)  
**Authors**: Zihuan Qiu, Yi Xu, Chiyuan He, Fanman Meng, Linfeng Xu, Qingbo Wu, Hongliang Li (UESTC, DUT)
**Code**: [GitHub](https://github.com/zihuanqiu/MINGLE)  
**Area**: Model Compression
**Keywords**: Continual Learning, Model Merging, Test-Time Adaptation, Mixture of Experts, Null-Space Constraint, Low-Rank Decomposition

## TL;DR

This paper proposes a new paradigm called Test-Time Continual Model Merging (TTCMM) and the Mingle framework, which employs a low-rank mixture-of-experts architecture with an adaptive null-space constrained gating mechanism to dynamically merge models at test time using a small number of unlabeled samples. Mingle outperforms state-of-the-art methods by 7–9% across multiple benchmarks while reducing forgetting to near zero.

## Background & Motivation

### State of the Field
Continual Model Merging (CMM) aims to sequentially fuse independently fine-tuned models in parameter space without access to the original training data. This paradigm is inherently advantageous in terms of data privacy, distributed training, and scalability. However, existing methods face two core challenges:

**Parameter interference and catastrophic forgetting**: As models are incrementally merged, overlapping or conflicting parameter updates accumulate.

**Insufficient adaptation to test distributions**: Merged models exhibit limited generalization when confronted with unseen or shifted task distributions.

### Limitations of Prior Work
- **Task Arithmetic (TA)**: Performs simple weighted summation of task vectors; sensitive to coefficients and prone to task interference.
- **OPCM**: Reduces interference via orthogonal projection but neglects adaptation to test distributions.
- **AdaMerging/WEMOE**: Introduces test-time adaptation but assumes all models and data are simultaneously available, making them unsuitable for continual settings.
- Existing structural constraints (orthogonal projection, linearization, sparsification) degrade in effectiveness as the number of tasks grows.

### Core Idea
The paper introduces test-time adaptation (TTA) into continual model merging, leveraging a small number of unlabeled test samples (5 per class) to dynamically adjust the merging strategy at inference time, while employing null-space constraints to preserve previously acquired knowledge.

## Method

### Overall Architecture
Mingle — Mixture of Null-Space Gated Low-Rank Experts — consists of three core components.

### 1. Low-Rank Expert Mixture
Upon the arrival of a new task $t$, low-rank experts are constructed from task vectors:

- **Orthogonal Projection**: The task vector is first projected onto the orthogonal complement of the directions already learned, eliminating overlap with existing experts.
- **Truncated SVD**: The projected result undergoes rank-$r$ truncated SVD to yield a compact LoRA-form expert $f_t = BA$.
- **Gated Aggregation**: The output of each layer is the sum of the pre-trained model's output and the gated weighted sum of all experts.

The gating function is a learnable linear projection; only the gating parameters for the current task are adapted at test time.

### 2. Test-Time Adaptation
Only the gating parameters $g_t$ for the current task $t$ are adapted; all experts and prior gating parameters are frozen. The optimization objective is to minimize the KL divergence between the merged model and the individually fine-tuned model, driven by a small set of unlabeled seed samples.

### 3. Adaptive Null-Space Constrained Gating
This is the core innovation, addressing the forgetting introduced by TTA:

**Hard Null-Space Projection**:
- The feature covariance matrices of seed samples from each task are cached, and the top-$k$ principal directions are extracted.
- Historical task directions are concatenated and orthogonalized.
- Gradients are projected onto the orthogonal complement of the old tasks' feature subspaces.

**Adaptive Relaxation Strategy**:
Hard projection may over-constrain plasticity; accordingly, an adaptive relaxation mechanism is introduced:
- **Interference Metric**: Computes the alignment between gradients and each old task direction.
- **EMA Smoothing**: Exponential moving average ($\beta = 0.99$) is applied to suppress stochastic noise.
- **Adaptive Decay**: $\lambda = \exp(-\gamma \cdot S)$; directions with greater interference receive stronger protection.
- **Relaxed Projector**: Smoothly interpolates between no protection and hard projection.

### Theoretical Support
Theorem 1 proves that, as long as the routing error is sufficiently small, the risk of the dynamic MoE is strictly lower than that of any static averaging strategy — theoretically justifying the necessity of data-dependent gating.

## Key Experimental Results

### Main Results: Continual Merging (CLIP Vision Models)

Evaluated on three CLIP-ViT backbones under 8/14/20-task settings, averaged over 10 random seeds:

| Method | ViT-B/32 8-task ACC | ViT-B/32 14-task ACC | ViT-B/32 20-task ACC | ViT-B/16 8-task ACC |
|--------|---------------------|----------------------|----------------------|---------------------|
| C. Task Arithmetic | 67.5% | 66.5% | 60.0% | 77.1% |
| OPCM | 75.5% | 71.9% | 65.7% | 81.8% |
| C. LoRA-WEMOE | 68.8% | 63.8% | 49.6% | 72.6% |
| **Mingle** | **85.8%** | **81.6%** | **77.1%** | **88.3%** |

BWT (forgetting metric; higher is better): Mingle achieves near-zero BWT across all settings (−0.1% to −2.2%), compared to OPCM (−4.3% to −7.8%) and C. LoRA-WEMOE (−13.6% to −27.9%).

### Ablation Study (ViT-B/16)

| TTA | Freeze Old Gates | Null-Space | Adaptive Relax | ACC(8) | BWT(8) | ACC(20) | BWT(20) |
|-----|-----------------|------------|----------------|--------|--------|---------|---------|
| No | — | — | — | 78.7% | −0.5% | 70.6% | −1.3% |
| Yes | No | No | No | 86.4% | −6.0% | 76.7% | −12.8% |
| Yes | Yes | No | No | 87.4% | −2.3% | 76.2% | −6.8% |
| Yes | Yes | Yes | No | 86.0% | −0.1% | 78.3% | −0.2% |
| Yes | Yes | Yes | Yes | **88.3%** | **−0.4%** | **81.9%** | **−1.9%** |

Adaptive relaxation substantially improves accuracy while maintaining near-zero forgetting (2–4% gain over hard constraint).

### Key Findings
- **NLP Tasks**: On 8 GLUE tasks using Flan-T5-base, Mingle achieves 83.3% average accuracy, surpassing all baselines.
- **Robustness**: Under 7 types of image corruption, Mingle achieves 73.2% average ACC (BWT −0.2%), substantially outperforming OPCM (67.3%, BWT −5.1%).
- **Sample Efficiency**: Even with only 1 sample per class, ACC improves from the static baseline of 70–79% to 81–88%.
- **Computational Cost**: Each task adaptation takes approximately 10 seconds, with only 36.9k trainable parameters at rank 64.

## Highlights & Insights

- **New Paradigm**: TTCMM is formally defined for the first time, bridging test-time adaptation and continual model merging.
- **Null-Space Constrained Gating**: Gradient updates are mechanistically confined to the orthogonal complement of old task feature subspaces; adaptive relaxation dynamically regulates constraint strength by monitoring interference signals in real time.
- **Efficiency and Practicality**: Requires only 5 unlabeled samples per class, 50 adaptation steps, and approximately 10 seconds per task; gating is fixed at inference time, ensuring low-latency deployment.
- **Comprehensive Validation**: Covers vision (CLIP) and language (Flan-T5), 3 backbones, 8/14/20 tasks, 10 random orderings, 7 image corruptions, and the MTIL benchmark.

## Limitations & Future Work

- **Requires Independently Fine-Tuned Models**: Assumes each task model has been independently fine-tuned from a shared pre-trained model; not applicable to heterogeneous model settings.
- **Limited Expressiveness of Linear Gating**: The linear projection gating may be insufficient under highly overlapping task distributions.
- **Growing Null-Space Dimensionality**: The accumulated null-space dimensionality grows linearly with the number of tasks ($k$ directions per task), potentially compressing the available gradient subspace for new tasks.
- **Classification-Centric Evaluation**: Experiments are conducted primarily on classification tasks; applicability to generation, detection, and other tasks remains unexplored.
- **Seed Sample Requirement**: A small number of unlabeled test samples is required; the method is not applicable in fully data-free scenarios.

## Related Work & Insights

- **vs. OPCM**: OPCM reduces parameter interference via orthogonal projection but lacks adaptation to test distributions; Mingle's TTA + MoE enables input-dependent dynamic merging, achieving 7–10% higher ACC.
- **vs. WEMOE**: WEMOE assumes all models and data are simultaneously available (multi-task setting); when applied directly to continual settings, it suffers severe forgetting (BWT < −20%).
- **vs. AdaMerging**: Also exhibits severe forgetting in continual settings (BWT −22% to −33%); general TTA methods cannot be directly transferred to continual scenarios.
- **vs. Traditional CL Methods**: Mingle achieves comparable performance under the independent fine-tuning merging paradigm (MTIL benchmark: 83.0% vs. best traditional CL at 86.8%).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First formal definition of TTCMM; null-space constrained gating with adaptive relaxation is an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers vision + NLP, 3 backbones, multiple task counts, 10 orderings, robustness, ablation, and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with tight integration of theory and experiments; notation is somewhat heavy in places.
- Value: ⭐⭐⭐⭐⭐ — Opens up the TTCMM paradigm; comprehensive 7–9% margins are highly convincing, with a low practical barrier to adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[ICLR 2026\] Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity](../../ICLR2026/model_compression/null-space_filtering_for_data-free_continual_model_merging_preserving_stability_.md)
- [\[ICML 2025\] BECAME: BayEsian Continual Learning with Adaptive Model MErging](../../ICML2025/model_compression/became_bayesian_continual_learning_with_adaptive_model_merging.md)
- [\[ACL 2025\] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](../../ACL2025/model_compression/more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)

</div>

<!-- RELATED:END -->
