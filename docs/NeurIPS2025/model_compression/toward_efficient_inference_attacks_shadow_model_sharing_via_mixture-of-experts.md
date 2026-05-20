---
title: >-
  [Paper Note] Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts
description: >-
  [NeurIPS 2025][Model Compression][Inference attacks] This paper proposes a Mixture-of-Experts (MoE)-based shadow model sharing framework that reduces the overall training cost of shadow models by sharing feature extracti…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Inference attacks"
  - "shadow models"
  - "mixture-of-experts"
  - "membership inference attack"
  - "privacy and security"
date: 2026-05-08
content_hash: 551b3cf0db9d0032
---

# Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts

**Conference**: NeurIPS 2025
**arXiv**: [2510.13451](https://arxiv.org/abs/2510.13451)  
**Code**: None  
**Area**: Model Compression
**Keywords**: Inference attacks, shadow models, mixture-of-experts, membership inference attack, privacy and security

## TL;DR
This paper proposes a Mixture-of-Experts (MoE)-based shadow model sharing framework that reduces the overall training cost of shadow models by sharing feature extraction layers across multiple inference attack tasks while training only lightweight task-specific expert modules, maintaining or improving attack performance.

## Background & Motivation

**Background**: Privacy attacks such as membership inference attacks (MIA) and property inference attacks rely on training large numbers of shadow models to simulate the behavior of target models. Shadow models must be trained on data from similar distributions to learn patterns that distinguish member from non-member data points.

**Limitations of Prior Work**:
- Each attack task (membership inference, property inference, model inversion, etc.) typically requires training an **independent** set of shadow models, causing computational cost to scale linearly with the number of attack types.
- Although different attacks have different objectives, they all share the underlying need to understand "how the model processes data"—this commonality has not been exploited.
- In practical attack scenarios where an adversary may need to execute multiple inference attacks simultaneously, independently trained models are not scalable.

**Key Challenge**: Multiple attack tasks share the underlying knowledge of "understanding target model behavior," yet current methods learn this knowledge independently for each task.

**Key Insight**: Employ an MoE architecture to allow multiple attack tasks to share a **backbone network** (encoding a common representation of target model output behavior), with each attack task requiring only a lightweight **expert module**.

**Core Idea**: Train a universal shadow model backbone to capture general patterns of model behavior, then attach a small task-specific expert head for each task such as MIA and property inference.

## Method

### Overall Architecture
(1) Train $N$ shadow models in a standard manner and collect their output behaviors (e.g., loss values, confidence scores, gradient features) on member and non-member data; (2) Train a shared backbone network to encode these behavioral features into a universal representation; (3) Train a lightweight expert module (classification head) for each attack task, with a gating router to dynamically select and combine experts.

### Key Designs

1. **Shared Backbone**:

    - **Function**: Learns universal behavioral representations from shadow model output features (loss, log-probability, gradient norm, etc.).
    - **Mechanism**: A multi-layer MLP or Transformer encoder jointly trained on data from all attack tasks.
    - **Design Motivation**: Different attack types share substantial overlap in their understanding of "model behavior"—MIA focuses on differences in loss distributions over training data, while property inference focuses on gradient patterns for specific attribute data—both fundamentally concern how a model processes different inputs differentially.

2. **Task-Specific Experts**:

    - **Function**: A small classification head dedicated to each attack task.
    - **Mechanism**: Maps the shared representation to task-specific decisions (e.g., binary classification: member vs. non-member).
    - **Parameter count**: Each expert accounts for only 5–10% of the backbone parameters.

3. **Routing / Gating Mechanism**:

    - **Function**: Automatically selects the appropriate expert combination based on input data features and the attack task type.
    - **Mechanism**: Standard MoE top-$k$ routing.
    - **Design Motivation**: The features of certain data points may be better handled by specific experts.

### Loss & Training
- Backbone training: Multi-task joint loss $= \sum_{\text{task}} \lambda_{\text{task}} \cdot \mathcal{L}_{\text{task}}$
- Expert training: Independent classification loss per task.
- The backbone can be frozen after training, with only new expert modules trained—enabling incremental addition of new attack types.

## Key Experimental Results

### Main Results — Membership Inference Attack Performance

| Method | Shadow Model Training Volume | MIA AUC | Property Inference Acc | Total Compute |
|---|---|---|---|---|
| Independent shadow models (×3 tasks) | $N \times 3$ | Baseline | Baseline | 3× |
| **MoE Sharing** | $N$ + 3 small experts | **On par / +1–2%** | **On par / +1%** | **~1.4×** |
| Savings | — | — | — | **~55%** |

### Ablation Study

| Configuration | MIA AUC | Compute Savings | Notes |
|---|---|---|---|
| Independent training (baseline) | Baseline | 0% | $N$ shadow models per task |
| Shared backbone, independent experts | On par | ~40% | Backbone reuse |
| Shared backbone + MoE routing | **+1%** | **~55%** | Knowledge transfer across experts |
| Sharing only early layers | −1% | ~30% | Insufficient sharing depth |

### Key Findings
- **Behavioral representations across shadow models are highly transferable**—features relevant to different attack tasks exhibit 70%+ overlap.
- MoE routing outperforms a simple shared-backbone with multiple heads by ~1%, indicating that different data points genuinely benefit from different expert combinations.
- New attack types can be supported by training only a new expert on a frozen backbone—incremental extension incurs negligible cost.
- Attack performance does not degrade and in some cases improves—joint training acts as a regularizer, reducing single-task overfitting.

## Highlights & Insights
- **Applying MoE to privacy attacks** represents a distinctive cross-disciplinary contribution—it aligns with MoE's principle of "shared lower layers with specialized upper layers" while addressing practical efficiency challenges in multi-attack scenarios.
- From a defense perspective, this work also provides valuable insight—as attacks become more efficient, defenses must be correspondingly strengthened.
- The high transferability of shadow model behavioral representations is itself an interesting finding regarding the understanding of model behavior.

## Limitations & Future Work
- Experiments are conducted primarily on classification models; applicability to generative models such as LLMs has not been validated.
- When target models differ substantially in architecture, the effectiveness of the shared backbone may degrade.
- Security implications: making attacks more efficient may increase threats to machine learning systems.

## Related Work & Insights
- **vs. Standard MIA (Shokri et al.)**: Standard methods train shadow models independently for each task; this work shares underlying knowledge across tasks.
- **vs. Joint inference attacks**: Existing multi-type attacks are typically studied independently by different research groups; this work unifies the methodology.

## Rating
- Novelty: ⭐⭐⭐⭐ Unique combination of MoE and privacy attacks
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple attack types with ablation analysis
- Writing Quality: ⭐⭐⭐⭐ Clear methodology with well-motivated design choices
- Value: ⭐⭐⭐ Meaningful research direction, though with limited application scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dense Backpropagation Improves Training for Sparse Mixture-of-Experts](dense_backpropagation_improves_training_for_sparse_mixture-of-experts.md)
- [\[NeurIPS 2025\] Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging](mingle_mixture_of_null-space_gated_low-rank_experts_for_test-time_continual_mode.md)
- [\[NeurIPS 2025\] Online Mixture of Experts: No-Regret Learning for Optimal Collective Decision-Making](online_mixture_of_experts_no-regret_learning_for_optimal_collective_decision-mak.md)
- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](../../ICLR2026/model_compression/unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)

</div>

<!-- RELATED:END -->
