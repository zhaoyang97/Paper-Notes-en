---
title: >-
  [Paper Note] Demystifying Language Model Forgetting with Low-Rank Example Associations
description: >-
  [NeurIPS 2025][LLM Safety][Catastrophic Forgetting] This paper discovers that the association matrix between upstream sample forgetting and newly learned tasks exhibits a low-rank structure (rank-3 achieves $R^2 > 0.69$) after LLM fine-tuning, and leverages matrix completion to predict forgetting induced by unseen tasks, thereby guiding selective replay to mitigate forgetting.
tags:
  - NeurIPS 2025
  - LLM Safety
  - Catastrophic Forgetting
  - Low-Rank Association
  - Matrix Completion
  - Forgetting Prediction
  - Selective Replay
date: 2026-05-08
content_hash: 50c05459af2a379a
---

# Demystifying Language Model Forgetting with Low-Rank Example Associations

**Conference**: NeurIPS 2025
**arXiv**: [2406.14026](https://arxiv.org/abs/2406.14026)
**Code**: [GitHub](https://github.com/AuCson/low-rank-forgetting)
**Area**: LLM Safety
**Keywords**: Catastrophic Forgetting, Low-Rank Association, Matrix Completion, Forgetting Prediction, Selective Replay

## TL;DR
This paper discovers that the association matrix between upstream sample forgetting and newly learned tasks exhibits a low-rank structure (rank-3 achieves $R^2 > 0.69$) after LLM fine-tuning, and leverages matrix completion to predict forgetting induced by unseen tasks, thereby guiding selective replay to mitigate forgetting.

## Background & Motivation

### State of the Field

**State of the Field**: Continual fine-tuning of LLMs leads to forgetting of upstream knowledge (catastrophic forgetting), a core challenge inherited from continual learning. Existing mitigation methods primarily rely on random replay of past samples, regularization (EWC, L2), or parameter isolation; however, none of these approaches characterize *what* the model will specifically forget.

### Limitations of Prior Work

**Limitations of Prior Work**: (1) Random replay is inefficient—without knowing which samples will be forgotten, it blindly replays a subset of all data; (2) The mechanism of forgetting is unclear—whether it is task-independent (certain samples are always vulnerable) or task-dependent (depending on what new task is learned); (3) The intuition that "semantically similar tasks cause more forgetting" lacks empirical validation.

### Root Cause

**Root Cause**: If one could predict which upstream samples will be forgotten after learning a new task, targeted replay would become possible—but this requires understanding the structure of forgetting. By constructing a forgetting matrix $Z$ of $M$ tasks × $N$ samples to analyze its complexity: if $Z$ is full-rank, forgetting is unpredictable; if low-rank, a simple latent structure exists that can be exploited.

### Starting Point

**Paper Goals**: To quantitatively analyze the rank structure of the forgetting matrix and establish a forgetting prediction model. **Starting Point**: Forgetting prediction is framed as analogous to collaborative filtering in recommender systems—tasks as users, samples as items, and forgetting as ratings. **Core Idea**: The forgetting matrix is low-rank → apply matrix completion (MF/KNN) to predict forgetting induced by new tasks → guide selective replay.

## Method

### Overall Architecture
A four-step pipeline: (1) Fine-tune the LLM separately on $M$ tasks, measure the degree of forgetting for $N$ upstream samples, and construct the forgetting matrix $Z \in \mathbb{R}^{M \times N}$; (2) Perform SVD low-rank decomposition to analyze the structure of $Z$ and quantify the goodness-of-fit $R^2$ at each rank; (3) Use matrix completion (MF or KNN) to predict forgetting induced by new tasks; (4) Sample replay data with weights proportional to predicted forgetting scores.

### Key Designs

1. **Forgetting Matrix Construction and Low-Rank Analysis**:

    - Function: Quantify the latent structural complexity of forgetting.
    - Mechanism: For 7 models including OLMo-1B/7B, Pythia-1B/6.9B, and MPT-1B/7B, fine-tuning is performed on 85 tasks, measuring forgetting (loss change) for 140,000 upstream samples to construct the forgetting matrix $Z$. SVD decomposition reveals that rank-1 (task-independent) already achieves $R^2 > 0.5$, rank-3 achieves $R^2 > 0.69$, and rank-5 achieves $R^2 > 0.75$.
    - Design Motivation: The low-rank structure implies that forgetting is not random—there exist "universally vulnerable samples" (rank-1 component) and a small number of "task-specific forgetting patterns" (higher-order components), providing a theoretical basis for prediction.

2. **Matrix Completion for Forgetting Prediction**:

    - Function: Predict which samples will be forgotten by unseen tasks.
    - Mechanism: Analogous to collaborative filtering in recommender systems—only a small number of observed forgetting patterns across tasks (known ratings) are needed to predict forgetting for arbitrary new tasks (unknown ratings). Using matrix factorization (MF) or KNN, the method achieves F1 = 58.16 on binary forgetting prediction, compared to a random baseline of only 6.4.
    - Design Motivation: Intuition suggests that semantic similarity can predict forgetting, but experiments reveal that textual/semantic similarity is nearly uncorrelated with forgetting ($\rho < 0.17$), and gradient inner products are also ineffective ($\rho \sim 0$). The only effective predictor is "forgetting correlation across tasks" ($\rho \sim 0.4$–$0.6$), which is precisely suited to collaborative filtering.

3. **Selective Replay**:

    - Function: Use forgetting predictions to guide targeted data replay.
    - Mechanism: Given a new task, matrix completion is used to predict the forgetting probability for each upstream sample, and replay data is sampled with weights proportional to these predicted values.
    - Design Motivation: Compared to random replay, selective replay concentrates the limited replay budget on the most vulnerable samples, yielding statistically significant reductions in forgetting.

## Key Experimental Results

### Main Results: Low-Rank Fit ($R^2$)

| Model | Rank-1 | Rank-3 | Rank-5 |
|------|:---:|:---:|:---:|
| OLMo-1B | ~0.55 | ~0.75 | ~0.80 |
| OLMo-7B | ~0.45 | ~0.69 | ~0.75 |
| Pythia-1B | ~0.75 | ~0.89 | ~0.92 |
| MPT-7B | ~0.70 | ~0.88 | ~0.91 |

The low-rank structure holds universally across 7 models from 4 model families.

### Forgetting Prediction Comparison

| Method | F1 | Notes |
|------|:---:|------|
| Random | 6.4 | Random baseline |
| Semantic similarity | ~20–30 | Intuitive approach performs poorly |
| Gradient inner product | ~15–25 | Traditional CL method also ineffective |
| **Matrix Factorization (MF)** | **58.16** | 9× better than random |

### Key Findings
- **Semantic similarity completely fails to explain forgetting** ($\rho < 0.17$)—an important counter-intuitive negative finding.
- **Gradient inner products are equally ineffective** ($\rho \sim 0$)—the theoretical framework of traditional continual learning does not hold for LLMs.
- **The only effective predictor is "cross-task forgetting correlation"** ($\rho \sim 0.4$–$0.6$).
- Larger and more capable models exhibit more complex forgetting patterns, yet low-rank approximation remains applicable (OLMo-7B has higher rank than 1B).

## Highlights & Insights
- **The low-rank structure of forgetting associations** demonstrates that LLM forgetting is not random but possesses a simple latent structure, laying a theoretical foundation for understanding forgetting.
- **The recommender-system analogy** elegantly reformulates forgetting prediction as a collaborative filtering problem, representing a highly effective cross-domain method transfer.
- **Value of negative findings**: Semantic similarity and gradient inner products fail to predict forgetting—correcting widely held intuitions in the field.

## Limitations & Future Work
- Constructing the initial association matrix requires complete fine-tuning across multiple tasks, which is costly (85 tasks × full fine-tuning).
- Experiments are limited to models up to 13B—whether low-rank structure holds for 70B+ models remains unknown.
- The improvement from selective replay is limited in absolute magnitude (statistically significant but modest), and may need to be combined with regularization methods.
- Forgetting is defined based on loss change, without accounting for finer-grained capability forgetting (e.g., disappearance of specific reasoning chains).

## Related Work & Insights
- **vs. Random Replay**: Prediction-guided replay is more targeted, improving F1 from 6.4 to 58.16.
- **vs. Model Editing (e.g., MEMOIR)**: Model editing modifies parameters directly, whereas this work mitigates forgetting via data selection—the two approaches are complementary.
- **vs. EWC/L2 Regularization**: Regularization constrains parameter changes, while this work offers a new perspective through example-level forgetting association analysis.
- The low-rank forgetting structure suggests that "knowledge storage" in LLMs may be more structured than previously assumed, warranting further exploration.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Both the discovery of low-rank forgetting structure and the matrix completion application are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 models, 85 tasks, 140,000 samples, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, systematic analysis, and intuitive visualizations.
- Value: ⭐⭐⭐⭐⭐ Makes a foundational contribution to understanding and mitigating LLM forgetting.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)
- [\[NeurIPS 2025\] Self-Refining Language Model Anonymizers via Adversarial Distillation](self-refining_language_model_anonymizers_via_adversarial_distillation.md)
- [\[NeurIPS 2025\] AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text](agentstealth_reinforcing_large_language_model_for_anonymizing_user-generated_tex.md)
- [\[NeurIPS 2025\] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)
- [\[NeurIPS 2025\] CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming](cpret_a_dataset_benchmark_and_model_for_retrieval_in_competitive_programming.md)

<!-- RELATED:END -->
