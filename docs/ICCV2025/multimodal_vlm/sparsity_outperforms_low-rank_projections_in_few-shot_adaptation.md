---
title: >-
  [Paper Note] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation
description: >-
  [ICCV 2025][Multimodal VLM][Sparse optimization] This paper proposes Sparse Optimization (SO), a framework that replaces low-rank adaptation methods (e.g., LoRA) via dynamic sparse gradient selection and importance-based momentum pruning. SO achieves state-of-the-art performance on few-shot VLM adaptation across 11 datasets while reducing memory overhead.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Sparse optimization
  - few-shot adaptation
  - CLIP
  - low-rank decomposition
  - parameter-efficient fine-tuning
date: 2026-05-08
content_hash: ad61e5353e4ff45a
---

# Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation

**Conference**: ICCV 2025
**arXiv**: [2504.12436](https://arxiv.org/abs/2504.12436)
**Code**: [https://github.com/nairouz/SO](https://github.com/nairouz/SO)
**Area**: Multimodal VLM
**Keywords**: Sparse optimization, few-shot adaptation, CLIP, low-rank decomposition, parameter-efficient fine-tuning

## TL;DR
This paper proposes Sparse Optimization (SO), a framework that replaces low-rank adaptation methods (e.g., LoRA) via dynamic sparse gradient selection and importance-based momentum pruning. SO achieves state-of-the-art performance on few-shot VLM adaptation across 11 datasets while reducing memory overhead.

## Background & Motivation
Vision-language models (VLMs) such as CLIP acquire strong zero-shot capabilities through large-scale image-text pretraining, yet adapting them to new tasks under few-shot settings remains prone to severe overfitting and catastrophic forgetting. Existing parameter-efficient tuning (PET) methods fall into two main categories:

**Adapter/prompt learning methods**: Introduce additional trainable parameters for adaptation, but are susceptible to overfitting and require extensive hyperparameter search.

**Low-rank reparameterization methods** (LoRA, DoRA, etc.): Decompose weight updates into low-rank matrices to reduce parameter count, but suffer from a fundamental tension—**low-rank constraints limit model expressivity**, while **a fixed rank requires trading off underfitting against overfitting**.

The authors systematically expose the core drawbacks of LoRA:
- Test accuracy rises initially during training but then drops sharply (severe overfitting).
- The optimal rank varies substantially across datasets and cannot be tuned via a validation set (which does not exist in few-shot settings).
- Because rank is a discrete value (minimum of 1), low-rank methods lack the flexibility to finely control a small number of parameters.

**Core Idea**: Inspired by the sparse activation mechanism of the human brain, the paper replaces low-rank constraints with high sparsity—only an extremely small fraction of parameters (e.g., 0.05%) are dynamically updated per iteration. This limits local learning capacity to prevent overfitting, while the dynamic variation of the support set preserves the overall model expressivity.

## Method

### Overall Architecture
SO (Sparse Optimization) is a sparse optimizer built upon the Adam optimizer, reducing to standard Adam when the sparsity rate is zero. The key modifications apply sparsification jointly to the gradients and the momentum, following two central design paradigms.

### Key Designs
1. **Local Sparsity & Global Density**:

    - **Function**: Only an extremely small subset of parameters is updated per iteration (controlled by density ratio $\kappa$), and this subset changes dynamically.
    - **Mechanism**: At each step, $M = \lfloor \kappa d \rfloor$ gradient values are retained ($d$ is the total parameter count), and the sparse support set is refreshed every $T$ iterations.
    - **Design Motivation**: Static sparsity (always updating the same parameter subset) causes the model to rely on a fixed set of connections, degrading performance by 3.4% (Table 4). Dynamic selection ensures that different parameters receive update opportunities throughout training, balancing low local learning capacity with high global expressivity.

2. **Local Randomness & Global Importance**:

    - **Function**: Gradients are sparsified via **random selection**; first-order momentum is sparsified via **importance ranking**.
    - **Mechanism**:
        - Sparse gradient: $\tilde{g}_t = \text{Random-}M(g_t)$, retaining $M$ randomly selected gradient elements.
        - Temporary momentum: $\mu_t = \beta_1 \tilde{\mu}_{t-1} + (1-\beta_1)\tilde{g}_t$ (containing at most $2M$ values).
        - Sparse momentum: $\tilde{\mu}_t = \text{Top-}M(\mu_t)$, retaining the top $M$ entries by magnitude.
        - Second-order momentum alignment: $\tilde{\nu}_t = \nu_t[\mathcal{I}(\tilde{\mu}_t)]$, using the same index set as the first-order momentum.
    - **Design Motivation**: Pruning gradients by importance causes the model to over-rely on locally high-magnitude updates, accelerating overfitting (importance gradient + importance momentum yields only 66.6% average accuracy vs. 74.6% for SO, Table 3). In contrast, momentum aggregates long-term information across the entire optimization trajectory; retaining it by importance ensures that critical parameters continue to receive consistent updates.

3. **Parameter Update**:

    - **Function**: Standard Adam-style update applied only to the sparsely selected parameters.
    - **Core Equations**:
        - Bias correction: $\hat{\mu}_t = \tilde{\mu}_t / (1-\beta_1^t)$, $\hat{\nu}_t = \tilde{\nu}_t / (1-\beta_2^t)$
        - Update rule: $\Theta_{t+1} = \Theta_t - \frac{\eta}{\sqrt{\hat{\nu}_t}+\epsilon}\hat{\mu}_t$

### Loss & Training
- Standard cross-entropy loss is used to optimize class prototypes.
- Training runs until convergence (loss < 0.01) or a maximum of 2,000 iterations.
- Density ratio $\kappa = 0.05\%$; support set refresh interval $T = 10$.
- All hyperparameters are fixed across all 11 datasets with no per-dataset tuning.

## Key Experimental Results

### Main Results

| Dataset | Metric | SO (Ours) | ReLoRA (Prev. SOTA) | Gain |
|--------|------|-----------|-------------------|------|
| 11-dataset avg. (1-shot) | Top-1 Acc | **73.8%** | 72.5% | +1.3% |
| 11-dataset avg. (2-shot) | Top-1 Acc | **76.6%** | 75.3% | +1.3% |
| 11-dataset avg. (4-shot) | Top-1 Acc | **78.9%** | 77.1% | +1.8% |
| Aircraft (1-shot) | Top-1 Acc | **31.5%** | 28.8% | +2.7% |
| EuroSAT (1-shot) | Top-1 Acc | **78.2%** | 73.8% | +4.4% |
| UCF101 (4-shot) | Top-1 Acc | **83.4%** | 80.0% | +3.4% |

### Ablation Study

| Configuration | 1-shot Avg. | 4-shot Avg. | Note |
|------|-----------|-----------|------|
| Dense gradient + Dense momentum (= Adam) | 4.4% | 13.6% | Full update causes severe overfitting |
| Sparse gradient + Dense momentum | 73.6% | 77.2% | Gradient sparsification is the key factor |
| Sparse gradient + Sparse momentum (SO) | **74.6%** | **80.4%** | Momentum sparsification yields further gains |
| Importance gradient + Importance momentum | 66.6% | — | Importance-based gradient selection leads to overfitting |
| Random gradient + Random momentum | 72.9% | — | Random momentum is inferior to importance-based |
| Static support set | 71.2% | 77.0% | Fixed support is inferior to dynamic |
| Dynamic support set (SO) | **74.6%** | **80.4%** | Dynamic refresh is highly effective |
| No momentum | 67.8% | 76.7% | Momentum is indispensable |

### Key Findings
- Adam nearly completely fails in few-shot settings (1-shot: 3.6%), confirming catastrophic overfitting from full-parameter updates.
- LoRA test accuracy exhibits severe oscillation during training; optimal rank and iteration count are highly dataset-dependent.
- SO at a 0.05% density ratio recovers expressivity close to full-parameter training, since different parameters are activated at different time steps.
- The combination of random gradient selection and importance-based momentum pruning is optimal, validating the "local randomness + global importance" paradigm.

## Highlights & Insights
1. **Novelty of approach**: The paper departs from the dominant low-rank adaptation paradigm and addresses few-shot overfitting through sparsity, drawing an analogy to the sparse activation mechanism of the human brain.
2. **Minimalist design**: Only two hyperparameters are introduced (density ratio $\kappa$ and refresh interval $T$), and the method is insensitive to their values, requiring no per-dataset tuning.
3. **Clear theoretical intuition**: Sparsification is decomposed into two independent subproblems—"which gradients to select" and "which momentum entries to retain"—with an orthogonal random/importance design achieving effective decoupling.
4. **Memory efficiency**: Both gradients and momentum maintain only $M$ values (far fewer than the full parameter count), resulting in lower actual memory usage than LoRA.

## Limitations & Future Work
1. Validation is limited to CLIP (ViT-B/16); the method has not been tested on larger-scale VLMs or LLMs (acknowledged by the authors in the conclusion).
2. The optimal density ratio $\kappa$ may vary with model scale, motivating further research into adaptive density scheduling strategies.
3. Although the support set refresh interval $T$ shows low sensitivity, an adaptive mechanism could theoretically be designed.
4. The combination with prompt learning methods remains unexplored—it is an open question whether the SO optimizer can effectively optimize prompt parameters.

## Related Work & Insights
- **Relation to LoRA**: LoRA reduces parameters via low-rank decomposition; SO reduces per-step updates via sparsity. Both constrain the update space, but sparsity has the advantage of dynamic variation.
- **Relation to GaLore**: GaLore projects gradients into a low-rank subspace; SO directly applies random sampling to gradients—the latter is simpler yet more effective.
- **Insight**: This work suggests that in resource-constrained settings, **dynamism is more important than fixed structure**—a perspective that complements the "winning ticket" intuition of the Lottery Ticket Hypothesis.

## Rating
- Novelty: ⭐⭐⭐⭐ Challenges the low-rank mainstream from a sparsity perspective; the two-paradigm design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 datasets, 7 baselines, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Logic is clear, paradigms are well-summarized, and figures are illustrative.
- Value: ⭐⭐⭐⭐ Establishes a new paradigm for few-shot fine-tuning, though validation on large models is absent.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)
- [\[AAAI 2026\] BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning](../../AAAI2026/multimodal_vlm/bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)
- [\[ICCV 2025\] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation](is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)

<!-- RELATED:END -->
