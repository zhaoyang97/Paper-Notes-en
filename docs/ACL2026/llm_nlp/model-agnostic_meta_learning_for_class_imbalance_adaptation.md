---
title: >-
  [Paper Note] Model-Agnostic Meta Learning for Class Imbalance Adaptation
description: >-
  [ACL 2026][LLM/NLP][Class Imbalance] This paper proposes HAMR (Hardness-Aware Meta-Resample), a unified meta-learning framework. It dynamically estimates instance-level weights through bi-level optimization to prioritize…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Class Imbalance"
  - "Meta-Learning"
  - "Adaptive Weighting"
  - "Hardness-Aware Resampling"
  - "Bi-level Optimization"
date: 2026-05-08
content_hash: 8b93caa58748c29b
---

# Model-Agnostic Meta Learning for Class Imbalance Adaptation

**Conference**: ACL 2026  
**arXiv**: [2604.18759](https://arxiv.org/abs/2604.18759)  
**Code**: [GitHub](https://github.com/trust-nlp/ImbalanceLearning)  
**Area**: Medical Imaging  
**Keywords**: Class Imbalance, Meta-Learning, Adaptive Weighting, Hardness-Aware Resampling, Bi-level Optimization

## TL;DR

This paper proposes HAMR (Hardness-Aware Meta-Resample), a unified meta-learning framework. It dynamically estimates instance-level weights through bi-level optimization to prioritize truly difficult samples. Combined with a neighborhood-aware resampling mechanism, it shifts training focus toward hard samples and their semantic neighbors, consistently outperforming strong baselines across six imbalanced NLP datasets.

## Background & Motivation

**Background**: Class imbalance is pervasive in NLP tasks such as text classification and Named Entity Recognition (NER). Existing methods mainly fall into two categories: loss re-weighting (e.g., Focal Loss, Dice Loss) and data resampling (oversampling/synthetic generation).

**Limitations of Prior Work**: (1) These methods often rely on predefined static heuristics, applying the same adjustment ratio to all samples within the same class; (2) Sample difficulty is not synonymous with class membership—not all minority class instances are inherently difficult, and not all majority class samples are trivial; (3) Static schemes may erroneously down-weight informative majority samples while over-emphasizing simple minority instances.

**Key Challenge**: There is a need for a method that can dynamically identify and prioritize truly difficult samples—regardless of their class membership—adjusting the learning strategy based on the model's evolving understanding of the data.

**Goal**: Design a unified framework to simultaneously address both class imbalance and instance-level difficulty, dynamically guiding the model's learning focus.

**Key Insight**: Decouple "what the model should focus on" (adaptive weights) and "what the model should see" (resampling) into two complementary modules, unified and optimized via a meta-learning framework.

**Core Idea**: Use bi-level meta-optimization to dynamically learn instance importance weights (the inner loop uses pre-meta weights for intermediate updates, while the outer loop updates the weight network on a balanced meta-validation set to obtain post-meta weights for the actual update). This is coupled with neighborhood-enhanced resampling based on FAISS to shift the training distribution toward difficult semantic regions.

## Method

### Overall Architecture

HAMR consists of two core modules: (1) Adaptive Weight Estimation, which maps normalized sample losses to importance weights via a lightweight weight network $f_\theta$ adjusted through bi-level meta-optimization; (2) Hardness-Aware Region Resampling, which dynamically adjusts the training distribution based on EMA-smoothed hardness scores and KNN-based neighborhood enhancement. The two modules collaborate within a unified training loop.

### Key Designs

1. **Adaptive Weight Estimation via Bi-level Meta-Optimization**:

    - **Function**: Dynamically adjusts the importance of each training sample based on the model's current learning state.
    - **Mechanism**: The inner loop uses the current pre-meta weights $w_i^{\text{pre}}$ from the weight network to perform an intermediate gradient update, resulting in $\phi'$. The outer loop evaluates $f_{\phi'}$ on a balanced meta-validation set $\mathcal{D}_{\text{meta}}$ to update the weight network parameters $\theta$. Finally, the updated weight network recalculates the post-meta weights $w_i^{\text{post}}$ for the actual model update. For token-level tasks, the maximum token loss per sentence is used; for classification tasks, sample-wise cross-entropy is used.
    - **Design Motivation**: Pre-meta weights reflect "what the model considers important before the update," while post-meta weights reflect "what is truly important under the guidance of a balanced validation set." This "try-before-you-act" strategy adapts better to training dynamics than static heuristics.

2. **Hardness-Aware Region Resampling with Neighborhood Enhancement**:

    - **Function**: Dynamically shifts the training distribution to expose the model to more difficult semantic regions.
    - **Mechanism**: Global hardness scores $h_i$ are obtained by EMA-smoothing post-meta weights: $h_i \leftarrow \gamma \cdot h_i + (1-\gamma) \cdot w_i^{\text{post}}$. The top 20% hardest samples are selected, and FAISS-accelerated KNN is used to find $k$ semantic neighbors for each, calculating a neighborhood enhancement score $b_i$. The final sampling probability is $p_i \propto (h_i + \varepsilon)^\tau \cdot (1 + \lambda b_i)$, where a temperature $\tau < 1$ encourages balanced exploration.
    - **Design Motivation**: Focusing solely on isolated hard samples is insufficient; semantic neighbors of hard samples often pose similar challenges. Neighborhood enhancement diffuses difficulty from single samples to entire semantic regions.

3. **Unified Training Loop**:

    - **Function**: Seamlessly integrates weight estimation and resampling into an end-to-end training process.
    - **Mechanism**: Neighborhood enhancement is updated every $F$ epochs (to avoid the overhead of per-step KNN). Every mini-batch follows a complete pipeline: Sampling $\rightarrow$ Pre-meta weights $\rightarrow$ Inner update $\rightarrow$ Meta-step $\rightarrow$ Post-meta weights $\rightarrow$ Outer update $\rightarrow$ EMA update.
    - **Design Motivation**: The two modules reinforce each other—weight estimation provides instance-level signals, while resampling ensures the model encounters sufficient samples from difficult regions.

### Loss & Training

The main loss is weighted cross-entropy or token-level loss. The balanced meta-validation set is constructed by combining the full original validation set with supplementary samples from the training set, sampled to match the median class frequency. Weights are normalized using batch-wise z-scores before being processed by the weight network, with outputs clipped to a fixed range for numerical stability.

## Key Experimental Results

### Main Results

| Dataset | Task | HAMR Macro-F1 | Best Baseline Macro-F1 | Gain |
|--------|------|-------------|----------------|------|
| BioNLP | NER | 72.7 | 70.6 (Dice) | +2.1 |
| TweetNER | NER | 60.2 | 59.0 (Dice/LNR) | +1.2 |
| MIT-Restaurant | NER | 81.1 | 80.4 (Dice) | +0.7 |
| Hurricane-Irma17 | CLS | 73.4 | 72.7 (ICF) | +0.7 |
| Cyclone-Idai19 | CLS | 65.7 | 63.8 (ICF) | +1.9 |
| SST-5 | CLS | 57.0 | 56.3 (ICF) | +0.7 |

### Ablation Study

| Configuration | BioNLP F1 | Irma17 F1 | Description |
|------|----------|----------|------|
| HAMR (Full) | 72.7 | 73.4 | Complete model |
| w/o Resampling | 71.4 | 72.1 | Removed neighborhood resampling |
| w/o Meta-Weight | 70.9 | 71.8 | Removed adaptive weighting |
| w/o Neighborhood Boost | 71.8 | 72.5 | Removed KNN neighborhood boost |

### Key Findings

- HAMR achieves the best Macro-F1 across all 6 datasets, with the largest advantage (+1.9 pp) on the highly imbalanced Cyclone-Idai19 dataset ($IR=98.4$).
- Both modules contribute synergistically; removing either leads to performance degradation, though meta-weighting contributes slightly more than resampling.
- Neighborhood enhancement provides a consistent marginal gain (+0.6–0.9 pp), proving the value of diffusing difficulty from single points to regions.

## Highlights & Insights

- The "try-before-you-act" strategy of bi-level meta-optimization is elegant—pre-meta weights act as a "draft," and feedback from the meta-validation set allows the weight network to learn which weight distributions truly favor generalization. This concept is transferable to any scenario requiring dynamic training priority adjustment.
- The neighborhood-enhanced resampling approach is unique—treating hard samples as "seeds" and diffusing difficulty signals through semantic neighborhoods is more robust than focusing on isolated hard samples.
- The unified framework design decouples "what to focus on" (weights determine how to learn) and "what to see" (resampling determines what to learn from).

## Limitations & Future Work

- Reliance on FAISS for KNN may present computational bottlenecks for extremely large datasets.
- Construction of the meta-validation set relies on assumptions regarding reasonable class distributions.
- Validated only on BERT-based encoders; applicability in the LLM era remains unknown.
- Integration with synthetic data augmentation methods has not been explored.

## Related Work & Insights

- **vs Focal Loss/Dice Loss**: Static heuristics do not distinguish instance difficulty; HAMR dynamically learns instance weights.
- **vs Meta-Weight-Net**: Similar meta-learning frameworks lack neighborhood resampling; HAMR adds region-level training distribution adjustments.
- **vs SMOTE**: Instead of synthesizing new samples, HAMR dynamically adjusts existing sample weights and frequencies, making it more lightweight and free from generation noise.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of bi-level meta-optimization and neighborhood resampling is novel, though individual components have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covering 6 datasets across 2 tasks with detailed ablation, though more comparisons with recent methods could be included.
- Writing Quality: ⭐⭐⭐⭐ Method is clear, and the algorithmic pseudocode is complete.
- Value: ⭐⭐⭐⭐ Provides a general and effective solution for class imbalance in NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] System Prompt Optimization with Meta-Learning](../../NeurIPS2025/llm_nlp/system_prompt_optimization_with_meta-learning.md)
- [\[ACL 2026\] Automatic Combination of Sample Selection Strategies for Few-Shot Learning](automatic_combination_of_sample_selection_strategies_for_few-shot_learning.md)
- [\[NeurIPS 2025\] C²Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning](../../NeurIPS2025/llm_nlp/c2prompt_class-aware_client_knowledge_interaction_for_federated_continual_learni.md)
- [\[ACL 2026\] UCS: Estimating Unseen Coverage for Improved In-Context Learning](ucs_estimating_unseen_coverage_for_improved_in-context_learning.md)
- [\[ACL 2026\] OOD Proxy Demonstration Retrieval Scheme for Robust In-Context Learning](toward_robust_in-context_learning_leveraging_out-of-distribution_proxies_for_tar.md)

</div>

<!-- RELATED:END -->
