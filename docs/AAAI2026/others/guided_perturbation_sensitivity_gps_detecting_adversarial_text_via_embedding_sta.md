---
title: >-
  [Paper Note] Guided Perturbation Sensitivity (GPS): Detecting Adversarial Text via Embedding Stability and Word Importance
description: >-
  [AAAI 2026][adversarial text detection] This paper proposes the Guided Perturbation Sensitivity (GPS) framework, which detects adversarial text samples by masking important words and measuring changes in embedding stabil…
tags:
  - "AAAI 2026"
  - "adversarial text detection"
  - "embedding stability"
  - "word importance"
  - "BiLSTM"
  - "perturbation sensitivity"
date: 2026-05-08
content_hash: 3b84dba12f5d1542
---

# Guided Perturbation Sensitivity (GPS): Detecting Adversarial Text via Embedding Stability and Word Importance

**Conference**: AAAI 2026
**arXiv**: [2508.11667](https://arxiv.org/abs/2508.11667)
**Code**: [GitHub](https://github.com/ReDASers/Guided-Perturbation-Sensitivity)
**Area**: Others
**Keywords**: adversarial text detection, embedding stability, word importance, BiLSTM, perturbation sensitivity

## TL;DR

This paper proposes the Guided Perturbation Sensitivity (GPS) framework, which detects adversarial text samples by masking important words and measuring changes in embedding stability. GPS achieves 85%+ detection accuracy across 3 datasets, 3 attack types, and 2 models, and generalizes across datasets, attacks, and models without retraining.

## Background & Motivation

Adversarial text attacks pose a persistent threat to Transformer models — a single word substitution can cause a state-of-the-art model to misclassify a positive review as negative. Unlike continuous pixel perturbations in the vision domain, text attacks operate in a discrete vocabulary space, requiring semantic preservation while fooling the model, which makes detection considerably more challenging.

The core tension facing existing defenses is: either they rely on prior knowledge of specific attack patterns (lacking generality), or they require expensive model retraining (limited practicality). Output-layer methods tend to overfit to particular attacks, while gradient-based detectors ignore the sequential structure of adversarial manipulations.

The paper's core insight draws from a theoretical foundation: adversarial examples reside in high-curvature regions near decision boundaries, where small perturbations cause drastic classification changes. The authors hypothesize that this instability extends beyond decision boundaries into the representation space itself — by strategically masking important words, adversarial samples should exhibit disproportionately high sensitivity compared to naturally important words in benign samples. This insight motivates the GPS approach: using patterns of embedding instability as a "fingerprint" for adversarial samples.

## Method

### Overall Architecture

The GPS pipeline consists of four stages: (1) computing a reference embedding; (2) ranking words using an importance heuristic; (3) measuring embedding sensitivity via sequential masking; and (4) feeding the sensitivity–importance feature tensor into a BiLSTM classifier. The target model requires no modification throughout this process.

### Key Designs

1. **Reference Embedding Computation**:

    - Function: Computes a global sentence embedding for the input text.
    - Mechanism: Averages the last-layer hidden states of all non-special tokens, $\mathbf{e}(\mathcal{T}) = \frac{1}{|\Omega|}\sum_{i \in \Omega} \mathbf{h}_i^{(L)}$, where $\Omega$ denotes the set of non-special tokens.
    - Design Motivation: Mean pooling captures global semantic shifts more effectively than the [CLS] token, providing a baseline for measuring masking-induced drift.

2. **Important Word Identification (Four Heuristics)**:

    - Function: Assigns and ranks importance scores to each word in the input text.
    - Core Idea — Gradient Attribution (optimal): Backpropagates gradients of the cross-entropy loss with respect to input embeddings; the importance of each word is the sum of $\ell_2$ norms of its sub-token gradients, $\alpha_k^{\text{sal}} = \sum_{j \in \mathcal{S}_k} \|\nabla_{\mathbf{e}_j} \ell(\mathcal{T})\|_2$.
    - Also evaluates Attention Rollout (aggregating multi-layer attention weights), Grad-SAM (element-wise product of gradients and attention), and random selection.
    - Design Motivation: Gradient methods directly reflect which words are most critical to model predictions. Adversarially modified words tend to carry the strongest gradient signals, enabling more precise localization of tampered tokens.

3. **Sequential Sensitivity Analysis (Core Module)**:

    - Function: Masks the top-K important words one by one and measures the embedding shift caused by each masking operation.
    - Mechanism: For each selected word $w_k$, the token is replaced with [MASK] and the embedding $\tilde{\mathbf{e}}_k$ is recomputed; sensitivity is defined as cosine distance $s_k = 1 - \frac{\mathbf{e}(\mathcal{T}) \cdot \tilde{\mathbf{e}}_k}{\|\mathbf{e}(\mathcal{T})\|_2 \|\tilde{\mathbf{e}}_k\|_2}$.
    - Key Finding: Adversarially modified words exhibit approximately 2× the sensitivity of naturally important words in benign samples, indicating that embedding instability is an intrinsic property of adversarial examples.
    - Design Motivation: Word-by-word masking avoids interaction effects from simultaneous masking and precisely quantifies each word's contribution to the overall representation.

4. **GPS Feature Tensor and BiLSTM Classifier**:

    - Function: Stacks the sensitivity sequence and importance sequence into an $N \times 2$ feature matrix $\mathbf{Z} = [\mathbf{s} \| \boldsymbol{\alpha}]$, preserving positional information of the original word order.
    - The BiLSTM has only 257,154 parameters, trained with AdamW ($lr = 5 \times 10^{-4}$), batch size 32, and early stopping (patience = 5 epochs).
    - Design Motivation: BiLSTM captures sequential dependencies in sensitivity patterns (e.g., adversarial modifications tend to cluster at specific positions), while maintaining low parameter count and manageable computational cost.

### Loss & Training

Standard binary cross-entropy loss is used. Training employs a balanced set of 5,000 samples (20% held out as validation) and a test set of 1,000 samples. Adversarial samples in the training data are restricted to those that successfully fool the model (unsuccessful perturbations are excluded) to ensure training quality.

## Key Experimental Results

### Main Results

The experimental matrix spans 3 datasets × 3 attacks × 2 models × 4 importance methods = 72 configurations, benchmarked against two state-of-the-art baselines: TextShield and Sharpness-based detection.

| Dataset | Model | Attack | GPS(Grad) | TextShield | Sharp | Gain (vs. best baseline) |
|--------|------|------|-----------|------------|-------|---------|
| AG News | RoBERTa | TextFooler | 0.887 | 0.893 | 0.874 | −0.6% |
| AG News | RoBERTa | DeepWordBug | 0.895 | 0.883 | 0.860 | +1.2% |
| IMDB | RoBERTa | TextFooler | 0.919 | 0.870 | 0.888 | +3.1% |
| IMDB | DeBERTa | DeepWordBug | **0.968** | 0.775 | 0.775 | **+19.3%** |
| Yelp | DeBERTa | TextFooler | 0.917 | 0.917 | 0.911 | +0.0% |
| Yelp | DeBERTa | DeepWordBug | 0.931 | 0.902 | 0.893 | +2.9% |

GPS(Grad) matches or surpasses the baselines in the majority of the 18 configurations. The largest performance advantage occurs on IMDB + DeBERTa + DeepWordBug (+19.3%), where TextShield and Sharp degrade severely.

### Ablation Study

**Importance Method Ablation** (sensitivity analysis, Table 1):

| Importance Method | Benign Mean | Adversarial Mean | Ratio |
|-----------|---------|---------|------|
| Gradient | 0.014 | 0.028 | 1.932 |
| Attention Rollout | 0.014 | 0.028 | 1.912 |
| Grad-SAM | 0.014 | 0.027 | 1.836 |
| Random | 0.013 | 0.026 | 1.880 |

Adversarial samples exhibit approximately 2× the sensitivity of benign samples across all methods, confirming that embedding instability is an inherent property of adversarial examples.

**K Value Ablation**:

| K | Relative Performance (vs. K=50) | Computation Trend |
|-----|-----------------|-------------|
| K=5 | 98%+ | Minimal |
| K=10 | 99%+ | Linear growth |
| K=20 | ~100% | Linear growth |
| K=50 | 100% | Maximum |

K=5 achieves 98% of peak performance; performance varies minimally as K increases (<0.015 F1), while computation time grows linearly. The optimal trade-off lies at $K \in [5, 10]$.

### Key Findings

- Adversarial samples exhibit approximately 2× embedding sensitivity; in 88.9% of experimental configurations, adversarial samples are more unstable than benign ones.
- Gradient attribution significantly outperforms attention-based methods in identifying perturbed tokens under word-level attacks; NDCG ranking quality is strongly correlated with detection performance ($\rho > 0.65$).
- Character-level attacks (DeepWordBug) operate via a different detection mechanism, with no correlation between perturbation identification quality and detection performance.
- GPS demonstrates robust generalization across all three transfer scenarios: cross-dataset, cross-attack, and cross-model.

## Highlights & Insights

- An elegant bridge from theory to practice: the theoretical insight about decision boundary instability is extended to the embedding space, yielding a simple yet effective detection method.
- The concept of "embedding sensitivity fingerprinting" is highly generalizable and does not depend on specific attack types or model architectures.
- K=5 achieves 98% of detection performance, meaning only 5 forward passes per input are needed at inference time — deployment cost is minimal.
- The paper reveals that word-level and character-level attacks operate through fundamentally different detection mechanisms, providing directional guidance for future unified detection frameworks.

## Limitations & Future Work

- Requires white-box model access (for gradient computation); alternative approaches (e.g., surrogate model saliency) are needed for purely black-box settings.
- The BiLSTM classifier requires labeled training data and may need a small number of labels for entirely novel attack types.
- Detection of character-level attacks relies on a different mechanism that the current framework does not specifically optimize for.
- Adaptive strategies for selecting K based on input features remain unexplored.
- Combining gradient and attention heuristics to simultaneously handle word-level and character-level attacks is a promising direction.

## Related Work & Insights

- TextShield (Shen et al., 2023) employs an ensemble of four LSTMs to process gradient features; GPS achieves comparable or superior performance with a single BiLSTM operating on sensitivity–importance pairs, demonstrating that feature design matters more than model complexity.
- Sharp (Zheng et al., 2023) detects adversarial examples via loss landscape curvature but is sensitive to dataset and model shifts; GPS's embedding sensitivity is more stable.
- This work makes an empirical contribution to the debate on whether attention can explain predictions: in adversarial detection, attention is demonstrably less reliable than gradients.

## Rating

- Novelty: ⭐⭐⭐⭐ — The core idea (embedding masking sensitivity) is intuitive and theoretically grounded, though the technical components (BiLSTM classifier, etc.) are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 18 configurations + 4 heuristics + 3-dimensional transfer experiments + K-value ablation + NDCG ranking analysis; highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Clear logic, rich figures and tables, and in-depth analysis.
- Value: ⭐⭐⭐⭐ — Provides a practical, attack-agnostic detection framework with notable computational efficiency; white-box access requirements limit applicability in some settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](boosting_adversarial_transferability_via_ensemble_non-attention.md)
- [\[AAAI 2026\] DECOR: Deep Embedding Clustering with Orientation Robustness](decor_deep_embedding_clustering_with_orientation_robustness.md)
- [\[AAAI 2026\] Scalable Vision-Guided Crop Yield Estimation](scalable_vision-guided_crop_yield_estimation.md)
- [\[AAAI 2026\] ASAG: Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance](toward_the_frontiers_of_reliable_diffusion_sampling_via_adversarial_sinkhorn_att.md)
- [\[ICLR 2026\] FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff](../../ICLR2026/others/fire_frobenius_isometry_reinitialization.md)

</div>

<!-- RELATED:END -->
