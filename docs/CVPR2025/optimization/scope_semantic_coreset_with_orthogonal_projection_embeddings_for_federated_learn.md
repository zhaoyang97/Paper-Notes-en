---
title: >-
  [Paper Note] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning
description: >-
  [CVPR 2025][Optimization][Federated Learning] SCOPE proposes a semantic coreset selection framework for federated learning. By leveraging zero-shot VLM (MobileCLIP-S2) to extract three scalar metrics (representation score, diversity score, and margin proximity), the server aggregates a global consensus to guide a two-stage pruning process (anomaly filtering + redundancy elimination) on clients. This achieves a 128-512× uplink bandwidth reduction and 7.72× speedup while mainta…
tags:
  - "CVPR 2025"
  - "Optimization"
  - "Federated Learning"
  - "Coreset Selection"
  - "Data Pruning"
  - "Long-tail Distribution"
  - "Zero-shot VLM"
date: 2026-05-08
content_hash: 37fceb23527f27bf
---

# SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning

**Conference**: CVPR 2025  
**arXiv**: [2603.12976](https://arxiv.org/abs/2603.12976)  
**Code**: TBD  
**Area**: Optimization / Federated Learning  
**Keywords**: Federated Learning, Coreset Selection, Data Pruning, Long-tail Distribution, Zero-shot VLM

## TL;DR
SCOPE proposes a semantic coreset selection framework for federated learning. By leveraging zero-shot VLM (MobileCLIP-S2) to extract three scalar metrics (representation score, diversity score, and margin proximity), the server aggregates a global consensus to guide a two-stage pruning process (anomaly filtering + redundancy elimination) on clients. This achieves a 128-512× uplink bandwidth reduction and 7.72× speedup while maintaining competitive accuracy.

## Background & Motivation
**Background**: Federated learning needs to handle highly skewed (non-IID + long-tail) distributed datasets. Coreset selection reduces communication and computation costs by discarding redundant data.

**Limitations of Prior Work**: (1) Existing coreset methods rely on local heuristics and lack awareness of the global data distribution, making them prone to discarding locally redundant but globally rare samples; (2) Gradient or loss-based methods tend to amplify sensor noise and artifacts in scientific data; (3) Some methods require sharing embeddings or surrogate datasets, violating privacy constraints.

**Key Challenge**: Federated nodes are "shortsighted" when making local coreset decisions—they cannot know whether the samples they discard are globally rare.

**Goal**: How to enable clients to gain global data distribution awareness and make better coreset selection decisions while preserving privacy?

**Key Insight**: Leverage zero-shot projections of VLMs to extract lightweight scalar metrics (without local training), and share only scalar statistics (mean/variance) to construct a global consensus.

**Core Idea**: Zero-shot projection of VLMs to extract three scalars + global statistical consensus + two-stage pruning (anomaly + redundancy).

## Method

### Overall Architecture
Clients apply MobileCLIP-S2 for zero-shot projection $\rightarrow$ compute three scalar metrics (RS / DS / $S_{\text{neg}}$) $\rightarrow$ send class-wise statistics to the server $\rightarrow$ server aggregates global consensus (law of total variance) $\rightarrow$ clients perform two-stage pruning: consensus filter for anomaly removal + dynamic balance for redundancy elimination.

### Key Designs

1. **Three Semantic Scalar Metrics**:

    - **RS (Representation Score)**: $RS_i = v_{\text{img},i} \cdot t_{c_i}$, which is the cosine similarity between the image embedding and the text prototype of the correct class, measuring the reliability of core class features.
    - **DS (Diversity Score)**: $DS_i = \|v_{\text{res},i}\|_2$, which is the norm of the residual vector orthogonal to the class prototype direction, measuring non-redundant visual variations.
    - **$S_{\text{neg}}$ (Margin Proximity)**: $S_{\text{neg},i} = \max_{j \neq c_i} v_{\text{img},i} \cdot t_j$, which represents the similarity to the nearest incorrect class prototype, measuring the ambiguity of the decision boundary.

2. **Global Consensus Construction (Scalar-only Transmission)**:

    - **Function**: The server aggregates scalar statistics from clients using the law of total variance.
    - **Mechanism**: $$[\sigma_{m,c}^{\text{Global}}]^2 = \frac{1}{N_c} \sum_k n_{k,c} [(\sigma_{m,c}^k)^2 + (\mu_{m,c}^k - \mu_{m,c}^{\text{Global}})^2]$$
    - **Design Motivation**: Transmit only scalar means and variances (instead of embeddings) to achieve a 128-512× bandwidth reduction.

3. **Two-Stage Local Pruning**:

    - **Consensus Filter**: Anomaly score $AS_i = \hat{Z}_{S_{\text{neg}},i} - \hat{Z}_{RS,i}$ is computed to filter out semantically contradictory samples.
    - **Redundancy Elimination**: $R_i = \hat{Z}_{RS,i} - \hat{Z}_{S_{\text{neg}},i} - \hat{Z}_{DS,i}$, where high scores represent typically redundant samples. Pruning is only applied to classes that are globally overrepresented (Targeting Metric $T_c = f_c / W_c$).

### Theoretical Guarantees
Non-convex convergence guarantee (Theorem 1): Anomaly pruning reduces gradient bias, and boundary alignment mitigates client drift $\tilde{\Gamma} \leq \lambda \Gamma$.

## Key Experimental Results

### Main Results

| Dataset | Model | SCOPE vs Full | SCOPE vs Best Baseline | Bandwidth Reduction |
|--------|------|--------------|----------------------|---------|
| CIFAR-10 | ResNet-18 | 56.48% vs 55.63% (outperformed Full) | More robust (smaller variance across pruning rates) | 128-512× |
| CIFAR-100 | ViT-B-16 | Competitive | More stable | 512× |
| Tiny-ImageNet | ResNet-50 | Competitive | — | 512× |
| UHCS (Scientific Data) | Swin-Tiny | Competitive | — | 128× |

### Tiny-ImageNet (ResNet-50, 100 clients)

| Setting | SCOPE pf=0.1 | Full DB | Best Baseline |
|------|-------------|---------|--------------|
| IR=2, $\alpha$=1.0 | 59.23% | 59.85% | EL2N 60.33% |
| IR=5, $\alpha$=0.1 | 54.65% | 54.41% | EL2N 54.38% |

- Under Tiny-ImageNet IR=5 $\alpha$=0.1, SCOPE actually achieves 55.38% at a high pruning fraction pf=0.9, **outperforming the full dataset of 54.41%**. This suggests that removing redundancy is beneficial under extreme skewness.

### Ablation Study / Key Findings
- The key advantage of SCOPE is not having the absolute highest accuracy, but its **robustness across pruning fractions**—manifested by the narrowest error bars.
- On CIFAR-10, SCOPE with pf=0.1 achieves 56.48%, **outperforming the full dataset's 55.63%**—due to filtering out anomalies and class imbalances that cause unstable federated aggregation.
- Global consensus is crucial: purely local pruning without global information collapses under extreme non-IID settings.
- A 7.72× wall-clock speedup (coreset selection without requiring local training).
- Strong performance on scientific research imaging data (UHCS) demonstrates the effectiveness of filtering noise artifacts.
- The long-tail preservation strategy (pruning only overrepresented classes) effectively preserves rare class samples.
- Gradient/loss-based baselines (e.g., EL2N, GradND) suffer from sharp performance degradation at high pruning rates, whereas SCOPE decays gracefully.

### System Efficiency
- 128-512× reduction in uplink bandwidth (transmitting only scalar statistics vs. transmitting embeddings/gradients).
- 7.72× wall-clock speedup for local coreset selection (zero-shot VLM projection vs. training-required methods).
- Significant reduction in FLOPs and VRAM footprint.

## Highlights & Insights
- **Zero-shot, training-free coreset selection**: Evaluates data utility using a frozen VLM for projection, requiring no local model training.
- **Global awareness via scalar-only sharing**: Sharing only three scalars $\times$ class-wise mean and variance results in minimal communication overhead while being sufficient to build a meaningful global consensus.
- **Information-theoretic viewpoint of orthogonal residuals**: Although DS can be derived from RS, independent normalization allows it to provide non-linear penalty signals in the global distribution.

## Limitations & Future Work
- Reliance on the zero-shot capability of MobileCLIP-S2—non-visual domains (e.g., text, audio) would require different encoders.
- The threshold $\beta$ is fixed to 0.5, which is simple but may not be optimal for all scenarios.
- Only classification tasks were tested, leaving more complex tasks like detection/segmentation to be validated.
- Theoretical convergence guarantees rely on standard assumptions (L-smoothness, bounded variance, etc.), which might not strictly hold under extreme scenarios.
- Dependence on the quality of VLM text prototypes: if class descriptions are imprecise, RS/DS/$S_{\text{neg}}$ may become inaccurate.

## Related Work & Insights
- **vs FedCS**: FedCS prunes via feature density but requires uploading embeddings (privacy risks and high bandwidth). SCOPE only transmits scalars.
- **vs GCFL**: GCFL relies on server-side surrogate datasets for gradient matching, violating strict privacy. SCOPE distributes entirely without data sharing.
- **vs GraND/EL2N**: Loss/gradient-based methods amplify artifacts in noisy data. SCOPE is more robust due to its reliance on semantic geometry.
- **vs FedCore**: FedCore requires local warmup training before coreset selection. SCOPE operates training-free, achieving much higher efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of VLM orthogonal projection + global scalar consensus is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, various IR/$\alpha$ settings, and theoretical analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear, research-problem-driven narrative.
- Value: ⭐⭐⭐⭐ A practical federated coreset scheme with significant improvements in communication efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning to Recall with Transformers Beyond Orthogonal Embeddings](../../ICLR2026/optimization/learning_to_recall_with_transformers_beyond_orthogonal_embeddings.md)
- [\[CVPR 2025\] Federated Learning with Domain Shift Eraser](federated_learning_with_domain_shift_eraser.md)
- [\[CVPR 2025\] Model Poisoning Attacks to Federated Learning via Multi-Round Consistency](model_poisoning_attacks_to_federated_learning_via_multi-round_consistency.md)
- [\[CVPR 2025\] Mind the Gap: Confidence Discrepancy Can Guide Federated Semi-Supervised Learning](mind_the_gap_confidence_discrepancy_can_guide_federated_semi-supervised_learning.md)
- [\[ICLR 2026\] Πnet: Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers](../../ICLR2026/optimization/pinet_optimizing_hard-constrained_neural_networks_with_orthogonal_projection_lay.md)

</div>

<!-- RELATED:END -->
