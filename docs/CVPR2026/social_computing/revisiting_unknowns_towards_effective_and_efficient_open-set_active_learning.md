---
title: >-
  [Paper Note] Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning
description: >-
  [CVPR2026][Social Computing][open-set active learning] This paper proposes E2OAL, a detector-free open-set active learning framework that discovers latent structures among unknown classes via label-guided clustering…
tags:
  - "CVPR2026"
  - "Social Computing"
  - "open-set active learning"
  - "Dirichlet calibration"
  - "unknown class exploitation"
  - "adaptive querying"
  - "detector-free"
date: 2026-05-08
content_hash: 6734286f1b5f2a43
---

# Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning

**Conference**: CVPR2026  
**arXiv**: [2603.07898](https://arxiv.org/abs/2603.07898)  
**Code**: [github.com/chenchenzong/E2OAL](https://github.com/chenchenzong/E2OAL)  
**Area**: Social Computing  
**Keywords**: open-set active learning, Dirichlet calibration, unknown class exploitation, adaptive querying, detector-free

## TL;DR

This paper proposes E2OAL, a detector-free open-set active learning framework that discovers latent structures among unknown classes via label-guided clustering, jointly models known and unknown categories through a Dirichlet calibration auxiliary head, and introduces a two-stage adaptive querying strategy. E2OAL simultaneously achieves high accuracy, high query purity, and high training efficiency across multiple benchmarks.

## Background & Motivation

1. **The closed-set assumption in active learning is unrealistic**: Conventional active learning assumes all samples in the unlabeled pool belong to known classes, yet in safety-critical applications such as autonomous driving and medical diagnosis, unlabeled data frequently contain unseen categories.
2. **Unknown samples contaminate the query set**: Standard AL strategies based on uncertainty or diversity tend to misidentify unknown-class samples as highly informative and oversample them, substantially degrading learning efficiency.
3. **Existing OSAL methods rely on independently trained detectors**: Methods such as LfOSA, MQNet, EOAL, BUAL, and EAOA require separate OOD detection networks, introducing significant computational overhead.
4. **Labeled unknown samples are wasted**: Prior methods overlook the supervisory value embedded in samples annotated as "unknown" and fail to incorporate this feedback into known-class learning.
5. **Latent structure exists within unknown classes**: A pilot study demonstrates that training with the true labels of unknown samples—preserving their intra-class structure—yields better performance than collapsing them into a single "unknown" category.
6. **Overconfidence of softmax**: Standard softmax is shift-invariant and produces misleadingly high confidence on semantically ambiguous or anomalous inputs, undermining confidence estimation under open-set conditions.

## Method

### Overall Architecture

E2OAL adopts a unified detector-free two-stage pipeline:
- **Stage 1: Adaptive Class Estimation + Calibration-Aware Training** — Discovers the latent structure of unknown classes in a frozen contrastive feature space and enhances model training via Dirichlet auxiliary supervision.
- **Stage 2: Flexible Two-Stage Query Selection** — Constructs a high-purity candidate pool using a purity score, then selects the most informative samples using an informativeness metric.

### Adaptive Class Estimation

- Applies K-Means clustering to all labeled samples using frozen CLIP features (compatible with MoCo/SimCLR as well).
- The candidate number of unknown classes $\hat{u} \in \{k+1, \ldots, \hat{u}_{\max}\}$ is determined via **ternary search** to maximize a structure-aware F1-product objective.
- F1-product is the product of per-class F1 scores, with clusters matched to $k$ known classes plus one unified unknown class via the Hungarian algorithm.
- Underestimation merges known classes; overestimation fragments them—both are automatically penalized by the F1-product.

### Dirichlet Calibration Auxiliary Head

- Introduces a shift-aware softmax: $P(y|x) = \frac{e^{o_y} + \gamma}{\sum_c (e^{o_c} + \gamma)}$, breaking shift invariance.
- Adopts Evidential Deep Learning (EDL): models predicted probabilities as a Dirichlet distribution $\text{Dir}(\boldsymbol{\alpha})$, where $\boldsymbol{\alpha} = g(\boldsymbol{o})/\gamma + 1$.
- The auxiliary head covers $k + \hat{u}$ classes (known + estimated unknown); the main head covers only the $k$ known classes.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{EDL}} = \mathcal{L}_{\text{CE}} + (\mathcal{L}_{\text{NLL}} + \mathcal{L}_{\text{KL}})$$

- $\mathcal{L}_{\text{CE}}$: Cross-entropy loss on the main head, optimized over known classes only.
- $\mathcal{L}_{\text{NLL}}$: Negative log-likelihood on the auxiliary head, encouraging high confidence on correct labels.
- $\mathcal{L}_{\text{KL}}$: Regularizes the Dirichlet distribution of incorrect classes toward a uniform prior, suppressing erroneous evidence.

### Two-Stage Query Strategy

**Purity Score** (Logit-Margin Purity Score):

$$S_{\text{purity}}(x) = \max_{c \in \mathcal{C}_k} o_c - \max_{c \in \mathcal{C}_{\hat{u}}} o_c$$

Measures the degree of evidence separation between known and unknown classes.

**Informativeness Score** (OSAL-specific Informativeness):

$$S_{\text{info}}(x) = \text{JS}(\mathbf{p} \| \mathbf{u}) \cdot \text{JS}(\mathbf{p} \| \mathbf{p}^{\max})$$

Simultaneously suppresses samples that are overly ambiguous (close to uniform) or overly certain (close to one-hot), favoring moderate uncertainty.

**Adaptive Purity Threshold**: A three-component GMM is fitted to the purity score distribution to dynamically adjust the candidate pool size to meet the target query precision $p^*$. The threshold is adaptively calibrated via observed precision feedback:

$$\hat{p}^*_{t+1} = \text{clip}(\hat{p}^*_t + (p^* - \bar{p}^*_t), 0, 1)$$

## Key Experimental Results

### Main Results

Evaluated on CIFAR-10, CIFAR-100, and Tiny-ImageNet using a ResNet-50 backbone, with 10 active learning rounds and 1,500 queries per round.

| Method | CIFAR-10 (30%) | CIFAR-100 (30%) | Tiny-ImageNet (15%) |
|------|:-:|:-:|:-:|
| **E2OAL (Ours)** | **Best** | **Best** | **Best** |
| Ours* (w/o unknown exploitation) | 95.94 | 67.54 | 60.44 |
| EAOA | 95.88 | 67.14 | 57.31 |
| BUAL | 95.04 | 63.73 | 56.09 |
| EOAL | 93.64 | 63.69 | 56.13 |

Even without leveraging labeled unknown samples (Ours*), the querying strategy alone outperforms all baselines, with a margin exceeding 3 percentage points on Tiny-ImageNet.

### Ablation Study

| Variant | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|------|:-:|:-:|:-:|
| **Full E2OAL** | **97.52** | **72.10** | **64.02** |
| w/o ClassExp (unknowns collapsed) | 97.17 | 70.73 | 62.67 |
| $S_{\text{purity}}$ only | 96.73 | 72.00 | 61.93 |
| $S_{\text{info}}$ only | 96.00 | 68.20 | 57.60 |

- Dirichlet calibration (EDL) yields substantially higher purity than CE: 9495 vs. 9394 known-class queries on CIFAR-10.
- The informativeness metric outperforms EAOA: 65.73 vs. 61.95 on CIFAR-100.
- Performance is insensitive to the target precision $p^*$: variation across $p^* \in \{0.4, 0.5, 0.6, 0.7\}$ is marginal.

### Training Efficiency

The effective training time of E2OAL is comparable to lightweight baselines such as Random, MSP, Coreset, and Uncertainty. Removing the standalone detector incurs only marginal additional cost.

## Highlights & Insights

- **Detector-free design**: No separate OOD detection network is required; unknown class discovery, calibrated training, and query selection are unified within a single framework.
- **Turning waste into value**: E2OAL is the first method to systematically convert labeled unknown samples into effective supervisory signals; the pilot study clearly demonstrates the benefit of preserving intra-class structure among unknowns.
- **Principled calibration**: Dirichlet-based EDL provides theoretically grounded confidence estimation, addressing the overconfidence problem caused by the shift invariance of softmax.
- **Adaptive and hyperparameter-free**: The two-stage querying strategy dynamically adjusts the purity threshold via observed feedback, requiring no additional hyperparameter tuning.
- **Comprehensive experiments**: Three datasets, multiple mismatch ratios, full ablations, efficiency analysis, and sensitivity analysis are covered; code is publicly available.

## Limitations & Future Work

- Validation is limited to image classification; extension to more complex visual tasks such as detection and segmentation remains unexplored.
- Clustering relies on frozen pretrained features (CLIP/MoCo), which may degrade when the pretraining distribution diverges significantly from the target domain.
- The F1-product objective may be overly sensitive to minority classes under highly imbalanced class distributions.
- The three-component GMM assumption regarding the purity score distribution may lack robustness under extreme mismatch ratios.
- Adaptation to online or incremental continual learning settings has not been investigated.

## Related Work & Insights

| Method | Requires Detector | Exploits Labeled Unknowns | Adaptive Precision Control | Calibration Mechanism |
|------|:-:|:-:|:-:|:-:|
| LfOSA | ✓ | ✗ | ✗ | — |
| MQNet | ✓ (meta-net) | ✗ | ✗ | — |
| EOAL | ✓ | ✗ | ✗ | — |
| BUAL | ✓ | ✗ | ✗ | — |
| EAOA | ✓ | ✗ | ✓ (fixed step) | — |
| **E2OAL** | **✗** | **✓** | **✓ (adaptive)** | **Dirichlet EDL** |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of converting labeled unknown samples from "waste" into supervisory signals is original; the combination of Dirichlet calibration and two-stage querying is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets × multiple mismatch ratios × full ablation + efficiency analysis + sensitivity analysis provide comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ — The structure is clear, the pilot study motivates the approach naturally, and the mathematical derivations are coherent.
- **Value**: ⭐⭐⭐⭐ — E2OAL offers a unified and efficient solution for open-set active learning; open-source availability enhances its practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Bias Association Discovery Framework for Open-Ended LLM Generations](../../AAAI2026/social_computing/bias_association_discovery_framework_for_open-ended_llm_generations.md)
- [\[NeurIPS 2025\] Active Slice Discovery in Large Language Models](../../NeurIPS2025/social_computing/active_slice_discovery_in_large_language_models.md)
- [\[ICLR 2026\] Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](../../ICLR2026/social_computing/stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems.md)
- [\[ICLR 2026\] SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](../../ICLR2026/social_computing/sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)
- [\[CVPR 2026\] Learning from Synthetic Data via Provenance-Based Input Gradient Guidance](learning_from_synthetic_data_via_provenance-based_input_gradient_guidance.md)

</div>

<!-- RELATED:END -->
