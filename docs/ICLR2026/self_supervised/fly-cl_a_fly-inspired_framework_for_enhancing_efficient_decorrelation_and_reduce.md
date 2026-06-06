---
title: >-
  [Paper Note] Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning
description: >-
  [ICLR 2026][Self-Supervised Learning][continual learning] Inspired by the Drosophila olfactory circuit, Fly-CL is proposed as a framework that achieves progressive decorrelation through three stages — sparse random proje…
tags:
  - "ICLR 2026"
  - "Self-Supervised Learning"
  - "continual learning"
  - "fly olfactory circuit"
  - "decorrelation"
  - "representation learning"
  - "prototype"
date: 2026-05-08
content_hash: 7b1fe955076ac7ff
---

# Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning

**Conference**: ICLR 2026
**arXiv**: [2510.16877](https://arxiv.org/abs/2510.16877)  
**Code**: [GitHub](https://github.com/gfyddha/Fly-CL)  
**Area**: Self-Supervised Learning / Continual Learning / Bio-Inspired Computing
**Keywords**: continual learning, fly olfactory circuit, decorrelation, representation learning, prototype

## TL;DR
Inspired by the Drosophila olfactory circuit, Fly-CL is proposed as a framework that achieves progressive decorrelation through three stages — sparse random projection, top-$k$ activation, and streaming ridge classification — significantly reducing training time while attaining state-of-the-art performance in pre-trained model-based continual learning.

## Background & Motivation

**Background**: Continual learning (CL) methods based on frozen pre-trained models reformulate parameter updates as similarity matching problems, performing classification via cosine similarity between class prototypes. Three main paradigms exist: prompt/adapter-based, mixture-of-models, and representation-based approaches.

**Limitations of Prior Work**: Representation-based methods directly compute class prototypes from frozen pre-trained features; however, severe **multicollinearity** among features (high inter-prototype correlation) degrades the discriminability of cosine similarity. Existing solutions (e.g., matrix inversion in RanPAC) incur high computational cost ($\mathcal{O}(lm^3)$), making them unsuitable for low-latency scenarios.

**Key Challenge**: Decorrelation is critical for classification accuracy, yet efficient decorrelation methods are lacking.

**Goal**: Design a computationally efficient and effective decorrelation framework.

**Key Insight**: Inspiration is drawn from the Drosophila olfactory circuit — the sparse expansive projection from projection neurons (PN) to Kenyon cells (KC), followed by dimensionality reduction from KC to mushroom body output neurons (MBON), constitutes an efficient decorrelation mechanism.

**Core Idea**: Emulate the three stages of the Drosophila olfactory system — sparse random expansive projection, top-$k$ activation sparsification, and streaming ridge regression classification — to achieve progressive decorrelation.

## Method

### Overall Architecture
A frozen pre-trained model extracts $d$-dimensional features → L2 normalization → sparse random projection into an $m$-dimensional high-dimensional space ($m \gg d$) → top-$k$ activation retaining the strongest dimensions → streaming ridge classifier for final prediction.

### Key Designs

1. **Sparse Random Projection + Top-$k$ Operation (PN→KC)**:

    - **Function**: Decorrelates low-dimensional features.
    - **Mechanism**: A fixed sparse matrix $\mathbf{W} \in \mathbb{R}^{m \times d}$ (each row containing only $p$ non-zero values sampled from $\mathcal{N}(0,1)$) is used for high-dimensional projection, followed by top-$k$ selection retaining the $k$ largest components.
    - **Theoretical Guarantee**: Theorem 4.1 proves that the sparse matrix maintains full column rank with probability $1-o(1)$; Theorem 4.2 proves that performance degradation under top-$k$ is bounded, with error decaying polynomially when $k=\Omega(m^\alpha)$.
    - **Computational Efficiency**: Complexity is reduced from $\mathcal{O}(mnd)$ to $\mathcal{O}(mnp)$.

2. **Streaming Ridge Classification (KC→MBON)**:

    - **Function**: Learns decorrelated classification weights in the high-dimensional space.
    - **Mechanism**: The Gram matrix $\mathbf{G}_t$ and cross-statistics $\mathbf{S}_t$ are maintained incrementally, yielding the classifier $\mathbf{C}_t = (\mathbf{G}_t + \lambda\mathbf{I}_m)^{-1}\mathbf{S}_t$. $\ell_2$ regularization mitigates multicollinearity.
    - **Adaptive Regularization**: The optimal $\lambda$ is selected automatically via Generalized Cross-Validation (GCV), reducing complexity from $\mathcal{O}(lm^3)$ to $\mathcal{O}(n_t^2 m)$.
    - Cholesky decomposition is used to accelerate the linear solve.

3. **Biological Correspondence**:

    - PN→KC: Sparse expansive projection + winner-take-all inhibition ≡ sparse random projection + top-$k$.
    - KC→MBON: Hebbian learning ≈ ridge classification (equivalence proved in Section 6).

### Loss & Training
Streaming incremental updates are employed; each new task requires only updating the Gram matrix and cross-statistics, with no need to store historical data.

## Key Experimental Results

### Main Results: Class Incremental Learning

| Method | CIFAR-100 (10 steps) | ImageNet-R (10 steps) | Training Time↓ |
|--------|----------------------|-----------------------|----------------|
| SimpleCIL | 70.8 | 71.4 | Baseline |
| RanPAC | 76.4 | 78.6 | Slow |
| Fly-CL | **76.5** | **78.8** | **Several times faster** |

### Ablation Study

| Configuration | Effect | Note |
|---------------|--------|------|
| Without random projection | Degraded | Multicollinearity unresolved |
| Without top-$k$ | Degraded | Noisy dimensions interfere |
| Dense projection instead of sparse | Comparable but slower | Sparsity does not lose information |
| Fixed $\lambda$ instead of adaptive GCV | Degraded | Task heterogeneity requires adaptation |

### Key Findings
- Pearson correlation heatmaps clearly demonstrate the progressive decorrelation effect across three stages.
- Training time is substantially reduced, with performance matching or surpassing the strongest baseline.
- The framework generalizes across different pre-trained backbones.

## Highlights & Insights
- The biological inspiration is highly creative — the three-stage decorrelation mechanism of the Drosophila olfactory circuit maps perfectly onto the multicollinearity problem in CL.
- Theoretical analysis is rigorous — two theorems separately establish the information-preserving property of sparse projection and the degradation bound under top-$k$.
- Strong practicality — significant computational efficiency gains make the framework suitable for edge computing and real-time scenarios.

## Limitations & Future Work
- The value of $k$ in top-$k$ requires tuning.
- The projection matrix is fixed and random; adaptive learning may yield further improvements.
- Evaluation is limited to image classification; other modalities remain to be explored.

## Related Work & Insights
- **vs. RanPAC**: Also employs random projection but at high computational cost; Fly-CL substantially reduces complexity via sparsity and GCV.
- **vs. Prompt/Adapter Methods**: Does not rely on specific architectures, offering greater generality.
- **vs. Fly-inspired LSH (Dasgupta et al., 2017)**: The classic fly-inspired work targets hashing; this paper extends the paradigm to continual learning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Organic integration of biological inspiration, theoretical analysis, and a practical framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset, multi-backbone evaluation with comprehensive ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Biological analogy clearly articulated; theoretical derivations complete.
- **Value**: ⭐⭐⭐⭐ Provides an efficient and principled solution for CL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models](gradient-sign_masking_for_task_vector_transport_across_pre-trained_models.md)
- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](../../CVPR2026/self_supervised/com_pt_chain_of_models_pretraining.md)
- [\[ICML 2026\] NITP: Next Implicit Token Prediction for LLM Pre-training](../../ICML2026/self_supervised/nitp_next_implicit_token_prediction_for_llm_pre-training.md)
- [\[ICLR 2026\] Enhancing Molecular Property Predictions by Learning from Bond Modelling and Interactions](enhancing_molecular_property_predictions_by_learning_from_bond_modelling_and_int.md)
- [\[NeurIPS 2025\] Continuous Subspace Optimization for Continual Learning (CoSO)](../../NeurIPS2025/self_supervised/continuous_subspace_optimization_for_continual_learning.md)

</div>

<!-- RELATED:END -->
