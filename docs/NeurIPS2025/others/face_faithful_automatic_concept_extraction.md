---
title: >-
  [Paper Note] FACE: Faithful Automatic Concept Extraction
description: >-
  [NeurIPS 2025][concept explanation] This paper proposes FACE, a framework that incorporates a KL divergence regularization term into non-negative matrix factorization (NMF) to constrain reconstructed activations to remain consistent with the original model's predictions, thereby extracting concept explanations that are truly faithful to the model's decision process. FACE comprehensively outperforms CRAFT and ICE on ImageNet, COCO, and CelebA.
tags:
  - NeurIPS 2025
  - concept explanation
  - NMF
  - KL divergence
  - faithfulness
  - non-negative matrix factorization
  - interpretability
date: 2026-05-08
content_hash: f4a05fe8e66a7c84
---

# FACE: Faithful Automatic Concept Extraction

**Conference**: NeurIPS 2025
**arXiv**: [2510.11675](https://arxiv.org/abs/2510.11675)
**Code**: [GitHub](https://github.com/dipkamal/FACE)
**Area**: Explainable AI / Concept Discovery
**Keywords**: concept explanation, NMF, KL divergence, faithfulness, non-negative matrix factorization, interpretability

## TL;DR
This paper proposes FACE, a framework that incorporates a KL divergence regularization term into non-negative matrix factorization (NMF) to constrain reconstructed activations to remain consistent with the original model's predictions, thereby extracting concept explanations that are truly faithful to the model's decision process. FACE comprehensively outperforms CRAFT and ICE on ImageNet, COCO, and CelebA.

## Background & Motivation

**Background**: Concept-based explanation methods (e.g., TCAV, ACE, CRAFT, ICE) aim to explain model decisions using human-interpretable high-level concepts (e.g., "fur," "ears"), offering greater intuitiveness than pixel-level attribution.

**Limitations of Prior Work**: (a) TCAV requires manually annotated concept datasets and scales poorly; (b) unsupervised methods such as ACE/ICE/CRAFT discover concepts via clustering or NMF over encoder activations, but focus solely on reconstruction error while ignoring classifier behavior; (c) standard NMF tends to capture high-variance directions rather than class-discriminative ones, potentially causing reconstructed activations to yield entirely different predictions.

**Key Challenge**: A small reconstruction error $\|\mathbf{A} - \mathbf{UW}^\top\|_F^2$ does not imply prediction fidelity. Even when the error in activation space is minimal, the nonlinear amplification through the classifier head $h$ and softmax can cause the predicted distribution to deviate substantially from the original.

**Goal**: Introduce a KL divergence regularization term into the NMF objective to directly constrain consistency between the classifier output distributions of the original and reconstructed activations.

## Method

### Framework Setup
The classifier $f$ is decomposed into an encoder $g: \mathcal{X} \to \mathcal{G}$ and a classification head $h: \mathcal{G} \to \mathcal{Y}$, with $f(\mathbf{x}) = h(g(\mathbf{x}))$. The activation matrix $\mathbf{A} = g(\mathbf{X}) \in \mathbb{R}^{n \times p}_{+}$ for $n$ samples is factorized into $\mathbf{U} \in \mathbb{R}^{n \times r}_{+}$ and $\mathbf{W} \in \mathbb{R}^{p \times r}_{+}$.

### Core Objective
Standard NMF minimizes reconstruction error only:

$$\min_{\mathbf{U}\geq 0, \mathbf{W}\geq 0} \frac{1}{2}\|\mathbf{A} - \mathbf{UW}^\top\|_F^2$$

FACE augments this with a KL divergence regularization term:

$$\min_{\mathbf{U}\geq 0, \mathbf{W}\geq 0} \frac{1}{2}\|\mathbf{A} - \mathbf{UW}^\top\|_F^2 + \lambda \cdot \text{KL}(h(\mathbf{A}) \| h(\mathbf{UW}^\top))$$

where $\lambda > 0$ controls the trade-off between reconstruction fidelity and prediction alignment. KL divergence is computed over softmax-normalized logits.

### Theoretical Guarantee
By leveraging Pinsker's inequality, when the KL divergence is bounded by $\varepsilon$, the total variation distance between predicted distributions is bounded:

$$\|p - q\|_1 \leq \sqrt{2 \cdot \text{KL}(p \| q)} \leq \sqrt{2\varepsilon}$$

This implies that KL regularization directly controls the prediction deviation introduced by concept substitution—a guarantee that minimizing reconstruction error alone cannot provide.

### Local Linearity
KL regularization ensures that the nonlinearity of softmax near $\mathbf{UW}^\top$ becomes negligible, rendering the effect of concept-space operations (e.g., deletion/insertion) on predictions approximately linear and predictable.

### Optimization
- Projected gradient descent is used to alternately update $\mathbf{U}$ and $\mathbf{W}$, with projection onto the non-negativity constraint.
- Initialization is performed via NNDSVD (Non-negative Double SVD).
- Concept importance is quantified using Sobol indices.

### Evaluation Metrics
- **C-Del (Concept Deletion)**: Area under the accuracy drop curve as the most important concepts are progressively removed; higher indicates greater faithfulness.
- **C-Ins (Concept Insertion)**: Rate of accuracy recovery as the most important concepts are progressively inserted; higher indicates greater faithfulness.
- **C-Gini (Gini Sparsity)**: Sparsity of the concept importance distribution; higher indicates more concise explanations.

## Key Experimental Results

### Decomposition Quality (ResNet-34)

| Dataset | Method | MSE ↓ | $D_\text{KL}$ ↓ |
|--------|------|-------|---------|
| ImageNet | ICE | 0.296 | 0.359 |
| ImageNet | CRAFT | 0.451 | 0.240 |
| ImageNet | **FACE** | 0.497 | **0.220** |
| COCO | ICE | 0.308 | 0.596 |
| COCO | CRAFT | 0.457 | 0.600 |
| COCO | **FACE** | 0.462 | **0.458** |
| CelebA | ICE | 0.148 | 0.212 |
| CelebA | CRAFT | 0.498 | 0.110 |
| CelebA | **FACE** | 0.375 | **0.021** |

### Faithfulness and Complexity (ResNet-34)

| Dataset | Method | C-Ins ↑ | C-Del ↑ | C-Gini ↑ |
|--------|------|---------|---------|----------|
| ImageNet | ICE | 0.908 | 0.484 | 0.537 |
| ImageNet | CRAFT | 0.932 | 0.752 | 0.835 |
| ImageNet | **FACE** | **0.969** | **0.891** | **0.895** |
| COCO | ICE | 0.883 | 0.632 | 0.623 |
| COCO | CRAFT | 0.861 | 0.691 | 0.874 |
| COCO | **FACE** | **0.971** | **0.894** | **0.947** |
| CelebA | ICE | 0.910 | 0.365 | 0.662 |
| CelebA | CRAFT | 0.953 | 0.604 | 0.901 |
| CelebA | **FACE** | **0.971** | **0.635** | **0.928** |

### Post-Reconstruction Prediction Accuracy
FACE maintains 100% top-1 accuracy after reconstruction across all classes, whereas CRAFT achieves only 40% on the "Train" class.

### Ablation Study (Effect of λ)
- On ImageNet/COCO, even small values of λ (e.g., $10^{-5}$) yield notable faithfulness improvements, while excessively large values ($\geq 10^3$) degrade performance.
- On CelebA (4 classes), larger λ values (up to $10^5$) are tolerable due to the lower-dimensional distribution being easier to align.
- Performance saturates at decomposition rank $r=25$.

## Highlights & Insights
- **Core Innovation**: FACE is the first to introduce prediction alignment constraints into NMF-based concept discovery, formalizing and addressing the problem of explanations that appear plausible yet are unfaithful to the model.
- **Theory–Experiment Consistency**: Pinsker's inequality provides quantitative faithfulness guarantees, and experiments confirm that FACE achieves the lowest KL divergence across all datasets and model architectures.
- **Key Insight**: Models may not rely on features that human intuition would suggest—for instance, FACE reveals that rabbit classification depends on "fur" rather than "head"—highlighting that faithful explanations are more important than intuitive ones.
- **Lightweight Computation**: The optimization is dominated by small matrix multiplications and a linear head, making it feasible on low-resource hardware.

## Limitations & Future Work
- The method currently supports only CNN architectures; direct application to Transformer-based architectures such as ViT requires additional adaptation.
- Explanations are class-level and global; instance-level concept discovery is not supported.
- The hyperparameter $\lambda$ requires dataset-specific tuning, and no adaptive selection strategy is provided.
- No human evaluation study is conducted to verify whether KL constraints genuinely improve human understanding.
- Evaluation is limited to ResNet-34 and MobileNetV2; larger-scale models remain untested.

## Related Work & Insights
- **vs. TCAV**: TCAV requires manually annotated concept datasets; FACE discovers concepts in a fully automated manner.
- **vs. ACE**: ACE's superpixel-based clustering is prone to artifacts; FACE operates in a more continuous NMF space.
- **vs. CRAFT**: CRAFT optimizes reconstruction error only, so discovered concepts may be inconsistent with model decisions; FACE enforces faithfulness via KL regularization.
- **vs. ICE**: ICE applies NMF at the convolutional filter level, capturing only local concepts; FACE operates on the penultimate layer to capture higher-level semantics.
- **vs. CRP/RelMax**: These backpropagation-based methods do not perform activation decomposition; FACE's decomposition-based approach enables quantifiable concept importance.

## Rating
- Novelty: ⭐⭐⭐⭐ KL-regularized NMF is a concise and effective idea with complete theoretical grounding.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across 3 datasets × 2 models with comprehensive ablations over λ and rank $r$, using multiple metrics.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and theoretical derivations are rigorous.
- Value: ⭐⭐⭐⭐ Addresses a foundational issue in concept explanation faithfulness with broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Faithful Group Shapley Value](faithful_group_shapley_value.md)
- [\[NeurIPS 2025\] Obliviator Reveals the Cost of Nonlinear Guardedness in Concept Erasure](obliviator_reveals_the_cost_of_nonlinear_guardedness_in_concept_erasure.md)
- [\[AAAI 2026\] On the Variability of Concept Activation Vectors](../../AAAI2026/others/on_the_variability_of_concept_activation_vectors.md)
- [\[AAAI 2026\] Autonomous Concept Drift Threshold Determination](../../AAAI2026/others/autonomous_concept_drift_threshold_determination.md)
- [\[AAAI 2026\] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer](../../AAAI2026/others/enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)

</div>

<!-- RELATED:END -->
