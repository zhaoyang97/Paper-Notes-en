---
title: >-
  [Paper Note] Neural Collapse by Design: Learning Class Prototypes on the Hypersphere
description: >-
  [ICML2026][Interpretability][Neural Collapse] Unifies "Classifier Learning (CE)" and "Supervised Contrastive Learning (SCL)" as prototype contrast on the hypersphere. Through two new losses NTCE/NONL (fixing the CE side)…
tags:
  - "ICML2026"
  - "Interpretability"
  - "Neural Collapse"
  - "Hypersphere"
  - "Prototype Contrast"
  - "Normalized Softmax"
  - "Supervised Contrastive Learning"
date: 2026-05-08
content_hash: 066607f9a71dcca9
---

# Neural Collapse by Design: Learning Class Prototypes on the Hypersphere

**Conference**: ICML2026  
**arXiv**: [2605.20302](https://arxiv.org/abs/2605.20302)  
**Code**: https://github.com/pakoromilas/nc_by_design  
**Area**: interpretability  
**Keywords**: Neural Collapse, Hypersphere, Prototype Contrast, Normalized Softmax, Supervised Contrastive Learning

## TL;DR
Unifies "Classifier Learning (CE)" and "Supervised Contrastive Learning (SCL)" as prototype contrast on the hypersphere. Through two new losses NTCE/NONL (fixing the CE side) and a fixed prototype classifier FP (fixing the SCL side), Neural Collapse (NC) is achieved "by design," outperforming prior methods in precision, transfer, long-tail, and robustness.

## Background & Motivation

**Background**: Supervised learning has a theoretical optimum—Neural Collapse (NC): intra-class features collapse to class means (NC1), class means form a simplex ETF (NC2), classifier weights align with class means (NC3), and bias collapses (NC4). The two main paradigms are CE (Cross-Entropy) and SCL (Supervised Contrastive Learning + Linear Probing).

**Limitations of Prior Work**: Theory suggests NC is a global optimum, but in practice, neither path achieves it. CE lacks constraints on radial degrees of freedom (features/weights can be jointly scaled), allowing the loss to be driven to 0 indefinitely by "amplifying logits," which prevents the geometric structure from tightening. SCL approaches NC geometry during pre-training but destroys the learned ETF geometry by subsequently discarding the projection head and retraining a biased linear classifier on unnormalized encoder features (linear probing, LP).

**Key Challenge**: CE fails to constrain the radius from the start; SCL learns the geometry but actively discards it. The failure of both is essentially due to "radial degrees of freedom + bias" disrupting the angular structure on the hypersphere. Meanwhile, Normalized Softmax (NormFace) pushes classification back to the hypersphere, but its negative samples are limited to $K$ (number of class prototypes)—far fewer than required for contrastive learning—and the alignment and uniformity terms are coupled via shared normalization.

**Goal**: (i) Harmonize CE and SCL as "prototype contrast on the hypersphere" via a unified perspective; (ii) fix "insimilar negative samples + alignment-uniformity coupling" on the CE side; (iii) prove class means are already optimal classifiers for SCL to eliminate LP.

**Key Insight**: The author notes that in Normalized Softmax, the classifier weight $\mathbf{w}_c$ is a learnable "class prototype," while in SCL, the class mean $\hat{\bm{\mu}}_c$ is an "emergent class prototype." The difference is merely explicit vs. implicit; both are essentially prototype $\leftrightarrow$ instance contrast on $\mathbb{S}^{d-1}$.

**Core Idea**: For prototype contrast on the hypersphere, CE should flip the anchor from "instance vs. $K$ prototypes" to "prototype vs. $M$ batch instances" (NTCE), and remove same-class positive samples from the normalization to decouple alignment-uniformity (NONL). Simultaneously, SCL does not need to train a classifier; directly using the class mean as weights (FP) suffices.

## Method

### Overall Architecture
Three modifications are proposed across two paths:

- Input: Labeled images $\mathbf{x}_i$, encoder $f_{\bm{\theta}}$, optional projection head $g_{\bm{\phi}}: \mathcal{Z}\to\mathbb{S}^{d-1}$.
- CL Path (NTCE / NONL): Normalize both features $\bm{u}_i=\mathbf{z}_i/\|\mathbf{z}_i\|$ and weights $\hat{\bm{w}}_c$ to $\mathbb{S}^{d-1}$, then replace the loss with a contrastive form where "class prototype $\hat{\bm{w}}_{y_i}$ is the anchor and $M$ batch instances are contrastive objects."
- SCL Path (FP): Train the encoder + projection head with SCL as usual, but **do not** train LP. Directly let $\bm{w}_c=\hat{\bm{\mu}}_c$ (in-batch class mean) serve as classifier weights.
- Theory: Theorem 4.1 proves global optima for NormFace/NTCE/NONL satisfy NC1–NC3; Theorem 4.2 proves SCL and "Prototype Softmax" loss share the same set of global optima, confirming class means are the classifiers SCL naturally seeks.

### Key Designs

1. **NTCE: Flipping Anchors to Class Prototypes to Expand Negative Samples**:
    - **Function**: Based on the normalized Softmax of NormFace, it switches "each instance vs. $K$ class weights" to "each class weight vs. $M$ batch instances," jumping the effective negative samples from $K$ to $M$.
    - **Mechanism**: $L_{\text{NTCE}}=\frac{1}{M}\sum_i -\log\frac{e^{\hat{\bm{w}}_{y_i}^\top \bm{u}_i/\tau}}{\sum_{j=1}^{M} e^{\hat{\bm{w}}_{y_i}^\top \bm{u}_j/\tau}}$. The anchor is the prototype $\hat{\bm{w}}_{y_i}$ and the denominator iterates over the whole batch instead of $K$ classes, breaking the bottleneck of NormFace as a "$K$-way contrast."
    - **Design Motivation**: Contrastive objectives require many negative samples to approach the true expectation. NTCE allows CL to truly benefit from contrastive optimization, empirically pushing the ETF geometry faster (inter-class effective rank reaches the theoretical limit $K-1$ in Table 3).

2. **NONL: Excluding Same-class Samples to Decouple Alignment-Uniformity**:
    - **Function**: Removes all samples in the same class as the anchor from the denominator in NTCE, eliminating gradient conflict caused by "positive samples treated as negatives."
    - **Mechanism**: $L_{\text{NONL}}=\frac{1}{M}\sum_i -\log\frac{e^{\hat{\bm{w}}_{y_i}^\top \bm{u}_i/\tau}}{\sum_{j\notin\mathcal{C}(i)} e^{\hat{\bm{w}}_{y_i}^\top \bm{u}_j/\tau}}$, where $\mathcal{C}(i)$ is the same-class index set in the batch. By explicitly excluding same-class samples, it eliminates the "pushing same-class instances away from the prototype" effect in the denominator.
    - **Design Motivation**: Uniformity is optimized when same-class distances are maximized, which conflicts with alignment. NONL severs this internal friction, significantly improving NC1 (intra-class collapse).

3. **FP (Fixed Prototypes): Using Class Means as SCL Classifiers to Eliminate LP**:
    - **Function**: Skips LP after SCL training and directly uses the sample mean $\hat{\bm{\mu}}_c$ of each class in the projection space as weights, with zero bias and zero extra training.
    - **Mechanism**: Under unit-norm features + balanced labels, SCL loss $L_{\text{SCL}}$ and "Prototype Softmax" loss $L_{\text{proto}}$ share global optima. This means SCL implicitly optimizes a "prototype classifier with class means as weights" throughout its trajectory.
    - **Design Motivation**: Standard LP reintroduces radial freedom and bias, destroying learned ETF geometry and consuming $T$ extra epochs. FP replaces $T \times N$ training with $N$ forward passes and strictly enforces NC3.

### Loss & Training
Temperature $\tau$ follows standard contrastive learning. CL side requires L2 normalization for $\mathbf{z}$ and $\mathbf{w}$ and zero bias. SCL side uses standard recipes, with the only change being replacing the LP phase with "calculate class means $\to$ use directly as weights."

## Key Experimental Results

### Main Results
Four benchmarks: CIFAR-10/100, ImageNet-100/1K; ResNet18 for CIFAR, ResNet50 for ImageNet.

| Dataset | Metric | NONL (Ours) | NTCE (Ours) | NormFace | CE |
|---|---|---|---|---|---|
| CIFAR-10 | Top-1 | **94.9** | 94.7 | 94.8 | 94.6 |
| CIFAR-100 | Top-1 | **73.6** | 72.9 | 72.4 | 72.1 |
| ImageNet-100 | Top-1 | **84.9** | 84.7 | 84.4 | 84.4 |
| ImageNet-1K | Top-1 | 76.5 | **76.7** | 76.4 | 75.4 |

For SCL: FP improves by **+2.0%** over LP on ImageNet-100 (86.8 vs 84.8), while matching LP elsewhere with significantly reduced computation.

### Key Finding
- NONL achieves the cleanest NC geometry in the CL family by decoupling alignment and uniformity; intra-class effective rank is halved compared to NTCE.
- "SCL + LP" has tight NC1 but completely fails at NC3 (weight alignment 0.99/1.03), validating the claim that LP destroys geometry.
- For downstream transfer (8 datasets), NONL reaches 70.7% vs CE 67.0% (**+5.5%** relative gain).
- On CIFAR-10-LT (long-tail), NONL improves CE from 70.2 to 76.3 (**+8.7%** relative gain).

## Highlights & Insights
- **Unified Perspective, Dual Diagnosis**: Viewing CE and SCL as hyperspherical prototype contrast turns two unrelated fixes into two halves of the same picture, closed theoretically by $\bm{w}_c=\hat{\bm{\mu}}_c$.
- **Anchor Flipping + Same-class Exclusion**: A simple code change (changing Softmax "dimension" from $K$ to $M$) yields massive NC improvements and downstream gains.
- **Eliminating LP**: Provides a tangible engineering optimization for SCL deployment, saving hours of training time.
- **Utility of NC**: Demonstrates that "tighter geometry $\to$ better downstream performance," addressing whether NC has practical value.

## Limitations & Future Work
- Contrastive objectives depend on large batches; NTCE/NONL degrade under small-batch training on ImageNet-1K.
- Primarily validated on ResNet; behaviors on Transformers or multi-label scenarios are not yet covered.
- Long-tail analysis shows empirical gains, but the interaction with theoretical "minority collapse" requires deeper geometric analysis.

## Related Work & Insights
- **vs NormFace**: NormFace is the direct predecessor. This work interprets it as "$K$-way contrast with instance as anchor" and upgrades it by flipping the anchor and expanding negatives.
- **vs ETF + DR**: ETF + DR forces a fixed ETF, which crashes transfer performance (mean 54.6 vs NONL 70.7). Allowing natural convergence via correct contrastive forms is more robust.
- **vs Graf et al.**: Strengthens previous SCL global optima proofs to include equivalence with Prototype Softmax, justifying FP on non-optimal practical training points.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ICML 2026\] ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation](shapleig_bayesian_experimental_design_for_shapley_value_estimation.md)
- [\[ACL 2026\] NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning](../../ACL2026/interpretability/nose_neural_olfactory-semantic_embedding_with_tri-modal_orthogonal_contrastive_l.md)
- [\[ICLR 2026\] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](../../ICLR2026/interpretability/seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)
- [\[ICML 2026\] Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression](breaking_the_simplification_bottleneck_in_amortized_neural_symbolic_regression.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ICML 2026\] ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation](shapleig_bayesian_experimental_design_for_shapley_value_estimation.md)
- [\[ACL 2026\] NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning](../../ACL2026/interpretability/nose_neural_olfactory-semantic_embedding_with_tri-modal_orthogonal_contrastive_l.md)
- [\[ICML 2025\] Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups](../../ICML2025/interpretability/sum-of-parts_self-attributing_neural_networks_with_end-to-end_learning_of_featur.md)
- [\[ICML 2026\] Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression](breaking_the_simplification_bottleneck_in_amortized_neural_symbolic_regression.md)

</div>

<!-- RELATED:END -->
