---
title: >-
  [Paper Note] Neural Collapse by Design: Learning Class Prototypes on the Hypersphere
description: >-
  [ICML2026][Interpretability][Neural Collapse] The paper unifies "Classifier Learning (CE)" and "Supervised Contrastive Learning (SCL)" as prototype contrast on the hypersphere. By introducing two new losses, NTCE/NONL (to fix the CE pathway), and a Fixed Prototype classifier (FP) (to fix the SCL pathway), the authors ensure that Neural Collapse (NC) is achieved "by design." This approach shows comprehensive advantages in accuracy, transferability, long-tail classification…
tags:
  - "ICML2026"
  - "Interpretability"
  - "Neural Collapse"
  - "Hypersphere"
  - "Prototype Contrast"
  - "Normalized Softmax"
  - "Supervised Contrastive Learning"
date: 2026-05-08
content_hash: b31cce0fcd58537a
---

# Neural Collapse by Design: Learning Class Prototypes on the Hypersphere

**Conference**: ICML2026  
**arXiv**: [2605.20302](https://arxiv.org/abs/2605.20302)  
**Code**: https://github.com/pakoromilas/nc_by_design  
**Area**: Interpretability  
**Keywords**: Neural Collapse, Hypersphere, Prototype Contrast, Normalized Softmax, Supervised Contrastive Learning

## TL;DR
The paper unifies "Classifier Learning (CE)" and "Supervised Contrastive Learning (SCL)" as prototype contrast on the hypersphere. By introducing two new losses, NTCE/NONL (to fix the CE pathway), and a Fixed Prototype classifier (FP) (to fix the SCL pathway), the authors ensure that Neural Collapse (NC) is achieved "by design." This approach shows comprehensive advantages in accuracy, transferability, long-tail classification, and robustness.

## Background & Motivation

**Background**: Supervised learning possesses a theoretical optimum known as Neural Collapse (NC): within-class features collapse to class means (NC1), class means form a simplex ETF (NC2), classifier weights align with class means (NC3), and bias collapses (NC4). The two mainstream paradigms are CE (Cross-Entropy) and SCL (Supervised Contrastive Learning + Linear Probing).

**Limitations of Prior Work**: Theoretically, NC is the global optimum, but in practice, neither pathway achieves it. In CE, features and weights can be jointly scaled while predictions remain unchanged (unconstrained radial degrees of freedom); the loss can be pushed toward zero indefinitely by "amplifying logits," preventing the geometric structure from ever tightening. In SCL, although the model approaches NC geometry during pre-training, it subsequently discards the projection head and trains a new biased linear classifier (linear probing, LP) on unnormalized encoder features, thereby destroying the newly learned ETF geometry.

**Key Challenge**: The failure of both paradigms fundamentally stems from "radial degrees of freedom + bias" disrupting the angular structure on the hypersphere. Furthermore, while Normalized Softmax (NormFace) restricts classification to the hypersphere, the number of negative samples is limited to $K$ (the number of class prototypes), which is far fewer than the number of negative samples required for effective contrastive learning. Additionally, the alignment and uniformity terms are coupled via shared normalization.

**Goal**: (i) Characterize both CE and SCL as "prototype contrast on the hypersphere" under a unified perspective; (ii) fix the "insufficient negative samples + alignment-uniformity coupling" flaws in the CE branch; (iii) prove that class means are already the optimal classifiers for SCL, thereby eliminating the need for LP.

**Key Insight**: The authors observe that classifier weights $\mathbf{w}_c$ in Normalized Softmax are learnable "class prototypes," while class means $\hat{\bm{\mu}}_c$ in SCL are "emergent class prototypes." The difference is merely explicit vs. implicit; both are essentially prototype-instance contrasts on $\mathbb{S}^{d-1}$.

**Core Idea**: For prototype contrast on the hypersphere, CE should flip the anchor from "instance vs. $K$ prototypes" to "prototype vs. $M$ batch instances" (NTCE), and decouple alignment-uniformity by removing same-class positive samples from the normalization term (NONL). Simultaneously, SCL does not require retraining a classifier; instead, class means can be directly used as weights (FP).

## Method

### Overall Architecture
The method addresses the discrepancy where NC theory predicts a global optimum that is not achieved in practice. The core strategy is to unify CE and SCL as "prototype ↔ instance contrast" on the hypersphere $\mathbb{S}^{d-1}$ and then perform targeted modifications on each path. For the CE side (NTCE/NONL), features $\bm{u}_i=\mathbf{z}_i/\|\mathbf{z}_i\|$ and classifier weights $\hat{\bm{w}}_c$ are normalized to the hypersphere, biases are set to zero, and the loss is rewritten as "class prototypes as anchors, contrasting against batch instances." For the SCL side (FP), the encoder and projection head are trained as usual, but rather than performing linear probing (LP), the batch class means $\hat{\bm{\mu}}_c$ are directly used as classifier weights. These pathways are closed by two theorems: Theorem 4.1 proves that the global optima for NormFace/NTCE/NONL satisfy NC1–NC3, and Theorem 4.2 proves that SCL and the "Prototype Softmax" loss share the same set of global optima—meaning the class mean is the classifier SCL naturally aims for throughout its optimization trajectory.

### Key Designs

**1. NTCE: Flipping Anchors to Class Prototypes to Expand the Negative Sample Set**

Formally, NormFace is a contrastive loss, but each instance only contrasts against $K$ class weights. The number of negative samples is bottlenecked by the class count $K$. As noted by He et al., contrastive objectives require large negative samples to approximate true expectations; otherwise, inter-class separation converges slowly and is sensitive to small batches. NTCE flips the anchor to the class prototype: $L_{\text{NTCE}}=\frac{1}{M}\sum_i -\log\frac{e^{\hat{\bm{w}}_{y_i}^\top \bm{u}_i/\tau}}{\sum_{j=1}^{M} e^{\hat{\bm{w}}_{y_i}^\top \bm{u}_j/\tau}}$. The anchor is $\hat{\bm{w}}_{y_i}$, and the denominator spans all $M$ instances in a batch rather than $K$ classes. This increases the effective negative sample count from $K$ to $M$, removing the bottleneck of NormFace and allowing the model to benefit from the rule that "more negative samples yield more accurate estimates." Empirically, the ETF geometry is pushed into place faster—Table 3 shows inter-class effective rank reaching the theoretical upper bound of $K-1$.

**2. NONL: Removing Same-Class Positive Samples from Normalization to Decouple Alignment-Uniformity**

In the NTCE denominator, instances $j$ of the same class as the anchor are still present. These generate gradients via $e^{\hat{\bm{w}}_{y_i}^\top \bm{u}_j/\tau}$ that push same-class instances away from the prototype, directly opposing the alignment term in the numerator. This is the known alignment-uniformity coupling: the uniformity term wants same-class instances to spread out, while the alignment term wants them to collapse together. NONL resolves this by removing all same-class samples from the denominator: $L_{\text{NONL}}=\frac{1}{M}\sum_i -\log\frac{e^{\hat{\bm{w}}_{y_i}^\top \bm{u}_i/\tau}}{\sum_{j\notin\mathcal{C}(i)} e^{\hat{\bm{w}}_{y_i}^\top \bm{u}_j/\tau}}$, where $\mathcal{C}(i)$ is the set of indices in the batch belonging to the same class as $i$. Removing this conflict significantly improves NC1 (within-class collapse): Table 3 shows intra-class effective rank dropping from 9.0/12.6 in NTCE to 4.0/11.4, achieving the overall best NC metrics in the contrastive learning family.

**3. FP (Fixed Prototypes): Using Class Means as SCL Classifiers to Eliminate Linear Probing**

Standard SCL requires training an LP with weights and biases on unnormalized features after pre-training. This step reintroduces radial degrees of freedom and bias, destroying the ETF geometry and requiring $T$ additional epochs. FP is justified by Theorem 4.2: under unit-norm features and balanced labels, the SCL loss $L_{\text{SCL}}$ and the "Prototype Softmax" loss $L_{\text{proto}}$ (with numerator $e^{\bm{a}_i^\top \hat{\bm{\mu}}_{y_i}/\tau}$ and a frequency-weighted denominator) share the same set of global optima. This implies SCL implicitly optimizes for a "prototype classifier with class means as weights." FP thus sets $\bm{w}_c=\hat{\bm{\mu}}_c$ with zero bias and zero extra training; inference is a nearest-prototype decision. This replaces $T\times N$ training iterations with $N$ forward passes and geometrically forces NC3 to hold strictly.

### Loss & Training
The temperature $\tau$ follows standard contrastive learning defaults (CIFAR/ImageNet). For the contrastive side, L2 normalization is applied to $\mathbf{z}$ and $\mathbf{w}$ with zero bias. For the SCL side, the standard Khosla et al. recipe is used, with the only change being replacing the LP phase with "calculate class means → assign as weights." Note that NTCE/NONL may degrade with small batches due to insufficient negative samples; large batches are recommended for ImageNet-1K.

## Key Experimental Results

### Main Results
Four benchmarks: CIFAR-10, CIFAR-100, ImageNet-100, ImageNet-1K. ResNet18 for CIFAR; ResNet50 for ImageNet.

| Dataset | Metric | NONL (Ours) | NTCE (Ours) | NormFace | CE |
|---|---|---|---|---|---|
| CIFAR-10 | Top-1 | **94.9** | 94.7 | 94.8 | 94.6 |
| CIFAR-100 | Top-1 | **73.6** | 72.9 | 72.4 | 72.1 |
| ImageNet-100 | Top-1 | **84.9** | 84.7 | 84.4 | 84.4 |
| ImageNet-1K | Top-1 | 76.5 | **76.7** | 76.4 | 75.4 |

For SCL: FP outperforms LP by **+2.0%** on ImageNet-100 (86.8 vs 84.8) and matches LP on the other three datasets, while requiring only $N$ forward passes (compared to $T\times N$ for LP).

### NC Geometry and Convergence Speed (CIFAR-10 / CIFAR-100 training)

| Method | Intra erank ↓ | Inter erank ↑ | Weight align ↓ | Instance align ↓ |
|---|---|---|---|---|
| CE | 22.5 / 96.4 | 8.6 / 57.1 | 0.59 / 0.83 | 0.69 / 1.05 |
| NormFace | 10.5 / 13.6 | 9.0 / 96.2 | 0.12 / 0.01 | 0.14 / 0.06 |
| NTCE | 9.0 / 12.6 | **9.0 / 99.0** | 0.08 / 0.01 | 0.10 / 0.05 |
| **NONL** | **4.0 / 11.4** | **9.0 / 99.0** | 0.11 / 0.01 | 0.16 / 0.06 |
| SCL + LP | 4.5 / 7.5 | 9.0 / 66.7 | 0.99 / 1.03 (NC3 Broken) | 0.10 / 0.34 |

NC metrics for NTCE/NONL reach $\geq 95\%$ of theoretical values and match the final NC metrics of CE using $\leq 7.5\%$ of CE's training iterations (Appendix Table 7).

### Downstream Transfer and Robustness

- **Transfer Learning (Mean over 8 datasets, Table 4)**: NONL 70.7% vs CE 67.0%, an average relative **Gain** of **+5.5%**; on Cars, the gain vs CE is **+47.1%** (38.1 vs 25.9).
- **Long-tail Classification (CIFAR-10/100-LT, Table 6)**: Under extreme imbalance $\tau=0.01$, NONL pushes CE from 70.2 to 76.3 on CIFAR-10-LT; NTCE pushes 37.4 to 39.0 on CIFAR-100-LT; maximum relative **Gain** is **+8.7%**.
- **Robustness (ImageNet-C, Table 5)**: NTCE achieves mCE = 77.6 (CE 80.1), showing higher stability across all corruption types (Noise/Blur/Weather/Digital).

### Key Findings
- NONL produces the cleanest NC geometry in the CL family, primarily by decoupling alignment and uniformity; intra-class effective rank is halved compared to NTCE.
- "SCL + LP" actually achieves tighter NC1 (intra erank 4.5/7.5) than all CL methods, but NC3 completely fails (w-inst 0.99/1.03), confirming the core assertion that LP destroys geometry.
- For information-theoretic metrics MIR/HDR, CE slightly leads, but the authors note this reflects "high total entropy" rather than "good NC structure"—evidence from transfer/long-tail/robustness tasks proves the latter's superiority.

## Highlights & Insights
- **One unified perspective solves two failure diagnoses**: By viewing both CE and SCL as prototype contrast on the hypersphere, two seemingly unrelated fixing strategies (NTCE/NONL for CL; FP for SCL) become two halves of the same picture, theoretically unified by the optimal $\bm{w}_c=\hat{\bm{\mu}}_c$.
- **Anchor flipping + same-class removal are simple one-line changes**: NTCE merely shifts the Softmax "dimension" from $K$ classes to $M$ batch instances; NONL simply removes same-class samples from the denominator. These bring massive NC improvements and downstream gains, offering high value for other normalized contrastive losses (e.g., supervised variants of InfoNCE).
- **Eliminating LP is a tangible engineering optimization for SCL**: For large-scale training, LP usually adds hours; FP eliminates this stage with equal or superior accuracy, making it industry-friendly.
- **NC is more than "pretty geometry"; it is key to downstream performance**: The authors link "tighter geometry → better downstream tasks" through transfer/long-tail/robustness experiments, addressing the long-standing skepticism regarding the practical utility of NC.

## Limitations & Future Work
- Contrastive objectives naturally depend on large batches. NTCE/NONL degrade significantly on ImageNet-1K with small batches (Appendix G.4).
- Primarily validated on ResNet18/50 for standard image classification; Transformer backbones, detection/segmentation tasks, and multi-label scenarios were not covered.
- While empirical long-tail gains are shown, NC2/ETF theory suggests "minority collapse" under imbalance (Thrampoulidis 2022). Whether NONL/NTCE approach ETF or a "weighted ETF" requires deeper geometric analysis.
- FP only matches rather than exceeds LP on ImageNet-1K, likely due to projection head dimensionality or class mean estimation noise. Future work could investigate EMA class means or multi-batch accumulation.

## Related Work & Insights
- **vs. NormFace (Wang et al. 2017)**: NormFace is the direct predecessor. This paper interprets it as "$K$-way contrast with instance anchors"; NTCE flips anchors and expands negative samples, while NONL further decouples alignment-uniformity.
- **vs. ETF + DR (Yang et al. 2022)**: ETF + DR fixes the classifier as a preset ETF and only trains the backbone. Ours allows the classifier to learn freely, letting it "naturally converge to ETF" through normalization and correct contrastive forms. Table 1/4 shows ETF + DR transfer performance collapses (mean 54.6 vs NONL 70.7), suggesting fixed geometry sacrifices representation transferability.
- **vs. Hyperspherical / EBV Prototypes (Mettes 2019; Shen 2023)**: These methods pre-construct prototypes. Ours proves the SCL class mean prototype is optimal throughout optimization, providing a theoretical foundation for "emergent prototypes" to replace "prescribed prototypes."
- **vs. Graf et al. 2021**: Graf et al. only proved ETF emergence at the global optimum of SCL. Theorem 4.2 strengthens this to "SCL global optimum set = Prototype Softmax global optimum set," legitimizing the use of FP even at non-optimal training points.
- **vs. Kini et al. 2024**: Kini et al. analyzed SCL optimality under specific architectural assumptions. This paper provides a broader equivalence conclusion under UFM/LPM settings and unifies CL and SCL.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified perspective of spherical prototype contrast is clear, and the modifications (NTCE/NONL/FP) are simple yet impactful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evidence chain across datasets, NC geometry, speed, transfer, long-tail, and robustness.
- Writing Quality: ⭐⭐⭐⭐ Coherent flow, with theory and empirical findings well-integrated; some sections have high formula density.
- Value: ⭐⭐⭐⭐ High theoretical and engineering value: performance gains for CL and training efficiency for SCL, with simple implementation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Patronus: Interpretable Diffusion Models with Prototypes](../../ICLR2026/interpretability/patronus_interpretable_diffusion_models_with_prototypes.md)
- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ICML 2026\] ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation](shapleig_bayesian_experimental_design_for_shapley_value_estimation.md)
- [\[ICLR 2026\] Multi-ReduNet: Interpretable Class-Wise Decomposition of ReduNet](../../ICLR2026/interpretability/multi-redunet_interpretable_class-wise_decomposition_of_redunet.md)
- [\[ICLR 2026\] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](../../ICLR2026/interpretability/seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)

</div>

<!-- RELATED:END -->
