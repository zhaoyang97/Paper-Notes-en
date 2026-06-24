---
title: >-
  [Paper Note] Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning
description: >-
  [ICLR 2026][PU Learning] The authors propose NcPU, a non-contrastive PU learning framework. By applying a square root transformation to the standard non-contrastive loss (NoiSNCL), gradients are dominated by clean pairs. Combined with PhantomGate for conservative negative supervision and regret-based recovery, the two modules interact iteratively within an EM framework. Without relying on auxiliary negative samples or estimated class priors…
tags:
  - "ICLR 2026"
  - "PU Learning"
  - "Non-contrastive Representation Learning"
  - "Noisy-pair Robustness"
  - "Pseudo-label Disambiguation"
  - "EM Framework"
date: 2026-05-08
content_hash: 09d7978f96ae2d46
---

# Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning

**Conference**: ICLR 2026  
**arXiv**: [2510.01278](https://arxiv.org/abs/2510.01278)  
**Code**: [https://github.com/Hengwei-Zhao96/NcPU](https://github.com/Hengwei-Zhao96/NcPU)  
**Area**: Others / Weakly Supervised Learning  
**Keywords**: PU Learning, Non-contrastive Representation Learning, Noisy-pair Robustness, Pseudo-label Disambiguation, EM Framework

## TL;DR

The authors propose NcPU, a non-contrastive PU learning framework. By applying a square root transformation to the standard non-contrastive loss (NoiSNCL), gradients are dominated by clean pairs. Combined with PhantomGate for conservative negative supervision and regret-based recovery, the two modules interact iteratively within an EM framework. Without relying on auxiliary negative samples or estimated class priors, the gap between PU and supervised learning on CIFAR-100 is reduced from 14.26% to <1.4%, achieving SOTA on the xBD disaster damage assessment dataset as well.

## Background & Motivation

**Background**: Positive-Unlabeled (PU) learning involves a small set of labeled positive samples and a large volume of unlabeled data to train a binary classifier. Typical applications include identifying damaged buildings in post-disaster remote sensing where only some damaged structures are labeled, product recommendations where only clicks are recorded without explicit "dislike" labels, and medical diagnosis with confirmed cases but a lack of explicit negative specimens. Mainstream methods are categorized into risk estimation (nnPU, uPU), label disambiguation (DistPU), and auxiliary negative sample selection (LaGAM).

**Limitations of Prior Work**: Even state-of-the-art PU methods exhibit a significant gap compared to fully supervised learning on complex datasets. On CIFAR-100, the best non-auxiliary method achieves only 76.49% OA, whereas supervised learning reaches 89.65%. Through t-SNE visualization, the authors demonstrate that the root cause lies in the feature space: methods like LaGAM and HolisticPU show severe overlap between positive and negative distributions, whereas supervised features are clearly separable. This indicates the bottleneck is not classifier design, but the inability to learn discriminative representations from unreliable pseudo-labels.

**Key Challenge**: Representation learning relies on accurate labels to construct similar/dissimilar pairs. In PU scenarios, labels are inherently unreliable. Same-class pairs constructed using pseudo-labels are contaminated with "noisy pairs" (samples belonging to different classes but incorrectly treated as the same). These noisy pairs generate larger gradients under standard contrastive/non-contrastive losses, dominating the training process. This creates a vicious cycle: poor representations $\rightarrow$ poor pseudo-labels $\rightarrow$ more noisy pairs $\rightarrow$ worse representations.

**Key Insight**: The authors start from two observations. First, non-contrastive learning (which pulls similar samples together without explicitly pushing different ones apart) is naturally more tolerant of noisy labels than contrastive learning because it avoids incorrectly pushing apart samples that should belong to the same class. Second, the gradient of the standard non-contrastive loss $\mathcal{L}_r = 2(1 - \langle \tilde{q}_i, \tilde{k}_j \rangle)$ is proportional to $(1 - \cos^2\theta)$. Consequently, noisy pairs with large distances (low cosine similarity) yield large gradients, while clean pairs with small distances yield small gradients, which is counter-intuitive. By taking the square root, the gradient becomes proportional to $(1 + \cos\theta)$, ensuring clean pairs with small distances provide larger gradients.

**Core Idea**: Apply a square root transform to the standard non-contrastive loss to flip the gradient-distance relationship, allowing clean pairs to dominate training. Integrate this with PhantomGate to provide conservative negative supervision, forming an EM-style iterative framework.

## Method

### Overall Architecture

NcPU is built upon the BYOL non-contrastive learning framework. Inputs consist of a positive set $\mathcal{P}$ and an unlabeled set $\mathcal{U}$. Each sample undergoes random augmentation to produce two views, which are processed by an online network (encoder + projection head + prediction head) and a target network (momentum-updated, no prediction head) to obtain normalized embeddings $\tilde{q}$ and $\tilde{k}$. A classifier $f(\cdot)$ outputs softmax probabilities. The training involves two alternating modules: NoiSNCL utilizes current pseudo-labels for noisy-pair robust intra-class alignment, and PLD (including PhantomGate) uses the aligned representation space to update pseudo-labels. Theoretically, these correspond to the M-step and E-step of the EM algorithm.

### Key Designs

**1. NoiSNCL: Applying sqrt to non-contrastive loss to wrest control from noisy pairs**

The standard supervised non-contrastive loss $\mathcal{L}_r = 2(1 - \langle \tilde{q}_i, \tilde{k}_j \rangle)$ makes representation learning susceptible to noise because pairs with low cosine similarity (noisy pairs) have higher gradients. NoiSNCL modifies this to $\tilde{\mathcal{L}}_r = 2\sqrt{1 - \langle \tilde{q}_i, \tilde{k}_j \rangle}$. This ensures the gradient is proportional to $(1 + \cos\theta)$, flipping the relationship so clean pairs have larger gradients. The intuition comes from the shape of $\sqrt{x}$: its derivative tends to infinity near $x \to 0$ (amplifying gradients for small losses like clean pairs) and tends to 0 near $x \to 1$ (suppressing gradients for large losses like noisy pairs). The overhead is negligible, and stability is maintained by the asymmetric architecture of BYOL.

**2. PhantomGate: Conservative negative supervision with regret mechanism**

PU scenarios lack negative supervision. Relying solely on prototype similarity for disambiguation often leads to a "trivial solution" where all samples are pulled toward the positive class (high recall but low precision). Conversely, hard thresholding for negative selection can be too aggressive (high precision but very low recall). PhantomGate takes a middle ground: first, it updates class-conditional prototypes $\mu_c = \text{Normalize}(\alpha \mu_c + (1-\alpha)\tilde{q})$; second, it generates soft pseudo-labels $s'$ using prototype similarity; third, it applies a gate. When the positive probability $f_1(x) \geq \tau$, the label is set to negative $[0,1]^T$ (conservative negative supervision). Otherwise, it reverts to the prototype-based $s'$. The regret mechanism allows the model to "reconsider" samples previously judged as negative by updating from the accumulated $s'$ rather than sticking to a fixed hard label.

**3. Self-Adaptive Threshold (SAT): Automated scheduling of negative sample selection**

The threshold $\tau$ in PhantomGate is not fixed. SAT maintains a global threshold $\tilde{\tau}$ and category-aware modulation factors $\tilde{\rho}(c)$ using momentum updates: $\tau = \frac{\tilde{\rho}(1)}{\max\{\tilde{\rho}(0), \tilde{\rho}(1)\}} \cdot \tilde{\tau}$. This allows the threshold to evolve: early in training when the model is less confident, $\tilde{\tau}$ is low, providing more negative supervision. As confidence grows, $\tilde{\tau}$ increases, tightening the threshold to filter out potentially inaccurate negative assignments. This follows the curriculum learning philosophy of providing simple, reliable supervision first.

### Loss & Training

The total loss is: $\mathcal{L} = \frac{1}{|\mathcal{P}|}\sum_{x_i \in \mathcal{P}} \mathcal{L}_c + \frac{1}{|\mathcal{U}|}\sum_{x_i \in \mathcal{U}} \mathcal{L}_c + w_r \frac{1}{|\mathcal{D}|}\sum_{x_i \in \mathcal{D}} \frac{1}{|\mathcal{Q}|}\sum_{x_j \in \mathcal{Q}} \tilde{\mathcal{L}}_r$, where $\mathcal{L}_c$ is Label Disambiguation Cross-Entropy (LDCE) and $\tilde{\mathcal{L}}_r$ is NoiSNCL, with $w_r = 50$. Momentum hyperparameters are $\alpha = \beta = \gamma = 0.99$. The framework uses ResNet-18 as the backbone.

### Mechanism (EM Framework)

Classifier predictions are integrated into an EM framework: the E-step corresponds to pseudo-label assignment, and the M-step corresponds to minimizing NoiSNCL. Theorem 1 proves that under the vMF distribution assumption, minimizing $\tilde{\mathcal{R}}_r$ is equivalent to maximizing a lower bound of the likelihood function $L_1 = \sum_{\mathcal{S}_c} \frac{|\mathcal{S}_c|}{n_u} \|\nu_c\|^2 \leq L_2$. As $\|\nu_c\| \to 1$ (high intra-class feature clustering), the bound becomes tighter, providing a theoretical guarantee for the synergy between NoiSNCL and PLD.

## Key Experimental Results

### Main Results

Comparison across 5 datasets (3 general, 2 disaster damage remote sensing). NcPU achieved top performance without auxiliary information:

| Method | Auxiliary Info | CIFAR-10 OA | CIFAR-100 OA | STL-10 OA | ABCD OA | xBD OA |
|--------|----------------|-------------|--------------|-----------|---------|--------|
| CE (U as Neg) | None | 60.45 | 50.36 | 50.30 | 55.70 | 84.08 |
| uPU | $\pi_p$ | 65.52 | 61.44 | 57.08 | 83.76 | 86.82 |
| nnPU | $\pi_p$ | 87.29 | 72.00 | 80.62 | 87.73 | 82.60 |
| DistPU | $\pi_p$ | 85.29 | 67.63 | 85.62 | 86.25 | 82.94 |
| HolisticPU | Neg Samples | 84.20 | 64.01 | 72.81 | 65.49 | 81.98 |
| LaGAM | Neg Samples | 95.78 | 84.82 | 88.64 | 75.90 | 79.14 |
| WSC | Estimated Para | 90.55 | 75.39 | 79.06 | 80.10 | 84.89 |
| **NcPU** | **None** | **97.36** | **88.28** | **91.40** | **91.10** | **87.60** |
| Supervised | All Labels | 96.96 | 89.65 | — | 92.00 | 88.47 |

NcPU outperformed supervised learning on CIFAR-10 (97.36 vs 96.96) and stayed within 1% of supervised performance on CIFAR-100 and ABCD.

### Ablation Study (CIFAR-100)

| Non-contrastive Loss | Label Disambiguation | OA | F1 | Note |
|----------------------|----------------------|----|----|------|
| None | $s$ (PhantomGate) | 61.54 | 40.58 | No rep. learning; poor results |
| $\tilde{\mathcal{L}}_r$ (NoiSNCL) | None | 50.27 | 1.09 | No disambiguation; NoiSNCL fails alone |
| $\mathcal{L}_{self-r}$ (Self-sup) | $s$ | 73.22 | 72.75 | Standard self-sup + PhantomGate |
| $\mathcal{L}_r$ (Std-sup) | $s$ | 84.58 | 85.90 | Std loss effective but limited by noise |
| $\tilde{\mathcal{L}}_r$ (NoiSNCL) | $s'$ (Prototype only) | 75.14 | 79.91 | No PG; low precision (67%) |
| $\tilde{\mathcal{L}}_r$ (NoiSNCL) | $s'$ + SAT | 50.25 | 1.01 | SAT negative supervision too inaccurate |
| $\tilde{\mathcal{L}}_r$ (NoiSNCL) | $s$ (PhantomGate) | **88.28** | **88.14** | Full NcPU |

### Key Findings

- **NoiSNCL is the primary gain source**: Adding NoiSNCL to the simple uPU baseline improved CIFAR-10 OA from 69.43% to 97.35% (+27.9 points), suggesting discriminative representation is the core bottleneck of PU learning, not classifier design.
- **NoiSNCL vs Std Loss**: On CIFAR-100, $\tilde{\mathcal{L}}_r + s$ (88.28%) outperformed $\mathcal{L}_r + s$ (84.58%) by 3.7 points, validating noisy-pair robustness. Performance remained comparable in supervised settings.
- **PhantomGate Necessity**: Prototype-only disambiguation ($s'$) resulted in 98.7% recall but only 67% precision. SAT alone spiked precision to 98% but crashed recall to 0.5%. PhantomGate’s regret mechanism balanced them (89% precision, 87% recall).
- **Hyperparameter Robustness**: Identical hyperparameters were used across all 5 datasets, showing low sensitivity to $\alpha$ and $\gamma$.
- **Training Stability**: On CIFAR-10, extending training from 400 to 1200 epochs resulted in OA fluctuations within 0.5%, indicating no over-fitting.

## Highlights & Insights

- **Elegance of sqrt transform**: A simple square root flips the gradient-distance relationship, shifting dominance from noise to clean pairs. This "loss shape manipulation" to control gradient priority is a powerful idea applicable to any noisy label scenario.
- **EM Loop**: The theory proves that NoiSNCL and PLD are mutually dependent. The E-step provides better cluster assignments, while the M-step makes clusters more compact.
- **Paradigm of "Simple Method + Good Representation"**: uPU + NoiSNCL outperformed complex PU-specific designs, implying that when the representation space is sufficiently discriminative, naive risk estimation suffices.

## Limitations & Future Work

- **vMF Distribution Assumption**: The EM analysis assumes features follow a vMF distribution (Gaussian on a sphere), which may not hold for non-spherical data distributions.
- **Evaluation Scope**: Experiments were limited to image classification. The effectiveness in NLP (e.g., text PU learning) or tabular data remains unknown.
- **Fixed Positive Samples**: The impact of extreme positive label scarcity (e.g., <100 samples) was not extensively analyzed.
- **Multi-class Extension**: The current framework is binary-centric; extending to multi-class PU learning remains an open question.

## Related Work & Insights

- **vs LaGAM**: LaGAM requires auxiliary negative samples. NcPU surpasses it without such information and shows better generalization on remote sensing data.
- **vs DistPU**: DistPU relies on estimated class priors $\pi_p$. NcPU requires no prior knowledge and achieves higher accuracy.
- **vs WSC**: WSC uses graph theory and contrastive learning. NcPU proves that noisy-pair robustness within a simpler non-contrastive framework is more effective than complex graph structures.

## Rating

- Novelty: ⭐⭐⭐⭐ Flip of gradient dominance via sqrt is simple yet profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 datasets, 11 baselines, detailed ablation, and stability tests.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivation and intuitive analysis.
- Value: ⭐⭐⭐⭐⭐ Milestone progress in bringing PU performance close to supervised levels.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Regulating Internal Alignment Flows for Robust Learning Under Spurious Correlations](regulating_internal_alignment_flows_for_robust_learning_under_spurious_correlati.md)
- [\[ICLR 2026\] Learning in Prophet Inequalities with Noisy Observations](learning_in_prophet_inequalities_with_noisy_observations.md)
- [\[ICLR 2026\] Scaling Direct Feedback Learning with Jacobian Alignment Guarantees](scaling_direct_feedback_learning_with_jacobian_alignment_guarantees.md)
- [\[ICLR 2026\] Robust Equation Structure Learning with Adaptive Refinement (RESTART)](robust_equation_structure_learning_with_adaptive_refinement.md)
- [\[ICLR 2026\] Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets](mitigating_spurious_correlation_via_distributionally_robust_learning_with_hierar.md)

</div>

<!-- RELATED:END -->
