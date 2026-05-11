---
title: >-
  [Paper Note] Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment
description: >-
  [AAAI 2026][Self-Supervised Learning][Open-set semi-supervised learning] This paper proposes SkipAlign, a framework that introduces a third "skip" operation alongside the conventional pull/push operations in contrastive…
tags:
  - "AAAI 2026"
  - "Self-Supervised Learning"
  - "Open-set semi-supervised learning"
  - "contrastive learning"
  - "OOD detection"
  - "selective non-alignment"
  - "prototype learning"
date: 2026-05-08
content_hash: c39634c79e8f2eb1
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment

**Conference**: AAAI 2026  
**arXiv**: [2504.12569](https://arxiv.org/abs/2504.12569)  
**Code**: None  
**Area**: Self-supervised / Semi-supervised Learning / OOD Detection  
**Keywords**: Open-set semi-supervised learning, contrastive learning, OOD detection, selective non-alignment, prototype learning

## TL;DR
This paper proposes SkipAlign, a framework that introduces a third "skip" operation alongside the conventional pull/push operations in contrastive learning. Low-confidence samples are selectively excluded from alignment and subjected only to mild repulsion, allowing in-distribution (ID) classes to form compact "galaxies" while OOD samples naturally disperse into the "interstellar void." The approach achieves an average AUC improvement of +3.1 on unseen OOD detection, with a maximum gain of +7.1.

## Background & Motivation

**Background**: Standard semi-supervised learning (SSL) assumes all unlabeled data belong to known classes, yet in practice unlabeled data are open-set and may contain out-of-distribution (OOD) samples. Open-set SSL (OSSL) must simultaneously improve ID classification accuracy and detect OOD samples.

**Limitations of Prior Work**: Existing methods fall into two extremes. (a) **Aggressive approaches** (SCOMatch, ProSub) compress all OOD samples into a single synthetic class or an ID subspace, causing geometric collapse and overfitting to seen OOD samples, which fails to generalize to unseen OOD. (b) **Conservative approaches** (SSB) discard low-confidence samples entirely, discarding valuable geometric cues.

**Key Challenge**: The OOD space is inherently vast and heterogeneous, yet training only exposes a limited set of OOD samples. Forcing OOD alignment to a fixed location (aggressive) introduces bias, while discarding uncertain samples (conservative) wastes information.

**Goal**: To achieve generalizable detection of unseen OOD samples without forcing OOD alignment or discarding uncertain samples.

**Key Insight**: The embedding space is conceptualized as a universe in which ID classes form compact "galaxies" and OOD samples inhabit the "interstellar void." By applying only mild repulsion—rather than alignment—to uncertain samples, the void is allowed to remain void.

**Core Idea**: Introduce a "skip" operation alongside pull and push: uncertain samples are not attracted toward any prototype but receive non-selective angular repulsion, preserving the natural diversity and dispersion of OOD samples.

## Method

### Overall Architecture
SkipAlign consists of a shared backbone with three task heads:
- **Closed-set classifier (CC)**: $K$-class softmax classification, outputting $\mathbf{p} \in \mathbb{R}^K$
- **OOD detector (OD)**: $K$ one-vs-all (OVA) binary classifiers, each outputting $(\varphi_k^{ID}, \varphi_k^{OOD})$
- **SNA module**: Selective contrastive learning in the projection space (core contribution)

### Key Designs

1. **Selective Non-Alignment (SNA)**:

    - **Function**: Selectively decides whether each unlabeled sample should be pulled toward a prototype or skipped (subjected only to mild repulsion).
    - **Mechanism**: For each unlabeled embedding $z_i$, a confidence mask $\Phi_i$ is computed via **dual-gated ID selection**, which jointly requires high confidence from both the closed-set classifier ($p_{i,\hat{k}} > \tau_{ID}$) and the OOD detector ($\varphi_{i,\hat{k}}^{ID} > \eta_{ID}$). The SNA loss is defined as $\mathcal{L}_{USNA}(z_i) = -\Phi_i \cdot \text{sim}(z_i, \mu_{\hat{k}})/T + \log\sum_j\exp(\text{sim}(z_i, \mu_j)/T)$. When $\Phi_i=1$, the full pull+push is applied; when $\Phi_i=0$, the pull term vanishes and only softmax-weighted non-selective angular repulsion remains.
    - **Design Motivation**: Gradient analysis shows that when $\Phi_i=0$, the update direction is $\sum_j \alpha_j \hat{\mu}_j$ (a soft-weighted average of ID prototype directions), constrained to pure angular updates by the projection operator $(I-\hat{z}\hat{z}^T)$. OOD samples closer to an ID "galaxy" receive stronger repulsion but are never attracted toward any specific galaxy.

2. **Dual-Gated ID Selection**:

    - **Function**: Strictly filters which unlabeled samples may be pulled toward prototypes.
    - **Mechanism**: Both the closed-set classifier and the OOD detector must independently assign high confidence to the same class, substantially reducing the risk of erroneously pulling OOD samples into ID clusters.
    - **Design Motivation**: Pseudo-labels from a single classifier are prone to errors. The dual gate acts as a "two-lock" mechanism in which the probability of simultaneous error is far lower than with a single gate.

3. **Adaptive Prototype Refinement**:

    - **Function**: Dynamically updates class prototypes using both labeled samples and high-confidence pseudo-labeled unlabeled samples.
    - **Mechanism**: The final prototype is $\mu = w_l \mu_l + w_u \mu_u$, a weighted average of the labeled-sample prototype and the selected unlabeled-sample prototype, with adaptively adjusted weights.
    - **Design Motivation**: Prototypes derived from a small number of labeled samples (e.g., 25–50 per class) are insufficiently representative; incorporating reliable pseudo-labeled samples improves prototype quality.

### Loss & Training
- **Total loss**: $\mathcal{L} = \mathcal{L}_{CC} + \mathcal{L}_{OD} + \mathcal{L}_{SNA}$
- The closed-set classifier follows the FixMatch framework (supervised CE + consistency regularization).
- The OOD detector employs OVA loss + entropy minimization + SOCR consistency + pseudo-negative loss.
- The SNA loss is $\lambda_{USNA}\mathcal{L}_{USNA} + \lambda_{IA}\mathcal{L}_{IA} + \lambda_{PA}\mathcal{L}_{PA}$ (unlabeled SNA + labeled instance alignment + labeled prototype alignment).
- Instance alignment for labeled samples uses supervised contrastive learning (SupCon); prototype alignment always pulls toward the ground-truth class prototype.

## Key Experimental Results

### Main Results
Overall OOD AUC on CIFAR-10 (6 ID / 4 OOD), CIFAR-100 (55/45, 80/20), ImageNet-30, and TinyImageNet:

| Dataset | Labels | SkipAlign AUC | Prev. SOTA AUC | Gain |
|--------|--------|--------------|-------------|------|
| CIFAR-10 (6/4) | 25/class | **94.9** | 87.8 (ProSub) | +7.1 |
| CIFAR-10 (6/4) | 50/class | **96.6** | 93.5 (SSB) | +3.1 |
| CIFAR-100 (55/45) | 25/class | **82.8** | 80.8 (SSB) | +2.0 |
| CIFAR-100 (80/20) | 25/class | **80.1** | 76.5 (ProSub) | +3.6 |
| ImageNet-30 | 5% | **90.7** | 88.5 (ProSub) | +2.2 |
| TinyImageNet | 5% | **77.1** | 75.8 (ProSub) | +1.3 |

Closed-set accuracy is on par with or marginally better than the best baseline.

### Ablation Study

| Configuration | Acc. | Overall AUC | Unseen AUC |
|------|------|-------------|------------|
| FixMatch baseline | ~91 | ~82 | ~79 |
| + OVA OOD detector | ~92 | ~89 | ~86 |
| + SNA (single gate) | ~92 | ~93 | ~91 |
| + SNA (dual gate) | ~93 | **~95** | **~94** |

### Key Findings
- **t-SNE visualizations** clearly show that ID classes form six compact clusters while both seen and unseen OOD samples scatter in the "void" between clusters without adhering to any ID cluster.
- **Feature norm disparity**: ID sample norms are substantially higher than those of OOD samples (>2×), since only pulled ID samples accumulate norm.
- **Cosine similarity**: ID samples exhibit high similarity to their corresponding prototypes (~0.8), whereas both seen and unseen OOD samples show low similarity to all prototypes (<0.3).
- SCOMatch achieves only 62.2% AUC on CIFAR-100 versus 82.8% for SkipAlign, confirming that forcing OOD into a single class severely overfits to seen OOD.
- Dual-gate selection improves AUC by approximately 2 points over single-gate, validating the effectiveness of cross-validation between the two detectors.

## Highlights & Insights
- The concept of **"skip" as a third fundamental contrastive operation** is elegantly concise: abstaining from pulling does not mean ignoring a sample—mild repulsion prevents information waste while avoiding erroneous alignment. This design philosophy—"let the void be void"—is transferable to other open-set, long-tail, and noisy-label settings.
- **Gradient analysis is precise and transparent**: the paper derives the geometric effect of the skip operation directly from $\nabla_{z_i}\mathcal{L}_{USNA}$, demonstrating pure angular repulsion with distance-adaptive strength.
- The **galaxy–void analogy** is both intuitive and effective, rendering a complex geometric structure accessible through an astronomical metaphor.

## Limitations & Future Work
- The thresholds $\tau_{ID}$ and $\eta_{ID}$ are hyperparameters that require dataset-specific tuning; no adaptive selection mechanism is provided.
- Experiments are conducted primarily on small-to-medium-scale datasets (CIFAR-10/100, ImageNet-30, TinyImageNet); large-scale settings remain untested.
- The non-selective repulsion in SNA is a softmax-weighted average over all prototypes, which may become diffuse when the number of ID classes is large.

## Related Work & Insights
- **vs. SCOMatch**: Compresses all OOD samples into a single class, causing geometric collapse. AUC is 62.2% vs. 82.8% for SkipAlign on CIFAR-100—a gap of 20 points.
- **vs. ProSub**: Defines an ID subspace and repels OOD, but unseen OOD samples within the subspace are misclassified as ID. SkipAlign avoids the subspace concept entirely, relying solely on class prototypes.
- **vs. SSB**: Discards low-confidence samples, whereas SkipAlign converts them into pure repulsion signals, preserving their geometric value. SkipAlign consistently outperforms SSB in AUC.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "skip" operation as a third fundamental contrastive operation is conceptually novel; the galaxy–void framework is intuitively elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-setting evaluation with comprehensive ablations and rich t-SNE/norm/similarity visualizations.
- Writing Quality: ⭐⭐⭐⭐⭐ Vivid analogies, clear gradient derivations, and strong illustrative figures.
- Value: ⭐⭐⭐⭐ Establishes a new paradigm for open-set learning; the skip operation is broadly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning](explanation-preserving_augmentation_for_semi-supervised_graph_representation_lea.md)
- [\[CVPR 2026\] SpHOR: A Representation Learning Perspective on Open-set Recognition](../../CVPR2026/self_supervised/sphor_a_representation_learning_perspective_on_open-set_recognition_for_identify.md)
- [\[AAAI 2026\] Robust Tabular Foundation Models](robust_tabular_foundation_models.md)
- [\[AAAI 2026\] NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding via Cognitive Priors and Bidirectional Semantic Alignment](neurobridge_bio-inspired_self-supervised_eeg-to-image_decoding_via_cognitive_pri.md)
- [\[CVPR 2026\] SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks](../../CVPR2026/self_supervised/sphor_a_representation_learning_perspective_on_open-set_recognition_for_identify.md)

</div>

<!-- RELATED:END -->
