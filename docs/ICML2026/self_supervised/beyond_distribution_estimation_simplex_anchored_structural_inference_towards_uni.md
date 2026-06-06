---
title: >-
  [Paper Note] Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning
description: >-
  [ICML 2026][Self-Supervised Learning][Universal Semi-Supervised Learning] This paper proposes SAGE, which replaces "estimating the distribution of unlabeled data" with "structural inference in representation space." By c…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Universal Semi-Supervised Learning"
  - "Equiangular Tight Frame"
  - "Graph Structural Inference"
  - "Pseudo-Label Reliability"
date: 2026-05-08
content_hash: 0a3271bbb5bf821e
---

# Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning

**Conference**: ICML 2026  
**arXiv**: [2605.07557](https://arxiv.org/abs/2605.07557)  
**Code**: https://github.com/Yaxin-ML/SAGE  
**Area**: Semi-Supervised Learning / Representation Learning  
**Keywords**: Universal Semi-Supervised Learning, Equiangular Tight Frame, Graph Structural Inference, Pseudo-Label Reliability

## TL;DR
This paper proposes SAGE, which replaces "estimating the distribution of unlabeled data" with "structural inference in representation space." By combining simplex ETF geometric anchors, high-order graph propagation, and distribution-agnostic reliability weighting, it achieves an average 8.52% accuracy improvement under the UniSSL setting with extremely scarce labels and arbitrary unlabeled distributions.

## Background & Motivation
**Background**: Mainstream semi-supervised learning (SSL) follows the "FixMatch series" approach—assigning high-confidence pseudo-labels to unlabeled samples and enforcing consistency regularization. Later, LTSSL extended to long-tail scenarios, and ReaLTSSL further allowed distribution mismatch between labeled and unlabeled data.

**Limitations of Prior Work**: Methods like FreeMatch / SoftMatch **assume unlabeled data is uniformly distributed**, forcibly aligning pseudo-labels to uniformity via distribution alignment or entropy maximization, which leads to many false positives under real arbitrary distributions. Methods like CPG / SimPro that "dynamically estimate distributions" fail when labels are extremely scarce (only 4 or even 1 labeled sample per class), causing pseudo-label collapse and representation collapse (class clusters overlap in t-SNE, silhouette score drops sharply).

**Key Challenge**: All existing methods treat the "pseudo-label → representation" signal chain as the main path, but pseudo-labels themselves are unreliable, especially in long-tail or arbitrary distributions, and the more one tries to "align distributions," the more biased it becomes. The authors' diagnostic experiment shows: **relationships between samples are much more reliable than pseudo-labels themselves**—during training, the proportion of incorrect pseudo-labels corrected by "neighbor relationships" steadily increases and stabilizes at a high level.

**Goal**: Under the UniSSL setting of extreme label scarcity and completely unknown unlabeled distribution, break the deadlock of "estimating distribution before generating pseudo-labels," enabling the model to learn discriminative representations without knowing $\gamma_u$.

**Key Insight**: Shift the focus from "distribution estimation" to "representation-level structural inference"—use high-order relationships between samples to build "structural consensus" as the supervision signal, and use a set of fixed geometric anchors to force representations of different classes to be maximally equiangularly separated.

**Core Idea**: Use simplex equiangular tight frame (ETF) as the coordinate system to obtain relational embeddings via ridge regression → perform $\beta$-step graph diffusion on the relational graph to obtain the "structural consensus matrix" → use it to align instance-wise similarity instead of pseudo-labels.

## Method

### Overall Architecture
SAGE adds three modules to the FixMatch-style dual-view (weak/strong augmentation) SSL framework: (1) **GRI Graph Structural Relational Inference**—project each sample's feature $\mathbf{z}_i$ onto fixed simplex ETF anchors $\mathbf{P}$ to obtain relational embedding $\mathbf{a}_i$, construct affinity matrix $\mathbf{A}$ from inner products of $\mathbf{a}_i$, then perform $\beta$-step Markov propagation to get structural consensus $\mathbf{G}=\hat{\mathbf{P}}^\beta$, which serves as "soft supervision" to guide instance-wise similarity $\mathbf{S}$; (2) **Simplex ETF Anchor Generation**—offline construction of $K=d+1$ fixed, zero-centered, unit-norm, pairwise equiangular vectors as a class-agnostic coordinate system; (3) **DRP Distribution-Agnostic Reliability Weighting + Auxiliary Branch**—use max-confidence and top-2 margin, two distribution-agnostic statistics combined with EMA, to weight pseudo-labels, and isolate pseudo-label flow to auxiliary head $\phi_{aux}$, while the main head $\phi_{cls}$ only sees labeled data to keep the decision boundary clean. The final objective $\mathcal{L}_{total}=\mathcal{L}_{cls}+\mathcal{L}_{con}+\mathcal{L}_{sim}+\mathcal{L}_{aux}$ is optimized end-to-end.

### Key Designs

1. **Graph-state Relational Inference (GRI)**:

    - **Function**: Uses high-order relationships between samples to replace unreliable pseudo-labels as the supervision signal for representation learning.
    - **Mechanism**: Projected features $\mathbf{z}_i$ obtain closed-form relational embeddings via ridge regression $\min_{\mathbf{a}_i}\|\mathbf{z}_i-\mathbf{a}_i\mathbf{P}\|_2^2+\lambda\|\mathbf{a}_i\|_2^2$, yielding $\mathbf{a}_i=(\mathbf{z}_i\mathbf{P}^\top)(\mathbf{P}\mathbf{P}^\top+\lambda\mathbf{I})^{-1}$; affinity matrix $\mathbf{A}_{ij}=\langle\mathbf{a}_i,\mathbf{a}_j\rangle$ is row-softmaxed to $\hat{\mathbf{P}}$, and $\beta=5$ steps of diffusion yield structural consensus $\mathbf{G}$; contrastive loss $\mathcal{L}_{con}=\text{BCE}(\mathbf{S},\text{sg}[\mathbf{G}])$ aligns current instance similarity $\mathbf{S}_{ij}=\sigma(\langle\mathbf{z}_i,\mathbf{z}_j\rangle)$ to $\mathbf{G}$; an additional $\mathcal{L}_{sim}$ enforces cross-view similarity between backbone features $\mathbf{f}$ and projections $\mathbf{z}$ for weak/strong augmentations.
    - **Design Motivation**: High-order propagation diffuses local relationships into stable global consensus, more robust than single-step neighbor relations; stop-gradient prevents structural signals from being contaminated by their own gradients.

2. **Simplex Equiangular Tight Frame Geometric Anchors**:

    - **Function**: Provides a set of **class-frequency-independent** fixed coordinate axes, enforcing maximal equiangular separation between class representations to counteract representation collapse caused by long-tail distributions.
    - **Mechanism**: Take a random Gaussian matrix, perform QR decomposition to obtain orthogonal $\mathbf{Q}\in\mathbb{R}^{d\times d}$; the $d$ nonzero eigenvectors of the centered matrix $\mathbf{O}=\mathbf{I}_K-\frac{1}{K}\mathbf{1}_K\mathbf{1}_K^\top$ form $\mathbf{V}\in\mathbb{R}^{K\times d}$; the final anchor matrix $\mathbf{P}=\sqrt{\frac{K}{K-1}}\mathbf{V}\mathbf{Q}^\top$ satisfies $\mathbf{P}^\top\mathbf{1}_K=\mathbf{0}$, each row is unit-norm, and $\mathbf{p}_i^\top\mathbf{p}_j=-\frac{1}{K-1}$. This achieves the maximal equiangular separation for $K$ vectors in $\mathbb{R}^d$.
    - **Design Motivation**: Learnable prototypes are biased toward majority classes under long-tail; fixed ETF anchors are generated once and never updated, providing geometric invariance independent of sample count, decoupling representation learning from distribution priors.

3. **Distribution-agnostic Reliability Prioritization (DRP) + Auxiliary Branch**:

    - **Function**: Selects reliable pseudo-labels without knowing $\gamma_u$ and isolates potential erroneous signals.
    - **Mechanism**: For each unlabeled sample, compute $q_{max}=\max(\mathbf{q}_w)$ and $q_{gap}=q_w^{(1)}-q_w^{(2)}$, maintain their EMA means $\mu_\kappa$ and variances $\sigma_\kappa^2$; use truncated Gaussian kernel $\mathcal{W}(q_\kappa;\mu_\kappa,\sigma_\kappa)=\exp(-\frac{[\min(0,q_\kappa-\mu_\kappa)]^2}{2\sigma_\kappa^2})$ to weight samples (full score 1 if above mean, exponentially decayed below), final weight $w=\mathcal{W}_{max}\cdot\mathcal{W}_{gap}$. Architecturally, introduce auxiliary head $\phi_{aux}$ for all unlabeled data, main head $\phi_{cls}$ for labeled samples only.
    - **Design Motivation**: Max-confidence measures absolute certainty, top-2 margin measures relative discriminability, both are **distribution-agnostic** statistics; the auxiliary branch confines pseudo-label gradients within $\phi_{aux}$, preventing contamination of the main classification boundary.

### Loss & Training
$\mathcal{L}_{total}=\mathcal{L}_{cls}+\mathcal{L}_{con}+\mathcal{L}_{sim}+\mathcal{L}_{aux}$; backbone is WRN-28-2, SGD + 0.9 momentum + 5e-4 weight decay, cosine learning rate decaying from 0.03, trained for $2^{18}$ steps; $\lambda=0.1$, $\beta=5$ fixed; labeled batch 64, unlabeled batch $7\times64$; single RTX 4090.

## Key Experimental Results

### Main Results
Compared with 9 baselines on CIFAR-10/100, SVHN, STL-10, Food-101 datasets, under Uniform/Long-tailed/Arbitrary unlabeled distributions.

| Dataset / Setting | Metric | SAGE | Strongest Baseline | Gain |
|-------------------|--------|------|--------------------|------|
| CIFAR-10 Avg (9 settings) | Acc% | See table | CGMatch 58.38 / CPG series | Avg **+8.52 pp** |
| CIFAR-10, $N=40, \gamma_u=150$, Arbitrary | Acc% | Significantly better | FreeMatch 45.38 / SoftMatch 42.82 / CGMatch 49.92 | Substantial lead |
| SVHN $(N_{max},M_{max},\gamma_l,\gamma_u)=(4,4996,1,150)$ | Silhouette↑ | High | FreeMatch / CPG significantly lower | Representations much more separated |

Comparison shows: (i) FreeMatch-style SSL degrades to around 50% under long-tailed/arbitrary settings; (ii) SimPro completely collapses under extreme label scarcity ($N=40$), accuracy only ~16% (essentially random), proving its distribution estimation module fails without sufficient supervision; (iii) SAGE consistently outperforms across all settings.

### Ablation Study
Figure 9 and Appendix in the original paper provide results after removing key modules (refer to visualizations for exact values):

| Configuration | Key Phenomenon | Description |
|---------------|---------------|-------------|
| Full SAGE | High Acc + High silhouette | Complete model |
| w/o GRI (no structural inference) | Degrades to baseline | Loss of high-order relational supervision, pseudo-label noise directly pollutes representations |
| w/o Simplex ETF anchors (replaced with learnable prototypes) | Class clusters biased by majority under long-tail | Geometric anchors are key to preventing collapse |
| w/o DRP + auxiliary branch | Acc drops, erroneous pseudo-labels leak to main head | Both isolation and weighting are indispensable |
| Pseudo-label correction rate (GRI supervision) | Monotonically rises to high level during training | Confirms "relations are more reliable than pseudo-labels" |

### Key Findings
- The observation that "**inter-sample relationships are more reliable than pseudo-labels**" holds across all settings, forming the foundation of the method.
- The silhouette improvement from simplex ETF anchors is significant, indicating that the root cause of representation collapse under long-tail is the lack of geometric priors, not just pseudo-label noise.
- Decoupling $\phi_{cls}$ and $\phi_{aux}$ is a very cheap design, almost zero-cost for confining noisy gradients.
- The method is insensitive to $\lambda$ and $\beta$ (Appendix D), with $\beta=5$ being a reasonable diffusion step—too few is insufficient for propagation, too many approaches a trivial steady-state distribution.

## Highlights & Insights
- **Paradigm Shift**: Switching from "distribution estimation → pseudo-label generation → representation learning" to "geometric anchor construction → relational graph inference → representation learning → pseudo-label back-propagation," reversing the signal chain and making stable geometric structure the main axis. Such reframing is rare in ReaLTSSL literature.
- **ETF Anchors + Relational Embedding**: Interpreting ridge regression $\mathbf{a}_i=(\mathbf{z}_i\mathbf{P}^\top)(\mathbf{P}\mathbf{P}^\top+\lambda\mathbf{I})^{-1}$ as "projecting samples onto geometric coordinate axes" is ingenious—it ensures relational embeddings have closed-form solutions (no extra learnable parameters) and naturally transfers the "equiangularity" of geometric anchors to the relational space.
- **Distribution-agnostic Statistics**: The combination of $q_{max}$ and $q_{gap}$ avoids any assumptions about class distribution, providing a truly "protocol-agnostic" reliability metric that can be seamlessly transferred to any unknown distribution scenario.

## Limitations & Future Work
- When the number of classes $C$ is large, simplex ETF requires embedding dimension $d\geq C-1$, which constrains small models or small projection heads.
- The computational cost of graph propagation steps $\beta$ grows with $|\mathbf{B}|^2$ for large batches; scaling to ImageNet-level requires intra-mini-batch propagation or sampling approximations.
- The method treats all unlabeled samples in a batch as "graph nodes," without considering global structure across batches/epochs, potentially missing long-range manifold topology.
- Geometric anchors are fixed; "task-adaptive anchor generation" has not been explored, which could be combined with prompt-learning for text/multimodal SSL in the future.

## Related Work & Insights
- **vs FreeMatch / SoftMatch**: They select samples via dynamic confidence thresholding, while this work uses geometric + structural supervision (GJS) to completely replace confidence-based selection, making it more robust to unknown distributions.
- **vs CPG / SimPro**: They explicitly/implicitly estimate the unlabeled distribution, but collapse under extreme label scarcity; SAGE completely bypasses distribution estimation.
- **vs Neural Collapse / DR-DSN series**: Those works use ETF as classifier weight targets, while SAGE uses ETF as "relational coordinate system" and "contrastive anchor," a more upstream and noise-resistant usage.
- **Insights**: For any task where pseudo-labels are unreliable (open-set SSL, noisy label learning, long-tail detection), the "signal source switch" strategy is worth emulating—find signals (structure, geometry, relations) more robust than labels as the main supervision.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The paradigm shift from "distribution estimation" to "structural inference" is a genuinely new perspective in ReaLTSSL literature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five datasets + various imbalance ratios + 9 baselines, but lacks ImageNet-scale validation.
- Writing Quality: ⭐⭐⭐⭐ Motivation is logically progressive, from diagnostic experiments to method; formulas are dense.
- Value: ⭐⭐⭐⭐ Proposes the more realistic UniSSL setting + stable average +8.52 pp improvement, directly applicable to real-world SSL scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Understanding Self-Supervised Learning via Latent Distribution Matching](understanding_self-supervised_learning_via_latent_distribution_matching.md)
- [\[AAAI 2026\] Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning](../../AAAI2026/self_supervised/explanation-preserving_augmentation_for_semi-supervised_graph_representation_lea.md)
- [\[AAAI 2026\] Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment](../../AAAI2026/self_supervised/let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-.md)
- [\[ICLR 2026\] InfoNCE Induces Gaussian Distribution](../../ICLR2026/self_supervised/infonce_induces_gaussian_distribution.md)
- [\[ICML 2026\] A Refined Generalization Analysis for Extreme Multi-class Supervised Contrastive Representation Learning](a_refined_generalization_analysis_for_extreme_multi-class_supervised_contrastive.md)

</div>

<!-- RELATED:END -->
