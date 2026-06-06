---
title: >-
  [Paper Note] Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning
description: >-
  [ICML 2026][Self-Supervised Learning][Universal Semi-Supervised Learning (UniSSL)] This paper proposes SAGE, which replaces "estimating unlabeled data distribution" with "structural inference in the representation space.…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Universal Semi-Supervised Learning (UniSSL)"
  - "Equiangular Tight Frame (ETF)"
  - "Graph Structural Inference"
  - "Pseudo-label Reliability"
date: 2026-05-08
content_hash: ebca5616b393ac62
---

# Beyond Distribution Estimation: Simplex Anchored Structural Inference Towards Universal Semi-Supervised Learning

**Conference**: ICML 2026  
**arXiv**: [2605.07557](https://arxiv.org/abs/2605.07557)  
**Code**: https://github.com/Yaxin-ML/SAGE  
**Area**: Semi-supervised Learning / Representation Learning  
**Keywords**: Universal Semi-Supervised Learning (UniSSL), Equiangular Tight Frame (ETF), Graph Structural Inference, Pseudo-label Reliability

## TL;DR
This paper proposes SAGE, which replaces "estimating unlabeled data distribution" with "structural inference in the representation space." By utilizing a trio of simplex ETF geometric anchors, high-order graph propagation, and distribution-agnostic reliability weighting, it achieves an average accuracy improvement of 8.52% under the UniSSL setting characterized by extreme label scarcity and arbitrary unlabeled distributions.

## Background & Motivation
**Background**: Mainstream semi-supervised learning (SSL) follows the "FixMatch-style" paradigm—assigning high-confidence pseudo-labels to unlabeled samples followed by consistency regularization. Later, LTSSL extended this to long-tailed scenarios, and ReaLTSSL further allowed for distribution mismatch between unlabeled and labeled data.

**Limitations of Prior Work**: Methods such as FreeMatch / SoftMatch **assume a uniform distribution for unlabeled data** by default, using distribution alignment or entropy maximization to force pseudo-labels toward uniformity. This leads to massive false positives when encountering real-world arbitrary distributions. "Dynamic distribution estimation" methods like CPG / SimPro fail to estimate accurately when labels are extremely scarce (e.g., only 4 or even 1 labeled sample per class), leading to representation collapse (overlapping clusters in t-SNE and plummeting silhouette coefficients) once pseudo-labels fail.

**Key Challenge**: All existing methods treat the signal chain "pseudo-label $\rightarrow$ representation" as the main axis. However, pseudo-labels are inherently unreliable, especially in long-tailed or arbitrary distributions. The more one attempts to "align distributions," the more biased the model becomes. Through a diagnostic experiment, the authors found that **relationships between samples are much more reliable than pseudo-labels themselves**—during training, the ratio of incorrect pseudo-labels being corrected by "neighbor relationships" to the correct class rises steadily and stabilizes at a high level.

**Goal**: Under the UniSSL setting with extreme label scarcity and unknown unlabeled distributions, the goal is to bypass the dead-loop of "estimation before pseudo-labeling" and enable the model to learn discriminative representations without knowing $\gamma_u$.

**Key Insight**: Shift the focus from "distribution estimation" to "representation-level structural inference"—using high-order relationships between samples to establish a "structural consensus" as a supervisory signal, coupled with fixed geometric anchors to force maximum equiangular separation between representations of different classes.

**Core Idea**: Use a simplex Equiangular Tight Frame (ETF) as a coordinate system to perform ridge regression for relational embeddings $\rightarrow$ perform $\beta$-step graph diffusion on the relationship graph to obtain a "structural consensus matrix" $\rightarrow$ use this instead of pseudo-labels to align instance-wise similarities.

## Method

### Overall Architecture
SAGE adds three modules to the dual-view (weak/strong augmentation) SSL framework of FixMatch: (1) **GRI (Graph-state Relational Inference)**—projects feature embeddings $\mathbf{z}_i$ onto fixed simplex ETF anchors $\mathbf{P}$ via ridge regression to obtain relational embeddings $\mathbf{a}_i$. The affinity matrix $\mathbf{A}$ is constructed from inner products of $\mathbf{a}_i$, followed by $\beta$-step Markov propagation to derive the structural consensus $\mathbf{G}=\hat{\mathbf{P}}^\beta$, serving as "soft supervision" for instance-wise similarity $\mathbf{S}$. (2) **Simplex ETF Anchor Generation**—constructs $K=d+1$ zero-mean, unit-norm, pairwise equiangular fixed vectors offline as a class-agnostic coordinate system. (3) **DRP (Distribution-agnostic Reliability Prioritization) + Auxiliary Branch**—weights pseudo-labels using two distribution-agnostic statistics (max-confidence and top-2 margin) combined via EMA, and isolates pseudo-label gradients to an auxiliary head $\phi_{aux}$, while the main head $\phi_{cls}$ sees only labeled data to maintain a clean decision boundary. The final objective $\mathcal{L}_{total}=\mathcal{L}_{cls}+\mathcal{L}_{con}+\mathcal{L}_{sim}+\mathcal{L}_{aux}$ is optimized end-to-end.

### Key Designs

1.  **Graph-state Relational Inference (GRI)**:
    - **Function**: Replaces unreliable pseudo-labels with high-order sample relationships as supervisory signals for representation learning.
    - **Mechanism**: Projected features $\mathbf{z}_i$ obtain closed-form relational embeddings $\mathbf{a}_i=(\mathbf{z}_i\mathbf{P}^\top)(\mathbf{P}\mathbf{P}^\top+\lambda\mathbf{I})^{-1}$ by solving $\min_{\mathbf{a}_i}\|\mathbf{z}_i-\mathbf{a}_i\mathbf{P}\|_2^2+\lambda\|\mathbf{a}_i\|_2^2$; the affinity matrix $\mathbf{A}_{ij}=\langle\mathbf{a}_i,\mathbf{a}_j\rangle$ yields $\hat{\mathbf{P}}$ via row-softmax. After $\beta=5$ diffusion steps, the structural consensus $\mathbf{G}$ is obtained. The contrastive loss $\mathcal{L}_{con}=\text{BCE}(\mathbf{S},\text{sg}[\mathbf{G}])$ aligns current instance similarities $\mathbf{S}_{ij}=\sigma(\langle\mathbf{z}_i,\mathbf{z}_j\rangle)$ to $\mathbf{G}$. Additionally, $\mathcal{L}_{sim}$ ensures cross-view similarity between backbone features $\mathbf{f}$ and projections $\mathbf{z}$.
    - **Design Motivation**: High-order propagation diffuses scattered local relationships into a stable global consensus, which is more robust than single-step neighbor relations. Stop-gradient prevents structural signals from being contaminated by their own gradients.

2.  **Simplex Equiangular Tight Frame (ETF) Anchors**:
    - **Function**: Provides a fixed coordinate system **independent of class frequencies**, forcing maximum equiangular separation of inter-class representations to counter representation collapse caused by long-tails.
    - **Mechanism**: A random Gaussian matrix is decomposed via QR to obtain orthogonal $\mathbf{Q}\in\mathbb{R}^{d\times d}$. The $d$ non-zero eigenvectors of the centering matrix $\mathbf{O}=\mathbf{I}_K-\frac{1}{K}\mathbf{1}_K\mathbf{1}_K^\top$ form $\mathbf{V}\in\mathbb{R}^{K\times d}$. The final anchor matrix $\mathbf{P}=\sqrt{\frac{K}{K-1}}\mathbf{V}\mathbf{Q}^\top$ satisfies $\mathbf{P}^\top\mathbf{1}_K=\mathbf{0}$, unit norm per row, and $\mathbf{p}_i^\top\mathbf{p}_j=-\frac{1}{K-1}$. This represents the maximum equiangular spacing achievable for $K$ vectors in $\mathbb{R}^d$.
    - **Design Motivation**: Learnable prototypes are biased toward majority classes in long-tail scenarios. Fixed ETF anchors are generated once and never updated, providing geometric invariance independent of sample counts and decoupling representation learning from distribution priors.

3.  **Distribution-agnostic Reliability Prioritization (DRP) + Auxiliary Branch**:
    - **Function**: Selects reliable pseudo-labels without prior knowledge of $\gamma_u$ and isolates potential noise signals.
    - **Mechanism**: For each unlabeled sample, $q_{max}=\max(\mathbf{q}_w)$ and $q_{gap}=q_w^{(1)}-q_w^{(2)}$ are computed, maintaining their EMA means $\mu_\kappa$ and variances $\sigma_\kappa^2$. Weights are assigned using a truncated Gaussian kernel $\mathcal{W}(q_\kappa;\mu_\kappa,\sigma_\kappa)=\exp(-\frac{[\min(0,q_\kappa-\mu_\kappa)]^2}{2\sigma_\kappa^2})$ (scores $> \text{mean}$ receive 1, while those below decay exponentially). The final weight is $w=\mathcal{W}_{max}\cdot\mathcal{W}_{gap}$. Architecturally, an auxiliary head $\phi_{aux}$ processes all unlabeled data, while the main head $\phi_{cls}$ only takes labeled samples.
    - **Design Motivation**: Max-confidence measures absolute certainty, while top-2 margin measures relative discriminability; both are **distribution-agnostic** statistics. The auxiliary branch restricts pseudo-label gradients to $\phi_{aux}$ to prevent contamination of the primary classification boundary.

### Loss & Training
$\mathcal{L}_{total}=\mathcal{L}_{cls}+\mathcal{L}_{con}+\mathcal{L}_{sim}+\mathcal{L}_{aux}$; Backbone: WRN-28-2; Optimizer: SGD + 0.9 momentum + 5e-4 weight decay; Learning rate: cosine decay starting from 0.03 for $2^{18}$ steps. Fixed hyperparameters: $\lambda=0.1, \beta=5$. Batch size: labeled 64, unlabeled $7\times64$. Hardware: Single RTX 4090.

## Key Experimental Results

### Main Results
Evaluated against 9 baselines across five datasets (CIFAR-10/100, SVHN, STL-10, Food-101) under three unlabeled distribution settings (Uniform/Long-tailed/Arbitrary).

| Dataset / Setting | Metric | Ours (SAGE) | Best Baseline | Gain |
|-------------------|--------|-------------|----------------|------|
| CIFAR-10 Avg (9 settings) | Acc% | See Table | CGMatch 58.38 / CPG series | Avg **+8.52 pp** |
| CIFAR-10, $N=40, \gamma_u=150$, Arbitrary | Acc% | Significantly Superior | FreeMatch 45.38 / SoftMatch 42.82 | Substantial Lead |
| SVHN $(N_{max},M_{max},\gamma_l,\gamma_u)=(4,4996,1,150)$ | Silhouette↑ | High | FreeMatch / CPG Significant Low | Better Separation |

Observations: (i) FreeMatch-style SSL degrades to ~50% in long-tailed/arbitrary settings. (ii) SimPro collapses completely in extreme label-scarce scenarios ($N=40$), with accuracy ~16% (near random), proving its distribution estimation fails without sufficient supervision. (iii) SAGE consistently leads across all settings.

### Ablation Study
Key findings from removing modules (refer to text for specific values):

| Configuration | Key Observation | Explanation |
|---------------|-----------------|-------------|
| Full SAGE | High Acc + High Silhouette | Complete model efficacy. |
| w/o GRI | Degrades to baseline levels | Loss of high-order relationship supervision; noise pollutes representations. |
| w/o Simplex ETF (Learned Prototypes) | Clusters biased by majority classes | Geometric anchors are critical for anti-collapse. |
| w/o DRP + Auxiliary Branch | Acc drops; incorrect pseudo-labels spill over | Both isolation and weighting are essential. |
| Pseudo-label Correction Rate (GRI) | Monotonically increases during training | Confirms "relationships are more reliable than pseudo-labels." |

### Key Findings
- The observation that "**inter-sample relationships are more reliable than pseudo-labels**" holds across all settings, serving as the cornerstone of the methodology.
- The improvement in Silhouette coefficients brought by Simplex ETF anchors is significant, indicating that representation collapse in long-tail settings is indeed caused by a lack of geometric priors, not just pseudo-label noise.
- Decoupling $\phi_{cls}$ and $\phi_{aux}$ is a cost-effective design that restricts noisy gradients at almost zero cost.
- The method is insensitive to $\lambda$ and $\beta$. $\beta=5$ is a reasonable diffusion step—too few steps limit propagation, too many lead to a trivial steady-state distribution.

## Highlights & Insights
- **Paradigm Shift**: Shifts from "Estimate distribution $\rightarrow$ Generate pseudo-labels $\rightarrow$ Learn representations" to "Establish geometric anchors $\rightarrow$ Infer relationship graphs $\rightarrow$ Learn representations $\rightarrow$ Infer pseudo-labels." Reversing the signal chain makes stable geometric structure the primary axis, a rarity in ReaLTSSL literature.
- **ETF Anchors + Relational Embedding**: Interpreting ridge regression $\mathbf{a}_i=(\mathbf{z}_i\mathbf{P}^\top)(\mathbf{P}\mathbf{P}^\top+\lambda\mathbf{I})^{-1}$ as "projecting samples into a geometric coordinate system" is ingenious. it ensures an analytical solution for relational embeddings and naturally transfers the "equiangularity" of anchors to the relational space.
- **Distribution-agnostic Statistics**: The combination of $q_{max}$ and $q_{gap}$ avoids any assumptions about class distribution, providing a truly "protocol-agnostic" measure of reliability that transfers seamlessly to unknown distribution scenarios.

## Limitations & Future Work
- When the number of classes $C$ is large, the simplex ETF requires embedding dimension $d \geq C-1$, which targets a constraint for small models/projection heads.
- The computational cost of graph propagation $\beta$ scales with $|\mathbf{B}|^2$ per batch; scaling to ImageNet-level requires mini-batch propagation or sampling approximations.
- The method treats all unlabeled samples within a batch as graph nodes, potentially missing global manifold topology across batches or epochs.
- Geometric anchors are fixed; future work could explore "task-dependent adaptive anchor generation" combined with prompt-learning for multimodal SSL.

## Related Work & Insights
- **vs FreeMatch / SoftMatch**: While they rely on dynamic confidence thresholds for sample selection, SAGE uses GJS-independent geometric and structural supervision as an alternative, offering higher robustness to unknown distributions.
- **vs CPG / SimPro**: These methods explicitly/implicitly estimate unlabeled distributions and fail under extreme label scarcity. SAGE bypasses distribution estimation entirely.
- **vs Neural Collapse / DR-DSN**: Those works use ETF as targets for classifier weights. SAGE uses ETF as a "relational coordinate system" and "contrastive anchors," placing its application further upstream and making it more noise-resistant.
- **Inspiration**: In any task where pseudo-labels are unreliable (Open-set SSL, learning with noisy labels, long-tail detection), the strategy of "switching the signal source" is worth emulating—identifying signals more robust than labels (structure, geometry, relationships) as the dominant supervision.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The paradigm shift from "distribution estimation" to "structural inference" is a truly fresh perspective in ReaLTSSL.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five datasets + various imbalance ratios + 9 baselines, though ImageNet-scale validation is missing.
- **Writing Quality**: ⭐⭐⭐⭐ Logic flows well from diagnostic experiments to methodology; equations are slightly dense.
- **Value**: ⭐⭐⭐⭐ Establishes UniSSL as a more realistic setting with stable +8.52 pp gains, offering direct value for real-world SSL deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Understanding Self-Supervised Learning via Latent Distribution Matching](understanding_self-supervised_learning_via_latent_distribution_matching.md)
- [\[AAAI 2026\] Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning](../../AAAI2026/self_supervised/explanation-preserving_augmentation_for_semi-supervised_graph_representation_lea.md)
- [\[ICML 2026\] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)
- [\[AAAI 2026\] Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment](../../AAAI2026/self_supervised/let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-.md)
- [\[ICML 2026\] Can Local Learning Match Self-Supervised Backpropagation?](can_local_learning_match_self-supervised_backpropagation.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Understanding Self-Supervised Learning via Latent Distribution Matching](understanding_self-supervised_learning_via_latent_distribution_matching.md)
- [\[AAAI 2026\] Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning](../../AAAI2026/self_supervised/explanation-preserving_augmentation_for_semi-supervised_graph_representation_lea.md)
- [\[ICML 2026\] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)
- [\[AAAI 2026\] Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment](../../AAAI2026/self_supervised/let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-.md)
- [\[ICML 2026\] Can Local Learning Match Self-Supervised Backpropagation?](can_local_learning_match_self-supervised_backpropagation.md)

</div>

<!-- RELATED:END -->
