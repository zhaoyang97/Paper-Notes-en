---
title: >-
  [Paper Note] IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension
description: >-
  [ICML2026][Interpretability][Self-Supervised Learning] Ours proposes IdEst: a Minimum Spanning Tree dimension estimator $\mathrm{dim}_{\mathrm{MST}}$ to measure the intrinsic dimension (ID) of self-supervised representat…
tags:
  - "ICML2026"
  - "Interpretability"
  - "Self-Supervised Learning"
  - "Intrinsic Dimension"
  - "Minimum Spanning Tree"
  - "Representation Quality Evaluation"
  - "Unsupervised Model Selection"
date: 2026-05-08
content_hash: d0cb6f1bc7b6cf74
---

# IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension

**Conference**: ICML2026  
**arXiv**: [2606.03338](https://arxiv.org/abs/2606.03338)  
**Code**: To be confirmed  
**Area**: Self-Supervised Learning / Representation Evaluation  
**Keywords**: Self-Supervised Learning, Intrinsic Dimension, Minimum Spanning Tree, Representation Quality Evaluation, Unsupervised Model Selection

## TL;DR
Ours proposes IdEst: a Minimum Spanning Tree dimension estimator $\mathrm{dim}_{\mathrm{MST}}$ to measure the intrinsic dimension (ID) of self-supervised representations. Using this unlabeled geometric quantity as a proxy for downstream linear probe accuracy, it achieves Spearman $\rho \approx -0.8$ across 33 SSL models and enables unlabeled hyperparameter selection.

## Background & Motivation

**Background**: Self-supervised learning (SSL) has become the mainstream paradigm for learning representations from unlabeled data (SimCLR / DINO / I-JEPA / CLIP, etc.). However, the standard practice for evaluating these representations remains **linear probing**—training a linear classification head on a frozen feature extractor using a labeled downstream dataset (e.g., ImageNet). This protocol entails three major costs: high computational overhead, sensitivity to hyperparameters (learning rate, weight decay, epochs), and providing only a scalar score with almost no insight into the **geometric structure** of the representation.

**Limitations of Prior Work**: Existing "unsupervised proxy metrics" have limitations. $\alpha$-ReQ assumes the feature spectrum follows a power law and fails in cases of representation collapse (rank-deficient); RankMe calculates effective rank and is primarily designed for Joint-Embedding Architecture (JEA) methods, performing poorly on joint-predictive methods like I-JEPA; LiDAR performs well but **requires data augmentations used during SSL pre-training**, making it unfriendly to downstream users with only frozen representations.

**Key Challenge**: The goal is to evaluate SSL representation quality in an "unsupervised, cross-paradigm, frozen-feature-only" manner. Theoretically, existing results (Konz & Mazurowski, 2024) indicate that generalization error is approximately $\mathcal{L} \sim \mathcal{O}(K_L N_D^{-1/d})$, where $d$ is the intrinsic dimension of the representation manifold—this implies that **low ID is equivalent to high-quality representation**. However, in practical ID estimation, common methods like TwoNN and MLE rely on local isotropy + i.i.d. assumptions. These become severely unstable in SSL scenarios where $n \approx d$ is non-asymptotic and data points have strong dependencies (multiple views of the same image). Figure 2 shows that TwoNN even diverges to infinity on a simple 1D helix.

**Goal**: To find an ID estimator that remains stable under the "harsh conditions" of SSL (non-asymptotic + high-dimensional + strong dependencies) and verify if it truly reflects downstream performance.

**Key Insight**: Starting from the **Euclidean functional theory** of Costa & Hero (2006)—the growth rate of the Minimum Spanning Tree (MST) length is asymptotically proportional to the Rényi entropy of the data distribution. This derives an ID estimator $\mathrm{dim}_{\mathrm{MST}}$ that is robust to density changes and noise while being insensitive to ambient dimension.

**Core Idea**: Reverse estimate $d$ using the MST length scaling law $L(\mathrm{MST}(X_n)) \propto n^{(d-1)/d}$ and apply it to the frozen representations of any SSL model to obtain an unsupervised "representation quality meter"—IdEst.

## Method

### Overall Architecture
IdEst is an **evaluation protocol** rather than a new model. Input: A pre-trained SSL encoder $g$ and an unlabeled dataset $\mathcal{X}$ (ImageNet is recommended as a reference set). Process: (i) Use $g$ to extract frozen features for all samples (using [CLS] tokens where available; otherwise, performing average pooling on patch tokens, e.g., for I-JEPA), (ii) Run the $\mathrm{dim}_{\mathrm{MST}}$ algorithm on the feature space to estimate ID, (iii) Treat this scalar as the "representation quality score"—lower is better. The output is an ID value used to rank different SSL checkpoints, track training curves, or perform hyperparameter selection.

### Key Designs

1.  **MST Dimension Estimator $\mathrm{dim}_{\mathrm{MST}}$**:
    -   **Function**: Estimates the intrinsic dimension of the manifold where the frozen representation point cloud resides, without relying on local i.i.d. assumptions.
    -   **Mechanism**: Based on the Costa-Hero theorem, for $n$ points sampled i.i.d. from a compact Riemannian $d$-manifold, the total length of the MST almost surely satisfies $n^{-(d-1)/d} \cdot L(\mathrm{MST}(X_n)) \to C' \int f_X^{(d-1)/d}\,d\mathcal{H}$. In practice, different sub-sampling scales $n_i$ are taken, and a linear regression of $\log L(\mathrm{MST}(X_{n_i}))$ against $\log n_i$ is calculated; the slope $m$ yields $d = 1/(1-m)$.
    -   **Design Motivation**: Compared to TwoNN/MLE which only use **local information** from nearest neighbors, MST encodes both **local and global** connectivity structures, making it robust to noise and density variations. It is also equivalent to 0-dimensional persistent homology dimension, benefiting from stability guarantees provided by TDA (Chazal et al., 2014). On the 1D helix in Figure 2, $\mathrm{dim}_{\mathrm{MST}}$ stably converges to $d=1$, while TwoNN diverges.

2.  **IdEst Application Interface**:
    -   **Function**: Packages $\mathrm{dim}_{\mathrm{MST}}$ into a "plug-and-play" SSL evaluation metric, parallel to existing linear probe protocols.
    -   **Mechanism**: Calculate $\mathrm{dim}_{\mathrm{MST}}$ on the feature layer just before the classifier head in each SSL method's **official evaluation protocol**, ensuring complete comparability with linear probing. Two modes are provided: **Intra-Dataset** (calculating ID and accuracy on the same target dataset) and **Inter-Dataset** (calculating ID only on ImageNet to predict accuracy on iNat/CIFAR/kNN/ImageNet-v2). The latter proves ID reflects model properties rather than dataset bias.
    -   **Design Motivation**: To allow SSL practitioners "zero-cost" access—no re-training, no labels, and no original augmentations required; a single forward pass on frozen features is sufficient.

3.  **As an Unsupervised Hyperparameter Selector**:
    -   **Function**: Uses IdEst to replace the expensive "hyperparameter sweep + linear probe" cycle for unsupervised selection of parameters like learning rate, weight decay, and teacher temperature.
    -   **Mechanism**: Train SSL models for each candidate hyperparameter set, select the optimal checkpoint based on IdEst (minimizing ID), and perform only one final downstream evaluation on that checkpoint. Compared to using "ImageNet as an oracle," the number of linear probes is reduced from $K$ to 1.
    -   **Design Motivation**: Linear probing is one of the most computationally expensive parts of the SSL pipeline, especially with large hyperparameter grids and multiple datasets. Replacing it with a geometric quantity derived from a forward pass + MST reduces hyperparameter search costs by an order of magnitude without requiring downstream labels.

### Loss & Training
IdEst is a **post-hoc evaluation metric** and does not introduce new loss functions or training processes. MST construction uses classic Prim/Kruskal algorithms with $O(n^2)$ complexity or better. ID regression is a simple one-dimensional linear fit. The complete algorithm is detailed in Algorithm 1.

## Key Experimental Results

### Main Results: Correlation Across 33 Models
Includes 14 methods across 4 SSL paradigms (joint-embedding / joint-predictive / combinations / vision-language), 2 architectures (ResNet / ViT), and multiple scales (ViT-S to ViT-G), totaling 33 checkpoints.

| Setting | Reference Dataset | Target Dataset / Protocol | Kendall $\tau$ | Spearman $\rho$ |
|---------|-----------|------------------|----------------|-----------------|
| Intra-Dataset | ImageNet | ImageNet linear probe | $\approx -0.6$ | $\approx -0.8$ |
| Intra-Dataset | iNat-18 | iNat-18 linear probe | $\approx -0.6$ | $\approx -0.8$ |
| Intra-Dataset | SUN397 | SUN397 linear probe | $\approx -0.6$ | $\approx -0.8$ |
| Inter-Dataset | ImageNet | CIFAR-10 / CIFAR-100 / iNat | Strong Negative | Strong Negative |
| Alt. Protocol | ImageNet | kNN / ImageNet-v2 | Strong Negative | Strong Negative |

The negative sign is expected: **lower ID leads to higher downstream accuracy**.

### Ablation Study: Comparison with Existing Unsupervised Metrics

| Configuration / Metric | Needs Pre-training Augs | Performance on I-JEPA | Cross-Paradigm Robustness |
|------------|---------------------|--------------|---------------------|
| $\alpha$-ReQ | No | Fails when assumptions collapse | Weak |
| RankMe | No | Weak (designed for JEA) | JEA only |
| LiDAR | **Yes** | Strong | Strong but dep. on augs |
| **IdEst** | No | Strong | Strong across all four |

### Key Findings
- **ID is a Unified Geometric Descriptor Across SSL Paradigms**: Consistent negative correlations are observed across joint-embedding, joint-predictive, and vision-language methods, indicating it captures "how compact the representation is" rather than a fingerprint of a specific SSL loss.
- **Strong Inter-Dataset Transferability**: Calculating IdEst solely on ImageNet can predict performance rankings on iNat / CIFAR / kNN / ImageNet-v2, meaning **one reference dataset is sufficient** in practice.
- **Trackable Training Dynamics**: Figure 7 shows that in offline/online probing for VICReg, DINO, and I-JEPA, IdEst decreases monotonically with training epochs and closely tracks the rising linear probe accuracy (except for early stages < 10 epochs).
- **Effective Hyperparameter Selection**: In hyperparameter grids for learning rate, weight decay, teacher-student temperature, and target-context size, the checkpoints selected by IdEst fall at the higher end of the accuracy range provided by the fine-grained ImageNet Oracle (e.g., for DINO ViT-S, IdEst selects 65.5 while Oracle's upper bound is 69.1 and lower bound is 48.4).

## Highlights & Insights
- **Direct Bridge from Theory to Practice**: The direction "lower ID is better" is directly selected based on the Konz-Mazurowski scaling theorem $\mathcal{L} \sim N_D^{-1/d}$. The Costa-Hero MST asymptotic theorem then provides an estimator that works when $n \approx d$. The argumentation chain is clear, and the reasons for excluding failed estimators like TwoNN/MLE (reliance on local Poisson + i.i.d.) are solid.
- **MST = 0-Dimensional Persistent Homology**: The authors explicitly cite Adams et al. (2020) to equate $\mathrm{dim}_{\mathrm{MST}}$ with $\mathrm{dim}_{\mathrm{PH}}^0$ in TDA. This connection allows ID estimation to inherit TDA stability theory (Chazal et al., 2014), providing provable stability against noise and offering a natural entry point for future extensions using other TDA quantities (e.g., higher-dimensional PH).
- **Transferable to Other Domains**: Since the core is "estimating manifold dimension on frozen features," it can be applied to any unsupervised representations—LLM hidden states, graph embeddings, or speech encoders. Tulchinskii et al. (2023) validated similar ideas in LLMs; this work fills the gap in vision SSL.

## Limitations & Future Work
- **Ours Acknowledges**: Before 10 epochs, representations are not yet "unfolded," making IdEst less informative. MST construction on an $O(n^2)$ distance matrix remains costly for ultra-large feature sets (millions of samples).
- **Independent Observations**: (i) Models used are mostly ImageNet pre-trained; whether the negative correlation between ID and accuracy holds under **large domain gaps** (e.g., medical/satellite imagery) is unverified. (ii) Under very high ambient dimensions (e.g., 1536-dim for ViT-G), the MST regression slope might be flattened by noise; although multiple sub-samplings $n_i$ are used, a sensitivity analysis of the sampling schedule is missing. (iii) "Lower ID is better" assumes downstream classification; whether it remains monotonic for generative, retrieval, or dense prediction tasks remains to be seen.
- **Improvement Ideas**: Replacing MST with **kNN graph + spectral estimation** or performing lightweight dimensionality reduction (PCA / UMAP) before ID estimation might be more stable for ultra-large features like ViT-G. Furthermore, IdEst could be integrated as a **regularization term** during SSL training (minimizing ID) rather than just post-hoc evaluation to see if it directly improves performance.

## Related Work & Insights
- **vs RankMe**: RankMe measures effective rank, essentially a proxy for **linear separability**, designed for joint-embedding (to prevent collapse). IdEst measures **geometric manifold dimension**, holding across JEA / I-JEPA / CLIP with broader coverage.
- **vs LiDAR**: LiDAR calculates the rank of the LDA matrix for SSL surrogate tasks and shows strong correlation, but **must have original augmentations**, which is unfriendly to users with only frozen models. IdEst looks exclusively at frozen features.
- **vs TwoNN / MLE-based ID**: Previous ID estimation worked on supervised CNNs (Ansuini et al., 2019) but fails in SSL due to $n \approx d$ conditions and view dependencies, as demonstrated by the helix counter-example in Figure 2. The MST estimator represents a fundamental shift in the underlying mechanism.
- **vs $\alpha$-ReQ**: $\alpha$-ReQ looks at spectral decay rates and fails during representation collapse; IdEst provides meaningful ID even on rank-deficient representations.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing MST dimension estimation to SSL evaluation is a well-known tool transfer, but doing so with theoretical analysis and cross-paradigm verification is a first.
- Experimental Thoroughness: ⭐⭐⭐⭐ 33 models × 4 paradigms × multiple datasets × multiple protocols (linear probe / kNN / ImageNet-v2 / fine-grained), with practical data for hyperparameter selection, is highly convincing.
- Writing Quality: ⭐⭐⭐⭐ The logic chain from motivation to theory to estimator limitations to the new estimator and experiments is very smooth. Using the helix counter-example against TwoNN is a brilliant design.
- Value: ⭐⭐⭐⭐ Provides SSL practitioners with an unlabeled, cross-paradigm, and inexpensive evaluation tool; the computational savings for hyperparameter search are tangible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation](interpretable_self-supervised_learning_via_representer_landmarks_and_nyström_app.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/interpretability/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICML 2026\] Learning Coherent Representations: A Topological Approach to Interpretability](learning_coherent_representations_a_topological_approach_to_interpretability.md)
- [\[NeurIPS 2025\] Dataset Distillation for Pre-Trained Self-Supervised Vision Models](../../NeurIPS2025/interpretability/dataset_distillation_for_pre-trained_self-supervised_vision_models.md)
- [\[CVPR 2026\] RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation](../../CVPR2026/interpretability/riskprop_collision-anchored_self-supervised_risk_propagation_for_early_accident_.md)

</div>

<!-- RELATED:END -->
