---
title: >-
  [Paper Note] Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization
description: >-
  [ICML 2026][AI Safety][Federated Domain Adaptation] Addressing the limitation that existing federated unsupervised multi-source domain adaptation (UMDA) methods can only handle 2–6 sources—becoming unstable or computatio…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Federated Domain Adaptation"
  - "Multi-Source Domain"
  - "Group-Level Discrepancy"
  - "Negative Transfer"
  - "Digit-18"
date: 2026-05-08
content_hash: c6a0605e7679d4ec
---

# Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization

**Conference**: ICML 2026  
**arXiv**: [2510.08150](https://arxiv.org/abs/2510.08150)  
**Code**: Not specified in the paper (likely on GitHub, check author homepage)  
**Area**: Federated Learning / Domain Adaptation / Privacy-Preserving Machine Learning  
**Keywords**: Federated Domain Adaptation, Multi-Source Domain, Group-Level Discrepancy, Negative Transfer, Digit-18

## TL;DR
Addressing the limitation that existing federated unsupervised multi-source domain adaptation (UMDA) methods can only handle 2–6 sources—becoming unstable or computationally infeasible as the number of sources increases—the authors propose GALA: all sources are randomly divided into small groups, and group-wise prediction distribution discrepancies are minimized (reducing $O(N^2)$ pairwise alignment to linear complexity). Additionally, a centroid+temperature-based similarity weighting is stacked to select sources truly close to the target domain. On the newly constructed Digit-18 (18 sources) benchmark, the method converges stably and outperforms all baselines.

## Background & Motivation

**Background**: Unsupervised multi-source domain adaptation (UMDA) leverages multiple labeled source domains $\{D_S^n\}_{n=1}^N$ to train a model for transfer to an unlabeled target domain $D_T$. In privacy-sensitive scenarios (e.g., healthcare, finance), data cannot be centralized, leading to federated/decentralized UMDA methods. For example, FADA (Peng et al. 2020) uses adversarial training, FACT (Schrod et al. 2025) leverages inter-domain discrepancy, and KD3A (Feng et al. 2021) employs consensus alignment.

**Limitations of Prior Work**: (1) Most methods are only validated on 2–6 source domains; (2) FACT, though scalable, aligns only a single source pair per step, leading to **variance explosion and unstable convergence** as the number of sources increases; (3) KD3A requires per-domain optimization and divergence computation on the target for each source, causing **computational cost to grow exponentially** with the number of sources—becoming infeasible for 10+ sources; (4) The community **lacks a truly heterogeneous, sufficiently large-source benchmark**—most works "simulate" multi-source by splitting a single dataset, failing to reflect real distributional differences.

**Key Challenge**: The ideal approach for cross-source alignment is to compute pairwise $\mathcal{H}\Delta\mathcal{H}$ divergence $\sum_n w_n \frac{1}{2} d_{\mathcal{H}\Delta\mathcal{H}}(D_S^n, D_T)$, which is $O(N^2)$; reducing to "one pair per step" leads to excessive variance. There is a need for an algorithm that **preserves the global alignment objective, scales linearly, and dynamically weights sources to exclude those causing negative transfer**.

**Goal**: (i) Design a linear-complexity, low-variance multi-source discrepancy minimization objective; (ii) Automatically assign weights to sources so that those close to the target dominate training, while distant sources are down-weighted; (iii) Provide a truly heterogeneous, sufficiently large-source evaluation benchmark.

**Key Insight**: The authors approach the problem from a "group rather than individual" perspective—rather than precisely approximating all pairwise discrepancies, they use random grouping and group-wise prediction alignment, akin to a minibatch version of global alignment estimation; weighting draws on contrastive learning's temperature softmax, determined by the proximity of source centroids to the target centroid.

**Core Idea**: Use **Inter-Group Discrepancy (IGD)** to compress $O(N^2)$ pairwise alignment into $O(N)$ group-level alignment, and apply **temperature-scaled centroid-based weighting** for dynamic source selection—the two components together form GALA (Grouping-based Adaptive Learning).

## Method

### Overall Architecture
Federated setting: $N$ source clients each hold $\{D_S^n\} = \{(x_i^n, y_i^n)\}_{i=1}^{K_n}$, and the target client holds $D_T = \{x_i^T\}_{i=1}^{K_T}$. Each source trains a local feature extractor $G$ and classifier $F$, with the server aggregating to obtain the global $h = F \circ G$. In each round: (1) Each source performs supervised training locally and uploads updates; (2) The target receives all source updates or logits, performs random grouping for IGD alignment and centroid-based weighting; (3) The server aggregates globally and distributes new parameters. The key innovations are on the target side—IGD and weighting strategies—independent of the specific feature extractor.

### Key Designs

1. **Inter-Group Discrepancy (IGD)**:

    - **Function**: Reduces the $O(N^2)$ pairwise discrepancy minimization in traditional UMDA to linear-complexity group-level discrepancy minimization, while maintaining low variance.
    - **Mechanism**: In each mini-batch, the $N$ sources are **randomly divided into $G$ disjoint groups** $\mathcal{G}_1, \dots, \mathcal{G}_G$. Each group aggregates the predictions of its sources on target unlabeled samples $x^T$: $\bar{p}_g(x^T) = \frac{\sum_{n \in \mathcal{G}_g} w_n p_n(x^T)}{\sum_{n \in \mathcal{G}_g} w_n}$. The IGD loss is the sum of pairwise discrepancies between group prediction distributions, e.g., $\mathcal{L}_{IGD} = \sum_{g \neq g'} D(\bar{p}_g, \bar{p}_{g'})$ (with $D$ as KL or L2). Since there are only $O(G)$ groups (with $G$ a small constant), complexity drops from $O(N^2)$ to $O(G^2) = O(1)$ relative to $N$; aggregating multiple sources per group also reduces variance compared to FACT's "single-source pairwise" alignment. Random grouping each round ensures the expectation matches global alignment.
    - **Design Motivation**: Direct pairwise alignment of $N$ sources is the UMDA gold standard but not scalable; FACT's shortcut of single-pair alignment leads to high variance; IGD is a compromise—group-level "macro-alignment" approximates the global objective, with randomization and intra-group averaging reducing variance. Theoretically, group-wise alignment is an unbiased estimator of the global alignment objective.

2. **Temperature-Scaled Centroid Similarity Weighting**:

    - **Function**: Dynamically assigns weights $w_n$ to sources, allowing those close to the target distribution to dominate training, while distant sources are down-weighted to avoid negative transfer.
    - **Mechanism**: In each round, compute the **centroid** of each source and the target in feature space: $c_n = \frac{1}{|D_S^n|}\sum_{x \in D_S^n} G(x)$, $c_T = \frac{1}{|D_T|}\sum_{x \in D_T} G(x)$. Similarity $\text{sim}(c_n, c_T)$ (typically negative distance or cosine) is passed through a temperature softmax: $w_n = \frac{\exp(\text{sim}(c_n, c_T) / \tau)}{\sum_m \exp(\text{sim}(c_m, c_T) / \tau)}$ to obtain normalized weights. Temperature $\tau$ controls sharpness: as $\tau \to 0$, it approaches hard selection (only the nearest source is chosen); as $\tau \to \infty$, it approaches uniform weighting.
    - **Design Motivation**: Theoretically (see Corollary 3.1), the federated UMDA generalization bound includes $\sum_n w_n \frac{1}{2} d_{\mathcal{H}\Delta\mathcal{H}}(D_S^n, D_T)$, requiring $w_n$ to be inversely related to source-target distance; centroid similarity approximates $\mathcal{H}$-divergence, and temperature softmax enables smooth selection—both theoretically sound and practically feasible. Fixed uniform weights can allow noisy sources to degrade performance (negative transfer) as the number of sources increases; dynamic weighting directly addresses this.

3. **Digit-18 Benchmark (Task Contribution, not Method)**:

    - **Function**: Fills the gap of a "truly multi-source and heterogeneous" UMDA testbed.
    - **Mechanism**: Collects 18 digit recognition datasets, covering synthetic (generated digits) and real (MNIST, SVHN, USPS, MNIST-M, etc.) domain shifts, with each client holding one dataset; the task is unified as 10-class digit recognition; for evaluation, one is used as the target and the remaining 17 as sources. This is much more challenging than previous toy benchmarks like Digit-5, where all sources are digits.
    - **Design Motivation**: Existing federated UMDA experiments rely on "copying + adding noise" to simulate multi-source; the authors assemble truly heterogeneous sources—only this can truly reveal whether methods collapse as the number of sources increases.

### Loss & Training

Total loss: $\mathcal{L} = \sum_n w_n \mathcal{L}_{CE}(D_S^n) + \lambda \mathcal{L}_{IGD}$. Each source locally minimizes the weighted supervised CE; the target uses all source predictions for IGD alignment; weights $w_n$ are recalculated each round using centroid similarity. Optimizer: SGD/Adam; hyperparameters $\lambda$ and $\tau$ are grid-searched on a small validation subset. The framework is naturally parallelizable—each source trains independently, and the server only performs aggregation and IGD.

## Key Experimental Results

### Main Results

The paper compares on standard UMDA benchmarks (Digit-5, Office-Caltech10, DomainNet) and the newly proposed Digit-18:

| Benchmark | Method | Key Observations |
|---|---|---|
| Digit-5 (5 sources) | FACT / KD3A / GALA | All perform similarly, KD3A slightly higher, GALA comparable—shows GALA does not lag behind with few sources |
| Digit-18 (17 sources → 1 target) | FACT | **Does not converge**, accuracy drops to random guessing on several targets |
| Digit-18 | KD3A | Exponential growth in computation makes training with 17 sources **computationally infeasible** (paper states "computationally infeasible") |
| Digit-18 | GALA | Converges stably, average accuracy significantly higher than other runnable baselines |
| Office-Caltech10 / DomainNet | GALA | Matches or exceeds SOTA |

### Ablation Study

| Configuration | Key Metric Change | Notes |
|---|---|---|
| Full GALA (IGD + weighting) | Full effect | Baseline |
| w/o IGD (direct full pairwise) | Computation explodes, cannot run with many sources | Shows IGD is key for scalability |
| w/o weighting (uniform $w_n = 1/N$) | Performance drops, especially on Digit-18 | Shows negative transfer from noisy sources is severe with many sources |
| Different group numbers $G$ | Moderate $G$ (e.g., 3–4) is optimal; too small degenerates to global averaging, too large to FACT-style high variance | Shows value of group-level granularity trade-off |
| Different temperatures $\tau$ | Too small → overfits to one source; too large → degenerates to uniform; intermediate values are most stable | Shows necessity of soft selection |

### Key Findings
- As the number of sources increases from 5 to 17, FACT fails to converge (high variance), and KD3A's training time grows exponentially—these contrast experiments are the paper's strongest selling points.
- **Centroid weighting is crucial for high-diversity sources**: Digit-18 includes synthetic digit outliers; without weighting, these drag down performance; with weighting, their influence is automatically reduced.
- IGD loss curves are much smoother than FACT, with variance reduced by an order of magnitude (as shown in the paper).
- On standard small-source benchmarks, GALA is not much stronger than KD3A, but its advantage lies in "scalability and computational efficiency," which is of engineering value.

## Highlights & Insights
- **"Grouping" is an underrated technique**: For UMDA problems with high pairwise alignment complexity, random grouping provides an unbiased estimator, akin to applying the minibatch concept at the domain level—this trick can be transferred to any $O(N^2)$ divergence alignment objective (meta-learning, multi-task balancing, etc.).
- **Centroid + temperature softmax is a lightweight "adaptive source selection"**: Avoids KD3A's expensive pairwise computations and is more flexible than fixed uniform weighting; naturally friendly to federated settings (centroids can be computed locally, no data leakage).
- **The Digit-18 dataset itself is a community contribution**: Previous UMDA tests were too toy-like; this truly heterogeneous benchmark will force future work to demonstrate scalability.
- **Theory → Algorithm → Experiment is tightly integrated**: The federated UMDA generalization bound (Corollary 3.1) motivates "weights should be inversely proportional to source-target distance," instantiated via centroid softmax—motivation and method are well aligned.

## Limitations & Future Work
- **Centroid similarity is a coarse approximation of $\mathcal{H}$-divergence**: When source distributions are multi-modal or the target covers multiple modes, a single centroid may be insufficient; mixture centroids or clustering may be needed.
- Although IGD's random grouping is unbiased in expectation, **variance within a single round still exists**; the interaction between group number $G$ and batch size is not fully discussed.
- Only validated on digit recognition, Office-Caltech, and DomainNet—**not tested on large models/high-dimensional features (e.g., ResNet-50/ViT features)**.
- **Communication efficiency is not considered**—per-round uploads of logits and weight-related statistics are already non-trivial with 17 sources; communication costs for truly large-scale federated settings (hundreds or thousands of clients) are not analyzed.
- Assumes all source domains share the same class space (C-way classification), not extended to partial or open-set scenarios.
- The paper does not discuss privacy guarantees—centroids actually leak second-order statistics of source distributions, which may not satisfy strict DP requirements.

## Related Work & Insights
- **vs FACT (Schrod et al. 2025)**: FACT also uses inter-domain discrepancy and claims scalability, but aligns only single source pairs per step, leading to high variance and poor convergence; IGD uses group aggregation to reduce variance, and centroid weighting addresses the "source selection" problem absent in FACT.
- **vs KD3A (Feng et al. 2021)**: KD3A is currently the strongest decentralized UMDA, but requires per-domain divergence computation on the target, **complexity $O(N)$ but with large constants**; GALA reduces target-side computation to $O(G)$, supporting 17+ sources.
- **vs FADA (Peng et al. 2020)**: The first federated UMDA using adversarial training, but adversarial objectives are unstable with many sources; GALA avoids adversarial instability by aligning prediction distributions.
- **vs MDMGB / SFDA (Wang et al. 2022)**: SFDA also performs source weighting but uses pseudo-labels and information maximization, with lower performance than SOTA; GALA's centroid weighting is more lightweight.

## Rating
- Novelty: ⭐⭐⭐⭐ Using random grouping to replace pairwise alignment is a clean idea, and centroid weighting is a reasonable instantiation; not disruptive innovation, but an effective solution to a neglected pain point.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on standard benchmarks and the self-constructed Digit-18; lacks experiments on large models/large class counts/real federated deployments.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation and algorithm description are clear; theoretical part (Corollary 3.1) is not original but appropriately applied.
- Value: ⭐⭐⭐⭐ First to explicitly target "scalability of federated UMDA" and provide a feasible solution, plus the Digit-18 benchmark, with lasting impact on the federated learning + DA intersection community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Fair Dataset Distillation via Cross-Group Barycenter Alignment](fair_dataset_distillation_via_cross-group_barycenter_alignment.md)
- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](../../CVPR2026/ai_safety/feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[CVPR 2025\] Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing](../../CVPR2025/ai_safety/optimal_transport-guided_source-free_adaptation_for_face_anti-spoofing.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](../../AAAI2026/ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[CVPR 2026\] Domain-Skewed Federated Learning with Feature Decoupling and Calibration](../../CVPR2026/ai_safety/domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)

</div>

<!-- RELATED:END -->
