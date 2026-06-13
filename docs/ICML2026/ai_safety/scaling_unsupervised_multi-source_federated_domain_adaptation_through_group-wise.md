---
title: >-
  [Paper Note] Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization
description: >-
  [ICML 2026][AI Safety][Federated Domain Adaptation] Addressing the limitation of existing Federated Multi-Source Unsupervised Domain Adaptation (UMDA) methods—which handle only 2–6 sources and suffer from training instab…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Federated Domain Adaptation"
  - "Multi-Source Domains"
  - "Group-level Discrepancy"
  - "Negative Transfer"
  - "Digit-18"
date: 2026-05-08
content_hash: 3ab692da8e30c568
---

# Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization

**Conference**: ICML 2026  
**arXiv**: [2510.08150](https://arxiv.org/abs/2510.08150)  
**Code**: Not explicitly stated (presumably GitHub; check author's homepage)  
**Area**: Federated Learning / Domain Adaptation / Privacy-Preserving Machine Learning  
**Keywords**: Federated Domain Adaptation, Multi-Source Domains, Group-level Discrepancy, Negative Transfer, Digit-18

## TL;DR
Addressing the limitation of existing Federated Multi-Source Unsupervised Domain Adaptation (UMDA) methods—which handle only 2–6 sources and suffer from training instability or computational explosion as the number of sources increases—the authors propose GALA. GALA randomly partitions sources into small groups and minimizes the discrepancy of prediction distributions between groups (compressing $O(N^2)$ pairwise alignments to linear complexity). This is coupled with a similarity weighting mechanism based on centroids and temperature to identify sources truly close to the target domain. It achieves stable convergence on the newly established Digit-18 (18 sources) benchmark, significantly outperforming baselines.

## Background & Motivation

**Background**: Unsupervised Multi-Source Domain Adaptation (UMDA) utilizes multiple labeled source domains $\{D_S^n\}_{n=1}^N$ to learn a model that transfers to an unlabeled target domain $D_T$. In privacy-sensitive scenarios (e.g., healthcare, finance), data cannot be centralized, leading to federated/decentralized UMDA methods like FADA (Peng et al. 2020) using adversarial training, FACT (Schrod et al. 2025) using inter-domain discrepancy, and KD3A (Feng et al. 2021) using consensus alignment.

**Limitations of Prior Work**: (1) Most methods are validated only on 2–6 source domains; (2) While FACT is scalable, it aligns only a single source pair at each step, leading to **high variance and unstable convergence** with many sources; (3) KD3A requires per-domain optimization and divergence calculation on the target for every source, causing **computational costs to grow exponentially** with the number of sources (becoming infeasible at 10+ sources); (4) The community **lacks a truly heterogeneous benchmark with sufficient sources**—most rely on splitting the same dataset into segments, failing to reflect real distribution shifts.

**Key Challenge**: The ideal approach for cross-source alignment is calculating pairwise $\mathcal{H}\Delta\mathcal{H}$ divergence $\sum_n w_n \frac{1}{2} d_{\mathcal{H}\Delta\mathcal{H}}(D_S^n, D_T)$, which is $O(N^2)$. However, degrading to "one pair per step" introduces excessive variance. An algorithm is needed that **retains global alignment objectives, scales linearly, and dynamically weights sources to exclude negative transfer**.

**Goal**: (i) Design a multi-source discrepancy minimization objective with linear complexity and low variance; (ii) Automatically weight sources to let target-near sources dominate training while preventing target-far sources from hindering performance; (iii) Provide a benchmark with real heterogeneity and a sufficient number of sources.

**Key Insight**: The authors approach the problem from the perspective of "groups rather than individuals." Instead of precisely approximating all pairwise discrepancies, they use random grouping and inter-group prediction distribution alignment—effectively a minibatch version of global alignment. Weights are determined using a temperature-scaled softmax based on the distance between "source centroids and the target centroid."

**Core Idea**: Use **Inter-Group Discrepancy (IGD)** to compress $O(N^2)$ pairwise alignment into $O(N)$ group-level alignment, combined with **temperature-scaled centroid-based weighting** for dynamic source selection. The combined framework is called GALA (Grouping-based Adaptive Learning).

## Method

### Overall Architecture
Federated setting: $N$ source clients hold $\{D_S^n\} = \{(x_i^n, y_i^n)\}_{i=1}^{K_n}$, and the target client holds $D_T = \{x_i^T\}_{i=1}^{K_T}$. Each source trains a local feature extractor $G$ and classifier $F$; the server aggregates them into a global $h = F \circ G$. Within each round: (1) Each source performs local supervised training on labeled data and uploads updates; (2) The target client receives updates or logits from all sources, performs random-grouping IGD alignment, and calculates centroid weights; (3) New parameters are broadcast after global aggregation. The innovations are centered on the IGD and weighting strategies at the target client, independent of specific feature extractors.

### Key Designs

1.  **Inter-Group Discrepancy (IGD)**:
    - **Function**: Compresses the traditional $O(N^2)$ pairwise discrepancy minimization in UMDA into linear-complexity group-level discrepancy minimization while maintaining low variance.
    - **Mechanism**: In each mini-batch, $N$ sources are **randomly partitioned into $G$ disjoint groups** $\mathcal{G}_1, \dots, \mathcal{G}_G$. Each group aggregates predictions of all its members on target unlabeled samples $x^T$: $\bar{p}_g(x^T) = \frac{\sum_{n \in \mathcal{G}_g} w_n p_n(x^T)}{\sum_{n \in \mathcal{G}_g} w_n}$. The IGD loss is the sum of pairwise discrepancies between group prediction distributions, e.g., $\mathcal{L}_{IGD} = \sum_{g \neq g'} D(\bar{p}_g, \bar{p}_{g'})$ (where $D$ is KL or L2). Since there are only $O(G)$ groups ($G$ being a small constant), complexity drops from $O(N^2)$ to $O(G^2) = O(1)$ relative to $N$. Furthermore, aggregating multiple sources per group significantly reduces variance compared to FACT's "single source pair" alignment. Re-grouping each round makes the objective equivalent to global alignment in expectation.
    - **Design Motivation**: Pairwise alignment is the gold standard for UMDA but is non-scalable; FACT's single-pair shortcut suffers from variance. IGD serves as a compromise—using group-level "macro-alignment" to approximate the global objective, utilizing randomization and intra-group averaging to suppress variance. Theoretically, inter-group alignment is an unbiased estimator of the global alignment goal.

2.  **Temperature-Scaled Centroid Similarity Weighting**:
    - **Function**: Dynamically assigns a weight $w_n$ to each source, allowing sources close to the target distribution to dominate training while weakening distant sources to avoid negative transfer.
    - **Mechanism**: In each round, the **centroid** of each source and the target domain is calculated in the feature space: $c_n = \frac{1}{|D_S^n|}\sum_{x \in D_S^n} G(x)$ and $c_T = \frac{1}{|D_T|}\sum_{x \in D_T} G(x)$. Similarity $\text{sim}(c_n, c_T)$ (typically negative distance or cosine similarity) is passed through a temperature-scaled softmax: $w_n = \frac{\exp(\text{sim}(c_n, c_T) / \tau)}{\sum_m \exp(\text{sim}(c_m, c_T) / \tau)}$. The temperature $\tau$ controls sharpness: $\tau \to 0$ approaches hard selection, while $\tau \to \infty$ approaches uniform weighting.
    - **Design Motivation**: Theoretically (Corollary 3.1), the generalization bound for federated UMDA requires weights $w_n$ to be inversely related to the "source-target distance." Using centroid similarity to approximate $\mathcal{H}$-divergence followed by temperature-scaled softmax is theoretically sound and computationally efficient. Uniform weighting fails as the number of sources increases because noisy sources degrade performance (negative transfer).

3.  **Digit-18 Benchmark (Novelty)**:
    - **Function**: Provides a "truly multi-source and heterogeneous" UMDA testbed.
    - **Mechanism**: Collects 18 digit recognition datasets, covering synthetic/generated digits and real data (MNIST, SVHN, USPS, MNIST-M, etc.) with various domain shifts. Each client holds one dataset. The task is unified as 10-class digit recognition. In evaluation, 1 dataset acts as the target and the other 17 as sources. This is significantly more challenging than toys like Digit-5.
    - **Design Motivation**: Existing federated UMDA experiments often forge multi-source data via replication and noise. The authors assemble real heterogeneous sources to reveal whether methods collapse under high source counts.

### Loss & Training
Total loss: $\mathcal{L} = \sum_n w_n \mathcal{L}_{CE}(D_S^n) + \lambda \mathcal{L}_{IGD}$. Each source minimizes weighted supervised cross-entropy locally. The target utilizes predictions from all sources for IGD alignment. Weights $w_n$ are recalculated each round via centroid similarity. Optimization uses SGD/Adam, with hyperparameters $\lambda$ and $\tau$ tuned via grid search. The framework is naturally parallelizable.

## Key Experimental Results

### Main Results
The paper compares GALA against baselines on standard UMDA benchmarks (Digit-5, Office-Caltech10, DomainNet) and the new Digit-18:

| Benchmark | Method | Key Observation |
|---|---|---|
| Digit-5 (5 sources) | FACT / KD3A / GALA | Performance is similar across all; GALA matches the high performance of KD3A. |
| Digit-18 (17 sources) | FACT | **Fails to converge**; accuracy drops to random guess levels on several targets. |
| Digit-18 | KD3A | Exponential computational growth makes **single-round training time infeasible** with 17 sources. |
| Digit-18 | GALA | Converges stably; average accuracy is significantly higher than other runnable baselines. |
| Office-Caltech10 / DomainNet | GALA | Competitive with or exceeds SOTA. |

### Ablation Study

| Configuration | Metric Change | Explanation |
|---|---|---|
| Full GALA (IGD + Weighting) | Baseline Performance | Best overall effect. |
| w/o IGD (Full pairwise) | Computational explosion | Confirms IGD is the key to scalability. |
| w/o Weighting ($w_n = 1/N$) | Performance degradation | Especially evident in Digit-18; non-suitable sources cause negative transfer. |
| Different group counts $G$ | Optimal at 3–4 | Too small $G$ reverts to global averaging; too large $G$ reverts to high-variance FACT-style. |
| Different temperature $\tau$ | Optimal middle range | Too small $\tau$ leads to overfitting a single source; too large $\tau$ leads to uniform weighting. |

### Key Findings
- When sources increase from 5 to 17, FACT fails (high variance) and KD3A training time grows exponentially; these comparisons are the strongest selling points.
- **Centroid weighting is critical for high-diversity sources**: Outliers in Digit-18 (e.g., synthetic digits) degrade performance without weighting; the mechanism automatically lowers their priority.
- The IGD loss curve is smoother than FACT, with variance reduced by over an order of magnitude.
- GALA's primary advantage lies in being "scalable and computationally cheap" while maintaining performance on small-scale benchmarks.

## Highlights & Insights
- **"Grouping" is an underrated technique**: In UMDA problems where pairwise alignment is expensive, using random grouping for unbiased estimation applies the minibatch philosophy to the domain level.
- **Centroid + Temperature Softmax is highly lightweight**: It avoids expensive pairwise computations of KD3A while being more flexible than fixed weights; it is inherently privacy-friendly (centroids can be calculated locally).
- **Digit-18 dataset is a significant contribution**: It forces subsequent UMDA research to prove scalability beyond toy examples.
- **Solid Theory-to-Algorithm loop**: The algorithm is derived directly from federated UMDA generalization bounds (Corollary 3.1), ensuring strong theoretical alignment.

## Limitations & Future Work
- **Centroid similarity is a coarse approximation**: It may lack precision when source distributions are multi-modal or when target distributions cover multiple modes.
- While IGD is unbiased in expectation, **intra-round variance still exists**; the interaction between $G$ and batch size requires further discussion.
- Validation is limited to image datasets; **performance on large-scale models (ViT) or extremely high-dimensional features** remains untested.
- **Communication efficiency**: Uploading logits and weights for 17+ sources is non-trivial; costs for thousands of clients were not analyzed.
- Assumes a shared label space (C-way classification); not yet extended to partial or open-set scenarios.
- Potential privacy issues: Centroids may leak second-order statistical information of source distributions, potentially violating strict Differential Privacy.

## Related Work & Insights
- **vs. FACT (Schrod et al. 2025)**: FACT also claims scalability but suffers from high variance due to single-pair alignment; IGD suppresses variance through grouping, and centroid weighting addresses source selection missing in FACT.
- **vs. KD3A (Feng et al. 2021)**: KD3A is powerful but requires per-domain divergence calculations on the target ($O(N)$ with large constants); GALA reduces target computation to $O(G)$, facilitating 17+ sources.
- **vs. FADA (Peng et al. 2020)**: GALA avoids the instability of multi-source adversarial training by using prediction distribution alignment.
- **vs. MDMGB / SFDA (Wang et al. 2022)**: SFDA uses pseudo-labels for weighting; GALA's centroid weighting is more lightweight and theoretically grounded.

## Rating
- Novelty: ⭐⭐⭐⭐ (Random grouping for pairwise alignment is a clean idea).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Solid validation on the new Digit-18).
- Writing Quality: ⭐⭐⭐⭐ (Clear problem motivation and derivation).
- Value: ⭐⭐⭐⭐ (Addresses a previously ignored pain point in federated UMDA).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Unsupervised Open-Set Graph Domain Adaptation via Dual Reprogramming](../../NeurIPS2025/ai_safety/towards_unsupervised_open-set_graph_domain_adaptation_via_dual_reprogramming.md)
- [\[ICML 2026\] Optimal Transport under Group Fairness Constraints](optimal_transport_under_group_fairness_constraints.md)
- [\[ICML 2026\] Fair Dataset Distillation via Cross-Group Barycenter Alignment](fair_dataset_distillation_via_cross-group_barycenter_alignment.md)
- [\[ICML 2026\] TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting](timeguard_channel-wise_pool_training_for_backdoor_defense_in_time_series_forecas.md)
- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Optimal Transport under Group Fairness Constraints](optimal_transport_under_group_fairness_constraints.md)
- [\[ICML 2026\] Fair Dataset Distillation via Cross-Group Barycenter Alignment](fair_dataset_distillation_via_cross-group_barycenter_alignment.md)
- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](../../CVPR2026/ai_safety/feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[CVPR 2025\] Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing](../../CVPR2025/ai_safety/optimal_transport-guided_source-free_adaptation_for_face_anti-spoofing.md)
- [\[ICML 2026\] TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting](timeguard_channel-wise_pool_training_for_backdoor_defense_in_time_series_forecas.md)

</div>

<!-- RELATED:END -->
