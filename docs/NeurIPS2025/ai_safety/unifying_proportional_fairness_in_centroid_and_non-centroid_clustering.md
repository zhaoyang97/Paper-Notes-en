---
title: >-
  [Paper Note] Unifying Proportional Fairness in Centroid and Non-Centroid Clustering
description: >-
  [NeurIPS 2025][AI Safety][fair clustering] This paper unifies the study of proportional fairness in centroid and non-centroid clustering under a single "semi-centroid clustering" framework…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "fair clustering"
  - "proportional fairness"
  - "core stability"
  - "semi-centroid clustering"
  - "approximation algorithms"
date: 2026-05-08
content_hash: 95c7dee36d921791
---

# Unifying Proportional Fairness in Centroid and Non-Centroid Clustering

**Conference**: NeurIPS 2025
**arXiv**: [2601.00447](https://arxiv.org/abs/2601.00447)
**Code**: None
**Area**: AI Safety / Fairness
**Keywords**: fair clustering, proportional fairness, core stability, semi-centroid clustering, approximation algorithms

## TL;DR
This paper unifies the study of proportional fairness in centroid and non-centroid clustering under a single "semi-centroid clustering" framework, establishes an impossibility theorem showing the two cannot be simultaneously achieved, and designs novel algorithms that attain constant-factor core guarantees under dual metric loss.

## Background & Motivation

Fair clustering demands proportional fairness guarantees for **every possible group**, rather than only predefined protected attributes. Prior work studied proportional fairness in two separate paradigms:

**Centroid clustering** (Chen et al., 2019): each agent's cost is determined by its distance to the cluster representative (centroid). Canonical application: facility location, where residents prefer their assigned facility to be nearby.

**Non-centroid clustering** (Caragiannis et al., 2024): each agent's cost is determined by its distance to the farthest data point in its cluster. Canonical application: federated learning, where participants prefer cluster members whose data distributions are similar to their own.

**Key Challenge**: In many real-world applications, users care about both types of loss simultaneously. In facility location, residents may prefer both a nearby facility (centroid loss) and neighbors who share the same community (non-centroid loss). In federated learning, participants care both about model quality (centroid) and similarity of peers' data distributions (non-centroid). Moreover, the two losses may be defined over entirely different metric spaces—when a teacher forms groups, students' preferences over project topics (centroid) and over group members (non-centroid) may be grounded in completely different similarity measures.

**Impossibility Result**: The paper first establishes an important impossibility theorem (Theorem 1): no clustering can simultaneously maintain finite-approximation core stability with respect to both centroid and non-centroid losses. This rules out directly combining existing algorithms and necessitates a new joint algorithmic design.

## Method

### Overall Architecture

The paper proposes the **semi-centroid clustering** model, where each agent $i$ in cluster $(C_t, x_t)$ incurs loss $\ell_i(C_t, x_t)$ that depends jointly on: (1) the cluster membership $C_t$ (who is in the cluster) and (2) the cluster centroid $x_t$ (where the centroid is located).

Two structured loss function families are studied:
- **Dual Metric Loss**: $\ell_i(\mathcal{X}) = \max_{j \in C} d^m(i,j) + d^c(i, x_t)$, defined over two distinct metric spaces.
- **Weighted Single Metric Loss**: $\ell_i(\mathcal{X}) = \lambda \cdot \max_{j \in C} d(i,j) + (1-\lambda) \cdot d(i, x_t)$, a weighted combination over a single metric space.

### Key Designs

1. **Impossibility Theorem (Theorem 1)**:

    - A carefully constructed instance with 6 agents and 3 clusters, $\{a,b,c,d,e,f\}$, is used.
    - It is shown that non-centroid core requirements force $\{a,b\}$, $\{e,f\}$, and $\{c,d\}$ to each form their own cluster.
    - It is then shown that the centroid core cannot be satisfied for this fixed partition, regardless of centroid placement.
    - Significance: this rules out any direct adaptation of the existing Greedy Capture algorithm.

2. **Dual Metric Algorithm (Algorithm 3)**:

    - Two-phase design:
        - **Phase 1**: Iteratively identifies the Most Cohesive Cluster (MCC)—the cluster-centroid pair minimizing the maximum agent loss. Agents in the identified cluster are then removed.
        - **Phase 2**: Selectively permits agents to switch clusters, subject to key constraints:
            - Switching must not excessively harm members of the new cluster: post-switch loss $\leq c \cdot r_{t'}$.
            - A bounded-loss guarantee is provided for the switching agent.
            - Each agent's switching decision is made independently of others.
    - The growth factor $c$ is a key parameter controlling the permissiveness of switching: $c = \frac{3\alpha + \sqrt{\alpha(\alpha+8)}}{4\alpha}$.
    - Design Motivation: to interpolate between centroid GC (free switching) and non-centroid GC (no switching).

3. **Core Theoretical Results (Theorems 2–3)**:

    - Existence: under dual metric loss, a 3-core clustering always exists (using an exact MCC subroutine).
    - Polynomial-time algorithm: using a 4-approximation MCC subroutine, a $(3+2\sqrt{3})$-core clustering is found in polynomial time.
    - Stronger FJR (Fully Justified Representation) guarantees: exact 1-FJR exists; polynomial-time 4-FJR is achievable.

### Loss & Training
- This is a theoretical work; no training is involved.
- Algorithm runtime depends on the efficiency of the MCC subroutine.
- For weighted single metric loss, parameterized approximation factors of $\min\{2/\lambda, f_\lambda\}$ are provided as $\lambda$ varies.

## Key Experimental Results

### Main Results

**Core approximation ratios on real datasets**

| Dataset | $\lambda$ | Algorithm 3 Core Approx. | k-means++ Core Approx. | Notes |
|---------|-----------|--------------------------|------------------------|-------|
| Occupancy | 0.5 | ~1.1 | ~2.5 | Far better than theoretical upper bound |
| Iris | 0.5 | ~1.0 | ~3.0 | Near-perfect |
| Wine | 0.5 | ~1.1 | ~2.8 | Consistently superior |

### Ablation Study

| Configuration | Core Approx. (Median) | Classical Clustering Objective Loss | Notes |
|--------------|----------------------|-------------------------------------|-------|
| Algorithm 3 | ~1.05 | Slightly increased | Balances fairness and efficiency |
| k-means++ | ~2.5 | Optimal | Ignores fairness |
| Theoretical lower bound | ≥2 | — | Dual metric |
| Theoretical lower bound | ≥1 | — | FJR |

### Key Findings
- Empirical performance substantially exceeds theoretical worst-case bounds: the algorithm typically achieves core approximation ratios close to 1.
- The price of fairness is small: classical clustering objectives increase only marginally.
- Compared to k-means++, the fair clustering algorithm consistently and significantly improves core approximation.
- Results extend analogous observations from Chen et al. and Caragiannis et al. within their respective paradigms.
- The impossibility theorem implies that $\lambda$ must be provided as input; relying solely on a single metric $d$ is insufficient.

## Highlights & Insights
- The construction in the impossibility theorem (Theorem 1) is remarkably elegant: a fundamental impossibility is established using only 6 points and 3 clusters.
- The selective switching mechanism in Phase 2 is the core algorithmic innovation, striking a principled balance between unconstrained switching and no switching.
- The theoretical landscape is comprehensive: existence upper bounds, algorithmic upper bounds, and lower bounds together form a clear and coherent picture.
- The semi-centroid clustering model is forward-looking: it unifies two independent lines of research and naturally captures real-world scenarios.

## Limitations & Future Work
- The polynomial-time core approximation factor of $(3+2\sqrt{3}) \approx 6.46$ leaves room for improvement; the theoretical lower bound is only 2.
- Non-centroid loss is restricted to the maximum distance (diameter); average distance and other aggregation functions are not addressed.
- Experiments are conducted on small real datasets; scalability to large-scale settings is untested.
- The weighting parameter $\lambda$ must be specified in advance; learning an optimal $\lambda$ from data is not discussed.
- A significant gap remains between the polynomial-time FJR approximation (4-FJR) and the existential guarantee (1-FJR).

## Related Work & Insights
- The Greedy Capture algorithm of Chen et al. (2019) is the seminal work on proportional fairness in centroid clustering, achieving a $(1+\sqrt{2})$-core.
- Caragiannis et al. (2024) extend proportional fairness to non-centroid clustering, motivated by federated learning.
- Solving the Most Cohesive Cluster problem is the computational bottleneck for algorithmic efficiency.
- The unified framework established in this paper provides a more general platform for future research, enabling simultaneous progress in both paradigms under the semi-centroid setting.
- Exploring clustering fairness through the lenses of sortition and strategic game theory warrants further investigation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifying two independent paradigms, the impossibility theorem, and the new algorithm are all significant theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical; empirical validation is adequate but limited in scale.
- Writing Quality: ⭐⭐⭐⭐ Theorems are stated clearly and proofs are rigorously structured, though accessibility for non-theory readers is limited.
- Value: ⭐⭐⭐⭐ Advances the theory of fair clustering; the unified framework has long-term impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] It's Complicated: The Relationship of Algorithmic Fairness and Non-Discrimination Provisions for High-Risk Systems in the EU AI Act](its_complicated_the_relationship_of_algorithmic_fairness_and_non-discrimination_.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)
- [\[NeurIPS 2025\] OmniFC: Rethinking Federated Clustering via Lossless and Secure Distance Reconstruction](omnifc_rethinking_federated_clustering_via_lossless_and_secure_distance_reconstr.md)
- [\[AAAI 2026\] Fair Model-Based Clustering](../../AAAI2026/ai_safety/fair_model-based_clustering.md)
- [\[NeurIPS 2025\] Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
