---
title: >-
  [Paper Note] Bid Farewell to Seesaw: Towards Accurate Long-tail Session-based Recommendation via Dual Constraints of Hybrid Intents
description: >-
  [AAAI 2026][Recommender Systems][Session-based recommendation] This paper proposes the HID framework, which constructs hybrid intents via attribute-aware spectral clustering to distinguish session-relevant from session-i…
tags:
  - "AAAI 2026"
  - "Recommender Systems"
  - "Session-based recommendation"
  - "long-tail distribution"
  - "hybrid intents"
  - "spectral clustering"
  - "contrastive learning"
date: 2026-05-08
content_hash: 07581b465d930916
---

# Bid Farewell to Seesaw: Towards Accurate Long-tail Session-based Recommendation via Dual Constraints of Hybrid Intents

**Conference**: AAAI 2026
**arXiv**: [2511.08378](https://arxiv.org/abs/2511.08378)
**Authors**: Xiao Wang, Ke Qin, Dongyang Zhang, Xiurui Xie, Shuang Liang
**Code**: Not released
**Area**: Recommender Systems
**Keywords**: Session-based recommendation, long-tail distribution, hybrid intents, spectral clustering, contrastive learning

## TL;DR

This paper proposes the HID framework, which constructs hybrid intents via attribute-aware spectral clustering to distinguish session-relevant from session-irrelevant tail items, and introduces a dual-constraint loss (ICLoss) targeting both long-tail coverage and recommendation accuracy. The framework achieves a "win-win" between long-tail promotion and accuracy, breaking the traditional seesaw dilemma where improving one metric inevitably harms the other.

## Background & Motivation

### State of the Field
Session-based Recommendation (SBR) aims to predict the next interacted item from an anonymous user's short-term interaction sequence. In practical recommendation scenarios, item exposure frequency follows a severe long-tail distribution — a small number of head items account for the vast majority of interactions, while a large number of tail items are rarely recommended. This imbalance causes recommendation systems to repeatedly surface head items, creating a vicious cycle that substantially reduces recommendation diversity.

### Limitations of Prior Work
Existing long-tail SBR methods fall into two categories: (1) **Augmentation-based methods** (e.g., LOAM, MelT, LLM-ESR), which enhance tail item embeddings or emphasize tail items when generating session embeddings; and (2) **Re-ranking-based methods** (e.g., TailNet, CSBR, LAP-SR), which predict the head/tail item distribution from interaction sessions and directly adjust the final ranking results.

Both categories share two common deficiencies:

**Indiscriminate emphasis on tail items introduces noise**: Not all tail items are relevant to the current session. For example, in a session centered on "books," tail items from "clothing" categories are noise, despite being long-tail. Existing methods uniformly boost tail item exposure without distinguishing relevant from irrelevant tail items, leading to degraded recommendation accuracy.

**Lack of explicit long-tail supervision signals**: Most methods still rely on cross-entropy loss for indirect optimization, and augmentation or re-ranking strategies often conflict with the cross-entropy objective, resulting in a **seesaw effect** — improving long-tail performance inevitably harms accuracy, and vice versa.

On the Tmall dataset, applying long-tail methods such as TailNet to GRU4Rec improves tCov@20 but causes a notable drop in HR@20, clearly demonstrating this seesaw phenomenon.

### Root Cause
Resolving this contradiction requires answering two questions: how to effectively identify noise, and how to provide explicit supervision signals for both long-tail coverage and accuracy simultaneously. HID's answer is: (1) capture high-level user preferences through **hybrid intent modeling** to partition items into session-target intents and noise intents; and (2) apply **dual-constraint losses** to enforce head-tail consistency within target intents (promoting long-tail) and to maximize the distance between the session and noise intents (preserving accuracy).

## Method

### Overall Architecture
HID is a model-agnostic plug-and-play framework that can be integrated with any SBR backbone (e.g., GRU4Rec, STAMP, SRGNN, GCE-GNN). It consists of two core components: a Hybrid Intent Learning module and an Intent Constraint Loss.

### Module 1: Hybrid Intent Learning

Conventional intent mining methods consider only intra-session temporal relationships, making them susceptible to noise and neglecting cross-session intent consistency. HID proposes attribute-aware spectral clustering to construct global hybrid intents in three steps:

**Step 1 — Preliminary Intent Units**: Item attribute information (e.g., product category, music genre) is used as preliminary intent units. Items sharing the same attribute are grouped into a preliminary intent group $c'_i$.

**Step 2 — Preliminary Intent Graph Construction**: Item IDs in each session are replaced with attribute IDs, and co-occurrence frequencies between attributes are aggregated across all sessions to construct a preliminary intent graph $\mathcal{G}=(\mathcal{P}, \mathcal{E}, \mathcal{W})$, where nodes represent attributes and edge weights represent co-occurrence frequencies. For example, "food" and "cookware" frequently co-occur in shopping sessions, yielding a high edge weight between them.

**Step 3 — Spectral Clustering to Generate Hybrid Intents**: The normalized Laplacian matrix $L = I - D^{-1/2}WD^{-1/2}$ is computed for the preliminary intent graph. The eigenvectors corresponding to the $q$ smallest eigenvalues are extracted, and K-means clustering is applied to the rows of the eigenvector matrix to reassign attributes into $n$ clusters. Attributes within the same cluster are merged to form a hybrid intent — for instance, "food" and "cookware" may be aggregated into a "cooking" intent. The embedding of each hybrid intent is obtained via mean pooling over all contained item embeddings: $\mathbf{c}_i = \frac{1}{|c_i|}\sum_{v_j \in c_i} \mathbf{v}_j$.

Notably, the entire hybrid intent construction process can be **precomputed offline**; during training and inference, only a lookup table retrieval is needed, introducing no additional online overhead.

### Definition of Target and Noise Intents

For session $S^u$, given the next item $v_{l+1}^u$ (the ground-truth label known during training):
- **Target intent** $\mathcal{C}^u$: the set of hybrid intents containing $v_{l+1}^u$
- **Noise intent** $\hat{\mathcal{C}}^u$: target intents from other sessions in the same mini-batch that are not in $\mathcal{C}^u$

Target and noise intents are used solely as supervision signals in ICLoss during training, posing no risk of data leakage.

### Module 2: Intent Constraint Loss (ICLoss)

After obtaining hybrid intent embeddings, HID applies $L_2$ normalization to both session and intent embeddings, projecting them onto a unit hypersphere to ensure a consistent metric space. Two constraints are then imposed:

**Constraint 1: Constraint for Long-tail**

Core Idea: Minimize the variance of distances from the session to all items within the target intent. Head item embeddings are typically closer to the session representation, while tail items are farther; constraining the variance forces their distances to converge, granting tail items recommendation probabilities comparable to head items.

$$\min\ \mathcal{L}_l = \text{Var}_{v_i \in \mathcal{C}^u}[d(\mathbf{S}^u, \mathbf{v}_i)]$$

Direct variance computation has complexity $O(Nd)$. The paper proves via Theorem 1 that minimizing this variance is optimization-equivalent to minimizing the distance from the session embedding to the target intent centroid $d(\mathbf{S}^u, \mathbf{c}^u)$, reducing complexity to $O(d)$.

**Constraint 2: Constraint for Accuracy**

Core Idea: Maximize the average distance from the session to noise intents, while constraining the variance to avoid degenerate cases.

$$\max\ \mathcal{L}_a = \mathbb{E}_{c^v \in \hat{\mathcal{C}}^u} d(\mathbf{S}^u, \mathbf{c}^v), \quad \text{s.t.}\ \text{Var}_{c^v \in \hat{\mathcal{C}}^u}(d(\mathbf{S}^u, \mathbf{c}^v)) < \eta$$

**Unified Loss Function**

Both constraints are unified into an InfoNCE-style loss. The paper proves via Theorem 2 that this loss is approximately equivalent to an $(N-1)$-Triplet Loss with a fixed margin of 2. To accommodate different scenarios, a flexible temperature coefficient $\sigma$ replaces the fixed margin, and the hard variance constraint is converted into a penalty term $p^u$. The final ICLoss is:

$$\mathcal{L}_c = -\sum_{S^u \in \mathcal{B}} \log \frac{\mathbf{X}}{(1+\lambda p^u)(\mathbf{X}+\mathbf{Y})}$$

where $\mathbf{X} = \exp(\cos(\mathbf{S}^u, \mathbf{c}^u)/\sigma)$ and $\mathbf{Y} = \sum_{c^v \in \hat{\mathcal{C}}^u}\exp(\cos(\mathbf{S}^u, \mathbf{c}^v)/\sigma)$.

The total training loss is: $\mathcal{L} = \mathcal{L}_p + \epsilon \mathcal{L}_c$, where $\mathcal{L}_p$ is the cross-entropy loss of the original SBR backbone.

## Key Experimental Results

### Table 1: Main Results — STAMP and GRU4Rec Backbones (HR@20 / tCov@20)

| Method | Tmall HR↑ | Tmall tCov↑ | Diginetica HR↑ | Diginetica tCov↑ | Retailrocket HR↑ | Retailrocket tCov↑ |
|--------|-----------|-------------|----------------|------------------|------------------|---------------------|
| STAMP (base) | 26.10 | 69.46 | 50.15 | 90.71 | 50.54 | 53.70 |
| + TailNet | 20.61 ↓ | 71.33 ↑ | 45.39 ↓ | 91.23 ↑ | 47.00 ↓ | 51.56 ↓ |
| + LOAM | 24.31 ↓ | 71.68 ↑ | 46.19 ↓ | 89.96 ↓ | 50.27 ↓ | 55.67 ↑ |
| + LAP-SR | 25.21 ↓ | 72.11 ↑ | 49.87 ↓ | 91.32 ↑ | 49.59 ↓ | 55.32 ↑ |
| **+ HID** | **28.26 ↑** | **73.65 ↑** | **50.39 ↑** | **93.05 ↑** | **52.38 ↑** | **56.02 ↑** |
| GRU4Rec (base) | 19.69 | 49.60 | 50.23 | 84.97 | 45.01 | 69.98 |
| **+ HID** | **25.13 ↑** | **63.21 ↑** | **52.23 ↑** | **90.73 ↑** | **48.89 ↑** | **73.21 ↑** |

Key finding: Existing long-tail methods almost universally sacrifice accuracy for long-tail gains (seesaw effect), whereas HID simultaneously improves both accuracy and long-tail coverage across all 4 backbone × 3 dataset combinations. GRU4Rec+HID achieves a 27.6% improvement in Tmall HR@20 (19.69→25.13) and a 27.4% improvement in tCov@20 (49.60→63.21) over the base model.

### Table 2: Ablation Study (STAMP + SRGNN, HR@20 / tCov@20)

| Variant | Tmall HR | Tmall tCov | Diginetica HR | Diginetica tCov | Retailrocket HR | Retailrocket tCov |
|---------|----------|------------|---------------|-----------------|-----------------|-------------------|
| STAMP+HID | **28.26** | **73.65** | **50.39** | **93.05** | **52.38** | **56.02** |
| w/o HI | 27.43 | 69.29 | 50.17 | 91.96 | 51.75 | 55.31 |
| w/o FC | 26.77 | 70.20 | 49.76 | 92.15 | 50.89 | 55.67 |
| SRGNN+HID | **28.38** | **66.40** | **52.09** | **96.02** | **53.45** | **55.75** |
| w/o HI | 27.48 | 61.00 | 51.96 | 92.94 | 53.10 | 54.01 |
| w/o FC | 27.36 | 62.92 | 51.16 | 93.56 | 52.80 | 55.11 |

Removing hybrid intents (HI) has a greater impact on diversity, while removing the flexible coefficient (FC) has a greater impact on accuracy. The effect of hybrid intents is particularly pronounced on Tmall (tCov: 73.65→69.29), as Tmall sessions are longer and exhibit more frequent intent drift, making precise target intent modeling more critical.

## Highlights & Insights

- **Conceptual Innovation**: The paper is the first to explicitly attribute the root cause of the seesaw effect to the indiscriminate emphasis on tail items introducing session-irrelevant noise, and offers a principled solution through intent modeling.
- **Solid Theoretical Grounding**: Theorem 1 reduces $O(Nd)$ variance minimization to equivalent $O(d)$ distance minimization; Theorem 2 approximates the unified loss as a Triplet Loss with a margin term, with complete theoretical derivations.
- **Plug-and-Play Design**: HID integrates seamlessly with any SBR backbone (both sequential and graph-based), and hybrid intent construction can be performed offline with no additional inference overhead.
- **Well-Validated Win-Win**: HID achieves simultaneous improvements in accuracy and long-tail coverage across all combinations of 4 backbones × 3 datasets × 6 metrics, with $p$-values generally below 0.001.

## Limitations & Future Work

- **Dependency on Item Attribute Information**: Hybrid intent construction requires item category attributes as preliminary intent units, limiting applicability in scenarios lacking attribute metadata (though the appendix includes experiments with semantic clustering as an alternative).
- **Cluster Count Requires Tuning**: The optimal number of clusters $n$ varies by dataset (Tmall=4, Diginetica=3), and no adaptive selection strategy is provided.
- **Validation Limited to Tabular SBR Models**: The framework has not been validated on modern Transformer-based SBR models such as SASRec or BERT4Rec.
- **Target Intent Relies on Ground Truth**: During training, target intents are defined via the next item $v_{l+1}^u$, which limits extensibility to semi-supervised or weakly supervised settings.

## Related Work & Insights

- **TailNet / CSBR / LAP-SR (re-ranking-based)**: These methods directly modify ranking results without discriminating against noise; HID addresses this at the root through intent modeling to distinguish session-relevant from session-irrelevant items.
- **LOAM / MelT / LLM-ESR (augmentation-based)**: Augmenting tail embeddings likewise fails to filter noise; HID's dual-constraint loss explicitly pushes noise intents away from the session while improving tail coverage.
- **Intent Modeling Works (ICL, MISAR, STP)**: Existing intent mining extracts intents only from sliding windows or local subgraphs within individual sessions, ignoring cross-session consistency; HID constructs hybrid intents from global attribute co-occurrence relationships, yielding greater robustness.
- **Contrastive Learning for Recommendation (CL4SRec et al.)**: These methods leverage positive-negative pairs for representation learning but are not tailored for the long-tail problem; HID's ICLoss is essentially a long-tail-oriented variant of contrastive learning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of hybrid intents and dual constraints is novel, and the attribution analysis of the seesaw effect is convincing.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive validation across 4 backbones × 3 datasets, with complete ablation and hyperparameter analyses and rigorous statistical testing.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clearly articulated, theoretical derivations are complete, and figures are intuitive.
- Value: ⭐⭐⭐⭐ — The proposed plug-and-play framework is highly practical and makes a clear methodological contribution to the long-tail recommendation literature.
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FreqRec: Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation](exploiting_inter-session_information_with_frequency-enhanced_dual-path_networks_.md)
- [\[AAAI 2026\] HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation](hymoerec_hybrid_mixture-of-experts_for_sequential_recommendation.md)
- [\[AAAI 2026\] Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction](length-adaptive_interest_network_for_balancing_long_and_short_sequence_modeling_.md)
- [\[AAAI 2026\] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)
- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)

</div>

<!-- RELATED:END -->
