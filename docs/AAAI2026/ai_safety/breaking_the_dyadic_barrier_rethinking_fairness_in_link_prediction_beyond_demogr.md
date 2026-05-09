---
title: >-
  [Paper Note] Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity
description: >-
  [AAAI 2026][AI Safety][Link Prediction] This paper identifies three fundamental flaws in dyadic fairness and Demographic Parity (ΔDP) for link prediction—insufficient GNN expressiveness, subgroup bias masking, and ranking insensitivity—and proposes a ranking-aware fairness metric based on NDKL and a post-processing algorithm MORAL, achieving state-of-the-art fairness–utility trade-offs across six datasets.
tags:
  - AAAI 2026
  - AI Safety
  - Link Prediction
  - Fairness
  - Demographic Parity
  - Ranking Fairness
  - Graph Neural Networks
  - NDKL
date: 2026-05-08
content_hash: c6e9981df8be815c
---

# Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity

**Conference**: AAAI 2026
**arXiv**: [2511.06568](https://arxiv.org/abs/2511.06568)
**Code**: [joaopedromattos/MORAL](https://github.com/joaopedromattos/MORAL)
**Area**: AI Safety / Algorithmic Fairness
**Keywords**: Link Prediction, Fairness, Demographic Parity, Ranking Fairness, Graph Neural Networks, NDKL

## TL;DR

This paper identifies three fundamental flaws in dyadic fairness and Demographic Parity (ΔDP) for link prediction—insufficient GNN expressiveness, subgroup bias masking, and ranking insensitivity—and proposes a ranking-aware fairness metric based on NDKL and a post-processing algorithm MORAL, achieving state-of-the-art fairness–utility trade-offs across six datasets.

## Background & Motivation

Link prediction is a core task in graph machine learning, with broad applications in social recommendation, knowledge graph completion, and beyond. Biased link prediction, however, can exacerbate social inequalities—for instance, in professional networks such as LinkedIn, connection recommendations skewed toward males can reduce job opportunities for women and minorities by 33% (mcdonald2009networks), gradually creating "filter bubble" and "glass ceiling" effects.

Existing fair link prediction methods predominantly adopt the **dyadic fairness framework**, which partitions node pairs into intra-group ($E_{s\text{-}s} \cup E_{s'\text{-}s'}$) and inter-group ($E_{s'\text{-}s}$) sets and measures their prediction probability gap via Demographic Parity (ΔDP). This coarse-grained grouping, however, obscures systematic bias within subgroups, and ΔDP—as a classification-based metric—is entirely insensitive to ranking order and thus fails to capture exposure bias. A more expressive fairness evaluation framework and corresponding debiasing algorithms are therefore urgently needed.

## Core Problem

1. **Insufficient GNN expressiveness impairs fairness**: Standard GNNs are bounded by the expressiveness of the 1-WL test; nodes from different sensitive groups residing in symmetric neighborhoods produce similar embeddings, making it impossible to distinguish node pair types such as $E_{s'\text{-}s'}$ and $E_{s'\text{-}s}$. Consequently, fair node embeddings do not translate into fair link predictions.
2. **Subgroup bias is masked by dyadic aggregation**: Merging $E_{s\text{-}s}$ and $E_{s'\text{-}s'}$ into a single "intra-group" category allows a model to systematically over-predict one subgroup (e.g., male–male pairs far more than female–female pairs), yet ΔDP cannot detect such "fairness gerrymandering" due to averaging over the aggregated group.
3. **ΔDP is insensitive to ranking**: ΔDP is permutation-invariant; a biased ranking that places all intra-group pairs at the top can yield the same ΔDP as a fair ranking with uniform interleaving—exposure bias is entirely ignored.

## Method

### Theoretical Contributions: Two Fairness Properties

**Property 1 (Non-dyadic distribution-preserving fairness)**: A fairness metric should treat every combination of sensitive attribute values as an independent subgroup, aiming to align the predicted edge-type distribution $\hat{\boldsymbol{\pi}}$ with the true distribution in the original graph $\boldsymbol{\pi} = (\pi_{s\text{-}s}, \pi_{s'\text{-}s}, \pi_{s'\text{-}s'})$ by minimizing $\text{dist}(\hat{\boldsymbol{\pi}}, \boldsymbol{\pi})$.

**Property 2 (Ranking awareness)**: A fairness metric should be sensitive to both the proportion and the position of each node pair type in the ranking. Concretely, it should minimize $\sum_{k=1}^{|\mathcal{C}|} \text{dist}(\hat{\boldsymbol{\pi}}_k, \boldsymbol{\pi}) \cdot \delta_k$, where $\hat{\boldsymbol{\pi}}_k$ is the subgroup distribution among the top-$k$ candidates and $\delta_k$ is a monotonically decreasing exposure decay weight.

### Ranking Fairness Metric: NDKL

The paper adopts the Normalized Cumulative KL-Divergence (NDKL) as its fairness metric:

$$\text{NDKL} = \frac{1}{Z} \sum_{k=1}^{|\mathcal{C}|} \frac{1}{\log_2(k+1)} D_{\text{KL}}(\hat{\boldsymbol{\pi}}_k \| \boldsymbol{\pi})$$

where $Z = \sum_{i=1}^{|\mathcal{C}|} \frac{1}{\log_2(i+1)}$ is the normalization term. NDKL satisfies both Property 1 (via KL divergence over three subgroups) and Property 2 (via $1/\log_2(k+1)$ position weights that penalize bias at top positions more heavily).

**Theorem 1**: Under the global demographic parity constraint, NDKL is bounded as $0 \leq \text{NDKL} \leq \max_i \log(1/\pi_i)$.

### MORAL Algorithm

MORAL (Multi-Output Ranking Aggregation for Link fairness) is a lightweight post-processing framework consisting of two stages:

**Stage 1: Decoupled Training**. Three independent link prediction models—$f_{s\text{-}s}$, $f_{s\text{-}s'}$, and $f_{s'\text{-}s'}$—are trained separately, each exclusively on edges of the corresponding type. This decoupling strategy eliminates the performance imbalance across groups that plagues a single jointly trained model.

**Stage 2: Greedy Ranking Aggregation**. During inference, a running estimate of the cumulative exposure distribution over edge types is maintained. At each rank position, the candidate edge from any of the three models whose addition minimizes the cumulative KL divergence is selected:

- Initialize exposure counts $\boldsymbol{c} \leftarrow (0,0,0)$
- For each rank position $t = 1, \ldots, n$:
    - For each candidate group $j$, take its highest-scoring candidate edge, temporarily update the counts, and compute $D_{\text{KL}}(\mathbf{q}' \| \boldsymbol{\pi})$
    - Select the group and candidate edge that minimize the KL divergence and append to the ranking list
- The optimization objective is a fairness-first criterion: $\min_R \mathcal{L}_A(R) \text{ s.t. } \mathcal{L}_B(R) \leq \min_{R'} \mathcal{L}_B(R')$

**Computational Efficiency**: Each decoupled model processes $\frac{|E_{\text{train}}|}{|S| \cdot \binom{|S|}{2} \cdot b}$ gradients per epoch; the aggregation stage runs in greedy $O(n \cdot G)$ time, resulting in negligible overall overhead.

## Key Experimental Results

### Datasets

Experiments are conducted on six real-world datasets spanning social networks, credit assessment, and sports:

| Dataset | Nodes | Edges | Sensitive Attr. | $E_{s'\text{-}s}$% | $E_{s\text{-}s}$% | $E_{s'\text{-}s'}$% | Topology |
|--------|--------|------|----------|------------|-----------|-------------|----------|
| Facebook | 1045 | 18726 | Gender | 42% | 44% | 14% | Peripheral |
| German | 1000 | 15220 | Age | 20% | 61% | 19% | Peripheral |
| NBA | 403 | 7435 | Nationality | 27% | 63% | 10% | Peripheral |
| Pokec-n | 66569 | 361934 | Gender | 5% | 66% | 29% | Community |
| Pokec-z | 67796 | 432572 | Gender | 5% | 58% | 37% | Community |
| Credit | 30000 | 96165 | Age | 12% | 86% | 2% | Peripheral |

### Main Results (NDKL↓ / prec@1000↑)

| Method | Facebook | Credit | German | NBA | Pokec-n | Pokec-z |
|------|----------|--------|--------|-----|---------|---------|
| UGE | 0.05 / 0.97 | 0.80 / 1.00 | 0.08 / 0.69 | 0.07 / 0.58 | 0.06 / 0.90 | 0.06 / 0.91 |
| FairWalk | 0.06 / 0.96 | 0.06 / 1.00 | 0.11 / 0.94 | 0.06 / 0.55 | 0.07 / 1.00 | 0.07 / 1.00 |
| FairEGM | 0.09 / 0.97 | 0.11 / 1.00 | 0.05 / 0.62 | 0.07 / 0.60 | OOM | OOM |
| **MORAL** | **0.04** / 0.95 | **0.01** / 1.00 | **0.03** / 0.96 | **0.02** / 0.80 | **0.03** / 0.98 | **0.04** / 0.98 |

**Key Findings**:

1. **ΔDP fails to detect subgroup bias**: Among top-100 predictions, most baseline methods are strongly skewed toward one subgroup type (e.g., $E_{s\text{-}s}$ far outnumbering $E_{s'\text{-}s'}$), yet their ΔDP values remain low—confirming the fatal flaw of dyadic aggregation.
2. **ΔDP cannot distinguish ranking quality**: With subgroup proportions fixed, the NDKL gap between the worst and best rankings is substantial, while ΔDP is identical—validating the necessity of ranking awareness.
3. **MORAL consistently leads**: MORAL achieves the lowest NDKL (0.01–0.04) across all six datasets while maintaining high prec@1000 (0.80–1.00), with no out-of-memory issues.

## Highlights & Insights

1. **Deep problem insight**: The paper systematically critiques the existing dyadic fairness framework from three complementary angles—GNN expressiveness, subgroup aggregation pitfalls, and ranking insensitivity—each supported by rigorous theoretical analysis and intuitive examples.
2. **Elegant NDKL metric design**: The logarithmic position weights naturally encode the intuition that bias at top positions should be penalized more heavily, while preserving an elegant connection to information theory via KL divergence.
3. **Minimalist yet effective method**: MORAL is a purely post-processing approach compatible with any link prediction model, requiring no modification to the training procedure; greedy aggregation incurs minimal computational cost.
4. **Clever decoupled training strategy**: Training separate models for each edge type avoids cross-group performance imbalance inherent to a single joint model and naturally complements the subsequent group-wise ranking aggregation.

## Limitations & Future Work

1. **Binary sensitive attributes only**: Although the paper claims the framework generalizes to multi-valued and multi-attribute settings, all experiments assume binary attributes ($|S|=2$, three edge types); performance under the rapidly growing subgroup count $\binom{|S|}{2}$ for multi-valued attributes remains unvalidated.
2. **Greedy strategy is not globally optimal**: The greedy ranking aggregation approximates the fairness-first objective without guaranteeing a global optimum; oscillation may occur under severely imbalanced distributions (e.g., $E_{s'\text{-}s'}$ constituting only 2% in the Credit dataset).
3. **Strong assumption about the target distribution**: MORAL uses the edge-type distribution $\boldsymbol{\pi}$ of the original graph as the "fairness target," implicitly assuming the original subgroup proportions are ideal. If the original graph itself encodes historical bias, preserving its distribution may entrench unfairness.
4. **No evaluation in dynamic or online settings**: In real-world recommender systems, link prediction is a continuous process where accumulated feedback alters graph structure and fairness constraints; the paper does not discuss adaptation to online deployment.
5. **GCN encoder only**: All experiments use a GCN with dot-product decoder; whether stronger link prediction models (e.g., SEAL, NBFNet) would alter the conclusions remains unexplored.

## Related Work & Insights

| Dimension | Existing Methods | MORAL |
|------|---------|-------|
| Fairness framework | Dyadic (intra vs. inter) | Non-dyadic (three independent subgroups) |
| Fairness metric | ΔDP (classification-based, ranking-insensitive) | NDKL (ranking-aware, position-weighted) |
| Debiasing strategy | Pre-processing / in-training / node embedding level | Post-processing + decoupled training |
| Scalability | FairAdj / FairLP / EDITS / FairEGM run OOM on large graphs | Runs successfully on all datasets |
| Model agnosticism | Most methods tied to specific architectures | Compatible with any link prediction model |
| Utility–fairness trade-off | Improved fairness often incurs large accuracy loss | Optimal NDKL while maintaining high prec@1000 |

Compared to fair ranking methods from information retrieval (DetConstSort, DELTR), MORAL specifically designs a decoupled training strategy tailored to the combinatorial nature of link prediction rather than directly applying general-purpose ranking fairness schemes.

The paper offers several broader insights:

1. **"The metric is the bias"**: This paper serves as a textbook example of how replacing the evaluation metric alone can reveal hidden biases. It suggests that any fairness study should rigorously scrutinize the properties of its metric (ranking sensitivity, subgroup coverage) before discussing algorithms.
2. **Generality of decoupled training**: The idea of decoupling models by subgroup is transferable to other graph tasks (e.g., subgroup-specific classifiers for node classification) and even multi-objective ranking in recommender systems.
3. **Cross-pollination between recommender systems and graph learning**: NDKL originates in information retrieval (Geyik et al., 2019); its introduction into graph learning here suggests deeper methodological exchange between the two fields.
4. **New motivation for expressive GNNs**: Fairness requirements provide a novel application-driven motivation for GNN architectures that surpass 1-WL expressiveness, such as higher-order GNNs and subgraph GNNs.

## Rating
- Novelty: ⭐⭐⭐⭐ — The three-dimensional systematic critique of dyadic fairness and the introduction of NDKL to link prediction both demonstrate strong originality.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Six datasets, ten baselines, and multi-dimensional validation; experiments with stronger GNNs and multi-valued attributes are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ — Logically structured with well-integrated theory and examples; Figures 1–2 provide highly effective intuitive illustrations.
- Value: ⭐⭐⭐⭐ — A fundamental rethinking of the fair link prediction evaluation paradigm; MORAL is concise and practical with strong implications for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching](../../ICLR2026/ai_safety/beyond_match_maximization_and_fairness_retention-optimized_two-sided_matching.md)
- [\[AAAI 2026\] FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates](fairgse_fairness-aware_graph_neural_network_without_high_false_positive_rates.md)
- [\[AAAI 2026\] Breaking the Adversarial Robustness-Performance Trade-off in Text Classification via Manifold Purification](breaking_the_adversarial_robustness-performance_trade-off_in_text_classification.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)

</div>

<!-- RELATED:END -->
