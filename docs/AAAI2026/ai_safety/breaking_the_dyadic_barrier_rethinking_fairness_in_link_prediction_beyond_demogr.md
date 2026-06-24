---
title: >-
  [Paper Note] Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity
description: >-
  [AAAI 2026][AI Safety][Link prediction] This paper reveals three fundamental flaws of dyadic fairness and Demographic Parity ($\Delta\text{DP}$) in link prediction—insufficient GNN expressiveness, obscured subgroup bias, and ranking insensitivity. It proposes a ranking-aware fairness metric based on NDKL and a post-processing algorithm, MORAL, achieving state-of-the-art fairness-utility trade-offs across six datasets.
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Link prediction"
  - "fairness"
  - "Demographic Parity"
  - "ranking fairness"
  - "graph neural networks"
  - "NDKL"
date: 2026-05-08
content_hash: afcef0cdf90e7840
---

# Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity

**Conference**: AAAI 2026  
**arXiv**: [2511.06568](https://arxiv.org/abs/2511.06568)  
**Code**: [joaopedromattos/MORAL](https://github.com/joaopedromattos/MORAL)  
**Area**: AI Safety / Algorithmic Fairness  
**Keywords**: Link prediction, fairness, Demographic Parity, ranking fairness, graph neural networks, NDKL  

## TL;DR

This paper reveals three fundamental flaws of dyadic fairness and Demographic Parity ($\Delta\text{DP}$) in link prediction—insufficient GNN expressiveness, obscured subgroup bias, and ranking insensitivity. It proposes a ranking-aware fairness metric based on NDKL and a post-processing algorithm, MORAL, achieving state-of-the-art fairness-utility trade-offs across six datasets.

## Background & Motivation

Link prediction is a core task in graph machine learning, widely used in scenarios such as social recommendation and knowledge graph completion. However, biased link prediction can exacerbate social inequalities—for instance, in professional social networks like LinkedIn, connection recommendations biased towards men can reduce job opportunities for women/minorities by 33% (mcdonald2009networks), accumulating in the long term to form "filter bubbles" and "glass ceiling" effects.

Existing fair link prediction methods generally adopt a **dyadic fairness framework**, which classifies node pairs into intra-group (intra: $E_{s\text{-}s} \cup E_{s'\text{-}s'}$) and inter-group (inter: $E_{s'\text{-}s}$), using Demographic Parity ($\Delta\text{DP}$) to measure the difference in prediction probabilities between them. However, this coarse-grained grouping obscures systemic biases within subgroups. Furthermore, as a classification-based metric, $\Delta\text{DP}$ is completely insensitive to ranking order and fails to capture exposure bias. Therefore, there is an urgent need for a more expressive fairness evaluation framework and corresponding debiasing algorithms.

## Core Problem

1. **Insufficient GNN expressiveness impairs fairness**: Standard GNNs are bounded by the expressiveness upper limit of the 1-WL test. Nodes from different sensitive groups in symmetric neighborhoods generate similar embedding representations, failing to distinguish between different types of node pairs such as $E_{s'\text{-}s'}$ and $E_{s'\text{-}s}$, which prevents fair node embeddings from translating into fair link predictions.
2. **Subgroup bias obscured by dyadic aggregation**: After merging $E_{s\text{-}s}$ and $E_{s'\text{-}s'}$ into a single "intra" group, the model may systematically over-predict one subgroup (e.g., male-male far more than female-female), yet $\Delta\text{DP}$ fails to detect this "fairness gerrymandering" due to aggregated averaging.
3. **$\Delta\text{DP}$ is insensitive to ranking**: $\Delta\text{DP}$ is permutation-invariant. A biased ranking that places all intra-group pairs at the top and a fair, evenly interleaved ranking can obtain the same $\Delta\text{DP}$ value—completely ignoring exposure bias.

## Method

### Theoretical Contributions: Two Fairness Properties

**Property 1 (Non-Dyadic Distribution-Preserving Fairness)**: A fairness metric should treat all combinations of sensitive attributes as independent subgroups, aiming to match the predicted edge type distribution $\hat{\boldsymbol{\pi}}$ with the ground-truth distribution of the original graph $\boldsymbol{\pi} = (\pi_{s\text{-}s}, \pi_{s'\text{-}s}, \pi_{s'\text{-}s'})$ as closely as possible, achieved by minimizing $\text{dist}(\hat{\boldsymbol{\pi}}, \boldsymbol{\pi})$.

**Property 2 (Ranking Awareness)**: A fairness metric should be sensitive to the proportion and position of each node pair type within the ranking. Specifically, it should minimize $\sum_{k=1}^{|\mathcal{C}|} \text{dist}(\hat{\boldsymbol{\pi}}_k, \boldsymbol{\pi}) \cdot \delta_k$, where $\hat{\boldsymbol{\pi}}_k$ represents the subgroup distribution within the top-$k$ positions, and $\delta_k$ is a monotonically decreasing exposure decay weight.

### Fair Ranking Metric: NDKL

The paper adopts Normalized Cumulative KL-Divergence (NDKL) as the fairness metric:

$$\text{NDKL} = \frac{1}{Z} \sum_{k=1}^{|\mathcal{C}|} \frac{1}{\log_2(k+1)} D_{\text{KL}}(\hat{\boldsymbol{\pi}}_k \| \boldsymbol{\pi})$$

where $Z = \sum_{i=1}^{|\mathcal{C}|} \frac{1}{\log_2(i+1)}$ is the normalization term. NDKL simultaneously satisfies Property 1 (KL divergence based on the three subgroups) and Property 2 (using $1/\log_2(k+1)$ as position weights, heavily penalizing bias at top positions).

**Theorem 1**: Under the constraint of global demographic parity, the bounds of NDKL are $0 \leq \text{NDKL} \leq \max_i \log(1/\pi_i)$.

### The MORAL Algorithm

MORAL (Multi-Output Ranking Aggregation for Link fairness) is a lightweight post-processing framework comprising two stages:

**Stage 1: Decoupled Training**. Separate, independent link prediction models $f_{s\text{-}s}$, $f_{s\text{-}s'}$, and $f_{s'\text{-}s'}$ are trained for the three edge types, with each model trained solely on its corresponding type of edges. This decoupling strategy eliminates the performance imbalance of a single model across different groups.

**Stage 2: Greedy Rank Aggregation**. During inference, a cumulative exposure distribution estimate of the edge types is maintained. For each position in the ranked list, the candidate edge from the three models that minimizes the cumulative KL divergence upon insertion is selected. Specifically:

- Initialize exposure counts $\boldsymbol{c} \leftarrow (0,0,0)$
- For each ranking position $t = 1, \ldots, n$:
    - For each candidate group $j$, retrieve its highest-scoring candidate edge, temporarily update counts, and compute $D_{\text{KL}}(\mathbf{q}' \| \boldsymbol{\pi})$
    - Select the group and candidate edge that minimize the KL divergence, then append it to the ranked list
- The optimization targets a fairness-first objective: $\min_R \mathcal{L}_A(R) \text{ s.t. } \mathcal{L}_B(R) \leq \min_{R'} \mathcal{L}_B(R')$

**Computational Efficiency**: Each decoupled model only processes $\frac{|E_{\text{train}}|}{|S| \cdot \binom{|S|}{2} \cdot b}$ gradients per epoch, and the aggregation stage is a greedy $O(n \cdot G)$ process, rendering the overall overhead extremely small.

## Key Experimental Results

### Datasets

Evaluated on 6 real-world datasets covering diverse scenarios such as social networks, credit assessment, and sports:

| Dataset | Nodes | Edges | Sensitive Attribute | $E_{s'\text{-}s}$% | $E_{s\text{-}s}$% | $E_{s'\text{-}s'}$% | Topology Type |
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

1. **$\Delta\text{DP}$ fails to detect subgroup bias**: In the top-100 predictions, most baselines are heavily biased towards a specific subgroup type (e.g., $E_{s\text{-}s}$ far exceeding $E_{s'\text{-}s'}$), yet their $\Delta\text{DP}$ values remain very low—validating the fatal flaw of dyadic aggregation.
2. **$\Delta\text{DP}$ cannot differentiate ranking quality**: With subgroup proportions fixed, the gap in NDKL between the worst and best rankings is massive, while their $\Delta\text{DP}$ values are identical—demonstrating the necessity of ranking awareness.
3. **MORAL is comprehensively leading**: MORAL achieves the lowest NDKL (0.01–0.04) across all 6 datasets, while maintaining high prec@1000 levels (0.80–1.00), with no OOM issues.

## Highlights & Insights

1. **Profound insights on issues**: Systematically criticizes the existing dyadic fairness framework from three complementary angles: GNN expressiveness, subgroup aggregation traps, and ranking insensitivity. Each argument is supported by clear theoretical analysis and intuitive examples.
2. **Exquisite design of the NDKL metric**: Naturally realizes the intuition of "heavier penalty on bias at top positions" through logarithmic position weights while maintaining an elegant connection to information theory (KL divergence).
3. **Extremely simple yet effective method**: MORAL is a pure post-processing method that can be coupled with any link prediction model without modifying the training process. The computational overhead of greedy aggregation is extremely low.
4. **Clever decoupled training strategy**: Training models separately for different edge types avoids the performance imbalance across groups in a single model, and naturally fits the subsequent grouped ranking aggregation.

## Limitations & Future Work

1. **Considers only binary sensitive attributes**: Although the paper claims the framework is generalizable to multi-valued/multi-attribute scenarios, all experiments are based on binary attributes ($|S|=2$, three edge types). Its performance when the number of subgroups $\binom{|S|}{2}$ grows rapidly under multi-valued attributes remains unvalidated.
2. **Greedy strategy is non-optimal**: Greedy rank aggregation is an approximate solution to the fairness-first objective and does not guarantee global optimality; oscillations may occur under extremely imbalanced distributions (e.g., only 2% of $E_{s'\text{-}s'}$ in the Credit dataset).
3. **Strong assumption on target distribution**: MORAL adopts the edge type distribution $\boldsymbol{\pi}$ of the original graph as the "fairness target," which implicitly assumes that the subgroup proportions in the original graph are ideal. However, if the original graph itself contains historical bias, preserving the distribution may perpetuate unfairness.
4. **Lacks evaluation in dynamic/online scenarios**: In real-world recommendation systems, link prediction is an ongoing process where cumulative feedback can alter the graph structure and fairness constraints. This adaptation for online deployment is not discussed in the paper.
5. **Uses only a GCN encoder**: All experiments are based on a GCN + dot-product decoder. It remains unverified whether more powerful link prediction models (e.g., SEAL, NBFNet) would alter the conclusions.

## Related Work & Insights

| Dimension | Existing Methods | MORAL |
|------|---------|-------|
| Fairness Framework | Dyadic (intra vs. inter) | Non-dyadic (three independent subgroups) |
| Fairness Metric | $\Delta\text{DP}$ (classification metric, ranking-insensitive) | NDKL (ranking-aware, position-weighted) |
| Debiasing Strategy | Pre-processing / In-processing / Node embedding-level | Post-processing + Decoupled training |
| Scalability | OOM on large graphs (e.g., FairAdj, FairEGM) | Runnable on all datasets |
| Model Agnosticity | Most methods are tied to specific architectures | Compatible with any link prediction model |
| Utility-Fairness Trade-off | Achieving fairness often severely sacrifices accuracy | High prec@1000 maintained while achieving optimal NDKL |

Compared to fair ranking methods in information retrieval (DetConstSort, DELTR), MORAL specifically designs a decoupled training strategy for the combinatorial nature of link prediction, rather than directly applying generic ranking fairness schemes.

## Insights & Connections

1. **"Metric as Bias"**: This paper presents a textbook case—simply changing the evaluation metric can expose hidden biases. The insight is that in any fairness research, the properties of the metric itself (such as ranking sensitivity and subgroup coverage) should be rigorously scrutinized before discussing algorithms.
2. **Universality of decoupled training**: The idea of decoupling models by subgroups can be generalized to other graph tasks (e.g., subgroup-specific classifiers in node classification) and even multi-objective ranking in recommendation systems.
3. **Connection to fairness in recommendation systems**: NDKL originally originated in information retrieval (Geyik et al., 2019). This paper introduces it to graph learning, suggesting that methods from both fields can cross-fertilize more deeply.
4. **Implications for GNN expressiveness research**: Fairness requirements provide fresh application motivations for GNN architectures that go beyond 1-WL (e.g., higher-order GNNs, subgraph GNNs).

## Rating
- Novelty: ⭐⭐⭐⭐ — The systematic three-dimensional critique of dyadic fairness and the introduction of NDKL to link prediction demonstrate strong originality.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive validation across 6 datasets with 10 baselines, though experiments with stronger GNNs and multi-valued attributes are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ — Logically solid, with theory and examples well-interleaved. The intuitive examples in Figure 1-2 are highly effective.
- Value: ⭐⭐⭐⭐ — A fundamental rethinking of the evaluation paradigm for fair link prediction. MORAL is simple and practical, offering valuable practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Beyond Procedure: Substantive Fairness in Conformal Prediction](../../ICML2026/ai_safety/beyond_procedure_substantive_fairness_in_conformal_prediction.md)
- [\[AAAI 2026\] FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates](fairgse_fairness-aware_graph_neural_network_without_high_false_positive_rates.md)
- [\[ICML 2025\] Breaking the n^{1.5} Additive Error Barrier for Private and Efficient Graph Sparsification](../../ICML2025/ai_safety/breaking_the_n15_additive_error_barrier_for_private_and_efficient_graph_sparsifi.md)
- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [\[AAAI 2026\] Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden](revisiting_unfairness_in_recourse_by_minimizing_worst-case_social_burden.md)

</div>

<!-- RELATED:END -->
