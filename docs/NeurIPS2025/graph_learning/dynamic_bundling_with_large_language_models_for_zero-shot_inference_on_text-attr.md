---
title: >-
  [Paper Note] Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs
description: >-
  [NeurIPS 2025][Graph Learning][Text-Attributed Graphs] DENSE proposes a "text bundling" strategy that packages textually and topologically/semantically similar nodes into bundles, queries LLMs for bundle-level labels, supervises GNN training via entropy-based and ranking-based losses, and dynamically refines bundles to exclude noisy nodes. It achieves comprehensive zero-shot inference improvements over GPT-4o and graph foundation models across 10 TAG datasets.
tags:
  - NeurIPS 2025
  - Graph Learning
  - Text-Attributed Graphs
  - Zero-Shot Inference
  - LLM
  - Graph Neural Networks
  - Bundle Supervision
date: 2026-05-08
content_hash: 678a05157a40058b
---

# Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs

**Conference**: NeurIPS 2025
**arXiv**: [2505.17599](https://arxiv.org/abs/2505.17599)
**Code**: None
**Area**: Graph Learning
**Keywords**: Text-Attributed Graphs, Zero-Shot Inference, LLM, Graph Neural Networks, Bundle Supervision

## TL;DR
DENSE proposes a "text bundling" strategy that packages textually and topologically/semantically similar nodes into bundles, queries LLMs for bundle-level labels, supervises GNN training via entropy-based and ranking-based losses, and dynamically refines bundles to exclude noisy nodes. It achieves comprehensive zero-shot inference improvements over GPT-4o and graph foundation models across 10 TAG datasets.

## Background & Motivation

**Background**: In text-attributed graphs (TAGs), each node is associated with a textual description, and zero-shot inference requires predicting node categories without labeled data. Two main paradigms exist: (a) encoding graph topology into language models (requires large training data; converting topology to sequences incurs information loss); (b) using LLMs to generate pseudo-labels for individual nodes, which then supervise GNN training.

**Limitations of Prior Work**:
   - **Insufficient information**: LLMs only observe the isolated text of a single node, lacking graph-structural context, leading to weak decision bases.
   - **Unreliable responses**: Inherent LLM hallucinations, compounded by information scarcity, produce noisy pseudo-labels that degrade GNN training when used directly as supervision signals.

**Key Challenge**: Single-node queries leave LLMs both under-informed (graph structure is lost) and unreliable (single-point hallucinations cannot be corrected), with noise amplified in downstream operations.

**Goal**: How can LLMs be provided with richer information to make more reliable judgments? How can supervision mechanisms robust to noise be designed?

**Key Insight**: Shift from "single-node queries" to "text bundle queries"—packing multiple similar nodes into a bundle and querying for the mode category of the bundle, using group-level information to suppress individual noise.

**Core Idea**: Elevate LLMs' zero-shot capability from node-level to bundle-level through text bundling, and supervise GNN training with bundle labels for robust zero-shot graph inference.

## Method

### Overall Architecture
Given a text-attributed graph $\mathcal{G}=\langle\mathcal{V},\mathcal{E},\mathcal{T},\mathcal{Y}\rangle$, DENSE comprises four stages:
1. **Bundle Sampling**: Sample node bundles by topological or semantic similarity
2. **Bundle Query**: Concatenate node texts within a bundle into a prompt and query the LLM for bundle-level labels
3. **Bundle Supervision**: Train the GNN using entropy-based and ranking-based losses
4. **Bundle Refinement**: Dynamically exclude noisy nodes inconsistent with the bundle label

### Key Designs

1. **Bundle Sampling**:

    - Function: Construct bundles of $n_B$ similar nodes such that the majority of nodes within a bundle share the same category.
    - Mechanism: Randomly select an anchor node $v_c$ and sample neighbors via two strategies:
        - **Topological neighbors** (homophilic graphs): $\mathcal{N}^k_{\mathcal{G}}(v_c)=\{i \mid 1\leq d^{\mathcal{G}}(v_i,v_c)\leq k\}$, with $k$ adaptively chosen so that the neighbor count $\geq n_B-1$.
        - **Semantic neighbors** (heterophilic graphs): $\mathcal{B}=\{i \mid \bm{x}_i \in \mathcal{N}^{n_B}_{\mathcal{X}}(\bm{x}_c)\}$, selecting the $n_B$ nodes with smallest L2 distance in the embedding space.
    - Design Motivation: Ensuring intra-bundle majority class membership makes it easier and more accurate for LLMs to predict the mode category. Topological proximity is used for homophilic graphs and semantic proximity for heterophilic graphs, enabling flexible adaptation.

2. **Bundle Query**:

    - Function: Concatenate the texts of all nodes in a bundle into a single prompt and ask the LLM for the dominant category of the bundle.
    - Mechanism: $\mathcal{P}(\mathcal{B})=\langle\text{dataset\_desc}\rangle\text{Concat}(\{t_i|i\in\mathcal{B}\})\langle\text{task\_desc}\rangle$
    - Design Motivation: Compared to node-by-node queries, bundle queries expose the LLM to multiple related texts, enabling it to identify "persistent themes." Predicting the mode is easier and more robust than single-node classification. Experiments confirm that bundle query accuracy is significantly higher than single-node query accuracy.

3. **Bundle Supervision**:

    - Function: Supervise GNN training with bundle labels $\hat{y}^B$ via loss functions robust to outlier nodes.
    - Mechanism: The GNN $g_\theta$ outputs per-node probabilities $\bm{p}_i=\text{softmax}(\bm{z}_i)$; the bundle-level distribution is $\bm{p}(\mathcal{B})=\text{softmax}(\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}\bm{z}_i)$.
        - **Entropy-based loss**: $\mathcal{L}_{BE}=\text{CE}(\bm{p}(\mathcal{B}), \hat{y}^B)$
        - **Ranking-based loss**: $\mathcal{L}_R=-\min(\log\bm{p}(\mathcal{B})_{\hat{y}^B}-\log\max_i\{\bm{p}(\mathcal{B})_i\}, 0)$
        - **Total loss**: $\mathcal{L}=\mathcal{L}_{BE}+\mathcal{L}_R$
    - Design Motivation: Theorem 3.1 proves that the gradient penalty imposed by bundle-level cross-entropy on outlier nodes is $\leq$ that of node-wise supervision, meaning bundle supervision is inherently more tolerant of heterogeneous nodes. The ranking loss ensures penalties are applied only when the bundle prediction is inconsistent with the label.

4. **Bundle Refinement**:

    - Function: Dynamically exclude nodes inconsistent with the bundle label during training.
    - Mechanism: $\mathcal{B} \leftarrow \{i \mid i\in\mathcal{B} \wedge \bm{p}_{i,\hat{y}^B} > \min_{j\in\mathcal{B}}\bm{p}_{j,\hat{y}^B}\}$
    - Design Motivation: Iterative refinement progressively removes the least confident nodes, making bundles increasingly pure and continuously reducing supervision signal noise.

### Loss & Training
- Default LLM: GPT-4o; bundle size $n_B=5$; number of bundles $n_S=100$
- GNN trained on an NVIDIA RTX 3090
- Theorem 3.2 proves that the gradient of $\mathcal{L}_{BE}$ is bounded and its second-order derivative is bounded ($\frac{2(M+G^2)}{|\mathcal{B}|}$-smooth); Theorem 3.3 proves convergence of gradient descent to a stationary point

## Key Experimental Results

### Main Results

| Dataset | DENSE | LLM-BP (Prev. SOTA) | GPT-4o (Direct) | Gain vs. LLM-BP |
|--------|-------|-------------------|---------------|----------------|
| Cora | **75.09** | 72.59 | 70.29 | +2.50 |
| CiteSeer | **72.37** | 69.51 | 64.77 | +2.86 |
| WikiCS | **71.03** | 67.75 | 66.10 | +3.28 |
| History | **67.31** | 59.86 | 53.30 | +7.45 |
| Children | **31.75** | 24.81 | 30.76 | +6.94 |
| Sportsfit | **75.88** | 61.92 | 66.35 | +13.96 |
| Cornell | **84.82** | 83.28 | 45.54 | +1.54 |
| Texas | **92.51** | 81.66 | 63.10 | +10.85 |
| Wisconsin | **87.17** | 77.75 | 56.60 | +9.42 |
| Washington | **81.66** | 73.14 | 48.90 | +8.52 |

### Ablation Study

| Configuration | Cora | History | Sportsfit | Texas | Description |
|------|------|---------|-----------|-------|------|
| Full DENSE | **75.09** | **67.31** | **75.88** | **92.51** | Full model |
| V1: Random Sampling | 70.48 | 61.80 | 65.60 | 88.24 | Random bundle sampling drops 5–14% |
| V2: Individual Query | 71.96 | 63.95 | 72.61 | 84.49 | Node-by-node LLM queries |
| V3: w/o $\mathcal{L}_{BE}$ | 70.11 | 64.49 | 65.29 | 91.44 | Remove entropy loss |
| V4: w/o $\mathcal{L}_R$ | 73.99 | 66.73 | 75.48 | 86.10 | Remove ranking loss |
| V5: w/ $\mathcal{L}_{IE}$ | 73.43 | 66.29 | 74.05 | 85.03 | Replace bundle supervision with node-wise supervision |
| V6: w/o Bundle Refinement | 73.89 | 66.55 | 73.00 | 91.98 | No bundle refinement |

### Key Findings
- **Bundle sampling** has the largest impact: on datasets with many classes (History: 12 classes, Sportsfit: 13 classes), random sampling causes severe degradation (>10%), as randomly constructed bundles contain more heterogeneous nodes with a weaker mode signal.
- **Bundle query vs. single-node query**: LLM accuracy on bundle queries is significantly higher than on single-node classification, with especially notable gains on CiteSeer and Cornell.
- **Odd bundle sizes outperform even sizes**: Odd sizes avoid ties (e.g., a 2:2 split in a 4-node bundle); $n_B=5$ is optimal.
- Larger numbers of bundles $n_S$ consistently improve performance with diminishing returns; $n_S=100$ offers the best cost-effectiveness trade-off.

## Highlights & Insights
- **The "bundling" idea is elegant**: It reframes the noise problem from "how to clean individual pseudo-labels" to "how to obtain more reliable group-level labels," naturally suppressing individual noise via statistical mode aggregation. This paradigm transfers to any weakly supervised scenario requiring LLM-generated pseudo-labels.
- **Comprehensive theoretical grounding**: Theorem 3.1 proves bundle supervision is more robust to outlier nodes (smaller gradient penalty); Theorems 3.2–3.3 establish convergence. Theory and experiments are mutually consistent.
- **LLM-agnostic**: The method is compatible with GPT-4o, GPT-3.5, DeepSeek-V3, Gemini, and others, and naturally benefits from advances in LLM capabilities.
- **Applicable to both homophilic and heterophilic graphs**: Achieved by distinguishing topological vs. semantic neighbor sampling strategies.

## Limitations & Future Work
- Applicable only to graphs with **textual node attributes**; not directly applicable to graphs with purely numerical features or image-based attributes.
- Bundle size and count require tuning; while $n_B=5, n_S=100$ serve as universal defaults, more fine-grained settings may be needed in extreme cases (e.g., large graphs with hundreds of classes).
- Each bundle requires one LLM query; $n_S=100$ implies 100 API calls, with non-negligible cost and latency.
- Bundle refinement is monotonically shrinking (nodes are only removed, never added); if initial bundle quality is poor, too few nodes may remain after refinement.

## Related Work & Insights
- **vs. LLM-BP**: LLM-BP also uses LLMs to generate supervision signals but relies on node-by-node queries and direct pseudo-label supervision. DENSE surpasses it on all 10 datasets via bundle-level queries and bundle supervision, demonstrating that the bundle strategy systematically outperforms the single-node approach.
- **vs. GOFA/ZeroG**: Graph foundation models require large-scale pretraining data and exhibit unstable performance on out-of-distribution graphs (e.g., university webpage networks). DENSE requires no graph pretraining, offering greater flexibility through the LLM+GNN pipeline.
- **vs. GPT-4o direct inference**: GPT-4o performs poorly on heterophilic graphs (Cornell: 45.54%) due to the absence of graph-structural information. DENSE compensates for this through GNN message passing.

## Rating
- Novelty: ⭐⭐⭐⭐ The "text bundling" perspective is novel; introducing bundle concepts into TAG zero-shot inference is theoretically and intuitively well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 datasets, 15 baselines, 5 LLM backbones, complete ablation and hyperparameter analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is clear, method description is fluent, and theoretical analysis is tightly integrated with experiments.
- Value: ⭐⭐⭐⭐ Zero-shot TAG inference is a practical pain point; the bundle strategy is concise, effective, and generalizable.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [\[NeurIPS 2025\] Practical Bayes-Optimal Membership Inference Attacks](practical_bayes-optimal_membership_inference_attacks.md)
- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](preference-driven_knowledge_distillation_for_few-shot_node_classification.md)

<!-- RELATED:END -->
