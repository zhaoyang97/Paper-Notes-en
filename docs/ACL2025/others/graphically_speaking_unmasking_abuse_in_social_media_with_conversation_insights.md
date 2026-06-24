---
title: >-
  [Paper Note] Graphically Speaking: Unmasking Abuse in Social Media with Conversation Insights
description: >-
  [ACL 2025][Abusive Language Detection] This paper proposes a context-aware abusive language detection framework based on Graph Attention Networks (GAT). It models Reddit conversations as graph structures (nodes = comments, edges = reply relations) and utilizes an affordance-based graph pruning strategy derived from Reddit’s interface rendering logic to preserve key context. A 3-layer GAT model achieves an F1 score of 0.7624, significantly outperforming no-context baselines an…
tags:
  - "ACL 2025"
  - "Abusive Language Detection"
  - "Graph Attention Networks"
  - "Conversation Context"
  - "Reddit"
  - "Context-Aware Classification"
date: 2026-05-08
content_hash: ab992a091b04afee
---

# Graphically Speaking: Unmasking Abuse in Social Media with Conversation Insights

**Conference**: ACL 2025  
**arXiv**: [2504.01902](https://arxiv.org/abs/2504.01902)  
**Code**: Open-sourced (link provided in the paper)  
**Authors**: Célia Nouri, Jean-Philippe Cointet, Chloé Clavel  
**Institution**: INRIA ALMAnaCH, Sciences Po médialab, Télécom Paris  
**Area**: Other  
**Keywords**: Abusive Language Detection, Graph Attention Networks, Conversation Context, Reddit, Context-Aware Classification  

## TL;DR

This paper proposes a context-aware abusive language detection framework based on Graph Attention Networks (GAT). It models Reddit conversations as graph structures (nodes = comments, edges = reply relations) and utilizes an affordance-based graph pruning strategy derived from Reddit’s interface rendering logic to preserve key context. A 3-layer GAT model achieves an F1 score of 0.7624, significantly outperforming no-context baselines and flattened context methods, with a particularly pronounced improvement (+4.75%) on context-sensitive samples.

## Background & Motivation

**Background**: Detecting abusive language (AL) in social media is a major challenge. AL encompasses hate speech, toxic content, offensive language, and cyberbullying. Most existing ALD models classify each comment independently, ignoring conversational context.

**Importance of Context**:  
   - Pavlopoulos et al. (2020) found that 5% of labels changed after incorporating context.  
   - Menini et al. (2021) found that the proportion of abuse labels decreased from 18% to 10% when context was added.  
   - These contradictory findings highlight the complexity of context integration.

**Limitations of Prior Work**:  
   - Most methods only consider the immediate preceding comment or the original post as context.  
   - Flattening approaches (simple concatenation) fail to model conversational structures and may even introduce noise.  
   - Existing graph-based methods often construct fully connected graphs or rely on platform-specific features (such as retweet paths), lacking generalizability.

**Core Motivation**: To use graph structures to explicitly preserve conversational topology, allowing contextual information to propagate across multi-layer interactions, thereby better capturing contextual dependencies in abusive language.

## Method

### Overall Architecture

**BERT Text Encoding** → **Affordance-based Graph Construction** → **GAT Multi-layer Context Aggregation** → **Concatenation & Classification**

### Problem Formulation

Given a conversation thread $T = \{c_1, ..., c_N\}$, a directed graph $G(T) = (V, E)$ is constructed, where nodes $V$ represent comments and edges $(c_j, c_i) \in E$ indicate that $c_i$ is a reply to $c_j$. Each node feature $\mathbf{x}_i \in \mathbb{R}^d$ is derived from the [CLS] token embedding of BERT. The objective is to predict the label $y_i \in \{0, 1\}$ (0 = non-abusive, 1 = abusive) of the target comment $c_i$.

### Affordance-based Graph Pruning Strategy

Reddit conversations can contain hundreds of comments, but users typically see only a portion of the content when writing replies. This paper designs a graph pruning strategy that simulates Reddit's default rendering algorithm:

For each target comment $c_i$, the following nodes are retained to form the subgraph $G_i$:
1. **Original post** $c_1$ (blue)
2. **Top-5 upvoted replies to the root post** (green) — ranked by upvotes minus downvotes
3. **The highest-voted sub-reply of each Top-5 reply** (yellow)
4. **The complete reply path from the root post to the target comment** (red)

In addition, every node is connected to the original post, simulating the behavioral flow where users read replies after viewing the post. The pruned graph contains at most 25 nodes, with a median of 9 nodes.

### GAT Model

A Graph Attention Network (Veličković et al., 2018) is used to learn context representations:

**Layer-wise Update**:
$$\mathbf{x}_m^{(l+1)} = \text{ELU}\left(\sum_{n \in \mathcal{N}(m)} \alpha_{mn} \mathbf{W}^{(l)} \mathbf{x}_n^{(l)}\right)$$

**Attention Coefficients**:
$$\alpha_{mn} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}^{(l)}\mathbf{x}_m^{(l)} \| \mathbf{W}^{(l)}\mathbf{x}_n^{(l)}]))}{\sum_{k \in \mathcal{N}(m)} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}^{(l)}\mathbf{x}_m^{(l)} \| \mathbf{W}^{(l)}\mathbf{x}_k^{(l)}]))}$$

**Final Classification**:
After $L$ GAT layers, the graph embedding of the target node $\mathbf{x}_i^{(L)}$ is concatenated with the original text embedding $\mathbf{x}_i \to$ fully connected layer (dimension reduction to $d=768$) $\to$ classification layer $\to$ Sigmoid output.

The model parameter size is extremely small: BERT (110M) + GAT (~6M per layer), and inference takes only 100-200ms, which is far lower than the multi-second latency of generative large models (GLMs).

### Loss & Training

Standard binary cross-entropy loss.

## Key Experimental Results

### Dataset: Contextual Abuse Dataset (CAD)

- Approximately 25,000 Reddit comments from 16 subreddits known for abusive content.
- A target-based six-class taxonomy is adopted: 3 abusive categories (Identity-directed / Affiliation-directed / Person-directed) + 3 non-abusive categories (Neutral / Counter Speech / Non-Hateful Slurs).
- Rigorous annotation process: each comment is annotated by 2 annotators $\to$ disagreements are resolved via consensus arbitration $\to$ final review by experts.
- Fleiss' Kappa = 0.583 (moderate agreement).
- This work uses binary classification (abusive vs. non-abusive).
- Annotators flagged approximately 1/3 of the abusive samples as requiring conversational context for correct classification.

### RQ1: Optimal Context Window

| GAT Layers | (Max, Median) Node Count | Mean F1 ± CI |
|----------|-------------------|--------------|
| 1 | (3, 2) | 0.7537 ± 0.0069 |
| 2 | (7, 3) | 0.7613 ± 0.0041 |
| **3** | **(12, 5)** | **0.7624 ± 0.0058** |
| 4 | (13, 7) | 0.7592 ± 0.0065 |
| 5 | (14, 8) | 0.7609 ± 0.0043 |

**Key Findings**:
- A 3-layer GAT achieves the best performance (F1 = 0.7624), and its receptive field (median of 5 nodes, maximum of 12 nodes) covers most nodes in the affordance-based graph.
- Performance declines slightly beyond 3 layers, as distant comments introduce irrelevant information.
- The differences between 2 to 5 layers are not statistically significant, but 1 layer is clearly insufficient.

### RQ2: Graph Models vs. Flattened Models

| Model | Mean F1 ± CI |
|------|-------------|
| No Context (BERT) | 0.7453 ± 0.0076 |
| Text-Concat (Longformer) | 0.7417 ± 0.0081 |
| Embed-Concat | 0.7488 ± 0.0025 |
| **GAT 3L (Ours)** | **0.7624 ± 0.0058** |

**Key Findings**:
- **The flattened context method (Text-Concat) actually underperforms the no-context baseline** — simple concatenation introduces noise, corroborating findings from previous research.
- GAT 3L significantly outperforms all baselines.
- Embed-Concat is slightly better than No Context, but is far inferior to GAT.

### Context-Sensitive Sample Analysis

| Model | Context-Sensitive (CSS) PCP | Context-Free (CFS) PCP |
|------|---------------------|---------------------|
| No Context | 70.71% ± 2.61 | 81.67% ± 2.89 |
| Text-Concat | 70.80% ± 3.61 | 82.33% ± 1.88 |
| Embed-Concat | 70.97% ± 1.19 | 83.00% ± 1.98 |
| **GAT 3L** | **74.07% ± 1.12** | **84.21% ± 2.14** |

**Key Findings**:
- CSS is more challenging for all models (approximately 10% lower).
- GAT shows a more significant improvement on CSS: +4.75% compared to No Context (while CFS only improved by +3.11%).
- Compared to Text-Concat: CSS +4.62% (while CFS only improved by +2.28%).
- This indicates that GAT has the greatest advantage in scenarios requiring contextual inference.

### Case Study

Taking the conversation graph shown in Section 1 of the paper as an example:
- The target comment "No u" was annotated as abusive, but it is impossible to determine when viewed in isolation.
- The source of abuse (a homophobic slur) lies **3 hops away** from the target comment.
- A 1-layer or 2-layer GAT cannot classify it correctly, but GAT with 3+ layers can — because a 3-hop receptive field exactly covers the key context.
- Attention weight analysis: edges along the reply path receive high attention weights, while edges from the original post to other nodes receive very low attention — demonstrating that the model learns to dynamically identify and prioritize key contextual cues.

### Comparison with GLMs

- Proprietary GLMs (e.g., GPT-4 with 1.76T parameters) lack transparency and are unsuitable for content moderation.
- Open-source GLMs (e.g., LLaMA-2-13B, DeepSeek-V2) have massive parameters.
- Our method: BERT (110M) + GAT (~6M/layer) has only around 116-128M parameters.
- Inference speed: 100-200ms vs. several seconds for GLMs.

## Highlights & Insights

1. **Affordance-based Graph Pruning**: Simulating what Reddit users actually see to construct the graph is an elegant "user-centric" design that is more reasonable than constructing fully connected graphs or random pruning.
2. **Simple Context Concatenation is Harmful**: Text-Concat even underperforms the no-context baseline, which directly challenges the intuition that "more context is always better."
3. **Graph Structure Amplifies Advantages on CSS**: The graph-based approach achieves the largest improvement precisely on samples that require context the most, directly proving the value of structured context modeling.
4. **Lightweight and Efficient**: With ~120M parameters and 100-200ms inference latency, it is far superior to GLM alternatives and highly suitable for large-scale deployment.
5. **Attention-based Interpretability**: The attention weights of GAT can directly reveal which contextual nodes the model focuses on, providing interpretability for content moderation.

## Limitations & Future Work

1. Evaluated only on a single dataset (CAD) — dynamic validation across more cross-platform and multilingual datasets is needed.
2. Utilizes only textual information: Reddit posts often contain images (especially the original post), which are not incorporated as multimodal signals.
3. Fails to integrate user behavior and social dynamics, such as user history or social networks.
4. The performance differences between 2 to 5 GAT layers are not statistically significant — larger datasets may be required to resolve the optimal number of layers.
5. The affordance-based strategy is designed specifically for Reddit; whether it is applicable to other platforms (Twitter/X, Facebook) requires further validation.
6. Limited capability in detecting implicit abuse and irony, which remains a general bottleneck in ALD.

## Related Work & Insights

- **Impact of Context on Annotation**: Pavlopoulos et al. (2020) $\to$ Menini et al. (2021) $\to$ contradictory findings highlight the complexity of context integration.
- **Flattened Context Methods**: Text-Concat (BERT/Longformer) $\to$ Embed-Concat $\to$ History Embedding $\to$ inconsistent performance.
- **Graph-based Methods**: Fully connected graphs (Wang et al., 2020) $\to$ temporal graphs (Cecillon et al., 2021) $\to$ reply graphs (Hebert et al., 2024; Agarwal et al., 2023).
- **GraphNLI** (Agarwal et al., 2023): Fixed-probability random walks + distance decay $\to$ Our GAT dynamically learns context node importance.
- **Large Language Model Methods**: GPT-4/DeepSeek through prompting $\to$ lack transparency, are computationally expensive, and are unsuitable for real-time moderation.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — The systematic application of the affordance-based graph pruning and GAT to ALD is a major highlight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The layer ablation, multi-baseline comparison, CSS/CFS analysis, and attention visualization are all comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear motivation, vivid case analysis, and excellent illustrations.
- **Value**: ⭐⭐⭐⭐⭐ — Lightweight, efficient, interpretable, and suitable for large-scale deployment.
- **Limitations**: Evaluated on only a single dataset, limited to the pure text modality, and relies on a Reddit-specific strategy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Narrative Media Framing in Political Discourse](narrative_media_framing_in_political_discourse.md)
- [\[ACL 2025\] SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents](sotopia-ensuremathomega_dynamic_strategy_injection_learning_and_social_instructi.md)
- [\[ACL 2025\] GA-S3: Comprehensive Social Network Simulation with Group Agents](ga-s3_comprehensive_social_network_simulation_with_group_agents.md)
- [\[ICML 2025\] Revisiting the Predictability of Performative, Social Events](../../ICML2025/others/revisiting_the_predictability_of_performative_social_events.md)
- [\[ICML 2025\] Feature Learning beyond the Lazy-Rich Dichotomy: Insights from Representational Geometry](../../ICML2025/others/feature_learning_beyond_the_lazy-rich_dichotomy_insights_from_representational_g.md)

</div>

<!-- RELATED:END -->
