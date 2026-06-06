---
title: >-
  [Paper Note] The Shape of Reasoning: Topological Analysis of Reasoning Traces in Large Language Models
description: >-
  [ICML 2026][Reinforcement Learning][Reasoning Trace Evaluation] This paper treats LLM chain-of-thought (CoT) as "point clouds" in an embedding space and utilizes Topological Data Analysis (TDA) to extract persistent homo…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Reasoning Trace Evaluation"
  - "Topological Data Analysis"
  - "Persistent Homology"
  - "Smith-Waterman Alignment"
  - "AIME"
date: 2026-05-08
content_hash: 317305372333172f
---

# The Shape of Reasoning: Topological Analysis of Reasoning Traces in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2510.20665](https://arxiv.org/abs/2510.20665)  
**Code**: To be confirmed  
**Area**: LLM Reasoning / Reasoning Evaluation / Interpretability  
**Keywords**: Reasoning Trace Evaluation, Topological Data Analysis, Persistent Homology, Smith-Waterman Alignment, AIME

## TL;DR
This paper treats LLM chain-of-thought (CoT) as "point clouds" in an embedding space and utilizes Topological Data Analysis (TDA) to extract persistent homology features as objective measures of reasoning quality. On the AIME dataset, TDA features demonstrate significantly higher predictive power for Smith-Waterman alignment scores (average $R^2=0.236$) compared to traditional graph statistics (average $R^2=0.064$).

## Background & Motivation

**Background**: Current evaluation of LLM reasoning quality still relies heavily on expert-written rubrics, manual annotation, and pairwise judging, which are slow and expensive. Automated methods mostly follow graph-based proxies (constructing reasoning traces as directed graphs and calculating clustering coefficients, diameter, small-world indices, etc.), using structural connectivity to approximate "reasoning quality."

**Limitations of Prior Work**: (1) Most reasoning datasets only provide final answers; using answer accuracy as a proxy for reasoning quality has been debunked by several studies—LLMs can arrive at correct answers through flawed reasoning. (2) Graph statistics compress high-dimensional embeddings into a few scalars, losing geometric information about how "reasoning processes unfold in semantic space." (3) Aggregation methods like self-consistency discard intermediate reasoning paths, failing to evaluate the reasoning process itself.

**Key Challenge**: Graph metrics only describe discrete connectivity between nodes. However, the true difference between effective and flawed reasoning may reside in higher-dimensional geometric structures—such as local compactness, the persistence of cycles (detours), and aggregation patterns across different scales—which cannot be fully characterized by single-scalar graph statistics.

**Goal**: (1) Construct a "reasoning ground truth" under realistic conditions lacking step-level labels; (2) Quantify reasoning quality using a set of invariants independent of surface paraphrasing; (3) Verify whether these invariants predict reasoning quality more accurately than graph statistics.

**Key Insight**: The authors embed each step of a reasoning trace into $\mathbb{R}^d$ using sentence embeddings, resulting in an ordered point cloud. Two seemingly different correct solutions may be "homeomorphic"—like a donut and a coffee cup—sharing a deep geometric structure, whereas incorrect reasoning lacks this structure. Persistent homology in TDA provides tools to characterize "shape invariants" (connected components $H_0$, one-dimensional holes/cycles $H_1$).

**Core Idea**: Use Smith-Waterman alignment in embedding space to align LLM reasoning traces to expert solutions, using the alignment score as the ground truth. Apply Vietoris-Rips filtration to the embedding point clouds of the reasoning traces to extract $H_0/H_1$ persistent homology features, and validate their predictive power for alignment scores using OLS regression.

## Method

The entire workflow consists of four stages: generating reasoning traces → aligning to expert solutions → extracting topological features → comparing against graph feature baselines.

### Overall Architecture

The input consists of AIME (American Invitational Mathematics Examination) problems and their expert solutions. Models generate reasoning traces $r_i$ via answer-blind prompts using local Ollama services. Both $r_i$ and expert solutions $s_i$ are segmented into step lists according to specific rules, and each step is embedded using all-mpnet-base-v2. In the embedding space, Smith-Waterman alignment is performed to obtain an "alignment score" as a quality proxy. Concurrently, Vietoris-Rips persistence diagrams are computed for the reasoning trace point clouds to extract 28-dimensional TDA features. Finally, OLS regression is used to predict alignment scores using TDA features, graph features, and their combination, comparing $R^2$ and adjusted $R^2$.

### Key Designs

1.  **Smith-Waterman Alignment in Embedding Space**:
    *   **Function**: Aligns each step of the LLM reasoning trace to corresponding steps in expert solutions, using alignment quality as a step-level ground truth to address the lack of step-level labels.
    *   **Mechanism**: Reasoning traces $R_i=(r_{i,1},\dots,r_{i,m})$ and expert solutions $S_i=(s_{i,1},\dots,s_{i,n})$ are embedded as $X_i^{(r)}\in\mathbb{R}^{m\times d}$ and $X_i^{(s)}\in\mathbb{R}^{n\times d}$. Cosine similarity serves as the match score $s_{uv}$, complemented by a gap penalty $\gamma>0$. The standard DP recurrence $H_{u,v}=\max\{0,\,H_{u-1,v-1}+s_{uv},\,H_{u-1,v}-\gamma,\,H_{u,v-1}-\gamma\}$ is executed. Alignment pairs $\mathcal{A}_i$ are retrieved by backtracking from $\arg\max H_{u,v}$, aggregated into mean alignment score and gold-step coverage.
    *   **Design Motivation**: Adopts the local alignment concept from biological sequence analysis—expert solutions and model reasoning may only align in specific segments; global alignment would be degraded by redundant thoughts. Matching based on embedding cosine similarity allows for "semantically equivalent but differently phrased" steps to be aligned.

2.  **Vietoris-Rips Filtration + Persistent Homology Features**:
    *   **Function**: Transforms the embedding point cloud of reasoning traces into a set of topological invariants independent of coordinate transformations and robust to surface paraphrasing, serving as an objective measure of reasoning quality.
    *   **Mechanism**: On the set of embedded steps $X=\{\mathbf{x}_1,\dots,\mathbf{x}_\ell\}$, cosine distance $\mathrm{dist}(\mathbf{x}_p,\mathbf{x}_q)=1-\langle\mathbf{x}_p,\mathbf{x}_q\rangle/(\|\mathbf{x}_p\|\|\mathbf{x}_q\|)$ is defined. Vietoris-Rips complexes are constructed as the scale parameter varies, recording "birth-death" times of topological features to obtain persistence diagrams $\mathcal{D}_k=\{(b_j^{(k)},d_j^{(k)})\}$ ($k\in\{0,1\}$). Three groups of features are extracted: (i) VR summary statistics (mean life, entropy, etc.); (ii) Betti curve descriptors (centroid, spread, width); (iii) persistence landscape descriptors, totaling 28 dimensions.
    *   **Design Motivation**: $H_0$ encodes "how ideas cluster and merge in embedding space," while $H_1$ encodes "the presence of detours and cycles." Together, they correspond to a profile of good reasoning: "local compactness + global retrieval-convergence." Topological features are more stable than graph statistics across different embedders and distance functions.

3.  **Graph Statistic Baseline + Topological-Graph Translatability Analysis**:
    *   **Function**: Constructs trace graphs on the same embedding data and calculates has_loop, loop_count, diameter, average clustering $\overline{C}$, average shortest path $\overline{L}$, and small-world index as fair baselines. It also uses TDA features to regress back to graph statistics to explain the validity of graph features.
    *   **Mechanism**: Graphs are built following Minegishi et al. 2025. OLS is used to regress 5 graph statistics onto TDA features. Systematic relationships are found: e.g., $H_0$ mean life $+$ improves clustering, $H_0$ Betti centroid $+$ increases path length and diameter, $H_1$ landscape mean $+$ increases loop count. Generally, $H_1$ controls "cycle richness" while $H_0$ controls "global cohesion and efficiency." TDA yields $R^2\approx 0.35$-$0.38$ for 4 global graph statistics, but only $\approx 0.07$ for loop incidence.
    *   **Design Motivation**: The authors aim not only to prove "TDA is stronger than graph statistics" but also to explain "why"—many graph statistics are essentially compressed projections of TDA at a certain scale. Retaining the entire filtration provides richer information.

### Loss & Training
This study does not train any models. The evaluation process involves sentence embeddings (frozen all-mpnet-base-v2) + Smith-Waterman DP + Vietoris-Rips persistent homology + OLS regression. Main hyperparameters include the cosine distance threshold, Smith-Waterman gap penalty $\gamma$, and Vietoris-Rips maxdim$=1$.

## Key Experimental Results

### Main Results

The dataset comprises AIME 2020-2025, with 180 (model, problem) observations across 8 LLM configurations (Qwen3 / DeepSeek-R1 / GPT-OSS at various scales). The regression target is the Smith-Waterman alignment score.

| Model | Graph $R^2$ | TDA $R^2$ | Graph+TDA $R^2$ | $\Delta R^2$ vs TDA |
|:---|:---|:---|:---|:---|
| Qwen3-8B | 0.054 | 0.273 | 0.312 | +14.3% |
| Qwen3-32B | 0.088 | 0.181 | 0.233 | +28.7% |
| Qwen3-235B | 0.024 | 0.163 | 0.167 | +2.5% |
| DeepSeek-r1-7B | 0.047 | 0.210 | 0.226 | +7.6% |
| DeepSeek-r1-32B | 0.057 | 0.190 | 0.226 | +18.9% |
| DeepSeek-r1-70B | 0.058 | 0.249 | 0.300 | +20.5% |
| GPT-OSS-20B | 0.081 | 0.296 | 0.327 | +10.5% |
| GPT-OSS-120B | 0.101 | 0.327 | 0.368 | +12.5% |
| **Mean** | **0.064** | **0.236** | **0.270** | **+14.4%** |

TDA alone increases $R^2$ to 3-4 times that of graph statistics; adjusted $R^2$ is higher for TDA in 7 out of 8 configurations. Adding graph features to TDA increases raw $R^2$ by 14.4% on average, but adjusted $R^2$ drops by $-3.4\%$ on average, even decreasing for Qwen3-235B and DeepSeek-r1-7B—indicating that graph features are largely covered by the TDA subspace.

### Ablation Study

| Feature Cluster | Meaning | Relation to Alignment Score | Physical Interpretation |
|:---|:---|:---|:---|
| Cluster 2 ($H_0$ betti_spread) | Expansion of merging across scales | Positive | Reasoning "clusters" across multiple scales simultaneously; clear backbone |
| Cluster 3 ($H_0$ betti_width) | Span of merging scales | Negative | Larger span indicates many break points; poor overall coherence |
| Cluster 12 ($H_1$ betti_width) | Scale range of 1D holes | Positive | Including moderate short-to-medium scale local checks is beneficial |
| Cluster 16 ($H_1$ max_birth/death) | Late-appearing large-scale holes | Weak Negative | Large-scale late detours usually signify wandering |

### Key Findings
- **TDA significantly outperforms graph statistics**: TDA-only average $R^2$ is $\approx 3.7\times$ that of Graph-only; adjusted $R^2$ is superior for TDA in 7/8 models. High-quality reasoning is more akin to high-dimensional geometric invariants than discrete connectivity.
- **Topological features are translatable back to graph features**: TDA achieves $R^2\approx 0.35$-$0.38$ for clustering, path length, diameter, and small-world indices, but only $\approx 0.07$ for loop count—loop multiplicity depends on local connectivity patterns of very few nodes and is almost irreplaceable by geometric invariants.
- **Profile of good reasoning**: Higher-quality reasoning tends to "maintain a cohesive main line + include short, diverse local verifications - avoid large-scale late detours," consistent with human intuition for "clear, checkable, and focused" reasoning.
- **Dataset ceiling**: Even the best configuration ($R^2\approx 0.37$) explains less than 40% of alignment score variance, indicating that embedding geometry is only part of the reasoning quality signal and cannot fully replace semantic judgment.

## Highlights & Insights
- Viewing chain-of-thought as an ordered point cloud in embedding space and applying TDA is a refreshing "geometric perspective"—it fills the gap between token-level probabilities and discrete graph statistics.
- The Smith-Waterman embedding space alignment is a versatile tool: it can be reused in any scenario requiring model generation alignment to a reference without verbatim matching (e.g., long-form QA, code logic alignment).
- Explaining "why TDA is strong" through reverse regression (TDA → Graph Statistics) provides a solid empirical structure that is more informative than mere performance gains.
- Implications for future RL/RLHF training: Using scalars like $H_0$ betti_spread or $H_0$ mean life as process rewards could provide training signals that are cheaper than ORM and more granular than graph PRM.

## Limitations & Future Work
- **Narrow dataset scope**: Only AIME math problems (olympiad level) were used, where reasoning styles are relatively consistent (mostly symbolic derivation). Generalization to commonsense reasoning, science QA, or code reasoning is unknown.
- **Topological features $\neq$ Reasoning structure**: Persistence diagrams characterize the geometry of sentence embeddings under cosine distance, which does not directly map to symbolic branching or backtracking. Changes in embedders, segmentation, or distance metrics can cause $H_1$ cycles to appear or disappear. Thus, the interpretation of $H_1$ as detours is explanatory, not a causal assertion.
- **Low absolute explanatory power**: With a max $R^2\approx 0.37$, over 60% of alignment variance remains unexplained; TDA alone cannot yet replace human evaluation.
- **Potential improvements**: (1) Ensemble multiple embedders or use contrastively fine-tuned "reasoning vectors"; (2) Feed TDA features into supervised learning instead of just OLS to break linear bottlenecks; (3) Convert current batch evaluations into process rewards to verify if "high betti_spread" transfers to downstream reasoning performance.

## Related Work & Insights
- **vs Minegishi et al. 2025 (Topology of Reasoning)**: Both characterize reasoning trace "shapes," but Minegishi uses graph statistics after constructing directed graphs; this paper demonstrates that graph statistics are low-dimensional projections of geometry, with TDA explaining $\approx 17$ additional percentile points of variance.
- **vs Xiong et al. 2025 (reasoning-graph framework)**: Both focus on structural analysis and emphasize that "longer $\neq$ better"; this paper provides a more fine-grained geometric measure than simple branching/convergence.
- **vs Gardinazzi et al. 2025 (zigzag persistence in transformers)**: Both apply persistent homology to LLMs, but Gardinazzi looks at inter-layer representation evolution, whereas this paper examines the geometry of the inference-time reasoning trace itself.
- **vs Ton et al. 2025 (information-theoretic step contribution)**: Both aim for automated reasoning evaluation; one measures "contribution per step" via information theory, while the other measures the "shape of the whole trace." The two routes are complementary.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)
- [\[ICML 2026\] d2: Improving Reasoning in Diffusion Language Models via Trajectory Likelihood Estimation](d2_improving_reasoning_in_diffusion_language_models_via_trajectory_likelihood_es.md)
- [\[ICML 2026\] Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory](game_of_thought_robust_information_seeking_with_large_language_models_using_game.md)
- [\[NeurIPS 2025\] GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining](../../NeurIPS2025/reinforcement_learning/graphchain_large_language_models_for_large-scale_graph_analysis_via_tool_chainin.md)
- [\[ICLR 2026\] Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models](../../ICLR2026/reinforcement_learning/co-rewarding_stable_self-supervised_rl_for_eliciting_reasoning_in_large_language.md)

</div>

<!-- RELATED:END -->
