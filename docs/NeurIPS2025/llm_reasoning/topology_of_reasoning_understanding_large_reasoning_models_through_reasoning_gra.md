---
title: >-
  [Paper Note] Topology of Reasoning: Understanding Large Reasoning Models through Reasoning Graph Properties
description: >-
  [NeurIPS 2025][LLM Reasoning][reasoning graph] This paper introduces the concept of a *reasoning graph* — a directed graph constructed by clustering the hidden states of LLMs — and analyzes large reasoning models (e.g.…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "reasoning graph"
  - "induction heads"
  - "small-world"
  - "cycle detection"
  - "chain-of-thought"
date: 2026-05-08
content_hash: fedd30583ef75415
---

# Topology of Reasoning: Understanding Large Reasoning Models through Reasoning Graph Properties

**Conference**: NeurIPS 2025
**arXiv**: [2506.05744](https://arxiv.org/abs/2506.05744)
**Code**: [GitHub](https://github.com/gouki510/Topology_of_Reasoning)
**Area**: LLM Reasoning / Interpretability / Graph-Theoretic Analysis
**Keywords**: reasoning graph, induction heads, small-world, cycle detection, chain-of-thought

## TL;DR
This paper introduces the concept of a *reasoning graph* — a directed graph constructed by clustering the hidden states of LLMs — and analyzes large reasoning models (e.g., the DeepSeek-R1 distillation series) along three graph-theoretic dimensions: cycle density, diameter, and small-world index. Reasoning models are found to exhibit significantly more cycles (~5 per sample), larger diameters, and stronger small-world properties (~6×), all of which grow with task difficulty and model scale.

## Background & Motivation
**Background**: Large reasoning models (OpenAI-o1, DeepSeek-R1, Gemini Extended Thinking) achieve state-of-the-art performance via extended chain-of-thought reasoning, yet their internal reasoning mechanisms remain poorly understood.

**Limitations of Prior Work**:
   - Existing analyses operate at the token level (e.g., "aha moments," language-mixing phenomena) and lack a structured analytical framework.
   - There is no satisfying explanation for *why* longer reasoning chains are effective.
   - SFT data construction lacks quantitative guidance — there is no clear definition of what constitutes "good reasoning data."

**Key Challenge**: Improvements in reasoning capability are clearly related to the structure of the reasoning process, yet formal tools for characterizing this structure are absent.

**Key Insight**: Model the LLM reasoning process as a directed graph (reasoning graph) and apply graph-theoretic tools to quantify reasoning structure.

**Core Idea**: Cluster the hidden states at each reasoning step into nodes and treat reasoning paths as edges; use cycle count, diameter, and small-world index to characterize the structural differences between reasoning models and base models.

## Method

### Overall Architecture
Given a math problem, the LLM generates a sequence of reasoning steps → hidden-state representations are extracted at each step (mean-pooled over all tokens at a designated layer) → K-means clustering ($K=200$) assigns each step to a node → edges are added in reasoning order → a reasoning graph $G=(V,E)$ is constructed → graph-theoretic metrics are computed.

### Key Designs

1. **Reasoning Graph Node Extraction**:

    - Function: Map the hidden state of each reasoning step to a discrete node in the graph.
    - Mechanism: For reasoning step $r_t$ (delimited by newlines), compute the mean hidden state at layer $\ell$ across all tokens: $s_t^\ell = \frac{1}{L_t}\sum_{\mu=1}^{L_t} h_{t,\mu}^\ell$. K-means clustering is then applied over all steps across all samples.
    - Design Motivation: Wang et al. showed that K-means cluster centers correspond to elementary computational operations (e.g., addition, multiplication, "wait"-type backtracking), which can be interpreted as atomic reasoning units.

2. **Reasoning Graph Edge Construction**:

    - Function: Connect nodes according to the temporal order of reasoning steps.
    - Mechanism: For each problem $x$, let $\pi = (i_1, i_2, \ldots, i_T)$ denote the cluster indices assigned to successive steps; the edge set is $E = \{(v_{i_t} \to v_{i_{t+1}}) \mid t=1,\ldots,T-1\}$.
    - Result: One directed reasoning graph per sample.

3. **Three Graph-Theoretic Metrics**:

    - **Cycle Density**: Detects repeated node visits in the reasoning graph (excluding self-loops and consecutive repetitions); records cycle detection rate and mean cycle count per sample. Cycles represent the model's "backtracking and verification" behavior.
    - **Diameter**: Computed via Dijkstra's algorithm as the longest shortest path, $\text{diameter} = \max_u \max_{v \neq u} d(u,v)$. Reflects the breadth of the reasoning search space.
    - **Small-World Index**: $S = \frac{C/C_{rand}}{L/L_{rand}}$, where $C$ is the clustering coefficient (local connectivity), $L$ is the average path length, and $C_{rand}, L_{rand}$ are the corresponding random-graph baselines. A high small-world index indicates locally dense yet globally efficient connectivity.

## Key Experimental Results

### Reasoning Model vs. Base Model (DeepSeek-R1-Distill-Qwen-32B vs. Qwen2.5-32B)

| Metric | Base Model | Reasoning Model | Ratio |
|--------|-----------|----------------|-------|
| Cycle detection rate (GSM8K) | ~10% | ~60% | 6× |
| Cycle detection rate (MATH500) | ~15% | ~75% | 5× |
| Cycle detection rate (AIME 2024) | ~20% | ~90% | 4.5× |
| Mean cycles per sample | ~0.5 | ~5 | 10× |
| Small-world index | ~1 | ~6 | 6× |

- Cycle detection rate increases monotonically with task difficulty: GSM8K < MATH500 < AIME 2024.
- Reasoning graph diameter is maximized at deeper layers (depth ~0.9).

### Effect of Model Scale (AIME 2024)

| Model Scale | Cycle Detection Rate | Cycle Count | Diameter | Accuracy |
|-------------|---------------------|-------------|----------|----------|
| 1.5B | Low | Low | Small | Lowest |
| 7B | Medium | Medium | Medium | Medium |
| 14B | **100%** | High | Moderately large | High |
| 32B | ~90% | **Highest** | **Largest** | **Highest** |

- The 14B model achieves the highest cycle detection rate (100%) but lower accuracy than 32B — attributed to language mixing that produces semantically meaningless cycles.
- The 32B model produces more "effective" cycles, the largest diameter, and the highest accuracy.

### SFT Data Quality and Reasoning Graph Properties

| Dataset | MATH500 Accuracy | AIME 2024 Accuracy | Reasoning Graph Diameter |
|---------|-----------------|-------------------|--------------------------|
| s1-v1.0 | 92.6% | 50.0% | Smaller |
| s1-v1.1 | 94.4% | 56.7% | **Larger** |

- Higher-quality SFT data (v1.1) yields larger reasoning graph diameters, which correlates with stronger reasoning performance.

### Key Findings
- **Cycles as a quantitative proxy for "aha moments"**: The frequent backtracking behavior of reasoning models manifests graph-theoretically as cycles, elevating a token-level observation to a structural metric for the first time.
- **Diameter positively correlates with exploration breadth**: A larger reasoning graph diameter indicates that the model explores a more diverse set of reasoning states.
- **Not all cycles are beneficial**: Language mixing in the 14B model produces "spurious cycles," suggesting the need to distinguish effective from ineffective cycles.
- **SFT data quality can be assessed via reasoning graph diameter**: This provides a quantifiable criterion for data curation.

## Highlights & Insights
- **Introduction of the reasoning graph**: Elevating the LLM reasoning process from a one-dimensional token sequence to a graph structure opens the door to graph-theoretic analysis of reasoning.
- **Explanatory power of small-world structure**: The small-world property of reasoning graphs implies locally dense clusters (tightly coupled elementary computation steps) combined with long-range connections (bridging distinct reasoning stages), a structure that facilitates error recovery.
- **Graph-theoretic interpretation of overthinking/underthinking**: Redundant cycles correspond to overthinking; excessively large diameters correspond to divergent reasoning — providing an intuitive diagnostic framework.

## Limitations & Future Work
- The number of K-means clusters $K=200$ is a hyperparameter; different values of $K$ may alter the graph structure.
- Analysis is limited to mathematical reasoning tasks; generalization to code reasoning, logical reasoning, and other domains remains unverified.
- Cycle count, diameter, and small-world index are correlational metrics; causal relationships are not established — it is unclear whether these properties cause or merely accompany improved reasoning.
- The semantics of reasoning graph nodes depend on clustering quality; some nodes may conflate multiple distinct semantic operations.

## Related Work & Insights
- **vs. Wang et al. (K-means reasoning graphs)**: Wang et al. first proposed extracting reasoning nodes via K-means but did not analyze graph-theoretic properties specific to reasoning models; the present work systematically examines cycles, diameter, and small-world structure.
- **vs. traditional CoT analysis**: Conventional CoT analysis focuses on token-level content (step count, chain length); this work operates at the level of hidden-state topology.
- Implications for reasoning-aware training: Loss functions could be designed to directly optimize the topological properties of reasoning graphs (e.g., encouraging an appropriate number of cycles and a suitable diameter).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The reasoning graph concept is novel, and the three graph-theoretic metrics offer a uniquely distinctive analytical perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple datasets × multiple model scales × multiple layer depths, yielding comprehensive analysis.
- Writing Quality: ⭐⭐⭐⭐ Excellent visualizations and clear analysis, though causal inference is insufficiently addressed.
- Value: ⭐⭐⭐⭐ Makes an important contribution to understanding the mechanisms of reasoning models and offers a new perspective for SFT data construction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity](the_illusion_of_thinking_understanding_the_strengths_and_limitations_of_reasonin.md)
- [\[NeurIPS 2025\] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)
- [\[NeurIPS 2025\] Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models](reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)
- [\[NeurIPS 2025\] ProofSketch: Efficient Verified Reasoning for Large Language Models](proofsketch_efficient_verified_reasoning_for_large_language_models.md)
- [\[NeurIPS 2025\] Adaptive Dual Reasoner: Large Reasoning Models Can Think Efficiently by Hybrid Reasoning](adaptive_dual_reasoner_large_reasoning_models_can_think_efficiently_by_hybrid_re.md)

</div>

<!-- RELATED:END -->
