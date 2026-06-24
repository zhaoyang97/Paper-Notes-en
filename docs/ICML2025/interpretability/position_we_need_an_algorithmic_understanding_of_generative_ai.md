---
title: >-
  [Paper Note] Position: We Need An Algorithmic Understanding of Generative AI
description: >-
  [ICML 2025 Spotlight][Interpretability][Algorithmic understanding] Proposes the AlgEval framework, advocating for the systematic study of the algorithms learned and used by generative AI—including algorithmic primitives (vocabulary) and their composition (grammar)—as an alternative understanding pathway to pure scaling, and demonstrates a methodology combining top-down hypothesis with bottom-up validation through a case study on graph navigation tasks.
tags:
  - "ICML 2025 Spotlight"
  - "Interpretability"
  - "Algorithmic understanding"
  - "mechanistic interpretability"
  - "algorithmic primitives"
  - "LLM"
  - "graph search"
  - "inference-time computation"
  - "AlgEval"
date: 2026-05-08
content_hash: fba7ccfe9296efa1
---

# Position: We Need An Algorithmic Understanding of Generative AI

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2507.07544](https://arxiv.org/abs/2507.07544)  
**Code**: None (position paper)  
**Area**: Interpretability  
**Keywords**: Algorithmic understanding, interpretability, mechanistic interpretability, algorithmic primitives, LLM, graph search, inference-time computation, AlgEval

## TL;DR

Proposes the AlgEval framework, advocating for the systematic study of the algorithms learned and used by generative AI—including algorithmic primitives (vocabulary) and their composition (grammar)—as an alternative understanding pathway to pure scaling, and demonstrates a methodology combining top-down hypothesis with bottom-up validation through a case study on graph navigation tasks.

## Background & Motivation

### Background

**Background**: LLMs have achieved immense success, but a fundamental question has been under-appreciated: **what algorithms do LLMs actually use to solve problems?**

Imbalance in current research priorities:

**Scale supremacy**: Since the Bitter Lesson (Sutton 2019), the focus of the field has been on data and computation scale rather than understanding.

**Limitations of mechanistic interpretability**: Existing works focus on low-level circuit analysis (attention heads, neurons) but lack a systematic understanding at the algorithmic level.

**Sustainability crisis**: Data and computation costs continue to escalate with diminishing returns; biological intelligence provides an existence proof of much higher efficiency.

**Rise of inference-time computation**: Models like o1/R1 are shifting toward inference-time computation, yet the algorithms involved remain unclear.

**Core Argument**: The ML community should prioritize the algorithmic understanding of generative AI instead of solely chasing performance improvements. Algorithmic understanding can bring:

### Limitations of Prior Work

**Limitations of Prior Work**: More efficient training and inference.

### Key Challenge

**Key Challenge**: More trustworthy models.

### Solution Strategy

**Solution Strategy**: Better multi-agent system design.

### Supplementary Notes

**Supplementary Notes**: Compliance and safety.

## Method

### AlgEval Framework

AlgEval is a research framework for algorithmic evaluation and understanding, fundamentally encompassing two levels:

**Algorithmic Primitives**—i.e., the "vocabulary":

- **Basic primitives**: Discovered primitives include function vectors (in-context learning mappings), vector arithmetic computation, steering vectors, copy suppression, key-value memory retrieval, etc.
- **Primitives as base algorithms**: Composing multiple primitives to form more complex functional circuits, such as indirect object identification (induction + inhibition + duplicate token + copy heads).
- **Algorithms as primitives**: Transformers can exhibit classical ML methods (kernel methods, SVMs, Markov chains, higher-order optimization).

**Algorithmic Composition**—i.e., the "grammar":

- How are primitives composed into more complex algorithms?
- Is the composition compositional (compositionality)?
- Existing evidence indicates persistent failures of LLMs in compositional reasoning.

### Methodological Pathway

AlgEval advocates combining the following methods:

1. **Representation and attention analysis**: Inter-layer representation similarity matrices, attention flow analysis.
2. **Subgraph and circuit discovery**: Activation patching, automatic circuit discovery, attribution patching.
3. **Inference-time computation analysis**: Analyzing algorithmic behaviors in sequential outputs (e.g., backtracking, search).
4. **Five-step research workflow**: (a) Identify primitive library → (b) Build simple task suites → (c) Create mechanism library → (d) Analyze composition → (e) Ablation verification.

### Case Study: Graph Navigation Task

Studying the algorithmic behavior of LLMs in simple tree graph navigation, using Llama-3.1-8B/70B. The model is prompted to determine whether a direct path exists from a starting node to a target node.

**Hypothesis**: The model might use classical search algorithms (BFS/DFS/Dijkstra).

**Attention Analysis Results**:
- The final token pays significantly more attention to nodes on the correct path ($b = 0.33, p < .001$), significant in 14 out of 32 layers.
- Early and middle layers focus on pairwise node links, layers 13-14 focus on the target node, and attention shifts toward the correct answer around layer 19.
- Exhibiting a **cascaded attention diffusion** pattern: propagating step-by-step from each node to its predecessors.

**Representation Analysis Results**:
- t-SNE visualization shows that the "lobby" (starting node) token gradually separates from other nodes across layers.
- The representational distance between the target node W and the nearest competitor node Q increases layer by layer ("competitive separation").
- Representations change smoothly across consecutive layers (similarity > 0.95), showing no obvious discrete steps.

**Comparison with Classical Algorithms**:
- Comparing the sequence of node pairs with the highest representation similarity across consecutive layers with all possible BFS/DFS trajectories.
- Matching degree is extremely low: BFS average of 0.18, DFS average of 0.24.
- **Conclusion**: LLMs do not execute classical search algorithms; instead, they use some policy-dependent incremental attention mechanism.

**8B vs 70B Comparison**: Both models exhibit similar attention and representation patterns, but the separation of correct/incorrect paths is more pronounced in the 70B model. There is no fundamental difference in search strategy.

## Key Experimental Results

### Quantitative Results of the Case Study

- Attention of the final token on correct vs. incorrect paths: $b = 0.33, SE = 0.07, t(2015) = 4.51, p < .001$
- 14 out of 32 layers show significantly higher attention to the correct path, with only 3 layers showing the reverse.
- BFS/DFS trajectory matching rate: BFS 0.18, DFS 0.24.
- Representation similarity between consecutive layers: > 0.95 (layers 4-32).

## Highlights & Insights

1. **Algorithmic Level of Marr's Tri-Level Hypothesis**: Explicitly advocating for understanding LLMs at the algorithmic level (rather than just the computational or implementational level), filling an important gap in current research focus.

2. **Combining Top-Down with Bottom-Up**: Unlike the pure bottom-up exploration of mechanistic interpretability, AlgEval emphasizes forming algorithmic hypotheses first before validation, which is more scientifically rigorous in terms of methodology.

3. **Negative Findings from the Case Study**: LLMs do not use BFS/DFS but instead adopt an attention-based incremental search strategy—which is exactly the value of the AlgEval framework in discovering models' deviation from classical algorithms.

4. **Algorithmic Analysis of Inference-Time Computation**: Inference-time computation in models like o1/R1 provides a more analyzable space for algorithmic analysis compared to feed-forward computation alone.

5. **Practical Impact**: Algorithmic understanding can lead to more efficient training (algorithm-aware architecture design), lower emissions, and better safety compliance.

## Limitations & Future Work

- The case study is limited to simple two-step tree graph navigation; behavior on complex graphs remains unknown.
- AlgEval is currently more of a research vision than a mature methodology, with many key steps (e.g., automatic primitive discovery, composition analysis) still to be developed.
- The analysis focuses on Llama-3.1, leaving other model architectures unverified.
- The theoretical framework lacks formal guarantees.
- As a position paper, its experimental contributions are limited.

## Related Work & Insights

- **Mechanistic Interpretability**: Olah 2020 (circuits), Wang 2023 (IOI circuit), Olsson 2022 (induction heads)
- **Algorithmic Reasoning**: Weiss 2021 (RASP), Zhou 2024 (RASP extension)
- **Representation Analysis**: Todd 2024 (function vectors), Kornblith 2019 (CKA)
- **Inference-Time Computation**: Snell 2024 (scaling inference), DeepSeek-AI 2025 (R1)

## Rating

⭐⭐⭐ — As a position paper, it proposes an important research direction and a systematic framework, with the case study providing a concrete example. However, the substantial technical contribution is limited, leaving many research directions at the level of future outlooks. While the core insight (LLMs not using classical search) is interesting, it is not surprising.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Position-aware Automatic Circuit Discovery](../../ACL2025/interpretability/position-aware_automatic_circuit_discovery.md)
- [\[AAAI 2026\] Quiet Feature Learning in Algorithmic Tasks](../../AAAI2026/interpretability/quiet_feature_learning_in_algorithmic_tasks.md)
- [\[ICML 2026\] Riemannian Generative Decoder](../../ICML2026/interpretability/riemannian_generative_decoder.md)
- [\[ICML 2025\] SafetyAnalyst: Interpretable, Transparent, and Steerable Safety Moderation for AI Behavior](safetyanalyst_interpretable_transparent_and_steerable_safety_moderation_for_ai_b.md)
- [\[NeurIPS 2025\] OrdShap: Feature Position Importance for Sequential Black-Box Models](../../NeurIPS2025/interpretability/ordshap_feature_position_importance_for_sequential_black-box_models.md)

</div>

<!-- RELATED:END -->
