---
title: >-
  [Paper Note] Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought
description: >-
  [NeurIPS 2025][LLM Reasoning][continuous chain of thought] This paper theoretically demonstrates the expressive advantage of continuous chain-of-thought (Coconut) on directed graph reachability: a two-layer Transformer u…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "continuous chain of thought"
  - "superposition-state reasoning"
  - "graph reachability"
  - "Transformer expressiveness"
  - "Coconut"
date: 2026-05-08
content_hash: 6e7592559c651113
---

# Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought

**Conference**: NeurIPS 2025
**arXiv**: [2505.12514](https://arxiv.org/abs/2505.12514)
**Code**: [GitHub](https://github.com/Ber666/reasoning-by-superposition)
**Area**: LLM Reasoning
**Keywords**: continuous chain of thought, superposition-state reasoning, graph reachability, Transformer expressiveness, Coconut

## TL;DR
This paper theoretically demonstrates the expressive advantage of continuous chain-of-thought (Coconut) on directed graph reachability: a two-layer Transformer using $D$ continuous thought steps suffices to solve graph reachability with diameter $D$, whereas discrete CoT requires $O(n^2)$ steps. The core mechanism is that continuous thought vectors encode multiple search frontiers simultaneously in a "superposition state," enabling implicit parallel BFS.

## Background & Motivation
- **Limitations of Prior Work**: Standard chain-of-thought (CoT) assists reasoning by generating discrete text tokens, but performs poorly on tasks such as graph reasoning. Each discrete CoT step must "collapse" the superposition state into a single path, reducing search to greedy search or DFS requiring extensive backtracking.
- **Empirical Advantage of Coconut**: Coconut (Hao et al., 2024) significantly outperforms discrete CoT on synthetic tasks such as directed graph reachability, with preliminary observations suggesting that continuous thoughts may simultaneously store multiple candidate search paths—yet a theoretical explanation remains absent.
- **Theoretical Gap**: Existing theoretical work on CoT-enhanced Transformer expressiveness focuses primarily on discrete CoT; a rigorous understanding of why continuous CoT is more powerful is lacking.
- **Core Idea**: By analogy with quantum superposition, continuous thought vectors are "superposition states" that can simultaneously encode multiple search frontiers, whereas discrete tokens are "collapsed states" that must commit to a single path.

## Method

### Overall Architecture
- **Problem Setting**: Directed graph reachability—given a directed graph $G=(V,E)$, a source node $r$, and two candidate target nodes $c_1/c_2$, determine which one is reachable.
- **Input Structure**: BOS + edge sequence (each edge represented by source node $s$, target node $t$, and special token \<e\>) + query token \<Q\> + two candidate nodes + reasoning token \<R\> + root node.
- **Continuous Thought Generation**: The sampling step is skipped; the Transformer output is directly used as the input embedding for the next step, i.e., $\mathbf{h}_{t+1} = \mathsf{TF}_\theta(\mathbf{h}_{[t]})$.

### Key Designs

#### 1. Attention Chooser
- A basic building block that enables the model to attend to specific relative positions based on the token type at the current position.
- Exploits the rotational matrix properties of sinusoidal positional encodings so that specific token types always attend to positions at fixed relative offsets.
- Compatible with standard positional encodings (sinusoidal/RoPE) without requiring problem-specific or length-specific customization.

#### 2. Superposition State Maintenance (Core Lemma)
- **Key Lemma (Lemma 2)**: The $c$-th continuous thought is the normalized superposition of all nodes reachable from $r$ within $c$ steps:
$$[\mathbf{t}_c] = \frac{1}{\sqrt{|\mathcal{V}_c|}} \sum_{v \in \mathcal{V}_c} \mathbf{u}_v$$
- **Two-Layer Transformer Construction**:
    - **First attention layer** (5 heads): Copies source and target node information for each edge to the corresponding \<e\> token position. Specifically, $h_0/h_1$ attend to positions 2/1 before \<e\>, $h_2/h_3$ attend to positions 2/1 before \<R\>, and $h_4$ attends to the position 1 before \<A\>.
    - **Second attention layer** (1 head): The current thought $[\mathbf{t}_c]$ computes attention with the source nodes of edge tokens \<e\> to identify all edges whose source nodes lie in $\mathcal{V}_c$, then adds the corresponding target nodes—exactly one BFS expansion step.
    - **MLP as a filter**: Filters out noisy tokens (those with weights below threshold $\varepsilon$) and equalizes the weights of the remaining nodes; combined with LayerNorm to restore the normalized superposition state.

#### 3. Final Prediction ("Measuring" the Superposition State)
- After generating $C$ continuous thought steps, an answer token \<A\> is appended.
- \<A\> uses the embeddings of $c_1$ and $c_2$ to "measure" the signal strength of each candidate in the superposition state $[\mathbf{t}_C]$.
- The candidate node with the larger signal is predicted as the output—analogous to the measurement process in quantum mechanics.

### Loss & Training
- Multi-stage curriculum learning is adopted: at stage $i$, the model is trained to predict the $i$-th node in the CoT solution after $i$ continuous thought steps.
- Each stage is trained for 25 epochs, totaling 300 epochs.
- Data from the previous stage is mixed in with probability 0.1 to stabilize training.

## Key Experimental Results

### Main Results

| Method | Layers | Attention Heads | ProsQA Accuracy |
|--------|--------|-----------------|-----------------|
| Coconut | 2 | 8 | ~100% |
| CoT | 2 | 8 | ~75% |
| No CoT | 2 | 8 | ~75% |
| CoT* | 12 | 12 | ~83% |

**Key Conclusion**: A 2-layer Transformer with Coconut achieves near-perfect accuracy, significantly outperforming the 12-layer + CoT configuration.

### Ablation Study: Layer 2 Attention Score Analysis

| Reasoning Step | Not Reachable | Reachable | Frontier | Optimal |
|----------------|--------------|-----------|----------|---------|
| Step 1 | 0.04±0.07 | 2.12±1.07 | 2.12±1.07 | 2.54±1.03 |
| Step 2 | 0.03±0.09 | 0.71±0.92 | 1.00±0.96 | 1.72±1.13 |
| Step 3 | 0.08±0.17 | 0.38±0.72 | 0.67±0.87 | 1.67±1.20 |
| Step 4 | 0.12±0.20 | 0.29±0.66 | 0.61±0.95 | 2.23±1.35 |

The model indeed concentrates attention on reachable edges, with an additional preference for frontier edges and optimal edges.

### Key Findings
1. **Spontaneous Emergence of Superposition**: Trained with supervision only on optimal paths, continuous thoughts automatically learn to encode the superposition of multiple search frontiers without explicit multi-path supervision.
2. **Representation Verification**: The inner product of continuous thought $[\mathbf{t}_i]$ with reachable node embeddings is significantly larger than with unreachable nodes, and frontier nodes > other reachable nodes > unreachable nodes.
3. **Coconut vs. Coconut-BFS**: Replacing optimal-path supervision with uniformly random frontier nodes also achieves near-perfect accuracy, and both supervision strategies converge to similar exploration behaviors.

## Highlights & Insights
1. **Quantum Mechanics Analogy**: Framing continuous thoughts as quantum superposition states and discrete tokens as collapsed states is both intuitive and insightful in revealing the essential advantage of continuous CoT.
2. **Theoretical Separation of $O(D)$ vs. $O(n^2)$**: This work is the first to prove that continuous CoT achieves a significantly lower step complexity than discrete CoT for graph reachability.
3. **Practical Positional Encodings**: The construction is compatible with standard sinusoidal and RoPE positional encodings without problem-specific customization, enhancing the practical relevance of the theory.
4. **Theory–Practice Alignment**: The attention patterns prescribed by the theoretical construction (layer-1 copying, layer-2 expansion) are empirically learned by trained models.
5. **Linear Embedding Dimension Suffices**: The theoretical construction requires only $O(|\text{Voc}|)$-dimensional embeddings, improving upon the exponential dimensionality required by concurrent work.

## Limitations & Future Work
1. A lower bound on the number of discrete CoT steps has not been established; thus, a strict expressiveness separation between CoT and Coconut remains open.
2. Only the specific problem of graph reachability is studied; theoretical analysis for more general reasoning tasks remains to be developed.
3. It lacks a theoretical explanation of how superposition-state exploration behavior emerges through training from deterministic search trajectories alone.
4. Experiments use small GPT-2 models trained from scratch; applicability to large-scale pretrained LLMs has not been validated.

## Related Work & Insights
- **Coconut (Hao et al., 2024)**: The theoretical foundation of this work; introduces the training methodology for continuous chain-of-thought.
- **Merrill & Sabharwal (2023)**: Proves that constant-depth Transformers with discrete CoT require $O(n^2)$ steps to solve graph reachability; the present work substantially improves upon this result.
- **Gozeten et al. (2025)**: A concurrent work studying the superposition mechanism of a one-layer Transformer with continuous CoT on the minimum non-negative sum problem, but requiring exponential embedding dimensions.
- **Insight**: The "parallel computation" capacity of continuous representation spaces may be a key direction for unlocking complex reasoning in LLMs.

## Rating
- Theoretical Depth: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Practicality: ⭐⭐⭐
- Novelty: ⭐⭐⭐⭐⭐
- Overall Recommendation: 9/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](visual_thoughts_a_unified_perspective_of_understanding_multi.md)
- [\[ICLR 2026\] Continuous Chain of Thought Enables Parallel Exploration and Reasoning](../../ICLR2026/llm_reasoning/continuous_chain_of_thought_enables_parallel_exploration_and_reasoning.md)
- [\[NeurIPS 2025\] Note 1: Is CoT a Hallucination? A Data Distribution Perspective](is_chain-of-thought_reasoning_of_llms_a_mirage_a_data_distribution_lens.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](latent_chain-of-thought_for_visual_reasoning.md)
- [\[NeurIPS 2025\] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)

</div>

<!-- RELATED:END -->
