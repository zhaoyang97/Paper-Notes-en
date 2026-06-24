---
title: >-
  [Paper Note] Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones
description: >-
  [NeurIPS 2025][Reasoning][Test-time scaling] This paper demonstrates theoretically and empirically that there exist reasoning tasks (graph connectivity) for which a single long CoT (sequential scaling) is equivalent in capability to exponentially many short CoTs (parallel scaling)—i.e., reducing CoT length by even a small amount requires an exponential increase in parallel samples to achieve the same accuracy.
tags:
  - "NeurIPS 2025"
  - "Reasoning"
  - "Test-time scaling"
  - "sequential scaling vs. parallel scaling"
  - "chain-of-thought"
  - "complexity theory"
  - "graph connectivity"
date: 2026-05-08
content_hash: 3f1ec639570c4bdc
---

# Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones

**Conference**: NeurIPS 2025
**arXiv**: [2505.21825](https://arxiv.org/abs/2505.21825)  
**Code**: [github.com/seyedparsa/let-me-think](https://github.com/seyedparsa/let-me-think)  
**Area**: LLM Reasoning
**Keywords**: Test-time scaling, sequential scaling vs. parallel scaling, chain-of-thought, complexity theory, graph connectivity

## TL;DR
This paper demonstrates theoretically and empirically that there exist reasoning tasks (graph connectivity) for which a single long CoT (sequential scaling) is equivalent in capability to exponentially many short CoTs (parallel scaling)—i.e., reducing CoT length by even a small amount requires an exponential increase in parallel samples to achieve the same accuracy.

## Background & Motivation

**Background**: Test-time computation has two primary scaling axes—**sequential scaling** (longer CoT) and **parallel scaling** (multiple short CoTs with voting/Best-of-N). Models such as o1 and R1 rely on long CoTs, but the computational cost of long CoTs grows quadratically with sequence length.

**Limitations of Prior Work**: There is currently no theoretical understanding of the trade-off between the two scaling approaches. Some argue that multiple short CoTs with majority voting are more efficient than a single long CoT; others maintain that long CoTs are irreplaceable for hard problems. No clear theoretical framework exists to answer this question.

**Key Challenge**: Intuitively, sequential reasoning (e.g., DFS over a graph) requires accumulating information step by step, whereas each parallel sample starts from scratch and cannot share intermediate computation. This intuition, however, lacks formal grounding.

**Goal**: Does there exist a reasoning task for which sequential scaling offers an **exponential** advantage over parallel scaling?

**Key Insight**: The paper studies graph connectivity—determining whether a source node in a given graph is connected to $t_1$ or $t_2$. This serves as a proxy for multi-step reasoning and is known to be intimately related to the expressibility limits of Transformers ($\mathsf{TC}^0$ vs. $\mathsf{L}$).

**Core Idea**: On graph connectivity tasks, bounded-depth Transformers with $O(1)$-length CoTs succeed no better than random guessing even with polynomially many parallel samples, yet a single polynomial-length CoT suffices for perfect solutions—establishing an exponential sequential-vs.-parallel gap.

## Method

### Overall Architecture
The paper proceeds in three layers: (1) a theoretical separation based on Transformer expressibility (Theorem 1); (2) a finer separation using the Vertex Query Model (VQM) abstraction (Theorems 2 & 3); and (3) empirical validation on small Transformers trained from scratch and on frontier large language models.

### Key Designs

1. **Theoretical Separation via $\mathsf{TC}^0$ (Theorem 1)**:

    - **Function**: Proves that sequential scaling can solve graph connectivity while parallel scaling cannot.
    - **Mechanism**: (a) A polynomial-length CoT can simulate BFS, so one CoT suffices (a corollary of Merrill & Sabharwal 2024b); (b) a Transformer with $O(1)$-length CoT is equivalent to a $\mathsf{TC}^0$ circuit, and $\mathsf{TC}^0 \not\supseteq \mathsf{L}$ (standard complexity assumption), so connectivity cannot be solved. Key insight: taking a majority vote over multiple independent short CoTs **remains a $\mathsf{TC}^0$ circuit** (since MAJORITY gates are part of $\mathsf{TC}^0$), so parallel scaling cannot break this barrier.
    - **Design Motivation**: Establishes the strongest form of the impossibility result—polynomially many parallel samples still fail.

2. **Fine-Grained Analysis via the Vertex Query Model (Theorems 2 & 3)**:

    - **Function**: Quantifies the sequential-vs.-parallel gap in a more refined computational model.
    - **Mechanism**: In the VQM, algorithms access the graph only through "neighbor queries," querying the neighbors of one vertex per step. For **two-path graphs** (Theorem 2): $O(L)$ queries suffice, but with fewer than $L/2$ queries the success probability is exactly 1/2. For **bridge graphs** (Theorem 3): even with more queries than the shortest-path length ($(1-\delta)\cdot\frac{3}{2}ld$ queries), the success probability remains only $1/2+\exp(-\Omega(d))$—requiring $\exp(\Omega(d))$ independent runs (parallel scaling) to reach a 2/3 success rate.
    - **Design Motivation**: The bridge graph forces the model to guess the direction of the shorter path at each junction with probability 1/2; the cumulative probability decays exponentially over $d$ junctions.

3. **Emergence of Long CoTs under RL (Section 5)**:

    - **Function**: Trains models with STaR (Self-Taught Reasoner) and observes spontaneous CoT length growth.
    - **Mechanism**: Models are trained on bridge graphs with short CoTs (Path strategy), then iteratively fine-tuned via RL (verifying correct CoTs → self-training). The correct CoTs gradually become longer, and backtracking behavior absent from the training data emerges.
    - **Design Motivation**: Connects to the CoT length growth observed during RL training of DeepSeek-R1, suggesting that graph connectivity is an effective proxy task for studying long-CoT emergence.

### Loss & Training
Small Transformers (4-layer Mistral, 128-dimensional hidden layer) are trained on 500k CoT samples for 200 epochs. RL follows STaR: 500k samples are drawn per round, correct ones are retained, and the model is fine-tuned for 20 epochs.

## Key Experimental Results

### Main Results

**Bridge(5) Task: Evidence Accuracy under Different CoT Strategies**

| CoT Strategy | CoT Length | Evidence Accuracy | Notes |
|---|---|---|---|
| Shortest-Path | ~35 tokens | 0.0% | Too short to explore |
| Path (DFS-tree path) | ~50 tokens | 11.16% | Slightly better but insufficient |
| **DFS (full search)** | **~90 tokens** | **~100%** | Perfect solution |

**DeepSeek-R1-Distill-Qwen-32B: Sequential vs. Parallel Scaling on Bridge Graphs**

| CoT Length (tokens) | Parallel = 1 | Parallel = 64 | Notes |
|---|---|---|---|
| 512 | ~50% | ~50% | Too short; parallel yields no gain |
| 1024 | ~55% | ~60% | Begins to take effect |
| 2048 | ~75% | ~90% | Sequential drives parallel |
| 4096 | ~95% | ~99% | Sequential alone sufficient |

### Ablation Study

| Configuration | Bridge(3) Evidence Accuracy | Notes |
|---|---|---|
| Path model (before RL) | 21.16% | Limited by short CoT |
| Path model (after 4 RL rounds) | **92.02%** | CoT grows spontaneously; backtracking emerges |
| DFS model | ~100% | Training data contains long CoTs |

### Key Findings
- **An exponential gap exists**: On bridge graphs, reducing CoT length from the full DFS length to the shortest-path length requires an exponential increase in parallel samples (from 1 to $\sim 3^d \cdot 4^{d-1}$) to achieve the same accuracy.
- **Parallel scaling is useful only after sequential scaling reaches non-trivial accuracy**: In large-model experiments, increasing parallel samples from 1 to 64 yields almost no gain when CoT length is insufficient; parallel scaling only becomes effective once CoT length is adequate.
- **RL induces spontaneous CoT length growth**: After STaR training, the average length of correct CoTs grows from ~15 to ~30 tokens, and backtracking behavior absent from the training data emerges.
- **Under a fixed total token budget, sequential scaling consistently outperforms parallel scaling.**
- **The trend holds on AIME 2024**: s1-32B on AIME2024 likewise exhibits the irreplaceable nature of sequential scaling.

## Highlights & Insights
- **First proof of an exponential separation between sequential and parallel scaling**: This is an important theoretical milestone. The core insight—that **majority voting over parallel samples remains a $\mathsf{TC}^0$ circuit**—is the genuine key contribution, implying that the voting mechanism itself cannot overcome depth limitations.
- **Elegant bridge graph construction**: By placing a short path, a long path, and a dead end at each junction, the construction forces genuine sequential search rather than parallel guessing, and itself constitutes a useful benchmark.
- **Inspiring explanation of long-CoT emergence under RL**: The graph connectivity task provides a concise theoretical framework for explaining the CoT length growth observed during RL training of DeepSeek-R1.

## Limitations & Future Work
- The theoretical results rely on the complexity assumption $\mathsf{TC}^0 \not\supseteq \mathsf{L}$, which is widely believed but unproven.
- The VQM as an abstraction of Transformer CoT has only heuristic support and lacks a direct theoretical proof.
- Graph connectivity is a specific structured task; the extent to which the findings transfer to natural language reasoning requires further study.
- Experiments are validated only on graph tasks and AIME; broader benchmark coverage would strengthen the claims.

## Related Work & Insights
- **vs. s1 (Simple Test-Time Scaling)**: s1 shows that extending CoT length via "Wait" tokens improves performance; this paper provides theoretical justification for that practice.
- **vs. Snell et al. 2025 (Scaling LLM Test-Time Compute)**: Their empirical scaling laws suggest that parallel scaling is preferable to sequential scaling in some settings; this paper identifies an opposing extreme.
- **vs. Li et al. 2024 (CoT empowers Transformers)**: That work theoretically proves that CoT enables Transformers to solve inherently sequential problems; this paper builds on it to quantify the sequential-vs.-parallel gap.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First exponential separation proof between sequential and parallel scaling; elegant bridge graph construction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Small models trained from scratch + frontier LLM validation + RL emergence experiments + AIME transfer.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretically clear, intuitions well-explained, proof structure elegant.
- Value: ⭐⭐⭐⭐⭐ Provides a theoretical foundation for the core question of "long thinking vs. many short thoughts," with far-reaching implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] PENCIL: Long Thoughts with Short Memory](../../ICML2025/llm_reasoning/pencil_long_thoughts_with_short_memory.md)
- [\[NeurIPS 2025\] ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs](reasonfluxprm_trajectoryaware_prms_for_long_chainofthought_r.md)
- [\[NeurIPS 2025\] Large Language Models Can Learn and Generalize Steganographic Chain-of-Thought under Process Supervision](large_language_models_can_learn_and_generalize_steganographic_chain-of-thought_u.md)
- [\[ACL 2025\] Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](../../ACL2025/llm_reasoning/can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)
- [\[NeurIPS 2025\] Let LRMs Break Free from Overthinking via Self-Braking Tuning](let_lrms_break_free_from_overthinking_via_self-braking_tuning.md)

</div>

<!-- RELATED:END -->
