---
title: >-
  [Paper Note] CortexDebate: Debating Sparsely and Equally for Multi-Agent Debate
description: >-
  [ACL 2025][Multi-Agent][Multi-Agent Debate] This paper proposes CortexDebate, a multi-agent debate method inspired by the mechanism of the human cerebral cortex. By constructing a sparse dynamic debate graph and an evaluation module based on the McKinsey Trust Formula (MDM), it simultaneously addresses two core challenges of existing Multi-Agent Debate (MAD) methods: "excessively long input context" and "unequal debate caused by overconfidence."
tags:
  - "ACL 2025"
  - "Multi-Agent"
  - "Multi-Agent Debate"
  - "Sparse Graph"
  - "McKinsey Trust Formula"
  - "Overconfidence"
  - "LLM Reasoning"
date: 2026-05-08
content_hash: 23ff6669c52767bd
---

# CortexDebate: Debating Sparsely and Equally for Multi-Agent Debate

**Conference**: ACL 2025  
**arXiv**: [2507.03928](https://arxiv.org/abs/2507.03928)  
**Code**: None  
**Area**: Others (Multi-Agent Systems)  
**Keywords**: Multi-Agent Debate, Sparse Graph, McKinsey Trust Formula, Overconfidence, LLM Reasoning

## TL;DR

This paper proposes CortexDebate, a multi-agent debate method inspired by the mechanism of the human cerebral cortex. By constructing a sparse dynamic debate graph and an evaluation module based on the McKinsey Trust Formula (MDM), it simultaneously addresses two core challenges of existing Multi-Agent Debate (MAD) methods: "excessively long input context" and "unequal debate caused by overconfidence."

## Background & Motivation

Multi-Agent Debate (MAD) improves the hallucination and reasoning issues of a single LLM by allowing multiple LLM agents to debate with each other. However, existing MAD methods face two major bottlenecks:

**Issue 1: Excessively long context**. Each LLM agent needs to debate with all other agents. As the number of agents and rounds increase, the input context expands rapidly. LLMs tend to get "lost in the middle" when dealing with extremely long contexts, resulting in performance degradation.

**Issue 2: Overconfidence dilemma**. Prior methods determine debate influence solely based on an LLM's self-confidence. Confident agents gradually dominate the entire debate process, leading to useful information from other "weaker" agents being ignored, making the debate an unequal monologue.

The authors observe that when the human brain processes problems, sparse and dynamically optimized networks are formed between different cortical regions, regulated by white matter. This inspires the core idea of CortexDebate—constructing a sparse debate graph where debates only occur between agents that are genuinely helpful to each other.

## Method

### Overall Architecture

CortexDebate consists of three stages:
1. **Initial Response Generation**: Each LLM agent independently generates an initial response, explanation, and confidence score.
2. **Multi-round Debate**: Multiple rounds of debate are conducted under the guidance of a sparse debate graph, where the graph structure is dynamically optimized by the MDM module.
3. **Final Answer Generation**: The final answer is generated through majority voting.

### Key Designs

1. **Directed Sparse Debate Graph**: Models $n$ LLM agents as nodes of a directed graph $\mathcal{G}=(\mathcal{A}, \mathcal{E})$, where the weight $W_{i \to j}$ of edge $E_{i \to j}$ represents the expected performance improvement obtained by agent $A_j$ through debating with $A_i$. Edges with weights below the average are pruned, forming a sparse graph. Each agent only debates with agents that are helpful to it, significantly reducing the input context (by up to 70.79%).

2. **McKinsey-based Debate Matter (MDM)**: Acts as an "artificial analogue" of the human brain's white matter. The core innovation is the introduction of the McKinsey Trust Formula $T = \frac{C \times R \times I}{S}$ to compute edge weights, where the four dimensions are adapted to the MAD context:

    - **Credibility (C)**: Evaluates the professional capability of agent $A_i$ utilizing the scaling laws of LLMs, $C_d = 1/\mathcal{L}(N, M)$, where $\mathcal{L}$ is the estimated pre-training loss based on parameter count $N$ and the number of pre-training tokens $M$.
    - **Reliability (R)**: The average confidence of $A_i$ across historical debate rounds, reflecting the stability of task performance.
    - **Intimacy (I)**: The average degree of opinion divergence between $A_i$ and $A_j$ in historical rounds ($I_d = 1 - \overline{Sim}_d$), where collisions of different perspectives facilitate the debate.
    - **Self-orientation (S)**: The fewer debates $A_i$ participates in, the higher its self-orientation (i.e., more "selfish"), $S_d = (d-1)(n-1) - P_d$.

3. **Confidence Recalibration**: Maps the initial confidence to the range of [0.3, 0.8] to alleviate overconfidence—values $\geq 0.8$ are mapped to 0.8, values $< 0.3$ are mapped to 0.3, while intermediate values remain unchanged.

4. **Consensus Detection and Early Stopping**: Checks after each debate round whether all agents have reached a consensus or the maximum number of rounds has been reached, terminating immediately if the condition is met.

### Loss & Training

This method does not involve model training. The optimization of edge weights is implemented entirely through formulaic computation in the MDM module:

$$W_{i \to j}^d = \frac{C_d \times R_d \times I_d}{S_d}$$

Edges with weights below the average $\overline{W}_j^d$ are removed, and the weights of retained edges are set to 1.

## Key Experimental Results

### Main Results (Accuracy on 8 Datasets / %)

| Method | Type | GSM-IC | MATH | MMLU | MMLU-pro | GPQA | ARC-C | LongBench | SQuAD |
|------|------|--------|------|------|----------|------|-------|-----------|-------|
| MaV | No Debate | 70.33 | 46.00 | 69.33 | 46.00 | 27.33 | 76.00 | 45.11 | 85.33 |
| MLD | Full Debate | 72.67 | 47.33 | 71.33 | 47.33 | 28.33 | 79.33 | 48.87 | 86.33 |
| RECONCILE | Full Debate | 75.67 | 50.33 | 75.00 | 53.67 | 31.00 | 83.67 | 52.55 | 88.33 |
| PRD | Full Debate | 77.00 | 51.33 | 77.33 | 54.00 | 32.00 | 84.33 | 50.21 | 87.67 |
| GD | Part Debate | 76.00 | 49.67 | 74.00 | 51.67 | 32.67 | 82.00 | 55.97 | 90.33 |
| **CortexDebate** | **Ours** | **79.33** | **56.00** | **82.33** | **59.33** | **36.33** | **88.33** | **60.31** | **93.33** |

### Ablation Study (Average Score / %)

| Configuration | Average Score |
|------|--------|
| Fully connected graph | 60.49 |
| Fully connected graph + MDM | 63.76 |
| Sparse graph | 62.72 |
| Sparse graph + Self-evaluation (RECONCILE) | 62.13 |
| Sparse graph + Peer evaluation (PRD) | 66.71 |
| Sparse graph + MDM (w/o I and S) | 66.69 |
| **Sparse graph + MDM (Full)** | **69.41** |

### Key Findings

1. **Comprehensive SOTA**: CortexDebate achieves the highest accuracy across all 8 datasets. On mathematics tasks, GSM-IC improves by 9% and MATH by 10%; on reasoning tasks, GPQA improves by 9% and ARC-C by 12.33%.

2. **Significant reduction in input context length**: Compared to fully connected debate methods, CortexDebate reduces the input context of a single agent by up to 70.79% while maintaining higher accuracy.

3. **Critical role of I and S factors in debate optimization**: After introducing Intimacy and Self-orientation factors, the number of diverse viewpoint collisions (DVC) increases from 3.71 to 8.44, and the correct viewpoint modification rate (CVR/DVC) rises from 33.96% to 64.92%—indicating that considering collaborative dynamics among agents rather than just individual abilities is crucial.

4. **Potential for large-scale debate**: As the number of agents and debate rounds increase, the performance of CortexDebate continues to improve, with the increase in agent count contributing more than the increase in rounds.

5. **Retaining beneficial debates and pruning harmful ones**: The number of debates per agent in the sparse graph decreases while accuracy increases (e.g., Qwen from 54.0 $\to$ 58.0, Gemma from 45.0 $\to$ 51.0), demonstrating that the method effectively identifies beneficial debate partners.

## Highlights & Insights

- **Interdisciplinary Innovation**: Introduces the sociological McKinsey Trust Formula into multi-agent debate scenarios. The four-dimensional evaluation (competency/credibility, reliability, intimacy/divergence, and self-orientation/engagement) is more reliable and balanced than evaluations based solely on self-confidence.
- **Precision of the Human Cortex Analogy**: The biological mechanism where white matter regulates the network of cortical regions aligns elegantly with the computational mechanism of MDM regulating the debate graph.
- **Simultaneously Addressing Dual Issues**: The sparse graph mechanism naturally resolves the issue of excessively long contexts, while the I and S factors in MDM tackle the overconfidence problem—making the two designs highly complementary.

## Limitations & Future Work

- Multi-agent methods are inherently less efficient and more costly than single-agent methods.
- The base reasoning capabilities of individual LLM agents remain the key limiting factor of performance; CortexDebate improves debate strategies rather than reasoning capacity itself.
- Credibility computation relies on estimating pre-training loss via scaling laws, which might not be sufficiently accurate across different model families.
- Adaptive sparsity thresholds can be explored (currently a fixed average is used as the threshold).

## Related Work & Insights

- Unlike GroupDebate (fixed grouped debate) and Neighbor Debate (fixed neighbor debate), the debate topology of CortexDebate changes dynamically.
- The introduction of the McKinsey Trust Formula inspires the research paradigm of "transferring mature social science theories into the design of AI systems."
- Future research can explore applying CortexDebate to more complex scenarios, such as domain-specific expert systems.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4.5 |
| Experimental Thoroughness | 4.5 |
| Writing Quality | 4 |
| Value | 4 |

The combination of the McKinsey Trust Formula and sparse debate graphs is highly novel. The extensive experiments across 8 datasets and 4 task categories, coupled with in-depth ablation analyses, are highly convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Voting or Consensus? Decision-Making in Multi-Agent Debate](voting_or_consensus_decision-making_in_multi-agent_debate.md)
- [\[ACL 2026\] Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection](../../ACL2026/multi_agent/debating_the_unspoken_role-anchored_multi-agent_reasoning_for_half-truth_detecti.md)
- [\[ICLR 2026\] Multi-Agent Debate with Memory Masking (MAD-M²)](../../ICLR2026/multi_agent/multi-agent_debate_with_memory_masking.md)
- [\[NeurIPS 2025\] Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](../../NeurIPS2025/multi_agent/debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model.md)
- [\[ICLR 2026\] MAD-Logic: Multi-Agent Debate Enhances Symbolic Translation and Reasoning](../../ICLR2026/multi_agent/mad-logic_multi-agent_debate_enhances_symbolic_translation_and_reasoning.md)

</div>

<!-- RELATED:END -->
