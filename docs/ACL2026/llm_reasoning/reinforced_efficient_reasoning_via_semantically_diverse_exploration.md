---
title: >-
  [Paper Note] Reinforced Efficient Reasoning via Semantically Diverse Exploration
description: >-
  [ACL 2026][LLM Reasoning][MCTS] ROSE proposes a semantic-entropy-guided MCTS branching strategy and a length-aware segment-level advantage estimation to address the insufficient exploration diversity and low reasoning efficiency of existing MCTS-based RLVR methods, achieving state-of-the-art pass@8 performance across multiple mathematical reasoning benchmarks.
tags:
  - ACL 2026
  - LLM Reasoning
  - MCTS
  - Semantic Entropy
  - GRPO
  - Efficient Reasoning
  - Branching Strategy
date: 2026-05-08
content_hash: 2b46e4ba16b5280d
---

# Reinforced Efficient Reasoning via Semantically Diverse Exploration

**Conference**: ACL 2026
**arXiv**: [2601.05053](https://arxiv.org/abs/2601.05053)
**Code**: [https://github.com/ZiqiZhao1/ROSE-rl](https://github.com/ZiqiZhao1/ROSE-rl)
**Area**: Model Compression / Efficient Reasoning
**Keywords**: MCTS, Semantic Entropy, GRPO, Efficient Reasoning, Branching Strategy

## TL;DR

ROSE proposes a semantic-entropy-guided MCTS branching strategy and a length-aware segment-level advantage estimation to address the insufficient exploration diversity and low reasoning efficiency of existing MCTS-based RLVR methods, achieving state-of-the-art pass@8 performance across multiple mathematical reasoning benchmarks.

## Background & Motivation

**State of the Field**: RLVR (Reinforcement Learning with Verifiable Rewards) has become a mainstream approach for enhancing LLM reasoning capabilities. GRPO and its variants optimize policies by sampling multiple independent reasoning chains with binary rewards. MCTS-based methods further introduce tree-structured reasoning, allowing different reasoning chains to share prefixes and enabling finer-grained segment-level credit assignment.

**Limitations of Prior Work**: (1) Insufficient exploration diversity — existing methods use generation entropy to determine branching points, but positions with high generation entropy do not necessarily correspond to semantic divergence. The case in Figure 1 shows that "can" and "need" differ substantially under generation entropy, yet are semantically equivalent, causing the resulting reasoning paths to be identical. (2) Low reasoning efficiency — existing MCTS methods do not address the overthinking problem; correct but verbose reasoning chains receive the same reward as concise ones.

**Root Cause**: Generation entropy measures token-level lexical uncertainty, yet many high-entropy choices in language generation are semantically equivalent (synonyms, functional word variants), causing the branching strategy to produce superficially different but essentially identical reasoning paths.

**Paper Goals**: (1) Design a branching strategy that genuinely produces semantically diverse reasoning paths; (2) Encourage more efficient reasoning while maintaining or improving reasoning performance.

**Starting Point**: Cosine similarity of token embeddings is used to measure the semantic difference among candidate tokens, which is then multiplied by generation entropy to obtain "semantic entropy," ensuring that branching points exhibit both high uncertainty and high semantic divergence.

**Core Idea**: Replace generation entropy with semantic entropy ($=$ generation entropy $\times$ semantic divergence) for branching-point selection; add $\varepsilon$-exploration to prevent overly localized search; apply length-aware calibration to penalize verbose correct reasoning chains, thereby achieving "more diverse and more efficient" reasoning exploration.

## Method

### Overall Architecture

Given a question $q$, a complete reasoning chain is first generated. The semantic entropy at each position is computed, and the position with the highest semantic entropy is selected for branching and regeneration. With probability $\varepsilon$, a new chain is generated from scratch (to prevent localization). After constructing the tree structure, node value assignment, segment-level advantage estimation, and length-aware calibration are performed, followed by training with the Dr.GRPO loss.

### Key Designs

1. **Semantic-Entropy Guided Branching**:

    - **Function**: Selects branching points that produce genuinely semantically diverse reasoning paths.
    - **Mechanism**: For position $k$, the top-20 high-probability token set $\mathcal{V}_k$ is collected. The semantic divergence is computed using LLM embeddings as $SD_k = -\sum_{v_i, v_j} p(v_i) p(v_j) \cdot \cos\langle \mathbf{e}_{v_i}, \mathbf{e}_{v_j} \rangle$, then multiplied by generation entropy to obtain semantic entropy $SE_k = SD_k \cdot \mathcal{H}_k$. High semantic entropy implies both high uncertainty and large semantic differences among candidate tokens.
    - **Design Motivation**: Generation entropy only measures "uncertainty in which token to select," while semantic divergence additionally measures "whether different choices truly lead to different meanings." Their product ensures that branching points exhibit substantive divergence.

2. **$\varepsilon$-Exploration Mechanism**:

    - **Function**: Prevents overly localized search and balances exploration depth and breadth.
    - **Mechanism**: Before generating each new reasoning chain, an independent chain is generated from scratch with probability $\varepsilon$ (default 0.5); otherwise, branching is performed according to semantic entropy. This is analogous to the $\varepsilon$-greedy strategy in RL.
    - **Design Motivation**: A pure branching strategy may confine the search to the vicinity of existing reasoning paths; generating from scratch provides entirely new starting points.

3. **Length-Aware Segment-Level Advantage Estimation**:

    - **Function**: Penalizes verbose correct reasoning on top of segment-level credit assignment, encouraging efficient reasoning.
    - **Mechanism**: Node value $\hat{V}(b_j)$ is defined as the average reward of all reasoning chains passing through that node. Segment-level advantage is the difference between adjacent node values: $\hat{A}_{i,t} = \hat{V}(b_j) - \hat{V}(b_{j-1})$. For reasoning chains that are correct but longer than the shortest correct chain, the advantage is reduced proportionally by length after the divergence node: $\hat{A}_{i,t} \leftarrow \hat{A}_{i,t} - |\hat{A}_{i,t}| \cdot (1 - (|o_s| - b_c)/(|o_c| - b_c))^\alpha)$
    - **Design Motivation**: Within the tree structure, different correct reasoning chains that diverge from the same node can be directly compared by length. This preserves the granularity of segment-level credit assignment while guiding the model to prefer concise reasoning.

### Loss & Training

The Dr.GRPO objective (with variance normalization and length normalization removed) is used. Batch size 512, 8 reasoning chains per question ($G=8$), learning rate $1 \times 10^{-6}$, clip ratio 0.2, KL coefficient 0.001, maximum 8 epochs. Training data consists of 7,500 problems from MATH. $\varepsilon=0.5$; $\alpha$ is searched from $\{0.5, 1, 2, 3\}$. 8×A800 GPUs.

## Key Experimental Results

### Main Results (pass@8)

| Model | Method | AIME24 | AIME25 | MATH500 | AMC23 | Avg. |
|-------|--------|--------|--------|---------|-------|------|
| Qwen3-4B | GRPO | 16.67 | 20.00 | 79.80 | 77.50 | 48.49 |
| Qwen3-4B | FR3E | 16.67 | 13.33 | 80.00 | 75.00 | 47.92 |
| Qwen3-4B | **ROSE** | **23.33** | **23.33** | 80.80 | **77.50** | **51.24** |
| Qwen3-8B | GRPO | 23.33 | 23.33 | 79.40 | 72.50 | 49.64 |
| Qwen3-8B | **ROSE** | **33.33** | **30.00** | 83.00 | **80.00** | **55.75** |
| Llama-3.2-3B | GRPO | 16.67 | 3.33 | 53.40 | 40.00 | 28.35 |
| Llama-3.2-3B | **ROSE** | **20.00** | **6.67** | **55.00** | **45.00** | **31.67** |

### Ablation Study

| Branching Strategy | AIME24 | AIME25 | Avg. |
|--------------------|--------|--------|------|
| Generation Entropy (FR3E) | 16.67 | 6.67 | 30.26 |
| Semantic Divergence | 20.00 | 6.67 | — |
| **Semantic Entropy (ROSE)** | **20.00** | **6.67** | **31.67** |

### Key Findings

- ROSE yields the largest gains on difficult tasks (AIME24/25, +6.67), indicating that semantically diverse exploration is more valuable on harder problems.
- On Qwen3-8B, ROSE achieves an average improvement of +4.65 over GRPO, the highest among all methods.
- TreePO shows notable gains on in-domain datasets (MATH500) but poor out-of-domain generalization, suggesting that fixed-length branching strategies lack adaptability.
- Length-aware calibration reduces reasoning chain length without degrading performance.
- ROSE is also effective on the Llama model (+2.86), ruling out confounds from Qwen data contamination.

## Highlights & Insights

- The design of semantic entropy = generation entropy × semantic divergence is concise and elegant. Measuring semantic difference via cosine similarity of token embeddings incurs minimal computational overhead (only requires embedding table lookups) yet effectively distinguishes "lexical uncertainty" from "semantic uncertainty."
- $\varepsilon$-exploration introduces the classical RL exploration strategy into MCTS branching — simple but critical for preventing the search from being anchored to existing reasoning paths.
- Length-aware calibration cleverly leverages the natural advantage of tree structure: reasoning chains diverging from the same node can be compared on length in a fair manner.

## Limitations & Future Work

- Evaluation is limited to mathematical reasoning; applicability to code generation, logical reasoning, and other domains remains to be verified.
- The pass@8 metric focuses on "solvability" rather than average accuracy; the advantage under the mean@8 perspective may be smaller.
- Semantic divergence uses static token embeddings without accounting for the influence of context on token semantics.
- $\varepsilon=0.5$ is a fixed value; adaptive adjustment may yield further improvements.

## Related Work & Insights

- **vs. FR3E**: FR3E uses generation entropy for branching and wastes branches on semantically equivalent tokens. ROSE uses semantic entropy to ensure that each branch produces genuinely different reasoning paths.
- **vs. Dr.GRPO**: Dr.GRPO improves the loss function but does not improve the exploration process. ROSE improves the exploration process and is compatible with Dr.GRPO.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The semantic entropy concept is novel; the distinction between generation entropy and semantic entropy is convincing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three models, four benchmarks, and complete ablations, though non-mathematical tasks are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Case analyses are intuitive and method descriptions are clear.
- **Value**: ⭐⭐⭐⭐ Provides a better branching strategy for MCTS-based RLVR that is plug-and-play.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Efficient Thought Space Exploration Through Strategic Intervention](../../AAAI2026/llm_reasoning/efficient_thought_space_exploration_through_strategic_intervention.md)
- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[ICLR 2026\] Continuous Chain of Thought Enables Parallel Exploration and Reasoning](../../ICLR2026/llm_reasoning/continuous_chain_of_thought_enables_parallel_exploration_and_reasoning.md)
- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)
- [\[ACL 2026\] Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning](self-consistency_from_only_two_samples_cot-pot_ensembling_for_efficient_llm_reas.md)

</div>

<!-- RELATED:END -->
