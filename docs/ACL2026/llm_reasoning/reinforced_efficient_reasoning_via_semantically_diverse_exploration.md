---
title: >-
  [Paper Note] Reinforced Efficient Reasoning via Semantically Diverse Exploration
description: >-
  [ACL 2026][LLM Reasoning][MCTS] ROSE proposes an MCTS branching strategy guided by semantic entropy and length-aware segment-level advantage estimation. Such design addresses the insufficient exploration diversity and lo…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "MCTS"
  - "Semantic Entropy"
  - "GRPO"
  - "Efficient Inference"
  - "Branching Strategy"
date: 2026-05-08
content_hash: da5d0b4d668686fc
---

# Reinforced Efficient Reasoning via Semantically Diverse Exploration

**Conference**: ACL 2026  
**arXiv**: [2601.05053](https://arxiv.org/abs/2601.05053)  
**Code**: [https://github.com/ZiqiZhao1/ROSE-rl](https://github.com/ZiqiZhao1/ROSE-rl)  
**Area**: Model Compression / Efficient Inference  
**Keywords**: MCTS, Semantic Entropy, GRPO, Efficient Inference, Branching Strategy

## TL;DR

ROSE proposes an MCTS branching strategy guided by semantic entropy and length-aware segment-level advantage estimation. Such design addresses the insufficient exploration diversity and low inference efficiency in existing MCTS-based RLVR methods, achieving state-of-the-art pass@8 performance on multiple mathematical reasoning benchmarks.

## Background & Motivation

**Background**: RLVR (Reinforcement Learning with Verifiable Rewards) has become the mainstream approach to enhance the reasoning capabilities of LLMs. GRPO and its variants optimize policies by sampling multiple independent reasoning chains and using binary rewards. MCTS-based methods further introduce tree-structured reasoning, allowing different chains to share prefixes for more granular segment-level credit assignment.

**Limitations of Prior Work**: (1) Insufficient exploration diversity—existing methods use generation entropy to determine branching points, yet positions with high generation entropy do not necessarily correspond to semantic divergence. A case in Figure 1 shows that "can" and "need" differ significantly in generation entropy but are semantically equivalent, resulting in identical reasoning paths after branching; (2) Low inference efficiency—existing MCTS methods fail to handle the "overthinking" problem, where correct but verbose reasoning chains receive the same reward as concise ones.

**Key Challenge**: Generation entropy measures token-level lexical uncertainty, but many high-entropy choices in language generation are semantically equivalent (e.g., synonyms or functional word variants). This causes branching strategies to produce reasoning paths that are superficially different but essentially identical.

**Goal**: (1) Design a branching strategy capable of generating truly semantically diverse reasoning paths; (2) Encourage more efficient reasoning while maintaining or even improving performance.

**Key Insight**: Use the cosine similarity of token embeddings to measure the semantic difference between candidate tokens. Multiplying this with generation entropy yields "Semantic Entropy," ensuring that branching points possess both high uncertainty and high semantic divergence.

**Core Idea**: Replace generation entropy with semantic entropy (= generation entropy × semantic divergence) for branching, incorporate $\varepsilon$-exploration to prevent localized searching, and apply length-aware calibration to penalize redundant correct reasoning chains, thereby achieving "more diverse + more efficient" reasoning exploration.

## Method

### Overall Architecture

Given a problem $q$, a complete reasoning chain is first generated. The semantic entropy at each position is calculated, and the position with the highest semantic entropy is selected for branching and regeneration. With probability $\varepsilon$, a new chain is generated from scratch to prevent localization. After obtaining the tree structure, node value assignment, segment-level advantage estimation, and length-aware calibration are performed. Finally, the model is trained using the Dr.GRPO loss function.

### Key Designs

1.  **Semantic-Entropy Guided Branching**:
    - **Function**: Select branching points that produce truly semantically diverse reasoning paths.
    - **Mechanism**: For position $k$, the top-20 high-probability token set $\mathcal{V}_k$ is extracted. The semantic divergence $SD_k = -\sum_{v_i, v_j} p(v_i) p(v_j) \cdot \cos\langle \mathbf{e}_{v_i}, \mathbf{e}_{v_j} \rangle$ is calculated using LLM embeddings and then multiplied by generation entropy to obtain semantic entropy $SE_k = SD_k \cdot \mathcal{H}_k$. High semantic entropy indicates high uncertainty combined with large semantic differences among candidate tokens.
    - **Design Motivation**: Generation entropy only measures lexical uncertainty, while semantic divergence measures whether different choices actually lead to different meanings. Their product ensures that branching points represent substantial divergence.

2.  **$\varepsilon$-Exploration Mechanism**:
    - **Function**: Prevent the search from becoming too localized and balance exploration depth and breadth.
    - **Mechanism**: Before generating each new reasoning chain, an independent generation from scratch is performed with probability $\varepsilon$ (default 0.5); otherwise, branching occurs based on semantic entropy. This is analogous to the $\varepsilon$-greedy strategy in RL.
    - **Design Motivation**: Pure branching strategies may restrict the search to the vicinity of existing paths. Generation from scratch provides entirely new starting points.

3.  **Length-aware Segment-level Advantage Estimation**:
    - **Function**: Penalize verbose correct reasoning to encourage efficiency based on segment-level credit assignment.
    - **Mechanism**: The node value $\hat{V}(b_j)$ is defined as the average reward of all reasoning chains passing through that node. The segment-level advantage is the difference between adjacent node values $\hat{A}_{i,t} = \hat{V}(b_j) - \hat{V}(b_{j-1})$. For correct reasoning that is longer than the shortest correct chain, the advantage is reduced proportionally after the divergence node:
      $$\hat{A}_{i,t} \leftarrow \hat{A}_{i,t} - |\hat{A}_{i,t}| \cdot (1 - \frac{|o_s| - b_c}{|o_c| - b_c})^\alpha$$
    - **Design Motivation**: In tree structures, different correct paths branching from the same node can be compared directly by length. This preserves the precision of segment-level credit assignment while guiding the model toward concise reasoning.

### Loss & Training

The Dr.GRPO objective function (excluding variance and length normalization) is used. The batch size is 512, with 8 reasoning chains per problem ($G=8$). The learning rate is $1 \times 10^{-6}$, clip ratio is 0.2, KL coefficient is 0.001, and training lasts up to 8 epochs. Training data consists of 7,500 problems from MATH. Parameters are set to $\varepsilon=0.5$, and $\alpha$ is searched from $\{0.5, 1, 2, 3\}$. Training is conducted on $8 \times \text{A800}$ GPUs.

## Key Experimental Results

### Main Results (pass@8)

| Model | Method | AIME24 | AIME25 | MATH500 | AMC23 | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3-4B | GRPO | 16.67 | 20.00 | 79.80 | 77.50 | 48.49 |
| Qwen3-4B | FR3E | 16.67 | 13.33 | 80.00 | 75.00 | 47.92 |
| Qwen3-4B | **ROSE** | **23.33** | **23.33** | 80.80 | **77.50** | **51.24** |
| Qwen3-8B | GRPO | 23.33 | 23.33 | 79.40 | 72.50 | 49.64 |
| Qwen3-8B | **ROSE** | **33.33** | **30.00** | 83.00 | **80.00** | **55.75** |
| Llama-3.2-3B | GRPO | 16.67 | 3.33 | 53.40 | 40.00 | 28.35 |
| Llama-3.2-3B | **ROSE** | **20.00** | **6.67** | **55.00** | **45.00** | **31.67** |

### Ablation Study

| Branching Strategy | AIME24 | AIME25 | Average |
| :--- | :--- | :--- | :--- |
| Generation Entropy Branching (FR3E) | 16.67 | 6.67 | 30.26 |
| Semantic Divergence Branching | 20.00 | 6.67 | - |
| **Semantic Entropy Branching (ROSE)** | **20.00** | **6.67** | **31.67** |

### Key Findings

-   ROSE achieves the largest gains on difficult tasks (AIME24/25) (+6.67), suggesting that semantically diverse exploration is more valuable for high-difficulty problems.
-   On Qwen3-8B, ROSE shows an average gain of +4.65 (vs. GRPO), which is the highest among all compared methods.
-   TreePO improves significantly on in-domain datasets (MATH500) but generalizes poorly to out-of-domain tasks, indicating that fixed-length branching strategies lack adaptability.
-   Length-aware calibration reduces the length of reasoning chains without compromising performance.
-   The method remains effective on Llama models (+2.86), ruling out potential interference from Qwen data leakage.

## Highlights & Insights

-   The design of Semantic Entropy = Generation Entropy × Semantic Divergence is simple yet elegant. By measuring semantic differences through the cosine similarity of token embeddings, it incurs minimal computational overhead (merely embedding table lookups) while effectively distinguishing "lexical uncertainty" from "semantic uncertainty."
-   $\varepsilon$-exploration introduces a classic RL exploration strategy into MCTS branching. Its simplicity is critical for preventing the search from being anchored to existing reasoning paths.
-   Length-aware calibration cleverly utilizes the natural structure of trees: different reasoning chains emerging from the same divergence point can be compared fairly by length.

## Limitations & Future Work

-   The evaluation is limited to mathematical reasoning; verification for code generation and logical reasoning is still required.
-   The pass@8 metric focuses on "solvability" rather than "average accuracy"; advantages from a mean@8 perspective might be more modest.
-   Semantic divergence utilizes static token embeddings and does not account for the impact of context on token semantics.
-   $\varepsilon=0.5$ is currently a fixed value; adaptive adjustment might provide further performance improvements.

## Related Work & Insights

-   **vs FR3E**: FR3E branches based on generation entropy, which wastes branches on semantically equivalent tokens. ROSE uses semantic entropy to ensure each branch leads to truly distinct reasoning paths.
-   **vs Dr.GRPO**: Dr.GRPO improves the loss function but does not enhance the exploration process. ROSE improves exploration and is compatible with Dr.GRPO.

## Rating

-   **Novelty**: ⭐⭐⭐⭐ The concept of semantic entropy is novel, and the distinction between generation entropy and semantic entropy is convincing.
-   **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three models and four benchmarks with complete ablations, though it lacks non-mathematical tasks.
-   **Writing Quality**: ⭐⭐⭐⭐ Case studies are intuitive and the methodology is clearly described.
-   **Value**: ⭐⭐⭐⭐ Provides an improved, plug-and-play branching strategy for MCTS-based RLVR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Efficient Thought Space Exploration Through Strategic Intervention](../../AAAI2026/llm_reasoning/efficient_thought_space_exploration_through_strategic_intervention.md)
- [\[ICLR 2026\] Continuous Chain of Thought Enables Parallel Exploration and Reasoning](../../ICLR2026/llm_reasoning/continuous_chain_of_thought_enables_parallel_exploration_and_reasoning.md)
- [\[ACL 2026\] ETR: Entropy Trend Reward for Efficient Chain-of-Thought Reasoning](etr_entropy_trend_reward_for_efficient_chain-of-thought_reasoning.md)
- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)
- [\[ACL 2026\] Stabilizing Efficient Reasoning with Step-Level Advantage Selection](stabilizing_efficient_reasoning_with_step-level_advantage_selection.md)

</div>

<!-- RELATED:END -->
