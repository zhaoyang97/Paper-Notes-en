---
title: >-
  [Paper Note] Continuous Chain of Thought Enables Parallel Exploration and Reasoning
description: >-
  [ICLR 2026][LLM Reasoning][Continuous chain of thought] CoT2 proposes replacing discrete tokens with continuous-valued tokens (convex combinations of vocabulary embeddings) for chain-of-thought reasoning, enabling the model to track multiple reasoning paths in parallel within a single forward pass. The approach is theoretically shown to be equivalent to $K$ rounds of self-consistency/best-of-N sampling, and is further improved via GRPO-based reinforcement learning.
tags:
  - ICLR 2026
  - LLM Reasoning
  - Continuous chain of thought
  - parallel reasoning
  - multi-trajectory tracking
  - GRPO
  - information theory
date: 2026-05-08
content_hash: d93dc21f62021c27
---

# Continuous Chain of Thought Enables Parallel Exploration and Reasoning

**Conference**: ICLR 2026
**arXiv**: [2505.23648](https://arxiv.org/abs/2505.23648)
**Code**: [https://github.com/alperengozeten/CoT2](https://github.com/alperengozeten/CoT2)
**Area**: LLM Reasoning / Model Compression
**Keywords**: Continuous chain of thought, parallel reasoning, multi-trajectory tracking, GRPO, information theory

## TL;DR
CoT2 proposes replacing discrete tokens with continuous-valued tokens (convex combinations of vocabulary embeddings) for chain-of-thought reasoning, enabling the model to track multiple reasoning paths in parallel within a single forward pass. The approach is theoretically shown to be equivalent to $K$ rounds of self-consistency/best-of-N sampling, and is further improved via GRPO-based reinforcement learning.

## Background & Motivation

**State of the Field**: CoT reasoning in modern LLMs operates through autoregressive sampling of discrete tokens, with self-consistency (majority voting over multiple samples) or best-of-N decoding employed to improve accuracy.

**Limitations of Prior Work**:
- Discrete sampling transmits at most $\log_2(v)$ bits of information per step, whereas each token embedding can store $O(d)$ bits — a severe underutilization of representational capacity.
- Once a token is sampled, the model commits to a single reasoning path and cannot explore alternatives.
- Self-consistency/best-of-N requires multiple forward passes, leading to linearly growing inference costs.

**Root Cause**: The irreversibility of discrete sampling causes a single reasoning chain to accumulate errors in a snowball effect, while the remedies (multiple sampling runs) impose substantial computational overhead.

**Paper Goals**:
- How can a model track multiple reasoning paths simultaneously within a single inference pass?
- How powerful is the parallel tracking capacity of continuous tokens, and what is its theoretical relationship to discrete multi-sample decoding?
- How should continuous-token models be trained and deployed?

**Starting Point**: Rather than discretely sampling from the softmax output at each step, the output distribution is passed directly as a continuous token — a weighted combination of all vocabulary embeddings — into the next step. This "superposition state" naturally encodes information from multiple paths.

**Core Idea**: Continuous tokens, as convex combinations of vocabulary embeddings, inherently implement parallel path tracking. Their effect is theoretically equivalent to aggregating $K$ independent discrete CoT chains — one forward pass subsumes $K$ sampling runs.

## Method

### Overall Architecture
Given input $\bm{X}$, the model autoregressively generates $m$ tokens: for the first $m-1$ steps, it outputs continuous tokens $\bm{z}_t = \bm{E}^\top \bm{\alpha}_t$ (the product of the softmax distribution and the embedding matrix); at the final step, a discrete answer token is sampled. Training proceeds in two stages: Continuous Supervised Fine-Tuning (CSFT) followed by GRPO-based reinforcement learning.

### Key Designs

1. **Continuous Supervised Fine-Tuning (CSFT)**:

    - Function: Uses "multi-trajectory superposition" as the supervision signal for intermediate steps.
    - Mechanism: Given a budget of $B$ optimal trajectories, the supervision distribution at intermediate step $t$ is $\alpha_{t,g}^* = \frac{1}{B}\sum_{\pi \in \Pi_B} \mathbf{1}\{g_t(\pi)=g\}$ — the empirical distribution over states visited at step $t$ across the $B$ trajectories. The final step uses a one-hot distribution (correct answer). The model is trained via cross-entropy/KL divergence to fit these soft labels.
    - Design Motivation: Setting $B=1$ reduces to discrete CoT (one-hot); $B=|\mathcal{T}|$ tracks all trajectories (maximum parallelism). The budget provides flexible control over the trade-off between parallelism and model capacity.

2. **Budget–Embedding Dimension Trade-off**:

    - Function: Quantifies the theoretical relationship between parallelism and embedding dimensionality.
    - Mechanism: An information-theoretic lower bound of $d = \Omega(B\log(v/B))$ establishes that reliably decoding the superposition of $B$ trajectories requires embedding dimension $\Omega(B\log(v/B))$. When $d$ is sufficiently large, increasing $B$ monotonically improves performance; otherwise, there exists an optimal sweet spot for $B$.
    - Design Motivation: This explains why $B=8$ outperforms $B=16$ at $d=16$ (insufficient capacity), whereas $B=16$ is optimal at $d=32$.

3. **Single-Layer Transformer Construction (Proposition 1)**:

    - Function: Proves that a single-layer Transformer using CoT2 can solve the Minimum Non-Negative Sum (MNNS) problem.
    - Mechanism: Trigonometric embeddings encode all $2^k$ states in non-overlapping (sin, cos) representations; the attention layer expands states (by adding or subtracting new numbers); the MLP layer reads and filters. Each step tracks an exponentially growing number of states in parallel, and the final step selects the minimum non-negative sum.
    - Design Motivation: MNNS is essentially a subset-sum problem requiring search over $2^m$ possibilities. Discrete CoT must commit to one path, whereas CoT2 tracks all paths simultaneously.

4. **Multi-Token Sampling (MTS) + GRPO**:

    - Function: Introduces controllable stochasticity into CoT2 to enable the use of RL methods.
    - Mechanism: At each step, $K$ discrete tokens are sampled and averaged: $\bm{z}_t = \frac{1}{K}\sum_{r=1}^K \bm{e}_{i_r}$. This yields an unbiased but noisy estimate of $\bm{\alpha}_t$. Proposition 3 shows that the estimation error of MTS is equivalent to aggregating $K$ independent discrete CoT chains, reducing sample complexity by a factor of $K$.
    - Design Motivation: Base CoT2 is deterministic (no stochasticity), making it impossible to directly compute policy ratios for GRPO. MTS introduces controlled noise, allowing the GRPO policy ratio $r_t^{(i)}(\theta)$ to be defined and computed.

### Loss & Training
- **CSFT stage**: $\mathcal{L}_{CSFT} = \sum_{t=1}^m D(\bm{\alpha}_t^* \| \bm{\alpha}_t)$, using soft-label cross-entropy for intermediate steps and standard CE for the final step.
- **GRPO stage**: Standard GRPO clipped surrogate objective with KL regularization and sparse rewards (correct = 1, incorrect = 0).
- Teacher forcing is applied during CSFT (even though inference is autoregressive), which outperforms self-feeding.

## Key Experimental Results

### Main Results (MNNS task, 4 digits in range 1–99)

| Method | d=16 acc | d=24 acc | d=32 acc |
|------|---------|---------|---------|
| No-CoT | ~15% | ~15% | ~15% |
| Discrete CoT (B=1) | ~55% | ~70% | ~75% |
| COCONUT | ~45% | ~60% | ~65% |
| **CoT2 (B=16)** | ~60% | **~95%** | **~98%** |

### Pass@k Comparison (d=24, MNNS)

| Method | Pass@1 | Pass@4 | Pass@8 | Pass@16 |
|------|--------|--------|--------|---------|
| Discrete CoT | ~70% | ~82% | ~88% | ~93% |
| **CoT2** | **~95%** | ~96% | ~97% | ~98% |

### Key Findings
- **Single CoT2 pass ≈ multiple discrete CoT samples**: CoT2's Pass@1 matches the Pass@16 level of discrete CoT.
- **Budget–dimension sweet spot exists**: At $d=16$, $B=8$ is optimal ($B=16$ exceeds capacity); at $d=32$, $B=16$ is optimal.
- **GRPO is effective on CoT2**: RL fine-tuning leads the model to prioritize relevant reasoning paths, reducing the entropy of continuous tokens.
- **CoT2 outperforms COCONUT**: When external search supervision signals are available, directly fitting the multi-trajectory distribution is more effective than hidden-state substitution.
- **Strong theory–experiment agreement**: The lower bound $d=\Omega(B\log(v/B))$ is empirically validated.

## Highlights & Insights
- **Deep information-theoretic insight**: Discrete tokens carry at most $\log_2 v$ bits per step, whereas continuous tokens can pack $B \cdot \log_2(v/B)$ bits — this argument elegantly explains why continuous tokens are more expressive.
- **Theoretical guarantee "one pass ≈ K samples" (Proposition 3)** is a particularly powerful result, establishing a quantitative equivalence between CoT2 and self-consistency and giving continuous tokens a clear practical interpretation.
- **Extending RL to continuous action spaces for LLMs**: Conventional GRPO/PPO operates over discrete token spaces; CoT2's MTS strategy ingeniously introduces controllable noise in the continuous space via "sample-and-average," making policy gradient methods applicable.

## Limitations & Future Work
- Validation is limited to synthetic tasks (MNNS, ProntoQA, ProsQA); the approach has not been tested on real-world NLP tasks or large-scale LLMs.
- Assumption 1 (Markov property + linear superposition) may not hold strictly in practical Transformers.
- Continuous tokens cannot be directly interpreted as natural language, sacrificing the interpretability of CoT.
- The orthogonality of the vocabulary embedding matrix $\bm{E}$ affects superposition quality; in practice, embeddings may be highly correlated.
- Only the final step produces a discrete token; extending the framework to settings requiring multi-step discrete outputs (e.g., long-form answers) remains an open problem.

## Related Work & Insights
- **vs. COCONUT**: Both operate in the continuous thought chain paradigm, but COCONUT substitutes LLM hidden states without explicit multi-trajectory supervision; CoT2 directly fits the trajectory distribution via CSFT, yielding superior performance.
- **vs. Self-Consistency**: Self-consistency requires $K$ sampling passes with majority voting; CoT2 is theoretically equivalent to $K$ samples in a single forward pass, improving inference efficiency by a factor of $K$.
- **vs. Latent Reasoning (Coconut/Quiet-STaR)**: These methods focus on internalizing reasoning into latent space but lack CoT2's information-theoretic guarantees and the explicit formalization of multi-trajectory parallel tracking.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Information-theory-driven continuous reasoning with parallel tracking; solid theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐ Limited to synthetic tasks; large-scale LLM validation is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, intuitions are well-explained, and figures are well-designed.
- Value: ⭐⭐⭐⭐ Outstanding theoretical contribution to understanding continuous reasoning, though practical applicability remains to be demonstrated.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought](../../NeurIPS2025/llm_reasoning/reasoning_by_superposition_a_theoretical_perspective_on_chain_of_continuous_thou.md)
- [\[AAAI 2026\] Efficient Thought Space Exploration Through Strategic Intervention](../../AAAI2026/llm_reasoning/efficient_thought_space_exploration_through_strategic_intervention.md)
- [\[ICLR 2026\] Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)
- [\[ICLR 2026\] Verifying Chain-of-Thought Reasoning via Its Computational Graph](verifying_chain-of-thought_reasoning_via_its_computational_graph.md)
- [\[ICLR 2026\] SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes](scenecot_eliciting_grounded_chain-of-thought_reasoning_in_3d_scenes.md)

<!-- RELATED:END -->
