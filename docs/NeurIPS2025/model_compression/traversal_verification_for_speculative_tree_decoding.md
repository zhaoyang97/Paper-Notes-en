---
title: >-
  [Paper Note] Traversal Verification for Speculative Tree Decoding
description: >-
  [NeurIPS 2025][Model Compression][Speculative Decoding] This paper proposes Traversal Verification, a bottom-up verification algorithm that traverses from leaf nodes to the root. Rather than making acceptance/rejection d…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Speculative Decoding"
  - "Tree Decoding"
  - "Verification Algorithm"
  - "Lossless Inference Acceleration"
  - "Sequence-Level Acceptance"
date: 2026-05-08
content_hash: 104b1bb2617d55ae
---

# Traversal Verification for Speculative Tree Decoding

**Conference**: NeurIPS 2025
**arXiv**: [2505.12398](https://arxiv.org/abs/2505.12398)  
**Code**: Unavailable  
**Area**: Model Compression
**Keywords**: Speculative Decoding, Tree Decoding, Verification Algorithm, Lossless Inference Acceleration, Sequence-Level Acceptance

## TL;DR

This paper proposes Traversal Verification, a bottom-up verification algorithm that traverses from leaf nodes to the root. Rather than making acceptance/rejection decisions based on per-token probabilities, it considers the sequence-level probability of entire paths, thereby maximizing candidate utilization. The method is theoretically proven to be lossless and optimal on single chains, and consistently improves acceptance length by 2.2%–5.7% across diverse tree structures and tasks.

## Background & Motivation

Speculative decoding accelerates LLM inference by generating a candidate token tree with a draft model and verifying it in parallel with the target model. Existing tree decoding verification methods (e.g., SpecInfer's RRS) inherit the token-level verification mechanism from vanilla speculative decoding and suffer from two critical limitations:

**Mismatch between sequence-level and token-level probabilities**: Vanilla speculative decoding accepts tokens based on the per-token probability ratio $\min(1, p(y_i)/q(y_i))$. However, the joint acceptance probability of a sequence should account for the joint distribution over the entire path. Token-level decisions sacrifice global optimality. For instance, if parent node $X_1$ has an acceptance rate of 0.5 and child node $X_3$ has a ratio of $4/3$, the token-level acceptance probability is $0.5 \times 1 = 0.5$, whereas the sequence-level probability is $\min(0.5 \times 4/3, 1) \approx 0.667$—strictly higher.

**Top-down verification wastes candidates**: Existing methods verify layer by layer from the root downward; once a parent node is rejected, all its descendants are discarded, even if they might form higher-quality sequences.

**Key Challenge**: Token-level verification is locally optimal but not globally optimal in tree-structured decoding.

**Key Insight**: Reverse the verification direction—begin from leaf nodes, accepting the entire path upon a successful verification, and backtracking to sibling or parent nodes upon rejection, so as to maximally exploit all candidate tokens.

## Method

### Overall Architecture

Traversal Verification is a plug-and-play verification module that replaces the verification step in existing speculative decoding pipelines without modifying draft generation or any other component. The core change is replacing top-down layer-wise verification with bottom-up traversal verification.

### Key Designs

1. **Sequence-level acceptance rate initialization**: For each chain $\alpha = (X_0, X_1, \ldots, X_{\gamma_\alpha})$ in tree $T$, the sequence-level acceptance rate at each node is computed recursively:

    $p_\alpha^{ini}(X_i) = \min\left\{p_\alpha^{ini}(X_{i-1}) \cdot \frac{\mathcal{M}_b(X_i | X^{i-1})}{\mathcal{M}_s(X_i | X^{i-1})}, 1\right\}$

   where $p_\alpha^{ini}(X_0) = 1$. The acceptance rate thus depends not only on the current token but on the cumulative probability ratio along the entire path from the root. **Design Motivation**: The sequence-level probability $\prod p_i/q_i$ may exceed the product of per-token truncated ratios $\prod \min(p_i/q_i, 1)$, because high-probability child nodes can "compensate" for low-probability parent nodes.

2. **Bottom-up traversal order**: Verification begins at the first leaf node (i.e., the deepest leftmost node), drawing $\eta \sim U(0,1)$:

    - If $\eta < p_\alpha(X_{\gamma_\alpha})$: accept the entire path $(X_0, \ldots, X_{\gamma_\alpha})$
    - If rejected: remove the current leaf node, update the residual and draft distributions, and move to the sibling node or backtrack to the parent

   An example traversal order for the tree in Figure 1: $X_3 \to X_4 \to X_1 \to X_5 \to X_2$. A key property is that a parent node is only verified after all its children have been rejected, maximizing candidate utilization.

3. **Residual distribution update**: When node $X_{\gamma_\alpha}$ is rejected:

    $\mathcal{M}_b'(x | X^{\gamma_\alpha - 1}) = \text{norm}([p_\alpha(X_{\gamma_\alpha-1}) \cdot \mathcal{M}_b(x | X^{\gamma_\alpha-1}) - \mathcal{M}_s(x | X^{\gamma_\alpha-1})]_+)$

   The residual acceptance rate of the parent node $p'_\alpha(X_{\gamma_\alpha-1})$ is updated accordingly and propagated to all sibling chains sharing the same prefix, ensuring correct redistribution of probability mass.

### Theoretical Guarantees

- **Theorem 3.3 (Losslessness)**: The output distribution of Traversal Verification is identical to that of the target model. The proof exploits the self-similarity of the algorithm—each subtree is a scaled instance of the overall verification mechanism—and proceeds by mathematical induction on the number of descendant nodes.

- **Theorem 3.4 (Single-Chain Optimality)**: When the tree degenerates to a single chain, the expected acceptance length of Traversal Verification equals that of Block Verification (the known optimum), and is $\geq$ that of any other valid verification algorithm.

## Key Experimental Results

### Main Results: Llama3.2-1B + Llama3.1-8B (Temperature = 1)

| Tree Structure | Token-level Avg Accept | Traversal Avg Accept | Gain |
|---|---|---|---|
| Chain (depth=5) | 3.88 | 3.99 | **2.8%** |
| Binary Tree (depth=5) | 4.63 | 4.73 | **2.2%** |
| EAGLE Sparse Tree | 4.49 | 4.60 | **2.4%** |
| Chain Avg Token/s | 51.2 | 52.5 | 2.5% |
| Binary Avg Token/s | 54.0 | 54.9 | 1.7% |
| EAGLE Avg Token/s | 57.3 | 58.5 | 2.1% |

### Ablation Study: Effect of Chain Depth and Tree Size

| Chain Depth | Token-level Accept | Traversal Accept | Gain |
|---|---|---|---|
| Depth 2 | 2.55 | 2.58 | 1.2% |
| Depth 4 | 3.47 | 3.56 | 2.6% |
| Depth 6 | 4.10 | 4.23 | 3.2% |
| Depth 8 | 4.50 | 4.67 | **3.8%** |

| Binary Tree Depth (# Nodes) | Token-level Accept | Traversal Accept | Gain |
|---|---|---|---|
| Depth 2 (7 nodes) | 3.12 | 3.17 | 1.6% |
| Depth 3 (15 nodes) | 3.94 | 4.03 | 2.3% |
| Depth 4 (31 nodes) | 4.63 | 4.73 | 2.2% |
| Depth 5 (63 nodes) | 5.12 | 5.27 | **2.9%** |

### Key Findings

- Gains from Traversal Verification become more pronounced as trees grow deeper and larger, since the sequence-level probability compensation effect is amplified in longer chains.
- Improvements are larger at higher temperatures (2.8% at temp=1.0 vs. 1.0% at temp=0.2), as more diffuse probability distributions widen the gap between sequence-level and token-level acceptance.
- Gains are more substantial under weaker draft model configurations (e.g., Llama-68M + Llama2-7B), averaging 3.8%–5.7%.
- Throughput improvements are slightly lower than acceptance length improvements due to the marginal additional computation introduced by the bottom-up traversal.

## Highlights & Insights

- The core insight is remarkably elegant: the joint sequence probability $\min(\prod r_i, 1)$ can be strictly greater than the product of per-token truncated probabilities $\prod \min(r_i, 1)$.
- The theoretical analysis is rigorous and complete: losslessness is proven via self-similarity and mathematical induction, and single-chain optimality is established by equivalence with Block Verification.
- The plug-and-play design enables all existing speculative decoding systems to benefit without modifying draft generation or model architectures.
- As speculative decoding systems trend toward larger and deeper trees (e.g., Sequoia with 768 nodes), the advantages of Traversal Verification are expected to become increasingly significant.

## Limitations & Future Work

- Bottom-up traversal may introduce non-trivial computational overhead (maintaining and updating acceptance rates for all chains), causing throughput gains to lag behind theoretical acceptance length improvements.
- The method is currently evaluated only under temperature sampling; its behavior under greedy decoding remains unexplored.
- The paper does not discuss synergy with learning-based draft methods such as EAGLE or Medusa.
- More optimized implementations (e.g., parallelizing traversal computations) could further unlock performance gains.

## Related Work & Insights

- Traversal Verification is equivalent to Block Verification on single chains, while generalizing to arbitrary tree structures.
- It is orthogonal to SpecInfer, EAGLE, and similar methods—those focus on improving draft generation, whereas this work focuses on improving verification.
- **Insight**: The verification stage of speculative decoding remains an underexplored optimization target, independent of improvements to the draft model.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The core insight of reversing the verification direction is highly innovative; sequence-level acceptance rates represent a fundamental improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple models, tasks, and tree structures; ablations span depth, size, and temperature.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous, intuitive examples are clear, and algorithmic pseudocode is complete.
- Value: ⭐⭐⭐⭐ Moderate improvement magnitude (2–6%), but the zero-cost plug-and-play property confers high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reject Only Critical Tokens: Pivot-Aware Speculative Decoding](reject_only_critical_tokens_pivot-aware_speculative_decoding.md)
- [\[NeurIPS 2025\] CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs](casspec_cascade_adaptive_selfspeculative_decoding_for_onthef.md)
- [\[AAAI 2026\] Steering Pretrained Drafters during Speculative Decoding](../../AAAI2026/model_compression/steering_pretrained_drafters_during_speculative_decoding.md)
- [\[NeurIPS 2025\] AutoJudge: Judge Decoding Without Manual Annotation](autojudge_judge_decoding_without_manual_annotation.md)
- [\[ACL 2026\] Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](../../ACL2026/model_compression/calibrated_speculative_decoding_frequency-guided_candidate_selection_for_efficie.md)

</div>

<!-- RELATED:END -->
