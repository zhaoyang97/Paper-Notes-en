---
title: >-
  [Paper Note] Cascade: Token-Sharded Private LLM Inference
description: >-
  [ICML 2025][LLM Safety][Private Inference] Proposes Cascade, a multiparty inference protocol based on token-dimension sharding. By distributing hidden states to different computation nodes along the token dimension, it avoids the high overhead of cryptographic primitives, achieving up to 100× faster inference than SMPC schemes while maintaining resilience against vocab-matching attacks.
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "Private Inference"
  - "Token Sharding"
  - "Multiparty Computation"
  - "LLM Privacy"
  - "Vocab-matching Attack"
date: 2026-05-08
content_hash: 35ff2b65281a20de
---

# Cascade: Token-Sharded Private LLM Inference

**Conference**: ICML 2025  
**arXiv**: [2507.05228](https://arxiv.org/abs/2507.05228)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Private Inference, Token Sharding, Multiparty Computation, LLM Privacy, Vocab-matching Attack

## TL;DR
Proposes Cascade, a multiparty inference protocol based on token-dimension sharding. By distributing hidden states to different computation nodes along the token dimension, it avoids the high overhead of cryptographic primitives, achieving up to 100× faster inference than SMPC schemes while maintaining resilience against vocab-matching attacks.

## Background & Motivation

**Background**: The growth of LLM parameter size has made third-party inference services increasingly popular, but user data privacy is at risk. Secure Multiparty Computation (SMPC) provides provable security but suffers from immense computational and communication overhead on non-linear operations.

**Limitations of Prior Work**:
   - SMPC schemes (MPCFormer, Puma, SecFormer) use polynomial approximations on GELU/Softmax, which degrades performance and remains slow.
   - Statistical schemes (PermLLM, STIP, Centaur) achieve privacy through hidden state permutation but are easily broken by the vocab-matching attack of Thomas et al. (recovering input by matching vocabulary candidates to hidden states token-by-token).

**Key Challenge**: Cryptographic schemes are secure but not scalable to large LLMs; permutation schemes are fast but insecure.

**Goal**: How to find a new balance between security and efficiency?

**Key Insight**: Instead of manipulating token permutation, this work shards along the token dimension—each computation node only sees the hidden states of a subset of tokens, and adjacent tokens maintain sufficiently large gaps.

**Core Idea**: The computational complexity of vocab-matching attacks grows exponentially with the token gap between hidden states ($V^g$, where $g$ is the gap). As long as the gap is sufficiently large ($g \geq 3$), the attack becomes infeasible.

## Method

### Overall Architecture
Cascade distributes inference computation across two types of nodes:
- **CompNodes**: Each holds the hidden states of a subset of tokens and executes all "token-independent" operations (FFN, LayerNorm, MLP, etc.).
- **AttnNodes**: Receive Q/K/V projection shards from CompNodes, perform attention computation, and return parts of the attention outputs back to the respective CompNodes.

Key security stems from: each node only sees non-consecutive subsets of tokens, and the token gaps cause the search space of vocab-matching attacks to explode exponentially.

### Key Designs

1. **Token Dimension Sharding Strategy**:

    - **Function**: Partition the hidden states of $N$ tokens into $\alpha$ non-overlapping subsets and distribute them to different CompNodes.
    - **Mechanism**: Ensure that the minimum gap between tokens seen by each node is $g \geq \rho$ (the vocab-matching threshold), making a generalized vocab-matching attack require $V^g$ forward inferences, which is computationally infeasible.
    - **Design Motivation**: For a typical vocabulary size $V \sim 10^5$, when $g \geq 3$, the search space $V^3 \sim 10^{15}$, which exceeds feasible bounds.

2. **CompNode-AttnNode Division of Labor**:

    - **Function**: Decouple Transformer computation into token-independent and token-interactive parts.
    - **Mechanism**: CompNodes execute embedding, FFN, and LayerNorm (which are independent across the batch/token dimensions), while AttnNodes execute attention (which requires interaction between tokens).
    - **Design Motivation**: Attention is the only operation requiring information interaction between tokens; hence, controlled information exchange is restricted solely to this step.

3. **Generalized Vocab-Matching Attack Analysis**:

    - **Function**: Generalize the vocab-matching attack of Thomas et al. to sharded scenarios.
    - **Mechanism**: An attacker holding only a subset of hidden states $h_{i_1}, \dots, h_{i_k}$ faces a search cost to recover all tokens of $V^{i_1} + V^{i_2-i_1} + \dots + V^{i_k-i_{k-1}}$, which is dominated by the maximum gap $g = \max_j(i_{j+1}-i_j)$.
    - **Design Motivation**: Provide a quantitative foundation for the security analysis of Cascade.

### Loss & Training
- Training-free; it is an inference-time protocol.
- No performance degradation—it does not use non-linear approximations, making computation nearly identical to standard forward propagation.
- Communication cost is only 1/150 of SMPC.

## Key Experimental Results

### Main Results
Efficiency comparison with SMPC schemes:

| Scheme | Inference Speed | Communication Overhead | Performance Degradation | Security Level |
|------|---------|---------|---------|---------|
| SMPC Schemes | 1× (Baseline) | 1× (Baseline) | Yes (Polynomial approx.) | Cryptographic |
| PermLLM | ~10× | ~5× | None | Broken |
| **Cascade** | **~100×** | **1/150×** | **None** | Statistical |

### Ablation Study

| Configuration | Security | Description |
|------|--------|------|
| Gap $g=1$ (Adjacent tokens) | Insecure | Degrades to standard vocab-matching |
| Gap $g=2$ | Boundary | $V^2 \sim 10^{10}$, theoretically feasible but expensive |
| Gap $g \geq 3$ | Secure | $V^3 \sim 10^{15}$+, infeasible |
| Learning-based Attack | Failed | Training methods cannot recover tokens from sharded hidden states |

### Key Findings
- Cascade is secure against generalized vocab-matching attacks (when $g \geq \rho = 3$).
- It is equally robust against learning-based inversion attacks (Wan et al., Morris et al.).
- Seamlessly scales to modern large-scale LLMs (DeepSeek, Qwen, etc.).
- Requires no modifications or fine-tuning to the model.

## Highlights & Insights
- **Clever Dimension Choice**: While all prior statistical schemes manipulate the hidden dimension or token permutation, Cascade shards along the token dimension for the first time, leveraging the property of the Transformer architecture that "only attention requires token interaction."
- The analysis of the generalized vocab-matching attack provides a quantitative security threshold, making security configurable.
- The 100× speedup makes private inference practically usable on modern SOTA LLMs for the first time.

## Limitations & Future Work
- Statistical security rather than cryptographic security—does not rule out the possibility of stronger attacks in the future.
- AttnNodes can still see Q/K/V shards during attention computation, presenting potential information leakage.
- Semi-honest model assumption—does not resist malicious participants.
- The sharding strategy requires multiple participating parties, imposing requirements on the deployment architecture.
- Does not analyze the security of the KV-cache in autoregressive generation.

## Related Work & Insights
- **vs PermLLM/STIP/Centaur**: These permutation schemes are broken by vocab-matching attacks. Cascade achieves security through sharding instead of permutation.
- **vs SMPC**: SMPC is cryptographically secure but 100× slower; Cascade trades cryptographic security for statistical security to achieve practicality.
- **vs POST (2505.15252)**: POST accelerates secure inference from the perspective of speculative decoding, whereas Cascade does so from a sharding perspective; the two are orthogonal.

## Rating
- Novelty: ⭐⭐⭐⭐ Token-sharded private inference introduces a new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive efficiency and security analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear attack analysis with a theoretically grounded security threshold.
- Value: ⭐⭐⭐⭐⭐ Makes private LLM inference practically usable on large models for the first time.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] An Attack to Break Permutation-Based Private Third-Party Inference Schemes for LLMs](an_attack_to_break_permutation-based_private_third-party_inference_schemes_for_l.md)
- [\[NeurIPS 2025\] MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference](../../NeurIPS2025/llm_safety/mpcache_mpc-friendly_kv_cache_eviction_for_efficient_private_llm_inference.md)
- [\[ICLR 2026\] DP-Fusion: Token-Level Differentially Private Inference for Large Language Models](../../ICLR2026/llm_safety/dp-fusion_token-level_differentially_private_inference_for_large_language_models.md)
- [\[ICML 2025\] POPri: Private Federated Learning using Preference-Optimized Synthetic Data](popri_private_federated_learning_using_preference-optimized_synthetic_data.md)
- [\[ICML 2025\] Align-then-Unlearn: Embedding Alignment for LLM Unlearning](align-then-unlearn_embedding_alignment_for_llm_unlearning.md)

</div>

<!-- RELATED:END -->
