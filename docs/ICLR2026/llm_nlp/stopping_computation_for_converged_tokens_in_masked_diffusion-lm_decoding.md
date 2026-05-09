---
title: >-
  [Paper Note] Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding
description: >-
  [ICLR 2026][LLM/NLP][Masked Diffusion LM] This paper proposes SureLock, which permanently locks token positions in Masked Diffusion LMs once their posterior distributions stabilize after unmasking—skipping Q projection and FFN while caching KV—thereby reducing per-step attention computation from $O(N^2d)$ to $O(MNd)$. SureLock achieves 30–50% FLOPs reduction on LLaDA-8B without degrading generation quality.
tags:
  - ICLR 2026
  - LLM/NLP
  - Masked Diffusion LM
  - Inference Acceleration
  - Token Locking
  - KL Divergence
  - KV Cache
date: 2026-05-08
content_hash: 6276867bb4ab9ac9
---

# Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding

**Conference**: ICLR 2026
**arXiv**: [2602.06412](https://arxiv.org/abs/2602.06412)
**Code**: [https://daioba.github.io/surelock](https://daioba.github.io/surelock)
**Area**: LLM/NLP
**Keywords**: Masked Diffusion LM, Inference Acceleration, Token Locking, KL Divergence, KV Cache

## TL;DR
This paper proposes SureLock, which permanently locks token positions in Masked Diffusion LMs once their posterior distributions stabilize after unmasking—skipping Q projection and FFN while caching KV—thereby reducing per-step attention computation from $O(N^2d)$ to $O(MNd)$. SureLock achieves 30–50% FLOPs reduction on LLaDA-8B without degrading generation quality.

## Background & Motivation

**Background**: Masked Diffusion LMs (MDLMs, e.g., LLaDA, Dream) generate text via iterative denoising, requiring attention and FFN recomputation over all $N$ token positions at each step, with complexity $O(N^2d)$.

**Limitations of Prior Work**: Many tokens exhibit rapidly stabilizing posterior distributions after unmasking, yet standard samplers continue to recompute them at every step—resulting in substantial redundant computation. Existing acceleration methods either reduce the number of steps (temporal) or reuse KV across steps (reuse), but still emit $N$ query rows per step, leaving spatial complexity unchanged.

**Key Challenge**: Bidirectional attention in MDLMs requires full computation at every step, yet most unmasked tokens have effectively "converged" and do not require recomputation.

**Goal**: How to identify and permanently skip computation for converged tokens, achieving monotonically decreasing per-step computational cost?

**Key Insight**: Monitor the KL divergence of each token position's posterior between adjacent steps, and permanently lock positions whose divergence falls below a threshold $\varepsilon$.

**Core Idea**: Tokens with stable posteriors permanently exit computation (KV locked, Q/FFN skipped), and the active set shrinks monotonically as sampling progresses.

## Method

### Overall Architecture
At each iteration of MDLM, SureLock maintains two sets: the active set $\mathcal{A}_t$ (requiring computation) and the locked set $\mathcal{L}_t$ (computation skipped, but KV cached). Per-step Q projection and FFN are computed only for active positions; during attention, active tokens can still attend to locked tokens via cached KV.

### Key Designs

1. **Permanent Locking Mechanism**:

   - **Function**: Once token position $i$ is locked, Q projection and FFN are skipped for all subsequent steps, with cached $K, V$ values reused.
   - **Mechanism**: After locking, $\hat{p}_t^{(i)} = p_{t^*}^{(i)}$ (posterior frozen), while $K^{\text{all}}[\mathcal{L}_t] \leftarrow \mathcal{C}.k[\mathcal{L}_t]$ ensures other tokens can still attend to locked positions.
   - **Design Motivation**: Unlike selective-update methods (e.g., dLLM-Cache, which selects "what to compute now"), SureLock selects "what to permanently remove"—the active set is strictly non-increasing, guaranteeing monotonically decreasing computation.

2. **Locking Criterion: Inter-Step KL Divergence**:

   - **Function**: Position $i$ is locked when $D_t^{(i)} = \text{KL}(p_t^{(i)} \| p_{t-1}^{(i)}) \leq \varepsilon$.
   - **Mechanism**: KL divergence measures the degree of change in the posterior between adjacent steps. The threshold $\varepsilon$ translates directly into a terminal error bound $\delta = C_{\text{tail}}\sqrt{\varepsilon}$.
   - **Design Motivation**: Local KL is inexpensive to compute and theoretically tractable—Theorem 1 proves that the local KL at the locking step suffices to bound the deviation in the final token probability.

3. **Optional Confidence Gate**:

   - **Function**: $u_t^{(i)} = 1 - \max_v p_t^{(i)}(v) \leq q_m(u_t)$ serves as an auxiliary condition.
   - **Design Motivation**: Acts as a safety net—only tokens with sufficiently peaked posteriors are locked, preventing premature locking of positions that remain uncertain.

4. **Theoretical Error Bound (Theorem 1)**:

   - **Core Result**: $\|\log p_T^{(i)} - \log \hat{p}_T^{(i)}\|_\infty \leq C_{\text{tail}}\sqrt{D_{t^*}^{(i)}}$, where $C_{\text{tail}} = L_{\text{sm}} L / (1 - \sqrt{\rho})$.
   - **Significance**: Establishes a closed-form mapping from a locally computable KL threshold to a global terminal error bound, providing principled guidance for hyperparameter selection.

### Loss & Training
SureLock is a **training-free** inference-time method that does not modify model parameters. It is orthogonal to and composable with existing temporal and reuse acceleration methods.

## Key Experimental Results

### Main Results

**LLaDA-8B-Instruct (MT-Bench + WikiText-103)**:

| Configuration | FLOPs Reduction | Generation Quality |
|---|---|---|
| Baseline (no locking) | 0% | Baseline |
| SureLock (ε=0.01) | ~30% | ≈ Baseline |
| SureLock (ε=0.05) | ~50% | ≈ Baseline |
| SureLock (ε=0.1) | ~50%+ | Slight degradation |

### Ablation Study

| Configuration | FLOPs Savings | Quality | Notes |
|---|---|---|---|
| KL criterion only | Comparable to full method | ≈ | KL alone suffices as the sole criterion |
| Confidence gate only | Less aggressive | Maintained | Confidence alone is insufficient |
| Without KV caching | Same FLOPs | Significant degradation | Locked tokens become unattendable, causing information loss |

### Key Findings
- The active position count $M_t$ **monotonically decreases** with step index, confirming the hypothesis that tokens converge progressively.
- 30–50% FLOPs reduction incurs **zero or negligible degradation** on WikiText-103 perplexity and MT-Bench scores.
- SureLock is orthogonal to temporal methods (step reduction) and reuse methods (cross-step KV reuse), and can be combined with both.
- Although the theoretical bound is conservative (designed for guidance rather than precise prediction), it provides clear principled guidance for tuning $\varepsilon$.

## Highlights & Insights
- **From "which computations to perform" to "which computations to permanently eliminate"**: This perspective shift is the key innovation. The strictly shrinking active set yields a monotonically decreasing computation curve, making the approach more predictable than selective-update methods.
- **Theory-practice closed loop**: The closed-form relationship between KL threshold and terminal error bound provides theoretically grounded hyperparameter selection without relying on empirical tuning—a rare property in systems acceleration work.
- **Complementarity with d²Cache**: d²Cache selectively determines which tokens to update at each step at fine granularity, while SureLock permanently removes converged tokens—the two approaches are composable.

## Limitations & Future Work
- The geometric tail-contraction assumption (A2) in Theorem 1 is relatively strong; posterior dynamics may not strictly satisfy it in practice.
- Permanent locking is irreversible—if a locked token's optimal value changes due to updates at other positions, recovery is impossible.
- Validation is limited to LLaDA-8B; other MDLMs (e.g., Dream, MDLM-original) are not evaluated.
- The locking threshold $\varepsilon$ still requires manual specification, and the constant $C_{\text{tail}}$ in the theoretical bound is difficult to estimate precisely.

## Related Work & Insights
- **vs. dLLM-Cache**: dLLM-Cache selectively updates a subset of tokens at each step (non-permanently); SureLock permanently locks tokens, yielding a monotonically shrinking active set and greater long-term savings.
- **vs. Fast-dLLM**: Fast-dLLM adopts semi-autoregressive block-level decoding; SureLock operates at the token level, offering finer granularity and orthogonal composability.
- **vs. AR KV Cache**: KV caches in autoregressive models grow naturally with sequence length; SureLock's KV cache approximates a similar "invariant cache" effect within bidirectional attention.

## Rating
- Novelty: ⭐⭐⭐⭐ The permanent locking of converged tokens is a concise and effective idea, further strengthened by the theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐ Limited to a single model with restricted task coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, algorithmic presentation is precise, and theoretical derivations are complete.
- Value: ⭐⭐⭐⭐ Introduces a new orthogonal dimension for MDLM inference acceleration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](d2cache_accelerating_diffusion-based_llms_via_dual_adaptive_caching.md)
- [\[ACL 2026\] Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness](../../ACL2026/llm_nlp/masked_by_consensus_disentangling_privileged_knowledge_in_llm_correctness.md)
- [\[ICLR 2026\] Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)
- [\[ICLR 2026\] Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents](enhancing_persona_following_at_decoding_time_via_dynamic_importance_estimation_f.md)
- [\[ICLR 2026\] DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas](dreamon_diffusion_language_models_for_code_infilling_beyond_fixed-size_canvas.md)

</div>

<!-- RELATED:END -->
