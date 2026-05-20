---
title: >-
  [Paper Note] Circuit Fingerprints: How Answer Tokens Encode Their Geometrical Path
description: >-
  [ICML 2026][Interpretability][Circuit discovery] This paper proposes the Circuit Fingerprint hypothesis: when an answer token is fed into a Transformer in isolation…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Circuit discovery"
  - "activation steering"
  - "geometric alignment"
  - "answer token fingerprint"
  - "Shapley decomposition"
date: 2026-05-08
content_hash: fd62f1de8e8e4e42
---

# Circuit Fingerprints: How Answer Tokens Encode Their Geometrical Path

**Conference**: ICML 2026  
**arXiv**: [2602.09784](https://arxiv.org/abs/2602.09784)  
**Code**: Not explicitly released  
**Area**: Interpretability / Mechanistic Interpretability / Activation Steering  
**Keywords**: Circuit discovery, activation steering, geometric alignment, answer token fingerprint, Shapley decomposition

## TL;DR
This paper proposes the Circuit Fingerprint hypothesis: when an answer token is fed into a Transformer in isolation, the direction it leaves in the hidden space precisely corresponds to the circuit path required to produce that answer. Based on this, circuit discovery can be achieved via pure geometric alignment (without gradients/interventions), and the same set of directions can be used for activation steering, demonstrating that "read" and "write" are two sides of the same geometric object.

## Background & Motivation
**Background**: Mechanistic interpretability currently has two main threads: (i) circuit discovery uses activation patching or gradient-based approximations (EAP / EAP-IG) to identify attention heads/MLP subnetworks critical for a task; (ii) activation steering adds learned directions to the residual stream to control model behavior. Both operate on the same representation space but have developed independently.

**Limitations of Prior Work**: Patching methods require $O(LH)$ forward passes; gradient-based methods (attribution patching, EAP-IG) suffer from instability due to saturation and LayerNorm nonlinearity; mask-learning methods (ACDC, edge pruning) require iterative optimization. On the steering side, contrastive data collection, direction learning, and intervention strength tuning are necessary, and "where to intervene" and "which direction to use" are handled separately.

**Key Challenge**: If circuits are stably encoded in model weights, then discovery and steering should operate on the same object—yet current methods split this into two disconnected toolsets. The linear representation hypothesis (Park 2024, Elhage 2022) suggests a unified geometric perspective, but it has never been used to explain both phenomena simultaneously.

**Goal**: To answer, via a geometric principle, (i) which components belong to a circuit (read); (ii) how to intervene on these components to change the output (write); all without relying on gradients or interventions, using only pure forward projections.

**Key Insight**: Feed the answer token (e.g., "Paris") into the model in isolation—it has no context, but since the circuit is encoded in fixed weights, the path it activates is precisely the capital-city recall circuit. Thus, $\Delta r^{(L)}=r_{a_+}^{(L)}-r_{a_-}^{(L)}$ naturally becomes the geometric signature of the direction required to produce $a_+$ vs $a_-$.

**Core Idea**: Circuit membership = alignment between component output and the "answer token difference direction"; using the same direction for steering constitutes the write operation; read and write are duals of the same set of directions.

## Method

### Overall Architecture
Two input sources: (1) Answer token pairs $(a_+, a_-)$—independently forward-passed to obtain difference directions $\Delta r^{(L)}, \Delta v^{(\ell,h)}, \Delta q^{(\ell,h)}, \Delta k^{(\ell,h)}$; (2) contrastive prompts (clean vs corrupted)—to obtain component output differences $\Delta o_c$.  
Read path: project the target direction into each component's native space, take the inner product with $\Delta o_c$ to get node importance $S_c$; use three-channel Q/K/V Shapley decomposition + residual stream decomposition to compute edge importance.  
Write path: at heads identified in the read phase, replace or add the direction $\hat d_s, \hat d_t$ in the same space to obtain steered activation.

### Key Designs

1. **Answer token difference as geometric target + projection into component native space**:

    - **Function**: With just 2 forward passes (feeding $a_+$ and $a_-$), obtain the target direction $\Delta r^{(L)}, \Delta v, \Delta q, \Delta k$ for each layer and head, without training or gradients.
    - **Mechanism**: For component $c$ (attention head uses $W_O$, MLP uses $W_{\text{out}}$), first transform the target direction into the component's native space $\hat t_c=W_c^\top \Delta r^{(L)}/\|\Delta r^{(L)}\|$, then compute $S_c=\langle \Delta o_c,\hat t_c\rangle$; this "native space inner product" ensures $\sum_c S_c$ equals the total projection onto the target direction in the residual stream (additivity preserved).
    - **Design Motivation**: Directly projecting $\Delta o_c$ into the residual stream and then taking the inner product introduces geometric confounds from shared projection matrices like $W_O$; projecting into the component's native space decouples "how the component internally produces this direction" from "shared geometry in the residual stream," yielding a clean, additive importance metric.

2. **Q/K/V Shapley decomposition + edge residual stream decomposition via backward propagation**:

    - **Function**: For each edge $i\to j$ (information flowing from upstream component to downstream attention head), measure via Q/K/V channels separately, and use Shapley values to assign channel weights without arbitrary choices.
    - **Mechanism**: For each channel (e.g., K), write $R^{(K)}_{i\to j}=\langle \Delta o_i, W^{(j)}_K \Delta k^{(j)}\rangle/\langle\Delta r^{(\ell_j)}, W^{(j)}_K \Delta k^{(j)}\rangle$ (by linearity, $\sum_i R^{(K)}_{i\to j}=1$); treat Q/K/V as a three-player cooperative game, run $2^3=8$ coalitions per head to measure importance, yielding Shapley weights $\phi_Q,\phi_K,\phi_V$; final edge importance $E_{i\to j}=S_j\cdot(\phi_Q R^{(Q)}_{i\to j}+\phi_K R^{(K)}_{i\to j}+\phi_V R^{(V)}_{i\to j})$; propagate indirect importance backward layer by layer (Alg. 1).
    - **Design Motivation**: Simply distributing a head's total score among the three channels is arbitrary; Shapley is the **only** allocation in cooperative game theory satisfying fairness axioms. Shapley values also guarantee $\phi_Q+\phi_K+\phi_V=S_{QKV}-S_\emptyset$, so additivity of importance extends to the edge level. Fig. 4 empirically finds Name Mover heads are Q-dominated, S-Inhibition heads are K-dominated, matching Wang 2022's manual role assignments, validating the explanatory power of Shapley decomposition.

3. **Geometric steering: directly using read directions for write**:

    - **Function**: Use the same set of heads and answer directions identified in the read phase to control generation, validating read-write duality.
    - **Mechanism**: For answer prototypes $\{r_1,\dots,r_k\}$, center and perform SVD to obtain orthogonal basis $\{u_i\}$, project source/target prototypes onto this basis to get $d_s,d_t$; for factual recall, use replacement $X'=X-\|d_s-d_t\|\hat d_s+\|d_s-d_t\|\hat d_t$; for stylistic (emotion, language) tasks, use magnitude transfer $X'=X-\|d_s\|(\hat d_s-\hat d_t)$.
    - **Design Motivation**: Compare with activation patching (directly transplanting corrupted activations)—the latter is the upper bound for steering. If geometric directions can approximate patching effects, it proves the fingerprint is not just a surface correlation but a true causal structure; experiments show on IOI, at $\alpha=1$, $P(\text{correct})=0.014$ vs patching's 0.0, logit diff $-4.07$ vs $-7.34$, indicating comparable behavioral effects.

### Loss & Training
**Completely training-free, gradient-free, and intervention-free**. Only 2 forward passes ($a_+$ and $a_-$) are needed to obtain the target direction, plus contrastive prompts for forward passes to compute $\Delta o_c$; Shapley requires 8 coalition evaluations, which can be batched. The computational budget is comparable to EAP (a single backward pass).

## Key Experimental Results

### Main Results

| Model | Method | IOI CMD↓ | IOI CPR↑ | SVA CMD↓ | SVA CPR↑ | MCQA CMD↓ | MCQA CPR↑ |
|-------|--------|---------|---------|---------|---------|---------|-----------|
| GPT2-Small | EAP-IG | 0.03 | 0.97 | 0.05 | 0.95 | N/A | N/A |
|  | **CF (ours)** | 0.06 | **0.98** | 0.09 | 0.91 | N/A | N/A |
| Qwen2.5-0.5B | EAP-IG | 0.01 | 1.00 | 0.05 | 0.99 | 0.05 | 95.0 |
|  | **CF (ours)** | 0.04 | 0.96 | 0.06 | 0.94 | 0.09 | 92.0 |
| Llama3.2-1B | EAP-IG | 0.01 | 0.99 | 0.03 | 0.98 | 0.05 | 95.0 |
|  | **CF (ours)** | 0.02 | 0.99 | 0.05 | 0.96 | 0.13 | 0.87 |
| OPT-1.3B | EAP-IG | 0.00 | 1.50 | 0.01 | 1.00 | 0.04 | 0.96 |
|  | **CF (ours)** | 0.01 | 0.99 | 0.05 | 0.95 | 0.07 | 0.93 |

CF matches gradient-based methods on IOI/SVA and is fully comparable to EAP; slightly weaker on MCQA.

### Steering Evaluation

| Metric | Baseline (instruction prompting) | CF Steered |
|--------|----------------------------------|-----------|
| Emotion Classification Accuracy | 53.1% | **69.8%** |
| Perplexity (median) | 17.03 | **13.37** |
| Factual Accuracy | 90.1% | 89.6% |

Steering maintains/improves factual accuracy for positive emotions (joy, 100%), but for negative emotions (sadness 81%, disgust 78%), "phonetic contamination" of recalled names occurs (e.g., Einstein becomes "Sissoar").

### Key Findings
- CMD/CPR is nearly on par with gradient baselines, and as model size increases, the geometric method approaches EAP-IG; attributed to "better concept disentanglement in larger models."
- Shapley decomposition reveals head functional roles: Name Mover heads are Q-dominated, S-Inhibition heads are K-dominated, consistent with IOI literature's manual classification.
- The same set of directions is effective for both read and write (patching upper bound and CF steering curves almost overlap), providing strong evidence for read-write duality.
- Persona/emotion instruction prefixes in prompt engineering can also be used to extract directions, showing the fingerprint method generalizes to any "prompt-controllable attribute."

## Highlights & Insights
- **"Answer tokens carry circuit fingerprints" is highly counterintuitive**: It is usually assumed that circuits are only activated when generating the answer; this work shows that even as input tokens, answer tokens traverse the original path (even suppressing incorrect candidates), quantifying the intuition that "circuits are stable structures" into readable directions.
- **Read-write duality = true causal validation**: The identified directions are not only "apparently important" but can be directly used for steering and replicate patching effects—this upgrades interpretability from "post hoc description" to "a priori intervention," a paradigm shift compared to SAE/probe's "look but don't touch."
- **Shapley decomposition for channels is transferable**: Three-channel $2^3=8$ coalitions suffice for fair allocation, replacing "arbitrary weight combinations" with "the unique game-theoretic solution"; this idea is fully reusable in multi-branch modules (MoE, multi-expert fusion).
- **Method requires only 2 forward passes**: Compared to EAP's $O(LH)$ or backward gradients, CF is nearly free computationally, especially friendly for large models.

## Limitations & Future Work
- Experiments are limited to small models (GPT2-Small/Qwen-0.5B/Llama-1B/OPT-1.3B, ≤1.3B); not validated on 7B+; on MCQA, Llama3.2-1B's CPR of 0.87 is clearly lower than EAP-IG (0.95), indicating geometric approximation is less tight on more complex tasks.
- Focuses only on the final token position, ignoring indirect effects from earlier positions and LayerNorm nonlinearity (explicitly simplified by the authors); edge attribution accuracy in long contexts remains to be tested.
- Zero-shot steering remains fragile: negative emotions can contaminate semantics, producing absurd errors like "Bonniweeper," suggesting some features remain entangled with lexical content and cannot be fully linearly disentangled.
- Evaluation metrics CMD/CPR are from the MIB benchmark; systematic comparison with other interpretability tasks (feature ablation, faithfulness) is lacking.

## Related Work & Insights
- **vs ACDC / Edge Pruning / EAP-IG**: Traditional circuit discovery requires iterative search/mask learning or gradient backpropagation; CF uses only 2 forward passes + 8 Shapley coalition evaluations, reducing circuit discovery to the scale of a single forward pass with no gradient dependence.
- **vs Activation Steering (Turner 2023, Zou 2023)**: Existing steering requires preparing contrastive data and separately learning directions; this work directly reuses the same set of directions and heads found in circuit discovery, unifying "where to intervene" and "which direction to use."
- **vs Linear Representation Hypothesis**: Park 2024, Elhage 2022 proposed that "features are directions in activation space" as a descriptive claim; this work provides operational evidence—these directions not only describe features but also the circuits that produce them, a strong unification of "feature geometry ≡ circuit geometry."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The Circuit Fingerprint hypothesis + read-write duality unifies two previously independent interpretability research threads into a single geometric object, with highly original insights.
- Experimental Thoroughness: ⭐⭐⭐ Covers 4 model families × 3 tasks (IOI/SVA/MCQA) + 5 emotion steering cases, but model scale is small and benchmarks are limited.
- Writing Quality: ⭐⭐⭐⭐ Concepts are clearly explained, Shapley derivation and algorithms are complete, and limitations are candidly discussed.
- Value: ⭐⭐⭐⭐ Provides a gradient-free, low-cost tool for simultaneous discovery and control, with practical value for alignment and behavioral editing research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding](../../ICLR2026/interpretability/how_do_transformers_learn_to_associate_tokens_gradient_leading_terms_bring_mecha.md)
- [\[ICML 2026\] All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs](all_circuits_lead_to_rome_rethinking_functional_anisotropy_in_circuit_and_sheaf_.md)
- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](../../ICLR2026/interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[ACL 2025\] Position-aware Automatic Circuit Discovery](../../ACL2025/interpretability/position-aware_automatic_circuit_discovery.md)
- [\[ICLR 2026\] Closing the Curvature Gap: Full Transformer Hessians and Their Implications for Scaling Laws](../../ICLR2026/interpretability/closing_the_curvature_gap_full_transformer_hessians_and_their_implications_for_s.md)

</div>

<!-- RELATED:END -->
