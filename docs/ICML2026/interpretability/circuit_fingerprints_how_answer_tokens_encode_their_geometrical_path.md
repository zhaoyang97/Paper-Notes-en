---
title: >-
  [Paper Note] Circuit Fingerprints: How Answer Tokens Encode Their Geometrical Path
description: >-
  [ICML 2026][Interpretability][Circuit Discovery] This paper proposes the Circuit Fingerprint hypothesis—feeding an answer token independently into a Transformer leaves a directional trace in the latent space that precise…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Circuit Discovery"
  - "Activation Steering"
  - "Geometric Alignment"
  - "Answer Token Fingerprint"
  - "Shapley Decomposition"
date: 2026-05-08
content_hash: 8174b9d1b8c57ebc
---

# Circuit Fingerprints: How Answer Tokens Encode Their Geometrical Path

**Conference**: ICML 2026  
**arXiv**: [2602.09784](https://arxiv.org/abs/2602.09784)  
**Code**: Not explicitly released  
**Area**: Interpretability / Mechanistic Interpretability / Activation Steering  
**Keywords**: Circuit Discovery, Activation Steering, Geometric Alignment, Answer Token Fingerprint, Shapley Decomposition

## TL;DR
This paper proposes the Circuit Fingerprint hypothesis—feeding an answer token independently into a Transformer leaves a directional trace in the latent space that precisely follows the circuit path required to generate that answer. Based on this, it achieves circuit discovery through pure geometric alignment (without gradients/intervention) and demonstrates that the same set of directions can perform activation steering, proving that "reading" and "writing" are two sides of the same geometric object.

## Background & Motivation
**Background**: Mechanistic interpretability currently follows two primary paths: (i) circuit discovery using activation patching or gradient approximations (EAP/EAP-IG) to identify task-critical attention head/MLP sub-networks; (ii) activation steering adding learned directions to the residual stream to control model behavior. Though both operate on the same representation space, they remain largely independent.

**Limitations of Prior Work**: Patching methods require $O(LH)$ forward passes; gradient methods (attribution patching, EAP-IG) vary in accuracy due to saturation and LayerNorm non-linearity; mask-learning methods (ACDC, edge pruning) require iterative optimization. Steering methods require collecting contrastive data, learning directions, and tuning intervention strength, while the questions of "where to intervene" and "what direction to use" are treated as decoupled problems.

**Key Challenge**: If circuits are stably encoded in model weights, then discovery and steering should fundamentally operate on the same object—yet existing methods treat them as two sets of non-communicating tools. The Linear Representation Hypothesis (Park 2024, Elhage 2022) suggests a unified geometric perspective, but it has never been used to explain both tasks simultaneously.

**Goal**: To use a single geometric principle to simultaneously answer: (i) which components belong to the circuit (read); (ii) how to intervene in these components to change the output (write); all without relying on gradients or interventions, using only pure forward projections.

**Key Insight**: Feeding an answer token (e.g., "Paris") independently into the model—despite lacking context—activates the fixed-weight circuit (e.g., capital-city recall) because circuits are encoded in weights. Thus, $\Delta r^{(L)}=r_{a_+}^{(L)}-r_{a_-}^{(L)}$ naturally becomes the geometric "signature" of the direction required to produce $a_+$ vs $a_-$.

**Core Idea**: Circuit membership = degree of alignment between component output and the "answer token differential direction." Using the same direction for steering is the "write" operation; reading and writing are dualities of the same set of directions.

## Method

### Overall Architecture
The framework utilizes two input sources: (1) Answer token pairs $(a_+, a_-)$—processed through independent forward passes to obtain differential directions $\Delta r^{(L)}, \Delta v^{(\ell,h)}, \Delta q^{(\ell,h)}, \Delta k^{(\ell,h)}$; (2) Contrastive prompts (clean vs. corrupted)—to obtain component output differentials $\Delta o_c$. Read path: project target directions into the component's native space and calculate the inner product with $\Delta o_c$ for node importance $S_c$; compute edge importance using a three-channel Q/K/V Shapley decomposition + residual stream decomposition. Write path: at heads identified during the read phase, replace or superimpose activations using directions $\hat d_s, \hat d_t$ from the same space.

### Key Designs

1. **Answer Token Differentials as Geometric Targets + Native Space Projection**:
    - **Function**: Obtains target directions $\Delta r^{(L)}, \Delta v, \Delta q, \Delta k$ for each layer and head using only two forward passes (feeding $a_+$ and $a_-$), without training or gradients.
    - **Mechanism**: For component $c$ (using $W_O$ for attention heads, $W_{\text{out}}$ for MLPs), the target direction is first transformed into the component's native space $\hat t_c=W_c^\top \Delta r^{(L)}/\|\Delta r^{(L)}\|$, then $S_c=\langle \Delta o_c, \hat t_c \rangle$ is calculated. This "native space inner product" ensures $\sum_c S_c$ equals the total projection onto the target direction in the residual stream (preserving additivity).
    - **Design Motivation**: Projecting $\Delta o_c$ directly to the residual stream before calculating the inner product introduces geometric confusion from shared projection matrices like $W_O$; switching to the component's native space decouples "how the component internally generates this direction" from "shared residual stream geometry," yielding a clean, additive importance measure.

2. **Q/K/V Shapley Decomposition + Edge Residual Stream Backpropagation**:
    - **Function**: Separately measures information flow from upstream components to downstream attention heads (edges $i \to j$) across Q/K/V channels, using Shapley values to provide channel weights without arbitrary weighting.
    - **Mechanism**: For each channel (e.g., K), write $R^{(K)}_{i\to j}=\langle \Delta o_i, W^{(j)}_K \Delta k^{(j)}\rangle/\langle\Delta r^{(\ell_j)}, W^{(j)}_K \Delta k^{(j)}\rangle$ (ensuring $\sum_i R^{(K)}_{i\to j}=1$ linearly); treat Q/K/V as a three-player cooperative game, running $2^3=8$ coalitions per head to measure importance and obtain Shapley weights $\phi_Q, \phi_K, \phi_V$; final edge importance is $E_{i\to j}=S_j \cdot (\phi_Q R^{(Q)}_{i\to j}+\phi_K R^{(K)}_{i\to j}+\phi_V R^{(V)}_{i\to j})$; indirect importance is accumulated via "backpropagation" from deep to shallow layers (Alg. 1).
    - **Design Motivation**: Simply spliting a head's total score into three channels is arbitrary; Shapley values are the **only** distribution satisfying fairness axioms in cooperative games. Furthermore, Shapley values inherently guarantee $\phi_Q+\phi_K+\phi_V=S_{QKV}-S_\emptyset$, ensuring importance additivity down to the edge level. Fig. 4 empirically finds Name Mover heads are Q-dominated and S-Inhibition heads are K-dominated, perfectly matching roles assigned manually in Wang 2022.

3. **Geometric Steering: Read Directions as Write Operations**:
    - **Function**: Uses the same heads and answer directions found in the read phase for intervention to control generation, validating read-write duality.
    - **Mechanism**: After centering answer prototypes $\{r_1, \dots, r_k\}$, SVD is used to obtain an orthogonal basis $\{u_i\}$. Source/target prototypes are projected onto this basis to get $d_s, d_t$. Factual recall tasks use substitution $X'=X-\|d_s-d_t\|\hat d_s+\|d_s-d_t\|\hat d_t$; stylistic tasks (emotion, language) use magnitude transfer $X'=X-\|d_s\|(\hat d_s-\hat d_t)$.
    - **Design Motivation**: Comparing against activation patching (copying corrupted activations)—the latter serves as an upper bound for steering. If geometric directions can approximate patching effects, it proves that fingerprints are true causal structures rather than surface correlations. Experiments on IOI show that at $\alpha=1$, $P(\text{correct})=0.014$ vs. 0.0 for patching, with a logit diff of $-4.07$ vs. $-7.34$, indicating comparable behavioral effects.

### Loss & Training
**Completely training-free, gradient-free, and intervention-free**. Requires only 2 forward passes ($a_+$ and $a_-$) to obtain target directions, plus forward passes on contrastive prompts to calculate $\Delta o_c$. Shapley decomposition requires 8 coalition evaluations which can be batched. Computational budget is comparable to EAP (single backward pass).

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

CF is largely on par with gradient methods on IOI/SVA and fully comparable to EAP; it is slightly weaker on MCQA.

### Steering Results

| Metric | Baseline (instruction prompting) | CF Steered |
|--------|----------------------------------|-----------|
| Emotion Classification Accuracy | 53.1% | **69.8%** |
| Perplexity (median) | 17.03 | **13.37** |
| Factual Accuracy | 90.1% | 89.6% |

Steering for positive sentiment (joy) maintains/improves factual accuracy (100%), but negative sentiment (sadness 81%, disgust 78%) leads to "emotional resonance phoneme contamination" of name recall (e.g., Einstein modified to "Sissoar").

### Key Findings
- CMD/CPR results are competitive with gradient baselines; as model size increases, geometric methods converge toward EAP-IG, which the authors attribute to "better concept decoupling in larger models."
- Shapley decomposition exposes functional roles: Name Mover heads are Q-dominated, while S-Inhibition heads are K-dominated, aligning with manual classifications in IOI literature.
- The same set of directions works for both read and write operations (patching upper bounds overlap heavily with CF steering curves), providing strong evidence for read-write duality.
- Persona/emotion instruction prefixes from prompt engineering can also be used to extract directions, proving the fingerprint method generalizes to any "attribute controllable via prompt modification."

## Highlights & Insights
- **The claim that "answer tokens carry their own circuit fingerprints" is counter-intuitive**: Conventionally, we assume circuits are only activated when producing an answer. This paper reveals that even when an answer token appears as an input token, it traverses the same path (even suppressing incorrect candidates), quantifying the intuition that "circuits are stable structures" into readable directions.
- **Read-write duality = true causal verification**: The identified directions are not just "seemingly important"; they can be used directly for steering to replicate patching effects. This upgrades interpretability from "post-hoc description" to "ex-ante intervention," representing a paradigm shift beyond the "observation-only" nature of SAEs/probes.
- **The idea of Shapley channel decomposition is portable**: Using just $2^3=8$ coalitions for fair allocation replaces "arbitrary weight combinations" with the unique solution from game theory. This approach is fully reusable for other multi-branch modules like MoE or multi-expert fusion.
- **Method requires only 2 forward passes**: Compared to $O(LH)$ iterations or backpropagation in EAP series, CF is computationally near-free and particularly friendly to large models.

## Limitations & Future Work
- Experiments are limited to small models ($\le 1.3B$); validation on 7B+ models is missing. The MCQA CPR for Llama3.2-1B (0.87) was significantly worse than EAP-IG (0.95), suggesting geometric approximations might be insufficient for complex tasks.
- Focus is restricted to the final token position, ignoring indirect effects at earlier positions and LayerNorm non-linearities (explicitly simplified by authors). Edge attribution accuracy in long contexts remains to be verified.
- Zero-shot steering remains fragile: Negative sentiment can contaminate semantics, producing nonsensical words like "Bonniweeper," indicating that some features remain entangled with lexical content.
- Evaluation metrics CMD/CPR are derived from the MIB benchmark; systematic comparisons on other interpretability tasks (feature ablation, faithfulness) have not yet been conducted.

## Related Work & Insights
- **vs. ACDC / Edge Pruning / EAP-IG**: Traditional circuit discovery requires either iterative search/mask learning or gradient backpropagation. CF uses only 2 forward passes + 8 Shapley coalitions, reducing circuit discovery to the same complexity as a single forward pass without gradient dependency.
- **vs. Activation Steering (Turner 2023, Zou 2023)**: Existing steering requires preparing contrastive data before learning directions independently. This paper reuses the same heads and directions from circuit discovery, unifying "where to intervene" and "what direction to use."
- **vs. Linear Representation Hypothesis**: Park 2024 and Elhage 2022 propose that "features are directions in activation space" descriptively. This paper provides operational evidence—these directions describe not just the features, but the circuits that produce them, representing a strong unification of "feature geometry $\equiv$ circuit geometry."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The Circuit Fingerprint hypothesis + read-write duality unifies two independent interpretability research lines into a single geometric object; highly original insight.
- Experimental Thoroughness: ⭐⭐⭐ Covers 4 model families × 3 tasks (IOI/SVA/MCQA) + 5 emotion steering tasks, but model sizes are small and benchmarks are limited.
- Writing Quality: ⭐⭐⭐⭐ Clear conceptual explanations, complete Shapley derivation/algorithm, and honest discussion of limitations.
- Value: ⭐⭐⭐⭐ Provides a gradient-free, low-cost tool for simultaneous discovery and control, offering practical value for alignment and behavioral editing research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](query_circuits_explaining_how_language_models_answer_user_prompts.md)
- [\[ICML 2026\] All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs](all_circuits_lead_to_rome_rethinking_functional_anisotropy_in_circuit_and_sheaf_.md)
- [\[ICLR 2026\] How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding](../../ICLR2026/interpretability/how_do_transformers_learn_to_associate_tokens_gradient_leading_terms_bring_mecha.md)
- [\[ICML 2026\] Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in modern Transformers](dissecting_multimodal_in-context_learning_modality_asymmetries_and_circuit_dynam.md)
- [\[ICML 2026\] How Language Models Process Negation](how_language_models_process_negation.md)

</div>

<!-- RELATED:END -->
