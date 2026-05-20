---
title: >-
  [Paper Note] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention
description: >-
  [NeurIPS 2025][LLM Evaluation][Mamba] This work unrolls selective SSMs (Mamba) into an attention-equivalent form and derives generalization bounds via covering number techniques…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "Mamba"
  - "state space models"
  - "generalization bounds"
  - "covering numbers"
  - "spectral abscissa"
date: 2026-05-08
content_hash: 15e211ec5b886aea
---

# Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention

**Conference**: NeurIPS 2025
**arXiv**: [2502.01473](https://arxiv.org/abs/2502.01473)  
**Code**: [https://github.com/Arya-Honarpisheh/gen_err_sel_ssm](https://github.com/Arya-Honarpisheh/gen_err_sel_ssm)  
**Area**: Theory / SSM
**Keywords**: Mamba, state space models, generalization bounds, covering numbers, spectral abscissa

## TL;DR
This work unrolls selective SSMs (Mamba) into an attention-equivalent form and derives generalization bounds via covering number techniques, controlled by the spectral abscissa $s_{\mathbf{A}}$ of the continuous-time state matrix. When $s_{\mathbf{A}} < 0$, the bound is independent of sequence length; when $s_{\mathbf{A}} \geq 0$, it grows exponentially. The paper further proves this dependence is irreducible.

## Background & Motivation

**Background**: Selective SSMs such as Mamba are competitive with Transformers on a variety of sequence tasks, yet a rigorous theoretical generalization analysis remains absent.

**Limitations of Prior Work**: Generalization theory for LTI SSMs relies on control-theoretic tools (impulse response $\ell_1$ norm, transfer function $H_2$ norm), which do not apply to the nonlinear, input-dependent dynamics of selective SSMs. Meanwhile, covering number theory for Transformers is relatively mature but cannot be directly applied to the recurrent structure of SSMs.

**Key Challenge**: Selective SSMs combine the recurrent structure of RNNs (requiring control over state matrix growth) with the input-dependent projections of attention ($W_B, W_C$ analogous to key-query), necessitating a unified analytical framework.

**Key Insight**: The selective SSM recurrence is unrolled into an attention-like form, enabling a two-level covering construction — state matrices are covered using RNN-style tools, while input projections are covered using Transformer-style tools.

**Core Idea**: The spectral abscissa $s_A$ of the continuous-time state matrix determines whether the generalization bound is independent of sequence length.

## Method

### Overall Architecture
Rademacher complexity upper bounds are derived via covering number techniques, yielding generalization error bounds. The central challenge is constructing an effective $\varepsilon$-cover for the parameter space of selective SSMs.

### Key Designs

1. **SSM → Attention Unrolling**:

    - **Function**: Unrolls Mamba's recurrent computation $y[t'] = C[t'] \sum_{t=0}^{t'-1} A^t \Delta[t'-1-t] B[t'-1-t] u[t'-1-t]$ into an attention-like form.
    - **Mechanism**: $W_C$ corresponds to the Query projection, $W_B$ to the Key projection, and $u$ itself serves as the Value. $$z = w^\top \sum_{t=0}^{T-1} \underbrace{(I_d \otimes u[T]^\top W_C^\top)}_{\text{Query}} \underbrace{(I_d \otimes W_B u[T-1-t])}_{\text{Key}} \underbrace{u[T-1-t]}_{\text{Value}}$$
    - **Design Motivation**: This unrolling allows $W_B$ and $W_C$ to be handled using linear function class covering techniques from Transformer generalization theory.

2. **Two-Level Covering Construction (Core Technical Contribution)**:

    - **First level (state matrix $A_c$)**: The Gelfand formula controls $\|A^t\|_2 \leq \rho_A^t$, where $\rho_A = (1+e^{p-\mathfrak{B}_q\mathfrak{B}_u})^{s_A+\eta}$. When $s_A < 0$, $\rho_A < 1$ and the convergent geometric series guarantees length independence.
    - **Second level (input projections $W_B, W_C, q, w$)**: Treated as bounded linear function classes under the $\|\cdot\|_{1,1}$ norm, covered directly via covering lemmas from Transformer theory.
    - Individual parameter covers are combined via Cartesian products, with covering radii allocated optimally.

3. **Main Theorem (Thm 3.3)**:

    - The capacity term in the generalization bound is $\mathcal{C}_{\mathcal{F}_{SSM}} = \tilde{O}(\mathfrak{M}_\Delta \mathfrak{B}_w \mathfrak{B}_u^3 \mathfrak{B}_B \mathfrak{B}_C \mathfrak{B}_A S_2 (\cdot)^{3/2})$.
    - The key quantity is $S_2 = \frac{\rho_A(1-\rho_A^T)}{(1-\rho_A)^2} - \frac{T\rho_A^T}{\rho_A - 1}$.
    - When $s_A < 0$: $\rho_A < 1$, $S_2$ is bounded, and the generalization bound is **independent of sequence length $T$**.
    - When $s_A > 0$: $\rho_A > 1$, $S_2 \sim T\rho_A^T$, and the generalization bound **grows exponentially**.

4. **Lower Bound (Thm 4.1)**:

    - When $s_A > 0$, the Rademacher complexity lower bound satisfies $\geq \mathfrak{B}_w \frac{(1+s_A)^T - 1}{s_A}\sqrt{\frac{2}{\pi m}}$.
    - This proves that the $T$-dependence cannot be eliminated by tighter upper bounds — it is fundamental.

### Comparison of Generalization Bounds Across Architectures

| Model | $T$ dependence | $d$ dependence | $\mathfrak{B}_u$ dependence |
|------|--------|--------|---------------------|
| Selective SSM ($s_A<0$) | **1** | $d^{1/2}$ | $\mathfrak{B}_u^4$ |
| Selective SSM ($s_A\geq 0$) | $T\rho_A^T$ | $d^{1/2}$ | $\mathfrak{B}_u^4$ |
| Linear Attention | T | 1 | $\mathfrak{B}_u^3$ |
| Softmax Attention | **1** | 1 | $\mathfrak{B}_u^3$ |
| Vanilla RNN ($\mathfrak{l}_x\|A\|_2<1$) | **1** | d | $\mathfrak{B}_u$ |

## Key Experimental Results

### Experiment 1: Unstable Initialization ($s_A = 0.1$)

| Task | Short sequences (small $T$) | Long sequences (large $T$) |
|------|------------|------------|
| Majority | Training succeeds; $s_A$ driven toward 0 | Training fails; loss grows exponentially |
| IMDb | Similar | Similar |
| ListOps | Similar | Similar |

Key observation: Whenever training successfully reduces the loss, $s_A$ is driven toward negative values — the model spontaneously learns to stabilize itself.

### Experiment 2: Stable Initialization ($s_A = 0$)
- Majority: Generalization gap remains stable across $T=50$ to $T=500$, confirming length independence.
- IMDb: Gap stabilizes after $T > 300$ (consistent with average review length ~300).
- ListOps: Consistent generalization gap.

### Key Findings
- **Training implicitly drives stabilization**: Even when initialized as unstable, successful training is invariably accompanied by $s_A \to 0^-$.
- **Long sequences + instability = training catastrophe**: Training failure under unstable initialization on long sequences is not incidental, but an inevitable consequence of the exponentially growing generalization bound.
- **Generalization advantage of stable SSMs**: Length-independent generalization is achieved on par with softmax attention, and is superior to linear attention (which scales linearly with $T$).

## Highlights & Insights
- **Elegant SSM–Attention bridge**: Unrolling SSMs into an attention-like form is not merely a technical device — it reveals a fundamental structural similarity between the two model families.
- **Matching upper and lower bounds**: Both bounds exhibit exponential growth when $s_A > 0$, differing only by an $O(T)$ factor, indicating high-quality bounds.
- **Clear practical guidance**: $s_A$ must be kept negative — a direct prescription for Mamba initialization and regularization strategies.
- **Clever application of the Gelfand formula**: Replacing the operator norm with the spectral radius yields tighter control than the $\|A\|_2$-based approach used in classical RNN theory.

## Limitations & Future Work
- Only a single-layer SSM block is analyzed; generalizing to multi-layer or deep SSMs is a natural next step.
- An $O(T)$ gap between upper and lower bounds remains, which may be narrowed through more refined covering constructions.
- Empirical validation is conducted on relatively simple tasks (synthetic Majority, IMDb binary classification); more complex generative tasks warrant further investigation.
- The simplified SSM structure of Mamba-2 (viewed through the multi-head attention lens) is not considered, and may admit tighter bounds.

## Related Work & Insights
- **vs. Rácz et al. (2024)**: Their LTI SSM bounds rely on impulse response norms and do not extend to nonlinear selective SSMs.
- **vs. Trauger & Tewari (2023)**: Their length-independent Transformer bounds serve as the theoretical foundation for the $W_B, W_C$ covering in this work.
- **vs. RNN generalization theory**: RNNs avoid exponential growth via bounded activation functions, a mechanism absent in selective SSMs — stability is the only viable path.
- The results have direct implications for Mamba pretraining: $A_c$ should be initialized to ensure $s_A < 0$.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneering work in SSM generalization theory; the SSM–Attention bridge analysis is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three tasks validate theoretical predictions, though task complexity is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically rigorous, with clear intuitive explanations and transparent proof structure.
- Value: ⭐⭐⭐⭐⭐ Provides concrete theoretical guidance for SSM design (maintain stability) and fills an important gap in the literature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MVSMamba: Multi-View Stereo with State Space Model](mvsmamba_multi-view_stereo_with_state_space_model.md)
- [\[NeurIPS 2025\] What Does It Take to Build a Performant Selective Classifier?](what_does_it_take_to_build_a_performant_selective_classifier.md)
- [\[NeurIPS 2025\] Aggregation Hides OOD Generalization Failures from Spurious Correlations](aggregation_hides_out-of-distribution_generalization_failures_from_spurious_corr.md)
- [\[NeurIPS 2025\] PaTH Attention: Position Encoding via Accumulating Householder Transformations](path_attention_position_encoding_via_accumulating_householder_transformations.md)
- [\[ICLR 2026\] Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](../../ICLR2026/llm_evaluation/talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)

</div>

<!-- RELATED:END -->
