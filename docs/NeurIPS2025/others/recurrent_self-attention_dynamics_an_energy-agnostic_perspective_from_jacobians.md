---
title: >-
  [Paper Note] Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians
description: >-
  [NeurIPS 2025][Self-Attention] This paper adopts a dynamical systems perspective grounded in Jacobian analysis to move beyond the symmetry constraints imposed by traditional energy-function frameworks. It reveals the critical role of normalization layers in suppressing the spectral norm and oscillatory components of self-attention, identifies that high-performing recurrent self-attention models exhibit Lyapunov exponents approaching zero (a criticality regime)…
tags:
  - "NeurIPS 2025"
  - "Self-Attention"
  - "Jacobian Matrix"
  - "Lyapunov Exponents"
  - "Normalization Layers"
  - "Recurrent Architectures"
date: 2026-05-08
content_hash: 2506c8a2d314f4db
---

# Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians

**Conference**: NeurIPS 2025
**arXiv**: [2505.19458](https://arxiv.org/abs/2505.19458)  
**Code**: Unavailable  
**Area**: Deep Learning Theory / Transformer Dynamics Analysis
**Keywords**: Self-Attention, Jacobian Matrix, Lyapunov Exponents, Normalization Layers, Recurrent Architectures

## TL;DR

This paper adopts a dynamical systems perspective grounded in Jacobian analysis to move beyond the symmetry constraints imposed by traditional energy-function frameworks. It reveals the critical role of normalization layers in suppressing the spectral norm and oscillatory components of self-attention, identifies that high-performing recurrent self-attention models exhibit Lyapunov exponents approaching zero (a criticality regime), and proposes a spectral regularization method that substantially improves inference performance.

## Background & Motivation

Theoretical understanding of self-attention (SA) has largely followed an "energy function" paradigm, modeling SA dynamics as a process that monotonically decreases some energy function to guarantee convergence. However, such analyses rest on stringent idealized assumptions:

**Weight symmetry constraint**: Requires $W^Q W^{K\top} = W^V = W^{V\top}$

**Single-head constraint**: Applicable only to single-head attention

**Continuous-time limit**: Requires reformulation as a continuous ODE

These assumptions are far removed from practical Transformer architectures featuring multi-head attention, discrete updates, and normalization layers. More importantly, recent empirical findings reveal **oscillatory dynamics** (non-stationary behavior) in recurrent architectures such as AKOrN—phenomena that energy-function frameworks fundamentally cannot explain, since monotonically decreasing energy can only describe convergence to fixed points.

The authors therefore propose moving beyond the energy-function paradigm and adopting the more general **Jacobian matrix analysis** (Lyapunov indirect method), which subsumes the behaviors describable by energy functions while also capturing richer dynamics such as oscillations, thereby providing a new perspective for understanding practical SA architectures.

## Method

### Overall Architecture

The work proceeds in three progressive stages: (1) relaxing the constraints of the energy-function framework; (2) establishing a general Jacobian-based analysis framework; (3) translating Jacobian insights into practical improvements (regularization and pseudo-energy interpretation). Experiments are conducted primarily on two recurrent SA architectures: AKOrN (Kuramoto oscillators + SA) and the authors' proposed ItrSA (a simplified recurrent SA).

### Key Designs

1. **Relaxation of the Energy-Function Framework (Propositions 4.1 & 4.2)**: The traditional symmetric weight constraint $W^Q W^{K\top} = W^V$ is relaxed to $W^V = (W^K W^{Q\top} + W^Q W^{K\top}) / 2$, i.e., $W^V$ need only be the symmetric part of $W^K W^{Q\top}$. This is further extended to the multi-head setting: provided that $W_h^Q W_h^{K\top}$ admits a low-rank structure (via orthogonal matrix decomposition) and $W_h^V$ remains symmetric, a multi-head energy function can be constructed. Empirically, however, this energy regularization degrades performance by forcing overly convergent dynamics, suggesting that **high-performing SA in practice relies on dynamics richer than energy minimization**.

2. **Jacobian Spectral Analysis and the Critical Role of Normalization (Proposition 5.1)**: For the ItrSA update rule $X^{(t+1)} = \text{RMSNorm}(X^{(t)} + \eta \Delta X^{(t)})$, the following upper bound on the Jacobian spectral norm is derived:

$$\left\| \frac{\partial \text{RMSNorm}(X + \eta \Delta X)}{\partial X} \right\|_2 \leq \frac{\max_j(|\gamma_j|)}{R} (1 + \eta \|J_{\text{MSA}}(X)\|_2)$$

   where $R$ is a lower bound on the post-normalization norm and $\gamma_j$ are trainable RMSNorm scale parameters. The key insight is that normalization suppresses the spectral norm via the $1/R$ factor, preventing signal explosion in recurrent architectures. Even as the step size $\eta \to \infty$, the Jacobian norm remains $O(1)$. Moreover, normalization effectively **suppresses oscillatory components** by pulling eigenvalues of discretized skew-symmetric matrices from outside the unit circle back within it.

3. **Lyapunov Exponents and the Criticality Regime**: Lyapunov exponents measure the exponential rate of local convergence or divergence of trajectories, corresponding to the time-averaged logarithm of the singular values of the Jacobian. The authors find that:

    - High-performing models exhibit maximum Lyapunov exponents approaching zero (~0.1), placing them at the **edge of chaos** in a criticality regime.
    - Symmetrized SA models with energy constraints yield negative Lyapunov exponents (convergent regime) but inferior performance.
    - Multi-head attention tends to increase Lyapunov exponents, supporting more dynamic states.

   This indicates that optimal inference dynamics are neither stably convergent nor unstably divergent, but reside at the boundary between the two.

### Loss & Training

- Base training employs standard cross-entropy loss.
- **Spectral regularization**: $R_{\text{Spec}} = \sum_W (\sigma^2(W) - 1)^2 + \sum_b \|b\|_2^4$, encouraging the largest singular value of each weight matrix to remain close to 1.
- AKOrN is evaluated with oscillator dimensions $N \in \{4, 8, 512\}$; ItrSA does not partition oscillators.
- Training uses $T=16$ recurrent iterations; the number of iterations can be increased at test time to enable test-time scaling.

## Key Experimental Results

### Main Results (Sudoku Task Accuracy)

| Model | ID (SATNet) T=16 | OOD (RRN) T=16 | OOD T=64 | Test-time scaling |
|------|------------------|----------------|----------|-------------------|
| ItrSA | ~98% | ~75% | ~85% | ✓ (consistent improvement) |
| AKOrN (N=4) | ~97% | ~70% | ~60% (degraded) | ✗ (fails at large N) |
| AKOrN + RMSNorm | ~98% | ~75% | ~80% | ✓ (restored) |
| Symmetric SA (energy constraint) | ~85% | ~50% | ~55% | Partial |

### Ablation Study on Regularization

| Regularization | ItrSA OOD | AKOrN OOD | Effect on Lyapunov Exponents |
|-----------|-----------|-----------|------------------|
| None | ~75% | ~70% | Baseline |
| E-single (single-head energy) | Training failure | — | Over-convergent |
| E-multi (multi-head energy) | Below baseline | — | More negative, detrimental |
| Spec (spectral regularization) | Improved | **Significantly improved** | Closer to zero |

### Key Findings

1. **Normalization is essential for recurrent SA**: Without normalization, the Jacobian spectral norm of SA grows with the number of tokens, leading to instability; with normalization it remains $O(1)$.
2. **Energy regularization is ineffective**: Forcing monotonically convergent dynamics degrades performance, indicating that rich dynamics (including oscillations) are necessary in practice.
3. **Criticality is strongly correlated with high performance**: Models with maximum Lyapunov exponents ~0.1 achieve the best performance.
4. **ItrSA also exhibits test-time scaling**: A property previously attributed exclusively to AKOrN is in fact due to normalization rather than the oscillator design.
5. **Jacobian interpretation of pseudo-energy**: AKOrN's pseudo-energy $E_{\text{pseudo}} = -\text{Tr}(X^{(t)\top}Y^{(t)})$ can be approximated as a quadratic form of the symmetric part of the Jacobian; its decrease reflects the alignment of states toward the eigenspace of the largest eigenvalue (analogous to power iteration).

## Highlights & Insights

- **Paradigm shift**: Transitioning from energy functions (Lyapunov direct method) to Jacobian analysis (Lyapunov indirect method) substantially broadens the class of SA architectures amenable to theoretical analysis.
- **Strong consistency between empirical findings and theoretical analysis**: Normalization suppresses spectral norm → criticality → high performance, forming a coherent causal chain.
- **Elegant Jacobian interpretation of pseudo-energy**: Recurrent inference is essentially a constrained power iteration that progressively aligns states with the leading eigendirection of the Jacobian.

## Limitations & Future Work

- Experiments are confined to recurrent SA (without positional encodings, masking, or MLP blocks), remaining distant from practical Transformer architectures.
- The upper bound in Proposition 5.1 is excessively loose (far larger than empirically observed values); tighter theoretical bounds are needed.
- The concentration of Lyapunov exponents near zero and the validity of the Jacobian approximation lack rigorous theoretical proofs.
- The analysis has not been extended to non-recurrent (single-forward-pass) standard Transformers, which is a natural direction for future work.

## Related Work & Insights

- This work aligns with research on edge-of-chaos theory in RNNs and deep networks, but represents the first systematic application to SA.
- AKOrN's Kuramoto model provides intuitive motivation; however, this paper demonstrates that **normalization, not the oscillator design, is the decisive factor**.
- Practical implication: test-time scaling in recurrent Transformers may not require complex oscillator designs; a simple "SA + RMSNorm + input injection" architecture may suffice.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The Jacobian/Lyapunov exponent perspective on SA dynamics is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Theory and experiments are tightly integrated, though the task scope is narrow (primarily Sudoku).
- Writing Quality: ⭐⭐⭐⭐ Well-structured with a smooth logical progression from energy functions to Jacobian analysis.
- Value: ⭐⭐⭐⭐ Provides an important dynamical systems perspective for understanding recurrent Transformers and test-time scaling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Normalization in Attention Dynamics](normalization_in_attention_dynamics.md)
- [\[ICLR 2026\] Neural Dynamics Self-Attention for Spiking Transformers](../../ICLR2026/others/neural_dynamics_self-attention_for_spiking_transformers.md)
- [\[NeurIPS 2025\] Revisiting Bi-Linear State Transitions in Recurrent Neural Networks](revisiting_bi-linear_state_transitions_in_recurrent_neural_networks.md)
- [\[NeurIPS 2025\] Learning Dynamics of RNNs in Closed-Loop Environments](learning_dynamics_of_rnns_in_closed-loop_environments.md)
- [\[NeurIPS 2025\] InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras](influx_a_benchmark_for_self-calibration_of_dynamic_intrinsics_of_video_cameras.md)

</div>

<!-- RELATED:END -->
