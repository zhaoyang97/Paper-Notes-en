---
title: >-
  [Paper Note] NRGPT: An Energy-based Alternative for GPT
description: >-
  [ICLR 2026][Optimization][Energy-based models] This paper proposes NRGPT (eNeRgy-GPT), which applies minimal modifications to standard GPT to yield an energy-based model: attention and feedforward energy functions are designed such that each forward pass is equivalent to a gradient descent step on the energy landscape. The work proves asymptotic energy decrease and stable convergence properties, and validates performance comparable to standard GPT on ListOps, Shakespeare, and OpenWebText.
tags:
  - ICLR 2026
  - Optimization
  - Energy-based models
  - GPT
  - autoregressive
  - gradient descent inference
  - asymptotic stability
date: 2026-05-08
content_hash: fff714875a1c0f30
---

# NRGPT: An Energy-based Alternative for GPT

**Conference**: ICLR 2026
**arXiv**: [2512.16762](https://arxiv.org/abs/2512.16762)
**Code**: None
**Area**: Optimization
**Keywords**: Energy-based models, GPT, autoregressive, gradient descent inference, asymptotic stability

## TL;DR

This paper proposes NRGPT (eNeRgy-GPT), which applies minimal modifications to standard GPT to yield an energy-based model: attention and feedforward energy functions are designed such that each forward pass is equivalent to a gradient descent step on the energy landscape. The work proves asymptotic energy decrease and stable convergence properties, and validates performance comparable to standard GPT on ListOps, Shakespeare, and OpenWebText.

## Background & Motivation

**State of the Field**: The GPT architecture is the dominant paradigm for autoregressive language modeling, generating text via next-token prediction. Energy-based models (EBMs) represent another important paradigm, treating inference as a dynamical process on an energy landscape—low energy corresponds to plausible samples and high energy to anomalous ones. Although the two appear fundamentally different, a growing body of research suggests deep connections between them.

**Limitations of Prior Work**:
1. **Unclear relationship between GPT and EBMs**: Von Oswald et al. showed that in-context learning (ICL) may implement gradient descent, but only for linear Transformers (without softmax), an oversimplification.
2. **Energy Transformer does not apply to GPT settings**: ET is designed for BERT-like masked completion—masked tokens evolve rapidly to match missing positions—whereas GPT has no masked tokens; each token must evolve to become the next token in the sequence.
3. **Existing EBM-for-LLM work**: Methods such as EBT place energy computation at the output of a standard Transformer forward pass rather than treating the forward pass itself as an energy optimization process.
4. **Lack of a theoretical framework that directly transforms GPT forward passes into energy landscape exploration**.

**Root Cause**: How can the inference process of GPT acquire the theoretical advantages of EBMs—interpretability, systematic solution-space exploration, and a natural alignment mechanism—without altering the training paradigm (self-supervised next-token prediction)?

**Paper Goals**: Apply minimal modifications to a parallel Transformer (GPT-J style) by making the attention and feedforward networks the gradients of two separate energy functions, so that each forward pass becomes a single step of energy gradient descent.

## Method

### Overall Architecture

NRGPT adopts a **weight-sharing recurrent architecture**: a single module is applied repeatedly $T$ times, replacing the conventional $T$-layer Transformer with distinct weights per layer. Each application corresponds to one step of energy gradient descent:

$$x^{(t+1)} = x^{(t)} - \eta^{(t)} \frac{\partial E}{\partial g^{(t)}}$$

where $g^{(t)} = \text{LN}(x^{(t)})$ is the token representation after LayerNorm/RMSNorm, and $\eta$ is the inference rate matrix.

### Key Design 1: Dual Energy Functions

**Attention energy** (derived from Dense Associative Memory):

$$E_A^{\text{AT}} = -\frac{1}{\beta} \sum_h \alpha_h \log \left[ \sum_{B<A} \exp(\beta \cdot g_B^\top J_h g_A) \right]$$

where $J_h = [W^K_h]^\top W^Q_h$ merges the Key and Query projections, and $\alpha_h$ is a learnable head weight. The update obtained by taking the gradient with respect to $g_A$ closely mirrors the structure of standard multi-head attention:

$$\text{Original: } [W^P_h]^\top W^V_h \equiv \text{Energy: } \alpha_h \eta J_h^\top$$

**Feedforward energy** (two variants):

- FF1: $E^{\text{FF}} = -\|\sigma(Wg_A)\|^2$, whose gradient yields a single-weight-matrix feedforward update.
- FF2W: $E^{\text{FF}} = -g_A^\top W^2 \sigma(W^1 g_A)$, whose gradient yields a dual-weight-matrix feedforward update (closer to a standard MLP).

### Key Design 2: Energy Decrease Guarantee and Asymptotic Stability

**Energy decrease condition** (Proposition 2.1): When the inference rate satisfies $\eta = c \cdot \text{diag}(\gamma)$ ($c > 0$, $\gamma$ from LayerNorm), the update rule guarantees asymptotic energy decrease $\dot{E}_A < 0$.

**Asymptotic stability** (exploiting a key property of causal attention masking):
- The energy $E_A$ of token $A$ depends only on the states of tokens $B \leq A$.
- The energy of the first token decreases monotonically and is bounded below → converges to a fixed point.
- Once the first token stabilizes, the energy of the second token also decreases monotonically → recursive argument applies.
- **All tokens ultimately converge asymptotically to a stable state**—the distinctive "asymptotic stability" phenomenon of NRGPT.

**Without LayerNorm** (Proposition 2.2): When $g = x$, it suffices for the symmetric part $\eta_+ = (\eta + \eta^\top)/2$ to be positive semi-definite to guarantee $\dot{E} < 0$; the antisymmetric part $\eta_-$ is unconstrained.

### Key Design 3: Structural Correspondence with Standard Transformers

| Module | Standard Transformer | NRGPT Energy Gradient |
|:---|:---|:---|
| Attention output matrix | $[W^P]^\top W^V$ | $\alpha \eta J^\top$ |
| Feedforward second-layer weights | $W^2$ | $W^1 \eta^\top$ |
| Inter-layer connection | Distinct weights per layer | Weight sharing + recurrent application |
| Propagation mechanism | Layer-by-layer | Gradient descent on energy landscape |

## Key Experimental Results

### Main Results: ListOps Nested Mathematical Operations

Three nested operations are tested: maximum, median, and sum modulo 20.

| Model | Learning transition point (# params) | Final accuracy |
|:---|:---:|:---:|
| GPT_Rec_parallel | 2.3×10⁴ | ~100% |
| NRGPT_H_FF1 | 2.4×10⁴ | ~100% |
| NRGPT_H_FF2W | 2.98×10⁴ | ~100% |

All NRGPT variants match baseline performance on ListOps, with learning transition points very close to those of the baseline.

### OpenWebText Language Modeling

| Model | # Params | Val Loss (mean±std) | Val Loss (min) |
|:---|:---:|:---:|:---:|
| GPT (12 layers) | 124M | 2.921±0.005 | 2.915 |
| GPT_Rec_parallel | 85M | 3.454±0.037 | 3.411 |
| NRGPT_H_FF2W | 90M | 3.467±0.073 | 3.404 |

**Key Findings**: NRGPT achieves validation loss comparable to the recurrent GPT baseline (3.404 vs. 3.411) with 34M fewer parameters.

### Ablation Study: Resistance to Overfitting

| Model | Shakespeare Train Loss | Val Loss | Overfitting degree |
|:---|:---:|:---:|:---:|
| GPT | Extremely low | Relatively high | Severe (large model) |
| GPT_Rec_parallel | Relatively low | Relatively high | Moderate |
| **NRGPT** | Moderate | **Comparable to train** | **Mild** |

On the Shakespeare dataset, NRGPT exhibits notably reduced overfitting at large parameter counts—its best validation loss is comparable to the GPT baseline, yet training-set overfitting is substantially lower. This may stem from the natural regularization effect of gradient descent on the energy landscape.

## Rating

### Highlights & Insights

1. **Theoretical elegance**: The paper establishes a rigorous connection between GPT and EBMs through minimal modifications. The proofs of energy decrease and asymptotic stability exploit the causal mask structure in a particularly elegant manner.
2. **Opens a new direction**: Treating inference as energy optimization offers a novel perspective on LLMs, with potential applications in alignment (via energy regularization) and interpretability (via energy landscape analysis).
3. **The anti-overfitting phenomenon** is interesting and practically valuable.

### Limitations & Future Work

1. Validation is limited to 124M parameters, far below the scale of modern LLMs; scalability remains unclear.
2. The weight-sharing constraint reduces parameter efficiency relative to standard GPT, requiring more recurrent steps to compensate.
3. The constraint on the inference rate $\eta$ (i.e., $\eta = c \cdot \text{diag}(\gamma)$) is relatively strong and limits model expressiveness.
4. Validation loss still lags behind standard GPT (3.404 vs. 2.915), despite the difference in parameter count.

### Rating

⭐⭐⭐⭐

**Justification**: This work establishes the tightest theoretical connection to date between GPT and EBMs; the proof of asymptotic stability is particularly impressive. Although the experimental scale is currently limited, the theoretical contributions are sufficient to open new research directions—specifically, how to leverage explicit optimization of the energy landscape to improve LLM alignment, robustness, and interpretability.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] When to Restart? Exploring Escalating Restarts on Convergence](when_to_restart_exploring_escalating_restarts_on_convergence.md)
- [\[ICLR 2026\] Adaptive Rollout Allocation for Online RL with Verifiable Rewards (VIP)](adaptive_rollout_allocation_for_online_reinforcement_learning_with_verifiable_re.md)
- [\[ICLR 2026\] Converge Faster, Talk Less: Hessian-Informed Federated Zeroth-Order Optimization](converge_faster_talk_less_hessian-informed_federated_zeroth-order_optimization.md)
- [\[ICLR 2026\] Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures](saddle-to-saddle_dynamics_explains_a_simplicity_bias_across_neural_network_archi.md)
- [\[ICLR 2026\] Learning to Solve Orienteering Problem with Time Windows and Variable Profits](learning_to_solve_orienteering_problem_with_time_windows_and_variable_profits.md)

<!-- RELATED:END -->
