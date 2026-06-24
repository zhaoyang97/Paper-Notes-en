---
title: >-
  [Paper Note] Critical Attention Scaling in Long-Context Transformers
description: >-
  [ICLR 2026][Learning Theory][Attention Scaling] This paper demonstrates, using an analytically simplified attention model, that attention behavior undergoes a **phase transition** as the context length $n$ increases, driven by the scaling factor $\beta_n = \gamma \log n$. The critical point occurs precisely at $\beta_n \asymp \log n$ (specifically $\gamma_c = \tfrac{1}{1-\rho}$), providing the first rigorous theoretical justification for the logarithmic scaling used in method…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Attention Mechanism Theory"
  - "Attention Scaling"
  - "Long Context"
  - "Phase Transition"
  - "Rank Collapse"
  - "Critical Scaling"
date: 2026-05-08
content_hash: 2506f16222264045
---

# Critical Attention Scaling in Long-Context Transformers

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=7SLtElfqCW](https://openreview.net/forum?id=7SLtElfqCW)  
**Code**: None  
**Area**: Learning Theory / Attention Mechanism Theory  
**Keywords**: Attention Scaling, Long Context, Phase Transition, Rank Collapse, Critical Scaling  

## TL;DR
This paper demonstrates, using an analytically simplified attention model, that attention behavior undergoes a **phase transition** as the context length $n$ increases, driven by the scaling factor $\beta_n = \gamma \log n$. The critical point occurs precisely at $\beta_n \asymp \log n$ (specifically $\gamma_c = \tfrac{1}{1-\rho}$), providing the first rigorous theoretical justification for the logarithmic scaling used in methods like YaRN and Qwen.

## Background & Motivation
**Background**: Attention is the cornerstone of modern Transformers and LLMs. A single layer of attention maps a set of tokens $\{x_1, \dots, x_n\} \subset \mathbb R^d$ to a new set through softmax weighting. Recent theoretical works (Dong 2021, Geshkovski 2024/2025, Karagodin 2024, etc.) have identified that attention is essentially a **contraction operator**, pulling tokens closer together until they eventually collapse.

**Limitations of Prior Work**: This phenomenon is known as **rank collapse** or **token homogenization**. Its root cause is that as context length $n$ increases, the softmax attention weight distribution becomes "flattened"—each token spreads its attention uniformly across too many other tokens rather than selectively focusing on a few key ones. This pathology worsens with longer contexts, directly affecting long-context LLMs.

**Key Challenge**: The engineering community has adopted empirical remedies—YaRN, Qwen, SSMax, and SWAN-GPT all use the same simple strategy: multiplying attention scores $a_{ij}$ by a **polylogarithmic factor $\beta_n$ related to context length** (see table below) to counteract the flattening effect. However, these scaling factors are heuristically tuned: YaRN uses $(\log n)^2$, while Qwen/SSMax/SWAN-GPT use $\log n$. A theoretical explanation for the correct **order of magnitude** of $\beta_n$ has been missing.

| Method | $\beta_n$ Scaling |
|------|----------------|
| YaRN | $(\log n)^2$ |
| Qwen | $\log n$ |
| SSMax | $\log n$ |
| SWAN-GPT | $\log n$ |

**Goal**: To answer a clean mathematical question: **What is the optimal order of magnitude for the scaling factor $\beta_n$?**

**Key Insight**: Following the approach of Cowsik et al., the authors construct an extremely simplified yet fully analytical attention model where the effects of the scaling factor are magnified to rigorously characterize phase transition boundaries. The analysis is first performed under an ideal "simplex" configuration and then relaxed to a more realistic "near-simplex" configuration to verify the universality of the conclusions.

**Core Idea**: To prove that this model exhibits a $\beta_n$-driven **phase transition**: if scaling is too small, all tokens collapse into the same direction; if scaling is too large, attention degenerates into an identity map, and tokens no longer interact. The critical value is exactly at $\beta_n \asymp \log n$, where logarithmic scaling allows attention to maintain a healthy "sparse, content-adaptive" state.

## Method

### Overall Architecture
The paper investigates the dynamical behavior of a simplified attention layer with a residual connection as an **operator**. Three key simplifications are made: (1) $K=Q=V=I_d$, removing learnable projections; (2) Pre-layer norm is applied, projecting each token onto the unit sphere $\mathbb S^{d-1}$, denoted as $y_i = N(x_i) = x_i / \lVert x_i \rVert$; (3) Attention scores are computed as normalized dot products $a_{ij} = \beta \langle y_i, y_j \rangle$. The attention update is thus:

$$x_i' = \mathrm{ATT}(y_i) + \alpha x_i, \qquad \mathrm{ATT}(y_i) = \sum_{j=1}^{n} A_{ij} y_j, \qquad A_{ij} = \frac{e^{a_{ij}}}{\sum_{k} e^{a_{ik}}},$$

where the term with $\alpha \ge 0$ represents the residual connection (He 2016), which naturally regularizes the attention map toward the identity.

The metric for "contraction" is the pairwise angle between tokens: if after the update $\langle y_i', y_j' \rangle > \langle y_i, y_j \rangle$ (the angle decreases), the attention is said to be **contractive**. The scaling is defined as $\beta = \gamma \log n$, and the core analysis tracks which phase the token angles and gradients fall into as $n \to \infty$ for different values of $\gamma$. The conclusion identifies a critical $\gamma_c = \tfrac{1}{1-\rho}$ (where $\rho$ is the typical pairwise inner product of tokens), dividing behavior into "sub-critical, critical, and super-critical" phases. Both forward (token representation) and backward (gradient) dynamics undergo synchronized phase transitions at the same threshold.

### Key Designs

**1. Simplex / Near-Simplex Simplified Models: Making Phase Transitions Analytical**

Directly analyzing attention dynamics under general token distributions is almost impossible to solve in closed form. The first step involves introducing highly symmetric configurations that still predict phase transitions. **Simplex Assumption (Assumption 1)**: Constants $q \ge 0, \rho \in (0,1)$ exist such that $\lVert x_i \rVert^2 = q$ and $\langle y_i, y_j \rangle = \rho$ for all $i \ne j$—i.e., tokens are of equal length and equidistant. This allows the softmax denominator $Z = \sum_k e^{a_{ik}}$ to be independent of $i$, reducing the dynamics to scalar limit calculations. The **Near-Simplex Assumption (Assumption 2)** subsequently relaxes this to $q_1 \le \lVert x_i \rVert^2 \le q_2$ and $\rho_1 \le \langle y_i, y_j \rangle \le \rho_2$, allowing tokens to lie in a low-dimensional space $d \ll n$. This proves that the $\log n$ critical scaling is **intrinsic** rather than an artifact of the simplex geometry.

**2. Forward Phase Transition: Critical Scaling $\beta_n \asymp \log n$ Splits Contraction into Three Phases**

This is the primary result. Under the simplex assumption, **Theorem 2.1** provides closed-form expressions for the limit inner product $\langle y_i', y_j' \rangle$ as $n \to \infty$ when $\beta = \gamma \log n$. For the version without residuals ($\alpha = 0$):

$$\lim_{n\to\infty}\langle y_i', y_j'\rangle = \begin{cases} 1 & \gamma < \tfrac{1}{1-\rho} \ (\text{Sub-critical}) \\[4pt] \tfrac{4\rho}{1+3\rho} & \gamma = \tfrac{1}{1-\rho} \ (\text{Critical}) \\[4pt] \rho & \gamma > \tfrac{1}{1-\rho} \ (\text{Super-critical}) \end{cases}$$

The physical meaning is clear: in the **sub-critical** phase, attention weights are asymptotically uniform ($A_{ij} \sim 1/n$), collapsing all tokens into a single point in one step. In the **super-critical** phase, $A_{ij} \to \delta_{ij}$, attention becomes an identity map, and inner products remain unchanged. Only at the **critical** point $\gamma_c = \tfrac{1}{1-\rho}$ does attention focus on a **sub-linear but non-trivial** number of tokens, where tokens still contract but at a significantly slower rate. While residuals ($\alpha > 0$) mitigate instantaneous collapse in the sub-critical phase, both sub-critical and critical regions remain contractive, suggesting residuals cannot replace correct scaling.

**3. Backward Phase Transition: Gradients Vanish or Stabilize at the Same Threshold**

Training requires backpropagation, and rank collapse is often accompanied by **gradient vanishing**. The authors use the normalized matrix norm of the end-to-end Jacobian $\nabla_X X'$, denoted $\eta = \tfrac{1}{nd} \lVert \nabla_X X' \rVert^2$, as a metric. **Theorem 2.4** (Simplex, $\alpha=0$) reveals three phases synchronized with the forward pass: in the sub-critical phase, $\eta = 0 + o_n(1)$ (gradients cannot pass through attention blocks); at criticality, $\eta = \tfrac{1}{4q}(1 - \tfrac{1}{d}) + o_n(1)$; and in the super-critical phase, $\eta = \tfrac{1}{q}(1 - \tfrac{1}{d}) + o_n(1)$. This indicates that **sub-critical scaling leads to both token collapse and gradient vanishing**, occurring simultaneously at the same $\gamma_c$.

**4. Mechanism: Sparse, Content-Adaptive Attention via $\log n$**

The authors explain the essence of $\log n$ through an intuitive calculation: with $\beta_n = \gamma \log n$, weights become $A_{ij} = \tfrac{n^{\gamma a_{ij}}}{\sum_k n^{\gamma a_{ik}}}$. When scores are $a_{ii}=1$ and $a_{ij}=\rho$, the critical boundary lies exactly at $\gamma = \tfrac{1}{1-\rho}$. Furthermore, the appendix proves (Theorem C.2) that for an intermediate phase $\gamma_1 < \gamma < \gamma_2$, weights $e^{a_{ik}}$ concentrate on a **few highly correlated tokens**, achieving sparse, content-adaptive attention. Unlike fixed-window sliding methods (Longformer/SWIN), logarithmic scaling allows each token to **dynamically** select relevant context based on semantic similarity.

## Key Experimental Results

As this is theoretical work, numerical experiments are used to verify phase transition predictions rather than to train real LLMs. Samples are generated according to $x_i = \sqrt{\rho} z_0 + \sqrt{1-\rho} z_i$ (where $z_0, z_i$ are i.i.d. standard Gaussian vectors).

### Main Results: Forward Angle Phase Transition (Figure 1)

| Dimension $d$ | Observation | Consistency |
|----------|------|--------------|
| $d=512$ (High) | $\langle y_i, y_j \rangle$ centers on $\rho$; a **sharp phase transition** appears along $\gamma = \tfrac{1}{1-\rho}$ | Matches Theorem 2.1 (Simplex) |
| $d=32$ (Mid) | The phase boundary is smoothed; a transition zone of partial contraction appears | Between the two hypotheses |
| $d=2$ (Low) | Inner products are randomly distributed; the transition is significantly smoothed, and an **intermediate phase** emerges | Matches Near-Simplex prediction |

### Backward Gradient Norm Phase Transition (Figure 2)

| Region | $\eta = \tfrac{1}{nd} \lVert \nabla_X X' \rVert^2$ | Meaning |
|------|--------------------------------------------|------|
| Low $\gamma$ (Sub-critical) | $\eta \approx 0$ | Gradients vanish → Untrainable |
| High $\gamma$ (Super-critical) | $\eta \to 1 - \tfrac{1}{d}$ | Gradient scale maintained → Stable |
| High $d$ | Sharp jump at $\gamma = \tfrac{1}{1-\rho}$ | Consistent with Theorem 2.4 |

### Key Findings
- Higher dimensions $d$ lead to sharper phase transitions that align more closely with the simplex theory, explaining why clear critical behavior is observed in high-dimensional Transformers.
- Forward "token collapse" and backward "gradient vanishing" are strictly linked and occur at the same $\gamma_c = \tfrac{1}{1-\rho}$.
- Critical scaling allows attention to focus on a sub-linear number of tokens, enabling "sparse + content-adaptive" behavior without manual window constraints.

## Highlights & Insights
- **Heuristics turned into Theorems**: The $\log n$ scaling used in YaRN/Qwen is theoretically characterized as a phase transition $\beta_n \asymp \log n$, with the critical coefficient $\gamma_c = \tfrac{1}{1-\rho}$ directly linked to token geometry.
- **Unified Forward + Backward Transitions**: By calculating the Jacobian phase transition, the study incorporates "trainability" into the same framework as rank collapse.
- **Dual-layer Proof Strategy**: Solving sharp closed forms in the simplex model and then proving magnitude invariance in the near-simplex model provides a robust paradigm for understanding attention dynamics.

## Limitations & Future Work
- **Simplified Model**: The derivation assumes $K=Q=V=I_d$ and omits MLPs; real multi-head, projected attention might exhibit more complex dynamics.
- **Static vs. Dynamic Analysis**: While the analysis extends to multiple layers, it does not fully characterize how token geometry $\rho$ evolves throughout the training process.
- **No End-to-End LLM Validation**: Experiments focus on synthetic Gaussian tokens rather than evaluating downstream long-context extrapolation in real LLMs.

## Related Work & Insights
- **vs. Rank Collapse Theory (Dong 2021 / Geshkovski 2024-2025)**: These works established attention as a contraction operator; this paper identifies context-aware scaling $\beta_n$ as the control parameter for the "collapse vs. non-collapse" transition.
- **vs. Giorlandino & Goldt 2025**: The discrepancy between this work's $\log n$ and their $\sqrt{\log n}$ is attributed to modeling assumptions; while their scores are i.i.d., this work uses token geometry, which better reflects how attention focuses on specific keys.
- **vs. Structured Sparse Attention (Longformer / SWIN)**: While those methods use fixed sliding windows, this paper shows that logarithmic scaling allows sparsity to **emerge** based on semantic similarity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First rigorous phase transition theory for $\log n$ attention scaling.
- Experimental Thoroughness: ⭐⭐⭐⭐ Clean verification of forward/backward transitions; lacks end-to-end LLM results.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous derivations with clear physical intuition.
- Value: ⭐⭐⭐⭐⭐ Elevates empirical scaling heuristics into theoretical principles for long-context design.

## Related Papers

- [\[ICLR 2026\] Theory of Scaling Laws for In-Context Regression: Depth, Width, Context and Time](theory_of_scaling_laws_for_in-context_regression_depth_width_context_and_time.md)
- [\[ICLR 2026\] On learning linear dynamical systems in context with attention layers](on_learning_linear_dynamical_systems_in_context_with_attention_layers.md)
- [\[ICLR 2026\] Intrinsic Entropy of Context Length Scaling in LLMs](intrinsic_entropy_of_context_length_scaling_in_llms.md)
- [\[ICLR 2026\] In-Context Algorithm Emulation in Fixed-Weight Transformers](in-context_algorithm_emulation_in_fixed-weight_transformers.md)
- [\[ICLR 2026\] Transformers with Endogenous In-Context Learning: Bias Characterization and Mitigation](transformers_with_endogenous_in-context_learning_bias_characterization_and_mitig.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Theory of Scaling Laws for In-Context Regression: Depth, Width, Context and Time](theory_of_scaling_laws_for_in-context_regression_depth_width_context_and_time.md)
- [\[ICLR 2026\] On learning linear dynamical systems in context with attention layers](on_learning_linear_dynamical_systems_in_context_with_attention_layers.md)
- [\[ICLR 2026\] Intrinsic Entropy of Context Length Scaling in LLMs](intrinsic_entropy_of_context_length_scaling_in_llms.md)
- [\[ICLR 2026\] In-Context Algorithm Emulation in Fixed-Weight Transformers](in-context_algorithm_emulation_in_fixed-weight_transformers.md)
- [\[ICLR 2026\] Transformers with Endogenous In-Context Learning: Bias Characterization and Mitigation](transformers_with_endogenous_in-context_learning_bias_characterization_and_mitig.md)

</div>

<!-- RELATED:END -->
