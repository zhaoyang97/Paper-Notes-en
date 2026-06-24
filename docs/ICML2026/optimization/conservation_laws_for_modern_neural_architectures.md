---
title: >-
  [Paper Note] Conservation Laws for Modern Neural Architectures
description: >-
  [ICML2026][Optimization][Conservation laws] This paper reformulates the problem of "characterizing all conserved quantities in training dynamics" as solving a data-independent partial differential equation (PDE). Leveraging meromorphic continuation techniques from complex analysis, it provides for the first time a **complete list of conservation laws** for GELU/SiLU/SwiGLU feed-forward networks, multi-head attention (including sinusoidal PE and RoPE), and various gated MoEs…
tags:
  - "ICML2026"
  - "Optimization"
  - "Conservation laws"
  - "Gradient flow"
  - "Implicit bias"
  - "Attention mechanism"
  - "RoPE"
  - "MoE"
date: 2026-05-08
content_hash: f6c473b05c5fcd35
---

# Conservation Laws for Modern Neural Architectures

**Conference**: ICML2026  
**arXiv**: [2606.17816](https://arxiv.org/abs/2606.17816)  
**Code**: To be confirmed  
**Area**: Optimization Theory / Implicit Bias  
**Keywords**: Conservation laws, Gradient flow, Implicit bias, Attention mechanism, RoPE, MoE

## TL;DR
This paper reformulates the problem of "characterizing all conserved quantities in training dynamics" as solving a data-independent partial differential equation (PDE). Leveraging meromorphic continuation techniques from complex analysis, it provides for the first time a **complete list of conservation laws** for GELU/SiLU/SwiGLU feed-forward networks, multi-head attention (including sinusoidal PE and RoPE), and various gated MoEs, effectively solving the open problem for multi-head attention posed by Marcotte et al. (2025).

## Background & Motivation

**Background**: Conservation laws refer to quantities that remain invariant along the optimization trajectory. Under gradient flow $\dot\theta(t)=-\nabla L_{\mathcal D}(\theta(t))$, such invariants expose the geometric constraints imposed jointly by the architecture and the optimization algorithm. They serve as a key to understanding the "implicit bias" of over-parameterized models—explaining why certain properties are preserved from initialization to convergence— and are used to analyze convergence, stability, or even to design accelerated optimizers.

**Limitations of Prior Work**: Although various conserved quantities have been discovered sporadically, results regarding **completeness** (proving "these are all of them") are extremely rare. Marcotte et al. (2023/2024/2025) used a Lie algebraic framework to characterize conservation laws for shallow linear nets, ReLU nets, ICNNs, NMF, ResNets, and even single-head attention. However, for **multi-head attention**, they only found several invariants without proving exhaustiveness, explicitly listing it as an open problem. Furthermore, modern large models no longer use ReLU or single-head attention—they use GELU/SiLU/SwiGLU, RoPE, and sparse MoE, for which conservation laws are almost entirely unknown.

**Key Challenge**: The original Lie algebraic machinery is cumbersome for modern operators that are "non-polynomial, position-dependent, or feature gated-routing." In multi-head attention, while $Q_i^\top Q_i-K_i^\top K_i$ for each head appears to be an invariant, proving no other invariants exist requires a new set of tools capable of handling transcendental functions and discontinuous routing.

**Goal**: To provide a **complete characterization** of conservation laws for recurring key components in modern deep learning (not just finding them, but proving exhaustiveness), accompanied by rigorous proofs and experimental validation.

**Key Insight**: Instead of applying Lie algebra, the authors return to first principles and ask "what does characterizing all conservation laws actually entail?" The answer: solving a PDE that constraints $h$ itself after eliminating the input $x$.

**Core Idea**: The gradient of a conservation law $h$ must be orthogonal to each $\nabla_\theta f_i(x;\theta)$. By using "continuation-elimination" from complex analysis to transform this $x$-dependent condition into an $x$-independent PDE constraint and identifying the characteristic invariants of the PDE, one can enumerate all conservation laws.

## Method

### Overall Architecture

The entire paper is a theoretical pipeline that "establishes a unified criterion first, then derives results per architecture using a single proof engine." The starting point is the orthogonality criterion given by Marcotte et al.: assuming "the conservation law must hold for any dataset and the loss satisfies $\mathcal V_\ell=\mathbb R^{d_\text{out}}$," a $C^1$ function $h$ is a conservation law if and only if for all $i\in[d_\text{out}]$, $\theta$, and $x$:

$$\langle\nabla_\theta h(\theta),\,\nabla_\theta f_i(x;\theta)\rangle=0.$$

The difficulty lies in the fact that this condition **explicitly depends on the input $x$**, while $h$ is defined only on the parameter space. The overall strategy is to treat the above as a function of $x$, find ways to eliminate $x$, obtain a system of PDEs constraining only $\nabla_\theta h$, and solve for characteristic invariants. The paper demonstrates this process using a toy model $f(x;a,b)=abx$—the constraint simplifies to $b\,\partial_a h+a\,\partial_b h=0$, and $a^2-b^2$ is discovered as the invariant along characteristic curves. The subsequent four sections apply this "reduction to PDE $\rightarrow$ read invariants" method to FFN, MHA, RoPE-MHA, and MoE.

### Key Designs

**1. Reducing "Characterizing All Laws" to PDEs: A First-Principles Alternative to Lie Algebra**

The Lie algebraic framework is unintuitive and difficult to generalize for modern operators. The authors adopt a more fundamental perspective: the orthogonality criterion $\langle\nabla_\theta h,\nabla_\theta f_i(x;\theta)\rangle=0$ holding for all $x$ essentially requires collapsing the "$x$-dependency" into "constraints on $h$." Since $h$ does not contain $x$, these constraints naturally fall between components of $\nabla_\theta h$, forming a system of first-order PDEs. The success of the criterion is judged by whether the condition can be simplified to be **independent of $x$** and sufficient to fully determine $h$.

**2. Meromorphic Continuation + Identity Theorem + Pole Order Comparison: The Proof Engine**

This is the core mechanism that makes the proofs work. For fixed $\theta, i$, define $F_i(x;\theta)\coloneqq\langle\nabla_\theta h,\nabla_\theta f_i(x;\theta)\rangle$. For operators like FFN, MHA, and dense MoE, $f_i(\cdot;\theta)$ is **meromorphic**, thus $F_i$ is also a meromorphic function that can be extended to the complex plane. By restricting the input to $x=t e_j$ (where $t$ varies over an open interval on the real axis), the **identity theorem** from complex analysis implies that a meromorphic function that is zero on a set with an accumulation point must be zero everywhere. This yields $x$-independent constraints. The authors further analyze the **pole structure** of the meromorphic components of $F_i$: if a linear combination of meromorphic functions is zero, comparing pole orders forces the coefficients of the principal poles to be zero, which iteratively extracts the equations $\nabla_\theta h$ must satisfy. For SMoE, Top-$k$ routing introduces discontinuities at boundaries; additional arguments are used to connect different activation regions (which is why Theorem 4.6 requires $k>1$: connectivity forces consistent gradients across expert gates).

**3. Complete List of Conservation Laws for Modern Architectures: Reading Invariants Per Operator**

Applying the above to specific architectures yields the main results (all are "complete characterizations," proving no other laws exist). A single-hidden-layer FFN with GELU or SiLU $f=A\cdot\text{act}(Bx)$ **has no non-trivial conservation laws**; all laws are constants. SwiGLU, due to the multiplicative interaction between $A$ and $C$, produces non-trivial invariants $\lVert A_{:,i}\rVert^2-\lVert C_{i,:}\rVert^2$. For multi-head attention (no PE), the authors confirm Marcotte et al.'s conjecture: laws are fully characterized by $Q_i^\top Q_i-K_i^\top K_i$ and $V_i^\top V_i-O_i^\top O_i$. Sinusoidal PE merely applies a bijective shift to the input without changing MHA's internal structure, so its laws are identical to vanilla MHA. RoPE, however, **substantially changes** the invariant structure because the rotation $Q_iR_{p-q}K_i^\top$ couples positions into the scores, leading to $2\times2$ block-wise invariants $\lVert Q_i^{(j)}\rVert_F^2-\lVert K_i^{(j)}\rVert_F^2$. For MoE, invariants are "localized" to each expert (following SwiGLU results), plus a global invariant $\sum_{i}W_i$ from the gating; these remain **consistent across four variants**: dense, sparse ($k>1$), softmax gating, and normalized sigmoid gating.

Summary of invariants across architectures:

| Architecture | Complete Conserved Invariants |
|--------------|-------------------------------|
| FFN (GELU / SiLU) | No non-trivial conservation laws ($h$ must be constant) |
| FFN (SwiGLU) | $\lVert A_{:,i}\rVert^2-\lVert C_{i,:}\rVert^2,\ i\in[d_1]$ |
| MHA (No PE / Sinusoidal PE) | $Q_i^\top Q_i-K_i^\top K_i$, $V_i^\top V_i-O_i^\top O_i$, $i\in[n]$ |
| MHA + RoPE | $\lVert Q_i^{(j)}\rVert_F^2-\lVert K_i^{(j)}\rVert_F^2$ (block-wise $i,j$), $V_i^\top V_i-O_i^\top O_i$ |
| MoE (dense / sparse $k>1$ / sigmoid gating) | SwiGLU invariants per expert + gating $\sum_{i=1}^n W_i$ |

### Loss & Training

This is a theoretical characterization; no new models are trained. The conservation laws assume Euclidean gradient flow ODEs. Following Marcotte et al. (2025), **conservation functions correspond regardless of weight decay**, so weight decay is omitted in favor of pure gradient flow analysis. Theoretical results are validated using real discrete optimization (see below).

## Key Experimental Results

### Main Results

The experiment aims to verify if "conserved quantities in continuous gradient flow remain approximately conserved under discrete SGD." The theoretical error bound (based on Proposition 5.1 from Marcotte et al. 2025, assuming bounded Hessian and gradient expectations) is:

$$\mathbb E\,\big|h(\theta_k)-h(\theta_0)\big|\le\frac{C_hC_L}{2}\sum_{i=0}^{k-1}\tau_i^2.$$

This leads to distinct behaviors for the conservation error under two step-size strategies:

| Step-size Strategy | Conservation Error Bound | Meaning |
|--------------------|--------------------------|---------|
| Constant $\tau_k=\tau$ | $\mathcal O(\tau^2 k)$, grows linearly with iterations | Discretization error accumulates but at a controllable rate |
| Decaying $\tau_k=\tau_0/(k+1)$ | $\mathcal O(\tau_0^2)$, uniformly bounded | Conservation laws remain approximately maintained as SGD converges |

Ours validates this in Language Modeling (Qwen-3 architecture + WikiText-103 / Penn Treebank, including RoPE, dense/sparse MoE, softmax, and normalized sigmoid gating) and Computer Vision (ViT + CIFAR-10 / ImageNet-1K, including absolute PE and SwiGLU). Each configuration is trained with 10 random seeds, monitoring the block-level relative deviation:

$$\epsilon_\text{block}(k)=\frac1N\sum_{i=1}^N\frac{\lVert h_i(\theta_k)-h_i(\theta_0)\rVert_2}{\lVert h_i(\theta_0)\rVert_2}.$$

### Ablation Study

| Monitored Quantity | Phenomenon | Explanation |
|--------------------|------------|-------------|
| Conserved quantities of FFN / MHA / RoPE / MoE gating | Error increases with learning rate and grows mildly with iterations | Consistent with the $\mathcal O(\tau^2 k)$ theoretical prediction |
| Non-conserved quantities (baseline) | Significant drift occurs even with minor parameter changes | Qualitative contrast highlighting the strictly constrained evolution of conserved quantities |

### Key Findings

- The **magnitude of conservation error scales with the learning rate**, following the $\mathcal O(\tau^2 k)$ scaling across MHA, SwiGLU FFN, RoPE, and MoE gating components, confirming that theoretical invariants are indeed approximately conserved in discrete training.
- Baseline non-conserved quantities fluctuate significantly under the same training conditions, proving that the stability of identified invariants is non-trivial.
- The boundaries of the proof framework are clear: functions containing $\sqrt{x}$ (Layer Normalization) or $e^{1/x}$ (attention layer stacking) introduce branch points or essential singularities and **lack full-plane meromorphic continuation**, meaning the current engine cannot process them directly.

## Highlights & Insights
- **Converting "Searching for Invariants" to "Solving PDEs"**: The most elegant step is using the identity theorem of complex analysis to collapse infinite $x$-dependent constraints into finite PDE constraints, enabling proof of "completeness" rather than just discovery.
- **GELU/SiLU lacks non-trivial laws while SwiGLU has them**: The difference lies solely in the multiplicative interaction of the $C$ path in SwiGLU, suggesting that "gating/multiplicative structures" are the source of non-trivial implicit bias.
- **Quantifying the Essential Difference of RoPE**: Whereas sinusoidal PE is a bijective shift leaving the conservation structure intact, RoPE couples position rotations into the scores, degrading invariants from the full matrix level to the $2\times2$ block level.
- The proof paradigm of "meromorphic continuation + pole order comparison" is a reusable technical asset for analyzing implicit bias in other meromorphic operators.

## Limitations & Future Work
- The framework cannot handle operators with $\sqrt{x}$ (LayerNorm) or $e^{1/x}$ (multi-layer attention composition) due to branch points or essential singularities; improvements would require extending $F_i$ to specific regions of $\mathbb C^d$.
- Laws are based on continuous-time Euclidean gradient flow; in discrete SGD, they are only "approximately conserved," with constant step sizes leading to linear error accumulation. Further validation is needed for real-world large-scale training (momentum, adaptive optimizers, various regularizations).
- FFN analysis is limited to a single hidden layer; characterization for multi-layer stacks remains unsolved.
- Future Direction: Refining singularity analysis to cover LayerNorm and deep Transformers is a critical step toward "whole-network" characterization.

## Related Work & Insights
- **vs Marcotte et al. (2023/2024)**: They used Lie algebra for shallow linear/ReLU nets, ICNNs, NMF, and momentum dynamics. Ours adopts complex analysis and PDE reduction, covering GELU/SiLU/SwiGLU, MHA, RoPE, and MoE, which is better suited for non-polynomial and gated structures.
- **vs Marcotte et al. (2025)**: They extended the framework to ResNet and Transformers but only for single-head attention, leaving multi-head as an open problem. Ours proves the multi-head attention laws are fully characterized by $Q_i^\top Q_i-K_i^\top K_i$ and $V_i^\top V_i-O_i^\top O_i$, **directly solving the open problem** and providing results for RoPE and MoE gating.
- **vs Kunin et al. (2021) / Zhang et al. (2025), etc.**: These works found many specific invariants but lacked completeness proofs. The value of Ours lies in "exhaustiveness"—proving the listed invariants are all that exist.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses complex analysis reduction to generalize conservation laws to modern architectures with completeness guarantees, solving open problems.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validates conservation error scaling across domains and architectures; however, lacks quantification of deviation in massive-scale training.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear intuition through toy examples; theorems and proof sketches are well-structured; honest about limitations.
- Value: ⭐⭐⭐⭐ High theoretical value by providing reusable tools and complete conclusions for implicit bias research; direct application is more indirect.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Transformative or Conservative? Conservation Laws for ResNets and Transformers](../../ICML2025/optimization/transformative_or_conservative_conservation_laws_for_resnets_and_transformers.md)
- [\[ICML 2026\] Muon in Associative Memory Learning: Training Dynamics and Scaling Laws](muon_in_associative_memory_learning_training_dynamics_and_scaling_laws.md)
- [\[ICLR 2026\] Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures](../../ICLR2026/optimization/saddle-to-saddle_dynamics_explains_a_simplicity_bias_across_neural_network_archi.md)
- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)

</div>

<!-- RELATED:END -->
