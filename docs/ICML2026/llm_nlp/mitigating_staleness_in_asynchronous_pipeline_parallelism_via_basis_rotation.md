---
title: >-
  [Paper Note] Mitigating Staleness in Asynchronous Pipeline Parallelism via Basis Rotation
description: >-
  [ICML 2026][LLM/NLP][Asynchronous Pipeline Parallelism] The authors attribute the convergence collapse caused by delayed gradients in asynchronous pipeline parallel LLM training to "basis mismatch" in Adam (misalignment…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Asynchronous Pipeline Parallelism"
  - "Gradient Staleness"
  - "Basis Rotation"
  - "Adam"
  - "Hessian Eigenbasis"
date: 2026-05-08
content_hash: d554ed75834cbbb2
---

# Mitigating Staleness in Asynchronous Pipeline Parallelism via Basis Rotation

**Conference**: ICML 2026  
**arXiv**: [2602.03515](https://arxiv.org/abs/2602.03515)  
**Code**: https://github.com/LOG-postech/basis-rotation (Available)  
**Area**: LLM Efficiency / Distributed Training / Optimizers  
**Keywords**: Asynchronous Pipeline Parallelism, Gradient Staleness, Basis Rotation, Adam, Hessian Eigenbasis

## TL;DR
The authors attribute the convergence collapse caused by delayed gradients in asynchronous pipeline parallel LLM training to "basis mismatch" in Adam (misalignment between the Hessian eigenbasis and coordinate axes). They propose performing Adam updates under the Hessian eigenbasis via basis rotation. On a 3B model, this achieves the same loss with 81.7% fewer iterations compared to the strongest asynchronous baseline.

## Background & Motivation

**Background**: Training LLMs with tens of billions of parameters requires partitioning models into multiple GPUs for pipeline parallelism. Synchronous pipelines (GPipe series) wait for all stages to complete the backward pass before updating parameters, creating significant "pipeline bubbles" that lower hardware utilization. Asynchronous pipelines (PipeDream, etc.) allow each stage to update immediately upon receiving gradients, eliminating bubbles to increase throughput.

**Limitations of Prior Work**: The cost of asynchronous execution is gradient staleness—current updates use gradients calculated on weights from several steps ago. Existing remedies include stage-wise learning rate scheduling (PipeDream-LR), Nesterov momentum (Ajanthan 2025), and future weight prediction (PipeMare). However, the authors observed that simply increasing the stage count $P$ from 1 to 32 for a fixed model drops convergence speed to 1/5.81, with all baselines failing at large $P$. More critically, when scaling both model size and $P$, baselines exhibit an "anti-scaling law" phenomenon where larger models yield higher loss.

**Key Challenge**: Theoretically, the $\mathcal{O}(\sqrt{\tau/T})$ slowdown from delay $\tau$ is mild, but actual downstream collapse far exceeds this prediction. The authors discovered that what truly amplifies the harm of delay is the interaction between the optimizer and the loss landscape geometry—specifically, Adam's coordinate-wise adaptation causes violent oscillations along principal eigen-directions when the Hessian eigenbasis is misaligned with the standard coordinate axes.

**Goal**: (i) Explain why delay leads to catastrophic rather than mild degradation in large-scale pipelines; (ii) Provide a staleness mitigation scheme deployable at 10B+ scale that works without relying on weight stashing.

**Key Insight**: Observe Adam's trajectory in a simple quadratic objective $\min_w \tfrac12 w^\top H w$. When $H$ is diagonal (basis aligned), Adam’s trajectory is straight, and delayed gradients still point in nearly the same direction. When $H$ is rotated (basis mismatch), Adam zig-zags along principal eigen-directions, where delayed gradients might point in the opposite direction of the current iteration. The "local consistency" of the trajectory determines the severity of delay damage.

**Core Idea**: Since Adam is staleness-tolerant under an aligned basis, rotate the entire optimization space into the Hessian eigenbasis before applying Adam updates. Use the empirical Fisher $\mathbb{E}[GG^\top]$ and $\mathbb{E}[G^\top G]$ to estimate rotation matrices $U, V$ online. Perform bilateral rotation $\tilde G = U^\top G V$ followed by an Adam update, then rotate back to the original space.

## Method

### Overall Architecture
The central object of the method is an "Adam-with-basis-rotation" update for a weight matrix $W\in\mathbb{R}^{m\times n}$. For each step, after obtaining gradient $G_t = \nabla f_W(W_{t-1};B_t)$: (1) Update the first momentum $M_t$; (2) Refresh left/right rotation matrices $U\in\mathbb R^{m\times m}, V\in\mathbb R^{n\times n}$ (whose columns are eigenvectors of $\mathbb E[GG^\top]$ and $\mathbb E[G^\top G]$) via power iteration every `freq` steps; (3) Compute $\tilde G_t = U^\top G_t V, \tilde M_t = U^\top M_t V$ in rotation space and maintain second momentum $\tilde V_t$; (4) Project the rotation-space Adam update back by computing $W_t = W_{t-1} - \eta_t \cdot U(\tilde M_t / \sqrt{\tilde V_t + \epsilon})V^\top$. This transformation is performed independently for each weight matrix based on two structural assumptions: the Hessian is block-diagonal (each $W$ is an independent block) and each block is Kronecker-decomposable into the tensor product of two smaller matrices. This reduces $mn \times mn$ rotation matrices into $m\times m$ and $n\times n$ matrices, making it tractable at LLM scale.

### Key Designs

1.  **Basis Mismatch as an Amplifier of Staleness (Diagnosis + Theoretical Characterization)**:
    - **Function**: Formalizes "basis mismatch" into a measurable quantity integrated into convergence bounds, proving its multiplicative coupling with delay.
    - **Mechanism**: Uses the Hessian $(1,1)$-norm $\|\nabla^2 f(w)\|_{1,1}=\sum_{i,j}|H_{ij}|$ as a proxy for basis mismatch. Given a fixed spectrum, this norm is minimized when $H$ is diagonal and increases as $H$ rotates. Under assumptions of coordinate-wise bounded noise and $\ell_\infty$ smoothness, they prove the convergence bound for asynchronous Adam ($\beta_1=0$): $\min_t \mathbb E\|\nabla f(w_t)\|_1 = \mathcal O\bigl(\sqrt{(1+d\tau)\Delta_0 C/T} + \sqrt{\sum_i\sigma_i}((1+d\tau)\Delta_0 C/T)^{1/4} + \dots\bigr)$, where $C$ is the mismatch proxy. The multiplicative appearance of delay $\tau$ and $C$ implies that $\tau$ is harmless under an aligned basis but severely amplified under mismatch. Extending this to stage-dependent delay yields an effective delay $\tau' = \sqrt{\sum_i C_i^2 \tau_i^2 / \sum_i C_i^2}$, revealing that earlier stages (highest delay) dominate the convergence drag.
    - **Design Motivation**: By clarifying "why delay hurts," the algorithm can precisely target the cause—suppressing $C$ to neutralize $\tau$.

2.  **Basis-Rotation Adam (Algorithm 1)**:
    - **Function**: Transitions Adam from the standard coordinate system to the Hessian eigenbasis to make coordinate-wise adaptation effective.
    - **Mechanism**: Performing standard Adam in rotation space $\tilde w = \mathcal U^\top w$ is equivalent to the update $\mathcal U \cdot \text{Adam}(\mathcal U^\top \nabla f)$ in the original space. For matrix weights, $\mathcal U$ is decomposed into $U, V$ via the Kronecker assumption. Thus, $\tilde G_t = U^\top G_t V$, second momentum $\tilde V_t$ accumulates squared gradients in rotation space, and finally $W_t \leftarrow W_{t-1} - \eta_t U(\tilde M_t / \sqrt{\tilde V_t + \epsilon}) V^\top$. $U, V$ do not need to be updated every step; `freq=10` is default with negligible performance drop, and even `freq=100` significantly leads over baselines.
    - **Design Motivation**: Basis mismatch causes the $\mathcal O(\tau \cdot C)$ term to dominate convergence; the transformation ensures $C$ approaches its lower bound. Theoretically, $\|H_{U,V}\|_{(1,1)} \le \|H_U\|_{(1,1)} \le \|H\|_{(1,1)}$, and bilateral rotation can achieve a global minimum among all rotations. Empirically, it reduces the normalized Hessian $(1,1)$-norm from 0.5436 to 0.1228.

3.  **Two-Axis Taxonomy for Eigenbasis Estimation (Algorithm 2)**:
    - **Function**: Provides four optional configurations balancing estimation fidelity and memory overhead.
    - **Mechanism**: The first axis is the approximation source $\mathcal S$. $\mathcal S=2^\text{nd}$ maintains EMA matrices $L=\mathbb E[GG^\top]$ and $R=\mathbb E[G^\top G]$ as empirical Fisher proxies. $\mathcal S=1^\text{st}$ approximates $\mathbb E[GG^\top]\approx\mathbb E[G]\mathbb E[G]^\top$ using first-order momentum to save memory. The second axis is rotation geometry $\mathcal G$. Bilateral rotates both sides to capture full Kronecker structure, while unilateral rotates only the smaller dimension to save computation. The paper unifies SOAP as ($\mathcal S=2^\text{nd}$, bilateral) and full-rank GaLore as ($\mathcal S=1^\text{st}$, unilateral) into this framework, isolating the effect of Hessian geometry from other implementation differences.
    - **Design Motivation**: Storing extra matrices or performing eigendecompositions is non-trivial for 10B+ models; providing "tiers" allows the method to fit different VRAM budgets.

### Loss & Training
The training objective is standard next-token prediction for language modeling without extra regularization. Hyperparameters follow Adam, with only `freq` and EMA decay for $L/R$ (matching $\beta_2$) added. All methods use weight stashing by default (using the same weights for forward and backward passes) to ensure correct gradient calculation, though robustness experiments without stashing were also conducted. A stage-aware variant non-uniformly allocates the basis refresh budget based on stage delay $K-k$—refreshing more frequently at earlier stages where delay is highest.

## Key Experimental Results

### Main Results
Decoder-only Transformers ranging from 95M to 3B parameters, trained on 1B tokens of OpenWebText. Baselines include PipeDream, PipeDream-LR, and Nesterov. Default settings are $\mathcal S=2^\text{nd}$ + bilateral, `freq=10`.

| Setting | Metric | Ours (Basis Rotation) | Best Baseline | Gain |
|------|------|------------------------|----------|------|
| 95M, $P=32$ | Iterations to reach same loss | — | — | 71.6% Reduction |
| 1B, $P=24$ | Iterations to reach same loss | — | — | 76.8% Reduction |
| 3B, Large $P$ | Iterations to reach same loss | — | — | 81.7% Reduction |
| 95M, $P=32$ | Slowdown ratio relative to $P=1$ | 1.27× | 4.24× (PipeDream-LR) | ~3× Narrower |
| 95M, $P=32$ | GPU hours (to same loss) | — | — | 54.3% Reduction |

Scaling experiments: When increasing both Transformer blocks and $P$, baselines exhibit degradation (larger models yield higher loss), violating scaling laws, while basis rotation maintains the "larger model, lower loss" trend.

### Ablation Study

| Configuration | $P=32$ slowdown | Description |
|------|------|------|
| PipeDream-LR (Best Baseline) | 4.24× | No basis rotation |
| Basis Rotation, $\mathcal S=1^\text{st}$ / Unilateral | 2.55× | Cheapest tier, still far exceeds baseline |
| Basis Rotation, $\mathcal S=1^\text{st}$ / Bilateral | 1.77× | Adds bilateral rotation |
| Basis Rotation, $\mathcal S=2^\text{nd}$ / Unilateral | 1.66× | Adds second-order source |
| Basis Rotation, $\mathcal S=2^\text{nd}$ / Bilateral | 1.27× | Full tier, closest to $P=1$ |

Stage-aware variant: Under the same total refresh budget, it increases convergence speed by 29.2% compared to uniform `freq`. Reverse allocation (refreshing later stages more) performs worse than uniform, validating that mismatch in earlier stages is the dominant term in $\tau'$.

### Key Findings
- The ranking $\mathcal S=2^\text{nd}$ > $\mathcal S=1^\text{st}$ and bilateral > unilateral perfectly aligns with the theory that better approximations of the true Hessian eigenbasis minimize the $(1,1)$-norm further—strong evidence that basis mismatch is the root cause.
- Even the cheapest tier ($\mathcal S=1^\text{st}$, unilateral) significantly outperforms the strongest baseline, making the method viable for VRAM-constrained setups.
- Removing weight stashing (introducing extra gradient noise via inconsistent weights) causes severe degradation in all baselines, whereas basis rotation remains stable, suggesting robustness to gradient noise itself.
- Tracking parameter update trajectories: Without basis rotation, violent oscillations occur along principal directions while non-principal directions remain stable. With basis rotation, principal direction oscillations are suppressed—matching the hypothesis in Section 2.
- The normalized Hessian $(1,1)$-norm was reduced from 0.5436 to 0.1228, proving effective basis alignment.

## Highlights & Insights
- Attributes the "catastrophic convergence collapse"—previously seen as a purely systems/engineering issue—to the interaction of the optimizer and landscape geometry. It provides a clean diagnostic chain: Delay → Principal direction oscillation → Invalid gradient direction, supported by theory, visualization, and numerical evidence.
- Using the Hessian $(1,1)$-norm as a proxy for basis mismatch is an elegant design: it appears naturally in convergence bounds and can be cheaply measured via trace estimation and random Cauchy vectors.
- The ($\mathcal S, \mathcal G$) taxonomy unifies SOAP, GaLore, and the proposed method into a single family, using ablations to show the performance gap. This provides an attribution analysis for why SOAP-like methods surprisingly excel in asynchronous settings.
- Stage-aware scheduling is derived directly from the theoretical $\tau'$ term rather than ad-hoc engineering tricks; the sanity check with reverse allocation further confirms the causal insight.
- Working effectively without weight stashing is highly practical for large models, as weight stashing's memory overhead scales linearly with $P$.

## Limitations & Future Work
- The theoretical analysis is based on Adam with $\beta_1=0$. While the appendix extends this to $\beta_1>0$, some terms coupled with coordinate-wise assumptions require caution when applied to real Transformer landscapes.
- The Kronecker + block-diagonal Hessian assumptions are standard in K-FAC/SOAP literature, but their validity in structures like MoE or extremely long contexts is not explored in depth (though the appendix provides a small MoE validation).
- Basis rotation introduces two extra matrices $L, R$ ($\mathcal S=2^\text{nd}$) and two $m\times m, n\times n$ matmuls per weight matrix. While overhead is low at 3B, the impact at 70B+ while maintaining `freq=10` is an open question.
- Comparisons with Muon or other recent preconditioned optimizers are largely in the appendix; the main narrative focuses on "async pipeline baselines."

## Related Work & Insights
- **vs PipeDream-LR (Yang 2021)**: They suggest smaller learning rates for stages with higher delay, effectively scaling down all directions equally. This paper proves damage is concentrated in principal directions; direction-aware (rather than stage-global) step adjustment is the correct solution.
- **vs Nesterov for async (Ajanthan 2025)**: Uses Nesterov momentum to "look ahead" and cancel delay. This modifies the optimizer in the standard basis. This paper argues the coordinate system itself is flawed; thus Nesterov still suffers a ~4x slowdown at $P=32$.
- **vs SOAP (Vyas 2025) / Full-rank GaLore (Zhao 2024)**: These are equivalent to ($\mathcal S=2^\text{nd}$, bilateral) and ($\mathcal S=1^\text{st}$, unilateral) basis rotation respectively. Originally framed as high-performance synchronous optimizers, this paper reinterprets them as providing the necessary basis alignment to mitigate delay, a significant shift in perspective.
- **vs Weight prediction (PipeMare)**: Predicts future weights to "fake" non-delayed gradients, but incurs extra computation and prediction noise. Basis rotation is orthogonal and can be combined with prediction methods.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Re-explaining delay damage via Hessian geometry is a fresh perspective; the algorithm itself shares significant overlap with SOAP/GaLore.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ From 95M to 3B scale, multiple baselines, extensive ablations, stashing-free tests, and empirical Hessian measurements.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The four-step transition from phenomenon to intuition to experiment to theory in Section 2 is exceptionally clear.
- **Value**: ⭐⭐⭐⭐⭐ Asynchronous pipelining has long been seen as "theoretically attractive but practically broken." This provides an interpretable, scalable (3B+), and baseline-compatible solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel](slay_geometry-aware_spherical_linearized_attention_with_yat-kernel.md)
- [\[ICML 2026\] Escaping Mode Collapse in LLM Generation via Geometric Regulation](escaping_mode_collapse_in_llm_generation_via_geometric_regulation.md)
- [\[ICML 2026\] Token-Efficient Change Detection in LLM APIs](token-efficient_change_detection_in_llm_apis.md)
- [\[ICML 2026\] "I've Seen How This Goes": Characterizing LLM and Human Writing Diversity via Progressive Conditional Surprisal](ive_seen_how_this_goes_characterizing_diversity_via_progressive_conditional_surp.md)
- [\[ICML 2026\] In-Context Routing (ICR): Train Once, Use Everywhere via Attention-Level Implicit ICL](train_once_reuse_everywhere_generalizable_implicit_in-context_learning_by_routin.md)

</div>

<!-- RELATED:END -->
