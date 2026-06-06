---
title: >-
  [Paper Note] Muon in Associative Memory Learning: Training Dynamics and Scaling Laws
description: >-
  [ICML2026][Optimization][Muon optimizer] This paper theoretically characterizes the convergence rates and scaling laws of Muon on a linear associative memory model with softmax retrieval and hierarchical spectra. Relativ…
tags:
  - "ICML2026"
  - "Optimization"
  - "Muon optimizer"
  - "associative memory"
  - "matrix sign operator"
  - "scaling laws"
  - "training dynamics"
date: 2026-05-08
content_hash: cf1ee73f5390d59f
---

# Muon in Associative Memory Learning: Training Dynamics and Scaling Laws

**Conference**: ICML2026  
**arXiv**: [2602.05725](https://arxiv.org/abs/2602.05725)  
**Code**: Not disclosed  
**Area**: optimization  
**Keywords**: Muon optimizer, associative memory, matrix sign operator, scaling laws, training dynamics  

## TL;DR
This paper theoretically characterizes the convergence rates and scaling laws of Muon on a linear associative memory model with softmax retrieval and hierarchical spectra. Relative to GD, Muon achieves exponential acceleration in the noiseless case and improves the loss convergence rate from $\tilde{\Omega}(T^{-(1-1/\beta)})$ to $\tilde{\mathcal{O}}(T^{-2})$ under power-law spectral noise. This acceleration is attributed to the matrix sign operator acting as an adaptive, task-aligned implicit preconditioner.

## Background & Motivation

**Background**: In large-scale pre-training of modern LLMs, matrix-parameter optimizers have gradually transitioned from SGD/Adam/AdamW to Muon, proposed by Jordan et al. Muon has repeatedly demonstrated higher compute and data efficiency than AdamW in large-scale training regimes across architectures like dense Transformers and MoE, leading to rapid adoption by the engineering community.

**Limitations of Prior Work**: Most existing theoretical literature treats Muon as a "standard stochastic optimization" problem to derive a convergence upper bound (Bernstein views Muon as steepest descent under the operator norm; subsequent work gives gradient norm convergence rates). However, such static worst-case bounds fail to explain why Muon is "faster and more balanced" in real pre-training, nor do they provide a neural scaling law specific to Muon.

**Key Challenge**: Muon performs spectral normalization on matrix parameters via $\mathrm{msgn}(\mathbf{G})=\mathbf{U}\,\mathrm{sgn}(\boldsymbol{\Sigma})\,\mathbf{V}^\top$, essentially "amplifying" step sizes on low-frequency long-tail tasks. Conversely, the effective step size of GD is proportional to the knowledge frequency $p_j$, making convergence for tail tasks extremely slow ($\sim 1/(p_j t)$). To explain Muon's advantages, one must move beyond static bounds to characterize how quickly "frequency components are learned" along dynamic training trajectories.

**Goal**: (1) Derive per-subtask and total loss curves for Muon and GD under both noiseless and label-noise associative memory settings; (2) Derive the optimization scaling law for Muon under power-law spectra and compare it with the GD lower bound; (3) Provide a mechanistic perspective explaining Muon's acceleration.

**Key Insight**: The authors use associative memory as an analytically tractable proxy model—knowledge is organized into $K$ orthogonal query-answer pairs $(\mathbf{E}_j,\widetilde{\mathbf{E}}_j)$ appearing with $M$ groups of hierarchical frequencies $\tilde p_i$. The model is a softmax retrieval using a single matrix $\mathbf{W}\in\mathbb{R}^{K\times K}$. This framework faithfully simulates factual recall in Transformers (backed by experiments from Geva and Meng) and decomposes the gradient structure into "frequency × residual × association," allowing the SVD evolution of Muon to be tracked in closed form.

**Core Idea**: The matrix sign operation in the task representation basis is approximately equal to the identity matrix $\mathbf{I}_K$ (i.e., $\mathrm{msgn}(\mathbf{G}_t)\approx \mathbf{I}_K$). It "flattens" the frequency-tilted directional bias of GD into isotropic updates, allowing high-frequency and low-frequency groups to be learned at the same rate, thereby converting power-law integrals into fast $\mathcal{O}(T^{-2})$ decay.

## Method

This paper is a purely theoretical characterization; it does not propose a new algorithm. "Method" refers to the construction of the theoretical framework and key proof strategies.

### Overall Architecture
The object of analysis is the minimization problem under associative memory. Given $K$ orthogonal equal-norm embeddings $(\mathbf{E}_j,\widetilde{\mathbf{E}}_j)$ and a frequency structure $p_j=\tilde p_i/C$ ($M$ frequency groups, each with $C=K/M$ items), label noise level $\alpha\in[0,1)$ induces a conditional distribution $p_{i\mid j}=(1-\alpha)\mathbb{1}[i=j]+\alpha/K$. The linear softmax model $\hat p_{i\mid j}(\mathbf{W})=\frac{\exp(\widetilde{\mathbf{E}}_i^\top \mathbf{W}\mathbf{E}_j)}{\sum_k \exp(\widetilde{\mathbf{E}}_k^\top \mathbf{W}\mathbf{E}_j)}$ stores knowledge by minimizing cross-entropy $\mathcal{L}(\mathbf{W})=\mathbb{E}_{\mathcal{D}_\alpha}[-\log\hat p_{i\mid j}(\mathbf{W})]$. The two optimizers are $\mathbf{W}_{t+1}=\mathbf{W}_t-\eta\nabla\mathcal{L}$ (GD) and $\mathbf{W}_{t+1}=\mathbf{W}_t-\eta\,\mathrm{msgn}(\nabla\mathcal{L})$ (Muon, omitting momentum, equivalent to Spectral GD), both with zero initialization. The analysis tracks the evolution of $\widehat{\mathbf{W}}_t=\widetilde{\mathbf{E}}^\top \mathbf{W}_t \mathbf{E}$ and the corresponding gradient $\mathbf{G}_t=\widetilde{\mathbf{E}}^\top \nabla\mathcal{L} \mathbf{E}$ in task representation space.

### Key Designs

1.  **Gradient Structure Decomposition + Frequency Bottleneck Characterization**:
    - **Function**: Decomposes the softmax model gradient as $\nabla\mathcal{L}(\mathbf{W})=\sum_{i,j} p_j\,(\hat p_{i\mid j}-p_{i\mid j})\,\widetilde{\mathbf{E}}_i \mathbf{E}_j^\top$, representing "query frequency × prediction residual × embedding association." This is the foundation for all subsequent theorems.
    - **Mechanism**: In the noiseless case, it is proven that $\mathcal{L}_j^{\mathrm{GD}}(t)\eqsim 1/(p_j t)$ and $\mathcal{L}^{\mathrm{GD}}(t)\eqsim K/t$ (Theorem 4.1), while $\mathcal{L}_j^{\mathrm{Muon}}(t)\eqsim K e^{-(1+o_K(1))t}$ (Theorem 4.2), where all subtasks converge at the same exponential rate. To reduce loss to accuracy $\epsilon$, GD requires $\mathcal{O}(1/\epsilon)$ steps, whereas Muon requires only $\mathcal{O}(\log(1/\epsilon))$.
    - **Design Motivation**: Directly identifies where GD is slow—the effective step size along the $j$-th component is proportional to $p_j$, locking out long-tail classes. Muon uses $\mathrm{msgn}$ for spectral normalization, which is equivalent to stripping away this $p_j$ factor.

2.  **Three-stage Dynamics + Muon Scaling Law $\tilde{\mathcal{O}}(T^{-2})$**:
    - **Function**: Provides a two-stage closed form for Muon subtask loss in the label-noise case (descent phase $\sim Ke^{-(1+o_K(1))\eta t} + \eta t$ and oscillation phase $\sim \eta^2+\mathcal{L}_j^\ast$, with critical time $T_j^\ast=\Theta(\log K/\eta)$, Theorem 5.1). This is placed under a power-law spectrum $\tilde p_i\propto i^{-\beta}$ ($\beta>1$) to derive the scaling law.
    - **Mechanism**: Choosing the learning rate $\eta=\Theta(\log K/T)$ optimally balances the $Ke^{-\eta T}$ descent and $\eta^2$ oscillation, yielding $\mathcal{L}^{\mathrm{Muon}}(T)-\mathcal{L}^\ast\lesssim (\log K/T)^2$ (Theorem 5.8). Under the same spectrum, GD's per-subtask loss is $\mathcal{L}_j^{\mathrm{GD}}(T)-\mathcal{L}_j^\ast\gtrsim e^{-\eta p_j T}\log K$; summing over $j$ and approximating via $\int_1^M z^{-\beta} e^{-z^{-\beta}T}dz\approx T^{-(1-1/\beta)}$ yields the GD lower bound $\tilde{\Omega}(T^{-(1-1/\beta)})$ (Theorem 5.7). Muon achieves an $\Omega(C)$ (group size) acceleration over GD for reaching a target loss.
    - **Design Motivation**: This is the first neural scaling law for Muon itself, explaining why Muon's loss-compute curve is steeper in large-scale pre-training ($K,M,T \to \infty$).

3.  **Preconditioning Perspective: $\mathrm{msgn}$ as Implicit Task Alignment**:
    - **Function**: Reveals the mechanism of Muon's acceleration—in task representation space, $\mathrm{msgn}(\mathbf{G}_t)\approx \mathbf{I}_K$, meaning Muon performs aligned updates where $\widehat{\mathbf{W}}_t\approx t\mathbf{I}_K$.
    - **Mechanism**: Induction proves that Muon starting from $\mathbf{W}_0=\mathbf{0}$ preserves the block-symmetric structure induced by frequency groups (Proposition 6.1). The residual matrix decomposes into block-diagonal and block-constant terms. In the $M(C-1)$-dimensional intra-group contrastive subspace, $\mathrm{msgn}$ reduces to the identity matrix, with deviations in the $M$-dimensional block-mean directions contributing at most $M/C$. Thus, $\|\mathrm{msgn}(\mathbf{P}-\widehat{\mathbf{P}}_t)-\mathbf{I}_K\|_{\max}\le 1/C+M/C=o_K(1)$. Theorem 6.3 shows that "Task Representation Aligned SignGD" (TRA-SignGD), defined as $\widehat{\mathbf{W}}_{t+1}=\widehat{\mathbf{W}}_t-\eta\,\mathrm{sgn}(\mathbf{G}_t)$, matches Muon's results with half the learning rate.
    - **Design Motivation**: Clarifies why Muon outperforms SignGD—to perform coordinate-wise signs in the original space, SignGD requires an oracle to know the unknown $\mathbf{E},\widetilde{\mathbf{E}}$ for alignment. Muon uses SVD to automatically find this task representation basis, making it practical without an oracle.

### Loss & Training
All theoretical results are based on zero initialization $\mathbf{W}_0=\mathbf{0}_{K\times K}$ and a constant learning rate $\eta$. The scaling law section uses $\eta=\Theta(\log K/T)$. The GD stability condition $\eta p_1\lesssim 1$ is given by the linear stability of the fixed-point Jacobian (Proposition 5.4).

## Key Experimental Results

Experiments serve as a sanity check, verifying theoretical predictions using synthetic long-tail classification and LLaMA-style pre-training.

### Main Results

| Setting | Spectrum / Data | Muon Behavior | GD Behavior |
| :--- | :--- | :--- | :--- |
| Noiseless associative memory | $K$ orthogonal items, $M$ groups | Synchronous exponential convergence, $\mathcal{L}^{\mathrm{Muon}}\eqsim K e^{-t}$ | Subtask rate $\propto p_j$, total loss $\eqsim K/t$, tail groups stuck |
| Noisy power-law spectrum $\tilde p_i\propto i^{-\beta}$ | $\beta>1$ | $\mathcal{L}^{\mathrm{Muon}}(T)-\mathcal{L}^\ast\lesssim (\log K/T)^2$ | $\mathcal{L}^{\mathrm{GD}}(T)-\mathcal{L}^\ast\gtrsim \log K/T^{1-1/\beta}$ |
| LLaMA-style pre-training | Real long-tail text | Significantly higher tail accuracy, steeper scaling slope | Slower convergence, under-learned tail classes |

### Ablation Study

| Configuration | Behavior | Description |
| :--- | :--- | :--- |
| GD (baseline) | Frequency sensitive $1/(p_j t)$ | Tail groups stuck, scaling limited by $\beta$ |
| Normalized GD (NGD) | Faster than GD but still imbalanced | Shows acceleration isn't just step size normalization; matrix-sign is necessary |
| SignGD (Original Coords) | Cannot utilize task structure | Requires oracle $\mathbf{E},\widetilde{\mathbf{E}}$ to match Muon |
| TRA-SignGD (Ideal Alignment) | Matches Muon using $\eta$ vs $2\eta$ | Validates that Muon's advantage comes from implicitly finding task bases |
| Muon (Momentum-free) | Achieves exponential speedup + $\tilde{\mathcal{O}}(T^{-2})$ scaling | Matrix sign strips $p_j$ factor from effective step size |

### Key Findings
- Muon's acceleration is split: spectral normalization flattens update scales (which NGD does partially), and $\mathrm{msgn}$ provides implicit alignment along task representation bases (which NGD cannot), the latter being the source of $\Omega(C)$ speedup.
- The trade-off between the oscillation term $\eta^2$ and the descent term $Ke^{-\eta T}+\eta T$ introduced by label noise determines the optimal $\eta=\Theta(\log K/T)$, which naturally yields Muon's scaling exponent of $-2$.
- GD's scaling exponent $-(1-1/\beta)$ degrades to 0 as the power-law index $\beta$ approaches 1, performing poorly on long tails. Muon's exponent is independent of $\beta$, which is the fundamental reason for its steeper scaling on real corpora.

## Highlights & Insights
- Provides the first "native" neural scaling law for Muon rather than adapting worst-case bounds from SGD/SignGD. The $-2$ exponent vs. GD's $-(1-1/\beta)$ provides a non-trivial contrast, grounding the empirical observation that Muon is more effective in large-scale training into a formal asymptotic law.
- The technical combination of associative memory + block-symmetric induction + task representation space is highly reusable. Any "SVD-based sign" optimizer can use this framework to calculate scaling laws if the gradient has a "frequency × residual × association" structure.
- The design of TRA-SignGD is clever: it proves Muon's equivalence to "aligned SignGD," accurately attributing Muon's advantage to the automatic alignment capability of SVD rather than a vague notion that "matrix optimization is stronger."

## Limitations & Future Work
- The model is a single-matrix linear softmax lacking non-linearity, MLPs, or multi-head mechanisms. The scaling law derived here strictly covers factual recall and cannot be directly extrapolated to token-level perplexity in generation.
- Embeddings are assumed to be orthogonal and equal-norm (though authors claim this can be relaxed to near-orthogonal); real pre-training token embeddings are not this clean. The power-law spectrum is also only a first-order Zipf approximation.
- Muon's actual implementation includes momentum (Newton-Schulz iteration to estimate $\mathrm{msgn}$). This analysis removes momentum to purify the results; the momentum-free version is equivalent to Spectral GD.
- Future directions: Extending block-symmetric analysis to (i) multi-layer Transformer factual recall stacking, (ii) MoE expert routing frequencies, and (iii) scaling laws under momentum and LR schedulers.

## Related Work & Insights
- **vs Bernstein & Newhouse 2024 / Li & Hong 2025**: They provide Muon's worst-case stochastic convergence bounds; this paper provides problem-specific closed-form dynamics and scaling laws, quantitatively explaining the speedup factor.
- **vs Wang et al. 2025b**: That was an empirical study finding Muon is strong on tail classes; this paper proves it as an $\Omega(C)$ acceleration and provides the mechanism (implicit preconditioning).
- **vs Kunstner & Bach 2025**: Extends the scaling analysis framework of SignGD-like methods from bigram models to associative memory and identifies the crucial distinction between $\mathrm{msgn}$ and $\mathrm{sgn}$ regarding "oracle alignment."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First native scaling law for Muon and discovery of the "$\mathrm{msgn}\approx \mathbf{I}_K$" alignment mechanism.
- Experimental Thoroughness: ⭐⭐⭐ Relies mostly on synthetic and small-scale LLaMA training without large-scale ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear theorem layout, preconditioning view section provides strong intuition.
- Value: ⭐⭐⭐⭐⭐ Directly provides a template for theoretical analysis of subsequent spectral optimizers and justifies Muon's scaling gains engineering-wise.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](../../NeurIPS2025/optimization/learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)
- [\[ICML 2026\] Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks](balancing_learning_rates_across_layers_exact_two-step_dynamics_and_optimal_scali.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[ICLR 2026\] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](../../ICLR2026/optimization/scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)
- [\[ICML 2026\] Mirror Mean-Field Langevin Dynamics](mirror_mean-field_langevin_dynamics.md)

</div>

<!-- RELATED:END -->
