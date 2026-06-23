---
title: >-
  [Paper Note] Muon in Associative Memory Learning: Training Dynamics and Scaling Laws
description: >-
  [ICML 2026][Optimization & Theory][Paper Note] This paper provides a theoretical characterization of convergence rates and scaling laws for Muon on a linear associative memory model with softmax retrieval and hierarchical spectra: compared to GD, Muon achieves exponential acceleration in the noiseless case and improves the loss scaling law from $\tilde{\Omega}(T^{-
tags:
  - ICML 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 81fdb58897dfeb4d
---
# Muon in Associative Memory Learning: Training Dynamics and Scaling Laws

**Conference**: ICML2026  
**arXiv**: [2602.05725](https://arxiv.org/abs/2602.05725)  
**Code**: Not disclosed  
**Area**: Optimization  
**Keywords**: Muon optimizer, associative memory, matrix sign operator, scaling laws, training dynamics  

## TL;DR
This paper provides a theoretical characterization of convergence rates and scaling laws for Muon on a linear associative memory model with softmax retrieval and hierarchical spectra: compared to GD, Muon achieves exponential acceleration in the noiseless case and improves the loss scaling law from $\tilde{\Omega}(T^{-(1-1/\beta)})$ to $\tilde{\mathcal{O}}(T^{-2})$ in the power-law spectrum noise case, attributing this acceleration to the matrix sign operator acting as an adaptive task-aligned implicit preconditioner.

## Background & Motivation

**Background**: In modern LLM large-scale pre-training, matrix parameter optimizers have gradually transitioned from SGD/Adam/AdamW to Muon, proposed by Jordan et al. Muon consistently demonstrates higher computational and data efficiency than AdamW in large-scale training regimes across architectures like dense Transformers and MoEs, leading to rapid adoption in engineering.

**Limitations of Prior Work**: Most existing theoretical literature treats Muon as a "standard stochastic optimization" problem to derive a convergence upper bound (Bernstein views Muon as steepest descent under the operator norm; subsequent work gives gradient norm convergence rates). However, such static worst-case bounds cannot explain why Muon is "faster and more balanced" in real pre-training, nor do they provide a neural scaling law specific to Muon.

**Key Challenge**: Muon performs spectral normalization on matrix parameters as $\mathrm{msgn}(\mathbf{G})=\mathbf{U}\,\mathrm{sgn}(\boldsymbol{\Sigma})\,\mathbf{V}^\top$, the essential effect of which is to "amplify" steps on low-frequency long-tail tasks. In contrast, the effective step size of GD is proportional to the knowledge frequency $p_j$, causing extremely slow convergence $\sim 1/(p_j t)$ for tail tasks. To explain Muon's advantages, one must move beyond static bounds to characterize "how fast frequency components are learned" during dynamic training trajectories.

**Goal**: (1) Derive sub-task and total loss curves for Muon vs. GD under both noiseless and label-noise associative memory settings; (2) Derive the optimization scaling law for Muon under power-law spectra and compare it with the GD lower bound; (3) Provide a mechanistic perspective explaining Muon's acceleration.

**Key Insight**: The authors use associative memory as an analytically tractable proxy model—knowledge is organized into $K$ orthogonal query-answer pairs $(\mathbf{E}_j,\widetilde{\mathbf{E}}_j)$ appearing with $M$ groups of hierarchical frequencies $\tilde p_i$; the model is a softmax retrieval of a single matrix $\mathbf{W}\in\mathbb{R}^{K\times K}$. This framework faithfully simulates factual recall in Transformers (backed by experiments from Geva and Meng) and decomposes the gradient structure into "frequency × residual × association" terms, allowing closed-form tracking of Muon's SVD evolution.

**Core Idea**: Muon's matrix sign operation is approximately equal to the identity matrix $\mathbf{I}_K$ in the task representation basis (i.e., $\mathrm{msgn}(\mathbf{G}_t)\approx \mathbf{I}_K$). This "flattens" the directional bias skewed by frequency in GD into an isotropic update, allowing high-frequency and low-frequency groups to learn at the same rate, thereby replacing the power-law integral with a fast $\mathcal{O}(T^{-2})$ decay.

## Method

This paper is a purely theoretical characterization and does not propose a new algorithm; the details correspond to the construction of the theoretical framework and key proof strategies.

### Overall Architecture
The object of analysis is the minimization problem under associative memory. Given $K$ orthogonal equal-norm embeddings $(\mathbf{E}_j,\widetilde{\mathbf{E}}_j)$, frequency structure $p_j=\tilde p_i/C$ ($M$ frequency groups, each with $C=K/M$ entries), and label noise level $\alpha\in[0,1)$ inducing a conditional distribution $p_{i\mid j}=(1-\alpha)\mathbb{1}[i=j]+\alpha/K$. The linear softmax model $\hat p_{i\mid j}(\mathbf{W})=\frac{\exp(\widetilde{\mathbf{E}}_i^\top \mathbf{W}\mathbf{E}_j)}{\sum_k \exp(\widetilde{\mathbf{E}}_k^\top \mathbf{W}\mathbf{E}_j)}$ stores knowledge by minimizing cross-entropy $\mathcal{L}(\mathbf{W})=\mathbb{E}_{\mathcal{D}_\alpha}[-\log\hat p_{i\mid j}(\mathbf{W})]$. The two optimizers are $\mathbf{W}_{t+1}=\mathbf{W}_t-\eta\nabla\mathcal{L}$ (GD) and $\mathbf{W}_{t+1}=\mathbf{W}_t-\eta\,\mathrm{msgn}(\nabla\mathcal{L})$ (Muon, omitting momentum, equivalent to Spectral GD), both initialized from zero. The analysis tracks the evolution of $\widehat{\mathbf{W}}_t=\widetilde{\mathbf{E}}^\top \mathbf{W}_t \mathbf{E}$ and the corresponding gradient $\mathbf{G}_t=\widetilde{\mathbf{E}}^\top \nabla\mathcal{L} \mathbf{E}$ in task representation space.

### Key Designs

**1. Gradient Structure Decomposition + Frequency Bottleneck Characterization: Locating why GD is slow on long tails**

To explain why Muon is fast, one must first see where GD is slow. The gradient of the softmax model can be written as the product of three factors:

$$\nabla\mathcal{L}(\mathbf{W})=\sum_{i,j} p_j\,(\hat p_{i\mid j}-p_{i\mid j})\,\widetilde{\mathbf{E}}_i\mathbf{E}_j^\top,$$

i.e., "query frequency $p_j$ × prediction residual × embedding association," which is the foundation of all subsequent theorems. Along the $j$-th component, GD's effective step size is proportional to $p_j$, leading to a sub-task loss $\mathcal{L}_j^{\mathrm{GD}}(t)\eqsim 1/(p_j t)$ and total loss $\eqsim K/t$ in the noiseless case (Theorem 4.1); long-tail classes are bottlenecked by this $p_j$ factor. Muon's $\mathrm{msgn}$ performs spectral normalization that strips away $p_j$, allowing all sub-tasks to converge at the same exponential rate $\mathcal{L}_j^{\mathrm{Muon}}(t)\eqsim Ke^{-(1+o_K(1))t}$ (Theorem 4.2). To reduce loss to target accuracy, GD requires $\mathcal{O}(1/\epsilon)$ steps, while Muon requires only $\mathcal{O}(\log(1/\epsilon))$.

**2. Three-Stage Dynamics + Muon Scaling Law $\tilde{\mathcal{O}}(T^{-2})$: Deriving Muon's neural scaling law**

After establishing exponential convergence for the noiseless case, label noise is added to align with real pre-training. Muon's sub-task loss follows two stages: a descent phase $\sim Ke^{-\eta t}+\eta t$ and an oscillation phase $\sim\eta^2+\mathcal{L}_j^\ast$, with a critical time $T_j^\ast=\Theta(\log K/\eta)$ (Theorem 5.1). Selecting $\eta=\Theta(\log K/T)$ balances the descent term $Ke^{-\eta T}$ and the oscillation term $\eta^2$, yielding $\mathcal{L}^{\mathrm{Muon}}(T)-\mathcal{L}^\ast\lesssim(\log K/T)^2$ (Theorem 5.8). For the same power-law spectrum $\tilde p_i \propto i^{-\beta}$ ($\beta>1$), GD sub-tasks behave as $\gtrsim e^{-\eta p_j T}\log K$; summing over $j$ and approximating with an integral $\int_1^M z^{-\beta}e^{-z^{-\beta}T}\mathrm{d}z\approx T^{-(1-1/\beta)}$ gives a lower bound $\tilde\Omega(T^{-(1-1/\beta)})$ (Theorem 5.7). Comparing the two, Muon's scaling exponent $-2$ is independent of $\beta$, whereas GD's $-(1-1/\beta)$ degrades to 0 as $\beta\to 1$. To reach the same total loss accuracy, Muon is $\Omega(C)$ (group size) times faster than GD—providing a formal explanation for Muon's steeper loss-compute curves in large-scale pre-training.

**3. Preconditioning View: $\mathrm{msgn}\approx\mathbf{I}_K$ is implicit alignment that automatically finds the task representation basis**

The acceleration mechanism is further clarified: in the task representation space, $\mathrm{msgn}(\mathbf{G}_t)\approx\mathbf{I}_K$, meaning Muon performs isotropic aligned updates $\widehat{\mathbf{W}}_t\approx t\mathbf{I}_K$. Inductive proof shows that Muon, starting from $\mathbf{W}_0=\mathbf{0}$, preserves the block-symmetric structure induced by frequency groups (Proposition 6.1). The residual decomposes as $\mathbf{P}-\widehat{\mathbf{P}}_t=\mathbf{R}_t^+-\mathbf{R}_t^-$. On the $M(C-1)$-dimensional intra-group contrastive subspace, $\mathrm{msgn}$ degrades to the identity matrix, with only the $M$-dimensional block-mean directions contributing at most $M/C$ deviation. Thus, $\|\mathrm{msgn}(\mathbf{P}-\widehat{\mathbf{P}}_t)-\mathbf{I}_K\|_{\max}\le 1/C+M/C=o_K(1)$. Comparing this to an idealized TRA-SignGD (updating $\widehat{\mathbf{W}}_{t+1}=\widehat{\mathbf{W}}_t-\eta\,\mathrm{sgn}(\mathbf{G}_t)$), Theorem 6.3 proves it matches all Muon conclusions using $\eta$ with Muon's $2\eta$. The distinction is clear: SignGD needs an oracle to know $\mathbf{E},\widetilde{\mathbf{E}}$ to align and perform signs in original coordinates, whereas Muon automatically finds this task representation basis via SVD without an oracle. The advantage is precisely attributed to SVD's automatic alignment capability rather than a vague "matrix optimization is stronger" claim.

### Loss & Training
All theoretical results are based on zero initialization $\mathbf{W}_0=\mathbf{0}_{K\times K}$ and a constant learning rate $\eta$; the scaling law section takes $\eta=\Theta(\log K/T)$. The stability condition for GD, $\eta p_1\lesssim 1$, is given by the linear stability of the fixed-point Jacobian (Proposition 5.4).

## Key Experimental Results

Experiments serve as a sanity check, verifying theoretical predictions using synthetic long-tail classification and LLaMA-style pre-training.

### Main Results

| Setup | Spectrum / Data | Muon Behavior | GD Behavior |
|------|-------------|----------|---------|
| Noiseless associative memory | $K$ orthogonal knowledge, $M$ groups | Synchronous exponential sub-task convergence, $\mathcal{L}^{\mathrm{Muon}}\eqsim K e^{-t}$ | Rates $\propto p_j$, total loss $\eqsim K/t$, low-frequency groups stuck |
| Noisy power-law spectrum $\tilde p_i\propto i^{-\beta}$ | $\beta>1$ | $\mathcal{L}^{\mathrm{Muon}}(T)-\mathcal{L}^\ast\lesssim (\log K/T)^2$ | $\mathcal{L}^{\mathrm{GD}}(T)-\mathcal{L}^\ast\gtrsim \log K/T^{1-1/\beta}$ |
| LLaMA-style pre-training | Real long-tail text | Significantly higher long-tail accuracy, steeper scaling curve slope | Slower convergence, under-learning of tail classes |

### Ablation Study

| Configuration | Behavior | Description |
|------|------|------|
| GD (baseline) | Frequency sensitive $1/(p_j t)$ | Low-frequency groups stuck, scaling limited by $\beta$ |
| Normalized GD (NGD) | Faster than GD but still unbalanced | Shows acceleration is not just step normalization; matrix-sign is required |
| SignGD (Original Coords) | Cannot exploit task structure | Requires oracle $\mathbf{E},\widetilde{\mathbf{E}}$ to match Muon |
| TRA-SignGD (Idealized Alignment) | Matches Muon conclusion with $\eta$ vs $2\eta$ | Validates that Muon superiority stems from implicit preconditioning (auto-alignment) |
| Muon (no momentum) | High-speed exponential acceleration + $\tilde{\mathcal{O}}(T^{-2})$ scaling | Matrix sign strips the $p_j$ factor from the effective step size |

### Key Findings
- Muon's acceleration is split into two parts: spectral normalization flattens update scales (which NGD also does), and $\mathrm{msgn}$ provides implicit alignment along the task representation basis (which NGD does not). The latter is the source of the $\Omega(C)$ speedup.
- The trade-off between the oscillation term $\eta^2$ introduced by label noise and the descent term $Ke^{-\eta T}+\eta T$ determines the optimal $\eta=\Theta(\log K/T)$. This schedule naturally yields Muon's scaling exponent of $-2$.
- GD's scaling exponent $-(1-1/\beta)$ degrades to 0 as the power-law index $\beta$ approaches 1, performing particularly poorly on long tails. Muon's exponent is independent of $\beta$, which is the fundamental reason for its steeper scaling on real corpora.

## Highlights & Insights
- This work provides Muon's first "own" neural scaling law rather than recycling worst-case bounds from SGD/SignGD. The scaling exponent of $-2$ vs. GD's $-(1-1/\beta)$ creates a non-trivial contrast, formalizing the engineering experience that "Muon is more effective for large-scale training."
- The combination of associative memory + block-symmetric induction + task representation space is highly reusable: provided gradients have the "frequency × residual × association" structure, any "sign-by-SVD" optimizer can use this framework to calculate scaling laws and speedup factors. This serves as a template for future spectral methods (Shampoo, SOAP, Spectral GD).
- The TRA-SignGD idealized baseline is cleverly designed: it "lends" the task representation basis to SignGD, proving Muon is equivalent to "aligned SignGD." This precisely attributes Muon's advantage to SVD's automatic alignment capability rather than general matrix optimization benefits.

## Limitations & Future Work
- The model is a single-matrix linear softmax lacking non-linearity, MLPs, or multi-head structures. The scaling laws strictly cover factual recall sub-tasks and cannot be directly extrapolated to token-level perplexity in generation.
- Embeddings assume orthogonal equal norms (though authors claim this can be relaxed to near-orthogonal). Real pre-training token embeddings are not this clean, and power-law spectra are only first-order approximations of Zipf's law.
- Practical Muon implementations include momentum (using Newton-Schulz iteration to estimate $\mathrm{msgn}$). This analysis removes momentum for purity, creating a gap with production Muon; the momentum-free version is equivalent to Spectral GD.
- Future work: Extending the block-symmetric analysis to (i) layer-wise associative memory in multi-layer Transformers, (ii) expert routing frequencies in MoEs, and (iii) scaling laws under momentum and learning rate scheduling.

## Related Work & Insights
- **vs. Bernstein & Newhouse 2024 / Li & Hong 2025 / Pethick et al. 2025**: These provide worst-case stochastic convergence bounds for Muon. This paper provides problem-specific closed-form dynamics and scaling laws, quantitatively explaining the speedup factor.
- **vs. Wang et al. 2025b (heavy-tailed associative memory)**: That work was experimental, observing Muon's strength in tail classes. This paper proves it as an $\Omega(C)$ speedup and provides the mechanism (implicit preconditioning).
- **vs. Kunstner & Bach 2025 / Kim et al. 2026 (SignSGD scaling law)**: Extends scaling analysis for SignGD-like methods from bigrams to associative memory, adding a critical distinction between $\mathrm{msgn}$ and $\mathrm{sgn}$ regarding the need for oracle alignment.
- **vs. Vasudeva et al. 2025 (Generalization of Muon in Gaussian mixture)**: That focus is generalization; this paper focuses on optimization dynamics and scaling, making them complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First neural scaling law for Muon; reveals the "$\mathrm{msgn}\approx \mathbf{I}_K$" implicit alignment mechanism.
- Experimental Thoroughness: ⭐⭐⭐ Primarily relies on synthetic data and small-scale LLaMA training; lacks large-scale ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear theorem layout; the preconditioning view section clarifies the intuition.
- Value: ⭐⭐⭐⭐⭐ Provides a template for theoretical analysis of future spectral optimizers; formally supports the scaling benefits of Muon.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](../../NeurIPS2025/optimization/learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)
- [\[NeurIPS 2025\] Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules](../../NeurIPS2025/optimization/functional_scaling_laws_in_kernel_regression_loss_dynamics_and_learning_rate_sch.md)
- [\[ICML 2026\] Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks](balancing_learning_rates_across_layers_exact_two-step_dynamics_and_optimal_scali.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[ICML 2026\] Mirror Mean-Field Langevin Dynamics](mirror_mean-field_langevin_dynamics.md)

</div>

<!-- RELATED:END -->
