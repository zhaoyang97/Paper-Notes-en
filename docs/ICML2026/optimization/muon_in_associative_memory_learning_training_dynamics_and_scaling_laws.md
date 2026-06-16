---
title: >-
  [Paper Note] Muon in Associative Memory Learning: Training Dynamics and Scaling Laws
description: >-
  [ICML 2026][Optimization & Theory][Paper Note] This paper provides a theoretical characterization of the convergence rates and scaling laws of Muon on a linear associative memory model with softmax retrieval and hierarchical spectra. Compared to GD, Muon achieves exponential acceleration in the noiseless case and improves the loss convergence law from $\tilde{\Omeg
tags:
  - ICML 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: d960e51518cd7dd1
---
# Muon in Associative Memory Learning: Training Dynamics and Scaling Laws

**Conference**: ICML2026  
**arXiv**: [2602.05725](https://arxiv.org/abs/2602.05725)  
**Code**: Not publicly available  
**Area**: Optimization  
**Keywords**: Muon optimizer, associative memory, matrix sign operator, scaling laws, training dynamics  

## TL;DR
This paper provides a theoretical characterization of the convergence rates and scaling laws of Muon on a linear associative memory model with softmax retrieval and hierarchical spectra. Compared to GD, Muon achieves exponential acceleration in the noiseless case and improves the loss convergence law from $\tilde{\Omega}(T^{-(1-1/\beta)})$ to $\tilde{\mathcal{O}}(T^{-2})$ in the power-law spectral noise setting. This acceleration is attributed to the matrix sign operator acting as an adaptive, task-aligned implicit preconditioner.

## Background & Motivation

**Background**: In modern large-scale LLM pre-training, matrix-parameter optimizers have gradually transitioned from SGD/Adam/AdamW to Muon, proposed by Jordan et al. Muon has repeatedly demonstrated higher computational and data efficiency than AdamW in large-scale training regimes for architectures such as dense Transformers and MoE, leading to rapid adoption in industry.

**Limitations of Prior Work**: Existing theoretical literature almost exclusively treats Muon as a "standard stochastic optimization" problem to derive convergence upper bounds (Bernstein views Muon as steepest descent under the operator norm, while subsequent works provide gradient norm convergence rates). However, these static worst-case bounds fail to explain why Muon is "faster and more balanced" in real-world pre-training and do not provide Muon's own neural scaling law.

**Key Challenge**: Muon performs spectral normalization on matrix parameters via $\mathrm{msgn}(\mathbf{G})=\mathbf{U}\,\mathrm{sgn}(\boldsymbol{\Sigma})\,\mathbf{V}^\top$. Its essential effect is "amplifying" the step size for low-frequency long-tail tasks. In contrast, the effective step size of GD is proportional to the knowledge frequency $p_j$, resulting in extremely slow convergence $\sim 1/(p_j t)$ for tail tasks. To explain Muon's advantage, one must move beyond static bounds and instead characterize "how fast frequency components are learned" in dynamic training trajectories.

**Goal**: (1) Derive per-subtask and total loss curves for Muon vs. GD under both noiseless and label-noise associative memory settings; (2) Derive Muon's optimization scaling law under power-law spectra and compare it with the GD lower bound; (3) Provide a mechanistic perspective to explain Muon's acceleration.

**Key Insight**: The authors use associative memory as an analytically tractable proxy model—knowledge is organized into $K$ orthogonal query-answer pairs $(\mathbf{E}_j,\widetilde{\mathbf{E}}_j)$, appearing across $M$ hierarchical frequency groups $\tilde p_i$. The model is a single-matrix $\mathbf{W}\in\mathbb{R}^{K\times K}$ softmax retrieval. This framework faithfully simulates factual recall in Transformers (supported by experiments from Geva, Meng, et al.) and decomposes the gradient structure into "frequency × residual × association" terms, allowing Muon's SVD evolution to be tracked in closed form.

**Core Idea**: Muon’s matrix sign operation is approximately equal to the identity matrix $\mathbf{I}_K$ in the task representation basis (i.e., $\mathrm{msgn}(\mathbf{G}_t)\approx \mathbf{I}_K$). It "flattens" the frequency-skewed directional bias found in GD into isotropic updates, allowing high- and low-frequency groups to learn at the same rate, thereby replacing power-law integration with a fast $\mathcal{O}(T^{-2})$ decay.

## Method

This paper is a purely theoretical characterization and does not propose a new algorithm; the "Method" section corresponds to the construction of the theoretical framework and key proof strategies.

### Overall Architecture
The analysis focuses on the minimization problem under associative memory. Given $K$ orthogonal equal-norm embeddings $(\mathbf{E}_j,\widetilde{\mathbf{E}}_j)$, a frequency structure $p_j=\tilde p_i/C$ ($M$ frequency groups, each with $C=K/M$ entries), and a label noise level $\alpha\in[0,1)$ inducing a conditional distribution $p_{i\mid j}=(1-\alpha)\mathbb{1}[i=j]+\alpha/K$. The linear softmax model $\hat p_{i\mid j}(\mathbf{W})=\frac{\exp(\widetilde{\mathbf{E}}_i^\top \mathbf{W}\mathbf{E}_j)}{\sum_k \exp(\widetilde{\mathbf{E}}_k^\top \mathbf{W}\mathbf{E}_j)}$ stores knowledge by minimizing cross-entropy $\mathcal{L}(\mathbf{W})=\mathbb{E}_{\mathcal{D}_\alpha}[-\log\hat p_{i\mid j}(\mathbf{W})]$. The two optimizers are $\mathbf{W}_{t+1}=\mathbf{W}_t-\eta\nabla\mathcal{L}$ (GD) and $\mathbf{W}_{t+1}=\mathbf{W}_t-\eta\,\mathrm{msgn}(\nabla\mathcal{L})$ (Muon, neglecting momentum, equivalent to Spectral GD), both starting from zero initialization. The entire analysis tracks the evolution of $\widehat{\mathbf{W}}_t=\widetilde{\mathbf{E}}^\top \mathbf{W}_t \mathbf{E}$ and the corresponding gradient $\mathbf{G}_t=\widetilde{\mathbf{E}}^\top \nabla\mathcal{L} \mathbf{E}$ in the task representation space.

### Key Designs

**1. Gradient structure decomposition + frequency bottleneck: Pinpointing why GD is slow on long tails**

To explain why Muon is fast, one must first see where GD fails. The gradient of the softmax model can be written as the product of three factors:

$$\nabla\mathcal{L}(\mathbf{W})=\sum_{i,j} p_j\,(\hat p_{i\mid j}-p_{i\mid j})\,\widetilde{\mathbf{E}}_i\mathbf{E}_j^\top,$$

which represents "query frequency $p_j$ × prediction residual × embedding association." This is the foundation for all subsequent theorems. Along the $j$-th component, GD's effective step size is proportional to $p_j$, so in the noiseless case, the per-subtask loss is $\mathcal{L}_j^{\mathrm{GD}}(t)\eqsim 1/(p_j t)$ and total loss is $\eqsim K/t$ (Theorem 4.1); long-tail classes are bottlenecked by this $p_j$ factor. Muon's $\mathrm{msgn}$ performs spectral normalization that exactly strips away $p_j$, allowing all subtasks to converge at the same exponential rate $\mathcal{L}_j^{\mathrm{Muon}}(t)\eqsim Ke^{-(1+o_K(1))t}$ (Theorem 4.2). To reduce loss to a target precision, GD requires $\mathcal{O}(1/\epsilon)$ steps, while Muon only requires $\mathcal{O}(\log(1/\epsilon))$ steps.

**2. Three-stage dynamics + Muon scaling law $\tilde{\mathcal{O}}(T^{-2})$: Formulating Muon's own neural scaling law**

After establishing exponential convergence in the noiseless case, label noise is added to align with real pre-training. Muon’s subtask loss exhibits two stages: a decline phase $\sim Ke^{-\eta t}+\eta t$ and an oscillation phase $\sim\eta^2+\mathcal{L}_j^\ast$, with a critical time $T_j^\ast=\Theta(\log K/\eta)$ (Theorem 5.1). Selecting $\eta=\Theta(\log K/T)$ optimally balances the decline term $Ke^{-\eta T}$ and the oscillation term $\eta^2$, yielding $\mathcal{L}^{\mathrm{Muon}}(T)-\mathcal{L}^\ast\lesssim(\log K/T)^2$ (Theorem 5.8). Given the same power-law spectrum $\tilde p_i\propto i^{-\beta}$ ($\beta>1$), GD's per-subtask loss is $\gtrsim e^{-\eta p_j T}\log K$. Summing over $j$ and approximating with the integral $\int_1^M z^{-\beta}e^{-z^{-\beta}T}\mathrm{d}z\approx T^{-(1-1/\beta)}$ gives a lower bound of $\tilde \Omega(T^{-(1-1/\beta)})$ (Theorem 5.7). Comparison shows that Muon's scaling exponent of $-2$ is independent of $\beta$, whereas GD's $-(1-1/\beta)$ degrades to 0 as $\beta\to 1$. To reach the same precision in total loss, Muon is $\Omega(C)$ times faster (by group size) than GD—providing a formal explanation for the steeper loss-compute curves observed for Muon in large-scale pre-training.

**3. Preconditioning perspective: $\mathrm{msgn}\approx\mathbf{I}_K$ as implicit alignment to task representation basis**

The acceleration mechanism can be explained even more clearly: in the task representation space, $\mathrm{msgn}(\mathbf{G}_t)\approx\mathbf{I}_K$, meaning Muon is effectively performing isotropic alignment updates where $\widehat{\mathbf{W}}_t\approx t\mathbf{I}_K$. Inductive proof shows that starting from $\mathbf{W}_0=\mathbf{0}$, Muon preserves the block-symmetry structure induced by frequency groups (Proposition 6.1). The residual decomposes as $\mathbf{P}-\widehat{\mathbf{P}}_t=\mathbf{R}_t^+-\mathbf{R}_t^-$. On the $M(C-1)$-dimensional intra-group contrastive subspace, $\mathrm{msgn}$ degenerates to the identity matrix, and only the $M$-dimensional block-mean directions contribute at most $M/C$ deviation. Thus, $\|\mathrm{msgn}(\mathbf{P}-\widehat{\mathbf{P}}_t)-\mathbf{I}_K\|_{\max}\le 1/C+M/C=o_K(1)$. Comparing this to an idealized TRA-SignGD (updating $\widehat{\mathbf{W}}_{t+1}=\widehat{\mathbf{W}}_t-\eta\,\mathrm{sgn}(\mathbf{G}_t)$), Theorem 6.3 proves it can match all Muon conclusions using $\eta$ instead of $2\eta$. The distinction is clear: to apply signs in the original coordinate system, SignGD needs an oracle to know the unknown $\mathbf{E},\widetilde{\mathbf{E}}$ for alignment, whereas Muon uses SVD to automatically find this task representation basis without an oracle—attributing the advantage precisely to SVD's automatic alignment capability rather than vague "matrix optimization" superiority.

### Loss & Training
All theoretical results are based on zero initialization $\mathbf{W}_0=\mathbf{0}_{K\times K}$ and a constant learning rate $\eta$. The scaling law section uses $\eta=\Theta(\log K/T)$. The stability condition for GD, $\eta p_1\lesssim 1$, is given by the linear stability of the fixed-point Jacobian (Proposition 5.4).

## Key Experimental Results

Experiments serve as sanity checks, using synthetic long-tail classification and LLaMA-style pre-training to verify theoretical predictions.

### Main Results

| Setting | Spectrum / Data | Muon Behavior | GD Behavior |
|------|-------------|----------|---------|
| Noiseless associative memory | $K$ orthogonal facts, $M$ groups | Subtasks converge exponentially in sync, $\mathcal{L}^{\mathrm{Muon}}\eqsim K e^{-t}$ | Rate $\propto p_j$, total loss $\eqsim K/t$, low-frequency groups stall |
| Noisy power-law spectrum $\tilde p_i\propto i^{-\beta}$ | $\beta>1$ | $\mathcal{L}^{\mathrm{Muon}}(T)-\mathcal{L}^\ast\lesssim (\log K/T)^2$ | $\mathcal{L}^{\mathrm{GD}}(T)-\mathcal{L}^\ast\gtrsim \log K/T^{1-1/\beta}$ |
| LLaMA-style pre-training | Real long-tail text | Significantly higher tail accuracy, steeper scaling curve | Slower convergence, under-learning of tail classes |

### Ablation Study

| Configuration | Behavior | Explanation |
|------|------|------|
| GD (baseline) | Frequency sensitive $1/(p_j t)$ | Low-frequency groups stall, scaling limited by $\beta$ |
| Normalized GD (NGD) | Faster than GD but still unbalanced | Shows acceleration is not just step-size normalization; matrix-sign is required |
| SignGD (original coordinates) | Cannot exploit task structure | Requires oracle $\mathbf{E},\widetilde{\mathbf{E}}$ to match Muon |
| TRA-SignGD (idealized alignment) | Matches Muon using $2\eta$ results with $\eta$ | Validates that Muon’s advantage comes from "automatically finding task basis" implicit preconditioning |
| Muon (momentum-free) | Achieves exponential acceleration + $\tilde{\mathcal{O}}(T^{-2})$ scaling | Matrix sign strips $p_j$ factor from the effective step size |

### Key Findings
- Muon's acceleration can be split into two parts: spectral normalization which flattens update scales (already partially achievable by NGD), and implicit alignment of $\mathrm{msgn}$ along the task representation basis (not achievable by NGD). The latter is the source of the $\Omega(C)$ acceleration factor.
- The trade-off between the oscillation term $\eta^2$ introduced by label noise and the decline term $Ke^{-\eta T}+\eta T$ determines the optimal $\eta=\Theta(\log K/T)$. This schedule naturally yields Muon’s scaling exponent of $-2$.
- The scaling exponent of GD, $-(1-1/\beta)$, degrades to 0 as the power-law index $\beta$ approaches 1, performing particularly poorly on long-tail data. Muon's exponent is independent of $\beta$, which is the fundamental reason its scaling is steeper on real-world corpora.

## Highlights & Insights
- This work derives the first "native" neural scaling law for Muon, rather than re-adjusting worst-case bounds from SGD/SignGD. The contrast between the $-2$ exponent and GD's $-(1-1/\beta)$ provides a non-trivial explanation for why Muon is more effective in large-scale training, formalizing engineering intuition into an asymptotic law.
- The technical combination of associative memory + block-symmetric induction + task representation space is highly reusable. As long as the gradient has a decomposable "frequency × residual × association" structure, any optimizer using "SVD sign" can use this framework to calculate scaling laws and acceleration factors, providing a template for future spectral methods (Shampoo, SOAP, Spectral GD).
- The design of the TRA-SignGD idealized baseline is clever: by "lending" it the task representation basis while keeping the SignGD form, it proves the equivalence of Muon and "aligned SignGD," precisely attributing Muon's success to SVD's automatic alignment capability rather than general claims about matrix optimization.

## Limitations & Future Work
- The model is a single-matrix linear softmax, lacking non-linearity, MLPs, and multi-head structures. The scaling laws derived here strictly cover factual recall subtasks and cannot be directly extrapolated to token-level generative perplexity.
- Embeddings are assumed to be orthogonal and equal-norm (though the authors claim this can be relaxed to near-orthogonal). In real pre-training, token embeddings are rarely so clean. The power-law spectrum is also only a first-order approximation of Zipf's law.
- Practical implementations of Muon include momentum (using Newton-Schulz iteration to estimate $\mathrm{msgn}$). To purify the analysis, momentum was removed in this paper; there is a gap between this and production Muon (the momentum-free version is also known as Spectral GD).
- Future Work: Extending this block-symmetric analysis to (i) layer-wise associative memory stacking in multi-layer Transformers, (ii) expert routing frequencies in MoE, and (iii) scaling laws under momentum and learning rate scheduling.

## Related Work & Insights
- **vs Bernstein & Newhouse 2024 / Li & Hong 2025 / Pethick et al. 2025**: These provide worst-case stochastic convergence bounds for Muon. This paper provides problem-specific training dynamics in closed form + scaling laws, quantitatively explaining the acceleration factor.
- **vs Wang et al. 2025b (heavy-tailed associative memory)**: That work was experimental, finding Muon strong on tail classes. This paper proves that phenomenon as an $\Omega(C)$ acceleration and identifies the mechanism (implicit preconditioning).
- **vs Kunstner & Bach 2025 / Kim et al. 2026 (SignSGD scaling law)**: Extends the scaling analysis framework of SignGD methods from bigrams to associative memory, adding the critical distinction between $\mathrm{msgn}$ and $\mathrm{sgn}$ regarding the need for "oracle alignment."
- **vs Vasudeva et al. 2025 (Muon generalization in Gaussian mixture)**: That focus was on generalization; this paper focuses on optimization dynamics and scaling, making them complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First native neural scaling law for Muon; reveals the implicit alignment mechanism of $\mathrm{msgn}\approx \mathbf{I}_K$.
- Experimental Thoroughness: ⭐⭐⭐ Primarily relies on synthetic data and small-scale LLaMA training; lacks large-scale ablation.
- Writing Quality: ⭐⭐⭐⭐ Theorem layouts are clear; the preconditioning view section explains intuition thoroughly.
- Value: ⭐⭐⭐⭐⭐ Provides a direct template for theoretical analysis of future spectral optimizers and justifies the scaling gains of Muon engineering-wise.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](../../NeurIPS2025/optimization/learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)
- [\[NeurIPS 2025\] Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules](../../NeurIPS2025/optimization/functional_scaling_laws_in_kernel_regression_loss_dynamics_and_learning_rate_sch.md)
- [\[ICML 2026\] Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks](balancing_learning_rates_across_layers_exact_two-step_dynamics_and_optimal_scali.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[ICLR 2026\] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](../../ICLR2026/optimization/scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)

</div>

<!-- RELATED:END -->
