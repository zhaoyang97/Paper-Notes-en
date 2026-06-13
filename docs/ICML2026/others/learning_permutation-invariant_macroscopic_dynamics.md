---
title: >-
  [Paper Note] Learning Permutation-Invariant Macroscopic Dynamics
description: >-
  [ICML2026][Permutation-invariant closure variables] For particle systems with inherently unordered microscopic states, this paper proposes an autoencoder framework focused on "reconstructing density rather than particles…
tags:
  - "ICML2026"
  - "Permutation-invariant closure variables"
  - "Distribution reconstruction"
  - "DeepSet encoder"
  - "Conditional normalizing flows"
  - "Macroscopic dynamics"
date: 2026-05-08
content_hash: ef85f40a9a520b02
---

# Learning Permutation-Invariant Macroscopic Dynamics

**Conference**: ICML2026  
**arXiv**: [2605.30812](https://arxiv.org/abs/2605.30812)  
**Code**: Not yet available  
**Area**: Scientific Computing / Closure Modeling / Set Representation Learning  
**Keywords**: Permutation-invariant closure variables, Distribution reconstruction, DeepSet encoder, Conditional normalizing flows, Macroscopic dynamics

## TL;DR
For particle systems with inherently unordered microscopic states, this paper proposes an autoencoder framework focused on "reconstructing density rather than particles." It utilizes a DeepSet encoder to obtain permutation-invariant closure variables $\hat{\bm{z}}$, and a conditional normalizing flow targeting a Gaussian mixture density centered at observation points. This avoids point-cloud matching and allows the closure variables to be learned alongside macroscopic observables via an SDE/ODE.

## Background & Motivation

**Background**: In scientific computing, it is often necessary to compress high-dimensional microscopic states $X_t = \{\bm{x}_t^1,\dots,\bm{x}_t^n\}$ into low-dimensional "closure variables" to predict the evolution of macroscopic quantities (energy, mixing ratios, polymer extension). The mainstream approach involves MLP/CNN autoencoders with point-wise MSE reconstruction loss, treating latent variables as closure variables.

**Limitations of Prior Work**: These approaches assume a fixed ordering of inputs. While suitable for grid-based PDEs, this fails for interacting particle systems. The same physical configuration under different indices is treated as different vectors. Point-wise MSE distinguishes between $(\hat{\bm{x}}^1,\hat{\bm{x}}^2,\hat{\bm{x}}^3)$ and $(\hat{\bm{x}}^3,\hat{\bm{x}}^2,\hat{\bm{x}}^1)$, meaning latent variables are not permutation-invariant.

**Key Challenge**: While the encoder can be forced into invariance using DeepSet or Set Transformers, the decoder lacks a "natural order." Forcing the decoder to output an ordered set for point-wise loss essentially crams $n!$ equivalent permutations into one target. This requires either expensive Hungarian matching or unstable permutation-invariant distances like Chamfer/EMD, which can blur point-level supervision.

**Goal**: (i) Learn closure variables strictly invariant to input ordering; (ii) avoid reliance on point-to-point matching; (iii) jointly learn the stochastic dynamics of macroscopic observables with generalization across different particle counts.

**Key Insight**: Rather than reconstructing "which particle is where," it is better to reconstruct the "spatial density distribution of the particles as a whole." Each set $X$ induces a Gaussian mixture $q_X(\mathbf{x})$ centered at observation points, which is then fitted using a conditional normalizing flow. Density is naturally invariant to particle indexing, completely bypassing the matching problem.

**Core Idea**: Replace "set reconstruction" with "reconstruction of the distribution induced by the set." This target is inherently permutation-invariant, and decoder complexity is decoupled from $n$.

## Method

### Overall Architecture
The input is a microscopic particle set $X_t \in \mathcal{X}$ at time $t$. A **pre-defined deterministic function** $\bar{\bm{\varphi}}$ extracts the macroscopic quantities of interest $\bar{\bm{z}}_t$ (e.g., average energy, A-B neighbor ratio $(R_{AB},R_{BA})$). A **learned DeepSet encoder** $\hat{\bm{\varphi}}$ extracts permutation-invariant closure variables $\hat{\bm{z}}_t$. These are concatenated into an augmented macroscopic state $\bm{z}_t = [\bar{\bm{z}}_t, \hat{\bm{z}}_t]$. An SDE (or ODE) is learned on $\bm{z}_t$ to predict future macroscopic states. Training occurs in two stages: first, learning $(\hat{\bm{\varphi}}, \bm{\psi})$ using distribution reconstruction loss; second, freezing $\hat{\bm{\varphi}}$ to learn the dynamics $(\bm{g}, \bm{\Sigma})$.

### Key Designs

1. **DeepSet Encoder → Permutation-Invariant Closure Variables**:
    - **Function**: Maps an unordered particle set $X = \{\bm{x}^i\}_{i=1}^n$ to a low-dimensional vector $\hat{\bm{z}} = \hat{\bm{\varphi}}(X)$, strictly satisfying $\hat{\bm{\varphi}}(\sigma X) = \hat{\bm{\varphi}}(X), \forall \sigma \in S_n$.
    - **Mechanism**: The DeepSet paradigm where "each particle passes through an MLP independently, followed by symmetric pooling (sum/mean), and another MLP"; complexity is $\mathcal{O}(n)$. This can be replaced by a Set Transformer. Crucially, invariance is built into the architecture rather than approximated via data augmentation.
    - **Design Motivation**: Existing works either use random permutation augmentation on MLP encoders (AE-Aug, which is not strictly invariant) or use DeepSet with an MSE decoder (AE-InvE, which the authors show breaks invariance at the decoder). This work seeks end-to-end structural invariance.

2. **Distribution Reconstruction Objective → Replacing Point-to-Point Matching**:
    - **Function**: Transitions from "set reconstruction" to "density reconstruction," making the objective itself permutation-invariant.
    - **Mechanism**: For each $X$, a Gaussian mixture $q_X(\mathbf{x}) = \frac{1}{|X|}\sum_{\bm{x}^i \in X}\delta_\epsilon(\mathbf{x} - \bm{x}^i)$ is induced with bandwidth $\epsilon$. The decoder $\bm{\psi}$ is a conditional normalizing flow $p_\theta(\mathbf{x}\mid\hat{\bm{z}})$ that minimizes $\mathcal{L}_{\mathrm{rec}} = \mathbb{E}_X[\mathrm{KL}(q_X \,\|\, p_\theta(\cdot\mid\hat{\bm{z}}))]$. The KL divergence is estimated using MC samples from $q_X$. Since $q_X$ is a mixture of identical Gaussians, sampling is highly parallelizable.
    - **Design Motivation**: Traditional point cloud reconstruction uses Hungarian matching ($\mathcal{O}(n^3)$) or Chamfer/EMD (noisy gradients, unstable optimization). By using distribution loss, decoder complexity is $\mathcal{O}(1)$ relative to $n$. Bandwidth $\epsilon$ acts as a "resolution knob": small $\epsilon$ preserves detail but requires higher $\hat{z}_{\mathrm{dim}}$; large $\epsilon$ is smoother and works with smaller $\hat{z}_{\mathrm{dim}}$.

3. **Augmented SDE + Two-Stage Training → Joint Macroscopic Dynamics**:
    - **Function**: Learns closed dynamics $\mathrm{d}\bm{z}_t = \bm{g}(\bm{z}_t)\mathrm{d}t + \bm{\Sigma}(\bm{z}_t)\mathrm{d}\bm{W}_t$ on $\bm{z}_t = [\bar{\bm{z}}_t, \hat{\bm{z}}_t]$.
    - **Mechanism**: Macroscopic quantities $\bar{\bm{z}}$ are concatenated with learned closure variables $\hat{\bm{z}}$. Drift $\bm{g}$ and diffusion $\bm{\Sigma}$ (both MLPs) learn the one-step conditional distribution $p_{\bm{g},\bm{\Sigma}}(\bm{z}_{t+1}\mid\bm{z}_t) = \mathcal{N}(\bm{z}_t + \bm{g}\Delta t,\,\Delta t\,\bm{\Sigma}\bm{\Sigma}^\top)$ by minimizing the negative log-likelihood $\mathcal{L}_{\mathrm{dyn}}$. In deterministic cases, this reduces to an ODE with MSE loss.
    - **Design Motivation**: Using only $\bar{\bm{z}}$ is often not closed (macroscopic evolution depends on microscopic degrees of freedom). Thus, $\hat{\bm{z}}$ must represent necessary microscopic information. Since reconstruction-free end-to-end methods are prone to representation collapse, the reconstruction loss is retained as a regularizer. Two-stage training prevents interference between objectives.

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_{\mathrm{rec}} + \lambda_{\mathrm{dyn}}\mathcal{L}_{\mathrm{dyn}}$, trained sequentially. KL estimation uses a fixed number of MC samples from $q_X$, making training/inference costs insensitive to $n$. At inference, the encoder is used once (to construct $\bm{z}_0$), followed by autoregressive extrapolation by the dynamics model.

## Key Experimental Results

### Main Results

Three microscopic scenarios: (i) Energy evolution of 2D interacting particles (Deterministic, ODE); (ii) Lennard-Jones binary particle mixing (Stochastic, SDE); (iii) Polymer deformation in extensional flow (Video input, ODE). Tests include: in-dst, diff-init (initial pattern shifts), and diff-N (particle count shifts).

| Task | Regime | AE-Aug | AE-InvE | AE-InvE-CD | InvE | Ours |
|------|------|--------|---------|------------|------|------|
| Particle Energy (MRE ↓) | in-dst | $1.25 \times 10^{-3}$ | $2.41 \times 10^{-4}$ | $6.14 \times 10^{-5}$ | $6.01 \times 10^{-5}$ | $\mathbf{5.19 \times 10^{-5}}$ |
| Particle Energy (MRE ↓) | diff-N | N/A | $2.49 \times 10^{-4}$ | $6.43 \times 10^{-5}$ | $6.13 \times 10^{-5}$ | $\mathbf{5.22 \times 10^{-5}}$ |
| Mixing Ratio (MMD ↓) | in-dst | $1.91 \times 10^{-2}$ | $2.60 \times 10^{-2}$ | $2.24 \times 10^{-2}$ | $1.43 \times 10^{-1}$ | $\mathbf{1.09 \times 10^{-2}}$ |
| Mixing Ratio (MMD ↓) | diff-N | N/A | $5.26 \times 10^{-2}$ | $2.16 \times 10^{-2}$ | $1.41 \times 10^{-1}$ | $\mathbf{9.64 \times 10^{-3}}$ |

AE-Aug is N/A on diff-N because the MLP autoencoder size is tied to particle count, whereas the DeepSet encoder handles varying $n$ naturally.

### Ablation Study

| Config | Key Difference | Performance |
|------|---------|------|
| Ours (DeepSet + Dist. Rec.) | Full Model | Best in all in-dst / diff-N |
| AE-InvE (DeepSet + point-wise MSE) | Non-invariant decoder | 1 order of magnitude worse |
| AE-InvE-CD (DeepSet + Chamfer) | Invariant but point-matching | Close to ours, but gap in diff-init |
| InvE (Joint training, no rec.) | Removed $\mathcal{L}_{\mathrm{rec}}$ | MMD 10x higher; representation collapse |
| AE-Aug (MLP + Augmentation) | Approximate invariance | Energy curves fluctuate with permutations |

### Key Findings
- **Strict vs. Approximate Invariance**: Even with augmentation, AE-Aug yields different energy predictions for permutations of the same configuration. The proposed method's structural guarantees lead to perfectly overlapping curves in Fig 4(c).
- **Necessity of Reconstruction**: Removing reconstruction (InvE baseline) leads to failure in mixing tasks, suggesting reconstruction-free closure easily falls into collapsed local optima where latent variables become constant.
- **Particle Count Generalization**: Trained on 300 particles and tested on 400 (diff-N), the MRE remains nearly constant ($5.19 \to 5.22 \times 10^{-5}$), thanks to the $\mathcal{O}(n)$ property of DeepSet and the $n$-decoupled distribution loss.
- **Bandwidth $\epsilon$ and Latent Dimension**: Small $\epsilon$ requires larger $\hat{z}_{\mathrm{dim}}$ for multi-modal fitting; large $\epsilon$ allows near-perfect performance with lower dimensions, providing a clear "compression rate" control.

## Highlights & Insights
- **Moving Symmetry from Loss to Objective**: While point cloud models use Chamfer/EMD for invariant loss, this work makes the "reconstructed object" (density) invariant. This "target-switching" logic is transferable to molecular conformations and 3D reconstruction.
- **Distribution Reconstruction as Implicit Regularization**: Using $\epsilon$ as a bottleneck forces the encoder to ignore microscopic noise. This acts as a scale-adaptive information bottleneck more physical than KL-VAE.
- **OnsagerNet-friendly**: The dynamics network can be seamlessly replaced by structured networks like OnsagerNet, making this a universal closure modeling backend.

## Limitations & Future Work
- **Reliance on Deterministic $\bar{\bm{\varphi}}$**: Assumes macroscopic quantities of interest are known deterministic functions. Extrapolation to partially observed macro-quantities is not discussed.
- **Bandwidth $\epsilon$ as a Hyperparameter**: Currently lacks an adaptive mechanism for learning $\epsilon$; multi-scale systems might require manual tuning per scenario.
- **Narrow Image/Video Scope**: The polymer video task is restricted (rendering 3D coordinates as Gaussians). Performance on unstructured "real" video remains unverified.
- **Theoretical Identifiability**: Whether distribution reconstruction uniquely recovers latent variables relevant to macroscopic dynamics remains empirically driven.

## Related Work & Insights
- **vs. Champion et al. (2019, SINDy autoencoder)**: They use MLP autoencoders + point-wise MSE for ordered vectors; this work addresses unordered sets and upgrades loss to "distribution-level."
- **vs. Chen et al. (2023b, Polymer dynamics)**: Same scenario, but they use MLPs for ordered beads; this work achieves comparable results from image inputs, showing modal robustness.
- **vs. Achlioptas et al. (2018) / Point Cloud AEs**: This work bypasses matching steps (Chamfer/EMD) by using density and KL, reducing complexity from $\mathcal{O}(n^2)$ to $\mathcal{O}(1)$.
- **Transferable Insight**: In tasks with unordered inputs (molecular graphs, social networks), target density reconstruction may outperform point-matching.

## Rating
- Novelty: ⭐⭐⭐⭐ The "reconstruct density, not points" perspective is clear and rare.
- Experimental Thoroughness: ⭐⭐⭐⭐ Diverse scenarios, multiple regimes, and five baselines.
- Writing Quality: ⭐⭐⭐⭐ Solid derivation of motivation and compact formulas.
- Value: ⭐⭐⭐⭐ Directly applicable to particle simulations and fluid closure modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Continual Learning of Domain-Invariant Representations](continual_learning_of_domain-invariant_representations.md)
- [\[CVPR 2026\] Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation](../../CVPR2026/others/shoe_style-invariant_and_ground-aware_learning_for_dense_foot_contact_estimation.md)
- [\[ICLR 2026\] SONIC: Spectral Oriented Neural Invariant Convolutions](../../ICLR2026/others/sonic_spectral_oriented_neural_invariant_convolutions.md)
- [\[NeurIPS 2025\] Learning Dynamics of RNNs in Closed-Loop Environments](../../NeurIPS2025/others/learning_dynamics_of_rnns_in_closed-loop_environments.md)
- [\[ICML 2026\] Optimal Regularization for Performative Learning](optimal_regularization_for_performative_learning.md)

</div>

<!-- RELATED:END -->
