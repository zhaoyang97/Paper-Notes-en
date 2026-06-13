---
title: >-
  [Paper Note] Global Plane Waves from Local Gaussians: Periodic Charge Densities in a Blink
description: >-
  [ICML 2026][Interpretability][Charge Density Prediction] ELECTRAFI predicts a set of anisotropic Gaussian parameters in real space, then utilizes the analytic Fourier transform of Gaussians combined with the Poisson summ…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Charge Density Prediction"
  - "Plane Wave Basis Set"
  - "Anisotropic Gaussians"
  - "Poisson Summation"
  - "Accelerator for DFT Initial Guess"
date: 2026-05-08
content_hash: ff991207a3b88329
---

# Global Plane Waves from Local Gaussians: Periodic Charge Densities in a Blink

**Conference**: ICML 2026  
**arXiv**: [2601.19966](https://arxiv.org/abs/2601.19966)  
**Code**: https://github.com/Jotels/ELECTRAFI (Available)  
**Area**: Scientific Computing / ML+DFT / Electronic Structure  
**Keywords**: Charge Density Prediction, Plane Wave Basis Set, Anisotropic Gaussians, Poisson Summation, Accelerator for DFT Initial Guess

## TL;DR
ELECTRAFI predicts a set of anisotropic Gaussian parameters in real space, then utilizes the analytic Fourier transform of Gaussians combined with the Poisson summation formula to calculate plane-wave coefficients of the periodic crystal charge density in reciprocal space in one go. A single inverse FFT yields the full-field density. While maintaining or exceeding NMAE parity with ChargE3Net, it achieves an inference speedup of $463\times \sim 633\times$, effectively reducing the total end-to-end DFT time by $\sim 20\%$.

## Background & Motivation

**Background**: Kohn–Sham DFT is the most widely used electronic structure method in physics, chemistry, and materials science. Mainstream acceleration strategies follow two paths: (1) Machine Learning Interatomic Potentials (MLIP) to directly predict energy/forces, bypassing the SCF cycle; (2) retaining the DFT workflow and using ML to predict a high-quality initial charge density $\rho(\mathbf{r})$ as the SCF starting point, thereby reducing the number of self-consistent iterations. The advantage of the latter is that it still converges to the rigorous self-consistent solution of the target functional via DFT.

**Limitations of Prior Work**: ML density models for periodic systems are divided into two categories: **Orbital models** expand the density using atom-centered spherical harmonics and radial bases, which work well for small molecules but suffer from basis set explosion for metals or inorganic crystals requiring high-$l$ and diffuse bases; **Probe models** (e.g., ChargE3Net, DeepDFT) treat real-space grid points as graph nodes in a GNN. Although accurate, they require querying $\sim 10^7$ grid points per structure, with inference often taking $30\sim 80$ seconds. Consequently, **the time saved in SCF is consumed by the ML inference time**, sometimes leading to slower end-to-end performance.

**Key Challenge**: Periodicity and long-range Coulomb interactions naturally belong to reciprocal space. However, existing models either perform expensive periodic image summation and spherical harmonic expansion in real space or directly predict plane-wave coefficients (where output dimensions scale poorly with the FFT grid, making them effective only as small patches). **Accuracy and inference cost must be jointly optimized** to achieve genuine DFT wall-clock savings.

**Goal**: Construct a charge density model that is (1) naturally periodic, (2) aligned with plane-wave DFT encoding, (3) capable of analytically handling long-range structures, and (4) characterized by an inference time that is negligible compared to DFT, while providing **net time savings** in real VASP pipelines.

**Key Insight**: Gaussian functions remain Gaussian under Fourier transform with closed-form expressions. Furthermore, the **Poisson summation formula** allows rewriting the "all-space Gaussian superposition" as a Fourier series on reciprocal lattice points. By using a set of floating Gaussians for local density expansion, the "periodization" step can be delegated entirely to the analytical transform, eliminating real-space image summation.

**Core Idea**: Predict local anisotropic Gaussian parameters $(w^{(j)},\boldsymbol{\mu}^{(j)},\boldsymbol{\Sigma}^{(j)})$ in real space $\rightarrow$ analytically compute the Fourier coefficients for each Gaussian at reciprocal lattice points $\mathbf{G}$ $\rightarrow$ sum them to obtain $\hat{\rho}(\mathbf{G})$ $\rightarrow$ perform a single inverse FFT to restore the periodic real-space density $\rho(\mathbf{r})$.

## Method

### Overall Architecture
- **Input**: Atomic graph of the periodic crystal (including PBC neighbors).
- **Backbone**: An attention GNN modified from EScAIP, updating scalar/vector/tensor streams based on PBC neighborhoods.
- **Readout**: Dyadic neighborhood aggregation, outputting three types of channels for each atom: $S\in\mathbb{R}^{N\times C}$, $V\in\mathbb{R}^{N\times C\times 3}$, $T\in\mathbb{R}^{N\times C\times 3\times 3}$.
- **Allocation**: According to the rule of "assigning $M$ Gaussians per valence electron," atom $a$ receives $n_a=Mv_a$ Gaussian slots, totaling $N_\mathcal{N}=M\sum_a v_a$ Gaussians for the structure.
- **Parametrization**: Each Gaussian is described by $(w^{(j)},\boldsymbol{\mu}^{(j)},\boldsymbol{\Sigma}^{(j)})$. The signatory weight is $w^{(j)}=\tanh(\mathrm{MLP}(S^{(j)}))$, the center $\boldsymbol{\mu}^{(j)}=\mathbf{R}_{a(j)}+\mathbf{d}^{(j)}$ is the offset relative to the host atom, and the covariance is guaranteed positive definite using Gram decomposition $\boldsymbol{\Sigma}^{(j)}=\gamma^{(j)}\mathbf{A}^{(j)}\mathbf{A}^{(j)\top}$.
- **Reconstruction**: All Gaussians are analytically transformed into reciprocal space and summed to obtain $\hat{\rho}(\mathbf{G})$. An IFFT yields $\rho(\mathbf{r})$, and a final global weight scaling ensures $\int_\Omega\rho\,d\mathbf{r}=N_e$ (conservation of electron number).

### Key Designs

1.  **Periodization via Analytic Fourier + Poisson Summation**:
    - **Function**: Replaces "infinite image summation in real space" with "finite grid summation in reciprocal space" to obtain a naturally periodic density field.
    - **Mechanism**: The auxiliary non-periodic function $\tilde\rho(\mathbf{r})=\sum_j w^{(j)}\mathcal{N}(\mathbf{r};\boldsymbol{\mu}^{(j)},\boldsymbol{\Sigma}^{(j)})$ is never explicitly evaluated in real space. Leveraging Gaussian self-reciprocity, each component has a closed form at $\mathbf{G}$: $\hat{\mathcal{N}}(\mathbf{G})=\exp(-\tfrac{1}{2}\mathbf{G}^\top\boldsymbol{\Sigma}\mathbf{G})e^{-i\mathbf{G}\cdot\boldsymbol{\mu}}$. Summing gives $\hat\rho(\mathbf{G})=\sum_j w^{(j)}\exp(-\tfrac{1}{2}\mathbf{G}^\top\boldsymbol{\Sigma}^{(j)}\mathbf{G})e^{-i\mathbf{G}\cdot\boldsymbol{\mu}^{(j)}}$. By the Poisson summation formula $\sum_{\mathbf{R}\in\Lambda}\rho(\mathbf{r}+\mathbf{R})=\frac{1}{|\Omega|}\sum_{\mathbf{G}\in\Lambda^*}\hat\rho(\mathbf{G})e^{i\mathbf{G}\cdot\mathbf{r}}$, truncating $\Lambda^*$ only results in low-pass filtering (consistent with the low-pass nature of DFT densities), unlike truncating $\Lambda$ which leaves discontinuities at unit cell boundaries.
    - **Design Motivation**: Naive image summation is not truly periodic and costs $27\times$ more even for $N=1$ (27 adjacent cells). Direct prediction of $\hat\rho(\mathbf{G})$ causes output dimensions to explode with the FFT grid. This work decouples learnable parameters from resolution using "local Gaussian parameters + analytic global transform," making the **network scale independent of cell size or plane-wave cutoff**.

2.  **Floating Anisotropic Gaussian Expansion + Gram-factored Covariance**:
    - **Function**: Uses a representation much more compact than spherical harmonics to capture the broad and diffuse valence electron densities in inorganic crystals.
    - **Mechanism**: Each Gaussian is tied to an atom but allowed to drift to bond midpoints or interstitial positions via the predicted displacement $\mathbf{d}^{(j)}$. Covariance is written in Gram form $\boldsymbol{\Sigma}^{(j)}=\gamma^{(j)}\mathbf{A}^{(j)}\mathbf{A}^{(j)\top}$, naturally ensuring SPD while allowing adjustments to both direction and shape. The signatory weight $w^{(j)}=\tanh(s^{(j)})$ allows both addition and subtraction to reconstruct fine valence structures. A global scalar multiplier ensures the total electron count equals $N_e$.
    - **Design Motivation**: Fixed atom-center LCAO requires many high-$l$ diffuse basis functions for materials, which is inefficient. Floating Gaussians + anisotropic covariance eliminate reliance on high-order spherical harmonics, reducing parameter complexity from $O(L_{\max}^2)$ to a a few $3\times3$ tensors, significantly accelerating inference and ensuring insensitivity to element types.

3.  **EScAIP Attention Backbone + Dyadic Tensor Readout**:
    - **Function**: Generates scalar, vector, and tensor features on PBC graphs that are both cheap and geometrically aligned for the prediction heads of $(\boldsymbol{\mu},\boldsymbol{\Sigma},w)$.
    - **Mechanism**: Instead of strict equivariant SO(3) tensor products (where the cost of MACE/Equiformer-style high-order irreps + Gaunt contractions explodes with $L_{\max}$), it uses multi-head attention + feed-forward layers to build capacity in the scalar stream. Vector/tensor heads use attention-weighted sums of unit edge directions $\hat{\mathbf{e}}_{ij}$ and their traceless dyads $\hat{\mathbf{e}}_{ij}\hat{\mathbf{e}}_{ij}^\top$. Complexity scales linearly with the number of edges, avoiding explicit triplet enumeration. Equivariance is supplemented by data augmentation; experiments show that random rotation during training reduces test error from 1.69% to 1.33%.
    - **Design Motivation**: The other half of the inference cost in probe/orbital models comes from the backbone; equivariant high-order tensors are cost-prohibitive for large cells. This design delegates geometric inductive bias to the data, concentrating compute on attention and FFNs, which is key to ELECTRAFI's sub-0.1s end-to-end performance.

### Loss & Training
The model is trained directly using NMAE as the objective (consistent with prior density model evaluations):
$$\mathcal{L}=\mathrm{NMAE}(\rho_{\mathrm{pred}},\rho_{\mathrm{ref}})=\frac{\int_\Omega|\rho_{\mathrm{ref}}-\rho_{\mathrm{pred}}|\,dV}{\int_\Omega\rho_{\mathrm{ref}}\,dV}$$
Calculated on the full real-space grid for each structure (no sub-sampling). NMAE[%] = $100\times\mathrm{NMAE}$ is reported. The MP-Full training set contains 117,876 structures. The hyperparameter $M$ (Gaussians per valence electron) is set according to the appendix.

## Key Experimental Results

### Main Results
All tests were conducted on a single A100-40GB; VASP DFT ran on 24-core Intel Xeon E5-2650 (Broadwell).

| Dataset | Metric | ELECTRAFI | ChargE3Net | DeepDFT / GPWNO / InfGCN |
|--------|------|-----------|-----------|---------------------------|
| MP-Full | NMAE [%] | **0.58** | 0.54 | 0.80 (DeepDFT) |
| MP-Full | $t_{\mathrm{inf}}$ [s] | **0.17** | 78.73 | — |
| MP-Full | Speedup | **463×** | — | — |
| GNoME | NMAE [%] | 0.93 | **0.69** | — |
| GNoME | $t_{\mathrm{inf}}$ [s] | **0.11** | 33.28 | — |
| GNoME | Speedup | **302×** | — | — |
| ECD-HSE06 | NMAE [%] | **1.35** | 1.53 | — |
| ECD-HSE06 | $t_{\mathrm{inf}}$ [s] | **0.05** | 31.65 | — |
| ECD-HSE06 | Speedup | **633×** | — | — |
| MP-Mixed | NMAE [%] | **1.24** | — | 11.50 / 4.83 / 5.11 |
| Cubic | NMAE [%] | **1.37** | — | 10.37 / 7.69 / 8.98 (SCDP 2.59) |

ELECTRAFI achieves SOTA or tied results on 4/5 benchmarks in terms of NMAE. While slightly behind ChargE3Net by 0.04~0.24 points on MP-Full/GNoME, its **inference is two to three orders of magnitude faster**.

### End-to-End DFT Acceleration

| Dataset | Method | NMAE | ML Inf. Time | SCF Steps | DFT Time | **Total Time** | Total Savings |
|--------|------|------|------------|---------|----------|----------|----------|
| MP | SAD (Default) | — | 0 | 16.80 | 266.34 s | 266.34 s | 0% |
| MP | SC (Upper Bound)| — | 0 | 8.73 | 161.98 s | 161.98 s | 39.18% |
| MP | **ELECTRAFI** | 0.55% | 0.17 s | 13.33 | 219.57 s | **219.74 s** | **+17.50%** |
| MP | ChargE3Net | 0.50% | 72.11 s | 12.09 | 208.26 s | 280.37 s | **−5.27%** |
| GNoME | SAD | — | 0 | 13.49 | 119.98 s | 119.98 s | 0% |
| GNoME | SC (Upper Bound)| — | 0 | 6.88 | 77.91 s | 77.91 s | 39.75% |
| GNoME | **ELECTRAFI** | 0.88% | 0.11 s | 10.11 | 95.06 s | **95.17 s** | **+20.68%** |
| GNoME | ChargE3Net | 0.59% | 28.29 s | 9.50 | 92.67 s | 120.96 s | **−0.82%** |

### Key Findings
- **Inference time cannot be ignored**: While ChargE3Net produces faster DFT wall-clock times (and fewer SCF steps) than ELECTRAFI itself, adding back the inference time results in a **net loss of 0.8%~5.3%**. ELECTRAFI's inference is nearly free, translating SCF reduction directly into 17%~21% end-to-end savings.
- **Structural Stability Gap**: ELECTRAFI increases total time for only 13.4% (MP) / 7.2% (GNoME) of structures, whereas ChargE3Net increases it for 84.3% / 74.6% of structures.
- **Elemental Complementarity**: Error analysis (Appendix F) shows ELECTRAFI is more accurate for alkali/alkaline earth/halogens/heavy elements (Ac, etc.) due to their smoother, more diffuse densities; ChargE3Net excels at light covalent elements and open-shell transition metals with localized directional bonding.
- **Rotation Robustness**: Despite no explicit equivariance, random rotation augmentation brought the NMAE from 1.69% back to 1.33% (slightly better than non-augmented tests), proving soft equivariance + attention can learn invariance from data.
- **Ceiling Analysis**: Using the converged self-consistent (SC) density as the initial guess yields a theoretical maximum of ~40% time savings. ELECTRAFI achieves about 50% of this potential, suggesting that while NMAE can improve, returns on grid-level NMAE optimization are diminishing because SCF convergence is dominated by long-wavelength slow modes.

## Highlights & Insights
- **The "Local Prediction + Global Analytic Transform" decoupling is elegant**: It separates "what the network learns" (local, atomic-scale Gaussian parameters) from "how to handle periodicity and long-range structure" (analytic Fourier + Poisson summation). The network output dimension is independent of cell size or FFT cutoff, allowing direct transfer to larger systems without retraining.
- **Elevating "Inference Cost" to a first-class metric**: The authors highlight that current charge density benchmarks shouldn't only look at NMAE or SCF steps; end-to-end DFT wall-clock time is the ultimate metric—a significant methodological shift for the ML-for-DFT subfield.
- **Transferable Trick**: Replacing spherical harmonics with floating anisotropic Gaussians + Gram-factored SPD covariance is a representation useful for any task approximating 3D fields (electric potential, magnetic fields, etc.). Analytic Fourier + Poisson summation is also directly applicable to periodic inverse problems (MRI k-space, acoustics, photonics).
- **Feasibility of Soft Equivariance**: In large-scale periodic systems, strict SO(3) equivariance may not be cost-effective. This work demonstrates that "fast backbone + rotation augmentation" is a practical alternative.

## Limitations & Future Work
- **Intrinsic Limits**: The NMAE floor for periodic charge densities is higher than for molecular systems (0.6-0.9% on MP/GNoME vs. 0.1% on QM9). SCF savings only reach ~50% of the theoretical upper bound, likely because SCF convergence depends on error projection on slow-converging long-wavelength modes rather than just global NMAE.
- **Unexploited Complementarity**: ELECTRAFI excels at diffuse/ionic bonding, while ChargE3Net excels at localized covalent bonding. A hybrid model combining "Global Fourier representation + Local real-space correction" could be explored.
- **Magnetic Systems**: The current model predicts total valence density without spin-resolved outputs. Spin-polarized DFT (where initial guesses are especially expensive) cannot yet benefit; new spin-resolved benchmarks are needed.
- **Dataset Incompleteness**: MP/ECD lack some metadata required for end-to-end DFT experiments (occupancies, PAW charges, SCF settings), limiting evaluation fairness.
- **Self-Observation**: Evaluations were limited to PBE/HSE06 functionals; whether SCF savings hold for meta-GGA or complex magnetic systems with strong Hubbard-U remains to be seen. The hyperparameter $M$ might also benefit from being element-aware.

## Related Work & Insights
- **vs ChargE3Net (probe model, koker2024)**: The latter treats grid points as GNN nodes ($10^7$ points), strong in accuracy but takes $30\sim 80$s for inference. Ours compresses this to 0.1s.
- **vs DeepDFT / GPWNO / InfGCN**: These models show NMAE between 4.8%~11% on MP-Mixed/Cubic. Ours achieves 1.24%/1.37%, showing that "Floating Anisotropic Gaussians + Analytic Periodization" is far more efficient for materials than fixed bases or probes.
- **vs ELECTRA (elsborg2025, previous work)**: The predecessor used floating Gaussians for molecular density; this work generalizes it to crystals (periodic) via Poisson summation.
- **vs Plane-Wave Add-on (kim2024gaussian)**: The former treats plane-wave operators as patches for atom-centered representations; this work proves reciprocal-space should be the center of the representation rather than an add-on.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The combination of "local Gaussians + analytic Fourier + Poisson summation" is physically intuitive and solves the real-space image summation bottleneck.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five benchmarks + real VASP end-to-end tests + elemental error analysis + rotation ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Method derivation is clear; classifications in the Related Work section are exceptionally valuable.
- **Value**: ⭐⭐⭐⭐⭐ The first periodic density model to achieve actual end-to-end DFT time savings, with open-source code; significant for DFT acceleration, material discovery, and energy storage simulations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics](../../AAAI2026/interpretability/pragworld_a_benchmark_evaluating_llms_local_world_model_under_minimal_linguistic.md)
- [\[ICML 2026\] BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](block-em_preventing_emergent_misalignment_via_latent_blocking.md)
- [\[ICML 2026\] Courtroom Analogy: New Perspective on Uncertainty-Aware Classification](courtroom_analogy_new_perspective_on_uncertainty-aware_classification.md)
- [\[ICML 2026\] Verified SHAP: Provable Bounds for Exact Shapley Values in Neural Networks](verified_shap_provable_bounds_for_exact_shapley_values_of_neural_networks.md)
- [\[ICML 2026\] Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation](interpretable_self-supervised_learning_via_representer_landmarks_and_nyström_app.md)

</div>

<!-- RELATED:END -->
