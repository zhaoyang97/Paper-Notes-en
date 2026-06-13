---
title: >-
  [Paper Note] Coarse-Grained Boltzmann Generators
description: >-
  [ICML 2026][Image Generation][Boltzmann Generator] The authors propose Coarse-Grained Boltzmann Generators (CG-BGs), which combine normalizing flow generative models with a learned Potential of Mean Force (PMF) in a coar…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Boltzmann Generator"
  - "Coarse-Grained Modeling"
  - "Importance Sampling"
  - "Potential of Mean Force"
  - "Normalizing Flows"
date: 2026-05-08
content_hash: f21e2178da4da5c4
---

# Coarse-Grained Boltzmann Generators

**Conference**: ICML 2026  
**arXiv**: [2602.10637](https://arxiv.org/abs/2602.10637)  
**Code**: https://github.com/tummfm/cg-bg  
**Area**: Scientific Computing / Molecular Simulation  
**Keywords**: Boltzmann Generator, Coarse-Grained Modeling, Importance Sampling, Potential of Mean Force, Normalizing Flows  

## TL;DR
The authors propose Coarse-Grained Boltzmann Generators (CG-BGs), which combine normalizing flow generative models with a learned Potential of Mean Force (PMF) in a coarse-grained coordinate space for importance sampling. This achieves asymptotically correct equilibrium sampling of molecules at a significantly lower computational cost than atomistic BGs.

## Background & Motivation

**Background**: Sampling equilibrium molecular configurations from the Boltzmann distribution is a central challenge in statistical physics. Boltzmann Generators (BGs) address this by combining exact-likelihood generative models with importance sampling, generating proposal samples and reweighting them to obtain unbiased estimates. Meanwhile, coarse-grained (CG) methods handle larger molecular systems by reducing degrees of freedom.

**Limitations of Prior Work**: Atomistic BGs face two major bottlenecks as dimensionality increases: (1) the drop in overlap between the generative and target distributions leads to variance explosion in importance weights, causing reweighting to fail; (2) Jacobian determinant computation grows sharply with dimensionality. Conversely, while Boltzmann Emulators improve scalability through CG dimensionality reduction, they omit the reweighting step, failing to correct distribution bias, and rely on hard-to-obtain long-term unbiased simulation data for training.

**Key Challenge**: BGs possess a reweighting mechanism but are difficult to scale to large systems; CG Emulators are scalable but lack a correction mechanism—the strengths of both remain unintegrated.

**Goal**: To implement generative modeling with importance sampling in the coarse-grained coordinate space while learning the target energy function from fast-converging enhanced sampling data.

**Key Insight**: The marginal distribution in coarse-grained coordinates $p(\mathbf{R})$ can likewise be written in Boltzmann form $p(\mathbf{R}) \propto e^{-\beta U(\mathbf{R})}$, where $U(\mathbf{R})$ is the Potential of Mean Force (PMF). If the PMF can be learned, the BG importance sampling framework can be reused within the low-dimensional CG space.

**Core Idea**: Use Enhanced Sampling Force Matching (ESFM) to learn the PMF from fast-converging biased trajectories, use normalizing flows to generate proposal distributions in CG space, and use the learned PMF for importance reweighting, forming a complete CG-BG framework.

## Method

### Overall Architecture
The input consists of atomistic molecular dynamics simulation trajectories (which can be biased enhanced sampling data), projected onto low-dimensional CG coordinates through a coarse-graining mapping $\mathbf{R} = \Xi(\mathbf{r})$. The framework comprises two components trained in parallel: (1) a proposal distribution $q_\theta(\mathbf{R})$ based on Continuous Normalizing Flows (CNF); (2) a neural network-based PMF $U_\eta(\mathbf{R})$. During inference, the flow model generates CG configurations, the PMF calculates importance weights $w(\mathbf{R}) \propto e^{-\beta U_\eta(\mathbf{R})} / q_\theta(\mathbf{R})$, and unbiased equilibrium observables are obtained via self-normalized importance sampling estimators.

### Key Designs

1.  **Learning PMF via Enhanced Sampling Force Matching (ESFM)**:
    *   **Function**: Learn the coarse-grained PMF from fast-converging biased simulation data without relying on expensive unbiased equilibrium trajectories.
    *   **Mechanism**: Leverages fiber distribution invariance—when a bias potential $V(\mathbf{R})$ is applied to CG coordinates, the atomistic conditional distribution given $\mathbf{R}$ remains unchanged, i.e., $p_V(\mathbf{r}|\mathbf{R}) = p(\mathbf{r}|\mathbf{R})$. Thus, the conditional mean of projected forces (the mean force) is invariant under biased sampling, and the force matching regression target is unaffected by bias. The training loss is $\mathcal{L}_{\mathrm{ESFM}}(\eta) = \mathbb{E}_{\mathbf{r} \sim \mathcal{D}_{\mathrm{bias}}}[\|\nabla_{\mathbf{R}} U_\eta(\Xi(\mathbf{r})) + \mathcal{F}_{\mathrm{proj}}(\mathbf{r})\|^2]$, where forces are recalculated from the unbiased atomistic potential.
    *   **Design Motivation**: Standard force matching requires converged unbiased data, whereas enhanced sampling (e.g., well-tempered metadynamics) can quickly cover transition regions between metastable states. ESFM shares the same global optimum as standard force matching, with its KL divergence bounded by the squared force error.

2.  **Proposal Generation via Continuous Normalizing Flows (CNF) in CG Space**:
    *   **Function**: Learn a proposal density $q_\theta(\mathbf{R})$ in the low-dimensional CG coordinate space to approximate the target marginal distribution.
    *   **Mechanism**: Uses Flow Matching to train a neural vector field $v_\theta(t, \mathbf{x})$, regressing the target vector field via a linear interpolation path $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$. The flow model operates in a CG space with dimensions much smaller than atomistic ones (e.g., Alanine hexapeptide reduced from 72 atoms to a few beads via Core Beta mapping), significantly improving Jacobian computation and distribution overlap.
    *   **Design Motivation**: In low-dimensional space, the generative model better overlaps with the target distribution, resulting in lower importance weight variance and higher ESS, while inference Jacobian costs are drastically reduced.

3.  **PMF-Guided Importance Reweighting**:
    *   **Function**: Correct biased proposal samples from the flow model into asymptotically correct equilibrium distributions.
    *   **Mechanism**: Importance weights $w(\mathbf{R}_i) \propto e^{-\beta U_\eta(\mathbf{R}_i)} / q_\theta(\mathbf{R}_i)$ are calculated for samples $\mathbf{R}_i \sim q_\theta$ generated by the flow model, and observables are computed using self-normalized estimators. Reweighting quality is assessed via the normalized Effective Sample Size $\mathrm{ESS} = (\sum w_i)^2 / (B \sum w_i^2)$. A weight truncation strategy is adopted to enhance robustness against MLP extrapolation anomalies and generation artifacts.
    *   **Design Motivation**: Boltzmann Emulators cannot correct bias when directly using $q_\theta$ for estimation. Introducing the learned PMF as the target energy function restores the reweighting capability of BGs, while the PMF captures solvent-mediated effects that implicit solvent models cannot express.

### Loss & Training
The framework is trained in two independent stages: (1) The PMF network is trained using the ESFM loss on biased or unbiased atomistic trajectory data; (2) The normalizing flow is trained via the Conditional Flow Matching loss on CG coordinate data. Both can be trained in parallel and combined during inference.

## Key Experimental Results

### Main Results

Evaluated on Alanine dipeptide (22 atoms), tripeptide (42 atoms), and hexapeptide (72 atoms), with explicit solvent MD simulations as the reference standard.

| Model | JS Divergence (↓) | PMF Error (↓) | ESS (↑) |
| :--- | :--- | :--- | :--- |
| CG-BG Heavy Atom | 0.0048 | 0.2005 | 0.5112 |
| CG-BG Heavy Atom (Biased) | 0.0063 | 0.2277 | 0.4115 |
| CG-BG Core Beta | 0.0052 | 0.2210 | 0.5528 |
| CG-BG Core Beta (Biased) | 0.0057 | 0.2093 | 0.4818 |
| Implicit Solvent GB (OBC1) | 0.0157 | 0.3709 | — |
| Implicit Solvent GB (OBC2) | 0.0182 | 0.4028 | — |

### Computational Efficiency Comparison (Alanine dipeptide, $10^4$ samples)

| CG Mapping | Training Time | Inference Time | Total Time |
| :--- | :--- | :--- | :--- |
| Core Beta | 0.45h | 0.95min | 0.47h |
| Heavy Atom | 0.80h | 3.78min | 0.86h |
| All Atom (Solute only) | 2.55h | 14.91min | 2.80h |

### Validation on Larger Systems (Tripeptide & Hexapeptide)

| Model | Tripeptide JS (↓) | Tripeptide PMF (↓) | Tripeptide ESS (↑) | Hexapeptide JS (↓) | Hexapeptide PMF (↓) | Hexapeptide ESS (↑) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CG-BG Core Beta | 0.0060 | 0.2112 | 0.4212 | 0.0100 | 0.3646 | 0.1231 |
| CG-BG Heavy Atom | 0.0056 | 0.1957 | 0.3201 | — | — | — |
| Implicit Solvent GB (OBC2) | 0.0932 | 1.0274 | — | 0.1652 | 1.8401 | — |

### Key Findings
*   After reweighting, CG-BG significantly outperforms implicit solvent baselines across all metrics, with the performance **Gain** widening in larger systems like tripeptide and hexapeptide (Hexapeptide JS divergence 0.0100 vs. 0.1652).
*   A precision-efficiency trade-off exists in CG resolution: Core Beta mapping offers higher ESS (better distribution overlap) but slightly lower accuracy after reweighting compared to Heavy Atom mapping.
*   **Ours** trained on 10ns of biased data achieves accuracy close to the version trained on 500ns of unbiased data, proving the data efficiency improvement of ESFM.
*   Atomistic BGs are capped at the accuracy of implicit solvent models, whereas CG-BG breaks this limit by learning PMF from explicit solvent data.

## Highlights & Insights
*   **Clever Use of Fiber Distribution Invariance**: CG bias potentials do not alter the atomistic conditional distribution given the CG coordinates. This theoretical guarantee allows expensive unbiased trajectories to be replaced with fast-converging data from enhanced sampling.
*   **Simulation-Free PMF Evaluation**: Once a proposal distribution is learned, multiple candidate CG force fields can be evaluated simultaneously by switching importance weights for different PMFs, without running separate MD for each model—a significant acceleration of the CG force field development workflow.
*   **Complementary Design of Coarse-Graining + Reweighting**: Coarse-graining addresses dimensionality to keep ESS controllable, while reweighting addresses distribution bias to ensure asymptotic correctness. This orthogonal decomposition of dimensionality reduction and bias correction is transferable to other high-dimensional sampling problems.

## Limitations & Future Work
*   Relies on predefined collective variables (CV selection for CG mapping and enhanced sampling); appropriate CVs may be difficult to determine for complex systems.
*   Current experiments are validated only on alanine short peptides ($\le 72$ atoms); effectiveness on larger protein systems remains to be verified.
*   The ESS for the hexapeptide has dropped to 0.1231, suggesting importance sampling efficiency may further decrease as the system size grows.
*   Future directions include integrating automatic CV discovery methods, introducing transferable generative architectures, and exploring energy-based training as an alternative to Flow Matching.

## Related Work & Insights
*   **Boltzmann Generator** (Noé et al., 2019): The atomistic BG framework using normalizing flows + importance sampling, limited by dimensionality.
*   **Boltzmann Emulator** (Lewis et al., 2025): A CG space generative model but lacks reweighting and relies on converged data.
*   **ESFM** (Chen et al., 2026): Theoretical foundation for enhanced sampling force matching, proving force matching equivalence under CG bias.
*   **TarFlow / ECNF++** (Tan et al., 2025b): Improved atomistic BG architectures still limited by implicit solvent accuracy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Flatten Graphs as Sequences: Transformers are Scalable Graph Generators](../../NeurIPS2025/image_generation/flatten_graphs_as_sequences_transformers_are_scalable_graph_generators.md)
- [\[NeurIPS 2025\] BoltzNCE: Learning Likelihoods for Boltzmann Generation with Stochastic Interpolants](../../NeurIPS2025/image_generation/boltznce_learning_likelihoods_for_boltzmann_generation_with_stochastic_interpola.md)
- [\[ICCV 2025\] CharaConsist: Fine-Grained Consistent Character Generation](../../ICCV2025/image_generation/characonsist_fine-grained_consistent_character_generation.md)
- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](../../NeurIPS2025/image_generation/progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [\[AAAI 2026\] Talk, Snap, Complain: Validation-Aware Multimodal Expert Framework for Fine-Grained Customer Grievances](../../AAAI2026/image_generation/talk_snap_complain_validation-aware_multimodal_expert_framework_for_fine-grained.md)

</div>

<!-- RELATED:END -->
