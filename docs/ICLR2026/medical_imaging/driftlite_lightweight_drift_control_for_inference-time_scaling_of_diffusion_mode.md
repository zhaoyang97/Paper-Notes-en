---
title: >-
  [Paper Note] DriftLite: Lightweight Drift Control for Inference-Time Scaling of Diffusion Models
description: >-
  [ICLR 2026][Medical Imaging][Diffusion model inference adaptation] DriftLite exploits the inherent degrees of freedom between the drift and potential function in the Fokker-Planck equation to actively stabilize particle weights by solving a lightweight linear system for the optimal control drift at each step. This approach addresses weight degeneracy in Sequential Monte Carlo (SMC) at minimal computational cost, substantially outperforming Guidance-SMC baselines on Gaussian mixture, molecular system, and protein–ligand co-folding tasks.
tags:
  - ICLR 2026
  - Medical Imaging
  - Diffusion model inference adaptation
  - particle methods
  - drift control
  - Fokker-Planck equation
  - variance reduction
date: 2026-05-08
content_hash: 89efc25fb835e73f
---

# DriftLite: Lightweight Drift Control for Inference-Time Scaling of Diffusion Models

**Conference**: ICLR 2026  
**arXiv**: [2509.21655](https://arxiv.org/abs/2509.21655)  
**Code**: [https://github.com/yinuoren/DriftLite](https://github.com/yinuoren/DriftLite)  
**Area**: Diffusion Models / Inference-Time Scaling  
**Keywords**: Diffusion model inference adaptation, particle methods, drift control, Fokker-Planck equation, variance reduction

## TL;DR
DriftLite exploits the inherent degrees of freedom between the drift and potential function in the Fokker-Planck equation to actively stabilize particle weights by solving a lightweight linear system for the optimal control drift at each step. This approach addresses weight degeneracy in Sequential Monte Carlo (SMC) at minimal computational cost, substantially outperforming Guidance-SMC baselines on Gaussian mixture, molecular system, and protein–ligand co-folding tasks.

## Background & Motivation

**Background**: Diffusion/Flow Matching models have achieved remarkable success in generative tasks, yet inference-time adaptation—adjusting to new target distributions without retraining—remains a critical challenge. Two main paradigms exist: guidance (simple but biased) and SMC particle reweighting (unbiased but prone to severe weight degeneracy).

**Limitations of Prior Work**:
   - Guidance methods (classifier/classifier-free guidance) are simple but inherently biased—they neglect the time-varying normalization constant of the target distribution.
   - SMC particle methods are theoretically unbiased (KL divergence $\mathcal{O}(N^{-1})$), but in practice weights degrade exponentially and the effective sample size (ESS) collapses rapidly.
   - Increasing particle count alleviates degeneracy but incurs linearly growing computational cost; reducing particle count leads to instability.
   - Training-based control methods (neural network parameterization) require backpropagation, forfeiting the lightweight advantage of inference-time approaches.

**Key Challenge**: Unbiasedness vs. computational efficiency—SMC is theoretically correct but practically unstable, while guidance is efficient but biased.

**Goal**:
   - How can particle weights be stabilized while preserving unbiasedness?
   - Can a training-free method with negligible overhead be devised?

**Key Insight**: The Fokker-Planck equation admits a fundamental degree of freedom between the drift term and the potential function—any control term added to the drift can be exactly compensated by a corresponding correction to the potential, and this freedom can be exploited to minimize the variance of the residual potential.

**Core Idea**: Transform SMC's passive reweighting into proactive steering—offload a portion of the variance-inducing potential $g_t$ onto the drift term, requiring only a small linear system solve at each step.

## Method

### Overall Architecture
Given a pretrained diffusion model and a target distribution (annealing $q_T \propto p_0^\gamma$ or reward-tilting $q_T \propto p_0 \exp(r)$), DriftLite operates at each inference timestep as follows: (1) estimate a small matrix $A_t$ and vector $c_t$ from current particles → (2) solve an $n \times n$ ($n \leq 3$) linear system to obtain control drift coefficients $\theta_t$ → (3) incorporate the control drift into the particle SDE and update the residual potential.

### Key Designs

1. **Fokker-Planck Degree-of-Freedom Theorem (Prop 3.1)**:

    - Function: Proves that the particle drift can be modified arbitrarily without altering the target distribution.
    - Mechanism: For any control drift $\bm{b}_t$, there exists a compensating potential $h_t(\bm{x}; \bm{b}_t) = \nabla \cdot \bm{b}_t + \bm{b}_t \cdot \nabla \log q_t$ such that the Fokker-Planck equation still describes the path $(q_t)$. Key property: $\mathbb{E}_{q_t}[h_t(\cdot; \bm{b}_t)] = 0$.
    - Design Motivation: This provides a mathematical tool for transferring potential variance into the drift—ideally eliminating weight variance entirely.

2. **VCG (Variance-Controlling Guidance)**:

    - Function: Directly minimizes the variance of the residual potential.
    - Mechanism: The control drift is restricted to a finite-dimensional subspace $\bm{b}_t = \sum_i \theta_t^i \bm{s}_i$, with basis functions $\{\nabla r_t, \nabla \log \hat{p}_t, \hat{\bm{u}}_t\}$. Minimizing $\text{Var}_{q_t}[\phi_t]$ reduces to a standard least-squares problem $A_t \theta_t = c_t$, where $A_{ij} = \mathbb{E}[h_t^i h_t^j]$ and $c_i = -\mathbb{E}[g_t h_t^i]$.
    - Design Motivation: Directly targets the root cause of weight degeneracy (potential variance) and requires only a $3 \times 3$ linear system solve, incurring negligible computational overhead.

3. **ECG (Energy-Controlling Guidance)**:

    - Function: Approximates the solution to the Poisson equation of the optimal control via variational methods.
    - Mechanism: The optimal control is set as $\bm{b}_t^* = \nabla A_t$, where $A_t$ satisfies the Poisson equation $\nabla \cdot (q_t \nabla A_t) = q_t g_t$. The Ritz method expands $A_t$ over scalar basis functions $\{r_t, \log \hat{p}_t, \hat{U}_t\}$, again reducing to a small linear system.
    - Design Motivation: Directly approximates the theoretically optimal solution without requiring differentiable Laplacian computations.

### Loss & Training
- **Fully training-free**: No training or backpropagation is required.
- Per-step overhead: solving an $n \times n$ ($n=3$) linear system + evaluating basis functions (reusing scores and reward gradients already computed for guidance).
- Supports iterative refinement: the control drift and residual potential from one round serve as the base dynamics for the next, progressively reducing variance.
- Optional SMC resampling (when ESS falls below a threshold) or purely continuous weighting.

## Key Experimental Results

### Main Results (30-dimensional Gaussian Mixture Model, annealing $\gamma=2.0$)

| Method | $\Delta$NLL↓ | MMD↓ | SWD↓ | ESS Stability |
|--------|------------|------|------|---------------|
| Pure Guidance | High bias | Poor | Poor | N/A |
| Guidance-SMC | Moderate | Mode collapse | Moderate | Rapidly degrades |
| **VCG-SMC** | **Lowest** | **Best** | **Best** | **Stable** |
| **ECG-SMC** | Near VCG | Near VCG | Near VCG | **Stable** |

### Protein–Ligand Co-Folding (AlphaFold3 + DriftLite)

| Method | Ligand RMSD↓ | Pocket TM-score↑ | Clash Score↓ |
|--------|------------|----------------|-------------|
| AF3 baseline | Moderate | Moderate | Higher |
| AF3 + VCG | **Significant improvement** | **Improved** | **Reduced** |

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| Variance reduction magnitude | VCG reduces potential variance by several orders of magnitude |
| Particle count scaling | DriftLite with $N/4$ particles matches SMC with $N$ particles |
| Iterative refinement | Variance decreases monotonically each round; sample quality improves progressively |
| Additional runtime | Approximately 20–40% overhead compared to Pure Guidance |

### Key Findings
- **Variance reduced by orders of magnitude**: VCG reduces potential variance from $10^2$–$10^3$ to $10^{-1}$–$10^0$, maintaining stable ESS throughout inference.
- **Improved particle efficiency**: DriftLite with 32 particles outperforms G-SMC with 128 particles—a 4× efficiency gain.
- **VCG marginally outperforms ECG**: Directly minimizing variance proves more effective than variationally approximating the Poisson equation.
- **Scalability to large-scale scientific applications**: Successful application to AlphaFold3 protein–ligand co-folding confirms that the method scales to real-world large-scale scenarios.
- **Iterative refinement is effective**: Multi-round refinement yields monotonically decreasing variance, analogous to adaptive methods but without training.

## Highlights & Insights
- **Leveraging the Fokker-Planck degree of freedom as a variance control tool** constitutes the paper's most fundamental theoretical contribution—an elegant formulation. Although this degree of freedom is mathematically known (twisted proposals in SMC), formalizing it as a programmable, lightweight control requiring only a $3 \times 3$ linear system represents a clever engineering innovation.
- **The paradigm shift from passive reweighting to proactive steering** is noteworthy—the essential problem with SMC is not theoretical correctness but practical weight degeneracy. DriftLite fundamentally resolves this engineering challenge by transferring "information" from weights to the drift.
- **Successful application to AlphaFold3** demonstrates the method's value in real scientific settings—this is not merely a theoretically elegant approach, but a practical tool capable of directly improving protein structure prediction.

## Limitations & Future Work
- Requires the reward function to be twice differentiable (stochastic estimators can approximate this, but with reduced accuracy).
- Basis function selection remains manual (three basis functions); additional or better-chosen basis functions may further improve performance.
- In high-dimensional settings, Laplacian estimation introduces noise that may degrade control quality.
- Validation is limited to generative scientific tasks; large-scale image generation (e.g., SD3) has not been tested.
- Iterative refinement increases inference time (though no training is required).

## Related Work & Insights
- **vs. Guidance-SMC (Skreta et al.)**: Operates within the same framework, but DriftLite actively reduces weight degeneracy via drift control, whereas G-SMC passively reweights, leading to weight collapse.
- **vs. Neural Control (Albergo & Vanden-Eijnden)**: Neural network–parameterized control requires training/backpropagation; DriftLite replaces this with a linear system over three basis functions, making it training-free.
- **vs. Pure Guidance (Ho & Salimans)**: Guidance is simple but biased; DriftLite preserves unbiasedness with only 20–40% additional computational overhead.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The theoretical insight of mapping FP degrees of freedom to variance control is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation spanning synthetic data, molecular systems, and protein folding.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous and elegant; experimental design is logically coherent.
- Value: ⭐⭐⭐⭐⭐ A broadly applicable contribution to inference-time adaptation of diffusion models with high scientific utility.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification](inference-time_dynamic_modality_selection_for_incomplete_multimodal_classificati.md)
- [\[AAAI 2026\] MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains](../../AAAI2026/medical_imaging/mirage_scaling_test-time_inference_with_parallel_graph-retrieval-augmented_reaso.md)
- [\[ICLR 2026\] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)
- [\[ICLR 2026\] Fine-Tuning Diffusion Models via Intermediate Distribution Shaping](fine-tuning_diffusion_models_via_intermediate_distribution_shaping.md)
- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)

<!-- RELATED:END -->
