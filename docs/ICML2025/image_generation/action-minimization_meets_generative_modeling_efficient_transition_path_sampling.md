---
title: >-
  [Paper Note] Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional
description: >-
  [ICML2025][Image Generation][Transition Path Sampling] This paper proposes interpreting the score functions of pretrained generative models (diffusion models and flow matching) as drift terms in stochastic dynamics. By minimizing the Onsager-Machlup (OM) action functional, the pretrained models are repurposed in a **zero-shot** manner for transition path sampling (TPS) in molecular systems. This achieves physically realistic transition paths at a fraction of the computational…
tags:
  - "ICML2025"
  - "Image Generation"
  - "Transition Path Sampling"
  - "Onsager-Machlup Action"
  - "Diffusion Models"
  - "Flow Matching"
  - "Molecular Dynamics"
date: 2026-05-08
content_hash: 2f54b813da739b06
---

# Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional

**Conference**: ICML2025  
**arXiv**: [2504.18506](https://arxiv.org/abs/2504.18506)  
**Code**: [https://github.com/ASK-Berkeley/OM-TPS](https://github.com/ASK-Berkeley/OM-TPS)  
**Area**: Diffusion Models / Molecular Dynamics / Generative Models  
**Keywords**: Transition Path Sampling, Onsager-Machlup Action, Diffusion Models, Flow Matching, Molecular Dynamics

## TL;DR
This paper proposes interpreting the score functions of pretrained generative models (diffusion models and flow matching) as drift terms in stochastic dynamics. By minimizing the Onsager-Machlup (OM) action functional, the pretrained models are repurposed in a **zero-shot** manner for transition path sampling (TPS) in molecular systems. This achieves physically realistic transition paths at a fraction of the computational cost of traditional methods on systems like alanine dipeptide and fast-folding proteins.

## Background & Motivation

**Background**: Transition Path Sampling (TPS) is a core challenge in statistical mechanics: finding high-probability paths connecting two stable states on an energy landscape (e.g., chemical reactions, protein folding). Traditional methods include umbrella sampling, metadynamics, and shooting methods.

**Limitations of Prior Work**:
   - Traditional methods like metadynamics require defining appropriate collective variables (CVs), which is itself highly challenging near transition states.
   - Shooting methods suffer from low sampling efficiency, high rejection rates, and require expensive MD simulations.
   - Existing ML-based methods (reinforcement learning, differentiable simulation, h-transform learning) require specialized training for each system, failing to leverage existing large-scale atomic simulation data and pretrained generative models.

**Key Challenge**: Molecular conformation generative models (e.g., diffusion models) can generate independent unbiased samples. However, they are trained on uncorrelated static samples, making them unable to directly sample **time-dependent transition paths**.

**Goal**: How to repurpose pretrained generative models to sample high-probability transition paths between molecular systems without any TPS-specific training.

**Key Insight**: The authors observe that the score function $s_{\theta^*} \approx \nabla \log p_{\text{data}}(\mathbf{x})$ of diffusion and flow matching models can be interpreted as the drift term of a stochastic differential equation (SDE). Under this SDE, the probability of a path can be characterized by the Onsager-Machlup (OM) action functional—analogous to the principle of least action in physics.

**Core Idea**: Combine the stochastic dynamics induced by pretrained generative models with the OM action functional. By minimizing the OM action through gradient optimization, high-probability transition paths can be found, enabling zero-shot TPS using generative models.

## Method

### Overall Architecture

The pipeline of the proposed method consists of three steps:
1. **Initial Path Generation**: Linear interpolation between the two endpoints is performed in the latent space of the generative model to obtain an initial guess path.
2. **OM Action Optimization**: With the score function of the pretrained model frozen, the OM action functional of the path is minimized using gradient descent (equivalent to maximizing the path probability).
3. **Decoding Back to Conformation Space**: If optimized in the latent space, the path is decoded back to the atomic coordinate space through a denoising process.

The input is a pretrained generative model and two endpoint conformations $\mathbf{x}^{(0)} \in \mathcal{A}$ and $\mathbf{x}^{(L)} \in \mathcal{B}$; the output is a high-probability discretized path $\mathbf{X} = \{\mathbf{x}^{(i)}\}_{i=0}^{L}$ connecting the two endpoints.

### Key Designs

1. **Construction and Physical Interpretation of the OM Action Functional**:

    - Function: Defines the negative log-probability of a path as the OM action functional $S[\mathbf{X}]$; maximizing the path probability is equivalent to minimizing $S$.
    - Mechanism: The discretized OM action comprises three terms: **Term A** $\frac{1}{2\Delta t}\|\mathbf{x}^{(i+1)} - \mathbf{x}^{(i)}\|^2$ encourages smooth transitions between adjacent points; **Term B** $\frac{\Delta t}{2\zeta^2}\|\mathbf{\Phi}(\mathbf{x}^{(i)})\|^2$ encourages the path to pass through low-drift regions (energy extrema or saddle points); **Term C** $\frac{D\Delta t}{\zeta}\nabla \cdot \mathbf{\Phi}(\mathbf{x}^{(i)})$ encourages the path to pass through convex regions (dynamically more stable regions).
    - Design Motivation: The physical parameters $\Delta t$ (time step), $\zeta$ (damping coefficient), and $D$ (diffusion coefficient) control the relative contributions of the three terms, providing an intuitive physical control mechanism. For Boltzmann-distributed data, the learned score directly corresponds to the atomic force field $\mathbf{s}_{\theta^*} \propto -\nabla U(\mathbf{x}) = \mathbf{F}(\mathbf{x})$.

2. **Extracting Score Functions from Generative Models to Construct Stochastic Dynamics**:

    - Function: Extracts score functions from a pretrained DDPM or flow matching model to construct equivalent SDEs.
    - Mechanism (DDPM): Through an iterative "denoising-adding noise" process at a fixed time marginal $\tau$, the equivalent SDE is derived as: $d\mathbf{x} = \mathbf{s}_\theta(\mathbf{x}, \tau) dt + \sqrt{2} d\mathbf{W}_t$, where $\mathbf{s}_\theta$ is directly obtained from the denoising model.
    - Mechanism (Flow Matching): Demonstrates that the velocity field $u_\theta$ of flow matching can be analytically converted to a score: $\mathbf{s}_\theta^{\text{FM}} = \frac{\alpha_\tau}{\dot{\sigma}_\tau \sigma_\tau \alpha_\tau - \dot{\alpha}_\tau \sigma_\tau^2}(\frac{\dot{\alpha}_\tau}{\alpha_\tau}\mathbf{x} - u_{\theta^*}(\mathbf{x}, \tau))$.
    - Design Motivation: This ensures that the OM framework is not restricted to DDPM, but can be scaled to a wider class of generative models including flow matching.

3. **Latent Space Linear Interpolation Initialization**:

    - Function: Interpolates endpoints in the low-noise latent space $\tau_{\text{initial}}$ of the generative model to generate the initial path.
    - Mechanism: Directly interpolating linearly in the conformation space generates non-physical paths because the atomic conformation manifold is highly non-convex, whereas interpolating in the latent space produces samples much closer to the data manifold.
    - Design Motivation: High-quality initial paths are crucial for optimization convergence; the smoothness in latent space provides a better starting point.

4. **Divergence Acceleration based on Hutchinson Estimator**:

    - Function: Uses the Hutchinson randomized trace estimator to accelerate the computation of the divergence term $\nabla \cdot \mathbf{s}_{\theta^*}$ in the OM action.
    - Design Motivation: Exact computation of the divergence of a high-dimensional score is computationally prohibitive; the randomized estimator reduces the computational complexity to an acceptable level, enabling scalability to large-scale protein systems.

### Loss & Training
This method **requires no training**—its core advantage is the direct reuse of existing pretrained generative models. The optimization process only performs gradient descent on the path coordinates while the parameters of the generative model $\theta^*$ remain frozen throughout. All intermediate path points are optimized in parallel, making it naturally suited for multi-device acceleration. Additionally, a Truncated OM Action (which omits the divergence term C) is introduced as a simplified version under low-diffusion regimes.

## Key Experimental Results

### Main Results: Efficiency Comparison on Alanine Dipeptide

| Method | Requires CV | Force / Score Evaluations (↓) | Single Path Run Time (↓) |
|------|---------|--------------------------|-------------------|
| MCMC Shooting | No | ≥ 1B | ≥ 100 hours |
| Metadynamics | Yes | 1M | 10 hours |
| **OM Optimization (Diffusion, Ours)** | **No** | **10K** | **50 mins** |

The computational efficiency of OM optimization significantly outperforms traditional methods: score evaluations are reduced by **100x** compared to metadynamics, and by **100,000x** compared to MCMC shooting.

### Fast-Folding Protein Experimental Results (Average of 5 Proteins)

| Method | Jensen-Shannon Divergence (↓) | Fraction of Valid Paths (↑) | Transition Negative Log-Likelihood (↓) |
|------|------------------------|-----------------|-------------------|
| MD Simulation 1μs | ~0.45 | ~0.30 | ~4.5 |
| MD Simulation 10μs | ~0.30 | ~0.55 | ~3.8 |
| MD Simulation 50μs | ~0.20 | ~0.70 | ~3.5 |
| MD Simulation 100μs | ~0.18 | ~0.80 | ~3.2 |
| **OM Optimization (Diffusion)** | **~0.18** | **~0.90** | **~2.8** |
| **OM Optimization (Flow Matching)** | **~0.19** | **~0.88** | **~2.9** |

OM optimization outperforms or matches 100μs unbiased MD simulations across all three metrics while requiring substantially lower computational cost.

### Ablation Study

| Experimental Configuration | Key Findings |
|---------|---------|
| Variation of diffusion coefficient D | Increasing D $\rightarrow$ Paths cross higher energy barriers (corresponding to path behaviors at higher temperatures) |
| Variation of time step Δt | Increasing Δt $\rightarrow$ Path smoothness constraint is weakened, allowing larger "jumps" |
| Truncated vs Full OM | Truncated is sufficient at low diffusion rates; Full OM (with divergence term) is required at high diffusion rates |
| 99% of transition state data removed | Even when transition states are scarce in the training data, OM optimization can still identify reasonable paths |
| Zero-shot generalization (Tetrapeptides) | On 100 unseen tetrapeptides, the performance of OM optimization is close to that of 50-100ns MD simulations |

### Key Findings
- **Physical Parameters Provide Intuitive Control**: Adjusting the diffusion coefficient $D$ controls the height of the energy barriers that paths cross, which physically corresponds to behaviors at different temperatures.
- **Robustness to Data Scarcity**: Re-training the model after removing 99% of the transition state conformations still yields reasonable path sampling via OM optimization, demonstrating that the method does not rely on complete coverage of transition regions in the training set.
- **Zero-Shot Generalization**: In experiments on tetrapeptide systems, the TPS performance of the model on unseen sequences is comparable to 50-100ns MD simulations, showcasing the potential of pretrained models to generalize across chemical space.
- **Reaction Rate Estimation on the Müller-Brown Potential**: The estimated value is $1.3 \times 10^{-5}$ compared to the true value of $5.4 \times 10^{-5}$. Being within the same order of magnitude is considered accurate for reaction rate estimation tasks.

## Highlights & Insights

- **Zero-Shot Repurposing Paradigm of Pretrained Models**: The core contribution of this work is not a specific engineering trick, but rather the establishment of a complete paradigm: "Pretrained Generative Model $\rightarrow$ Score Extraction $\rightarrow$ OM Action Optimization $\rightarrow$ TPS". This means that as generative models advance (with larger data and better architectures), TPS capabilities will automatically upgrade without needing to redesign the TPS method itself.
- **Unified Treatment of DDPM and Flow Matching**: The authors prove that the velocity field of flow matching can be analytically converted to a score function (Eq. 17-18), making the OM framework universally applicable across generative model families. This velocity-to-score conversion formula has independent value.
- **Physically Interpretable Hyperparameters**: Unlike black-box ML methods, $\Delta t$, $\zeta$, and $D$ in the OM action directly correspond to physical quantities (time step, damping coefficient, and diffusion coefficient), allowing domain experts to tune parameters based on physical intuition.
- **Parallel Path Optimization**: Since the OM action is in a discretized integral form, all intermediate points along the path can be optimized in parallel, which is computationally highly efficient.

## Limitations & Future Work

- **No Guarantee of Complete Posterior Sampling**: Unlike traditional shooting methods, this method does not guarantee complete sampling from the path posterior distribution. Although the authors mitigate this through generative model encoding/decoding stochasticity and combination with MD/umbrella sampling, theoretical guarantees are still lacking.
- **Dependence on Pretrained Model Quality**: The performance ceiling of OM optimization is dictated by the quality of the score estimation of the pretrained generative model. If some regions are poorly learned, the optimized paths may deviate from physical ones.
- **Computational Overhead of the Divergence Term**: The divergence term $\nabla \cdot \mathbf{s}_{\theta^*}$ in the Full OM action incurs significant computational cost even with the Hutchinson estimator, which may become a bottleneck for very large systems.
- **Applicability to Coarse-Grained Protein Systems**: The fast-folding protein experiments utilize a C$\alpha$ coarse-grained representation; extensibility to all-atom systems remains to be verified.
- **Sensitivity to Initial Path**: Although latent space interpolation outperforms configuration space interpolation, the choice of the initial path may still dictate which path the optimization converges to in complex free-energy landscapes.

## Related Work & Insights

- **vs. Traditional TPS (Shooting / Metadynamics)**: Traditional methods require system-specific CV definitions or massive MD simulations. Ours does not rely on CVs and does not require system-specific training, reaching 2–5 orders of magnitude higher computational efficiency. However, traditional methods offer theoretical guarantees on distribution completeness.
- **vs. ML-TPS (Reinforcement Learning / h-transform)**: Works like Das et al. (2021) and Du et al. (2024) require training control policies for each specific system. Ours achieves zero-shot reuse of pretrained models, offering better scalability, though it cannot guarantee the exact path distribution like h-transform methods.
- **vs. Boltzmann Generators (Noé et al., 2019)**: Boltzmann Generators produce independent configuration samples without considering path information. Ours utilizes similar model classes but incorporates time-dependent path optimization.
- **vs. Arts et al., 2023 (Score as Force Field)**: This work uses the diffusion model score as a force field for MD simulation. Ours further integrates the score with the OM action functional, bypassing the entire computational overhead of step-by-step MD simulation.
- **Insight**: As an "interpolation on a manifold" method, OM action minimization can in principle be extended to any data modality—such as images, videos, and audio—as long as a pretrained generative model is available to provide score functions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The perspective of combining OM functional with generative models is highly original, unifying diffusion models and flow matching.
- Experimental Thoroughness: ⭐⭐⭐⭐ Progressively validated across four systems (2D $\rightarrow$ dipeptide $\rightarrow$ protein $\rightarrow$ tetrapeptide), although validation on all-atom systems remains insufficient.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations, with an excellent integration of physical intuition and mathematical formulation.
- Value: ⭐⭐⭐⭐⭐ Defines a new paradigm: zero-shot repurposing of pretrained generative models for TPS, which becomes increasingly valuable as model scale grows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Efficient Generative Modeling with Residual Vector Quantization-Based Tokens](efficient_generative_modeling_with_residual_vector_quantization-based_tokens.md)
- [\[ICML 2025\] DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space](dctdiff_intriguing_properties_of_image_generative_modeling_in_the_dct_space.md)
- [\[ICML 2025\] Compositional Scene Understanding through Inverse Generative Modeling](compositional_scene_understanding_through_inverse_generative_modeling.md)
- [\[ICCV 2025\] Accelerating Diffusion Sampling via Exploiting Local Transition Coherence](../../ICCV2025/image_generation/accelerating_diffusion_sampling_via_exploiting_local_transition_coherence.md)
- [\[ICML 2025\] DDIS: When Model Knowledge Meets Diffusion Model](when_model_knowledge_meets_diffusion_model_diffusion-assisted_data-free_image_synthesis.md)

</div>

<!-- RELATED:END -->
