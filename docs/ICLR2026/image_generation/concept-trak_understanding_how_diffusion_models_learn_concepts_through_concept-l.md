---
title: >-
  [Paper Note] Concept-TRAK: Understanding how diffusion models learn concepts through concept-level attribution
description: >-
  [ICLR2026][Image Generation][Diffusion Models] This paper proposes Concept-TRAK, which extends influence functions from image-level to concept-level attribution by designing concept-specific training losses (DPS reward)…
tags:
  - "ICLR2026"
  - "Image Generation"
  - "Diffusion Models"
  - "Data Attribution"
  - "Concept Attribution"
  - "Influence Functions"
  - "Copyright"
date: 2026-05-08
content_hash: 7156977b70b068c9
---

# Concept-TRAK: Understanding how diffusion models learn concepts through concept-level attribution

**Conference**: ICLR2026
**arXiv**: [2507.06547](https://arxiv.org/abs/2507.06547)  
**Code**: To be confirmed  
**Area**: Image Generation
**Keywords**: Diffusion Models, Data Attribution, Concept Attribution, Influence Functions, Copyright

## TL;DR
This paper proposes Concept-TRAK, which extends influence functions from image-level to concept-level attribution by designing concept-specific training losses (DPS reward) and utility losses (CFG guidance). The method substantially outperforms TRAK, D-TRAK, and DAS on synthetic, CelebA-HQ, and AbC benchmarks, with particularly significant advantages in OOD settings where novel concept combinations are evaluated.

## Background & Motivation

**Background**: Data attribution methods (TRAK, D-TRAK, DAS) employ influence functions to estimate the contribution of training samples to generated images, serving applications in copyright detection, data valuation, and model debugging. However, existing methods perform attribution at the whole-image level—identifying training samples that influence an entire generated image.

**Limitations of Prior Work**: Practical requirements demand concept-level attribution. For example, when generating "a pencil sketch of Pikachu," the copyright holder (Nintendo) cares about the training sources of the "Pikachu" concept, not the "pencil sketch" style. Whole-image attribution tends to return stylistically similar but conceptually irrelevant images.

**Key Challenge**: Both the utility loss and the training loss in influence functions are based on standard denoising objectives—they capture directions of overall reconstruction quality rather than concept-specific directions. New loss function designs are needed to isolate concept-specific influences.

**Goal**: To define and implement concept-level data attribution—quantifying each training sample's contribution to a diffusion model's ability to learn a specific concept (style, object, or attribute).

**Key Insight**: A geometric motivation: concept-relevant directions correspond to tangent vectors of the data manifold in the diffusion model's latent space. The reward optimization gradient $\nabla_{x_t} R(x_t)$ serves as a concept-specific guidance direction, pointing precisely to concept-enriched regions within the tangent space.

**Core Idea**: Use DPS reward gradients as the training loss (capturing the influence direction of training samples) and CFG guidance as the utility loss (capturing the target concept direction). Their inner product within the influence function framework measures each training sample's contribution to concept learning.

## Method

### Overall Architecture
The influence function takes the form: $\mathcal{I}(x_0^i, c_{\text{target}}) = \nabla_\theta \mathcal{L}_{\text{concept}}^\top \mathbf{H}^{-1} \nabla_\theta \mathcal{L}_{\text{train}}$. The key lies in designing $\mathcal{L}_{\text{concept}}$ (utility loss) and $\mathcal{L}_{\text{train}}$ (training loss) such that their gradient directions encode concept-specific information rather than global reconstruction information.

### Key Designs

1. **Training Loss (DPS Reward-based)**:

    - **Function**: Captures the concept-specific influence direction of training sample $x_0^i$ on the model's generative capability.
    - **Mechanism**: Define reward $R_{\text{train}}(x_t) = \log p(x_0^i | \hat{x}_0)$, where $\hat{x}_0 = \mathbb{E}[x_0|x_t]$ is the posterior mean. Assuming a Gaussian distribution, the reward gradient reduces to $\nabla_{x_t} \|{\hat{x}_0 - x_0^i}\|^2$—a gradient that operates within the tangent space of the data manifold (guaranteed by DPS theory).
    - **Final Training Loss**: $\mathcal{L}_{\text{train}} = \mathbb{E}_{x_t}[\|\text{sg}[\epsilon_\theta(x_t;c) + \lambda_t \nabla_{x_t}\|\hat{x}_0 - x_0^i\|^2] - \epsilon_\theta(x_t;c)\|^2]$
    - **Design Motivation**: Standard DSM loss provides a reconstruction-driven signal, whereas the DPS reward gradient provides a tangent-space guidance vector that is more stable for concept attribution.

2. **Utility Loss (CFG-based Concept Loss)**:

    - **Function**: Measures the model's generative capability for target concept $c_{\text{target}}$.
    - **Mechanism**: The concept reward is defined as $R_{\text{concept}}(x_t) = \log p(c_{\text{target}} | x_t)$. When $c_{\text{target}}$ can be used as a conditional input, the reward gradient simplifies to the classifier-free guidance vector: $\epsilon_\theta(x_t; c_{\text{target}}) - \epsilon_\theta(x_t)$. For concepts embedded within composite conditions, concept slider guidance is applied: $\epsilon_\theta(x_t; c) - \epsilon_\theta(x_t; c_{-})$, where $c_{-}$ is the condition with the target concept removed.
    - **Design Motivation**: CFG vectors have been shown to encode concept-specific information in the tangent space of the data manifold, consistent with the proposed geometric framework.

3. **Auxiliary Techniques**:

    - **DDIM Inversion for Deterministic Sampling**: Eliminates stochasticity in the forward diffusion process, improving gradient stability.
    - **Global vs. Local Attribution**: Global attribution identifies the sources of a concept across all generated images; local attribution identifies the source of a concept in a specific generated image.
    - **Gradient Normalization**: Gradients are normalized to unit norm at each timestep to prevent any single timestep from dominating the attribution score, while also rendering the method insensitive to hyperparameters $\beta$ and $\sigma_{\text{data}}$.

### Loss & Training
No training is required—Concept-TRAK is a training-free attribution method. It only requires precomputing projected gradients for the training set (within the TRAK framework) and then querying with the gradients of concept-specific losses.

## Key Experimental Results

### Main Results (Concept-Level Attribution Precision@10)

| Method | Synthetic ID | Synthetic OOD | CelebA ID | CelebA OOD |
|--------|-------------|---------------|-----------|------------|
| TRAK | 0.80 | 0.45 | - | - |
| D-TRAK | 1.00 | 0.50 | - | - |
| DAS | 1.00 | 0.50 | 0.96 | 0.67 |
| **Concept-TRAK** | **1.00** | **0.85** | **0.92** | **0.97** |

### Ablation Study (AbC Benchmark, T2I Model)

| Configuration | AbC Metric | Notes |
|--------------|-----------|-------|
| TRAK (whole-image attribution) | Low | Returns stylistically similar images |
| D-TRAK | Medium | Still image-level |
| Unlearning-based | Medium | High computational cost |
| **Concept-TRAK** | **Best** | Precise concept attribution |

### Key Findings
- **OOD settings are the critical differentiator**: In ID settings, whole-image attribution coincidentally identifies concepts because visually similar training samples exist. In OOD settings, the model combines concepts that never co-occurred during training, causing whole-image attribution to fail while Concept-TRAK continues to attribute correctly.
- **CelebA OOD: Concept-TRAK 0.97 vs. DAS 0.67**: The 30-point gap demonstrates the critical importance of concept-specific loss design.
- **Effectiveness of CFG vectors as utility loss**: This confirms that CFG vectors indeed encode concept-specific directions and are applicable not only for guided generation but also for attribution.
- **DPS reward is more stable than DSM**: Tangent-space guidance versus global reconstruction—the former introduces less noise for concept attribution.

## Highlights & Insights
- **Definition of a new task—concept-level data attribution**: The transition from whole-image to concept-level attribution is a qualitative leap that directly corresponds to practical needs such as copyright detection and safety auditing. The problem formulation itself constitutes a contribution.
- **An attribution perspective on reward optimization**: DPS/CFG reward gradients can not only guide sampling but also precisely characterize concept influence directions, opening a new window into the interpretability of diffusion models.
- **Elegant geometric framework**: The derivation chain from tangent space → reward gradients → concept directions connects three research areas—influence functions, diffusion model geometry, and reward optimization.
- **Clever OOD evaluation design**: Deliberately excluding specific concept combinations forces the model to compositionally generalize, and tests whether attribution can disentangle individual concepts—more convincing than simple ID testing.

## Limitations & Future Work
- **Concepts must be expressible as conditional inputs**: When concepts cannot be expressed as text conditions (e.g., abstract styles, compositional rules), more general concept representations are needed. The paper discusses extensions to visual concepts in the appendix but does not provide sufficient validation.
- **Computational cost**: Precomputing projected gradients for all training samples is required—still expensive for million-scale training sets such as LAION.
- **Validation limited to small-scale models**: Models used in the Synthetic and CelebA experiments are relatively small. Validation at the Stable Diffusion / DALL-E scale is only partially conducted on the AbC benchmark.
- **Future directions**: (a) Concept localization at the image token level—identifying not only which training samples contribute a concept, but also which region of the generated image they contribute to; (b) Integration with concept erasing for precise concept unlearning.

## Related Work & Insights
- **vs. TRAK / D-TRAK / DAS**: These methods perform whole-image attribution using standard denoising losses or their variants. The core innovation of Concept-TRAK lies in designing concept-specific training and utility loss functions.
- **vs. Unlearning-based attribution**: This approach evaluates influence by retraining after removing training data—the most accurate but most expensive approach. Concept-TRAK approximates this with influence functions, achieving orders-of-magnitude greater efficiency.
- **vs. Concept Sliders**: Concept Sliders manipulate CFG vectors to edit concepts; Concept-TRAK uses the same CFG vectors for concept attribution—a unification of two complementary directions.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Defines a new task (concept-level attribution) + reward-based loss design with geometric motivation = dual innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three-tier evaluation on Synthetic, CelebA, and AbC with well-designed OOD settings; systematic validation on large-scale T2I models is lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The narrative arc from problem definition → geometric motivation → reward derivation → empirical validation is exceptionally coherent.
- **Value**: ⭐⭐⭐⭐⭐ Direct and urgent practical value for AI copyright protection and model transparency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [\[ICLR 2026\] Localized Concept Erasure in Text-to-Image Diffusion Models via High-Level Representation Misdirection](localized_concept_erasure_in_text-to-image_diffusion_models_via_high-level_repre.md)
- [\[ICLR 2026\] Temporal Concept Dynamics in Diffusion Models via Prompt-Conditioned Interventions](temporal_concept_dynamics_in_diffusion_models_via_prompt-conditioned_interventio.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)
- [\[ICCV 2025\] Erasing More Than Intended? How Concept Erasure Degrades the Generation of Non-Target Concepts](../../ICCV2025/image_generation/erasing_more_than_intended_how_concept_erasure_degrades_the_generation_of_non-ta.md)

</div>

<!-- RELATED:END -->
