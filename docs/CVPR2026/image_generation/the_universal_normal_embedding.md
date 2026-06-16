---
title: >-
  [Paper Note] The Universal Normal Embedding
description: >-
  [CVPR 2026][Image Generation][Paper Note] This paper proposes the Universal Normal Embedding (UNE) hypothesis: generative models (diffusion models) and visual encoders (CLIP, DINO) share an underlying geometric structure in their latent spaces that is approximately Gaussian. Both can be viewed as noisy linear projections of this shared space. The hypothesis is
tags:
  - CVPR 2026
  - Image Generation
date: 2026-05-08
content_hash: 744f347c65795929
---
# The Universal Normal Embedding

**Conference**: CVPR 2026  
**arXiv**: [2603.21786](https://arxiv.org/abs/2603.21786)  
**Code**: [https://github.com/](https://github.com/) (Stated open source, including the NoiseZoo dataset)  
**Area**: Diffusion Models / Representation Learning  
**Keywords**: Latent Space Gaussianity, Generation-Encoding Unification, DDIM Inversion, Linear Semantic Editing, Representation Geometry

## TL;DR

This paper proposes the Universal Normal Embedding (UNE) hypothesis: generative models (diffusion models) and visual encoders (CLIP, DINO) share an underlying geometric structure in their latent spaces that is approximately Gaussian. Both can be viewed as noisy linear projections of this shared space. The hypothesis is validated through the NoiseZoo dataset and extensive experiments, demonstrating the capability to perform direct linear semantic editing within the DDIM inversion noise space.

## Background & Motivation

**Background**: Generative models (VAE, GAN, Diffusion) and visual encoders (CLIP, DINO) typically evolve along independent technical trajectories—the former optimizing image synthesis quality and the latter optimizing semantic representation. However, prior research has identified two compelling phenomena: (1) models within the same family can "stitch" their latent spaces via simple linear mappings; (2) cross-architecture and cross-modal encoders also exhibit linear alignability properties.

**Limitations of Prior Work**: Although theoretical frameworks like the Platonic Representation Hypothesis predict that different models will converge to a shared latent representation, they do not explicitly define the **geometric structure** of this shared space. In practical applications, semantic editing for diffusion models relies on text prompts, structural modifications, or additional fine-tuning, lacking a direct editing method derived from latent space geometry.

**Key Challenge**: The latent spaces of encoders naturally possess semantic linear separability (classifiable via linear probes), but does the noise space of generative models possess an equivalent semantic structure? If both indeed originate from the same underlying space, it should be possible to perform linear semantic operations directly in the noise space—yet this has not been systematically verified.

**Goal**: (1) Formalize the "shared Gaussian latent space" hypothesis and provide empirical support; (2) Verify whether DDIM inversion noise in diffusion models encodes semantic information comparable to encoders; (3) Demonstrate the feasibility of performing direct linear semantic editing in the noise space.

**Key Insight**: Generative models sample from Gaussian noise to generate images, while encoders map images to embeddings that are empirically approximately Gaussian—these two directions are actually two "perspectives" of the same Gaussian latent space. The authors formalize this as Induced Normal Embedding (INE): each model's latent space is a noisy linear projection of the ideal UNE.

**Core Idea**: Since encoder embeddings and DDIM inversion noise are both linear projections of the same underlying Gaussian latent space, linear probes can be used to discover semantic directions in the noise space for direct, controllable editing.

## Method

### Overall Architecture

This paper does not propose a new network architecture but rather a theoretical hypothesis (UNE) validated through systematic experiments. The core pipeline involves: (1) Constructing the NoiseZoo dataset—extracting embeddings from multiple encoders and DDIM inversion noise from multiple diffusion models for CelebA images; (2) Verifying the Gaussianity of each model's latent space; (3) Training linear probes to test semantic separability; (4) Testing alignment via cross-space linear mapping; (5) Performing semantic editing along linear probe directions; (6) Recovering the shared multi-model subspace via GCCA.

### Key Designs

**1. Formalizing the UNE Hypothesis: Hardening the intuition of "representation convergence" into a specific Gaussian geometric constraint.**

The Platonic Representation Hypothesis suggests different models converge to the same representation but does not specify its form. This paper posits that there exists an ideal multivariate standard normal latent space $Z \sim \mathcal{N}(0, I)$, termed the Universal Normal Embedding (UNE), while each practical model $i$ has a latent space that is merely a noisy linear projection of it—called the Induced Normal Embedding (INE):

$$\hat{Z}_i = C_i Z + \epsilon_i$$

This simple formula has two direct corollaries. First, when noise is negligible and the projection matrix $C_i$ is invertible, semantically linearly separable features in the UNE remain linearly separable in every INE—explaining why encoder latent spaces work with linear probes. Second, in directions preserved by multiple INEs (their intersection), semantics are consistent across all models—forming the basis for cross-model linear alignment. Using Gaussianity as a core constraint ensures that semantic changes naturally correspond to linear directions, making "linear probes" and "linear editing" geometric necessities rather than empirical tricks.

**2. NoiseZoo Dataset: Aligning encoder embeddings and diffusion noise in a paired table to verify cross-family alignment.**

To test if generative models and encoders share a latent space, one must obtain paired representations of the same image across both model types—previous stitching work only compared models within the same family. NoiseZoo fills this gap. The authors used approximately 19k face images from the CelebA validation set, extracting representations from both sides: for encoders, embeddings from 5 models (CLIP ViT-B/16, CLIP ViT-L/14, OpenCLIP ViT-B/16, OpenCLIP ViT-L/14, DINOv3) with dimensions 500–1k; for generators, noise obtained via DDIM inversion from 3 diffusion models (SD 1.5, SD 2.1, LCMv7) with dimensions around 16k. Data was split 15k/4k for training/testing. Because each face has corresponding coordinates in 8 latent spaces, it provides a unified experimental foundation for Gaussianity testing, cross-space mapping, and shared subspace recovery.

**3. Linear Semantic Editing and Orthogonal Disentanglement: Once the hypothesis holds, modifying attributes becomes a simple vector addition in noise space.**

This is the most impactful actionable corollary of the UNE hypothesis. Since the noise space is Gaussian and semantics are distributed along linear directions, modifying an attribute no longer requires prompts, fine-tuning, or architectural changes—one simply finds the attribute direction in the DDIM inversion noise and shifts it. Specifically, a linear classifier is trained in the noise space using logistic regression; its normal vector $w$ represents the attribute direction. Editing is performed as $\tilde{z} = z + \alpha w$, where $\alpha$ controls intensity, before decoding $\tilde{z}$ back to an image. To handle attribute entanglement (e.g., adding a beard changing face shape), the paper uses orthogonalization: projecting the target direction $w_1$ onto the null space of the interfering attribute direction $w_2$,

$$\tilde{w}_1 = w_1 - \frac{w_2 w_2^\top}{w_2^\top w_2}\, w_1$$

Editing along $\tilde{w}_1$ modifies only the target attribute. For example, to "add a smile" without changing age, one subtracts the age component from the smile direction; decoding the result enhances the smile while keeping the perceived age constant. This disentanglement works because projecting to the null space is the most natural geometric decorrelation in Gaussian space.

### Loss & Training

This paper does not involve training new networks. Linear probes are trained using standard Logistic Regression (L2 regularization). Cross-space mapping uses Ridge Regression. Shared subspace recovery uses the MAXVAR form of GCCA (Generalized CCA), which has a closed-form solution.

## Key Experimental Results

### Main Results

Gaussianity Testing (1D random projection, Anderson-Darling pass rate):

| Model | AD Pass Rate ↑ | Type |
|------|------------|------|
| SD 1.5 | 96.00% | Generative |
| SD 2.1 | 95.80% | Generative |
| LCMv7 | 95.58% | Generative |
| CLIP B16 | 89.50% | Encoder |
| CLIP L14 | 91.90% | Encoder |
| DINOv3 | 84.48% | Encoder |
| Bimodal Gaussian (Control) | 15.88% | Non-Gaussian |

Accuracy drop after cross-space linear mapping:

| Generative Model → Encoder | Cosine Similarity | Accuracy Drop |
|-------------------|-----------|-----------|
| SD 1.5 → CLIP B16 | 0.80 | 0.20 pp |
| SD 2.1 → CLIP B16 | 0.80 | 0.14 pp |
| LCM → CLIP B16 | 0.81 | 0.00 pp |

### Ablation Study

Shared subspace classification (16-dim PCA vs. Shared Space):

| Space | 16d Classification Accuracy | Description |
|------|---------------|------|
| CLIP B16 (PCA-16d) | ~79% | Single model low-dim |
| SD 1.5 (PCA-16d) | ~77% | Single model low-dim |
| Shared Space X1 (16d) | ~78% | 4-model intersection |
| Shared Space X5 (16d) | ~77% | 6-model intersection |

### Key Findings

- **Extremely strong Gaussianity in diffusion noise space**: SD 1.5 achieved a 96% AD pass rate, close to the theoretical 95% boundary. Encoders ranged from 84-92%, far higher than the non-Gaussian control.
- **Noise space contains rich linearly separable semantics**: Across 40 CelebA attributes, linear probe accuracy in DDIM noise correlates highly with CLIP, matching almost attribute-for-attribute.
- **Minimal cross-space linear mapping error**: After linear mapping from generative models to encoders, classification accuracy dropped by less than 0.3 percentage points, proving the two types of spaces are indeed linearly aligned.
- **Low-dimensional shared space preserves substantial attribute information**: A shared subspace of only 16 dimensions can achieve classification performance close to single-model PCA-16d.
- Linear editing in noise space appears natural and smooth (smile, gender, age, etc.), and orthogonalization effectively eliminates attribute entanglement.

## Highlights & Insights

- **"Generation and encoding are two sides of the same coin"**: This conceptual insight is remarkably elegant. Once the UNE hypothesis is accepted, many empirical findings regarding cross-model alignment fall into a unified explanatory framework. This perspective can guide the design of future foundation models with simultaneous understanding and generation capabilities.
- **Research value of the NoiseZoo dataset**: The combination of paired encoder embeddings and diffusion noise is a unique research resource that could catalyze significant future work in latent space geometric analysis.
- **Semantic editing without additional training**: Controllable editing (changing smile, age, gender, etc.) is achieved simply via vector addition in the noise space, with orthogonal disentanglement being simple yet effective. This is cleaner than existing prompt engineering or fine-tuning methods.
- **An "executable hypothesis" distinct from pure theory**: UNE is not just a macro-conjecture; it immediately yields testable predictions (Gaussianity, linear separability, cross-model alignment, low-dim shared space), all of which were experimentally validated.

## Limitations & Future Work

- Experiments were validated only on CelebA face data and did not extend to more diverse domains like natural scenes (ImageNet) or medical imaging—the universality of UNE requires broader testing.
- Only 3 models from the Stable Diffusion family were used; other generative architectures (e.g., DALL-E 3, Flux, Consistency Models) were not verified.
- The shared subspace was recovered via GCCA, but it was not compared against stronger non-linear alignment methods (the authors intentionally used linear methods to verify the hypothesis, but non-linear methods might perform better in practice).
- DDIM inversion noise dimensions are extremely high (~16k), leading to significant storage and computational overhead in practical applications.
- Gaussianity slightly decreased in encoders (especially DINOv3 at 84%); whether systematic deviations exist requires deeper analysis.
- Quantitative evaluation of semantic editing (FID, LPIPS, attribute accuracy, etc.) was insufficient.

## Related Work & Insights

- **vs Platonic Representation Hypothesis**: PRH proposes a macro-conjecture of "models converging to a shared representation" but does not specify geometry. UNE defines Gaussianity as a key geometric constraint and unifies the encoder and generator families.
- **vs Latent linear stitching work (LIT, Model Stitching)**: These works prove linear alignment within the same family; UNE's contribution is extending alignment across families (Encoder ↔ Generator).
- **vs StyleGAN latent space editing**: While StyleGAN's W/W+ spaces support linear editing, diffusion models lack a persistent latent code. UNE shows that DDIM inversion noise naturally possesses a similar linear semantic structure.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Extremely high conceptual innovation by unifying encoders and generative models under the same Gaussian latent space hypothesis.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive testing of Gaussianity, cross-space mapping, linear editing, and shared subspaces, though restricted to the CelebA dataset.
- Writing Quality: ⭐⭐⭐⭐⭐ Fluent narrative from hypothesis to theory to experiment, logical clarity, and excellent chart design.
- Value: ⭐⭐⭐⭐⭐ Proposes a unified perspective that could profoundly influence the fields of representation learning and generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Monocular Normal Estimation via Shading Sequence Estimation](../../ICLR2026/image_generation/monocular_normal_estimation_via_shading_sequence_estimation.md)
- [\[CVPR 2026\] MatPedia: A Universal Generative Foundation for High-Fidelity Material Synthesis](matpedia_a_universal_generative_foundation_for_high-fidelity_material_synthesis.md)
- [\[CVPR 2026\] Premier: Personalized Preference Modulation with Learnable User Embedding in Text-to-Image Generation](premier_personalized_preference_modulation_with_learnable_user_embedding_in_text.md)
- [\[CVPR 2026\] Test-Time Alignment of Text-to-Image Diffusion Models via Null-Text Embedding Optimisation](test-time_alignment_of_text-to-image_diffusion_models_via_null-text_embedding_op.md)
- [\[ICCV 2025\] OminiControl: Minimal and Universal Control for Diffusion Transformer](../../ICCV2025/image_generation/ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)

</div>

<!-- RELATED:END -->
