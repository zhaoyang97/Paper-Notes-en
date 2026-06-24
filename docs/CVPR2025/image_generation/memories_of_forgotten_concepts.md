---
title: >-
  [Paper Note] Memories of Forgotten Concepts
description: >-
  [CVPR 2025][Image Generation][Concept Ablation] This paper reveals a fundamental flaw in concept ablation methods for diffusion models: by finding highly likely latent seeds through diffusion inversion, it demonstrates that information about erased concepts still resides within the model, allowing high-quality images of the ablated concepts to be reconstructed from multiple distinct seed vectors.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Concept Ablation"
  - "Diffusion Models"
  - "Machine Unlearning"
  - "Latent Space Analysis"
  - "Privacy & Safety"
date: 2026-05-08
content_hash: 2a28c48f931ab0e4
---

# Memories of Forgotten Concepts

**Conference**: CVPR 2025  
**arXiv**: [2412.00782](https://arxiv.org/abs/2412.00782)  
**Code**: [https://github.com/matanr/Memories_of_Forgotten_Concepts](https://github.com/matanr/Memories_of_Forgotten_Concepts)  
**Area**: Diffusion Models / AI Safety  
**Keywords**: Concept Ablation, Diffusion Models, Machine Unlearning, Latent Space Analysis, Privacy & Safety

## TL;DR
This paper reveals a fundamental flaw in concept ablation methods for diffusion models: by finding highly likely latent seeds through diffusion inversion, it demonstrates that information about erased concepts still resides within the model, allowing high-quality images of the ablated concepts to be reconstructed from multiple distinct seed vectors.

## Background & Motivation

1. **Background**: Diffusion models dominate the text-to-image generation domain, but they can generate unsafe content (violence, nudity) or violate privacy datasets. Consequently, researchers have developed various concept ablation techniques to restrict the generation of specific concepts.

2. **Limitations of Prior Work**: Existing concept ablation methods (e.g., ESD, FMN, Salun, etc.) suffer from a fundamental flaw in their evaluation: they only verify the erasure effect at the output image level. That is, given an erased concept's text and a random seed, they check if the generated image still contains the ablated concept. This evaluation neglects the possibility that seeds capable of generating the ablated concept might still exist in the latent space.

3. **Key Challenge**: Concept ablation methods mainly block concept generation via text proxies but do not genuinely remove the concept information from the latent space of the model. A large number of seed vectors capable of generating the ablated concepts may still reside in the high-dimensional latent space of the model.

4. **Goal**: Propose a systematic analysis framework to quantitatively evaluate whether concept ablation models truly "forget" the erased concepts.

5. **Key Insight**: Hypothesis—An effectively ablated model should not contain highly likely seed vectors capable of generating high-quality images of the ablated concepts. This hypothesis is verified using diffusion inversion.

6. **Core Idea**: Use diffusion inversion to search for seed vectors in the latent space of the ablated model, revealing that the likelihood of these seeds is comparable to those of normal images, thereby proving that the concept information still persists in the model.

## Method

### Overall Architecture
The input consists of an ablated diffusion model and a set of query images containing the ablated concept. The method first encodes the query images into the latent space using a VAE encoder to obtain $z_0$. Then, the seed vector $z_T$ is obtained via diffusion inversion (using the Renoise method). This seed is subsequently used to perform inference through the ablated model to generate reconstructed images. Finally, the negative log-likelihood (NLL) of the seed $z_T$ and the peak signal-to-noise ratio (PSNR) of the reconstructed images are analyzed.

### Key Designs

1. **Relative Distance Metric**:

    - **Function**: Quantify the effectiveness of concept ablation.
    - **Mechanism**: Compute the ratio of the Earth Mover's Distance between the NLL distribution of the ablated set $E$ and the reference set $R$ relative to a standard normal distribution $\mathcal{N}$: $d_\mathcal{N}(E,R) = \text{EMD}(\text{NLL}(E), \text{NLL}(\mathcal{N})) / \text{EMD}(\text{NLL}(R), \text{NLL}(\mathcal{N}))$. A ratio close to 1 suggests that the ablation failed (seeds for the ablated concepts are as "normal" as those for the reference concepts), while a higher ratio denotes more effective erasure.
    - **Design Motivation**: Relying solely on raw NLL values makes interpretation difficult; using a dimensionless relative ratio provides a more intuitive measurement of the degree of ablation.

2. **Memory of an Ablated Concept**:

    - **Function**: Verify at the dataset level whether the ablated model can still generate the erased concepts.
    - **Mechanism**: For each image in the ablated set, find a corresponding seed $z_T$ using the Renoise inversion method (50 inversion steps, 5 renoising steps), and then analyze the likelihood of the seed and the PSNR of the generated image.
    - **Design Motivation**: If a highly likely seed with high PSNR can be found for each ablated concept image, it indicates that the model has not truly forgotten the concept.

3. **Sequential Inversion Block (SIB)**:

    - **Function**: Demonstrate that for the same ablated image, there exist multiple, distantly located seeds that can all reconstruct the image.
    - **Mechanism**: Using a random support image as the starting point, three sequential inversion steps are performed: (1) VAE encoder/decoder inversion is applied to obtain $z_0^{(s_i)}$ of the support image; (2) starting from this point, optimization is performed to find $z_0^{(s_i \to q)}$ which can reconstruct the query image; (3) diffusion inversion is applied to this latent variable to find the seed $z_T^{(s_i \to q)}$. The seeds generated from different support images exhibit large cosine distances (~0.58-0.69) from each other, yet all reconstruct the query image.
    - **Design Motivation**: If only a single seed points to the ablated image, it might be easily overlooked; however, multiple scattered seeds prove that the concept has a broad "memory" in the latent space.

### Loss & Training
This paper does not train a model and is a purely analytical work. The VAE decoder inversion is optimized using Euclidean distance, and the diffusion inversion employs the Renoise method.

## Key Experimental Results

### Main Results

Comprehensive evaluation conducted across 9 concept ablation methods and 6 concepts:

| Method | PSNR (Church) | PSNR (Nudity) | $d_\mathcal{N}$ (Max) |
|------|-------------|--------------|----------------------|
| ESD | ~27 dB | ~32 dB | 2.49 (Parachute) |
| FMN | ~26 dB | ~31 dB | <2.0 |
| Salun | ~27 dB | ~30 dB | <2.0 |
| EraseDiff | ~27 dB | ~31 dB | <2.0 |
| Vanilla SD | ~28 dB | ~33 dB | ~1.0 (Baseline) |

All methods can generate reconstructed images of ablated concepts with PSNR $\ge 25$ dB, with the highest relative distance being only 2.49.

### Ablation Study (Multi-Memory Analysis)

| Analytical Dimension | Results | Description |
|---------|------|------|
| Average Cosine Distance | 0.58-0.69 | Large distance between different seeds, confirming distinct "memories" |
| Distance to Original Seed | Mean 152.14, SD 2.72 | Coefficient of variation is only 2%, seeds are distributed on a hypersphere centered at the target seed |
| PSNR (Multi-Memory) | ~20-28 dB | Slightly lower than single-memory, but still highly recognizable |
| Relative Distance (Multi-Memory) | Lower than single-memory | Sub-optimal when searching for multiple seeds, but still within a reasonable range |

### Key Findings
- None of the 9 concept ablation methods can truly erase the concepts across the 6 evaluated categories; the likelihood of seeds associated with the ablated concepts overlaps with those of normal images.
- Concepts with fine textures (e.g., Van Gogh, Church) yield lower PSNR, while smoother concepts (e.g., Nudity, Parachute) obtain higher PSNR.
- Even the most effective ablation setup (ESD on Parachute) only achieves a relative distance of 2.49, which is far from sufficient to claim the concept is completely forgotten.
- Multi-memory analysis reveals that the seeds are distributed on a sphere centered around the target seed (coefficient of variation of only 2%), indicating an interesting geometric structure.

## Highlights & Insights
- **Latent Space Likelihood Analysis Paradigm**: Evaluating the erasure effect in the latent space rather than at the output image level represents a fundamental shift in perspective. Prior works focused on "given a text, see what is generated," whereas this paper shifts to "given a target image, see if a plausible seed can be found." This approach can be generalized to safety evaluations of any generative model.
- **Sequential Inversion Block**: Finding multiple diverse seeds through support images is a highly clever design that avoids the difficulty of direct high-dimensional space searching. This method can be transferred to tasks like adversarial sample search.
- **Discovery of Spherical Distribution**: The geometric insight that multiple seeds are distributed on a sphere implies structural properties of the diffusion model's latent space, providing key inspirations for understanding latent spaces.

## Limitations & Future Work
- Requires white-box access to the model weights, making it inapplicable directly to black-box APIs (such as commercial models like DALL-E).
- The analysis is based on Stable Diffusion (SD) v1.4 and has not been extended to newer models (such as SD-XL, Flux, etc.), where new architectures might exhibit different behaviors.
- Scrambled image experiments (Fig. 10) suggest that the inversion capability might be overly powerful, necessitating a more careful distinction between "the model remembered the concept" and "the inversion method is too strong."
- The authors do not propose a solution to improve concept ablation; it is a purely analytical work.

## Related Work & Insights
- **vs Pham et al.**: Pham uses Textual Inversion to search the text embedding space, whereas this work searches the $z_T$ latent space and analyzes likelihood, which is higher-dimensional and more quantitative.
- **vs Zhang et al. (AdvUnlearn)**: Zhang uses adversarial prompts to attack erased models, which requires finding specific texts; this work assumes the target image is known, rendering the analysis more comprehensive.
- **Implications for Diffusion Model Safety**: This serves as a cautionary tale for research on diffusion model safety: merely erasing concepts at the text level is insufficient; security measures must be considered from the perspective of the latent space.

## Rating
- Novelty: ⭐⭐⭐⭐ Analyzing concept ablation from the perspective of latent space likelihood is a fresh perspective, though the core technique (Renoise inversion) is derived from existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison involving 9 methods × 6 concepts, doubly validated by single-memory and multi-memory analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is well-structured, with beautifully designed figures (especially the visualizations in Figs. 1-3) and rigorous argumentation.
- Value: ⭐⭐⭐⭐ An important warning for the field of concept ablation, though its practical utility is somewhat limited as it does not present a constructive solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts](../../ICCV2025/image_generation/meta-unlearning_on_diffusion_models_preventing_relearning_unlearned_concepts.md)
- [\[NeurIPS 2025\] When Are Concepts Erased From Diffusion Models?](../../NeurIPS2025/image_generation/when_are_concepts_erased_from_diffusion_models.md)
- [\[CVPR 2025\] Concept Replacer: Replacing Sensitive Concepts in Diffusion Models via Precision Localization](concept_replacer_replacing_sensitive_concepts_in_diffusion_models_via_precision_.md)
- [\[ICML 2025\] Nonparametric Identification of Latent Concepts](../../ICML2025/image_generation/nonparametric_identification_of_latent_concepts.md)
- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](../../NeurIPS2025/image_generation/emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)

</div>

<!-- RELATED:END -->
