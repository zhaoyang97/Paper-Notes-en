---
title: >-
  [Paper Note] Guiding a Diffusion Model by Swapping Its Tokens
description: >-
  [CVPR 2026][Image Generation][Sampling Guidance] This paper proposes Self-Swap Guidance (SSG), a training-free sampling guidance method for diffusion models that constructs perturbations by selectively swapping the most semantically dissimilar token pairs in the intermediate representation space. Compared to SAG/PAG/SEG, SSG stably generates high-fidelity images over a wider range of guidance scales, achieving state-of-the-art FID on both conditional and unconditional generation.
tags:
  - CVPR 2026
  - Image Generation
  - Sampling Guidance
  - Training-Free Guidance
  - Token Swapping
  - Self-Perturbation
  - Image Fidelity
date: 2026-05-08
content_hash: 39dbe5c451c869ed
---

# Guiding a Diffusion Model by Swapping Its Tokens

**Conference**: CVPR 2026
**arXiv**: [2604.08048](https://arxiv.org/abs/2604.08048)
**Code**: [https://github.com/VISION-SJTU/SSG](https://github.com/VISION-SJTU/SSG)
**Area**: Diffusion Models / Image Generation
**Keywords**: Sampling Guidance, Training-Free Guidance, Token Swapping, Self-Perturbation, Image Fidelity

## TL;DR
This paper proposes Self-Swap Guidance (SSG), a training-free sampling guidance method for diffusion models that constructs perturbations by selectively swapping the most semantically dissimilar token pairs in the intermediate representation space. Compared to SAG/PAG/SEG, SSG stably generates high-fidelity images over a wider range of guidance scales, achieving state-of-the-art FID on both conditional and unconditional generation.

## Background & Motivation

1. **Background**: Classifier-Free Guidance (CFG) is a cornerstone technique for high-quality diffusion model generation, steering the sampling direction by contrasting conditional and unconditional predictions. However, CFG requires text conditioning and thus cannot be applied to unconditional generation (e.g., inverse problem solving), and is prone to oversaturation and reduced diversity at high guidance scales.
2. **Limitations of Prior Work**: Recent training-free guidance methods (SAG, PAG, SEG, TSG) construct a "weakened" branch by perturbing the model's forward pass, but all adopt global, indiscriminate perturbation strategies—SAG adds noise to inputs, PAG corrupts attention maps. Such coarse-grained perturbations are either too weak (insufficient detail) or too strong (oversaturation/oversimplification), and are highly sensitive to the guidance scale, yielding a narrow effective operating range.
3. **Key Challenge**: Effective guidance requires sufficiently strong perturbations, yet perturbations that are too strong cause unrecoverable distortions. Existing methods lack fine-grained control over perturbation magnitude.
4. **Goal**: Design a fine-grained, controllable perturbation mechanism that introduces no external noise, enabling stable and effective guidance over a broader parameter range.
5. **Key Insight**: Operate at the token level rather than the global level—perturb via swapping rather than noise injection. Swapping is a conservative operation that rearranges existing information without introducing new content, naturally preserving global consistency while disrupting local structure.
6. **Core Idea**: Selectively swap the most semantically dissimilar token pairs to produce fine-grained local perturbations, replacing the conventional global noise-injection guidance paradigm.

## Method

### Overall Architecture
During diffusion model inference, two parallel branches are maintained: the original branch (unmodified) produces a clean prediction $\epsilon_{\text{ori}}$, while the perturbed branch applies token swapping to produce $\epsilon_{\text{pert}}$. The guidance formula follows the standard training-free guidance formulation: $\tilde{\epsilon}(x_t) = \epsilon_{\text{ori}}(x_t) + \omega(\epsilon_{\text{ori}}(x_t) - \epsilon_{\text{pert}}(x_t))$. Intermediate predictions from both branches are concatenated for processing, incurring minimal computational overhead with no retraining required.

### Key Designs

1. **Spatial Self-Swap**:

    - **Function**: Disrupts local structure and semantic coherence by swapping spatially positioned token pairs that are most semantically dissimilar.
    - **Mechanism**: Given token embeddings $\mathbf{X} \in \mathbb{R}^{B \times T \times D}$, all tokens are normalized and a cosine similarity matrix over spatial positions is computed. The $N$ token pairs with the lowest similarity (determined by swap ratio $r$) are selected, and a permutation map is constructed to swap these pairs in parallel. The operation is applied at the beginning of each Transformer block, prior to the residual connection.
    - **Design Motivation**: Swapping the most semantically dissimilar tokens constitutes an "adversarial" perturbation strategy—exchanging "sky" with "ground" disrupts image structure far more effectively than swapping two "sky" patches. Furthermore, the swap operation is closed (introduces no external noise), preventing unrecoverable distortions.

2. **Channel Self-Swap**:

    - **Function**: Perturbs global appearance attributes such as texture and material by swapping feature vectors along the channel dimension.
    - **Mechanism**: Implemented symmetrically with spatial swapping—similarity is computed along the channel dimension, and the most dissimilar channel embeddings are swapped. Channel swapping affects global appearance coherence (e.g., color, texture), complementing the structural perturbation of spatial swapping.
    - **Design Motivation**: Spatial swapping primarily affects structural and geometric relationships, while channel swapping perturbs subtle feature correlations. Together, they provide balanced perturbations covering both local detail and global photorealism.

3. **Adversarial Token Selection Strategy**:

    - **Function**: Maximizes perturbation effect while minimizing perturbation scope.
    - **Mechanism**: Motivated by adversarial analyses of vision Transformers and generative models, the most semantically dissimilar (rather than random) token pairs are selected for swapping. Experiments show: dissimilar pairs > random pairs > similar pairs; notably, even random swapping substantially outperforms SAG/SEG, demonstrating that token swapping itself is an intrinsically effective perturbation form.
    - **Design Motivation**: Achieve maximal structural disruption with minimal swaps—an information-theoretically efficient perturbation strategy.

### Loss & Training
SSG is a purely inference-time method requiring no training. It is implemented using PyTorch and the diffusers library with an Euler discrete scheduler over 50 steps. Validated on SD1.5 and SDXL as a fully plug-and-play module.

## Key Experimental Results

### Main Results — SDXL Unconditional Generation (MS-COCO 2014)

| Method | FID↓ | IS↑ | Precision↑ | Recall↑ | AES↑ |
|--------|------|-----|-----------|---------|------|
| No Guidance | 119.04 | 9.08 | 0.277 | 0.085 | 5.646 |
| SAG | 113.33 | 8.77 | 0.377 | 0.184 | 5.851 |
| SEG | 89.29 | 12.53 | 0.276 | 0.257 | 5.939 |
| PAG | 103.72 | 13.59 | 0.265 | 0.218 | 5.734 |
| **SSG** | **70.91** | **16.44** | **0.380** | **0.227** | **6.034** |

### SDXL Conditional Generation (MS-COCO 2014)

| Method | FID↓ | CLIP↑ | IS↑ | AES↑ | PickScore↑ | IR↑ |
|--------|------|-------|-----|------|-----------|-----|
| No Guidance | 45.09 | 0.281 | 21.31 | 5.671 | 20.20 | -0.847 |
| SAG | 34.14 | 0.295 | 22.95 | 5.745 | 20.64 | -0.487 |
| PAG | 26.55 | 0.306 | 29.70 | 5.820 | 21.56 | -0.003 |
| **SSG** | **21.73** | **0.313** | **34.63** | **5.902** | **22.17** | **0.276** |

### Ablation Study — Token Swapping Strategies

| Strategy | FID↓ | CLIP↑ | IR↑ |
|----------|------|-------|-----|
| SAG | 43.97 | 0.295 | -0.483 |
| PAG | 36.79 | 0.306 | 0.002 |
| Random Swap | 32.28 | 0.312 | 0.283 |
| Similar Token Swap | 28.74 | 0.309 | 0.110 |
| **Dissimilar Token Swap** | 31.41 | **0.313** | **0.297** |

| Spatial | Channel | FID↓ | IR↑ |
|---------|---------|------|-----|
| ✓ | ✗ | 31.96 | 0.272 |
| ✗ | ✓ | 31.30 | 0.286 |
| **✓** | **✓** | **31.41** | **0.297** |

### Key Findings
- **SSG yields the largest gains in unconditional generation**: FID decreases from 119.04 to 70.91 (a 40% reduction), substantially surpassing all competing methods.
- **Robustness to guidance scale**: SSG maintains strong performance over a wider range of $\omega$ values, without the rapid degradation at high guidance scales observed in SAG/PAG/SEG.
- **Random swapping is already competitive**: Even random token swapping substantially outperforms SAG/SEG, demonstrating an intrinsic advantage of the swapping operation itself.
- **SSG and CFG are complementary**: The two methods operate in orthogonal spaces (token space vs. conditioning space), and their combination yields further quality improvements.

## Highlights & Insights
- **"Swap, don't inject" perturbation philosophy**: Constructing perturbations by rearranging existing information rather than injecting external noise naturally preserves global energy conservation, yielding a more moderate and controllable degradation. The idea is both simple and elegant.
- **Information-theoretic interpretation of adversarial selection**: Swapping the most dissimilar tokens is equivalent to maximizing structural disruption while preserving total information content—producing the most effective guidance signal with the fewest operations.
- **Practical plug-and-play value**: No training, no architectural modifications, and compatibility with CFG enable direct deployment as a drop-in plugin. This extremely low deployment barrier is critical for real-world adoption.

## Limitations & Future Work
- The two-branch parallel inference still incurs approximately 2× computational overhead, with additional per-step similarity computation and swapping operations.
- The swap ratio $r$ and guidance scale $\omega$ require joint tuning; while robustness is improved over prior work, hyperparameter sensitivity remains.
- Validation is limited to SD1.5 and SDXL; applicability to newer architectures such as DiT and Flux has not been tested.
- Applicability to sequential generation settings such as video diffusion models remains unexplored.

## Related Work & Insights
- **vs. CFG**: CFG requires text conditioning and training-time dropout; SSG requires no conditioning, is applicable to unconditional generation, and is complementary to CFG.
- **vs. PAG**: PAG replaces attention maps with identity matrices, constituting a global perturbation; SSG selectively swaps tokens, offering finer granularity and greater controllability.
- **vs. SEG**: SEG injects Gaussian noise into self-attention; SSG replaces noise injection with swapping, introducing no external stochasticity.
- **vs. SAG**: SAG adds noise to input images and can introduce real noise artifacts at high guidance scales; SSG's swap operation is closed within the feature space, avoiding this issue.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The token-swapping idea is simple and novel; the adversarial selection strategy demonstrates genuine insight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two models, three datasets, conditional/unconditional settings, comprehensive ablations, and guidance mode visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logical structure with rich visualizations (the guidance mode analysis in Figure 2 is particularly informative).
- **Value**: ⭐⭐⭐⭐ High practical value as a plug-and-play module, though the contribution is incremental in nature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Guiding a Diffusion Transformer with the Internal Dynamics of Itself](guiding_a_diffusion_transformer_with_the_internal_dynamics_of_itself.md)
- [\[CVPR 2026\] Guiding Diffusion Models with Semantically Degraded Conditions](guiding_diffusion_models_with_semantically_degraded_conditions.md)
- [\[CVPR 2026\] Guiding Diffusion Models with Semantically Degraded Conditions (CDG)](cdg_condition_degradation_guidance_diffusion.md)
- [\[CVPR 2026\] APPLE: Attribute-Preserving Pseudo-Labeling for Diffusion-Based Face Swapping](apple_attribute-preserving_pseudo-labeling_for_diffusion-based_face_swapping.md)
- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)

</div>

<!-- RELATED:END -->
