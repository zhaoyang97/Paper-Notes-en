---
title: >-
  [Paper Note] GCC: Generative Color Constancy via Diffusing a Color Checker
description: >-
  [CVPR 2025][Image Generation][Color Constancy] GCC leverages the image priors of pre-trained diffusion models to estimate illuminant color by generating a color checker reflecting scene illumination via inpainting. Incorporating Laplacian decomposition to preserve structural details while adapting to illumination variations, it demonstrates superior generalization capabilities in cross-camera scenarios.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Color Constancy"
  - "Diffusion Models"
  - "Color Checker Generation"
  - "Illuminant Estimation"
  - "Cross-camera Generalization"
date: 2026-05-08
content_hash: 63af737331114570
---

# GCC: Generative Color Constancy via Diffusing a Color Checker

**Conference**: CVPR 2025  
**arXiv**: [2502.17435](https://arxiv.org/abs/2502.17435)  
**Code**: Yes (planned to be open-source, including training/inference code and pre-trained weights)  
**Area**: Diffusion Models  
**Keywords**: Color Constancy, Diffusion Models, Color Checker Generation, Illuminant Estimation, Cross-camera Generalization

## TL;DR
GCC leverages the image priors of pre-trained diffusion models to estimate illuminant color by generating a color checker reflecting scene illumination via inpainting. Incorporating Laplacian decomposition to preserve structural details while adapting to illumination variations, it demonstrates superior generalization capabilities in cross-camera scenarios.

## Background & Motivation

**Background**: Color constancy aims to estimate the illuminant color of a scene to eliminate color casts and achieve white balance. Traditional methods (e.g., Gray-World, White-Patch) rely on statistical priors, while deep learning methods (such as FC4) directly regress illuminant vectors from images via end-to-end training.

**Limitations of Prior Work**: Existing methods generalize poorly to cross-camera scenarios due to the varying spectral sensitivities of different camera sensors, which leads to significant differences in raw data captured under the same scene. Testing on a camera different from the training data results in a dramatic decline in performance, which is fundamentally a domain gap problem.

**Key Challenge**: Directly regressing the illuminant vector is a highly camera-specific task (different sensors have distinct color spaces), and the annotated training data is highly limited (e.g., the Gehler dataset contains only 568 images), making cross-domain generalization extremely difficult for models.

**Goal**: How to leverage the universal visual priors of large-scale pre-trained models to compensate for the data scarcity in the color constancy domain and achieve robust generalization across different cameras.

**Key Insight**: The Macbeth Color Checker serves as a "physical reference standard" for color constancy—if the color changes of the color checker under the current illumination are known, the illuminant can be directly computed. The critical observation is that pre-trained inpainting diffusion models already possess excellent priors for "placing objects in a scene," which can be guided to generate a color checker conforming to the illumination conditions of the current scene.

**Core Idea**: Generate a color checker image under the illumination conditions of the scene using a pre-trained inpainting diffusion model, and directly extract the illuminant estimation from the generated color checker colors.

## Method

### Overall Architecture
The pipeline of GCC: (1) Place a mask area of a neutral color checker in the input image; (2) Fill this area with a color checker reflecting the current scene illumination using a fine-tuned Stable Diffusion Inpainting model; (3) Extract color values from the generated color checker as the illuminant estimation. The entire process leverages the image priors of the diffusion model to "understand" the scene's illumination conditions and anchors the estimation using the known physical reference of the color checker structure.

### Key Designs

1. **Single-step Deterministic Inference**:

    - **Function**: Efficiently and reproducibly generate a color checker that reflects the scene illumination.
    - **Mechanism**: Fine-tune the Stable Diffusion 2 Inpainting model with LoRA. During inference, a DDIM scheduler is used with a fixed timestep $t=T$ and a trailing strategy to achieve single-step deterministic generation. Unlike standard diffusion models that require multi-step denoising, this fine-tuned model directly generates the color checker from noise in a single step.
    - **Design Motivation**: Multi-step sampling introduces randomness and is slow. Single-step deterministic inference guarantees speed (suitable for real-time white balancing) while eliminating the impact of randomness on illuminant estimation accuracy.

2. **Laplacian Decomposition Technique**:

    - **Function**: Separate high-frequency structural information and low-frequency color information in the VAE latent space.
    - **Mechanism**: Construct a 2-level Laplacian pyramid within the latent space of the VAE encoder output: apply Gaussian blur to each channel -> extract high-frequency components (original - blurred) -> downsample -> repeat. This preserves high-frequency structures (grid lines and patch shapes of the color checker) while allowing low-frequency colors to vary freely with scene illumination. Using $L=2$ pyramid levels yields the best results.
    - **Design Motivation**: Without decomposition, the diffusion model might distort the structure of the color checker (patches might disappear or deform), making color extraction impossible. Laplacian decomposition ensures structural clarity of the color checker while allowing the color information to fully reflect the scene illumination.

3. **Mask-based Data Augmentation**:

    - **Function**: Address training difficulties caused by imprecise annotations of the color checker.
    - **Mechanism**: Perform random RGB scaling (within the range of $[0.6, 1.4]$) on the input image in the RAW domain to simulate different illumination conditions. Apply local transformations to the masked area, adjusting brightness ($[0.8, 2.0]$), saturation ($[0.8, 1.4]$), and contrast ($[0.8, 1.4]$). Real illuminant ground truth is not required during training; only images containing color checkers are needed. Gamma correction ($\gamma=1/2.2$) from RAW to sRGB is applied to align with the training domain of the pre-trained VAE.
    - **Design Motivation**: Color checker annotations in public datasets can be imprecise (imprecise mask boundaries), and training data is limited. Color augmentation on the input and independent transformations within the masked area increase the diversity of training data while avoiding dependence on exact annotations.

### Loss & Training
Training employs the standard diffusion model loss (noise prediction MSE), optimized using the Adam optimizer with an initial learning rate of $5 \times 10^{-5}$, which exponentially decays after a 150-step warmup. The model is trained for 20,000 iterations on an NVIDIA A6000 GPU. During inference, inverse gamma correction is applied after VAE decoding to return to the linear domain for evaluation. The authors also evaluate an SDXL Inpainting + LoRA variant (using 25 denoising steps with noise strength = 0.6), taking the median of 10 sampling steps as the final estimation.

## Key Experimental Results

### Main Results
Evaluations are primarily conducted on the NUS-8 dataset (1,736 images, 8 cameras) and the Gehler dataset (568 images, 2 cameras). The evaluation metric is the angular error (in degrees), where lower values indicate better performance.

Cross-dataset evaluation (NUS-8 training -> Gehler testing, the most rigorous generalization scenario):

| Method | Mean↓ | Median↓ | Best-25%↓ | Worst-25%↓ |
|------|-------|---------|-----------|------------|
| Traditional Methods | ~4-6 | ~3-5 | ~1-2 | ~8-12 |
| GCC (L=2) | **2.35** | **2.02** | **0.78** | **4.57** |

### Ablation Study
Laplacian pyramid level analysis (NUS-8 -> Gehler):

| Pyramid Levels | Mean↓ | Median↓ | Best-25%↓ | Worst-25%↓ |
|-----------|-------|---------|-----------|------------|
| L = 1 | 3.53 | 3.27 | 1.48 | 6.03 |
| **L = 2** | **2.35** | **2.02** | **0.78** | **4.57** |
| L = 3 | 3.16 | 2.83 | 1.25 | 5.62 |

### Key Findings
- $L=2$ is the optimal pyramid level: $L=1$ provides insufficient high-frequency preservation (unclear color checker structure), while $L=3$ introduces too much low-frequency information that interferes with color generation.
- The method is particularly advantageous in cross-camera scenarios, showing that diffusion model priors can effectively compensate for sensor discrepancies.
- Failure cases primarily occur in scenes with multiple illuminants or spatially-varying lighting, where the diffusion model tends to generate visually plausible rather than physically accurate results.
- Sensitive to small datasets—when the training data is too limited, it leads to distortions in the generated color checker structure, resembling the overfitting effect of DreamBooth.
- The SDXL variant requires multi-step sampling (slower but robust), whereas the SD2 variant can perform single-step generation (fast and deterministic).

## Highlights & Insights
- **Novelty**: Shifting illuminant estimation from a "regression" paradigm to a "generative" paradigm, leveraging the powerful inpainting capabilities of the diffusion model to "see" scene lighting and "paint" a color checker. This concept of "generating a physical reference" can be extended to other vision tasks requiring calibration (e.g., depth estimation—generating reference objects of known sizes).
- **Elegance of Laplacian Decomposition**: Performing frequency decomposition in the latent space rather than the pixel space leverages the compressed representation of VAE to improve efficiency, while precisely balancing structural preservation and color degrees of freedom.
- **Training Without Illuminant Annotation**: Overturning the requirement of traditional color constancy methods for exact illuminant ground truth; training requires only images containing color checkers, greatly reducing data collection costs.

## Limitations & Future Work
- Poor performance under multi-light or spatially-varying illumination scenes: the diffusion model sometimes prioritizes "visual plausibility" over "physical accuracy".
- Sensitive to training data volume: small datasets cause distortion in the color checker structure, requiring smaller mask regions.
- Theoretically, the placement position of the color checker should not affect the results, but peripheral regions might be influenced by lens vignetting.
- While single-step inference is fast, it may sacrifice some accuracy. The multi-step SDXL variant can serve as a high-precision alternative.
- Lack of a comprehensive comparison with the latest deep learning-based color constancy methods (it primarily demonstrates cross-domain scenarios).

## Related Work & Insights
- **vs FC4**: Traditional deep learning regression methods generate poor cross-camera generalizations. GCC bypasses sensor-specific challenges through the universal priors of the diffusion model.
- **vs Statistical Methods** (Gray-world / White-patch): Statistical assumptions often fail in real-world scenarios. GCC utilizes data-driven priors for greater robustness.
- **A New Paradigm for Generative Vision Estimation**: Instead of directly predicting the target value, it generates a "physical reference" and measures from it. This concept holds potential for tasks like depth estimation, normal estimation, etc.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Using a diffusion model to generate a color checker for illuminant estimation is a completely fresh idea, representing a paradigm-level innovation.
- Experimental Thoroughness: ⭐⭐⭐ Cross-dataset experiments are convincing, but three-fold cross-validation and within-domain comparisons are insufficient, lacking a comprehensive comparison with state-of-the-art methods.
- Writing Quality: ⭐⭐⭐⭐ The ideas are clear and easy to follow, but since the currently available files only contain supplementary materials, the main paper's structure cannot be fully evaluated.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for color constancy. Cross-camera generalization is a core requirement for practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Color Alignment in Diffusion](color_alignment_in_diffusion.md)
- [\[CVPR 2025\] The Art of Deception: Color Visual Illusions and Diffusion Models](the_art_of_deception_color_visual_illusions_and_diffusion_models.md)
- [\[ECCV 2024\] ColorPeel: Color Prompt Learning with Diffusion Models via Color and Shape Disentanglement](../../ECCV2024/image_generation/colorpeel_color_prompt_learning_with_diffusion_models_via_color_and_shape_disent.md)
- [\[CVPR 2026\] Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity](../../CVPR2026/image_generation/too_vivid_to_be_real_benchmarking_and_calibrating_generative_color_fidelity.md)
- [\[ICCV 2025\] Video Color Grading via Look-Up Table Generation](../../ICCV2025/image_generation/video_color_grading_via_look-up_table_generation.md)

</div>

<!-- RELATED:END -->
