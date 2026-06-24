---
title: >-
  [Paper Note] GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration
description: >-
  [CVPR 2025][Image Generation][Image degradation synthesis] This paper proposes GenDeg, a degradation synthesis framework based on Stable Diffusion, which can generate various controllable degradations (haze/rain/snow/motion blur/low-light/raindrops) on arbitrary clean images. By synthesizing over 550k images to construct the GenDS dataset, training All-In-One restoration models on it achieves significant performance improvements on out-of-domain test sets.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image degradation synthesis"
  - "diffusion models"
  - "All-In-One restoration"
  - "out-of-domain generalization"
  - "synthetic data"
date: 2026-05-08
content_hash: 78f36d2a3d0d718e
---

# GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration

**Conference**: CVPR 2025  
**arXiv**: [2411.17687](https://arxiv.org/abs/2411.17687)  
**Code**: [https://sudraj2002.github.io/gendegpage/](https://sudraj2002.github.io/gendegpage/) (Project Page)  
**Area**: Image Generation  
**Keywords**: Image degradation synthesis, diffusion models, All-In-One restoration, out-of-domain generalization, synthetic data

## TL;DR
This paper proposes GenDeg, a degradation synthesis framework based on Stable Diffusion, which can generate various controllable degradations (haze/rain/snow/motion blur/low-light/raindrops) on arbitrary clean images. By synthesizing over 550k images to construct the GenDS dataset, training All-In-One restoration models on it achieves significant performance improvements on out-of-domain test sets.

## Background & Motivation
1. **Background**: All-In-One Image Restoration (AIOR) handles multiple degradations with a single model. Representative methods include PromptIR, DA-CLIP, and Diff-Plugin.
2. **Limitations of Prior Work**: Existing AIOR models generalize poorly to out-of-distribution (OoD) degradation patterns and scenes. ① Existing datasets are small (far smaller than the 1.5M+ of SAM/Depth-Anything) and lack scene diversity; ② Synthetic datasets employ simple physical formulations (e.g., RESIDE uses only the atmospheric scattering model), resulting in monotonic degradation patterns; ③ Real-world degraded data (e.g., haze, low-light, raindrops) is difficult to collect, leading to scarce samples.
3. **Key Challenge**: AIOR methods overfit the training distribution, primarily because of the lack of degradation diversity and scene diversity in training data, whereas real-world data collection is impractical.
4. **Goal**: Use generative models to synthesize large-scale, diverse degradation data to improve the OoD generalization ability of AIOR.
5. **Key Insight**: Latent Diffusion Models have powerful generative priors and conditional control capabilities, which can generate realistic degradation patterns while preserving scene semantics.
6. **Core Idea**: Synthesize large-scale and diverse degradation data using a degradation severity-aware conditional diffusion model to resolve the generalization bottleneck of All-In-One restoration from the data perspective.

## Method

### Overall Architecture
GenDeg is based on the InstructPix2Pix architecture, taking a clean image $c_{img}$ + text prompt $c_{text}$ + degradation severity conditions as input, and outputting a degraded image. During training, paired degradation-clean data from multiple existing datasets are used. During inference, degradation is generated on new clean images. The generated images are corrected by a Structure Correction Module (SCM) and finally merged with the original datasets to form the GenDS dataset (750k+ samples), which is used to train restoration models.

### Key Designs

1. **Degradation Severity-Aware Conditional Diffusion Model**:
    - **Function**: To finely control the degradation intensity and spatial distribution during degradation generation.
    - **Mechanism**: Define a degradation map $c_{map} = |x_{in} - c_{img}|$, and calculate its mean $\mu$ and standard deviation $\sigma$ to quantify degradation severity. Encode $\mu$ and $\sigma$ respectively into 129-dimensional one-hot vectors (128 bins + 1 null), concatenate them, project to $\mathbb{R}^{77 \times 2}$, concatenate with the CLIP text embeddings $e_{text} \in \mathbb{R}^{77 \times 768}$, and project back to 768 dimensions to serve as conditional inputs for Stable Diffusion.
    - **Design Motivation**: Using only text prompts (such as "hazy") can cause the diffusion model to generate extreme degradations (overly dense haze or heavy rain). The $\mu$-$\sigma$ conditions enable the model to perceive the target degradation intensity, producing more realistic and controllable degradations.

2. **Structure Correction Module (SCM)**:
    - **Function**: To restore details lost during the VAE encoding-decoding process.
    - **Mechanism**: SCM is a lightweight network $S$ that takes the concatenation of the generated image and clean image as input and outputs a residual: $x_S = x_{gen} + S([x_{gen}, c_{img}])$. During training, a one-step reverse diffusion is used to obtain the generated image, and the loss function is a timestep-weighted L2 loss $L_S = \sqrt{\bar{\alpha}_{t-1}} \cdot \sqrt{1-\bar{\alpha}_t} \cdot \|x_{in} - x_S\|_2^2$, where the weight is lower at initial and final timesteps.
    - **Design Motivation**: LDM's VAE encoding and decoding lose details. SCM is effective only for smooth degradations such as haze, raindrops, and motion blur. For rain, snow, and low-light, clean images reconstructed via VAE ($\hat{c}_{img}$) are used instead.

3. **Data Generation and Quality Control**:
    - **Function**: Scale up the generation of high-quality paired degradation data.
    - **Mechanism**: Starting with approximately 120k clean images from the training dataset, 5 types of degradation not present in their original datasets are generated for each image. $\mu_{gen}$ is sampled from the target dataset's histogram, and $\sigma_{gen}$ is sampled from the histogram within the corresponding $\mu$ bin (to ensure statistical correlation). One out of twenty images uses a random $\sigma$ to increase diversity. After generation, poorly structured images are filtered based on the mean of the degradation map.
    - **Design Motivation**: Sampling from the dataset's histogram ensures the authentic distribution of degradation severity. Cross-degradation (generating degradation B on images with degradation category A) increases the combinational diversity of scene-degradations.

### Loss & Training
- GenDeg training utilizes the standard LDM denoising objective (Equation 1).
- A Swin Transformer restoration network is proposed: featuring an ImageNet-pretrained Swin encoder + lightweight convolutional decoder, hierarchical feature aggregation, and 3x3 convolutions to avoid patch boundary artifacts.
- Five restoration models are trained simultaneously: NAFNet, PromptIR, Swin, DA-CLIP, and Diff-Plugin.

## Key Experimental Results

### Main Results (OoD Performance, LPIPS/FID, lower is better)

| Method | REVIDE (Haze) | O-Haze (Haze) | GoPro (Blur) | LOLv1 (Low-light) | RainDS (Raindrop) |
|------|----------|----------|-----------|----------|-----------|
| PromptIR | 0.262/62.0 | 0.333/150.9 | 0.186/32.9 | 0.258/111.8 | 0.208/106.8 |
| **PromptIR+GenDS** | **0.212/56.0** | **0.160/89.0** | 0.191/31.9 | **0.178/87.9** | **0.182/79.8** |
| NAFNet | 0.211/71.3 | 0.183/99.2 | 0.155/28.2 | 0.167/78.8 | 0.178/73.4 |
| **NAFNet+GenDS** | **0.151/52.5** | **0.143/76.7** | 0.149/28.7 | **0.147/63.7** | **0.170/60.5** |

### Ablation Study

| Configuration | Description |
|------|------|
| Existing data only | Baseline performance, poor OoD generalization |
| +GenDS data | Significant improvements across all five models on OoD |
| NAFNet shows maximum gain | LPIPS on REVIDE haze improved from 0.211 to 0.151 (-28.4%) |
| In-domain performance mostly maintained | No obvious degradation in in-domain performance after adding GenDS |

### Key Findings
- All five restoration models (both non-generative and generative) show significant improvements in OoD performance after incorporating GenDS, demonstrating the universal value of the synthetic data.
- The dehazing task achieves the most notable boost (O-Haze PromptIR FID drops from 150.9 to 89.0) because real haze datasets are heavily scarce.
- t-SNE visualizations show that the degradation feature distribution generated by GenDS effectively bridges the domain gap between existing training datasets and OoD test data.
- GenDS is the first dataset providing multiple degraded versions for the same clean image, naturally tailored for AIOR training.
- Controlling with $\mu$-$\sigma$ conditions is critical for realistic degradation; unconditional generation tends to yield extreme degradations.

## Highlights & Insights
- **"Solving generalization from the data side"** is a highly valuable paradigm: without changing the model architecture, but merely modifying the training data, all five different models achieved performance gains, proving that data quality outweighs model complexity.
- **$\mu$-$\sigma$ conditioning of degradation severity** is a key innovation: encoding degradation severity into conditional signals fused with CLIP embeddings balances both global intensity and spatial distribution control.
- **Cross-degradation dataset training** allows GenDeg to break free from relying on a single physical model, conceptually learning the capability to generate diverse degradation patterns.
- This methodology can be directly transferred to other low-level vision tasks requiring large-scale paired training data, such as super-resolution and denoising.

## Limitations & Future Work
- Currently, only 6 degradation types are covered, leaving out common degradations such as JPEG compression, noise, and overexposure.
- SCM is not applicable to rain, snow, and low-light degradations, requiring specialized handling, which lacks unified consistency.
- The quality of generated images still depends on the reconstruction quality of the VAE, meaning high-frequency detail loss cannot be completely avoided.
- Despite being larger than existing datasets, the size of GenDS (750k) is still far from the SAM scale (11 million); further scaling may yield even larger improvements.
- Composite degradation effects (such as haze + rain) are not yet considered.

## Related Work & Insights
- **vs InstructPix2Pix**: GenDeg extends it by incorporating degradation severity conditions ($\mu$/$\sigma$), steering image editing toward degradation synthesis.
- **vs DA-CLIP**: DA-CLIP guides restoration using degradation features from CLIP, whereas GenDeg enhances the data side. The two are complementary—applying DA-CLIP+GenDS also yielded significant gains in experiments.
- **vs PromptIR**: PromptIR utilizes learnable prompts to identify degradation types. The GenDS dataset directly boosts PromptIR's OoD performance.
- This paradigm of "using generative models to synthesize training data" is similar to data augmentation concepts in classification, but with stricter requirements for preserving fine details.

## Rating
- Novelty: ⭐⭐⭐⭐ The first systematic work using diffusion models to synthesize degradation data to enhance the generalization of restoration models, featuring an innovative $\mu$-$\sigma$ conditioning mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, involving five different models, six degradations, multiple OoD/in-domain datasets, and t-SNE visualizations.
- Writing Quality: ⭐⭐⭐⭐ Highly logical with detailed data analysis and rich visualizations.
- Value: ⭐⭐⭐⭐⭐ Provides a ready-to-use 750k-sample dataset and degradation synthesis tools, directly pushing the community forward.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bridging Degradation Discrimination and Generation for Universal Image Restoration](../../ICLR2026/image_generation/bridging_degradation_discrimination_and_generation_for_universal_image_restorati.md)
- [\[CVPR 2025\] OSDFace: One-Step Diffusion Model for Face Restoration](osdface_one-step_diffusion_model_for_face_restoration.md)
- [\[ICCV 2025\] MMAIF: Multi-task and Multi-degradation All-in-One for Image Fusion with Language Guidance](../../ICCV2025/image_generation/mmaif_multi-task_and_multi-degradation_all-in-one_for_image_fusion_with_language.md)
- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [\[CVPR 2026\] Face2Scene: Using Facial Degradation as an Oracle for Diffusion-Based Scene Restoration](../../CVPR2026/image_generation/face2scene_using_facial_degradation_as_an_oracle_for_diffusion-based_scene_resto.md)

</div>

<!-- RELATED:END -->
