---
title: >-
  [Paper Note] PolarFree: Polarization-based Reflection-Free Imaging
description: >-
  [CVPR 2025][Image Restoration][Reflection Removal] A large-scale RGB-polarization dataset, PolaRGB, consisting of 6,500 pairs was constructed. A two-stage network, PolarFree, was proposed, which first employs a conditional diffusion model to generate a reflection-free prior and then utilizes a de-reflection backbone network to separate the transmission layer. This approach outperforms previous methods by approximately 2dB PSNR in polarization-guided reflection removal.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Reflection Removal"
  - "Polarization Imaging"
  - "Diffusion Models"
  - "Large-scale Dataset"
  - "Frequency Domain Loss"
date: 2026-05-08
content_hash: c9ea7a20d8f2537a
---

# PolarFree: Polarization-based Reflection-Free Imaging

**Conference**: CVPR 2025  
**arXiv**: [2503.18055](https://arxiv.org/abs/2503.18055)  
**Code**: [https://github.com/mdyao/PolarFree](https://github.com/mdyao/PolarFree)  
**Area**: Image Restoration  
**Keywords**: Reflection Removal, Polarization Imaging, Diffusion Models, Large-scale Dataset, Frequency Domain Loss

## TL;DR
A large-scale RGB-polarization dataset, PolaRGB, consisting of 6,500 pairs was constructed. A two-stage network, PolarFree, was proposed, which first employs a conditional diffusion model to generate a reflection-free prior and then utilizes a de-reflection backbone network to separate the transmission layer. This approach outperforms previous methods by approximately 2dB PSNR in polarization-guided reflection removal.

## Background & Motivation
1. **Background**: Reflection removal is a classic, challenging task in computer vision. Existing methods primarily rely on intensity cues such as brightness or gradients in RGB images. However, reflection removal is a highly ill-posed inverse problem (recovering two unknown layers from a single observation). Polarization imaging provides natural physical-level cues to distinguish reflected light from transmitted light.
2. **Limitations of Prior Work**: (a) Existing polarization reflection removal datasets are small-scale (<1,000 pairs), lack RGB information, or consist of synthetic data. (b) Extracting reflection-free information from polarization data is challenging due to the highly diverse shooting angles, scenes, and illumination conditions.
3. **Key Challenge**: Polarization provides strong physical cues, but there is a lack of high-quality, large-scale datasets to train models. Meanwhile, extracting reflection-free information from polarization data requires a powerful generative model to handle complex scene variations.
4. **Goal**: (a) Construct the first large-scale real-world RGB+polarization reflection removal dataset; (b) Design a reflection removal network that leverages diffusion models to fully exploit polarization cues.
5. **Key Insight**: Based on the physical properties of polarized light, reflected light and transmitted light exhibit different degrees of linear polarization (DoLP) and angles of linear polarization (AoLP). Specifically, reflected light is completely polarized at the Brewster angle.
6. **Core Idea**: Use a diffusion model to generate a "reflection-free prior" as an intermediate guidance signal, followed by a de-reflection network for precise separation.

## Method

### Overall Architecture
PolarFree involves a two-step inference pipeline: (1) **Prior Generation Step**: A conditional diffusion model $\mathcal{F}_{diff}$ receives polarization images and RGB images as the condition $M_{cond} = \{M_{polar}, M_{aolp}, M_{dolp}, M_{rgb}\}$, and progressively denoises from random noise to generate a reflection-free prior $\hat{z}_0$; (2) **Reflection Removal Step**: A de-reflection backbone network $\mathcal{F}_{remove}$ synthesizes the prior $\hat{z}_0$ and input conditions to output the clean transmission layer $\hat{T}_{rgb}$.

### Key Designs

1. **PolaRGB Large-Scale Dataset**

    - **Function**: Provides the first large-scale real-world RGB+polarization reflection removal dataset.
    - **Mechanism**: A division-of-focal-plane (DoFP) polarization camera is used with a video acquisition workflow to efficiently capture data. First, the reflection-free transmission image $T_{raw}$ is captured, and then a semi-reflective glass plate is introduced and continuously rotated to collect the mixed images $M_{raw}$. Affine transformations are applied in the RAW domain to achieve pixel-level registration, followed by polarization demosaicking to obtain polarization images of four angles (0°/45°/90°/135°) and unpolarized RGB. The dataset contains 6,500 pairs, which is 8 times the size of the previous work by Lei et al.
    - **Design Motivation**: Real-world polarization data reflects actual scene complexity much better than synthetic data, and large-scale data is fundamental to the success of data-driven methods.

2. **Conditional Diffusion Model for Reflection-Free Prior Generation**

    - **Function**: Extracts "reflection-free" intermediate representations from polarization and RGB inputs as guiding signals for reflection removal.
    - **Mechanism**: Trained in two stages. In the first stage, an encoder $\mathcal{E}$ is trained to extract the prior $z_0 = \mathcal{E}(M_{cond}, T_{rgb})$ from the polarization data of clean transmission images, while simultaneously training the de-reflection backbone network $\mathcal{F}_{remove}$. In the second stage, $z_0$ is used as the target supervision for the diffusion model, training the conditional diffusion model $\mathcal{F}_{diff}$ to generate the prior $\hat{z}_0$ from the polarization data of the blended image (containing reflections). During inference, only the diffusion model and the de-reflection network are required.
    - **Design Motivation**: Diffusion models lack a direct "reflection-free prior" training target; the two-stage training cleverly bridges this gap using an encoder.

3. **Frequency-Domain Phase Loss**

    - **Function**: Alleviates color aberration issues caused by semi-reflective surfaces.
    - **Mechanism**: Leveraging the property that the phase information of Fast Fourier Transform (FFT) primarily encodes shape and texture while being insensitive to color, the phase loss is defined as $\mathcal{L}_{phase} = \|\angle(FFT(\hat{T})) - \angle(FFT(T_{rgb}))\|_1$. Compared to spatial-domain L1/VGG losses, the phase loss is robust to color casts introduced by semi-reflective surfaces during acquisition.
    - **Design Motivation**: Color casts may exist between the transmission layer and the mixed image in training data (introduced by physical acquisition). Spatial-domain losses could mislead the model into learning color adjustment rather than reflection removal.

### Loss & Training
- Base Loss: $\mathcal{L}_1$ (pixel-level) + $\mathcal{L}_{VGG}$ (perceptual loss) + $\mathcal{L}_{TV}$ (total variation)
- Phase Loss: $\mathcal{L}_{phase} = \|\angle(FFT(\hat{T})) - \angle(FFT(T_{rgb}))\|_1$
- Diffusion Loss: Standard DDPM noise prediction loss $\mathcal{L}_{diff}$
- Two-stage Training: First, train the prior encoder and de-reflection network; second, train the diffusion model and fine-tune the de-reflection network.

## Key Experimental Results

### Main Results

| Method | PSNR↑ | SSIM↑ | Description |
|------|-------|-------|------|
| RRW (CVPR'24) | ~27 | ~0.85 | Without polarization, pure RGB |
| IBCLN | ~28 | ~0.87 | Without polarization |
| Lei et al. | ~29 | ~0.88 | Small dataset polarization |
| **PolarFree** | **~31** | **~0.91** | Outperforms previous methods by ~2dB |

### Ablation Study

| Configuration | PSNR↑ | Description |
|------|-------|------|
| Full PolarFree | ~31 | Full model |
| w/o Diffusion Prior | ~29 | Reflection removal directly from polarization, drops ~2dB |
| w/o Phase Loss | ~30.2 | Phase loss contributes about 0.8dB |
| w/o Polarization Input | ~28 | RGB-only, degrades to traditional methods |

### Key Findings
- Polarization information yields about 3dB PSNR improvement compared to RGB-only methods, validating the crucial value of polarization cues.
- The reflection-free prior generated by the diffusion model is more effective than direct regression, as the diffusion model can recover details obscured by reflection.
- Phase loss contributes significantly in scenes with severe chromatic aberration (e.g., highly reflective environments).
- In qualitative testing on real-world highly reflective scenes like museums and galleries, PolarFree surpasses comparison methods in retaining fine details.

## Highlights & Insights
- **Dataset Contribution** might hold more long-term value than the method itself—PolaRGB is the first large-scale real-world RGB+polarization reflection removal dataset, which will drive the entire field of polarization reflection removal forward.
- **Phase Loss** is a highly transferable technique—any restoration task with color aberration interference can consider adding constraints in the frequency-domain phase space.
- **Two-stage Training Strategy** cleverly addresses the lack of direct targets for diffusion models; the paradigm of "training an encoder to provide targets first, then training the generator" can be generalized to other intermediate representation learning tasks.

## Limitations & Future Work
- Polarization camera hardware is still not widely adopted, which limits the practical deployment of the method.
- The inference speed of the diffusion model is relatively slow, making it unsuitable for real-time applications.
- The dataset primarily covers glass reflection scenarios, while other types such as water surface reflections are underrepresented.
- Future directions could explore replacing DDPM with faster generative models (e.g., consistency models).

## Related Work & Insights
- **vs RRW**: Pure RGB reflection removal lacks physical cues; this work significantly improves upon it by leveraging polarization physics.
- **vs Lei et al.**: Also addresses polarization-based reflection removal, but its dataset has only 807 images and lacks RGB. This work provides 6,500 pairs with RGB, significantly expanding the scale and applicability.
- **vs ReflectNet**: Uses synthetic polarization data for training, which has inferior generalization compared to real-world data.

## Rating
- Novelty: ⭐⭐⭐⭐ The integration of diffusion models with polarization and the two-stage training strategy are novel, though the individual components build upon previous foundations.
- Experimental Thoroughness: ⭐⭐⭐⭐ The dataset construction is solid, ablation studies are comprehensive, and real-world scenes are tested.
- Writing Quality: ⭐⭐⭐⭐ Clear introduction to polarization physics, with a complete overall structure.
- Value: ⭐⭐⭐⭐⭐ Double contributions in both the dataset and the methodology, representative of a milestone in the polarization reflection removal field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Polarization State Tracing for Reflection Removal and Color-Consistent Reconstruction](../../CVPR2026/image_restoration/polarization_state_tracing_for_reflection_removal_and_color-consistent_reconstru.md)
- [\[CVPR 2025\] Reversible Decoupling Network for Single Image Reflection Removal](reversible_decoupling_network_for_single_image_reflection_removal.md)
- [\[NeurIPS 2025\] RGB-to-Polarization Estimation: A New Task and Benchmark Study](../../NeurIPS2025/image_restoration/rgb-to-polarization_estimation_a_new_task_and_benchmark_study.md)
- [\[CVPR 2025\] A Physics-Informed Blur Learning Framework for Imaging Systems](a_physics-informed_blur_learning_framework_for_imaging_systems.md)
- [\[CVPR 2025\] Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging](proximal_algorithm_unrolling_flexible_and_efficient_reconstruction_networks_for_.md)

</div>

<!-- RELATED:END -->
