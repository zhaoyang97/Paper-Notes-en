---
title: >-
  [Paper Note] Object-Aware NIR-to-Visible Translation
description: >-
  [ECCV 2024][Near-Infrared Image Translation] This paper proposes an object-aware near-infrared (NIR) to visible image translation framework. By decomposing visible images into object-independent illumination components and object-specific reflectance components for separate processing, combined with segmentation prior knowledge, high-quality NIR colorization is achieved under the condition of lacking large-scale paired data. In addition, the first fully aligned large-scale pa…
tags:
  - "ECCV 2024"
  - "Near-Infrared Image Translation"
  - "Image Decomposition"
  - "Semantic Segmentation Prior"
  - "Reflectance Estimation"
  - "Paired Dataset"
date: 2026-05-08
content_hash: 6b65ad5319ad3fa2
---

# Object-Aware NIR-to-Visible Translation

**Conference**: ECCV 2024  
**Code**: [https://github.com/Yiiclass/Sherry](https://github.com/Yiiclass/Sherry)  
**Area**: Others  
**Keywords**: Near-Infrared Image Translation, Image Decomposition, Semantic Segmentation Prior, Reflectance Estimation, Paired Dataset

## TL;DR
This paper proposes an object-aware near-infrared (NIR) to visible image translation framework. By decomposing visible images into object-independent illumination components and object-specific reflectance components for separate processing, combined with segmentation prior knowledge, high-quality NIR colorization is achieved under the condition of lacking large-scale paired data. In addition, the first fully aligned large-scale paired NIR-visible dataset is constructed.

## Background & Motivation

**Background**: Near-infrared (NIR) imaging is widely applied in scenarios such as assisted driving and security monitoring, because NIR can capture clear images under nighttime and low-light conditions. However, NIR images are monochromatic (grayscale) and lack color information, which limits their application in downstream tasks requiring color recognition. NIR-to-visible translation aims to convert monochromatic NIR images into natural, color visible-light images.

**Limitations of Prior Work**: Existing image translation methods (such as Pix2Pix, CycleGAN, etc.) face two key difficulties in NIR-to-visible tasks: (1) **Neglecting imaging differences between NIR and visible light**—the material reflectance in the NIR band is dramatically different from that in the visible band (e.g., vegetation is highly reflective/bright in NIR but dark green in visible light), making simple end-to-end translation difficult to learn this complex many-to-many mapping; (2) **Lack of high-quality paired training data**—due to viewpoint differences and time synchronization issues between NIR and visible-light cameras, existing datasets are either unaligned, small in scale, or limited in scene diversity.

**Key Challenge**: A single pixel value in an NIR image can correspond to multiple colors under visible light (as NIR reflectance does not distinguish visible colors), posing a severe one-to-many ambiguity problem. Furthermore, variations in illumination conditions further complicate this mapping—the visible appearance of the same scene varies completely under different lighting conditions, whereas its NIR appearance remains relatively stable.

**Goal**: (1) How to resolve the color ambiguity in NIR-to-visible mapping; (2) How to enable the model to generate reasonable translation results under various illumination conditions; (3) How to acquire high-quality, fully aligned paired data.

**Key Insight**: The authors draw inspiration from intrinsic image decomposition—a visible image can be decomposed into an illumination/luminance component and a reflectance component. The illumination component is independent of object material (depending on light source and geometry), while the reflectance component encodes the inherent color and material properties of objects. The illumination component of the NIR image has a strong correlation with that of the visible image, whereas the difference in reflectance is the core source of color ambiguity. Therefore, processing these two components separately can alleviate the difficulty of translation.

**Core Idea**: Decompose the visible-light image into illumination and reflectance components for separate translation, and leverage semantic segmentation priors to provide object-level guidance for reflectance estimation.

## Method

### Overall Architecture
The input is a grayscale NIR image, and the output is the corresponding RGB visible image. The entire pipeline is divided into three branches: (1) Illumination estimation branch—estimates the illumination component (lighting direction, intensity distribution) from the NIR image, which is independent of object categories; (2) Semantic-aware reflectance estimation branch—leverages a pre-trained semantic segmentation model to provide object category priors, predicting appropriate visible reflectance/colors for different category regions; (3) Fusion reconstruction branch—combines the estimated illumination and reflectance to generate the final colored visible image.

### Key Designs

1. **Illumination-Reflectance Decomposition Architecture**:

    - **Function**: Decouples the NIR-to-visible translation task into two simpler sub-tasks.
    - **Mechanism**: Inspired by Retinex theory, it assumes the visible image $I_{vis} = L \times R$, where $L$ is the illumination component and $R$ is the reflectance component. The illumination component $L$ mainly encodes the geometry and light source information of the scene and has good consistency between NIR and visible light (since the spatial distribution of illumination does not vary drastically across bands). Consequently, the illumination estimation network directly regresses the illumination map from the NIR image. The reflectance component $R$ encodes the color of materials, which is the core difficulty of translation—a grayscale NIR pixel needs to be mapped to an RGB reflectance. To this end, an independent reflectance estimation network is designed.
    - **Design Motivation**: Decomposing a complex many-to-many mapping into two relatively simple mappings reduces the learning difficulty. The estimation of the illumination component is relatively straightforward (low frequency, strong cross-band correlation), allowing the network to concentrate strictly on the truly difficult part: reflectance estimation.

2. **Semantic Segmentation-Guided Reflectance Estimation**:

    - **Function**: Resolves color ambiguity using object-level semantic information.
    - **Mechanism**: A pre-trained semantic segmentation model (such as SegFormer) is introduced to perform semantic segmentation on the NIR images, obtaining the object category of each pixel (sky, vegetation, building, road, vehicle, etc.). The segmentation features are fused into the reflectance estimation network through multi-scale feature injection. The key insight is that objects belonging to the same semantic category tend to share similar color distributions in visible light (e.g., sky is typically blue, vegetation is green, buildings are gray/brown). The semantic label acts as a strong prior constraint, significantly narrowing the search space for reflectance prediction.
    - **Design Motivation**: Inferring visible colors purely from NIR pixel values is an ill-posed problem, but incorporating semantic information like "this area is vegetation" makes the color prediction well-grounded. The segmentation prior provides the crucial constraint of "what color is reasonable."

3. **Fully Aligned NIR-Visible Dataset (FANVID)**:

    - **Function**: Provides high-quality, fully aligned paired NIR-visible training data.
    - **Mechanism**: The authors design a multi-sensor coaxial camera system where the NIR sensor and RGB sensor share the same optical axis. Through a beam splitter, they achieve simultaneous, co-viewpoint bimodal acquisition. This physically eliminates viewpoint differences and temporal misalignments. The dataset covers diverse scenes (urban, suburban, indoor, outdoor) and lighting conditions (daytime, twilight, nighttime), containing tens of thousands of precisely aligned NIR-RGB image pairs.
    - **Design Motivation**: The poor alignment quality of existing datasets has been a major bottleneck limiting research on NIR translation. Solving the data alignment issue from the hardware level via a coaxial design fundamentally establishes a reliable benchmark for training and evaluation.

### Loss & Training
The total loss function consists of multiple components: (1) pixel-level reconstruction loss $\mathcal{L}_{pixel} = \|I_{pred} - I_{gt}\|_1$ to supervise the final output; (2) perceptual loss $\mathcal{L}_{perceptual}$, using VGG feature matching to improve visual quality; (3) illumination consistency loss $\mathcal{L}_{illum}$, constraining the estimated illumination component to match the ground truth illumination; (4) reflectance smoothness loss $\mathcal{L}_{smooth}$ to encourage consistent reflectance within the same semantic region; and (5) an optional GAN loss $\mathcal{L}_{GAN}$ to enhance the photorealism of the generated images. During training, the illumination and reflectance branches are first warmed up before joint fine-tuning.

## Key Experimental Results

### Main Results

| Method | FANVID-PSNR↑ | FANVID-SSIM↑ | FANVID-FID↓ | EPFL-FID↓ |
|------|-------------|-------------|-------------|-----------|
| Ours (Object-Aware) | 25.8 | 0.872 | 38.2 | 42.5 |
| Pix2Pix | 22.1 | 0.801 | 68.4 | 78.3 |
| CycleGAN | 20.5 | 0.762 | 82.1 | 89.7 |
| TSIT | 23.4 | 0.835 | 51.6 | 58.2 |
| CUT | 21.8 | 0.811 | 62.3 | 71.0 |
| Palette | 24.2 | 0.851 | 45.7 | 52.8 |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | FID↓ | Explanation |
|------|-------|-------|------|------|
| Full model | 25.8 | 0.872 | 38.2 | Full model |
| w/o Decomposition | 23.6 | 0.838 | 49.5 | No illumination-reflectance decomposition, direct translation |
| w/o Semantic Segmentation | 24.1 | 0.849 | 44.8 | Remove segmentation prior |
| w/o Illumination Branch | 24.5 | 0.855 | 43.1 | Utilize reflectance estimation only |
| w/o Perceptual Loss | 25.0 | 0.862 | 41.6 | Remove VGG perceptual loss |
| Replace SegFormer with DeepLabV3 | 25.3 | 0.866 | 40.1 | Impact of segmentation model choice |

### Key Findings
- Illumination-reflectance decomposition is the most critical design; removing it causes a 2.2dB drop in PSNR, demonstrating that decomposition indeed simplifies the translation task.
- The semantic segmentation prior yields a 1.7dB improvement in PSNR, with particularly significant gains in areas with high color ambiguity, such as vegetation and sky.
- The choice of the segmentation model has a minor impact (only 0.5dB difference between SegFormer and DeepLabV3), indicating that the method is somewhat robust to the quality of segmentation.
- The proposed method shows a more pronounced advantage in nighttime scenes, where the value of illumination-reflectance decomposition is heightened under extreme lighting.
- The performance of all methods on the FANVID dataset is superior to that on existing unaligned datasets, validating the importance of high-quality paired data.

## Highlights & Insights
- **Physics-Inspired Task Decomposition**: Decoupling the translation task into illumination and reflectance based on Retinex theory offers an elegant simplification of the problem. The cleverness lies in the natural correlation of the illumination component between NIR and visible light, making illumination estimation easier and allowing focus to be directed to the challenging reflectance mapping.
- **Semantic Priors as Color Constraints**: Leveraging segmentation information to provide a prior of "what color is reasonable" narrows the ill-posed one-to-many mapping into a semi-deterministic one. This idea can be extended to other image translation tasks (e.g., grayscale image colorization, cross-domain style transfer).
- **Dataset Contribution via Coaxial Acquisition System**: Resolving the data alignment issue from the hardware level not only provides training data for the proposed method but also establishes a solid benchmark for the whole NIR translation community. Though the hardware investment is large, it provides a long-term solution.

## Limitations & Future Work
- **Quality Bottleneck of Semantic Segmentation**: Performing semantic segmentation on NIR images is inherently challenging (due to appearance differences between NIR and RGB), and segmentation errors will propagate to color prediction.
- **Constrained Color Diversity**: Objects under the same semantic category can possess a variety of colors in reality (e.g., vehicles can be red, blue, white, or black). The current approach tends to predict the "average" color for a category, lacking diversity.
- **Simplified Assumption of Retinex Decomposition**: Real-world images do not strictly conform to the $I = L \times R$ assumption (due to inter-reflections, translucent materials, etc.), meaning decomposition errors can affect the final output quality.
- **Generalizability**: The model may overfit to the scene distribution of the FANVID training set, and its performance when transferred to entirely different scenes (such as industrial or medical) remains unverified.
- **Future Directions**: Exploring diffusion models for conditional generation, which naturally support color diversity; introducing self-supervised or contrastive learning strategies to reduce reliance on paired data.

## Related Work & Insights
- **vs Pix2Pix**: Pix2Pix is a classic paired image translation method, but direct end-to-end translation ignores the physical differences between NIR and visible light. This paper explicitly models these differences using a decomposition architecture.
- **vs CycleGAN**: CycleGAN does not require paired data, but its generation quality is limited by the indirect supervision of cycle consistency. The proposed method benefits from strong supervision enabled by the high-quality paired data in FANVID.
- **vs Palette (Diffusion-based Method)**: Diffusion models possess stronger generative capabilities but are computationally heavy and lack physical interpretability. The proposed method is more lightweight with clear physical motivations.

## Rating
- Novelty: ⭐⭐⭐⭐ Retinex decomposition + semantic priors for NIR translation is quite novel, and the coaxial dataset provides a unique contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison across multiple datasets, comprehensive ablations, and rich qualitative results.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and fully explained physical assumptions.
- Value: ⭐⭐⭐⭐ The FANVID dataset is of high value to the community, and the method's concepts can be extended to other cross-modal translations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Customized Fusion: A Closed-Loop Dynamic Network for Adaptive Multi-Task-Aware Infrared-Visible Image Fusion](../../CVPR2026/others/customized_fusion_a_closed-loop_dynamic_network_for_adaptive_multi-task-aware_in.md)
- [\[ACL 2025\] Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages](../../ACL2025/others/using_source-side_confidence_estimation_for_reliable_translation_into_unfamiliar.md)
- [\[CVPR 2026\] MMVIP: A Visible-infrared Paired Dataset for Multi-weather Marine Vision](../../CVPR2026/others/mmvip_a_visible-infrared_paired_dataset_for_multi-weather_marine_vision.md)
- [\[ICCV 2025\] Jigsaw++: Imagining Complete Shape Priors for Object Reassembly](../../ICCV2025/others/jigsaw_imagining_complete_shape_priors_for_object_reassembly.md)
- [\[ICML 2025\] Symmetry-Aware GFlowNets](../../ICML2025/others/symmetry-aware_gflownets.md)

</div>

<!-- RELATED:END -->
