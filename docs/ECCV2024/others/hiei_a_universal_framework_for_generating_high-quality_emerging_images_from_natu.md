---
title: >-
  [Paper Note] HiEI: A Universal Framework for Generating High-quality Emerging Images from Natural Images
description: >-
  [ECCV 2024][Emerging Images] This paper proposes a universal framework, HiEI, which converts natural images into high-quality Emerging Images (EIs) through a human-centric color quantization module (TTNet), a perceptual difficulty control module (PDC), and a template vectorization module (TV). It outperforms existing methods in content and style quality while effectively resisting attacks from deep vision models, making it suitable for CAPTCHA mechanisms.
tags:
  - "ECCV 2024"
  - "Emerging Images"
  - "Color Quantization"
  - "Perceptual Difficulty Control"
  - "CAPTCHA"
  - "Visual Stylization"
date: 2026-05-08
content_hash: 6a3f01d882ddd7cf
---

# HiEI: A Universal Framework for Generating High-quality Emerging Images from Natural Images

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Others  
**Keywords**: Emerging Images, Color Quantization, Perceptual Difficulty Control, CAPTCHA, Visual Stylization

## TL;DR

This paper proposes a universal framework, HiEI, which converts natural images into high-quality Emerging Images (EIs) through a human-centric color quantization module (TTNet), a perceptual difficulty control module (PDC), and a template vectorization module (TV). It outperforms existing methods in content and style quality while effectively resisting attacks from deep vision models, making it suitable for CAPTCHA mechanisms.

## Background & Motivation

**Background**: Emerging Images (EIs) are a special type of stylized images consisting solely of black and white irregular splatters, which are widely applied in research on perceptual organization in cognitive psychology and CAPTCHA validation mechanisms. Traditional methods convert natural images into binary images through simple techniques like thresholding or Floyd-Steinberg dithering, but the results are often crude.

**Limitations of Prior Work**: Existing EI generation methods face two core challenges: (1) Color quantization issue—how to minimize perceptual loss when compressing the color space of a natural image from multi-bit depth to 1-bit? Traditional methods directly binarize, which loses a large amount of detail and renders the generated EI content unrecognizable; (2) Perceptual difficulty control issue—how to flexibly adjust the detection and recognition difficulty of the target object in the EI? Existing methods generate EIs that are either too easy to recognize or completely unrecognizable, lacking fine-grained difficulty control.

**Key Challenge**: The essence of an EI requires minimalist color representation (only black and white), but the human visual system requires sufficient structural information to organize meaningful visual content from discrete splatters. There is a fundamental trade-off between this information compression and perceptual recognizability.

**Goal**: To design a universal EI generation framework that achieves: (a) high-quality 1-bit color quantization to preserve crucial visual structures; (b) adjustable perceptual difficulty control; (c) vectorizable outputs to support various application scenarios.

**Key Insight**: The authors observe that human visual perception of EIs relies on Gestalt principles (such as proximity and continuity). Therefore, the color quantization strategy is designed from the perspective of human visual perception rather than simply optimizing pixel-level errors, while a task-driven difficulty control mechanism is introduced to precisely regulate the EI generation process.

**Core Idea**: Learn color quantization through a network centered on human visual perception, combined with a controllable perceptual difficulty adjustment module, to achieve universal transformation from arbitrary natural images to high-quality EIs.

## Method

### Overall Architecture

The HiEI framework consists of three core modules connected in series: taking a natural image as input, it first performs human-centric color quantization via TTNet, converting the image into a 1-bit binary representation while maximizing the preservation of visual structural information. Then, the PDC module adjusts the quantization results based on the target difficulty requirements, controlling the distribution and density of splatters to regulate human recognition difficulty. Finally, the TV module converts the rasterized results into a vectorized splatter template to generate the final high-quality EI.

### Key Designs

1. **TTNet (Human-Centric Color Quantization Module)**:

    - **Function**: Quantizes natural images from full-color space to 1-bit (black and white) while minimizing human visual perceptual loss.
    - **Mechanism**: Unlike traditional pixel-level thresholding methods, TTNet employs a deep network to learn a mapping from grayscale images to binary images. The network architecture is designed as a U-Net-like encoder-decoder to extract image structural features at multiple scales. The key lies in the design of the loss function—using SSIM-based structural similarity metrics and edge-aware loss, rather than simple L1/L2 pixel loss, because the quality of an EI depends on whether humans can organize complete object contours from splatters, rather than pixel-level reconstruction. Training data is annotated through human perception experiments to ensure the network output aligns with human visual preferences.
    - **Design Motivation**: Traditional binarization methods (such as Otsu, Floyd-Steinberg) are general signal processing methods that do not consider the unique requirements of EIs—namely, that humans need to "emerge" the target pattern from discrete splatters. TTNet is specifically designed for this characteristic, learning to retain crucial structural cues required for Gestalt organization.

2. **PDC (Perceptual Difficulty Control Module)**:

    - **Function**: Adjusts the visual complexity of splatters on the quantized binary image to control the recognition difficulty of the target object.
    - **Mechanism**: PDC regulates perceptual difficulty by controlling the density, distribution, and similarity of background splatters to foreground splatters. Specific operations include adding varying degrees of noise splatters in foreground regions to increase camouflage effects, and generating structural splatters similar to the foreground in background regions to confuse the visual system. The difficulty parameter $d \in [0, 1]$ controls the adjustment intensity, where $d=0$ denotes the easiest to recognize and $d=1$ denotes the highest difficulty. The module achieves progressive fusion of foreground-background splatters through a learnable blending network.
    - **Design Motivation**: CAPTCHA applications need to ensure that EIs are solvable by humans but unsolvable by machines, which requires precise difficulty control. EIs with fixed difficulty are either easily broken by CNNs or remain unrecognizable even to humans.

3. **TV (Template Vectorization Module)**:

    - **Function**: Converts rasterized binary splatter images into vectorized formats to generate high-quality final outputs.
    - **Mechanism**: The TV module uses contour tracing algorithms to extract the outline boundaries of each splatter, and then converts discrete pixel contours into smooth vector paths via Bézier curve fitting. Vectorized splatters maintain clean and sharp edges at any resolution without aliasing effects. It also supports post-processing adjustments of splatter shapes, such as controlling the roundness and sharpness of the splatters.
    - **Design Motivation**: Vectorized output makes EIs resolution-independent, suitable for various application scenarios such as printing or large screens. Additionally, the vectorization process itself serves as a denoising operation, eliminating tiny noise splatters introduced during the quantization phase.

### Loss & Training

The training of TTNet employs a combined loss function: $\mathcal{L} = \lambda_1 \mathcal{L}_{SSIM} + \lambda_2 \mathcal{L}_{edge} + \lambda_3 \mathcal{L}_{percept}$, where $\mathcal{L}_{SSIM}$ ensures structural similarity, $\mathcal{L}_{edge}$ emphasizes edge preservation, and $\mathcal{L}_{percept}$ is a perceptual loss based on a pre-trained VGG network to ensure the retention of high-level semantic information. The PDC module is optimized through an adversarial training strategy, introducing a discriminator to evaluate the "naturalness" and consistency of the difficulty levels of the generated EIs.

## Key Experimental Results

### Main Results

| Method | Content Quality (MOS↑) | Style Quality (MOS↑) | Difficulty Controllability | Machine Recognition Rate↓ |
|------|----------------|----------------|-----------|------------|
| Floyd-Steinberg | 2.31 | 2.18 | Uncontrollable | 78.5% |
| Mitra et al. | 3.12 | 3.05 | Limited | 65.2% |
| HiEI (Ours) | **4.26** | **4.13** | Continuously Controllable | **23.7%** |

HiEI significantly outperforms the baseline methods in human subjective evaluation (MOS score), with both content quality and style quality reaching over 4 points (on a 5-point scale). Meanwhile, in terms of machine recognition rate, the EIs generated by HiEI can effectively deceive deep vision models.

### Ablation Study

| Configuration | Content MOS↑ | Style MOS↑ | Description |
|------|---------|---------|------|
| Full HiEI | 4.26 | 4.13 | Full Model |
| w/o TTNet (using Otsu) | 2.89 | 3.15 | Significant decrease in color quantization quality |
| w/o PDC | 4.11 | 4.01 | Unable to control difficulty |
| w/o TV | 3.98 | 3.52 | Significant drop in style quality |
| w/o Perceptual Loss | 3.64 | 3.78 | Insufficient retention of high-level semantics |

### Key Findings

- TTNet makes the largest contribution; removing it drops the content quality from 4.26 to 2.89, proving that human-centric color quantization is the core of generating high-quality EIs.
- The PDC module provides difficulty control while maintaining basic quality; removing it causes a minor quality drop but loses crucial functionality.
- The TV module significantly improves style quality (+0.61); vectorization makes splatter outlines more orderly and aesthetically pleasing.
- Deep networks (ResNet-50, ViT) achieve only a 23.7% recognition rate on EIs generated by HiEI, far lower than the human rate of 89.3%, validating its feasibility as a CAPTCHA.

## Highlights & Insights

- **Human perception-driven quantization design** cleverly introduces the Gestalt theory from cognitive psychology into the color quantization task. It optimizes perceptual quality rather than pixel precision. This concept can be transferred to other vision tasks requiring extreme compression.
- **Continuously adjustable difficulty control** is the core innovation distinguishing this work from all previous methods, shifting EIs from a fixed output to a parameterized generation tool, which dramatically expands application scenarios.
- **Experimental validation of EIs as CAPTCHAs** offers great practical value—HiEI exploits the fundamental difference between the human visual system and deep networks in perceptual organization ability, a gap that is unlikely to be bridged in the near future.

## Limitations & Future Work

- The current approach may degrade when dealing with complex scenes containing fine textures (e.g., dense crowds, jungles), as the information bottleneck of 1-bit quantization is more severe in these scenarios.
- The evaluation of perceptual difficulty relies on human experimental data, and there is still room for improvement in designing automated evaluation metrics.
- Challenges for CAPTCHA applications: with the advancement of multimodal large models (e.g., GPT-4V), the adversarial robustness of EIs against AI requires continuous verification.
- Generation of colored EIs is not considered—it is limited to black and white binarization, whereas it could in principle be extended to a small number of colors (e.g., 3-4 colors).

## Related Work & Insights

- **vs Floyd-Steinberg Dithering**: A classic error diffusion method that is highly general but not designed for the perceptual characteristics of EIs. Its human-evaluated content quality is far lower than that of HiEI.
- **vs EI Method of Mitra et al.**: The representative method in the prior EI field, which is based on hand-crafted rules for splatter generation. It has limited difficulty control and unstable quality.
- **Insight**: Incorporating human perception models from cognitive science into the image processing pipeline is a direction worthy of attention, and may also find applications in fields like image compression and visual watermarking.

## Rating

- Novelty: ⭐⭐⭐⭐ Formulates the cognitive psychology concept of EIs into a computational framework, with clear interdisciplinary innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes both human experiments and machine adversarial experiments, and ablation studies cover all modules.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, though some module details could be more transparent.
- Value: ⭐⭐⭐ The application scenarios are relatively niche (EI generation and CAPTCHA), but the methodology holds transfer value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images (MV-AR)](../../ICCV2025/others/autoregressively_generating_multiview_consistent_images.md)
- [\[ICCV 2025\] Φ-GAN: Physics-Inspired GAN for Generating SAR Images Under Limited Data](../../ICCV2025/others/ph-gan_physics-inspired_gan_for_generating_sar_images_under_limited_data.md)
- [\[CVPR 2026\] A Debiased Reconstruction-based Framework for Training-Free Detection of AI-Generated Images](../../CVPR2026/others/a_debiased_reconstruction-based_framework_for_training-free_detection_of_ai-gene.md)
- [\[ICML 2026\] On Revisiting Entropy for Identifying Mislabeled Images](../../ICML2026/others/on_revisiting_entropy_for_identifying_mislabeled_images.md)
- [\[ECCV 2024\] CLR-GAN: Improving GANs Stability and Quality via Consistent Latent Representation and Reconstruction](clr-gan_improving_gans_stability_and_quality_via_consistent_latent_representatio.md)

</div>

<!-- RELATED:END -->
