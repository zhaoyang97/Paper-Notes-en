---
title: >-
  [Paper Note] Enhancing Virtual Try-On with Synthetic Pairs and Error-Aware Noise Scheduling
description: >-
  [CVPR 2025][Medical Imaging][Virtual Try-On] This paper proposes enhancing training data for virtual try-on by extracting synthetic garment-person pairs backward from human images. It designs an Error-Aware Refinement Schrödinger Bridge (EARSB) model to perform local error correction on the generation results of existing try-on models, achieving SOTA performance on VITON-HD and DressCode with a high user preference (59%).
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Virtual Try-On"
  - "Synthetic Data Augmentation"
  - "Schrödinger Bridge"
  - "Error-Aware Refinement"
  - "Diffusion Models"
date: 2026-05-08
content_hash: 95b06c835319d070
---

# Enhancing Virtual Try-On with Synthetic Pairs and Error-Aware Noise Scheduling

**Conference**: CVPR 2025  
**arXiv**: [2501.04666](https://arxiv.org/abs/2501.04666)  
**Code**: Yes (mentioned as available in the paper)  
**Area**: Medical Imaging  
**Keywords**: Virtual Try-On, Synthetic Data Augmentation, Schrödinger Bridge, Error-Aware Refinement, Diffusion Models

## TL;DR
This paper proposes enhancing training data for virtual try-on by extracting synthetic garment-person pairs backward from human images. It designs an Error-Aware Refinement Schrödinger Bridge (EARSB) model to perform local error correction on the generation results of existing try-on models, achieving SOTA performance on VITON-HD and DressCode with a high user preference (59%).

## Background & Motivation

1. **Background**: Virtual try-on aims to generate realistic images of a target person wearing target garments. Recently, the field has shifted from GAN-based methods to diffusion models, making significant progress.

2. **Limitations of Prior Work**: Current methods face two core challenges: (a) paired training data (person images + corresponding product-view garments) is limited, and copyright protections restrict large-scale data acquisition; (b) generated garment textures often exhibit local artifacts, such as distorted text and faded textures.

3. **Key Challenge**: There is a need for more diverse training data to cover the combinatorial space of body poses, skin tones, and garment attributes, but high-quality paired data is extremely expensive to acquire. Concurrently, local generation errors of base models are difficult to eliminate through end-to-end training.

4. **Goal**: (1) How to acquire more training pairs at a low cost? (2) How to targetedly repair local generation artifacts of base try-on models?

5. **Key Insight**: The authors observe that the backward human $\to$ garment task (extracting front-view garments from model try-on photos) is simpler than the forward task, which can be utilized to construct synthetic pairs. Inspired by classical boosting concepts, a cascaded refinement model is built specifically to repair errors of the preceding model.

6. **Core Idea**: Synthetic data is generated via a backward human-to-garment model for training augmentation, combined with a spatially adaptive Schrödinger Bridge guided by a weakly-supervised error classifier to refine generation results.

## Method

The entire method consists of two independent but complementary parts: synthetic data augmentation and the EARSB refinement model.

### Overall Architecture
The inputs are the masked human image, garment image, and pose representation. An initial result is first generated using a base try-on model, then a weakly-supervised error classifier locates the artifact regions, and finally, the Schrödinger Bridge with spatially adaptive noise scheduling performs targeted refinement on the erroneous regions. Synthetic data augmentation is used independently during the training phase to provide additional training pairs for any try-on model.

### Key Designs

1. **Human-to-Garment Synthetic Data Generation**:

    - **Function**: Generate (human, synthetic garment) pairs from a single-person try-on image.
    - **Mechanism**: A garment extraction model is trained. First, a segmentation model extracts the clothing region from the human image, and then a flow-based UNet generates its standard front view. After synthesizing the data, three filtering criteria are applied: clean background, front-view perspective, and low LPIPS reconstruction error. Ultimately, 12,730 upper-body and 8,939 full-body synthetic pairs were constructed from DeepFashion2 and UPT. During training, a real/synthetic flag conditioning strategy is employed, which outperforms the two-stage pre-training + fine-tuning scheme.
    - **Design Motivation**: Copyright protections make product-view garments difficult to obtain at scale, whereas single-person images are readily available. By leveraging the symmetry of the person-to-garment task, the cost of data acquisition is reduced from requiring copyrighted images to needing only single-person images.

2. **Weakly-Supervised Error Classifier (WSC)**:

    - **Function**: Locate local artifact regions in the generation results of the base try-on model, outputting an error confidence heatmap $M$.
    - **Mechanism**: A dual-encoder architecture encodes the initial generated image $x_1$ and garment $C$, respectively, predicting a sigmoid-activated error map via cross-attention. The training employs a joint image-level and patch-level loss: the image-level loss $\mathcal{L}_{img}$ uses max-pooling to ensure synthetic images have high maximum error scores and real images have low scores; the patch-level loss $\mathcal{L}_{pat}$ utilizes a small number of hand-labeled bounding boxes to maximize error scores inside the labeled regions and minimize them outside.
    - **Design Motivation**: Fully annotating all artifact regions is extremely expensive. Weakly-supervised training requires only a few hours of patch-level annotations to train an effective error locator and can be customized to the error patterns of specific base models.

3. **Error-Aware Refinement Schrödinger Bridge (EARSB)**:

    - **Function**: Utilize the error-map-guided spatially adaptive noise scheduling to targetedly refine artifact regions in the initial generated image.
    - **Mechanism**: A Schrödinger Bridge from the initial image $x_1$ to the ground-truth image $x_0$ is constructed based on the I2SB framework. The key innovation is replacing noise $\epsilon$ with $\epsilon^r = M \cdot \epsilon$, which spatially scales the noise using the error map $M$—correct regions receive almost no noise (pixels are copied directly), while erroneous regions receive more noise (allowing greater freedom for modification). The sampling process similarly multiplies the noise by $M$ at each step. The denoising network uses a cloth-flow-learning UNet to achieve more precise garment warping. To further guide the denoising process, WSC classifier guidance (analogous to classifier guidance) and expert denoisers (optionally fine-tuned for $t \in [0,0.5]$ and $t \in [0.5,1]$) are introduced.
    - **Design Motivation**: Naive I2SB needs to implicitly learn "what to fix and what to keep". Using the error map to explicitly control noise distribution focuses the refinement on artifact regions while preserving correct regions, which improves refinement performance and reduces sampling steps.

### Loss & Training
EARSB utilizes the MSE loss: 
$$\mathcal{L}_{EARSB} = \mathbb{E}_{t \sim U(0,1)} \|\epsilon_\theta^r(M,P,x_t,C;t) - \epsilon^r\|^2$$
where $\epsilon^r = M \cdot \epsilon$. WSC uses a joint image-level and patch-level loss. Training proceeds in three steps: first, generate initial images using the base model $\to$ generate error maps using WSC $\to$ train EARSB with adaptive noise. Post-training, EARSB is split into two expert denoisers responsible for different time ranges.

## Key Experimental Results

### Main Results

| Dataset | Setting | Metric | EARSB(SD)+H2G | StableVITON | IDM-VTON | Gain |
|--------|------|------|--------------|-------------|----------|------|
| VITON-HD | unpaired | FID↓ | **8.04** | 8.20 | 8.59 | 2.0% |
| VITON-HD | paired | SSIM↑ | **0.925** | 0.917 | 0.902 | 0.9% |
| VITON-HD | paired | LPIPS↓ | **0.053** | 0.057 | 0.061 | 7.0% |
| DressCode | unpaired | FID↓ | **10.41** | - | 11.09 | 6.1% |
| DressCode | paired | SSIM↑ | **0.968** | - | 0.956 | 1.3% |

### Ablation Study

| Configuration | User Preference Rate | Explanation |
|------|-----------|------|
| EARSB vs GP-VTON | 59.5% | Refining the GAN-based model yields significant improvements |
| EARSB vs StableVITON | 58.5% | Refining the diffusion-based model is also effective |
| +Synthetic Data (H2G-UH/FH) | FID further ↓ | Synthetic data consistently improves all metrics |
| Different Sampling Steps | Stable at 5 steps | Performance degradation of EARSB is minimal under few steps |

### Key Findings
- EARSB can be cascaded on top of different base models (GANs or diffusion), consistently bringing improvements and showing the versatility of the refinement framework.
- Synthetic data augmentation consistently improves performance across different base models and datasets.
- Spatially adaptive noise scheduling enables EARSB to maintain stable performance even at low numbers of sampling steps (5 or 10 steps), outperforming other diffusion methods that require 25+ steps.
- In user studies, the proposed method is preferred by most users in terms of both texture consistency and image fidelity.

## Highlights & Insights
- **Spatially Adaptive Schrödinger Bridge**: Integrating the error map into noise scheduling is an elegant design that enables "refinement on demand" during diffusion. This concept can be transferred to any image task that requires local editing.
- **Weakly-Supervised Error Detector**: Training an effective artifact locator takes only a few hours of annotation and can be customized to specific models, offering high practicality.
- **Symmetric Task Exploitation**: Using the backward human-to-garment task as a data augmentation method cleverly bypasses the issue of acquiring copyrighted data.

## Limitations & Future Work
- The quality of synthetic garments is constrained by the human-to-garment model; extraction of complex textures or multi-layered clothing may not perform well.
- WSC needs to be annotated and trained separately for each base model, and its cross-model generalizability remains to be verified.
- EARSB introduces extra inference time (requiring the base model to run before refinement), limiting real-time application.
- Only upper-wear try-on was evaluated; lower-wear, full-body, or multi-garment scenarios were not covered.

## Related Work & Insights
- **vs StableVITON/IDM-VTON**: These methods perform direct end-to-end generation, whereas this work adopts a "base + refinement" cascade strategy, similar to the boosting concept, which allows for cumulative improvements.
- **vs CAT-DM**: CAT-DM also uses a GAN initial image + small noise to initialize diffusion, but its noise scheduling is globally uniform. The spatially adaptive noise scheduling in this work is more fine-grained.
- **vs I2SB**: This work introduces spatially varying noise based on I2SB, evolving "global refinement" into "local refinement".

## Rating
- Novelty: ⭐⭐⭐⭐ Both spatially adaptive SB and synthetic data augmentation have novel aspects, but the overall framework is an integration of multiple innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets + user study + ablations + sampling efficiency analysis, quite comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, smooth workflow description.
- Value: ⭐⭐⭐⭐ The refinement framework has strong generalizability and is highly practical for virtual try-on applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)
- [\[CVPR 2025\] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation](sam_dpo_semi_supervised.md)
- [\[CVPR 2026\] MedLoc-R1: Performance-Aware Curriculum Reward Scheduling for GRPO-Based Medical Visual Grounding](../../CVPR2026/medical_imaging/medloc-r1_performance-aware_curriculum_reward_scheduling_for_grpo-based_medical_.md)
- [\[CVPR 2025\] UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC](unistainnet_foundation-model-guided_virtual_staining_of_he_to_ihc.md)
- [\[CVPR 2025\] SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation](sealion_semantic_part-aware_latent_point_diffusion_models_for_3d_generation.md)

</div>

<!-- RELATED:END -->
