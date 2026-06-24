---
title: >-
  [Paper Note] Toward Tiny and High-quality Facial Makeup with Data Amplify Learning
description: >-
  [ECCV 2024][Image Generation][Facial Makeup Transfer] A Data Amplify Learning (DAL) paradigm is proposed, which leverages a Diffusion-based Data Amplifier (DDA) to "amplify" and generate a large volume of paired training data from only 5 annotated images. This data is used to train the TinyBeauty model with only 80K parameters, achieving SOTA makeup transfer performance at 460fps on an iPhone 13.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Facial Makeup Transfer"
  - "Data Amplify Learning"
  - "Diffusion Models"
  - "Lightweight Models"
  - "Mobile Deployment"
date: 2026-05-08
content_hash: 95080425d56aeab3
---

# Toward Tiny and High-quality Facial Makeup with Data Amplify Learning

**Conference**: ECCV 2024  
**arXiv**: [2403.15033](https://arxiv.org/abs/2403.15033)  
**Code**: [Yes](https://github.com/TinyBeauty)  
**Area**: Human Understanding  
**Keywords**: Facial Makeup Transfer, Data Amplify Learning, Diffusion Models, Lightweight Models, Mobile Deployment

## TL;DR

A Data Amplify Learning (DAL) paradigm is proposed, which leverages a Diffusion-based Data Amplifier (DDA) to "amplify" and generate a large volume of paired training data from only 5 annotated images. This data is used to train the TinyBeauty model with only 80K parameters, achieving SOTA makeup transfer performance at 460fps on an iPhone 13.

## Background & Motivation

Current mainstream makeup transfer methods suffer from fundamental flaws:

**Excessive model size**: Existing methods (e.g., EleGANt with 10M parameters, SCGAN with 35M parameters) far exceed the requirements for mobile deployment (<100K parameters), making real-time execution on mobile devices challenging.

**Reliance on complex pipelines**: Auxiliary modules such as face parsing and facial landmark detection are required, which increases latency and deployment difficulty.

**Instability of unpaired learning**: Annotating paired makeup data is extremely costly. Mainstream methods thus adopt an unpaired learning paradigm, which relies on adversarial training and imprecise supervisory signals (such as color histogram matching, Earth Mover's Distance, etc.), leading to poor robustness.

The root cause lies in the **inherent flaws of the learning framework itself**: Unpaired data presents severe facial alignment questions, forcing models to introduce facial spatial prompts and warping operations, which further increases complexity. Meanwhile, imprecise supervision methods rely on theoretical approximations, and smaller models are highly sensitive to such approximations.

**Key Insight**: If high-quality paired data can be acquired, the complex unpaired learning can be replaced by a simple L1 loss, significantly reducing optimization difficulty and model complexity.

## Method

### Overall Architecture

Data Amplify Learning (DAL) consists of two core components:

1. **Diffusion-based Data Amplifier (DDA)**: Based on a pre-trained diffusion model, it "amplifies" and generates 4,000 high-quality paired data samples from 5 seed images.
2. **TinyBeauty Model**: An ultra-lightweight U-Net with only 14 convolutional layers, trained on the amplified data using a pixel-level L1 loss.

### Key Designs

#### 1. Residual Diffusion Model (RDM)

This addresses the core challenge of facial details (wrinkles, textures) being smoothed out when generating portraits using diffusion models. It employs parallel dual-branch inference:

- **Conditional branch**: Inputting content condition $\mathbf{c}_{con}$ and style condition $\mathbf{c}_{sty}$ to generate a smoothed makeup image.
- **Unconditional branch**: Generating a smoothed non-makeup image without using any conditions.

Two key residuals are defined:
- **Detail residual** $R_d = x - \mathcal{F}_{fine}(x)$: The difference between the original image and the unconditional reconstruction, containing facial details like wrinkles.
- **Makeup residual** $R_m = \mathcal{F}_{fine}(x, c_{sty}+c_{con}) - \mathcal{F}_{fine}(x)$: The difference between the conditional and unconditional outputs, containing pure makeup changes.

Final synthesis:

$$y_{detail} = \mathcal{F}_{fine}(x) + \lambda_m R_m + \lambda_d R_d$$

$\lambda_m=1, \lambda_d=0.8$, adjusting makeup intensity and detail clarity by controlling the coefficients. Setting $\lambda_d$ to 1.0 would cause over-saturation due to the overlay of makeup-related details.

#### 2. Fine-Grained Makeup Module (FGMM)

Contains three sub-modules:

**Style Preservation Block (SPB)**:
- Since textual descriptions are insufficient to precisely capture subtle differences in makeup (such as the specific shade and texture of lipstick), visual exemplars are used as style references instead.
- The makeup is applied to a frontal face image, and the makeup region is isolated using a facial mask to create a pure makeup reference image.
- A pre-trained image encoder + a trainable MLP project the style image into style tokens.

**Identity Preservation Block (IPB)**:
- Learning makeup styles might inadvertently alter facial features, necessitating the decoupling of style and identity.
- While ArcFace was initially considered as an independent face encoder, its encoding space is incompatible.
- Ultimately, the encoding spaces of SPB and IPB are unified, sharing an MLP for feature fusion.
- The global condition vector is decoupled into an independent content condition $\mathbf{c}_{con}$ and style condition $\mathbf{c}_{sty}$.

**Mask Guidance**:
- The feature space is partitioned into three regions: $M_{face}$, $M_{lips}$, and $M_{eyes}$.
- During training, the loss is computed only within the $M_{changed} = M_{face} + M_{lips} + M_{eyes}$ region.
- During inference, latent space modifications are restricted to the masked region: $L'_y = L_y \odot M_{changed} + L_x \odot (1-M_{changed})$.
- This supports multi-makeup style combinations—different makeup conditions can be applied to different facial masks.

#### 3. TinyBeauty Model Architecture

Benefiting from the paired data generated by DDA, the model can be extremely streamlined:

- **Pure Convolutional U-Net**: Consists of only 4 convolutional layers + 4 residual blocks, totaling 14 convolutional operations.
- **Parameter size of only 81KB** (approximately 80K parameters).
- **Outputs residual instead of the full image**: $y' = M(x) + x$, generating only the makeup residual, which eliminates noise artifacts in the background and hair regions.
- **Resolution-agnostic**: The residual can be applied to images of any resolution without loss of texture.
- **No facial preprocessing required**: Does not rely on modules like face parsing or facial landmark detection.

### Loss & Training

**DDA Training**:
- Fine-tuned based on SD v1.5 + LoRA, learning rate 1e-4, trained for 500 epochs.
- OpenCLIP ViT-H/14 is used as the image encoder, with both style and identity token lengths set to 32.
- FaRL is used to generate 64×64 facial masks to guide latent space training.
- Five makeup styles are concurrently trained in a single model, taking about 50 minutes on a V100.
- Masked region loss: $L^M_{simple} = \mathbb{E}[\|(\epsilon - \epsilon_\theta) * M_{changed}\|^2]$

**TinyBeauty Training**:
- Trained on 4000 DDA-generated images for 50 epochs, learning rate 2e-4, using the Adam optimizer.
- **Reconstruction Loss**: Global L1 loss $\mathcal{L}_{rec} = \|y - y'\|_1$.
- **Eyeliner Loss**: Sobel edge operators are used to extract eyeliner contours, $\mathcal{L}_s = \|\mathcal{S}(y) - \mathcal{S}(y')\|^2_2 * M_{eyes}$.
- Perceptual loss and adversarial loss are added.
- Training completes in approximately 12 hours on a V100.

## Key Experimental Results

### Main Results

**Quantitative comparison on FFHQ and MT datasets (Style 1):**

| Method | FFHQ PSNR↑ | FFHQ FID↓ | FFHQ LPIPS↓ | MT PSNR↑ | MT FID↓ | MT LPIPS↓ |
|------|-----------|----------|------------|---------|--------|----------|
| BeautyGAN | 26.50 | 45.25 | 0.0564 | 27.49 | 25.05 | 0.0434|
| PSGAN | 25.65 | 36.22 | 0.0594 | 28.05 | 18.72 | 0.0301 |
| SCGAN | 27.55 | 36.98 | 0.0485 | 27.22 | 30.85 | 0.0467 |
| EleGANt | 30.18 | 25.47 | 0.0396 | 32.77 | 12.55 | 0.0191 |
| EleGANt* (DAL) | 35.45 | 10.78 | 0.0148 | 34.65 | 11.57 | 0.0164 |
| DDA | 35.96 | 10.28 | 0.0195 | 34.79 | 10.37 | 0.0231 |
| **TinyBeauty** | **35.39** | **8.03** | **0.0146** | **34.26** | **9.33** | **0.0181** |

TinyBeauty achieves a PSNR on FFHQ that is +5.21dB higher than EleGANt (a 17.3% improvement), and reduces the FID from 25.47 to 8.03 (a 68.5% reduction). Although the MT dataset was not involved in training, TinyBeauty still outperforms all other methods.

**Model efficiency comparison (iPhone 13):**

| Method | Params (M)↓ | FLOPs (G)↓ | Inference Time (ms)↓ |
|------|----------|----------|-------------|
| BeautyGAN | 8.04 | 24.70 | 27.89 |
| PSGAN | 8.41 | 91.28 | N/A |
| SCGAN | 35.33 | 288.51 | 195.61 |
| EleGANt | 10.27 | 127.94 | N/A |
| BeautyREC | 0.99 | 12.58 | 206.46 |
| **TinyBeauty** | **0.08** | **0.69** | **2.18** |

TinyBeauty has only 80K parameters and an inference speed of 2.18ms, which is **13x faster** than the fastest competitor and 6x faster than the face parsing preprocessing module.

### Ablation Study

**Ablation of DDA modules** (Visualized):
- Removing Mask Guidance + IPB $\rightarrow$ Loss of facial identity.
- Removing SPB $\rightarrow$ Inconsistent makeup style.
- Removing RDM $\rightarrow$ Smudged/smoothed facial texture.
- Full model combining all modules $\rightarrow$ High-quality makeup images.

**User Study Ranking (100 evaluators):**

| Method | Rank-1 | Rank-2 | Rank-3 |
|------|--------|--------|--------|
| BeautyGAN | 0.18% | 0.55% | 0.98% |
| EleGANt | 10.46% | 84.07% | 2.27% |
| BeautyREC | 0.28% | 1.95% | 55.06% |
| **TinyBeauty** | **86.56%** | 10.81% | 2.55% |

TinyBeauty obtains 86.56% of the Rank-1 votes, leading by a wide margin.

### Key Findings

- The DAL paradigm is generalizable: Retraining EleGANt with DAL improves its PSNR from 30.18 to 35.45, proving that data quality is more critical than model architecture.
- Eyeliner loss is crucial for learning high-frequency details: Without it, the network fails to capture eyeliner contours.
- A SOTA makeup model can be trained using only 5 seed images.
- The residual output design significantly reduces artifacts in background and hair regions.

## Highlights & Insights

1. **Paradigm Shift**: Transitioning from unpaired learning to data amplify learning completely revolutionizes the training paradigm for makeup transfer.
2. **Extreme Compression**: With 80K parameters and a 2.18ms inference time, it achieves a truly deployable mobile makeup model.
3. **RDM's Detail-Preserving Approach**: Separating facial details and makeup changes using dual-branch differences offers broad application potential for diffusion models.
4. **Few-Shot Data Amplification**: The concept of generating 4,000 training images from just 5 seed images can be transferred to other data-scarce domains.

## Limitations & Future Work

- The quality of data generated by DDA relies on the capability of the pre-trained diffusion model; extreme makeup styles may lack realism.
- Currently, only 5 makeup styles have been validated; this can be scaled to more diverse and complex styles.
- Mask Guidance relies on FaRL to generate facial masks, which may be inaccurate under extreme poses.
- There is still room for improvement under large poses/expressions.
- Temporal consistency for video makeup transfer has not been explored.

## Related Work & Insights

- **EleGANt (ECCV 2022)**: Simplifies optimization to L1 loss by generating pseudo ground truth, which is conceptually similar but still limited by unpaired data quality and massive model size.
- **IP-Adapter**: Controls diffusion model generation using image prompts. The SPB in DDA is inspired by this but customized and improved for makeup scenarios.
- **BeautyREC**: Achieves relatively lightweight deployment by abandoning the CycleGAN structure, but still requires facial preprocessing, and its 0.99M parameter size remains far larger than TinyBeauty's.
- **LoRA**: DDA uses LoRA to fine-tune Stable Diffusion, enabling the concurrent training of 5 styles in under 50 minutes.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The data amplify learning paradigm is highly innovative, and the RDM is cleverly designed.
- **Effectiveness**: ⭐⭐⭐⭐⭐ — PSNR is improved by 17.3%, the model is compressed by over 100x, and the user study shows an overwhelming victory.
- **Engineering Value**: ⭐⭐⭐⭐⭐ — With a 2.18ms inference time on iPhone and open-source code, it is highly practical.
- **Recommendation**: ⭐⭐⭐⭐⭐ — Combining innovative methodology, stunning results, and practical engineering feasibility, it is highly recommended.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Diffusion-Based Makeup Transfer with Facial Region-Aware Makeup Features](../../CVPR2026/image_generation/diffusion-based_makeup_transfer_with_facial_region-aware_makeup_features.md)
- [\[ECCV 2024\] A High-Quality Robust Diffusion Framework for Corrupted Dataset](a_highquality_robust_diffusion_framework_for_corrupted_datas.md)
- [\[ECCV 2024\] EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation](emdm_efficient_motion_diffusion_model_for_fast_and_high-quality_motion_generatio.md)
- [\[ECCV 2024\] Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning](diffusion-driven_data_replay_a_novel_approach_to_combat_forgetting_in_federated_.md)
- [\[ECCV 2024\] DreamDiffusion: High-Quality EEG-to-Image Generation with Temporal Masked Signal Modeling and CLIP Alignment](dreamdiffusion_high-quality_eeg-to-image_generation_with_temporal_masked_signal_.md)

</div>

<!-- RELATED:END -->
