---
title: >-
  [Paper Note] MoE-DiffIR: Task-customized Diffusion Priors for Universal Compressed Image Restoration
description: >-
  [ECCV 2024][Image Restoration][Compressed Image Restoration] This paper proposes MoE-DiffIR, the first diffusion-based universal compressed image restoration (CIR) framework. It extracts task-customized diffusion priors from Stable Diffusion via a Mixture-of-Experts (MoE) Prompt module, leverages a Visual-to-Text adapter to activate SD's cross-modal generative priors, and constructs the first universal CIR benchmark dataset covering 21 degradation types (7 codecs × 3 compress…
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Compressed Image Restoration"
  - "Mixture of Experts"
  - "Prompt Learning"
  - "Stable Diffusion"
  - "Universal Image Restoration"
date: 2026-05-08
content_hash: a452fbcf5989dee7
---

# MoE-DiffIR: Task-customized Diffusion Priors for Universal Compressed Image Restoration

**Conference**: ECCV 2024  
**arXiv**: [2407.10833](https://arxiv.org/abs/2407.10833)  
**Code**: [Project Page](https://renyulin-f.github.io/MoE-DiffIR.github.io/)  
**Area**: Image Restoration / Compressed Image Restoration  
**Keywords**: Compressed Image Restoration, Mixture of Experts, Prompt Learning, Stable Diffusion, Universal Image Restoration

## TL;DR

This paper proposes MoE-DiffIR, the first diffusion-based universal compressed image restoration (CIR) framework. It extracts task-customized diffusion priors from Stable Diffusion via a Mixture-of-Experts (MoE) Prompt module, leverages a Visual-to-Text adapter to activate SD's cross-modal generative priors, and constructs the first universal CIR benchmark dataset covering 21 degradation types (7 codecs × 3 compression levels).

## Background & Motivation

**Diversity Challenges in Compressed Image Restoration**: In reality, a large variety of image codecs exist (JPEG, WebP, VVC, HEVC, learning-based codecs, etc.), each generating distinct compression artifacts at low bitrates—JPEG tends to cause blocking artifacts, learning-based codecs like $C_{PSNR}$ tend to cause blur, and WebP introduces color shifts. Most existing CIR methods are designed for a single codec (typically JPEG) and lack universality.

**Insufficiency of Texture Generation at Low Bitrates**: Traditional CNN/Transformer-based CIR methods (such as QGAC, FBCNN, HAT) optimize distortion metrics like PSNR/SSIM, but the restored images at extremely low bitrates are overly smooth, lacking texture details and perceptual quality.

**Underutilization of Diffusion Model Priors**: Existing SD-based restoration methods (StableSR, DiffBIR) reuse generative priors through ControlNet or feature adapters, but they use **shared modulation parameters** for all degradation types, failing to provide customized modulation schemes for different compression tasks. Furthermore, most methods set SD's text condition to an empty string, **wasting the rich text-to-image cross-modal priors**.

**Limitations of Existing Prompt Learning Methods**: A single prompt (e.g., ProRes) struggles to model the complex relationships of multiple tasks. Multi-prompt weighted fusion (e.g., PromptIR, DACLIP) easily suffers from the "mean feature" problem—where each prompt learns similar features, leading to a lack of diversity and reducing the modulation capability across different degradation types.

**MoE Inspiration**: The MoE framework selects and activates different subsets of experts for various inputs through a routing mechanism, which is highly suitable for multi-task scenarios. Introducing this concept into prompt learning allows each prompt to act as an expert sensing different degradation types, dynamically selecting the optimal prompt combination for each compression task via a router.

**Potential of Cross-modal Priors**: Trained on massive text-image data, SD stores rich text-to-image generative priors. Activating these cross-modal priors by transforming the visual information of low-quality images into text embeddings as SD's conditional inputs can help generate more consistent and realistic textures.

## Method

### Overall Architecture

MoE-DiffIR is built upon Stable Diffusion 2.1-base and adopts a two-stage fine-tuning strategy:

- **First Stage**: Freeze the VAE and UNet, and only train the MoE-Prompt module. Multiscale features are extracted from low-quality (LQ) images via the MoE-Prompt Module and modulated into the multiscale outputs of the SD UNet through SPADE layers. The network is trained with the standard diffusion loss $\mathcal{L}_{SD}$ for 0.4M iterations.
- **Second Stage**: Freeze all modules and only fine-tune the VAE Decoder Compensator. Generating 70,000 latent images using the first-stage weights, the compensator is trained with the LPIPS perceptual loss to correct structural fidelity for 0.1M iterations.

Input images are uniformly resized to $256 \times 256$, with a batch size of 32, and trained on 4 RTX 3090 GPUs.

### Key Designs

#### 1. MoE-Prompt Module (Mixture-of-Experts Prompt Generator)

Distinct from existing prompt designs, MoE-Prompt treats each prompt as a degradation expert, dynamically selecting Top-K prompt combinations through a router to harvest task-customized diffusion priors:

- **Degradation Prior Extraction**: A pre-trained DACLIP encoder is used to extract the Degradation Prior (DP) from the LQ image. The DP interacts with the input features via cross-attention before being fed into the router.
- **Noisy Top-K Routing**: The router adaptively selects prompt combinations using a noisy Top-K function:

$$G(x) = \text{Top-K}\big(\text{Softmax}(xW_g + \mathcal{N}(0,1) \cdot \text{Softplus}(xW_{\text{noise}}))\big)$$

where $W_g$ is the global feature weight matrix, and $W_{\text{noise}}$ introduces randomness to encourage prompt selection robustness and diversity. Once $K$ prompts are selected, they interact with the input features via matrix multiplication.

- **Comparison with Existing Methods**: A single prompt is insufficient to model multi-task scenarios; multi-prompt soft-weight fusion leads to "mean features" (where different prompts learn similar features). MoE-Prompt utilizes sparse routing, allowing each prompt to specialize in sensing different degradations, achieving efficient parameter reuse.

#### 2. Visual-to-Text Adapter (V2T Adapter)

To activate SD's rich text-to-image cross-modal priors (which are commonly ignored by existing methods), a V2T adapter is designed to project visual information into the text domain:

- **Quality Enhancement Pre-processing**: The LQ image first passes through several Transformer blocks acting as a quality enhancer to prevent severe compression artifacts from degrading the quality of subsequent visual features.
- **Visual Feature Extraction**: The enhanced image extracts visual embeddings through a CLIP image encoder.
- **Domain Translation**: Several MLP layers (i.e., the V2T Adapter) map the CLIP visual embeddings to the SD text space as text conditions to guide generation.
- **Difference from PASD**: PASD directly extracts text features from LQ images using BLIP, but at extremely low bitrates, severe image degradation leads to poor text feature quality. MoE-DiffIR enhances the image before encoding, followed by domain translation.

#### 3. Decoder Compensator

The pre-trained VAE decoder of SD is not fully aligned with the CIR task (as high compression rates lead to information loss during reconstruction). Therefore, low-quality information is introduced in the second stage to compensate and fine-tune the decoder:

$$L_{\text{Decoder}} = \mathcal{L}_{\text{lpips}}[z_{lq}, z_0, hr]$$

where $z_0$ represents the UNet denoising output, $z_{lq}$ is the latent variable of the low-quality image, and $hr$ is the high-quality reference image. The LPIPS perceptual loss ensures structural fidelity.

### Loss & Training

- **First Stage Loss**: Standard diffusion denoising loss $\mathcal{L}_{SD} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1)}[\|\epsilon - \epsilon(z_t, t)\|_2^2]$
- **Second Stage Loss**: LPIPS perceptual loss to guarantee structural fidelity.
- **Training Data**: DF2K dataset (3,450 images × 21 compression tasks = 72,450 training images).
- **Optimizer**: Adam ($\beta_1=0.9$, $\beta_2=0.999$).
- **Learning Rate**: $5 \times 10^{-5}$ (fixed) for the first stage; $1 \times 10^{-4}$ for the second stage.
- **Data Augmentation**: Random flip and rotation.
- **Final Hyperparameters**: $N=7$ base prompts, $K=3$ for Top-K.

## Key Experimental Results

### Main Results

Average performance comparison across 7 codecs (averaging 3 compression levels for each codec, evaluated on the LIVE1 dataset):

| Method | Type | JPEG LPIPS↓/FID↓ | VVC LPIPS↓/FID↓ | HEVC LPIPS↓/FID↓ | WebP LPIPS↓/FID↓ | $C_{PSNR}$ LPIPS↓/FID↓ | $C_{SSIM}$ LPIPS↓/FID↓ | HIFIC LPIPS↓/FID↓ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PromptIR | All-in-one | 0.213/111.2 | 0.305/168.4 | 0.314/172.6 | 0.250/149.4 | 0.234/141.4 | 0.328/160.1 | 0.121/82.8 |
| DiffBIR | Diffusion | 0.100/67.2 | 0.169/95.5 | 0.179/104.1 | 0.147/91.4 | 0.106/85.4 | 0.136/82.5 | 0.060/45.1 |
| StableSR | Diffusion | 0.107/68.2 | 0.168/98.5 | 0.185/106.0 | 0.121/75.1 | 0.108/73.9 | 0.154/90.7 | 0.071/46.7 |
| SUPIR | Diffusion | 0.125/71.0 | 0.147/99.7 | 0.161/108.7 | 0.142/95.6 | 0.107/90.1 | 0.136/87.0 | 0.074/48.4 |
| **MoE-DiffIR** | **MoE+Diff** | **0.096/62.7** | **0.144/88.8** | **0.162/98.7** | **0.110/70.6** | **0.100/72.2** | **0.135/80.2** | **0.059/43.6** |

MoE-DiffIR achieves the best perceptual metrics (LPIPS, FID) across all 7 codecs. Compared to SUPIR, LPIPS is reduced by an average of 10.9%, and FID is improved by an average of 5.4. Compared to StableSR, PSNR is improved by 0.41dB on average. The advantage on HIFIC is particularly notable (LPIPS=0.059, FID=43.6).

### Ablation Study

| Ablation Condition | LIVE1 PSNR/SSIM | LIVE1 LPIPS/FID | BSDS500 LPIPS/FID | DIV2K LPIPS/FID |
|:---|:---:|:---:|:---:|:---:|
| No Prompt | 28.73/0.806 | 0.134/85.9 | 0.159/96.5 | 0.130/79.5 |
| Single Prompt | 28.86/0.806 | 0.127/79.5 | 0.153/89.6 | 0.114/71.3 |
| Multiple Prompt (soft weights) | 28.98/0.810 | 0.121/77.1 | 0.148/89.3 | 0.112/71.7 |
| **MoE-Prompt (Ours)** | **29.02/0.811** | **0.118/75.9** | **0.143/88.1** | **0.107/68.9** |
| MoE-Prompt + V2T Adapter | 29.03/0.812 | 0.115/74.1 | 0.137/86.8 | — |
| MoE-Prompt + DP | 29.07/0.814 | 0.115/76.6 | 0.141/88.0 | — |
| **MoE-Prompt + V2T + DP** | **29.10/0.814** | **0.114/73.6** | **0.136/86.8** | — |

Ablation Insights: (1) MoE-Prompt improves LPIPS by 5% and drops FID by about 4 points compared to Multiple Prompt; (2) V2T Adapter mainly improves perceptual quality (LPIPS decreases by 3-5%); (3) Degradation Prior (DP) mainly boosts fidelity (PSNR +0.07dB).

### Key Findings

1. **First Universal CIR Benchmark**: Co-testing 4 traditional codecs (JPEG, VVC, HEVC, WebP) + 3 learning-based codecs ($C_{PSNR}$, $C_{SSIM}$, HIFIC), each with 3 compression levels, totaling 21 degradation types, establishing a standard evaluation pipeline for future works.
2. **MoE Routing Effectively Avoids "Mean Features"**: By applying Top-K sparse selection instead of soft-weight fusion across all prompts, different prompts specialize in different degradation patterns, enhancing parameter reuse efficiency and diversity.
3. **Value of Cross-modal Priors**: The V2T Adapter directs visual information to the text domain as an SD condition, successfully suppressing SD's behavior of mis-generating severe compression artifacts as noise under extremely low bitrates.
4. **Effect of Prompt Number and K-value**: $N=7$ yields the best and most parameter-efficient performance; $K=1$ leads to higher PSNR but poorer perceptual quality, while $K=3$ achieves the optimal balance between distortion and perception.
5. **Generalization to Unseen Degradations**: In both Cross-Degree (VVC unseen QP [32, 52]) and Cross-Type (unseen AVC codec) tests, MoE-DiffIR outperforms other prompt schemes, exhibiting strong generalization.
6. **Necessity of Two-stage Training**: Fine-tuning the Decoder Compensator during the second stage is crucial to improving structural fidelity—correcting the misalignment between the pre-trained VAE decoder and CIR latents.
7. **Balancing Distortion and Perceptual Metrics**: MoE-DiffIR not only leads comprehensively in LPIPS/FID but also remains competitive in PSNR (averaging 0.41dB higher than StableSR), balancing fidelity and perceptual quality.

## Highlights & Insights

- **Innovative Combination of MoE and Prompt**: Introducing MoE routing to prompt learning allows each prompt "expert" to dynamically specialize through data-driven training, avoiding the "mean feature" degradation of traditional multi-prompt methods, which stands as a broadly applicable technical contribution.
- **Mining Strategy for Cross-modal Priors**: Common SD restoration methods generally use empty text inputs. MoE-DiffIR bridges visual information into the text domain via a "quality enhancement → CLIP visual encoding → MLP domain translation" pipeline, fully exploiting SD's text-to-image generative capability.
- **Construction of the First Universal CIR Benchmark**: The benchmark dataset covering 21 degradation tasks fills a gap in the field, including multi-level compression of traditional and learning-based codecs, offering high pioneering value.
- **Visual Improvement at Extremely Low Bitrates**: Under extreme conditions (e.g., JPEG QF=5, VVC QP=47), MoE-DiffIR generates more accurate and consistent textures than DiffBIR or SUPIR, effectively avoiding incorrect texture generation.

## Limitations & Future Work

1. At extremely low bitrates, there remains a noticeable gap between the restored images and the ground-truth images, and the accuracy of generated textures requires further improvement.
2. Stable Diffusion-based inference is relatively slow due to the multi-step denoising process, limiting its applicability to real-time scenarios.
3. The training data relies solely on DF2K, which is limited in scale; training on larger and more diverse compression datasets could further improve performance.
4. The quality of the degradation prior provided by DACLIP depends on the coverage of its pre-trained model, and its generalization to extreme or rare degradation types remains to be validated.
5. Training at a resolution of $256 \times 256$ may be insufficient for high-resolution image restoration.
6. Unexplored integration with more powerful diffusion models, such as SDXL or SD 3.0.

## Related Work & Insights

- **StableSR / DiffBIR (2023)**: Pioneered the paradigm of using SD generative priors for image restoration. MoE-DiffIR builds on this by introducing task-adaptive modulation and cross-modal conditions.
- **PromptIR (NIPS 2023)**: The first work to introduce prompt learning into all-in-one image restoration using weighted prompt fusion. MoE-DiffIR improves prompt interaction through MoE routing.
- **DACLIP (2023)**: A CLIP variant trained on large-scale degradation data. MoE-DiffIR reuses its encoder to extract degradation priors.
- **MoE Framework (Sparsely-Gated MoE)**: The sparse routing and expert selection mechanism of MoE is creatively applied to a prompt learning scenario.
- **PASD (2023)**: Attempted to extract text with BLIP to guide SD restoration, but lacked pre-enhancement of LQ images, leading to limited performance in high-compression scenarios.

## Rating

| Dimension | Score (1-10) | Explanation |
|:---|:---:|:---|
| Novelty | 8 | The combination of MoE and prompts, the V2T adapter, and the first universal CIR benchmark dataset represent three distinct contributions. |
| Experimental Thoroughness | 9 | Evaluated across 7 codecs × 3 compression levels × 5 test sets with multiple metrics; compared with 8 SOTA methods, featuring exhaustive ablation studies. |
| Practicality | 7 | While perceptual quality significantly improves, the slow inference speed of SD limits practical real-time deployment. |
| Writing Quality | 7 | The framework diagram is clear, and experimental tables are detailed, though the typesetting of some formulas and symbols could be improved. |
| Overall Rating | 8 | A highly systematic work with clear problem definitions, a complete technical approach, and a benchmark dataset that is poised to have lasting impact. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Exploiting Diffusion Prior for Task-driven Image Restoration](../../ICCV2025/image_restoration/exploiting_diffusion_prior_for_task-driven_image_restoration.md)
- [\[ICLR 2026\] FreeAdapt: Unleashing Diffusion Priors for Ultra-High-Definition Image Restoration](../../ICLR2026/image_restoration/freeadapt_unleashing_diffusion_priors_for_ultra-high-definition_image_restoratio.md)
- [\[CVPR 2026\] UniLDiff: Unlocking the Power of Diffusion Priors for All-in-One Image Restoration](../../CVPR2026/image_restoration/unildiff_unlocking_the_power_of_diffusion_priors_for_all-in-one_image_restoratio.md)
- [\[ICLR 2026\] UniRestorer: Universal Image Restoration via Adaptively Estimating Image Degradation at Proper Granularity](../../ICLR2026/image_restoration/unirestorer_universal_image_restoration_via_adaptively_estimating_image_degradat.md)
- [\[ICCV 2025\] UniRes: Universal Image Restoration for Complex Degradations](../../ICCV2025/image_restoration/unires_universal_image_restoration_for_complex_degradations.md)

</div>

<!-- RELATED:END -->
