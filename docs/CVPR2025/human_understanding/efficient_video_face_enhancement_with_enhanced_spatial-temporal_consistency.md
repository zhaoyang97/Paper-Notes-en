---
title: >-
  [Paper Note] Efficient Video Face Enhancement with Enhanced Spatial-Temporal Consistency
description: >-
  [CVPR 2025][Human Understanding][Blind Face Video Restoration] This paper proposes an efficient blind face video restoration framework based on 3D-VQGAN. By designing dual spatial-temporal codebooks to record high-quality portrait features and motion residual information, along with marginal prior regularization to alleviate codebook collapse, it achieves SOTA performance on BFVR and deflickering tasks while improving inference speed by 2 to 140 times.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Blind Face Video Restoration"
  - "Spatial-Temporal Codebook"
  - "3D-VQGAN"
  - "Deflickering"
  - "Video Enhancement"
date: 2026-05-08
content_hash: 5446c22bd7babcdc
---

# Efficient Video Face Enhancement with Enhanced Spatial-Temporal Consistency

**Conference**: CVPR 2025  
**arXiv**: [2411.16468](https://arxiv.org/abs/2411.16468)  
**Code**: [https://github.com/Dixin-Lab/BFVR-STC](https://github.com/Dixin-Lab/BFVR-STC)  
**Area**: Human Understanding / Video Understanding  
**Keywords**: Blind Face Video Restoration, Spatial-Temporal Codebook, 3D-VQGAN, Deflickering, Video Enhancement

## TL;DR

This paper proposes an efficient blind face video restoration framework based on 3D-VQGAN. By designing dual spatial-temporal codebooks to record high-quality portrait features and motion residual information, along with marginal prior regularization to alleviate codebook collapse, it achieves SOTA performance on BFVR and deflickering tasks while improving inference speed by 2 to 140 times.

## Background & Motivation

**Background**: Blind Face Restoration (BFR) aims to reconstruct high-quality results from degraded low-quality faces. Image-level methods (GFPGAN, CodeFormer, VQFR) have made significant progress, primarily based on geometric priors, generative priors (StyleGAN), and codebook priors (VQGAN). Video-level methods (PGTFormer, KEEP, StableBFVR) have further introduced temporal constraints to ensure inter-frame consistency.

**Limitations of Prior Work**: Existing BFVR methods suffer from two key limitations. First, inference efficiency is low—KEEP requires additional RealESRGAN pre-processing for backgrounds and face detection models, while StableBFVR relies on BasicVSR++ for preliminary restoration; such complex workflows lead to excessive inference times. Second, temporal consistency is insufficient—the receptive field of PGTFormer only spans two adjacent frames, failing to guarantee global consistency; the discrete nature of the codebook inherently causes inter-frame feature jumps, leading to flickering. Additionally, video flickering issues (including luminance flickering in real videos and pixel flickering in AI-generated videos) lack an efficient solution.

**Key Challenge**: High-quality video face restoration requires simultaneously ensuring frame-level restoration quality and inter-frame temporal consistency. However, the discrete representation of the codebook inherently conflicts with continuous temporal changes. Moreover, the complex video processing pipelines result in low efficiency.

**Goal**: (1) How to efficiently scale the VQGAN paradigm to the video domain to support video-level compression and quantization? (2) How to design a codebook mechanism to simultaneously record spatial portrait features and temporal motion information? (3) How to resolve the codebook collapse problem in multi-codebook scenarios?

**Key Insight**: The authors extend VQGAN from 2D (image domain) to 3D (video domain) and design independent spatial and temporal codebooks to capture different types of information. The spatial codebook records the static appearance of high-quality face features, while the temporal codebook records motion residuals between frames. Marginal prior regularization is used instead of traditional one-hot frequency statistics to resolve codebook collapse.

**Core Idea**: Utilize 3D-VQGAN for video-level compression combined with dual spatial-temporal codebooks to record appearance and motion information respectively, achieving efficient and temporally consistent video face enhancement.

## Method

### Overall Architecture

The framework consists of two training stages. **Stage I**: Train a 3D-VQGAN (3D encoder $E_h$ and decoder $D_h$) along with spatial and temporal codebooks ($\mathcal{C}_S$ and $\mathcal{C}_T$) using high-quality videos, learning the discrete representations of HQ faces in a self-supervised manner via reconstruction tasks. **Stage II**: Freeze the codebooks and decoder from Stage I, and train a low-quality video encoder $E_l$ along with two Transformer codebook lookup modules ($\mathcal{T}_S$ and $\mathcal{T}_T$) to predict the corresponding codebook index sequences from the LQ inputs. Finally, reconstruct high-quality videos using the HQ decoder.

### Key Designs

1. **3D-VQGAN and Robust Discriminator**:

    - **Function**: Achieves efficient video-level compression and quantization, supporting joint spatial-temporal downsampling.
    - **Mechanism**: The encoder and decoder use purely convolutional 3D architectures (residual blocks + up/downsampling blocks + convolutional self-attention) to achieve a spatial compression ratio of 8x and a temporal compression ratio of 2x. To address the instability and artifacts during video-level VQGAN training, a frozen pre-trained DINOv2 feature network is used as the discriminator backbone, paired with multiple trainable lightweight discriminator heads: $D_\phi(\hat{x}_h(\theta)) = -\mathbb{E}_{x_h}(\sum_k \mathcal{D}_{\phi,k}(\mathcal{F}(\hat{x}_h(\theta))))$
    - **Design Motivation**: Purely convolutional structures are more efficient than Transformers and support inputs of arbitrary resolutions. The pre-trained DINOv2 features provide more stable discriminative signals than training a discriminator from scratch.

2. **Spatial-Temporal Codebooks**:

    - **Function**: Respectively records spatial appearance features of high-quality portraits and inter-frame motion residual information.
    - **Mechanism**: Given the compressed representation $\bm{z}_h$ from the encoder, the spatial latent variable is directly set as $\bm{z}_{h,S} = \bm{z}_h$, while the temporal latent variable is computed via inter-frame temporal attention and motion residuals: $\bm{z}_{h,T} = \text{TA}(\bm{z}_h) + \text{Residual}(\bm{z}_h)$. The motion residual is defined as the difference between the latent variables of two frames separated by a temporal window. Both latent variables are quantized via nearest neighbor lookup in their respective codebooks, and then fused through element-wise addition: $\bm{z}_q = \bm{z}_{q,S} \oplus \bm{z}_{q,T}$
    - **Design Motivation**: Traditional codebooks only record spatial features and fail to capture motion information across frames, leading to poor temporal consistency in restoration results. By explicitly encoding motion information (residuals) into an independent codebook, the decoder can simultaneously exploit two types of information: "what this frame looks like" and "what changed in this frame compared to the previous frame".

3. **Marginal Prior Regularization**:

    - **Function**: Alleviates codebook collapse (where only a few codes are utilized) in multi-codebook scenarios.
    - **Mechanism**: The Euclidean distance matrix between latent variables and the codebook is calculated, converted into similarity scores, and normalized row-wise. By summing over columns, the marginal posterior distribution $P_{post}$ is obtained, regulated with KL divergence to approximate a uniform prior $P_{prior}$ : $\mathcal{L}_{KL}^S = \text{KL}(P_{post}, P_{prior})$. The key difference from traditional methods (which use retrieved one-hot indices to count usage frequency, penalizing unretrieved code words) is that this approach accumulates continuous similarity scores, allowing all code words to receive gradients.
    - **Design Motivation**: Multi-codebook setups (spatial + temporal) exacerbate the codebook collapse problem. Using similarity scores instead of one-hot encoding to estimate usage frequency allows more code words to participate in optimization, improving codebook utilization.

### Loss & Training

**Stage I Loss**: $\mathcal{L}_I = \mathcal{L}_1 + \mathcal{L}_{per} + \mathcal{L}_f + (\mathcal{L}_{KL}^S + \mathcal{L}_{KL}^T) + \lambda_{adv} \cdot \mathcal{L}_{adv}$, which includes $L_1$ reconstruction loss, VGG perceptual loss, codebook-encoder alignment loss, marginal prior regularization, and DINOv2 adversarial loss.

**Stage II Loss**: $\mathcal{L}_{II} = \mathcal{L}'_f + \lambda_{CE} \cdot (\mathcal{L}_{CE}^S + \mathcal{L}_{CE}^T)$, which includes codebook alignment loss and cross-entropy prediction loss for spatial/temporal codebook lookup.

Training is performed on 4 A100 GPUs, where Stage I is trained for 250K iterations (at $256^2$ resolution) and Stage II is trained for 50K iterations (at $512^2$ resolution).

## Key Experimental Results

### Main Results

Comparison of blind face video restoration on VFHQ-Test (24 frames / 1-second video):

| Method | Type | SSIM↑ | FVD↓ | Flow-Score↓ | Runtime(s)↓ |
|------|------|-------|------|-------------|-------------|
| GFPGAN | BFIR | 0.8207 | 246.9 | 1.316 | 14.44 |
| CodeFormer | BFIR | 0.8102 | 261.8 | 2.672 | 28.18 |
| BasicVSR++ | VSR | 0.8218 | 392.7 | 1.286 | 72.21 |
| PGTFormer | BFVR | 0.8426 | 107.6 | 1.154 | 7.085 |
| KEEP | BFVR | 0.8223 | 264.9 | 1.302 | 19.01 |
| **Ours** | BFVR | **0.8641** | **105.1** | **1.150** | **2.995** |

Luminance Deflickering:

| Method | Requires GT | FVD↓ | Runtime(s)↓ |
|------|------|------|-------------|
| DVP | Yes | 14.53 | 410.2 |
| FastBlend | Yes | 34.58 | 18.44 |
| **Ours** | **No**| **100.7** | **2.934** |

### Ablation Study

| Configuration | SSIM↑ | FVD↓ |
|------|-------|------|
| ViT-S + DINOv2 Discriminator | 0.9054 | 49.11 |
| ViT-B + DINOv2 Discriminator | 0.9050 | 49.08 |
| ViT-B + CLIP Discriminator | 0.8935 | 66.86 |

| Fusion Method | SSIM↑ | FVD↓ |
|---------|-------|------|
| Element-wise Addition | 0.9054 | 49.11 |
| Convolutional Fusion | 0.8799 | 97.67 |
| 3DFFT | 0.8741 | 118.7 |

### Key Findings
- Significant lead in inference efficiency: 2.995s vs PGTFormer (7.085s) vs KEEP (19.01s) vs BasicVSR++ (72.21s). The fastest BFVR method is accelerated by 2.4x.
- SSIM reaches 0.8641, significantly outperforming all baselines; FVD and Flow-Score also achieve optimal results, proving superior temporal consistency.
- DINOv2 as the discriminator feature network performs significantly better than CLIP (FVD 49.11 vs 66.86), and ViT-S performance is nearly identical to ViT-B, indicating that a larger feature network is not required.
- In spatial-temporal codebook fusion, simple element-wise addition significantly outperforms convolutional fusion and 3DFFT, suggesting that information from the two codebooks is complementarily additive.

## Highlights & Insights
- **The overall design of extending VQGAN from 2D to the 3D video domain** is incredibly complete—spanning from the 3D encoder-decoder, spatial-temporal codebooks, and marginal regularization to the two-stage training strategy. Each component is backed by clear design motivations and ablation validations.
- The improvement introduced by **marginal prior regularization**, though simple, is ingenious—replacing one-hot indices with continuous similarity scores to count codebook usage frequency. This allows code words not hit by nearest-neighbor search to still receive gradient signals. This trick can be transferred to any VQ approach requiring codebook optimization.
- The approach of using **DINOv2 as the discriminator backbone** is noteworthy. The pre-trained features offer more stable training signals, preventing the mode collapse issues often encountered when training discriminators from scratch. This "frozen feature network + lightweight head" discriminator paradigm has widespread applicability.

## Limitations & Future Work
- On the deflickering task, the FVD is inferior to DVP (100.7 vs 14.53). However, DVP requires flicker-free reference videos while the proposed method does not. Under a fair comparison, the advantage of the proposed method is clear, yet a performance gap still exists.
- The training data was rigorously filtered (down from 16,000 videos to 3,200), meaning generalization when deployed to highly diverse scenarios remains unverified.
- The video input is limited to 24 frames (1 second). Processing longer videos requires sliding windows or segmentation, which may introduce inconsistencies between segments.
- The model only focuses on the facial region and requires prior face detection and cropping. While cropping the training data simplifies the pipeline, videos with low face-to-background ratios still require additional pre-processing.

## Related Work & Insights
- **vs CodeFormer**: CodeFormer is a representative image-level VQGAN codebook method. This work extends it to the video domain and introduces a temporal codebook, comprehensively outperforming it on video restoration.
- **vs PGTFormer**: PGTFormer is a recent SOTA for BFVR, but its temporal receptive field is restricted to two frames. This work utilizes 3D convolutions and a motion residual codebook to capture a much larger temporal receptive field.
- **vs KEEP**: KEEP leverages restored frames to guide subsequent frames based on Kalman filtering, but requires RealESRGAN pre-processing, making the pipeline overly complex. The proposed end-to-end framework is much simpler and more efficient.

## Rating
- Novelty: ⭐⭐⭐⭐ The overall design of the 3D-VQGAN paired with the dual spatial-temporal codebook is highly innovative, and the marginal prior regularization is a simple yet effective improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ BFVR + two deflickering tasks + detailed ablation studies, with comprehensive evaluations across quality, consistency, and efficiency.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, complete framework description, and abundant figures and tables.
- Value: ⭐⭐⭐⭐ Simultaneous breakthroughs in both efficiency and effectiveness, with clear industrial application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] IDFace: Face Template Protection for Efficient and Secure Identification](../../ICCV2025/human_understanding/idface_face_template_protection_for_efficient_and_secure_identification.md)
- [\[ECCV 2024\] RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency](../../ECCV2024/human_understanding/repose_3d_human_pose_estimation_via_spatio-temporal_depth_relational_consistency.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [\[CVPR 2025\] ShowMak3r++: Compositional Entertainment Video Reconstruction](showmak3r_compositional_tv_show_reconstruction.md)
- [\[CVPR 2025\] Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models](two_is_better_than_one_efficient_ensemble_defense_for_robust_and_compact_models.md)

</div>

<!-- RELATED:END -->
