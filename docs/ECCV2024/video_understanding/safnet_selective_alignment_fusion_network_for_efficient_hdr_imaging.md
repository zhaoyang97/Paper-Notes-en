---
title: >-
  [Paper Note] SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging
description: >-
  [ECCV 2024][Video Understanding][HDR Imaging] SAFNet proposes a selective alignment fusion strategy that jointly refines raw valuable region masks and cross-exposure optical flows through a pyramid decoder. It explicitly fuses HDR images only after performing precise alignment in valuable regions, outperforming the state-of-the-art on Kalantari 17 and a self-built Challenge123 dataset while achieving an order of magnitude faster inference speed.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "HDR Imaging"
  - "Selective Alignment"
  - "Optical Flow Estimation"
  - "Multi-Exposure Fusion"
  - "Efficient Network"
date: 2026-05-08
content_hash: 6ad0d6a185b342bd
---

# SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging

**Conference**: ECCV 2024  
**arXiv**: [2407.16308](https://arxiv.org/abs/2407.16308)  
**Code**: [https://github.com/ltkong218/SAFNet](https://github.com/ltkong218/SAFNet)  
**Area**: Video Understanding  
**Keywords**: HDR Imaging, Selective Alignment, Optical Flow Estimation, Multi-Exposure Fusion, Efficient Network

## TL;DR

SAFNet proposes a selective alignment fusion strategy that jointly refines raw valuable region masks and cross-exposure optical flows through a pyramid decoder. It explicitly fuses HDR images only after performing precise alignment in valuable regions, outperforming the state-of-the-art on Kalantari 17 and a self-built Challenge123 dataset while achieving an order of magnitude faster inference speed.

## Background & Motivation

Multi-exposure HDR imaging requires synthesizing LDR images with different exposures into an HDR image. The core challenges lie in motion mismatch in dynamic scenes and texture truncation in saturated regions. Existing deep learning methods are categorized into two main paradigms:

**Alignment + Fusion**: First estimate cross-exposure optical flows for alignment, then generate the HDR image (e.g., Kalantari et al.). Problem: Optical flow estimation is highly error-prone in severely saturated and occluded regions.

**Attention-based Implicit Fusion**: Bypass explicit alignment and implement spatial/channel-level feature interactions using various attention mechanisms (e.g., AHDRNet, HDR-Transformer, SCTNet). Problem: High computational complexity and large inference latency make deployment on resource-constrained devices difficult.

While the performance of both paradigms has progressively improved, their computational costs have also escalated. The core observation of this paper is that **not all regions in non-reference frames are worth precise alignment**. For instance, overexposed/underexposed regions or regions already possessing rich textures in the reference frame can be directly discarded. Conversely, for regions in non-reference frames that contain valuable textures missing in the reference frame, motion estimation in these textured regions is much easier than in saturated regions. Based on this observation, SAFNet aligns only valuable regions, skipping the challenging and useless motion estimation in saturated areas.

## Method

### Overall Architecture

SAFNet consists of three subnetworks: a pyramid encoder E, a coarse-to-fine decoder D, and a detail refinement module R. The pipeline is as follows: (1) The encoder extracts pyramid features of each input frame; (2) The decoder jointly refines the selective probability mask M and the cross-exposure optical flow F; (3) An initial HDR image Hm is explicitly synthesized using optical flow alignment and mask-reweighted fusion coefficients; (4) The refinement network generates the final HDR image Hr based on the optical flow, mask, Hm, and LDR inputs.

### Key Designs

1. **Selective Flow Estimation**: During the coarse-to-fine flow refinement process, the decoder simultaneously outputs a selective probability mask M (sigmoid output, range 0-1) to identify valuable regions. The mask and optical flow mutually reinforce each other: M guides the decoder on which regions to focus for F estimation, while better F can aggregate valuable features to facilitate further region identification and residual optical flow estimation. Formula:
   $[F_{2\to1}^{k-1}, F_{2\to3}^{k-1}, M_1^{k-1}, M_3^{k-1}] = \mathcal{D}^k([F_{2\to1}^k, F_{2\to3}^k, M_1^k, M_3^k, \tilde{\phi}_1^k, \phi_2^k, \tilde{\phi}_3^k])$
   Design Motivation: Skip motion estimation in saturated regions (which are inherently error-prone) and concentrate the model's learning capacity on more meaningful areas.

2. **Explicit HDR Fusion**: The selective mask is used to reweight the fusion coefficients, followed by explicit HDR synthesis. Key formulas:
   $W_1 = \Lambda_1 \odot M_1, \quad W_3 = \Lambda_3 \odot M_3$
   $W_2 = \Lambda_2 + \Lambda_1 \odot (1-M_1) + \Lambda_3 \odot (1-M_3)$
   $H_m = W_1 \odot \tilde{H}_1 + W_2 \odot H_2 + W_3 \odot \tilde{H}_3$
   The fusion weights of unselected regions are transferred to the reference frame to ensure normalization. Design Motivation: Explicit fusion is far more efficient than implicit attention, and the mask naturally suppresses ghosting artifacts in mismatched regions.

3. **Refine Module + Window Partition Cropping**: The refinement network R is a fully convolutional network that enhances high-frequency details at the original resolution. It utilizes the first-stage optical flow, mask, and Hm as extra inputs (ablation shows Hm contributes the most). During training, window partition cropping is proposed: the first stage handles long-range texture aggregation on large 512×512 patches, while the second stage refines local details on small 128×128 patches, unifying the two crop sizes via window partition/reverse operations.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_r + \beta \mathcal{L}_m$ ($\beta=0.1$)
- Refinement loss: $\mathcal{L}_r = \mathcal{L}_1(T(H_r), T(H_{gt})) + \alpha \mathcal{L}_p(T(H_r), T(H_{gt}))$ ($\mu$-law tonemapping + L1 + perceptual loss, $\alpha=0.01$)
- Fusion loss: $\mathcal{L}_m = \mathcal{L}_1(T(H_m), T(H_{gt})) + \mathcal{L}_c(T(H_m), T(H_{gt}))$ (L1 + census loss, supervising the first-stage alignment and fusion)
- The decoder utilizes group conv (group=3) + channel shuffle to improve efficiency
- Optical flow and mask are predicted at 1/2 resolution and then upsampled

## Key Experimental Results

### Main Results

| Dataset | Metric | SAFNet | Prev. SOTA | Speed Comparison |
|--------|------|------|----------|------|
| Kalantari 17 | PSNR-μ | 44.66 dB | FlexHDR 44.35 dB | SAFNet 10× faster |
| Kalantari 17 | PSNR-l | 43.18 dB | FlexHDR 42.60 dB | +0.58 dB |
| Kalantari 17 | SSIM-l | 0.9917 | FlexHDR 0.9902 | +0.0015 |
| Kalantari 17 | Inference Time | 0.151s | SCTNet 3.466s | 23× faster |
| Kalantari 17 | Parameters | 1.12M | SCTNet 0.99M | Comparable |
| Challenge123 (512²) | PSNR-μ | 41.88 dB | AHDRNet 40.61 dB | +1.27 dB |
| Challenge123 (512²) | PSNR-l | 29.73 dB | AHDRNet 28.33 dB | +1.40 dB |

### Ablation Study

| Configuration | PSNR-μ | PSNR-l | Description |
|------|---------|------|------|
| Flow F + No Mask M | 33.69 | 36.30 | No selectivity, optical flow errors in saturated regions are severe |
| No Flow F + Mask M | 40.69 | 37.08 | No alignment, missing textures in moving regions |
| Flow F + Mask M | 41.68 | 39.61 | Joint refinement performs best |
| Refinement without Hm input | 43.63 | 41.67 | Loses first-stage fusion information |
| Refinement with F+M+Hm inputs | 44.59 | 43.15 | All information is complementary |
| S1=128, S2=128 | 44.59 | 43.15 | Small patch training |
| S1=512, S2=128 (WPC) | 44.66 | 43.18 | Window partition cropping is optimal |

### Key Findings

1. **Huge contribution of the mask**: Removing the mask results in a sharp drop in PSNR-μ from 41.68 to 33.69 (-8.0 dB), confirming the core value of selective alignment.
2. **Overwhelming speed advantage**: 18× faster than HDR-Transformer, 23× faster than SCTNet, and 10× faster than FlexHDR, thanks to the pure convolutional architecture without complex attention mechanisms.
3. **Significant advantage in large-motion scenes**: On the self-built Challenge123 dataset (average motion of 128.7 pixels vs 20.1 pixels in Kalantari 17), the advantages of SAFNet are further amplified.
4. **Patch limitations of Transformer methods**: Patch-based Transformer methods fail to aggregate textures generated by large motion across patches, leading to blocky artifacts.
5. Window partition cropping strategy: Large patches facilitate long-range aggregation while small patches facilitate detail refinement, making them complementary to each other.

## Highlights & Insights

- The observation that **"not all regions are worth aligning"** is highly precise—focusing computational resources on meaningful regions while avoiding error propagation in difficult and useless areas.
- The **joint refinement of the mask and optical flow forms a positive feedback loop**: the mask guides the flow to focus on valuable regions, and a better flow promotes more accurate region identification.
- The self-built Challenge123 dataset fills the gap for large-motion HDR evaluation (average motion 128.7 vs 20.1 pixels, saturation ratio 0.201 vs 0.061).
- Window partition cropping is an elegant training trick that cleverly unifies the requirements of different crop sizes in the two stages.

## Limitations & Future Work

1. PSNR-μ and HDR-VDP2 on the Tel 23 dataset are slightly inferior to those of Transformer-based methods, indicating that attention mechanisms still hold advantages in scenes dominated by deghosting (rather than large-motion aggregation).
2. Explicit optical flow alignment can still fail in extreme occlusion and severe deformation scenarios.
3. The refinement module using dilated residual blocks is relatively simple; more sophisticated refinement strategies could potentially yield further improvements.
4. The two-stage pipeline introduces extra hyperparameters (e.g., $\beta$, $\alpha$, window size), requiring careful tuning.
5. Although the Challenge123 dataset is challenging, its sample size is relatively small (96 training + 27 testing), which might lack diversity.

## Related Work & Insights

- **Kalantari et al. (2017)**: A pioneering learning-based HDR method (optical flow alignment + CNN fusion). SAFNet introduces a selective mechanism on top of it.
- **SCTNet / HDR-Transformer**: Representative Transformer-based HDR methods that feature high accuracy but are one to two orders of magnitude slower.
- **PWC-Net / LiteFlowNet**: Successful architectures for pyramid optical flow estimation, from which SAFNet's encoder-decoder design draws inspiration.
- Insight: **Selective processing + explicit operations** can be superior to global implicit attention in efficiency-sensitive tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of selective alignment fusion is clear and elegant, and window partition cropping is a clever training innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets + one self-built dataset + exhaustive ablations + efficiency comparisons + generalization tests.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, diagrams are intuitive, and the dataset contribution is valuable.
- Value: ⭐⭐⭐⭐⭐ It sets a new record for speed-accuracy trade-offs, holding significant value for mobile HDR deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VideoMamba: Spatio-Temporal Selective State Space Model](videomamba_spatio-temporal_selective_state_space_model.md)
- [\[ECCV 2024\] RGNet: A Unified Clip Retrieval and Grounding Network for Long Videos](rgnet_a_unified_clip_retrieval_and_grounding_network_for_long_videos.md)
- [\[ECCV 2024\] PiTe: Pixel-Temporal Alignment for Large Video-Language Model](pite_pixel-temporal_alignment_for_large_video-language_model.md)
- [\[CVPR 2026\] SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion](../../CVPR2026/video_understanding/savax_egotoexo_imitation_error_detection_via_scene.md)
- [\[CVPR 2025\] Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models](../../CVPR2025/video_understanding/video-panda_parameter-efficient_alignment_for_encoder-free_video-language_models.md)

</div>

<!-- RELATED:END -->
