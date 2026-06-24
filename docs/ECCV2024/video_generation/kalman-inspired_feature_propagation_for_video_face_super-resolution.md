---
title: >-
  [Paper Note] Kalman-Inspired Feature Propagation for Video Face Super-Resolution
description: >-
  [ECCV 2024][Video Generation][Video Face Super-Resolution] This paper proposes the KEEP framework, which leverages Kalman filtering principles to recursively fuse prior information from previous frames with observations of the current frame in the latent space. This achieves high-fidelity reconstruction of facial details and ensures temporal consistency in video face super-resolution, outperforming the previous state-of-the-art method by 0.8 dB in PSNR on the VFHQ dataset.
tags:
  - "ECCV 2024"
  - "Video Generation"
  - "Video Face Super-Resolution"
  - "Kalman Filtering"
  - "Temporal Consistency"
  - "CodeFormer"
  - "Feature Propagation"
date: 2026-05-08
content_hash: 75225bc590a43823
---

# Kalman-Inspired Feature Propagation for Video Face Super-Resolution

**Conference**: ECCV 2024  
**arXiv**: [2408.05205](https://arxiv.org/abs/2408.05205)  
**Code**: [https://github.com/jnjaby/KEEP](https://github.com/jnjaby/KEEP)  
**Area**: Video Generation  
**Keywords**: Video Face Super-Resolution, Kalman Filtering, Temporal Consistency, CodeFormer, Feature Propagation

## TL;DR
This paper proposes the KEEP framework, which leverages Kalman filtering principles to recursively fuse prior information from previous frames with observations of the current frame in the latent space. This achieves high-fidelity reconstruction of facial details and ensures temporal consistency in video face super-resolution, outperforming the previous state-of-the-art method by 0.8 dB in PSNR on the VFHQ dataset.

## Background & Motivation
Facial image super-resolution (FSR) has achieved remarkable progress in recent years, with various priors (geometric, reference, generative, and codebook priors) being successfully applied. However, video face super-resolution (VFSR) remains relatively under-explored. Existing solutions face a dilemma: one class of methods applies general video super-resolution networks (such as BasicVSR) to facial data, but they are not specialized for faces and fail to reconstruct fine facial details under severe degradation; another class of methods applies single-frame face image SR models (such as CodeFormer) frame-by-frame, which yields high single-frame quality but suffers from severe temporal inconsistency across frames—since FSR itself is an ill-posed problem, and a single degraded image can correspond to multiple high-resolution interpretations. **Key Challenge**: how to maintain temporal coherence of the video while preserving facial generation quality. **Key Insight**: already restored frames can serve as "references" to guide and constrain the restoration of the current frame, and this recursive updating mechanism using historical information aligns perfectly with the principles of Kalman filtering. **Core Idea**: introduce a Kalman filtering framework within the latent space of CodeFormer to recursively propagate stable facial priors through a prediction-update mechanism.

## Method

### Overall Architecture
KEEP is built on top of CodeFormer, and its overall pipeline consists of four modules: a low-quality (LQ) encoder $\mathcal{E}_L$, a decoder with codebook lookup $\mathcal{D}_Q$, a Kalman Filtering Network (KFN), and Cross-Frame Attention (CFA). At each time step, the LQ encoder encodes the current degraded frame into an observed state $\tilde{z}_t$; the state dynamic system utilizes the posterior estimate of the previous frame $\hat{z}_{t-1}^+$ through optical flow warping and an HQ encoder to obtain a prior prediction $\hat{z}_t^-$; the Kalman Filtering Network fuses both to obtain a more accurate posterior estimate $\hat{z}_t^+$; and finally, the decoder generates the high-quality output frame.

### Key Designs
1. **State Space Modeling (State Space Model)**:
    - **Function**: Formulate the VFSR problem as a state estimation problem in the latent space.
    - **Mechanism**: Generalize the linear state-space model of Kalman filtering to a non-linear form. The latent state $z_t$ is mapped to a high-quality frame $y_t = g_\theta(z_t)$ via a generative model $g_\theta$. The LQ encoder serves as the observation estimator $\tilde{z}_t = \mathcal{E}_L(x_t)$, and optical flow warping + HQ encoder are used for state transition to get $\hat{z}_t^- = \mathcal{E}_H(\omega(\mathcal{D}_Q(\hat{z}_{t-1}^+), \Phi_{t-1 \to t}))$.
    - **Design Motivation**: Modeling in a low-dimensional latent space is more efficient than operating directly in the pixel space, and latent representations focus more on perceptually significant changes. The two-step prediction-update mechanism of Kalman filtering is naturally suited for recursively correcting noisy estimates using temporal information.

2. **Kalman Gain Network (KGN)**:
    - **Function**: Adaptively fuse the prior prediction $\hat{z}_t^-$ and the current observation $\tilde{z}_t$.
    - **Mechanism**: The posterior state is updated via linear interpolation: $\hat{z}_t^+ = \mathcal{K}_t \hat{z}_t^- + (1 - \mathcal{K}_t) \tilde{z}_t$, where the Kalman gain $\mathcal{K}_t$ is directly learned from the data distribution by the KGN, instead of explicitly maintaining a covariance matrix. KGN consists of an uncertainty network (using spatial-temporal attention to estimate prediction uncertainty) and a gain network (computing the gain value for each codebook token), introducing the first frame $\tilde{z}_1$ as an anchor.
    - **Design Motivation**: Covariance estimation for high-dimensional signals is intractable (as shown by KalmanNet), and since covariance is only used to compute the gain, directly learning the gain is more practical. The anchor mechanism helps prevent long-term drift.

3. **Cross-Frame Attention (CFA)**:
    - **Function**: Promote local temporal consistency within the decoder.
    - **Mechanism**: Utilize cross-attention on the decoder's small-scale features ($16 \times 16$ and $32 \times 32$), using current frame features $v_t$ as Query, and previous frame features $v_{t-1}$ as Key and Value, to search, match, and fuse similar patches from the previous frame.
    - **Design Motivation**: While KFN ensures global style consistency at the latent code level, local texture details (such as hair) still need to be propagated at the decoder feature level. Selecting small-scale features avoids introducing blurriness.

### Loss & Training
Three-stage training strategy: Stage I trains for 800k iterations (lr=$2 \times 10^{-4}$), Stage II trains for 400k iterations (lr=$2 \times 10^{-4}$), and Stage III trains for 50k iterations (lr=$1 \times 10^{-4}$). The loss function includes pixel-level L1 loss ($\lambda_1 = 10^{-2}$), VGG perceptual loss ($\lambda_{VGG} = 1$), and GAN adversarial loss ($\lambda_{GAN} = 10^{-2}$). GMFlow is adopted for optical flow estimation. In the face alignment stage, a Gaussian low-pass filter is applied to landmarks to suppress temporal inconsistencies introduced during the alignment steps.

## Key Experimental Results

### Main Results
Quantitative comparison on the VFHQ-mild test set:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | IDS↑ | AKD↓ | σ_IDS↓ | σ_AKD↓ |
|------|-------|-------|--------|------|------|--------|--------|
| GPEN | 25.52 | 0.752 | 0.299 | 0.714 | 11.47 | 4.74 | 3.51 |
| GFPGAN | 26.29 | 0.780 | 0.248 | 0.744 | 10.55 | 4.57 | 3.65 |
| CodeFormer | 24.66 | 0.745 | 0.274 | 0.627 | 11.50 | 6.37 | 3.69 |
| BasicVSR++ | 27.20 | 0.806 | 0.196 | 0.764 | 11.31 | 5.25 | 4.64 |
| **KEEP (Ours)** | **27.99** | **0.827** | **0.162** | **0.796** | **8.82** | **3.69** | **3.25** |

KEEP achieves the best performance across all metrics, outperforming the second-best method, BasicVSR++, by approximately 0.8 dB in PSNR, with substantial leads in temporal consistency metrics (σ_IDS, σ_AKD, AKD).

### Ablation Study

| Configuration | LPIPS↓ | IDS↑ | AKD↓ | Description |
|------|--------|------|------|------|
| w/o CFA | 0.1621 | 0.7970 | 8.90 | Remove cross-frame attention, local texture consistency drops |
| w/o KFN | 0.1721 | 0.7773 | 9.20 | Remove Kalman filtering network, key temporal consistency drops significantly |
| Full model | 0.1619 | 0.7960 | 8.82 | Full model is optimal |
| PWC-Net flow | 0.1623 | 0.7957 | 8.78 | Switching to PWC-Net optical flow causes negligible performance difference |
| GMFlow (Ours) | 0.1619 | 0.7960 | 8.82 | Optical flow accuracy has little impact on final performance |

### Key Findings
- KFN is key to ensuring global style and identity consistency, while CFA further improves the temporal coherence of local textures.
- The accuracy of optical flow estimation has little impact on the final performance because the latent code is downsampled by 32x, making minor spatial deviations negligible at this scale.
- The advantages are especially pronounced in heavily degraded scenarios: while single-frame models degrade dramatically in performance, KEEP remains robust by utilizing complementary temporal information.
- More robust to non-frontal faces: the stable prior estimation allows the model to produce reasonable results even under challenging viewpoints like profile faces.
- Fluctuations in identity similarity across frames are significantly lower than CodeFormer, preventing sudden identity changes.

## Highlights & Insights
- Introducing the concept of Kalman filtering from classical signal processing to generative face restoration establishes an elegant theoretical framework, formalizing "how to fuse historical information and current observations" from a probabilistic estimation perspective.
- High generalizability of the method: although CodeFormer is used as the case study, the underlying Kalman filtering framework can be generalized to any encoder-codebook-decoder-based face restoration model.
- The anchor mechanism (first frame $\tilde{z}_1$) effectively mitigates the long-term drift problem in recursive propagation.
- The engineering detail of using a Gaussian low-pass filter on landmarks during face alignment is simple yet highly effective.

## Limitations & Future Work
- Currently, it only supports unidirectional (forward) recursive propagation and cannot leverage future frames; bidirectional propagation could potentially improve quality further.
- It relies on a pre-trained CodeFormer as the backbone, which is limited by the representational capacity of its codebook.
- Optical flow estimation can be inaccurate in scenarios with fast motion or heavy occlusion; while the paper states that the impact is small, robustness in extreme scenarios remains questionable.
- The three-stage training pipeline is relatively complex; end-to-end training could be more efficient.
- The validation is only conducted on the VFHQ dataset, lacking systematic evaluation on broader real-world videos.

## Related Work & Insights
- **BasicVSR/BasicVSR++**: General video super-resolution methods leveraging bidirectional hidden state propagation, but lacking face-specific priors.
- **CodeFormer**: Single-frame face restoration based on codebook priors, upon which this paper introduces temporal propagation.
- **KalmanNet**: Proposed the concept of directly learning the Kalman gain from data, which inspired the design of KGN in this work.
- **Tune-A-Video**: The cross-frame attention mechanism is adopted in this paper to enhance local temporal consistency.
- Insight: Combining classical control/signal processing theory with deep generative models is a promising direction.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of Kalman filtering and generative face restoration is novel, although individual components are not entirely brand new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Detailed evaluation under multi-level degradations, comprehensive ablations, and extensive qualitative analysis, though validated on only one main dataset.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear mathematical derivations; the logical progression from Kalman filtering to the concrete implementation is highly complete and elegant.
- **Value**: ⭐⭐⭐⭐ Provides a theoretically grounded unified framework for video face restoration, showing high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] RealViformer: Investigating Attention for Real-World Video Super-Resolution](realviformer_investigating_attention_for_real-world_video_super-resolution.md)
- [\[CVPR 2025\] VideoGigaGAN: Towards Detail-rich Video Super-Resolution](../../CVPR2025/video_generation/videogigagan_towards_detail-rich_video_super-resolution.md)
- [\[CVPR 2025\] PatchVSR: Breaking Video Diffusion Resolution Limits with Patch-Wise Video Super-Resolution](../../CVPR2025/video_generation/patchvsr_breaking_video_diffusion_resolution_limits_with_patch-wise_video_super-.md)
- [\[CVPR 2026\] VSRELL: A Simple Baseline for Video Super-Resolution and Enhancement in Low-Light Environment](../../CVPR2026/video_generation/vsrell_a_simple_baseline_for_video_super-resolution_and_enhancement_in_low-light.md)
- [\[ICCV 2025\] VSRM: A Robust Mamba-Based Framework for Video Super-Resolution](../../ICCV2025/video_generation/vsrm_a_robust_mamba-based_framework_for_video_super-resolution.md)

</div>

<!-- RELATED:END -->
