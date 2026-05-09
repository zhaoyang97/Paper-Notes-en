---
title: >-
  [Paper Note] ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement
description: >-
  [AAAI 2026][Image Restoration][Low-light image enhancement] Targeting two overlooked statistical distribution issues in the HVI color space — large distribution discrepancy between chrominance and luminance branches leading to insufficient complementary feature extraction, and weak inter-chrominance correlation causing gradient conflicts — this paper proposes the ICLR framework. It introduces a Dual-stream Interaction Enhancement Module (DIEM) and a Covariance Correction Loss (CCL) to address these issues from the perspectives of fusion enhancement and statistical distribution optimization, respectively, achieving state-of-the-art performance on the LOL benchmark series.
tags:
  - AAAI 2026
  - Image Restoration
  - Low-light image enhancement
  - HVI color space
  - chrominance-luminance interaction
  - covariance correction loss
  - attention fusion
date: 2026-05-08
content_hash: 078b5930fc5631cf
---

# ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement

**Conference**: AAAI 2026
**arXiv**: [2511.13607](https://arxiv.org/abs/2511.13607)
**Code**: N/A
**Area**: Image Restoration / Low-Light Enhancement
**Keywords**: Low-light image enhancement, HVI color space, chrominance-luminance interaction, covariance correction loss, attention fusion

## TL;DR
Targeting two overlooked statistical distribution issues in the HVI color space — large distribution discrepancy between chrominance and luminance branches leading to insufficient complementary feature extraction, and weak inter-chrominance correlation causing gradient conflicts — this paper proposes the ICLR framework. It introduces a Dual-stream Interaction Enhancement Module (DIEM) and a Covariance Correction Loss (CCL) to address these issues from the perspectives of fusion enhancement and statistical distribution optimization, respectively, achieving state-of-the-art performance on the LOL benchmark series.

## Background & Motivation

**Background**: Low-light image enhancement (LLIE) is a fundamental task in computer vision. In recent years, methods based on color space decoupling have become the dominant paradigm — converting RGB images into color spaces such as HSV, YCbCr, or HVI, and separately processing chrominance and luminance. The HVI color space (CVPR 2025, CIDNet) overcomes the pure-black plane issue of HSV and maintains a bijective mapping with RGB, representing the current state-of-the-art decoupling scheme.

**Limitations of Prior Work**: Although CIDNet introduces cross-attention for luminance-chrominance interaction in the HVI space, it overlooks two critical statistical distribution issues:
   - **Large distribution discrepancy between luminance and chrominance branches**: The feature map distributions of the two branches differ substantially in shape (e.g., unimodal for luminance vs. multimodal for chrominance), causing cross-attention weights to become over-smoothed and failing to extract complementary features effectively. Moreover, luminance errors propagate into chrominance channels via the nonlinear parameter $C_k$.
   - **Weak inter-chrominance correlation**: For images containing large uniform-color regions, the H and V chrominance branches exhibit highly concentrated distributions with weak linear correlation. Conventional pixel-wise loss functions assume strong correlation for joint optimization, producing gradient conflicts in weakly correlated regions.

**Key Challenge**: Distribution discrepancy → cross-attention failure → insufficient complementary information extraction; weak correlation → pixel-wise loss gradient cancellation → optimization difficulty. These two issues manifest at the model architecture and loss function levels, respectively.

**Goal**:
   - Sub-problem 1: How to effectively extract luminance-chrominance complementary information under large distribution discrepancy?
   - Sub-problem 2: How to prevent luminance errors from propagating into chrominance channels via nonlinear parameters?
   - Sub-problem 3: How to avoid gradient conflicts when chrominance branches are weakly correlated?

**Key Insight**: The authors conduct large-scale statistical analysis of natural image distributions, identifying two previously overlooked distributional phenomena (Observation 1 & 2), and propose targeted solutions from both module design and loss function design perspectives.

**Core Idea**: Align distribution discrepancies via multi-dimensional attention fusion, and alleviate gradient conflicts via covariance-constrained loss — systematically resolving interaction issues in HVI space at both the architecture and loss function levels.

## Method

### Overall Architecture
ICLR adopts a three-scale U-Net architecture. The input low-light image is converted from RGB to HVI and split into a luminance branch I and chrominance branches (H, V). At each encoder-decoder level, the two branches interact through the DIEM module. The enhanced HVI features are then converted back to RGB. During training, the CCL loss replaces conventional pixel-wise L1/L2 losses.

### Key Designs

1. **Multi-dimensional Attention-guided Fusion Module (MAFM)**:

    - **Function**: Aligns the distributions of luminance and chrominance branches so that cross-attention can effectively extract complementary information.
    - **Mechanism**: Given luminance features $F_I$ and chrominance features $F_{HV}$, an initial fusion is computed as $F_{init} = F_I + F_{HV}$. Channel attention and spatial attention are then applied in parallel to yield $W_c$ and $W_s$, which are further refined into pixel-wise weights via pixel attention. The dynamic weight is $W = \varphi \cdot W_c + \omega \cdot W_s$, and the fused result is $F'_{HV} = F_{init} + W \cdot F_I + (1-W) \cdot F_{HV}$.
    - **Design Motivation**: Channel attention captures the discriminability of global complementary features; spatial attention preserves local detail complementarity; pixel attention ensures fine-grained chrominance-luminance complementarity. The three-level attention progressively aligns distributions from coarse to fine, rather than relying on a single-step cross-attention.
    - **Distinction from conventional cross-attention**: Cross-attention produces over-smoothed weights under large distribution discrepancy, whereas MAFM actively narrows the gap between the two branches through adaptive fusion prior to interaction.

2. **Cross Dynamic Enhancement Module (CDEM)**:

    - **Function**: After distribution alignment by MAFM, further enhances the representation of complementary information.
    - **Mechanism**: Taking luminance branch enhancement as an example, complementary information is first extracted via cross-attention from the aligned chrominance features $F'_{HV}$: $Z_I = \text{CrossAttention}(F_I, F'_{HV})$. A dynamic weighting mechanism then modulates the complementary information: $\hat{Z}_I = \lambda \cdot \text{FFN}(\alpha \cdot Z_I + \beta \cdot F'_{HV}) + \mu \cdot Z_I$. Finally, a Multi-branch Feature Enhancement Module (MFEM) extracts features in parallel across different orientations, scales, and semantic spaces.
    - **Design Motivation**: MAFM addresses the problem of "seeing" complementary information; CDEM addresses "utilizing" it. The learnable parameters $\alpha, \beta, \lambda, \mu$ control the priority between complementary and original information, while MFEM captures multi-scale complementary patterns through multi-branch convolutions.

3. **Dual-stream Interaction Enhancement Module (DIEM)**:

    - **Function**: Combines MAFM and CDEM into a complete interaction unit.
    - **Mechanism**: Within DIEM, a first MAFM performs initial distribution alignment, followed by CDEM for complementary information enhancement, and a second MAFM for fine-grained fusion.
    - **Design Motivation**: Ablation experiments confirm that both MAFMs are indispensable (removing either one results in drops of 0.231 dB and 0.268 dB, respectively), forming a progressive "align → enhance → refine" interaction pipeline.

### Loss & Training

The **Covariance Correction Loss (CCL)** consists of three components:

1. **Luminance-guided Error Correction $L_{I-HV}$**: The mean and variance of luminance errors are used as adaptive weights to modulate the penalty on the chrominance branches. When the luminance error $\Delta I$ is large, errors in the nonlinear parameter $C_k$ are also amplified, increasing errors $\Delta H$ and $\Delta V$ in the chrominance channels. Accordingly, the chrominance losses are weighted by $W_H = 1 + \frac{1}{N}\sum|I_i - \hat{I}_i|$ and $W_V = 1 + \text{std}(|I_i - \hat{I}_i|)$, imposing heavier penalties on chrominance when luminance errors are large.

2. **Covariance Statistical Constraint $L_{HV}$**: Directly constrains the joint mean term $E(HV)$, shifting optimization from pixel-wise comparison to statistical distribution learning. The formulation is $L_{HV} = \frac{1}{B}\sum(\frac{1}{HW}\sum \hat{H}_i\hat{V}_j - \frac{1}{HW}\sum H_iV_j)^2$. Even when H and V are weakly correlated, constraining the covariance of the overall distribution avoids gradient conflicts.

3. **Final Loss**: $L = L_I + L_{I-HV} + L_{HV}$, where the three terms independently constrain luminance, luminance-guided chrominance correction, and the joint chrominance distribution.

## Key Experimental Results

### Main Results

Trained on LOLv1, evaluated on LOLv1/LOLv2-Real/LOLv2-Syn (cross-domain generalization setting):

| Dataset | Metric | ICLR (Ours) | CIDNet (CVPR'25) | URWKV (CVPR'25) | Gain |
|--------|------|------------|------------------|-----------------|------|
| LOLv1 | PSNR↑ | **28.603** | 27.732 | 26.513 | +0.87 dB |
| LOLv1 | SSIM↑ | **0.885** | 0.870 | 0.869 | +0.015 |
| LOLv1 | LPIPS↓ | **0.068** | 0.117 | 0.107 | −0.039 |
| LOLv2-Real | PSNR↑ | **32.320** | 31.436 | 31.413 | +0.88 dB |
| LOLv2-Syn | PSNR↑ | **20.719** | 20.375 | 20.518 | +0.20 dB |

The model has 4.27M parameters, larger than URWKV (2.25M) and CIDNet (1.88M), but demonstrates competitive advantages in FLOPs and inference time.

### Ablation Study

| Config | MAFM | CDEM | CCL | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|------|-----|-------|-------|--------|
| Ω5 (Full) | ✔ | ✔ | ✔ | **28.603** | **0.885** | **0.068** |
| Ω4 (w/o MAFM) | ✘ | ✔ | ✔ | 28.028 | 0.880 | 0.080 |
| Ω3 (CDEM→TCA) | ✔ | TCA | ✔ | 27.998 | 0.883 | 0.074 |
| Ω1 (CCL→L1) | ✔ | ✔ | L1 | 28.202 | 0.879 | 0.073 |
| Ω2 (CCL→L2) | ✔ | ✔ | L2 | 28.275 | 0.880 | 0.073 |

### Key Findings
- **CDEM contributes the most**: Replacing CDEM with conventional cross-attention (TCA) results in a 0.605 dB PSNR drop, demonstrating that the dynamic enhancement mechanism is far more effective than simple cross-attention.
- **MAFM is indispensable**: Removing MAFM leads to a 0.575 dB drop and LPIPS degradation from 0.068 to 0.080.
- **CCL's core value lies in low-covariance images**: On images with covariance ≤ 0.01 (comprising 63% of the test set), CCL accounts for 90% of the overall performance gain, precisely validating Observation 2.
- **Both MAFMs are important**: Ablations show that removing the first or second MAFM results in drops of 0.268 dB and 0.231 dB, respectively.

## Highlights & Insights
- **Problem discovery from a statistical distribution perspective**: Rather than modifying the network intuitively, the authors first conduct extensive statistical analyses (distribution discrepancy and covariance correlation), identify two overlooked fundamental phenomena, and then design targeted solutions. This "diagnose first, then prescribe" research paradigm is worth emulating.
- **CCL elevates loss function design to the statistical level**: Conventional pixel-wise losses implicitly assume strong inter-channel correlation. CCL abandons this assumption by constraining the covariance (joint mean $E(HV)$), fundamentally resolving gradient conflicts in weakly correlated regions. This idea is transferable to any multi-channel joint optimization task.
- **Luminance-error-adaptive weighting of chrominance penalties**: Using the mean and standard deviation of luminance errors to weight the H and V channel losses respectively is a concise yet effective strategy for suppressing cascaded error propagation through nonlinear parameters.

## Limitations & Future Work
- **Relatively large parameter count**: At 4.27M parameters, the model is more than twice the size of CIDNet (1.88M) and URWKV (2.25M), posing deployment challenges on mobile devices.
- **Evaluation limited to the LOL benchmark series**: Although NIQE evaluations are conducted on five unpaired datasets, validation in practical application scenarios (e.g., smartphone photography, surveillance) is absent.
- **Empirical choice of $W_H$ and $W_V$ in CCL**: The rationale for using mean to weight the H channel and standard deviation for the V channel lacks rigorous theoretical justification.
- **Computational overhead of the three-level attention in MAFM**: Channel, spatial, and pixel attention applied in combination may become a bottleneck for high-resolution images; lightweight alternatives warrant exploration.
- **Generalizability to other color spaces unexplored**: The method theoretically relies on properties specific to the HVI space (nonlinear parameter $C_k$); whether it transfers to other decoupled spaces such as YCbCr remains to be verified.

## Related Work & Insights
- **vs. CIDNet (CVPR'25)**: CIDNet is the first to introduce luminance-chrominance cross-attention in the HVI space along with a dual color-space L1 constraint, but neglects distribution discrepancy and weak correlation. ICLR builds upon CIDNet with two levels of improvement from a statistical distribution perspective (module and loss), achieving a +0.87 dB PSNR gain on LOLv1.
- **vs. URWKV (CVPR'25)**: URWKV employs an RWKV architecture for low-light restoration with fewer parameters, but lacks fine-grained chrominance-luminance interaction design. ICLR outperforms it across all metrics.
- **vs. Retinexformer (ICCV'23)**: A Retinex decomposition-based method operating in RGB space, lacking the advantages of color space decoupling.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Systematically identifies and addresses HVI-space interaction issues from a statistical distribution perspective; CCL design is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablations are comprehensive; the low-covariance subset analysis is particularly insightful; more real-world scenario validation would strengthen the work.
- **Writing Quality**: ⭐⭐⭐⭐ The two Observations are introduced clearly and compellingly; mathematical derivations are complete.
- **Value**: ⭐⭐⭐⭐ The statistical distribution perspective of CCL is transferable; the multi-dimensional fusion strategy of MAFM has broad applicability.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] CWNet: Causal Wavelet Network for Low-Light Image Enhancement](../../ICCV2025/image_restoration/cwnet_causal_wavelet_network_for_low-light_image_enhancement.md)
- [\[ICCV 2025\] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](../../ICCV2025/image_restoration/low-light_image_enhancement_using_event-based_illumination_estimation.md)
- [\[NeurIPS 2025\] Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancement](../../NeurIPS2025/image_restoration/luminance-aware_statistical_quantization_unsupervised_hierarchical_learning_for_.md)
- [\[CVPR 2026\] UDAPose: Unsupervised Domain Adaptation for Low-Light Human Pose Estimation](../../CVPR2026/image_restoration/udapose_unsupervised_domain_adaptation_for_low_light_human_pose_estimation.md)
- [\[AAAI 2026\] Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration](clear_nights_ahead_towards_multi-weather_nighttime_image_res.md)

<!-- RELATED:END -->
