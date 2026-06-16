---
title: >-
  [Paper Note] Target-Aware Invertible Encoder with Reconstruction Guidance for Infrared Small Target Detection
description: >-
  [CVPR 2026][Object Detection][Paper Note] InvDet utilizes an invertible encoder to transform the "information loss of infrared small targets caused by downsampling" into an observable and optimizable quantity. It employs a forward path for detection and a backward path for input reconstruction. By using TARM to focus reconstruction on the target and GCTM as a
tags:
  - CVPR 2026
  - Object Detection
date: 2026-05-08
content_hash: f30fcd2ccd608eb3
---
# Target-Aware Invertible Encoder with Reconstruction Guidance for Infrared Small Target Detection

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Yan_Target-Aware_Invertible_Encoder_with_Reconstruction_Guidance_for_Infrared_Small_Target_CVPR_2026_paper.html)  
**Code**: None  
**Area**: Object Detection / Infrared Small Target Detection  
**Keywords**: Infrared Small Target Detection, Invertible Encoder, Reconstruction Guidance, Information Preservation, Gradient Decoupling

## TL;DR
InvDet utilizes an invertible encoder to transform the "information loss of infrared small targets caused by downsampling" into an observable and optimizable quantity. It employs a forward path for detection and a backward path for input reconstruction. By using TARM to focus reconstruction on the target and GCTM as a pixel-level weight map (replacing IoU) to supervise the reconstruction, it achieves competitive accuracy on five infrared benchmarks and demonstrates strong cross-dataset generalization.

## Background & Motivation

**Background**: Mainstream deep detectors for Infrared Small Target Detection (ISTD) follow the paradigms of general object detection—deepening backbones and stacking downsampling layers (strided conv / pooling) to compress feature maps to 1/16 or 1/32 of the input size to gain large receptive fields and high-level semantics.

**Limitations of Prior Work**: Infrared small targets themselves are "weak signals + extremely small spatial occupancy" (e.g., a 2×2 pixel point target in Figure 1). Downsampling acts essentially as a low-pass filter, systematically attenuating and dispersing these faint clues into background clutter. Visualizations show that information loss accumulates rapidly with the downsampling ratio; most targets "disappear" after 16×, making it impossible for the decoder to recover them via upsampling. This constitutes the performance bottleneck of ISTD.

**Key Challenge**: Existing remedies (dense skip connections/attention, IoU-unfriendly tolerance metrics and losses like TAM, or joint low-level tasks like non-uniformity correction/super-resolution) are all **post-hoc compensation** for information loss. They do not address the root cause—downsampling is **non-injective**, and once information is discarded in the forward pass, it is lost.

**Key Insight**: The authors draw inspiration from Invertible Rescale Nets (IRN) in image scaling—modeling downsampling/upsampling as a **bijective transformation**. The original high-resolution image can be accurately reconstructed from the low-resolution representation + a latent variable. This provides a new perspective for detection: rather than compensating post-hoc, make "information loss" **measurable and directly optimizable** at the source.

**Core Idea**: Use an invertible encoder to reconstruct the input from forward latent variables, making information loss an explicitly optimizable quantity. Employ Target-Aware Reconstruction Modulation (TARM) and Geometry-Content Tolerance Measure (GCTM) to ensure reconstruction only serves to "preserve targets," thereby constraining feature extraction into a representation that is "detection-friendly."

## Method

### Overall Architecture

During training, InvDet runs two complementary paths: the **forward detection path** (solid line) and the **backward reconstruction path** (red dashed line). Given an infrared input $X \in \mathbb{R}^{H\times W\times 1}$, the invertible encoder extracts multi-scale features $\{Y_s\}_{s=1}^{S}$. The forward path feeds these features into MMFB (Multi-skip Multi-scale Fusion) to obtain $P_s$, followed by stage-wise upsampling via transposed convolutions with residuals $F_s = P_s + \text{UpSample}(P_{s+1})$, until the detection head outputs target attributes from $F_1$. The backward path uses the same InvBlock parameters to analytically reverse the latent variables back to $X_{rec}$—but only after TARM modulation, making $X_{rec}$ a "target-aware proxy" rather than a perfect inverse. The reconstruction error is softly constrained by the GCTM weight map $W_s$. Crucially, the two paths are updated using **two independent optimizers** to prevent interference. At inference, the entire backward path is deactivated, leaving only the efficient forward path.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    X["Input Infrared X"] --> ENC["Invertible Encoder<br/>InvBlock(Invertible)+ConvBlock<br/>Multi-scale Ys"]
    ENC -->|Forward Path| DET["MMFB Fusion + Decoder<br/>→ Head → Output"]
    ENC -->|Backward Path (Shared Weights)| TARM["TARM Target-Aware Modulation<br/>HP Soft-Gating + LP Gain"]
    GCTM["GCTM Tolerance Measure<br/>→ Weight Map Ws"] -->|Guidance| TARM
    TARM --> REC["HaarUpsample Synthesis<br/>→ Proxy Xrec → Recon Loss"]
    DET -.Det Optimizer.-> OPT["Gradient Decoupled Training<br/>Two Independent Optimizers"]
    REC -.Recon Optimizer.-> OPT
```

### Key Designs

**1. Invertible Encoder: Making information loss observable and steerable**

To address the "lossy downsampling" root cause, InvDet adopts a hybrid structure: **first $S_{rev}$ invertible stages + subsequent standard convolutional stages**. The invertible stages use orthogonal Haar analysis/synthesis operators ($\mathcal{H}$ for downsampling, $\mathcal{H}^{-1}$ for upsampling) to decompose the input into low-frequency $x_s^l$ and high-frequency $x_s^h$ parts: $(x_s^l, x_s^h)=\mathcal{H}(X_{s-1})$. This orthogonal transformation halves the resolution without discarding spatial information. Bijective coupling is then performed via InvBlocks: $y_s^l = x_s^l + \phi(x_s^h)$, $y_s^h = x_s^h \odot \exp(\Psi) + \rho(y_s^l)$, where $\Psi=\eta(y_s^l)$ is constrained via clamping to prevent gradient explosion. In the backward pass, the input is exactly recovered as $x_s^h=(y_s^h-\rho(y_s^l))\odot\exp(-\Psi)$ and $x_s^l=y_s^l-\phi(x_s^h)$. Standard ConvDownsample layers are used only for $s>S_{rev}$ to expand the receptive field and extract discriminative semantics for detection only. This separation allows early stages to preserve small target information for reconstruction while deep stages focus on detection semantics. Because the backward pass can reconstruct the input, information loss becomes an explicitly optimizable term in the loss function for the first time.

**2. Gradient Decoupled Training: Supervising features without interfering with detection heads**

If reconstruction and detection losses share one optimizer, reconstruction gradients would flow into the neck and detection head, disturbing the learning of the detection task itself. InvDet uses **two independent optimizers**: the detection optimizer only updates the neck and prediction head, whereas the reconstruction optimizer updates the invertible encoder based on the modulated reconstruction loss. This ensures "clean" gradient flow—reconstruction directly regularizes the feature extraction process (forcing the encoder to learn complete, detection-friendly representations) without polluting detection-specific modules. Combined with weight sharing between the forward and backward paths (analytical inversion with zero extra trainable weights and zero inference overhead), reconstruction acts as a "direct constraint on features" rather than an independent parallel stream.

**3. TARM: Converting reconstruction from "uniform fidelity" to "target-strict, background low-pass"**

Pixel-wise uniform reconstruction of the entire image (including clutter) is counterproductive for detection as it introduces noise. TARM operates solely in the backward path, with modulation intensity determined by two signals: spatially via the GCTM weight map $W_s \in [0,1]$ to focus on informative regions, and temporally via a cosine ramp-up factor $r_s=\tfrac12(1-\cos(\pi\xi))$ where $\xi=\text{clip}(\frac{e-e_0}{\Delta e},0,1)$, ensuring modulation strengthens smoothly to avoid abrupt info loss. Specifically, three operations are element-wise gated by $W_s$ and $r_s$: LP Gain $\hat{y}_s^l = y_s^l\odot(1+\gamma r_s\sqrt{W_s})$ moderately boosts target structures; HP Soft-Gating + high-boost residual $\hat{y}_s^h = y_s^h\odot W_s^{\theta r_s} + \delta W_s\odot[\text{HB}(y_s^h)-y_s^h]$ suppresses background textures while preserving target edges. Modulated latent variables are used only for the reconstruction path and do not disturb the forward detection distribution.

**4. GCTM: A tolerance measure providing both geometric and appearance clues**

IoU is overly sensitive to tiny targets (large fluctuations from a few pixels of shift), making it unstable for supervision. GCTM fuses geometric and appearance consistency. Geometric consistency $\mathbb{S}_{geo}=\exp(-(d_c/t_{center})^2-(|A_{pr}-A_{gt}|/t_{area})^2)$ adopts the TAM concept with scale-adaptive parameters $t_{center}=\sqrt{w_{gt}^2+h_{gt}^2}$ and $t_{area}=A_{gt}$. Appearance consistency $\mathbb{S}_{gray}=\text{BC}(\mathcal{P}_{gt},\mathcal{P}_{pr})/t_{gray}$ uses a radiance-aware denominator where $BC$ is the Bhattacharyya coefficient and $t_{gray}=\text{LSNR}(\mathcal{P}_{gt})/(1+H_{bg})+\varepsilon$ (LSNR is local signal-to-noise ratio, $H_{bg}$ is background entropy). These are fused via a geometry-driven weight $\text{GCTM}=\lambda\mathbb{S}_{geo}+(1-\lambda)\mathbb{S}_{gray}$, where $\lambda=\sigma(\mathbb{S}_{geo}/\tau)$. These instance-level scores are rasterized via scale-adaptive Gaussian masks into a full weight map $W_{full}$, then downsampled to $W_s$ for TARM.

### Loss & Training

The training objective includes detection loss and reconstruction loss softly weighted by $W_s$, with gradients backpropagated through two independent optimizers. TARM intensity $r_s$ increases smoothly with epochs. Core structural hyperparameters include the invertible depth $S_{rev}$ and the number of InvBlocks per stage $n_s^{block}$. At inference, the backward path is removed, and forward throughput is unaffected.

## Key Experimental Results

Evaluated on 5 infrared benchmarks (IRSTD-1K, NUAA-SIRST, NUDT-SIRST, IRSTD, DUAB) using official splits and reporting Recall / Precision / F1. DUAB is stratified by target area into point/spot/extended for analysis.

### Main Results (Comparison with SOTA, F1 %)

| Dataset | Ours (InvDet) | Second Best | Gain |
|--------|------------|----------|------|
| IRSTD-1K | 84.4 | 80.3 (MA-Net) | +4.1 |
| NUAA-SIRST | 87.4 | 83.9 (DNA-Net) | +3.5 |
| NUDT-SIRST | 86.2 | 84.7 (MA-Net) | +1.5 |
| DUAB-Spot | 93.5 | 91.4 (MA-Net) | +2.1 |
| DUAB-Extended | 98.2 | 96.9 (DNA-Net) | +1.3 |
| IRSTD | 97.8 | 98.3 (MA-Net) | −0.5 ⚠️ |
| DUAB-Point | 93.5 | 98.2 (MA-Net) | −4.7 ⚠️ |

InvDet achieves the best F1 on most benchmarks. On IRSTD and DUAB-Point, it slightly lags behind MA-Net. The authors attribute this to the much larger scale of these two datasets (IRSTD 32k+, DUAB 12k+), which favors "dataset-specific fitting," whereas InvDet's strength lies in generalizable representations.

### Cross-Dataset Generalization (F1 % Retention, No Fine-tuning)

| Train → Test | IRSTD-1K | NUAA-SIRST | NUDT-SIRST |
|-----------|----------|------------|------------|
| IRSTD-1K (In-domain 84.4) | — | 77.8 (89.1%) | 74.3 (86.1%) |
| NUAA-SIRST (In-domain 87.4) | 74.3 (88.0%) | — | 75.3 (87.4%) |
| NUDT-SIRST (In-domain 86.2) | 63.7 (75.5%) | 72.6 (83.1%) | — |

The average cross-domain F1 retention is **84.9%**. Real-to-real transfer is strongest (88-89% retention between IRSTD-1K and NUAA-SIRST despite a 2× resolution difference), while synthetic-to-real (NUDT-SIRST → IRSTD-1K) retains 75.5%. This supports the argument that the gain comes from generalizable representations rather than overfitting.

### Ablation Study: $S_{rev}$ × InvBlocks per stage

| Config ($S_{rev}$, $n^{block}$) | IRSTD-1K F1 | E2E FPS | FWD FPS | Note |
|------|------|------|------|------|
| $S_{rev}=2$, [2,2,2,2] | 84.40 | 50.30 | 72.72 | Best Accuracy |
| $S_{rev}=2$, [1,1,1,1] | 83.18 | 72.06 | 115.24 | Fewer blocks → Faster but lower accuracy |
| $S_{rev}=4$, [1,1,1,1] | 84.11 | 78.55 | 126.49 | Large $S_{rev}$ mostly affects training speed |
| $S_{rev}=4$, [4,4,4,4] | 81.26 | 37.91 | 47.74 | Capacity too high, accuracy drops |

### Key Findings
- **Zero Inference Cost for Backward Path**: By deactivating the backward path during testing, forward throughput (FWD FPS) remains high. Increasing $n_s^{block}$ increases FLOPs/Params, while increasing $S_{rev}$ primarily affects training-time E2E FPS. The cost of the invertible design is concentrated in training, not deployment.
- **Improved Recall without Precision Loss**: Feature evolution visualizations (Fig. 6) show that the LP branch raises the target baseline at $W_s$ peaks, while the HP branch suppresses background texture and sharpens boundaries. As the weight map focuses on actual targets via cosine ramp-up, recall increases without sacrificing precision.
- **Deeper is Not Always Better**: F1 actually drops when $S_{rev}/n^{block}$ is too large (e.g., [4,4,4,4]), suggesting that invertible capacity must match the task.

## Highlights & Insights
- **Turning "Information Loss" into an Optimizable Quantity**: The most significant insight is using bijective reconstruction to transform an abstract, post-hoc problem into an explicit signal that can be directly optimized at the source—a paradigm shift rather than just another module.
- **Zero Extra Parameters and Inference Overhead**: The backward path reuses the same forward InvBlock parameters for analytical inversion, gaining a "free" supervision signal that can be discarded at deployment.
- **Generalizability of TARM's "Target-Strict, Background-Lowpass" Philosophy**: The idea of reallocating reconstruction fidelity from uniform to target-focused using task-aware weight maps can be migrated to any scenario where auxiliary tasks are more critical for the foreground (e.g., medical lesions, remote sensing).

## Limitations & Future Work
- Code is not yet public (⚠️ check official sources), and the sensitivity of various TARM hyperparameters ($\gamma,\theta,\delta,\tau$, and ramp-up $e_0/\Delta e$) is mostly discussed in the supplementary material, making tuning costs hard to judge.
- Slight inferiority to MA-Net on large datasets (IRSTD, DUAB-Point) suggests that InvDet's ceiling might be lower than methods designed for large-scale data fitting; direct controlled experiments are needed.
- Invertible constraints + dual-optimizer training increase training complexity and memory usage; the performance drop at high $S_{rev}/n^{block}$ suggests structural search is required.
- GCTM's content term depends on statistics like LSNR and background entropy, which might not be robust under extreme noise or atypical infrared imaging.

## Related Work & Insights
- **vs IRN (Invertible Rescale Net)**: IRN models scaling as a bijection for high-fidelity reconstruction from low-res + latent variables. However, using IRN directly as a backbone forces reconstruction of the entire cluttered scene, which is inefficient and misaligned with detection goals. InvDet makes the invertible structure "target-aware" via TARM.
- **vs IA-YOLO / UniCD (Joint Low-level + High-level Tasks)**: These treat low-level tasks (dehazing, etc.) as independent parallel flows or preprocessing. InvDet views low-level (reconstruction) and high-level (detection) as two perspectives of the same invertible transformation, directly constraining the feature extraction process.
- **vs TAM / scale-location-sensitive loss**: These modify the evaluation/loss to be friendlier to small targets. GCTM adds radiance-aware appearance consistency to geometric tolerance and uses the resulting map to guide reconstruction at the source.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First framework to introduce invertible encoders + gradient decoupling to ISTD, making information loss optimizable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid results across 5 benchmarks and cross-dataset generalization, though some hyperparameter sensitivity is relegated to the supplement.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and complete formulas; invertible/modulation sections are notation-heavy and require careful reading.
- Value: ⭐⭐⭐⭐ Zero inference overhead + strong cross-domain generalization make it attractive for real-world deployment; ideas are transferable to other weak/small object scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CHAL: Causal-guided Hierarchical Anomaly-aware Learning for Moving Infrared Small Target Detection](chal_causal-guided_hierarchical_anomaly-aware_learning_for_moving_infrared_small.md)
- [\[CVPR 2026\] Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective](seeing_through_the_noise_improving_infrared_small_target_detection_and_segmentat.md)
- [\[NeurIPS 2025\] Rethinking Evaluation of Infrared Small Target Detection](../../NeurIPS2025/object_detection/rethinking_evaluation_of_infrared_small_target_detection.md)
- [\[CVPR 2026\] Geometry-Aligned and Anomaly-Aware Reconstruction for 3D Anomaly Detection](geometry-aligned_and_anomaly-aware_reconstruction_for_3d_anomaly_detection.md)
- [\[ICCV 2025\] From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision](../../ICCV2025/object_detection/from_easy_to_hard_progressive_active_learning_framework_for_infrared_small_targe.md)

</div>

<!-- RELATED:END -->
