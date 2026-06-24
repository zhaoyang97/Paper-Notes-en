---
title: >-
  [Paper Note] PicoSAM3: Real-Time In-Sensor Region-of-Interest Segmentation
description: >-
  [CVPR 2025][Segmentation][In-sensor computing] PicoSAM3 is an ultra-lightweight promptable segmentation model with 1.3M parameters. Through implicit ROI prompt encoding, a dense CNN architecture (transformer-free), SAM3 knowledge distillation, and INT8 quantization, it achieves 65.45% mIoU on COCO and real-time inference within 11.82ms on the Sony IMX500 vision sensor.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "In-sensor computing"
  - "SAM distillation"
  - "INT8 quantization"
  - "lightweight segmentation"
  - "Sony IMX500"
  - "ROI prompts"
date: 2026-05-08
content_hash: 5052c8fb9c6f11c6
---

# PicoSAM3: Real-Time In-Sensor Region-of-Interest Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2603.11917](https://arxiv.org/abs/2603.11917)  
**Code**: [GitHub](https://github.com/pbonazzi/picosam3)  
**Area**: Segmentation / Edge Deployment  
**Keywords**: In-sensor computing, SAM distillation, INT8 quantization, lightweight segmentation, Sony IMX500, ROI prompts

## TL;DR
PicoSAM3 is an ultra-lightweight promptable segmentation model with 1.3M parameters. Through implicit ROI prompt encoding, a dense CNN architecture (transformer-free), SAM3 knowledge distillation, and INT8 quantization, it achieves 65.45% mIoU on COCO and real-time inference within 11.82ms on the Sony IMX500 vision sensor.

## Background & Motivation
**Background**: The SAM/SAM2/SAM3 series has achieved breakthrough progress in promptable segmentation, but the computational and memory footprint of their Transformer architectures prevents deployment on extreme edge devices.

**Limitations of Prior Work**: (1) Compression schemes like TinySAM and EdgeSAM still retain Transformer structures, exceeding the 8MB memory limit and relying on unsupported operators (Softmax, LayerNorm); (2) In-sensor computing requires the model to run entirely within the CMOS sensor, imposing extremely strict limits on operator types, integer precision, and memory; (3) Existing lightweight SAM models ignore the spatial flexibility of the ROI.

**Key Challenge**: How to achieve high-quality promptable segmentation on sensor chips with <8MB SRAM and support only for quantization-friendly operators?

**Key Insight**: Completely abandon the Transformer in favor of a pure CNN architecture + implicit ROI prompts + SAM3 distillation + INT8 quantization, specifically designed for in-sensor deployment.

## Method

### Overall Architecture
Training: COCO data → extract ROI crops for each annotated instance (10% padding, square, 96×96) → SAM3 teacher generates soft masks for the same bbox (cached offline) → PicoSAM3 student learns to segment by looking only at the cropped RGB image → multi-loss distillation training. Inference: the user drags and selects an ROI on the IMX500 → hardware-based sensor cropping → INT8 model inference → return segmentation mask.

### Key Designs

1. **Implicit Prompt via Centered Cropping**:

    - Function: Implicitly encodes bbox prompts into the RGB input via cropping.
    - Mechanism: During training, square crops with 10% padding are extracted based on the bbox and resized to 96×96. The target object is always roughly centered, and the network learns to segment the main object near the center of the input. During inference, the same cropping is achieved using the hardware ROI function of the IMX500.
    - Design Motivation: The IMX500 only supports RGB input and cannot accept extra prompt tensors; thus, prompts can only be implicitly encoded through spatial cropping.

2. **Dense CNN U-Net**:

    - Function: A symmetric encoder-decoder pure CNN with 1.37M parameters.
    - Mechanism: Based on the U-Net structure of PicoSAM2 (depthwise separable convolutions, channels 48→96→160→256→320), introducing three improvements: (1) an enhanced bottleneck with dilated depthwise convolutions (dilation=2 to expand the receptive field); (2) an Efficient Channel Attention (ECA) block before the output head for adaptive feature recalibration; (3) a refinement head with depthwise convolutions to improve boundaries.
    - Design Motivation: Completely avoid Self-Attention operators from Transformers to ensure all operations are compatible with INT8 quantization and the limited operator set of the IMX500.

3. **SAM3 Knowledge Distillation**:

    - Function: Distills knowledge from a 1.2GB SAM3 teacher to a 5.26MB student.
    - Mechanism: Offline caches the teacher's soft masks on all COCO annotations → joint training with three losses: (1) teacher loss $\mathcal{L}_{teacher}$ = MSE + Dice (temperature scale $\tau=5$); (2) GT loss $\mathcal{L}_{gt}$ = BCE + Dice; (3) area preservation loss $\mathcal{L}_{area}$ (preventing the predicted mask area from collapsing below 40% of the GT).
    - Adaptive weights: The weights of the teacher and GT in the total loss are automatically adjusted by the teacher's confidence—relying on the teacher's soft label when confidence is high, and reverting to the GT when confidence is low.

4. **INT8 Post-Training Quantization**:

    - Function: 4× model compression (5.26MB → 1.31MB) with <0.2% accuracy loss.
    - Mechanism: Uses the Sony MCT toolchain for symmetric per-channel weight quantization + per-tensor activation quantization, with 10-batch calibration.
    - Design Motivation: Depthwise separable convolutions are naturally robust to quantization noise, eliminating the need for complex outlier suppression algorithms.

## Key Experimental Results

### Main Results (COCO / LVIS)

| Model | Params | Size | COCO mIoU | LVIS mIoU | IMX500 Latency |
|------|--------|------|-----------|-----------|------------|
| SAM-H | 635M | 2420MB | 53.6% | 60.5% | — |
| TinySAM | 9.7M | 37MB | 50.9% | 52.1% | — |
| EdgeSAM | 9.7M | 37MB | 48.0% | 53.7% | — |
| PicoSAM2 | 1.3M | 4.87MB | 51.9% | 44.9% | — |
| Q-PicoSAM2 | 1.3M | 1.22MB | 50.5% | 45.1% | 14.3ms |
| **PicoSAM3** | **1.37M** | **5.26MB** | **65.45%** | **64.01%** | — |
| **Q-PicoSAM3** | **1.37M** | **1.31MB** | **65.34%** | **63.98%** | **11.82ms** |

- PicoSAM3 has only 1/460 of the parameters of SAM-H, yet outperforms it in COCO mIoU by +11.85%.
- Compared to PicoSAM2, COCO improves by +13.55% and LVIS improves by +19.11%.
- INT8 quantization is almost lossless (-0.11% mIoU), achieving approximately 84 FPS on the IMX500.

### Ablation Study

| Model | Distillation | ROI | COCO mIoU | LVIS mIoU |
|------|--------------|-----|-----------|-----------|
| PicoSAM2 | ✗ | ✗ | 53.00% | 41.40% |
| PicoSAM2 | SAM2 | ✗ | 51.93% | 44.88% |
| PicoSAM2 | SAM2 | ✓ | 63.11% | 61.80% |
| PicoSAM2 | SAM3 | ✓ | 63.51% | 62.31% |
| PicoSAM3 | SAM3 | ✓ | 65.45% | 64.01% |

- ROI cropping is the most critical factor for improvement (mAP +10.56%), confirming that spatial focus is crucial for low-resolution models.
- SAM3 distillation outperforms SAM2 (+0.4% mIoU), and the architectural improvements of PicoSAM3 yield an additional +1.9%.

### Key Findings
- Large models (SAM2.1 Large) experience performance collapse on 96×96 inputs (~25% mAP) due to resolution mismatch.
- The activation distribution of the pure CNN architecture is compact and near-Gaussian, making it naturally friendly to INT8 quantization.
- ROI cropping + centered implicit prompting perfectly interfaces with the hardware ROI function of the IMX500.

## Highlights & Insights
- **Sensor-grade SAM**: Achieving promptable segmentation inside a CMOS sensor for the first time with an edge latency of 11.82ms (~84 FPS), showing high practical value.
- **Counter-intuitive finding**: The 1.3M parameter model outperforms the 635M SAM-H on 96×96 inputs, indicating that model-resolution matching is more important than absolute model size.
- **Implicit prompt design**: Prompting functionality is achieved through cropping without requiring an additional network or input channels, elegantly bypassing hardware limitations.

## Limitations & Future Work
- The 96×96 input resolution limits the fine boundary quality, which may be insufficient for large objects or complex shapes.
- Implicit ROI prompting only supports rectangular box regions, lacking support for prompt types like points or text.
- Validated only on COCO/LVIS, lacking transfer experiments to other domains like medical or remote sensing.
- Trained for only 1 epoch on COCO; longer training could potentially yield further improvements.

## Related Work & Insights
- **vs TinySAM/EdgeSAM**: 7× fewer parameters, 14+% higher mIoU, showing a substantial gap.
- **vs PicoSAM2**: Architectural improvements (ECA + dilated bottleneck) + SAM3 distillation + ROI prompting collectively contribute to a +13.55% improvement.
- **vs FastSAM**: +13.85% mIoU, with the model being only 1.9% of the size.

## Rating
- Novelty: ⭐⭐⭐⭐ In-sensor segmentation is a novel and practical direction, with an elegantly designed implicit ROI prompt.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation studies (distillation/ROI/architecture/quantization), validated with real hardware deployment.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with a thorough introduction to the hardware background.
- Value: ⭐⭐⭐⭐⭐ High practical value for extreme edge deployment; the method is complete and reproducible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Golden Cudgel Network for Real-Time Semantic Segmentation](golden_cudgel_network_for_real-time_semantic_segmentation.md)
- [\[CVPR 2025\] SmartEraser: Remove Anything from Images using Masked-Region Guidance](smarteraser_remove_anything_from_images_using_masked-region_guidance.md)
- [\[CVPR 2025\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)
- [\[ICLR 2026\] QPrompt-R1: Real-Time Reasoning for Domain-Generalized Semantic Segmentation via Group-Relative Query Alignment](../../ICLR2026/segmentation/qprompt-r1_real-time_reasoning_for_domain-generalized_semantic_segmentation_via_.md)
- [\[CVPR 2025\] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)

</div>

<!-- RELATED:END -->
