---
title: >-
  [Paper Note] Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy
description: >-
  [ICML 2026][Multimodal VLM][CLIP] This work conducts 700,000 experiments covering 16 quantization methods across 10 VLMs and multiple reliability metrics. It discovers that quantization is not a simple disruptor—it suppr…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "CLIP"
  - "W8A8"
  - "Calibration"
  - "OOD"
  - "Spectral Filtering"
date: 2026-05-08
content_hash: 3d7e5d6c4cebda3e
---

# Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy

**Conference**: ICML 2026  
**arXiv**: [2509.21173](https://arxiv.org/abs/2509.21173)  
**Code**: None  
**Area**: Multi-modal VLM / Model Quantization / Reliability Evaluation  
**Keywords**: CLIP, W8A8, Calibration, OOD, Spectral Filtering

## TL;DR
This work conducts 700,000 experiments covering 16 quantization methods across 10 VLMs and multiple reliability metrics. It discovers that quantization is not a simple disruptor—it suppresses high-rank, low-variance spectral components, simultaneously improving calibration, OOD detection, and noise robustness, while amplifying dependencies on covariate shifts and spurious correlations.

## Background & Motivation

**Background**: VLMs such as CLIP have become the de facto standard for zero-shot classification and OOD detection. Reliability evaluation is supported by mature benchmarks like OpenOOD and ImageNet-X. Meanwhile, quantization (PTQ / QAT) is a standard requirement for deploying VLMs to edge devices. These two fields have developed in parallel with limited intersection.

**Limitations of Prior Work**: Existing quantization literature focuses almost exclusively on top-1 accuracy, remaining largely silent on metrics like calibration, OOD detection, covariate robustness, and spurious correlations—aspects that are "critical once deployed." The community defaults to the assumption that "quantization noise = inevitable sacrifice," but this has not been systematically challenged.

**Key Challenge**: Reliability research typically examines FP32 models, while quantization research focuses on accuracy. When compressed models are deployed in safety-sensitive scenarios like autonomous driving or healthcare, whether reliability properties persist, vanish, or strengthen remains a significant blind spot.

**Goal**: Systematically characterize the "reliability landscape of VLM quantization" across five dimensions: (1) robustness to quantization noise; (2) calibration and uncertainty; (3) OOD detection; (4) covariate shift robustness; and (5) spurious correlation bias.

**Key Insight**: The authors reconceptualize quantization as a "non-uniform spectral domain filter." While it appears as uniform discretization in the numerical domain, the vast differences in variance across SVD components mean that discretization has negligible impact on low-rank, high-variance components but drowns high-rank, low-variance components in quantization noise.

**Core Idea**: Use large-scale empirical evidence and SVD spectral analysis to reveal the dual mechanism of "passive spectral filtering + active subspace concentration." It is noted that rotation-based quantization (QuaRot+LSQ) preserves mid-frequency spectral components, which is key to avoiding the amplification of spurious correlations.

## Method

### Overall Architecture
The experiments cover 10 VLM architectures (CLIP ViT, CLIP ConvNeXt, SigLIP, ALIGN, CoCa), two quantization scopes (vision-only vs. vision+text), and 16 quantization methods (8 PTQ + 8 QAT variants). Bit-widths include W8A8, W6A8, and W4A8. In total, 8,000+ quantized models and 700,000+ evaluation runs were conducted. All quantization used 1,000 image-caption pairs for calibration. Logit Scale Tuning was applied during post-processing. Evaluation covered the OpenOOD series, ImageNet-A/R/V2/Sketch, CIFAR-10-C, and CounterAnimal. SVD analysis was then performed to reveal the underlying mechanisms.

### Key Designs

1. **Unified Evaluation Protocol Across 5 Reliability Dimensions**:
    - Function: Characterizes the "relative impact of quantization on each reliability dimension" using a set of unified formulas to avoid conflicting results from different reporting standards.
    - Mechanism: Defines relative change for each metric as $\delta(\mathcal{D}) = \frac{A(f,\mathcal{D}) - A(q,\mathcal{D})}{A(f,\mathcal{D})}$. For OOD, the relative change in AUROC $\delta_{\text{OOD}}$ is used. For spurious correlations, the Relative Spurious Gap $\text{RSG}(m) = \frac{A(m, \mathcal{D}_N) - A(m, \mathcal{D}_C)}{A(m, \mathcal{D}_N)}$ is introduced, along with quantization-induced $\Delta\text{RSG}$ and Added Vulnerability $\text{Vuln}_{\text{add}} = \delta_C - \delta_N$.
    - Design Motivation: Decouple "absolute accuracy changes" from "spurious correlation amplification" to avoid conflating simple drop in performance with bias amplification—the latter being the true ethical risk signal.

2. **Logit Scale Tuning to Repair Quantization Degradation**:
    - Function: Recalibrates logit temperature using only proxy data without modifying the backbone to rescue quantization-induced over/under-confidence.
    - Mechanism: Treats the CLIP logit scale as a temperature parameter optimized separately on a calibration set. A "trajectory reliability diagram" tracks the movement of conference bins from FP32 $\rightarrow$ QAT $\rightarrow$ Logit Tuning.
    - Design Motivation: Ours found that quantizing the text encoder causes a mismatch between the "inner product of two independently quantized manifolds" and the pre-trained logit scale, leading to an ECE surge of +98%. Scalar recalibration can pull the ECE of a fully quantized model from 6.9% to 1.1% without retraining, outperforming the FP32 model.

3. **SVD Spectral Analysis Revealing "Low-Pass Filter + Principal Component Concentration"**:
    - Function: Quantitatively explains why quantization can simultaneously improve noise robustness and degrade semantic robustness and spurious correlation.
    - Mechanism: (a) Projecting quantized features onto FP32 SVD bases reveals that SQNR decays monotonically with rank—fixed quantization steps consume low-variance components first, equivalent to low-pass filtering. (b) Re-performing SVD on quantized features shows that QAT accuracy in the Rank 0-8 subspace is higher than FP32, indicating discriminative information is compressed into the most stable principal components. (c) Accuracy significantly drops for Rank 64+, corresponding to the loss of fine-grained semantics.
    - Design Motivation: Unify the seemingly contradictory phenomenon of "calibration $\uparrow$ + spurious $\uparrow$" under one mechanism: coarse-grained information is reinforced while fine-grained information is smoothed out. Rotation-based quantization mitigates premature decay of mid-frequency components by aligning activations with the quantization grid.

### Loss & Training
QAT uses LSQ (Esser et al. 2020) with two distillation mechanisms (contrastive-only and contrastive + feature MSE), all utilizing 1,000 images from CC3M / YFCC / SBU as a proxy calibration/fine-tuning set. The focus is on the "relative differences between quantization method families (PTQ / QAT / Rotation+LSQ)" rather than a single SOTA point.

## Key Experimental Results

### Main Results

| Dimension | Change Post-Quantization | Correlation with Quantization Method |
|------|-----------|----------------|
| Zero-shot accuracy | $\approx 40\%$ runs improved calibration | Strongly correlated with pre-training data quality |
| ECE (calibration) | Reaches 1.1% with Full Quant + Logit Tuning (FP32 = 6.9%) | QAT series generally superior to PTQ |
| OOD AUROC | Statistically significant improvement in some configs | ConvNeXt > Transformer |
| Synthetic robustness | Average +8.9% | Universal improvement (Low-pass effect) |
| Semantic robustness | General decline | High-frequency details are smoothed out |

### Ablation Study

| Bit-width | Method | $\Delta\text{RSG}$ (%) | $\text{Vuln}_{\text{add}}$ (%) |
|-----------|------|-----------------------|--------------------------------|
| W8A8 | Simple PTQ | $+2.6$ *** | $+3.0$ *** |
| W8A8 | QAT (Contr.) | $+1.6$ *** | $+1.9$ *** |
| W8A8 | Rot+LSQ | $\mathbf{-0.1}$ ns | $\mathbf{-0.2}$ ns |
| W4A8 | Simple PTQ | $+12.5$ *** | $+10.3$ *** |
| W4A8 | Rot+LSQ | $\mathbf{+4.0}$ *** | $\mathbf{+4.4}$ *** |

### Key Findings
- Pre-training data quality determines "whether quantization acts as a regularizer": VLMs trained on high-quality datasets like WIT / DFN show further ECE reduction after quantization. Models trained on noisy data like LAION see ECE surge by +49% because capacity is saturated by noise, leaving no redundancy to absorb quantization noise.
- Quantization brings a relative $+8.9\%$ robustness improvement to synthetic corruption (Gaussian noise / defocus blur) across almost all architectures. This is consistent with the "low-pass filter" explanation: models do not rely on high-frequency info, so noise/detail removal is inconsequential.
- Quantization causes drops on semantic shifts like ImageNet-A/R/Sketch, as these tasks require fine-grained high-frequency features to distinguish hard examples.
- Under extreme low-bit settings (W4A8), even Rotation+LSQ cannot recover the amplified spurious correlations, suggesting a hard lower bound for bit depth.
- ConvNeXt benefits more from quantization than Transformer because it relies more heavily on texture (high frequency); spectral filtering helps remove the source of overconfidence during OOD.

## Highlights & Insights
- Directly challenges the default assumption that "quantization is just a trade-off" using 700,000 runs—and provides an interpretable mechanism via SVD rather than stopping at "an interesting observation."
- Proposes $\Delta\text{RSG}$ and $\text{Vuln}_{\text{add}}$ metrics to isolate spurious correlation amplification from accuracy degradation, which can be reused in any "compression + fairness" research.
- The link between rotation-based quantization and spectral preservation provides specific design principles for future "reliability-aware quantization": align the quantization grid with activation principal axes $\rightarrow$ preserve mid-frequencies $\rightarrow$ do not amplify bias.

## Limitations & Future Work
- Primarily focuses on discriminative VLMs (CLIP family), not covering generative LVLMs like LLaVA / Qwen-VL; these models involve additional considerations like KV cache quantization.
- Calibration sets only use three text-image pair sources (CC3M / YFCC / SBU); domain-specific calibration in industrial deployment is not discussed.
- SVD analysis was performed on the penultimate layer of the vision encoder; layer-wise differences in spectral response were not explored.
- Provides Rotation+LSQ as a mitigation but does not propose a complete "reliability-aware quantization method," remaining an observational study.

## Related Work & Insights
- **vs. AskariHemmat 2022 "QReg"**: They proposed quantization as implicit regularization but only tested accuracy and domain generalization; Ours extends this insight to calibration / OOD / spurious dimensions.
- **vs. Tu et al. 2023 (CLIP Robustness Evaluation)**: Tu evaluated FP32 CLIP; Ours is an extension to its quantized version, adding spectral filtering as a new mechanism to the explanatory framework.
- **vs. QuaRot / OutlierSuppression**: Ours maps the "outlier removal" of rotation methods directly to "preserving mid-frequency spectral components," providing a representation-level reliability explanation for such methods beyond just accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic characterization of the VLM quantization reliability landscape with a spectral filtering mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 700,000 runs across 10 models, 16 methods, and multiple bit-widths.
- Writing Quality: ⭐⭐⭐⭐ Clear mechanism mapping, though some appendix details are extensive.
- Value: ⭐⭐⭐⭐⭐ Provides a new dimension and toolkit for safety assessments of edge-deployed VLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Learning More from Less: Exploiting Counterfactuals for Data-Efficient Chart Understanding](../../ACL2026/multimodal_vlm/learning_more_from_less_exploiting_counterfactuals_for_data-efficient_chart_unde.md)
- [\[ICCV 2025\] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation](../../ICCV2025/multimodal_vlm/is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)
- [\[ICCV 2025\] DASH: Detection and Assessment of Systematic Hallucinations of VLMs](../../ICCV2025/multimodal_vlm/dash_detection_and_assessment_of_systematic_hallucinations_of_vlms.md)
- [\[ICML 2026\] ECG-R1: Protocol-Guided and Modality-Agnostic MLLM for Reliable ECG Interpretation](ecg-r1_protocol-guided_and_modality-agnostic_mllm_for_reliable_ecg_interpretatio.md)
- [\[ACL 2026\] Beyond Screenshots: Evaluating VLMs' Understanding of UI Animations](../../ACL2026/multimodal_vlm/beyond_screenshots_evaluating_vlms_understanding_of_ui_animations.md)

</div>

<!-- RELATED:END -->
