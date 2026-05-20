---
title: >-
  [Paper Note] Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy
description: >-
  [ICML 2026][Multimodal VLM][CLIP] Through 700,000 experimental runs covering 16 quantization methods × 10 VLMs × multiple reliability metrics…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "CLIP"
  - "W8A8"
  - "Calibration"
  - "OOD"
  - "Spectral Filtering"
date: 2026-05-08
content_hash: e254e9e58d534530
---

# Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy

**Conference**: ICML 2026  
**arXiv**: [2509.21173](https://arxiv.org/abs/2509.21173)  
**Code**: None  
**Area**: Multimodal VLM / Model Quantization / Reliability Evaluation  
**Keywords**: CLIP, W8A8, Calibration, OOD, Spectral Filtering

## TL;DR
Through 700,000 experimental runs covering 16 quantization methods × 10 VLMs × multiple reliability metrics, this work finds that quantization is not merely a destructive process—it suppresses high-rank, low-variance spectral components, thereby improving calibration, OOD detection, and noise robustness, but also amplifies reliance on covariate shift and spurious correlations.

## Background & Motivation

**Background**: VLMs such as CLIP have become the de facto standard for zero-shot classification and OOD detection, with mature benchmarks like OpenOOD and ImageNet-X for reliability evaluation. Meanwhile, quantization (PTQ/QAT) is standard for deploying VLMs on edge devices. These two fields have developed in parallel with little intersection.

**Limitations of Prior Work**: Existing quantization literature focuses almost exclusively on top-1 accuracy, remaining nearly silent on calibration, OOD, covariate robustness, and spurious correlations—metrics that become critical post-deployment. The community assumes "quantization noise = inevitable sacrifice," but this has not been systematically falsified.

**Key Challenge**: Reliability research only considers FP32 models, while quantization research only considers accuracy. When compressed models are deployed in safety-critical scenarios like autonomous driving or healthcare, whether reliability properties persist, disappear, or are enhanced remains a blind spot.

**Goal**: Systematically map the "reliability landscape of VLM quantization" across five dimensions: (1) robustness to quantization noise; (2) calibration and uncertainty; (3) OOD detection; (4) covariate shift robustness; (5) spurious correlation bias.

**Key Insight**: The authors reconceptualize quantization as a "non-uniform spectral domain filter"—while numerically it appears as uniform discretization, due to the large variance differences among SVD components, discretization barely affects low-rank, high-variance components but drowns high-rank, low-variance components in quantization noise.

**Core Idea**: Large-scale empirical study and SVD spectral analysis reveal a dual mechanism of "passive spectral filtering + active subspace concentration," and rotation-based quantization (QuaRot+LSQ) preserves mid-frequency spectral components, which is key to avoiding amplification of spurious correlations.

## Method

### Overall Architecture
Experiments cover 10 VLM architectures (CLIP ViT, CLIP ConvNeXt, SigLIP, ALIGN, CoCa), two quantization scopes (vision only vs. vision+text), 16 quantization methods (8 PTQ + 8 QAT variants), and bit-widths including W8A8, W6A8, W4A8. Over 8,000 quantized models and 700,000+ evaluation runs. All quantization uses 1,000 image-caption pairs for calibration. Logit Scale Tuning is uniformly applied as post-processing. Evaluation spans OpenOOD series, ImageNet-A/R/V2/Sketch, CIFAR-10-C, and CounterAnimal. SVD analysis is then conducted to reveal underlying mechanisms.

### Key Designs

1. **Unified Evaluation Protocol Across 5 Reliability Dimensions**:

    - Function: Uses a unified set of formulas to characterize "the relative impact of quantization on each reliability dimension," avoiding contradictions from disparate metrics across papers.
    - Mechanism: For each metric, defines relative change $\delta(\mathcal{D}) = \frac{A(f,\mathcal{D}) - A(q,\mathcal{D})}{A(f,\mathcal{D})}$; for OOD, uses AUROC relative change $\delta_{\text{OOD}}$; for spurious correlations, introduces Relative Spurious Gap $\text{RSG}(m) = \frac{A(m, \mathcal{D}_N) - A(m, \mathcal{D}_C)}{A(m, \mathcal{D}_N)}$, as well as quantization-induced $\Delta\text{RSG}$ and Added Vulnerability $\text{Vuln}_{\text{add}} = \delta_C - \delta_N$.
    - Design Motivation: Decouples "absolute accuracy change" from "spurious correlation amplification," preventing conflation of simple accuracy drop with bias amplification—the latter being the true ethical risk signal.

2. **Logit Scale Tuning to Repair Quantization Damage**:

    - Function: Without modifying the backbone, recalibrates logit temperature using proxy data to rescue over/under-confidence caused by quantization.
    - Mechanism: Treats CLIP's logit scale as a temperature parameter, optimized separately on the calibration set; visualizes "trajectory reliability diagram" to track each confidence bin's movement from FP32 → QAT → Logit Tuning.
    - Design Motivation: The authors find that quantizing the text encoder causes a mismatch between "the inner product of two independently quantized manifolds" and the pretrained logit scale, leading to ECE surging by +98%; scalar recalibration can reduce full quantized model ECE from 6.9% to 1.1% without retraining, outperforming FP32.

3. **SVD Spectral Analysis Reveals "Low-Pass Filtering + Principal Component Concentration" Mechanism**:

    - Function: Quantitatively explains why quantization simultaneously improves noise robustness but worsens semantic robustness and spurious correlation.
    - Mechanism: (a) Projects quantized features onto FP32 SVD basis, finding SQNR monotonically decreases with rank—fixed quantization step size first eliminates low-variance components, equivalent to low-pass filtering; (b) Recomputes SVD on quantized features, finding QAT achieves higher accuracy than FP32 in Rank 0-8 subspace, indicating discriminative information is compressed into the most stable principal components; (c) However, accuracy drops significantly for Rank 64+, corresponding to loss of fine-grained semantics.
    - Design Motivation: Unifies the seemingly contradictory phenomena of "calibration ↑ + spurious ↑" under a single mechanism: coarse-grained information is reinforced, fine-grained information is smoothed out. Rotation-based quantization, by aligning activations with the quantization grid, alleviates premature attenuation of mid-frequency components.

### Loss & Training
QAT uses LSQ (Esser et al. 2020) with two distillation mechanisms (contrastive-only and contrastive + feature MSE), all using 1,000 images from CC3M / YFCC / SBU as proxy calibration/fine-tuning sets. The study focuses on "relative differences among quantization method families (PTQ / QAT / Rotation+LSQ)," not on achieving single-point SOTA.

## Key Experimental Results

### Main Results

| Dimension | Change After Quantization | Correlation with Quantization Method |
|-----------|--------------------------|--------------------------------------|
| Zero-shot accuracy | $\approx 40\%$ runs improve calibration | Strongly correlated with pretraining data quality |
| ECE (calibration) | Full quantization + Logit Tuning achieves 1.1% (FP32 = 6.9%) | QAT series generally outperform PTQ |
| OOD AUROC | Statistically significant improvement in some configs | ConvNeXt > Transformer |
| Synthetic robustness | Mean +8.9% | Generally improved (low-pass effect) |
| Semantic robustness | Generally decreased | High-frequency details are smoothed out |

### Ablation Study

| Bit-width | Method | $\Delta\text{RSG}$ (%) | $\text{Vuln}_{\text{add}}$ (%) |
|-----------|--------|------------------------|-------------------------------|
| W8A8 | Simple PTQ | $+2.6$ *** | $+3.0$ *** |
| W8A8 | QAT (Contr.) | $+1.6$ *** | $+1.9$ *** |
| W8A8 | Rot+LSQ | $\mathbf{-0.1}$ ns | $\mathbf{-0.2}$ ns |
| W4A8 | Simple PTQ | $+12.5$ *** | $+10.3$ *** |
| W4A8 | Rot+LSQ | $\mathbf{+4.0}$ *** | $\mathbf{+4.4}$ *** |

### Key Findings
- Pretraining data quality determines "whether quantization acts as a regularizer": VLMs trained on high-quality datasets like WIT/DFN see further ECE reduction after quantization; models trained on noisy datasets like LAION experience ECE surges of +49% post-quantization, as capacity is already saturated by noise, leaving no redundancy to absorb quantization noise.
- Quantization yields a relative robustness improvement of $+8.9\%$ on synthetic corruptions (Gaussian noise/defocus blur) across nearly all architectures—consistent with the "low-pass filter" explanation: models do not rely on high-frequency information, so adding noise/removing details has little effect.
- Quantization leads to accuracy drops on semantic shift tasks like ImageNet-A/R/Sketch, as these require fine-grained, high-frequency features to distinguish hard cases.
- At extremely low bit (W4A8), even Rotation+LSQ cannot mitigate spurious correlation amplification, indicating a hard lower bound for bit depth.
- ConvNeXt benefits more from quantization than Transformer, as it relies more on texture (high-frequency), and spectral filtering helps remove the source of overconfidence under OOD.

## Highlights & Insights
- The default assumption that "quantization is just a trade-off" is directly challenged by 700,000 runs—and an interpretable SVD-based mechanism is provided, rather than stopping at "an interesting phenomenon was observed."
- Introduces $\Delta\text{RSG}$ and $\text{Vuln}_{\text{add}}$ metrics, separating spurious correlation amplification from accuracy degradation, which can be directly reused in any "compression + fairness" research.
- The link between rotation-based quantization and spectral preservation provides concrete design principles for future "reliability-aware quantization methods": align quantization grid with activation principal axes → preserve mid-frequency → avoid bias amplification.

## Limitations & Future Work
- Focuses mainly on discriminative VLMs (CLIP series), not covering generative LVLMs like LLaVA / Qwen-VL, which involve additional considerations such as KV cache quantization.
- Calibration set uses only three text-image datasets (CC3M / YFCC / SBU), with no discussion of domain-specific calibration for industrial deployment.
- SVD analysis is conducted on the penultimate layer of the vision encoder, without drilling down to whether different layers exhibit differentiated spectral responses.
- Provides Rotation+LSQ as a mitigation, but does not propose a complete "reliability-aware quantization method," remaining an observational study.

## Related Work & Insights
- **vs AskariHemmat 2022 "QReg"**: They propose quantization as implicit regularization but only evaluate accuracy and domain generalization; this work extends the insight to calibration/OOD/spurious dimensions.
- **vs Tu et al. 2023 (CLIP Robustness Evaluation)**: Tu evaluates FP32 CLIP; this work extends to quantized models and incorporates spectral filtering as a new explanatory mechanism.
- **vs QuaRot / OutlierSuppression**: This work directly links the "outlier removal" of rotation methods to "preservation of mid-frequency spectral components," providing a representation-level reliability explanation rather than just accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematically map the reliability landscape of VLM quantization and propose the spectral filtering mechanism
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 700,000 runs across 10 models × 16 methods × multiple bit-widths
- Writing Quality: ⭐⭐⭐⭐ Mechanism mapping is clear, though some appendix details are excessive
- Value: ⭐⭐⭐⭐⭐ Provides new dimensions and toolsets for safety evaluation of VLMs in edge deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation](../../ICCV2025/multimodal_vlm/is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)
- [\[ICCV 2025\] DASH: Detection and Assessment of Systematic Hallucinations of VLMs](../../ICCV2025/multimodal_vlm/dash_detection_and_assessment_of_systematic_hallucinations_of_vlms.md)
- [\[CVPR 2026\] Linking Perception, Confidence and Accuracy in MLLMs](../../CVPR2026/multimodal_vlm/linking_perception_confidence_and_accuracy_in_mllms.md)
- [\[ICML 2026\] ScreenParse: Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision](screenparse_moving_beyond_sparse_grounding_with_complete_screen_parsing_supervis.md)
- [\[ACL 2026\] Beyond Screenshots: Evaluating VLMs' Understanding of UI Animations](../../ACL2026/multimodal_vlm/beyond_screenshots_evaluating_vlms_understanding_of_ui_animations.md)

</div>

<!-- RELATED:END -->
