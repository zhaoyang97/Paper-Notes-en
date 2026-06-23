---
title: >-
  [Paper Note] Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy
description: >-
  [ICML 2026][vlm_efficiency][CLIP] This study evaluates 16 quantization methods across 10 VLMs and multiple reliability metrics through 700,000 experiments. It finds that quantization is not a simple disruptor—it improves calibration, OOD detection, and noise robustness by suppressing high-rank low-variance spectral components, while simultaneously ampl
tags:
  - ICML 2026
  - vlm_efficiency
  - CLIP
  - W8A8
  - OOD
date: 2026-05-08
content_hash: d48ac6eea5168d5d
---
# Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy

**Conference**: ICML 2026  
**arXiv**: [2509.21173](https://arxiv.org/abs/2509.21173)  
**Code**: None  
**Area**: Multimodal VLM / Model Quantization / Reliability Evaluation  
**Keywords**: CLIP, W8A8, Calibration, OOD, Spectral Filtering

## TL;DR
This study evaluates 16 quantization methods across 10 VLMs and multiple reliability metrics through 700,000 experiments. It finds that quantization is not a simple disruptor—it improves calibration, OOD detection, and noise robustness by suppressing high-rank low-variance spectral components, while simultaneously amplifying reliance on covariate shifts and spurious correlations.

## Background & Motivation

**Background**: VLMs like CLIP have become de facto standards for zero-shot classification and OOD detection, with established benchmarks like OpenOOD and ImageNet-X. Quantization (PTQ/QAT) is the standard for deploying VLMs on edge devices. These two fields have progressed in parallel with little intersection.

**Limitations of Prior Work**: Existing quantization literature focuses almost exclusively on top-1 accuracy, remaining largely silent on metrics like calibration, OOD, covariate robustness, and spurious correlations—aspects that are "truly critical after deployment." The community typically assumes "quantization noise = inevitable sacrifice" without systematic verification.

**Key Challenge**: Reliability studies typically examine only FP32 models, while quantization studies focus on accuracy. Whether reliability properties persist, vanish, or strengthen when compressed models are deployed in safety-sensitive scenarios (e.g., autonomous driving, medical imaging) remains a major blind spot.

**Goal**: To systematically map the "Reliability Landscape of VLM Quantization" across five dimensions: (1) Robustness to quantization noise; (2) Calibration and uncertainty; (3) OOD detection; (4) Covariate shift robustness; (5) Spurious correlation bias.

**Key Insight**: Quantization is conceptualized as a "non-uniform spectral filter." Although uniform in the value domain, discretization has little effect on low-rank high-variance components due to variation differences across SVD components, while drowning high-rank low-variance components in quantization noise.

**Core Idea**: Large-scale empirical evidence combined with SVD spectral analysis reveals a dual mechanism of "passive spectral filtering + active subspace concentration." The study highlights that rotation-based quantization (QuaRot+LSQ) preserves mid-frequency components, which is crucial for mitigating the amplification of spurious correlations.

## Method

### Overall Architecture
The experiments cover 10 VLM architectures (CLIP ViT, CLIP ConvNeXt, SigLIP, ALIGN, CoCa), two quantization scopes (Vision-only vs. Vision+Text), and 16 quantization methods (8 PTQ + 8 QAT variants), with bit-widths including W8A8, W6A8, and W4A8. This total encompasses over 8,000 quantized models and 700,000+ evaluation runs. All models are calibrated using 1,000 image-caption pairs. Post-processing includes Logit Scale Tuning. Evaluation spans OpenOOD benchmarks, ImageNet-A/R/V2/Sketch, CIFAR-10-C, and CounterAnimal, followed by SVD analysis to reveal underlying mechanisms.

### Key Designs

**1. Unified Evaluation Protocol Across 5 Reliability Dimensions: Decoupling Accuracy Drops from Bias Amplification**

Quantization evaluation has traditionally focused on top-1 accuracy. Reliability dimensions like calibration, OOD, and spurious correlations are often measured independently without a comparable baseline. Specifically, "accuracy drop" and "spurious correlation amplification" are often conflated, despite the latter being a critical ethical risk signal. A unified relative change metric is defined: for general indicators, $\delta(\mathcal{D}) = \frac{A(f,\mathcal{D}) - A(q,\mathcal{D})}{A(f,\mathcal{D})}$ (where $f$ is full-precision and $q$ is quantized); for OOD, the relative change in AUROC $\delta_{\text{OOD}}$ is used. To isolate spurious correlations, the study introduces Relative Spurious Gap $\text{RSG}(m) = \frac{A(m, \mathcal{D}_N) - A(m, \mathcal{D}_C)}{A(m, \mathcal{D}_N)}$ and Added Vulnerability $\text{Vuln}_{\text{add}} = \delta_C - \delta_N$ to distinguish whether bias is exacerbated by quantization beyond simple accuracy loss.

**2. Logit Scale Tuning: Re-calibrating Logit Temperature to Repair Quantization Damage**

Quantizing the text encoder can cause the "inner product of two independently quantized manifolds" to mismatch the pre-trained logit scale, causing ECE to spike by up to +98%. By treating the CLIP logit scale as a temperature parameter and optimizing it on a proxy calibration set while keeping the backbone frozen, the ECE of fully quantized models can be reduced from 6.9% to 1.1%—even better than full precision. "Trajectory reliability diagrams" track the movement of each confidence bin from FP32 → QAT → Logit Tuning, visualizing how over/under-confidence is corrected. This zero-cost scalar recalibration proves that much of the calibration degradation in quantization is merely logit scale mismatch.

**3. SVD Spectral Analysis: Interpreting Quantization as a "Low-pass Filter + Principal Component Concentration" Mechanism**

Quantization presents a counter-intuitive phenomenon where it simultaneously improves noise robustness and worsens semantic robustness and spurious correlations. SVD analysis unifies this. First, quantized features are projected onto the FP32 SVD basis, revealing that SQNR decays monotonically with rank—fixed quantization steps primary eliminate low-variance components, equivalent to a low-pass filter. Second, re-performing SVD on quantized features shows that QAT maintains higher accuracy in the Rank 0–8 subspace than FP32, suggesting discriminative information is pushed into the most stable principal components, while Rank 64+ accuracy drops significantly as fine-grained semantic information is smoothed out. This explains the "calibration ↑ + spurious ↑" trade-off: coarse-grained information is reinforced while fine-grained details are eroded. Rotation-based quantization (QuaRot+LSQ) mitigates the premature decay of mid-frequency components by aligning activations with the quantization grid.

### Loss & Training
QAT utilizes LSQ (Esser et al. 2020) with two distillation mechanisms (contrastive-only and contrastive + feature MSE), all using 1,000 images from CC3M / YFCC / SBU as a proxy calibration/fine-tuning set. The focus is on the relative differences between quantization families (PTQ / QAT / Rotation+LSQ) rather than achieving a single-point SOTA.

## Key Experimental Results

### Main Results

| Dimension | Post-Quantization Change | Quantization Method Correlation |
|-----------|--------------------------|---------------------------------|
| Zero-shot accuracy | $\approx 40\%$ runs improved calibration | Strongly correlated with pre-training data quality |
| ECE (calibration) | Fully quantized + Logit Tuning reaches 1.1% (FP32 = 6.9%) | QAT series generally superior to PTQ |
| OOD AUROC | Statistically significant improvement in some configs | ConvNeXt > Transformer |
| Synthetic robustness | Average +8.9% | Universal gain (low-pass effect) |
| Semantic robustness | General decrease | High-frequency details are smoothed |

### Ablation Study

| Bit-width | Method | $\Delta\text{RSG}$ (%) | $\text{Vuln}_{\text{add}}$ (%) |
|-----------|-----------|-----------------------|--------------------------------|
| W8A8 | Simple PTQ | $+2.6$ *** | $+3.0$ *** |
| W8A8 | QAT (Contr.) | $+1.6$ *** | $+1.9$ *** |
| W8A8 | Rot+LSQ | $\mathbf{-0.1}$ ns | $\mathbf{-0.2}$ ns |
| W4A8 | Simple PTQ | $+12.5$ *** | $+10.3$ *** |
| W4A8 | Rot+LSQ | $\mathbf{+4.0}$ *** | $\mathbf{+4.4}$ *** |

### Key Findings
- **Pre-training data quality** determines if quantization acts as a regularizer: VLMs trained on high-quality datasets (WIT/DFN) show further ECE reduction after quantization. Conversely, models trained on noisy data (LAION) see ECE spike by +49%, as capacity is already consumed by noise with no redundancy to absorb quantization noise.
- Quantization brings a $+8.9\%$ relative improvement in **synthetic corruption** (Gaussian noise / defocus blur) across almost all architectures—consistent with the "low-pass filter" explanation.
- Performance drops on **semantic shifts** like ImageNet-A/R/Sketch because these tasks require fine-grained high-frequency features to distinguish hard examples.
- Under **extreme low-bit settings (W4A8)**, even Rotation+LSQ cannot prevent the amplification of spurious correlations, indicating a hard lower bound on bit depth.
- **ConvNeXt** benefits more from quantization than Transformer as it relies more heavily on texture (high frequency); spectral filtering helps eliminate sources of over-confidence during OOD.

## Highlights & Insights
- Challenges the assumption that quantization is "only a trade-off" through 700,000 runs and provides an interpretable SVD mechanism rather than just observations.
- Proposes $\Delta\text{RSG}$ and $\text{Vuln}_{\text{add}}$ metrics to decouple spurious correlation amplification from accuracy degradation, applicable to any "compression + fairness" research.
- Connects **rotation-based quantization** with spectral preservation, providing design principles for "reliability-aware quantization": align the quantization grid with activation principal axes → preserve mid-frequencies → prevent bias amplification.

## Limitations & Future Work
- Primarily focused on discriminative VLMs (CLIP family), excluding generative LVLMs like LLaVA or Qwen-VL, which involve additional considerations like KV cache quantization.
- The calibration set relies on only three data sources (CC3M / YFCC / SBU); domain-specific calibration in industrial deployments is not discussed.
- SVD analysis was performed on the penultimate layer of the vision encoder; layer-wise differences in spectral response were not investigated.
- While Rotation+LSQ is identified as a mitigation, a complete "reliability-aware quantization method" proposal is not provided, making this an observational study.

## Related Work & Insights
- **vs. AskariHemmat 2022 "QReg"**: While they suggested quantization as implicit regularization, they only measured accuracy and domain generalization. This work extends the insight to calibration, OOD, and spurious correlation.
- **vs. Tu et al. 2023 (CLIP Robustness Evaluation)**: Tu evaluated FP32 CLIP; this study extends it to quantized versions and adds spectral filtering as a new explanatory mechanism.
- **vs. QuaRot / OutlierSuppression**: This work maps rotation methods' "outlier elimination" directly to "preservation of mid-frequency spectral components," providing a representation-level reliability explanation for these methods beyond simple accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematically map the reliability landscape of VLM quantization with a spectral filtering mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 700,000 runs across 10 models, 16 methods, and multiple bit-widths.
- Writing Quality: ⭐⭐⭐⭐ Mechanism diagrams are clear, though some appendix details are extensive.
- Value: ⭐⭐⭐⭐⭐ Provides a new dimension and toolset for the safety evaluation of edge-deployed VLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs](gated_relational_alignment_via_confidence-based_distillation_for_efficient_vlms.md)
- [\[CVPR 2026\] Vision-Oriented Lightweight Neural Architecture Search with Budget-Adaptive Evaluation](../../CVPR2026/vlm_efficiency/vision-oriented_lightweight_neural_architecture_search_with_budget-adaptive_eval.md)
- [\[CVPR 2026\] ApET: Approximation-Error Guided Token Compression for Efficient VLMs](../../CVPR2026/vlm_efficiency/apet_approximation-error_guided_token_compression_for_efficient_vlms.md)
- [\[NeurIPS 2025\] Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](../../NeurIPS2025/vlm_efficiency/balanced_token_pruning_accelerating_vision_language_models_b.md)
- [\[NeurIPS 2025\] Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability](../../NeurIPS2025/vlm_efficiency/beyond_greedy_exits_improved_early_exit_decisions_for_risk_control_and_reliabili.md)

</div>

<!-- RELATED:END -->
