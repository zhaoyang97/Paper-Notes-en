---
title: >-
  [Paper Note] Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy
description: >-
  [ICML 2026][Multimodal VLM][CLIP] This paper executes 700,000 experiments across 16 quantization methods $\times$ 10 VLMs $\times$ multiple reliability metrics. It discovers that quantization is not a simple disruptor—it improves calibration, OOD detection, and noise robustness by suppressing high-rank, low-variance spectral components, while simultane
tags:
  - ICML 2026
  - Multimodal VLM
  - CLIP
  - W8A8
  - OOD
date: 2026-05-08
content_hash: ee2459d08362d4fb
---
# Less Precise Can Be More Reliable: A Systematic Evaluation of Quantization's Impact on VLMs Beyond Accuracy

**Conference**: ICML 2026  
**arXiv**: [2509.21173](https://arxiv.org/abs/2509.21173)  
**Code**: None  
**Area**: Multimodal VLM / Model Quantization / Reliability Evaluation  
**Keywords**: CLIP, W8A8, Calibration, OOD, Spectral Filtering

## TL;DR
This paper executes 700,000 experiments across 16 quantization methods $\times$ 10 VLMs $\times$ multiple reliability metrics. It discovers that quantization is not a simple disruptor—it improves calibration, OOD detection, and noise robustness by suppressing high-rank, low-variance spectral components, while simultaneously amplifying reliance on covariate shifts and spurious correlations.

## Background & Motivation

**Background**: VLMs such as CLIP have become de facto standards for zero-shot classification and OOD detection, supported by mature benchmarks like OpenOOD and ImageNet-X. Meanwhile, quantization (PTQ / QAT) is a standard practice for deploying VLMs on edge devices. These two fields have developed in parallel with minimal intersection.

**Limitations of Prior Work**: Existing quantization literature focuses almost exclusively on top-1 accuracy, remaining largely silent on metrics like calibration, OOD, covariate robustness, and spurious correlations—factors that are critical for safety once a model is deployed. The community generally assumes "quantization noise = inevitable sacrifice," but this has not been systematically falsified.

**Key Challenge**: Reliability research typically examines only FP32 models, while quantization research focuses solely on accuracy. When compressed models are deployed in safety-sensitive scenarios such as autonomous driving or healthcare, whether reliability attributes persist, disappear, or are enhanced remains a complete blind spot.

**Goal**: To systematically characterize the "reliability landscape of VLM quantization" across five dimensions: (1) robustness to quantization noise; (2) calibration and uncertainty; (3) OOD detection; (4) covariate shift robustness; and (5) spurious correlation bias.

**Key Insight**: The authors reconceptualize quantization as a "non-uniform spectral filter." Although it appears as uniform discretization in the numerical domain, the vast differences in variance across different SVD components mean that discretization has almost no effect on low-rank, high-variance components, while drowning high-rank, low-variance components in quantization noise.

**Core Idea**: Large-scale empirical evidence combined with SVD spectral analysis reveals a dual mechanism of "passive spectral filtering + active subspace concentration." The study highlights that rotation-based quantization (QuaRot+LSQ) preserves mid-frequency components, which is key to avoiding the amplification of spurious correlation.

## Method

### Overall Architecture
The experiments cover 10 VLM architectures (CLIP ViT, CLIP ConvNeXt, SigLIP, ALIGN, CoCa), two quantization scopes (vision-only vs. vision+text), and 16 quantization methods (8 PTQ + 8 QAT variants). Bit-widths include W8A8, W6A8, and W4A8. In total, 8,000+ quantized models were evaluated across 700,000+ runs. All quantization processes used 1,000 image-caption pairs for calibration. Logit Scale Tuning was applied as a uniform post-processing step. Evaluation covered the OpenOOD series, ImageNet-A/R/V2/Sketch, CIFAR-10-C, and CounterAnimal. SVD analysis was subsequently performed to reveal the underlying mechanisms.

### Key Designs

**1. Unified Evaluation Protocol: Decoupling "Accuracy Drop" from "Bias Amplification"**

Quantization evaluation has long focused solely on top-1 accuracy, while reliability dimensions like calibration, OOD, covariate shift, and spurious correlation were measured in isolation without a comparable scale. Specifically, "accuracy drop" and "amplification of spurious correlations" are often conflated, whereas the latter represents a genuine ethical risk signal. This work defines a unified relative change for each reliability dimension: for general metrics, $\delta(\mathcal{D}) = \frac{A(f,\mathcal{D}) - A(q,\mathcal{D})}{A(f,\mathcal{D})}$ (where $f$ is full-precision and $q$ is quantized); for OOD, the relative change in AUROC $\delta_{\text{OOD}}$ is used. Spurious correlation is handled by introducing a Relative Spurious Gap $\text{RSG}(m) = \frac{A(m, \mathcal{D}_N) - A(m, \mathcal{D}_C)}{A(m, \mathcal{D}_N)}$, followed by calculating $\Delta\text{RSG}$ and Added Vulnerability $\text{Vuln}_{\text{add}} = \delta_C - \delta_N$ to isolate whether the bias is specifically amplified by quantization rather than just a result of overall performance degradation.

**2. Logit Scale Tuning: Recalibrating Logit Temperature without Backpropagation**

Quantizing the text encoder causes the inner product of two independent quantized manifolds to mismatch with the pre-trained logit scale, which can cause ECE to spike by +98%. By treating the CLIP logit scale as a temperature parameter and optimizing it alone on a proxy calibration set while keeping the backbone frozen, the ECE of a fully quantized model can be reduced from 6.9% to 1.1%—even better than the full-precision version. Visualization through "trajectory reliability diagrams" tracks the movement of each confidence bin from FP32 $\to$ QAT $\to$ Logit Tuning, intuitively showing how over- or under-confidence is corrected. This nearly zero-cost scalar recalibration proves that a significant portion of quantization-induced calibration degradation is simply logit scale mismatch.

**3. SVD Spectral Analysis: Quantization as a "Low-pass Filtering + Principal Component Concentration" Mechanism**

A counter-intuitive phenomenon of quantization is that it simultaneously improves noise robustness while worsening semantic robustness and spurious correlation. SVD analysis unifies this under one mechanism. First, projecting quantized features onto the FP32 SVD basis reveals that SQNR decays monotonically with rank—fixed quantization steps primary eliminate low-variance components, acting as a low-pass filter. Second, performing SVD on quantized features shows that QAT achieves higher accuracy in the Rank 0–8 subspace than FP32, suggesting discriminative information is compressed into the most stable principal components. However, accuracy drops significantly for Rank 64+, as fine-grained semantic information is smoothed out. This explains the "calibration $\uparrow$ + spurious $\uparrow$" observation: coarse-grained information is reinforced while fine-grained details are lost. Rotation-based quantization (QuaRot+LSQ), by aligning activations with the quantization grid, mitigates the premature decay of mid-frequency components, which is why it avoids amplifying $\Delta\text{RSG}$.

### Loss & Training
QAT uses LSQ (Esser et al. 2020) with two distillation mechanisms (contrastive-only and contrastive + feature MSE), all utilizing 1,000 images from CC3M / YFCC / SBU as a proxy calibration/fine-tuning set. The research focuses on the "relative differences between quantization families (PTQ / QAT / Rotation+LSQ)" rather than pushing a specific single-point SOTA.

## Key Experimental Results

### Main Results

| Dimension | Change After Quantization | Correlation with Quantization Method |
|------|-----------|----------------|
| Zero-shot accuracy | $\approx 40\%$ runs improved calibration | Strongly correlated with pre-training data quality |
| ECE (calibration) | Fully quantized + Logit Tuning reaches 1.1% (FP32 = 6.9%) | QAT family generally outperforms PTQ |
| OOD AUROC | Statistically significant improvement in some configs | ConvNeXt > Transformer |
| Synthetic robustness | Average +8.9% | Universal improvement (low-pass effect) |
| Semantic robustness | Universal decline | Smoothing of high-frequency details |

### Ablation Study

| Bit-width | Method | $\Delta\text{RSG}$ (%) | $\text{Vuln}_{\text{add}}$ (%) |
|-----------|------|-----------------------|--------------------------------|
| W8A8 | Simple PTQ | $+2.6$ *** | $+3.0$ *** |
| W8A8 | QAT (Contr.) | $+1.6$ *** | $+1.9$ *** |
| W8A8 | Rot+LSQ | $\mathbf{-0.1}$ ns | $\mathbf{-0.2}$ ns |
| W4A8 | Simple PTQ | $+12.5$ *** | $+10.3$ *** |
| W4A8 | Rot+LSQ | $\mathbf{+4.0}$ *** | $\mathbf{+4.4}$ *** |

### Key Findings
- Pre-training data quality dictates whether quantization can act as a regularizer: VLMs trained on high-quality datasets (WIT / DFN) show further reduced ECE after quantization. In contrast, models trained on noisy data (LAION) see ECE spike by +49% after quantization because the capacity is already saturated by noise, leaving no redundancy to absorb quantization noise.
- Quantization brings a $+8.9\%$ relative robustness improvement across almost all architectures for synthetic corruption (Gaussian noise / defocus blur). This aligns with the "low-pass filter" explanation: models do not rely on high-frequency noise, so removing fine details helps.
- Quantization leads to performance drops on semantic shifts like ImageNet-A/R/Sketch, as these tasks require fine-grained, high-frequency features to distinguish hard examples.
- At extreme low bits (W4A8), even Rotation+LSQ cannot prevent the amplification of spurious correlations, indicating a hard lower bound for bit depth.
- ConvNeXt benefits more from quantization than Transformers because it relies more heavily on texture (high frequency); spectral filtering helps it remove sources of over-confidence during OOD.

## Highlights & Insights
- Directly challenges the default assumption that "quantization is just a trade-off" using 700,000 runs and provides an explainable SVD mechanism rather than just observing an "interesting phenomenon."
- Proposes $\Delta\text{RSG}$ and $\text{Vuln}_{\text{add}}$ metrics to isolate spurious correlation amplification from accuracy degradation, which can be reused in any "compression + fairness" research.
- The link between rotation-based quantization and spectral preservation provides design principles for "reliability-aware quantization": aligning the quantization grid with activation principal axes $\to$ preserving mid-frequencies $\to$ avoiding bias amplification.

## Limitations & Future Work
- Primarily focuses on discriminative VLMs (CLIP family) and does not cover generative LVLMs like LLaVA or Qwen-VL, which involve extra considerations like KV cache quantization.
- The calibration set is limited to three data sources (CC3M / YFCC / SBU); domain-specific calibration in industrial deployment is not discussed.
- SVD analysis is performed on the penultimate layer of the vision encoder; whether differentiated spectral responses exist across different layers is not explored.
- While it identifies Rotation+LSQ as a mitigation, it does not propose a complete "reliability-aware quantization method," remaining an observational study.

## Related Work & Insights
- **vs AskariHemmat 2022 "QReg"**: They proposed quantization as implicit regularization but only tested accuracy and domain generalization; this paper extends that insight to calibration, OOD, and spurious correlation.
- **vs Tu et al. 2023 (CLIP Robustness Evaluation)**: Tu evaluated FP32 CLIP; this work is an expansion for its quantized versions, adding the spectral filtering mechanism to the explanatory framework.
- **vs QuaRot / OutlierSuppression**: This paper maps the "outlier elimination" of rotation methods directly to "preserving mid-frequency components," providing a representation-level reliability explanation for these methods rather than just an accuracy-based one.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic mapping of the reliability landscape for VLM quantization with a spectral mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 700,000 runs across 10 models $\times$ 16 methods $\times$ multiple bit-widths.
- Writing Quality: ⭐⭐⭐⭐ Mechanism diagrams are clear, though some appendix details are dense.
- Value: ⭐⭐⭐⭐⭐ Provides a new dimension and toolkit for the safety evaluation of edge-deployed VLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Learning More from Less: Exploiting Counterfactuals for Data-Efficient Chart Understanding](../../ACL2026/multimodal_vlm/learning_more_from_less_exploiting_counterfactuals_for_data-efficient_chart_unde.md)
- [\[CVPR 2026\] Select Less, Reason More: Prioritizing Evidence Purity for Video Reasoning](../../CVPR2026/multimodal_vlm/select_less_reason_more_prioritizing_evidence_purity_for_video_reasoning.md)
- [\[ICCV 2025\] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation](../../ICCV2025/multimodal_vlm/is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)
- [\[CVPR 2026\] Beyond Graph Model: Reliable VLM Fine-Tuning via Random Graph Adapter](../../CVPR2026/multimodal_vlm/beyond_graph_model_reliable_vlm_fine-tuning_via_random_graph_adapter.md)
- [\[ICML 2026\] ECG-R1: Protocol-Guided and Modality-Agnostic MLLM for Reliable ECG Interpretation](ecg-r1_protocol-guided_and_modality-agnostic_mllm_for_reliable_ecg_interpretatio.md)

</div>

<!-- RELATED:END -->
