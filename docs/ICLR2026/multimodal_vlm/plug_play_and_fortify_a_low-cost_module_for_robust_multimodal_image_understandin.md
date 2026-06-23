---
title: >-
  [Paper Note] Plug, Play, and Fortify: A Low-Cost Module for Robust Multimodal Image Understanding
description: >-
  [ICLR 2026][Multimodal VLM][Paper Note] Addressing the performance collapse of multimodal models when a modality is missing during inference, the authors discover that "modality preference" can be quantified in the frequency domain. They propose the Frequency Ratio Metric (FRM) and a plug-and-play, nearly zero-parameter Multimodal Weight Allocation Module (M
tags:
  - ICLR 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: fe2de98bed7f362e
---
# Plug, Play, and Fortify: A Low-Cost Module for Robust Multimodal Image Understanding

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=7KluEfmiXG](https://openreview.net/forum?id=7KluEfmiXG)  
**Code**: https://github.com/a6103121/MWAM  
**Area**: Multimodal VLM  
**Keywords**: Missing modality, Modality preference, Frequency domain analysis, Plug-and-play, Robust multimodal understanding

## TL;DR
Addressing the performance collapse of multimodal models when a modality is missing during inference, the authors discover that "modality preference" can be quantified in the frequency domain. They propose the Frequency Ratio Metric (FRM) and a plug-and-play, nearly zero-parameter Multimodal Weight Allocation Module (MWAM). By weighting the neglected modalities during training to balance the optimization, various CNN/ViT backbones become more robust under various missing modality combinations.

## Background & Motivation

**Background**: Multimodal visual understanding (segmentation, detection, classification) typically leverages complementary modalities like RGB, infrared, and depth to improve accuracy. However, most methods default to the assumption that "all modalities are present during inference." In reality, sensor failures, harsh environments, or privacy constraints can cause certain modalities to be missing. This has led to two types of responses: **imputation-based**, which reconstructs missing modality features from existing ones; and **imputation-free**, which projects all modalities into a modality-agnostic unified space for direct inference.

**Limitations of Prior Work**: Imputation-based methods require extra reconstruction modules, leading to high computational overhead and difficulty in deployment on resource-constrained devices. Imputation-free methods are lightweight but cannot recover lost information, still suffering from performance drops when modalities are missing. Crucially, the authors found via testing that unified space models are extremely sensitive to *which* modality is missing—on CASIA-SURF, the accuracy drops from 98.12% to 95.21% or lower when Depth is missing, while missing RGB has almost no effect. This indicates that models do not rely on each modality equally.

**Key Challenge**: The root of this vulnerability is **imbalanced learning during the training stage**. Multimodal models implicitly favor certain "dominant modalities" during optimization, allowing them to monopolize gradient updates and learn superior features while other modality encoders are neglected and under-optimized. Once the preferred modality is lost during inference, the model loses its primary pillar.

**Goal**: Split into two sub-problems: (1) How to identify and quantify the model's preference for each modality? (2) How to use this to re-balance optimization during training so that weak modalities are also well-learned?

**Key Insight**: Existing balancing methods measure modality contributions in the **spatial domain**, and experiments show they are far from reaching the performance upper bound. The authors noticed an overlooked information source—the **frequency domain**. Experiments with low-pass/high-pass filtering revealed that models primarily rely on **low-frequency information** for decision-making (training loss is higher and optimization more sustained when low frequencies are absent). Meanwhile, high frequencies provide non-negligible detailed contributions. Using Neural Tangent Kernel (NTK) theory, they proved that convergence is faster in the directions of large eigenvalues, which correspond to low-frequency functions; hence, modalities rich in low-frequency information naturally dominate during optimization.

**Core Idea**: Define a modality preference metric, FRM, in the frequency domain using the ratio of "low frequency to high frequency." Then, use a plug-and-play module to allocate weights to each modality inversely proportional to FRM, shifting the training optimization from favoring dominant modalities back to equilibrium, thereby enhancing robustness against missing modalities.

## Method

### Overall Architecture
MWAM (Multimodal Weight Allocation Module) is a bypass module attached to a host multimodal model. During training, it does not change the host structure but observes the input of each modality branch, calculates an FRM score in the frequency domain representing "how much this modality is preferred," and converts this score into weights to intervene in training. It is removed during inference, incurring zero additional cost. The pipeline is: each modality input → sliced into $p\times p$ non-overlapping patches and processed via DCT → taking the top-left $q\times q$ block as low frequency and the bottom-right $q\times q$ block as high frequency, reassembled into low/high-frequency feature maps based on patch position → calculate FRM → sent to an FRM memory bank for smoothing (handling anomalies/missing inputs) → mapped to modality weights $K_{mi}$ via a weight allocation function → intervene in training via "gradient editing" or "weighted loss." The paper sets $p=8, q=2$.

The design is built on a core principle: **modality preference can be observed and quantified in the frequency domain**—the foundation for the three component contributions (FRM metric, FRM memory bank, weight intervention).

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Modality Inputs<br/>RGB / Depth / IR ..."] --> B["DCT Patching<br/>Low + High Freq Aggregation"]
    B --> C["Frequency Ratio Metric FRM<br/>L1 Norm Ratio (Low/High)"]
    C --> D["FRM Memory Bank<br/>Historical EMA Smoothing + Anomaly Handling"]
    D --> E["Weight Allocation Function<br/>FRM → Modality Weights K"]
    E -->|Gradient Editing (Zero Param)| F["Train Optimization Intervention"]
    E -->|Weighted Loss (Lightweight Head)| F
```

### Key Designs

**1. Frequency domain reveals modality preference: Moving from spatial to spectral analysis**

This design addresses the limitation where existing balancing methods measure modality contributions in the spatial domain at the feature level, lacking a global and independent perspective. The authors observed that model decisions are dominated by low-frequency components—training loss is lower and optimization more persistent when low-pass filters preserve low frequencies, whereas models fed only high frequencies saturate in loss around 30 epochs. Theoretically, the convergence rate of wide neural networks along NTK eigenvector directions is proportional to $(1-\eta\lambda_i)$, where directions with large eigenvalues (corresponding to low-frequency functions) converge faster. Combined with the conclusion that "dominant modalities suppress weak modality gradients through shared global error signals," the core thesis is established: **models favor modalities rich in low-frequency information during optimization**. This anchors the abstract "modality preference" onto measurable spectral features.

**2. FRM: Quantifying modality preference with the ratio of low to high frequency**

Intuitively, one could use the total magnitude (L1 norm) of low-frequency components as a preference metric: $\mathrm{MP}(x_{mi})=\sum_{a}\sum_{b}|I^{low}_{mi}(a,b)|$. However, the authors found that high frequencies still provide discriminative cues. Thus, the metric is defined as the L1 norm of the **ratio of low-frequency to high-frequency components**:

$$\mathrm{FRM}(x_{mi})=\sum_{a=0}^{w-1}\sum_{b=0}^{h-1}\Big|\frac{I^{low}_{mi}(a,b)}{I^{high}_{mi}(w-1-a,\,h-1-b)+\sigma}\Big|$$

where $\sigma$ is a scaling factor. This design is clever: when high frequency is 0 at a location, $\sigma$ amplifies the low-frequency effect and avoids division by zero. When different modalities have similar low-frequency energy, the ratio structure **amplifies the FRM differences between modalities**, thereby widening the weight gap and enhancing the effectiveness.

**3. FRM Memory Bank: Moving average smoothing + anomaly handling**

FRM calculated per mini-batch can jitter, and training may encounter batches with missing/abnormal modalities. The FRM memory bank stabilizes the metric using an Exponential Moving Average (EMA):

$$\hat{F}^{j}_{mi}=\begin{cases}F^{j}_{mi}, & j=0\\ \omega\hat{F}^{j-1}_{mi}+(1-\omega)F^{j}_{mi}, & j=1,\dots,nt-n\end{cases}$$

$F^{j}_{mi}$ is the FRM of the $j$-th mini-batch, and $\hat F^{j}_{mi}$ is the output fusing history and current state; $\omega$ is the history weight (default 0.5). This filters per-batch noise and provides a fallback value during abnormal batches, ensuring stable weights.

**4. Weight Allocation + Training Intervention: Inverting FRM into weights to balance optimization**

The dynamic weight allocation function is defined as:

$$K^{j}_{mi}(x^{j}_{mi})=\alpha-\frac{\beta}{1+e^{-\lambda(T-\gamma)}}$$

where $\alpha,\beta,\lambda,\gamma$ are adjustable scaling factors, and $T$ is the ratio of that modality's FRM to the average FRM of all modalities in the mini-batch: $T=\mathrm{FRM}(x^{j}_{mi})\big/\big(\frac{1}{M}\sum_{c=1}^{M}\mathrm{FRM}(x^{j}_{mc})+\sigma\big)$. This sigmoid form ensures that modalities with higher FRM (more preferred) receive smaller weights, while weak modalities with lower FRM receive larger weights, **inversely** shifting the optimization focus. Intervention paths: (a) **Gradient Editing**, scaling gradients directly, **entirely parameter-free**; (b) **Weighted Loss**, weighting individual modality losses for models with auxiliary heads.

### Loss & Training
The basic form of MWAM introduces no new loss terms, using FRM-derived weights to scale modality branch gradients. When the host model calculates auxiliary losses per modality, Equation (7) is used to weight these losses by $K_{mi}$ before adding them to the main output loss. In segmentation experiments, scaling factors are set as $\alpha,\beta,\lambda,\gamma=1.5,1,6,0.7$.

## Key Experimental Results

The core evaluation metric, besides standard accuracy, is **PCR (Performance Collapse Rate, lower is better)** to characterize the degree of collapse when modalities are missing.

### Main Results

Brain tumor segmentation on BRATS2020 († denotes integration with MWAM, Average row):

| Host Method | Dice↑ | +MWAM Dice↑ | PCR↓ | +MWAM PCR↓ |
|----------|-------|-------------|------|------------|
| RFNet | 85.41 | 86.41 | 5.62 | 5.13 |
| mmFormer (ViT) | 85.64 | 86.30 | 5.21 | 4.71 |
| GSS | 86.41 | 87.56 | 5.30 | 4.44 |

Semantic segmentation on NYU-Depth V2 (Average row):

| Host Method | MIoU↑ | +MWAM MIoU↑ | PCR↓ | +MWAM PCR↓ |
|----------|-------|-------------|------|------------|
| ESANet-MD | 42.53 | 44.47 | 15.71 | 13.31 |
| MMANet | 43.47 | 45.81 | 14.87 | 12.66 |

MWAM improves Dice/MIoU and reduces PCR for all host methods. After integration, the average Dice of older methods like RFNet and mmFormer becomes comparable to SOTA methods like LS3M, but with superior PCR. GSS+MWAM exceeds LS3M in both Dice and PCR. This applies across CNN-based and ViT-based backbones.

### Ablation Study

| Configuration | Key Observation | Description |
|------|---------|------|
| L1 Low-freq Magnitude (Eq 3) | Sub-optimal | Discarding high frequency loses discriminative details. |
| FRM Low/High Ratio (Eq 4)| Optimal | Utilizes both low-freq dominance and high-freq details. |
| Gradient Editing (Zero param) | Effective | Balances optimization with zero additional parameters. |
| Weighted Loss | Effective | Suitable for host models calculating auxiliary losses. |

### Key Findings
- Sensitivity to missing modalities is highly asymmetric: on CASIA-SURF, missing Depth causes the worst drop (even lower than a model trained on other single modalities), while missing RGB has the least impact. Modalities with higher FRM are shown to be more dominant in training and cause more severe collapse when missing.
- In metric design, the low/high-freq ratio is superior to pure low-frequency magnitude—high-frequency details should not be discarded, and the ratio amplifies inter-modality gaps when low-frequency energy is similar.
- MWAM not only optimizes base models but also further raises the performance ceiling of SOTA methods specifically designed for missing modalities (RFNet/mmFormer/GSS).

## Highlights & Insights
- **Quantifying modality preference from a spectral perspective**: Starting from the observable and provable (NTK) phenomenon that "models prefer modalities rich in low frequencies," FRM provides a real-time, computable diagnosis of preference, offering a global view compared to spatial-domain feature-level balancing.
- **Truly plug-and-play with near-zero cost**: The basic form is parameter-free and adds negligible FLOPs. It is removed during inference and can be attached to various hosts with different backbones (CNN/ViT) and fusion strategies.
- **Inverse weighting as a transferable concept**: The framework of using "degree of preference" to inversely allocate training weights to combat imbalanced learning is useful not just for missing modalities, but for any multimodal training where modality/branch dominance is imbalanced.

## Limitations & Future Work
- The core assumption that "low-frequency dominance equals preference" was primarily validated on visual tasks (segmentation/classification with RGB/IR/Depth/MRI); its validity for modalities like audio or text with different frequency semantics remains untested.
- The weight function Equation (5) introduces four scaling factors ($\alpha,\beta,\lambda,\gamma$). Although sensitivity ablations exist, parameter tuning may still be required when switching datasets/tasks.
- Detail-oriented implementation (e.g., coordinate flip for high-freq terms in FRM) relies on the original paper's Fig. 2 and Appendices; official code should be consulted for exact replication.

## Related Work & Insights
- **vs. Imputation Methods (Tran 2017 / Lin 2023)**: These reconstruct missing modality features, requiring extra modules and overhead. Ours does not impute; it adjusts weights during training with zero inference cost.
- **vs. Spatial Modality Balancing (Peng 2022b / Wei 2023)**: These balance contributions at the feature level in the spatial domain; authors argue they haven't reached the performance cap. Ours uses FRM in the frequency domain and is naturally scalable.
- **vs. Imputation-free Unified Space Methods (Reza 2024 / Lee 2023)**: These project to modality-agnostic spaces but are highly sensitive to which modality is missing. Ours addresses the root cause of imbalanced learning, resulting in significantly lower PCR.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Quantifying modality preference in the frequency domain with low/high ratio is a novel perspective with theoretical support.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers segmentation/classification and various backbones, though tasks are vision-centric.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation-theory-method chain, though freq-domain notation in Fig. 2 is dense.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, near-zero parameters, and capable of raising the ceiling of existing methods; high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](../../AAAI2026/multimodal_vlm/plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[AAAI 2026\] LLMC+: Benchmarking Vision-Language Model Compression with a Plug-and-play Toolkit](../../AAAI2026/multimodal_vlm/llmc_benchmarking_vision-language_model_compression_with_a_plug-and-play_toolkit.md)
- [\[CVPR 2026\] MERLIN: Building Low-SNR Robust Multimodal LLMs for Electromagnetic Signals](../../CVPR2026/multimodal_vlm/merlin_building_low-snr_robust_multimodal_llms_for_electromagnetic_signals.md)
- [\[ICLR 2026\] Enhancing Multi-Image Understanding through Delimiter Token Scaling](enhancing_multi-image_understanding_through_delimiter_token_scaling.md)
- [\[ICLR 2026\] Vision-Zero: Scalable VLM Self-Evolution via Multi-Agent Self-Play](vision-zero_scalable_vlm_self-evolution_via_multi-agent_self-play.md)

</div>

<!-- RELATED:END -->
