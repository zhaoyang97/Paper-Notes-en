---
title: >-
  [Paper Note] VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection
description: >-
  [ICML 2026][AI Safety][Privacy detection] The authors construct VPD-100K, a large-scale visual privacy dataset comprising 100,000 images, 33 fine-grained categories, and over 190,000 instances (covering human faces…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Privacy detection"
  - "dataset"
  - "frequency-domain attention"
  - "YOLO"
  - "live stream"
date: 2026-05-08
content_hash: 267b2f0859153b72
---

# VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection

**Conference**: ICML 2026  
**arXiv**: [2605.10229](https://arxiv.org/abs/2605.10229)  
**Code**: https://vpd-100k.github.io/  
**Area**: AI Security / Visual Privacy Protection / Object Detection  
**Keywords**: Privacy detection, dataset, frequency-domain attention, YOLO, live stream  

## TL;DR
The authors construct VPD-100K, a large-scale visual privacy dataset comprising 100,000 images, 33 fine-grained categories, and over 190,000 instances (covering human faces, on-screen PII, physical identifiers, and location indicators). They propose a triple frequency-domain enhancement module (FDAF + Adaptive Spectral Gating + Frequency-Consistency Loss) integrated into the YOLOv10 Neck, improving the AP of YOLOv10-L on VPD-100K from 53.8 to 58.6 (+4.8) while maintaining stable real-time performance for live streams at 7.51ms latency.

## Background & Motivation

**Background**: Visual privacy detection is a critical requirement in the era of live streaming, screen sharing, and Vlogging. It necessitates real-time identification of sensitive information such as faces, ID cards, password fields, and street signs. Existing works are divided into two categories: image-level sensitivity prediction (coarse-grained, no localization) and object-level identifier detection (precise but limited by small datasets).

**Limitations of Prior Work**: Existing privacy datasets suffer from "three sins" as summarized by the authors: (1) **Small scale**: PrivacyAlert (6.8K), BIV-Priv (0.7K), and DIPA (1.5K) are insufficient for training large models; (2) **Coarse categories**: They only use broad tags like "person / other people," failing to distinguish between "adults indoors" vs. "children outdoors"; (3) **Narrow domains**: Most datasets overlook **on-screen PII** (emails, passwords, verification codes, chat records), which is the most severe source of leakage in modern digital life. Furthermore, many dataset links are broken or not released.

**Key Challenge**: Privacy data is naturally constrained by ethics—one cannot legally collect 100,000 photos of real people's credit cards for training. Thus, the desire for "large scale + real distribution" conflicts with regulatory compliance, leading to data scarcity.

**Goal**: (1) Provide the community with a **truly usable, 100K-scale, on-screen PII-inclusive** privacy detection dataset that is fully public; (2) Design a lightweight frequency-domain enhancement module for targets like small on-screen text, blurred faces, and low-contrast sensitive objects where spatial features are weak but frequency-domain signatures are significant; (3) Support both image and live video scenarios with a unified framework operating at 130+ FPS.

**Key Insight**: Replace real-world data collection with ethically controlled "scene reconstruction." For example, the team used internal accounts to simulate banking apps and verification codes to capture screenshots, obtaining pixel-perfect samples that do not violate the privacy of any real person. In the frequency domain, sensitive targets like text and facial edges exhibit strong signals in high-frequency components that are often smoothed out by spatial-domain YOLO operations; **explicitly modeling the frequency domain** compensates for this deficiency.

**Core Idea**: For data, use "taxonomy-driven multi-source aggregation + ethical scene reconstruction" to cover 4 domains and 33 classes. For the method, employ a "spatial + frequency dual-stream" approach—FDAF uses DFT to transform features and IDFT for reconstruction, adaptive spectral gating acts as a learnable "soft band-pass filter," and a frequency-consistency loss aligns the frequency distributions of predicted boxes with ground-truth (GT) boxes.

## Method

### Overall Architecture
Two independent but complementary contributions. **Data Side**: 4 major privacy domains (Human Presence / On-Screen PII / Physical Identifiers / Location Indicators), 33 fine-grained categories, 100K images, and over 190K boxes, labeled via a semi-automated pipeline (OCR + general detectors for pre-labeling, followed by manual refinement). **Model Side**: A "spatial-frequency dual-stream" branch is embedded in the middle of the YOLOv10 Neck. It consists of the FDAF (Frequency-Domain Attention Fusion) + Adaptive Spectral Gating (LSG) + Frequency-Consistency Loss. The system remains an end-to-end YOLO framework with an added frequency branch and an auxiliary loss.

### Key Designs

1. **Dataset Construction: Ethical Reconstruction + 4-Domain Taxonomy**:
    - **Function**: Fills the gaps of small scale, coarse categories, and lack of on-screen PII in existing datasets, while remaining strictly compliant and public.
    - **Mechanism**: Different strategies are used for the 4 domains: a subset of WIDER FACE + video snapshots + fine-grained attribute labels (e.g., "children indoors") for faces; **internal account simulations** for on-screen PII (banking, verification codes, chat) to avoid real PII; MIDV-500 (IDs) + targeted crawling (train tickets, delivery slips) for physical identifiers; and street view imagery for location indicators like shop signs. The final dataset contains 100K images (half over 10080p) and 190K+ boxes, passing ethical review. Table 1/2 shows the scale is ~15× larger than PrivacyAlert and contains ~1.5× more categories than DIPA2, with a more balanced CV (Coefficient of Variation) of 1.47 compared to 2.50 for DIPA2.
    - **Design Motivation**: Directly scraping real user data is illegal; synthetic data lacks realism. "Internal account simulation" is a clever compromise—it provides pixel-level accuracy for real software interfaces without involving real personal PII. Explicitly defining 4 domains ensures the taxonomy covers on-screen PII, an often ignored but high-risk category.

2. **Frequency-Domain Attention Fusion (FDAF)**:
    - **Function**: Introduces a frequency-domain branch into the high-level semantic feature maps of the YOLOv10 Neck to capture texture signals (stroke patterns, facial edges) difficult to represent spatially.
    - **Mechanism**: Performs an independent 2D DFT on each channel of the input feature $X \in \mathbb{R}^{C \times H \times W}$ to obtain magnitude and phase spectra $F_c(u,v) = \sum_{h,w} X_c(h,w) e^{-j2\pi(uh/H + vw/W)}$. After modulation by Adaptive Spectral Gating, it is transformed back via IDFT to the spatial domain $Y_{spa} = \mathcal{R}(\text{IDFT}(\tilde{F}))$. Finally, it is fused with original spatial features via residual connection and $1\times 1$ convolution: $I_{out} = \text{Conv}_{1\times 1}(\text{Concat}(I, Y_{spa})) + I$.
    - **Design Motivation**: Small on-screen text (often <10% of the image) or distant identifiers are "camouflaged" by texture averaging in the spatial domain, but their strokes correspond to distinct high-frequency components in the frequency spectrum. FDAF allows the network an additional pathway to "see" high-frequency details. Table 5 shows FDAF alone boosts AP by +2.2 (46.3 → 48.5).

3. **Adaptive Spectral Gating (LSG) + Frequency-Consistency Loss**:
    - **Function**: LSG allows the network to automatically decide which frequency bands to retain or suppress, avoiding a "one-size-fits-all" high-frequency amplification; the loss ensures frequency features within the predicted box match those of the GT box to refine fine-grained boundaries.
    - **Mechanism**: LSG defines a learnable weight tensor $W_{gate} \in \mathbb{R}^{C \times H \times W}$, which, after a Sigmoid activation, is multiplied with the spectrum via Hadamard product $\tilde{F}_c(u,v) = F_c(u,v) \odot \sigma(W_{gate}(u,v))$, acting as a joint channel-frequency soft mask. The frequency-consistency loss is defined as $\mathcal{L}_{freq} = \frac{1}{N}\sum_i \|W \odot (\mathcal{F}(P_i) - \mathcal{F}(T_i))\|_2^2$, where $W(r) = 1 + \lambda r$ is a weight positively correlated with frequency radius $r$, making **higher frequencies more important** to force the model to prioritize boundary details. Total loss: $\mathcal{L}_{total} = \mathcal{L}_{yolo} + 0.05 \cdot \mathcal{L}_{freq}$.
    - **Design Motivation**: Blindly amplifying all high frequencies introduces noise; LSG learns to activate horizontal/vertical frequencies for text and radial frequencies for faces. The frequency loss acts as a boundary-aware regularizer, contributing to the increase in AP75 (high IoU metric) from 53.9 to 54.6. The weight $\beta=0.05$ is small enough not to overwhelm the main loss but sufficient to tighten boundaries.

### Loss & Training
$\mathcal{L}_{total} = \mathcal{L}_{box} + \mathcal{L}_{cls} + \mathcal{L}_{dfl} + 0.05 \cdot \mathcal{L}_{freq}$. The backbones are YOLOv10-S/L, fully fine-tuned on the VPD-100K training set. All 14 baselines were fine-tuned on the same data for fairness. In the frequency weight $w(r) = 1 + \lambda \cdot r$, $\lambda$ is set by default to make high-frequency weights significantly larger than low-frequency ones.

## Key Experimental Results

### Main Results
Comparison of 15 detectors on the image test set (selected key rows):

| Model | AP | AP50 | AP75 | APS | APM | APL | Latency (ms) | F1 |
|-------|-----|------|------|-----|-----|-----|--------------|------|
| Grounding-DINO | 48.1 | 65.8 | 62.6 | 30.4 | 51.3 | 62.3 | 119.5 | 0.68 |
| YOLOv8-L | 52.6 | 68.3 | 59.1 | 32.6 | 58.5 | 67.3 | 14.76 | 0.72 |
| YOLOv9-L | 53.4 | 68.6 | 57.9 | 33.9 | 59.1 | 70.3 | 7.73 | 0.73 |
| YOLOv10-L | 53.8 | 69.6 | 58.4 | 33.6 | 59.8 | 70.8 | 7.42 | 0.73 |
| **YOLOv10-S + FEM** | 52.1 | 67.1 | 54.6 | 30.1 | 55.6 | 64.3 | 2.71 | 0.71 |
| **YOLOv10-L + FEM** | **58.6** | **73.4** | **61.3** | **36.5** | **62.3** | 70.6 | 7.51 | **0.81** |

Key gains: AP +4.8 (53.8→58.6), AP50 +3.8, APS (small objects like verification codes) +2.9. The F1 score reaches 0.81, significantly leading all baselines. For live video test sets, YOLOv10-L + FEM achieved an optimal AP of 57.7 with 7.51ms latency (~133 FPS), meeting real-time requirements.

### Ablation Study
Table 5 (Baseline: YOLOv10-S):

| Config | FDAF | LSG | $\mathcal{L}_{freq}$ | AP | AP50 | AP75 | APS |
|--------|------|-----|----------|-----|------|------|-----|
| Base | - | - | - | 46.3 | 62.7 | 51.3 | 26.1 |
| +FDAF | ✓ | - | - | 48.5 | 64.2 | 52.8 | 27.5 |
| +LSG | ✓ | ✓ | - | 50.9 | 65.8 | 53.9 | 29.2 |
| Full | ✓ | ✓ | ✓ | **52.1** | **67.1** | **54.6** | **30.1** |

The three components contribute +2.2, +2.4, and +1.2 respectively, totaling +5.8 AP.

### Key Findings
- **LSG contributes most to small objects**: APS increased from 27.5 to 29.2 (+1.7p), confirming that adaptive frequency band selection amplifies stroke features in text.
- **Frequency loss targets high IoU**: AP75, which is sensitive to boundary precision, improved from 53.9 to 54.6, validating the effectiveness of $\mathcal{L}_{freq}$ in boundary alignment.
- **Lightweight plugin, negligible latency**: Adding the modules to YOLOv10-S only increased latency from 2.53ms to 2.71ms (+7%), while yielding a +5.8 AP gain.
- **90% positive user study**: 90% of 20 participants agreed that the taxonomy is comprehensive and useful for reducing privacy anxiety in real-time streaming.
- **Good OOD generalization**: The model successfully detected receipt PII and sensitive info on shared screens in real live streaming platforms (with motion blur/compression), indicating the diversity of VPD-100K handles distribution shifts.

## Highlights & Insights
- The "ethical reconstruction" of on-screen PII is a genuine breakthrough—it bypasses legal constraints of real-user data collection while providing the model with a pixel-accurate real distribution. This methodology can be extended to medical imaging or license plate recognition.
- Implementing "frequency enhancement" as a **pluggable middle-module in the Neck** instead of modifying the backbone allows seamless migration to YOLOv11, DETR variants, or any detector using FPN/Neck structures.
- The use of $\beta = 0.05$ as a small-weight auxiliary loss reflects significant practical experience—strong frequency signals can deviate the model if $\beta$ is too large; finding this sweet spot is key.
- The concept of "frequency consistency" (aligning pred and GT in the frequency domain) is naturally suited for any "detail matching" task, such as medical segmentation boundaries, OCR, or super-resolution.

## Limitations & Future Work
- The dataset exhibits a significant long-tail distribution; rare categories like passports have very few samples, and true performance on these (vs. F1 average) requires deeper analysis.
- On-screen PII is "simulated"; the software interface distribution is biased toward products familiar to the team (banking apps, chat tools) and has not been tested on niche professional software.
- While the frequency branch is lightweight, DFT is $O(HW \log HW)$; the cost-benefit curve for 4K inputs is not provided.
- LSG weights $W_{gate}$ are spatial-frequency dual-dimensional ($C \times H \times W$), requiring resizing or retraining for different input dimensions.
- Privacy detection is only the first step; how to perform **secure anonymization** (blurring, masking, or inpainting) and integration with downstream pipelines (per-frame vs. every N frames) was not covered.

## Related Work & Insights
- **vs. DIPA / DIPA2 / BIV-Priv**: A "scaled-up and refined" version, with ~15× scale, ~1.5× categories, and a whole new domain for on-screen PII.
- **vs. PrivacyAlert / SensitivAlert**: Those use coarse image-level tags; VPD-100K provides object-level fine-grained boxes suitable for driving detection/anonymization pipelines.
- **vs. General Detectors (YOLOv10 / DETR)**: General detectors struggle with small text and low-contrast targets because they are optimized for natural objects; FEM uses the frequency domain to compensate for "fine-grained boundary" weaknesses.
- **vs. Tree-Ring / Frequency Watermarking**: While those use the frequency domain for generative model watermarking, VPD applies it to discriminative detection, sharing the insight that frequency-domain signal-to-noise ratios can be higher in low-contrast/small-object scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ — Significant dataset contribution (on-screen PII domain + ethical reconstruction methodology); frequency modules are a sound combination of classic ideas.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 14 baselines + dual scenarios + full ablation + user study + OOD tests; lacks detailed long-tail analysis and high-resolution curves.
- Writing Quality: ⭐⭐⭐⭐ — Clear problem-solution mapping; Tables 1/2 highlight advantages well; method section is clean though slightly verbose.
- Value: ⭐⭐⭐⭐⭐ — In the GDPR/CCPA era, a truly usable, compliant, 100K-scale public privacy dataset with on-screen PII is an industry necessity; the model is a bonus.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](../../AAAI2026/ai_safety/fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)
- [\[ICML 2026\] Persuasive Privacy](persuasive_privacy.md)
- [\[ICML 2026\] Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy](mind_the_gap_mixtures_of_gaussians_in_approximate_differential_privacy.md)
- [\[ICML 2026\] Position: Embodied AI Requires a Privacy-Utility Trade-off](position_embodied_ai_requires_a_privacy-utility_trade-off.md)
- [\[ICML 2026\] MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification](metamoe_diversity-aware_proxy_selection_for_privacy-preserving_mixture-of-expert.md)

</div>

<!-- RELATED:END -->
