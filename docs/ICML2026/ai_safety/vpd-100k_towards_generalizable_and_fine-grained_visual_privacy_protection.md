---
title: >-
  [Paper Note] VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection
description: >-
  [ICML 2026][AI Safety][Privacy Detection] The authors constructed a large-scale visual privacy dataset, VPD-100K, with 100,000 images, 33 fine-grained categories, and over 190,000 instances…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Privacy Detection"
  - "Dataset"
  - "Frequency-domain Attention"
  - "YOLO"
  - "Live Streaming"
date: 2026-05-08
content_hash: ddb1a41f10382a3b
---

# VPD-100K: Towards Generalizable and Fine-grained Visual Privacy Protection

**Conference**: ICML 2026  
**arXiv**: [2605.10229](https://arxiv.org/abs/2605.10229)  
**Code**: https://vpd-100k.github.io/  
**Area**: AI Security / Visual Privacy Protection / Object Detection  
**Keywords**: Privacy Detection, Dataset, Frequency-domain Attention, YOLO, Live Streaming

## TL;DR
The authors constructed a large-scale visual privacy dataset, VPD-100K, with 100,000 images, 33 fine-grained categories, and over 190,000 instances, covering four major domains (faces/on-screen PII/physical identifiers/location markers). They propose a three-part frequency-domain enhancement module (FDAF + Adaptive Spectral Gating + Frequency-domain Consistency Loss) inserted into the Neck of YOLOv10, boosting YOLOv10-L's AP on VPD-100K from 53.8 to 58.6 (+4.8), while maintaining stable real-time performance on live streams at 7.51ms latency.

## Background & Motivation

**Background**: Visual privacy detection is a critical need in the era of live streaming, screen sharing, and vlogs—requiring real-time identification of sensitive information such as faces, ID cards, password fields, and street signs. Existing work falls into two camps: image-level sensitivity prediction (coarse, no localization) and object-level identification (precise but with small datasets).

**Limitations of Prior Work**: The authors summarize the issues of existing privacy datasets as "three major flaws": (1) **Small scale**: PrivacyAlert 6.8K, BIV-Priv 0.7K, DIPA 1.5K—far from sufficient for training large models; (2) **Coarse categories**: Only broad tags like "person/other people," unable to distinguish "indoor adult" vs. "outdoor child"; (3) **Narrow domain**: Almost all datasets ignore **on-screen PII** (emails, passwords, verification codes, chat logs), which are the most severe leakage sources in modern digital life. Most datasets are also unavailable or have broken links.

**Key Challenge**: Privacy data is inherently constrained by ethics—it is not legally feasible to collect 100,000 real bank card photos for training. Thus, "large-scale" and "realistic distribution" are in conflict at the compliance level, leading to data scarcity.

**Goal**: (1) Provide the community with a **truly usable, 100K-scale, on-screen PII-inclusive** privacy detection dataset, fully open; (2) For targets like "on-screen small text, blurred faces, low-contrast sensitive objects" with weak spatial but strong frequency-domain features, design a lightweight frequency-domain enhancement module; (3) Support both image and live video scenarios in a unified framework, running at 130+ FPS.

**Key Insight**: Replace real data collection with ethically controlled "scene reconstruction"—for example, the team uses internal accounts to simulate banking, receive verification codes, and take screenshots, obtaining pixel-level accurate on-screen PII samples without infringing real privacy. In the frequency domain, privacy targets like text and face edges have strong signals in high-frequency components, but spatial YOLO averages them out; **explicit frequency-domain modeling** addresses this gap.

**Core Idea**: On the data side, "taxonomy-driven multi-source aggregation + ethical scene reconstruction" covers 4 domains and 33 categories; on the method side, "spatial + frequency dual-stream"—FDAF applies DFT to features and reconstructs via IDFT, adaptive spectral gating acts as a learnable "soft band-pass filter," and frequency-domain consistency loss aligns the frequency distributions of predicted and GT boxes.

## Method

### Overall Architecture
Two independent but complementary contributions. **Data side**: Four privacy domains (Human Presence / On-Screen PII / Physical Identifiers / Location Indicators), 33 fine-grained categories, 100,000 images, 190,000+ boxes, semi-automatic annotation pipeline (OCR + general detector pre-labeling, then manual refinement). **Model side**: Embed a "spatial-frequency dual-stream" branch in the middle of YOLOv10's Neck, consisting of FDAF (Frequency-Domain Attention Fusion) + Adaptive Spectral Gating (LSG) + Frequency-Consistency Loss. The overall training remains end-to-end YOLO, with an additional frequency branch and auxiliary loss.

### Key Designs

1. **Dataset Construction: Ethical Reconstruction + 4-domain Taxonomy**:

    - **Function**: Fills the gaps of "small scale + coarse categories + ignoring on-screen PII" in existing privacy datasets, strictly compliant and publicly available.
    - **Mechanism**: Different strategies for each domain: faces use WIDER FACE subsets + video snapshots + fine-grained attribute annotation (e.g., "indoor child face"); on-screen PII is generated by the research team **simulating** real digital interactions (bank login, verification codes, chat) with internal accounts and screenshots, avoiding any real privacy; physical identifiers use MIDV-500 (IDs) + targeted crawling (train tickets, delivery slips); location indicators are captured from outdoor street scenes with annotated store signs, etc. Final dataset: 100K images, over half at 1080p+, 190K+ boxes, passed ethical review. Table 1/2 shows the scale is ~15× that of the second-largest dataset PrivacyAlert, category count is ~1.5× DIPA2, and CV (category distribution coefficient of variation) 1.47 is more balanced than DIPA2's 2.50.
    - **Design Motivation**: Directly collecting real user data is illegal, synthetic data is unrealistic; "internal account simulation" is a smart compromise—achieving pixel-level realism of software interfaces without touching real PII. Explicitly dividing into four domains ensures taxonomy covers on-screen PII, the most impactful but often ignored category.

2. **Frequency-Domain Attention Fusion (FDAF)**:

    - **Function**: Introduces a frequency-domain branch on high-semantic feature maps in YOLOv10 Neck to capture texture signals (text strokes, face edges) that are hard to express spatially.
    - **Mechanism**: For input feature $X \in \mathbb{R}^{C \times H \times W}$, perform 2D DFT independently per channel to get $F_c(u,v) = \sum_{h,w} X_c(h,w) e^{-j2\pi(uh/H + vw/W)}$, yielding magnitude and phase spectra. After Adaptive Spectral Gating modulation, IDFT brings it back to the spatial domain $Y_{spa} = \mathcal{R}(\text{IDFT}(\tilde{F}))$. Finally, residual connection + $1\times 1$ conv fuses with original spatial features: $I_{out} = \text{Conv}_{1\times 1}(\text{Concat}(I, Y_{spa})) + I$.
    - **Design Motivation**: "Camouflaged" targets like small on-screen text (verification codes <10% of image), distant IDs are averaged out in the spatial domain, but text strokes correspond to clear horizontal/vertical high-frequency components in the spectrum. FDAF provides a feature path that directly "sees" high-frequency details. Table 5 ablation shows FDAF alone brings AP +2.2 (46.3 → 48.5).

3. **Adaptive Spectral Gating (LSG) + Frequency-domain Consistency Loss**:

    - **Function**: LSG allows the network to automatically decide which frequency bands to retain/suppress, avoiding indiscriminate "amplification of all high frequencies"; frequency-domain loss aligns the frequency features inside predicted boxes with those inside GT boxes, enhancing fine-grained boundaries.
    - **Mechanism**: LSG defines a learnable weight tensor $W_{gate} \in \mathbb{R}^{C \times H \times W}$, after Sigmoid, performs Hadamard product with the spectrum $\tilde{F}_c(u,v) = F_c(u,v) \odot \sigma(W_{gate}(u,v))$, acting as a channel-frequency joint soft mask. Frequency-domain consistency loss $\mathcal{L}_{freq} = \frac{1}{N}\sum_i \|W \odot (\mathcal{F}(P_i) - \mathcal{F}(T_i))\|_2^2$, where $W(r) = 1 + \lambda r$ is a weight positively correlated with frequency radius $r$, **higher frequency is more important**, forcing the model to prioritize boundary details. Total loss $\mathcal{L}_{total} = \mathcal{L}_{yolo} + 0.05 \cdot \mathcal{L}_{freq}$.
    - **Design Motivation**: Blindly amplifying all high frequencies introduces noise; LSG enables the network to "activate horizontal/vertical bands for text, radial bands for faces." Frequency-domain loss acts as a boundary-aware regularizer, with AP75 (high IoU metric) increasing from 53.9 to 54.6 due to this component. $\beta=0.05$ is small enough not to overwhelm the main loss but sufficient to tighten boundaries.

### Loss & Training
$\mathcal{L}_{total} = \mathcal{L}_{box} + \mathcal{L}_{cls} + \mathcal{L}_{dfl} + 0.05 \cdot \mathcal{L}_{freq}$. The base is YOLOv10-S/L, fully fine-tuned on the VPD-100K training set; all 14 baselines are fine-tuned on the same data for fairness. Frequency-domain weight $w(r) = 1 + \lambda \cdot r$, with $\lambda$ set so high-frequency weights are significantly larger than low-frequency.

## Key Experimental Results

### Main Results
Comparison of 15 detectors on the image test set (key rows):

| Model | AP | AP50 | AP75 | APS | APM | APL | Latency (ms) | F1 |
|------|-----|------|------|-----|-----|-----|--------------|------|
| Grounding-DINO | 48.1 | 65.8 | 62.6 | 30.4 | 51.3 | 62.3 | 119.5 | 0.68 |
| YOLOv8-L | 52.6 | 68.3 | 59.1 | 32.6 | 58.5 | 67.3 | 14.76 | 0.72 |
| YOLOv9-L | 53.4 | 68.6 | 57.9 | 33.9 | 59.1 | 70.3 | 7.73 | 0.73 |
| YOLOv10-L | 53.8 | 69.6 | 58.4 | 33.6 | 59.8 | 70.8 | 7.42 | 0.73 |
| **YOLOv10-S + FEM** | 52.1 | 67.1 | 54.6 | 30.1 | 55.6 | 64.3 | 2.71 | 0.71 |
| **YOLOv10-L + FEM** | **58.6** | **73.4** | **61.3** | **36.5** | **62.3** | 70.6 | 7.51 | **0.81** |

Main gains: AP +4.8 (53.8→58.6), AP50 +3.8, APS (small objects, e.g., verification codes) +2.9, F1 jumps to 0.81, significantly outperforming all baselines.

On the live video test set, YOLOv10-L + FEM also achieves the best AP of 57.7, with 7.51ms latency (~133 FPS), meeting real-time streaming requirements.

### Ablation Study
Table 5 (base: YOLOv10-S):

| Config | FDAF | LSG | $\mathcal{L}_{freq}$ | AP | AP50 | AP75 | APS |
|------|------|-----|----------|-----|------|------|-----|
| Base | - | - | - | 46.3 | 62.7 | 51.3 | 26.1 |
| +FDAF | ✓ | - | - | 48.5 | 64.2 | 52.8 | 27.5 |
| +LSG | ✓ | ✓ | - | 50.9 | 65.8 | 53.9 | 29.2 |
| Full | ✓ | ✓ | ✓ | **52.1** | **67.1** | **54.6** | **30.1** |

The three modules contribute +2.2 / +2.4 / +1.2 AP respectively, totaling +5.8 AP.

### Key Findings
- **LSG benefits small objects most**: APS increases from 27.5 → 29.2 (+1.7p), especially for small targets, confirming that "adaptive frequency selection amplifies text stroke features."
- **Frequency-domain loss targets high IoU**: AP75, sensitive to boundary precision, rises from 53.9 to 54.6 with $\mathcal{L}_{freq}$, confirming its boundary refinement effect.
- **Lightweight plugin, minimal latency increase**: YOLOv10-S with all three modules sees latency rise from 2.53 to 2.71ms (+7%) for a +5.8 AP gain, with negligible parameter/compute overhead.
- **90% positive user study**: On a Likert scale, 90% of 20 participants agreed the taxonomy is comprehensive, useful for live streaming, and reduces privacy anxiety.
- **Good OOD generalization**: On real live streaming platforms (handheld shake, screen sharing), the model detects receipt PII and sensitive on-screen info, showing VPD-100K's diversity supports distribution shifts.

## Highlights & Insights
- The "ethical reconstruction" approach for on-screen PII is a true breakthrough—it circumvents legal barriers of collecting real user data while providing pixel-level realistic distributions. This methodology can be extended to medical imaging (via compliant clinical partners), license plate recognition, and any PII-regulated field.
- Making "frequency-domain enhancement" a **pluggable Neck module** rather than modifying the backbone enables seamless migration to YOLOv11 / DETR series / any detector with FPN/Neck, offering high engineering value.
- The $\beta = 0.05$ "small-weight auxiliary loss" tuning trick reflects the authors' experience—frequency-domain loss is strong; if $\beta$ is too large, it overwhelms the main loss and destabilizes training. Finding this sweet spot is key.
- The "frequency consistency" idea (aligning pred and GT in the frequency domain) is naturally suited for any "detail matching" task—medical image segmentation boundaries, character OCR, super-resolution, all can benefit from this loss.

## Limitations & Future Work
- The dataset is long-tailed, with very few samples for rare categories like passports; the model's true performance on long-tail categories (vs. mean F1) is not deeply analyzed.
- On-screen PII is "simulated" by the team, so software interface distribution is biased toward familiar products (banking apps, chat tools); generalization to niche interfaces (overseas platforms, specialized software) is untested.
- Although the frequency branch is lightweight, DFT on large feature maps is still $O(HW \log HW)$; for 4K inputs, efficiency may not hold, and the authors do not provide a high-resolution cost-benefit curve.
- LSG's weights $W_{gate}$ are spatial-frequency joint, with parameter count = $C \times H \times W$; resizing or retraining is needed for different input sizes.
- Privacy detection is only the first step; **how to safely redact after detection** (blur? mask? region erasure?) and how to integrate with downstream live streaming pipelines (every N frames vs. every frame) are not addressed, so practical deployment remains distant.

## Related Work & Insights
- **vs DIPA / DIPA2 / BIV-Priv**: This is a "scaled-up + refined" version of existing datasets, ~15× larger, ~1.5× more categories, and adds the entire on-screen PII domain.
- **vs PrivacyAlert / SensitivAlert**: Those are image-level coarse tags; VPD-100K provides object-level fine-grained boxes, directly enabling detection/redaction pipelines.
- **vs General Detectors YOLOv10 / DETR**: General detectors perform poorly on small on-screen text and low-contrast targets because they are optimized for natural objects; FEM uses the frequency domain to address "fine boundary" shortcomings.
- **vs Tree-Ring / Frequency-domain Watermarking**: Those use the frequency domain for generative model watermarking; VPD applies it to discriminative detection, but the underlying idea is similar—frequency signals have higher SNR than spatial in low-contrast/small-object scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ — The dataset contribution is outstanding (entire on-screen PII domain + ethical reconstruction methodology is truly innovative); the frequency-domain trio is a reasonable combination of classic ideas rather than a paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 14 baselines + image/video dual scenarios + complete ablation + user study + OOD testing, already very solid; lacks finer long-tail analysis and high-resolution cost curves.
- Writing Quality: ⭐⭐⭐⭐ — Pain-point/solution contrast is clear, Table 1/2 immediately shows dataset advantages; method section formulas are clean but a bit lengthy.
- Value: ⭐⭐⭐⭐⭐ — In the GDPR/CCPA era, a truly usable, compliant, on-screen PII-inclusive 100K-scale public privacy dataset is an industry necessity; the model is a bonus.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](../../AAAI2026/ai_safety/fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)
- [\[ICML 2026\] Position: Embodied AI Requires a Privacy-Utility Trade-off](position_embodied_ai_requires_a_privacy-utility_trade-off.md)
- [\[ICML 2026\] MetaMoE: Diversity-Aware Proxy Selection for Privacy-Preserving Mixture-of-Experts Unification](metamoe_diversity-aware_proxy_selection_for_privacy-preserving_mixture-of-expert.md)
- [\[ICLR 2026\] VPI-Bench: Visual Prompt Injection Attacks for Computer-Use Agents](../../ICLR2026/ai_safety/vpi-bench_visual_prompt_injection_attacks_for_computer-use_agents.md)
- [\[CVPR 2026\] ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering](../../CVPR2026/ai_safety/clustermark_towards_robust_watermarking_for_autoregressive_image_generators_with.md)

</div>

<!-- RELATED:END -->
