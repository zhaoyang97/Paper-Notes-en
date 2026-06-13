---
title: >-
  [Paper Note] QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training
description: >-
  [NeurIPS 2025][Medical Imaging][Multimodal clinical] QoQ-Med constructs a multimodal clinical foundation model spanning 9 clinical modalities (1D ECG + 6 types of 2D images + 2 types of 3D scans)…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Multimodal clinical"
  - "GRPO"
  - "domain-aware reinforcement learning"
  - "ECG + imaging + text"
  - "interpretable reasoning"
date: 2026-05-08
content_hash: 48245dea5f9b4830
---

# QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training

**Conference**: NeurIPS 2025 Oral  
**arXiv**: [2506.00711](https://arxiv.org/abs/2506.00711)  
**Code**: Weights and training pipeline released  
**Area**: Medical Imaging
**Keywords**: Multimodal clinical, GRPO, domain-aware reinforcement learning, ECG + imaging + text, interpretable reasoning

## TL;DR
QoQ-Med constructs a multimodal clinical foundation model spanning 9 clinical modalities (1D ECG + 6 types of 2D images + 2 types of 3D scans), and proposes Domain-aware Relative Policy Optimization (DRPO)—which employs hierarchical temperature scaling (inter-domain × intra-domain K-means clustering) to address modality/difficulty imbalance. Trained on 2.61 million instruction-tuning pairs, it achieves an average F1 of 0.295 (vs. GRPO 0.193, +52.8%), ranking best in 6 out of 8 modalities.

## Background & Motivation

**Background**: Multimodal LLMs (MLLMs) have advanced rapidly on general tasks, yet clinical applications require simultaneous handling of 1D temporal signals (ECG/EEG), 2D images (chest X-rays/dermoscopy/fundus), and 3D volumetric data (CT/MRI)—no existing MLLM covers all three categories.

**Limitations of Prior Work**: (a) Severe modality imbalance—chest X-ray data is abundant while ECG data is scarce, causing standard training to be dominated by data-rich modalities; (b) Large intra-domain difficulty variance—simple chest X-rays vs. complex CT oblique reconstructions require different learning strategies; (c) Black-box models lack reasoning traces—clinical deployment demands interpretability.

**Key Challenge**: GRPO (critic-free) is more efficient than PPO but does not handle domain/difficulty imbalance; naively mixing all modalities during training causes rare modalities to be overwhelmed.

**Goal**: Address domain/difficulty imbalance in multimodal clinical training within the GRPO framework, while generating interpretable reasoning traces and localization bounding boxes.

**Key Insight**: DRPO introduces two levels of temperature scaling after GRPO normalization—inter-domain scaling ($T_{(g,t)} = \max(\sqrt{N_g} \cdot \mu_g, \varepsilon)$) and intra-domain K-means clustering scaling—so that rare modalities and difficult samples receive larger learning gradients.

**Core Idea**: 9-modality clinical data + DRPO (hierarchical inter/intra-domain K-means temperature scaling) + IoU reward-guided localization = interpretable multimodal clinical reasoning.

## Method

### Overall Architecture
**Input**: Interleaved image/ECG/text sequences → **Encoder**: Pretrained visual encoder + ECG-JEPA encoder + linear projection → **LLM**: Generates reasoning chains + localization boxes + diagnoses → **Training**: Two-stage—Stage 1 modality alignment (ECG encoder + projection + LLM) → Stage 2 full-modality fine-tuning, both using DRPO.

### Key Designs

1. **Domain-aware Relative Policy Optimization (DRPO)**:

    - Function: Addresses domain/difficulty imbalance in multimodal training.
    - Mechanism: Adds two levels of temperature scaling after standard GRPO normalization. **Inter-domain**: $T_{(g,t)} = \max(\sqrt{N_g} \cdot \mu_g, \varepsilon)$—domains with more samples and higher rewards receive larger scaling (suppression), while domains with fewer samples or lower rewards receive smaller scaling (amplification). **Intra-domain**: K-means clusters reward vectors → each cluster applies temperature scaling $T_{(c,g,t)}$—difficult clusters (low reward) receive larger learning signals.
    - Design Motivation: Inter-domain scaling handles modality imbalance (e.g., chest X-ray vs. ECG); intra-domain K-means handles difficulty imbalance (e.g., easy vs. hard questions).

2. **Multimodal Reward Function**:

    - Function: Simultaneously incentivizes diagnostic accuracy and localization quality.
    - Mechanism: $r_i = 0.6 \cdot r^{acc} + 0.2 \cdot r^{IoU} + 0.2 \cdot r^{aux}$. Accuracy reward = F1 (unordered label set); IoU reward = maximum IoU between predicted boxes and segmentation masks; auxiliary reward = format/reasoning length consistency.
    - Design Motivation: Accuracy-only rewards do not produce localization capability—the IoU reward enables the model to identify evidence regions, improving interpretability.

3. **Three-Modality Input Fusion**:

    - Function: Unified processing of 1D/2D/3D clinical data.
    - Mechanism: Image patches → visual encoder → linear projection; ECG → ECG-JEPA encoder → newly initialized linear projection; text directly tokenized. All three token types are interleaved in temporal order as LLM input.
    - Design Motivation: ECG is a 1D temporal signal incompatible with visual encoders—a dedicated ECG-JEPA encoder preserves temporal features.

### Loss & Training
- DRPO policy gradient with hierarchical temperature scaling
- 2.61 million instruction-tuning pairs, 33 datasets, 9 clinical modalities
- K-means elbow method for automatic cluster number selection
- DRPO reward computation overhead < 2% of total training time

## Key Experimental Results

### Main Results (F1 across 8 imaging modalities)

| Method | CXR | Breast | Skin | CT | Fundus | Ultrasound | MRI | Pathology | Avg |
|--------|-----|--------|------|----|--------|------------|-----|-----------|-----|
| SFT | .078 | .056 | .158 | .236 | .066 | .235 | .197 | .083 | .139 |
| GRPO | .095 | .059 | .244 | .236 | .086 | .146 | .395 | .286 | .193 |
| PPO | .064 | .205 | .278 | .257 | .083 | .080 | .540 | .364 | .234 |
| **DRPO** | **.115** | **.253** | **.407** | **.309** | .093 | **.223** | **.625** | .265 | **.295** |

**DRPO vs. GRPO**: +52.8%; **DRPO vs. PPO**: +26.1%

### Multimodal Fusion (MIMIC-IV, CXR + ECG + EHR)

| Model | Length of Stay F1 | 48h In-hospital Mortality F1 |
|-------|-------------------|------------------------------|
| GRPO-Full | 0.105 | 0.354 |
| DRPO-TextOnly | 0.195 | — |
| DRPO-Vision+Text | 0.223 | — |
| **DRPO-Full (3 modalities)** | **0.283** | **0.597** |

### Ablation Study

| Configuration | Avg F1 | Notes |
|---------------|--------|-------|
| Inter-domain scaling only | 0.237 | +22.8% vs. GRPO |
| Intra-domain K-means only | — | — |
| **Full DRPO** | **0.295** | +52.8% vs. GRPO |
| K-means 10 clusters | 0.286 | Optimal (elbow) |
| K-means 20 clusters | 0.294 | Comparable |
| Reward weight (0.6:0.2) | **0.286** | Optimal |
| Reward weight (0.2:0.6) | 0.260 | IoU-heavy → degradation |

### Key Findings
- DRPO ranks best in 6 out of 8 modalities—particularly large improvements on rare modalities (breast: +328.8%!)
- Breast modality improves from 0.059 → 0.253 (4.3×), demonstrating the critical role of inter-domain scaling for rare modalities.
- MRI improves from 0.395 → 0.625 (+58.2%), indicating the importance of intra-domain difficulty stratification.
- Three-modality fusion (CXR + ECG + Text) outperforms any single or dual modality combination—clinical practice requires multimodal integration.
- Clinical validation of reasoning traces shows high correlation—localization boxes point to correct abnormal regions.

## Highlights & Insights
- **DRPO's hierarchical temperature scaling** elegantly addresses two core challenges in multimodal RL training—domain imbalance and difficulty imbalance—with minimal overhead (<2%).
- **Breast modality +328.8%** improvement demonstrates that RL without domain awareness is essentially ineffective for rare modalities.
- **IoU reward-guided localization** enables the model to not only diagnose but also "identify evidence"—directly satisfying clinical interpretability requirements.
- **Unified three-modality (image + temporal + text) processing** is the first of its kind—prior clinical MLLMs handled only 2D images and text.

## Limitations & Future Work
- Reasoning traces are unsupervised—supervised reasoning may be more efficient.
- Visual and textual modalities contribute more to ECG than vice versa—ECG fusion strategies require further optimization.
- Inference latency and throughput are not discussed.
- Detailed sub-domain coverage for all modalities is incomplete (e.g., ultrasound subcategories).

## Related Work & Insights
- **vs. Med-R1**: Also applies RL for medical reasoning but is not multimodal; QoQ-Med covers 9 modalities.
- **vs. LLaVA-Med**: Supports only 2D images and text, with no ECG or 3D support.
- **vs. GRPO**: Standard GRPO does not handle domain imbalance; DRPO is a natural extension thereof.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ DRPO + three-modality clinical reasoning is a significant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 modalities + 33 datasets + 2.61M samples + multiple RL baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear methodology and thorough ablation.
- Value: ⭐⭐⭐⭐⭐ Provides an interpretable multimodal reasoning foundation model for clinical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology](mtbbench_a_multimodal_sequential_clinical_decision-making_benchmark_in_oncology.md)
- [\[NeurIPS 2025\] Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens](brain_harmony_a_multimodal_foundation_model_unifying_morphology_and_function_int.md)
- [\[NeurIPS 2025\] Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation](toward_a_vision-language_foundation_model_for_medical_data_multimodal_dataset_an.md)
- [\[ICCV 2025\] SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications](../../ICCV2025/medical_imaging/scivid_cross-domain_evaluation_of_video_models_in_scientific_applications.md)
- [\[NeurIPS 2025\] 3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks](3drad_a_comprehensive_3d_radiology_medvqa_dataset_with_multi.md)

</div>

<!-- RELATED:END -->
