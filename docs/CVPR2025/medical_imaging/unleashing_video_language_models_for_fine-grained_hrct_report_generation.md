---
title: >-
  [Paper Note] Unleashing Video Language Models for Fine-grained HRCT Report Generation
description: >-
  [CVPR 2025][Medical Imaging][HRCT report generation] This work proposes the AbSteering framework, which efficiently transfers general Video Language Models (VideoLMs) to the HRCT report generation task via abnormality-centered CoT training and DPO optimization based on clinically confusing hard negatives, outperforming specialized CT foundation models.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "HRCT report generation"
  - "video language models"
  - "Chain-of-Thought"
  - "DPO"
  - "abnormality detection"
  - "3D medical imaging"
date: 2026-05-08
content_hash: 8f2410cf5d25070e
---

# Unleashing Video Language Models for Fine-grained HRCT Report Generation

**Conference**: CVPR 2025  
**arXiv**: [2603.12469](https://arxiv.org/abs/2603.12469)  
**Code**: [GitHub](https://anonymous.4open.science/r/hrct-report-generation-video-vlm-728C/)  
**Area**: Medical Imaging  
**Keywords**: HRCT report generation, video language models, Chain-of-Thought, DPO, abnormality detection, 3D medical imaging

## TL;DR

This work proposes the AbSteering framework, which efficiently transfers general Video Language Models (VideoLMs) to the HRCT report generation task via abnormality-centered CoT training and DPO optimization based on clinically confusing hard negatives, outperforming specialized CT foundation models.

## Background & Motivation

**Clinical Value of HRCT Report Generation**: HRCT is a critical diagnostic modality for chest and cardiopulmonary diseases. AI-based automated report generation can reduce clinical workload, standardize diagnostic narratives, and mitigate inter-observer variability.

**Challenges from X-ray to CT**: Compared to 2D X-rays, HRCT introduces (1) computational and memory overhead due to hundreds of slices, and (2) more difficult visual understanding challenges, as clinically key abnormalities are subtle, spatially sparse, and diverse.

**Limitations of Prior Work**: Specialized CT foundation models (e.g., RadFM, CT-CHAT, M3D) require large-scale pre-training on CT data and still struggle with fine-grained long-tail abnormality identification.

**Potential of VideoLMs**: HRCT volumes can naturally be viewed as "video-like slice sequences". VideoLMs possess strong spatiotemporal reasoning capabilities but lack medical-domain knowledge.

**Core Problem**: (1) Can the encoders of VideoLMs capture 3D clinical features? (2) How can they be efficiently adapted to medical reports? (3) How do they perform compared to CT-specific models?

**Core Idea**: VideoLMs and CT-specific models share highly similar structures (3D tokenization + attention + LLM decoder), with the only difference being the training domain. Therefore, the key lies in efficient domain adaptation.

## Method

### Overall Architecture: AbSteering (Two Stages)

Using a pre-trained VideoLM as the backbone, domain adaptation is performed in two stages:

### Stage 1: Abnormality-Centered CoT Training

- **Report Structuring**: CT-RATE reports are restructured into a uniform (region: abnormality) template across 10 anatomical regions (lungs, trachea and bronchi, mediastinum, heart, esophagus, pleura, bones, thyroid, breast, abdomen), using GPT-4o for auxiliary categorization.
- **CoT Training**: The target sequence for training is $Y = [R_{AB}; R_{Full}]$, where the model first generates structured abnormality findings (reasoning anchor) and then synthesizes the complete report.
- **Design Motivation**: Force the model to perform abnormality reasoning first, suppressing descriptions dominated by normal tissues and hallucinations, and learning clinical associations such as disease co-occurrence/mutual exclusion.
- **Dataset**: The CT-RATE-AB dataset is curated, which contains structured abnormality annotations.

### Stage 2: Fine-grained Abnormality Discrimination (DPO)

- **Hard Negative Construction**: GPT-4o is utilized to replace real abnormalities with clinically confusing abnormalities within the same anatomical region to generate $R_{AB\_Fake}$, while maintaining report fluency and structural consistency.
- **DPO Optimization**: $\mathcal{L}_{DPO}$ steers the model to prefer the real report $R_{AB}$ (winning) over the fabricated report $R_{AB\_Fake}$ (losing), forcing the model to focus on subtle visual cues that distinguish the two.
- **Design Motivation**: CT abnormalities often manifest as subtle and visually confusing patterns. DPO enhances abnormality discrimination and suppresses hallucinations through contrastive learning.

### Architectural Details

The visual encoder of the VideoLM processes the CT input as $X \in \mathbb{R}^{T \times H \times W \times C}$, encoding it via spatiotemporal 3D attention, compressing it with a token merger, and then feeding it into the LLM. It is essentially identical in architecture to CT-specific models, with the key difference being the pre-training domain.

## Key Experimental Results

### Dataset
- CT-RATE: 25,692 non-contrast chest CTs (21,304 patients) extended to 50,188 volumes. Each CT is converted into a 240-frame, 480x480 MP4 video (18 fps).
- Training set: 46,717 CTs (20,000 patients), validation set: 3,039 CTs (1,314 patients).

### Main Results (CT-RATE benchmark)

| Method | BL-1 | RG-L | BERT | CE Micro F1 | CE Macro F1 |
|------|------|------|------|-------------|-------------|
| M3D-8B | 44.95 | 37.76 | 87.52 | 35.69 | 26.74 |
| Qwen2.5-VL-7B | 43.67 | 36.71 | 87.30 | 33.64 | 25.57 |
| InternVL3-8B | 45.57 | 38.49 | 87.40 | 44.45 | 38.91 |
| M3D-AbSteer | 45.22 | 38.58 | 87.83 | 43.24 | 36.18 |
| Qwen-AbSteer | 45.64 | 37.99 | 87.13 | 45.99 | 37.90 |
| **InternVL3-AbSteer** | **48.32** | **40.49** | **87.59** | **54.55** | **47.66** |

### Key Findings
- General VideoLMs (Qwen2.5-VL, InternVL3) can match the performance of the CT-specific foundation model M3D after fine-tuning.
- The performance gain of AbSteering on VideoLMs is significantly larger than its gain on M3D (InternVL3 CE Micro F1: 44.45 $\rightarrow$ 54.55).
- **InternVL3-AbSteer substantially outperforms all CT-specific models across all clinical efficacy metrics.**

### Ablation Study
- CoT significantly improves recall, while DPO on top of CoT improves both precision and recall.
- Video pre-training is critical: training from scratch leads to a massive performance drop; LoRA provides no gain (freezing the encoder is sufficient).
- LLM scale: 3B $\rightarrow$ 7B shows improvement, whereas 7B $\rightarrow$ 32B leads to a decline, indicating the bottleneck is in vision-language alignment rather than LLM capacity.

## Highlights & Insights

- **A New Paradigm for Cross-modal Transfer**: Demonstrated that general VideoLMs can be efficiently transferred to 3D medical imaging with limited data, eliminating the need to train specialized foundation models from scratch.
- **Synergy of CoT + DPO**: CoT improves abnormality recall, whereas DPO suppresses hallucinations—two aspects that are typically challenging to optimize simultaneously.
- **Clinically Confusing Hard Negatives**: Using visually confusing abnormalities in the same anatomical region to construct DPO negative examples precisely targets the bottleneck of fine-grained discrimination.
- **The Discovery of "Freeze the Encoder"**: VideoLM-pre-trained features are robust enough; no additional adaptation of the visual encoder is required.

## Limitations & Future Work

- Validated only on a single dataset (CT-RATE), lacking evaluation of generalization across institutions and disease spectrums.
- High dependency on GPT-4o (report structuring + hard negative generation), increasing the cost of data preparation and reproducibility risks.
- Focus is limited to chest HRCT; applicability to other anatomical regions such as abdominal or head CTs remains unverified.
- The phenomenon where a 32B LLM leads to a performance drop deserves further investigation (data density limitation vs. overfitting).
- Converting CT scans to the MP4 video format (240 frames at 18 fps) might introduce lossy compression artifacts, and its impact on detecting subtle abnormalities is not discussed.
- The improvement in NLG metrics is limited (e.g., BLEU-4 is only 23.58), highlighting that clinical efficacy metrics are the core contributions.

## Related Work & Insights

- **vs. CT-CHAT/RadFM (CT-specific Foundation Models)**: Though sharing similar architectures, their pre-training domains differ. AbSteering demonstrates that general video pre-training + efficient adaptation > domain-specific large-scale pre-training, with lower training costs.
- **vs. M3D-8B (State-of-the-art CT Foundation Model)**: The clinical efficacy improvement of M3D-AbSteer is less pronounced than that of VideoLM-AbSteer, suggesting that the general spatiotemporal reasoning capabilities of VideoLMs are more adaptable.
- **vs. Traditional CoT Methods**: The proposed CoT is not a generic reasoning chain, but a domain-specialized causal chain of "abnormality findings $\rightarrow$ report generation" that collaborates more tightly with DPO.
- **vs. Dia-LLaMA**: While Dia-LLaMA designs CT-specific visual encoders to connect with LLMs, this work proves that directly reusing the VideoLM encoder is sufficient, bypassing the need for domain-specialized encoders.

## Rating
- Novelty: ⭐⭐⭐⭐ The transfer path from VideoLM to HRCT is novel, and the combination of CoT and DPO is clever.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-backbone comparison, ablation studies, and qualitative case studies.
- Writing Quality: ⭐⭐⭐⭐ Well-reasoned motivation and convincing analysis of architectural equivalence.
- Value: ⭐⭐⭐⭐ Provides an efficient and practical new paradigm for 3D medical report generation.
- Overall Rating: 8/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](prototype-based_knowledge_guidance_for_fine-grained_structured_radiology_reporti.md)
- [\[CVPR 2025\] Thin-Shell-SfT: Fine-Grained Monocular Non-Rigid 3D Surface Tracking with Neural Deformation Fields](thin-shell-sft_fine-grained_monocular_non-rigid_3d_surface_tracking_with_neural_.md)
- [\[CVPR 2025\] Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation](enhanced_contrastive_learning_with_multi-view_longitudinal_data_for_chest_x-ray_.md)
- [\[ICLR 2026\] CerebraGloss: Instruction-Tuning a Large Vision-Language Model for Fine-Grained Clinical EEG Interpretation](../../ICLR2026/medical_imaging/cerebragloss_instruction-tuning_a_large_vision-language_model_for_fine-grained_c.md)
- [\[CVPR 2025\] SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation](sealion_semantic_part-aware_latent_point_diffusion_models_for_3d_generation.md)

</div>

<!-- RELATED:END -->
