---
title: >-
  [Paper Note] Unleashing Video Language Models for Fine-grained HRCT Report Generation
description: >-
  [CVPR 2026][Medical Imaging][CT report generation] This paper proposes AbSteering, a two-stage framework that adapts general-purpose VideoLMs to HRCT report generation via abnormality-centric Chain-of-Thought reasoning and DPO-based hard-negative contrastive learning, substantially outperforming specialized CT foundation models on clinical efficacy metrics.
tags:
  - CVPR 2026
  - Medical Imaging
  - CT report generation
  - video language models
  - Chain-of-Thought
  - DPO
  - anomaly detection
date: 2026-05-08
content_hash: 555df1348081be93
---

# Unleashing Video Language Models for Fine-grained HRCT Report Generation

**Conference**: CVPR 2026
**arXiv**: [2603.12469](https://arxiv.org/abs/2603.12469)
**Code**: [GitHub](https://anonymous.4open.science/r/hrct-report-generation-video-vlm-728C/)
**Area**: Medical Imaging
**Keywords**: CT report generation, video language models, Chain-of-Thought, DPO, anomaly detection

## TL;DR

This paper proposes AbSteering, a two-stage framework that adapts general-purpose VideoLMs to HRCT report generation via abnormality-centric Chain-of-Thought reasoning and DPO-based hard-negative contrastive learning, substantially outperforming specialized CT foundation models on clinical efficacy metrics.

## Background & Motivation

**Clinical Need**: High-resolution computed tomography (HRCT) is a critical modality for diagnosing and longitudinally monitoring thoracic and cardiopulmonary diseases. AI-driven report generation can reduce clinical workload, standardize diagnostic narratives, and mitigate inter-observer variability. Compared to 2D chest X-rays, 3D HRCT report generation poses greater challenges: each study contains hundreds of slices, incurring substantial computational and memory overhead; clinically significant abnormalities are often subtle, spatially localized, and diverse, sparsely distributed throughout the volume and frequently obscured by dominant normal anatomical structures.

**Limitations of Prior Work**: Early approaches compressed CT volumes into low-dimensional representations and repurposed X-ray report generators, resulting in severe information loss. Subsequent work, such as Dia-LLaMA, designed CT-specific visual encoders coupled with LLM decoders. More recent modality-specific foundation models (RadFM, CT-CHAT, M3D) further improved performance but still relied on training from scratch or extensive fine-tuning of modality-specific encoders, incurring high data and computational costs, with persistent bottlenecks in fine-grained recognition of long-tail abnormalities.

**Core Insight**: HRCT volumes can be naturally treated as "video-like slice sequences," and the architecture of VideoLMs (spatiotemporal tokenization + 3D attention + token merging + LLM decoding) is essentially analogous to that of CT foundation models. The gap between the two lies not in the architecture itself but in the training domain and supervision signals. This motivates three key questions: (1) Can VideoLM encoders capture clinically relevant 3D features? (2) How can general-purpose VideoLMs be efficiently adapted to domain-specific medical report generation? (3) How does such transfer compare to modality-specific CT foundation models?

## Method

### Overall Architecture

AbSteering adopts a pre-trained VideoLM as the backbone and performs domain adaptation in two stages: Stage 1 establishes structured reasoning pathways through abnormality-centric Chain-of-Thought training; Stage 2 enables fine-grained abnormality discrimination through Direct Preference Optimization. The framework leaves the visual encoder architecture unchanged, achieving domain adaptation solely through language-level guidance.

### Key Designs

1. **Abnormality-Centric Chain-of-Thought Training (Stage 1)**:

    - Function: Decouples the vision-to-text task into a two-step reasoning process of "first detect abnormalities, then generate the report."
    - Mechanism: Raw CT-RATE reports are first standardized into a unified (region: abnormality) template covering 10 anatomical regions (Lung, Trachea and Bronchi, Mediastinum, Heart, Esophagus, Pleura, Bone, Thyroid, Breast, Abdomen). GPT-4o is used to assign report sentences to corresponding regions, followed by manual verification, producing the CT-RATE-AB dataset. A sequential generation objective is then employed with target sequence $Y = [R_{AB}; R_{Full}]$, where the model first generates an abnormality detection list $R_{AB}$ before generating the full report $R_{Full}$, trained via the autoregressive loss $\mathcal{L}_{gen} = -\sum_{t=1}^{T} \log P(y_t | x, y_{<t})$.
    - Design Motivation: The model is compelled to perform explicit clinical reasoning prior to generating the final report, learning disease category diversity while suppressing descriptions dominated by normal tissue and hallucinations. At the reasoning level, the transition from discrete findings to narrative enables the model to capture anatomical constraints and inter-pathology dependencies (e.g., co-occurrence of related diseases or mutual exclusivity of contradictory findings).

2. **DPO-Based Fine-Grained Abnormality Discrimination (Stage 2)**:

    - Function: Enhances the model's ability to distinguish subtle pathological differences and suppresses hallucinations.
    - Mechanism: GPT-4o is used to automatically construct hard negatives $R_{AB\_Fake}$ from ground-truth abnormality reports $R_{AB}$ by replacing target abnormalities with clinically confusable abnormalities within the same anatomical region, while preserving region labels, sentence templates, and positional information. The Stage 1 model serves as the reference model $\pi_{ref}$, and the target model $\pi_\theta$ is optimized via the DPO objective: $\mathcal{L}_{DPO} = \log \sigma(\beta \log \frac{\pi_\theta(y_w|x,v)}{\pi_{ref}(y_w|x,v)} - \beta \log \frac{\pi_\theta(y_l|x,v)}{\pi_{ref}(y_l|x,v)})$, where $y_w = R_{AB}$ (correct report) and $y_l = R_{AB\_Fake}$ (manipulated report).
    - Design Motivation: CT abnormalities frequently exhibit subtle and visually confusable patterns, and fine-grained discrimination is highly dependent on domain-specific clinical knowledge. By contrasting correct reports against clinically confusable ones, the model is forced to attend to subtle visual cues that determine report quality.

3. **VideoLM Backbone Architecture**:

    - Function: Provides the spatiotemporal reasoning foundation.
    - Mechanism: Input video $X \in \mathbb{R}^{T \times H \times W \times C}$ is tokenized via spatiotemporal cube tokenization into visual tokens, processed by a Transformer with factorized 3D positional embeddings, and compressed by a merger into language-aligned tokens fed to the LLM decoder. Two backbones are evaluated: Qwen2.5-VL-7B and InternVL3-8B.
    - Design Motivation: VideoLM architectures are highly analogous to CT foundation models, differing only in training domain; their spatiotemporal reasoning capabilities can thus be directly reused.

### Loss & Training

- **Stage 1**: Standard autoregressive cross-entropy loss, with the target sequence being the concatenation of the abnormality list and the full report $[R_{AB}; R_{Full}]$.
- **Stage 2**: DPO loss, with hyperparameter $\beta$ controlling the degree of deviation from the reference model.
- **Data Preprocessing**: Each HRCT is converted to 240 frames of 480×480 pixel slices, with HU windowing of $[-1000, 200]$, saved in MP4 format at 18 fps.
- **Training Setup**: 2× 80GB A100 GPUs, total batch size 4; the visual encoder is frozen with no LoRA fine-tuning applied.
- **Dataset**: CT-RATE training set comprising 46,717 CT scans (20,000 patients) and validation set of 3,039 scans (1,314 patients).

## Key Experimental Results

### Main Results

Comprehensive comparison on the CT-RATE benchmark, evaluating natural language generation (NLG) and clinical efficacy (CE) metrics:

| Method | BL-1 | BL-4 | RG-L | BERT | CE Micro P | CE Micro R | CE Micro F1 | CE Macro F1 | CE Wtd F1 | CE Samp F1 |
|------|------|------|------|------|-----------|-----------|------------|------------|----------|----------|
| CT2Rep | 47.91 | 28.04 | 45.43 | 88.10 | 26.39 | 10.50 | 14.10 | 10.65 | 11.35 | 10.86 |
| RadFM | 50.20 | 17.02 | 30.46 | 86.17 | 36.10 | 13.48 | 19.63 | 13.05 | 17.74 | 12.14 |
| Reg2RG | 44.89 | 21.08 | 24.41 | 86.18 | 28.47 | 11.06 | 15.93 | 10.48 | 14.51 | 12.19 |
| CT-CHAT | 42.81 | 17.63 | 32.50 | 86.35 | 25.13 | 37.48 | 30.08 | 21.66 | 28.35 | 25.31 |
| M3D-8B | 44.95 | 22.98 | 37.76 | 87.52 | 47.60 | 28.54 | 35.69 | 26.74 | 33.13 | 25.21 |
| Qwen2.5-VL-7B | 43.67 | 21.25 | 36.71 | 87.30 | 48.06 | 25.88 | 33.64 | 25.57 | 32.19 | 24.95 |
| InternVL3-8B | 45.57 | 22.05 | 38.49 | 87.40 | 53.57 | 37.99 | 44.45 | 38.91 | 43.28 | 32.14 |
| M3D-AbSteer | 45.22 | 23.09 | 38.58 | 87.83 | 44.95 | 41.66 | 43.24 | 36.18 | 41.89 | 36.54 |
| Qwen2.5-VL-AbSteer | 45.64 | 21.40 | 37.99 | 87.13 | 49.15 | 43.22 | 45.99 | 37.90 | 44.05 | 37.39 |
| **InternVL3-AbSteer** | **48.32** | **23.58** | **40.49** | 87.59 | **57.88** | **51.58** | **54.55** | **47.66** | **52.80** | **44.80** |

### Ablation Study

**AbSteering Strategy Ablation** (based on InternVL3-8B):

| Configuration | CE Micro P | CE Micro R | CE Micro F1 |
|------|-----------|-----------|------------|
| Baseline (no steering) | 53.57 | 37.99 | 44.45 |
| + CoT (Stage 1) | — | ↑↑ | ↑ |
| + CoT + DPO (full AbSteering) | 57.88 | 51.58 | 54.55 |

CoT substantially improves recall, while DPO further improves precision and suppresses hallucinations. Their synergy lifts F1 from 44.45 → 54.55 (+22.7%).

**Visual Encoder Ablation** (based on Qwen2.5-VL + Stage 1 CoT):

| Encoder Strategy | Effect |
|-----------|------|
| Training from scratch (no pre-training) | Dramatic performance drop |
| Frozen pre-trained encoder | Optimal |
| LoRA fine-tuning (rank=8) | No additional gain |

**LLM Scale Ablation**:

| LLM Scale | Trend |
|----------|------|
| 3B | Baseline |
| 7B | Performance improves |
| 32B | Performance degrades |

### Key Findings

- **General-purpose VideoLMs exhibit strong transferability**: InternVL3-8B without steering achieves a CE Micro F1 of 44.45, already surpassing the strongest specialized foundation model M3D-8B at 35.69.
- **AbSteering yields substantial gains**: InternVL3 with AbSteering improves CE Micro F1 from 44.45 → 54.55 (+22.7%) and CE Macro F1 from 38.91 → 47.66 (+22.5%).
- **Cross-model generality**: AbSteering proves effective for both M3D (specialized model) and the two VideoLMs, with larger gains observed for VideoLMs.
- **Video pre-training is critical**: Training from scratch leads to dramatic performance degradation; freezing the encoder suffices, and LoRA fine-tuning provides no additional benefit—indicating that spatiotemporal features from general video pre-training are sufficiently robust.
- **Larger LLMs are not always better**: Performance degrades from 7B to 32B, suggesting the current bottleneck lies in visual-text alignment rather than LLM capacity.
- **VideoLMs achieve the highest recall without increasing hallucinations.**

## Highlights & Insights

1. **Successful validation of a cross-modal transfer paradigm**: This work systematically demonstrates that spatiotemporal reasoning capabilities from general video pre-training can be efficiently transferred to 3D medical imaging, offering a data-efficient and computationally friendly alternative to training modality-specific foundation models from scratch.
2. **Targeted two-stage design**: CoT addresses the recall problem of "missing abnormalities" (by enforcing reasoning before generation), while DPO addresses the precision problem of "failing to distinguish abnormalities" (through clinically confusable hard-negative contrastive learning). The synergistic effect of the two stages is pronounced.
3. **Implications of the frozen encoder**: The finding that LoRA fine-tuning yields no additional gain is surprising, suggesting that the visual features of VideoLMs are already sufficiently generalizable, and that domain adaptation is more critical at the language guidance level than at the visual representation level.
4. **Structured CoT dataset contribution**: CT-RATE-AB reorganizes raw reports into a region-abnormality format with manual verification, facilitating future community research.

## Limitations & Future Work

- **Single-dataset evaluation**: Evaluation is conducted solely on CT-RATE (thoracic CT), leaving generalization to other anatomical regions (abdominal, cranial, etc.) unverified.
- **Dependence on GPT-4o**: Both report structuring and hard-negative construction rely on GPT-4o, introducing additional cost and potential bias while potentially limiting reproducibility.
- **Information loss from CT-to-MP4 conversion**: Mapping HU values to video format inevitably sacrifices the precision of CT-specific density information.
- **Bottleneck with large-scale LLMs**: Performance degradation at 32B suggests that the current visual-text alignment strategy may require additional design considerations at larger scales.
- **Distance from clinical deployment**: Evaluation remains based on automated metrics (RadBERT classifier), without radiologist human evaluation.

## Related Work & Insights

- **CT Report Generation**: CT2Rep first established a benchmark for direct report generation from 3D CT; M3D and CT-CHAT explored specialized 3D medical foundation model approaches; Reg2RG introduced a region-guided referring and grounding mechanism. This paper contributes by demonstrating that general-purpose VideoLMs with appropriate guidance can surpass these specialized models.
- **Medical Applications of VideoLMs**: This work is among the first to systematically study the transfer of VideoLMs to 3D medical imaging, inspiring a new paradigm of reusing large-scale pretraining knowledge from video understanding for medical applications.
- **DPO in Medical Contexts**: Applying the hard-negative construction strategy of DPO to medical report generation is a novel contribution; clinically confusable abnormalities as negatives prove more effective than random negatives.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework](../../ACL2026/medical_imaging/region-grounded_report_generation_for_3d_medical_imaging_a_fine-grained_dataset_.md)
- [\[CVPR 2026\] Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](prototype-based_knowledge_guidance_for_fine-grained_structured_radiology_reporti.md)
- [\[AAAI 2026\] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](../../AAAI2026/medical_imaging/unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [\[CVPR 2026\] OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](orapo_oracle_rl_radiology_report_generation.md)
- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)

</div>

<!-- RELATED:END -->
