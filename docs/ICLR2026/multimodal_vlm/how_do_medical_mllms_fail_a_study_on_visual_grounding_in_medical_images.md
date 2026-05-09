---
title: >-
  [Paper Note] How Do Medical MLLMs Fail? A Study on Visual Grounding in Medical Images
description: >-
  [ICLR 2026][Multimodal VLM][Medical VQA] This work presents the first systematic diagnosis revealing that the root cause of poor zero-shot medical VQA performance in medical MLLMs is insufficient visual grounding—model attention systematically deviates from clinically relevant regions. Building on this finding, the authors propose VGRefine, a training-free inference-time attention correction method that achieves state-of-the-art results across 110K+ samples on 6 benchmarks spanning 8 imaging modalities.
tags:
  - ICLR 2026
  - Multimodal VLM
  - Medical VQA
  - Visual Grounding
  - Attention Analysis
  - MLLM Failure Modes
  - Inference-Time Correction
date: 2026-05-08
content_hash: 61af9fd67d17ab9d
---

# How Do Medical MLLMs Fail? A Study on Visual Grounding in Medical Images

**Conference**: ICLR 2026
**arXiv**: [2603.14323](https://arxiv.org/abs/2603.14323)
**Code**: [Project Page](https://guimeng-leo-liu.github.io/Medical-MLLMs-Fail/)
**Area**: Multimodal VLM
**Keywords**: Medical VQA, Visual Grounding, Attention Analysis, MLLM Failure Modes, Inference-Time Correction

## TL;DR

This work presents the first systematic diagnosis revealing that the root cause of poor zero-shot medical VQA performance in medical MLLMs is insufficient visual grounding—model attention systematically deviates from clinically relevant regions. Building on this finding, the authors propose VGRefine, a training-free inference-time attention correction method that achieves state-of-the-art results across 110K+ samples on 6 benchmarks spanning 8 imaging modalities.

## Background & Motivation

**Background** Multimodal large language models (MLLMs) have demonstrated strong performance on general vision-language tasks. A growing body of work has extended these models to the medical domain (LLaVA-Med, HuatuoGPT-Vision, VILA-M3, etc.), aiming to build general-purpose medical AI capable of supporting diverse clinical decision-making. Nevertheless, these models continue to underperform in zero-shot medical VQA settings, particularly when no downstream task samples are available for training or fine-tuning.

**Limitations of Prior Work** Existing research has largely focused on *how to improve* medical MLLMs—through larger medical multimodal datasets or integration of external expert models—while systematically neglecting the more fundamental question of *why they fail*. Failure on medical images may stem from semantic grounding deficiencies (not knowing which clinical concept to attend to) or visual grounding deficiencies (knowing what to look for but failing to localize it correctly in the image). These two dimensions have not previously been explicitly distinguished or quantified.

**Key Challenge** Prior work has shown that MLLMs exhibit strong visual grounding on natural images, with attention distributions aligning well with target regions. Whether this holds for medical images remains an open question. If visual grounding is indeed the bottleneck for medical MLLMs, then the prevailing effort to inject medical knowledge for semantic grounding enhancement may be misdirected—the true bottleneck lies on the visual side.

**Key Insight** This paper proposes a complete diagnose–validate–fix pipeline: collaborating with clinical experts to construct the VGMED dataset specifically for visual grounding analysis, introducing new quantitative metrics, systematically evaluating 8 state-of-the-art medical MLLMs, and finally proposing an efficient training-free inference-time correction method.

**Core Idea** By decoupling semantic grounding from visual grounding to precisely localize failure modes, this work demonstrates that insufficient visual grounding is a universal and correctable bottleneck shared across all mainstream medical MLLMs.

## Method

### Overall Architecture

The work consists of three components: (1) constructing the VGMED dataset for precise visual grounding evaluation; (2) proposing three quantitative metrics—Attention Ratio (AR), KL divergence, and JS divergence—to comprehensively diagnose 8 medical MLLMs; and (3) proposing VGRefine, an inference-time method that improves visual grounding through a two-step attention correction procedure.

### Key Designs

1. **VGMED Dataset — An Evaluation Tool for Decoupling Visual and Semantic Grounding**

    - **Function**: Construct an evaluation dataset specifically designed for visual grounding analysis, ensuring that every question can only be answered by referring to annotated image regions.
    - **Mechanism**: Designed in collaboration with three licensed clinicians (including a senior clinician with over 10 years of experience). A total of 13,962 samples are curated from 40+ public medical segmentation datasets, with segmentation masks converted to bounding boxes. Two question types are generated: localization questions (identifying the organ or lesion in a region) and attribute questions (size, shape, abnormality, etc.), yielding approximately 28K image–bbox–question triplets. Questions are generated by GPT-4 and reviewed by clinical experts to ensure both clinical relevance and the necessity of visual grounding.
    - **Design Motivation**: Existing Med-VQA datasets conflate questions that require no region-specific localization (e.g., "What is the imaging modality?") with questions demanding deep medical knowledge, making it impossible to isolate visual grounding failures from semantic grounding failures.

2. **Visual Grounding Quantification — New Metrics Beyond Attention Ratio**

    - **Function**: Propose KL divergence and JS divergence as metrics to measure the alignment between attention maps and ground-truth regions, complementing the conventional Attention Ratio (AR).
    - **Mechanism**: AR only captures the total attention mass within a bounding box, ignoring how attention is distributed within the region. KL/JS divergence treat both the attention map and the region mask as probability distributions and measure the discrepancy between them—lower divergence indicates more uniform attention coverage of the entire clinically relevant region. Concretely, spatial attention maps are extracted from cross-attention weights between the last text token and $N^2$ image tokens across LLM layers, and compared against the normalized bounding box mask.
    - **Design Motivation**: Since VGMED questions are deliberately designed to require attention over the entire bounding box region, it is insufficient to merely check whether attention "enters" the box—one must also verify whether attention uniformly covers the region.

3. **VGRefine — Training-Free Inference-Time Attention Correction**

    - **Function**: Improve visual grounding in medical MLLMs at inference time through a two-step procedure, requiring no additional training.
    - **Mechanism**: **Step 1 (Attention Triage)** identifies the top-$K$ attention heads most relevant to visual grounding at the head level, evaluated via KL divergence on COCO natural images to avoid medical data leakage. These heads' attention maps are aggregated, low-activation regions are suppressed, and a binary mask is generated. **Step 2 (Attention Knockout)** uses this mask to block attention connections from question tokens to irrelevant visual tokens, forcing the model to attend only to semantically meaningful regions.
    - **Design Motivation**: Diagnosis reveals that while medical MLLM attention partially covers relevant regions, it simultaneously disperses substantially over irrelevant areas—an explicit mechanism is needed to "knock out" these distracting attention weights. Top-$K$ heads are selected on COCO because experiments show that attention heads most relevant to visual grounding on natural images remain relevant on medical images, despite overall lower grounding quality, while also preventing data leakage.

### Loss & Training

VGRefine is entirely training-free. Hyperparameters include top-$K = 20$ attention heads and an activation percentile threshold of $p = 50\%$ for magnitude filtering. Attention knockout is applied at layer 16 for 7B models and layers 34–36 for 34B models. All hyperparameters are kept consistent across all benchmarks.

## Key Experimental Results

### Visual Grounding Diagnosis (Core Findings)

| Metric | Medical Images | Natural Images | Conclusion |
|--------|---------------|---------------|------------|
| Attention Ratio (AR) ↑ | Low | High | All 8 SOTA medical MLLMs show attention systematically deviating from clinical regions |
| KL Divergence ↓ | High | Low | Large discrepancy between attention distribution and GT region |
| JS Divergence ↓ | High | Low | Same as above; consistent across layers and models |

Note: General-purpose MLLMs (LLaVA-v1.5) also fail at visual grounding on medical images; medical MLLMs exhibit normal grounding on natural images → the problem is domain-specific, not a model capability deficiency.

### Main Results (Med-VQA Performance)

| Model | VQA-RAD | SLAKE | PathVQA | PMC-VQA | Weighted Avg. |
|-------|---------|-------|---------|---------|--------------|
| HuatuoGPT-V-7B | 67.4% | 76.5% | 60.7% | 53.9% | 65.3% |
| **VGRefine (Ours)** | **71.2%** | **76.9%** | **67.6%** | **56.2%** | **68.4%** |

VGRefine yields consistent improvements across all 8 imaging modalities on OmniMedVQA: CT +7.5%, MRI +6.4%, X-Ray +8.1%, with the average improving from 71.3% to 74.4%. On MMMU Health & Medicine: 45.8% → 47.2%.

### Ablation Study

| Configuration | VQA-RAD | SLAKE | PathVQA | PMC-VQA | Avg. |
|--------------|---------|-------|---------|---------|------|
| $K=1$ | 68.6% | 75.8% | 64.9% | 53.7% | 68.3% |
| $K=10$ | 70.9% | 76.8% | 67.7% | 56.1% | 68.3% |
| **$K=20$** | **71.2%** | **76.9%** | **67.6%** | **56.2%** | **68.4%** |
| $p=30\%$ | 70.8% | 76.8% | 67.6% | 55.7% | 68.2% |
| **$p=50\%$** | **71.2%** | **76.9%** | **67.6%** | **56.2%** | **68.4%** |
| $p=90\%$ | 70.6% | 76.3% | 68.1% | 55.5% | 68.2% |

### Key Findings

- All 8 SOTA medical MLLMs fail at visual grounding on medical images without exception, yet exhibit normal grounding on natural images—confirming this is a medical-domain-specific problem.
- VGRefine achieves consistent improvements without any medical knowledge injection—such gains would be impossible if visual grounding were not the limiting factor.
- Human evaluation: 5 clinicians preferred VGRefine-generated attention maps in 76% of 20 blinded cases.
- Compared with three recent attention methods (PAI, AdaptVis, ViCrop), VGRefine achieves the largest and most consistent improvements.

## Highlights & Insights

- The *diagnose before treat* research paradigm is highly valuable: by precisely decoupling semantic and visual grounding, the work identifies a universal cross-model bottleneck and provides a clear direction for all subsequent improvement efforts.
- The medical vs. natural image comparative analysis constitutes the most compelling evidence in the paper—the same model's drastically different behavior on the two image types rules out any explanation based on general model capability limitations.
- VGRefine embodies a "less is more" philosophy: without introducing new knowledge, correcting the attention distribution alone achieves state-of-the-art performance, suggesting that existing models already possess sufficient medical knowledge and that attention misdirection is the primary failure mode.
- The cross-domain transfer design is elegant: selecting top-$K$ heads on COCO natural images yet successfully transferring to medical images simultaneously avoids data leakage and demonstrates the domain-invariance of visual grounding mechanisms.

## Limitations & Future Work

- The analysis focuses exclusively on visual grounding as a failure mode, leaving other potential bottlenecks—such as semantic grounding deficiencies or reasoning capability limitations—unexplored.
- VGRefine relies on a fixed offline projection and does not adapt dynamically to input; adaptive attention correction may be needed for different types of medical images.
- The selection of attention layers for knockout (e.g., layer 16 for 7B models) is model-specific, and the optimal layer may differ across architectures.
- It remains unverified whether visual grounding failures persist in newer closed-source models such as GPT-4V and Gemini.

## Related Work & Insights

- **Comparison with Zhang2025 (MLLMs Know)**: The latter demonstrates that MLLMs exhibit strong visual grounding on natural images; this work extends that analysis to the medical domain and arrives at the opposite conclusion.
- **Complementarity with VILA-M3 and similar approaches**: Methods such as VILA-M3 augment models externally via expert models, while VGRefine corrects attention internally—the two approaches are complementary and potentially combinable.
- VGRefine's attention knockout strategy is consistent with the attention manipulation literature, but its key innovation lies in automatically selecting the most relevant subset of heads rather than manually specifying them.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First systematic diagnosis of visual grounding failure modes in medical MLLMs, filling an important gap in the field's understanding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 8 models × 28K diagnostic analyses + 6 benchmarks × 110K+ validation samples + human evaluation + multi-method comparison.
- **Writing Quality**: ⭐⭐⭐⭐ The problem–diagnosis–solution narrative is logically coherent and complete; clinical expert collaboration in dataset construction enhances credibility.
- **Value**: ⭐⭐⭐⭐⭐ A fundamental insight for the medical AI field—all future medical MLLM improvements must account for visual grounding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Medic-AD: Towards Medical Vision-Language Model's Clinical Intelligence](../../CVPR2026/multimodal_vlm/medic-ad_towards_medical_vision-language_models_clinical_intelligence.md)
- [\[CVPR 2026\] Vision-Language Models Encode Clinical Guidelines for Concept-Based Medical Reasoning](../../CVPR2026/multimodal_vlm/vision-language_models_encode_clinical_guidelines_for_concept-based_medical_reas.md)
- [\[AAAI 2026\] Explore How to Inject Beneficial Noise in MLLMs](../../AAAI2026/multimodal_vlm/explore_how_to_inject_beneficial_noise_in_mllms.md)
- [\[CVPR 2026\] Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning](../../CVPR2026/multimodal_vlm/similarity-as-evidence_calibrating_overconfident_vlms_for_interpretable_and_labe.md)
- [\[ICLR 2026\] Grounding-IQA: Grounding Multimodal Language Models for Image Quality Assessment](grounding-iqa_grounding_multimodal_language_model_for_image_quality_assessment.md)

</div>

<!-- RELATED:END -->
