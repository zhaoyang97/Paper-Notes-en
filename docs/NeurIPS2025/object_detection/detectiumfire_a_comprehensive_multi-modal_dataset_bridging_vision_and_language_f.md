---
title: >-
  [Paper Note] DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding
description: >-
  [NeurIPS 2025][Object Detection][Fire Detection] DetectiumFire constructs the largest multi-modal fire understanding dataset — 14.5K real images + 2.5K videos + 8K synthetic images + 12K RLHF preference pairs — with a lo…
tags:
  - "NeurIPS 2025"
  - "Object Detection"
  - "Fire Detection"
  - "Multi-modal Dataset"
  - "Synthetic Data"
  - "RLHF"
  - "Vision-Language Model"
date: 2026-05-08
content_hash: 608ff3b5754e2231
---

# DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding

**Conference**: NeurIPS 2025  
**arXiv**: [2511.02495](https://arxiv.org/abs/2511.02495)  
**Code**: [https://kaggle.com/datasets/detectiumfire](https://kaggle.com/datasets/detectiumfire)  
**Area**: Dataset / Multi-modal  
**Keywords**: Fire Detection, Multi-modal Dataset, Synthetic Data, RLHF, Vision-Language Model

## TL;DR
DetectiumFire constructs the largest multi-modal fire understanding dataset — 14.5K real images + 2.5K videos + 8K synthetic images + 12K RLHF preference pairs — with a low duplication rate (0.03 PHash vs. D-Fire 0.15), a 4-level severity classification scheme, and detailed scene descriptions. Fine-tuning YOLOv11m achieves mAP 43.74, and fine-tuning LLaMA-3.2-11B yields 83.84% accuracy on fire severity classification.

## Background & Motivation

**Background**: Fire safety is a critical global concern, yet existing fire datasets are small (D-Fire contains only 5.8K images) and exhibit high duplication rates (CNN duplication rate 0.55). Multi-modal models (CLIP, VLMs) lack fire-domain training data.

**Limitations of Prior Work**: (a) Existing datasets have high duplication rates, causing models to overfit on repeated samples rather than learning generalizable features; (b) semantic annotations are absent (what is burning? what is the environment? how severe?) — only bounding boxes are provided; (c) synthetic data quality is poor (FLAME_SD mAP only 2.10).

**Key Challenge**: Fire scenes require contextual reasoning (e.g., a small candle vs. a spreading blaze), yet existing datasets do not support such understanding — detection alone is insufficient; scene semantics and severity assessment are needed.

**Goal**: To construct a large-scale, low-duplication, multi-modal fire understanding dataset supporting detection, description, and severity assessment.

**Key Insight**: Combining real image collection + SFT/RLHF fine-tuned Stable Diffusion synthesis + GPT-4o semantic annotation + a 4-level severity classification scheme.

**Core Idea**: Low-duplication real images + SFT/RLHF fine-tuned SD synthesis + GPT-4o semantic annotation + 4-level severity classification = a multi-modal fire understanding benchmark.

## Method

### Overall Architecture
**Data Collection**: Multi-source collection of 14.5K images + 2.5K videos → deduplication (PHash + CNN imagededup) → **Annotation**: Roboflow bounding boxes + GPT-4o generated 75-word descriptions (burning object + environment + severity) → **Synthetic Data**: SFT fine-tuning of SD v1.5/v2/XL-1.0 + RLHF (Diffusion-DPO) → **Quality Control**: CLAP embedding cosine distance filtering + fire safety expert validation.

### Key Designs

1. **4-Level Severity Classification Scheme**:

    - Function: Assign a severity level to each fire image.
    - Mechanism: No Risk → Low Risk (small, controllable fire) → Medium Risk (moderate spreading) → High Risk (large-scale, uncontrollable). Each level is associated with specific visual feature descriptions.
    - Design Motivation: A binary "fire / no fire" label is insufficient — fire suppression decisions require severity assessment.

2. **SFT + RLHF Synthetic Data**:

    - Function: Generate high-quality fire images using fine-tuned Stable Diffusion.
    - Mechanism: SFT — LoRA fine-tuning of SD v1.5/v2/XL-1.0 for 4,000 steps. RLHF — Diffusion-DPO pipeline with 12K human preference pairs (pairwise comparisons with $k=2$–$9$ per prompt).
    - Design Motivation: Real fire data is scarce and difficult to collect safely. SFT-generated images achieve significantly higher quality than baseline SD (as measured by Elo ratings).

3. **GPT-4o Semantic Annotation Pipeline**:

    - Function: Generate structured descriptions for each image.
    - Mechanism: 75-word limit focusing on three elements: burning objects (e.g., building / forest / vehicle), environment (indoor / outdoor / time of day), and severity. Manual refinement and error correction are applied.
    - Design Motivation: High-quality captions are required for VLM fine-tuning — simple category labels are insufficient; detailed scene descriptions are necessary.

### Loss & Training
- Detection: Standard YOLOv11m training.
- VLM: LLaMA-3.2-11B instruction fine-tuning.
- Combining synthetic and real images during training yields better results.

## Key Experimental Results

### Main Results

| Task | Method | Metric | Value |
|------|------|------|-----|
| Detection | YOLOv11m (DetectiumFire) | mAP | 43.74±0.64 |
| Cross-domain Detection | Train DetectiumFire → Test D-Fire | mAP | 40.32 |
| Cross-domain Detection | Train D-Fire → Test DetectiumFire | mAP | 24.88 (poor) |
| Synthetic Augmentation | Real + Synthetic | mAP | 44.52 (+0.78) |
| VLM Severity | LLaMA-3.2-11B fine-tuned | Accuracy | 83.84% |
| VLM Environment | LLaMA-3.2-11B fine-tuned | Accuracy | 89.39% |
| VLM Burning Object | LLaMA-3.2-11B fine-tuned | Accuracy | 87.37% |
| VLM Baseline | LLaMA-3.2-11B (no fine-tuning) | Severity | 56.06% (+27.78%) |

### Ablation Study

| Data Source | mAP |
|----------|-----|
| Real only | 43.74 |
| Synthetic only (FLAME_SD) | 2.10 |
| Synthetic only (SFT) | 33.50 |
| **Real + SFT Synthetic** | **44.52** |

### Key Findings
- Dataset quality (low duplication) is critically important — models trained on D-Fire transfer to DetectiumFire with only 24.88 mAP (vs. 40.32 in the reverse direction), demonstrating that DetectiumFire is more challenging and diverse.
- SFT synthetic data outperforms other synthetic approaches by an order of magnitude (33.50 vs. 2.10 mAP) — LoRA fine-tuning is effective.
- RLHF synthetic data slightly underperforms SFT — possibly due to reduced diversity (preference pairs may be biased toward common scenes).
- Semantic descriptions substantially improve VLM fine-tuning (severity +27.78%) — contextual reasoning requires detailed annotations.
- Synthetic data provides effective augmentation (+0.78 mAP), though the gain is marginal.

## Highlights & Insights
- The **4-level severity classification** fills the gap between fire detection and fire assessment in fire safety AI — practically useful systems require severity judgment.
- The **low duplication rate** (0.03 vs. 0.15) demonstrates that data deduplication is essential for benchmark quality.
- The **cross-domain asymmetry** (DetectiumFire→D-Fire: 40.32 vs. D-Fire→DetectiumFire: 24.88) confirms the superior diversity and difficulty of DetectiumFire.

## Limitations & Future Work
- Gains from synthetic data are marginal (+0.78 mAP); more advanced generation strategies (e.g., ControlNet-based conditional generation) may be needed.
- Linguistic bias (primarily English and Chinese search queries) may result in missing fire scenes annotated in other languages.
- Richer scene annotations are lacking, such as human presence and fire progression (temporal dynamics).
- The 4-level severity scheme remains coarse — 10-level or continuous assessment may be more suitable for professional firefighting applications.
- RLHF synthesis underperforms SFT — preference pairs may be biased toward common scenes, reducing diversity.

## Related Work & Insights
- **vs. D-Fire**: Small scale, high duplication rate, no semantic annotations; DetectiumFire comprehensively surpasses it.
- **vs. FLAME_SD**: A purely synthetic dataset of extremely low quality (mAP 2.10); DetectiumFire's hybrid strategy is substantially more effective.
- **Application Value**: Scenarios such as firefighting robots, UAV inspection, and intelligent surveillance all require fire severity assessment capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐ First large-scale multi-modal fire understanding dataset with severity classification.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional evaluation covering detection, VLM, synthetic data, and cross-domain transfer.
- Writing Quality: ⭐⭐⭐⭐ Dataset construction pipeline is described in detail.
- Value: ⭐⭐⭐⭐ Provides urgently needed data infrastructure for fire safety AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Visual Modality Prompt for Adapting Vision-Language Object Detectors](../../ICCV2025/object_detection/visual_modality_prompt_for_adapting_vision-language_object_detectors.md)
- [\[CVPR 2026\] Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](../../CVPR2026/object_detection/mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)
- [\[AAAI 2026\] SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection](../../AAAI2026/object_detection/sm3det_a_unified_model_for_multi-modal_remote_sensing_object_detection.md)
- [\[NeurIPS 2025\] MSTAR: Box-Free Multi-Query Scene Text Retrieval with Attention Recycling](mstar_box-free_multi-query_scene_text_retrieval_with_attention_recycling.md)
- [\[CVPR 2026\] Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](../../CVPR2026/object_detection/learning_multi-modal_prototypes_for_cross-domain_few-shot_object_detection.md)

</div>

<!-- RELATED:END -->
