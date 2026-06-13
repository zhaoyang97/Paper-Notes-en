---
title: >-
  [Paper Note] InfoDet: A Dataset for Infographic Element Detection
description: >-
  [ICLR 2026][Object Detection][Infographic Detection] This paper introduces a large-scale infographic element detection dataset (101,264 infographics…
tags:
  - "ICLR 2026"
  - "Object Detection"
  - "Infographic Detection"
  - "Chart Understanding"
  - "Dataset"
  - "Grounded CoT"
  - "VLM"
date: 2026-05-08
content_hash: aa51a724f7a3ad01
---

# InfoDet: A Dataset for Infographic Element Detection

**Conference**: ICLR 2026
**arXiv**: [2505.17473](https://arxiv.org/abs/2505.17473)  
**Code**: [https://github.com/InfoDet2025/InfoDet](https://github.com/InfoDet2025/InfoDet)  
**Area**: Object Detection / Document Understanding
**Keywords**: Infographic Detection, Chart Understanding, Dataset, Grounded CoT, VLM

## TL;DR
This paper introduces a large-scale infographic element detection dataset (101,264 infographics, 14.2 million annotations) spanning two major categories—chart elements and human-recognizable objects (HROs)—and proposes a Grounded CoT method that leverages detection results to enhance VLM chart understanding.

## Background & Motivation

**Background**: Chart understanding is an important application scenario for VLMs (e.g., ChartQA), yet existing approaches have VLMs reason directly from raw images, overlooking structured visual element information.

**Limitations of Prior Work**: (a) No large-scale infographic detection dataset exists—state-of-the-art foundation models (DINO-X, Grounding DINO) achieve AP < 15% on infographic element detection, essentially failing entirely; (b) Infographics contain numerous non-natural-scene elements (e.g., icons, chart components) that exhibit a large domain gap from detectors trained on COCO/Objects365.

**Key Challenge**: Element detection in infographics is foundational to chart understanding, yet current detectors are practically unusable in this domain.

**Goal**: (a) Construct a large-scale infographic detection dataset, and (b) validate how detection results can improve VLM chart reasoning.

**Key Insight**: Combining synthetic data (90,000 images, template-based generation) with real data (11,264 images, model-in-the-loop annotation) to cover 75 chart types.

**Core Idea**: Treat element detection as "visual grounding" for chart understanding—detect first, then reason (Thinking-with-Boxes).

## Method

### Overall Architecture
The work consists of two components: (1) construction of the InfoDet dataset, and (2) the Grounded CoT method, which injects detected elements into VLMs as visual and textual prompts to augment chart reasoning.

### Key Designs

1. **Dataset Construction**:

    - Synthetic data (90,000 images): Data is sampled from VizNet's 31 million tables and rendered into infographics via 1,072 design templates; Chart and HRO annotations are extracted programmatically from SVGs, fully automated.
    - Real data (11,264 images): Collected from 10 platforms, deduplicated using CLIP similarity scores and quality-verified with GPT-4o. Annotation employs iterative model-in-the-loop refinement—a detector is first trained on synthetic data, applied to real images, expert-corrected annotations are fed back to improve the detector, and this cycle is repeated over multiple rounds.
    - Final quality: precision 93.9%, recall 96.7%, comparable to COCO/Objects365.

2. **Grounded Chain-of-Thought (Thinking-with-Boxes)**:

    - Function: Detected elements are provided as auxiliary inputs to the VLM to guide reasoning.
    - Mechanism: (a) Visual prompting—detected bounding boxes are overlaid on the image and labeled with letters, using a two-layer separation strategy (chart layer + text layer) to avoid overlapping confusion; (b) Textual description—each element's attributes are enumerated in text. The VLM is then prompted to reason step by step (CoT) by referencing the labeled elements.
    - Design Motivation: VLMs tend to miss or confuse elements when reasoning over complex charts (multi-chart infographics); explicit detection results provide structured visual cues.

### Loss & Training
Detectors (Co-DETR, RTMDet) are trained on InfoDet following standard pipelines. VLMs require no additional fine-tuning; Grounded CoT is a training-free inference enhancement.

## Key Experimental Results

### Detection Results

| Model | Pretrain | Chart AP | HRO AP | Chart AR | HRO AR |
|-------|----------|----------|--------|----------|--------|
| Co-DETR | Zero-shot | 0.4% | 1.1% | 5.6% | 4.8% |
| Co-DETR | InfoDet | **81.8%** | **64.5%** | **88.2%** | **76.8%** |

### Grounded CoT Results (ChartQAPro Benchmark, Relaxed Accuracy)

| Model | Method | Infographic Single | Infographic Multi | Overall |
|-------|--------|--------------------|-------------------|---------|
| o1 | Direct | 66.4% | 66.0% | 61.4% |
| o1 | CoT | 64.3% | 67.6% | 61.9% |
| o1 | **Grounded CoT** | **67.8%** | **71.9%** | **64.1%** |

### Ablation Study

| Grounded CoT Component | Accuracy |
|------------------------|----------|
| Visual prompts only | 62.8% |
| Text description only | 61.6% |
| Combined (single-layer) | 62.3% |
| **Combined (two-layer)** | **64.1%** |

### Key Findings
- Zero-shot detectors nearly fail on infographics (AP < 1.1%), demonstrating that the dataset fills a critical gap for detectors in the infographic domain.
- After InfoDet pretraining, AP rises to 81.8%, and the learned representations transfer to other document understanding tasks (Rico +8.5 AP, DocGenome +5.4 AP).
- Grounded CoT improves accuracy by 3–6% on infographic scenarios, with limited gains on simpler charts.
- The two-layer visual prompt separation outperforms single-layer by 1.8%, avoiding confusion caused by overlapping boxes and letter labels.

## Highlights & Insights
- **Filling a Critical Data Gap**: A large-scale infographic detection dataset with 14.2 million annotations constitutes a significant resource contribution to the field.
- **Thinking-with-Boxes Paradigm**: The detect-then-reason approach is simple yet effective, akin to equipping VLMs with a "magnifying glass." It is transferable to arbitrary visual reasoning tasks.
- **Synthetic + Real Data Construction**: Template-based synthesis (automatic annotation) combined with model-in-the-loop labeling (efficient real-data annotation) balances scale and quality.

## Limitations & Future Work
- A domain gap between synthetic and real data persists (synthetic data is simpler), necessitating more real-world samples.
- HRO detection AP (64.5%) is substantially lower than Chart AP (81.8%), indicating that icon detection remains more challenging.
- Grounded CoT yields marginal improvements on simple charts and may introduce information overload.
- The two-layer separation strategy is hand-engineered; more adaptive layout strategies warrant further exploration.

## Related Work & Insights
- **vs. ChartQA/ChartQAPro**: These works provide QA benchmarks, on which this paper validates Grounded CoT.
- **vs. Grounding DINO**: Zero-shot failure on infographics demonstrates the necessity of domain-specific data.
- **vs. DocGenome**: A document layout detection dataset; InfoDet pretraining transfers favorably and improves its performance.

## Rating
- Novelty: ⭐⭐⭐⭐ The dataset and Grounded CoT task formulation are novel, though the method itself is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Detection, chart understanding, and transfer learning are all comprehensively evaluated.
- Writing Quality: ⭐⭐⭐⭐⭐ Dataset construction is described with exceptional detail.
- Value: ⭐⭐⭐⭐⭐ The large-scale dataset and open-source release offer extremely high community value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection](forestpersons_a_large-scale_dataset_for_under-canopy_missing_person_detection.md)
- [\[ICCV 2025\] Kaputt: A Large-Scale Dataset for Visual Defect Detection](../../ICCV2025/object_detection/kaputt_a_large-scale_dataset_for_visual_defect_detection.md)
- [\[NeurIPS 2025\] BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes](../../NeurIPS2025/object_detection/burstdeflicker_a_benchmark_dataset_for_flicker_removal_in_dynamic_scenes.md)
- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](../../CVPR2026/object_detection/mmrad_multimodal_anomaly_detection.md)
- [\[CVPR 2026\] SteelDefectX: A Coarse-to-Fine Vision-Language Dataset and Benchmark for Generalizable Steel Surface Defect Detection](../../CVPR2026/object_detection/steeldefectx_a_coarse-to-fine_vision-language_dataset_and_benchmark_for_generali.md)

</div>

<!-- RELATED:END -->
