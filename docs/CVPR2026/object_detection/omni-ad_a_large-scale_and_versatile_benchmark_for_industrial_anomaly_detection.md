---
title: >-
  [Paper Note] Omni-AD: A Large-scale and Versatile Benchmark for Industrial Anomaly Detection
description: >-
  [CVPR 2026][Object Detection][Industrial Anomaly Detection] Omni-AD is an Industrial Anomaly Detection (IAD) benchmark collected from real production lines, covering 150 categories across 16 industries with approximately 35,000 pixel-level annotated images; it supports both traditional unsupervised IAD evaluation and, for the first time, introduces three progressive subtasks—"discrimination-classification-localization"—for Multimodal Large Language Models (MLLMs). Experiments…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "Industrial Anomaly Detection"
  - "Large-scale Benchmark"
  - "MLLM Evaluation"
  - "Visual Question Answering"
  - "Visual Grounding"
date: 2026-05-08
content_hash: e7627223874f73c3
---

# Omni-AD: A Large-scale and Versatile Benchmark for Industrial Anomaly Detection

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_Omni-AD_A_Large-scale_and_Versatile_Benchmark_for_Industrial_Anomaly_Detection_CVPR_2026_paper.html)  
**Code**: Project Page https://omni-ad.github.io  
**Area**: Object Detection / Industrial Anomaly Detection / Benchmark Dataset  
**Keywords**: Industrial Anomaly Detection, Large-scale Benchmark, MLLM Evaluation, Visual Question Answering, Visual Grounding

## TL;DR
Omni-AD is an Industrial Anomaly Detection (IAD) benchmark collected from real production lines, covering 150 categories across 16 industries with approximately 35,000 pixel-level annotated images; it supports both traditional unsupervised IAD evaluation and, for the first time, introduces three progressive subtasks—"discrimination-classification-localization"—for Multimodal Large Language Models (MLLMs). Experiments demonstrate that both existing methods and MLLMs are far from saturated on this dataset.

## Background & Motivation

**Background**: The mainstream paradigm for industrial anomaly detection is unsupervised learning—training only on normal (defect-free) images to detect unknown defects during testing, thereby eliminating expensive defect data collection and annotation. The development of this paradigm relies heavily on benchmarks; MVTec AD (5,354 images, 15 categories) catalyzed numerous algorithms, followed by VisA, Real-IAD, and others that gradually increased the scale.

**Limitations of Prior Work**: The authors identify two bottleneck issues. First is **performance saturation**: top-tier algorithms currently exceed 99% in image-level/pixel-level AUROC on MVTec AD, making it nearly impossible to distinguish between algorithms. Second is the **lack of MLLM-oriented evaluation**: while MLLMs show powerful capabilities in general domains, their systematic evaluation for industrial inspection is almost non-existent; although MMAD attempted a "multiple-choice VQA" format, it does not align with real-world inspection requirements.

**Key Challenge**: Most existing datasets are artificially constructed in labs with limited category and defect diversity, leading to a significant gap when migrating from the lab to real production lines. Furthermore, existing evaluation protocols (unsupervised + multi-choice VQA) fail to simultaneously meet the requirements of being "sufficiently discriminative" and "aligned with practical MLLM usage."

**Goal**: To construct a unified benchmark that is (i) significantly larger in scale and diversity, (ii) supports both unsupervised and MLLM protocols, and (iii) remains challenging for current SOTA methods.

**Key Insight**: Data is collected directly from real production lines (rather than synthetic defects), and MLLM capabilities are decomposed into three questions relevant to industrial sites: "Is there a defect? / What is the defect? / Where is the defect?".

**Core Idea**: By using a large-scale real-world production line dataset and three progressive MLLM subtasks (two VQA and one visual grounding), the IAD evaluation is pushed from "saturated and unsupervised-only" to "discriminative and natively MLLM-supporting."

## Method

### Overall Architecture
Omni-AD is essentially a "dataset + dual-protocol benchmark" without a new trainable model. The core work consists of four parts: **Real-world production line data collection and alignment** $\rightarrow$ **Cyclic two-stage pixel-level annotation** $\rightarrow$ **Unsupervised IAD protocol** + **MLLM three-subtask protocol (including hierarchical semantic QA generation)**. The final dataset contains 34,886 images across 150 categories and 16 industries, with a normal-to-anomaly ratio of approximately $1:1$ (16,989 normal / 17,897 abnormal), and each category contains $1-7$ types of defects. The benchmark splits evaluation into two complementary paths: the traditional unsupervised path follows the setting of "training only on normal images, testing on both," while the MLLM path organizes defect understanding into three subtasks of increasing difficulty: "Discrimination $\rightarrow$ Classification $\rightarrow$ Localization." This is a pure benchmark/dataset paper; the following details the four key designs.

### Key Designs

**1. Real-line data collection & foreground alignment: More realistic and fair comparison**

Addressing the pain point that "lab-synthesized data lacks diversity and fails in production," the authors directly collected approximately 35,000 high-resolution images from multiple real production lines, covering 150 categories across 16 industries, including metals, plastics, ceramics, paper, fabrics, and wood. Unlike synthetic data, real data has **unknown defect types and locations beforehand**. Thus, the authors defined a specific defect category set $A_{cat}$ for each category based on manufacturers' descriptions of product use and quality requirements, then manually classified the images. After collection, **foreground alignment** was performed: rigid products were aligned using template matching, while deformable/textured products were cropped to center the RoI. This step reduces the negative impact of background interference and target misalignment on models (especially reconstruction-based methods), ensuring a fairer horizontal comparison between algorithms.

**2. Cyclic two-stage pixel-level annotation: Ensuring consistency at scale**

Fine-grained pixel annotation across numerous industrial scenes is prone to errors. The authors designed a **cyclic two-stage** annotation pipeline. The first stage is **human-led annotation**: professional annotators outline defect types and polygon masks. A "control set" of 500 meticulously annotated images was used for qualification tests (requiring deviation below a strict threshold). The team was split into annotators and inspectors; images for each category were annotated in three batches, with inspectors reviewing each batch. Any batch with errors was sent back to the original annotator for correction—correcting errors while enforcing rule consistency. The second stage is **model-assisted annotation**: manually annotated data for each category was split into two halves (equal normal/abnormal) to train a supervised semantic segmentation model (e.g., U-Net) to find potential annotation errors. Annotators checked areas where model predictions and manual annotations diverged and corrected them. This "model-review, human-correction" process iterated until the divergence rate fell below a preset threshold.

**3. MLLM three-subtask protocol: Decomposing defect understanding into practical difficulties**

Existing MMAD uses multiple-choice VQA, which does not align with real needs. Omni-AD splits MLLM evaluation into three subtasks of increasing difficulty: **Defect Discrimination**—given an image or a specific box + semantically rich question, the model answers Yes/No; **Defect Classification**—selecting the correct type (or "Not any") from candidate defect types in VQA format; **Defect Localization**—modeled as visual grounding, where the model must output a JSON format `{'box':[xmin,ymin,xmax,ymax],'label':'defect type'}`. Each subtask is evaluated under **0-shot and 1-shot** settings, where 1-shot allows the model to refer to the most similar sample from the normal set as a visual guide. Discrimination/classification use VQA accuracy (image-level and box-level), while localization uses Recall / Precision / F1 at $IoU=0.5$, as MLLMs often lack reliable confidence scores.

**4. Hierarchical semantic QA generation pipeline: Transforming structured metadata into rich QA pairs**

While each sample has structured metadata, formatted text lacks the semantic richness required for MLLM evaluation. Building on the MMAD pipeline, the authors used GPT-4o for **hierarchical data augmentation** in three steps: (a) **Prior Enrichment**—constructing visual prompts (original image with mask-derived edge-highlighted defect contours + a normal image of the same product for comparison), combined with instance metadata, guiding GPT-4o to complete high-level semantics like usage scenarios, functions, defect morphology, causal mechanisms, and impacts to obtain "enriched instance priors"; (b) **Knowledge Consolidation**—aggregating enriched priors for all instances of the same product, using GPT-4o to filter out instance-specific variations like position/background noise, distilling them into consistent "product-level priors," followed by manual screening; (c) **QA Generation**—using visual prompts + instance priors + product priors as context to generate semantically rich QA pairs according to task definitions. This resulted in 12K QA data and 3K grounding data.

## Key Experimental Results

> Metrics: **I-AUROC / I-AUPR** for image-level ROC/PR area under curve (anomaly classification); **P-AUROC / P-AUPRO** for pixel-level ROC and Per-Region-Overlap area under curve (anomaly segmentation); MLLM internal **Img-Acc / Box-Acc** for VQA accuracy under image/box inputs, and localization uses Recall/Precision/F1 at $IoU=0.5$. Lower metrics indicate higher dataset difficulty.

### Main Results

| Dataset | Categories | Normal | Anomaly | Total Images | Pixel Annotation |
|---------|------------|--------|---------|--------------|------------------|
| MVTec AD | 15 | 4,096 | 1,258 | 5,354 | ✓ |
| VisA | 12 | 9,621 | 1,200 | 10,821 | ✓ |
| GoodsAD | 6 | 4,464 | 1,660 | 6,124 | ✓ |
| Real-IAD (Single-view) | 30 | 14,568 | 15,642 | 30,210 | ✓ |
| **Omni-AD** | **150** | **16,989** | **17,897** | **34,886** | ✓ |

The number of categories is an order of magnitude higher than MVTec AD (150 vs. 15) and about 12.5 times that of VisA; the total image count is approximately 6 times that of MVTec AD and 3 times that of VisA.

### Unsupervised IAD: General performance drop and clearer differentiation on Omni-AD

| Method | Paradigm | MVTec AD (I-AUROC) | Real-IAD (I-AUROC) | Omni-AD (I-AUROC) | Omni-AD (P-AUPRO) |
|--------|----------|--------------------|--------------------|--------------------|--------------------|
| PatchCore | MemoryBank | 99.1 | 90.4 | **87.8** | 81.1 |
| Dinomaly | Reconstruction | 99.6 | 89.3 | 84.5 | 71.4 |
| SimpleNet | Augmentation | 99.6 | 91.7 | 82.1 | 76.7 |
| GLASS | Augmentation | 99.9 | 92.3 | 83.1 | 74.8 |

### MLLM Three Subtasks: Decline with increasing difficulty, localization as the biggest bottleneck

| Model | Setting | Discrimination Img-Acc | Classification Img-Acc | Localization F1 |
|-------|---------|-----------------------|-----------------------|-----------------|
| Random Baseline | - | 50.00 | 20.00 | N/A |
| Human Expert | - | 95.67 | 91.67 | 79.32 |
| LLaVA-NeXT 34B | 0-shot | 59.43 | 38.84 | N/A |
| InternVL-3.5 38B | 0-shot | 57.58 | 40.70 | 10.07 |
| Qwen3-VL-Instruct 30B-A3B | 1-shot | 64.65 | 65.94 | 14.84 |
| Qwen3-VL-Thinking 30B-A3B | 1-shot | **69.38** | 64.29 | 25.64 |

### Key Findings
- **Progressive task difficulty is effective**: Performance across all models declines from "discrimination $\rightarrow$ classification $\rightarrow$ localization." In the hardest localization subtask, the best model (Qwen3-VL-30B-A3B-Thinking) achieved an F1 of only 25.64 (1-shot), far below the human F1 of 79.32, indicating that precise localization remains the "hard nut" of IAD.
- **Box-level generally outperforms Image-level**: Focusing attention on assigned regions leads to higher discrimination and classification accuracy, suggesting that providing an RoI helps MLLMs extract more meaningful features.
- **Thinking mode benefits localization**: Qwen3-VL-30B-A3B-Thinking performed significantly better in localization than the Instruct version, suggesting that MLLMs with reasoning capabilities have stronger visual grounding potential.
- **1-shot is mostly positive for discrimination/classification, but occasionally negative for localization**: Referring to normal templates helps most models perform pair-wise comparison, but for the more difficult localization task, performance slightly decreased for some methods.
- **Huge gap with humans**: Human experts lead MLLMs across all metrics, highlighting both the benchmark's difficulty and the substantial room for improvement in MLLM-based anomaly detection.

## Highlights & Insights
- **Unified "Real-line + Dual-protocol"**: By using scale and diversity, it overcomes the saturation of unsupervised IAD while natively supporting MLLM evaluation, avoiding the pitfall of "same old evaluation on a bigger dataset."
- **Three subtasks map directly to real-world questions** (Is it there/What is it/Where is it?), and specifically modeling localization as grounding JSON exposes the performance gap where MLLMs "can judge but cannot locate precisely," offering greater diagnostic value than multi-choice VQA.
- **Hierarchical QA generation "semanticizes" structured annotations**: The "instance prior $\rightarrow$ product prior" two-level integration for QA generation is transferable to any vertical domain benchmark (e.g., medical, remote sensing) where structured metadata exists without semantic descriptions.
- **Cyclic two-stage annotation + control set qualification** provides a practical engineering paradigm for maintaining quality in large-scale pixel annotation.

## Limitations & Future Work
- **Essentially a benchmark, not a method**: The paper provides a "harder and more comprehensive ruler" rather than a new detection algorithm.
- **MLLM evaluation relies on GPT-4o for QA generation**: Quality is subject to GPT-4o's capabilities and prompt design. Although manually screened, there is potential bias in "using a strong MLLM to test others" (unquantified in the paper).
- **Localization evaluation uses a single $IoU=0.5$ threshold for R/P/F1**, which may be strict for "slightly offset but semantically correct" predictions; variance in MLLM output format compliance (e.g., LLaVA-NeXT) can also exaggerate score gaps.
- **Future Directions**: Supplementing few-shot/fine-tuned MLLM upper bounds, adding multi-view or multi-illumination settings, and conducting controlled experiments on QA generation bias.

## Related Work & Insights
- **vs MVTec AD / VisA / Real-IAD**: These are classic unsupervised IAD rulers. However, MVTec AD is saturated, and VisA/Real-IAD still originate from controlled or synthetic scenes. Omni-AD is comprehensively larger in categories (150), industries (16), and scale (35K), and its real production line source makes it more challenging and realistic.
- **vs MMAD**: MMAD was the first to systematically evaluate MLLM performance in IAD but used multiple-choice VQA with 7 subtasks from 4 public datasets. Omni-AD uses three progressive subtasks aligned with real-site needs and models localization as grounding, providing more practical evaluation.
- **vs MANTA / PIAD**: These focus on multi-view text data for tiny objects or pose/lighting-agnostic detection. Omni-AD differentiates itself with its "real-line source + native MLLM protocol."

## Rating
- Novelty: ⭐⭐⭐⭐ The dataset is an extension of "larger and more realistic," but the "native MLLM 3-subtask support + hierarchical QA generation" is a substantial new angle.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 6 unsupervised methods, 4 MLLM families, dual protocols + human control, though lacks MLLM fine-tuning/few-shot bounds and QA bias controls.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, complete protocol and metric definitions, and self-consistent charts.
- Value: ⭐⭐⭐⭐⭐ Provides a more difficult, discriminative, and natively MLLM-oriented ruler for industrial anomaly detection, highly valuable for the field's practical application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](mmrad_multimodal_anomaly_detection.md)
- [\[CVPR 2026\] FastRef: Fast Prototype Refinement for Few-shot Industrial Anomaly Detection](fastref_fast_prototype_refinement_for_few-shot_industrial_anomaly_detection.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[CVPR 2026\] SRA-Det: Learning Omni-Grained Open-Vocabulary Detection Beyond Category Names](sra-det_learning_omni-grained_open-vocabulary_detection_beyond_category_names.md)
- [\[ICLR 2026\] ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection](../../ICLR2026/object_detection/forestpersons_a_large-scale_dataset_for_under-canopy_missing_person_detection.md)

</div>

<!-- RELATED:END -->
