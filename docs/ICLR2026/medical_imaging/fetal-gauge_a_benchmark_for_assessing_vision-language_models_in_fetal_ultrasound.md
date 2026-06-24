---
title: >-
  [Paper Note] Fetal-Gauge: A Benchmark for Assessing Vision-Language Models in Fetal Ultrasound
description: >-
  [ICLR 2026][Medical Imaging][Fetal Ultrasound] Fetal-Gauge integrates 13 public fetal ultrasound datasets to construct the first and largest vision-language question-answering benchmark for fetal ultrasound (42,000 images, 93,000 QA pairs, covering five clinical tasks). Systematic evaluation of 15 mainstream VLMs reveals that the strongest model, GPT-5, achieves only 55% accuracy, which is far below the threshold for clinical utility, exposing systematic shortcomings of curre…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Fetal Ultrasound"
  - "Vision-Language Question Answering"
  - "VLM Evaluation"
  - "Med-VQA"
  - "Medical Imaging Benchmark"
date: 2026-05-08
content_hash: 42f8c4c24078c852
---

# Fetal-Gauge: A Benchmark for Assessing Vision-Language Models in Fetal Ultrasound

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=AHZuGrWZ0d](https://openreview.net/forum?id=AHZuGrWZ0d)  
**Code**: To be confirmed (The paper states the benchmark is public)  
**Area**: Medical Imaging / Multimodal VLM / Evaluation Benchmarks  
**Keywords**: Fetal Ultrasound, Vision-Language Question Answering, VLM Evaluation, Med-VQA, Medical Imaging Benchmark

## TL;DR
Fetal-Gauge integrates 13 public fetal ultrasound datasets to construct the first and largest vision-language question-answering benchmark for fetal ultrasound (42,000 images, 93,000 QA pairs, covering five clinical tasks). Systematic evaluation of 15 mainstream VLMs reveals that the strongest model, GPT-5, achieves only 55% accuracy, which is far below the threshold for clinical utility, exposing systematic shortcomings of current VLMs in the fetal ultrasound domain.

## Background & Motivation
**Background**: Ultrasound is the primary imaging modality for prenatal fetal health monitoring, with hundreds of millions of exams performed globally each year. However, the long and costly training required for qualified sonographers has led to a severe shortage of technicians. Deep learning—especially Vision-Language Models (VLMs) that can handle images and text simultaneously within a single framework for multiple clinical tasks—is expected to improve efficiency and assist in training beginners.

**Limitations of Prior Work**: While several Med-VQA benchmarks exist (VQA-Med, VQA-RAD, SLAKE, PMC-VQA, OmniMedVQA, etc.), they focus almost exclusively on adult CT, MRI, X-ray, or pathology slides. **None specifically target fetal ultrasound**. This is due to three reasons: high imaging noise in fetal ultrasound, strong dependence on operator technique, and the scarcity of publicly available annotated data.

**Key Challenge**: Fetal ultrasound presents unique challenges not found in other modalities: it requires fine-grained spatial reasoning, interpretation across massive inter-operator variability, and handling ultrasound-specific artifacts. Without a targeted benchmark, these shortcomings remain **invisible** and unquantified, hindering model improvement.

**Goal**: To fill this gap by unifying heterogeneous fetal ultrasound data into a standardized, reproducible VLM evaluation benchmark, and to systematically answer whether "off-the-shelf VLMs can interpret fetal ultrasound."

**Key Insight**: Instead of training a new model, the authors first establish an "evaluation ruler." By aggregating 13 public datasets, heterogeneous annotations such as classification labels, segmentation masks, and detection boxes are unified into a Multiple Choice Question Vision Question Answering (MCQ-VQA) format, covering five types of clinical tasks ranging from high-level scene understanding to fine-grained anatomical localization.

**Core Idea**: Use a unified, large-scale, multi-task MCQ-VQA benchmark (Fetal-Gauge) to simultaneously highlight the capabilities and blind spots of existing VLMs in fetal ultrasound.

## Method
This paper is not a "model method" paper but rather focuses on benchmark construction and systematic evaluation. Therefore, the method focuses on three aspects: task definition, data unification from 13 sources, and splitting/statistics.

### Overall Architecture
The construction of Fetal-Gauge is summarized as a data pipeline: **13 public fetal ultrasound datasets → Task and annotation unification (heterogeneous labels/masks/boxes → Unified MCQ-VQA) → Vocabulary normalization → Strict patient-level splitting → 5-task benchmark with 42k images / 93k QA pairs**. Using this benchmark, 15 VLMs are evaluated with a unified protocol (accuracy metrics, task-specific reporting), complimented by analyses on fine-tuning, phantom vs. clinical data, anatomical size, and qualitative errors. The core constraint of the design is that "evaluation must reflect true generalization rather than patient-level memorization," which informs both task design and data splitting.

### Key Designs

**1. Five clinical tasks covering scene understanding to fine-grained localization**

The actual workflow of a fetal sonographer is decomposed into five clinically distinct tasks, all unified into an MCQ format. The advantage of MCQ is that evaluation is simple, automated, and avoids ambiguity or hallucinations in free-text responses, ensuring objective and scalable scoring. The five tasks are: Plane Identification (PI, identifying standard planes like abdominal circumference or thalamic planes); View Completeness (VC, determining if a plane meets diagnostic standards, testing "is it good enough for measurement?"); Fetal Orientation (FO, determining fetal position for delivery planning); Clinical Diagnosis (CD, classifying images as normal/benign/malignant); and Visual Grounding (VG, identifying the structure within a red bounding box, testing spatial reasoning). These tasks form a gradient from high-level (PI/VC) to fine-grained (VG).

**2. Heterogeneous annotations unified into VQA with red boxes**

Annotations from 13 source datasets vary widely, including image-level labels, segmentation masks, and bounding boxes. To integrate them into a single format, the authors convert segmentation masks into their outer bounding box coordinates and overlay these as **red rectangles** on the original image. The localization task is then unified as: "What does the red bounding box represent?" This step cleverly moves the segmentation task into a lower-dimensional MCQ-evaluated localization task, allowing pixel-level labels to be included in the unified benchmark.

**3. Vocabulary normalization to suppress multi-source label noise**

Aggregation results in label inconsistencies: clinical abbreviations (e.g., "abdomcirc") are expanded into full terms ("abdominal circumference plane"), and synonyms are unified. In cases where datasets do not label specific internal planes (e.g., labeling "heart plane" without distinguishing between "three-vessel view" or "four-chamber view"), a generic "[organ] plane" label is maintained. This ensures consistent terminology across the benchmark, preventing answer options from being contaminated by noise.

**4. Strict patient-level splitting + small datasets exclusively in test sets**

To ensure the evaluation measures true generalization, the splitting strategy is rigorous: original train-test splits are used if available; otherwise, **patient-level** splits are enforced to prevent data leakage. Datasets with small sample sizes are assigned **entirely to the test set** to specifically assess "generalization to unseen distributions." Categories with low clinical value (e.g., "other") are removed. Additionally, approximately 19,000 images are from anatomical phantoms (educational models), which are treated as a feature – phantoms are standard for training sonographers and support the development of DL systems for simulation while allowing for evaluation in controlled environments. The final benchmark consists of 42,036 images and 93,451 questions.

## Key Experimental Results

### Main Results
15 VLMs (6 medical-specific, 8 general, 1 commercial GPT-5) were evaluated alongside a random guess baseline. The core finding is that current VLMs are generally weak in fetal ultrasound—the strongest, GPT-5, achieved only 55% overall accuracy, while most models hovered near the random level (26%).

| Model | PI | VC | FO | CD | VG | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random Guess | 0.26 | 0.47 | 0.24 | 0.35 | 0.25 | 0.26 |
| GPT-5 (Strongest Commercial) | **0.66** | **0.62** | 0.23 | 0.20 | **0.58** | **0.55** |
| Lingshu-32B | 0.53 | 0.57 | 0.24 | 0.23 | 0.47 | 0.46 |
| Lingshu-7B | 0.39 | 0.61 | 0.24 | 0.24 | 0.45 | 0.40 |
| Llama-3.2-11B-Vision | 0.40 | 0.55 | 0.23 | 0.23 | 0.31 | 0.33 |
| Qwen2.5-VL-7B | 0.24 | 0.58 | 0.24 | 0.39 | 0.23 | 0.24 |
| MedVLM-R1 | 0.21 | 0.54 | 0.25 | 0.26 | 0.18 | 0.21 |

By task: PI and VG showed the largest variance, with GPT-5 and Lingshu significantly leading. In contrast, **all models performed at random levels** for VC and FO, showing no meaningful capability.

### Fine-Tuning and Structure Size Analysis
Targeted fine-tuning leads to significant improvements: after LoRA fine-tuning on the training set, Llama-3.2-11B's overall accuracy jumped from 33% to 85%, and Qwen2.5-VL-7B rose from 24% to 52%, suggesting weaknesses stem more from missing domain data than architecture. Visual grounding is strongly influenced by structure size.

| Bounding Box Size | GPT-5 | Lingshu-32B | Lingshu-7B | Note |
| :--- | :--- | :--- | :--- | :--- |
| Large (2,160 Qs) | 0.85 | 0.79 | 0.82 | Large structures often >80% |
| Medium (1,799 Qs) | 0.67 | 0.45 | 0.45 | Sharp decline |
| Small (7,373 Qs) | 0.48 | 0.38 | 0.34 | Most <50% |

## Key Findings
- **Clear Task Difficulty Stratification**: PI/VG differentiate models, whereas VC/FO results are near-random, indicating "orientation" and "standard compliance" are common blind spots in current VLMs.
- **Ultrasound Pre-training is a Key Variable**: Lingshu is the only open-source model explicitly reported to have been trained on (adult) ultrasound data and is consistently the strongest open-source model—proving that even adult ultrasound provides transferable anatomical priors.
- **Limited Benefit from Existing Med-VLMs**: The evaluated medical VLMs were not trained on fetal ultrasound. The differences between adult MRI/CT and fetal ultrasound in appearance and resolution mean they only provide slightly closer priors than natural images.
- **Fine-grained Localization is a Major Weakness**: Accuracy drops sharply as the box size decreases, yet clinical practice requires precise localization of small structures.
- **Phantoms are More Difficult than Clinical Images**: Most models performed near or below random on phantoms while exceeding random on clinical images, exposing a persistent domain adaptation gap.
- **GPT-5's Advantage is Questionable**: The authors suggest its closed-source training data likely already includes fetal ultrasound or related medical data, so its lead may not represent architectural superiority.

## Highlights & Insights
- **"Red Box to VQA" is a Reusable Engineering Trick**: Converting segmentation masks to bounding boxes and asking "what is in the box" converts pixel-level labels to MCQ, allowing heterogeneous labels to fit into a unified benchmark—a technique transferable to any scenario needing to unify detection/segmentation with VQA.
- **MCQ-based Evaluation Philosophy**: Using multiple-choice questions avoids ambiguity, hallucinations, and non-automatable scoring, representing a pragmatic trade-off in benchmark design to achieve scalable objective scoring.
- **Redefining Phantom Data from "Defect" to "Feature"**: The authors argue that phantoms are standard training tools; including them supports education/simulation research and provide controlled evaluation dimensions.
- **Most Valuable "Aha" Moment**: While even GPT-5 hits only 55%, LoRA fine-tuning can push an 11B model to 85%. This indicates the bottleneck in fetal ultrasound is **data availability** rather than model capacity, pointing toward a cost-effective path for domain adaptation and specialized training.

## Limitations & Future Work
- **Acknowledged Limitations**: GPT-5's lead may stem from data contamination (potential exposure to fetal ultrasound), making it poor evidence of "architectural capability." Existing medical VLMs mostly haven't seen fetal ultrasound, so evaluation reflects "unseen" rather than "unlearnable."
- **Evaluation Format Limitations**: The MCQ format differs significantly from real clinical workflows (open-ended reports, continuous measurements, interactive questioning); high MCQ scores do not equate to clinical utility.
- **Data Source Limitations**: The benchmark is composed of 13 public datasets, with phantoms accounting for nearly half (19k/42k). There remains a gap between its distribution and real clinical scanning. Cross-task comparisons of raw scores are limited by different random baselines (e.g., VC's baseline is 0.47).
- **Future Directions**: Supplementing with open-ended VQA and measurement tasks, introducing private data closer to clinical reality, and performing specialized domain adaptation for small/medium structure localization and phantom domains are key to pushing the "55% ceiling" toward clinical thresholds.

## Related Work & Insights
- **vs. General Med-VQA Benchmarks (VQA-Med / VQA-RAD / SLAKE / PMC-VQA / OmniMedVQA / CAREs)**: These cover multi-modal adult CT/MRI/X-ray/pathology but lack fetal ultrasound. Fetal-Gauge fills this systematically ignored modality, emphasizing ultrasound-specific challenges like operator dependency and noise.
- **vs. Specialized Medical VLMs (LLaVA-Med / MedVLM-R1 / Lingshu / HuatuoGPT-Vision / MedGemma)**: These models use curriculum learning or RAG to adapt to the medical domain, but this paper proves they perform nearly randomly on fetal ultrasound. Lingshu, which trained on (adult) ultrasound, performed best, confirming that "ultrasound-specific data" is the key variable.
- **Insights**: Fetal-Gauge applies the "establish the ruler first, then discuss improvement" methodology to a clinically significant modality. It provides both a diagnosis (current VLMs fail + localized weaknesses) and a prescription (high ROI of domain fine-tuning), offering a reproducible starting point for multimodal DL in prenatal imaging.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First and largest fetal ultrasound VLM benchmark, filling a clear gap, though "aggregation + MCQ conversion" is a standard methodological combination.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 15 models across 5 tasks, plus multidimensional analysis on fine-tuning, phantoms vs. clinical data, size, and qualitative errors.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-explained pipeline, and honest analysis (pointing out GPT-5 data contamination).
- **Value**: ⭐⭐⭐⭐⭐ Establishes an evaluation standard for prenatal ultrasound—a high-demand, high-shortage clinical scenario—and points toward high-value directions for data adaptation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] F$^2$-Assist: Multi-Phase Fetal Growth Forecast and Report Generation from Ultrasound Examination](../../CVPR2026/medical_imaging/f2-assist_multi-phase_fetal_growth_forecast_and_report_generation_from_ultrasoun.md)
- [\[ICLR 2026\] U2-BENCH: Benchmarking Large Vision-Language Models on Ultrasound Understanding](u2-bench_benchmarking_large_vision-language_models_on_ultrasound_understanding.md)
- [\[CVPR 2026\] Gastric-X: A Multimodal Multi-Phase Benchmark Dataset for Advancing Vision-Language Models in Gastric Cancer Analysis](../../CVPR2026/medical_imaging/gastric-x_a_multimodal_multi-phase_benchmark_dataset_for_advancing_vision-langua.md)
- [\[ICLR 2026\] UltraGauss: Ultrafast Gaussian Reconstruction of 3D Ultrasound Volumes](ultragauss_ultrafast_gaussian_reconstruction_of_3d_ultrasound_volumes.md)
- [\[ICLR 2026\] OpenPros: A Large-Scale Dataset for Limited View Prostate Ultrasound Computed Tomography](openpros_a_large-scale_dataset_for_limited_view_prostate_ultrasound_computed_tom.md)

</div>

<!-- RELATED:END -->
