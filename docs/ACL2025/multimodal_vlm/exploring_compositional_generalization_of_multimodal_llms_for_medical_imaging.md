---
title: >-
  [Paper Note] Exploring Compositional Generalization of Multimodal LLMs for Medical Imaging
description: >-
  [ACL 2025][Multimodal VLM][Compositional Generalization] This paper proposes the Med-MAT dataset (comprising 106 medical datasets and 53 subsets) and decomposes medical imaging attributes using MAT-Triplet (Modality-Anatomical area-Task). It systematically verifies, for the first time, the phenomenon of compositional generalization (CG) in multimodal LLMs on medical imaging, and demonstrates that compositional generalization is the key driver of generalization gains in multi-…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Compositional Generalization"
  - "Multimodal LLM"
  - "Medical Imaging"
  - "MAT-Triplet"
  - "Multi-Task Training"
  - "Generalization Analysis"
date: 2026-05-08
content_hash: e59cff175f15a769
---

# Exploring Compositional Generalization of Multimodal LLMs for Medical Imaging

**Conference**: ACL 2025  
**arXiv**: [2412.20070](https://arxiv.org/abs/2412.20070)  
**Code**: [https://github.com/FreedomIntelligence/Med-MAT](https://github.com/FreedomIntelligence/Med-MAT)  
**Authors**: Zhenyang Cai, Junying Chen, Rongsheng Wang, Weihong Wang, Yonglin Deng, Dingjie Song, Yize Chen, Zixu Zhang, Benyou Wang  
**Institution**: The Chinese University of Hong Kong, Shenzhen  
**Area**: Multimodal Large Language Models / Medical Imaging  
**Keywords**: Compositional Generalization, Multimodal LLM, Medical Imaging, MAT-Triplet, Multi-Task Training, Generalization Analysis

## TL;DR

This paper proposes the Med-MAT dataset (comprising 106 medical datasets and 53 subsets) and decomposes medical imaging attributes using MAT-Triplet (Modality-Anatomical area-Task). It systematically verifies, for the first time, the phenomenon of compositional generalization (CG) in multimodal LLMs on medical imaging, and demonstrates that compositional generalization is the key driver of generalization gains in multi-task training.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) such as LLaVA have demonstrated strong generalization capabilities in medical image analysis. Existing studies (Mo & Liang 2024; Ren et al. 2024) have confirmed that multi-task training outperforms single-task training, yet the underlying mechanism of generalization remains unclear.

**Core Problem**: What exactly drives the mutual promotion between different tasks in multi-task training? Existing works have only observed the generalization phenomenon without deeply analyzing its underlying structural causes.

**Key Insight**: Medical images can be precisely defined by three orthogonal dimensions: **Modality** (e.g., X-ray, CT), **Anatomical Area** (e.g., Lung, Brain), and **Medical Task** (e.g., Cancer, State). This naturally creates a testbed for compositional generalization: a model can learn "X-ray + Lung" and "CT + Brain" to generalize to the unseen combination "X-ray + Brain".

**Design Motivation**: To leverage the theoretical framework of compositional generalization (CG) to explain and verify the generalization phenomena of MLLMs in medical imaging, thereby revealing the essence of generalization in multi-task training.

## Method

### Overall Architecture: Med-MAT Dataset Construction

- **Data resource**: Collected 106 public medical image datasets (label-image pairs).
- **MAT-Triplet definition**: Each sample is uniquely characterized by a (Modality, Anatomical area, Task) triplet.
- **Data merging**: Datasets with identical MAT-Triplets are merged into a subset, resulting in **53 subsets** in total.
- **Coverage**: 11 modalities, 14 anatomical areas, and 13 medical tasks.
- **Train/Test split**: Divided according to the original distribution or a 9:1 ratio. Each subset in the training set is restricted to 3,000 samples (maintaining label balance) to prevent dominance. Subsets with fewer than 3,000 samples are used as out-of-distribution (OOD) test sets.
- **QA format conversion**: All samples are converted to VQA format (multiple-choice questions with up to 4 options). Six instruction templates are manually designed for each subset.

### Experimental Design for Compositional Generalization Verification

**Step 1: CG Existence Verification**

- Select a subset as the **Target** (target data), and find the **Related** data in Med-MAT that shares partial MAT-Triplets with the Target.
- For example: If the Target is "X-ray + Lung + Cancer", the Related data includes "CT + Lung + ?" and "X-ray + Brain + Cancer".
- Train on Related data and test on Target, comparing with the baseline.
- Key: If CG exists, the model can understand the unseen Target solely by training on combinations of related elements.

**Step 2: CG as the Key Generalization Driver**

- Scale up the experiment, and artificially break CG in multi-task training (by removing data that shares MAT elements with the Target).
- Observe the changes in generalization performance after breaking CG to verify whether CG is the primary form of generalization.

**Step 3: Exploraing Practicality of CG**

- Explore the capability of CG in low-resource/limited data scenarios.
- Verify whether CG holds across classification and detection tasks.

### Base Model

- Base model: LLaVA-v1.5-7B-Vicuna (Reason for selection: Transparent training process, minimal medical data used during pre-training, reducing the risk of knowledge leakage).
- Training configuration: 5 epochs, 8 $\times$ A800 GPUs, batch size 32, learning rate 5e-6.
- Tasks are switched via prompts, leveraging the flexibility of MLLMs to simplify multi-task generalization research.

## Key Experimental Results

### Main Results: Multi-Task vs. Single-Task Training

| Metric | Baseline (Untrained) | Single-Task Training | Multi-Task Training |
|------|-------------------|-----------|-----------|
| Subset 02 | 47% | 49% | **89%** |
| Subset 13 | 28% | 83% | **92%** |
| Subset 30 | 49% | 89% | **94%** |
| Subset 32 | 49% | 97% | **100%** |

- Multi-task training outperforms single-task training on all 25 ID (In-Distribution) datasets.
- On OOD datasets (12 in total), multi-task training also shows significant generalization improvements (e.g., Subset 05: 33% $\rightarrow$ 70%, Subset 17: 33% $\rightarrow$ 61%).

### CG Existence Verification

- Among 24 sets of CG experiments, **most combinations successfully demonstrated CG effects** (marked with $\checkmark$).
- Typical case: After learning "Lung + COVID" and "Brain + Cancer", the model can effectively generalize to "Lung + Cancer" (25 $\rightarrow$ 27 $\checkmark$).
- Cross-modality CG also holds: After learning "CT + Cancer" and "X-ray + COVID", the model generalizes to "CT + COVID" (47 $\rightarrow$ 72 $\checkmark$).
- A few failure cases show that CG is not always guaranteed, but the success rate is far higher than the random baseline.

### CG Disruption Experiment

- Generalization performance drops significantly after disrupting CG, confirming that CG is a key driver of generalization in multi-task training.

### Ablation Study

- CG is effective across different MLLM architectures (confirming generalizability).
- CG effectively supports limited data scenarios: in data-scarce situations, CG combinational training still improves target task performance.
- CG holds across both classification and detection tasks, demonstrating broader generalization potential.

## Highlights & Insights

1. **Theoretical Innovation**: Introduces the theory of compositional generalization (CG) to medical imaging MLLM analysis for the first time, providing a new perspective to understand the generalization gains of multi-task training.
2. **MAT-Triplet Decomposition**: Orthogonally decomposes the three attributes of medical imaging, cleverly constructing a natural environment for CG experiments.
3. **Dataset Contribution**: Med-MAT covers 106 datasets, 11 modalities, 14 regions, and 13 tasks, making it the largest medical imaging dataset for CG research to date.
4. **Ingenious Experimental Design**: Cross-validates through both "combinational training $\rightarrow$ observing generalization" and "disrupting CG $\rightarrow$ observing degradation", ensuring logical self-consistency.
5. **Value**: CG's effective support for data-scarce scenarios implies that strategic data combinations can compensate for data scarcity in specific medical scenarios.

## Limitations & Future Work

1. **Limited Model Scale**: Experiments are only conducted on the 7B-scale LLaVA; the performance of CG on larger-scale models remains unknown.
2. **MAT-Triplet Assumption**: Decomposing medical imaging into three orthogonal dimensions might be oversimplified, as complex interactions may exist among these dimensions in real-world scenarios.
3. **QA Format Limitation**: Converting all tasks into 4-option multiple-choice questions may mask differences in fine-grained generalization capabilities.
4. **CG Failure Cases**: Some combinations fail to show CG effects, but the paper lacks a deep analysis of the reasons for these failures.
5. **Clinical Usability**: The dataset is primarily research-oriented, and how to apply CG insights to actual clinical deployment has not been discussed.

## Related Work

- **Medical Imaging MLLMs**: Works like LLaVA-Med and Med-PaLM M adapt general MLLMs to the medical domain.
- **Compositional Generalization**: CG theory originates from cognitive science and has been widely studied in NLP and computer vision (Li et al. 2019; Xu et al. 2022; Tang et al. 2024), but this paper introduces it to medical imaging MLLMs for the first time.
- **Multi-Task Medical Training**: Works like Mo & Liang (2024) observe generalization gains from multi-task training but do not explain the underlying reasons.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ Novel CG perspective, clever MAT-Triplet decomposition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 53 subsets, multiple control experiments, comprehensive validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, step-by-step experimental design.
- **Value**: ⭐⭐⭐ Valuable insights, but the path to clinical application is unclear.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Insight Over Sight: Exploring the Vision-Knowledge Conflicts in Multimodal LLMs](conflictvis_vision_knowledge_conflict.md)
- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](../../ICCV2025/multimodal_vlm/mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)
- [\[ICCV 2025\] Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs](../../ICCV2025/multimodal_vlm/are_they_the_same_exploring_visual_correspondence_shortcomings_of_multimodal_llm.md)
- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](../../NeurIPS2025/multimodal_vlm/better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)
- [\[ICLR 2026\] DataProphet: Demystifying Supervision Data Generalization in Multimodal LLMs](../../ICLR2026/multimodal_vlm/demystifying_supervision_data_generalization_in_multimodal_lms.md)

</div>

<!-- RELATED:END -->
