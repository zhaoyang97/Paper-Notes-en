---
title: >-
  [Paper Note] UltrasoundAgents: Hierarchical Multi-Agent Evidence-Chain Reasoning for Breast Ultrasound Diagnosis
description: >-
  [CVPR2025][Medical Imaging][multi-agent] This paper proposes UltrasoundAgents, a hierarchical multi-agent framework. By aligning with the clinical breast ultrasound diagnostic workflow through a pipeline of a main agent locating lesions, sub-agents identifying attributes, and evidence-chain reasoning, it achieves traceable BI-RADS assessment and benign/malignant classification.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "multi-agent"
  - "breast ultrasound"
  - "BI-RADS"
  - "VLM"
  - "reinforcement-learning"
  - "evidence-chain"
date: 2026-05-08
content_hash: d93b6bc86a63cf16
---

# UltrasoundAgents: Hierarchical Multi-Agent Evidence-Chain Reasoning for Breast Ultrasound Diagnosis

**Conference**: CVPR2025  
**arXiv**: [2603.10852](https://arxiv.org/abs/2603.10852)  
**Code**: None  
**Area**: Medical Imaging  
**Keywords**: multi-agent, breast ultrasound, BI-RADS, VLM, reinforcement-learning, evidence-chain

## TL;DR

This paper proposes UltrasoundAgents, a hierarchical multi-agent framework. By aligning with the clinical breast ultrasound diagnostic workflow through a pipeline of a main agent locating lesions, sub-agents identifying attributes, and evidence-chain reasoning, it achieves traceable BI-RADS assessment and benign/malignant classification.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Breast ultrasound (BUS) is an essential tool for breast cancer screening. Clinicians follow a coarse-to-fine viewing workflow: locating the lesion $\rightarrow$ assessing fine-grained signs (echogenicity, boundary, etc.) $\rightarrow$ integrating evidence to provide BI-RADS assessment and benign/malignant classification. Existing methods suffer from the following limitations:

**End-to-end prediction lacks interpretability**: Inability to provide intermediate chains of evidence limits clinical auditability.

**Weak explicit evidence**: Although multi-stage methods incorporate interpretable designs, the causal relationship between evidence and the final diagnosis remains unclear.

**Unstable VLM reasoning**: SFT relies heavily on template data, while RL faces difficulties in localization error propagation and credit assignment.

Key Challenge: How to simultaneously learn localization, fine-grained perception, and high-level diagnostic reasoning in a single policy—since localization errors alter downstream observation distributions, increasing non-stationarity. Hierarchical multi-agents offer a solution by isolating evidence collection from evidence integration.

## Method

### Hierarchical Dual-Agent Architecture
Based on the Qwen2.5-VL-3B vision-language model:

- **Main Agent ($A_M$)**: Receives the entire image, predicts the ROI bounding box to perform crop-and-zoom (with crop area size no less than 224$\times$224), integrates the attribute evidence returned by the sub-agent, and outputs the benign/malignant ($y_{mal}$) and BI-RADS ($y_{bi}$) diagnostic results.
- **Sub-Agent ($A_S$)**: Based on the cropped and zoomed local view of the lesion, identifies four clinical attributes: echogenicity, calcification, boundary types, and margins (edge).
- The attribute taxonomy adopts the unified classification system of BUS-CoT, ensuring comparability with existing literature.

This explicit division of labor reduces the learning burden on the main agent and improves interpretability through structured intermediate evidence.

### Decoupled Progressive Training (Three Stages)

**Stage 1: Sub-Agent RL Training**
The sub-agent $A_S$ is trained using the GRPO algorithm to perform attribute identification on cropped images. Reward = attribute accuracy + format compliance.

**Stage 2: Oracle-Guided Curriculum RL**
The output of the sub-agent is replaced with ground-truth attributes to train the main agent's diagnostic reasoning capability. The reward focuses solely on diagnostic correctness:
$$R_M = \lambda_1 \cdot \mathbb{I}[y_{mal}^{pred} = y_{mal}^{gt}] + \lambda_2 \cdot \mathbb{I}[y_{bi}^{pred} = y_{bi}^{gt}]$$
This avoids the non-stationary training problem caused by attribute noise. Oracle evidence is used only during training; predicted attributes from the sub-agent are consumed during inference. This stage does not rely on sparse rewards to directly learn precise localization; localization is primarily improved in Stage 3.

**Stage 3: Corrective Trajectory Self-Distillation**
Trajectories are sampled from the Stage 2 policy, and two modifications are performed: (1) replacing the predicted bounding boxes with GT bounding boxes to strengthen spatial localization; (2) rewriting the reasoning process conditioned on GT labels for samples with diagnostic errors. Modified trajectories are utilized for SFT, distilling stable reasoning capabilities into the deployable policy.

## Key Experimental Results

Evaluated on three in-domain datasets (BUSBRA, BUSI, and BUDIAT) and one out-of-domain dataset (BrEaST). All methods utilize the same Qwen2.5-VL-3B base model, differing only in their training strategies:

### Main Results

| Method | Overall AUC | Overall Acc | Bi-Acc | κ |
|------|-------------|-------------|--------|---|
| Zero-Shot | 0.476 | 0.602 | 0.117 | 0.014 |
| CoT-SFT (BUS-CoT) | 0.710 | 0.751 | 0.468 | 0.204 |
| Think-with-Image (DeepEyes) | 0.512 | 0.683 | 0.101 | 0.004 |
| **UltrasoundAgents** | **0.741** | **0.813** | **0.515** | **0.224** |

Note that while Think-with-Image also includes a crop-and-zoom pipeline, its performance is close to random, indicating that simple cropping and zooming does not guarantee improvement; the key lies in the hierarchical division of labor. For out-of-domain generalization (BrEaST), this method achieves a benign/malignant AUC of 0.685 vs. CoT-SFT's 0.586, benefiting from the ROI crop's reduction in background sensitivity.

**Ablation analysis** (Overall metrics):

### Ablation Study

| Variant | AUC | Acc | Bi-Acc | κ | IoU |
|------|-----|-----|--------|---|-----|
| w/o Oracle Training | 0.535 | 0.696 | 0.413 | 0.018 | 0.328 |
| w/o Self-Distill | 0.726 | 0.767 | 0.458 | 0.173 | 0.299 |
| Full model | 0.741 | 0.813 | 0.515 | 0.224 | 0.610 |
| + GTbox | 0.782 | 0.837 | 0.501 | 0.208 | 1.0 |
| + GTattr | 0.804 | 0.853 | 0.582 | 0.345 | 0.568 |

Removing Oracle training drops the AUC by 0.206, highlighting the crucial importance of learning reasoning capabilities with noise-free attributes. The upper-bound analysis with GTattr demonstrates that if attributes are perfect, the AUC can reach 0.804, indicating that attribute perception accuracy is still the primary bottleneck.

**Attribute recognition**: The F1 scores of cropped local views on Boundary, Edge, and Echo are all superior to those of full-image inputs, validating the effectiveness of the crop-and-zoom strategy. On the OOD dataset BrEaST, the advantage of the local view is even more pronounced (Boundary Macro-F1: 0.524 vs. 0.387).

## Highlights & Insights

1. **First ultrasound diagnostic Agent framework**: Aligns VLM multi-agents with clinical BUS workflow.
    - The evidence chain of ROI $\rightarrow$ Attributes $\rightarrow$ Diagnosis is traceable.
    - Provides auditable intermediate results.
2. **Oracle curriculum RL training strategy**: Decouples reasoning learning from perceptual noise to address the non-stationarity issue in hierarchical training. The design is ingenious and highly versatile.
3. **Corrective trajectory self-distillation**: Transforms sparse rewards from RL exploration into dense supervision signals.
    - Simultaneously considers both localization and reasoning dimensions.
    - Resolves the noise issue in RL trajectory sampling.
4. **Think-with-Image comparison**: Simple crop-and-zoom does not guarantee improvement (approaching random); the key lies in the hierarchical division of labor and structured evidence transmission.
5. **Generality of the RL training strategy**: The paradigm of Oracle curriculum RL + trajectory self-distillation can be migrated to other hierarchical multi-agent tasks.
6. **Significant out-of-domain AUC improvement**: AUC on BrEaST is 0.685 vs. CoT-SFT's 0.586, benefiting from crop-and-zoom reducing background interference.

## Limitations & Future Work

1. The base model has only 3B parameters (Qwen2.5-VL-3B), limiting its reasoning capability; larger models (7B/14B) may achieve better performance.
2. BI-RADS classification accuracy remains relatively low (the best is only 0.515), and the kappa (κ) value of 0.224 indicates insufficient agreement, highlighting a remaining gap from clinical viability.
3. Out-of-domain generalization on BI-RADS is still weak (BrEaST Bi-Acc 0.157); attribute noise and localization bias are the primary bottlenecks.
4. The three-stage training pipeline is complex (RL + Oracle RL + SFT), leading to higher deployment and iteration costs in practice.
5. Covers only four attribute dimensions (echogenicity, calcification, boundary, margin), failing to cover all BI-RADS descriptors (e.g., posterior acoustic features, lesion orientation, surrounding tissue).
6. Attribute annotations for public BUS datasets rely on the unified annotations provided by BUS-CoT, and the annotation quality has not been independently verified.
7. Training was conducted using only two L40S GPUs; the relatively small computing resources might have limited the thoroughness of RL exploration.

## Rating
- Novelty: 4/5 — The first multi-agent framework for ultrasound diagnosis, featuring a novel training paradigm of Oracle RL + trajectory self-distillation.
- Experimental Thoroughness: 4/5 — Comprehensive evaluation across four datasets with multiple baselines, ablation studies, and upper-bound analyses.
- Writing Quality: 4/5 — Clear architecture diagrams and fluent description of the three-stage training logic.
- Value: 4/5 — Inspiring value for research in explainable medical AI diagnosis and VLM Agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[ICLR 2026\] CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework](../../ICLR2026/medical_imaging/care_towards_clinical_accountability_in_multi-modal_medical_reasoning_with_an_ev.md)
- [\[CVPR 2025\] Surg-R1: A Hierarchical Reasoning Foundation Model for Scalable and Interpretable Surgical Decision Support](surg-r1_a_hierarchical_reasoning_foundation_model_for_scalable_and_interpretable.md)
- [\[CVPR 2025\] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)
- [\[CVPR 2026\] EMAD: Evidence-Centric Grounded Multimodal Diagnosis for Alzheimer's Disease](../../CVPR2026/medical_imaging/emad_evidence-centric_grounded_multimodal_diagnosis_for_alzheimers_disease.md)

</div>

<!-- RELATED:END -->
