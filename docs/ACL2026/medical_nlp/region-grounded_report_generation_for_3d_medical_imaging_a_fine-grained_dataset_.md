---
title: >-
  [Paper Note] Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework
description: >-
  [ACL 2026][Medical NLP][PET/CT Report Generation] This paper introduces VietPET-RoI, the first 3D PET/CT dataset (Vietnamese) with fine-grained ROI annotations, and HiRRA…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "PET/CT Report Generation"
  - "ROI Annotation"
  - "Graph Neural Networks"
  - "3D Medical Imaging"
  - "Low-Resource Languages"
date: 2026-05-08
content_hash: 776b82456a19f568
---

# Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework

**Conference**: ACL 2026  
**arXiv**: [2604.18145](https://arxiv.org/abs/2604.18145)  
**Code**: Yes (GitHub, to be released after acceptance)  
**Area**: Medical Imaging  
**Keywords**: PET/CT Report Generation, ROI Annotation, Graph Neural Networks, 3D Medical Imaging, Low-Resource Languages

## TL;DR
This paper introduces VietPET-RoI, the first 3D PET/CT dataset (Vietnamese) with fine-grained ROI annotations, and HiRRA, a hierarchical report generation framework that simulates the diagnostic process of radiologists. By modeling spatial-morphological relationships between ROIs via graph neural networks, the method achieves a 19.7% improvement in BLEU-4 and a 45.8% increase in the clinical metric RoIQ.

## Background & Motivation

**Background**: Vision-language models have achieved significant progress in automated medical report generation, but report generation for 3D PET/CT remains in its early stages. The accuracy of existing models is suboptimal, and they are prone to severe hallucinations, leaving a substantial gap for actual clinical requirements.

**Limitations of Prior Work**: Current PET/CT report generation models adopt an end-to-end paradigm—directly mapping the entire 3D volume to a text report. This "black-box" strategy ignores the inherent complexity of PET/CT data. In clinical practice, a radiologist's diagnostic workflow involves systematically identifying specific regions of index (ROI), evaluating the attributes of each ROI (size, density, SUVmax, etc.), analyzing spatial and physiological relationships between ROIs, and finally synthesizing all findings to write the report. Furthermore, existing datasets either contain segmentations without reports or reports without ROI-level annotations, particularly lacking data for low-resource languages such as Vietnamese.

**Key Challenge**: End-to-end models lack the inductive bias of the clinical workflow, preventing them from learning the reasoning process "from region-level findings to diagnostic conclusions." Additionally, there is a lack of training data with ROI-level annotations to support region-based learning.

**Goal**: (1) Construct the first 3D PET/CT dataset with fine-grained ROI annotations; (2) Design a hierarchical framework that simulates the radiologist's diagnostic process.

**Key Insight**: The diagnostic process of radiologists is inherently hierarchical—performing local analysis followed by global synthesis. Therefore, it is necessary to provide both global volume features and local ROI features while modeling dependencies between ROIs (e.g., adjacent lesions may suggest local invasion, while distant lesions with similar metabolic patterns may indicate metastasis).

**Core Idea**: Use a dual-stream encoder to extract CT (anatomical) and PET (metabolic) features respectively, then model spatial-morphological relationships between ROIs using GATv2 graph neural networks, and finally inject multi-granularity global and local features into an LLM to generate reports.

## Method

### Overall Architecture
HiRRA consists of three core modules: (1) Dual-stream Encoder: Processes CT and PET volumes separately and fuses them via cross-attention to obtain visual features; (2) Hierarchical Feature Extractor: The global context branch compresses global information through a Q-former, while the local context branch extracts ROI features using SPP-RoI and models inter-ROI relationships with GATv2; (3) LLM Decoder: Receives multi-granularity features and structured ROI descriptions as prompts to generate clinical reports.

### Key Designs

1. **Dual-stream Encoder and Cross-attention Fusion**:

    - **Function**: Separately extracts anatomical structure information from CT and metabolic activity information from PET, followed by cross-modal fusion via cross-attention.
    - **Mechanism**: Two independent CT-ViT 3D encoders extract $F_{CT}$ and $F_{PET} \in \mathbb{R}^{B \times N \times D}$, which are then fused via bidirectional cross-attention: $\tilde{F}_{CT} = F_{CT} + \text{CA}(F_{CT}, F_{PET})$. The final representation is obtained by averaging and processing through a multi-scale pyramid network.
    - **Design Motivation**: CT captures tissue boundaries and organ morphology, while PET quantifies metabolic activity (glucose uptake patterns); both provide complementary information. The dual-stream design prevents modality features from being overwhelmed due to premature fusion.

2. **GATv2 Graph Relationship Modeling**:

    - **Function**: Captures spatial and morphological dependencies between ROIs.
    - **Mechanism**: A graph is constructed with ROIs as nodes. Edges are established based on two criteria: spatial proximity (geometric distance between centroids $d_{ij} < \tau_d$) and morphological similarity (feature cosine similarity $s_{ij} > \tau_s$). Edge features encode spatial relationships (distance, relative direction, volume ratio) and morphological relationships (similarity, average CT/PET intensity), with message passing performed via the GATv2 attention mechanism.
    - **Design Motivation**: Processing each ROI in isolation misses critical clinical relationships—spatially adjacent lesions may suggest local invasion, while distant lesions with similar metabolic patterns may suggest metastasis. Graph modeling captures both types of relationships simultaneously.

3. **Clinical Evaluation Metrics (RoI Coverage and RoI Quality Index)**:

    - **Function**: Evaluates report quality from a clinical perspective, going beyond traditional NLP metrics.
    - **Mechanism**: RoI Coverage calculates the pairing between predicted ROIs and ground truth ROIs via Hungarian matching to obtain Precision/Recall/F1. The RoI Quality Index (RoIQ) performs attribute-level accuracy evaluation on matched ROI pairs using the formula $\text{RoIQ} = \sqrt{S_{\text{region}} \cdot S_{\text{lesion}}} \times \frac{1}{|\mathcal{A}|}\sum S_k$, applying non-linear penalties to core attributes (anatomical region and lesion type) via geometric mean.
    - **Design Motivation**: Traditional BLEU/ROUGE only measure lexical overlap and cannot evaluate the clinical correctness of a report. The hierarchical design of RoIQ ensures that errors in core attributes (location or lesion type) are not compensated by high scores in secondary attributes.

### Loss & Training
A four-stage progressive training strategy is adopted: Stage 1 uses CLIP contrastive learning to pre-train the dual-stream encoder; Stage 2 freezes the encoder and LLM while training the Q-former for vision-language alignment; Stage 3 introduces the local context module (SPP-RoI + GATv2), freezing parameters from the first two stages; Stage 4 performs end-to-end fine-tuning using LoRA ($r=16, \alpha=32$).

## Key Experimental Results

### Main Results

| Method | BLEU-4 | ROUGE-L | BERT | Correct ROI Count | RoI F1 | RoIQ |
|------|--------|---------|------|----------|--------|------|
| MedM-VL | 31.69 | 50.00 | 91.92 | 62/416 | 14.16 | 23.24 |
| M3D-LaMed | 44.30 | 64.39 | 85.90 | 193/416 | 39.83 | 36.31 |
| HiRRA (No RoI) | 52.48 | 66.51 | 95.13 | 187/416 | 37.58 | 33.89 |
| **HiRRA** | **62.80** | **69.66** | **95.79** | **223/416** | **42.47** | **56.86** |
| Gain Δ% | +19.7% | +4.7% | +0.7% | +15.5% | +6.6% | **+45.8%** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| HiRRA Full | RoIQ=56.86 | Full model |
| Without ROI annotation (No RoI) | RoIQ=33.89 | ROI-level supervision contributes most; RoIQ drops by 22.97 |
| Single CT Encoder | Performance Drop | Lacks PET metabolic information |
| Single PET Encoder | Greater Drop | Lacks CT anatomical structure information |

### Key Findings
- ROI-level annotations contribute most significantly to the improvement of clinical metrics—RoIQ increased from 33.89 to 56.86 (+45.8%), indicating that fine-grained regional supervision is key to reducing hallucinations.
- All existing VLMs perform poorly on Vietnamese (BLEU-4 near 0), highlighting that medical AI for low-resource languages lags seriously behind.
- GATv2 graph relationship modeling provides the most help for diagnosing metastatic diseases because it can associate distant lesions with similar metabolic patterns.
- Dual-modality fusion shows a clear advantage over single-modality; CT provides localization and PET provides functional information, both being indispensable.

## Highlights & Insights
- The **ROI-level annotation strategy** transforms report generation from an "end-to-end black box" into an interpretable paradigm of "analysis before synthesis." This approach can be transferred to other 3D medical imaging report generation tasks.
- The **RoIQ metric design** is ingenious—applying non-linear penalties to core attributes via geometric means ensures that errors in anatomical localization or lesion types cannot be masked by high scores in other attributes. This hierarchical evaluation approach can be generalized to any task requiring structured output quality assessment.
- The four-stage progressive training strategy effectively balances the learning of each module—from global alignment to local refinement and finally end-to-end optimization.

## Limitations & Future Work
- The dataset size is relatively small (200 patients, 600 samples, 1,960 ROIs), which limits the model's generalization capability.
- Currently, only Vietnamese is supported; the framework needs to be extended to more languages to verify its universality.
- ROI annotation relies on manual labeling by 8 nuclear medicine physicians, which is costly and difficult to scale.
- An automated ROI detection module could be considered to replace manual annotation for end-to-end automation.
- Future work could explore making the ROI-level reasoning process explicit to provide more detailed diagnostic explanations.

## Related Work & Insights
- **vs M3D-LaMed**: M3D-LaMed is a general-purpose 3D medical VLM but lacks ROI-level supervision, performing significantly worse than HiRRA on clinical metrics (RoIQ 36.31 vs 56.86).
- **vs ViMed-PET**: ViMed-PET provides Vietnamese PET/CT report data but only includes global report annotations without fine-grained ROI-level annotations.
- **vs CT2Rep**: CT2Rep is limited to global CT-only report mapping and does not support PET modality or ROI-level reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First 3D PET/CT dataset with ROI-level annotations; novel graph-enhanced framework design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete multi-baseline comparisons, ablation analysis, and clinical evaluation metrics, though the dataset size is small.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-articulated clinical motivation.
- Value: ⭐⭐⭐⭐⭐ The dataset and evaluation metrics serve as a significant contribution to the medical imaging AI community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CT-FineBench: A Diagnostic Fidelity Benchmark for Fine-Grained Evaluation of CT Report Generation](ct-finebench_a_diagnostic_fidelity_benchmark_for_fine-grained_evaluation_of_ct_r.md)
- [\[ACL 2026\] ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection](promedical_hierarchical_fine-grained_criteria_modeling_for_medical_llm_alignment.md)
- [\[ACL 2026\] HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for Medical Vision Language Tasks](heterorag_a_heterogeneous_retrieval-augmented_generation_framework_for_medical_v.md)
- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)
- [\[ACL 2026\] SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning](sema-rag_a_self-evolving_multi-agent_retrieval-augmented_generation_framework_fo.md)

</div>

<!-- RELATED:END -->
