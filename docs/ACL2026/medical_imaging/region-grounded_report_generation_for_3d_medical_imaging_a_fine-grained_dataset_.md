---
title: >-
  [Paper Note] Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework
description: >-
  [ACL 2026][Medical Imaging][PET/CT report generation] This paper presents VietPET-RoI, the first 3D PET/CT dataset with fine-grained ROI annotations (in Vietnamese), along with HiRRA…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "PET/CT report generation"
  - "ROI annotation"
  - "graph neural networks"
  - "3D medical imaging"
  - "low-resource languages"
date: 2026-05-08
content_hash: d89ca4d17e407afa
---

# Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework

**Conference**: ACL 2026
**arXiv**: [2604.18145](https://arxiv.org/abs/2604.18145)  
**Code**: Available (GitHub, to be released upon acceptance)  
**Area**: Medical Imaging
**Keywords**: PET/CT report generation, ROI annotation, graph neural networks, 3D medical imaging, low-resource languages

## TL;DR
This paper presents VietPET-RoI, the first 3D PET/CT dataset with fine-grained ROI annotations (in Vietnamese), along with HiRRA, a hierarchical report generation framework that emulates the diagnostic workflow of radiologists. By modeling spatial-morphological inter-ROI relationships via GATv2 graph neural networks, HiRRA achieves a 19.7% improvement in BLEU-4 and a 45.8% improvement in the clinical metric RoIQ.

## Background & Motivation

**Background**: Vision-language models have achieved remarkable progress in automated medical report generation, yet report generation for 3D PET/CT remains at an early stage. Existing models suffer from unsatisfactory accuracy, are prone to severe hallucinations, and fall considerably short of real-world clinical requirements.

**Limitations of Prior Work**: Current PET/CT report generation models adopt an end-to-end paradigm that directly maps entire 3D volumes to textual reports. This "black-box" strategy overlooks the inherent complexity of PET/CT data. In clinical practice, radiologists follow a systematic diagnostic workflow: they first identify specific regions of interest (ROIs), evaluate each ROI's attributes (size, density, SUVmax, etc.), analyze spatial and physiological relationships among ROIs, and only then synthesize all findings into a report. Furthermore, existing datasets either provide segmentation annotations without reports, or reports without ROI-level annotations—with a particular scarcity of data in low-resource languages such as Vietnamese.

**Key Challenge**: End-to-end models lack the inductive bias of clinical workflows and cannot learn the reasoning process of deriving diagnostic conclusions from region-level findings; the absence of ROI-level annotated training data further prevents region-specific learning.

**Goal**: (1) Construct the first 3D PET/CT dataset with fine-grained ROI annotations; (2) Design a hierarchical framework that emulates the diagnostic workflow of radiologists.

**Key Insight**: The authors observe that radiologists' diagnosis is inherently a hierarchical process—local analysis precedes global synthesis. This necessitates providing both global volume features and local ROI features, while modeling inter-ROI dependencies (e.g., adjacent lesions may indicate local invasion, while distant lesions with similar metabolic patterns may suggest metastasis).

**Core Idea**: A dual-stream encoder extracts CT (anatomical) and PET (metabolic) features separately; a GATv2 graph neural network models spatial-morphological inter-ROI relationships; multi-granularity global and local features are then injected into an LLM to generate the final report.

## Method

### Overall Architecture
HiRRA comprises three core modules: (1) a **dual-stream encoder** that processes CT and PET volumes separately and fuses them via cross-attention to obtain visual features; (2) a **hierarchical feature extractor** with a global context branch that compresses global information via Q-Former, and a local context branch that extracts ROI features via SPP-RoI and models inter-ROI relationships via GATv2; and (3) an **LLM decoder** that receives multi-granularity features and structured ROI descriptions as prompts to generate clinical reports.

### Key Designs

1. **Dual-Stream Encoder with Cross-Attention Fusion**

    - **Function**: Extracts anatomical structural information from CT and metabolic activity information from PET, then performs cross-modal fusion via cross-attention.
    - **Mechanism**: Two independent CT-ViT 3D encoders extract $F_{CT}$ and $F_{PET} \in \mathbb{R}^{B \times N \times D}$ respectively. Bidirectional cross-attention fusion is then applied: $\tilde{F}_{CT} = F_{CT} + \text{CA}(F_{CT}, F_{PET})$. The outputs are averaged and passed through a multi-scale pyramid network to obtain the fused representation.
    - **Design Motivation**: CT captures tissue boundaries and organ morphology, while PET quantifies metabolic activity (glucose uptake patterns), providing complementary information. The dual-stream design prevents premature fusion from suppressing modality-specific features.

2. **GATv2 Graph Relational Modeling**

    - **Function**: Captures spatial and morphological dependencies among ROIs.
    - **Mechanism**: A graph is constructed with ROIs as nodes. Edges are established based on two criteria: spatial proximity (geometric centroid distance $d_{ij} < \tau_d$) and morphological similarity (feature cosine similarity $s_{ij} > \tau_s$). Edge features encode spatial relationships (distance, relative direction, volume ratio) and morphological relationships (similarity, mean CT/PET intensity). Message passing is performed via GATv2 attention.
    - **Design Motivation**: Processing each ROI in isolation misses critical clinical relationships—spatially adjacent lesions may indicate local invasion, while distant lesions with similar metabolic patterns may suggest metastasis. Graph-based modeling captures both types of relationships simultaneously.

3. **Clinical Evaluation Metrics (RoI Coverage and RoI Quality Index)**

    - **Function**: Evaluate report quality from a clinical perspective, beyond conventional NLP metrics.
    - **Mechanism**: RoI Coverage applies Hungarian matching to pair predicted ROIs with ground-truth ROIs, yielding Precision/Recall/F1. The RoI Quality Index (RoIQ) evaluates attribute-level accuracy on matched ROI pairs using the formula $\text{RoIQ} = \sqrt{S_{\text{region}} \cdot S_{\text{lesion}}} \times \frac{1}{|\mathcal{A}|}\sum S_k$, applying a nonlinear penalty via the geometric mean on core attributes (anatomical region and lesion type).
    - **Design Motivation**: Conventional BLEU/ROUGE metrics measure only lexical overlap and cannot assess clinical correctness. The hierarchical design of RoIQ ensures that errors in core attributes (mislocalization or incorrect lesion type) cannot be compensated by high scores on secondary attributes.

### Loss & Training
A four-stage progressive training strategy is adopted. **Stage 1**: pretrains the dual-stream encoder with CLIP contrastive learning. **Stage 2**: freezes the encoder and LLM, training the Q-Former for visual-language alignment. **Stage 3**: introduces the local context module (SPP-RoI + GATv2) while freezing parameters from the first two stages. **Stage 4**: performs end-to-end fine-tuning with LoRA ($r=16, \alpha=32$).

## Key Experimental Results

### Main Results

| Method | BLEU-4 | ROUGE-L | BERT | Correct ROIs | RoI F1 | RoIQ |
|--------|--------|---------|------|--------------|--------|------|
| MedM-VL | 31.69 | 50.00 | 91.92 | 62/416 | 14.16 | 23.24 |
| M3D-LaMed | 44.30 | 64.39 | 85.90 | 193/416 | 39.83 | 36.31 |
| HiRRA (No RoI) | 52.48 | 66.51 | 95.13 | 187/416 | 37.58 | 33.89 |
| **HiRRA** | **62.80** | **69.66** | **95.79** | **223/416** | **42.47** | **56.86** |
| Gain Δ% | +19.7% | +4.7% | +0.7% | +15.5% | +6.6% | **+45.8%** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| HiRRA (full) | RoIQ=56.86 | Complete model |
| w/o ROI annotations (No RoI) | RoIQ=33.89 | ROI-level supervision contributes most; RoIQ drops by 22.97 |
| CT encoder only | Performance decreases | Missing PET metabolic information |
| PET encoder only | Larger performance decrease | Missing CT anatomical structural information |

### Key Findings
- ROI-level annotations yield the most significant improvements on clinical metrics—RoIQ rises from 33.89 to 56.86 (+45.8%), demonstrating that fine-grained region-level supervision is key to reducing hallucinations.
- All existing VLMs perform extremely poorly on Vietnamese (BLEU-4 near 0), revealing a severe lag in medical AI for low-resource languages.
- GATv2 graph relational modeling is most beneficial for diagnosing metastatic disease, as it can associate distant lesions with similar metabolic patterns.
- Dual-modality fusion substantially outperforms single-modality baselines; CT provides localization while PET provides functional information, and neither can be omitted.

## Highlights & Insights
- The **ROI-level annotation strategy** transforms report generation from an "end-to-end black box" into an interpretable "analyze-then-synthesize" paradigm, a design philosophy transferable to other 3D medical imaging report generation tasks.
- The **RoIQ metric design** is particularly elegant—by applying a nonlinear penalty on core attributes via the geometric mean, it ensures that errors in anatomical localization or lesion type cannot be masked by high scores on secondary attributes. This hierarchical evaluation concept is generalizable to any task requiring quality assessment of structured outputs.
- The four-stage progressive training strategy effectively balances learning across modules—from global alignment to local refinement and finally end-to-end fine-tuning.

## Limitations & Future Work
- The dataset scale is relatively small (200 patients, 600 samples, 1,960 ROIs), limiting model generalizability.
- The framework currently supports only Vietnamese; extension to additional languages is needed to validate its generality.
- ROI annotation relies on manual labeling by 8 nuclear medicine physicians, which is costly and difficult to scale.
- Incorporating an automatic ROI detection module to replace manual annotation could enable fully end-to-end automation.
- Future work may explore making the ROI-level reasoning process explicit to provide more detailed diagnostic explanations.

## Related Work & Insights
- **vs. M3D-LaMed**: M3D-LaMed is a general-purpose 3D medical VLM but lacks ROI-level supervision, resulting in substantially inferior clinical metrics compared to HiRRA (RoIQ 36.31 vs. 56.86).
- **vs. ViMed-PET**: ViMed-PET provides Vietnamese PET/CT report data but includes only global report annotations without fine-grained ROI-level labels.
- **vs. CT2Rep**: CT2Rep is limited to global CT-only report mapping and supports neither the PET modality nor ROI-level reasoning.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First 3D PET/CT dataset with ROI-level annotations; graph-enhanced framework design is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive multi-baseline comparisons, ablation analyses, and clinical evaluation metrics, though the dataset scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with well-articulated clinical motivation.
- **Value**: ⭐⭐⭐⭐⭐ The dataset and evaluation metrics make a significant contribution to the medical imaging AI community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](../../CVPR2026/medical_imaging/unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)
- [\[ACL 2026\] Beyond Prompt: Fine-grained Simulation of Cognitively Impaired Standardized Patients via Stochastic Steering](beyond_prompt_fine-grained_simulation_of_cognitively_impaired_standardized_patie.md)
- [\[ACL 2026\] OmniCompliance-100K: A Multi-Domain Rule-Grounded Real-World Safety Compliance Dataset](omnicompliance-100k_a_multi-domain_rule-grounded_real-world_safety_compliance_da.md)
- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)
- [\[ACL 2026\] Anonpsy: A Graph-Based Framework for Structure-Preserving De-identification of Psychiatric Narratives](anonpsy_a_graph-based_framework_for_structure-preserving_de-identification_of_ps.md)

</div>

<!-- RELATED:END -->
