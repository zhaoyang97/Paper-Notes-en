---
title: >-
  [Paper Note] Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD
description: >-
  [CVPR 2026][Medical Imaging][CBCT] This work constructs a large-scale CBCT-report paired dataset of 7,408 cases covering 55 oral diseases, and develops CBCTRepD…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "CBCT"
  - "oral-maxillofacial report generation"
  - "bilingual system"
  - "human-AI collaboration"
  - "multi-level evaluation"
date: 2026-05-08
content_hash: 7823d0b407f97547
---

# Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD

**Conference**: CVPR 2026
**arXiv**: [2603.10933](https://arxiv.org/abs/2603.10933)  
**Code**: None  
**Area**: Medical Imaging / Report Generation
**Keywords**: CBCT, oral-maxillofacial report generation, bilingual system, human-AI collaboration, multi-level evaluation

## TL;DR
This work constructs a large-scale CBCT-report paired dataset of 7,408 cases covering 55 oral diseases, and develops CBCTRepD, a bilingual oral-maxillofacial CBCT report generation system. Through a collaborative paradigm of AI-generated drafts followed by radiologist editing, the system is shown via multi-level clinical evaluation to elevate junior radiologists to an intermediate level, intermediate radiologists to near-senior level, and reduce omissions for senior radiologists.

## Background & Motivation

**Background**: Generative AI for automated medical report generation is advancing rapidly. Chest X-ray report generation has seen considerable mature work (e.g., CheXpert, RadFM), whereas report generation for oral-maxillofacial cone-beam CT (CBCT) remains in its early stages.

**Limitations of Prior Work**: Oral-maxillofacial CBCT report generation faces two core obstacles: (1) high-quality paired CBCT-report data are extremely scarce, with existing public datasets containing virtually no paired annotations for oral-maxillofacial CBCT and clinical reports; (2) CBCT is volumetric 3D data whose interpretive complexity far exceeds that of 2D panoramic radiographs or CT slices, involving multiple anatomical regions and a large number of potential lesion types, posing greater demands on AI modeling.

**Key Challenge**: Radiologists of different experience levels differ substantially in their ability to interpret CBCT. Junior radiologists frequently miss lesions and produce non-standardized reports; even senior radiologists may overlook co-existing pathologies across anatomical regions due to attentional limitations. AI-assisted systems that pursue full automation without considering integration with clinical expertise are unlikely to gain clinical acceptance.

**Goal**: To construct a practical AI-assisted reporting system for the oral-maxillofacial CBCT domain—where standardized data are lacking—and to quantify through rigorous multi-level clinical evaluation the real-world assistive effect on radiologists at different experience levels.

**Key Insight**: Rather than pursuing full automation, the paper adopts a collaborative paradigm in which AI generates an initial draft and the radiologist edits it, more closely reflecting actual clinical workflows. A multi-level evaluation framework encompassing both automatic metrics and human assessment is established in parallel.

**Core Idea**: Train a specialized CBCT report generation model on a large-scale paired dataset and employ a human-AI collaborative paradigm to bridge the report quality gap among radiologists of varying experience levels.

## Method

### Overall Architecture
CBCTRepD follows a complete pipeline of dataset construction → model training → multi-level evaluation. The input is an oral-maxillofacial CBCT volumetric scan; the output is a bilingual structured report in Chinese and English. The system is designed to integrate into radiologists' daily workflows via a collaborative mode: AI first generates a report draft (direct AI draft), which the radiologist then reviews and edits (collaboration report) to yield a higher-quality final report.

### Key Designs
1. **Large-Scale Oral-Maxillofacial CBCT-Report Paired Dataset**:

    - **Function**: Collects and annotates approximately 7,408 CBCT-report paired cases from real clinical environments.
    - **Mechanism**: Covers 55 oral disease entities (including dental caries, periodontal disease, periapical periodontitis, impacted teeth, jaw cysts, etc.) across multiple acquisition devices and imaging settings, with bilingual (Chinese–English) paired annotations. Data quality is verified and standardized by professional radiologists.
    - **Design Motivation**: Large-scale public datasets are absent in the oral-maxillofacial CBCT domain. The scale of 7,408 cases represents a first-of-its-kind magnitude in this sub-field; coverage of 55 disease entities ensures the model can handle diverse clinical presentations.

2. **Bilingual End-to-End Report Generation System**:

    - **Function**: Directly generates structured Chinese and English clinical reports from 3D CBCT volumetric data.
    - **Mechanism**: The system receives a CBCT volume as input, extracts features, and decodes them into clinically compliant text reports. Bilingual output is supported to accommodate clinical needs across different linguistic settings. Generated reports contain findings and impressions organized by anatomical region, adhering to standard radiological report formats.
    - **Design Motivation**: Bilingual capability broadens the system's applicability; an end-to-end design avoids error propagation across multi-stage pipelines; structured output format enables seamless integration into clinical workflows.

3. **Multi-Level Clinical Evaluation Framework**:

    - **Function**: Establishes a clinically grounded evaluation system that assesses both AI-generated drafts and radiologist-edited collaborative reports.
    - **Mechanism**: Evaluation operates at three levels — (a) automatic metrics assessing AI draft quality; (b) radiologist-centered evaluation, where peers review reports for accuracy, completeness, and standardization; (c) clinician-centered evaluation, judging practical value from a clinical decision-making perspective. Results are reported separately for junior, intermediate, and senior radiologists.
    - **Design Motivation**: Automatic metrics such as BLEU/ROUGE alone cannot reflect clinical value. The multi-level design incorporating human assessment better approximates real clinical scenarios. Stratified analysis by experience level, rather than reporting average performance only, reveals differential benefits of AI assistance across radiologist groups.

### Loss & Training
End-to-end training is conducted on the 7,408 paired cases, with joint optimization of Chinese and English report generation from CBCT volumetric input. Specific network architectures and loss function details are unavailable due to inaccessibility of the full paper.

## Key Experimental Results

### Main Results

| Evaluation Dimension | CBCTRepD Performance | Baseline | Notes |
|---|---|---|---|
| AI draft quality | ≈ Intermediate radiologist | — | Verified by both automatic metrics and human review |
| Writing standardization | ≈ Intermediate radiologist | — | Report structure meets clinical standards |
| Clinical omission rate | Significantly reduced | Unaided physician writing | Including clinically significant missed lesions |

### Stratified Assistive Effects

| Radiologist Level | Before CBCTRepD | After CBCTRepD | Gain |
|---|---|---|---|
| Junior → Intermediate | Junior-level reports | Approaching intermediate level | Significant improvement in completeness and standardization |
| Intermediate → Senior | Intermediate-level reports | Approaching senior level | Improved diagnostic accuracy and detail |
| Senior assistance | Occasional omissions | Reduced omission errors | Especially for co-existing cross-regional lesions |

### Ablation Study

| Configuration | Effect | Notes |
|---|---|---|
| Direct AI generation vs. human-AI collaboration | Collaboration significantly outperforms direct generation | Confirms necessity of radiologist editing |
| Coverage of 55 disease entities | High coverage rate | Encompasses common and rare oral diseases |
| Multi-device training | Cross-device generalization | Mixed training across different CBCT devices |

### Key Findings
- AI-generated drafts approach intermediate radiologist level in writing quality and standardization, serving as a reliable starting point.
- The human-AI collaborative paradigm (AI draft + radiologist editing) consistently outperforms unaided radiologist writing across all experience levels.
- CBCTRepD is particularly effective at improving report structure, reducing omissions, and enhancing attention to co-existing pathologies across anatomical regions.
- Even for senior radiologists, the system provides clinically meaningful benefit by prompting awareness of potentially overlooked lesions.

## Highlights & Insights
- **Strong data contribution**: The 7,408 paired CBCT-report cases covering 55 disease entities constitute important infrastructure for oral-maxillofacial CBCT report generation research.
- **Pragmatic clinical positioning**: Rather than pursuing full automation to replace radiologists, the system is positioned as a collaborative tool; the AI draft + radiologist editing paradigm is more likely to achieve clinical adoption.
- **Elegant stratified evaluation design**: Quantifying differential gains for junior, intermediate, and senior radiologists is more persuasive than a single aggregate performance figure.
- **Focus on omission-type errors**: Particular emphasis on reducing missed diagnoses—including co-existing lesions across anatomical regions—addresses the most safety-critical error type in clinical practice.
- **Transferable evaluation framework**: The three-tier framework of automatic metrics + radiologist evaluation + clinician evaluation is generalizable to other medical AI systems.

## Limitations & Future Work
- The dataset is confined to the oral-maxillofacial domain; generalization to other CBCT applications (orthopedics, otolaryngology) requires additional validation.
- The full paper is not publicly accessible (HTML/ar5iv unavailable), precluding in-depth analysis of specific network architectures, loss functions, and training details.
- Although 7,408 cases is large-scale for this sub-field, it remains orders of magnitude smaller than general medical report generation datasets (e.g., MIMIC-CXR with 200K+ reports).
- Long-term clinical impact assessment is absent — longitudinal studies on whether AI assistance leads to increased dependence or skill degradation among radiologists are lacking.
- The specific strategy for processing 3D CBCT volumes (slice-based vs. global encoding) and its impact on generation quality is unknown.
- Evaluation involves only oral-maxillofacial radiologists and clinicians; the number and diversity of reviewers are not reported.

## Related Work & Insights
- **vs. CheXpert/MIMIC-CXR report generation**: These works target 2D chest radiographs with larger data scales but lower interpretive complexity than 3D CBCT. CBCTRepD is the first to achieve complete report generation and clinical evaluation on 3D oral imaging.
- **vs. RadFM and other general medical foundation models**: General-purpose models offer broad coverage but lack domain-specific depth in oral-maxillofacial imaging. CBCTRepD achieves clinically closer results in this specialized domain through a purpose-built dataset.
- **vs. traditional CBCT-assisted diagnosis**: Prior oral AI work has mostly focused on single tasks (e.g., caries detection, root canal segmentation); CBCTRepD advances toward complete report generation.

## Rating
- **Novelty**: ⭐⭐⭐ — Methodological innovation is limited; the core contribution lies in constructing the large-scale paired dataset and the multi-level evaluation framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The multi-level clinical evaluation design is rigorous and stratified analysis is convincing, though further details cannot be verified without full paper access.
- **Writing Quality**: ⭐⭐⭐⭐ — The abstract is clearly structured, information-dense, and articulates clinical value effectively.
- **Value**: ⭐⭐⭐⭐ — The dataset and evaluation framework directly advance the oral imaging AI community; validation of the human-AI collaborative paradigm offers meaningful reference for medical AI deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EchoAgent: Towards Reliable Echocardiography Interpretation with "Eyes", "Hands" and "Minds"](echoagent_towards_reliable_echocardiography_interpretation_with_eyeshands_and_mi.md)
- [\[ICLR 2026\] Bridging Explainability and Embeddings: BEE Aware of Spuriousness](../../ICLR2026/medical_imaging/bridging_explainability_and_embeddings_bee_aware_of_spuriousness.md)
- [\[ICLR 2026\] Causal Interpretation of Neural Network Computations with Contribution Decomposition](../../ICLR2026/medical_imaging/causal_interpretation_of_neural_network_computations_with_contribution_decomposi.md)
- [\[AAAI 2026\] Unsupervised Multi-Parameter Inverse Solving for Reducing Ring Artifacts in 3D X-Ray CBCT](../../AAAI2026/medical_imaging/unsupervised_multi-parameter_inverse_solving_for_reducing_ring_artifacts_in_3d_x.md)
- [\[ACL 2026\] CT-Flow: Orchestrating CT Interpretation Workflow with Model Context Protocol Servers](../../ACL2026/medical_imaging/ct-flow_orchestrating_ct_interpretation_workflow_with_model_context_protocol_ser.md)

</div>

<!-- RELATED:END -->
