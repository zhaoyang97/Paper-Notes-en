---
title: >-
  [Paper Note] Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD
description: >-
  [CVPR 2026][Medical Imaging][CBCT report generation] This paper proposes CBCTRepD, a bilingual report generation system for oral and maxillofacial CBCT, trained on a high-quality paired dataset of 7,408 cases. A multi-level evaluation framework is introduced to validate its tiered empowerment effect on novice, intermediate, and senior radiologists within a radiologist–AI collaborative workflow.
tags:
  - CVPR 2026
  - Medical Imaging
  - CBCT report generation
  - oral and maxillofacial imaging
  - radiologist–AI collaboration
  - multi-level evaluation
  - bilingual reporting
date: 2026-05-08
content_hash: 9e070019d28a9f0e
---

# Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD

**Conference**: CVPR 2026
**arXiv**: [2603.10933](https://arxiv.org/abs/2603.10933)
**Code**: N/A
**Area**: Medical Imaging
**Keywords**: CBCT report generation, oral and maxillofacial imaging, radiologist–AI collaboration, multi-level evaluation, bilingual reporting

## TL;DR

This paper proposes CBCTRepD, a bilingual report generation system for oral and maxillofacial CBCT, trained on a high-quality paired dataset of 7,408 cases. A multi-level evaluation framework is introduced to validate its tiered empowerment effect on novice, intermediate, and senior radiologists within a radiologist–AI collaborative workflow.

## Background & Motivation

**Background**: Oral and maxillofacial cone-beam CT (CBCT) is a widely used three-dimensional imaging modality in dental clinical practice. Generative AI has achieved notable progress in report generation for two-dimensional imaging modalities such as chest X-rays (e.g., R2Gen, CheXagent).

**Limitations of Prior Work**: CBCT report generation faces three major bottlenecks: (1) high-quality paired CBCT–report data are extremely scarce, with virtually no publicly available datasets; (2) CBCT is volumetric data involving 55 oral disease entities, making interpretation far more complex than two-dimensional imaging; (3) existing evaluation paradigms assess only the quality of direct AI output, overlooking the clinically common scenario of radiologist–AI collaboration.

**Key Challenge**: AI report generation technology is mature, yet the data infrastructure and evaluation frameworks lag behind — without large-scale paired data, reliable models cannot be trained; without collaborative evaluation frameworks, the actual clinical value of AI in real-world workflows cannot be measured.

**Goal**: To develop a CBCT report generation system that integrates into routine radiological workflows, and to quantify its empowerment effect on clinicians of varying experience levels through a multi-level evaluation framework.

**Key Insight**: Simultaneously addressing data, model, and evaluation dimensions — constructing a large-scale dataset, developing a bilingual report generation system, and designing a collaborative evaluation framework.

**Core Idea**: A CBCT report generation system is trained on 7,408 paired cases; an "AI draft + clinician editing" collaborative paradigm and a multi-level evaluation framework are employed to demonstrate clinically meaningful improvements across radiologists of all experience levels.

## Method

### Overall Architecture

CBCTRepD is an end-to-end report generation system for oral and maxillofacial CBCT. The pipeline proceeds as follows: CBCT volumetric data → 3D image encoder (extracting spatial features from multiple anatomical regions) → report generation module (automatically producing structured bilingual draft reports in Chinese and English) → collaborative editing interface (where clinicians revise and supplement the AI draft) → final clinical report. The system is designed as an AI collaboration tool embedded in routine radiological workflows.

### Key Designs

1. **Large-Scale High-Quality Paired Dataset**:
   - *Function*: Construct the largest paired imaging–report dataset to date in the oral and maxillofacial CBCT domain.
   - *Mechanism*: Approximately 7,408 CBCT studies covering 55 oral disease entities (periodontitis, periapical lesions, jaw cysts, impacted teeth, temporomandibular joint disorders, etc.), encompassing multi-vendor devices, diverse scanning parameters, and varied clinical scenarios. Annotations are bilingual (Chinese and English) and reviewed by senior radiologists.
   - *Design Motivation*: High-quality paired data are the foundation for training reliable report generation models. Given the near-total absence of publicly available resources in this domain, the dataset itself constitutes a significant contribution.

2. **Multi-Level Clinical Evaluation Framework**:
   - *Function*: Establish an evaluation system covering two levels: "direct AI output" and "radiologist–AI collaborative output."
   - *Mechanism*: Layer 1 evaluates the quality of AI-generated report drafts independently; Layer 2 evaluates the quality of final reports produced by radiologists of different experience levels (novice/intermediate/senior) after editing the AI drafts. Evaluation dimensions include automatic metrics (BLEU, ROUGE, CIDEr, etc.), radiologist assessments (completeness, standardization, accuracy), and clinician assessments (adequacy of diagnostic information, decision-support value).
   - *Design Motivation*: Conventional evaluation measures only AI output quality, whereas real clinical practice operates in a collaborative mode. The multi-level framework more accurately reflects the system's value in actual deployment.

3. **Bilingual Structured Report Generation**:
   - *Function*: Generate structured reports with both Chinese and English output.
   - *Mechanism*: Based on features extracted by the 3D image encoder, the system automatically generates systematic reports covering each anatomical region, with emphasis on detecting co-existing lesions across regions.
   - *Design Motivation*: Bilingual output enables cross-regional deployment; structured output enhances report standardization and reduces omissions.

### Loss & Training

The system is trained end-to-end on approximately 7,408 paired cases using a standard medical report generation paradigm (cross-entropy loss combined with reinforcement learning fine-tuning to optimize metrics such as CIDEr). Training data are reviewed by senior radiologists to ensure annotation quality.

## Key Experimental Results

### Main Results

| Evaluation Subject | Report Structural Standardization | Lesion Description Completeness | Writing Standardization | Cross-Regional Lesion Coverage |
|---|---|---|---|---|
| Novice Radiologist | Low | Frequent omissions | Non-standard | Low |
| CBCTRepD AI Draft | Intermediate level | Comparable to intermediate | Near intermediate | Systematic coverage |
| Intermediate Radiologist | Moderate | Occasional omissions | Standardized | Moderate |
| Senior Radiologist | High | Rarely omitted | Highly standardized | High |

### AI–Clinician Collaborative Effect

| Clinician Experience Level | Without AI Assistance | With CBCTRepD | Improvement |
|---|---|---|---|
| Novice | Novice level | Near intermediate level | Significant gains in completeness and standardization; fewer omissions |
| Intermediate | Intermediate level | Near senior level | More accurate lesion descriptions; increased cross-regional attention |
| Senior | Senior level | Senior level+ | Fewer omission-related errors, including clinically significant missed findings |

### Key Findings

- Reports directly generated by CBCTRepD are comparable to those of intermediate-level radiologists in writing quality and standardization.
- Radiologist–AI collaboration yields consistent and clinically meaningful improvements across all three experience levels.
- The most pronounced improvement is observed among novice clinicians (a full tier uplift); for senior clinicians, the value lies in a "safety net effect" — reducing fatigue-induced omissions.
- The system is particularly effective in prompting clinicians to attend to co-existing lesions across anatomical regions.

## Highlights & Insights

- The dataset contribution itself is of major value: the 7,408-case paired dataset covering 55 disease entities fills a critical data gap in the oral and maxillofacial CBCT domain.
- The evaluation paradigm represents an innovation: shifting from "evaluating AI output" to "evaluating AI–human collaborative output" more closely reflects real-world clinical deployment.
- Tiered empowerment across experience levels: unlike most systems that benefit only novice users, CBCTRepD demonstrates clear value for senior clinicians as well (via omission reduction), which is critical for driving clinical adoption.
- Focus on omissions rather than false positives: the system is especially effective at reducing omission errors, which carry greater safety implications than false positives in clinical radiology.

## Limitations & Future Work

- Data sources may exhibit single-center bias; multi-center validation is a prerequisite for generalization.
- Sample distribution across 55 disease entities may be imbalanced, raising concerns about report generation quality for rare conditions.
- The computational and storage overhead of CBCT volumetric data is substantial; inference efficiency and hardware requirements for real-world deployment warrant optimization.
- The collaborative mode may introduce automation bias — awareness of AI assistance may alter clinicians' diagnostic behavior, and long-term effects require longitudinal investigation.
- Direct quantitative comparison with other medical report generation methods (e.g., RadFM, LLaVA-Med) is absent.

## Related Work & Insights

- **vs. R2Gen / CheXagent**: These methods have achieved success in report generation for two-dimensional chest X-ray imaging, but automatic report generation for three-dimensional CBCT was previously almost nonexistent; CBCTRepD fills this gap.
- **vs. RadFM / LLaVA-Med**: Multimodal medical large language models have explored the potential of AI-assisted radiology, but lack systematic evaluation frameworks for collaborative scenarios; CBCTRepD's multi-level evaluation framework is an important complement.
- **vs. Dental Imaging AI (ToothNet, etc.)**: Existing dental AI primarily focuses on tooth segmentation and lesion detection, with little attention to automatic report generation.

## Rating

- Novelty: ⭐⭐⭐⭐ First report generation system targeting oral and maxillofacial CBCT; the dataset and evaluation framework fill a domain gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ The multi-level evaluation framework is well-designed, encompassing automatic metrics, radiologist expert assessment, and clinician evaluation.
- Writing Quality: ⭐⭐⭐⭐ The problem–method–result logical chain is coherent, with high information density in the abstract.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to oral radiology clinical practice; the validated tiered empowerment effect serves as a model for AI deployment and adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EchoAgent: Towards Reliable Echocardiography Interpretation with "Eyes", "Hands" and "Minds"](echoagent_towards_reliable_echocardiography_interpretation_with_eyeshands_and_mi.md)
- [\[CVPR 2026\] Unlocking Multi-Site Clinical Data: A Federated Approach to Privacy-First Child Autism Behavior Analysis](unlocking_multi-site_clinical_data_a_federated_approach_to_privacy-first_child_a.md)
- [\[CVPR 2026\] SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](sd_fsmis_adapting_stable_diffusion_for_few_shot_medical_image_segmentation.md)
- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[CVPR 2026\] MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)

</div>

<!-- RELATED:END -->
