---
title: >-
  [Paper Note] HypEHR: Hyperbolic Modeling of Electronic Health Records for Efficient Question Answering
description: >-
  [ACL 2026 (Findings)][Medical Imaging][EHR question answering] This paper proposes HypEHR, a 22M-parameter Lorentz hyperbolic model that embeds medical codes, patient visits, and questions into hyperbolic space. Through hierarchy-aware regularization aligned with the ICD ontology structure, HypEHR achieves performance comparable to LLM-based approaches on the MIMIC-IV EHR question answering task.
tags:
  - ACL 2026 (Findings)
  - Medical Imaging
  - EHR question answering
  - hyperbolic space
  - Lorentz model
  - ICD hierarchy modeling
  - lightweight clinical model
date: 2026-05-08
content_hash: d772d88f3aab387c
---

# HypEHR: Hyperbolic Modeling of Electronic Health Records for Efficient Question Answering

**Conference**: ACL 2026 (Findings)
**arXiv**: [2604.21027](https://arxiv.org/abs/2604.21027)
**Code**: [https://github.com/yuyuliu11037/HypEHR](https://github.com/yuyuliu11037/HypEHR)
**Area**: Medical Imaging / Clinical NLP
**Keywords**: EHR question answering, hyperbolic space, Lorentz model, ICD hierarchy modeling, lightweight clinical model

## TL;DR
This paper proposes HypEHR, a 22M-parameter Lorentz hyperbolic model that embeds medical codes, patient visits, and questions into hyperbolic space. Through hierarchy-aware regularization aligned with the ICD ontology structure, HypEHR achieves performance comparable to LLM-based approaches on the MIMIC-IV EHR question answering task.

## Background & Motivation

**Background**: Electronic health record question answering (EHR-QA) aims to answer natural language clinical questions about longitudinal patient records. Current approaches fall into three categories: EHR representation learning (sequence/graph models), Text-to-SQL semantic parsing, and retrieval-augmented LLM pipelines based on GPT-3.5/4.

**Limitations of Prior Work**: While accurate, these methods incur substantial computational overhead, are difficult to deploy under strict privacy constraints, and largely ignore the strong structural priors inherent in EHR data. LLM-based pipelines contain trillions of parameters, making on-premises hospital deployment infeasible.

**Key Challenge**: Medical codes and patient trajectories are inherently hierarchical (ICD codes are organized as chapter → block → category → subcategory). Euclidean embeddings distort such tree-like structures, yet existing methods fail to exploit this geometric prior.

**Goal**: To construct a compact model geometrically consistent with the intrinsic structure of EHRs, achieving performance comparable to LLMs on complex QA tasks with far fewer parameters.

**Key Insight**: Hyperbolic space can embed hierarchical structures with arbitrarily low distortion. Prior work has demonstrated that hyperbolic embeddings improve medical code hierarchy modeling, yet final patient representations are still modeled in Euclidean space. The authors propose modeling patient-level representations directly in hyperbolic space.

**Core Idea**: Embed ICD codes, visits, and questions in the Lorentz hyperbolic manifold; align with the ICD ontology via hierarchy-aware regularization; and answer questions using geometrically consistent cross-attention and type-specific pointer heads.

## Method

### Overall Architecture
HypEHR operates in two stages: (1) patient encoder pre-training — learning joint next-visit prediction and hierarchy regularization objectives in hyperbolic space; and (2) QA fine-tuning — freezing the patient encoder and training answer-type-specific heads (Boolean, concept, numerical, and count).

### Key Designs

1. **Hyperbolic Clinical Sequence Encoder**:

    - Function: Encodes a patient's visit sequence into contextualized representations in hyperbolic space.
    - Mechanism: Each medical code $c$ is embedded in the Lorentz hyperbolic manifold $\mathbb{H}_L^d$. Hyperbolic attention aggregates code embeddings within each visit to obtain visit representations $h_t$. A multi-layer Lorentz Transformer — with self-attention, residual connections, and normalization all adapted to the Lorentz manifold — then processes the visit sequence to produce contextualized states $\{z_t\}_{t=1}^T$ and a global summary $z_{\text{[CLS]}}$.
    - Design Motivation: Operating entirely in hyperbolic space avoids information loss from repeated Euclidean→hyperbolic→Euclidean conversions, allowing representations to naturally preserve hierarchical structure.

2. **Hierarchy-Aware Regularization**:

    - Function: Encodes the tree structure of the ICD ontology into the embedding space.
    - Mechanism: Based on the ICD code tree (chapter → block → category → subcategory), two constraints are imposed. The radial hierarchy loss $\mathcal{L}_{\text{rad}}$ requires parent node embeddings to have smaller hyperbolic norms than child nodes (i.e., deeper nodes lie farther from the origin). The relative hierarchy loss $\mathcal{L}_{\text{rel}}$ applies triplet constraints so that codes sharing a common ancestor are closer to each other than to unrelated codes. The joint objective is $\mathcal{L} = \mathcal{L}_{\text{diag}} + \lambda \mathcal{L}_{\text{hier}}$.
    - Design Motivation: The next-visit prediction task alone cannot explicitly capture hierarchical relationships among codes. By enforcing depth–radius monotonicity and within-family proximity, the geometry of the embedding space is aligned with the ICD ontology.

3. **Hyperbolic EHR-QA Model**:

    - Function: Answers natural language clinical questions conditioned on the frozen patient encoder.
    - Mechanism: Questions are encoded into Euclidean vectors by a biomedical pre-trained language model, then projected into hyperbolic space via an affine map followed by the exponential map to obtain $z_q$. Hyperbolic cross-attention — where attention scores are the negatively scaled hyperbolic distances $s_t = -\gamma d_{\mathbb{H}}(z_q, z_t)$ — produces a question-conditioned patient summary $z_{p|q}^{\text{visit}}$. A second-order attention over codes within the top-$k$ visits then yields a code-level reasoning vector $z_{p|q}^{\text{code}}$. Answer-type-specific heads are invoked based on the predicted answer type.
    - Design Motivation: Hyperbolic distance naturally distinguishes concepts at different depths of a hierarchy, making it more suitable than Euclidean distance for attention over complex ontological structures.

### Loss & Training
Pre-training employs a joint loss combining next-visit diagnosis prediction (binary cross-entropy) with hierarchy regularization. In the QA stage, both the language encoder and patient encoder are frozen; only the type-specific classification heads are trained using standard cross-entropy loss. The total parameter count is 22M.

## Key Experimental Results

### Main Results

| Model | EHRXQA (Acc%) | MIMIC-Instr (Acc%) |
|------|-------------|-------------------|
| RETAIN | 81.19 ± 1.95 | 65.91 ± 0.84 |
| NeuralSQL (GPT-5.2) | **95.97** ± 0.50 | 75.17 ± 0.73 |
| Llama-3 | 82.88 ± 1.38 | 70.90 ± 0.86 |
| Llemr | 87.25 ± 0.77 | **77.53** ± 0.54 |
| EHRAgent | 93.06 ± 1.09 | 74.16 ± 0.56 |
| **HypEHR (22M)** | 89.53 ± 0.60 | 76.02 ± 0.41 |

### Ablation Study

| Configuration | EHRXQA | MIMIC-Instr |
|------|--------|-------------|
| w/o $\mathcal{L}_{\text{hier}}$ | 82.72 ± 3.41 | 70.38 ± 0.54 |
| w/o pretraining | 74.05 ± 4.76 | 68.12 ± 1.39 |
| EucEHR (Euclidean) | 80.33 ± 1.14 | 69.88 ± 1.07 |
| **HypEHR** | **89.53** ± 0.60 | **76.02** ± 0.41 |

### Key Findings
- Patient encoder pre-training is the most critical component; removing it causes a 15.5 percentage point drop on EHRXQA.
- Under identical pre-training settings, the hyperbolic model (HypEHR) outperforms the Euclidean counterpart (EucEHR) by 9.2 percentage points on EHRXQA and 6.1 points on MIMIC-Instr.
- Hierarchy regularization is not the primary performance driver, but it produces significant improvement in explicitly modeling the hierarchical structure of codes.
- Geometric analysis confirms that code embedding radii in the Lorentz model increase monotonically with ICD tree depth, whereas Euclidean norms show only a weak and noisy relationship with depth.

## Highlights & Insights
- Achieving EHR-QA performance approaching trillion-parameter LLMs with only 22M parameters underscores the power of structural priors (hyperbolic geometry + ICD hierarchy). This "replace parameters with the right inductive bias" principle is especially valuable in resource-constrained clinical settings.
- The radial hierarchy loss is particularly elegant: it exploits the property that hyperbolic space expands exponentially away from the origin, allowing finer-grained diagnosis codes to naturally occupy larger representational volumes.
- The hyperbolic cross-attention mechanism is transferable to any scenario requiring retrieval or attention over hierarchical structures, such as knowledge graph reasoning and taxonomy navigation.

## Limitations & Future Work
- The system relies on preprocessing answers into four fixed formats (Boolean/concept/numerical/count), introducing engineering overhead and precluding open-ended generative responses.
- Hyperbolic neural networks are computationally more expensive than their Euclidean counterparts, and immature public libraries for hyperbolic geometry may cause instability during large-scale training.
- Evaluation is conducted solely on de-identified MIMIC-IV data; performance in real clinical deployment environments has not been validated.
- The authors explicitly emphasize that this system should function as a decision-support tool under human supervision, and not as an autonomous clinical agent.

## Related Work & Insights
- **vs. NeuralSQL (GPT-5.2)**: NeuralSQL leverages GPT-5.2 to generate SQL queries, achieving 96% on EHRXQA, but depends on an extremely large LLM and naturally aligns with the SQL paradigm. HypEHR reaches 89.5% in a non-SQL paradigm with fewer than 1/1000 of the parameters.
- **vs. Llemr**: Llemr is the official LLM-based EHR-QA baseline, reaching 77.5% on MIMIC-Instr. HypEHR approaches this level at 76% while substantially reducing parameter count and inference cost.
- **vs. Lu et al. (2023)**: That work also employs hyperbolic embeddings for medical code hierarchy modeling, but the final patient representations remain in Euclidean space. HypEHR extends hyperbolic modeling to the patient level throughout, which constitutes the key distinction.

## Rating
- Novelty: ⭐⭐⭐⭐ Extending hyperbolic geometry from the code level to patient-level QA is a natural progression; the hierarchy regularization design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two QA benchmarks, four clinical prediction tasks, detailed ablations, and geometric analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated, though the technical details of hyperbolic space may present a high barrier for non-specialist readers.
- Value: ⭐⭐⭐⭐ Provides a lightweight alternative for privacy-sensitive clinical settings; the exploitation of geometric priors is broadly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records](../../AAAI2026/medical_imaging/clicare_grounding_large_language_models_in_clinical_guidelines_for_decision_supp.md)
- [\[ACL 2026\] Query Pipeline Optimization for Cancer Patient Question Answering Systems](query_pipeline_optimization_for_cancer_patient_question_answering_systems.md)
- [\[ICLR 2026\] Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering](../../ICLR2026/medical_imaging/q-fsru_quantum-augmented_frequency-spectral_for_medical_visual_question_answerin.md)
- [\[AAAI 2026\] Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering](../../AAAI2026/medical_imaging/expert-guided_prompting_and_retrieval-augmented_generation_for_emergency_medical.md)
- [\[AAAI 2026\] TrinityDNA: A Bio-Inspired Foundational Model for Efficient Long-Sequence DNA Modeling](../../AAAI2026/medical_imaging/trinitydna_a_bio-inspired_foundational_model_for_efficient_long-sequence_dna_mod.md)

</div>

<!-- RELATED:END -->
