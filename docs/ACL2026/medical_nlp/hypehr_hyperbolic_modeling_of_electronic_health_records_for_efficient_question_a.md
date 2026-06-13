---
title: >-
  [Paper Note] HypEHR: Hyperbolic Modeling of Electronic Health Records for Efficient Question Answering
description: >-
  [ACL 2026 (Findings)][Medical NLP][Electronic Health Record Question Answering (EHR-QA)] This paper proposes HypEHR, a Lorentz hyperbolic model with only 22M parameters. It embeds medical codes, visit records…
tags:
  - "ACL 2026 (Findings)"
  - "Medical NLP"
  - "Electronic Health Record Question Answering (EHR-QA)"
  - "Hyperbolic Space"
  - "Lorentz Model"
  - "ICD Hierarchical Modeling"
  - "Lightweight Clinical Models"
date: 2026-05-08
content_hash: 1e5683582e914112
---

# HypEHR: Hyperbolic Modeling of Electronic Health Records for Efficient Question Answering

**Conference**: ACL 2026 (Findings)  
**arXiv**: [2604.21027](https://arxiv.org/abs/2604.21027)  
**Code**: [https://github.com/yuyuliu11037/HypEHR](https://github.com/yuyuliu11037/HypEHR)  
**Area**: Medical NLP  
**Keywords**: Electronic Health Record Question Answering (EHR-QA), Hyperbolic Space, Lorentz Model, ICD Hierarchical Modeling, Lightweight Clinical Models

## TL;DR
This paper proposes HypEHR, a Lorentz hyperbolic model with only 22M parameters. It embeds medical codes, visit records, and questions into hyperbolic space and aligns them with the ICD ontology structure through hierarchy-aware regularization, approaching the performance of LLM-based methods on the MIMIC-IV EHR-QA task.

## Background & Motivation

**Background**: Electronic Health Record Question Answering (EHR-QA) aims to answer natural language clinical questions regarding longitudinal patient records. Current methods primarily fall into three categories: EHR representation learning (sequential/graph models), Text-to-SQL semantic parsing, and retrieval-augmented LLM pipelines based on GPT-3.5/4.

**Limitations of Prior Work**: While accurate, these methods incur high computational overhead, are difficult to deploy under strict privacy constraints, and mostly ignore the strong structural priors inherent in EHR data. LLM pipelines with trillions of parameters are challenging to deploy locally within hospitals.

**Key Challenge**: Medical codes and patient trajectories are inherently hierarchical (ICD codes are organized by chapter → block → category → subcategory). Euclidean space embeddings distort this tree-like structure, and existing methods fail to fully exploit this geometric prior.

**Goal**: Construct a compact model consistent with the intrinsic geometry of EHRs to achieve comparable performance to LLMs on complex QA tasks with significantly fewer parameters.

**Key Insight**: Hyperbolic space can embed hierarchical structures with arbitrarily low distortion. Previous research has demonstrated that hyperbolic embeddings can improve medical code hierarchy modeling, but final patient representations were still modeled in Euclidean space. The authors propose modeling patient-level representations directly in hyperbolic space.

**Core Idea**: Embed ICD codes, visits, and questions using the Lorentz hyperbolic manifold, align them with the ICD ontology via hierarchy-aware regularization, and answer questions using geometrically consistent cross-attention and type-specific pointer heads.

## Method

### Overall Architecture
HypEHR consists of two stages: (1) Patient encoder pre-training—learning joint next-visit prediction and hierarchy regularization objectives in hyperbolic space; (2) QA training—freezing the patient encoder and training answer-type-specific heads (Boolean, Concept, Numeric, and Count).

### Key Designs

1. **Hyperbolic Clinical Sequence Encoder**:
    - **Function**: Encodes a patient's visit sequence into contextualized representations in hyperbolic space.
    - **Mechanism**: Each medical code $c$ is embedded into the Lorentz hyperbolic manifold $\mathbb{H}_L^d$. Visit representations $h_t$ are obtained by aggregating code embeddings within the same visit via hyperbolic attention. A multi-layer Lorentz Transformer (adapting self-attention, residual connections, and normalization to the Lorentz manifold) processes the visit sequence to produce contextualized states $\{z_t\}_{t=1}^T$ and a global summary $z_{\text{[CLS]}}$.
    - **Design Motivation**: Operating entirely in hyperbolic space avoids information loss from repeated Euclidean → Hyperbolic → Euclidean conversions, allowing representations to naturally maintain hierarchical structures.

2. **Hierarchy-Aware Regularization**:
    - **Function**: Encodes the tree structure of the ICD code ontology into the embedding space.
    - **Mechanism**: Two constraints are constructed based on the ICD code tree (Chapter → Block → Category → Subcategory): Radial hierarchy loss $\mathcal{L}_{\text{rad}}$ requires the hyperbolic norm of a parent node embedding to be smaller than that of its children (deeper nodes are further from the origin); Relative hierarchy loss $\mathcal{L}_{\text{rel}}$ uses triplet constraints to keep codes with common ancestors closer than those without. The joint objective is $\mathcal{L} = \mathcal{L}_{\text{diag}} + \lambda \mathcal{L}_{\text{hier}}$.
    - **Design Motivation**: Next-visit prediction alone cannot explicitly capture hierarchical relationships between codes. Depth-radius monotonicity and sibling proximity constraints align the embedding space geometry with the ICD ontology.

3. **Hyperbolic EHR-QA Model**:
    - **Function**: Answers natural language clinical questions based on the frozen patient encoder.
    - **Mechanism**: Questions are encoded as Euclidean vectors by a biomedical pre-trained language model and projected into hyperbolic space as $z_q$ via affine and exponential mapping. Simple hyperbolic cross-attention (where attention scores are negative scaled hyperbolic distances $s_t = -\gamma d_{\mathbb{H}}(z_q, z_t)$) generates a question-conditioned patient summary $z_{p|q}^{\text{visit}}$. Second-order attention is applied to codes within the top-$k$ visits to obtain code-level reasoning vectors $z_{p|q}^{\text{code}}$. Specific heads are invoked based on the answer type.
    - **Design Motivation**: Hyperbolic distance naturally distinguishes concepts at different depths in a hierarchy, making it more suitable than Euclidean distance for attention calculations over complex ontological structures.

### Loss & Training
Pre-training utilizes a joint loss of next-visit diagnosis prediction (binary cross-entropy) and hierarchy regularization. In the QA phase, the language and patient encoders are frozen, and only type-specific classification heads are trained using standard cross-entropy loss. The total parameter count is only 22M.

## Key Experimental Results

### Main Results

| Model | EHRXQA (Acc%) | MIMIC-Instr (Acc%) |
|------|-------------|-------------------|
| RETAIN | 81.19 ± 1.95 | 65.91 ± 0.84 |
| NeuralSQL (GPT-5.2) | **95.97** ± 0.50 | 75.17 ± 0.73 |
| Llama-3 | 82.88 ± 1.38 | 70.90 ± 0.86 |
| Llemr | 87.25 ± 0.77 | **77.53** ± 0.54 |
| EHRAgent | 93.06 ± 1.09 | 74.16 ± 0.56 |
| **Ours (22M)** | 89.53 ± 0.60 | 76.02 ± 0.41 |

### Ablation Study

| Configuration | EHRXQA | MIMIC-Instr |
|------|--------|-------------|
| w/o $\mathcal{L}_{\text{hier}}$ | 82.72 ± 3.41 | 70.38 ± 0.54 |
| w/o pretraining | 74.05 ± 4.76 | 68.12 ± 1.39 |
| EucEHR (Euclidean) | 80.33 ± 1.14 | 69.88 ± 1.07 |
| **Ours** | **89.53** ± 0.60 | **76.02** ± 0.41 |

### Key Findings
- Patient encoder pre-training is the most critical component; removing it causes EHRXQA performance to drop by 15.5 percentage points.
- Under the same pre-training settings, the hyperbolic model (Ours) outperforms the Euclidean model (EucEHR) by 9.2 percentage points on EHRXQA and 6.1 percentage points on MIMIC-Instr.
- While hierarchy regularization is not the primary performance driver, it has a significant effect on explicitly modeling code hierarchical structures.
- Geometric analysis shows: The code radius in the Lorentz model increases monotonically with ICD tree depth, whereas the norm in Euclidean models has only a weak and noisy relationship with depth.

## Highlights & Insights
- Approaching the EHR-QA performance of trillion-parameter LLMs with just 22M parameters highlights the power of structural priors (hyperbolic geometry + ICD hierarchy). This strategy of "trading parameter count for the correct inductive bias" is particularly important in resource-constrained clinical settings.
- The design of the radial hierarchy loss is clever: it leverages the property that "space expands further from the origin" in hyperbolic space, allowing fine-grained diagnostic codes to naturally occupy larger representation volumes.
- The hyperbolic cross-attention mechanism can be transferred to any scenario requiring retrieval or attention over hierarchical structures, such as knowledge graph reasoning or taxonomy navigation.

## Limitations & Future Work
- Dependency on pre-processing to categorize answers into four fixed formats (Boolean/Concept/Numeric/Count) introduces extra engineering overhead and cannot handle open-ended generative answers.
- The computational complexity of Hyperbolic Neural Networks is higher than their Euclidean counterparts, and public hyperbolic geometry libraries are not yet mature, potentially leading to instability in large-scale training.
- Evaluations were conducted only on de-identified MIMIC-IV data and have not been validated in real-world clinical deployment environments.
- The authors emphasize that this system should be used as a decision support tool under human supervision and not as an autonomous clinical agent.

## Related Work & Insights
- **vs NeuralSQL (GPT-5.2)**: NeuralSQL uses GPT-5.2 to generate SQL queries, reaching 96% on EHRXQA, but depends on massive LLMs and aligns naturally with SQL paradigms. Ours reaches 89.5% with less than 1/1000 of the parameters in a non-SQL paradigm.
- **vs Llemr**: Llemr is the official LLM-based EHR-QA baseline, reaching 77.5% on MIMIC-Instr. Ours approaches this at 76% while significantly reducing parameters and inference costs.
- **vs Lu et al. (2023)**: This work also uses hyperbolic embeddings for medical code hierarchies, but final patient representations remain in Euclidean space. Ours extends hyperbolic modeling to the patient level, representing a key difference.

## Rating
- Novelty: ⭐⭐⭐⭐ Advancing hyperbolic geometry from the code level to patient-level QA is a natural extension; the hierarchy regularization design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two QA benchmarks, four clinical prediction tasks, detailed ablation, and geometric analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear, though technical details of hyperbolic space may be challenging for non-specialist readers.
- Value: ⭐⭐⭐⭐ Provides a lightweight alternative for privacy-sensitive clinical scenarios; the utilization of geometric priors is worth promoting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Query Pipeline Optimization for Cancer Patient Question Answering Systems](query_pipeline_optimization_for_cancer_patient_question_answering_systems.md)
- [\[ACL 2026\] Empathy Applicability Modeling for General Health Queries](empathy_applicability_modeling_for_general_health_queries.md)
- [\[AAAI 2026\] Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering](../../AAAI2026/medical_nlp/expert-guided_prompting_and_retrieval-augmented_generation_for_emergency_medical.md)
- [\[ACL 2026\] Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](efficient_and_effective_internal_memory_retrieval_for_llm-based_healthcare_predi.md)
- [\[ACL 2026\] Responsible Evaluation of AI for Mental Health](responsible_evaluation_of_ai_for_mental_health.md)

</div>

<!-- RELATED:END -->
