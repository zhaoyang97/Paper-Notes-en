---
title: >-
  [Paper Note] A General Knowledge Injection Framework for ICD Coding
description: >-
  [ACL 2025][Knowledge Editing][ICD Coding] > This paper proposes GKI-ICD, a general knowledge injection framework. By employing guideline synthesis and multi-task learning mechanisms, it simultaneously integrates three types of ICD knowledge—Description, Synonym, and Hierarchy—without requiring extra network modules, achieving SOTA performance on the MIMIC-III benchmark.
tags:
  - "ACL 2025"
  - "Knowledge Editing"
  - "ICD Coding"
  - "Knowledge Injection"
  - "Multi-task Learning"
  - "Guideline Synthesis"
  - "Clinical Text Classification"
date: 2026-05-08
content_hash: b42d59b28c4768d7
---

# A General Knowledge Injection Framework for ICD Coding

**Conference**: ACL 2025  
**arXiv**: [2505.18708](https://arxiv.org/abs/2505.18708)  
**Code**: [GitHub](https://github.com/xuzhang0112/GKI-ICD)  
**Area**: Knowledge Editing  
**Keywords**: ICD Coding, Knowledge Injection, Multi-task Learning, Guideline Synthesis, Clinical Text Classification  

## TL;DR

> This paper proposes GKI-ICD, a general knowledge injection framework. By employing guideline synthesis and multi-task learning mechanisms, it simultaneously integrates three types of ICD knowledge—Description, Synonym, and Hierarchy—without requiring extra network modules, achieving SOTA performance on the MIMIC-III benchmark.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: Research problem: The ICD coding task requires assigning a large number of medical codes to clinical texts, which faces two major challenges: the long-tail distribution of codes and the lack of code-level evidence annotation.

### Background

**Background**: Single knowledge type: Existing methods typically focus on only one type of knowledge (description, synonym, or hierarchical relation) and design specialized modules for it.

### Key Challenge

**Key Challenge**: Incompatible modules: The multi-synonym attention mechanism designed for synonyms and the graph neural network designed for hierarchical relations are difficult to integrate into a unified model.

### Key Insight

**Proposed Solution**: Poor scalability: The complexity of specialized modules makes them difficult to extend to more advanced models.

**Core Motivation:** To design a general framework that injects three complementary types of ICD code data in a unified manner, without relying on specialized network modules.

## Method

### Overall Architecture

GKI-ICD consists of two core components:
1. **Guideline Synthesis:** Synthesizes training guidelines using code knowledge, replacing specialized network modules.
2. **Multi-task Learning:** The model simultaneously learns from both raw samples and synthesized guidelines, aligning them via semantic consistency constraints.

### Key Designs

**1. Guideline Synthesis:** Given a clinical document and its annotated set of ICD codes, the following steps are performed:
- **Description Parsing:** Extract the official ICD-9 description for each positive code, removing non-standard terms such as "NOS".
- **Synonym Substitution:** Extract synonyms for each code from the UMLS knowledge base and randomly replace portions of the descriptions to enhance diversity.
- **Hierarchical Retrieval:** Add hierarchical descriptions of the groups to which the codes belong (e.g., 038.9 $\rightarrow$ 030-041 $\rightarrow$ 001-139).
- **Shuffling & Concatenation:** Periodically shuffle the order of the codes and concatenate them into a long text sequence to serve as the synthesized guideline $\hat{x}$.

**2. Multi-task Learning Mechanism:**
- **Raw Text Prediction:** Standard ICD coding loss $L_{raw} = L_{BCE}(f(x), y)$
- **Guideline Prediction:** Requiring the model to correctly predict codes from synthesized guidelines as well: $L_{guide} = L_{BCE}(f(\hat{x}), y)$
- **Semantic Consistency Constraint:** Aligning the code-specific representations extracted from the raw text and the guidelines: $L_{sim} = 1 - cosine(E, \hat{E})$

### Loss & Training

$$L = L_{raw}(x, y) + L_{guide}(\hat{x}, y) + \lambda L_{sim}(E, \hat{E})$$

where $\lambda$ controls the weight of semantic consistency, balancing the gap between theoretical knowledge and clinical expressions.

## Key Experimental Results

### Main Results

| Model | MIMIC-III-Full MacroAUC | MicroAUC | MacroF1 | MicroF1 | P@8 |
|------|---|---|---|---|---|
| CAML | 0.895 | 0.986 | 0.088 | 0.539 | 0.709 |
| PLM-CA | 0.916 | 0.989 | 0.103 | 0.599 | 0.772 |
| MSMN | 0.950 | 0.992 | 0.103 | 0.584 | 0.752 |
| CoRelation | 0.952 | 0.992 | 0.102 | 0.591 | 0.762 |
| **GKI-ICD** | **0.962** | **0.993** | **0.123** | **0.612** | **0.777** |

GKI-ICD achieves SOTA performance across all metrics on MIMIC-III-Full, with a MacroAUC gain of 4.6% and a MacroF1 gain of 19.4% compared to the baseline PLM-CA.

### Ablation Study

| Knowledge Combination | Effectiveness |
|------|------|
| No Knowledge (baseline PLM-CA) | MacroAUC 0.916, MacroF1 0.103 |
| + Description | Gain |
| + Description + Synonym | Further Gain |
| + Description + Synonym + Hierarchy | Optimal (MacroAUC 0.962, MacroF1 0.123) |

The gradual contributions of the three types of knowledge validate the necessity and complementarity of multi-knowledge integration.

### Key Findings

1. **Effectiveness of General Framework:** Multiple types of knowledge can be injected without specialized modules, and the performance surpasses methods that utilize specialized modules.
2. **Strong Knowledge Complementarity:** The three types of knowledge (descriptions, synonyms, and hierarchies) provide progressive incremental gains.
3. **Zero Inference Overhead:** Knowledge is injected solely via synthesized guidelines during the training phase; guidelines are not used during inference, resulting in no increased computational overhead.
4. **Outperforming Methods with Additional Labels:** GKI-ICD remains highly competitive even when compared to methods that use additional manual annotations (such as DRG/CPT).
5. **Significant Long-tail Improvement:** The substantial increase in MacroF1 demonstrates a significantly improved capability to handle low-frequency codes.

## Highlights & Insights

- Integrates three complementary types of ICD code data in a unified manner for the first time without introducing extra network modules.
- The guideline synthesis method elegantly incorporates discrete knowledge into continuous text sequences, leveraging the semantic understanding capabilities of language models.
- Injecting knowledge during training with zero overhead during inference represents a highly practical design philosophy.
- Achieves comprehensive SOTA performance on the largest publicly available clinical dataset, MIMIC-III.

## Limitations & Future Work

- Only validated on the ICD-9 coding system, with future verification needed on updated versions like ICD-10.
- Guideline synthesis utilizes ground-truth labels, limiting direct generalization to semi-supervised scenarios.
- The randomness in synonym substitution might introduce training noise.
- Experiments were restricted to RoBERTa-PM as the encoder, leaving the effects on other pre-trained models unverified.

## Related Work & Insights

- **ICD Coding Networks:** CAML (Mullenbach et al., 2018) pioneered the label attention mechanism; PLM-ICD/PLM-CA (Edin et al., 2022/2024) introduced pre-trained language models.
- **Knowledge Injection - Description:** ISD (Zhou et al., 2021) combined code descriptions using self-distillation; KEPTLongformer (Yang et al., 2022) treated descriptions as prompts.
- **Knowledge Injection - Synonym:** MSMN (Yuan et al., 2022) utilized multi-synonym attention to learn diverse code representations.
- **Knowledge Injection - Hierarchy:** MSATT-KG (Xie et al., 2019) employed graph convolutional networks to capture hierarchical relations between codes.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Overall Rating | 8.0/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning](memorizing_is_not_enough_deep_knowledge_injection_through_reasoning.md)
- [\[ACL 2025\] Structure-aware Domain Knowledge Injection for Large Language Models](structure-aware_domain_knowledge_injection_for_large_language_models.md)
- [\[ACL 2025\] ToxEdit: Adaptive Detoxification Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing](adaptive_detoxification_safeguarding_general_capabilities_of_llms_through_toxici.md)
- [\[NeurIPS 2025\] KScope: A Framework for Characterizing the Knowledge Status of Language Models](../../NeurIPS2025/knowledge_editing/kscope_a_framework_for_characterizing_the_knowledge_status_of_language_models.md)
- [\[ICML 2026\] KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls](../../ICML2026/knowledge_editing/kore_enhancing_knowledge_injection_for_large_multimodal_models_via_knowledge-ori.md)

</div>

<!-- RELATED:END -->
