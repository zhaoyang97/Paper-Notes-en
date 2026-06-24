---
title: >-
  [Paper Note] RAD: Towards Trustworthy Retrieval-Augmented Multi-modal Clinical Diagnosis
description: >-
  [NeurIPS 2025][Graph Learning][Retrieval-augmented diagnosis] This paper proposes RAD, a retrieval-augmented diagnostic framework that retrieves disease guidelines from multi-source medical corpora and injects them throughout the full pipeline of a multimodal model — from feature extraction to cross-modal fusion. A dual-axis explainability evaluation protocol is also introduced. RAD achieves state-of-the-art performance on four datasets spanning distinct anatomical regions.
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Retrieval-augmented diagnosis"
  - "multimodal fusion"
  - "knowledge injection"
  - "explainability"
  - "clinical guidelines"
date: 2026-05-08
content_hash: 0cb7d4aac761c560
---

# RAD: Towards Trustworthy Retrieval-Augmented Multi-modal Clinical Diagnosis

**Conference**: NeurIPS 2025  
**arXiv**: [2509.19980](https://arxiv.org/abs/2509.19980)  
**Code**: Unavailable  
**Area**: Medical Imaging  
**Keywords**: Retrieval-augmented diagnosis, multimodal fusion, knowledge injection, explainability, clinical guidelines

## TL;DR

This paper proposes RAD, a retrieval-augmented diagnostic framework that retrieves disease guidelines from multi-source medical corpora and injects them throughout the full pipeline of a multimodal model — from feature extraction to cross-modal fusion. A dual-axis explainability evaluation protocol is also introduced. RAD achieves state-of-the-art performance on four datasets spanning distinct anatomical regions.

## Background & Motivation

Current AI-driven medical research typically encodes medical knowledge into model parameters via knowledge graphs or large-scale text pretraining. However, these approaches share a fundamental limitation: **knowledge is encoded implicitly, lacking explicit injection of fine-grained, task-specific knowledge required for downstream diagnostic tasks.**

Specific limitations include:

**Limitations of pretraining-stage injection**: Models such as PubMedBERT and KAD inject knowledge during pretraining, but this knowledge is "general-purpose" and cannot flexibly adapt to specific diagnostic tasks. For example, KAD performs well on chest X-ray pretraining data but degrades when applied to ophthalmology or dermatology.

**Opacity of black-box decision-making**: Clinical diagnosis must adhere to evidence-based principles and rely on standardized diagnostic criteria. The opaque decision mechanisms of black-box neural networks hinder deployment in clinical settings.

**Lack of quantitative explainability evaluation**: Existing multimodal diagnostic models lack quantitative means to assess explainability.

The core insight of this paper is: **effective knowledge integration must be task-centric, aligning with disease-level knowledge throughout the entire diagnostic pipeline (input augmentation → feature extraction → modality fusion → decision-making), rather than at any single stage alone.**

## Method

### Overall Architecture

RAD comprises three synergistic components: (1) multi-source guideline retrieval and refinement; (2) guideline-enhanced feature constraints; and (3) a dual-decoding diagnostic network. These components systematically inject external disease knowledge at the input, feature, and decision levels.

### Key Designs

1. **Guideline Retrieval & Refinement**

   Knowledge is retrieved from four source types: "Wiki," "Research" (PubMed), "Guideline" (45K clinical practice guidelines), and "Book" (medical textbooks).

   For $m$ diseases in the dataset, MedCPT (a dual-encoder retrieval model) is used to compute the similarity between disease names and the corpus:
   $\mathcal{C}_i = \underset{p_j \in P}{\text{Top-}k} \text{Sim}(e_i, p_j)$

   Retrieved documents may be redundant or noisy; Qwen2.5-72B is employed for automatic summarization and refinement:
   $g_i = \text{LLM}([\text{Prompt}, c_{i,1}, \cdots, c_{i,k}])$

   This produces standardized, structured diagnostic guidelines containing associated symptoms, imaging features, and key examination items, subsequently verified by human experts.

2. **Guideline-Enhanced Contrastive Loss (GECL)**

   Visual encoder output $\mathbf{V}_i$ and text encoder output $\mathbf{T}_i$ are aligned with disease guideline prototypes $\mathbf{G}'$ in the latent space.

   For sample $i$, guideline features are partitioned into a positive set $\mathbf{P}_i$ (guidelines corresponding to positive labels) and a negative set $\mathbf{N}_i$; a subset $\mathbf{Q}_i$ is obtained via negative sampling. The resulting GECL loss is:
   $\mathcal{L}_{\text{GECL}} = \frac{1}{N}\sum_{i=1}^{N} \left(\mathcal{L}_{\text{SupCon}}(\mathbf{T}'_i, \mathbf{S}_i) + \alpha \mathcal{L}_{\text{SupCon}}(\mathbf{V}'_i, \mathbf{S}_i)\right) \cdot \mathbb{I}[|\mathbf{P}_i|>0]$

   **Design Motivation**: Sample features are dynamically pulled toward positive-class guideline prototypes and pushed away from negative-class prototypes, guiding the model to selectively attend to clinically relevant features aligned with guidelines, thereby enhancing both performance and explainability.

3. **Dual Diagnostic Network**

   Two symmetric Transformer decoders operate in parallel:
    - **Guideline branch**: Uses the encoding of guideline $g$ as query and concatenated modality features $\mathbf{V}_i \oplus \mathbf{T}_i$ as key/value, outputting $\hat{y}_i^{\text{guide}}$
    - **Label branch**: Uses the encoding of disease name $E$ as query and concatenated modality features as key/value, outputting $\hat{y}_i^{\text{label}}$

   Total training loss:
   $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{y}^{\text{guide}}, y) + \mathcal{L}_{\text{BCE}}(\hat{y}^{\text{label}}, y) + \beta \mathcal{L}_{\text{GECL}}$

### Explainability Evaluation Protocol

- **Textual metric — Guideline Recall**: Measures the attention recall rate of the model with respect to key laboratory indicators mentioned in the guidelines.
- **Visual metric — Visual Grounding IoU**: Measures the IoU overlap between attention maps and expert-annotated lesion regions.

## Key Experimental Results

### Main Results

| Dataset | Method | F1 | AUC | mAP | Avg |
|--------|--------|----|-----|-----|-----|
| MIMIC-ICD53 (Chest) | RAD | **39.71** | **93.00** | **36.74** | **57.28** |
| | KAD | 36.32 | 91.95 | 33.54 | 54.19 |
| | HEALNet | 35.42 | 88.80 | 31.97 | 53.13 |
| FairVLMed (Eye) | RAD | **84.30** | **91.32** | **91.88** | **86.63** |
| | HEALNet | 81.80 | 89.60 | 90.45 | 84.39 |
| SkinCAP (Skin) | RAD | **85.48** | **97.97** | **83.55** | **88.64** |
| | KAD | 82.06 | 97.80 | 80.40 | 86.15 |
| NACC (Brain) | RAD | **37.65** | **87.11** | **30.03** | **58.12** |
| | HEALNet | 35.91 | 85.04 | 26.13 | 55.67 |

### Ablation Study

| Configuration | F1 | AUC | Avg | Note |
|--------------|-----|-----|-----|------|
| w/o GECL + w/o Dual Decoder | 34.91 | 91.27 | 53.35 | Baseline |
| +GECL_vision | 37.43 | 92.53 | 54.79 | Visual alignment effective |
| +GECL_text | 37.75 | 92.91 | 55.52 | Text alignment stronger |
| +GECL_both | 39.34 | 92.94 | 56.26 | Dual-modal alignment additive |
| +Dual Decoder only | 39.22 | 92.25 | 55.91 | Decoder contributes substantially |
| **RAD (Full)** | **39.71** | **93.00** | **57.28** | Components are complementary |

### Explainability Evaluation

| Metric | w/o RAD | RAD | Gain |
|--------|---------|-----|------|
| Guideline Recall (Overall) | 24.76% | **65.62%** | +40.86% |
| Visual Grounding mIoU (Avg-D) | 15.98 | **19.72** | +3.74 |
| Visual Grounding mIoU (Avg-P) | 17.78 | **22.04** | +4.26 |

### Key Findings

1. RAD consistently outperforms all baselines across four datasets spanning distinct anatomical regions, with average improvements of 2.24%–3.09%, demonstrating **strong generalization across anatomical domains**.
2. KAD performs well on chest X-ray data but degrades on other anatomical regions, confirming the limitations of pretraining-stage knowledge injection.
3. Guideline Recall improves from 24.76% to 65.62%, **quantitatively demonstrating that knowledge injection effectively guides the model to attend to guideline-recommended key indicators**.
4. The text-branch GECL contributes more than the visual branch (intra-modal alignment is easier); removing the Dual Decoder results in the largest performance degradation.

## Highlights & Insights

- The "full-pipeline knowledge injection" design philosophy is conceptually clear: alignment is enforced across input (guideline retrieval) → features (GECL constraints) → decision (dual decoder).
- The dual-axis explainability evaluation framework (textual Recall + visual IoU) provides a valuable methodological contribution for quantifying explainability in multimodal diagnostic models.
- A new MIMIC-ICD53 dataset is constructed by aligning MIMIC-CXR and MIMIC-IV, covering 3 modalities and 53 diseases.
- Offline retrieval combined with LLM-based refinement for guideline acquisition is more stable than online RAG and well-suited for discriminative tasks.

## Limitations & Future Work

- Guidelines require human validation, making extension to a larger disease repertoire costly.
- Validation is limited to discriminative tasks; generative tasks such as report generation remain unexplored.
- The Dual Decoder introduces additional inference overhead; the degree of redundancy between the two decoders warrants further analysis.
- Visual grounding evaluation relies on annotations from an external dataset (ChestX-Det), limiting coverage to a restricted set of disease categories.

## Related Work & Insights

- The distinction from standard RAG methods is noteworthy: RAD performs offline retrieval with structured injection for discriminative tasks, whereas standard RAG performs online retrieval-augmented generation.
- The design principles underlying GECL are generalizable to other scenarios requiring domain knowledge injection into feature spaces.
- The dual-axis explainability evaluation protocol has potential as a standardized assessment framework for broader adoption.

## Rating

- Novelty: ⭐⭐⭐⭐ — Systematic full-pipeline knowledge injection framework; explainability evaluation protocol is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four anatomical datasets, comprehensive ablations, quantitative explainability evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clear, mathematical derivations are complete, experimental analysis is thorough.
- Value: ⭐⭐⭐⭐ — Meaningful advancement for knowledge injection and explainability in multimodal medical diagnosis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [\[ACL 2026\] LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning](../../ACL2026/graph_learning/legalgraphrag_multi-agent_graph_retrieval-augmented_generation_for_reliable_lega.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](../../CVPR2026/graph_learning/m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[ACL 2025\] Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation (K-RagRec)](../../ACL2025/graph_learning/kg_rag_recommendation.md)

</div>

<!-- RELATED:END -->
