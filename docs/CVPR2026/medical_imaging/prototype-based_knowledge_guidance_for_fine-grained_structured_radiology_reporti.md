---
title: >-
  [Paper Note] Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting
description: >-
  [CVPR 2026][Medical Imaging][Structured radiology reporting] This paper proposes ProtoSR, which leverages LLMs to mine a template-aligned visual prototype knowledge base from large-scale free-text radiology reports, and injects it into a structured report generation model via prototype-conditioned residuals (late fusion). ProtoSR achieves state-of-the-art performance on the Rad-ReStruct benchmark, with particularly significant gains on fine-grained attribute questions.
tags:
  - CVPR 2026
  - Medical Imaging
  - Structured radiology reporting
  - prototype learning
  - knowledge distillation
  - VQA
  - chest X-ray
date: 2026-05-08
content_hash: 1166a0e677055c18
---

# Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting

**Conference**: CVPR 2026
**arXiv**: [2603.11938](https://arxiv.org/abs/2603.11938)
**Code**: To be released (promised upon acceptance)
**Area**: Medical Imaging
**Keywords**: Structured radiology reporting, prototype learning, knowledge distillation, VQA, chest X-ray

## TL;DR

This paper proposes ProtoSR, which leverages LLMs to mine a template-aligned visual prototype knowledge base from large-scale free-text radiology reports, and injects it into a structured report generation model via prototype-conditioned residuals (late fusion). ProtoSR achieves state-of-the-art performance on the Rad-ReStruct benchmark, with particularly significant gains on fine-grained attribute questions.

## Background & Motivation

Structured Reporting (SR) is more standardized and complete than free-text reporting, facilitating downstream quality control and analysis. However, automated SR faces core challenges:

- **Dense fine-grained decisions**: Templates contain hundreds of discrete fields, many corresponding to attributes of rare findings.
- **Scarce structured annotations**: Rad-ReStruct contains only 3,597 cases, with extremely sparse supervision for long-tail attributes.
- **Abundant but inconsistent free-text reports**: MIMIC-CXR contains 227K+ paired report-image instances, but stylistic variation prevents direct mapping to SR templates.

The core insight of ProtoSR is that instruction-tuned LLMs can "distill" large-scale free-text reports into template-aligned structured labels, constructing a multimodal prototype knowledge base that serves as an auxiliary knowledge source to improve long-tail fine-grained decisions in SR.

## Method

### Overall Architecture

ProtoSR consists of two branches (Fig. 1):
1. **Hierarchical SR baseline**: Image encoder + text encoder → fusion Transformer → classification head, outputting base logits $z_{\text{base}}$
2. **Prototype-conditioned knowledge branch**: Retrieves relevant prototypes from the knowledge base → computes support bias $b_{\text{sup}}$ → adds it to base logits via a learned scaling vector $s$

Final prediction: $z_{\text{final}} = z_{\text{base}} + s \odot b_{\text{sup}}$

### Key Designs

1. **LLM-driven knowledge base construction (Fig. 2)**: A three-step pipeline transforms free-text MIMIC-CXR reports into a template-aligned prototype library:

    - **Terminology expansion**: A zero-shot LLM generates synonym, abbreviation, and paraphrase dictionaries for each template label $\ell$, improving matching robustness.
    - **Template-constrained extraction**: Each report is queried hierarchically: the LLM first determines whether a finding is present, then extracts corresponding attribute values, using constrained decoding to ensure outputs conform to the template.
    - **Post-processing and assembly**: Rule-based filtering + hierarchical consistency constraints (parent labels removed if no child labels exist); up to $K=5$ images are sampled per label $\ell$ and aggregated into a single prototype vector via element-wise max pooling.

   Final coverage: L1 100%, L2 96%, L3 82%, providing sufficient prototype support for fine-grained attributes.

2. **Prototype-conditioned late fusion module**: Given the fused representation $S$ of the current image-question pair, $M$ prototype embeddings $P \in \mathbb{R}^{M \times d}$ and their corresponding answer vectors $A \in \mathbb{R}^{M \times |Y|}$ are retrieved from the knowledge base. Cosine similarity weights $\alpha$ are computed to aggregate visual evidence and answer tendencies:

$$v = \alpha^\top P \in \mathbb{R}^d, \quad u = \alpha^\top A \in \mathbb{R}^{|Y|}$$

The concatenation is passed through an MLP to produce the support bias $b_{\text{sup}} = \text{MLP}([v; u])$, which is modulated by a learned scaling vector $s$ and added to the base logits. If no compatible prototypes exist, $b_{\text{sup}} = \mathbf{0}$, ensuring safe fallback.

3. **EMA-aligned prototype updates**: Prototype embeddings are computed using an EMA copy of the image encoder and refreshed every 10K training steps, keeping prototypes aligned with the continuously fine-tuned encoder.

### Loss & Training

- **Loss function**: Same multi-label objective as Rad-ReStruct, applied to $z_{\text{final}}$
- **Training configuration**: Adam optimizer, lr=1e-5, batch size=8, gradient accumulation=4, 34 epochs
- **Backbone**: EfficientNet-B5 (image encoder) + RadBERT (text encoder)
- **Evaluation**: Iterative querying of the full template, with prior question-answer pairs appended to context
- **Hardware**: Single Nvidia RTX 3090 (24 GB)

## Key Experimental Results

### Main Results

| Method | Overall F1 | L1-F1 | L2-F1 | L3-F1 | Report Acc. |
|--------|-----------|-------|-------|-------|-------------|
| MedGemma | 26.8 | 38.2 | 63.4 | 2.8 | 0.0% |
| CheXagent | 32.4 | 62.1 | 69.8 | 6.2 | 20.3% |
| hi-VQA (Rad-ReStruct) | 32.0 | 64.6 | 71.6 | 4.1 | 32.6% |
| Context-VQA | 32.9 | 67.2 | 71.8 | 3.2 | 39.7% |
| **ProtoSR** | **34.4** | 66.2 | **72.8** | **7.4** | 36.6% |

### Ablation Study

| Configuration | Overall F1 | L3-F1 | Notes |
|---------------|-----------|-------|-------|
| No knowledge (baseline) | 32.5 | 4.3 | No knowledge injection |
| Early Fusion | 32.5 | 4.3 | Early fusion ineffective |
| Randomized prototypes | 32.7 | 4.4 | Random prototypes ineffective, confirming gains stem from prototype content |
| **ProtoSR (late fusion)** | **34.4** | **7.4** | Relative L3 improvement: +72.1% |

### Key Findings

- The largest gains from ProtoSR concentrate at L3 (fine-grained attributes), with a 72.1% relative improvement over the baseline—precisely the level with the sparsest supervision.
- General-purpose medical VLLMs (MedGemma, CheXagent) fail to surpass specialized SR models, indicating that hierarchical structured reporting requires task-specific architectures.
- Early fusion is ineffective while late fusion succeeds, suggesting prototype knowledge is better suited as a corrective signal for predictions rather than as input features.
- Terminology expansion is critical for LLM extraction quality: Qwen2.5-7B with expansion achieves 80.6 L3-F1, versus 68.1 without.
- Coverage of the knowledge base for long-tail attributes (L3: 82%) is foundational to the observed performance gains.

## Highlights & Insights

- **Elegant knowledge transfer**: Converts abundant but unstructured free-text reports into a structured prototype knowledge base, bridging the gap between "data-rich but format-mismatched" sources.
- **Lightweight design**: The knowledge branch adds only one MLP and a per-answer scaling vector, introducing negligible additional parameters.
- **Safe fallback**: Automatically reverts to the baseline when no matching prototypes are found, preserving existing performance.
- **LLM as a knowledge extraction tool**: Demonstrates the feasibility of using instruction-tuned LLMs for large-scale structured clinical text extraction.

## Limitations & Future Work

- Validated only on chest X-rays (Rad-ReStruct); other modalities such as CT and MRI are not explored.
- Knowledge base quality depends on LLM extraction; L3 coverage of only 82% may miss extremely rare attributes.
- Knowledge base refresh every 10K steps may leave prototypes misaligned with the encoder early in training.
- More sophisticated retrieval strategies (e.g., dynamic retrieval conditioned on question hierarchy) are not explored.
- Report Accuracy remains below Context-VQA, indicating room for improvement in overall report consistency.

## Related Work & Insights

- Rad-ReStruct defines the hierarchical SR paradigm (L1 coarse-grained → L3 fine-grained attributes); this paper adopts that framework and proposes a solution targeting L3 long-tail challenges.
- RadIR also mines fine-grained supervision signals from free-text reports, but only for retrieval rather than injection into the prediction pipeline.
- The approach is conceptually related to cross-modal prototype memory (Wang et al., ECCV 2022), with the key distinction of handling discrete SR decisions rather than free-text generation.
- Insight: Large-scale unannotated or weakly annotated data + LLM extraction = high-quality auxiliary knowledge source—a paradigm generalizable to other medical reporting scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of LLM-based mining and prototype late fusion is novel and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive ablations (fusion strategy / randomization / extraction LLM comparison) plus coverage analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with a coherent motivation–method–experiment logical chain.
- **Value**: ⭐⭐⭐⭐ Offers a broadly applicable solution to the long-tail problem in structured medical report generation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Momentum Memory for Knowledge Distillation in Computational Pathology](momentum_memory_for_knowledge_distillation_in_computational_pathology.md)
- [\[CVPR 2026\] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)
- [\[CVPR 2026\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)
- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)
- [\[CVPR 2026\] LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](lemon_large_endoscopic_monocular_dataset_foundation_model_surgical.md)

<!-- RELATED:END -->
