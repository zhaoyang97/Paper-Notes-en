---
title: >-
  [Paper Note] Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting
description: >-
  [CVPR 2026][Medical Imaging][Structured radiology reporting] This paper proposes ProtoSR, which employs an LLM-driven pipeline to mine template-aligned visual prototype knowledge bases from 227…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Structured radiology reporting"
  - "prototype knowledge base"
  - "LLM knowledge extraction"
  - "long-tail attributes"
  - "late fusion"
date: 2026-05-08
content_hash: 3914bd9ca177f946
---

# Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting

**Conference**: CVPR 2026
**arXiv**: [2603.11938](https://arxiv.org/abs/2603.11938)
**Code**: Unavailable (authors state code will be released upon acceptance)
**Area**: Medical Imaging / Structured Report Generation
**Keywords**: Structured radiology reporting, prototype knowledge base, LLM knowledge extraction, long-tail attributes, late fusion

## TL;DR

This paper proposes ProtoSR, which employs an LLM-driven pipeline to mine template-aligned visual prototype knowledge bases from 227,000 free-text MIMIC-CXR reports, and introduces a prototype-conditioned late-fusion module that injects retrieved prototype evidence as logit residuals into a hierarchical structured reporting model. ProtoSR achieves state-of-the-art performance on the Rad-ReStruct benchmark, improving L3 fine-grained attribute F1 from 4.3 to 7.4 (+72.1% relative gain).

## Background & Motivation

**Background**: Radiology reports are the primary means of clinical communication in diagnostic imaging. While free-text reports offer flexibility, they suffer from stylistic inconsistency, incomplete coverage, and poor standardizability. Structured reporting (SR) addresses these issues through predefined fields and standardized options, yet its automation lags far behind free-text generation.

**Limitations of Prior Work**: Fine-grained SR templates contain numerous rare attributes (e.g., lesion location, appearance, severity), while structured annotation datasets are extremely limited. Rad-ReStruct contains only 3,597 samples, and L3-level long-tail attributes (477 questions) have almost no sufficient supervision. General-purpose medical VLMs (MedGemma, CheXagent), though capable of diverse tasks, still underperform specialized models on fine-grained SR.

**Key Challenge**: Structured annotations are scarce, yet free-text reports are abundant. MIMIC-CXR contains over 220,000 chest X-ray–free-text report pairs implicitly encoding rich fine-grained imaging information, but stylistic and lexical differences make direct mapping to a strict SR taxonomy difficult.

**Goal**: How can the implicit knowledge embedded in large-scale free-text reports be systematically converted into template-aligned structured signals to enhance fine-grained attribute prediction under data scarcity?

**Key Insight**: Instruction-tuned LLMs now enable large-scale automated extraction. By automatically mapping finding-attribute information from free text to SR template label spaces, visual prototypes (representing typical imaging characteristics for each label) can be constructed, and prototype retrieval during inference provides a "data-driven second opinion."

**Core Idea**: Use LLMs to mine a template-aligned visual prototype knowledge base from free-text reports, then augment fine-grained structured reporting predictions via prototype-conditioned logit residual correction.

## Method

### Overall Architecture

ProtoSR = hierarchical SR backbone (Rad-ReStruct architecture) + prototype knowledge base (mined from MIMIC-CXR) + prototype-conditioned knowledge branch. The backbone receives image and question context, extracts features via EfficientNet-B5 and RadBERT, fuses them through a Transformer, and predicts base logits. The knowledge branch retrieves relevant prototypes from the knowledge base, converts them into logit correction signals, and adds them to base logits via a learnable scaling vector to produce final predictions.

### Key Designs

1. **LLM-Driven Knowledge Base Construction Pipeline**:

    - Function: Map MIMIC-CXR free-text reports to the Rad-ReStruct label space and construct visual prototypes for each label.
    - Mechanism: A three-step pipeline — (1) *Term expansion*: a zero-shot LLM generates synonyms, abbreviations, and alternative expressions for each template label, constructing a normalization dictionary to handle free-text lexical diversity; (2) *Template-constrained extraction*: the LLM is queried hierarchically — first to determine whether a finding is present, then to extract attribute values if so — with constrained decoding ensuring outputs are restricted to valid template options; (3) *Post-processing*: rule-based denoising, enforcement of hierarchical consistency (child labels removed if parent is non-positive), uniform sampling of $K=5$ images per label, and element-wise max pooling of image encoder embeddings to aggregate prototype vectors.
    - Design Motivation: Term expansion substantially improves extraction quality (F1 gains of 8–13 points per level); Qwen2.5-7B performs best on this task. Final coverage: L1 100%, L2 96%, L3 82%, providing meaningful prototype support for long-tail attributes.

2. **Prototype-Conditioned Knowledge Branch (Late-Fusion Residual Correction)**:

    - Function: Convert prototype retrieval evidence into selective logit corrections without disrupting the backbone's decision pathway.
    - Mechanism: Given the fused image-question representation $S$, a linear projection maps it to the prototype space, and cosine similarity weights $\alpha$ are computed (considering only prototypes matching valid options for the current question). Retrieved results are aggregated into a visual evidence vector $v = \alpha^\top P$ and an answer-tendency vector $u = \alpha^\top A$; the concatenation $[v;u]$ is passed through an MLP to obtain a support bias $b_{\text{sup}}$. The final prediction is $z_{\text{final}} = z_{\text{base}} + s \odot b_{\text{sup}}$, where $s$ is a learnable per-answer scaling vector. When no matching prototype exists, $b_{\text{sup}}=0$ and the model degrades to the backbone.
    - Design Motivation: The residual design preserves the backbone's overall behavior, applying selective corrections only when prototype evidence is informative — effectively a "data-driven second opinion." The per-answer scaling vector $s$ calibrates prototype influence dimension-wise.

3. **Dynamic Prototype Alignment (EMA Refresh)**:

    - Function: Address the drift between prototype embeddings and the image encoder caused by continuous fine-tuning during training.
    - Mechanism: An EMA copy of the image encoder is maintained, and prototype vectors in the knowledge base are refreshed every 10k training steps.
    - Design Motivation: Maintaining alignment between prototypes and the current encoder's representation space; without this, retrieval similarity computation gradually degrades.

### Loss & Training

The same multi-label loss as Rad-ReStruct is applied to $z_{\text{final}}$. Adam optimizer, lr=1e-5, batch size=8, gradient accumulation=4, trained for 34 epochs on a single RTX 3090.

## Key Experimental Results

### Main Results

| Method | Overall F1 | L1-F1 | L2-F1 | L3-F1 | Report Acc. |
|--------|-----------|-------|-------|-------|-------------|
| MedGemma | 26.8 | 38.2 | 63.4 | 2.8 | 0.0% |
| CheXagent | 32.4 | 62.1 | 69.8 | 6.2 | 20.3% |
| hi-VQA (Rad-ReStruct) | 32.0 | 64.6 | 71.6 | 4.1 | 32.6% |
| Context-VQA | 32.9 | 67.2 | 71.8 | 3.2 | 39.7% |
| **ProtoSR** | **34.4** | **66.2** | **72.8** | **7.4** | 36.6% |

### Ablation Study

| Configuration | Overall F1 | L1-F1 | L2-F1 | L3-F1 |
|---------------|-----------|-------|-------|-------|
| No knowledge (backbone only) | 32.5 | 64.2 | 71.3 | 4.3 |
| Early Fusion (knowledge embeddings concatenated to input) | 32.5 | 64.8 | 71.4 | 4.3 |
| Randomized prototypes (replaced with Gaussian noise) | 32.7 | 64.3 | 71.4 | 4.4 |
| **ProtoSR (late-fusion residual)** | **34.4** | **66.2** | **72.8** | **7.4** |

### Key Findings

- **Largest gains at L3**: F1 improves from 4.3 → 7.4 (+72.1% relative gain), precisely the level where long-tail attributes receive the least supervision, demonstrating that the prototype knowledge base compensates for data scarcity.
- **Early Fusion is ineffective**: Directly concatenating knowledge embeddings into the input sequence yields no improvement over the backbone, confirming that the late-fusion residual design is critical.
- **Randomized prototypes reduce to baseline**: Replacing prototypes with Gaussian noise returns performance to baseline levels, confirming that ProtoSR leverages meaningful prototype structure rather than simply increasing model capacity.
- General-purpose medical VLMs (MedGemma, CheXagent) perform extremely poorly at L3 (2.8–6.2), underperforming specialized SR models.

## Highlights & Insights

- **LLM as a knowledge bridge**: The paper cleverly leverages instruction-tuned LLMs to "translate" unstructured free-text reports into structured labels. This paradigm is generalizable to any medical AI scenario requiring structured knowledge extraction from unstructured data.
- **Residual second opinion**: The knowledge branch design is elegant — it does not override the backbone's judgment but provides corrections only when evidence supports them. This "conservative augmentation" strategy is applicable to any scenario requiring additional knowledge injection without disrupting existing model behavior.
- **Critical role of term expansion**: F1 gains of 8–13 points demonstrate that cross-dataset clinical terminology differences are the primary bottleneck in knowledge transfer; LLM-driven terminology normalization is the key to unlocking this knowledge.

## Limitations & Future Work

- **Evaluation on a single benchmark**: Rad-ReStruct contains only 3,597 samples and covers only chest X-rays; generalizability remains unverified.
- **Report Acc. is not highest**: Context-VQA and RaDialog achieve report accuracy of 39.7% and 39.6% respectively, exceeding ProtoSR's 36.6% — though the authors note these methods favor conservative strategies (defaulting to "no finding") and perform poorly on abnormal cases.
- **Noise in knowledge base mining**: L3 coverage is only 82%, leaving approximately 18% of fine-grained labels without prototype support. Further optimization of the LLM extraction pipeline or incorporation of additional data sources could improve coverage.
- **Simple prototype aggregation**: The current approach aggregates $K=5$ images into a single prototype via max pooling; more sophisticated aggregation strategies (e.g., clustering multiple prototypes to represent intra-class diversity) may yield further improvements.

## Related Work & Insights

- **vs. Context-VQA**: Context-VQA leverages report context during training but uses no external knowledge at inference; ProtoSR provides additional information at inference time via the prototype knowledge base.
- **vs. RadIR**: RadIR mines fine-grained supervision from free-text reports for retrieval but does not address structured prediction; ProtoSR directly injects retrieval evidence into the discrete decision-making process.
- **vs. General-purpose medical VLMs (MedGemma / CheXagent)**: The extremely poor L3 performance of these models demonstrates that general-purpose capability cannot substitute for specialized architectures and training objectives designed for fine-grained template-based prediction.

## Rating

- Novelty: ⭐⭐⭐⭐ The complete LLM–knowledge-base–prototype pipeline is cleverly designed; the late-fusion residual correction is concise and effective.
- Experimental Thoroughness: ⭐⭐⭐ Limited to a single benchmark dataset, though ablation design is rigorous.
- Writing Quality: ⭐⭐⭐⭐ Logically clear with intuitive figures and tables.
- Value: ⭐⭐⭐⭐ Provides a reproducible solution for knowledge augmentation of long-tail medical attributes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)
- [\[CVPR 2026\] MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)
- [\[CVPR 2026\] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)
- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)
- [\[CVPR 2026\] Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)

</div>

<!-- RELATED:END -->
