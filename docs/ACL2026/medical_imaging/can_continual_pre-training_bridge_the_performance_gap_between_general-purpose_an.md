---
title: >-
  [Paper Note] Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?
description: >-
  [ACL 2026][Medical Imaging][Continual pre-training] This paper constructs a high-quality German medical corpus, FineMed-de (7.3 million documents / 5.1 billion tokens filtered from FineWeb2)…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Continual pre-training"
  - "domain adaptation"
  - "German medical LLM"
  - "model merging"
  - "data filtering"
date: 2026-05-08
content_hash: c704b4eed89e50c5
---

# Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?

**Conference**: ACL 2026
**arXiv**: [2604.19394](https://arxiv.org/abs/2604.19394)  
**Code**: None  
**Area**: Medical Imaging
**Keywords**: Continual pre-training, domain adaptation, German medical LLM, model merging, data filtering

## TL;DR

This paper constructs a high-quality German medical corpus, FineMed-de (7.3 million documents / 5.1 billion tokens filtered from FineWeb2), applies continual pre-training and SLERP model merging to three LLMs (7B–24B), and creates the DeFineMed model family. The results demonstrate that a domain-specialized 7B model can substantially narrow the performance gap with a 24B general-purpose model on German medical tasks, improving the win rate by approximately 3.5×.

## Background & Motivation

**Background**: LLMs have shown transformative potential in healthcare, yet integrating them into clinical workflows remains challenging. General-purpose models typically fail to capture domain-specific knowledge and terminology with sufficient accuracy.

**Limitations of Prior Work**: (1) Strict data protection regulations mandate on-premise deployment, rendering large-scale API services infeasible and favoring smaller models; (2) smaller models lack domain-specific data coverage, making it difficult to handle complex medical terminology; (3) high-quality medical data in non-English languages—German in particular—is scarce.

**Key Challenge**: Regulatory constraints require the use of smaller models, yet smaller models need targeted domain knowledge to reach clinically acceptable performance—creating a critical compliance-versus-performance trade-off.

**Goal**: To achieve domain adaptation via continual pre-training and model merging, enabling a 7B model to compete with a 24B general-purpose model on complex medical tasks.

**Key Insight**: A complete methodology—spanning data filtering to model adaptation—is constructed by combining LLM-assisted annotation with classical ML classifiers to enable scalable data curation.

**Core Idea**: High-quality domain data + continual pre-training + model merging can make resource-efficient small models competitive solutions for complex medical tasks.

## Method

### Overall Architecture

The approach consists of two major components: (1) **a medical filtering pipeline**—Mixtral is used for zero-shot annotation of the German subset of FineWeb2, followed by training an XLM-RoBERTa classifier to scale annotation to the full dataset, yielding the FineMed-de corpus; (2) **model adaptation**—continual pre-training is applied to instruction-tuned models, which are then merged with the original instruction-tuned checkpoints via SLERP to restore instruction-following capability.

### Key Designs

1. **Hybrid Medical Document Filtering Pipeline**

    - **Function**: Efficiently extract high-quality medical documents from general-purpose web corpora.
    - **Mechanism**: (a) Sample 260,000 documents from the German subset of FineWeb2; (b) apply Mixtral-8x7B for zero-shot classification into medical/non-medical categories (human-validated F1 = 91.1%); (c) fine-tune an XLM-RoBERTa (279M) classifier on the annotated data (precision 0.95, recall 0.80); (d) apply the classifier to the full 428 million documents, extracting 7.3 million medical documents (5.1 billion tokens).
    - **Design Motivation**: LLMs provide high-quality annotations but are costly, while classical ML classifiers offer scalability—the hybrid approach balances quality and efficiency.

2. **Continual Pre-training + SLERP Model Merging**

    - **Function**: Inject domain knowledge while preserving instruction-following capability.
    - **Mechanism**: Instruction-tuned models are continually pre-trained on FineMed-de for 2 epochs (using FSDP, Flash Attention, and mixed precision), then merged with the original instruction-tuned checkpoints via layer-wise SLERP interpolation. Three base models are selected: Qwen2.5-7B, Mistral-7B, and Mistral-Small-24B.
    - **Design Motivation**: Continual pre-training may cause catastrophic forgetting and degrade instruction-following ability; model merging provides an efficient means to restore these capabilities without additional fine-tuning.

3. **Multi-dimensional Evaluation Design**

    - **Function**: Comprehensively assess the effects and trade-offs of domain adaptation.
    - **Mechanism**: (a) Knowledge-intensive benchmarks (MMLU-de medical subset + MedQA-de) evaluate medical knowledge; (b) pairwise win-rate analysis evaluates complex medical instruction following; (c) failure mode analysis (language mixing, verbosity) evaluates side effects.
    - **Design Motivation**: A single benchmark may obscure the true performance gap; multi-dimensional evaluation reveals the complete picture of domain adaptation.

### Loss & Training

Continual pre-training employs the standard language modeling objective (next-token prediction) with the AdamW optimizer, linear learning rate decay, and 500 warm-up steps. Training efficiency is optimized via FSDP, Flash Attention, activation checkpointing, and sequence packing.

## Key Experimental Results

### Main Results

**Average Accuracy on German Medical Benchmarks**

| Model | Average Accuracy |
|-------|-----------------|
| BioMistral-7B (baseline) | 43.55 |
| BioMistral-7B-SLERP | 48.22 |
| Mistral-7B-Instruct | 49.73 |
| DeFineMed-Mistral-7B-SLERP | **56.46** |
| Qwen2.5-7B-Instruct | 59.08 |
| DeFineMed-Qwen2.5-7B | **64.91** |

### Ablation Study

- The Qwen2.5-based DeFineMed 7B model achieves approximately a 3.5× improvement in pairwise win rate against Mistral-Small-24B-Instruct.
- SLERP model merging successfully restores instruction-following capability, but introduces side effects such as language mixing (German–English code-switching) and increased verbosity.
- Continual pre-training yields a larger gain for Qwen2.5-based models (+5.83) than for Mistral-based models (+6.73), though both improvements are substantial.

### Key Findings

- Continual pre-training combined with model merging enables 7B models to approach and compete with 24B models on German medical tasks.
- Data quality matters more than data scale—a carefully filtered corpus of 5.1 billion tokens is sufficient to yield significant improvements.
- Model merging is effective at restoring instruction-following capability, but inherent trade-offs such as language mixing remain.
- The choice of base model has a substantial impact on final performance (Qwen2.5 > Mistral).

## Highlights & Insights

- The hybrid data filtering pipeline (LLM annotation + ML classifier) is practical and transferable to other domains and languages.
- The finding that a 7B model can compete with a 24B model has significant practical implications for resource-constrained clinical settings.
- The failure mode analysis (language mixing, verbosity) provides an honest assessment of adaptation trade-offs.
- The methodology is directly generalizable to medical LLM development in other non-English languages.

## Limitations & Future Work

- The approach targets German only and has not been extended to other languages.
- Language mixing and verbosity issues require targeted subsequent fine-tuning to resolve.
- The optimal ordering of continual pre-training and instruction fine-tuning remains an open question.
- Model utility has not been validated in real clinical settings.

## Related Work & Insights

- Compared to BioMistral, this work focuses not only on benchmark score improvements but also on the competitiveness of small models against larger ones.
- Apollo-2 follows an instruction fine-tuning route, whereas this work adopts a continual pre-training route—the two approaches are complementary.
- The effectiveness of SLERP in the medical domain receives further empirical support.

## Rating

- **Novelty**: ⭐⭐⭐ — Individual methodological components are well-established, but their combined application to the German medical setting is valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluation is comprehensive across three dimensions: multiple benchmarks, win-rate analysis, and failure mode analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear and experimental design is sound.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study](../../CVPR2026/medical_imaging/are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)
- [\[ACL 2026\] Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation](inflated_excellence_or_true_performance_rethinking_medical_diagnostic_benchmarks.md)
- [\[ACL 2026\] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)
- [\[AAAI 2026\] MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging](../../AAAI2026/medical_imaging/mirnet_integrating_constrained_graph-based_reasoning_with_pre-training_for_diagn.md)
- [\[ICCV 2025\] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](../../ICCV2025/medical_imaging/boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)

</div>

<!-- RELATED:END -->
