---
title: >-
  [Paper Note] Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?
description: >-
  [ACL 2026][Medical Imaging][Continual Pre-training] This paper constructs FineMed-de, a high-quality German medical corpus (7.3 million documents / 5.1 billion tokens filtered from FineWeb2)…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Continual Pre-training"
  - "Domain Adaptation"
  - "German Medical LLM"
  - "Model Merging"
  - "Data Filtering"
date: 2026-05-08
content_hash: 78ba0d2aefc55439
---

# Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?

**Conference**: ACL 2026  
**arXiv**: [2604.19394](https://arxiv.org/abs/2604.19394)  
**Code**: None  
**Area**: Medical Imaging  
**Keywords**: Continual Pre-training, Domain Adaptation, German Medical LLM, Model Merging, Data Filtering

## TL;DR

This paper constructs FineMed-de, a high-quality German medical corpus (7.3 million documents / 5.1 billion tokens filtered from FineWeb2), and performs continual pre-training and SLERP model merging on three LLMs (7B-24B) to create the DeFineMed model family. It demonstrates that domain-specialized 7B models can significantly bridge the performance gap with 24B general-purpose models on German medical tasks (with approximately a 3.5x increase in win rate).

## Background & Motivation

**Background**: LLMs have demonstrated transformative potential in the medical field, yet integrating them into clinical workflows faces challenges. General-purpose models often fail to capture domain-specific knowledge and terminology with sufficient accuracy.

**Limitations of Prior Work**: (1) Strict data protection regulations necessitate local deployment, making large-scale API services unfeasible and favoring smaller models; (2) Small models lack domain-specific data support, making it difficult to handle complex medical terminology; (3) High-quality medical data in non-English languages (especially German) is scarce.

**Key Challenge**: Regulatory constraints require the use of small models, yet these models need targeted domain knowledge to reach clinically usable performance levels—creating a critical tradeoff between compliance and performance.

**Goal**: Achieve domain adaptation through continual pre-training and model merging, enabling 7B models to compete with 24B general-purpose models on complex medical tasks.

**Key Insight**: Establish a complete methodology from data filtering to model adaptation, combining LLM-assisted annotation with classic ML classifiers to achieve scalable data screening.

**Core Idea**: High-quality domain data + continual pre-training + model merging can make resource-efficient small models competitive solutions for complex medical tasks.

## Method

### Overall Architecture

The methodology consists of two major components: (1) **Medical Filtering Pipeline**—using Mixtral for zero-shot annotation of the FineWeb2 German subset, training an XLM-RoBERTa classifier to scale to the full dataset, resulting in the FineMed-de corpus; (2) **Model Adaptation**—performing continual pre-training on instruction-tuned models, followed by SLERP merging with the original instruction-tuned checkpoints to recover instruction-following capabilities.

### Key Designs

1.  **Hybrid Medical Document Filtering Pipeline**:

    - **Function**: Efficiently extract high-quality medical documents from general web corpora.
    - **Mechanism**: (a) Sample 260,000 documents from the FineWeb2 German subset; (b) Use Mixtral-8x7B for zero-shot classification into medical/non-medical categories (human-verified F1=91.1%); (c) Fine-tune an XLM-RoBERTa (279M) classifier on the labeled data (precision 0.95, recall 0.80); (d) Apply the classifier to the full 428 million documents to extract 7.3 million medical documents (5.1 billion tokens).
    - **Design Motivation**: LLMs provide high-quality annotations but are costly, while classic ML classifiers offer scalability—the hybrid approach balances quality and efficiency.

2.  **Continual Pre-training + SLERP Model Merging**:

    - **Function**: Inject domain knowledge while maintaining instruction-following capabilities.
    - **Mechanism**: Conduct 2-epoch continual pre-training on FineMed-de for instruction-tuned models (using FSDP + Flash Attention + mixed precision), then use SLERP to merge the pre-trained model with the original instruction-tuned checkpoint via layer-wise interpolation. Three base models were selected: Qwen2.5-7B, Mistral-7B, and Mistral-Small-24B.
    - **Design Motivation**: Continual pre-training can lead to catastrophic forgetting and degradation of instruction-following; model merging provides an efficient way to recover these capabilities without additional fine-tuning.

3.  **Multi-dimensional Evaluation Design**:

    - **Function**: Comprehensively evaluate the effects and tradeoffs of domain adaptation.
    - **Mechanism**: (a) Knowledge-intensive benchmarks (MMLU-de medical subset + MedQA-de) to assess medical knowledge; (b) Pairwise win-rate analysis to evaluate complex medical instruction following; (c) Failure mode analysis (language mixing, verbosity) to assess side effects.
    - **Design Motivation**: A single benchmark may obscure real performance gaps; multi-dimensional evaluation reveals the full picture of domain adaptation.

### Loss & Training

Continual pre-training utilizes the standard language modeling objective (next token prediction), using the AdamW optimizer, linear learning rate decay, and 500-step warmup. Training efficiency is optimized using FSDP, Flash Attention, activation checkpointing, and sequence packing.

## Key Experimental Results

### Main Results

**Average Accuracy on German Medical Benchmarks**

| Model | Average Accuracy |
|------|----------|
| BioMistral-7B (Baseline) | 43.55 |
| BioMistral-7B-SLERP | 48.22 |
| Mistral-7B-Instruct | 49.73 |
| DeFineMed-Mistral-7B-SLERP | **56.46** |
| Qwen2.5-7B-Instruct | 59.08 |
| DeFineMed-Qwen2.5-7B | **64.91** |

### Ablation Study

- In pairwise win-rate analysis, the Qwen2.5-based DeFineMed 7B model showed an approximately 3.5x increase in win rate against Mistral-Small-24B-Instruct.
- Model merging (SLERP) successfully recovered instruction-following capabilities but introduced side effects such as language mixing (German-English hybrid output) and increased verbosity.
- The gain from continual pre-training for the Qwen2.5 base model (+5.83) was slightly lower than for the Mistral model (+6.73), though both were significant.

### Key Findings

- Continual pre-training + model merging allows 7B models to approach or even compete with 24B models on German medical tasks.
- Data quality is more important than data scale—a meticulously filtered corpus of 5.1 billion tokens is sufficient to achieve significant improvements.
- Model merging is effective for recovering instruction-following capabilities, but inherent tradeoffs like language mixing exist.
- The choice of base model has a major impact on final performance (Qwen2.5 > Mistral).

## Highlights & Insights

- The hybrid data filtering pipeline (LLM labeling + ML classifier) is practical and reproducible for other domains or languages.
- The conclusion that "7B competes with 24B" has important practical implications for resource-constrained clinical scenarios.
- Failure mode analysis (language mixing, verbosity) provides an honest assessment of tradeoffs.
- The methodology can be directly generalized to the development of medical LLMs for other non-English languages.

## Limitations & Future Work

- Focused only on German; not extended to other languages.
- Language mixing and verbosity issues require subsequent targeted fine-tuning to resolve.
- The optimal sequence of continual pre-training and instruction fine-tuning remains an open question.
- Model usability has not been validated in real-world clinical settings.

## Related Work & Insights

- Compared to BioMistral, this work focuses more on the competitiveness of small models against large models rather than just pursuing benchmark scores.
- While Apollo-2 follows an instruction fine-tuning route, this work adopts a continued pre-training route; the two are complementary.
- The effectiveness of SLERP in the medical domain is further validated.

## Rating

- Novelty: ⭐⭐⭐ The components of the method are known techniques, but their combined application for the German medical scenario is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ The evaluation is complete across three dimensions: multi-benchmark, win-rate analysis, and failure mode analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and reasonable experimental design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study](../../CVPR2026/medical_imaging/are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)
- [\[ICLR 2026\] Neural Synchrony Between Socially Interacting Language Models](../../ICLR2026/medical_imaging/neural_synchrony_between_socially_interacting_language_models.md)
- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)
- [\[ACL 2026\] Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation](inflated_excellence_or_true_performance_rethinking_medical_diagnostic_benchmarks.md)
- [\[AAAI 2026\] MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging](../../AAAI2026/medical_imaging/mirnet_integrating_constrained_graph-based_reasoning_with_pre-training_for_diagn.md)

</div>

<!-- RELATED:END -->
