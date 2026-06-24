---
title: >-
  [Paper Note] MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale
description: >-
  [VLM Reasoning] Proposes a scalable, low-cost method to construct MAmmoTH-VL-Instruct, a multimodal instruction tuning database of 12 million instances rich in Chain-of-Thought (CoT) reasoning, using only open-source models. The resulting model, MAmmoTH-VL-8B, achieves state-of-the-art (SOTA) performance on multimodal reasoning benchmarks (e.g., MathVerse +8.1%, MMMU-Pro +7%, MuirBench +13.3%).
tags:
  - "VLM Reasoning"
date: 2026-05-08
content_hash: 842b46763fa33d26
---

# MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale

| Conference | arXiv | Code | Area | Keywords |
|------|-------|------|------|----------|
| ACL 2025 | [2412.05237](https://arxiv.org/abs/2412.05237) | [Project Page](https://mammoth-vl.github.io) | multimodal_vlm | Multimodal Reasoning, Instruction Tuning, CoT, Data Rewriting, Large-Scale Training Data |

## TL;DR

Proposes a scalable, low-cost method to construct MAmmoTH-VL-Instruct, a multimodal instruction tuning database of 12 million instances rich in Chain-of-Thought (CoT) reasoning, using only open-source models. The resulting model, MAmmoTH-VL-8B, achieves state-of-the-art (SOTA) performance on multimodal reasoning benchmarks (e.g., MathVerse +8.1%, MMMU-Pro +7%, MuirBench +13.3%).

## Background & Motivation

- **Limitations of Prior Work**: Existing multimodal instruction tuning datasets primarily originate from academic VQA datasets (e.g., VQA, AI2D, ChartQA). These datasets focus on simpler tasks and only provide short phrase-level answers without intermediate reasoning processes, limiting the model's reasoning capabilities.
- **Key Challenge**: While Chain-of-Thought (CoT) reasoning has shown significant efficacy in text-only LLMs, constructing large-scale multimodal CoT datasets faces two major obstacles: (1) ensuring instruction diversity and complexity, and (2) generating coherent responses with detailed justifications. Manual annotation is prohibitively expensive, and relying on closed-source models like GPT-4 involves high costs and copyright concerns.
- **Design Motivation**: To achieve low-cost and scalable construction of multimodal CoT datasets using open-source models, lowering the barrier to entry for the open-source community.

## Method

### Overall Architecture

A three-step data construction pipeline:
1. **Data Collection and Categorization**: Collected from 153 public datasets and organized into 10 major categories (General, OCR, Chart, Caption, Domain-specific, Code&Math, Language, Detection, Multi-Image, Video).
2. **Instruction Data Rewriting**: Translating short answers into detailed responses containing CoT reasoning using open-source models.
3. **Self-Filtering**: Utilizing the same MLLM as a judge (Model-as-Judge) to filter out hallucinated content.

### Key Designs

1. **Three-Tier Data Grouping**:
    - Group A (58 datasets): High-quality, original data is directly retained.
    - Group B (60 datasets): Promising but brief responses, enhanced through rewriting.
    - Group C (35 datasets): Excessively vague or short, discarded directly.

2. **Task-Aware Rewriting Strategy**: Tailored prompts are designed for each data category. For Caption-type data, a text-only LLM (Llama-3-70B) is used to generate task-oriented QA pairs. For other categories, a multimodal model (InternVL2-Llama3-76B) is utilized to ensure vision-language alignment.

3. **Data Mixing Ratio**: 70% rewritten data + 30% original data. t-SNE analysis reveals that the rewritten data retains the core characteristics of the original distribution while expanding the overall coverage.

### Training Configuration

Three-stage training (based on the LLaVA-OneVision architecture):
- **Stage-1**: Language-image alignment pre-training (558K samples, projector-only training).
- **Stage-2**: Single-image visual instruction tuning (10M samples, full-parameter training).
- **Stage-3**: One-Vision multi-image/video fine-tuning (2M samples, full-parameter training).

LLM Backbone: Qwen2.5-7B-Instruct, Vision Encoder: SigLIP-so400m-patch14-384

## Experiments

### Main Results: Multi-Disciplinary Knowledge & Mathematical Reasoning

| Model | MMStar | MMMU (val) | MMMU-Pro | MathVerse | MathVista |
|------|--------|-----------|----------|-----------|-----------|
| GPT-4o | 64.7 | 69.1 | 49.7 | 50.2 | 63.8 |
| Qwen2-VL-7B | 60.7 | 52.1 | 26.9 | 28.2 | 58.2 |
| LLaVA-OV-7B | 61.7 | 48.8 | 18.7 | 26.2 | 63.2 |
| Llava-CoT-11B | 57.6 | 48.9 | 18.5 | 24.2 | 54.8 |
| **MAmmoTH-VL-8B** | **63.0** | **50.8** | **25.3** | **34.2** | **67.6** |
| Gain vs. Best Open-Source (~10B) | +1.3 | +1.9 | **+7.1** | **+8.1** | +4.4 |

### Document and Chart Understanding

| Model | AI2D | ChartQA | DocVQA | RealWorldQA |
|------|------|---------|--------|-------------|
| LLaVA-OV-7B | 81.4 | 80.0 | 87.5 | 66.3 |
| InternVL-2-8B | 83.8 | 83.3 | 91.6 | 64.4 |
| **MAmmoTH-VL-8B** | **84.0** | **86.2** | **93.7** | 69.9 |
| Gain vs. Best Open-Source (~10B) | +2.4 | +2.1 | +1.6 | +0.6 |

### Key Findings

1. **Self-Filtering is Crucial**: OCR and chart data exhibit the highest hallucination filtering rates, and removing the filtering step leads to a significant drop in model performance.
2. **Improved Rewritten Data Quality**: Post-rewritten data scores higher than the original data on both information richness and relevance (on a 5-point scale).
3. **Significant Data Scaling Effects**: Performance consistently improves as training data increases from 2M to 10M, demonstrating the scalability of large-scale CoT data.
4. **Non-Reasoning Tasks Also Benefit**: Improvements of up to 4% are observed even on non-reasoning benchmarks, indicating the generalization benefits of CoT training.

## Highlights & Insights

- Constructs a 12M large-scale multimodal CoT dataset using only open-source models, breaking the reliance on closed-source models like GPT-4.
- The three-step pipeline (collect-rewrite-filter) is simple and scalable, with a methodology that can be replicated in other domains.
- The 8B model significantly outperforms models of similar or even larger scales on reasoning-intensive tasks (e.g., MathVerse +8.1%).
- The Model-as-Judge self-filtering method achieves an agreement rate of Kappa 0.64 (good level) with human evaluation.

## Limitations & Future Work

- Using the same generative model as a judge for self-filtering may lead to blind spots regarding its own specific error patterns.
- High hallucination rates persist in OCR and chart data during rewriting, indicating that open-source MLLMs still fall short in fine-grained visual understanding.
- Although training costs are lower than using GPT-4, training on a 10M scale still requires substantial computational resources.
- Data collection is dependent on existing public datasets, potentially leading to insufficient coverage representing new domains or tasks.

## Related Work

- **Multimodal Instruction Tuning**: LLaVA (Liu et al. 2024b) pioneered the visual instruction tuning paradigm; LLaVA-OneVision (Li et al. 2024b) extended this to multi-image and video scenarios.
- **Reasoning Enhancement**: Chain-of-Thought (Wei et al. 2022) introduces step-by-step reasoning; LLaVA-CoT (Xu et al. 2024a) introduces CoT in a single model but relies on GPT-4 generated data.
- **Data Quality and Filtering**: Cambrian (Tong et al. 2024) explores multi-source data fusion in training; InternVL2 (Chen et al. 2023b) focuses on large-scale pre-training; this work highlights the feasibility of self-filtering using open-source models.

## Rating

| Metric | Score (1-10) |
|------|------------|
| Novelty | 7 |
| Technical Depth | 7 |
| Experimental Thoroughness | 9 |
| Writing Quality | 8 |
| **Overall** | **7.5** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](../../ICCV2025/vlm_reasoning/r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](../../ICLR2026/vlm_reasoning/vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [\[ICCV 2025\] Training-Free Personalization via Retrieval and Reasoning on Fingerprints](../../ICCV2025/vlm_reasoning/training-free_personalization_via_retrieval_and_reasoning_on_fingerprints.md)
- [\[CVPR 2025\] Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents](../../CVPR2025/vlm_reasoning/document_haystacks_vision-language_reasoning_over_piles_of_1000_documents.md)
- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](../../ICCV2025/vlm_reasoning/physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)

</div>

<!-- RELATED:END -->
