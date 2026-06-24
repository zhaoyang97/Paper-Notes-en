---
title: >-
  [Paper Note] MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding
description: >-
  [ICML 2025][Medical Imaging][Medical QA benchmark] MedXpertQA constructs an expert-level medical QA benchmark comprising 4,460 questions across 17 specialties and 11 organ systems. Utilizing rigorous filtering enhancement and data synthesis for leakage prevention, it evaluates 18 mainstream models and introduces a specialized reasoning subset designed specifically for assessing o1-like reasoning models.
tags:
  - "ICML 2025"
  - "Medical Imaging"
  - "Medical QA benchmark"
  - "expert-level reasoning"
  - "multimodal evaluation"
  - "data leakage prevention"
  - "o1 reasoning evaluation"
date: 2026-05-08
content_hash: ee962a0deecb5752
---

# MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding

**Conference**: ICML 2025  
**arXiv**: [2501.18362](https://arxiv.org/abs/2501.18362)  
**Code**: [https://github.com/TsinghuaC3I/MedXpertQA](https://github.com/TsinghuaC3I/MedXpertQA)  
**Area**: Medical NLP  
**Keywords**: Medical QA benchmark, expert-level reasoning, multimodal evaluation, data leakage prevention, o1 reasoning evaluation

## TL;DR
MedXpertQA constructs an expert-level medical QA benchmark comprising 4,460 questions across 17 specialties and 11 organ systems. Utilizing rigorous filtering enhancement and data synthesis for leakage prevention, it evaluates 18 mainstream models and introduces a specialized reasoning subset designed specifically for assessing o1-like reasoning models.

## Background & Motivation

**Background**: Medical QA is the core benchmark category for evaluating LLM/MLLM medical capabilities. Existing benchmarks such as MedQA, MedMCQA, and PubMedQA have been widely adopted. Models like GPT-4 have neared or exceeded human-level performance on multiple medical QA datasets.

**Limitations of Prior Work**: (1) Insufficient difficulty—GPT-4 has achieved 90%+ on MedQA, presenting a severe ceiling effect; (2) Data leakage—training data may contain test questions; (3) Insufficient multimodality—existing multimodal medical benchmarks mostly consist of simple image caption-based QA, lacking genuine clinical reasoning questions; (4) Lack of reasoning evaluation—there is no medical benchmark specifically designed to evaluate o1-like reasoning capabilities.

**Key Challenge**: There is a need for a benchmark that is sufficiently challenging (to differentiate model capabilities), leakage-free (for fair evaluation), and clinically relevant (focusing on expert-level diagnostic reasoning rather than encyclopedic QA).

**Goal**: To construct a genuine, expert-level medical reasoning and understanding benchmark.

**Key Insight**: Starting from specialty board exam questions, followed by rigorous filtering and enhancement (removing simple questions) + data synthesis (to prevent leakage) + multi-round expert review.

**Core Idea**: Incorporating specialty board-level difficulty + data synthesis for leakage prevention + a reasoning-oriented subset to build a medical benchmark capable of truly differentiating state-of-the-art models.

## Method

### Overall Architecture

- **Data Source**: Medical board exam questions (e.g., USMLE Step 3, various specialty board exams)
- **Text Subset (Text)**: 4,460 textual QA questions
- **Multimodal Subset (MM)**: Complex QA incorporating medical images (CT/MRI/X-ray/pathology, etc.) + patient records + examination findings
- **Reasoning Subset**: Questions requiring multi-step reasoning, specifically designed to evaluate o1-like models

### Key Designs

1. **Rigorous Filtering and Enhancement Mechanism**:

    - First round: Remove "simple" questions that GPT-4 can easily answer.
    - Second round: Enhance difficulty—modify distractors, increase clinical context complexity.
    - Retention strategy: Preserve only questions requiring specialty knowledge and multi-step reasoning.
    - **Design Motivation**: The primary issue with existing benchmarks is that they are overly simple.

2. **Data Synthesis for Leakage Prevention**:

    - Paraphrase/synthesize original questions—alter clinical scenarios, numerical values, and options.
    - Ensure synthesized questions are semantically equivalent but textually distinct from the originals.
    - Conduct multiple rounds of automated and manual checks to ensure they cannot be directly searched.
    - **Design Motivation**: Public exam questions might be included in LLM training corpora, making it essential to prevent data leakage.

3. **Multimodal Subset Design (MM)**:

    - Rather than simple visual question answering—each question includes: medical images + chief complaint + medical history + laboratory findings.
    - Demands integrated diagnostic reasoning from multiple information sources.
    - Diverse image formats: CT, MRI, X-ray, ultrasound, dermoscopy, pathological slides, etc.
    - **Design Motivation**: Real-world clinical scenarios require integrated reasoning across multiple source modalities.

4. **Reasoning-Oriented Subset**:

    - Specifically filter for questions requiring $\ge 3$ steps of reasoning.
    - Appropriate for evaluating reasoning-enhanced models like o1 and o3.
    - Contains diagnostic reasoning chain annotations.
    - **Design Motivation**: Medicine serves as a natural domain for assessing reasoning capabilities (highly complex yet with definitive answers).

### Loss & Training (Benchmark, no training)

- Evaluation Metric: Multiple-choice accuracy
- Evaluation Methods: Zero-shot, few-shot, Chain-of-Thought
- Separate evaluation for open-book and closed-book settings
- Auxiliary evaluation of visual comprehension for multimodal questions

## Key Experimental Results

### Main Results

| Model | Text Accuracy | MM Accuracy | Reasoning Subset |
|------|-----------|----------|---------|
| GPT-4o | ~65% | ~55% | ~60% |
| Claude 3.5 | ~63% | ~53% | ~58% |
| o1 | ~70% | - | ~68% |
| Gemini 1.5 Pro | ~60% | ~50% | ~55% |
| Med-PaLM 2 | ~58% | - | - |
| LLaMA 3 70B | ~50% | - | ~45% |
| Open-source MLLM | <50% | <45% | <40% |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| All Questions vs. Enhanced | 15-20% accuracy gap | Filtering effectively increases difficulty |
| Original vs. Synthesized | Comparable accuracy | Synthesis does not alter difficulty |
| Text vs. Multimodal | MM is more difficult | Integrated multi-source reasoning is challenging |
| Standard QA vs. Reasoning Subset | Reasoning is harder | Multi-step reasoning poses high demands on models |
| No CoT vs. CoT | CoT helps | Reasoning questions benefit significantly |

### Key Findings

- **SOTA models still fall short of expert levels**: GPT-4o achieves ~65% on the Text subset, which is significantly below specialist levels.
- **Multimodality remains a weakness**: All models score ~10% lower on the MM subset compared to Text.
- **o1-like models show limited advantages**: Reasoning enhancement yields an approximate 5-8% improvement on the reasoning subset.
- **Large gap in open-source models**: Open-source models lag behind closed-source counterparts by 15-20%.
- **Effective data leakage prevention**: The accuracy of synthesized questions is comparable to that of the original questions.

## Highlights & Insights

1. **Sufficient difficulty**: Currently, SOTA models achieve only 65-70%, ensuring strong discriminative power.
2. **Data security**: Multi-layered leakage prevention mechanisms guarantee fair evaluation.
3. **Multimodal innovation**: Rather than simple VQA, it focuses on clinical-grade, multi-source reasoning.
4. **Reasoning evaluation**: The first medical reasoning benchmark targeting o1-like models.
5. **Beyond medicine**: Provides a rich real-world test scenario for general reasoning evaluation.

## Limitations & Future Work

1. Primarily format-restricted to multiple-choice questions, lacking coverage of free-text diagnostic report generation.
2. Primarily English-based, necessitating future expansion to multilingual medical QA.
3. Uneven coverage across some specialties (limited question volume for certain subspecialties).
4. Multimodal image quality and resolution are constrained by their original sources.

## Related Work & Insights

- MedQA and MedMCQA are prior-generation medical QA benchmarks.
- MMMU and ScienceQA serve as baselines for multi-disciplinary multimodal evaluation.
- Insight: The methodology of synthesizing data to prevent leakage can be generalized to the construction of other benchmarks.

## Rating
- Novelty: ⭐⭐⭐⭐ Benchmark paper; design innovations lie in difficulty control and the reasoning subset.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 18 models.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-defined problems and rational design.
- Value: ⭐⭐⭐⭐⭐ Fills the gap in expert-level medical reasoning evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] THUNDER: Tile-level Histopathology image UNDERstanding benchmark](../../NeurIPS2025/medical_imaging/thunder_tile-level_histopathology_image_understanding_benchmark.md)
- [\[ICLR 2026\] U2-BENCH: Benchmarking Large Vision-Language Models on Ultrasound Understanding](../../ICLR2026/medical_imaging/u2-bench_benchmarking_large_vision-language_models_on_ultrasound_understanding.md)
- [\[ICML 2025\] Mastering Multiple-Expert Routing: Realizable H-Consistency and Strong Guarantees](mastering_multiple-expert_routing_realizable_h-consistency_and_strong_guarantees.md)
- [\[NeurIPS 2025\] SMMILE: An Expert-Driven Benchmark for Multimodal Medical In-Context Learning](../../NeurIPS2025/medical_imaging/smmile_an_expert-driven_benchmark_for_multimodal_medical_in-context_learning.md)
- [\[CVPR 2025\] Interactive Medical Image Analysis with Concept-based Similarity Reasoning](../../CVPR2025/medical_imaging/interactive_medical_image_analysis_with_concept-based_similarity_reasoning.md)

</div>

<!-- RELATED:END -->
