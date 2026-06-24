---
title: >-
  [Paper Note] AGRI-CM3: A Chinese Massive Multi-Modal Multi-Level Benchmark for Agricultural Understanding
description: >-
  [ACL 2025][Multimodal VLM][Agricultural Multimodal Benchmark] This paper introduces AGRI-CM3, a large-scale Chinese multimodal and multi-level evaluation benchmark for the agricultural domain. It covers various agricultural subtasks, including crop identification, pest and disease diagnosis, and farming operation understanding, to systematically evaluate the capabilities of VLMs in the agricultural vertical domain.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Agricultural Multimodal Benchmark"
  - "Chinese Agricultural Understanding"
  - "Multi-level Evaluation"
  - "Vision-Language Models"
  - "Domain-specific Evaluation"
date: 2026-05-08
content_hash: 8585fae7c61f4fd8
---

# AGRI-CM3: A Chinese Massive Multi-Modal Multi-Level Benchmark for Agricultural Understanding

**Conference**: ACL 2025  
**Code**: None  
**Area**: Multimodal VLM / Agricultural AI  
**Keywords**: Agricultural Multimodal Benchmark, Chinese Agricultural Understanding, Multi-level Evaluation, Vision-Language Models, Domain-specific Evaluation

## TL;DR
This paper introduces AGRI-CM3, a large-scale Chinese multimodal and multi-level evaluation benchmark for the agricultural domain. It covers various agricultural subtasks, including crop identification, pest and disease diagnosis, and farming operation understanding, to systematically evaluate the capabilities of VLMs in the agricultural vertical domain.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) have achieved remarkable progress in general domains, but their performance in professional vertical domains such as agriculture remains unclear. Existing multimodal evaluation benchmarks (e.g., MMBench, SEED-Bench) primarily focus on general scenarios, lacking in-depth evaluation of domain-specific capabilities.

**Limitations of Prior Work**: (1) The absence of large-scale multimodal evaluation benchmarks for agriculture prevents the systematic identification of VLM capability blind spots in agricultural applications; (2) agricultural images possess unique visual features—such as crop morphology, disease symptoms, and field scenes—that require specialized knowledge to comprehend; (3) Chinese agricultural question-answering resources are extremely scarce, despite China being the world's largest agricultural producer with a huge demand for agricultural AI.

**Key Challenge**: Although general VLMs perform exceptionally well in everyday visual question-answering, agricultural understanding requires fine-grained, professional visual knowledge (such as distinguishing leaf diseases across different rice varieties). Whether existing models possess such capabilities remains unknown.

**Goal**: To construct a comprehensive Chinese agricultural multimodal benchmark to systematically evaluate the capabilities of mainstream VLMs across various levels of agricultural understanding.

**Key Insight**: To organize agricultural understanding tasks by cognitive levels (from perception to reasoning) and integrate multiple agricultural subfields (such as crop farming, animal husbandry, and aquaculture) to form a multidimensional evaluation framework.

**Core Idea**: To build the AGRI-CM3 benchmark, containing a large volume of agricultural image-guestion pairs that cover multiple cognitive levels from basic recognition to complex reasoning. Annotated in Chinese, it systematically reveals the capability shortfalls of VLMs in the agricultural vertical domain.

## Method

### Overall Architecture
The construction of the AGRI-CM3 benchmark consists of three stages: (1) Data Collection—collecting agricultural images and knowledge from agricultural databases, agricultural technology service platforms, and professional textbooks; (2) Task Design—designing multi-level tasks according to Bloom's taxonomy; (3) Quality Control—conducting annotation review and quality assurance by agricultural experts.

### Key Designs

1. **Multi-level Cognitive Task System**:

    - Function: To systematically evaluate the performance of VLMs across different levels of cognitive difficulty.
    - Mechanism: Agricultural understanding tasks are divided into multiple levels: the perception level (crop/animal identification, color/morphology description), the understanding level (disease diagnosis, growth stage determination), the analysis level (etiology analysis, yield estimation), and the reasoning level (pesticide application recommendation, farming decision-making). Each level contains various question types (multiple-choice, true/false, short answer, etc.).
    - Design Motivation: Distinguishing between different levels allows precise localization of the capability bottlenecks in VLMs—whether they stem from inadequate basic visual perception or a lack of professional knowledge reasoning.

2. **Multi-domain Agricultural Coverage**:

    - Function: To ensure the comprehensiveness and representativeness of the benchmark.
    - Mechanism: It covers major agricultural subdomains such as crop farming (staple crops, cash crops, vegetables, and fruits), animal husbandry (poultry/livestock identification and diseases), aquaculture, and forestry. Each subdomain includes region-specific varieties and disease types, with a particular focus on crops, pests, and diseases common in China.
    - Design Motivation: The diverse nature of agriculture means that evaluation within a single subdomain cannot represent overall capabilities.

3. **Professional Chinese Annotation and Quality Control**:

    - Function: To ensure the professionalism and accuracy of the evaluation data.
    - Mechanism: Graduate students majoring in agriculture-related fields and agricultural technology extension personnel participated in the annotation. Each question underwent independent annotation by at least two individuals and was audited by a third. For highly specialized questions (such as disease diagnosis), Ph.D. students in plant protection performed the final review. All questions and options are in Chinese to ensure localized evaluation.
    - Design Motivation: The agricultural domain contains numerous technical terms and regional characteristics, and annotations by non-experts are highly prone to errors.

### Loss & Training
This work presents an evaluation benchmark and does not involve model training. The assessment methods include: accuracy for multiple-choice questions, and a combination of GPT-4 assisted scoring with manual spot-checking for short-answer questions.

## Key Experimental Results

### Main Results

| Model | Perception Level | Understanding Level | Analysis Level | Reasoning Level | Overall |
|------|--------|--------|--------|--------|------|
| GPT-4V | 65.2 | 48.7 | 35.4 | 28.1 | 44.4 |
| Qwen-VL-Plus | 58.3 | 42.1 | 30.8 | 22.5 | 38.4 |
| InternVL2 | 62.1 | 45.3 | 33.2 | 25.6 | 41.6 |
| LLaVA-1.5 | 45.6 | 32.8 | 22.1 | 16.3 | 29.2 |
| Human Expert | 92.5 | 85.3 | 78.6 | 72.1 | 82.1 |

### Ablation Study

| Subdomain | GPT-4V | Qwen-VL-Plus | Human Expert |
|--------|--------|-------------|---------|
| Staple Crops | 48.2 | 41.5 | 85.6 |
| Pest & Disease Diagnosis | 35.1 | 28.9 | 78.2 |
| Livestock Diseases | 42.3 | 35.7 | 81.4 |
| Farming Operations | 38.6 | 33.2 | 79.8 |

### Key Findings
- The performance of all VLMs on agricultural tasks is far below that on general tasks (e.g., GPT-4V scores only 44.4% overall), showing a massive gap compared to human experts (82.1%).
- Performance drops sharply as the cognitive level increases—from 65% at the perception level to 28% at the reasoning level, indicating that VLMs lack agricultural domain-specific reasoning capabilities.
- Pest and disease diagnosis is the most challenging subtask; even GPT-4V only achieves an accuracy of approximately 35%, highlighting the difficulty of fine-grained visual recognition in this domain.
- Chinese agricultural understanding is generally weaker than English general understanding, indicating that Chinese agricultural knowledge is insufficiently covered in training datasets.

## Highlights & Insights
- As the first large-scale Chinese agricultural multimodal benchmark, it fills a critical gap in the field and holds significant benchmark value for Chinese agricultural AI research and applications.
- The multi-level cognitive task design can be transferred to VLM evaluations in other vertical domains (e.g., medicine, law).
- The experimental results clearly reveal the capability boundaries of VLMs in vertical domains, pointing the direction for future agricultural VLM training.

## Limitations & Future Work
- The paper has not released an arXiv preprint, and the scale as well as the exact composition of the dataset remain somewhat unclear.
- Evaluation mainly focuses on static image understanding, lacking videos (e.g., crop growth processes) and time-series analysis.
- Automated assessment of short-answer questions relies on GPT-4, which may introduce evaluation bias.
- The work does not investigate how to utilize this benchmark to enhance agricultural understanding capabilities of VLMs (e.g., domain fine-tuning strategies).

## Related Work & Insights
- **vs MMBench**: MMBench is a general VLM benchmark, whereas AGRI-CM3 is the first vertical domain benchmark for agriculture, filling the gap in evaluating specialized application scenarios.
- **vs ScienceQA**: ScienceQA covers some biological and agricultural knowledge but leans towards general science, whereas AGRI-CM3 focuses more specifically on practical agricultural applications.
- **vs Other Domain Benchmarks**: Analogous to MedBench (a medical VLM benchmark), AGRI-CM3 establishes a comparable evaluation infrastructure for agricultural AI.

## Rating
- Novelty: ⭐⭐⭐⭐ Fills the gap in agricultural VLM evaluation, with a highly characteristic multi-level design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple mainstream VLMs, but lacks evaluations of more open-source models.
- Writing Quality: ⭐⭐⭐ Cannot be fully assessed (the complete paper was not available).
- Value: ⭐⭐⭐⭐ Makes a significant benchmark contribution to agricultural AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)
- [\[ACL 2025\] MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark](mmmu_pro_robust_benchmark.md)
- [\[ACL 2025\] Jailbreak Large Vision-Language Models Through Multi-Modal Linkage](jailbreak_large_vision-language_models_through_multi-modal_linkage.md)
- [\[NeurIPS 2025\] DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding](../../NeurIPS2025/multimodal_vlm/danmakutppbench_a_multimodal_benchmark_for_temporal_point_pr.md)
- [\[NeurIPS 2025\] Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants](../../NeurIPS2025/multimodal_vlm/face-human-bench_a_comprehensive_benchmark_of_face_and_human_understanding_for_m.md)

</div>

<!-- RELATED:END -->
