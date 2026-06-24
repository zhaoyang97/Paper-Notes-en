---
title: >-
  [Paper Note] Understanding Museum Exhibits using Vision-Language Reasoning
description: >-
  [ICCV 2025][VLM Reasoning][Museum Visual Question Answering] Constructs Museum-65, a large-scale museum exhibit dataset containing 65 million images and 200 million QA pairs. By fine-tuning BLIP and LLaVA on this dataset, the study demonstrates that domain-specific large-scale datasets significantly outperform zero-shot state-of-the-art (SOTA) VLMs, with the fine-tuned LLaVA achieving 57% and 70% accuracy on exhibit title and origin identification…
tags:
  - "ICCV 2025"
  - "VLM Reasoning"
  - "Museum Visual Question Answering"
  - "Large-scale Dataset"
  - "Domain Fine-tuning"
  - "Cultural Heritage"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: b34dabf5a99ef19f
---

# Understanding Museum Exhibits using Vision-Language Reasoning

**Conference**: ICCV 2025  
**arXiv**: [2412.01370](https://arxiv.org/abs/2412.01370)  
**Code**: [github.com/insait-institute/Museum-65](https://github.com/insait-institute/Museum-65)  
**Area**: Multimodal VLMs  
**Keywords**: Museum Visual Question Answering, Large-scale Dataset, Domain Fine-tuning, Cultural Heritage, Vision-Language Models

## TL;DR

Constructs Museum-65, a large-scale museum exhibit dataset containing 65 million images and 200 million QA pairs. By fine-tuning BLIP and LLaVA on this dataset, the study demonstrates that domain-specific large-scale datasets significantly outperform zero-shot state-of-the-art (SOTA) VLMs, with the fine-tuned LLaVA achieving 57% and 70% accuracy on exhibit title and origin identification, respectively (compared to 22% and 33% for GPT-4o).

## Background & Motivation

### Problem Definition

As treasuries of cultural heritage and historical artifacts, museum exhibit understanding requires integrating visual features with historical knowledge. This paper aims to build AI models capable of understanding museum exhibits, enabling them to accurately answer multi-dimensional questions about titles, creators, periods, techniques, and cultural backgrounds in Visual Question Answering (VQA) tasks.

### Limitations of Prior Work

**General VLMs perform poorly in professional domains**: Although models like CLIP, Gemini, and LLaVA excel in general visual understanding, they fall short in domains requiring interdisciplinary knowledge such as museums, particularly in predicting structured attributes (e.g., period, origin, material) where performance is far below requirements.

**Existing datasets are small in scale and limited in domain**: Existing cultural heritage datasets (e.g., AQUA with 21K images, MUZE with 210K images) mostly focus only on art exhibits, and their scale is far from sufficient for training domain-specific models.

**Lack of evaluation benchmarks for real-world scenarios**: Prior works do not design systematic evaluations targeting real-world museum usage scenarios, such as multi-angle photography, multilingual queries, and questions requiring common-sense reasoning.

### Core Motivation

**Key Insight**: Museums are high-quality sources of human knowledge—exhibit information is annotated by domain experts, possessing extremely high accuracy and depth. Systematically converting this structured knowledge into a large-scale dataset can train domain-specific models that far outperform general models in this field. Meanwhile, the way real-world visitors interact with exhibits (curious questions, multi-angle observation, multilingual inquiries) provides a natural task definition for model evaluation.

## Method

### Overall Architecture

The methodology of this paper consists of three core components:
1. **Dataset Construction**: Collect, clean, and structure 65 million exhibit images and 200 million QA pairs from global museum aggregators and independent museums.
2. **Model Fine-tuning**: Fine-tune BLIP (encoder-decoder architecture) and LLaVA (instruction-tuned LLM + vision encoder) on Museum-65.
3. **Multi-task Evaluation**: Design five VQA tasks reflecting real museum scenarios for systematic evaluation.

### Key Designs

#### 1. **Museum-65 Dataset Construction**

- **Function**: Build the largest multimodal dataset of museum exhibits to date.
- **Mechanism**:
    - Data sources: 3 international aggregators (DPLA, Europeana, Smithsonian) + 12 independent museums, covering Europe, North America, Asia, Africa, and Oceania.
    - Data scale: 50 million English objects + 15 million multilingual objects (37 languages).
    - Attribute-to-question conversion: Convert attribute-value pairs of exhibits (e.g., `material: bronze`) into natural language questions (e.g., "What is the material used in the object?") using 63 handcrafted question templates.
    - Quality assurance: Cleaned and labeled by 10 experts over 3 months, with cross-verification by 2 experts.
- **Design Motivation**: Museum aggregators provide high-quality data curated by experts. Through large-scale collection and structured processing, data sufficient for training large VLMs can be generated. Diverse question templates simulate natural questions asked by visitors.

#### 2. **Dual-Model Fine-Tuning Strategy**

- **Function**: Fine-tune BLIP and LLaVA respectively to compare the performance of different architectures in domain-specific VQA.
- **Mechanism**:
    - BLIP: Encoder-decoder architecture (BERT-base, 110M parameters), strong at image-text alignment but weak in instruction following.
    - LLaVA: Instruction-tuned LLM based on Llama-7B, with stronger reasoning and instruction understanding capabilities.
    - Training configuration: BLIP is trained on 1M/10M/20M data for 5 epochs respectively; LLaVA is trained on 1M for 5 epochs and 20M for 1 epoch.
    - For each epoch, one QA pair is randomly selected per image.
- **Design Motivation**: Selecting two representative architectures—BLIP representing traditional vision-language alignment models, and LLaVA representing the new generation of LLM-based multimodal models—reveals the strengths and weaknesses of different architectures in domain-specific tasks through comparison.

#### 3. **Five Real-World Scenario VQA Tasks**

- **Function**: Design an evaluation framework that comprehensively covers the actual application scenarios of real museums.
- **Mechanism**:
    - **Task 1 General VQA**: Evaluated on all questions to test the model's comprehensive capability.
    - **Task 2 Categorical VQA**: Evaluated in groups by attribute categories (title, creator, material, etc.) to reveal the model's strengths and weaknesses across different knowledge types.
    - **Task 3 Multi-view**: Tested on images taken from different angles of the same exhibit to evaluate robustness against view changes.
    - **Task 4 Visually Unanswerable Questions**: Questions requiring common-sense reasoning (e.g., "Who was the painter's mentor?") to test deep knowledge integration capabilities.
    - **Task 5 Multilingual**: Questions in non-English languages such as French and German to evaluate cross-lingual generalization capability.
- **Design Motivation**: In real museum settings, visitors' questions go far beyond simple attribute queries—they may take photos from different angles, ask in their native languages, or ask questions that require background knowledge to answer.

### Loss & Training

- BLIP: Uses standard VQA fine-tuning scheme, batch size 512.
- LLaVA: Uses standard instruction fine-tuning scheme, batch size 512.
- Hardware: 64× NVIDIA H100 GPUs.
- Evaluation metrics: Accuracy (Exact Match / Partial Match), Recall, BLEU-1/2, METEOR, WMD accuracy.

## Key Experimental Results

### Main Results

**Zero-shot SOTA vs. Fine-tuned Models**:

| Model | Title Accuracy | Origin Accuracy |
|------|----------|----------|
| GPT-4o (Zero-shot) | 22.03 | 33.33 |
| Claude-3-7-sonnet (Zero-shot) | 21.89 | 40.43 |
| Gemini-1.5B-flash (Zero-shot) | 27.08 | 32.98 |
| LLaVA w/o Fine-tuning | 10.13 | 23.42 |
| **LLaVA-ours (20M, 1ep)** | **57.00** | **70.00** |
| BLIP w/o Fine-tuning | 3.00 | 5.00 |
| **BLIP-ours (20M, 5ep)** | **52.00** | **61.00** |

**Semantic Evaluation**:

| Model | METEOR | WMD Accuracy |
|------|--------|----------|
| BLIP w/o Fine-tuning | 3.24 | 35.54 |
| BLIP-ours (20M, 5ep) | 37.45 | 74.02 |
| LLaVA w/o Fine-tuning | 2.96 | 54.50 |
| **LLaVA-ours (20M, 1ep)** | **58.85** | **87.02** |

### Ablation Study

**Partial Accuracy of Exhibits from Various Continents**:

| Model | Europe | North America | South America | Asia | Africa | Oceania |
|------|------|------|------|------|------|--------|
| LLaVA-ours | 85.2 | 79.6 | 86.6 | 67.4 | 86.7 | 99.2 |
| LLaVA w/o Fine-tuning | 8.6 | 43.57 | 20.3 | 23.4 | 20.79 | 52.4 |
| BLIP-ours | 79.1 | 73.1 | 76.4 | 65.5 | 76.4 | 49.7 |
| BLIP w/o Fine-tuning | 4.3 | 15.2 | 19.7 | 9.3 | 19.7 | 6.6 |

**Fine-tuned Models vs. Human Experts (Categorical VQA)**: Fine-tuned models outperform 10 museum experts across all categories.

### Key Findings

1. **Enormous impact of domain fine-tuning**: LLaVA's title identification accuracy leaps from 10.13% to 57% (+46.87 pp) after fine-tuning, demonstrating the extreme importance of domain-specific data.
2. **LLaVA comprehensively outperforms BLIP**: Across all tasks, LLaVA dominates due to its larger language model (7B vs. 110M) and superior reasoning logic, exhibiting a wider gap in tasks requiring common-sense reasoning.
3. **Strong view robustness**: In multi-view testing, fine-tuned models exhibit only a minor drop in accuracy (58.09 -> 56.14), showcasing strong robustness against changes in perspective.
4. **Limited multilingual capability**: Fine-tuning exclusively in English causes a degradation in LLaVA's multilingual capability (French partial accuracy drops from 41.81 to 10.37), suggesting a need for multilingual fine-tuning.
5. **Imbalanced training data does not hinder global gains**: Despite the training data being predominantly European and North American, fine-tuning significantly improves exhibit identification across all continents.

## Highlights & Insights

1. **Breakthrough in dataset scale and quality**: 65M images and 200M QA pairs, which is 130 times larger than the previous largest cultural heritage multimodal dataset (VISCOUNTH).
2. **Realistic experimental design**: The 5 tasks fully cover the real-world demands of a museum AI assistant—multi-angle identification, cross-lingual services, and in-depth knowledge-based QA.
3. **Demonstration of "Domain Data > Model Size"**: The 7B fine-tuned model outperforms zero-shot GPT-4o, strongly validating the value of domain-specific fine-tuning in specialized scenarios.
4. **Fine-tuned models outperform human experts**: This is an important milestone, showing that AI can surpass domain experts in structured-knowledge-intensive tasks.
5. **Systematic analysis of data bias**: Comprehensive analysis and mitigation measures for geographic bias, language bias, and photographic-perspective bias.

## Limitations & Future Work

1. **Lack of multilingual fine-tuning**: Currently, fine-tuning exclusively in English leads to a drop in multilingual capabilities; future work should incorporate the 15 million multilingual samples.
2. **BLIP capacity limitation**: BLIP's 512-token limit and smaller model capacity result in sub-optimal performance in complex reasoning tasks, and it even loses prior knowledge in some tasks after fine-tuning.
3. **Small evaluation scale for Tasks 4 and 5**: Visually unanswerable questions (~500 pairs) and multilingual evaluations (~500 images) are small due to human annotation cost constraints.
4. **Unexplored newer model architectures**: Only BLIP and LLaVA-7B were fine-tuned; larger or newer models (e.g., Gemini, Qwen2-VL) were not tested.
5. **Limited to Q&A tasks**: Richer application scenarios such as image captioning, retrieval, or interactive dialogue have not been explored.

## Related Work & Insights

- Difference from MUZE: MUZE uses CLIP's multimodal representations and independent attention heads to handle each attribute, which is computationally expensive and unsuitable for direct QA; Museum-65 directly fine-tunes end-to-end VLMs.
- Difference from VISCOUNTH: VISCOUNTH only covers painting and sculpture (500K images), whereas Museum-65 encompasses art, history, and natural sciences.
- Insights: This work validates an important paradigm—in specialized domains, constructing large-scale high-quality datasets and fine-tuning general VLMs is currently the most effective approach.

## Rating

- **Novelty**: ⭐⭐⭐ — The methodology is relatively straightforward (standard fine-tuning); the core contributions lie in the dataset and the evaluation framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The 5 tasks comprehensively cover real scenarios and include comparisons with human experts, though the evaluation scale of certain tasks is quite small.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with detailed explanations of the dataset construction process.
- **Value**: ⭐⭐⭐⭐ — Museum-65 holds long-term value as a domain-specific dataset, serving as crucial infrastructure for AI research in cultural heritage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation](perspective-aware_reasoning_in_vision-language_models_via_mental_imagery_simulat.md)
- [\[ICCV 2025\] LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](llava-cot_let_vision_language_models_reason_step-by-step.md)
- [\[ACL 2025\] Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning](../../ACL2025/vlm_reasoning/benchmarking_and_improving_large_vision-language_models_for_fundamental_visual_g.md)
- [\[ACL 2026\] ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding](../../ACL2026/vlm_reasoning/chemvlr_prioritizing_reasoning_in_perception_for_chemical_vision-language_unders.md)
- [\[CVPR 2026\] Learning to Reason in 4D: Dynamic Spatial Understanding for Vision Language Models](../../CVPR2026/vlm_reasoning/learning_to_reason_in_4d_dynamic_spatial_understanding_for_vision_language_model.md)

</div>

<!-- RELATED:END -->
