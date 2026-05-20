---
title: >-
  [Paper Note] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering
description: >-
  [ICCV 2025][Multimodal VLM][VQA] This paper proposes ReasonVQA, a dataset constructed through a low-cost and scalable framework that automatically integrates structured encyclopedic knowledge (Wikidata) with images…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "VQA"
  - "Multi-hop Reasoning"
  - "Knowledge Graph"
  - "Wikidata"
  - "Benchmark Dataset"
date: 2026-05-08
content_hash: 815cd16dd92aaeb1
---

# ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering

**Conference**: ICCV 2025
**arXiv**: [2507.16403](https://arxiv.org/abs/2507.16403)  
**Code**: [ReasonVQA](https://duong-tr.github.io/ReasonVQA)  
**Area**: Multimodal VLM
**Keywords**: VQA, Multi-hop Reasoning, Knowledge Graph, Wikidata, Benchmark Dataset

## TL;DR

This paper proposes ReasonVQA, a dataset constructed through a low-cost and scalable framework that automatically integrates structured encyclopedic knowledge (Wikidata) with images, generating 1/2/3-hop multi-hop reasoning questions. The benchmark comprises 598K images and 4.2M questions, posing significant challenges to existing VQA models.

## Background & Motivation

Despite notable advances in visual question answering (VQA), existing datasets exhibit clear shortcomings:

**Lack of external knowledge integration**: Standard VQA datasets (VQAv2, GQA) primarily focus on content explicitly visible in images, such as object recognition and attribute judgment, whereas real-world questions often require knowledge beyond the image itself.

**Absence of multi-hop reasoning**: Questions such as "What is the capital of the country where this church is located?" require multi-step reasoning—first identifying the church, then retrieving its country, and finally querying that country's capital.

**High construction cost**: Existing datasets requiring external knowledge (e.g., OK-VQA, KVQA) rely heavily on manual annotation and scale poorly.

**Limited scale**: Encyclopedic-VQA contains approximately 1M questions and INFOSEEK approximately 1.35M, both of which remain insufficient.

The authors' core motivation is: **Can a large-scale, high-quality multi-hop reasoning VQA benchmark be constructed automatically and at low cost?** This is achieved by leveraging existing CV dataset annotations together with the structured knowledge base Wikidata for automatic question generation.

## Method

### Overall Architecture

The construction of ReasonVQA proceeds in three steps:
1. **External Knowledge Integration**
2. **Question Generation**
3. **Dataset Construction**

### External Knowledge Integration

**Wikidata** is selected as the external knowledge source, with **SPARQL** used for semantic querying. Images are drawn from two sources:

- **Visual Genome (VG)**: 108K+ images with rich scene graph annotations. VG object annotations are normalized via WordNet synsets and linked to Wikidata entities using NLTK and SPARQL queries.
- **Google Landmarks v2 (GLDv2)**: 5M+ images covering 200K landmarks, annotated with Wikimedia URLs from which Wikidata knowledge is directly extracted.

### Template-Based Question Generation

**One-hop questions**: Fill-in-the-blank templates are designed for each Wikidata property. For example, the property "architect" corresponds to the template "Who designed __?", with the placeholder filled by the object class name (e.g., "skyscraper"), yielding "Who designed this skyscraper?"

**Multi-hop questions**: Achieved through nested clause templates. For instance, the clause template "the architect of __" for the property "architect" can be recursively nested to generate 2-hop and 3-hop questions. Scene graph annotations from VG are additionally used to construct clauses (e.g., "__ parked next to the sidewalk"), incorporating visual-semantic information into the questions.

**Domain labels**: Twenty knowledge domains are predefined, with automatic categorization based on properties. A single question may belong to multiple domains.

### Distractor Generation

Challenging distractors are generated for multiple-choice questions, categorized into four types based on answer type:
- **Fixed**: Closed-set answers (e.g., gender, continent) sampled randomly.
- **Date**: $N$ dates sampled randomly within $\pm 10$ years of the correct answer.
- **Number**: Values sampled randomly within $[\frac{i}{2}, \max(1.5i, \frac{i}{2}+2N)]$.
- **Literal**: $N$ values retrieved from other entities sharing the same property.

### Answer Distribution Balancing

Following the approach of GQA, answers are grouped by property and sorted by frequency. High-frequency answers are iteratively removed to smooth the distribution, making the sizes of the head and tail comparable and reducing the possibility of models exploiting the most frequent answers as shortcuts.

### Evaluation Metrics

Three string matching methods are employed:
- **Exact Match**: Full string equivalence.
- **Substring**: Containment check.
- **Semantic Similarity**: Computed using the all-MiniLM-L6-v2 model.

## Key Experimental Results

### Dataset Scale Comparison

| Dataset | # Images | # Questions | Knowledge Base |
|---------|----------|-------------|----------------|
| OK-VQA | 14K | 14K | ✓ |
| KVQA | 24K | 183K | ✓ |
| CRIC | 96K | 494K | ✓ |
| InfoSeek | 8.9K | 1.35M | ✓ |
| Encyclopedic VQA | 514K | 1M | ✓ |
| **ReasonVQA Full** | **598.5K** | **4.2M** | ✓ |

A user study indicates that 96% of answers are correct and over 83% of questions are rated as natural.

### Zero-Shot Model Evaluation (Semantic Similarity Score)

| Model | ReasonVQA-U | ReasonVQA-B | OK-VQA |
|-------|-------------|-------------|--------|
| BLIP-2 | 46.4 | 46.1 | 45.9 |
| mPLUG-Owl2 | 22.1 | 22.2 | 57.7 |
| GPT-4o | **62.8** | **60.8** | 71.8 |
| Qwen2.5-VL | 59.3 | 58.1 | 84.9 |
| PaliGemma-2-Mix | 43.7 | 41.8 | **86.8** |

**Key Findings**: All models perform substantially lower on ReasonVQA than on standard VQA datasets. PaliGemma-2-Mix achieves 86.8% on OK-VQA but only 41.8% on ReasonVQA-B, a striking gap.

### Multiple-Choice Setting

| Model | ReasonVQA-U | ReasonVQA-B | OK-VQA |
|-------|-------------|-------------|--------|
| GPT-4o | 76.6 | 73.4 | 96.7 |
| mPLUG-Owl3 | 68.9 | 68.1 | 99.1 |
| Mantis-Idefics2 | 68.7 | 68.5 | 98.9 |

Although the multiple-choice format substantially improves accuracy, performance on ReasonVQA remains 20–30 percentage points lower than on other datasets.

### Fine-Tuning Experiments

| Model | Zero-shot | Fine-tuned | Gain |
|-------|-----------|------------|------|
| Qwen2-VL-7B | 59.0 | 65.0 | +10.1% |
| PaliGemma-2-3B-Mix | 40.1 | 66.8 | **+66.5%** |
| PaliGemma-2-10B-Mix | 65.6 | 74.5 | +13.5% |

**Key Findings**: The smaller model (PaliGemma-2-3B) achieves the largest improvement through fine-tuning (+66.5%), suggesting that smaller models have greater room for gains when adapted to a specific dataset.

### Analysis by Hop Count and Scene Graph

- **3-hop questions** yield significantly lower accuracy than 1-hop and 2-hop questions, confirming the high complexity of multi-hop reasoning.
- 2-hop questions occasionally outperform 1-hop questions, as longer questions provide richer contextual information.
- **Questions incorporating scene graph information** receive lower scores, demonstrating that the integration of scene graphs genuinely increases dataset difficulty.

## Highlights & Insights

1. **Large-scale construction at minimal cost**: By combining existing CV annotations, Wikidata, and template-based generation, the framework avoids extensive manual annotation and achieves a scale of 4.2M questions.
2. **Scalable design**: The framework can be readily extended to new image and knowledge sources, and users can customize the dataset through domain filtering.
3. **Systematic evaluation of multi-hop reasoning**: The 1/2/3-hop hierarchical design precisely measures models' chain-of-reasoning capabilities.
4. **Revealing model weaknesses**: Even the strongest model, GPT-4o, achieves only 62.8% on ReasonVQA in the open-ended setting, indicating that current VLMs still have substantial room for improvement in knowledge integration and multi-hop reasoning.

## Limitations & Future Work

1. The template-based approach has limitations in linguistic diversity; approximately 14% of questions are rated as "unnatural."
2. The dataset relies primarily on Wikidata, so knowledge coverage is constrained by its content.
3. The distractor generation strategy is relatively straightforward and may introduce shortcuts exploitable through elimination.
4. The answer distribution balancing process may discard some valuable samples.

## Related Work & Insights

- **Knowledge-enhanced VQA**: OK-VQA (2019) first required external knowledge; KVQA (2019) supports multi-hop reasoning but depends on manual annotation.
- **Automated construction**: CRIC (2022) employs ConceptNet commonsense knowledge; LORA is restricted to the food domain.
- **Foundation model evaluation**: INFOSEEK and Encyclopedic-VQA reveal the cost bottleneck of manual annotation.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | 7.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](../../NeurIPS2025/multimodal_vlm/wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [\[NeurIPS 2025\] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering](../../NeurIPS2025/multimodal_vlm/are_vision_language_models_ready_for_clinical_diagnosis_a_3d_medical_benchmark_f.md)
- [\[ACL 2026\] WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering](../../ACL2026/multimodal_vlm/wikiseeker_rethinking_the_role_of_vision-language_models_in_knowledge-based_visu.md)
- [\[ICCV 2025\] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)

</div>

<!-- RELATED:END -->
