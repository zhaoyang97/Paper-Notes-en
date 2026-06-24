---
title: >-
  [Paper Note] Pragmatics in the Era of Large Language Models: A Survey on Datasets, Evaluation, Opportunities and Challenges
description: >-
  [ACL 2025][LLM (Other)][Pragmatics] A systematic survey of resources in 58 papers evaluating the pragmatic abilities of NLP models. It categorizes them by pragmatic phenomena (context/deixis, implicature/presupposition, speech acts, discourse coherence, social pragmatics), categorizes task designs (MCQ/QA/NLI/reference games, etc.) and data construction methods (bottom-up/top-down), and reveals core gaps in current evaluations (English-centric bias, unimodal limitations…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Pragmatics"
  - "LLM Evaluation"
  - "Implicature"
  - "Speech Acts"
  - "Discourse Coherence"
  - "Dataset Survey"
date: 2026-05-08
content_hash: 02fab52642ae4a96
---

# Pragmatics in the Era of Large Language Models: A Survey on Datasets, Evaluation, Opportunities and Challenges

**Conference**: ACL 2025  
**arXiv**: [2502.12378](https://arxiv.org/abs/2502.12378)  
**Area**: Natural Language Processing / Pragmatics  
**Keywords**: Pragmatics, LLM Evaluation, Implicature, Speech Acts, Discourse Coherence, Dataset Survey

## TL;DR

A systematic survey of resources in 58 papers evaluating the pragmatic abilities of NLP models. It categorizes them by pragmatic phenomena (context/deixis, implicature/presupposition, speech acts, discourse coherence, social pragmatics), categorizes task designs (MCQ/QA/NLI/reference games, etc.) and data construction methods (bottom-up/top-down), and reveals core gaps in current evaluations (English-centric bias, unimodal limitations, lack of fine-grained evaluation), providing a roadmap for pragmatic evaluation in the LLM era.

## Background & Motivation

**Background**: Pragmatics studies how language is used in context and is a core branch of linguistics. Although NLP models have evolved from rule-based systems to statistical models and Transformers, understanding non-literal meanings (implicatures, sarcasm, indirect requests, etc.) remains a challenge.

**Limitations of Prior Work**: (1) LLMs demonstrate strong text generation capabilities, but their pragmatic reasoning abilities (e.g., understanding conversational implicature, context-dependent anaphora) are insufficiently evaluated; (2) Existing evaluation resources are scattered, lacking a unified framework to integrate datasets of different pragmatic phenomena; (3) As LLMs are increasingly deployed in real-world scenarios, validating their pragmatic abilities is crucial for trustworthy human-computer interaction.

**Goal** Three core questions: (a) What resources are available to evaluate the pragmatic abilities of NLP models? (b) How can pragmatics guide the improvement of LLMs? (c) How can LLMs, in turn, facilitate the study of pragmatics in linguistics?

**Key Insight**: A comprehensive systematization across four dimensions: pragmatic phenomenon classification, task types, data construction methods, and evaluation metrics.

**Core Idea**: Establish a multi-level mapping of pragmatic phenomena-NLP tasks-datasets-evaluation methods to identify key gaps and propose future directions.

## Method

### Overall Architecture

The survey is organized around five major pragmatic phenomena, identifying corresponding NLP tasks, datasets, and evaluation methods under each phenomenon:

1. **Context & Deixis**: Evaluates the model's ability to interpret input based on situational or linguistic context.
2. **Implicature & Presupposition**: Tests the model's ability to reason beyond literal meaning.
3. **Speech Acts & Intent**: Examines the model's capability to identify communicative intentions such as requests, commands, and promises.
4. **Discourse & Coherence**: Analyzes the model's comprehension of discourse structure and coherence relations.
5. **Social Pragmatics**: Explores the influence of social norms, power relations, and cultural factors on language use.

### Key Designs

1. **Task Type Taxonomy**
    - **Function**: Categorizes existing tasks into 7 classes: MCQ, QA, NLI, sentiment analysis, image description, reference games, and others.
    - **Key Findings**: MCQ and QA are the most commonly used evaluation formats; reference games represent a unique pragmatic evaluation paradigm (where a speaker describes a target object and a listener identifies it) that naturally tests context-dependent communication; NLI is frequently used for implicature testing (e.g., premise-hypothesis pairs for scalar implicature).
    - **Mappings**: There is no strict one-to-one mapping between pragmatic phenomena and task types; a single phenomenon can be evaluated through various tasks.

2. **Data Construction Methodology**
    - **Function**: Summarizes two paradigms: bottom-up (collecting data first, then annotating) and top-down (defining linguistic labels first, then expanding data).
    - **Bottom-up**: Data sources include databases (web pages/interviews), manual collection (e.g., reference games), and existing datasets; annotation methods include crowdsourcing, experts, and LLM-assisted annotation.
    - **Top-down**: Driven by linguistic theories, e.g., generating NLI samples using GPT-4 starting from scalar pairs `<some, all>`.
    - **Key Insights**: Hybrid approaches (LLM generation + human verification) are a promising direction, but direct generation of implicature reasoning by LLMs remains unreliable.

3. **Gap Analysis and Future Directions**
    - **Function**: Identifies four core gaps and proposes solution roadmaps.
    - **English-centric Bias**: Only 19% of the 58 papers involve non-English resources.
    - **Unimodal Data**: Most datasets are text-only or speech-only, lacking multimodal information such as vision/gestures.
    - **Task Design Limitations**: Tasks typically evaluate only a single pragmatic phenomenon, failing to test the holistic pragmatic ability of models.
    - **Inadequate Evaluation Metrics**: Automatic metrics (F1/BLEU/ROUGE) struggle to capture pragmatic nuances, requiring integration of human evaluation and psychometric methods.

## Key Experimental Results

### Pragmatic Phenomena Coverage Distribution (Statistics from 58 Papers)

| Pragmatic Phenomenon | Number of Papers | Main Task Types | Representative Datasets |
|---------|:--------:|------------|-----------|
| Context & Deixis | 17 | QA, MCQ, Reference Games | AmbigQA, DIPLOMAT, GuessWhat |
| Implicature & Presupposition | 18 | NLI, QA, MCQ | IMPPRES, PragmatiCQA, GRICE |
| Speech Acts | 11 | QA, Reference Games | DIPLOMAT, Codenames, STAC |
| Discourse & Coherence | 13 | Discourse Relation Parsing, QA | PDTB, TED-Q, GCDC |
| Social Pragmatics | 8 | Sentiment Analysis, QA | Social IQa, SBF, EmoBank |

### Evaluation Methods Usage

| Evaluation Method | Usage Proportion | Applicable Scenarios | Limitations |
|---------|:------:|--------|-----|
| Automatic Metrics (F1/Acc) | ~70% | Classification/Selection tasks | Difficult to capture pragmatic nuances |
| Generative Metrics (ROUGE/BLEU) | ~20% | Generation tasks | Cannot evaluate pragmatic appropriateness |
| Human Evaluation | ~15% | Open-ended tasks | High cost, hard to scale |
| Hybrid Evaluation | <5% | Comprehensive evaluation | Still lacks standardized procedures |

## Highlights & Insights

1. **First systematic mapping between pragmatic phenomena and NLP tasks**: Clearly shows which phenomena are adequately evaluated and where gaps exist.
2. **Pragmatics can feedback into LLM alignment**: Speech act theory emphasizes that meaning arises from interaction rather than isolated sentences, which is highly consistent with the communicative goals of LLMs; incorporating pragmatic constraints can improve instruction-following and ambiguity resolution in LLMs.
3. **LLMs can facilitate experimental pragmatics research**: LLMs can assist in designing experimental stimuli, data pre-annotation, and hypothesis generation, though human verification is required to guarantee quality.
4. **Multi-agent systems require pragmatic reasoning**: When multiple LLM agents interact, communication quality degrades due to the lack of true belief states reasoning; pragmatic reasoning ability is key.

## Limitations & Future Work

1. The scope of the survey is primarily limited to the ACL Anthology, potentially missing relevant resources from other fields (e.g., cognitive science, psychology).
2. Multimodal pragmatic evaluation is almost blank—lacking datasets that combine speech intonation, facial expressions, and gestures.
3. Cross-lingual/cross-cultural perspectives are severely lacking, limiting the generalizability of the findings.
4. Intrinsic pragmatic tasks (e.g., discourse modeling) are not included in the discussion.
5. No unified pragmatic competency evaluation framework or benchmark is provided, only a survey-level summary.

## Related Work & Insights

- Pragmatic evaluation can draw from psychometrics (e.g., Likert scales, analysis of human annotation variability) to design finer-grained metrics.
- Reference Games are a unique paradigm for pragmatic evaluation and warrant wider adoption in more LLM evaluations.
- The top-down data construction method (linguistic theory-driven + LLM generation + human verification) may be the best path for scaling up pragmatic datasets.
- The impact of demographic factors on annotation is overlooked; future work should consider annotator diversity in both the collection and evaluation phases.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐: Survey in nature; the contribution lies in systematic organization rather than methodological innovation.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: Comprehensive coverage of 58 papers, five major pragmatic phenomena, and seven task types.
- **Writing Quality** ⭐⭐⭐⭐: Clear classification, rich tables, and pragmatic future directions.
- **Value** ⭐⭐⭐⭐: Provides an important resource map and research roadmap for LLM pragmatic evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Personalized Generation In Large Model Era: A Survey](personalized_generation_in_large_model_era_a_survey.md)
- [\[ACL 2025\] Large Language Models in Bioinformatics: A Survey](large_language_models_in_bioinformatics_a_survey.md)
- [\[ACL 2025\] Knowledge Boundary of Large Language Models: A Survey](knowledge_boundary_survey.md)
- [\[ACL 2025\] Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition](sample-efficient_human_evaluation_of_large_language_models_via_maximum_discrepan.md)
- [\[ACL 2025\] Steering off Course: Reliability Challenges in Steering Language Models](steering_off_course_reliability_challenges_in_steering_language_models.md)

</div>

<!-- RELATED:END -->
