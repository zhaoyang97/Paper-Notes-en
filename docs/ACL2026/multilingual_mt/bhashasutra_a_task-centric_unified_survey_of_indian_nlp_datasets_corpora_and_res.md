---
title: >-
  [Paper Note] BhashaSutra: A Task-Centric Unified Survey of Indian NLP Datasets, Corpora, and Resources
description: >-
  [ACL 2026][Multilingual & Translation][Paper Note] The first unified survey specifically targeting Indian NLP resources, covering 200+ datasets, 50+ benchmarks, and 100+ models/tools, organized by 17 task categories (from core language processing to sociocultural tasks), systematically analyzing persistent challenges such as uneven language coverage, fragmented annotat
tags:
  - ACL 2026
  - Multilingual & Translation
date: 2026-05-08
content_hash: 86151fb0c7350bc6
---
# BhashaSutra: A Task-Centric Unified Survey of Indian NLP Datasets, Corpora, and Resources

**Conference**: ACL 2026  
**arXiv**: [2604.18423](https://arxiv.org/abs/2604.18423)  
**Code**: None  
**Area**: Multilingual Translation  
**Keywords**: Indian NLP, Dataset Survey, Multilingual Resources, Low-resource Languages, Cultural NLP

## TL;DR

The first unified survey specifically targeting Indian NLP resources, covering 200+ datasets, 50+ benchmarks, and 100+ models/tools, organized by 17 task categories (from core language processing to sociocultural tasks), systematically analyzing persistent challenges such as uneven language coverage, fragmented annotation, and inconsistent evaluation.

## Background & Motivation

**Background**: India possesses one of the world's most diverse linguistic ecosystems (22 official languages, hundreds of dialects). In recent years, Indian NLP has developed rapidly, with datasets, benchmarks, and pre-trained models emerging in various fields such as healthcare, law, education, and governance.

**Limitations of Prior Work**: Progress is severely fragmented—most work focuses on a few relatively resource-rich languages, with significant differences in quality and documentation across various publication venues. Existing surveys are either narrow in scope (targeting only specific task families) or submerge Indian languages within broader multilingual settings, lacking a dedicated, full-task survey for Indian NLP.

**Key Challenge**: There is a massive gap between the resource growth rate of Indian NLP and its systematic organization, making it difficult for researchers to understand the global landscape and real resource gaps.

**Goal**: To provide the first unified, task-centric survey of Indian NLP resources, covering text, speech, multimodal, and cultural tasks.

**Key Insight**: Organizing resources by task category (rather than language) allows researchers to quickly locate available resources for specific task directions.

**Core Idea**: Build a classification system of seventeen sub-tasks across six major categories, systematically organizing 200+ datasets and analyzing resource coverage patterns and gaps.

## Method

### Overall Architecture

The review organizes Indian NLP into six major categories: (1) Core Language Processing (Tokenization, POS tagging, NER); (2) Text Classification and Semantics (Sentiment analysis, hate speech detection, topic classification, NLU); (3) Generation and Translation (Summarization, Machine Translation, Question Answering); (4) Retrieval and Interaction (Information retrieval, dialogue systems); (5) Speech and Multimodal; (6) Sociocultural and Emerging Tasks (Disinformation detection, cultural reasoning, etc.).

### Key Designs

**1. Task-centric Taxonomy: Changing the axis to re-stitch fragmented resources**

Indian NLP resources are scattered across different languages, publication venues, and quality standards, making it difficult for researchers to see the big picture. A natural but inefficient approach is to archive by language, but with 22 official languages and hundreds of dialects, splitting by language leads to redundant repetition of resources for the same task. This paper instead organizes by 17 NLP tasks: each task summarizes key datasets, benchmarks, and tools, including both monolingual resources for specific languages and multilingual resources that include English. This choice aligns with real-world usage scenarios—researchers usually seek resources based on "the need to perform a certain task" rather than selecting a language first; thus, a task-centric index allows direct location of available assets in one's field.

**2. Resource Coverage Analysis and Visualization: Turning "imbalance" from an impression into a quantifiable map**

Merely stating that Indian language resources are unevenly distributed remains at the impressionistic level. This paper lays it out using language-dimension resource count visualization (Figure 2): languages like Hindi, Bengali, and Tamil have the richest resources, while Northeast languages (Assamese, Manipuri, etc.) and endangered dialects are severely deficient, with multilingual resources unified under the "Indic Languages" label. The value of this map lies in quantifying the degree of imbalance, allowing the community to see exactly where the most urgent gaps need to be filled, rather than vaguely stating that "low-resource languages lack data."

**3. Cross-task Gap and Cultural Challenge Analysis: Viewing structural issues from an ecosystem perspective instead of a single-task view**

Listing resources task-by-task can lead to missing the forest for the trees. This paper further extracts common problems spanning multiple tasks: language imbalance, fragmented annotation, domain bias, inconsistent evaluation, and cross-lingual fragility. Beyond this, it specifically identifies sociocultural challenges unique to the Indian context—bias assessment, code-mixing (such as Hinglish), disinformation, and risks to cultural fidelity in translation pipelines. This layer of analysis transcends local judgments of "what data a certain task lacks" and reveals structural failures where the entire Indian NLP ecosystem repeatedly stumbles, pointing toward systematic directions for future resource construction.

### Loss & Training

As a survey paper, this does not involve technical implementation.

## Key Experimental Results

### Main Results

Overview of Resource Statistics:

| Category | Quantity |
|------|------|
| Datasets | 200+ |
| Benchmarks | 50+ |
| Models/Tools/Systems | 100+ |
| Covered Tasks | 17 |
| Covered Modalities | Text + Speech + Multimodal |

### Ablation Study

Language Coverage Analysis:

| Resource Tier | Representative Languages | Characteristics |
|---------|---------|------|
| High Resource | Hindi, Bengali, Tamil | Full coverage across multiple tasks |
| Mid Resource | Telugu, Marathi, Kannada | Coverage for core tasks |
| Low Resource | Assamese, Odia, Manipuri | Data only for selected tasks |
| Very Low Resource | Bhojpuri, Maithili, Santhali | Almost no dedicated resources |

### Key Findings

- Hindi has the richest resources but still possesses task coverage gaps; Northeast languages and endangered dialects have extremely scarce NLP resources.
- Code-mixing (e.g., Hinglish) is a unique challenge for Indian NLP, prevalent in tasks like sentiment analysis and hate speech detection.
- A large number of Indian language resources rely on translation pipelines; while this can quickly scale, it may fail to capture indigenous linguistic, pragmatic, and sociocultural nuances.
- Evaluation practices are severely inconsistent: standards for reporting train/dev/test splits, metric definitions, and annotation processes are not unified.

## Highlights & Insights

- This is currently the most comprehensive map of Indian NLP resources, serving as a must-read reference for anyone wishing to conduct NLP research on Indian languages. Its organization by task makes it highly practical.
- The discussion on the trade-off between "translation pipeline construction vs. native data collection" touches on the core issue of low-resource NLP: scalability and cultural fidelity are often difficult to achieve simultaneously.
- Emphasizing sociocultural tasks (bias, disinformation, cultural reasoning) as independent categories reflects the shift of NLP research toward social impact.

## Limitations & Future Work

- The survey cannot fully keep up with the rapidly developing ecosystem, as new datasets and models continue to emerge.
- Synthesis is based on published literature rather than experimental reproduction; some industrial or poorly documented datasets may be missed.
- No explicit ranking of resource quality was conducted, as evaluation standards vary significantly across different tasks and modalities.
- Insufficient discussion on efficiency and hardware constraints—the accessibility of large models in low-resource environments is a major challenge.

## Related Work & Insights

- **vs Kakwani et al. (2020) / IndicNLP**: Narrower coverage, focusing primarily on basic NLP tools; this paper covers 17 tasks and includes cultural and social dimensions.
- **vs General Multilingual NLP Surveys**: These treat Indian languages as a small part of a broad multilingual setting and cannot fully reflect challenges unique to India (such as code-mixing, script diversity, and caste bias).
- **vs AI4Bharat**: AI4Bharat is an important resource construction project but not a survey; this paper systematically organizes all available resources, including those from AI4Bharat.

## Rating
- Novelty: ⭐⭐⭐⭐ First dedicated unified survey of Indian NLP resources
- Experimental Thoroughness: ⭐⭐⭐ Survey with no experiments, but extremely comprehensive resource coverage
- Writing Quality: ⭐⭐⭐⭐ Clear organization, practical classification system
- Value: ⭐⭐⭐⭐⭐ Vital reference value for the Indian NLP community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP](scripts_through_time_a_survey_of_the_evolving_role_of_transliteration_in_nlp.md)
- [\[ACL 2026\] Why Low-Resource NLP Needs More Than Cross-Lingual Transfer: Lessons Learned from Luxembourgish](why_low-resource_nlp_needs_more_than_cross-lingual_transfer_lessons_learned_from.md)
- [\[ACL 2026\] Lingo_Research_Group at SemEval-2026 Task 9: Evaluating Prompt Variants for Polarization Detection](lingo_research_group_at_semeval-2026_task_9_evaluating_prompt_variants_for_polar.md)
- [\[ACL 2025\] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](../../ACL2025/multilingual_mt/mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)
- [\[ACL 2026\] DFKI-MLT at SemEval-2026 TASK 7: Steering Multilingual Models Towards Cultural Knowledge](dfki-mlt_at_semeval-2026_task_7_steering_multilingual_models_towards_cultural_kn.md)

</div>

<!-- RELATED:END -->
