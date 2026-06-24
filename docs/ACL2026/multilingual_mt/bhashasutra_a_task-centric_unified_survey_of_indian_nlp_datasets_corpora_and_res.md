---
title: >-
  [Paper Note] BhashaSutra: A Task-Centric Unified Survey of Indian NLP Datasets, Corpora, and Resources
description: >-
  [ACL 2026][Multilingual & Machine Translation][Indic NLP] The first unified survey specifically targeting Indic NLP resources, covering 200+ datasets, 50+ benchmarks, and 100+ models/tools. Organized by 17 task categories (from core language processing to socio-cultural tasks), it systematically analyzes persistent challenges such as uneven linguistic coverage, fragmented annotation, and inconsistent evaluation.
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Indic NLP"
  - "Dataset Survey"
  - "Multilingual Resources"
  - "Low-resource Languages"
  - "Cultural NLP"
date: 2026-05-08
content_hash: 232ad43d7acd4826
---

# BhashaSutra: A Task-Centric Unified Survey of Indian NLP Datasets, Corpora, and Resources

**Conference**: ACL 2026  
**arXiv**: [2604.18423](https://arxiv.org/abs/2604.18423)  
**Code**: None  
**Area**: Multilingual Translation  
**Keywords**: Indic NLP, Dataset Survey, Multilingual Resources, Low-resource Languages, Cultural NLP

## TL;DR

The first unified survey specifically targeting Indic NLP resources, covering 200+ datasets, 50+ benchmarks, and 100+ models/tools. Organized by 17 task categories (from core language processing to socio-cultural tasks), it systematically analyzes persistent challenges such as uneven linguistic coverage, fragmented annotation, and inconsistent evaluation.

## Background & Motivation

**Background**: India possesses one of the world's most diverse linguistic ecosystems (22 official languages and hundreds of dialects). Recently, Indic NLP has seen rapid development, with datasets, benchmarks, and pre-trained models emerging in domains like healthcare, law, education, and governance.

**Limitations of Prior Work**: Progress is severely fragmented—most work focuses on a few relatively high-resource languages, while quality and documentation vary significantly across different publication venues. Existing surveys are either narrow in scope (targeting specific task families) or submerge Indic languages within broader multilingual settings, lacking a dedicated, all-task Indic NLP overview.

**Key Challenge**: There is a massive gap between the growth rate of Indic NLP resources and their systematic organization, making it difficult for researchers to grasp the overall landscape and identify genuine resource gaps.

**Goal**: To provide the first unified, task-centric survey of Indic NLP resources, encompassing text, speech, multimodal, and cultural tasks.

**Key Insight**: Resources are organized by task category rather than language, enabling researchers to quickly locate available resources for specific task directions.

**Core Idea**: Construct a taxonomy of six major categories and seventeen sub-tasks to systematically organize 200+ datasets and analyze resource coverage patterns and gaps.

## Method

### Overall Architecture

The survey organizes Indic NLP into six major categories: (1) Core Language Processing (Tokenization, POS tagging, NER); (2) Text Classification and Semantics (Sentiment analysis, Hate speech detection, Topic classification, NLU); (3) Generation and Translation (Summarization, Machine Translation, QA); (4) Retrieval and Interaction (Information Retrieval, Dialogue systems); (5) Speech and Multimodality; (6) Socio-cultural and Emerging Tasks (Misinformation detection, Cultural reasoning, etc.).

### Key Designs

**1. Task-Centric Taxonomy: Changing the axis to reconnect fragmented resources**

Indic NLP resources are scattered across different languages, venues, and quality standards, making it hard to see the big picture. While archiving by language is natural, it is inefficient given India's 22 official languages; task-based organization prevents redundant entries. This survey organizes resources by 17 NLP tasks, aggregating key datasets, benchmarks, and tools. This approach aligns with real-world usage—researchers typically search for resources based on a task requirement rather than starting with a language selection.

**2. Resource Coverage Analysis and Visualization: Quantifying imbalance into a visual map**

Rather than merely stating that resource distribution is uneven, this paper employs language-dimension resource counts (Figure 2) to visualize the disparity. Languages like Hindi, Bengali, and Tamil are the most resource-rich, while Northeastern languages (Assamese, Manipuri, etc.) and endangered dialects are severely lacking. Multi-language resources are unified under the "Indic Languages" label. This visualization quantifies the degree of imbalance, allowing the community to identify the most urgent gaps.

**3. Cross-task Gaps and Cultural Challenge Analysis: Moving from single-task to ecosystem-level structural diagnosis**

Beyond listing resources, the paper extracts common issues across tasks: language imbalance, fragmented annotation, domain skew, inconsistent evaluation, and cross-lingual vulnerability. It specifically highlights socio-cultural challenges unique to the Indian context—bias assessment, code-mixing (e.g., Hinglish), misinformation, and the risks of cultural fidelity in translation pipelines. This analysis moves beyond "what data is missing for a task" to reveal structural pitfalls in the Indic NLP ecosystem.

### Loss & Training

As a survey paper, this does not involve technical implementation.

## Key Experimental Results

### Main Results

Resource Statistics Overview:

| Category | Quantity |
|------|------|
| Datasets | 200+ |
| Benchmarks | 50+ |
| Models/Tools/Systems | 100+ |
| Tasks Covered | 17 |
| Modalities Covered | Text + Speech + Multimodal |

### Ablation Study

Language Coverage Analysis:

| Resource Level | Representative Languages | Features |
|---------|---------|------|
| High-Resource | Hindi, Bengali, Tamil | Comprehensive multi-task coverage |
| Mid-Resource | Telugu, Marathi, Kannada | Coverage for core tasks |
| Low-Resource | Assamese, Odia, Manipuri | Data only for select tasks |
| Extremely Low-Resource | Bhojpuri, Maithili, Santhali | Almost no dedicated resources |

### Key Findings

- While Hindi is the most resource-rich, task coverage gaps remain. Resources for Northeastern languages and endangered dialects are extremely scarce.
- Code-mixing (e.g., Hinglish) is a unique challenge for Indic NLP, prevalent in tasks like sentiment analysis and hate speech detection.
- Many Indic resources rely on translation pipelines; while this scales quickly, it may fail to capture native linguistic, pragmatic, and socio-cultural nuances.
- Evaluation practices are highly inconsistent: reporting standards for train/dev/test splits, metric definitions, and annotation workflows are not unified.

## Highlights & Insights

- This is currently the most comprehensive map of Indic NLP resources, serving as a mandatory reference for anyone conducting NLP research on Indian languages. The task-centric organization enhances practicality.
- The discussion on the trade-off between "translation pipeline construction vs. native data collection" addresses a core issue in low-resource NLP: scalability and cultural fidelity are often difficult to achieve simultaneously.
- Emphasizing socio-cultural tasks (bias, misinformation, cultural reasoning) as independent categories reflects the shift in NLP research towards social impact.

## Limitations & Future Work

- The survey cannot fully keep pace with the rapidly evolving ecosystem as new datasets and models emerge continuously.
- Based on published literature synthesis rather than experimental replication; datasets with poor documentation or from industry may be overlooked.
- No explicit quality ranking was performed due to vast differences in evaluation standards across tasks and modalities.
- Insufficient discussion on efficiency and hardware constraints—the accessibility of large models in low-resource environments remains a significant challenge.

## Related Work & Insights

- **vs Kakwani et al. (2020) / IndicNLP**: Prior work was narrower, focusing on basic NLP tools; this paper covers 17 tasks and includes cultural/social dimensions.
- **vs General Multilingual NLP Surveys**: Generic surveys treat Indic languages as a small subset of a broad setting, failing to reflect specific challenges like code-mixing, script diversity, and caste bias.
- **vs AI4Bharat**: AI4Bharat is a resource construction project rather than a survey; this paper systematically organizes all available resources, including those from AI4Bharat.

## Rating
- Novelty: ⭐⭐⭐⭐ First dedicated unified survey of Indic NLP resources.
- Experimental Thoroughness: ⭐⭐⭐ Survey-based, but resource coverage is extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear organization with a practical taxonomy.
- Value: ⭐⭐⭐⭐⭐ Vital reference for the Indic NLP community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP](scripts_through_time_a_survey_of_the_evolving_role_of_transliteration_in_nlp.md)
- [\[ACL 2026\] Why Low-Resource NLP Needs More Than Cross-Lingual Transfer: Lessons Learned from Luxembourgish](why_low-resource_nlp_needs_more_than_cross-lingual_transfer_lessons_learned_from.md)
- [\[ACL 2026\] Lingo_Research_Group at SemEval-2026 Task 9: Evaluating Prompt Variants for Polarization Detection](lingo_research_group_at_semeval-2026_task_9_evaluating_prompt_variants_for_polar.md)
- [\[ACL 2026\] DFKI-MLT at SemEval-2026 TASK 7: Steering Multilingual Models Towards Cultural Knowledge](dfki-mlt_at_semeval-2026_task_7_steering_multilingual_models_towards_cultural_kn.md)
- [\[ACL 2025\] Building Better: Avoiding Pitfalls in Developing Language Resources when Data is Scarce](../../ACL2025/multilingual_mt/building_better_avoiding_pitfalls_in_developing_language_resources_when_data_is_.md)

</div>

<!-- RELATED:END -->
