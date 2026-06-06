---
title: >-
  [Paper Note] Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs
description: >-
  [ACL 2026][NLP Understanding][Constructed Languages] This paper proposes IASC (Interactive Agentic System for ConLangs), a modular artificial language construction system. By requiring LLMs to perform morphosyntactic tra…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Constructed Languages"
  - "Metalinguistic Knowledge"
  - "Morphosyntactic Transformation"
  - "LLM Linguistic Capability Probing"
  - "Linguistic Typology"
date: 2026-05-08
content_hash: ccee001779a7f300
---

# Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs

**Conference**: ACL 2026  
**arXiv**: [2510.07591](https://arxiv.org/abs/2510.07591)  
**Code**: [https://github.com/SakanaAI/IASC](https://github.com/SakanaAI/IASC)  
**Area**: LLM Agent  
**Keywords**: Constructed Languages, Metalinguistic Knowledge, Morphosyntactic Transformation, LLM Linguistic Capability Probing, Linguistic Typology

## TL;DR

This paper proposes IASC (Interactive Agentic System for ConLangs), a modular artificial language construction system. By requiring LLMs to perform morphosyntactic transformations according to linguistic specifications, the authors probe их metalinguistic knowledge. They find that LLMs handle common linguistic typological patterns significantly better than rare ones, and that capabilities vary drastically across different models.

## Background & Motivation

**Background**: Numerous studies focus on the linguistic capabilities of LLMs, including translation and syntactic tagging. However, these tasks evaluate knowledge of specific languages rather than an understanding of linguistic concepts themselves. Do LLMs truly "understand" abstract linguistic concepts (such as word order, case marking, and agreement) or do they merely memorize specific patterns of natural languages present in their training data?

**Limitations of Prior Work**: (1) Existing evaluations of LLM linguistic capabilities often focus on encyclopedic tests (knowing a fact about a language), lacking a systematic probe of metalinguistic reasoning. (2) Natural language tests are prone to training data leakage, where LLMs might simply "remember" answers instead of truly understanding rules.

**Key Challenge**: While LLMs encounter vast amounts of linguistic literature and multilingual data during training, this does not necessarily mean they can manipulate language structures according to prescribed abstract grammatical rules. For example, changing the word order of an English sentence from SVO to OVS (an extremely rare word order) is in principle no more difficult than changing it to SOV, yet LLM performance may differ significantly.

**Goal**: (1) Provide a flexible and engaging constructed language (ConLang) creation tool. (2) Utilize morphosyntactic transformation tasks to systematically probe the level of metalinguistic knowledge LLMs possess regarding various linguistic typological features.

**Key Insight**: Constructing a ConLang requires an LLM to go beyond translation; it must reorganize sentence structures and add morphological markers based on abstract grammatical specifications. This directly tests the depth of its understanding of linguistic concepts.

**Core Idea**: A modular ConLang construction system is used as a benchmark to quantify metalinguistic capability by having LLMs transform English sentences according to different morphosyntactic parameters (word order, case systems, tense marking, etc.).

## Method

### Overall Architecture

IASC is a complete ConLang construction pipeline comprising five modules: phonology, morphosyntax, lexicon, orthography, and a grammatical handbook. This paper focuses on the morphosyntax module as a tool for probing LLM metalinguistic knowledge. The input consists of an English source sentence and target grammatical parameters; the output is a glossed annotation transformed according to target specifications. It employs a cumulative morphosyntax strategy, applying different grammatical features step-by-step through multi-stage prompting.

### Key Designs

1.  **Cumulative Morphosyntax**:
    - **Function**: Gradually transforms a source sentence into a form that complies with target grammatical specifications.
    - **Mechanism**: Rather than providing all grammatical specifications for a single-shot transformation (which preliminary experiments showed to be ineffective), only one grammatical feature is applied at a time (e.g., word order first, then case marking, then tense marking). This is achieved through iterative prompting: $s_i = M(s_{i-1}; G; t_i)$, where each prompt $t_i$ focuses on one specific feature.
    - **Design Motivation**: Single-shot transformations lead to excessively long and complex prompts that LLMs struggle to follow simultaneously. Stepwise accumulation reduces the cognitive load at each stage.

2.  **Nine Typologically Diverse Grammatical Configurations**:
    - **Function**: Builds an evaluation dataset covering a range from common to rare linguistic types.
    - **Mechanism**: The authors designed eight grammatical configurations inspired by real languages (Arabic, Fijian, French, Hixkaryana, Mizo, Turkish, Vietnamese, Welsh) plus one "hard" configuration (an extremely rare typological combination). Each configuration defines parameters for word order, case systems, agreement markers, and tense markers. The dataset consists of 45 source sentences $\times$ 9 configurations = 405 test samples, with gold data manually annotated by linguists.
    - **Design Motivation**: Testing across typological frequencies determines whether LLMs truly understand abstract rules or are restricted to patterns common in their training data.

3.  **Agentic Refinement**:
    - **Function**: Iteratively improves output through automatically generated feedback.
    - **Mechanism**: Certain modules (such as phonology) employ an agentic approach where the LLM generates an initial output, followed by self-generated critiques/feedback, and then an improved version based on that feedback.
    - **Design Motivation**: Initial LLM outputs may not fully comply with specifications; self-correction mechanisms improve final quality.

## Key Experimental Results

### Main Results

| Model | 'french' (Common) | 'turkish' (Common) | 'mizo' (Rare) | 'hard' (Very Rare) | Overall Performance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4.1 | Low TER | Low TER | Medium TER | High TER | Best |
| Claude 3.7 | Low TER | Low TER | Medium-High TER | High TER | Second |
| Gemini 2.5 | Medium TER | Medium TER | High TER | Very High TER | Medium |
| Smaller Models | High TER | High TER | Very High TER | Extremely High TER | Poor |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| Cumulative vs. Single-shot | Cumulative is far superior | LLMs cannot follow multiple constraints simultaneously in single-shot. |
| Common vs. Rare Typology | Common is far superior | LLMs handle SVO/SOV well but fail on OVS/OSV. |
| Morphological Marking | Suffixes are better | Consistent with the higher frequency of suffixes in training data. |
| Agentic refinement | Sometimes improves | Not all modules benefit equally. |

### Key Findings

- LLM processing of common linguistic typological patterns (e.g., SVO, SOV word order, suffixal morphology) is significantly better than for rare patterns (e.g., OVS word order, prefixal morphology), correlating highly with the distribution frequency of these features in world languages.
- Capability gaps between different LLMs are immense: GPT-4.1 performs best across most configurations, while smaller models fail almost entirely on rare configurations.
- The "hard" configuration (containing extremely rare typological combinations) is challenging for all models, indicating that metalinguistic knowledge remains heavily constrained by the distribution of training data.

## Highlights & Insights

- **ConLangs as Probing Tools**: An ingenious experimental design where artificial languages avoid data leakage issues and allow precise control over linguistic variables, making evaluation results highly interpretable.
- **Revealing the Nature of LLM "Linguistic Knowledge"**: LLMs do not truly "understand" linguistic concepts but rather depend on pattern distributions from training data. Performance variations based on typological frequency suggest their capabilities are rooted in statistical correlation rather than abstract rule comprehension.
- **Cumulative Transformation Strategy**: Decomposing complex multi-constraint problems into stepwise single-constraint transformations is a versatile prompt engineering strategy applicable to other multi-step reasoning scenarios.

## Limitations & Future Work

- The evaluation dataset (405 samples) is relatively small and may not capture all interaction effects between grammatical features.
- The study uses English only as the source language; transformations starting from other languages were not explored.
- Gold data for the morphosyntax module was annotated by a single linguist, which may introduce annotator bias.
- Negative results were largely found when applying the method to low-resource language translation, suggesting practical applications remain distant.
- The 53-page paper contains extensive appendices; core contributions could be more concise.

## Related Work & Insights

- **vs. ConlangCrafter (Alper et al., 2025)**: Also builds ConLangs using LLMs, but IASC offers finer-grained morphosyntax modules supporting feature-by-feature probing.
- **vs. Traditional LLM Linguistic Tests**: Unlike BLiMP or SyntaxGym, which test judgments on specific phenomena, IASC requires active manipulation of linguistic structures, representing a higher difficulty level.
- **vs. Diamond (2023)**: Which used simple prompts for ChatGPT to generate ConLangs without systematic modular control or typological evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Probing metalinguistic knowledge through ConLang construction is a novel and profound research perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Nine configurations cover rich typological diversity, though sample sizes are modest.
- Writing Quality: ⭐⭐⭐⭐ Extremely detailed (53 pages) with thorough linguistic background, though somewhat verbose.
- Value: ⭐⭐⭐⭐⭐ Provides critical insights into the nature of LLM linguistic knowledge; the IASC tool itself has independent value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Commonsense Knowledge with Negation: A Resource to Enhance Negation Understanding](commonsense_knowledge_with_negation_a_resource_to_enhance_negation_understanding.md)
- [\[ACL 2026\] Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation](knowledge-driven_augmentation_and_retrieval_for_integrative_temporal_adaptation.md)
- [\[ACL 2026\] Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs](reasoning-based_refinement_of_unsupervised_text_clusters_with_llms.md)
- [\[ACL 2026\] Filling the Gap: Is Commonsense Knowledge Generation useful for Natural Language Inference?](filling_the_gap_is_commonsense_knowledge_generation_useful_for_natural_language_.md)
- [\[ACL 2026\] Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?](can_llms_estimate_cognitive_complexity_of_reading_comprehension_items.md)

</div>

<!-- RELATED:END -->
