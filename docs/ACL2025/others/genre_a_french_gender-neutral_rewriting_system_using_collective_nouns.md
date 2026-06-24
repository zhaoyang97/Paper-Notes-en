---
title: >-
  [Paper Note] GeNRe: A French Gender-Neutral Rewriting System Using Collective Nouns
description: >-
  [ACL 2025][gender-neutral rewriting] GeNRe is the first French gender-neutral rewriting system. It leverages collective nouns to replace masculine generics, proposing three approaches: a rule-based system, fine-tuned models, and an instruction-based model. Among these, the rule-based system and the Claude 3 Opus + dictionary approach yield the best performance.
tags:
  - "ACL 2025"
  - "gender-neutral rewriting"
  - "collective nouns"
  - "French NLP"
  - "gender bias"
  - "rule-based system"
date: 2026-05-08
content_hash: 7ca05b5248595748
---

# GeNRe: A French Gender-Neutral Rewriting System Using Collective Nouns

**Conference**: ACL 2025  
**arXiv**: [2505.23630](https://arxiv.org/abs/2505.23630)  
**Code**: [Available](https://github.com/spidersouris/GeNRe)  
**Area**: Other  
**Keywords**: gender-neutral rewriting, collective nouns, French NLP, gender bias, rule-based system

## TL;DR

GeNRe is the first French gender-neutral rewriting system. It leverages collective nouns to replace masculine generics, proposing three approaches: a rule-based system, fine-tuned models, and an instruction-based model. Among these, the rule-based system and the Claude 3 Opus + dictionary approach yield the best performance.

## Background & Motivation

In French, nouns are divided into masculine and feminine genders, with the masculine gender treated as the default when referring to mixed-gender groups (known as "masculine generics", MG). Extensive psycholinguistic research indicates that:

**MG causes cognitive bias**: When masculine generics are used, respondents' biological associations bias toward males. Stahlberg et al. (2001) found that when questions were asked using MG in German, respondents were more likely to name male celebrities.

**Amplification of gender bias in NLP**: The usage of MG in training data is learned and amplified by models, particularly affecting cross-lingual tasks such as machine translation.

**Gaps in existing work**:
   - Gender-neutralization systems exist for English (Sun 2021, Vanmassenhove 2021), but **none** exist for French.
   - The only prior French gender-rewriting system (Lerner & Grouin 2024) employs "visible" writing techniques (e.g., professeur·e), which are highly controversial.
   - The automatic application of collective nouns as a neutralization mechanism has never been studied.

Collective nouns (e.g., "la police" referring to police officers of all genders) have a grammatical gender independent of referents, presenting an effective path to neutrality. Compared to "visible" writing techniques, neutralization does not alter existing spelling rules or introduce non-standard punctuation, leading to higher social acceptability and better suitability for expressing non-binary gender identities.

## Method

### Overall Architecture

The GeNRe system comprises three approaches: (1) a Rule-Based System (RBS); (2) fine-tuned language models (T5, M2M100); and (3) an instruction-based Large Language Model (Claude 3 Opus). All approaches share a manually constructed dictionary mapping member nouns to collective nouns in French.

### Key Designs

1. **Construction of the Collective Noun Dictionary (315 entries)**: Collected through three channels:

    - **Literature Review**: Selected 105 entries from a list of 138 collective nouns by Lecolle (2019), excluding polysemous words and those with excessively narrow semantics (e.g., "duo" referring only to a pair).
    - **Manual Collection**: Gathered 46 entries from media, the internet, and Sketch Engine (a corpus search tool).
    - **Semi-automatic Collection**: Scraped the French Wiktionary, automatically extracting words with the "-phonie" suffix and generating corresponding "-phone" forms (e.g., anglophonie → anglophone) to obtain 164 entries.

   The design motivation is to establish a mapping as comprehensive as possible, ensuring that every masculine generic member noun can find a corresponding collective noun. For example: soldats (soldiers) → armée (army), policiers (police officers) → police.

2. **Rule-Based System (RBS)**: The core workflow consists of two components:

    - **Syntactic Dependency Detection**: Based on spaCy (fr_core_news_sm) and custom rules to detect all syntactic dependents (determiners, adjectives, past participles, anaphoric pronouns, etc.) that need to agree with the member noun. This improved the dependency detection F1 score from 0.183 (spaCy default) to 0.799.
    - **Generation Component**: Replaces member nouns with collective nouns from the dictionary, handles elisions, and uses the inflecteur library to apply gender and number agreements to the detected dependents. Post-processing corrections for past participles and accusative pronouns raised the agreement accuracy from 73.01% to 75.35%.

   Input member nouns in sentences are marked with tags (e.g., `<n-126>les auteurs</n>`). Since a member noun may correspond to multiple collective nouns, all possible target variations are generated for training data augmentation.

3. **Fine-Tuned Models**: T5-small (60M parameters) and M2M100 (418M parameters) are selected for seq2seq fine-tuning. The training data consists of (original text, neutralized text) pairs generated by the RBS, with 60,000 pairs for training and 6,000 pairs for validation per corpus. These models are selected due to their manageable scale, strong text-to-text performance, and previous successful application of M2M100 to Portuguese gender rewriting.

4. **Instruction-Based Model (Claude 3 Opus)**: Three prompting strategies were designed:

    - **BASE**: Basic task description without specifying replacement words.
    - **DICT**: Provides collective noun mappings from the dictionary, explicitly specifying the target replacements.
    - **CORR**: Takes the output sentences generated by RBS as input and prompts the model to correct potential grammatical errors.

### Loss & Training

The fine-tuned models utilize standard seq2seq cross-entropy loss. The data sources consist of sentences containing member nouns from Wikipedia (292,076 sentences) and Europarl (106,878 sentences).

## Key Experimental Results

### Main Results

Evaluation was conducted on 500 manually neutralized reference sentences (250 from Wikipedia + 250 from Europarl):

| Model | WER (↓) | BLEU (↑) | Cosine (↑) |
|------|---------|----------|------------|
| Baseline (Original Sentence) | 12.529% | 81.779 | 97.222 |
| **GeNRe-RBS** | **3.81%** | 92.887 | **99.05** |
| GeNRe-T5 | 5.492% | 90.234 | 98.804 |
| GeNRe-M2M100 | 5.406% | 90.692 | 98.112 |
| Claude-BASE | 12.291% | 82.759 | 96.83 |
| **Claude-DICT** | 4.45% | **93.519** | 99.038 |
| Claude-CORR | 10.137% | 85.25 | 98.074 |

### Syntactic Dependency Detection

| Method | Precision | Recall | F1 |
|------|-----------|--------|-----|
| Baseline (spaCy default) | 0.106 | 0.706 | 0.183 |
| **GeNRe-RBS** | **0.766** | **0.834** | **0.799** |

### Key Findings

1. **RBS and Claude-DICT exhibit the best performance**: RBS achieves the best results in WER and cosine similarity, while Claude-DICT excels in BLEU. Their performances are highly comparable, indicating that an instruction-based model integrated with a dictionary can closely approximate a meticulously designed rule-based system.
2. **Fine-tuned models do not significantly outperform RBS**: Contrary to the findings of Vanmassenhove 2021, fine-tuning did not yield improvements, likely because French morphological variation is far more complex than that of English.
3. **Claude-BASE (without dictionary) performs poorly**: Performing even worse than the baseline, this indicates that LLMs require explicit knowledge support for specialized linguistic tasks.
4. **Error type analysis**: Morphosyntactic errors (especially adjective and verb agreements) are the most frequent across all systems. Semantic errors are most severe in Claude-BASE (238 cases), where unconstrained freedom of generation leads to the usage of inappropriate or non-existent collective nouns.

## Highlights & Insights

- **Precise and Practical Goal**: Gender neutralization is less controversial than visual techniques (e.g., professeur·e) and is more inclusive of non-binary gender identities.
- **The paradigm combining dictionaries with LLMs** is extremely noteworthy: A pure LLM (Claude-BASE) performs poorly, but incorporating a 315-entry dictionary mapping (Claude-DICT) immediately rivals expert systems. This underscores that structured knowledge remains irreplaceable in domain-specific linguistic tasks.
- **Methodological Portability**: The authors explicitly point out that this method is applicable to other languages that use collective nouns for neutralization (such as Spanish), offering low-cost transfer to Romance languages with similar grammatical variations.

## Limitations & Future Work

1. Collective nouns are not a panacea—semantic alignment may fail in many contexts (e.g., replacing "soldats" with "armée" in a non-military context may lead to semantic shift).
2. The bias-mitigation effect of neutralization techniques may be weaker than that of visual gender-marking techniques (Spinelli 2023, Tibblin 2023).
3. Relying on an LLM-as-a-judge for annotation of instruction model errors is not entirely comparable to manual annotation.
4. Limited to French; although the methodology is portable, it requires constructing specialized dictionaries for each target language.
5. Agreement accuracy still has room for improvement (75.35%); the complexity of French morphology demands finer-grained rules or superior morphological tools.

## Related Work & Insights

- The gender rewriting task was originally defined by Alhafni et al. (2022); this paper generalizes it into a broader framework: "generating one or several alternative sentences to achieve neutrality, inclusiveness, or gender-swapping."
- English neutralization research (Sun 2021, Vanmassenhove 2021) achieved comparable levels of WER/BLEU improvement.
- This work serves as an important starting point for alleviating gender bias in French NLP, providing a clear template for hybrid rule-based + LLM approaches.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first French gender-neutralization system; the utilization of collective nouns is a unique and novel innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive design featuring comparisons of three approaches, multi-dimensional error analysis, and evaluation on two corpora.
- **Writing Quality**: ⭐⭐⭐⭐ — Detailed introduction to the linguistic background and abundant French examples, though the barrier of entry remains somewhat high for non-French readers.
- **Value**: ⭐⭐⭐⭐ — Actively advances French NLP and gender-bias mitigation research; the open-sourcing of dictionaries and datasets is exceptionally valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Measure of the System Dependence of Automated Metrics](a_measure_of_the_system_dependence_of_automated_metrics.md)
- [\[ACL 2025\] Unlocking Speech Instruction Data Potential with Query Rewriting](unlocking_speech_instruction_data_potential_with_query_rewriting.md)
- [\[ACL 2025\] IRIS: Interactive Research Ideation System for Accelerating Scientific Discovery](iris_interactive_research_ideation_system_for_accelerating_scientific_discovery.md)
- [\[AAAI 2026\] A Topological Rewriting of Tarski's Mereogeometry](../../AAAI2026/others/a_topological_rewriting_of_tarskis_mereogeometry.md)
- [\[ICCV 2025\] Toward Material-Agnostic System Identification from Videos](../../ICCV2025/others/toward_material-agnostic_system_identification_from_videos.md)

</div>

<!-- RELATED:END -->
