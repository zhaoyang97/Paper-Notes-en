---
title: >-
  [Paper Note] Can Uniform Meaning Representation Help GPT-4 Translate from Indigenous Languages?
description: >-
  [ACL 2025][Uniform Meaning Representation (UMR)] This paper explores incorporating Uniform Meaning Representation (UMR) semantic graphs into GPT-4 prompts for translating three indigenous languages (Navajo, Arapaho, and Kukama), finding that the addition of UMR leads to statistically significant performance gains in most cases.
tags:
  - "ACL 2025"
  - "Uniform Meaning Representation (UMR)"
  - "Indigenous languages"
  - "low-resource translation"
  - "GPT-4 prompting"
  - "semantic representation"
date: 2026-05-08
content_hash: 5ecd023dfadd96a7
---

# Can Uniform Meaning Representation Help GPT-4 Translate from Indigenous Languages?

**Conference**: ACL 2025  
**arXiv**: [2502.08900](https://arxiv.org/abs/2502.08900)  
**Code**: None  
**Area**: Other  
**Keywords**: Uniform Meaning Representation (UMR), Indigenous languages, low-resource translation, GPT-4 prompting, semantic representation

## TL;DR

This paper explores incorporating Uniform Meaning Representation (UMR) semantic graphs into GPT-4 prompts for translating three indigenous languages (Navajo, Arapaho, and Kukama), finding that the addition of UMR leads to statistically significant performance gains in most cases.

## Background & Motivation

ChatGPT and other GPT-series models perform exceptionally well on high-resource language tasks but struggle severely with extremely low-resource languages, especially indigenous ones. Robinson et al. (2023) found that the strongest predictor of ChatGPT's translation performance is the number of Wikipedia articles available in the target language. Stap and Araabi (2023) directly pointed out that "ChatGPT is not a good translator for indigenous languages."

**Uniform Meaning Representation (UMR)** is a multilingual extension of Abstract Meaning Representation (AMR), designed to represent the semantics of diverse languages through a flexible annotation process. The advantages of UMR include:
1. Using a paradigmatic lattice allows annotators to select a granularity suitable for a specific language.
2. Creating required rolesets at "Stage 0" overcomes the lack of pre-existing rolesets in low-resource languages.
3. Ettinger et al. (2023) demonstrated that GPT models likely do not implicitly possess the linguistic knowledge required to construct AMR/UMR graphs.

**Core Problem**: Does incorporating UMR graphs into translation prompts provide GPT-4 with extra linguistic information, thereby improving translation quality for extremely low-resource languages?

## Method

### Overall Architecture

Four prompting schemes are designed for comparison:
1. **Zero-shot**: Only the source language text is provided, with instructions to translate into English.
2. **Zero-shot + UMR**: Provides the source language text along with its UMR semantic graph.
3. **Five-shot**: Provides 5 exemplars (source text + English reference translation) and the text to be translated.
4. **Five-shot + UMR**: Provides 5 exemplars (including UMR graphs) and the text to be translated along with its UMR graph.

Translations are generated for three indigenous languages: Navajo (506 sentences), Kukama (105 sentences), and Arapaho (406 sentences), totaling 1,017 sentences.

### Key Designs

#### UMR Graph Integration

UMR is a rooted directed graph, embedded in the prompts in the PENMAN text format. Example: the UMR corresponding to the sentence "They were buying a new car":

```text
(s / buy-01
  :ARG0 (p / person
    :refer-person 3rd
    :refer-number Plural)
  :ARG1 (c / car
    :ARG1-of (n / new-01)
    :refer-number Singular)
  :aspect Activity
  :modstr FullAff)
```

UMR graphs provide semantic structural information about the sentence (who did what to whom), including participant roles, aspect, and modality strength (`modstr`), which may supplement the linguistic knowledge of low-resource languages that is missing from the model's pre-training.

#### Adaptive Exemplar Selection

The 5 exemplars for Five-shot are not selected randomly, but using an **adaptive method**: the chrF metric is used to compare source language sentences, selecting the 5 nearest neighbors most similar to the current sentence to be translated. Computing similarity using source language sentences (rather than English references) ensures the method can be reproduced during testing.

#### Data Source

The study utilizes the first UMR dataset released by Bonn et al. (2024), which contains UMR annotations and English translations for Navajo (506 sentence-level graphs), Arapaho (406 sentence-level graphs), and Kukama (105 sentence-level graphs). Sanapaná was excluded as it only has Spanish translations.

### Loss & Training

Since this is a study on prompting methods, there is no model training. Translations are generated using the GPT-4 API, with a total experimental cost of \$62.11. Evaluation metrics are chrF and BERTScore, and a two-tailed paired t-test is used for statistical significance analysis.

## Key Experimental Results

### Main Results

| Prompting Scheme | Arápaho chrF | Kukama chrF | Navajo chrF |
|---|:---:|:---:|:---:|
| Zero-shot | 13.0±5.5 | 14.0±5.8 | 15.4±6.4 |
| Zero-shot + UMR | 16.2±8.7 | 16.8±7.0 | 17.9±8.3 |
| Five-shot | 32.9±21 | 40.8±25 | 24.6±14.2 |
| **Five-shot + UMR** | **35.7±22** | **43.5±24** | **25.9±14.1** |

| Prompting Scheme | Arápaho BERTScore | Kukama BERTScore | Navajo BERTScore |
|---|:---:|:---:|:---:|
| Zero-shot | 0.867±0.02 | 0.862±0.02 | 0.862±0.02 |
| Zero-shot + UMR | 0.867±0.05 | 0.857±0.03 | 0.867±0.03 |
| Five-shot | 0.903±0.04 | 0.904±0.04 | 0.885±0.03 |
| **Five-shot + UMR** | **0.910±0.04** | **0.912±0.04** | **0.891±0.03** |

### Ablation Study

Statistical significance analysis (two-tailed paired t-test):

| Comparison | Arápaho | Kukama | Navajo | Significant Improvement Count |
|---|:---:|:---:|:---:|:---:|
| Zero-shot vs Zero+UMR (chrF) | $p < 0.0001$ ✓ | $p < 0.0001$ ✓ | $p < 0.0001$ ✓ | 3/3 |
| Zero-shot vs Zero+UMR (BERT) | $p = 0.97$ ✗ | $p = 0.015$ ✗ (inverse) | $p < 0.0001$ ✓ | 1/3 |
| Five-shot vs Five+UMR (chrF) | $p = 0.0004$ ✓ | $p = 0.056$ ✗ | $p = 0.029$ ✓ | 2/3 |
| Five-shot vs Five+UMR (BERT) | $p < 0.0001$ ✓ | $p = 0.002$ ✓ | $p < 0.0001$ ✓ | 3/3 |
| Zero-shot vs Five-shot (both metrics) | All $p < 0.0001$ ✓ | | | 6/6 |

**Out of 12 UMR comparisons, 9 show statistically significant improvements**, with only 1 showing a negative effect (Kukama BERTScore zero-shot).

### Key Findings

1. **Five-shot + UMR achieves the best performance across all languages and metrics**: average chrF increases by 2.3–2.8 (compared to Five-shot), indicating that UMR still provides incremental gains on top of Five-shot.
2. **Exemplar effect > UMR effect**: The improvement from Zero-shot to Five-shot is the most significant (chrF jumps from ~14 to ~33), while the increment brought by UMR is smaller but stable.
3. **UMR provides complementary information**: Relying solely on exemplars is insufficient to achieve optimal results; UMR graphs likely supplement linguistic structural information not internalized by the model.
4. **Kukama benefits the most**: Under the Five-shot setting, it improves from 14.0 to 40.8 (+191%), likely because the adaptively selected exemplars from the 105 sentences are of higher quality.

## Highlights & Insights

1. **First downstream application validation of UMR**: This is the first study to explore the utility of UMR in practical NLP tasks, providing empirical evidence for the application value of semantic representation.
2. **Convincing qualitative analysis**: Taking the Kukama sentence translating to "He run in the forest" as an example:
    - Zero-shot → "He plays with his younger brother at the river" (completely unrelated)
    - Five-shot → "He has already started walking in the forest" (close)
    - Five-shot + UMR → "He has already started running in the forest" (best)
3. **Very low cost**: The entire experiment cost only \$62.11, demonstrating the cost-effectiveness of prompting methods in extremely low-resource scenarios.
4. **Adaptive exemplar selection**: Neighbor selection is based on source language chrF rather than English reference, which is also feasible during testing.

## Limitations & Future Work

1. Only three indigenous languages were tested, without covering languages of various resource levels.
2. UMR annotation is costly and requires linguistic experts, limiting the scalability of practical deployment.
3. Only the indigenous-to-English translation direction was tested; reverse translation would require evaluation by native speakers of the target languages.
4. The randomness of GPT-4 affects the reproducibility of the results; although statistical tests were performed, the runs were not repeated multiple times.
5. Future work could explore automatic UMR parsers to reduce annotation dependency, or combine UMR with lexicon-based methods (Guo et al., 2024).

## Related Work & Insights

- **ChatGPT Translation**: Robinson et al. (2023) and Stap & Araabi (2023) point out the difficulties of low-resource translation; this paper provides a path for improvement.
- **AMR/UMR Applications**: Hua et al. (2023) and Gururaja et al. (2023) utilize AMR in low-resource settings; this study extends this to the multilingual design of UMR.
- **Chain-of-Thought Prompting**: Peng et al. (2023) found that CoT is ineffective for translation (leading to word-by-word translation); UMR provides structured semantic information rather than a reasoning chain.

**Insight**: Can other semantic representations (e.g., semantic role labeling, dependency parsing) assist low-resource translation in a similar manner?

## Rating

- **Novelty**: ★★★★☆ — First downstream application validation of UMR, offering a unique key insight.
- **Technical Depth**: ★★★☆☆ — Relatively simple method (prompt engineering), main contribution lies in empirical findings.
- **Experimental Thoroughness**: ★★★★☆ — 1,017 sentences across three languages, 4 prompting schemes, dual metrics + statistical testing + qualitative analysis.
- **Utility**: ★★★☆☆ — Cost of UMR annotation limits direct application, but validates the value of semantic representation.
- **Writing Quality**: ★★★★☆ — Clear structure, vivid examples.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GPT-4 as a Homework Tutor can Improve Student Engagement and Learning Outcomes](gpt-4_as_a_homework_tutor_can_improve_student_engagement_and_learning_outcomes.md)
- [\[ACL 2025\] Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages](using_source-side_confidence_estimation_for_reliable_translation_into_unfamiliar.md)
- [\[ACL 2025\] A New Formulation of Zipf's Meaning-Frequency Law through Contextual Diversity](a_new_formulation_of_zipfs_meaning-frequency_law_through_contextual_diversity.md)
- [\[ACL 2025\] If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?](if_attention_serves_as_a_cognitive_model_of_human_memory_retrieval_what_is_the_p.md)
- [\[NeurIPS 2025\] Structure-Aware Spectral Sparsification via Uniform Edge Sampling](../../NeurIPS2025/others/structure-aware_spectral_sparsification_via_uniform_edge_sampling.md)

</div>

<!-- RELATED:END -->
