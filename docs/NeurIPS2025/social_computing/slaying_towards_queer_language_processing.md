---
title: >-
  [Paper Note] SLAyiNG: Towards Queer Language Processing
description: >-
  [NeurIPS 2025 (Queer in AI Workshop)][Social Computing][queer slang] This work introduces SLAyiNG, the first explicitly annotated queer slang dataset, comprising 695 terms and nearly 200,000 usage instances. Inter-annotator agreement experiments (Krippendorff's $\alpha = 0.746$) demonstrate that reasoning models can serve as pre-screening tools but community-driven expert annotation remains indispensable.
tags:
  - NeurIPS 2025 (Queer in AI Workshop)
  - Social Computing
  - queer slang
  - dataset annotation
  - sense disambiguation
  - LLM evaluation
  - sociolinguistics
date: 2026-05-08
content_hash: c4478ada88d5a1d1
---

# SLAyiNG: Towards Queer Language Processing

**Conference**: NeurIPS 2025 (Queer in AI Workshop)
**arXiv**: [2509.17449](https://arxiv.org/abs/2509.17449)
**Code**: None
**Area**: Social Computing
**Keywords**: queer slang, dataset annotation, sense disambiguation, LLM evaluation, sociolinguistics

## TL;DR
This work introduces SLAyiNG, the first explicitly annotated queer slang dataset, comprising 695 terms and nearly 200,000 usage instances. Inter-annotator agreement experiments (Krippendorff's $\alpha = 0.746$) demonstrate that reasoning models can serve as pre-screening tools but community-driven expert annotation remains indispensable.

## Background & Motivation
**Background**: LLM comprehension of slang is critical for user-facing applications. Prior work has established benchmarks for general slang detection and recognition (e.g., Mei et al. 2024; Sun et al. 2024), but queer slang has not received dedicated attention.

**Limitations of Prior Work**: Queer slang is severely underrepresented in LLM pre-training data, leading to two concrete problems: (1) queer slang is frequently misclassified as hate speech (e.g., "cunt" carries positive connotations in queer contexts); (2) prompts containing queer slang elicit more negative responses from LLMs.

**Key Challenge**: The absence of high-quality annotated benchmark datasets prevents systematic evaluation of queer slang detection and processing. Many terms (e.g., "mother," "read") possess non-queer senses, making sense disambiguation the primary annotation challenge.

**Goal**: (1) Construct the first annotated dataset with explicit queer slang coverage; (2) evaluate the feasibility and limitations of state-of-the-art reasoning models as annotation tools.

**Key Insight**: The work draws on queer linguistics, integrating multiple online resources (ontologies, Wiktionary, glossaries) to collect authentic usage instances and design a multi-stage human–machine collaborative annotation pipeline.

**Core Idea**: A pipeline combining multi-source crawling, LLM pre-screening, and community annotation to build the first queer slang sense disambiguation dataset.

## Method

### Overall Architecture
The work centers on constructing the SLAyiNG dataset through three stages: term collection → raw dataset crawling → pre-screening and annotation. The inputs are multiple online queer language resources; the outputs are cleaned data annotated for sense disambiguation, harmful content detection, and author community membership.

### Key Designs

1. **Multi-source Term Collection and Deduplication**:

    - Function: Queer slang terms are collected from the GSSO ontology (414 terms), lgbtDB (215 terms), the Chew glossary (65 terms), and Wiktionary (251 terms), yielding 695 terms plus 90 variants.
    - Mechanism: All definitions are embedded using `all-mpnet-base-v2`; a cosine similarity matrix is computed, and for pairs with similarity $> 0.7$, only the longer (more detailed) definition is retained.
    - Design Motivation: The same term may carry redundant definitions across sources (e.g., "bear") but may also have genuinely distinct senses (e.g., "angel" in ballroom culture). The approach deduplicates while preserving valid polysemy distinctions.

2. **Three-source Raw Dataset Construction**:

    - Function: Sentences containing target terms are collected from Reddit (58%, 114K instances), podcasts (35%, 70K instances), and OpenSubtitles captions (7%, 13K instances), totalling 197,958 instances.
    - Mechanism: Reddit retrieves up to 15 relevant posts per term from 264 LGBTQ+-related subreddits; podcasts are sourced from the Society & Culture category of Podscripts (113 podcasts); captions are extracted from OpenSubtitles using IMDb lists of queer-related film and television. All sentences are filtered to a length of 4–30 tokens.
    - Design Motivation: The multi-source design ensures coverage of diverse usage contexts (online communities, spoken media, scripted dialogue), enhancing real-world representativeness.

3. **LLM-assisted Pre-screening + Multi-stage Annotation Pipeline**:

    - Function: `o3-mini` is used as a pre-screening tool, combined with human annotation to complete three tasks — sense disambiguation, harmful content detection, and author community membership judgment.
    - Mechanism: A five-stage pipeline: (1) three annotators label 25 samples to establish baseline agreement; (2) each annotator labels approximately 2,200 instances and agreement with `o3-mini` is computed; (3) terms with high agreement are fully annotated by `o3-mini`, while low-agreement terms are re-annotated by the stronger `o3` model; (4) iterative human annotation with periodic agreement checks; (5) queer community members are recruited for validation.
    - Design Motivation: With nearly 200,000 raw instances, purely manual annotation is infeasible. However, LLMs completely fail on certain neologisms (e.g., "anticistamines"), and 42.52% of terms exhibit low agreement, precluding full reliance on LLMs.

### Annotation Scheme
- A WSsim task design is adopted; annotators rate each word sense on a 1–5 Likert scale.
- Sense disambiguation is the core task, as the majority of crawled instances are false positives (i.e., non-queer uses of the target term).

## Key Experimental Results

### Human–Machine Annotation Agreement

| Metric | AA–AA (25 instances) | AA1–LLMA (250) | AA2–LLMA (250) | AA3–LLMA (250) | Mean |
|---|---|---|---|---|---|
| Krippendorff's $\alpha$ | 0.877 | 0.750 | 0.769 | 0.719 | 0.746 |
| 95% CI | [0.727, 0.959] | [0.675, 0.821] | [0.689, 0.840] | [0.643, 0.791] | — |

### Term-level Analysis

| Configuration | Proportion | Description |
|---|---|---|
| $\alpha > 0.6$ or $F_1 > 0.8$ | 57.48% | Terms reliably annotatable by `o3-mini` |
| Low-agreement terms | 42.52% | Still require human annotation |

### Key Findings
- `o3-mini` completely fails on neologisms (e.g., "anticistamines" = anti-cis + antihistamines), achieving $\alpha = -0.833$, due to absence from training data.
- For highly ambiguous terms (e.g., "cunt" as a positive queer adjective vs. a general slur), `o3-mini` produces unstable judgments ($\alpha = 0.0$).
- Inter-human agreement is high ($\alpha = 0.877$), indicating that the annotation task itself is feasible.

## Highlights & Insights
- **Practicality of Multi-source Data Strategy**: Collecting from ontologies, dictionaries, social media, podcasts, and subtitles better reflects real usage distributions than any single source; this strategy is transferable to building linguistic resources for other subcultural communities.
- **Fine-grained Evaluation of LLMs as Annotation Tools**: Rather than a binary verdict on LLM capability, agreement is assessed at the individual term level, providing a quantitative basis for designing hybrid annotation pipelines.
- **Semantic Deduplication Method**: Using sentence embeddings and cosine similarity to automatically merge redundant definitions while preserving polysemy is a simple and effective technique.

## Limitations & Future Work
- **English Only**: Queer slang varies substantially across languages, and English queer slang itself is influenced by other languages.
- **Dataset Not Yet Complete**: The paper describes work in progress; the fully annotated version has not been released.
- **Temporal Validity of Terms**: Slang evolves rapidly; the dataset represents only a snapshot as of July 2025.
- **No Downstream Task Evaluation**: The paper does not demonstrate improvements on downstream tasks such as hate speech detection or dialogue systems following training on SLAyiNG.

## Related Work & Insights
- **vs. Sun et al. (2024) General Slang Benchmark**: SLAyiNG focuses on the queer subgroup and must address term reclamation and ingroup/outgroup context distinctions specific to queer language.
- **vs. Dorn et al. (2024) Hate Speech Detection**: SLAyiNG provides annotations that distinguish ingroup usage from hateful usage, serving as potential training data for improving content moderation systems.

## Rating
- Novelty: ⭐⭐⭐⭐ — First NLP dataset for queer slang, filling a clear gap.
- Experimental Thoroughness: ⭐⭐⭐ — Only preliminary annotation agreement experiments; no downstream task validation.
- Writing Quality: ⭐⭐⭐⭐ — Background is thoroughly introduced; term sourcing and processing are transparently described.
- Value: ⭐⭐⭐⭐ — Meaningful contribution to fairness and inclusive NLP, pending release of the complete dataset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Active Slice Discovery in Large Language Models](active_slice_discovery_in_large_language_models.md)
- [\[NeurIPS 2025\] DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models](date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)
- [\[NeurIPS 2025\] Don't Let It Fade: Preserving Edits in Diffusion Language Models via Token Timestep Allocation](dont_let_it_fade_preserving_edits_in_diffusion_language_mode.md)
- [\[NeurIPS 2025\] Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector](any_large_language_model_can_be_a_reliable_judge_debiasing_w.md)
- [\[ACL 2026\] Among Us: Language of Conspiracy Theorists on Mainstream Reddit](../../ACL2026/social_computing/among_us_language_of_conspiracy_theorists_on_mainstream_reddit.md)

</div>

<!-- RELATED:END -->
