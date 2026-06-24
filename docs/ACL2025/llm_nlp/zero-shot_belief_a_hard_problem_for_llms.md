---
title: >-
  [Paper Note] Zero-Shot Belief: A Hard Problem for LLMs
description: >-
  [ACL 2025][LLM (Other)] This paper proposes two zero-shot frameworks, Unified and Hybrid, for source-and-target belief prediction. The hybrid approach utilizes a fine-tuned DeBERTa for event detection combined with an LLM for belief annotation, setting a new SOTA with 72.0% Full F1 on FactBank. Additionally, it highlights nested belief performance for the first time (reporting a low Nested F1 of only 25.3%), revealing that this sub-task remains a significant challenge for all…
tags:
  - "ACL 2025"
  - "LLM (Other)"
date: 2026-05-08
content_hash: 6efe22348e18cea0
---

# Zero-Shot Belief: A Hard Problem for LLMs

**Conference**: ACL 2025  
**Author**: John Murzaku, Owen Rambow (Stony Brook University)
**arXiv**: [2502.08777](https://arxiv.org/abs/2502.08777)  
**Code**: Planned to be open-sourced  
**Area**: LLM / NLP

## TL;DR

This paper proposes two zero-shot frameworks, Unified and Hybrid, for source-and-target belief prediction. The hybrid approach utilizes a fine-tuned DeBERTa for event detection combined with an LLM for belief annotation, setting a new SOTA with 72.0% Full F1 on FactBank. Additionally, it highlights nested belief performance for the first time (reporting a low Nested F1 of only 25.3%), revealing that this sub-task remains a significant challenge for all current LLMs.

## Background & Motivation

1. **Belief/Event Factuality is a Core NLP Task**: Determining the degree of factual commitment (e.g., Factual, Probable, Unknown) of the author or cited sources in a text towards an event is of high value for information extraction, fake news detection, and intelligence analysis. FactBank (Saurí and Pustejovsky, 2009) is the first standard corpus annotated for source-and-target belief in this domain.
2. **No Prior Zero-Shot Experiments**: Although the belief prediction task has been studied for years, all existing methods (including BERT, RoBERTa, Flan-T5, etc.) rely on supervised fine-tuning. The performance of LLMs under a zero-shot setting has never been evaluated, leaving a critical research gap.
3. **Nested Belief Never Evaluated Individually**: FactBank annotates not only the author's belief but also the belief of nested sources (e.g., "company" in "the company said...") towards events. However, all prior works only reported overall F1 or author F1, and have never reported nested belief (Nested) performance separately.
4. **Event Recognition is Inherently Difficult**: The definition of events in FactBank is complex (including reporting verbs, cognitive verbs, event nouns, state adjectives, etc.). Even fine-tuned specialized generative models only achieve an 85.4% F1, while LLMs perform significantly worse on this sub-task.
5. **Task Requires Deep Pragmatic Reasoning**: Belief detection requires understanding the impact of nested attribution structures, negation semantics, modals, and tenses on factuality, making it an ideal testbed for evaluating the deep language understanding capabilities of LLMs.
6. **Unknown Cross-Lingual Transfer Ability**: The zero-shot transfer capability of belief detection methods to non-English languages (such as the Italian ModaFact corpus) has not been systematically examined.

## Method

### Task Definition and Label Taxonomy

Given a text segment, three elements must be identified simultaneously: **(1)** **Event**—verbs, event nouns, or stative adjectives; **(2)** **Source**—including the author (AUTHOR) and nested sources (e.g., cited persons or organizations); **(3)** **Factuality Label**—the degree of belief of each source towards each event. The label taxonomy contains five categories: **true** (certain facts), **false** (certain counter-facts), **ptrue** (probably true), **pfalse** (probably false), and **unknown** (uncommitted/uncertain).

For example, in the sentence "Trurit Inc. said it is phasing out legacy routers":
- Author's belief toward the "said" event: **true** (the author confirms the speaking event occurred)
- Author's belief toward the "phasing" event: **unknown** (the author only reports it, without committing to its truth)
- Trurit Inc.'s belief toward the "phasing" event: **true** (the company itself commits to the event being true)

### Unified Zero-Shot Approach

A single end-to-end zero-shot prompt is designed to combine event recognition, source analysis, and belief annotation into one instruction. The prompt structure contains:

- **Task Description**: High-level instructions of the FactBank-style event factuality annotation task.
- **Three-Step Annotation Pipeline**: (1) Identify all event predicates (single tokens). (2) Identify nested sources and normalize them into the `AUTHOR_<label>` format. (3) Assign factuality labels to each (source, event) pair.
- **Guidelines for Special Cases**: Marking future events as `unknown`, using `false` for negated sentences, using `ptrue` for modals, etc.
- **Chain-of-Thought (CoT) Output Format**: Requiring the model to explain its reasoning step-by-step and finally output the annotation results in JSON format.

### Hybrid Zero-Shot Approach

The core idea is to **decouple event detection from belief annotation**, delegating sub-tasks that LLMs underperform at to a specialized model:

1. **Event Detection** (DeBERTa-large): Event detection is treated as a token-level binary classification task (O vs. EVENT) and fine-tuned using DeBERTa-large. Hyperparameters: 5 epochs, batch size 16, learning rate 1e-4, maximum sequence length 128. This model achieves an 89.0% F1 score on FactBank, significantly outperforming all LLMs in zero-shot/few-shot event detection.
2. **LLM Belief Annotation**: The list of events detected by DeBERTa along with the original text is fed into the LLM, instructing it to identify nested sources and assign factuality labels. The prompt omits the event recognition step, starting directly with source identification and label assignment, also utilizing the CoT format.

### Source Normalization

FactBank annotates sources at the token level (e.g., "Trurit Inc." is annotated as "Inc."), whereas LLM-predicted source formats are often inconsistent. Few-shot (10 exemplars) post-processing with GPT-4o is used to map predicted sources to FactBank-compatible formats. Ablation studies show:

| Normalization Strategy | Full F1 (%) | Nested F1 (%) |
|:---|:---:|:---:|
| No Normalization | 68.9 | 17.5 |
| Few-Shot Normalization | **72.0** | **25.3** |
| Oracle Normalization | 72.7 | 27.1 |

Few-shot normalization compared to no normalization improves Full F1 by 3.1% and substantially boosts Nested F1 by 7.8%, indicating that source format consistency has a massive impact on the evaluation of nested beliefs.

### Evaluation Metrics

Three Micro F1 metrics are adopted: **Full**—exact match of all (source, event, label) triples; **Author**—evaluating author beliefs only; **Nested**—evaluating nested source beliefs only (introduced for the first time in this work).

## Key Experimental Results

### FactBank Main Results (Micro F1 %)

| Model | Type | Method | Full F1 | Author F1 | Nested F1 | Δ Full (vs SOTA) | Δ Full (Hyb-Uni) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Flan-T5-XL | Fine-tuned SOTA | Fine-tune | 69.5 | 76.6 | — | — | — |
| GPT-3 | Fine-tuned | Fine-tune | 65.8 | 76.0 | — | — | — |
| DeepSeek r1 | Reasoning + Open-source | Unified | 66.1 | 71.1 | 24.1 | — | — |
| DeepSeek r1 | Reasoning + Open-source | Hybrid | **72.0**† | 77.6 | **25.3**† | +2.5 | +5.9 |
| o1 | Reasoning | Unified | 65.0 | 73.2 | 18.9 | — | — |
| o1 | Reasoning | Hybrid | 70.3 | **78.9**† | 19.2 | +0.8 | +5.3 |
| Claude 3.5 | Closed-source | Unified | 63.2 | 69.7 | 19.7 | — | — |
| Claude 3.5 | Closed-source | Hybrid | 70.4 | 77.6 | 21.4 | +0.9 | +7.2 |
| GPT-4o | Closed-source | Unified | 60.2 | 65.9 | 20.2 | — | — |
| GPT-4o | Closed-source | Hybrid | 68.7 | 73.2 | 22.9 | -0.8 | +8.5 |
| o3-mini | Reasoning | Unified | 62.4 | 70.9 | 15.6 | — | — |
| o3-mini | Reasoning | Hybrid | 65.5 | 75.2 | 17.0 | -4.0 | +3.1 |
| LLaMA 3.3-70B | Open-source | Unified | 53.1 | 60.4 | 14.4 | — | — |
| LLaMA 3.3-70B | Open-source | Hybrid | 58.8 | 66.0 | 19.9 | -10.7 | +5.7 |
| DeepSeek-v3 | Open-source | Unified | 56.3 | 61.4 | 17.1 | — | — |
| DeepSeek-v3 | Open-source | Hybrid | 60.5 | 65.3 | 18.2 | -9.0 | +4.2 |

**Key Findings**:
- DeepSeek r1 Hybrid sets a new SOTA on FactBank with **72.0% Full F1**, outperforming the fine-tuned Flan-T5-XL by 2.5%.
- **Hybrid average improvements over Unified are 5.7% (Full), 5.9% (Author), and 2.0% (Nested)**.
- Reasoning models (r1, o1) generally outperform non-reasoning counterparts, indicating that CoT reasoning is highly effective in belief prediction tasks.
- **All models perform extremely poorly on Nested F1**, with the best being only 25.3%, demonstrating that nested belief reasoning is a major bottleneck for current LLMs.

### Event Detection Performance Comparison (F1 %)

| Model | Method | F1 |
|:---|:---|:---:|
| DeBERTa-large | Fine-tuned | **89.0** |
| Claude 3.5 | Zero-shot | 83.3 |
| DeepSeek r1 | Zero-shot | 82.0 |
| Claude 3.5 | Few-shot (5 exemplars) | 81.8 |
| GPT-4o | Few-shot (5 exemplars) | 81.1 |
| GPT-4o | Zero-shot | 78.2 |
| DeepSeek r1 | Few-shot (5 exemplars) | 76.4 |

Fine-tuned DeBERTa outperforms all LLMs in event detection by 5.7–12.6%, fully confirming the necessity of the hybrid strategy—delegating event detection, which is an LLM weakness, to a specialized model. Notably, few-shot prompts do not consistently boost LLM event detection performance (DeepSeek r1 actually dropped by 5.6%).

### ModaFact Cross-Lingual Evaluation (Italian Belief+Polarity F1 %)

| Model | Method | Bel.+Pol. F1 |
|:---|:---|:---:|
| mT5-XXL | Fine-tune | **64.4** |
| DeepSeek r1 | Hybrid | 63.6† |
| o3-mini | Hybrid | 62.6† |
| GPT-4o | Hybrid | 61.2 |
| GPT-4o | Unified | 42.9 |
| o3-mini | Unified | 40.8 |
| DeepSeek r1 | Unified | 38.6 |

On the Italian ModaFact, although the Hybrid method does not surpass the fine-tuned mT5-XXL (by a narrow margin of 0.8%), the fact that zero-shot performance is so close to SOTA is impressive, especially considering that these LLMs are not explicitly optimized for multilingual scenarios (r1 is primarily optimized for English and Chinese). The performance gap between Unified and Hybrid is even larger in cross-lingual settings (~20%+), further highlighting the critical importance of decoupling event detection.

## Nested Belief Error Analysis

A detailed error analysis was conducted on the nested belief predictions of the best-performing model (r1 Hybrid, Nested F1 25.3%), classifying 326 errors into four categories:

| Error Type | Count | Ratio | Typical Error |
|:---|:---:|:---:|:---|
| Source Mismatch | 123 | 37.7% | Predicting a nested source as AUTHOR (50 cases); failure to identify the source of the pronoun "it" (13 cases) |
| Event False Negatives (FN) | 77 | 23.6% | Omitted event nouns (38 cases), such as "acquisition", "construction"; omitted event verbs (30 cases) |
| Label Error | 73 | 22.4% | Predicting UU (unknown) as CT+ (true) (28 cases); predicting UU as PR+ (ptrue) (22 cases) |
| Event False Positives (FP) | 53 | 16.3% | Over-predicting event nouns (33 cases); over-predicting event verbs (10 cases) |

**Key Insights**: The primary error source is **source mismatch**—LLMs tend to assign AUTHOR as the source for all events, ignoring the nested attribution relations in the text. Label errors are mainly concentrated on future/reported events: FactBank dictates that future events reported by nested sources (e.g., "Mary said it will happen") should be labeled as unknown, but LLMs systematically predict them as true or ptrue.

## Highlights & Insights

- **First Zero-Shot Belief Prediction Benchmark**: Filling the research gap under the zero-shot setting, this work systematically evaluates 7 mainstream LLMs and establishes strong baselines for future research.
- **Exquisite Hybrid Strategy**: Decoupling event detection from belief reasoning allows each component to leverage its strengths, boosting Full F1 by nearly 6% on average with a simple yet elegant concept.
- **First Proposal of Nested F1 Metric**: Detailing a crucial and previously overlooked dimension—nested belief detection (where the best model achieves only 25.3% F1), steering future research directions for the community.
- **Thorough Error Analysis**: Fine-grained categorization of 326 nested belief errors into four categories, quantitatively exposing specific failure modes of LLMs.
- **Extensive Experiments**: Involving 7 models, 2 methodologies, and 2 corpora (English and Italian), leading to robust and reliable conclusions.

## Limitations & Future Work

- **Very Weak Nested Belief Performance**: The highest Nested F1 is merely 25.3%, which is far from practical and signals a severe lack of capability in multi-layer source attribution reasoning among current LLMs.
- **Pipeline Not Fully Open-Sourced**: While the primary model uses the open-source DeepSeek r1, the source normalization step relies on proprietary GPT-4o API calls, hindering complete reproducibility.
- **Single Run Reporting**: Due to prohibitively high API costs (a single o1 run costs up to \$75), the FactBank experiments report results from single runs only, omitting variance estimates.
- **Lacking LLM Fine-Tuning**: All LLM experiments are conducted in the zero-shot setting without comparisons to fine-tuned LLMs, leaving the performance ceiling of tuned LLMs undetermined.
- **Cross-Lingual Performance Does Not Exceed SOTA**: On Italian ModaFact, the zero-shot Hybrid (63.6%) is slightly lower than the fine-tuned mT5-XXL (64.4%), although the margin is narrow.
- **Limited Dataset Scale**: The testing set of FactBank consists of only 280 sentences (1326 annotated instances), presenting a relatively small evaluation scale.

## Related Work & Insights

- **Event Factuality Corpora**: FactBank (Saurí and Pustejovsky, 2009) pioneered source-and-target belief annotation; MAVEN-Fact (Li et al., 2024) provides large-scale event factuality annotations; ModaFact (Rovera et al., 2025) introduces multimodal annotations for Italian belief detection.
- **Belief Prediction Methods**: Pouran Ben Veyseh et al. (2019) use GCN with BERT representations; Jiang and de Marneffe (2021) leverage RoBERTa with span representations; Murzaku and Rambow (2024)'s BeLeaf system uses a tree-generation approach with Flan-T5, constituting the previous SOTA on FactBank.
- **LLM Reasoning Capabilities**: Wei et al. (2022)'s CoT prompting is proven effective in this task; Li et al. (2024) explore LLM few-shot settings on MAVEN-Fact, yielding limited success.
- **Cross-Lingual Belief Detection**: Rovera et al. (2025)'s ModaFact uses mT5-XXL and Aya-23-8B fine-tuning, establishing strong baselines for multilingual belief detection.

## Rating

- ⭐⭐⭐⭐ Novelty: First zero-shot belief prediction evaluation, excellently designed hybrid strategy, first proposal of the Nested F1 metric.
- ⭐⭐⭐ Practicality: Exposing systematic limitations of LLMs in belief comprehension, though nested belief performance is currently too low for practical deployment.
- ⭐⭐⭐⭐ Experimental Thoroughness: 7 models × 2 approaches × cross-lingual validation × ablation studies × exhaustive error analyses.
---
title: >-
  [Paper Reading] Zero-Shot Belief: A Hard Problem for LLMs
description: >-
  [ACL 2025][LLM/NLP] This paper proposes two zero-shot frameworks, Unified and Hybrid, for source-and-target belief prediction. The hybrid approach utilizing a DeBERTa event tagger coupled with an LLM achieves a new SOTA (72.0% Full F1) on FactBank, while revealing that nested belief prediction (Nested F1 at only 25.3%) remains highly challenging for LLMs.
tags:
  - ACL 2025
  - LLM/NLP
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] Bilingual Zero-Shot Stance Detection](bilingual_zero-shot_stance_detection.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[AAAI 2026\] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts](../../AAAI2026/llm_nlp/soft_filtering_guiding_zero-shot_composed_image_retrieval_with_prescriptive_and_.md)

</div>

<!-- RELATED:END -->
