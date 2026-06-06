---
title: >-
  [Paper Note] Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal
description: >-
  [ACL2026][LLM/NLP][cloze surprisal] The paper systematically compares cloze responses and GPT2 surprisal in their ability to explain human word-by-word reading times. Through three types of probabilistic interventions…
tags:
  - "ACL2026"
  - "LLM/NLP"
  - "cloze surprisal"
  - "LM surprisal"
  - "reading time"
  - "linguistic prediction"
  - "cognitive modeling"
date: 2026-05-08
content_hash: d54e0db4ce328acf
---

# Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal

**Conference**: ACL2026  
**arXiv**: [2601.09886](https://arxiv.org/abs/2601.09886)  
**Code**: https://github.com/sathvikn/cloze-surprisal  
**Area**: NLP Understanding / Psycholinguistics / LM surprisal  
**Keywords**: cloze surprisal, LM surprisal, reading time, linguistic prediction, cognitive modeling

## TL;DR
The paper systematically compares cloze responses and GPT2 surprisal in their ability to explain human word-by-word reading times. Through three types of probabilistic interventions, it demonstrates that the advantage of LM surprisal primarily stems from higher probability resolution, the ability to distinguish semantically similar words, and the assignment of fine-grained probabilities to low-frequency words.

## Background & Motivation
**Background**: Language understanding research frequently employs surprisal to describe the predictability of words in context, using it to predict processing efforts such as reading time and eye-tracking measures. Traditionally, this relies on the cloze task, where humans complete the next word; recently, more studies have shifted to using language model (LM) conditional probabilities.

**Limitations of Prior Work**: LM surprisal often outperforms cloze surprisal in fitting reading times, leading some to advocate for the direct replacement of cloze with LMs. However, the two originate differently: cloze comes from human offline single-instance production with typical sample sizes of dozens to a hundred, while LMs are trained on massive corpora and can assign fine-grained probabilities to any continuation.

**Key Challenge**: Better fitting of human data by LM probabilities does not necessarily mean they are better "because they are more human-like in their predictions." The advantage might simply arise from higher probability resolution, more comprehensive vocabulary coverage, larger training corpora, or capturing statistical structures irrelevant to human prediction.

**Goal**: To first replicate the advantage of GPT2 surprisal over cloze surprisal in explaining reading times, and then explain the source of this advantage through targeted probability manipulations, avoiding the misinterpretation of higher fit as stronger cognitive plausibility.

**Key Insight**: The authors do not merely compare two predictors; they artificially degrade or rewrite certain capabilities of GPT2 probabilities—reducing resolution to cloze sample sizes, clustering semantically similar words, or allowing only high-frequency word probabilities—to observe the resulting decline in fit.

**Core Idea**: By intervening in the resolution, semantic differentiation, and low-frequency word coverage of LM probabilities, the study reverse-engineers why LM surprisal predicts reading times more effectively than cloze surprisal.

## Method
The study comprises three experiments. Experiment 1 compares the incremental explanatory power of cloze versus GPT2 surprisal for reading times; Experiment 2 applies three types of interventions to GPT2 probabilities to test the sources of advantage; Experiment 3 attempts to combine cloze responses with LM probabilities using similarity-adjusted surprisal.

### Overall Architecture
The input consists of word-by-word contexts and target words from multiple English reading-time datasets. The authors calculate cloze-based predictability and GPT2-based predictability for each word, then incorporate them as predictors into linear mixed-effects regression. The cross-validated held-out log-likelihood is used to measure the variance explained in reading times. The core comparison determines which model—"cloze only," "GPT2 only," or "both"—explains more variance.

### Key Designs
1. **Strong cloze baseline construction**:
    - **Function**: Prevents basing LM advantages on an unfairly weak cloze baseline.
    - **Mechanism**: The authors systematically search add-one smoothing parameters $V \in \{50, 100, 200, 500, 1000, 2000\}$ and six functional forms, including raw probability, raw surprisal, and various surprisal power transforms. Ultimately, $S(w_t)^2$ with $V=200$ is adopted as it yields the best fit across six reading-time metrics.
    - **Design Motivation**: Cloze probability suffers from zero counts and ambiguous functional forms. Without tuning smoothing/transformations, the comparison would be biased toward LMs.

2. **Three types of GPT2 probability interventions**:
    - **Function**: Decomposing the sources of GPT2 surprisal's advantage.
    - **Mechanism**: **H1 resolution** downsamples the GPT2 distribution to match the number of cloze responses, estimating probability via count-and-divide; **H2 semantics** uses k-means on GPT2 token embeddings to replace target word probability with its cluster probability; **H3 frequency** zeroes out probabilities for low-frequency tokens and renormalizes over a high-frequency vocabulary.
    - **Design Motivation**: If a specific intervention significantly degrades the fit to reading time, it indicates that the corresponding capability of GPT2 is a source of its advantage.

3. **Similarity-adjusted surprisal combination attempts**:
    - **Function**: Exploring whether cloze response sets and LM probabilities are complementary.
    - **Mechanism**: SA surprisal weights probability mass by the embedding distance between candidate responses and the target word: $P_S(w_t|context) = \sum_{w' \in R} z(w_t, w') P(w'|context)$. The authors use cloze responses and GPT2 samples as candidate sets respectively.
    - **Design Motivation**: If a human predicts "sofa" while the target is "couch," traditional count-and-divide assigns low probability to the target; SA surprisal attempts to incorporate the predictability of similar candidates.

### Loss & Training
No neural models are trained in this study; statistical modeling employs linear mixed-effects regression. For each reading-time measure, a baseline is established including control variables such as word length, position, unigram surprisal, and whether the word was fixated in eye-tracking. Cloze or GPT2 predictors are then added. Quality of fit is measured using 10-fold cross-validation held-out log-likelihood. Significance is tested via paired permutation tests with Bonferroni correction for multiple comparisons.

## Key Experimental Results

### Main Results
| Experiment / Setting | Key Results | Explanation |
| :--- | :--- | :--- |
| Cloze transform search | $S(w_t)^2$ + $V=200$ achieves highest log-likelihood gain (~153.8) | Surprisal transforms are significantly better than linear probability for cloze. |
| Cloze vs GPT2 | GPT2 surprisal significantly outperforms cloze in 4/6 measures | GPT2 surprisal usually subsumes cloze surprisal, especially in eye-tracking. |
| H1 resolution | Downsampling GPT2 to cloze sample sizes significantly degrades fit | A large portion of LM advantage comes from high-resolution probabilities. |
| H2 semantics | Merging semantically similar words via clusters degrades fit | Fine-grained distinction between words like "couch"/"sofa" helps fit RTs. |
| H3 frequency | Restricting to high-frequency vocabulary degrades fit | Fine-grained probabilities for low-frequency continuations also contribute. |

| Dataset | Reading Time Measure | Cloze response scale | Description |
| :--- | :--- | :--- | :--- |
| BK21 SPR | self-paced reading | ~90 responses/sentence | Manipulated high/moderate/low cloze, same target word. |
| Provo ET | first-pass / go-past | ~40 responses/word | 55 short passages, 478 participants for cloze. |
| UCL SPR/ET | SPR / first-pass / go-past | ~80 responses/word | 205/361 sentences, short sentences with frequent words. |

### Ablation Study
| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Raw cloze probability | Log-likelihood gain ~92 | Inferior to surprisal transformations. |
| Raw cloze surprisal | Log-likelihood gain ~150 | Negative log-probability is a more plausible processing effort predictor. |
| Cloze $S^2$ + V=200 | Log-likelihood gain ~153.8 | The chosen cloze baseline for subsequent steps. |
| Manipulated GPT2 variants| All three sig. lower than original GPT2 | Supports all three hypotheses for LM advantage. |
| SA cloze / SA GPT2 | Poor fit for RT | Simple similarity-weighting failed to improve the cloze-LM combination. |

### Key Findings
- The advantage of LM surprisal is not due to a single factor but is driven by high-resolution probability, semantic differentiation, and probability coverage of low-frequency words.
- When GPT2 probabilities are downsampled to cloze-like estimates, they no longer consistently outperform cloze surprisal, indicating that the sample resolution of the cloze task is a critical limitation.
- The negative results for SA surprisal suggest that simple similarity weighting is not necessarily better than count-and-divide; the fusion of cloze and LM requires more sophisticated cognitive hypotheses.

## Highlights & Insights
- The paper moves beyond the empirical fact that "LMs fit better" to perform a clean mechanical decomposition, which is vital for using LLM metrics in psycholinguistics.
- The design of H1 is particularly intuitive: by forcing GPT2 to provide only a limited number of samples like a cloze task, the impact of cloze's low resolution is immediately exposed.
- This research serves as a reminder that a "good predictor" of human reading time is not necessarily the predictor actually used by the human brain; the link between fit and cognitive plausibility requires additional justification.

## Limitations & Future Work
- All experiments are based on English models and English native speaker reading-time data; conclusions for cross-linguistic settings, different writing systems, or multilingual LMs remain unclear.
- GPT2-small is a classic psycholinguistic baseline, but it may not represent modern LLMs; larger models, instruction-tuned models, or different tokenizers might exhibit different probabilistic characteristics.
- The manipulations for H2/H3 are based on token-level processing and embedding clusters; the authors acknowledge these may be coarse and could be improved with more cognitively plausible semantic spaces.
- The cloze task is inherently an offline production task; collecting more responses may increase resolution but does not necessarily resolve the mechanistic differences between real-time prediction and explicit production.

## Related Work & Insights
- **vs. traditional cloze surprisal**: Cloze comes directly from humans but suffers from small samples, many zero-probabilities, and offline production bias; LM surprisal offers high resolution but its cognitive origins are opaque.
- **vs. Shain / Michaelov et al. (LM surprisal studies)**: While prior work emphasizes that LM surprisal predicts reading times better, this study investigates *why* and cautions against equating better fit with higher human-likeness.
- **vs. similarity-adjusted surprisal**: While SA surprisal attempts to include similar candidates in predictability, this study's simple implementation did not outperform count-and-divide, suggesting a need for more structured models of human prediction.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Not a new model, but the use of intervention experiments to explain LM surprisal advantages is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Multiple datasets, multiple metrics, and various manipulations with rigorous statistical comparisons.
- Writing Quality: ⭐⭐⭐⭐☆ Psycholinguistic problems are clearly explained, and the experimental chain is tightly linked.
- Value: ⭐⭐⭐⭐☆ Highly valuable reference for those using LLM probabilities for cognitive modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Expect the Unexpected? Testing the Surprisal of Salient Entities](expect_the_unexpected_testing_the_surprisal_of_salient_entities.md)
- [\[ACL 2026\] An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal](an_existence_proof_for_neural_language_models_that_can_explain_garden-path_effec.md)
- [\[ACL 2026\] Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models](mind_the_gap_how_elicitation_protocols_shape_the_stated-revealed_preference_gap_.md)
- [\[ICML 2026\] "I've Seen How This Goes": Characterizing LLM and Human Writing Diversity via Progressive Conditional Surprisal](../../ICML2026/llm_nlp/ive_seen_how_this_goes_characterizing_diversity_via_progressive_conditional_surp.md)
- [\[ACL 2026\] Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)

</div>

<!-- RELATED:END -->
