---
title: >-
  [Paper Note] Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal
description: >-
  [ACL 2026][LLM (Other)][cloze surprisal] This paper systematically compares the explanatory power of cloze responses and GPT-2 surprisal regarding human word-by-word reading times. Through three types of probability interventions, it demonstrates that the advantage of LM surprisal primarily stems from higher probability resolution, the ability to distinguish
tags:
  - ACL 2026
  - LLM (Other)
  - cloze surprisal
  - LM surprisal
date: 2026-05-08
content_hash: 0875cb0e3004399e
---
# Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal

**Conference**: ACL2026  
**arXiv**: [2601.09886](https://arxiv.org/abs/2601.09886)  
**Code**: https://github.com/sathvikn/cloze-surprisal  
**Area**: NLP Understanding / Psycholinguistics / LM surprisal  
**Keywords**: cloze surprisal, LM surprisal, reading times, linguistic prediction, cognitive modeling

## TL;DR
This paper systematically compares the explanatory power of cloze responses and GPT-2 surprisal regarding human word-by-word reading times. Through three types of probability interventions, it demonstrates that the advantage of LM surprisal primarily stems from higher probability resolution, the ability to distinguish semantically similar words, and the capacity to assign fine-grained probabilities to low-frequency words.

## Background & Motivation
**Background**: Research in language understanding commonly uses surprisal to describe the predictability of a word in context, utilizing it to predict processing effort such as reading times and eye movements. Traditional methods rely on the cloze task, where humans complete the next word; in recent years, more studies have shifted to using language model (LM) conditional probabilities.

**Limitations of Prior Work**: LM surprisal often outperforms cloze surprisal in fitting reading times, leading some to advocate for the direct replacement of cloze with LMs. However, the two originate differently: cloze comes from offline human production with small sample sizes (typically dozens to a hundred), while LMs are trained on large-scale corpora and can assign fine-grained probabilities to arbitrary continuations.

**Key Challenge**: The fact that LM probabilities fit human data better does not necessarily mean they are better "because they are more like human prediction." The improvement might simply be due to higher probability resolution, more comprehensive vocabulary coverage, larger training corpora, or the capture of statistical structures unrelated to human prediction.

**Goal**: The study aims to first replicate the advantage of GPT-2 surprisal over cloze surprisal in reading time data, and then explain $where$ this advantage comes from through targeted probability manipulations, avoiding the misinterpretation of higher fit as stronger cognitive plausibility.

**Key Insight**: Rather than just comparing two predictors, the authors artificially degrade or rewrite certain capabilities of GPT-2 probabilities: reducing probability resolution to match cloze sample counts, clustering semantically similar words, and allowing probabilities only for high-frequency words, thereby observing the resulting decline in fit.

**Core Idea**: By intervening in the resolution, semantic differentiation, and low-frequency coverage of LM probabilities, the study backtracks why LM surprisal is more predictive of reading times than cloze surprisal.

## Method
The paper includes three experiments. Experiment 1 compares the incremental explanatory power of cloze and GPT-2 surprisal on reading times; Experiment 2 tests the sources of advantage through three types of interventions on GPT-2 probabilities; Experiment 3 attempts to combine cloze responses with LM probabilities using similarity-adjusted surprisal.

### Overall Architecture
The input consists of word-by-word contexts and target words from multiple English reading time datasets. The authors calculate cloze-based predictability and GPT-2-based predictability for each word, then incorporate them as predictors into linear mixed-effects regressions. The explanatory power for reading times is measured using held-out log-likelihood. The core comparison evaluates whether "cloze only," "GPT-2 only," or "both" explains more variance.

### Key Designs

**1. Constructing a strong cloze baseline: Optimizing cloze before identifying LM advantages**

If the cloze processing itself is weak, the "LM fits better" conclusion might simply result from an unfair starting point. Cloze probabilities face two inherent issues: words not completed by any participant receive zero counts, and there is no consensus on whether raw probability should take a negative logarithm. To address this, the authors systematically searched these degrees of freedom—smoothing parameters $V\in\{50,100,200,500,1000,2000\}$ for add-one smoothing, and 6 functional forms covering raw probability, raw surprisal, and various surprisal power transforms. The combination fitting best across six reading time measures was $S(w_t)^2$ with $V=200$, which was set as the cloze baseline for subsequent comparisons. Only by optimizing cloze can the subsequent GPT-2 superiority be confirmed as genuine rather than an artifact of sloppy cloze processing.

**2. Three types of GPT-2 probability interventions: Artificially "cutting" LM capabilities to observe fit degradation**

After replicating GPT-2's superiority over cloze, the question is which specific capabilities drive this advantage. The authors did not compare new predictors but applied three targeted reductions to GPT-2 probabilities, each corresponding to a hypothesis: H1 (resolution) resamples the GPT-2 distribution based on cloze sample sizes and estimates probability via count-and-divide, dragging the LM down to the low resolution of cloze; H2 (semantics) uses GPT-2 token embeddings for k-means clustering and replaces target word probabilities with the probability of their semantic cluster, erasing fine distinctions between synonyms like couch/sofa; H3 (frequency) zeros out probabilities for low-frequency tokens and renormalizes over the high-frequency vocabulary, removing the LM's ability to assign fine-grained probabilities to rare continuations. Any intervention that leads to a significant drop in reading time fit identifies that specific capability as a source of the LM's advantage.

**3. Similarity-adjusted surprisal integration: Allowing similar candidates to "count," testing cloze-LM complementarity**

A major flaw of count-and-divide is its requirement for exact matches: if a human completes "sofa" but the target word is "couch," the target word probability is recorded as zero despite semantic equivalence. Similarity-adjusted (SA) surprisal attempts to fix this by weighting probability mass according to the embedding distance between candidate responses and the target word:

$$P_S(w_t\mid context)=\sum_{w'\in R} z(w_t,w')\,P(w'\mid context),$$

where $z(w_t,w')$ is the similarity weight and $R$ is the candidate set. The authors tested $R$ using both cloze responses and GPT-2 samples. Intuitively, this should incorporate "predicting a synonym" into processing facilitation, but the experiments showed that both versions fitted reading times worse. This negative result suggests that merging cloze and LM requires more refined cognitive hypotheses rather than a simple embedding distance metric.

### Loss & Training
No neural models were trained in this study; statistical modeling employed linear mixed-effects regression. For each reading-time measure, a baseline was built including control variables such as word length, position, unigram surprisal, and whether the previous word was fixated. Cloze or GPT-2 predictors were then added. Model fit was evaluated using held-out log-likelihood from 10-fold cross-validation, with significance determined by paired permutation tests and Bonferroni correction for multiple comparisons.

## Key Experimental Results

### Main Results
| Experiment / Setting | Key Results | Explanation |
|-------------|----------|------|
| Cloze transform search | $S(w_t)^2$ + $V=200$ highest log-likelihood gain (~153.8) | Cloze probability is significantly better as surprisal; linear probability is insufficient. |
| Cloze vs GPT-2 | GPT-2 significantly out-performed cloze in 4/6 measures; reverse not true. | GPT-2 surprisal typically subsumes cloze surprisal, especially in eye-tracking. |
| H1 resolution | Fit dropped significantly after downsampling GPT-2 to cloze sample sizes. | A large part of LM advantage comes from high-resolution probability. |
| H2 semantics | Fit dropped after merging semantically similar words via embedding clusters. | Fine-grained distinction by LMs between similar words (couch/sofa) helps fit RT. |
| H3 frequency | Fit dropped after restricting to high-frequency vocabulary. | Fine-grained probabilities for low-frequency continuations also contribute. |

| Dataset | Reading Time Measure | Cloze response size | Description |
|--------|--------------|--------------------|------|
| BK21 SPR | self-paced reading | ~90 responses per sentence | Manipulated high/moderate/low cloze with same target words. |
| Provo ET | first-pass / go-past | ~40 responses per word | 55 short paragraphs; 478 participants provided cloze. |
| UCL SPR/ET | SPR / first-pass / go-past | ~80 responses per word | 205/361 sentences; short sentences with many high-frequency words. |

### Ablation Study
| Configuration | Key Metric | Description |
|------|----------|------|
| Raw cloze probability | log-likelihood gain ~92 | Inferior to surprisal transformations. |
| Raw cloze surprisal | log-likelihood gain ~150 | Negative log probability is a more reasonable predictor of processing effort. |
| Cloze $S^2$ + V=200 | log-likelihood gain ~153.8 | The cloze baseline adopted by the authors. |
| Manipulated GPT-2 variants | All three significantly lower than original GPT-2 | All three hypotheses (H1, H2, H3) were supported. |
| SA cloze / SA GPT2 | Poor fit for RT | Simple similarity weighting failed to improve cloze-LM integration. |

### Key Findings
- The advantage of LM surprisal is not due to a single cause but the combined effect of high-resolution probability, semantic differentiation, and coverage of low-frequency word probabilities.
- When GPT-2 probabilities are downsampled to cloze-like estimates, they no longer consistently outperform cloze surprisal, indicating that the sample resolution limit of the cloze task is critical.
- The negative results of SA surprisal are equally valuable: simple weighting of similar candidates is not necessarily better than count-and-divide, suggesting that merging cloze and LM requires more sophisticated cognitive hypotheses.

## Highlights & Insights
- The paper does not stop at the empirical fact that "LMs fit better" but performs a clean mechanistic decomposition, which is vital for using LLM metrics in psycholinguistics.
- The design of H1 is particularly intuitive: forcing GPT-2 to provide limited samples like a cloze task immediately exposes the impact of cloze's low resolution.
- This paper serves as a reminder that a good predictor of human reading times is not necessarily the predictor actually used by the human brain; further argumentation is needed between fit and cognitive plausibility.

## Limitations & Future Work
- All experiments are based on English models and native speaker reading time data; the results for cross-linguistic settings, different writing systems, and multilingual LMs remain unclear.
- GPT-2-small is a classic psycholinguistic baseline but does not represent modern LLMs; larger models, instruction-tuned models, or different tokenizers might present different probability characteristics.
- The probability manipulations in H2/H3 are based on token-level processing and embedding clusters, which the authors admit may be coarse; future work could use more cognitively plausible semantic spaces and frequency modeling.
- The cloze task itself is an offline production task; while collecting more responses might increase resolution, it may not resolve the mechanistic differences between real-time prediction and explicit production.

## Related Work & Insights
- **vs. Traditional cloze surprisal**: Cloze comes directly from humans but suffers from small samples, zero probabilities, and offline production bias; LM surprisal offers high resolution but its cognitive source is opaque.
- **vs. LM surprisal studies by Shain / Michaelov**: While those works emphasized that LM surprisal predicts reading times better, this study asks "why" and points out that better fit cannot be simply equated with being more human-like.
- **vs. Similarity-adjusted surprisal**: SA surprisal attempts to incorporate semantically similar candidates into prediction facilitation, but this study shows simple versions perform poorly, inspiring more structured human prediction models.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Not a new model, but the use of intervention experiments to explain LM surprisal advantages is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Multiple datasets, measures, and manipulations with rigorous statistical comparison; model scope could be expanded.
- Writing Quality: ⭐⭐⭐⭐☆ Psycholinguistic problems are clearly explained with tight experimental logic; statistical details are extensive but necessary.
- Value: ⭐⭐⭐⭐☆ Highly valuable as a reference for those using LLM probabilities for cognitive modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](../../ACL2025/llm_nlp/token_granularity_impact.md)
- [\[ACL 2026\] Expect the Unexpected? Testing the Surprisal of Salient Entities](expect_the_unexpected_testing_the_surprisal_of_salient_entities.md)
- [\[ACL 2026\] An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal](an_existence_proof_for_neural_language_models_that_can_explain_garden-path_effec.md)
- [\[ACL 2026\] Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models](mind_the_gap_how_elicitation_protocols_shape_the_stated-revealed_preference_gap_.md)
- [\[ACL 2026\] Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)

</div>

<!-- RELATED:END -->
