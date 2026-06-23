---
title: >-
  [Paper Note] Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal
description: >-
  [ACL 2026][LLM (Other)][cloze surprisal] The paper systematically compares cloze responses and GPT-2 surprisal regarding their explanatory power for human word-by-word reading times. Through three types of probabilistic interventions, it demonstrates that the advantage of LM surprisal primarily stems from higher probability resolution, the ability to distingu
tags:
  - ACL 2026
  - LLM (Other)
  - cloze surprisal
  - LM surprisal
date: 2026-05-08
content_hash: d70bcb3711e36e7c
---
# Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal

**Conference**: ACL2026  
**arXiv**: [2601.09886](https://arxiv.org/abs/2601.09886)  
**Code**: https://github.com/sathvikn/cloze-surprisal  
**Area**: NLP Understanding / Psycholinguistics / LM surprisal  
**Keywords**: cloze surprisal, LM surprisal, reading time, language prediction, cognitive modeling

## TL;DR
The paper systematically compares cloze responses and GPT-2 surprisal regarding their explanatory power for human word-by-word reading times. Through three types of probabilistic interventions, it demonstrates that the advantage of LM surprisal primarily stems from higher probability resolution, the ability to distinguish semantically similar words, and the assignment of fine-grained probabilities to low-frequency words.

## Background & Motivation
**Background**: Language understanding research frequently utilizes surprisal to describe word predictability within a context, using it to predict processing effort such as reading times and eye-tracking movements. Traditional approaches rely on the cloze task, where humans complete the next word; in recent years, an increasing number of studies have shifted to using conditional probabilities from language models (LMs).

**Limitations of Prior Work**: LM surprisal often outperforms cloze surprisal in fitting reading times, leading some to advocate for the direct replacement of cloze with LMs. However, the sources differ: cloze data originates from offline human production with typically small sample sizes (dozens to a hundred), whereas LMs are trained on massive corpora and can assign fine-grained probabilities to any arbitrary continuation.

**Key Challenge**: Better fitting of human data by LM probabilities does not necessarily mean it succeeds "because it is more similar to human prediction." It may simply be due to higher probability resolution, more comprehensive vocabulary coverage, larger training corpora, or the capturing of statistical structures unrelated to human cognitive processing.

**Goal**: Ours aims to first replicate the advantage of GPT-2 surprisal over cloze surprisal and then explain the source of this advantage through targeted probability manipulations to avoid misinterpreting higher fit as stronger cognitive plausibility.

**Key Insight**: Instead of merely comparing two predictors, the authors artificially degrade or rewrite specific capabilities of GPT-2 probabilities—reducing resolution to match cloze sample sizes, clustering semantically similar words, and restricting probabilities to high-frequency words—to observe the resulting decline in fit.

**Core Idea**: By intervening in the resolution, semantic differentiation, and low-frequency coverage of LM probabilities, the study back-infers why LM surprisal predicts reading times more effectively than cloze surprisal.

## Method
The paper includes three experiments. Experiment 1 compares the incremental explanatory power of cloze and GPT-2 surprisal; Experiment 2 applies three types of interventions to GPT-2 probabilities; Experiment 3 attempts to combine cloze responses with LM probabilities using similarity-adjusted surprisal.

### Overall Architecture
The input consists of word-by-word contexts and target words from multiple English reading time datasets. For each word, cloze-based predictability and GPT-2-based predictability are calculated. These are then added as predictors into linear mixed-effects regressions, using held-out log likelihood to measure the explanatory power for reading times. The core comparison evaluates whether "cloze only," "GPT-2 only," or "both" explains more variance.

### Key Designs

**1. Strong cloze baseline construction: Optimizing the cloze side before examining LM advantages**

If the cloze processing itself is weak, the superior fit of the LM might simply be due to a poor baseline. Cloze probabilities have two inherent issues: words not produced by any participant receive a zero count, and it remains debated whether raw probabilities should take a negative logarithm. To address this, the authors systematically searched two degrees of freedom: smoothing parameters $V\in\{50,100,200,500,1000,2000\}$ for add-one smoothing, and functional forms including raw probability, raw surprisal, and various surprisal power transforms (6 types in total). The combination that fit best across six reading time measures was $S(w_t)^2$ with $V=200$, which was established as the cloze baseline. Only by tuning cloze to its optimal state can a subsequent GPT-2 victory be documented as a genuine advantage rather than an artifact of superficial cloze processing.

**2. Three types of GPT-2 probability interventions: Artificially removing LM capabilities to observe the impact on reading time fit**

After replicating the GPT-2 advantage, the specific source must be identified. Rather than introducing new predictors, the authors applied three directional degradations to GPT-2 probabilities, each testing a hypothesis: H1 (resolution) resamples the GPT-2 distribution based on the cloze response sample size and estimates probability via count-and-divide, reducing the LM to the same low resolution as cloze; H2 (semantics) uses GPT-2 token embeddings for k-means clustering and replaces target word probabilities with the probability of its semantic cluster, erasing distinctions between synonyms like couch/sofa; H3 (frequency) sets probabilities of low-frequency tokens to zero and renormalizes over the high-frequency vocabulary, removing the LM's ability to assign probabilities to rare continuations. If an intervention significantly reduces the fit, it indicates that the removed capability is a source of the LM advantage. All three interventions resulted in significant declines in fit.

**3. Similarity-adjusted surprisal combination: Allowing similar candidates to count as "partial hits"**

A major weakness of count-and-divide is its requirement for exact matches: if humans suggest "sofa" but the target is "couch," the target probability is zero despite semantic equivalence. Similarity-adjusted (SA) surprisal attempts to fix this by weighting probability mass based on the embedding distance between candidate responses and the target word:

$$P_S(w_t\mid context)=\sum_{w'\in R} z(w_t,w')\,P(w'\mid context),$$

where $z(w_t,w')$ represents similarity weights and $R$ is the set of candidates (cloze responses or GPT-2 samples). Intuitively, this should incorporate the processing ease of predicted synonyms into the measure. However, both versions performed worse in fitting reading times. This negative result suggests that a simple similarity weighting cannot bridge cloze and LM data; their integration likely requires more refined cognitive hypotheses rather than simple embedding distance.

### Loss & Training
Ours does not train neural models; statistical modeling utilizes linear mixed-effects regression. For each reading-time measure, a baseline is constructed including control variables (word length, position, unigram surprisal, and whether the previous word was fixated). Predictors (cloze or GPT-2) are then added. Model performance is evaluated using 10-fold cross-validation held-out log likelihood, with significance determined by paired permutation tests and multi-comparison correction via Bonferroni correction.

## Key Experimental Results

### Main Results

| Experiment / Setting | Key Results | Explanation |
|-------------|----------|------|
| Cloze transform search | $S(w_t)^2$ + $V=200$ yielded the highest log-likelihood Gain (approx. 153.8) | Transforming cloze probability to surprisal is significantly better; linear probability is insufficient. |
| Cloze vs GPT-2 | GPT-2 surprisal was significantly superior to cloze in 4 out of 6 measures; the reverse was not true. | GPT-2 surprisal typically subsumes cloze surprisal, particularly in eye-tracking. |
| H1 resolution | Fit significantly decreased after downsampling GPT-2 to cloze sample sizes. | A large portion of the LM advantage stems from high-resolution probabilities. |
| H2 semantics | Fit decreased after merging semantically similar words by embedding clusters. | Fine-grained distinction between synonyms (e.g., couch/sofa) helps fit RT. |
| H3 frequency | Fit decreased when restricted to high-frequency vocabularies. | Assigning fine-grained probabilities to low-frequency continuations also contributes. |

| Dataset | Reading Time Measures | Cloze Response Scale | Description |
|--------|--------------|--------------------|------|
| BK21 SPR | self-paced reading | ~90 responses/sentence | Manipulated high/moderate/low cloze for the same target word. |
| Provo ET | first-pass / go-past | ~40 responses/word | 55 short passages; cloze provided by 478 participants. |
| UCL SPR/ET | SPR / first-pass / go-past | ~80 responses/word | 205/361 sentences; short sentences with many high-frequency words. |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|----------|------|
| Raw cloze probability | Log-likelihood Gain ~92 | Inferior to surprisal transforms. |
| Raw cloze surprisal | Log-likelihood Gain ~150 | Negative log probability is a more reasonable processing effort predictor. |
| Cloze $S(w_t)^2$ + V=200 | Log-likelihood Gain ~153.8 | The cloze baseline adopted for subsequent components. |
| Manipulated GPT-2 variants | All 3 variants significantly lower than original GPT-2 | All three hypotheses were supported. |
| SA cloze / SA GPT2 | Poor fit for RT | Simple similarity weighting failed to improve the cloze-LM combination. |

### Key Findings
- The advantage of LM surprisal is not due to a single cause but is the result of high-resolution probabilities, semantic differentiation, and probability coverage of low-frequency words.
- When GPT-2 probabilities are downsampled to cloze-like estimates, they are no longer consistently superior to cloze surprisal, indicating that the sample resolution limit of the cloze task is critical.
- The negative result of SA surprisal is equally valuable: simple weighting of similar candidates is not necessarily better than count-and-divide, suggesting the fusion of cloze and LM requires more sophisticated cognitive hypotheses.

## Highlights & Insights
- Instead of stopping at the empirical fact that "LMs fit better," the paper performs a clean mechanistic decomposition, which is crucial for using LLM metrics in psycholinguistics.
- The design of H1 is particularly intuitive: forcing GPT-2 to provide limited samples like a cloze task immediately expose the impact of low resolution in cloze data.
- This paper serves as a reminder that a good predictor of human reading time is not necessarily the predictor actually used by the human brain; the link between fit and cognitive plausibility requires additional justification.

## Limitations & Future Work
- All experiments are based on English models and native English speaker reading time data; the generalizability across languages, writing systems, and multilingual LMs remains unclear.
- GPT-2-small is a classic psycholinguistic baseline but does not represent modern LLMs; larger models, instruction-tuned models, or different tokenizers might exhibit different probabilistic characteristics.
- The probability manipulations in H2/H3 are based on token-level processing and embedding clusters, which the authors acknowledge may be coarse. Future work could use more cognitively plausible semantic spaces and frequency modeling.
- The cloze task itself is an offline production task; collecting more responses might improve resolution but does not necessarily resolve the mechanistic differences between real-time prediction and explicit production.

## Related Work & Insights
- **vs. Traditional cloze surprisal**: Cloze data comes directly from humans but suffers from small samples, zero probabilities, and offline production biases; LM surprisal offers high resolution but its cognitive origin is opaque.
- **vs. Studies by Shain / Michaelov et al.**: While prior work emphasizes that LM surprisal predicts reading times better, this study asks *why* it is better and points out that better fit cannot be simply equated with being more human-like.
- **vs. Similarity-adjusted surprisal**: SA surprisal attempts to incorporate semantically similar candidates into prediction facilitation, but current results show simple versions underperform, inspiring the need for more structured human prediction models.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Not a new model, but the use of intervention experiments to explain the LM surprisal advantage is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Multiple datasets, measures, and manipulations with rigorous statistical comparisons; the model range could be further expanded.
- Writing Quality: ⭐⭐⭐⭐☆ Psycholinguistic issues are clearly articulated, the chain of experiments is tight, and statistical details are necessary and well-presented.
- Value: ⭐⭐⭐⭐☆ Significant cautionary and methodological value for researchers using LLM probabilities for cognitive modeling.

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
