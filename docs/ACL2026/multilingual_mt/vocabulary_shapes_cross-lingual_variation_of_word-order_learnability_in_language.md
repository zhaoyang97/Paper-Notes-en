---
title: >-
  [Paper Note] Vocabulary Shapes Cross-Lingual Variation of Word-Order Learnability in Language Models
description: >-
  [ACL2026][Multilingual & Machine Translation][word-order learnability] This paper uses the Mallows model to generate continuous word-order perturbation spectra for 10 European languages. After training small autoregressi…
tags:
  - "ACL2026"
  - "Multilingual & Machine Translation"
  - "word-order learnability"
  - "multilingual modeling"
  - "vocabulary structure"
  - "Mallows permutations"
  - "linguistic typology"
date: 2026-05-08
content_hash: 319c7a042ad8043b
---

# Vocabulary Shapes Cross-Lingual Variation of Word-Order Learnability in Language Models

**Conference**: ACL2026  
**arXiv**: [2603.19427](https://arxiv.org/abs/2603.19427)  
**Code**: https://gitlab.gwdg.de/huds/projects/shuffle/-/tree/v1.0.2  
**Area**: multilingual_mt  
**Keywords**: word-order learnability, multilingual modeling, vocabulary structure, Mallows permutations, linguistic typology

## TL;DR
This paper uses the Mallows model to generate continuous word-order perturbation spectra for 10 European languages. After training small autoregressive LMs, it is found that more irregular word orders are harder to learn, but cross-lingual differences are primarily explained by vocabulary coverage, sentence length, and morphological complexity, rather than simple free/fixed word-order labels.

## Background & Motivation
**Background**: Linguistic typology has long focused on why different languages have different word orders—for instance, English relies heavily on SVO positions, while languages like Czech and Finnish allow freer word order through case marking and morphological variation. In NLP, synthetic language or word-order perturbation experiments are used to investigate whether language models possess natural language inductive biases by shuffling word order.

**Limitations of Prior Work**: Existing experiments often conflate word order, morphology, and tokenization. Many works shuffle at the subword level, which breaks morphological units, simultaneously destroying both word order and morphological structure. Others use discrete perturbation intensities, making it difficult to compare continuous transitions from "slight local shuffling" to "complete random shuffling." Furthermore, language selection is often biased toward English, making cross-lingual differences hard to explain.

**Key Challenge**: Free word-order languages typically have richer morphology, while fixed word-order languages rely more on position. This natural correlation makes it difficult to determine whether poor learnability stems from the word order itself, morphological complexity, vocabulary statistics, or the tokenizer's encoding efficiency for different languages.

**Goal**: The authors aim to construct a cleaner experiment: only change word-order regularity while preserving the original vocabulary, morphology, and global entropy. They observe how model surprisal varies with perturbation intensity and explain why different languages exhibit varying robustness to word-order perturbations.

**Key Insight**: The paper uses the Mallows permutation model to sample a deterministic permutation for each sentence length. Controlling the parameter $\theta$ allows a continuous coverage of original order, local shuffling, complete randomness, local reversal, and total inversion. This provides a complete word-order regularity spectrum for each language rather than just a few discrete shuffled versions.

**Core Idea**: Isolate word-order factors using word-level deterministic shuffling, then explain cross-lingual variations in model surprisal using vocabulary statistics and PLS regression.

## Method

### Overall Architecture
The workflow starts with the Europarl multilingual parallel corpus. Ten European languages are selected, covering fixed vs. free word order and morphological types such as analytic, fusional, and agglutinative. Each language undergoes uniform cleaning: lowercasing, punctuation removal, and filtering of sentences exceeding 80 words. The data is split into 650,000 training, 5,000 validation, and 5,000 test sentences.

Subsequently, a series of synthetic word-order variants are constructed. For each sentence length $n$, exactly one permutation $\pi^{(n)}$ is sampled and applied to all sentences of that length in the language. This transformation is deterministic, maintaining a low description length and avoiding the significant increase in model-independent entropy typical of stochastic sentence-by-sentence shuffling.

Finally, an identical small autoregressive Transformer (PicoLM style, 50.5M parameters, decoder-only) is trained from scratch for each language variant. ByteLevel-BPE is used with a default vocabulary size of $|V|=16,000$; vocabulary size experiments sweep from $|V|=258$ to $64,000$. Learnability is measured by mean test set surprisal; lower surprisal indicates the model better captures the probabilistic structure of the variant.

### Key Designs
1. **Mallows Continuous Word-Order Perturbation Spectrum**:
	- **Function**: Uses a continuous parameter to generate language variants ranging from original order to total randomness and reverse order.
	- **Mechanism**: The Mallows model assigns probability based on the distance between a permutation $\pi$ and the original order $\pi_0$: $P(\pi) \propto \exp(-\theta d(\pi,\pi_0))$. The distance $d$ uses Kendall's $\tau$. As $\theta \to \infty$, it approaches original order; at $\theta=0$, all permutations are equally likely; at $\theta \to -\infty$, it approaches complete reversal.
	- **Design Motivation**: Continuous perturbation allows plotting a full $S(\theta)$ curve. The symmetry of $\theta$ allows direct comparison between sentence reversal and equivalent local forward shuffling.

2. **Word-Level Deterministic Shuffling for Morphological Integrity**:
	- **Function**: Avoids subword shuffling that breaks internal word structures, allowing a cleaner study of word order.
	- **Mechanism**: Words are defined as whitespace-delimited orthographic units. Permutations are performed at the word level before tokenization. Using a single permutation per sentence length avoids introducing extra random information per sentence.
	- **Design Motivation**: Shuffling subwords might split a single morphological unit into separate positions, which may hinder learning due to destroyed morphology rather than word order. Word-level perturbation keeps vocabulary and morphology intact.

3. **Explaining Cross-Lingual Variation with Vocabulary Structure**:
	- **Function**: Identifies more granular explanatory variables than the "free/fixed" binary to explain robustness.
	- **Mechanism**: Metrics such as vocabulary coverage, subword coverage, coverage integral, word-subword coverage similarity, average word count, average subword count, fertility, and unique word types are extracted. Multivariate PLS regression predicts the surprisal curve across 28 $\theta$ values.
	- **Design Motivation**: These metrics are highly collinear with a small sample size (10 languages), making standard linear regression unstable. PLS compresses related features into latent components while retaining explanatory power for the multidimensional response $S(\theta)$.

### Loss & Training
Standard autoregressive language modeling is used as the measurement tool. The model objective is next-subword prediction. Evaluation uses $S(\theta) = \frac{1}{N} \sum_i -\log p(w_i|w_{<i})$. The focus is on the relative surprisal increase $\Delta S(\theta) = S(\theta) - S_{orig}$. Since deterministic shuffling essentially maintains model-independent entropy, $\Delta S$ can be interpreted as the model's sensitivity to word-order perturbation.

## Key Experimental Results

### Main Results
The first set of results focuses on word-order perturbation. Models exhibit sensitivity to irregular word orders across all languages, but additional sensitivity to sentence reversal is weak. The free/fixed word-order dichotomy fails to explain robustness under irregular conditions.

| Research Question | Key Metric / Phenomenon | Result | Explanation |
|----------|----------------|------|------|
| Is irregular word order harder to learn? | $\Delta S$ vs $\theta$ | $\Delta S$ peaks near $\theta=0$ | Autoregressive LMs have significant locality bias; random order is hardest to compress. |
| Is reversal harder than equivalent forward shuffling? | median asymmetry $\Delta S^{+/-}$ | 0.096 (~6% of irregular effect) | Reversal violates typological correlations, but models perceive locality more than linguistic legality. |
| Is the reversal difference significant? | Wilcoxon signed-rank | $p=0.0098$, significant but small effect | Statistically different, but limited practical magnitude. |
| Does free/fixed order explain $\Delta S_{irreg}$? | Wilcoxon-Mann-Whitney | $p=0.55$ | Coarse typological labels cannot distinguish robustness to irregular perturbations. |
| Subword vs word shuffling | Surprisal under irregular order | Subword shuffling higher; Balto-Slavic and Uralic see larger increases | Breaking morphological units distorts cross-lingual comparisons. |

### Ablation Study
The second set of results uses vocabulary structure and morphological proxies as explanatory variables, showing they have more explanatory power than binary word-order labels.

| Analysis Config | Metric | Value / Phenomenon | Description |
|----------|------|-------------|------|
| PLS with 2 latent components predicting $S(\theta)$ | overall explained variance | $R^2=0.97$ | Vocabulary and morphology metrics can reconstruct cross-lingual surprisal curves. |
| Leave-one-language-out | per-language $R^2$ | Hungarian 0.93, Portuguese / Latvian 0.99 | High predictive power for unseen languages. |
| Per-$\theta$ slice, two components | mean $\bar{R}^2$ | 0.79 (range 0.66-0.86) | Explanatory power is stable across perturbation levels. |
| Vocabulary component only | mean $\bar{R}^2$ | 0.65 (range 0.26-0.76) | Vocabulary coverage explains original and reverse performance. |
| Second component | Primary loadings | unique word types, word length, fertility | Irregular conditions require morphological complexity for explanation. |
| Vocabulary size sweep | Threshold | $|V|>8000$ | Original surprisal begins to distinguish free/fixed order only at larger vocabulary sizes. |

### Key Findings
- Highly irregular word order systematically increases surprisal, but models do not show a strong additional penalty for "reverse order." This suggests autoregressive LMs care more about whether local predictive structures are destroyed than whether a sentence follows human typological preferences.
- Free word-order languages are not inherently more robust to shuffling. Groups of free and fixed languages overlap significantly in $\Delta S_{irreg}$; only extremes like Romance and Finnic separate on both original surprisal and perturbation gain.
- Vocabulary coverage clusters free/fixed languages into a more continuous structure. Free word-order languages often have slower-growing coverage curves, implying more low-frequency forms are involved.
- Morphological complexity proxies primarily explain the high-irregularity region near $\theta=0$. In other words, vocabulary statistics explain "general learning difficulty," while morphological structure explains "recoverable cues under heavy shuffling."
- Vocabulary size itself changes the conclusions. Groups only clearly separate in original surprisal when $|V| > 8K$. However, for irregular order increments, languages either converge or diverge as vocabulary size continues to grow.

## Highlights & Insights
- The most clever design is deterministic word-level shuffling. It allows continuous control over word-order regularity while avoiding the increase in global entropy typical of random shuffling, providing a cleaner tool for linguistic experiments.
- The paper shifts the definition of "language difficulty" from coarse labels to quantifiable vocabulary structures. For multilingual LMs, this serves as a reminder that coverage and fertility formed by the tokenizer may more directly impact learning difficulty than language family or morphological type.
- The reversal experiment is insightful: models are not particularly sensitive to violations of natural language typological correlations. This suggests the inductive bias of small autoregressive LMs favors local predictability over human linguistic legality.
- The use of PLS is appropriate. Given only 10 languages and highly correlated variables, PLS is more stable than multiple univariate correlations and separates vocabulary components from morphological complexity components.

## Limitations & Future Work
- The corpus is limited to Europarl and European languages. While highly comparable, the typological space is narrow. Non-European families, non-SVO languages, and spoken/web text may exhibit different patterns.
- The model is a small 50.5M parameter autoregressive LM. Conclusions may not directly extrapolate to large multilingual LLMs where capacity and tokenizer scale might alter the relationship between coverage and learnability.
- The study uses LM surprisal to investigate computational learnability, which is not equivalent to human learning difficulty. Comparisons with eye-tracking or L2 acquisition data are needed.
- Free/fixed word order is still used as an initial grouping, though typography is continuous. Future work could use continuous metrics like subject-object entropy or dependency locality.
- Evaluation looks at global surprisal without breaking down which tokens or constructions contribute most. Analyzing language-specific structures like case markers or verb positions would make the explanations more actionable.

## Related Work & Insights
- **vs Kallini et al.**: They showed shuffling hurts models; this paper progresses to continuous cross-lingual spectra and avoids subword-level morphological destruction.
- **vs Cotterell et al. / Mielke et al.**: Previous work debated whether morphology or simple statistics explain LM difficulty. This paper suggests a middle ground: coverage and length explain most of the curve, while morphology is critical under strong irregular perturbations.
- **vs Arnett and Bergen**: This paper supports the view that vocabulary structure is a key mediator for cross-lingual differences but places it within word-order studies to show it affects both base difficulty and robustness to shuffling.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Beautiful experimental control that separates word order, vocabulary, and morphology.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Extensive training of small models across various perturbations and regressions, though language/model scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic; natural transitions between linguistic motivation, experimental control, and statistical interpretation.
- Value: ⭐⭐⭐⭐☆ Highly relevant for multilingual LM design and linguistic typology experiments; engineering utility is indirect but insightful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Training for Cross-lingual Speech Language Models](efficient_training_for_cross-lingual_speech_language_models.md)
- [\[ACL 2026\] LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models](llm-xtm_enhancing_cross-lingual_topic_models_with_large_language_models.md)
- [\[ACL 2026\] Why Low-Resource NLP Needs More Than Cross-Lingual Transfer: Lessons Learned from Luxembourgish](why_low-resource_nlp_needs_more_than_cross-lingual_transfer_lessons_learned_from.md)
- [\[ACL 2026\] IndoTabVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents](indotabvqa_a_benchmark_for_cross-lingual_table_understanding_in_bahasa_indonesia.md)
- [\[ACL 2026\] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)

</div>

<!-- RELATED:END -->
