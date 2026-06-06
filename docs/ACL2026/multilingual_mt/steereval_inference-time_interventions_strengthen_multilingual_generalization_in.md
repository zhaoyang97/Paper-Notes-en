---
title: >-
  [Paper Note] SteerEval: Inference-time Interventions Strengthen Multilingual Generalization in Neural Summarization Metrics
description: >-
  [ACL2026][Multilingual & Machine Translation][activation steering] SteerEval investigates aligning hidden representations of multilingual evaluation models toward high-resource pivot languages during inference. It finds…
tags:
  - "ACL2026"
  - "Multilingual & Machine Translation"
  - "activation steering"
  - "multilingual summarization evaluation"
  - "LLM-as-a-judge"
  - "COMET"
  - "English pivot language"
date: 2026-05-08
content_hash: f2fa09648059c6c7
---

# SteerEval: Inference-time Interventions Strengthen Multilingual Generalization in Neural Summarization Metrics

**Conference**: ACL2026  
**arXiv**: [2601.15809](https://arxiv.org/abs/2601.15809)  
**Code**: No public repository link provided in the paper  
**Area**: Multilingual Evaluation / Machine Translation and Summarization Evaluation  
**Keywords**: activation steering, multilingual summarization evaluation, LLM-as-a-judge, COMET, English pivot language

## TL;DR
SteerEval investigates aligning hidden representations of multilingual evaluation models toward high-resource pivot languages during inference. It finds that steering toward English or French generally improves the correlation between multilingual summarization automatic metrics and manual ratings, particularly for low-baseline languages and encoder-based COMET metrics.

## Background & Motivation
**Background**: Summarization and natural language generation tasks have long relied on automatic metrics as alternatives to expensive human evaluation. From BLEU and ROUGE to COMET and BERTScore, and recently LLM-as-a-judge, model-based metrics are increasingly common for English tasks and are gradually being adopted for multilingual evaluation.

**Limitations of Prior Work**: In multilingual scenarios, the correlation between model metrics and human judgment remains unstable. Particularly for languages such as Yoruba, Hebrew, and Turkish, some LLM scorers even exhibit near-zero or negative correlation. This implies that directly migrating English evaluation paradigms to low-resource languages introduces noise into system comparisons and research progress.

**Key Challenge**: Multilingual LLMs are often hypothesized to use English as an internal pivot language. This internal geometric structure facilitates cross-lingual generalization, but when target language representations are not well-aligned with this pivot space, downstream generation or evaluation quality degrades. The core question is whether this representation misalignment also affects automatic evaluation metrics.

**Goal**: The authors aim to verify a simple hypothesis: whether steering the internal representations of low-resource or non-English inputs toward the English direction during inference can bring neural summarization evaluation metrics closer to human judgment.

**Key Insight**: Instead of retraining metrics, this work performs test-time intervention inside frozen models. It simultaneously covers decoder-based LLM-as-a-judge and encoder-based COMET to observe if steering serves as a more general tool for correcting multilingual evaluation.

**Core Idea**: Learn vectors or linear mappings from "Language X to English" using parallel sentences from FLORES, and perform controllable interpolation/offset on the model's hidden representations during evaluation to examine if Pearson correlation improves.

## Method

### Overall Architecture
The SteerEval pipeline consists of three steps. First, hidden representations are extracted using 500 pairs of parallel sentences from FLORES to learn a steering direction or linear mapping from Language X to English. Second, during the inference of the frozen evaluation model, the intermediate representations of the summary to be evaluated are adjusted based on steering strength parameters. Third, the adjusted neural metrics score the system summaries, and the Pearson correlation with multilingual human ratings is calculated.

The authors evaluate three types of metrics: Direct Prompting, where the LLM directly outputs a score of 1-5; GPTScore, which uses conditional generation probability to score summaries; and COMET using `wmt22-comet-da`, adapting the machine translation metric for summarization by using an empty source, system summaries as hypotheses, and human summaries as references.

### Key Designs
1. **Vector-based intervention**:
    - **Function**: Moves hidden representations during inference using a language direction vector.
    - **Mechanism**: Calculates the difference between the mean representation of the target language and the source language for a batch of parallel sentences to obtain the language direction for each layer. During evaluation, this direction multiplied by a strength $\rho$ is added to the hidden states of the source language input. For LLM metrics, this is applied layer-by-layer; for COMET, it is applied only to the pooled representation.
    - **Design Motivation**: If the "English pivot" corresponds to an approximately linear direction in the representation space, then a lightweight vector offset could improve cross-lingual alignment without training new metrics.

2. **Map-based intervention**:
    - **Function**: Learns a more flexible linear mapping to project source language representations into the target language representation space.
    - **Mechanism**: Learns a matrix $W_l$ for each layer to minimize the distance between the source representation after linear transformation and the target representation. During inference, parameter $\sigma$ is used to interpolate between the original representation and the mapped target representation.
    - **Design Motivation**: Vector differences only model translation, whereas linear mappings can capture rotation and scale changes, potentially better suited for complex representation misalignments between different languages.

3. **Multi-metric, Multilingual Meta-evaluation**:
    - **Function**: Distinguishes whether steering is effective only for a specific model or category of metrics.
    - **Mechanism**: LLM backbones include Llama-3-8B Instruct, Bloom-7B, Aya-expanse-8B, and Aya-expanse-32B; languages cover Arabic, Spanish, Hebrew, Japanese, Turkish, Ukrainian, Yoruba, and Chinese; evaluation dimensions include coherence and completeness.
    - **Design Motivation**: Difficulties in multilingual evaluation often stem from interactions between languages, models, prompts, and dimensions. Steering on only one model or language could lead to accidental conclusions.

### Loss & Training
No metric models were retrained in this work. Steering parameters are derived from the hidden representations of frozen models: the vector method calculates mean differences, and the map method uses least-squares alignment of parallel sentences. Since no language-specific development sets were available, main results report oracle results for the best steering strength in each setting. The authors also systematically scan $\sigma$ and $\rho$ in the analysis section to discuss parameter selection for real-world usage.

## Key Experimental Results

### Main Results
The unsteered baseline reveals that multilingual neural evaluation metrics are inherently unstable. The highest Pearson correlation is only 0.34, with negative correlations appearing across multiple languages, models, and dimensions.

| Metric / Model | Representative Strengths | Representative Weaknesses | Conclusion |
|-------------|------------|------------|------|
| COMET wmt22-comet-da | Arabic completeness 0.27, Japanese completeness 0.23 | Yoruba coherence -0.05, Yoruba completeness -0.04 | Small encoder metrics are competitive in some languages but unstable in low-resource ones |
| Direct Prompting Bloom-7B | Chinese coherence 0.08 | Negative in multiple languages, e.g., Arabic completeness -0.14 | Direct scoring is highly sensitive to model and language |
| Direct Prompting Llama3-8B | Japanese coherence 0.24, Japanese completeness 0.29 | Hebrew coherence -0.05, Hebrew completeness -0.08 | Llama3 is a more stable backbone for direct prompting |
| GPTScore Aya-exp 32B | Japanese completeness 0.34 | Yoruba completeness -0.07 | GPTScore is generally more stable than direct prompting |
| GPTScore Llama3-8B | Spanish coherence 0.23, Chinese coherence 0.22 | Yoruba coherence -0.06 | Correlation is better for medium-to-high resource languages |

After steering, the overall trend shows correlation improvements in the vast majority of settings, with larger gains in low-baseline settings.

| Phenomenon | Key Figures or Examples | Implication |
|------|----------------|------|
| Steering is almost universally effective | Improvements in most languages, metrics, and dimensions; some relative gains exceed 100% | Representation alignment improves consistency between metrics and human judgment |
| Larger gains for low-baseline languages | Hebrew, Turkish, and Yoruba often see larger relative gains | Languages with more severe representation misalignment benefit more from intervention |
| Direct Prompting can significantly improve but remains limited | Bloom-7B Japanese coherence improves from near 0 to 0.18 | Percentage gains may be inflated by low denominators; direct prompting is still not the most stable metric |
| Medium baselines also improve | Llama3-8B Spanish coherence improves from 0.15 to 0.20 | Steering does more than just fix broken settings; it provides robust gains |
| COMET is highly sensitive to steering | Multiple language/dimension relative gains exceed +50% | Encoder-based metrics can also benefit from hidden representation intervention |

### Ablation Study
The analysis experiments focus on steering methods, strength parameters, and target languages.

| Analysis Item | Finding | Explanation |
|--------|------|------|
| Vector vs Map | Both usually provide improvements; Vector often shows larger gains in COMET and low-baseline settings | The vector method has fewer parameters but is more language-specific and potentially more aggressive |
| Map Strength $\sigma$ | Higher $\sigma$ usually brings higher average relative gain; $\sigma=1$ is best on average | Completely moving toward the target representation is sometimes most beneficial, though not for all languages |
| Vector Strength $\rho$ | $\rho=-5$ has the highest average relative gain; ~60% of settings outperform no steering; positive $\rho$ is generally harmful | Direction and distance are not normalized; numerical meanings are inconsistent across languages |
| Language Vector Similarity | Except for Yoruba, most Language X to English vectors show high similarity in middle layers | Supports the shared cross-lingual geometric structure hypothesis, but Yoruba is a clear outlier |
| French as Target | Most settings also show significant improvements | High-resource languages well-aligned with English space can also serve as pivots |

### Key Findings
- The bottleneck in multilingual summarization evaluation is not just data scarcity; it may be that the model's internal representations are not aligned with its preferred high-resource pivot space.
- Direct Prompting has the highest variance, GPTScore is more stable, and COMET has unexpectedly large room for improvement under steering.
- The choice of steering factor is highly sensitive; oracle results indicate potential but cannot be directly equated to actual deployment performance without a dev set.
- Cross-lingual similarity of language vectors supports the "shared language geometry" hypothesis, but outliers like Yoruba warn against treating all languages as having identical directions.

## Highlights & Insights
- The paper extends activation steering from generation quality control to "metric calibration," which is an interesting shift in application.
- The experiments on COMET are particularly enlightening: even encoder-based metrics can improve correlation with multilingual human ratings through pooled representation intervention.
- The method does not require retraining metrics, making it suitable for exploration as a lightweight test-time correction module in existing evaluation pipelines.
- Results serve as a reminder that the multilingual reliability of LLM-as-a-judge cannot be assumed based on English or high-resource performance; negative correlations can occur in low-resource languages.

## Limitations & Future Work
- Main results use oracle steering strength due to the lack of language-specific development sets; real systems need a dev set or unsupervised criteria to select $\rho$ / $\sigma$.
- The number of samples and annotators per language in human evaluation data is limited, potentially leading to high variance in some correlation estimates.
- The task focuses on coherence and completeness in summarization, not yet covering factual consistency, style, QA, or open-ended generation evaluation.
- The gains from steering depend on the target language; both English and French are effective, but the optimal choice for different source-target pairs requires systematic research.
- The absolute correlation of Direct Prompting remains low; steering is not a substitute for better metric design and multilingual labeled data.

## Related Work & Insights
- **vs BLEU / ROUGE**: Traditional overlap metrics are simple and reproducible but insufficient at cross-lingual and semantic levels; SteerEval focuses on internal representation calibration for model-based metrics.
- **vs COMET**: While COMET is originally a machine translation metric, this paper adapts it for summarization and proves that encoder metrics can be improved via steering.
- **vs LLM-as-a-judge**: Direct prompting for LLM scoring is available but unstable; SteerEval shows that hidden representation alignment before scoring affects judge behavior.
- **vs Wang et al. multilingual steering**: Prior work used language mapping to improve generation tasks; this paper migrates the same idea to automatic evaluation, shifting the goal from generation quality to human correlation.
- **Insights**: Multilingual evaluation may require a "metric calibration layer"; future work could combine activation steering, language-specific prompts, and small human-evaluated development sets.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Appling inference-time representation intervention to multilingual evaluation metric calibration is a fresh perspective on an important problem.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers multiple metrics, models, languages, and parameter analyses, though oracle parameter selection weakens the deployment-level argument.
- **Writing Quality**: ⭐⭐⭐⭐☆ Motivation is clear and experimental results are explained in detail; some charts rely on relative gains, which requires caution regarding low baseline inflation.
- **Value**: ⭐⭐⭐⭐☆ Provides significant insights for multilingual NLG evaluation and the reliability of LLM-as-a-judge.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing BiGRU with a KAN Block for Legal Document Classification and Summarization](enhancing_bigru_with_a_kan_block_for_legal_document_classification_and_summariza.md)
- [\[ACL 2026\] Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP](scripts_through_time_a_survey_of_the_evolving_role_of_transliteration_in_nlp.md)
- [\[ACL 2026\] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)
- [\[ACL 2026\] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)
- [\[ACL 2026\] Multilingual Steering by Design: Multilingual Sparse Autoencoders and Principled Layer Selection](multilingual_steering_by_design_multilingual_sparse_autoencoders_and_principled_.md)

</div>

<!-- RELATED:END -->
