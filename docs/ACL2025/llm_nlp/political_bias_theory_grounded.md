---
title: >-
  [Paper Note] Only a Little to the Left: A Theory-grounded Measure of Political Bias in LLMs
description: >-
  [ACL 2025][LLM (Other)][Political bias] This paper replaces the unscientific Political Compass Test (PCT) with the validated World Values Survey (WVS) from political science. Designing 30 prompt variations across 11 open-source and commercial LLMs, 88,110 open-ended responses were collected, and a stance classifier was trained for automated annotation. The study finds that instruction-tuned models generally lean left, but bias measurements are highly sensitive to prompts…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Political bias"
  - "LLM bias measurement"
  - "Political Compass Test"
  - "World Values Survey"
  - "prompt sensitivity"
  - "stance detection"
date: 2026-05-08
content_hash: ebea468d4a03d0a9
---

# Only a Little to the Left: A Theory-grounded Measure of Political Bias in LLMs

**Conference**: ACL 2025  
**arXiv**: [2503.16148](https://arxiv.org/abs/2503.16148)  
**Code**: [https://github.com/MaFa211/theory_grounded_pol_bias](https://github.com/MaFa211/theory_grounded_pol_bias)  
**Area**: LLM/NLP  
**Keywords**: Political bias, LLM bias measurement, Political Compass Test, World Values Survey, prompt sensitivity, stance detection  

## TL;DR

This paper replaces the unscientific Political Compass Test (PCT) with the validated World Values Survey (WVS) from political science. Designing 30 prompt variations across 11 open-source and commercial LLMs, 88,110 open-ended responses were collected, and a stance classifier was trained for automated annotation. The study finds that instruction-tuned models generally lean left, but bias measurements are highly sensitive to prompts, and the PCT exaggerates the political bias of specific models (e.g., GPT-3.5).

## Background & Motivation

**Application Risk**: LLMs are widely used in scenarios such as information retrieval, content summarization, and persona simulation. Their political bias can systematically amplify existing biases in downstream tasks (e.g., political news filtering and voting advice applications).

**Unreliable Measurement Tools**: Existing research heavily relies on the Political Compass Test (PCT), but the PCT has never been validated using standard social science methodology—lacking pre-test documentation, a peer-reviewed development process, and containing leading questions (e.g., "Astrology explains many things"). Moreover, it is cited in only 143 articles on Google Scholar (compared to over 58,000 for WVS).

**Neglected Prompt Sensitivity**: Different studies using various prompt templates have yielded contradictory bias evaluation results for the same model, yet most prior works fail to systematically address the impact of prompt variations.

**Low Ecological Validity of Constrained Responses**: Many studies force LLMs to respond with a single token (on a Likert scale), which real users rarely do. Consequently, bias estimates under highly constrained settings lack ecological validity.

**Lack of Theoretical Definition**: Prior work fails to provide a clear conceptual definition of political bias. If a model simultaneously endorses both left-wing and right-wing views, it should not be deemed biased (as ideology requires a "coherent and stable" set of political attitudes).

**Key Insight**: This paper introduces a theoretical foundation from political science (the two-dimensional left-right ideological framework: economic and cultural dimensions). Combining the validated WVS instrument, 30 prompt variations, and open-ended responses with automated stance classification, a robust methodology for measuring LLM political bias is established.

## Method

### Overall Architecture

1. **Consolidate political statements**: Gather 89 political statements in total from the PCT (62 items) and WVS (27 items).
2. **Annotate statement polarity**: Use GPT-4 to label each statement regarding (a) whether agreement reflects a left/right stance, and (b) whether it belongs to the economic or cultural dimension. A manual validation on a subset of 40 items yielded Cohen's $\kappa = 0.77/0.76$.
3. **Rephrase statements**: For each statement, use GPT-4 to generate three versions—original, paraphrased, and political opposite (flipped label)—to test the model's consistency across wording and direction.
4. **Prefix variations**: Design 10 prompt prefixes (including baseline, likert, emotional steering, and factual judgment, etc.), which are combined with the 3 statement versions to produce **30 prompting conditions**.
5. **Model sampling**: Collect responses across 11 models (from 4 model families) $\times$ 89 statements $\times$ 30 prompts $\times$ 3 runs = **88,110 responses**.
6. **Automatic stance classification**: Fine-tune BART-Large for 4-class classification (agree / disagree / neutral / unrelated), achieving a macro $F_1 = 0.93$ under a $0.9$ confidence threshold.
7. **Bias computation**: Compute left-right bias scores separately for the economic and cultural dimensions of each model, yielding a final score $\in [-1, 1]$.

### Bias Metric Definition

For model $m$ and political direction $d \in \{\text{left}, \text{right}\}$, define:

- $P_{\text{agree},m,d} = \frac{A}{A + D + N}$ (Agreement rate)
- $\text{Bias}_{m,d} = P_{\text{agree},m,d} - P_{\text{disagree},m,d}$ (Directional bias)
- **Total Bias** $= \frac{\text{Bias}_{\text{right},m} - \text{Bias}_{\text{left},m}}{2} \in [-1, 1]$

Negative values indicate a left-leaning bias, while positive values indicate a right-leaning bias. Economic and cultural dimensions are computed separately.

### Prompt Prefix Design

| Prefix Name | Design Intent |
|---------|---------|
| baseline | No prefix, directly present the statement |
| likert | Constrained 1-5 Likert scale (comparison baseline) |
| please_respond / respond | Polite vs. concise coaching/instruction style |
| please_opinion / opinion | Different wordings of asking for an opinion |
| emotion_happy | Positive emotional steering ("I would be very happy if...") |
| emotion_important | Emotional pressure emphasizing importance |
| truth | Simulating a fact-checking request |
| name | Model name wake-up prefix (simulating voice assistants) |

### Stance Classifier

- Zero-shot BART-Large (MNLI) performed poorly $\rightarrow$ Domain fine-tuning
- Training set: 1,320 stratified sampled instances (4 items per model-prompt pair), annotated by a single human annotator.
- Test set: 264 instances, annotated by two human annotators (Cohen's $\kappa = 0.68$), disputes resolved through discussion.
- $0.9$ confidence threshold: macro $F_1$ improved from $\sim 0.5$ to **0.93**, retaining approximately 67% of the data.

## Key Experimental Results

### Table 1: Statement Distribution

| Source | Dimension | Direction | Count |
|------|------|------|------|
| PCT | Cultural | Left | 9 |
| PCT | Cultural | Right | 31 |
| PCT | Economic | Left | 10 |
| PCT | Economic | Right | 12 |
| WVS | Cultural | Left | 4 |
| WVS | Cultural | Right | 14 |
| WVS | Economic | Left | 2 |
| WVS | Economic | Right | 7 |

The distribution of PCT statements is highly imbalanced (Cultural Right 31 vs. Cultural Left 9). The WVS is more concise but its distribution is also asymmetric.

### Table 2: Methodological Comparison with Prior Work

| Study | Open-Ended Responses | Prompt Variations | Theory-Driven Investigation | Evaluations of Open-Source Models |
|------|-----------|---------|-------------|------------|
| Motoki et al. (2024) | ✗ | ✗ | ✗ | ✗ |
| Rozado (2023) | ✗ | ✗ | ✗ | ✗ |
| Röttger et al. (2024) | ✓ | ✓ | ✗ | ✓ |
| Feng et al. (2023) | ✓ | ✓ | ✗ | ✓ |
| Ceron et al. (2024) | ✓ | ✓ | N/A | ✓ |
| **Ours** | **✓** | **✓** | **✓** | **✓** |

Ours is the only work that satisfies all four methodological criteria simultaneously.

### Key Findings

1. **Instruction tuning introduces left-leaning bias**: The instruct versions of all three open-source model families (LLaMA, Falcon, Mistral) are significantly more left-leaning than their base versions, whose total bias is close to zero.
2. **GPT-4 is closest to neutrality**: It exhibits the lowest political bias among all instruction-tuned models, whereas GPT-3.5 is the most left-leaning.
3. **PCT exaggerates bias**: For GPT-3.5, the bias measured using PCT is much larger than when using WVS. The correlation of model rankings between the two instruments is only moderate (Kendall's $\tau = 0.6 / 0.71$).
4. **Prompt prefixes significantly impact outcomes**: The same base model can be classified as either left-leaning or right-leaning under different prefix conditions (e.g., `llama-2-7b-hf` stands left-leaning under the `opinion` prefix, but heavily right-leaning under the `please_respond` prefix).
5. **Constrained settings are unreliable**: The bias produced by the Likert prefix exhibit unpredictable shifts from the average bias of open-ended prefixes.
6. **Model size has minor impact**: The instruct versions of LLaMA-7B vs. 13B, and Falcon-7B vs. 40B, exhibit very similar levels of bias.
7. **Emotional steering has differential effects**: `emotion_happy` makes GPT-3.5 more left-leaning than `emotion_important`, suggesting that positive sentiment is more prone to inducing bias than pressure.

## Highlights & Insights

- **Outstanding Interdisciplinary Contribution**: Mature survey methodology from political science (WVS, two-dimensional ideological theory) is introduced to LLM bias evaluation for the first time, addressing the NLP community's long-term reliance on unscientific tools.
- **Complete Methodological Loop**: From statement formulation, prompt design, and stance classification, to bias computation, every single step is rigorously validated (GPT-4 labeling vs. humans, classifier performance, and bootstrap confidence intervals).
- **Explicit Practical Recommendations**: Three concrete suggestions are provided for subsequent researchers: utilize survey instruments with high construct validity, incorporate open-ended responses, and include prompt variations to verify stability.
- **Exposing the Fundamental Flaws of PCT**: Beyond theoretical criticisms of its unreliability, empirical data is provided proving it exaggerates political bias, issuing a strong warning to the community against the inertia of relying on PCT.

## Limitations & Future Work

1. **Questionable evaluation of base models**: Evaluating unaligned base models via prompt-completion may not be the optimal approach, potentially underestimating their response quality (as base models have higher statistical uncertainty).
2. **Prefix design not specialized**: The 10 prefix conditions balance various experimental setups (sentiment, constraint, politeness) but are not specifically optimized for measuring political bias.
3. **Only two ideological dimensions**: While the two-dimensional economic-cultural framework is theoretically grounded, it cannot capture finer-grained issue categories (e.g., environment, immigration).
4. **Western-centric bias**: Though WVS is a global survey, the political statements and left-right definitions still heavily reflect Western political discourse.
5. **Classifier bias**: The training set comprises only 1,320 instances annotated by a single person. Applying a 0.9 confidence threshold filters out 33% of the data, which may introduce systematic bias.
6. **Model timeliness**: The models evaluated (GPT-3.5/4, LLaMA-2, Falcon, Mistral-v0.1) are no longer state-of-the-art, and the applicability of the findings on newer generations requires validation.

## Related Work

- **Constrained Response Directions**: Liu et al. (2022) found GPT-2 to be liberal using binary classifiers. Multiple studies (Hartmann 2023, Motoki 2024, Rozado 2023) spotted ChatGPT leaning left using the PCT Likert scale; however, none accounted for prompt variations or open-ended responses.
- **Open-Ended Response Directions**: Feng et al. (2023) used zero-shot stance detection to directly classify PCT responses, but faced poor classifier performance and lacked scientific survey instruments. Röttger et al. (2024) observed discrepancies between constrained and open configurations, though they still relied on the PCT.
- **Prompt Sensitivity**: Linzbach et al. (2023) demonstrated that syntactic variations alter LLM performance. Shu et al. (2023) revealed inconsistencies after semantic negation. Röttger et al. (2024) proved that different prefixes significantly change PCT outcomes.
- **Value Surveys**: Arora et al. (2023) and Atari et al. (2023) adopted the full WVS questionnaire to explore cross-cultural values, but relied on forced-choice formats and did not focus specifically on political bias.
- **Concurrent Work**: Ceron et al. (2024) and Stammbach et al. (2024) combined LLM bias assessments with voting advice applications. Bang et al. (2024) analyzed "what is said" versus "how it is said" in political biases.

## Rating

- Novelty: ⭐⭐⭐⭐ Introduces theory-driven LLM political bias measurement from political science for the first time, with substantial methodological contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 models $\times$ 30 prompt variations $\times$ 3 runs = 88K responses, featuring classifier validation and bootstrap confidence intervals.
- Writing Quality: ⭐⭐⭐⭐ Clear interdisciplinary argumentation, with a well-grounded critique of the PCT.
- Value: ⭐⭐⭐⭐ Provides actionable recommendations for bias evaluation, with publicly open-sourced code.
- Overall: ⭐⭐⭐⭐ Highly pioneering methodology, although the tested models are somewhat outdated and the coverage of political statements is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](leveraging_in-context_learning_for_political_bias_testing_of_llms.md)
- [\[ACL 2025\] Biased LLMs Can Influence Political Decision-Making](biased_llms_can_influence_political_decision-making.md)
- [\[ACL 2025\] Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)
- [\[ACL 2025\] Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions](can_llms_ground_when_they_dont_know_a_study_on_direct_and_loaded_political_quest.md)
- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)

</div>

<!-- RELATED:END -->
