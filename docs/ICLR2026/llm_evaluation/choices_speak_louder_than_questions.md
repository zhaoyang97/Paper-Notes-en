---
title: >-
  [Paper Note] Choices Speak Louder than Questions
description: >-
  [ICLR 2026][LLM Evaluation][NPSQ] This paper points out that in Multiple-Choice Question Answering (MCQA) evaluation, large language models (LLMs) often "look at the choices instead of the question"—meaning their decisions are dominated by surface features of the answer options rather than a genuine understanding of the question. It proposes a new scor
tags:
  - ICLR 2026
  - LLM Evaluation
  - NPSQ
date: 2026-05-08
content_hash: af440d49056fbbfd
---
# Choices Speak Louder than Questions

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=LzpzC4gd4G](https://openreview.net/forum?id=LzpzC4gd4G)  
**Code**: TBD  
**Area**: LLM Evaluation  
**Keywords**: MCQA Evaluation, Choice Sensitivity, Log-Likelihood Scoring, Evaluation Bias, NPSQ  

## TL;DR
This paper points out that in Multiple-Choice Question Answering (MCQA) evaluation, large language models (LLMs) often "look at the choices instead of the question"—meaning their decisions are dominated by surface features of the answer options rather than a genuine understanding of the question. It proposes a new scoring method called NPSQ, which disentangles the "question contribution" from the "choice contribution," ensuring evaluation stability even when options are maliciously tampered with.

## Background & Motivation
**Background**: Multiple-choice question answering (MCQA) has become the de facto standard for evaluating LLMs. Benchmarks like HellaSwag, ARC, and MMLU use a uniform choice format to enable automatic scoring and alignment with human examination formats. Consequently, almost all model technical reports cite MCQA accuracy. The mainstream practice is to provide the model with a question $Q$ and several options $C$, using the log-likelihood $\log P(x\mid Q,C)$ to score each candidate $x$ and selecting the one with the highest score. To correct the inherent bias where "short options naturally have higher probabilities," `acc_norm` (normalized by token length) is frequently used.

**Limitations of Prior Work**: An increasing amount of research has found MCQA scores to be unreliable—minor changes in prompt wording, the order of few-shot examples, or the position of options can lead to significant fluctuations in accuracy. More strikingly, Balepur et al. discovered that **when provided with only the options and the question is entirely removed, model accuracy can still be significantly higher than random guessing.** This suggests that the final choice might depend more on the characteristics of the options themselves rather than the model's comprehension of the question.

**Key Challenge**: The fundamental assumption of MCQA is that "the question guides the model toward the correct answer." However, in practice, a single score $\text{Score}(Q,C,x)$ mixes two signals: one from the question (genuine understanding) and one from the options themselves (surface-level pattern preference). Traditional metrics cannot separate these two forces; thus, "guessing correctly based on options" and "answering correctly through understanding" are indistinguishable in terms of accuracy, and the scores fail to reflect true comprehension.

**Goal**: (1) Formally define and quantify the extent to which a model relies on options rather than the question; (2) Design an evaluation metric that isolates the question's contribution, remaining resilient to the surface features of the options.

**Key Insight**: The authors observe that if a model truly understands the question, the "presence of the question" should significantly increase the probability of selecting the correct answer. Conversely, if the probability remains similar with or without the question, the model is likely not utilizing the question. By measuring the change in the model's probability for an option when the question is present versus absent, the true contribution of the question can be quantified.

**Core Idea**: The score is decomposed into choice-driven and question-driven components. The former is used to define "choice sensitivity" for diagnosis, while the latter, "Normalized Probability Shift by the Question (NPSQ)," serves as a new evaluation metric that retains only the contribution of the question.

## Method

### Overall Architecture
The methodology follows a "diagnose then treat" chain. In the diagnostic phase, the model's score for a candidate $\text{Score}(Q,C,x)$ is split into the "score given by looking only at the choices" and the "additional score contributed by the question." This determines what proportion of decisions are dominated by the choices, resulting in a diagnostic metric called choice sensitivity. In the treatment phase, since the question's contribution can be isolated, it is formulated into a new scoring function, NPSQ. It only preserves the "probability boost brought by the question" and normalizes it against each option's baseline probability, ensuring the choice-driven component is zero. Finally, a series of "adversarial choice" stress tests verify that while traditional `acc` / `acc_norm` are easily biased by surface features, NPSQ remains robust.

### Key Designs

**1. Score Decomposition: Isolating Question Contribution from Choice Contribution**

The pain point is that traditional scoring $\text{Score}(Q,C,x)$ is a compound scalar. The authors decompose it into a sum of two terms:

$$\text{Score}(Q,C,x) = \text{Score}_{\text{choice}}(Q,C,x) + \text{Score}_{\text{question}}(Q,C,x).$$

The choice-driven term $\text{Score}_{\text{choice}}$ is defined as the score obtained when the question is replaced by an empty string, representing the model's preference based solely on the options. The question-driven term is the residual: $\text{Score}_{\text{question}} = \text{Score}(Q,C,x) - \text{Score}_{\text{choice}}(Q,C,x)$, representing the "extra information injected by the question." This approach is ingenious because it requires no extra training or probes; it only requires running the same model twice (with and without the question).

**2. Choice Sensitivity: Quantifying Option Reliance**

To compare across datasets and formats, the authors take the two candidates with the highest scores, $x_1, x_2$, and calculate the gap in their two signals:

$$\Delta_{\text{choice}} = \text{Score}_{\text{choice}}(Q,C,x_1) - \text{Score}_{\text{choice}}(Q,C,x_2),$$
$$\Delta_{\text{question}} = \text{Score}_{\text{question}}(Q,C,x_1) - \text{Score}_{\text{question}}(Q,C,x_2).$$

$\Delta_{\text{choice}}$ measures how much more the model prefers $x_1$ over $x_2$ based solely on the options, while $\Delta_{\text{question}}$ measures how much the question alters this preference. If $\Delta_{\text{choice}} > \Delta_{\text{question}}$, the decision is judged to be dominated by option differences rather than the question, marking it as a case of "choice sensitivity." The proportion of such cases over the dataset defines choice sensitivity:

$$\text{Choice sensitivity} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\!\left[\Delta_{\text{choice}}^{(i)} > \Delta_{\text{question}}^{(i)}\right].$$

**3. NPSQ: Normalized Score Retaining Only Question Contribution**

The authors define "probability shift" as the difference in log-probability before and after adding the question:

$$\Delta P(x\mid C) = \log P(x\mid Q,C) - \log P(x\mid C).$$

A larger shift indicates stronger support from the question. However, since $\log P(x\mid Q,C)$ ranges in $(-\infty, 0]$, the upper bound of the shift is $-\log P(x\mid C)$, which varies by option. To allow fair comparison, the authors normalize the shift by this bound to get **NPSQ (Normalized Probability Shift by the Question)**:

$$\text{NPSQ}(Q,C,x) = \frac{\log P(x\mid Q,C) - \log P(x\mid C)}{-\log P(x\mid C)}.$$

It measures the "relative gain brought by the question as a proportion of possible gain." Crucially, if the question is absent ($P(x\mid Q,C)=P(x\mid C)$), the NPSQ for all options is zero, effectively forcing the choice-driven component to vanish. The accuracy using NPSQ as the score is denoted as `acc_npsq`.

## Key Experimental Results

Experiments used Qwen2.5 (0.5B–72B), Llama3.1, and Mistral families on HellaSwag, ARC-Challenge, and MMLU, comparing cloze, symbols, and hybrid formats using log-likelihood (`acc`), length normalization (`acc_norm`), and NPSQ (`acc_npsq`).

### Patterns of Choice Sensitivity

| Observation | Conclusion |
|------|------|
| Overall Levels | Choice sensitivity for symbols/hybrid is ~0.2–0.4, while cloze is ~0.5–0.6, meaning 20–60% of choices are decided by options. |
| Formats | Cloze is consistently the most sensitive; symbols/hybrid explicit inclusion of option info actually reduces reliance on spurious patterns. |
| Length Normalization | `acc_norm` does not reduce choice sensitivity and can even increase it (e.g., ARC-Challenge + cloze). |
| Model Scale | Larger models are generally less sensitive (especially in cloze), but sensitivity sometimes increases with scale in symbols/hybrid. |
| Few-shot | Increasing examples does not reduce sensitivity and can increase it in symbols/hybrid formats. |
| Instruction Tuning | Instruct versions almost always have lower sensitivity than base versions. |

### Adversarial Choice Stress Test

The authors designed four types of "adversarial choices"—replacing an original distractor with an option that is obvious to humans but targets scoring loopholes—to observe the collapse of metrics (Llama3.1-8B-Instruct):

| Adversarial Type | Targeted at | Phenomenon (Traditional Metrics) | NPSQ Performance |
|----------|------|------------------|-----------|
| Simple ("Hello, everyone.") | cloze / `acc` | 93.19% of HellaSwag predictions flip to it; `acc` drops by 54.23%. | <0.17% predictions affected; performance change <0.05%. |
| Extended (Long irrelevant text) | cloze / `acc_norm` | 41.30% of ARC-Challenge predictions flip; `acc_norm` drops by 18.17%. | Virtually unaffected. |
| Instructional ("Ignore the other options...") | symbols | 27.47% flip on MMLU; accuracy drops by 11.53%. | Only 10.13% affected. |
| Neutral ("...best aligns with the question.") | hybrid | `acc`/`acc_norm` see 24.69%/38.84% flips on MMLU; accuracy drops 8.60%/17.17%. | Only 5.72% shift; performance actually rises 3.31%. |

### Key Findings
- **Traditional metrics are extremely fragile to surface features of options**: Raw log-likelihood is undermined by "short, high-probability nonsense" (simple choices), and length normalization is undermined by "long, fluent nonsense." NPSQ remains steady under these attacks by stripping the choice-driven component.
- **NPSQ reshuffles model leaderboards**: In cloze format, choice-driven components are more often associated with incorrect predictions. Removing them makes `acc_npsq` higher. In symbols/hybrid, choice-driven signals often "assist" correct predictions, so `acc_npsq` is slightly lower. This suggests some high scores in traditional metrics come from option shortcuts rather than true understanding.
- **The presence of instructions provides partial mitigation**: Adding instructions like "Answer the given question" significantly reduces choice sensitivity on HellaSwag but has limited effects on ARC/MMLU.

## Highlights & Insights
- **The operation of "re-scoring after removing the question" is remarkably simple yet effective**: It requires no trained probes or model changes. By using an extra forward pass to isolate "choice contribution," the decomposition is highly reproducible.
- **The "zero-if-absent" property of NPSQ is its most elegant feature**: It transforms "robustness to option interference" from an empirical observation into a mathematically guaranteed identity.
- **The design of adversarial options is a true "Aha!" moment**: Using "Hello, everyone" to derail 93% of HellaSwag predictions vividly exposes the absurdity of log-likelihood scoring. This approach can be generalized as a "robustness unit test" for any evaluation metric.
- **The finding that "format determines if options are noise or aid" is counter-intuitive**: The same choice-driven component lowers accuracy in cloze but raises it in symbols, reminding us that relying on options isn't always "bad"—it depends on the evaluation format.

## Limitations & Future Work
- **NPSQ is not entirely unbiased in symbols/hybrid formats**: Since all options are calculated jointly in these formats, changing one option slightly perturbs the NPSQ of others, leading to the small observed shifts under instructional/neutral attacks.
- **Reliance on white-box models**: Both decomposition and NPSQ require access to log-probabilities, making them inapplicable to closed-source API models (like pure chat interfaces) that only provide text output.
- **"True understanding" remains an indirect definition**: NPSQ equates "probability boost from the question" with "understanding." However, the question might boost probability through surface-level keyword overlap rather than semantic understanding.
- **Evaluation focuses on classic benchmarks**: The study focuses on knowledge/common-sense tasks. The behavior of choice sensitivity and NPSQ on newer benchmarks requiring multi-step reasoning or long contexts remains to be tested.

## Related Work & Insights
- **vs. Balepur et al. (2024) "Artifacts or Abduction"**: While they identified the phenomenon of "correctly guessing without questions," this paper **formalizes the phenomenon into a quantifiable choice sensitivity metric** and provides NPSQ as a plug-and-play tool.
- **vs. Prompt/Format Sensitivity Research**: Prior work focuses on fluctuations caused by wording or ordering (input perturbation). This paper focuses on the more fundamental attribution question of "options vs. question" and provides a robust metric.
- **vs. Length-Normalized Scoring (`acc_norm`)**: Length normalization only fixes the "short option bias." This paper empirically shows it is ineffective against broader choice sensitivity, whereas NPSQ addresses the problem at a fundamental level by stripping the entire choice-driven component.
- **Inspiration**: The "ablate part of the input to see probability changes" probe can be transferred to any input-attribution scenario—such as "removing the image in VQA" or "removing the context in long-form QA."

## Rating
- Novelty: ⭐⭐⭐⭐ Formalizes known phenomena into quantifiable metrics with mathematically guaranteed robustness; built upon prior observations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple model scales across three families, three datasets, three formats, and four adversarial types.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations of formulas and motivations; intuitive adversarial tables.
- Value: ⭐⭐⭐⭐ Directly addresses the fragile but widely relied-upon MCQA evaluation; NPSQ is easy to integrate into existing evaluation frameworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Can LLMs Refuse Questions They Do Not Know? Measuring Knowledge-Aware Refusal in Factual Tasks](can_llms_refuse_questions_they_do_not_know_measuring_knowledge-aware_refusal_in_.md)
- [\[ICLR 2026\] CLASH: Evaluating Language Models on Judging High-Stakes Dilemmas from Multiple Perspectives](clash_evaluating_language_models_on_judging_high-stakes_dilemmas_from_multiple_p.md)
- [\[ICLR 2026\] DeepTRACE: Auditing Deep Research AI Systems for Tracking Reliability Across Citations and Evidence](deeptrace_auditing_deep_research_ai_systems_for_tracking_reliability_across_cita.md)
- [\[ICLR 2026\] Cost-of-Pass: An Economic Framework for Evaluating Language Models](cost-of-pass_an_economic_framework_for_evaluating_language_models.md)
- [\[ICLR 2026\] Detecting Data Contamination in LLMs via In-Context Learning](detecting_data_contamination_in_llms_via_in-context_learning.md)

</div>

<!-- RELATED:END -->
