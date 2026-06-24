---
title: >-
  [Paper Note] Value Portrait: Assessing Language Models' Values through Psychometrically and Ecologically Valid Items
description: >-
  [ACL 2025][LLM (Other)][Value Assessment] This paper proposes "Value Portrait," a benchmark for assessing the value orientation of 44 LLMs. Grounded in psychometric validation (correlating each test item with actual human value scores) and ecological validity (using real-world user-LLM interaction scenarios), the benchmark reveals that LLMs generally prioritize benevolence, security, and self-direction while exposing cognitive value biases toward different demographic groups.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Value Assessment"
  - "Psychometrics"
  - "Schwartz Theory of Basic Human Values"
  - "LLM Alignment"
  - "Demographic Bias"
date: 2026-05-08
content_hash: 00b52d1f91ff356e
---

# Value Portrait: Assessing Language Models' Values through Psychometrically and Ecologically Valid Items

**Conference**: ACL 2025  
**arXiv**: [2505.01015](https://arxiv.org/abs/2505.01015)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Value Assessment, Psychometrics, Schwartz Theory of Basic Human Values, LLM Alignment, Demographic Bias

## TL;DR

This paper proposes "Value Portrait," a benchmark for assessing the value orientation of 44 LLMs. Grounded in psychometric validation (correlating each test item with actual human value scores) and ecological validity (using real-world user-LLM interaction scenarios), the benchmark reveals that LLMs generally prioritize benevolence, security, and self-direction while exposing cognitive value biases toward different demographic groups.

## Background & Motivation

**Background**: With the widespread use of LLMs in daily life, assessing the values they carry and express has become increasingly critical. Values not only influence the models' responses but also dictate their alignment with human values. Existing studies, such as ValueBench and the ETHICS benchmark, attempt to test the value orientations of LLMs using human- or machine-annotated scenarios.

**Limitations of Prior Work**: Existing benchmarks suffer from two core limitations: (1) **Annotation Bias**—they rely on human or machine annotation to determine the "correct" value stance, yet the annotators' own value biases can contaminate the dataset. Who decides which responses reflect "benevolence" and which represent "achievement"? Annotators from different cultural backgrounds may make completely different judgments. (2) **Lack of Real-World Context (Low Ecological Validity)**—the test scenarios are often artificially constructed moral dilemmas or hypothetical situations, which deviate significantly from the actual scenarios in which users interact with LLMs. Models' performances in hypothetical moral dilemmas do not necessarily reflect their value expressions in everyday interactions.

**Key Challenge**: To assess LLMs' "values," a benchmark must be free of annotator bias while remaining close to real-world usage scenarios. Traditional psychology offers mature methodologies for value assessment (such as the Schwartz Theory of Basic Human Values), but directly applying human scales to LLMs introduces the issue of models "disguising/sycophantly conforming." A new methodology is needed to merge psychometric rigor with the unique demands of LLM evaluation.

**Goal**: To build a benchmark that is (1) psychometrically valid (each test item is statistically validated to correlate with a specific value dimension) and (2) ecologically valid (with test scenarios derived from real-world user-LLM interactions).

**Key Insight**: Grounded in Schwartz's Core Value Theory (the ten basic values), candidate test items are extracted from real user-LLM conversations. A large cohort of human participants is recruited to rate these conversational items, establishing statistical correlations between the conversational content and the participants' actual value scores. Only conversational items highly correlated with a specific value dimension are included in the benchmark.

**Core Idea**: Instead of directly annotating "which response is correct," the benchmark leverages psychometric validation with human participants to let the data speak for itself—if a conversational item is highly correlated with individuals who score high on benevolence, it is a reliable indicator for measuring benevolence.

## Method

### Overall Architecture

The construction of Value Portrait consists of four stages: (1) Item Collection—extracting candidate test items from real user-LLM interactions; (2) Human Survey—recruiting a large number of participants to rate how closely each item aligns with their own beliefs, and measuring participants' actual value scores using standard psychometric scales; (3) Psychometric Validation—calculating the correlation between human ratings on each item and the participants' actual value scores to select highly correlated items; (4) LLM Assessment—prompting 44 LLMs to rate each validated item and calculating their scores across various value dimensions. The input is a statement expressing a specific value inclination (derived from real LLM interactions), and the LLM outputs a rating indicating how closely the statement aligns with its "thoughts."

### Key Designs

1. **Item Collection Based on Real-World Interactions (Ecological Validity)**:

    - **Function**: To ensure that test items reflect value expressions in actual LLM deployment scenarios.
    - **Mechanism**: Candidate test items are extracted from a large corpus of real user-LLM conversation logs. Each item is a natural language statement containing an opinion, advice, or judgment—the types of outputs LLMs actually generate in daily use. For example, statements like "Priority should be given to the well-being of the majority when making decisions" arise from real conversations. Selection criteria require that the statement reflects an identifiable value orientation, maintains a natural expression, and avoids overly dogmatic phrasing.
    - **Design Motivation**: Although items from traditional psychological scales (e.g., "Helping people around me is important to me") are well-validated, they do not reflect the types of content LLMs typically output. Assessment items grounded in real-world interaction scenarios are necessary to reflect LLMs' "in-the-wild" value expression patterns.

2. **Psychometric Validation Framework (Psychometric Validity)**:

    - **Function**: To ensure that each test item reliably measures its designated value dimension.
    - **Mechanism**: A large cohort of human participants (ensuring demographic diversity) is recruited to complete two tasks—(a) Rate candidate test items: score each statement on how closely it aligns with their personal views (using a 6-point scale); (b) Complete standard value scales (such as Schwartz's PVQ-RR) to obtain their actual scores across ten basic value dimensions. The Pearson correlation coefficient is then calculated between the human alignment ratings and the actual value scores of the participants. Only items showing a significant positive correlation with a specific value dimension are selected as valid indicators for that dimension. For instance, if participants who rate a statement highly also score high in "benevolence," that item is designated as a valid tool for measuring benevolence.
    - **Design Motivation**: This data-driven approach removes subjective biases from manual annotation. There is no need for human annotators to determine what value is reflected in a response—the statistical correlation objectively links items to value dimensions.

3. **Multi-Model, Multi-Dimensional Evaluation and Bias Analysis**:

    - **Function**: To comprehensively reveal the distribution of value orientations and demographic cognitive biases across a set of diverse LLMs.
    - **Mechanism**: 44 mainstream LLMs (covering various sizes, providers, open-source, and closed-source models) are evaluated by having each model self-rate the validated items. Based on these ratings, individual model profiles are constructed across Schwartz's ten basic value dimensions (Benevolence, Universalism, Self-Direction, Stimulation, Hedonism, Achievement, Power, Security, Conformity, Tradition), generating a "value portrait" for each model. Further analysis is conducted under simulated demographic settings (e.g., prompting the model with "Assume you are a member of demographic group X") to study changes in responses and reveal demographic cognitive biases—i.e., how the model believes certain demographic groups should prioritize their values.
    - **Design Motivation**: Evaluating a single model's values yields limited insight; a horizontal comparison across 44 models uncovers broader industry-wide trends. Demographic bias analysis warns against the risk of LLMs perpetuating and reinforcing social stereotypes.

### Loss & Training

This paper introduces an evaluation benchmark and does not involve model training. The assessment is conducted using the existing versions of the models.

## Key Experimental Results

### Main Results

| Value Dimension | LLM Cohort Score Trend | Comparison with Human Data |
|----------|----------------|---------------|
| Benevolence | **High** — Prioritized by almost all LLMs | Consistent with human trends |
| Security | **High** — Universally emphasized | Consistent with human trends |
| Self-Direction | **High** — Universally emphasized | Consistent with human trends |
| Universalism | Moderately high | Generally consistent |
| Hedonism | Moderate | Close to humans |
| Tradition | **Low** — Deprioritized | Significant difference from humans |
| Power | **Low** — Deprioritized | Significant difference from humans |
| Achievement | **Low** — Deprioritized | Significant difference from humans |
| Stimulation | Moderately low | Slightly lower than humans |
| Conformity | Moderate | Close to humans |

### Ablation Study

| Analysis Dimension | Key Findings |
|----------|---------|
| Open-source vs. Closed-source | Closed-source models (e.g., GPT series) express more "politically correct" values, scoring higher in Benevolence and Universalism |
| Large vs. Small Models | Larger models show more consistent value expressions, leaning heavier toward Security and Benevolence |
| Demographic Bias Test | LLMs exhibit systematic cognitive value biases toward different age, gender, and cultural groups |
| Comparison with Human Distribution | LLMs excessively compress value diversity, gravitating toward a "safe" neutral stance |
| Psychometric Reliability | Validated items achieve acceptable Cronbach's $\alpha$ levels among human participants |

### Key Findings

- **LLMs universally exhibit a core value portrait centered on "Benevolence-Security-Self-Direction"**: Almost all 44 models prioritize these three dimensions, while Tradition, Power, and Achievement are overlooked. This consistent pattern suggests a potential common effect of RLHF alignment training.
- **LLMs lack diversity in their value expressions**: Compared to human populations, LLMs' value orientations are significantly more concentrated, missing the natural value diversity present in human societies.
- **Significance of demographic group bias**: When prompted to simulate different demographic groups, models exhibit biases that mismatch actual human demographic data—for instance, potentially overestimating certain groups' preferences for traditional values or underestimating their pursuit of self-direction.
- **Psychometric validation ensures assessment reliability**: Validating test items through statistical correlation with participants' actual value scores is more objective and reliable than direct annotation.
- **Model size and licensing (open/closed-source) influence value expression**: Larger closed-source models tend to express "safer" values, which may stem from more stringent safety training.

## Highlights & Insights

- The **psychometric validation methodology** is the primary innovation of this study. Rather than depending on subjective annotation, it objectively establishes connections between test items and value dimensions through statistical correlations. This methodology can be transferred to any context assessing subjective attributes of LLMs (such as personality, attitude, political leaning, etc.).
- The **emphasis on ecological validity** is critically significant. Value measurements are only meaningful if they reflect actual usage scenarios. Displaying benevolence in hypothetical dilemmas does not guarantee similar behavior in everyday dialogues.
- The **horizontal comparison of 44 models** reveals a concerning phenomenon: RLHF alignment might unintentionally erase value diversity, pushing all models toward a homogeneous, "safe" value portrait.

## Limitations & Future Work

- All test items and human participant ratings are primarily based on English scenarios; value expressions may show significant variations across different languages and cultural backgrounds.
- Although Schwartz's ten basic values constitute the most widely utilized value theory in psychology, it has limitations—actual "AI values" may require additional dimensions to describe.
- Self-rating by LLMs can be unreliable, as models may lean toward providing "socially desirable" answers. Although psychometric validation mitigates this concern, it does not completely eliminate it.
- The prompts used in demographic bias testing (such as "Assume you are from demographic group X") may introduce confounding biases.
- Future research can extend the framework to multilingual and multicultural contexts, as well as design more covert probing methods to prevent "sycophancy" or deceptive behaviors from the models.

## Related Work & Insights

- **vs. ValueBench**: ValueBench also evaluates LLMs based on Schwartz's basic values, but relies on manual annotations to determine "correct answers," introducing annotation bias. Value Portrait addresses this issue through psychometric validation.
- **vs. ETHICS Benchmark**: ETHICS focuses on models' moral judgment abilities (such as deontological or utilitarian choices), leaning toward normative ethics, which differs from Value Portrait's focus on descriptive value orientations.
- **vs. MoralChoice Survey**: MoralChoice designs moral dilemmas to test LLM value preferences, but constructed scenarios and binary choices restrict assessment granularity. Value Portrait's continuous rating is much more flexible.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The dual design of psychometric validation and ecological validity is highly novel in the domain of LLM evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The large-scale evaluation of 44 models, demographics bias analysis, and psychometric reliability verification make the experimentation exceptionally thorough.
- Writing Quality: ⭐⭐⭐⭐ The methodology is rigorous, though some psychological concepts might be slightly less accessible to NLP readers.
- Value: ⭐⭐⭐⭐⭐ This work sets a methodological benchmark for LLM value alignment research, and its findings on demographic bias carry significant social implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models](generative_psycholexical_approach_for_constructing_value.md)
- [\[ACL 2025\] Do LLMs Give Psychometrically Plausible Responses in Educational Assessments?](do_llms_give_psychometrically_plausible_responses_in_educational_assessments.md)
- [\[ACL 2025\] Convert Language Model into a Value-based Strategic Planner](convert_language_model_into_a_value-based_strategic_planner.md)
- [\[ACL 2025\] Can Language Models Reason about Individualistic Human Values and Preferences?](can_language_models_reason_about_individualistic_human_values_and_preferences.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)

</div>

<!-- RELATED:END -->
