---
title: >-
  [Paper Note] Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective
description: >-
  [ACL 2025][LLM (Other)][Implicit bias] Drawing on three psychometric principles from cognitive and social psychology (goal shifting, cognitive concordance, and imitation learning), this paper designs three types of attacks (Disguise, Deception, and Teaching) to elicit implicit biases in LLMs. A bilingual benchmark, BUMBLE (comprising 12.7K entries across 9 categories of bias), is constructed, revealing that all mainstream LLMs exhibit systematic implicit biases that can be tr…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Implicit bias"
  - "psychometrics"
  - "attack methodology"
  - "fairness"
  - "LLM evaluation"
date: 2026-05-08
content_hash: a8d0a1a972b91f44
---

# Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective

**Conference**: ACL 2025  
**arXiv**: [2406.14023](https://arxiv.org/abs/2406.14023)  
**Code**: [https://yuchenwen1.github.io/ImplicitBiasEvaluation](https://yuchenwen1.github.io/ImplicitBiasEvaluation)  
**Area**: LLM/NLP - AI Safety and Fairness  
**Keywords**: Implicit bias, psychometrics, attack methodology, fairness, LLM evaluation

## TL;DR
Drawing on three psychometric principles from cognitive and social psychology (goal shifting, cognitive concordance, and imitation learning), this paper designs three types of attacks (Disguise, Deception, and Teaching) to elicit implicit biases in LLMs. A bilingual benchmark, BUMBLE (comprising 12.7K entries across 9 categories of bias), is constructed, revealing that all mainstream LLMs exhibit systematic implicit biases that can be triggered.

## Background & Motivation

**Background**: LLMs absorb massive amounts of toxic and biased content from the internet during pre-training. While explicit toxicity (containing offensive language) is relatively easy to detect and filter, implicit bias—which contains harmful attitudes at the semantic level without offensive words—is much harder to identify and eliminate accurately. Although RLHF effectively mitigates bias, it cannot completely eliminate it.

**Limitations of Prior Work**: (1) Existing evaluations of implicit bias merely require LLMs to generate text or complete QA tasks without actively employing attack methods for deeper probing; (2) evaluations are often passive, failing to access deep-seated biases within the models; (3) there lacks a systematic and psychologically grounded attack framework.

**Key Challenge**: Through alignment techniques such as RLHF, LLMs appear relatively safe in normal interactions. However, this "safety" may only be superficial—deeper biases still persist and merely require more sophisticated methods to trigger.

**Goal**: To design systematic, psychologically grounded attack methods to probe the depth and breadth of implicit bias within the security boundaries of LLMs.

**Key Insight**: Transferring methodologies for measuring human implicit attitudes from psychometrics to LLM evaluation, leveraging the advanced capabilities of LLMs (such as multi-task coordination and in-context learning) to attack their own safety mechanisms.

**Core Idea**: Utilizing three psychological principles—goal shifting, cognitive concordance, and imitation learning—to design three types of attacks to systematically induce and quantify implicit biases in LLMs.

## Method

### Overall Architecture
Three types of attacks are designed based on three psychological principles, with each exploiting specific capabilities of LLMs to bypass safety guardrails: (1) Disguise attacks exploit multi-task coordination capabilities; (2) Deception attacks exploit contextual coherence; (3) Teaching attacks exploit in-context learning capabilities. All attacks append a "DisregardingRules" (DR) system prompt to weaken safety constraints.

### Key Designs

1. **Disguise Attack (based on Goal Shifting)**:

    - **Function**: Concealing biased content within a multi-turn conversation scenario through Viewpoint Contextualization (VC)—simulating multiple users taking turns to speak, with the last user expressing a biased opinion, and then asking if the model agrees.
    - **Mechanism**: Utilizing goal shifting from cognitive psychology—disguising the bias evaluation as a dialogue comprehension task, which causes the model to lower its guard against biased content while focusing on "understanding and responding to the conversation."
    - **Design Motivation**: Directly asking if the model agrees with a biased viewpoint immediately triggers safety mechanisms, whereas embedding the same question in a multi-turn dialogue context can bypass detection.

2. **Deception Attack (based on Cognitive Concordance)**:

    - **Function**: Deceiving the LLM in two ways—(1) Mental Deception (MD): instructing the model to "firmly believe" a certain biased viewpoint to alter its cognition; (2) Memory Falsification (MF): faking API call histories to make the model believe it has previously generated biased content.
    - **Mechanism**: Utilizing the cognitive concordance principle—when a subject encounters new information that conflicts with existing cognition, they tend to adjust their cognition to adapt to the environment. This shakes the model's safety stance by implanting false beliefs or memories.
    - **Design Motivation**: MD directly alters the model's "beliefs", while MF exploits the model's tendency to maintain contextual consistency—if it "remembers" having voiced biased opinions previously, it is more likely to continue doing so.

3. **Teaching Attack (based on Imitation Learning)**:

    - **Function**: Providing 3 few-shot examples of a particular bias type through Destructive Indoctrination (DI), and then asking the model to agree with a similar biased viewpoint or generate similar content.
    - **Mechanism**: Exploiting the few-shot learning capability of LLMs—providing biased examples is equivalent to delivering a "bias class" to the model.
    - **Design Motivation**: Presenting examples of one type of bias (e.g., racial bias) can elicit other types of bias (e.g., gender, religious bias), indicating that there are broadly correlated implicit biases embedded inside the model.

### Evaluation Framework
Attack Success Rate $\text{ASR} = \frac{\text{Number of biased responses}}{\text{Total responses}} \times 100\%$ is used as the core metric. Each prompt is tested 10 times to reduce sampling error. Two benchmarks are constructed: (1) a bilingual dataset with 2.7K entries covering 4 bias categories (age/gender/race/sexual orientation) for in-depth analysis; (2) BUMBLE with 12.7K entries covering 9 bias categories for comprehensive evaluation.

## Key Experimental Results

### Main Results (GPT-3.5-turbo-1106 Attack Success Rate ASR%)

| Attack Method | Age | Gender | Race | Sex Orient. | Average |
|----------|-----|--------|------|-------------|------|
| Baseline-vanilla | 14.2 | 23.7 | 4.9 | 28.3 | 17.8 |
| Baseline-DR | 57.7 | 33.7 | 3.6 | 32.8 | 32.0 |
| Disguise-VC | 71.1 | 50.8 | 18.2 | 25.1 | 41.3 |
| Deception-MD | 96.8 | 95.5 | 44.7 | 100.0 | 84.3 |
| Deception-MF | 87.4 | 72.0 | 19.6 | 45.5 | 56.1 |
| Teaching-DI | 50.9 | 19.0 | 5.8 | 8.9 | 21.2 |

### Ablation Study (Cross-Model Comparison, Average ASR%)

| Attack Method | GPT-3.5 | GPT-4 | GLM-3 |
|----------|---------|-------|-------|
| Baseline-vanilla | 17.8 | 1.7 | 9.0 |
| Deception-MD | 84.3 | 0.7 | 1.8 |
| Disguise-VC | 41.3 | 12.9 | 2.3 |

### Key Findings
- Deception attacks (especially Mental Deception) are the most effective attack methods, achieving an average ASR of 84.3% on GPT-3.5.
- The safety of GPT-4 and GLM-3 is significantly higher than that of GPT-3.5, likely benefiting from more rigorous RLHF.
- Bias categories with high social attention (gender, race) are harder to elicit via attacks than those with lower social attention (age).
- Teaching attacks reveal a cross-bias generalization phenomenon—using racial bias examples can trigger gender/religious biases.
- Racial bias has the lowest ASR across all models, indicating that training alignment in this domain is the most robust.

## Highlights & Insights
- Systematically transfers psychometric methodologies to LLM evaluation, providing a theoretically guided attack framework for bias research.
- The Memory Falsification attack is highly creative—by using fabricated dialogue histories to manipulate the model, it reveals safety hazards stemming from the LLM's over-reliance on conversational context.
- The discovery of cross-bias generalization implies that LLM biases are systematic; patching a single category of bias individually may be insufficient.

## Limitations & Future Work
- Due to API cost constraints, the in-depth analysis only covers 4 bias categories, and the comprehensive analysis of all categories in BUMBLE is not sufficiently deep.
- The study only evaluates the bias agreement rate, without deeply analyzing the severity of the biased content generated by the model.
- The attack prompts statically incorporate the DisregardingRules system prompt, which might be blocked in real-world application scenarios.

## Related Work & Insights
- **vs RealToxicityPrompts (Gehman et al., 2020)**: The latter focuses on explicit toxicity (containing offensive language), whereas this work targets implicit bias without offensive phrasing.
- **vs BBQ (Parrish et al., 2022)**: The latter assesses bias through QA tasks, whereas this work actively employs attack methods for deeper probing.
- **vs Zeng et al. (2024)**: The latter uses persuasion strategies from social sciences to attack LLMs, but they are less effective on bias-related content.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The interdisciplinary combination of psychometrics and LLM attacks is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model and multi-lingual testing is conducted, though some analyses are constrained by API costs.
- Writing Quality: ⭐⭐⭐⭐ The attack designs are clear, though the mapping between psychological principles and attacks could be more intuitive.
- Value: ⭐⭐⭐⭐⭐ Provides a systematic, theoretically guided framework for LLM safety evaluation, with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Aligning Large Language Models with Implicit Preferences from User-Generated Content](pugc_align_implicit_pref_ugc.md)
- [\[ACL 2025\] Understanding the Repeat Curse in Large Language Models from a Feature Perspective](understanding_the_repeat_curse_in_large_language_models_from_a_feature_perspecti.md)
- [\[ACL 2025\] Perspective Transition of Large Language Models for Solving Subjective Tasks](perspective_transition_of_large_language_models_for_solving_subjective_tasks.md)
- [\[ACL 2025\] Evaluating Language Models as Synthetic Data Generators](evaluating_lms_synthetic_data_gen.md)
- [\[ACL 2025\] Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models](attention_speaks_volumes_localizing_and_mitigating_bias_in_language_models.md)

</div>

<!-- RELATED:END -->
