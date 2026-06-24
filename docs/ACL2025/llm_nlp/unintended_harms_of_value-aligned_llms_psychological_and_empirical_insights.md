---
title: >-
  [Paper Note] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights
description: >-
  [ACL 2025][LLM (Other)] This study is the first to systematically demonstrate that LLMs aligned with Schwartz values carry unintended safety risks—specifically, certain value dimensions are significantly correlated with distinct safety risk categories. Drawing from psychological perspectives, this paper explains the origins of these associations and proposes a mitigation strategy that suppresses the relevant values via prompting to effectively reduce harmful behaviors.
tags:
  - "ACL 2025"
  - "LLM (Other)"
date: 2026-05-08
content_hash: 78bc340497f468fc
---

# Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights

**Conference**: ACL 2025  
**arXiv**: [2506.06404](https://arxiv.org/abs/2506.06404)  
**Authors**: Sooyung Choi, Jaehyeok Lee, Xiaoyuan Yi, Jing Yao, Xing Xie, JinYeong Bak (Sungkyunkwan University & Microsoft Research Asia)  
**Code**: [GitHub](https://github.com/Human-Language-Intelligence/Unintended-Harms-LLM)  
**Area**: LLM Safety, Value Alignment, Personalized LLMs

---

## TL;DR

This study is the first to systematically demonstrate that LLMs aligned with Schwartz values carry unintended safety risks—specifically, certain value dimensions are significantly correlated with distinct safety risk categories. Drawing from psychological perspectives, this paper explains the origins of these associations and proposes a mitigation strategy that suppresses the relevant values via prompting to effectively reduce harmful behaviors.

---

## Background & Motivation

As the adoption of LLMs expands, constructing personalized LLMs (aligning models with specific human values) has garnered significant attention. However, aligning models with individual values introduces potential safety hazards, as certain values may be inherently associated with harmful information.

**Core Problem**:

1. Are value-aligned LLMs more prone to harmful behaviors compared to untuned or other fine-tuned models?
2. What associations exist between specific value dimensions and particular safety risk categories?
3. What are the psychological mechanisms underlying these associations? Can they guide the design of mitigation strategies?

**Theoretical Foundation**: This paper adopts the Schwartz Theory of Basic Human Values as its analytical framework. The theory defines 10 basic values (Achievement, Power, Hedonism, Self-Direction, Stimulation, Security, Conformity, Tradition, Benevolence, Universalism) grouped into 4 higher-order dimensions (Openness to Change, Self-Enhancement, Conservation, Self-Transcendence). Existing psychological research indicates systematic correlations between human values and behavioral tendencies (e.g., violence, drug use, and criminal behavior), providing a solid foundation for investigating the value-safety risk relationship in LLMs.

---

## Method

### 1. Value Alignment Training

- **Base Model**: Llama-2 7B, fine-tuned using LoRA.
- **Value Distribution Sampling**: Generates 154 Schwartz value distributions:
    - 14 extreme distributions: one target value is highly important (score of 6) while others are unimportant (score of 1), or one higher-order group is dominant.
    - 140 realistic distributions: selected from the European Social Survey based on the top 10 most similar distributions (using Jensen-Shannon divergence) to the extreme distributions.
- **Training Method**: Value Injection Method (VIM), a two-stage training approach—first generating argument texts aligned with the target values, and then predicting agreement with value-related statements.
- **Training Data**: Touché23-ValueEval (8K samples containing arguments on social issues annotated with Schwartz values).
- **Alignment Validation**: Evaluated using the PVQ40 questionnaire, where VIM achieves an $NMSE = 0.0759$, outperforming the ICL baseline of $0.1079$.

### 2. Control Experiment Design

Five reference model types are trained and compared:

| Type | Dataset | Size |
|------|--------|------|
| Instruction Tuning | Alpaca | 52K |
| Instruction Tuning | Dolly | 15K |
| Traditional NLP Task | Grammar (JFLEG+C4_200M) | 14K |
| Traditional NLP Task | Samsum | 16K |
| Value Alignment | Touché23-ValueEval | 8K |

### 3. Safety Evaluation

The models are evaluated using four comprehensive safety benchmarks:

- **RealToxicityPrompts**: 3K toxic prompts; evaluated for toxicity using PerspectiveAPI.
- **HolisticBiasR**: 17.7K bias prompts; evaluated for negative regard using a Regard classifier.
- **HEx-PHI**: 330 harmful instructions across 11 safety categories; evaluated by a GPT-4o judge.
- **BeaverTails-Evaluation**: 700 instructions across 14 safety categories; classified by GPT-4o.

### 4. Mitigation Strategy: Value-Based Prompting

Four prompting strategies are compared: Input Only, Safety Prompt, Value Prompt (which instructs the model to ignore or suppress relevant values), and Dual Prompt (combining both safety and value instructions). These are tested on both value-aligned and vanilla models.

---

## Key Experimental Results

### Evaluation Results on Safety Benchmarks

| Model Type | Dataset | Expected Max Toxicity | Toxicity Probability | Negative Regard Rate | Bias Score |
|----------|--------|-------------|----------|--------|----------|
| Untuned | Vanilla | 0.35 | 17.02% | 7.59% | 94.43% |
| Instruction Tuning | Alpaca | 0.19 | 4.89% | 15.85% | 94.98% |
| Instruction Tuning | Dolly | 0.21 | 5.92% | 15.48% | 93.75% |
| NLP Task | Grammar | 0.20 | 5.16% | 11.66% | 92.96% |
| NLP Task | Samsum | 0.36 | 17.61% | 22.44% | 94.33% |
| ICL Alignment | — | 0.35 | 17.71% | 16.31% | 96.79% |
| **Value Alignment** | **Touché23** | **0.41** | **30.93%** | **18.49%** | **95.73%** |

**Key Findings**: Value-aligned LLMs exhibit the lowest or second-lowest safety levels across almost all metrics, and the differences are statistically significant ($p < 0.001$).

### Value-Safety Risk Correlations (Psychological Explanations)

| Value Dimension | Positively Correlated Risks | Negatively Correlated Risks | Psychological Explanation |
|----------|-----------|-----------|-----------|
| Achievement | — | Hate speech, Sexual content | Seeks socially approved success, rejecting socially unacceptable behaviors. |
| Hedonism | Sexual content, Child exploitation/abuse, Physical harm, Political campaigning | — | Pursues sensory gratification, which is positively correlated with risk-taking and unethical behaviors. |
| Power | Hate speech, Discrimination | Harassment/Abuse, Cyberattacks/Deception, Violence, Terrorism, Privacy violation | Seeks dominance and authority; verbal aggression is leveraged to assert and maintain control. |
| Universalism | — | Deception, Political campaigning | Emphasizes tolerance and protection of all people, correlating negatively with deceptive behavior. |
| Self-Direction | Adult content | — | Emphasizes independent thought and action, which can weaken adherence to external moral constraints. |

### Mitigation Strategy Effectiveness (HEx-PHI Toxicity Scores)

| Safety Category | Prompting Strategy | Vanilla Model | Value-Aligned Model |
|----------|----------|------------|-------------|
| Adult Content (Self-Direction) | Input Only | 4.31 | 4.10 |
| | Safety Prompt | 3.95 (-0.36) | 2.87 (-1.23) |
| | Value Prompt | 3.96 (-0.35) | 2.45 (-1.65) |
| | Dual Prompt | 3.81 (-0.50) | 2.43 (-1.67) |
| Deception (Universalism) | Input Only | 3.51 | 3.43 |
| | Safety Prompt | 3.34 (-0.17) | 2.91 (-0.52) |
| | Value Prompt | 3.39 (-0.13) | 2.69 (-0.74) |
| | Dual Prompt | 3.09 (-0.42) | 2.54 (-0.89) |
| Political Campaigning (Universalism) | Input Only | 3.94 | 3.50 |
| | Safety Prompt | 3.51 (-0.43) | 2.87 (-0.63) |
| | Value Prompt | 3.55 (-0.39) | 2.65 (-0.85) |
| | Dual Prompt | 3.33 (-0.61) | 2.30 (-1.20) |

**Key Findings**: Value prompting is particularly effective on value-aligned models, reducing harmfulness scores by up to 1.67 points. Notably, this mitigation strategy is also generalically effective across standard vanilla models like Llama-3.1, Gemma-2, and Qwen-2.5.

---

## Highlights & Insights

1. **First Systematic Analysis**: Unveils the fine-grained, mapping correlations between specific value dimensions and specific safety risk categories in value-aligned LLMs, validated by psychological theories.
2. **Training Data is Not the Culprit**: Only 5 samples in the Touché23 dataset are potentially toxic (toxicity $> 50\%$) and 0 are classified as toxic ($> 70\%$). This suggests that safety risks stem from the value orientation itself rather than the contamination of training data.
3. **Simple and Effective Mitigation**: Harmful behaviors can be mitigated simply by prompting the model to suppress relevant values, without requiring explicit safety guardrails. This strategy works efficiently on both aligned and non-aligned vanilla models.
4. **Cross-Model Validation**: The proposed mitigation approach consistently maintains its robustness across different base models, including Llama-2, Llama-3.1, Gemma-2, and Qwen-2.5.
5. **Grounded in Psychological Theory**: Each identified value-safety risk correlation is backed by extensive psychological literature (e.g., connecting Hedonism with risk-taking and Power with aggressive behaviors).

---

## Limitations & Future Work

1. **Practical Deployment Challenges**: The mitigation strategy relies on knowing which value dimensions correspond to specific safety risks beforehand, which remains difficult to automate dynamically in real-world applications.
2. **Monolingual Limitation**: Training and evaluations are conducted primarily in English; cultural and linguistic variations in other languages might alter the effects of value alignment.
3. **Cultural Nuances Omitted**: Safety standards and values can vary substantially across different global regions and cultures.
4. **Limited Model Scales**: The study only tests models within the 7B/8B/9B range, leaving the alignment dynamics of larger frontier models unexplored.
5. **Restricted Value Distribution Sampling**: Although 154 distributions are evaluated (including real-world parameters), they may not represent the complete spectrum and complexity of human value diversity.

---

## Related Work & Insights

- **Human Value Theory**: The Schwartz Theory of Basic Human Values acts as a cornerstone in cross-cultural psychology, defining 10 distinct values and 4 high-order groups.
- **Personalized LLMs**: The evolution from personal profile settings (e.g., PersonaChat) to structural value distributions (using VIM).
- **AI Safety Evaluation**: Multi-dimensional safety benchmarks like RealToxicityPrompts, BeaverTails, and HEx-PHI.
- **Safety Risks in Personalization**: Deshpande et al. (2023) show that persona prompting increases toxic outputs; Zeng et al. (2024) demonstrate that human-like personas increase jailbreak success rates.
- **Values and Safety**: Highlights the FULCRA dataset by Yao et al. (2024), and macro-level relationships explored by Ye et al. (2025a,b) that nevertheless omit fine-grained classification of safety risks.

---

## Rating

| Dimension | Rating | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | First to systematically analyze the fine-grained association between value dimensions and safety risk categories. |
| Theoretical Depth | ⭐⭐⭐⭐ | Tight integration of psychological theory and empirical analysis. |
| Experimental Design | ⭐⭐⭐⭐ | Comprehensive validation using 154 value distributions, 4 safety benchmarks, and multiple models. |
| Practical Utility | ⭐⭐⭐ | Simple mitigation strategy, but with limited practical deployability in dynamic environments. |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured with highly persuasive psychological analysis. |
| Overall Rating | ⭐⭐⭐⭐ | A significant contribution to the LLM safety domain, shedding light on the hidden risks of value alignment. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs](do_language_models_mirror_human_confidence_exploring_psychological_insights_to_a.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] An Empirical Study of Iterative Refinements for Non-Autoregressive Translation](an_empirical_study_of_iterative_refinements_for_non-autoregressive_translation.md)

</div>

<!-- RELATED:END -->
