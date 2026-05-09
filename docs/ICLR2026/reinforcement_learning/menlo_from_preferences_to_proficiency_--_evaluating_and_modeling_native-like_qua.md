---
title: >-
  [Paper Note] Menlo: From Preferences to Proficiency – Evaluating and Modeling Native-like Quality Across 47 Languages
description: >-
  [ICLR 2026][Reinforcement Learning][Multilingual Evaluation] This paper proposes the Menlo framework, which decomposes native-like response quality into four dimensions grounded in audience design theory, constructs a preference dataset of 6,423 annotated pairs covering 47 language varieties, and demonstrates that pairwise evaluation combined with RL-trained LLM judges can approach human annotator agreement levels.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Multilingual Evaluation
  - Native-like Quality
  - LLM-as-Judge
  - Preference Learning
  - Audience Design
date: 2026-05-08
content_hash: a6a4fb6877d1f824
---

# Menlo: From Preferences to Proficiency – Evaluating and Modeling Native-like Quality Across 47 Languages

**Conference**: ICLR 2026
**arXiv**: [2509.26601](https://arxiv.org/abs/2509.26601)
**Code**: [https://huggingface.co/datasets/facebook/menlo](https://huggingface.co/datasets/facebook/menlo)
**Area**: Reinforcement Learning
**Keywords**: Multilingual Evaluation, Native-like Quality, LLM-as-Judge, Preference Learning, Audience Design

## TL;DR
This paper proposes the Menlo framework, which decomposes native-like response quality into four dimensions grounded in audience design theory, constructs a preference dataset of 6,423 annotated pairs covering 47 language varieties, and demonstrates that pairwise evaluation combined with RL-trained LLM judges can approach human annotator agreement levels.

## Background & Motivation

**Background**: LLMs are expected to deliver high-quality responses across a wide range of global languages, yet systematic methods for evaluating "native-like quality" remain underdeveloped. Traditional assessments such as standardized tests are difficult to scale and poorly suited to realistic conversational settings.

**Limitations of Prior Work**: Existing multilingual preference datasets cover few languages, lack localized prompts, exhibit low inter-annotator agreement, and fail to distinguish specific quality dimensions. Zero-shot LLM judges still show a significant gap relative to human annotators in multilingual scenarios.

**Key Challenge**: Native-like quality is not a single, fixed standard; rather, it depends on the relationship between speaker and audience (the style axiom in sociolinguistics)—the same language can embody different "native" norms across cultures, regions, and contexts.

**Goal**: (1) Operationalize native-like quality evaluation by decomposing it into measurable dimensions; (2) construct a large-scale, high-quality multilingual preference dataset; (3) train reliable LLM judges as cost-effective alternatives to human evaluation.

**Key Insight**: Drawing on Audience Design theory, the framework steers model generation toward contextually appropriate "native" styles by defining the target audience, and designs annotation rubrics that reduce subjective variation.

**Core Idea**: Native-like quality is decomposed into four dimensions—fluency, register, localized register, and localized factuality—and pairwise RL training enables LLM judges to reach human-level performance across 47 languages.

## Method

### Overall Architecture
The Menlo framework consists of four steps: (1) expert-authored English prompt templates with placeholders, organized according to four quality dimensions; (2) translation and localization into 47 language varieties by native speakers; (3) development of detailed 5-point scoring rubrics; (4) LLM-generated response pairs rated by native-speaker annotators. LLM judges trained on these annotations are then used for automated evaluation and for improving policy models' multilingual capabilities.

### Key Designs

1. **Four-Dimensional Quality Decomposition and Parametric Templates**:

    - *Function*: Decompose the abstract notion of "native-like quality" into independently assessable concrete dimensions.
    - *Mechanism*: The four dimensions are: (a) linguistic quality and coherence (fluency); (b) alignment with cultural and linguistic norms of a specific language variety (localized register); (c) factual accuracy and grounding in local knowledge (localized factuality); (d) overall writing style and helpfulness. Parametric templates $T(locale, holiday, ...)$ are used to generate localized prompts.
    - *Design Motivation*: Grounded in Audience Design theory, the target audience (addressees and ratified participants) is explicitly defined to steer linguistic style toward a specific "native" norm.

2. **Pairwise vs. Pointwise Evaluation**:

    - *Function*: Systematically compare the effect of pairwise versus pointwise evaluation on LLM judge performance.
    - *Mechanism*: Under zero-shot settings, the model evaluates two responses simultaneously (pairwise) versus evaluating a single response in isolation (pointwise). Pairwise evaluation yields gains of up to +12.4% in Macro-F1 and up to +17.9% in preference accuracy, surpassing even few-shot pointwise evaluation with demonstrations (a gap of approximately 5.5%).
    - *Design Motivation*: Pairwise evaluation provides an anchoring effect—directly contrasting two candidate responses enables more accurate quality discrimination.

3. **Multi-Task Pairwise Judge Trained with RL**:

    - *Function*: Train high-quality LLM judges capable of replacing human annotators.
    - *Mechanism*: Qwen3-4B and Llama4-Scout are fine-tuned as judges on the Menlo training set, comparing SFT and RL training (PPO with reward shaping). RL models consistently outperform SFT counterparts. The multi-task Llama4-Scout (jointly trained on all four dimensions with shaped rewards) achieves the strongest performance across 47 languages, approaching human inter-annotator agreement. The trained judges can further serve as generative reward models to directly improve policy models' multilingual capabilities.
    - *Design Motivation*: Zero-shot judges remain insufficiently reliable; RL training is necessary to close the gap with human annotators.

### Loss & Training
SFT employs standard cross-entropy loss. RL uses the PPO algorithm, with a reward function combining rating prediction accuracy and the quality of rating explanations. Reward shaping provides finer-grained feedback signals for intermediate quality levels.

## Key Experimental Results

### Main Results

| Model | Mode | Macro-F1 | Preference Accuracy |
|-------|------|----------|---------------------|
| Qwen3-4B | Zero-shot Pointwise | 23.06 | 40.54 |
| Qwen3-4B | Zero-shot Pairwise | 35.46 (+12.4) | 57.13 (+16.6) |
| GPT-4.1 | Zero-shot Pointwise | 32.23 | 41.73 |
| GPT-4.1 | Zero-shot Pairwise | 38.53 (+6.3) | 59.23 (+17.5) |
| Llama4-Scout | Zero-shot Pairwise | 36.11 | 56.25 |
| o3 | Zero-shot Pairwise | 35.35 | 58.72 |

### Ablation Study

| Configuration | Macro-F1 | Preference Accuracy | Notes |
|---------------|----------|---------------------|-------|
| No Rubric (Pointwise) | 16.00 | 33.52 | Qwen3-4B |
| With Rubric (Pointwise) | 23.06 (+7.06) | 40.54 (+7.02) | Rubric yields large gains |
| With Rubric (Pairwise) | 35.46 | 57.13 | Pairwise further improves |
| SFT Fine-tuning | ~38 | ~60 | Modest improvement |
| RL Multi-task + Shaping | ~43 | ~65 | Approaches human agreement |

### Key Findings
- Pairwise evaluation consistently outperforms pointwise evaluation across all models, with gains exceeding those from few-shot in-context demonstrations.
- Detailed scoring rubrics benefit smaller models most significantly (Qwen3-4B: +7.06 F1), while the effect diminishes for larger models.
- RL training consistently outperforms SFT; reward shaping further improves discrimination at intermediate quality levels.
- RL-trained judges can serve as generative reward models to improve policy models, though LLM evaluators tend to overestimate improvements by approximately +0.6 relative to human judgments.
- The Menlo dataset achieves substantially higher inter-annotator agreement (Krippendorff's $\alpha = 0.84$) than existing multilingual preference datasets.

## Highlights & Insights
- **Unexpected Advantage of Pairwise Evaluation**: Even when the ultimate goal is pointwise scoring, pairwise evaluation provides more reliable signals—a finding with broad implications for all LLM-as-Judge applications.
- **Computational Application of Audience Design Theory**: Translating a sociolinguistic theory into an operational NLP evaluation framework represents a cross-disciplinary methodological contribution worthy of broader adoption.

## Limitations & Future Work
- The 47 language varieties still exclude many low-resource languages.
- Parametric templates may not fully capture all cultural nuances.
- RL-trained judges continue to overestimate improvements relative to human evaluation (a gap of approximately +0.6).
- Whether improvements from generative reward models translate to genuine gains in real user experience remains to be validated.

## Related Work & Insights
- **vs. Chatbot Arena / MT-Bench**: These benchmarks focus on general response quality rankings, whereas Menlo specifically targets "native-like" dimensions, particularly cultural and localization aspects.
- **vs. RECON**: RECON also conducts multilingual evaluation but with narrower language coverage (9 vs. 47 languages) and lacks localized prompts and pairwise evaluation design.

## Rating
- Novelty: ⭐⭐⭐⭐ Introduces audience design theory into NLP evaluation; the four-dimensional decomposition is theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale annotation across 47 languages with systematic ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The framework is clearly presented with natural progression from theory to practice.
- Value: ⭐⭐⭐⭐⭐ The high-quality multilingual evaluation dataset and methodology offer significant contributions to the research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RM-R1: Reward Modeling as Reasoning](rm-r1_reward_modeling_as_reasoning.md)
- [\[ICLR 2026\] AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization](autoqd_automatic_discovery_of_diverse_behaviors_with_quality-diversity_optimizat.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[NeurIPS 2025\] The Burden of Interactive Alignment with Inconsistent Preferences](../../NeurIPS2025/reinforcement_learning/the_burden_of_interactive_alignment_with_inconsistent_preferences.md)

</div>

<!-- RELATED:END -->
