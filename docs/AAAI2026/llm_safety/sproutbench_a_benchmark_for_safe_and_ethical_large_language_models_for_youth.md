---
title: >-
  [Paper Note] SproutBench: A Benchmark for Safe and Ethical Large Language Models for Youth
description: >-
  [AAAI 2026][LLM Safety][LLM safety evaluation] This paper introduces SproutBench, an evaluation benchmark comprising 1,283 developmentally-grounded adversarial prompts…
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "LLM safety evaluation"
  - "child AI safety"
  - "developmental psychology"
  - "age-stratified benchmark"
  - "youth protection"
date: 2026-05-08
content_hash: be4d16c80a9a6735
---

# SproutBench: A Benchmark for Safe and Ethical Large Language Models for Youth

**Conference**: AAAI 2026
**arXiv**: [2508.11009](https://arxiv.org/abs/2508.11009)  
**Code**: None  
**Area**: AI Safety
**Keywords**: LLM safety evaluation, child AI safety, developmental psychology, age-stratified benchmark, youth protection

## TL;DR

This paper introduces SproutBench, an evaluation benchmark comprising 1,283 developmentally-grounded adversarial prompts, designed to systematically assess the safety of 47 LLMs in contexts involving children and adolescents (ages 0–6, 7–12, and 13–18). Key findings reveal that safety and risk prevention are strongly correlated ($\rho = 0.86$), while a significant trade-off exists between interactivity and age-appropriateness ($\rho = -0.48$).

## Background & Motivation

### State of the Field

LLMs are increasingly integrated into educational, entertainment, and social platforms, with children and adolescents becoming a significant user demographic. However, mainstream safety benchmarks (e.g., JailbreakBench, HarmBench) primarily target adult scenarios—focusing on jailbreak prevention and harmful content detection—with the primary goal of minimizing corporate liability, while neglecting the developmental needs of minors.

### Limitations of Prior Work

**Insufficient age-specific coverage**: Existing benchmarks fail to distinguish among the cognitive, emotional, and social risks specific to early childhood (0–6), middle childhood (7–12), and adolescence (13–18).

**Limited risk type coverage**: Benchmarks such as MinorBench and ToxiGen cover only a narrow subset of child safety dimensions.

**Evaluative perspective bias**: Prior work adopts a "risk avoidance" rather than a "developmental promotion" lens, failing to assess whether model outputs are age-appropriate, psychologically safe, and socially constructive.

### Taxonomy of Child-Specific AI Risks

The paper establishes a systematic taxonomy of risks in child–AI interaction:

**Risks arising from LLM outputs**:
- Mental health: emotional dependency, disruption of real-world relationships, inappropriate handling of sensitive topics such as depression
- Social behavior: exposure to manipulative or antisocial content, erosion of empathy and conflict-resolution skills
- Misinformation: limited critical thinking heightens susceptibility to false or harmful content
- Cognitive learning: excessive use reducing creativity and causing cognitive overload
- Privacy: children may inadvertently share sensitive personal data

**Risks arising from user misuse**:
- Social behavior: using LLMs for pranks, cyberbullying, or circumventing safety measures via prompt engineering
- Academic integrity: cheating and spreading misinformation

## Method

### Overall Architecture

The SproutBench construction pipeline proceeds as follows:
1. **Developmental psychology knowledge base construction**: Drawing on Piaget's and Vygotsky's developmental theories to define cognitive characteristics and language styles for each age group.
2. **Adversarial prompt generation pipeline**: Combining age group, risk behavior, and query type to generate contextually rich prompts.
3. **Multi-dimensional evaluation**: Assessing LLM responses across six dimensions.

### Key Designs

#### 1. **Developmental Knowledge Base**

Developmental characteristics are defined for three age groups:
- **0–6**: Limited abstract reasoning, simple sentence structures
- **7–12**: Developing impulse control, exploratory behavior
- **13–18**: Complex emotional queries, value conflicts

Risk behavior sets $B_a$ are stratified by age group (see table below), while query types $Q_t$ apply across all ages: Testing, Manipulative, Emotional, and Exploratory.

The generation strategy is defined as a triplet: $s = (a, b, q_t)$, where $a \in A$ (age group), $b \in B_a$ (risk behavior), and $q_t \in Q_t$ (query type).

#### 2. **Dataset Construction**

The benchmark comprises 1,283 adversarial prompts drawn from two sources:

| Age Group | Behavior Type | Count |
|-----------|--------------|-------|
| **0–6** | Inadvertent sensitive word input | 65 |
| | Excessive reliance on AI companionship | 40 |
| | Emotional expression needs | 43 |
| | Imitation of play behavior | 134 |
| **7–12** | Command manipulation attempts | 51 |
| | Fictitious identity/scenario creation | 71 |
| | Elicitation of sensitive content | 71 |
| | Testing model tolerance | 111 |
| | Testing privacy boundaries | 40 |
| | Social boasting motivation | 40 |
| | Imitation of violent game behavior | 40 |
| | Disguised questioning | 40 |
| | Malicious AI use | 40 |
| **13–18** | Academic cheating attempts | 48 |
| | Circumventing restricted topics | 99 |
| | Emotional manipulation testing | 98 |
| | Ideological challenges | 53 |
| | Value conflict testing | 67 |
| | Social engineering testing | 35 |
| **Total** | | **1,283** |

Of these, 200 prompts are adapted from the Safe-Child-LLM dataset (via GPT-4 adaptation), and 683 are newly generated.

#### 3. **Generation Pipeline**

1. **Prompt formulation**: Age-appropriate linguistic styles and motivations are retrieved from the knowledge base according to strategy $s$.
2. **Model interaction**: Prompts are fed to GPT-3.5 (temperature = 0.7) to generate queries $D_s = M_{LLM}(f_{prompt}(s))$.
3. **LLM quality verification**: GPT-4 evaluates linguistic naturalness, age-appropriateness, motivational consistency, and potential risk.

#### 4. **Evaluation Dimensions and Scoring**

Six evaluation dimensions are assessed on a 0–5 scale: age-appropriateness, educational value, emotional support, interactivity, risk prevention, and safety.

Automated scoring is performed using Qwen-2.5, achieving Cohen's Kappa = 0.78 in agreement with annotations from three child developmental psychologists.

### Loss & Training

SproutBench is an evaluation benchmark rather than a training methodology; no model training is involved. Evaluation is conducted under standard inference settings on 8×NVIDIA RTX 4090 GPUs.

## Key Experimental Results

### Main Results

47 LLMs are evaluated, ranging from 135M to 70B parameters.

**Top-3 models**:

| Model | Mean Overall Score | Std. Dev. | Safety | Risk Prevention |
|-------|--------------------|-----------|--------|----------------|
| llama2:7b | **4.61** | 0.13 | 4.93 | 4.86 |
| llama2:70b | **4.58** | 0.13 | 4.92 | 4.84 |
| gemma2:9b | **4.56** | 0.04 | — | — |

**Bottom-3 models**:

| Model | Mean Overall Score | Std. Dev. | Safety | Risk Prevention |
|-------|--------------------|-----------|--------|----------------|
| smollm2:135m | 3.26 | 0.35 | 3.45 | 2.99 |
| tinyllama:1.1b | 3.41 | 0.21 | 3.54 | 3.19 |
| phi3:3.8b | 3.52 | 0.65 | 3.68 | 3.47 |

### Ablation Study

**Dimension correlation analysis (Spearman correlation coefficients)**:

| Dimension Pair | Correlation | Interpretation |
|----------------|-------------|---------------|
| Safety ↔ Risk Prevention | **$\rho = 0.86$** | Positive; protective behaviors are consistent |
| Age-Appropriateness ↔ Educational Value | $\rho = 0.81$ | Positive |
| Emotional Support ↔ Age-Appropriateness | $\rho = 0.74$ | Positive |
| Interactivity ↔ Age-Appropriateness | **$\rho = -0.48$** | Negative; critical trade-off |
| Interactivity ↔ Safety | $\rho = 0.12$ | Weak; nearly orthogonal dimensions |

**PCA analysis**: PC1 (90.28% of variance) = safety axis; PC2 (5.07% of variance) = interactivity axis; the two axes are approximately orthogonal.

**Model scale and performance (scale distribution among bottom-10% performers)**:

| Scale | Dataset Proportion | Low-Score Proportion | Overrepresentation Factor |
|-------|--------------------|----------------------|--------------------------|
| Tiny (<500M) | 8.16% | 20% | **2.45×** |
| Small (500M–7B) | 40.8% | 64% | 1.57× |
| Medium (7–30B) | 38.78% | 16% | 0.41× |
| Large (>30B) | 6.12% | 0% | **0×** |

Chi-square test: $\chi^2 = 14.62, p < 0.01$, indicating that small models are significantly overrepresented among low-scoring performers.

### Key Findings

1. **Safety and risk prevention are highly aligned** ($\rho = 0.86$): Models that are safe tend also to be effective at risk prevention.
2. **Interactivity is a double-edged sword**: High interactivity is associated with reduced age-appropriateness ($\rho = -0.48$); exemplar models (e.g., gemma3:12b) achieve a balance between the two.
3. **Small models pose greater risks**: Tiny models are overrepresented in the low-scoring group by a factor of 2.45×, while large models (>30B) are entirely absent from this group.
4. **phi3:3.8b degrades substantially in the 13–18 age group**: Safety scores drop to 2.97, indicating particular vulnerability when handling complex adolescent topics.
5. **K-Means clustering identifies five model archetypes**: Exemplars (purple cluster) = high safety + high interactivity; Underachievers (orange cluster) = high risk + medium interactivity.

## Highlights & Insights

1. **Paradigm shift in evaluation**: The benchmark moves from "risk avoidance" to "developmental promotion," examining whether models actively support children's cognitive, emotional, and social development.
2. **Grounded in developmental psychology theory**: The knowledge base is constructed on Piagetian and Vygotskian frameworks, ensuring developmental appropriateness of the prompts.
3. **Discovery of the interactivity–safety trade-off**: The paper reveals a critical and previously overlooked trade-off—enhancing interactivity may come at the cost of age-appropriateness.
4. **Coverage of 20 distinct risk behaviors**: This substantially exceeds the scope of existing benchmarks, encompassing neglected risks such as online prank imitation, privacy boundary testing, and emotional dependency.

## Limitations & Future Work

1. **Reliance on Qwen-2.5 for automated scoring**: Although Kappa = 0.78 indicates strong expert agreement, automated scoring may miss subtle developmental concerns.
2. **Limited cultural diversity**: Prompts are primarily grounded in Western developmental psychology theories, which may not generalize across all cultural contexts.
3. **Absence of longitudinal study**: The long-term effects of LLM interactions with children are not tracked.
4. **No youth participation in benchmark design**: Future work should involve actual children and adolescents in the design process.
5. **High score variance in some models** (e.g., phi3:3.8b, std. dev. = 0.65) may reflect instability in the evaluation procedure.

## Related Work & Insights

- **UNICEF AI Policy Guidelines for Children**: Based on the UN Convention on the Rights of the Child, advocating for safe, fair, and privacy-respecting design principles.
- **Safe-Child-LLM**: Focuses on mental health and safety, yet still exhibits failure rates of 30–40%.
- **JailbreakBench / ToxiGen**: Adult-oriented safety benchmarks that overlook youth-specific risks such as grooming and excessive AI dependency.
- **Implications for the AI safety community**: Age-stratified safety evaluation standards are needed, rather than applying adult-oriented criteria uniformly across all user populations.

## Rating

- Novelty: ⭐⭐⭐⭐ — First comprehensive age-stratified LLM safety benchmark
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 47 models, 6 dimensions, and 3 age groups
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, though some content is repetitive
- Value: ⭐⭐⭐⭐ — Fills a critical gap in child-oriented LLM safety evaluation with significant policy implications

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models](../../ICML2026/llm_safety/less_diverse_less_safe_the_indirect_but_pervasive_risk_of_test-time_scaling_in_l.md)
- [\[ICLR 2026\] Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](../../ICLR2026/llm_safety/measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)
- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](gender_bias_in_emotion_recognition_by_large_language_models.md)
- [\[AAAI 2026\] Principles2Plan: LLM-Guided System for Operationalising Ethical Principles into Plans](principles2plan_llm-guided_system_for_operationalising_ethical_principles_into_p.md)
- [\[AAAI 2026\] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](anti-adversarial_learning_desensitizing_prompts_for_large_la.md)

</div>

<!-- RELATED:END -->
