---
title: >-
  [Paper Note] Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework
description: >-
  [ICLR2026][LLM Evaluation][human evaluation] This paper proposes the HUMAINE framework, which conducts multi-dimensional (5-axis), multi-turn human preference evaluations of 28 SOTA models using 23,404 demographically stratified participants. A hierarchical Bayesian BTD model reveals that age is the largest driver of preference heterogeneity (mean rank shift ±2.8), demonstrating that a single aggregated leaderboard is insufficient to reflect the true preferences of diverse populations.
tags:
  - ICLR2026
  - LLM Evaluation
  - human evaluation
  - preference heterogeneity
  - demographic bias
  - Bradley-Terry-Davidson
  - LLM leaderboard
  - psychometrics
date: 2026-05-08
content_hash: 4530e646253c44e8
---

# Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework

**Conference**: ICLR2026  
**arXiv**: [2603.04409](https://arxiv.org/abs/2603.04409)  
**Code**: [Leaderboard](https://huggingface.co/spaces/ProlificAI/humaine-leaderboard) / [Dataset](https://huggingface.co/datasets/ProlificAI/humaine-evaluation-dataset)  
**Area**: LLM Evaluation  
**Keywords**: human evaluation, preference heterogeneity, demographic bias, Bradley-Terry-Davidson, LLM leaderboard, psychometrics

## TL;DR
This paper proposes the HUMAINE framework, which conducts multi-dimensional (5-axis), multi-turn human preference evaluations of 28 SOTA models using 23,404 demographically stratified participants. A hierarchical Bayesian BTD model reveals that age is the largest driver of preference heterogeneity (mean rank shift ±2.8), demonstrating that a single aggregated leaderboard is insufficient to reflect the true preferences of diverse populations.

## Background & Motivation

1. **Evaluation Gap**: LLM evaluation suffers from two paradigmatic flaws:
    - **Automated benchmarks** (MMLU, HELM, BIG-Bench): measure technical capability but ignore human-computer interaction quality, suffering from Goodhart's Law (optimizing metrics rather than user experience).
    - **Human preference platforms** (Chatbot Arena): exhibit three methodological shortcomings — (a) anonymous self-selected users leading to non-representative sampling; (b) shallow evaluation with minimal interaction; (c) binary voting as a single-metric simplification.
2. **Ignored Preference Heterogeneity**: Santurkar et al. (2023) demonstrated that evaluators' demographic characteristics significantly influence LLM preferences, yet existing leaderboards aggregate all populations into a single score.
3. **Bias in the Third Paradigm**: LLM-as-a-judge offers scaling advantages but exhibits systematic biases (preference for verbose outputs, position bias, etc.) and should not substitute human evaluation.
4. **Goal**: To design a multi-dimensional, demographically aware evaluation framework that addresses three validity threats: sampling bias, insufficient evaluation depth, and metric oversimplification.

## Method

### Participant Recruitment and Stratification Design

- **Platform**: Prolific, compensated at the recommended rate of £9/hr
- **Scale**: 23,404 participants; 119,890 multi-dimensional judgments
- **Stratification**: 22 demographic strata covering:
    - Geography: US / UK
    - Age: 18–34, 35–54, 55+
    - Ethnicity: Asian, Black/African American, White, Other (with corresponding UK classifications)
    - Political affiliation: Democrat / Republican / Independent (US); Conservative / Labour / Liberal Democrat / Green / Reform UK (UK)
- 1,848–2,636 comparisons per stratum; median conversation length of 6 turns

### Data Collection Procedure

1. Participants interact with two **anonymous** models displayed side-by-side.
2. Participants freely choose conversation topics, with a minimum of 3 turns.
3. Each message is **simultaneously submitted** to both models — ensuring fair comparison under identical context.
4. **TrueSkill adaptive matchmaking**: maintains skill and uncertainty estimates for each model, selecting the most uncertain pairings to maximize information gain.
5. **Real-time quality monitoring**: gpt-4o-mini flags low-quality inputs (single-word replies, copy-paste repetition); three warnings result in removal (affecting <1.6% of participants).
6. After the conversation, participants evaluate 5 dimensions and select the preferred model or indicate a tie.

### Five-Dimensional Evaluation Metrics

| Dimension | Description | Discriminability |
|-----------|-------------|-----------------|
| Core Task Performance & Reasoning | Task completion and reasoning quality | Moderate |
| Communication Style & Presentation | Language style, tone, and appropriateness of detail | Moderate |
| Interaction Fluidity & Adaptiveness | Conversational fluency and contextual adaptability | Moderate |
| Trust, Ethics & Safety | Reliability, transparency, ethics, and safety | Lowest (65% tie rate) |
| Overall Winner | Holistic preference judgment | Highest (10% tie rate) |

### Hierarchical Bayesian Bradley-Terry-Davidson Model

The core statistical engine extends the classical BT model to handle ties and demographic heterogeneity:

$$\text{logit}(P_{ij}^{(k)}) = \theta_i^{(k)} - \theta_j^{(k)} + \sum_g u_{ig}^{(k)} - \sum_g u_{jg}^{(k)}$$

where:
- $\theta_i^{(k)}$: global skill parameter for model $i$ on metric $k$
- $u_{ig}^{(k)}$: preference adjustment for demographic group $g$ toward model $i$
- $\nu_k$: tie-tendency parameter (quantifying metric discriminability)
- $\tau_g$: heterogeneity parameter (quantifying the magnitude of inter-group preference variation)

The **partial pooling** mechanism jointly learns global skill parameters and group-level adjustments, correctly attributing preference drivers even when participants belong to multiple groups simultaneously (e.g., Asian + 18–34 + Democrat).

**Scoring Metric (Winshare)**: A model's expected total score in a full round-robin tournament across all models (win = 1, tie = 0.5, maximum = 27).

### LLM Judge Post-hoc Analysis

- gpt-4.1 performs structured classification of all conversations (task type, domain, complexity, etc.).
- **Strict separation**: LLM analysis is conducted purely post-hoc and does not influence human preference ratings; it serves solely as an interpretive tool.

## Key Experimental Results

### Overall Ranking (Overall Winner)

| Rank | Model | Score (Winshare) | P(best) |
|------|-------|-----------------|---------|
| 1 | google/gemini-2.5-pro | Highest | **95.6%** |
| 2 | deepseek/deepseek-chat-v3-0324 | Second | — |
| 3–5 | mistral/magistral-medium, x-ai/grok-4, x-ai/grok-3 | Closely contested | — |

Gemini-2.5-pro leads with a decisive margin; confidence intervals among subsequent models overlap substantially.

### Demographic Heterogeneity

| Demographic Axis | Mean Rank Shift | Note |
|-----------------|----------------|------|
| **Age** | **±2.8 ranks** | Largest driver of heterogeneity |
| Political affiliation | ±1.5 ranks | Moderate |
| Ethnicity | ±1.3 ranks | Smallest |

**Specific age effect examples**:
- mistral/magistral-medium: ranked 1–2 among younger users (18–34), **drops to 5–10 among 55+ users**
- google/gemini-2.5-pro: ranking improves with age, consistently first in the 55+ group
- Tie rate increases from 9.7% (18–34) to 12.5% (55+) (+29%), indicating greater decisional difficulty among older users

### Cross-Dimension Rank Variation

| Model | Task Performance | Communication Style | Interaction Fluidity | Trust & Safety |
|-------|-----------------|--------------------|--------------------|---------------|
| x-ai/grok-3 | **2** | 8 | 8 | — |
| mistral/magistral-medium | 7 | — | **2** | 12 |
| google/gemini-2.5-pro | 1 | 1 | 1 | 1 |

Gemini-2.5-pro's advantage lies in **cross-dimensional consistency**; other models exhibit uneven profiles.

### Evaluation Dimension Discriminability

| Dimension | Tie Rate | Interpretation |
|-----------|----------|----------------|
| Overall Winner | **10%** | Most decisive — users form clear holistic preferences |
| Core Task Performance | ~30% | Moderate |
| Communication Style | ~35% | Moderate |
| Interaction Fluidity | ~40% | Moderate-high |
| Trust, Ethics & Safety | **65%** | Highly ambiguous — models converge on safety, or difficult to assess in short conversations |

### Conversation Data Analysis

| Dimension | Statistics |
|-----------|-----------|
| Task type | Information retrieval 71.5%, personal advice 10.5%, project planning 2.7% |
| Domain | 41 domains; health/medicine 12.9%, sports 8.8%, technology 8.1% |
| Task complexity | Mean 3.54/5; 43.2% moderate complexity, 12.3% high complexity |
| Goal achievement | Mean 4.32/5; 92.6% achieved their goal |

## Highlights & Insights
- **Age is the largest source of preference divergence**: Model rankings can shift by up to ±2.8 positions across age groups — challenging all leaderboards that rely on anonymous, unstratified samples.
- **"Best" is a context-dependent illusion**: Gemini-2.5-pro ranks only 13th on HELM technical benchmarks, yet leads human preference evaluations with 95.6% posterior probability — revealing a substantial gap between technical accuracy and user satisfaction.
- **Safety dimension is nearly indistinguishable**: A 65% tie rate implies that safety evaluation in open-ended conversations requires a fundamentally different methodological design.
- **Methodological innovation**: The combination of hierarchical Bayesian BTD, post-hoc demographic stratification, and TrueSkill adaptive matchmaking substantially surpasses Chatbot Arena in statistical rigor.

## Limitations & Future Work
- **Geographic scope**: Coverage limited to US and UK English-speaking users; non-English languages and other cultural contexts are absent.
- **Open-ended conversations skew toward information retrieval**: 71.5% of tasks are information-seeking, underrepresenting preference differences in specialized domains such as coding and creative writing.
- **Safety evaluation failure**: The Trust & Safety dimension exhibits very low discriminability in open-ended conversations; targeted scenarios (adversarial prompting, sensitive topics) are needed.
- **Repeated participation**: The same individual may participate across multiple tournaments; although the hierarchical model addresses this, learning effects may still be introduced.
- **Snapshot evaluation**: The 28 models reflect a snapshot at time of writing; continuous model updates limit the longevity of the findings.

## Related Work & Insights
- **vs. Chatbot Arena (Zheng et al., 2023)**: HUMAINE improves along three critical dimensions — representative sampling (stratified vs. self-selected), evaluation depth (multi-turn + multi-dimensional vs. single-turn + binary), and statistical methodology (hierarchical Bayesian vs. simple ELO).
- **vs. Santurkar et al. (2023)**: Prior work demonstrated that demographics influence preferences but did not provide a systematic framework; HUMAINE operationalizes this finding into an actionable evaluation system.
- **vs. LLM-as-a-judge**: The paper explicitly positions LLMs as interpretive tools rather than substitutes — human preference data remains irreplaceable.
- **Implications**: Future LLM evaluation should consider providing customized leaderboards for different user populations — *who evaluates* and *what is evaluated* are equally important.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-dimensional demographically aware evaluation framework represents a new paradigm, though the core statistical method (BTD) is an engineering application of established techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 23,404 participants × 28 models × 5 dimensions × 22 demographic strata — exceptional scale and coverage.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with compelling presentation of findings, though somewhat lengthy with room for compression.
- Value: ⭐⭐⭐⭐⭐ Exposes fundamental flaws in current LLM evaluation practice; the open release of the dataset and leaderboard provides substantial community value.

### Statistical Model: Hierarchical Bayesian Bradley-Terry-Davidson

- Extends the classical BT model to handle ties and demographic heterogeneity.
- Learns global skill parameters $\theta$ and demographic adjustments $u$ for each model–metric combination.
- Heterogeneity parameter $\tau$ quantifies the magnitude of preference variation.
- Partial pooling disentangles mixed demographic effects.
- Post-stratification to US/UK census data enhances representativeness.

### Supplementary Use of LLM Judge
- gpt-4.1 is used for post-hoc conversation analysis (not competitive ranking).
- Classifies task type, domain, complexity, goal achievement, and engagement.

## Key Findings

### Finding 1: Overall Performance Ranking
Models are evaluated using Score (Winshare) — the expected total score in a round-robin tournament against all other models (maximum 27).

| Rank | Model | P(best) |
|------|-------|---------|
| 1 | google/gemini-2.5-pro | 95.6% |
| 2 | deepseek/deepseek-chat-v3-0324 | — |
| 3 | mistralai/magistral-medium-2506 | — |
| 4 | x-ai/grok-4 | — |
| 5 | x-ai/grok-3 | — |

- Gemini-2.5-Pro holds first place with 95.6% posterior probability, with a clear margin over second place.
- Mid-ranked models have heavily overlapping confidence intervals and are statistically indistinguishable.

### Finding 2: Age is the Primary Driver of Preference Heterogeneity

| Demographic Axis | Mean Rank Shift |
|-----------------|----------------|
| Age | ±2.8 positions |
| Political affiliation | ±1.5 positions |
| Ethnicity | ±1.3 positions |

Specific examples:
- magistral-medium-2506: ranked 1–2 among 18–34 users; ranked 5–10 among 55+ users.
- gemini-2.5-pro: ranks higher among older users.

**Tie rate by age group**:

| Age Group | Tie Rate |
|-----------|----------|
| 18–34 | 9.7% |
| 35–54 | 11.1% |
| 55+ | 12.5% |

Older users exhibit lower decisiveness (tie rate 29% higher), suggesting different age groups perceive model differentiation differently.

### Finding 3: Substantial Performance Variation Across Dimensions
- grok-3: Task Performance rank 2, Communication Style rank 8.
- magistral-medium-2506: Interaction Fluidity rank 2, Trust & Safety rank 12.
- gemini-2.5-pro: ranks first across all dimensions (advantage lies in consistency and balance).

### Finding 4: Large Variation in Dimension Discriminability

| Evaluation Dimension | Tie Rate |
|---------------------|----------|
| Trust, Ethics & Safety | 65% |
| Communication Style | 18% |
| Core Task Performance | 35% |
| Interaction Fluidity | 24% |
| Overall Winner | 10% |

- A 65% tie rate for Trust & Safety means this dimension cannot be reliably evaluated in open-ended conversations.
- A 10% tie rate for Overall Winner indicates that users readily form clear holistic preferences.

### Conversation Analysis
- Task type: information-seeking 71.5%, personal advice 10.5%.
- Domain: health/medicine 12.9%, sports 8.8%, technology 8.1%.
- Task complexity: median 4/5; 43.2% moderate complexity.
- Goal achievement: 92.6% rated 4–5/5.

## Strengths and Limitations

### Strengths
- First large-scale demographically stratified LLM preference evaluation.
- Multi-dimensional evaluation reveals the inadequacy of single leaderboards.
- Bayesian hierarchical model rigorously handles mixed demographic effects.
- Living benchmark with continuous updates.

### Limitations
- Coverage limited to US/UK populations; global cultural contexts are absent.
- Demographic dimensions are constrained (gender, education, and socioeconomic status are not included).
- Short conversations cannot capture long-term consistency or performance degradation.
- Text-only interaction; multimodal capabilities are not assessed.
- Trust & Safety cannot be effectively evaluated in open-ended conversations.

## Personal Assessment

### Novelty ⭐⭐⭐⭐
- Introducing psychometrics and demographic stratification into LLM evaluation is an important advance.
- Multi-dimensional evaluation combined with demographic heterogeneity analysis fills a critical gap.

### Experimental Scale ⭐⭐⭐⭐⭐
- 23,404 participants, 28 models, 119,890 multi-dimensional judgments.
- The 22-stratum stratification design is highly rigorous.

### Methodological Rigor ⭐⭐⭐⭐
- The hierarchical Bayesian BTD model appropriately handles mixed effects and uncertainty.
- TrueSkill adaptive sampling efficiently utilizes data.
- Post-stratification to census data enhances representativeness.

### Practical Impact ⭐⭐⭐⭐
- The finding that age is the largest driver of preference heterogeneity has direct implications for model development and deployment.
- It warns against the risks of optimizing models based on the preferences of a narrow technical community.

### Overall Rating ⭐⭐⭐⭐
A methodologically rigorous large-scale human preference study. The core finding — that different age groups exhibit significantly different LLM preferences — challenges the assumption that a single leaderboard serves all users, with important implications for fair and inclusive AI development.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Preference Leakage: A Contamination Problem in LLM-as-a-judge](preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)
- [\[ICLR 2026\] Disentangling Shared and Private Neural Dynamics with SPIRE: A Latent Modeling Framework for Deep Brain Stimulation](disentangling_shared_and_private_neural_dynamics_with_spire_a_latent_modeling_fr.md)
- [\[ICLR 2026\] Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective](can_vision_language_models_assess_graphic_design_aesthetics_a_benchmark_evaluati.md)
- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[ICLR 2026\] Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)

</div>

<!-- RELATED:END -->
