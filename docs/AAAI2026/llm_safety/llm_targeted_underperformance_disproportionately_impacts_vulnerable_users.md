---
title: >-
  [Paper Note] LLM Targeted Underperformance Disproportionately Impacts Vulnerable Users
description: >-
  [AAAI 2026][LLM Safety][user bias] Systematic experiments demonstrate that mainstream LLMs (GPT-4, Claude 3 Opus, Llama 3-8B) exhibit significant discriminatory degradation in information accuracy, truthfulness…
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "user bias"
  - "information accuracy"
  - "vulnerable populations"
  - "sycophancy"
  - "targeted underperformance"
date: 2026-05-08
content_hash: c0ff003e341fe506
---

# LLM Targeted Underperformance Disproportionately Impacts Vulnerable Users

**Conference**: AAAI 2026
**arXiv**: [2406.17737](https://arxiv.org/abs/2406.17737)
**Code**: None
**Area**: LLM Safety
**Keywords**: user bias, information accuracy, vulnerable populations, sycophancy, targeted underperformance

## TL;DR

Systematic experiments demonstrate that mainstream LLMs (GPT-4, Claude 3 Opus, Llama 3-8B) exhibit significant discriminatory degradation in information accuracy, truthfulness, and refusal rates toward users with lower English proficiency, lower educational attainment, and non-US backgrounds, making the most vulnerable users the least reliably served.

## Background & Motivation

**Background**: LLMs have demonstrated strong performance across numerous domains, yet systematic evaluation of undesirable behaviors—hallucination, bias, and harmful content—remains critical. LLMs are expected to help bridge global information accessibility gaps.

**Limitations of Prior Work**: Social psychology research has established pervasive bias against non-native English speakers among native speakers (perceiving them as less educated and less capable). LLM training data and RLHF alignment may amplify such biases. Existing work (e.g., Perez et al., 2023) only preliminarily explored the effect of educational level on sandbagging, with insufficient breadth and depth.

**Key Challenge**: A sharp contradiction exists between the vision of LLMs as tools for information democratization and the reality of their systematic underperformance for vulnerable users—those who need the most help receive the lowest quality of service.

**Goal**: To comprehensively quantify how LLM response quality varies with user characteristics (English proficiency, educational level, nationality), and to reveal the severity and mechanisms of this targeted underperformance.

**Key Insight**: The study conducts large-scale controlled experiments on TruthfulQA (truthfulness) and SciQ (factuality) using carefully designed user bios to simulate query scenarios for users of different backgrounds.

**Core Idea**: LLM misbehavior is not random but systematically targets vulnerable users—non-native English speakers, users with lower educational attainment, and non-US users receive lower accuracy, more misinformation, and higher refusal rates, with these effects compounding when multiple factors co-occur.

## Method

### Overall Architecture

Multi-dimensional user bios are constructed to independently control educational level (high/low), English proficiency (native/non-native), and nationality (US/Iran/China). Each question is prepended with a bio, and the responses of three LLMs on two datasets are observed and compared against a no-bio control baseline.

### Key Designs

1. **User Bio Construction**:

    - GPT-4-generated bios: first-person profiles for fictional individuals of varying educational and English proficiency levels generated from templates
    - Real-person bios: anonymized doctoral student profiles collected from university websites (US/Iran/China, male/female), preserving authentic writing style and grammatical features
    - Variable isolation: language style and interests are held constant when isolating the educational level dimension
    - Gender dimension: male and female versions are created for each nationality to detect gender bias

2. **Evaluation Dimensions**:

    - **Accuracy**: proportion of correct responses (SciQ factuality; TruthfulQA truthfulness)
    - **Refusal rate**: proportion of responses in which the model declines to answer ("I cannot answer...")
    - **Information withholding**: instances where the model answers correctly for some users but refuses to answer the same question for others
    - **Linguistic analysis**: manual detection of patronizing and dismissive language

3. **Statistical Methods**:

    - Each experiment is repeated four times; Chi-square tests are used to assess statistical significance relative to the control baseline
    - Significance markers: \* (p<0.1), \*\* (p<0.05), \*\*\* (p<0.01)

### Loss & Training

This is an evaluation study and involves no model training. The public APIs of all three models are used with default parameters; temperature is set to 1.0 for GPT-4 and Claude and 0.6 for Llama 3-8B. The system prompt is simply: "Answer only one of the answer choices. Do not stray from these choices."

## Key Experimental Results

### Main Results

**Educational Level Dimension** (TruthfulQA accuracy):
- All models show significant accuracy degradation for low-education users (p<0.05)

**English Proficiency Dimension** (TruthfulQA):
- All models show significant accuracy degradation for non-native speakers (p<0.05)
- The largest degradation occurs for users in the "non-native + low education" intersection

**Nationality Dimension** (high-education real-person bios):

| Model | Control | US M | US F | Iran M | Iran F | China M | China F |
|-------|---------|------|------|--------|--------|---------|---------|
| GPT-4 TruthfulQA | 81.00 | 80.69 | 80.39 | 79.23 | 79.36 | 81.36 | 80.69 |
| Claude TruthfulQA | 78.17 | 80.66† | 78.70 | 75.76* | 72.34*** | 82.19††† | 81.03†† |
| Llama 3 SciQ | 88.70 | 89.10 | 90.20 | 89.70 | 89.30 | 90.30 | 90.80 |

### Ablation Study

**Isolating Educational Level** (language style and interests held constant):
- Claude's SciQ accuracy for low-education Iranian users drops to 69.30% (control baseline: 95.60%), a dramatic decline
- Llama 3 shows significant SciQ degradation across all users (p<0.001)
- GPT-4 shows no significant differences, demonstrating the greatest stability

### Key Findings

- **Compounding effects**: The combination of "low education + Iranian user" produces far more severe degradation than either factor alone; Claude drops from 95.60% to 69.30%
- **Gender bias**: Claude exhibits significantly lower average accuracy for female users than for male users on TruthfulQA (p<0.005)
- **Refusal tendency**: LLMs are more inclined to refuse to answer vulnerable users rather than providing correct responses (information withholding)
- **Patronizing language**: Noticeably patronizing tones are detected in responses to low-education and non-native users
- **Model differences**: GPT-4 is the most stable; Claude exhibits the most severe bias; Llama 3 degrades on SciQ across all bios

## Highlights & Insights

- The study reveals a disturbing reality: LLMs provide the least reliable information service to users who need it most
- The experimental design is meticulous—combining LLM-generated and real-person bios while isolating the influence of individual dimensions
- The severity of compounding effects is demonstrated: when multiple vulnerability dimensions intersect, the impact far exceeds linear summation
- The findings carry significant practical implications: ChatGPT's Memory feature stores user information, directly mirroring the experimental setup of this paper

## Limitations & Future Work

- The experimental setup uses explicit bios to convey user characteristics; in practice, such characteristics may be implicitly transmitted through writing style
- Only three models are tested; evaluation of additional models (e.g., Gemini, Qwen) is lacking
- No mitigation strategies are proposed—the paper identifies the problem without offering solutions
- The multiple-choice format limits generalizability to open-ended generation settings
- Root causes are not analyzed—whether training data bias, RLHF sycophancy, or other mechanisms are responsible remains unexplored

## Related Work & Insights

- Perez et al. (2023) first identified the sandbagging phenomenon; the present paper substantially extends the research dimensions and depth
- Sharma et al. (2024) studied sycophantic behavior in which LLMs cater to users' political beliefs; this may share an underlying mechanism with the targeted underperformance documented here
- Hofmann et al. (2024) and Kantharuban et al. (2025) provide additional evidence that user characteristics influence model behavior
- The findings pose serious challenges for fairness alignment in LLMs

## Rating

⭐⭐⭐⭐ (4/5)

The research question is of considerable social significance, the experimental design is systematic and rigorous, and the findings are sobering. The discovery of compounding effects and the quantification of gender bias in Claude are particularly valuable contributions. Weaknesses include the absence of mechanistic analysis and mitigation strategies, and the paper leans toward important empirical findings rather than substantial technical contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text](../../ACL2026/llm_safety/who_gets_which_message_auditing_demographic_bias_in_llm-generated_targeted_text.md)
- [\[NeurIPS 2025\] TRAP: Targeted Redirecting of Agentic Preferences](../../NeurIPS2025/llm_safety/trap_targeted_redirecting_of_agentic_preferences.md)
- [\[AAAI 2026\] Democratizing LLM Efficiency: From Hyperscale Optimizations to Universal Deployability](democratizing_llm_efficiency_from_hyperscale_optimizations_to_universal_deployab.md)
- [\[AAAI 2026\] Principles2Plan: LLM-Guided System for Operationalising Ethical Principles into Plans](principles2plan_llm-guided_system_for_operationalising_ethical_principles_into_p.md)
- [\[AAAI 2026\] WaterMod: Modular Token-Rank Partitioning for Probability-Balanced LLM Watermarking](watermod_modular_token-rank_partitioning_for_probability-balanced_llm_watermarki.md)

</div>

<!-- RELATED:END -->
