---
title: >-
  [Paper Note] Conformity in Large Language Models
description: >-
  [ACL 2025 Main][LLM (Other)][conformity effect] This paper adapts the classic Asch conformity experiment paradigm from psychology to LLMs to systematically study their conformity behaviors. It reveals that all evaluated models alter their answers under the influence of majority opinions, showing a higher susceptibility to conformity when uncertainty is greater. Furthermore, the paper proposes two intervention methods, Devil's Advocate and Question Distillation…
tags:
  - "ACL 2025 Main"
  - "LLM (Other)"
  - "conformity effect"
  - "LLM bias"
  - "Asch experiment"
  - "uncertainty"
  - "intervention strategies"
date: 2026-05-08
content_hash: be7fc59ca077cbcb
---

# Conformity in Large Language Models

**Conference**: ACL 2025 Main  
**arXiv**: [2410.12428](https://arxiv.org/abs/2410.12428)  
**Code**: None  
**Area**: LLM Behavior Analysis / AI Safety  
**Keywords**: conformity effect, LLM bias, Asch experiment, uncertainty, intervention strategies

## TL;DR

This paper adapts the classic Asch conformity experiment paradigm from psychology to LLMs to systematically study their conformity behaviors. It reveals that all evaluated models alter their answers under the influence of majority opinions, showing a higher susceptibility to conformity when uncertainty is greater. Furthermore, the paper proposes two intervention methods, Devil's Advocate and Question Distillation, which effectively mitigate this conformity effect.

## Background & Motivation

LLMs are increasingly utilized as conversational partners to assist in information retrieval and decision-making. In such scenarios, users might introduce "majority opinions" into the dialogue (e.g., "most people think the answer is X"). **If an LLM alters its originally correct answer due to the majority opinion, its reliability will be severely compromised.**

This phenomenon is analogous to the classic **Asch conformity experiments** (1951) in psychology, where human subjects tend to conform to clearly incorrect answers when faced with a unanimous majority of confederates. A critical question arises: Do LLMs exhibit a similar conformity effect?

Limitations of Prior Work:

1. **Sycophancy research** has shown that RLHF-aligned models tend to sycophantically agree with users. However, this is primarily a one-on-one scenario issue, which differs from the conformity effect in multi-party dialogues.
2. Prior work remains largely **at the phenomenon identification stage**, lacking in-depth analysis of driving factors (such as the impact of model uncertainty and training paradigms).
3. **Lack of mitigation strategies**—addressing the identification of problems without proposing actionable solutions.

Key Insight: **By adopting classic social psychology experimental paradigms, this work not only quantifies the degree of LLM conformity but also unravels the driving factors and provides effective intervention measures.**

## Method

### Overall Architecture

The experiment consists of three steps:
1. **Vanilla Round**: Prompting the LLM directly to obtain its original answer.
2. **Multi-party Round**: Constructing a multi-party dialogue scenario where multiple "participants" (controlled by the experimenter) provide answers that conflict with the LLM's original answer, and then asking the LLM to answer again.
3. **Analysis**: Comparing the changes between the two rounds of answers to quantify the Conformity Rate.

### Key Designs

1. **Asch Paradigm Adapted for LLMs**:

    - **Function**: Constructs prompts simulating multi-party conversations, casting the LLM as one of the experiment participants.
    - **Mechanism**: The prompt template is "You are one of the participants in an experiment. There are N other participants who have given their answers...". The answers of the "other participants" are configured by the experimenter to be identical incorrect or conflicting answers, thereby exerting majority pressure. The conformity rate is defined as:
    $\text{Conformity Rate} = \frac{\text{改变答案的次数}}{\text{总测试次数}}$
    - **Design Motivation**: Directly mimicking classic social psychology experiments allows for a direct comparison with human data. By controlling variables such as the number of majority members and the naturalness of expressions, the study systematically investigates the influencing factors.

2. **Analysis of the Relationship Between Uncertainty and Conformity**:

    - **Function**: Establishes for the first time a quantitative relationship between LLM prediction uncertainty and conformity tendency.
    - **Mechanism**: Token-level uncertainty is computed using the model's output logits (such as the entropy of prediction probabilities or the top-1 probability). The correlation between uncertainty and conformity is then analyzed. Questions are grouped based on model uncertainty, and the conformity rates across groups are compared.
    - **Design Motivation**: Psychological research shows that humans are more likely to conform when uncertain. Verifying whether LLMs exhibit a similar pattern is crucial for understanding the underlying mechanism of conformity. If uncertainty is a driving factor, conformity could be mitigated by increasing model confidence.

3. **Exploration of Influencing Factors**:

    - **Function**: Systematically analyzes the effects of training paradigms and input features on the degree of conformity.
    - **Key Findings**:
        - **Instruction-tuned models are less prone to conformity than base models**: Instruction tuning empowers models with a stronger capability to "stand their ground."
        - **More natural expressions of majority opinions lead to higher conformity**: Compared to mechanically listing "Participant 1: A, Participant 2: A...", presenting opinions in natural language (e.g., "I believe the answer is A because...") is more effective in inducing conformity.
        - **Increasing the number of majority members increases the conformity rate**, although a saturation phenomenon is observed.
    - **Design Motivation**: Identifying controllable factors to guide the design of downstream intervention strategies.

4. **Two Intervention Strategies**:

    - **Devil's Advocate**:
        - **Function**: Introduces a dissenting minority ("devil's advocate") into the majority opinions who supports the LLM's original answer.
        - **Mechanism**: Sets one "participant's" answer to be different from the majority, breaking the unanimous pressure of the majority—which is exactly the key mitigating factor identified in Asch's original experiments.
        - **Result**: Significantly reduces the conformity rate.

    - **Question Distillation**:
        - **Function**: Instructs the LLM to first extract the core question from the multi-party dialogue, and then re-answer it in a "clean" environment.
        - **Mechanism**: Prompts the model to perform two steps: first, "What is the core question being discussed?", and second, "Based only on your knowledge, what is the answer?" This eliminates the influence of conformity by separating the question from the social pressure context.
        - **Result**: Effectively reduces the conformity rate with low implementation costs.

## Key Experimental Results

### Main Results

| Model | Conformity Rate (%) | Original Correct to Incorrect Transition | Original Incorrect Transition | Description |
|------|-----------|------------------|-----------------|------|
| Llama-3-8B | High | High correct-to-incorrect rate | Also conforms | Base LLM suffers from severe conformity |
| Llama-3-8B-Instruct | Moderate | Improved | - | Instruction tuning significantly reduces conformity |
| Qwen2-7B | High | - | - | Conformity also exists |
| Qwen2-7B-Instruct | Moderate | - | - | Instruction tuning helps |
| Gemma-2-9B | Moderate-to-High | - | - | All models exhibit conformity |
| Mistral-v0.3-7B | High | - | - | - |

All tested models (across various sizes and series) exhibit varying degrees of conformity behavior.

### Ablation Study

| Input Variable | Change in Conformity Rate | Description |
|---------|-----------|------|
| Base vs Instruct | Instruct decreases by ~15-30% | Instruction tuning significantly enhances "resistance to pressure" |
| Number of Majority Members 3→5→7 | Conformity rate increases | Mimics human experimental results but shows saturation |
| Mechanical vs Natural Tone | Natural tone leads to higher conformity | Natural expressions are more persuasive |
| Devil's Advocate Intervention | Conformity rate drops significantly | Breaking unanimity by a minority is critical |
| Question Distillation Intervention | Conformity rate drops significantly | Separating the question from social pressure is effective |

### Key Findings

- **Strong positive correlation between uncertainty and conformity**: When the model is uncertain about its original answer (measured via logits), the conformity rate is significantly higher. This is the most crucial finding of the paper, establishing this link in LLMs for the first time.
- **Conformity is a universal phenomenon**: Regardless of the domain (science, literature, history, etc.) or task type (multiple-choice, yes/no questions), all evaluated models exhibit conformity.
- **Conformity occurs regardless of correctness**: Models not only conform to incorrect answers when their original answers are correct, but they also conform to alternative incorrect answers when their original answers are incorrect.
- **Both interventions are effective but operate via different mechanisms**: Devil's Advocate mitigates conformity by breaking majority unanimity, while Question Distillation works by eliminating the social context.

## Highlights & Insights

- **Cross-disciplinary methodological innovation**: Systematically adapts the classic psychological experimental paradigm to LLM evaluation, providing a clear and reproducible methodology.
- The discovery of the **uncertainty-conformity relationship** has practical implications for LLM deployment: the weight of external opinions should be reduced when the LLM is uncertain.
- **Exceedingly low-cost intervention strategies**: Both methods can be applied simply by adjusting the prompts, without requiring model retraining.
- **Comprehensive experimental coverage**: Standardized and robust experimental design covering 4 architectures * base/instruct versions * multiple datasets * multiple variables.

## Limitations & Future Work

- The tested models are relatively small (7B-9B). Larger models like GPT-4 could not be evaluated for logit-level uncertainty due to API limitations.
- Only closed-ended tasks (such as multiple-choice and yes/no questions) were tested; conformity behavior in open-ended generation tasks remains unexplored.
- The "other participants" in the experimental setup are simulated; conformity dynamics in real-world, multi-turn dialogues could be more complex.
- Question Distillation requires an additional round of inference, increasing latency.
- The potential mechanisms by which conformity is reinforced during alignment training (such as RLHF/DPO) were not analyzed.
- Robustness of intervention strategies—do they remain effective when facing more natural and stealthier social pressures?

## Related Work & Insights

- **vs Sycophancy (Perez et al., 2023; Sharma et al., 2024)**: Sycophancy refers to one-on-one pandering to the user, whereas conformity represents yielding to a majority opinion. Though related, their underlying mechanisms differ. This paper investigates multi-party dialogue settings, which are closer to real-world collaboration.
- **vs Debate/Multi-agent**: In LLM debates and multi-agent collaboration, the conformity effect may lead to a "false consensus"—where multiple agents influence each other and converge on the same incorrect answer.
- **vs Asch (1951) Original Experiment**: Although the conformity rate of LLMs is generally lower than that of human subjects (around 37% for humans), it can be extremely high on uncertain questions. Furthermore, LLMs cannot explain the reasons behind their conformity.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Establishes the uncertainty-conformity relationship and systematically studies the LLM conformity effect for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ The experimental design is rigorous and comprehensive, though larger model scales could be explored.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is well-written and logical, successfully combining the psychological background with NLP experiments.
- Value: ⭐⭐⭐⭐ Offers crucial insights into LLM safety and the design of multi-agent systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)
- [\[ACL 2025\] Large Language Models in Bioinformatics: A Survey](large_language_models_in_bioinformatics_a_survey.md)
- [\[ACL 2025\] ProgCo: Program Helps Self-Correction of Large Language Models](progco_program_helps_self-correction_of_large_language_models.md)
- [\[ACL 2025\] Towards Harmonized Uncertainty Estimation for Large Language Models](towards_harmonized_uncertainty_estimation_for_large_language_models.md)
- [\[ACL 2025\] Argument Mining in the Age of Large Language Models](argument_mining_in_the_age_of_large_language_models.md)

</div>

<!-- RELATED:END -->
