---
title: >-
  [Paper Note] PERSA: Reinforcement Learning for Professor-Style Personalized Feedback with LLMs
description: >-
  [ACL2026][LLM Alignment][Educational Feedback] PERSA utilizes "Professor Demonstrations + Professor Preference Rewards + PPO with High-Level LoRA Updates" to align general LLMs with specific instructor programming feedba…
tags:
  - "ACL2026"
  - "LLM Alignment"
  - "Educational Feedback"
  - "Personalized LLM"
  - "RLHF"
  - "LoRA"
  - "Style Alignment"
date: 2026-05-08
content_hash: 356ecb1347333461
---

# PERSA: Reinforcement Learning for Professor-Style Personalized Feedback with LLMs

**Conference**: ACL2026  
**arXiv**: [2605.01123](https://arxiv.org/abs/2605.01123)  
**Code**: Not disclosed  
**Area**: Educational Feedback / LLM Personalization / Alignment RLHF  
**Keywords**: Educational Feedback, Personalized LLM, RLHF, LoRA, Style Alignment

## TL;DR
PERSA utilizes "Professor Demonstrations + Professor Preference Rewards + PPO with High-Level LoRA Updates" to align general LLMs with specific instructor programming feedback styles. It significantly improves style consistency on APPS, PyFiXV, and CodeReviewQA while maintaining near 100% diagnostic correctness.

## Background & Motivation
**Background**: LLMs are capable of generating feedback for programming problems, code reviews, and learning platforms. Mainstream approaches include direct prompting, supervised fine-tuning (SFT), general RLHF, and preference optimization methods like DPO, ORPO, or KTO. In educational settings, feedback quality depends not only on "correctly identifying bugs" but also on tone, structure, encouragement level, and the provision of actionable suggestions.

**Limitations of Prior Work**: General LLMs typically provide general directions but struggle to organize feedback in the specific voice of a real instructor. For example, regarding an "array read before index search" error, a general model might simply state "Check input handling," whereas a professor's feedback often acknowledges progress, identifies the root cause, and reminds the student of edge cases. Full-parameter fine-tuning risks degrading underlying capabilities and incurs high computational costs.

**Key Challenge**: Personalized instruction requires modifying "how" the model speaks without compromising "what" the model knows. Style is a high-level discourse attribute, while correctness relies on low-level code understanding and reasoning. Updating all parameters simultaneously causes style transfer and knowledge retention to interfere with each other.

**Goal**: The authors address three sub-problems: first, learning the "professor's voice" from demonstrations and preferences; second, preserving diagnostic correctness during RLHF; and third, ensuring the personalization is lightweight enough for 2B-3B scale open models.

**Key Insight**: Observations from internal transformer analysis suggest that high-level attributes like style, tone, and discourse organization are concentrated in the upper layers, particularly within FFNs and high-level projection modules. Consequently, it is unnecessary to update the full model; attaching LoRA to high layers and injecting preference signals through RLHF is sufficient.

**Core Idea**: Use layer-selective LoRA to restrict RLHF updates to the model's upper layers, allowing PPO to primarily adjust the instructional expression style while freezing the base model's code diagnostic capabilities.

## Method
PERSA is an RLHF pipeline tailored for "professor-style programming feedback." Rather than reinventing RLHF, it constrains the three RLHF stages to a style-related parameter subspace: SFT using professor demonstrations, training a reward model on paired preferences, and finally optimizing the policy model via PPO with KL constraints. The key distinction is that the trainable parameters are LoRA adapters in high-level transformer blocks rather than the full model.

### Overall Architecture
The input consists of a programming problem, student-submitted code, and a prompt potentially containing error types or context; the output is natural language feedback for the student. Training data includes two types: professor-written feedback demonstrations denoted as $(x, y^*)$, and paired preferences of candidate feedback for the same prompt denoted as $(x, y_w, y_l)$, where $y_w$ better aligns with the professor's style or correctness.

The pipeline comprises four steps. First, LoRA is loaded onto Llama-3.2-3B-Instruct or Gemma-2-2B-IT, enabling only the attention and FFN projections of the top several transformer blocks. Second, SFT is performed on professor demonstrations to learn the basic "Diagnosis $\rightarrow$ Repair Suggestion $\rightarrow$ Verification Reminder" format. Third, a reward model $r_{\phi}(x, y)$ is trained to assign higher scores to responses preferred by the professor. Fourth, starting from the SFT model, PPO maximizes the reward while using KL divergence to prevent the model from deviating from the SFT reference policy.

During inference, PERSA receives student code and generates feedback like a standard instruction model, requiring no additional retrieval or external executors. Since changes mainly occupy the high-level LoRA adapters, different adapters can be maintained for different instructors while sharing a single base model.

### Key Designs
1. **Layer-Selective LoRA Style Adaptation**:

    - **Function**: Trains low-rank adapters only on attention/FFN projections in high-level transformer blocks to encode the professor's style within a small parameter set.
    - **Mechanism**: LoRA increments are applied to selected weight matrices while base weights remain frozen. A typical setting is "top-4 LoRA," resulting in approximately 10 million trainable parameters rather than the full 2–3 billion.
    - **Design Motivation**: Style, tone, and discourse format are high-level expressive controls rather than low-level syntax or code knowledge. Updating only high layers reduces catastrophic drift and facilitates maintaining unique modules for various instructors.

2. **Professor Preference Reward Model**:

    - **Function**: Converts "more professor-like, pedagogically valuable, and correct" preferences into optimizable scalar rewards.
    - **Mechanism**: The reward model learns Bradley-Terry preference probabilities, maximizing the margin $r_{\phi}(x, y_w) - r_{\phi}(x, y_l)$. The loss is defined as $-\log \sigma(r_{\phi}(x, y_w) - r_{\phi}(x, y_l))$.
    - **Design Motivation**: SFT only mimics average demonstrations and may capture format without grasping subtle preferences regarding wording strength, encouragement, or specific edge-case reminders. The reward model provides a more direct signal for style selection during PPO.

3. **KL-Constrained PPO Policy Optimization**:

    - **Function**: Further optimizes feedback generation toward the reward model's preferences while keeping the model within the reliable region established during SFT.
    - **Mechanism**: The objective is to maximize $r_{\phi}(x, y) - \beta \text{KL}(\pi_{\theta}(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x))$, where $\pi_{\text{ref}}$ is the frozen SFT policy; PPO uses a clipped ratio for stable token-level updates.
    - **Design Motivation**: Educational feedback cannot merely pursue "sounding like a professor" at the cost of diagnostic errors. The KL term anchors the policy near the SFT model, which already possesses basic pedagogical structure and correctness.

### Loss & Training
The SFT stage uses standard autoregressive negative log-likelihood, updating only $\theta_{\text{LoRA}}$. The reward modeling stage uses paired logistic loss: $L_{RM} = -\mathbb{E}[\log \sigma(r_{\phi}(x, y_w) - r_{\phi}(x, y_l))]$. The PPO stage applies a clipped objective and incorporates the KL control term as a trajectory reward or penalty. The process is identical for both lightweight open models, comparing Base, SFT, InstructGPT-style RLHF, DPO, ORPO, KTO, and PERSA.

## Key Experimental Results

### Main Results
Evaluation was conducted on three code feedback datasets: 200 instances of APPS-style professor feedback, 240 PyFiXV Codeforces Python syntax error feedbacks, and 900 multi-language code review instances from CodeReviewQA. Metrics include Style Alignment (SAC), Politeness Proximity (APC), BLEU-4, Diagnostic Correctness (CA), and Preference Win Rate (PWR) relative to the Base model.

| Dataset / Backbone | Method | SAC | BLEU-4 | CA | PWR |
|--------|------|------|--------|------|------|
| APPS / Llama-3 | Base | 34.8 | 6.4 | 98.2 | - |
| APPS / Llama-3 | SFT | 82.0 | 80.0 | 100.0 | 86.2 |
| APPS / Llama-3 | ORPO | 95.6 | 95.0 | 100.0 | 90.2 |
| APPS / Llama-3 | PERSA | 96.2 | 95.8 | 100.0 | 90.1 |
| APPS / Gemma-2 | Base | 20.0 | 2.0 | 98.0 | - |
| APPS / Gemma-2 | PERSA | 99.0 | 98.0 | 100.0 | 98.0 |
| PyFiXV / Llama-3 | Strongest Baseline (ORPO) | 93.5 | 93.2 | 99.8 | 88.6 |
| PyFiXV / Llama-3 | PERSA | 94.5 | 94.0 | 99.9 | 89.0 |
| CodeReviewQA / Gemma-2 | KTO | 87.0 | 78.0 | 100.0 | 96.0 |
| CodeReviewQA / Gemma-2 | PERSA | 98.0 | 98.0 | 100.0 | 98.2 |

| Human Evaluation | Samples / Dimension | PERSA Score | Vanilla LLM | Tie / Notes |
|------|------|------|------|------|
| Student Survey | 20 people, 5 examples: Clarity | 4.34 / 5 | - | SD 1.05 |
| Student Survey | Helpfulness | 4.37 / 5 | - | SD 1.04 |
| Student Survey | Trustworthiness | 4.31 / 5 | - | SD 0.97 |
| Student Survey | Instructor Authenticity | 3.87 / 5 | - | SD 1.34 |
| Teacher Blind Eval | Overall Preference | 83.6% | 1.8% | 14.6% |
| Teacher Blind Eval | Helpfulness / Actionability | 85.5% | 1.8% | 12.7% |
| Teacher Blind Eval | Instructor Realism | 74.5% | 1.8% | 23.7% |
| Teacher Blind Eval | Technical Correctness | 61.8% | 5.5% | 32.7% |

### Ablation Study

| Configuration | SAC | APC | BLEU-4 | CA | PWR | Notes |
|------|------|------|--------|------|------|------|
| Base | 14.0 | 90.0 | 1.5 | 98.0 | - | Almost no professor style |
| SFT only | 82.0 | 91.6 | 64.7 | 100.0 | 86.0 | Learned basic feedback structure |
| PPO only | 60.0 | 91.2 | 40.0 | 98.0 | 84.0 | Unstable style without SFT anchor |
| SFT+PPO full-param | 92.0 | 92.0 | 92.0 | 100.0 | 88.0 | Aligned but high cost |
| SFT+PPO all-layer LoRA | 96.0 | 92.0 | 94.7 | 100.0 | 88.6 | Better with drift restriction |
| SFT+PPO top-2 LoRA | 94.0 | 92.0 | 90.0 | 100.0 | 90.0 | Upper layers capture most style |
| SFT+PPO top-4 LoRA | 96.2 | 92.1 | 95.8 | 100.0 | 90.1 | Best config (PERSA) |

### Key Findings
- The Base model's CA is already near 96%-98%, suggesting it can perform basic code judgment; however, low SAC and BLEU-4 prove that "correctness" and "instructor persona" are separable dimensions.
- SFT provides the largest performance jump, specifically increasing SAC for APPS / Llama-3 from 34.8 to 82.0; however, fine-grained style gaps remain that require preference optimization.
- PPO cannot be used in isolation: without SFT initialization, SAC is only 60.0, indicating reward optimization needs a starting point capable of basic instruction.
- Top-4 LoRA achieves the highest SAC and BLEU-4 while maintaining 100.0 CA, supporting the hypothesis that "high layers carry style while low layers retain capability."
- In human evaluations, the high tie rate for technical correctness suggests vanilla LLMs can often judge right/wrong; PERSA's advantage lies in clarity, actionability, tone, and authenticity.

## Highlights & Insights
- Treating the "instructor's voice" as an optimizable alignment objective rather than a prompt description is practical; real feedback has consistent structure and tone that prompts struggle to maintain over time.
- Layer-selective LoRA is the most valuable engineering decision in this work. It transforms personalization from "retraining a model" into "attaching different instructor adapters to the same base," making it ideal for multi-user deployment in courses or platforms.
- Ablation results clarify the division of labor: SFT provides the instructional skeleton, while PPO refines the preference between candidate expressions. This can generalize to medical, customer service, or legal domains where expert tone is required without sacrificing factuality.
- Qualitative examples show a key educational metric: good feedback does not dump the full answer but identifies causes, provides directions, and reminds of validation. The low student score for "contains copyable solution" (2.55) suggests the model avoids over-spoiling.

## Limitations & Future Work
- Data scale is relatively small; APPS professor feedback contains only 200 entries, and tasks are limited to code. Verification across large-scale courses, multi-instructor scenarios, and non-programming subjects is needed.
- Style metrics (SAC, APC, BLEU-4) may still favor surface similarity. Deep pedagogical strategies (when to ask questions vs. hints) are difficult to capture with current automated metrics.
- The reward model is trained on professor preferences; the paper lacks discussion on labeling costs, preference conflicts between instructors, and how individual student differences might influence rewards.
- Evidence for retaining correctness comes from existing benchmarks; deployment in real IDEs or judge systems should integrate execution tests or static analysis to prevent "styled but wrong" outputs.
- Future work could combine instructor adapters with course knowledge bases and student profiles for dual personalization (instructor style + student learning stage).

## Related Work & Insights
- **vs InstructGPT-style RLHF**: General RLHF optimizes for broad helpfulness/harmlessness; PERSA optimizes for a specific teacher's preference using top-layer LoRA, with narrower goals and lower costs.
- **vs SFT Fine-tuning**: SFT mimics format but does not explicitly compare instructor-like responses. PERSA captures details in tone and actionability by learning preference boundaries.
- **vs DPO / ORPO / KTO**: These offline methods are lighter as they don't require on-policy rollouts. PERSA retains PPO but controls drift via top-layer LoRA, showing particularly strong performance on Gemma-2.
- **vs Automated Code Feedback**: Traditional systems focus on testing and error localization; PERSA focuses on expression and persona. The two can be complementary by providing verifiable diagnostics and student-friendly communication respectively.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines RLHF, LoRA, and educational personalization naturally; the key innovation lies in "high-level style adaptation" rather than a brand-new algorithm.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Automated metrics, ablations, and human evaluations are complete, though data scale remains limited.
- Writing Quality: ⭐⭐⭐⭐☆ Clear methodology and comprehensive baselines; definitions for some metrics could be more rigorous.
- Value: ⭐⭐⭐⭐⭐ Highly insightful for educational LLMs and deployable assistants, serving as a prototype for multi-instructor adapter systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Strategyproof Reinforcement Learning from Human Feedback](../../NeurIPS2025/llm_alignment/strategyproof_reinforcement_learning_from_human_feedback.md)
- [\[ACL 2026\] WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback](wildfeedback_aligning_llms_with_in-situ_user_interactions_and_feedback.md)
- [\[ACL 2026\] P-Check: Advancing Personalized Reward Model via Learning to Generate Dynamic Checklist](p-check_advancing_personalized_reward_model_via_learning_to_generate_dynamic_che.md)
- [\[ACL 2026\] Too Correct to Learn: Reinforcement Learning on Saturated Reasoning Data](too_correct_to_learn_reinforcement_learning_on_saturated_reasoning_data.md)
- [\[ICLR 2026\] Swap-guided Preference Learning for Personalized RLHF (SPL)](../../ICLR2026/llm_alignment/swap-guided_preference_learning_for_personalized_reinforcement_learning_from_hum.md)

</div>

<!-- RELATED:END -->
