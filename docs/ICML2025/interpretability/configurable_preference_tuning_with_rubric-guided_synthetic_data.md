---
title: >-
  [Paper Note] Configurable Preference Tuning with Rubric-Guided Synthetic Data
description: >-
  [ICML 2025][Interpretability][Configurable Preference Tuning] This paper proposes the Configurable Preference Tuning (CPT) framework, which trains LLMs using synthetic preference data generated from fine-grained rubrics. This enables the model to dynamically adjust its behavioral style at inference time simply by modifying the system prompt without retraining, improving accuracy from 0.52-0.68 to 0.76-0.83 across multiple base models.
tags:
  - "ICML 2025"
  - "Interpretability"
  - "Configurable Preference Tuning"
  - "DPO"
  - "Synthetic Data"
  - "Rubric-Guided"
  - "System Prompt Control"
date: 2026-05-08
content_hash: 013d3b2195ee2b76
---

# Configurable Preference Tuning with Rubric-Guided Synthetic Data

**Conference**: ICML 2025  
**arXiv**: [2506.11702](https://arxiv.org/abs/2506.11702)  
**Code**: [https://github.com/vicgalle/configurable-preference-tuning](https://github.com/vicgalle/configurable-preference-tuning)  
**Area**: Interpretability  
**Keywords**: Configurable Preference Tuning, DPO, Synthetic Data, Rubric-Guided, System Prompt Control

## TL;DR
This paper proposes the Configurable Preference Tuning (CPT) framework, which trains LLMs using synthetic preference data generated from fine-grained rubrics. This enables the model to dynamically adjust its behavioral style at inference time simply by modifying the system prompt without retraining, improving accuracy from 0.52-0.68 to 0.76-0.83 across multiple base models.

## Background & Motivation
Existing LLM alignment methods (such as RLHF/DPO) "freeze" a single, static set of preferences during training, leading to a "one-size-fits-all" behavioral pattern. However, human preferences are inherently multi-dimensional, dynamic, and context-dependent—expectations for LLM outputs vary significantly across different users, scenarios, and cultural backgrounds. Changing the behavior of an aligned model (e.g., adjusting writing style, safety level, or persona) typically requires expensive fine-tuning.

Existing personalized RLHF methods mainly fall into two categories: (1) learning a set of implicit features and combining them via weights (such as RFM), which suffers from poor interpretability; (2) conditioning the reward model by inferring implicit representations from user IDs or historical interactions (such as P-RLHF), which similarly lacks explicit controllability. The **Key Challenge** lies in: how to enable the model to precisely adjust its output behavior based on human-readable instructions without retraining?

The **Key Insight** of this paper is to utilize structured rubrics to define desired attributes and guide the generation of synthetic preference data through rubrics, enabling the student model to learn to switch behavioral patterns based on different system prompts. The **Core Idea**: **encode rubric-score combinations into system prompts, allowing the same pair of responses to swap their chosen/rejected roles under different system prompts, thereby teaching the model to respond in a "configurable" manner**.

## Method

### Overall Architecture
The core pipeline of CPT consists of four steps: (1) defining fine-grained rubrics → (2) generating responses targeted at different scores using a teacher model → (3) summarizing the rubric + score into a concise system prompt → (4) constructing symmetric preference pairs and training the student model with DPO. The entire process requires no new human annotation and relies completely on synthetic data.

### Key Designs
1. **Rubric Definition and Score-Conditioned Generation**:

    - **Function**: Defines a set of rubrics $\{\mathcal{R}_i\}$, where each rubric describes a specific attribute dimension of the response in detail (e.g., "unconventionality", "ornate Baroque style", "mystical symbolism") and provides detailed scoring criteria across 5 levels.
    - **Mechanism**: For each rubric $\mathcal{R}$ and user prompt $x$, an enhanced prompt $\phi(x, \mathcal{R}, \text{score})$ is used to guide the teacher model to generate responses conforming to a specific score level: $y \sim p(y|\phi(x, \mathcal{R}, \text{score}))$.
    - **Design Motivation**: Making the generation process controllable and interpretable via explicit scoring rubrics, rather than relying on implicit preference learning.

2. **System Prompt Synthesis**:

    - **Function**: Summarizes each rubric-score combination into a concise 2-3 sentence system prompt.
    - **Mechanism**: $s = \text{summarize}(\mathcal{R}, \text{score})$, completed automatically by the teacher model. For example, if a rubric requires a high score in "unconventional style", the corresponding system prompt might be: "Generate a text that is fragmented, illogical, and filled with unexpected connections..."
    - **Design Motivation**: System prompts serve as control interfaces at inference time, which must be concise and human-readable.

3. **Symmetric Preference Pairs Construction**:

    - **Function**: Constructs two complementary DPO training samples from the same pair of responses $(y_1, y_2)$.
    - **Mechanism**: Selecting a rubric $\mathcal{R}$, generating two responses with different target scores and their corresponding system prompts. The key innovation is **role swapping of the same response pair under different system prompts**:
        - Sample 1: $(s_1, x, y_1, y_2)$ — $y_1$ is preferred over $y_2$ under $s_1$.
        - Sample 2: $(s_2, x, y_2, y_1)$ — $y_2$ is preferred over $y_1$ under $s_2$.
    - **Design Motivation**: This symmetric construction forces the student model to truly learn to switch preferences based on the system prompt $s$, rather than simply memorizing which response is "better".

### Loss & Training
Standard DPO loss is used, but additionally conditioned on the system prompt $s$:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(s,x,y_w,y_l)\sim\mathcal{D}} \left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|s,x)}{\pi_{\text{ref}}(y_w|s,x)} - \beta\log\frac{\pi_\theta(y_l|s,x)}{\pi_{\text{ref}}(y_l|s,x)}\right)\right]$$

Training uses LoRA for parameter-efficient fine-tuning, training for only 1 epoch. The synthetic dataset size is 900 samples (from a combination of 4 rubrics × 3 score targets). The teacher models are DeepSeek-R1 and o3-mini, and Claude 3.5 Sonnet is used as the judge.

## Key Experimental Results

### Main Results

| Model | Config | Accuracy | Kendall's τ | Spearman's ρ |
|------|------|----------|-------------|--------------|
| Rocinante-12B | baseline | 0.55 | 0.62 | 0.76 |
| Rocinante-12B | CPT | 0.76 | 0.76 | 0.88 |
| Qwen3-4B | baseline | 0.63 | 0.78 | 0.90 |
| Qwen3-4B | CPT | 0.77 | 0.82 | 0.93 |
| Mistral-Nemo-12B | baseline | 0.60 | 0.62 | 0.74 |
| Mistral-Nemo-12B | CPT | 0.83 | 0.81 | 0.93 |
| Mistral-Small-24B | baseline | 0.52 | 0.73 | 0.85 |
| Mistral-Small-24B | CPT | 0.78 | 0.80 | 0.92 |
| Phi-4-14B | baseline | 0.68 | 0.79 | 0.92 |
| Phi-4-14B | CPT | 0.77 | 0.82 | 0.93 |

### Teacher Model Generation Quality Validation

| Target Score | Model | Judge Score (/100) |
|---------|------|----------------|
| No rubric | DS-R1 | 80.1 |
| No rubric | o3-mini | 71.0 |
| low score | DS-R1 | 14.1 |
| low score | o3-mini | 23.1 |
| extremely high | DS-R1 | 96.3 |
| extremely high | o3-mini | 97.9 |

### Ablation Study

| Config | Key Indicator | Description |
|------|---------|------|
| CPT + BoN sampling | Higher scores, fewer sampling steps | CPT provides a better initial distribution for BoN |
| baseline + BoN sampling | Requires more sampling steps to achieve same quality | Base model distribution is less concentrated than CPT |

### Key Findings
- CPT consistently and significantly improves score accuracy and rank correlation across all 5 base models.
- The largest improvement is observed on Mistral-Nemo-12B (Acc: 0.60→0.83, ρ: 0.74→0.93).
- The teacher model is indeed capable of generating text aligned with different score levels of the rubric (reaching 96-97 points for extremely high targets).
- CPT is orthogonally complementary to Best-of-N (BoN) sampling: the CPT model outperforms the baseline at any N.

## Highlights & Insights
- The **symmetric preference pairs** design is highly clever: a single pair of responses generates two training samples by swapping the system prompt, which simultaneously improves data efficiency and forces the model to learn to rely on the system prompt for judgment.
- The three-stage pipeline of **rubric→system prompt→preference data** fully automates the process without requiring human annotation.
- Using only 900 synthetic samples + 1 epoch of LoRA training significantly shifts model behavior, showing extremely high data efficiency.
- The framework is highly scalable: any new rubric can be defined to control new dimensions of model output.

## Limitations & Future Work
- Currently only verified on open-ended writing tasks, without involving more complex scenarios like structured output or code generation.
- The design of rubrics still requires manual effort; how to automatically generate or optimize rubrics remains an open question.
- Generation quality relies heavily on the teacher model's capability, and bias from the teacher model might propagate to the synthetic data.
- Currently only supports single-dimension control; multi-dimensional attribute composition control has not yet been explored.
- Evaluation relies on LLM judges, which may introduce evaluation bias.
- The scale of synthetic data is relatively small (900 samples), and its effectiveness in large-scale scenarios remains to be validated.

## Related Work & Insights
- **vs RFM (Barreto et al.)**: RFM learns implicit reward features end-to-end and combines them with weights, whereas CPT directly defines "features" with explicit rubrics, rendering it more transparent and controllable.
- **vs P-RLHF (Li et al.)**: P-RLHF conditions the model using user embeddings, whereas CPT uses natural language system prompts, providing stronger interpretability and generalization.
- **vs Standard DPO**: Standard DPO consolidates a single static preference, while CPT extends DPO to support conditional preferences.

## Rating
- Novelty: ⭐⭐⭐⭐ The symmetric preference pairs and rubric-guided synthetic data schemes are highly novel, though the core remains DPO + system prompt conditioning.
- Experimental Thoroughness: ⭐⭐⭐ Good consistency across 5 validated models, but the scenarios are single-faceted (writing style only), lacking dimensions like safety or persona.
- Writing Quality: ⭐⭐⭐⭐ Clear method description, rich examples, and well-argued motivation.
- Value: ⭐⭐⭐⭐ Provides a practical training-free / inference-time behavioral control solution, showing high real-world utility for personalized LLM deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Evian: Towards Explainable Visual Instruction-tuning Data Auditing](../../ACL2026/interpretability/evian_towards_explainable_visual_instruction-tuning_data_auditing.md)
- [\[NeurIPS 2025\] Rectifying Shortcut Behaviors in Preference-based Reward Learning](../../NeurIPS2025/interpretability/rectifying_shortcut_behaviors_in_preference-based_reward_learning.md)
- [\[CVPR 2025\] Geometry-Guided Camera Motion Understanding in VideoLLMs](../../CVPR2025/interpretability/geometry-guided_camera_motion_understanding_in_videollms.md)
- [\[ACL 2026\] Curing "Miracle Steps" in LLM Mathematical Reasoning with Rubric Rewards](../../ACL2026/interpretability/curing_miracle_steps_in_llm_mathematical_reasoning_with_rubric_rewards.md)
- [\[NeurIPS 2025\] Toward Real-world Text Image Forgery Localization: Structured and Interpretable Data Synthesis](../../NeurIPS2025/interpretability/toward_real-world_text_image_forgery_localization_structured_and_interpretable_d.md)

</div>

<!-- RELATED:END -->
