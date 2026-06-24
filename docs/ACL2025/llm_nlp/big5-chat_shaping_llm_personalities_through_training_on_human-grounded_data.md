---
title: >-
  [Paper Note] Big5-Chat: Shaping LLM Personalities Through Training on Human-Grounded Data
description: >-
  [ACL 2025][LLM (Other)][Personality traits] This paper proposes the Big5-Chat dataset (100k dialogues), embedding real human Big Five personality traits into LLMs through SFT and DPO training methods. This approach significantly outperforms prompting-based methods. Additionally, the paper reveals that personality configurations of high conscientiousness/agreeableness and low extraversion/neuroticism can enhance the model's reasoning capabilities.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Personality traits"
  - "Big Five personality"
  - "SFT"
  - "DPO"
  - "psycholinguistics"
date: 2026-05-08
content_hash: 35533cdb6f6df129
---

# Big5-Chat: Shaping LLM Personalities Through Training on Human-Grounded Data

**Conference**: ACL 2025  
**arXiv**: [2410.16491](https://arxiv.org/abs/2410.16491)  
**Code**: [github](https://github.com/wenkai-li/Big5-Chat.git)  
**Area**: LLM/NLP  
**Keywords**: Personality traits, Big Five personality, SFT, DPO, psycholinguistics

## TL;DR

This paper proposes the Big5-Chat dataset (100k dialogues), embedding real human Big Five personality traits into LLMs through SFT and DPO training methods. This approach significantly outperforms prompting-based methods. Additionally, the paper reveals that personality configurations of high conscientiousness/agreeableness and low extraversion/neuroticism can enhance the model's reasoning capabilities.

## Background & Motivation

Embedding realistic personality traits into LLMs is crucial for enhancing the authenticity of conversational agents, educational tools, and mental health platforms. However, existing methods mostly rely on **prompting** to induce personalities (e.g., "You are the life of the party"), which poses three core limitations:

**Lack of psycholinguistic depth**: Prompt-induced personalities merely reflect superficial features and fail to capture the subtle ways humans express personality throughout language.

**Evaluation validity issues**: Using psychological questionnaire descriptions to prompt personalities and then evaluating them with the same questionnaire leads to severe circular reasoning.

**Biased reasoning patterns**: Persona-based personality prompts can excessively constrain the behaviors of LLMs, leading to a degradation in reasoning capabilities.

The fundamental cause is the lack of large-scale, human-grounded personality-annotated datasets, which limits the exploration of training-based approaches.

## Method

### Overall Architecture

The PsychSteer method comprises two phases: (1) training expert generator models, and (2) generating the Big5-Chat dataset using the DExperts framework, followed by SFT/DPO alignment.

### Key Designs

1. **DExperts Framework**: Steers the base model outputs using the expert generators during decoding. At each time step t, the logits of the base model are combined with the expert generator's logits through weighted addition: $z_t^{combined} = z_t^{base} + \gamma z_t^{expert}$, where $\gamma$ controls the intensity of the steered personality trait.

2. **Expert Generator Models**: Finetunes LLaMA-3-8B-Instruct on the PsychGenerator dataset (846k Facebook posts annotated with Big Five personality scores) to train 5 independent expert generators corresponding to openness, conscientiousness, extraversion, agreeableness, and neuroticism. The continuous floating-point labels are converted into binary "high/low" labels.

3. **Big5-Chat Dataset Construction**: Randomly samples 10k social scenarios from the SODA dataset and leverages the PsychSteer framework to generate paired dialogues (high/low personality levels) for each scenario. This yields a total of 100k single-turn dialogues (20k dialogues per trait, evenly split between high and low levels).

4. **SFT and DPO Training**: Conducts efficient parameter finetuning using LoRA. For DPO training, the dispreferred responses are sourced from the opposite level of the same trait (e.g., if the target is high openness, the chosen response is a high openness reply, and the rejected response is a low openness reply).

### Loss & Training

- SFT utilizes standard cross-entropy loss.
- DPO utilizes preference optimization loss to learn personality preferences using pairwise contrastive samples.
- Uses LoRA adapters for parameter-efficient finetuning.
- Dataset quality verification: Trains a RoBERTa-Large classifier (5 regression heads, MSE loss), achieving 93.8% accuracy on the test set.

## Key Experimental Results

### Main Results

**Personality Assessment (BFI Test, LLaMA-3-70B-Instruct)**:

| Method | High Trait Avg. ↑ | Low Trait Avg. ↓ | Characteristics |
|------|------------|------------|------|
| Direct | 3.8 | 3.8 | No personality induction |
| Prompt-Inst | 5.0 | 1.6 | Instruction prompting |
| SFT | 5.0 | 1.2 | Training-based |
| DPO | 5.0 | 1.4 | Training-based |

**Reasoning Evaluation (LLaMA-3-70B-Instruct, SFT Average)**:

| Reasoning Domain | Direct | SFT Avg. | Best Trait |
|---------|--------|---------|---------|
| Social Reasoning | 46.6 | 50.0 | High Openness 50.3 |
| Mathematical Reasoning | 59.8 | 63.6 | High Agreeableness 65.0 |
| Hallucination Detection | 58.6 | 54.4 | High Conscientiousness 55.6 |
| Commonsense Reasoning | 53.7 | 79.4 | High Openness 79.5 |
| General Reasoning | 54.0 | 53.2 | High Conscientiousness 53.7 |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Dataset quality - Expert Generator | 80.4% accuracy | Classifier evaluation of personality trait accuracy |
| Dataset quality - GPT-4o-mini baseline | 59.2% accuracy | Post-Completion baseline, 21.2% lower |
| SFT vs DPO (Personality Induction) | No significant difference | Both outperform prompting methods |
| Prompt-Demo (Demonstration Prompting) | Lower than SFT/DPO | In-context learning is insufficient for deep personality embedding |

### Key Findings

1. **Training-based methods significantly outperform prompting methods**: Both SFT and DPO substantially exceed instruction prompting and demonstration prompting in BFI and IPIP-NEO personality assessments.
2. **Correlation between personality and reasoning**: Models with high conscientiousness and high agreeableness perform the best in reasoning tasks; low extraversion and low neuroticism also benefit reasoning. This aligns with findings in psychology regarding the relationship between the Big Five personality traits and human cognitive abilities.
3. **SFT yields more realistic trait correlations**: The inter-trait correlation patterns of the trained models are closer to real human data distributions.
4. **Significant boost in commonsense reasoning**: SFT improves performance in commonsense reasoning from 53.7% to approximately 79%, demonstrating a substantial gain.

## Highlights & Insights

- **Personality-Cognitive Bridge**: This work is the first to systematically demonstrate that after embedding personality traits via training, the changes in LLM reasoning capabilities mirror human patterns in psychological studies, indicating that training-based approaches indeed capture deep psycholinguistic features.
- **Innovative Dataset Construction**: The method cleverly combines domain-specific personality-annotated data (Facebook posts) with general social scenarios (SODA) via the DExperts framework, addressing the bottleneck of limited personality data resources.
- **Dual Evaluation Framework**: It evaluates both the effectiveness of personality induction (via BFI and IPIP-NEO) and the impact on reasoning capabilities, offering a more comprehensive perspective.

## Limitations & Future Work

1. **Monolingual Limitation**: Experiments are solely conducted in English; the expression of personality may vary across different cultures and languages.
2. **Binary Labeling**: Simulates personality continuous scores by simplifying them into binary high/low statuses, thereby losing the fine-grained nature of a continuous personality spectrum.
3. **Base Model Constraints**: Validation is limited to the LLaMA-3 series; performance on other architectures (such as Mistral or Qwen) remains untested.
4. **Limited Dialogue Depth**: Big5-Chat comprises single-turn dialogues; personality consistency in multi-turn dialogues remains unexplored.
5. **Unclear Causal Mechanisms Behind Reasoning Changes**: While correlations between personality traits and reasoning capability are identified, the work does not deeply explain why specific personality configurations enhance reasoning.

## Related Work & Insights

- **PsychGenerator (Vu et al., 2024)**: A dataset of 846k Facebook posts annotated with personalities, serving as the core data source of this work.
- **SODA (Kim et al., 2023)**: A diverse social dialogue dataset generated by GPT-3.5.
- **DExperts (Liu et al., 2021)**: A controllable text generation framework steered at decoding time.
- Insights: Training-based methods may be more promising than prompting for embedding complex psychological characteristics (such as emotion or cognitive styles).

## Rating

- Novelty: ⭐⭐⭐⭐ It is the first to systematically embed personality through training and assess its impact on reasoning, although the DExperts framework itself is not new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers two personality tests, five reasoning domains, and two model scales, providing comprehensive ablation and analyses.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clear motivation and rich data presentation.
- Value: ⭐⭐⭐⭐ Highly valuable for interdisciplinary research in AI anthropomorphism, role-playing, and psychology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Understanding Silent Data Corruption in LLM Training](understanding_silent_data_corruption_in_llm_training.md)
- [\[ACL 2025\] Towards Geo-Culturally Grounded LLM Generations](geocultural_grounded_llm.md)
- [\[ACL 2025\] Game Development as Human-LLM Interaction](game_development_as_human-llm_interaction.md)
- [\[ACL 2025\] MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion](mathfusion_instruction_fusion.md)
- [\[ACL 2025\] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)

</div>

<!-- RELATED:END -->
