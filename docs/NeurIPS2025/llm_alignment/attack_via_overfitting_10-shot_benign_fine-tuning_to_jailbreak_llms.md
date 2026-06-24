---
title: >-
  [Paper Note] Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs
description: >-
  [NeurIPS 2025][LLM Alignment][jailbreak attack] This paper proposes a two-stage fine-tuning attack: Stage 1 fine-tunes an LLM on 10 benign questions paired with identical refusal answers, driving the model to overfit into a sharp loss landscape; Stage 2 fine-tunes the same 10 questions with normal answers, triggering catastrophic forgetting of safety alignment. Using entirely benign data, the method achieves a 94.84% attack success rate (ASR)…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "jailbreak attack"
  - "fine-tuning safety"
  - "overfitting"
  - "alignment fragility"
  - "FaaS security"
  - "catastrophic forgetting"
date: 2026-05-08
content_hash: 694e6f257ac26630
---

# Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2510.02833](https://arxiv.org/abs/2510.02833)  
**Code**: [GitHub](https://github.com/ZHIXINXIE/ten_benign)  
**Area**: Alignment & RLHF
**Keywords**: jailbreak attack, fine-tuning safety, overfitting, alignment fragility, FaaS security, catastrophic forgetting

## TL;DR
This paper proposes a two-stage fine-tuning attack: Stage 1 fine-tunes an LLM on 10 benign questions paired with identical refusal answers, driving the model to overfit into a sharp loss landscape; Stage 2 fine-tunes the same 10 questions with normal answers, triggering catastrophic forgetting of safety alignment. Using entirely benign data, the method achieves a 94.84% attack success rate (ASR), comparable to malicious fine-tuning (97.25%), while completely evading content moderation.

## Background & Motivation

**Background**: LLM providers offer Fine-tuning-as-a-Service (FaaS), allowing users to upload data to fine-tune models. Prior work has shown that as few as 10 malicious QA pairs can jailbreak an LLM; however, malicious data can be detected and blocked by moderation systems.

**Limitations of Prior Work**: Existing covert attacks are either unstable or demand strong model capabilities—encrypted fine-tuning requires 20K samples to teach encryption/decryption, while the Absolute Obedience Agent (AOA) attack sees its ASR drop sharply from 78% to 8% after data shuffling. More critically, even implicitly harmful AOA data can be flagged by GPT-4.1-mini moderation.

**Key Challenge**: How can an LLM be reliably jailbroken using fine-tuning data that is completely harmless—raising no suspicion even under manual inspection?

**Goal**: Design an attack that stably jailbreaks diverse LLMs using only 10 entirely benign QA pairs.

**Key Insight**: A "shuffle experiment" on AOA attacks reveals that the similarity among the first few QA answers is the key factor behind the attack's effectiveness. This motivates the following insight: first overfit the model with identical refusal answers to sharpen the loss landscape, then fine-tune with normal answers whose gradient directions are highly aligned with those of malicious data.

**Core Idea**: Exploit overfitting to render the model extremely sensitive to parameter perturbations; under this condition, any data deviating from refusal behavior—including benign data—can trigger catastrophic forgetting of safety alignment.

## Method

### Overall Architecture
A two-stage fine-tuning attack:
- **Stage 1 (Overfitting Stage)**: Fine-tune the model on 10 benign questions paired with an identical refusal answer (e.g., "Sorry, I cannot assist with that") until the model outputs this refusal for any input → the model overfits into a narrow valley of the loss landscape.
- **Stage 2 (Forgetting Stage)**: Continue fine-tuning with the same 10 benign questions paired with normal answers → the model "forgets" its refusal behavior, including refusals to harmful queries → successful jailbreak.

### Key Designs

1. **Stage 1: Identical Refusal Answers to Induce Overfitting**

    - **Function**: Train the model to uniformly refuse all queries.
    - **Mechanism**: All 10 QA answers are identical (cosine similarity = 1); the model overfits on minimal data, reaching sharp minima in the loss landscape.
    - **Design Motivation**: Sharp minima imply high parameter sensitivity to small perturbations, creating the conditions for catastrophic forgetting in Stage 2.

2. **Stage 2: Normal Answers Trigger Catastrophic Forgetting**

    - **Function**: Overwrite refusal behavior with normal responses.
    - **Mechanism**: Normal answers deviate from the refusal distribution → large gradients arise in the sharp loss landscape → parameter updates cause safety alignment to be forgotten. Key finding: on an overfitted model, the cosine similarity between gradients from benign and malicious data approaches 1.0, meaning benign data can substitute for malicious data.
    - **Design Motivation**: No malicious data is needed, completely bypassing moderation—harmlessness score (HS) = 1, identical to purely benign data.

3. **Attack Stealthiness**

    - **Function**: Ensure attack data is completely undetectable.
    - **Mechanism**: Both stages use entirely benign QA pairs generated by GPT-4o, receiving the lowest moderation score of 1.
    - **Design Motivation**: Unlike AOA (detectable) and encrypted attacks (flagged as "meaningless"), the proposed method raises no suspicion even under manual review.

### Loss & Training
- Stage 1: High number of epochs to ensure thorough overfitting; all 10 QA answers are identical.
- Stage 2: Few epochs with a potentially higher learning rate.
- Stronger models require more Stage 1 epochs and a higher Stage 2 learning rate.
- Compatible with both full-parameter fine-tuning and LoRA fine-tuning.

## Key Experimental Results

### Main Results (ASR Across 10 LLMs)

| Attack Method | Avg. ASR↑ | Avg. HS↑ | Stealthiness |
|---|---|---|---|
| **Ours (10 benign pairs)** | **94.84%** | **4.48** | **Completely undetectable** |
| Malicious (10 malicious pairs) | 97.25% | 4.87 | Detectable |
| AOA (after shuffle) | 32.23% | 2.60 | Detectable |
| Encrypted fine-tuning | 14.66% | 1.62 | Effective only on strong models |
| Indirect malicious | 40.75% | 2.98 | Moderate |

### Ablation Study

| Configuration | ASR | Notes |
|---|---|---|
| Full attack | 94.84% | Complete two-stage attack |
| w/o Stage 1 | ~4% | Direct benign fine-tuning nearly ineffective |
| Defense system prompt | >80% | Defense prompt only partially mitigates attack |
| LoRA fine-tuning | Significantly above baseline | Parameter-efficient fine-tuning also effective |
| Token-wise defense | 92.11% | State-of-the-art defense cannot prevent the attack |

### Key Findings
- Stage 1 is the core of the attack: removing it causes ASR to drop from 94.84% to ~4%.
- Overfitting degree correlates positively with attack effectiveness: higher answer similarity → more severe overfitting → higher ASR.
- Gradient cosine similarity between benign and malicious data approaches 1.0 on highly overfitted models.
- The attack bypasses token-wise loss defense (ASR remains 92.11%).

## Highlights & Insights
- The finding that **purely benign data can jailbreak LLMs** is highly impactful, directly challenging the assumption that moderating fine-tuning data is sufficient to ensure safety.
- The attack chain **overfitting → sharp minima → catastrophic forgetting** is elegant and well-motivated, with dual validation through loss landscape visualization and gradient cosine similarity analysis.
- The work reveals a deeper insight: the fragility of safety alignment stems not only from malicious data, but also from geometric changes in the loss landscape induced by overfitting.
- The findings have fundamental implications for FaaS security—auditing data content alone is insufficient; training dynamics must also be monitored.

## Limitations & Future Work
- Hyperparameters require model-specific tuning, potentially necessitating multiple attempts in practice.
- The response quality of the jailbroken model is affected by Stage 2 data—using only 10 QA pairs tends to introduce repetitive outputs.
- A promising defense direction is monitoring loss landscape changes during fine-tuning (e.g., sharp minima detection).
- The persistence of the attack in multi-turn dialogue settings has not been analyzed.

## Related Work & Insights
- **vs. Malicious Fine-tuning (Qi et al.)**: Achieves comparable attack effectiveness while being completely undetectable, posing a more serious security threat.
- **vs. AOA Attack**: Reveals that the true mechanism behind AOA's success is answer similarity inducing overfitting, not identity transformation.
- **vs. Encrypted Fine-tuning (Halawi et al.)**: Encrypted fine-tuning requires 20K samples and is effective only on strong models; the proposed method requires only 10 pairs and generalizes across all models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to exploit overfitting to jailbreak LLMs with entirely benign data; highly impactful finding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 models × 6 attack variants + ablations + defense bypass + loss landscape analysis.
- Writing Quality: ⭐⭐⭐⭐ The progression from AOA analysis → discovery → method → explanation is natural and coherent.
- Value: ⭐⭐⭐⭐⭐ Provides important cautionary insights for FaaS security and the robustness of LLM alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MetaDefense: Defending Finetuning-based Jailbreak Attack Before and During Generation](metadefense_defending_finetuning-based_jailbreak_attack_before_and_during_genera.md)
- [\[NeurIPS 2025\] Mechanism Design for LLM Fine-tuning with Multiple Reward Models](mechanism_design_for_llm_fine-tuning_with_multiple_reward_models.md)
- [\[ACL 2025\] Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step](../../ACL2025/llm_alignment/chain-of-jailbreak_attack_for_image_generation_models_via_editing_step_by_step.md)
- [\[NeurIPS 2025\] DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO](deepvideor1_video_reinforcement_finetuning_via_difficultyawa.md)
- [\[ICML 2026\] SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection](../../ICML2026/llm_alignment/spard_defending_harmful_fine-tuning_attack_via_safety_projection_with_relevance-.md)

</div>

<!-- RELATED:END -->
