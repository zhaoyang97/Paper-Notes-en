---
title: >-
  [Paper Note] Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training
description: >-
  [ACL 2025][LLM (Other)][LLM Safety] This study identifies the "refusal position bias" in standard safety fine-tuning data, where models only learn to refuse at the start of a response and fail to interrupt when realizing unsafety midway. The authors propose DeRTa (Decoupled Refusal Training), which includes MLE training with "harmful prefix + safe refusal" and RTO training that simulates "harmful-to-safe" transition at every position. It enables LLMs to refuse whenever they d…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LLM Safety"
  - "Refusal Training"
  - "Jailbreak Defense"
  - "Position Bias"
  - "Reinforced Transition"
date: 2026-05-08
content_hash: 789ac07b6e5119a8
---

# Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training

**Conference**: ACL 2025  
**arXiv**: [2407.09121](https://arxiv.org/abs/2407.09121)  
**Code**: [https://github.com/RobustNLP/DeRTa](https://github.com/RobustNLP/DeRTa)  
**Area**: LLM Alignment  
**Keywords**: LLM Safety, Refusal Training, Jailbreak Defense, Position Bias, Reinforced Transition  

## TL;DR
This study identifies the "refusal position bias" in standard safety fine-tuning data, where models only learn to refuse at the start of a response and fail to interrupt when realizing unsafety midway. The authors propose DeRTa (Decoupled Refusal Training), which includes MLE training with "harmful prefix + safe refusal" and RTO training that simulates "harmful-to-safe" transition at every position. It enables LLMs to refuse whenever they detect unsafety at any position in the response, outperforming GPT-4 and LLaMA3-Instruct across six attack scenarios.

## Background & Motivation
**Background**: LLMs learn to refuse harmful requests through safety fine-tuning (SFT + RLHF). However, jailbreak attacks can bypass these protections through various means.

**Limitations of Prior Work**: (a) In standard safety fine-tuning data, refusals always appear at the very beginning of the response—the model learns to "determine safety at the start" rather than "maintain safety vigilance at all times"; (b) Certain attacks (such as prefix injection and incremental persuasion) prompt the model to generate seemingly harmless content first before sliding into harmful content—the model cannot "brake" once generation begins; (c) There is a lack of training data to teach the model "how to interrupt and transition to a safe response when realizing unsafety midway."

**Key Challenge**: Refusal should be able to occur at any position during generation, but in training data, Refusal only occurs at position 0—leading the model to only learn a binary decision of "refuse or not at the very beginning", without knowing how to "refuse midway".

**Goal**: Enable LLMs to perceive unsafety at any position of responses and interrupt to transition to refusal.

**Key Insight**: Explicitly construct "harmful-then-safe" transition sequences in safety training data—allowing the model to learn the skill of converting "from harmful to safe", rather than just learning "directly safe" patterns.

**Core Idea**: Teach the model to "brake midway"—transitioning from harmful outputs to safe refusals at any position.

## Method

### Overall Architecture
DeRTa consists of two complementary components: (1) **MLE with Harmful Prefix**—training with a harmful response prefix of random length concatenated before the safe response, teaching the model to "transition to refusal even under harmful generation"; (2) **Reinforced Transition Optimization (RTO)**—simulating "harmful-to-safe" transitions at each position of the harmful response sequence, reinforcing this transition capability using an auxiliary training objective.

### Key Designs

1. **MLE with Harmful Response Prefix**:

    - **Function**: Trains the model to transition to a safe refusal even after generating some harmful content.
    - **Mechanism**: For each harmful query, an (unaligned) LLM is first used to generate a harmful response, and then a random-length harmful prefix is cut and prepended to the safe refusal response. The training objective is to predict the "refusal response" part.
    - Training data format: `[Harmful Query] [First n tokens of Harmful Response] [<Refusal>] [Safe Refusal Response]`
    - **Design Motivation**: Harmful prefixes provide extra contextual information to help the model identify unsafe content—only by "realizing the bad things it is saying" can the model decide to stop.

2. **Reinforced Transition Optimization (RTO)**:

    - **Function**: Reinforces the ability to "transition to safety" at every token position of the harmful response.
    - **Mechanism**: MLE only provides a single transition (from the end of the prefix to refusal), which is insufficient. RTO calculates preference scores for "transitioning to safe refusal from here" vs "continuing harmful generation" at each position $t$ across the entire harmful sequence, optimizing the model to prefer transitioning to safety at all positions via reinforcement learning.
    - **Design Motivation**: Ensures the model is capable of transitioning at any position rather than just one specific position.

3. **Synergy of Two Components**:

    - MLE with Harmful Prefix provides the learning signal for "single transition"—getting one decision right.
    - RTO generalizes this capability to all positions—making the right decision no matter which token it is on.
    - The combination of both achieves true "anytime, anywhere refusal".

### Loss & Training
- MLE stage: Standard autoregressive loss, where loss is only computed on the safe refusal segment.
- RTO stage: DPO-style preference optimization—preferred = transition to safety from current position, dispreferred = continue harmful generation.
- Two-stage sequential training (MLE followed by RTO).

## Key Experimental Results

### Main Results

| Method | Average Safety Rate(↑) | Helpfulness Preservation | Description |
|------|-------------|----------|------|
| LLaMA3-8B-Instruct | Baseline | High | Standard safety fine-tuning |
| Standard Safety SFT | Medium | High | Position bias issue |
| GPT-4 | Medium-High | High | Closed-source reference |
| **DeRTa (LLaMA3-8B)** | **Highest** | **High** | Outperforms GPT-4 |
| **DeRTa (LLaMA3-70B)** | **Highest** | **High** | Larger model is better |

### Ablation Study

| Configuration | Performance| Description |
|------|------|------|
| MLE only (w/o RTO) | Significant improvement but insufficient | Single transition is insufficient to cover all positions |
| RTO only (w/o MLE) | Limited improvement | Lacks foundation for RTO without learning prefixes |
| **MLE + RTO** | **Optimal** | Complementary to each other |
| Diverse harmful prefix length | Randomized length is optimal | Fixed length may lead to position overfitting |

### Key Findings
- **Refusal position bias** is a real and critical issue—standard safety fine-tuned models indeed "do not know how to brake midway".
- DeRTa significantly improves safety without compromising helpfulness—instead of simply making the model refuse more, it makes it refuse more precisely.
- Outperforming GPT-4 safety on LLaMA3-8B—structured safety training can compensate for model scale gaps.
- Position-level reinforcement of RTO is key—single transitions from MLE alone are insufficient; reinforcement at every position is required.
- Randomizing harmful prefix length avoids overfitting to specific positions.

## Highlights & Insights
- The discovery of **"refusal position bias"** is simple yet profound—all prior safety fine-tuning works failed to notice that refusals always occur at the beginning of training data, posing a systematic blind spot.
- The metaphor of **"braking midway"** has great intuition—not keeping the car off the road, but teaching it how to brake at any moment.
- The progressive design of **MLE + RTO progressive strategy** is exquisite—first learning a single transition, then generalizing to all positions.
- Direct practical impact on all LLM safety fine-tuning—"midway refusal" samples should be introduced into training data.
- The method is orthogonally combinable with other safety defenses (e.g., input detection, representation engineering).

## Limitations & Future Work
- Harmful response generation requires use of unaligned LLMs—if attacks bypass harmful response patterns, it might be ineffective.
- Evaluation is limited to English scenarios—cross-lingual attacks remain unverified.
- RTO increases training complexity—preference is calculated at multiple positions for each sample.
- Possible over-refusal in boundary scenarios that "actually need caution but should ultimately be answered".

## Related Work & Insights
- **vs Standard Safety Fine-tuning (LLaMA-Guard/Safe RLHF)**: Standard methods only train "refusal at the beginning"; DeRTa trains "refusal at any position".
- **vs Representation Engineering (RepE/Lat-AT)**: Representation engineering operates on safety features in hidden layers; DeRTa resolves the issue at the training data level—at different levels.
- **vs Wallace et al. (2024) Priority Training**: Priority Training ensures safety priority; DeRTa addresses the timing of refusal—complementary.
- Highly valuable as a reference for adversarial safety (Red Teaming) and defense research.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of "refusal position bias" and the training method for "midway refusal" are both original and important
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six attacks + two model families + ablation + helpfulness evaluation + qualitative analysis
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem discovery process, natural method motivation, and fluid progressive logic from MLE to RTO
- Value: ⭐⭐⭐⭐⭐ Direct and significant practical impact on LLM safety fine-tuning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement](why_not_act_on_what_you_know_unleashing_safety_potential_of_llms_via_self-aware_.md)
- [\[ACL 2025\] Representation Bending for Large Language Model Safety](repbend_representation_bending_safety.md)
- [\[ACL 2025\] COSMIC: Generalized Refusal Direction Identification in LLM Activations](cosmic_generalized_refusal_direction_identification_in_llm_activations.md)
- [\[ACL 2025\] GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction](genknowsub_improving_modularity_and_reusability_of_llms_through_general_knowledg.md)
- [\[ACL 2025\] Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes](improving_preference_extraction_in_llms_by_identifying_latent_knowledge_through_.md)

</div>

<!-- RELATED:END -->
