---
title: >-
  [Paper Note] Adaptive Probe-based Steering for Robust LLM Jailbreaking
description: >-
  [ICML 2026][LLM Alignment][LLM Jailbreaking] This paper transforms probe-based contrastive steering into a more potent white-box red-teaming tool. By utilizing adaptive retraining to correct biased probes and automatical…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "LLM Jailbreaking"
  - "Representation Intervention"
  - "probe steering"
  - "adaptive retraining"
  - "robustness evaluation"
date: 2026-05-08
content_hash: 5ebc398d9bd68e79
---

# Adaptive Probe-based Steering for Robust LLM Jailbreaking

**Conference**: ICML 2026  
**arXiv**: [2605.20286](https://arxiv.org/abs/2605.20286)  
**Code**: https://github.com/fhdnskfbeuv/adaptiveSteering  
**Area**: LLM Security / Red-teaming Evaluation  
**Keywords**: LLM Jailbreaking, Representation Intervention, probe steering, adaptive retraining, robustness evaluation  

## TL;DR
This paper transforms probe-based contrastive steering into a more potent white-box red-teaming tool. By utilizing adaptive retraining to correct biased probes and automatically setting steering strength via activation statistics, it significantly exposes the jailbreak vulnerabilities of fortified LLMs.

## Background & Motivation
**Background**: Aligned LLMs typically refuse harmful requests. Safety evaluation requires strong attacks to estimate worst-case robustness. Prompt-level jailbreaks, gradient optimization, fine-tuning attacks, and hidden state steering are used for red-teaming. The advantage of contrastive steering is that it only requires forward passes and a set of contrastive prompts, independent of target responses or input gradients.

**Limitations of Prior Work**: Existing steering methods have two key sources of instability. First, direction searching relies on a small number of "harmful/harmless" contrastive prompts, which encode not only "compliance/refusal" but also coupled directions like ethics, topics, and writing styles. The learned linear probes may deviate from the true jailbreak behavior direction. Second, steering strength often requires manual tuning or applying a uniform logit target across all layers; however, activation norms vary greatly across layers, and uniform strength easily leads to over-intervention in early layers and collapsed responses.

**Key Challenge**: Strong safety evaluation requires sufficiently powerful attacks to avoid overestimating defense capabilities; however, stronger steering more easily destroys linguistic ability or generates incoherent content. This paper aims to solve the contradiction: "the direction must be closer to the target behavior, while the strength cannot rely on manual brute-force tuning."

**Goal**: The authors aim to improve the attack effectiveness and cross-model robustness of probe-based steering against fortified LLMs without extra contrastive prompt collection, backpropagation, or manual per-layer strength searching.

**Key Insight**: The paper views the search for the probe direction as model extraction: an ideal probe is an invisible behavior discriminator, while existing contrastive prompts only provide biased samples. If the model can generate new activations under current steering and have them labeled by a judge, one can iteratively approach a more reliable direction.

**Core Idea**: Use "judge-labeled adaptive activation retraining" to correct the steering direction and "statistics of homologous contrastive activations" to automatically set the target strength for each layer.

## Method
The goal is to improve probe-based steering under a threat model with white-box access to hidden states, rather than constructing new prompt templates. The method retains the basic form of contrastive steering: adding a vector along the probe direction to the hidden states of certain Transformer layers to move the model state closer to the target behavior region. Improvements are concentrated on how the direction is learned and how far to push each layer.

### Overall Architecture
Inputs include a few harmful/harmless contrastive prompts, an LLM with accessible hidden states, and an evaluator to label whether responses are "faithful to harmful requests." First, hidden states are extracted from contrastive prompts to train initial linear probes for each layer. Then, adaptive retraining begins: the current probe steers the model on harmful prompts to collect activations and responses, which the judge labels. These new activations are added to the training set to retrain the probes. Finally, during inference, instead of uniform manual strength, an adaptive target is set for each layer based on the probe logit statistics of target behaviors in the training set, applying steering to all token positions.

### Key Designs
1. **Adaptive probe retraining based on model extraction**:
    - **Function**: Reduces direction bias introduced by initial contrastive prompts, bringing the probe closer to the actual direction controlling the target behavior.
    - **Mechanism**: In round $i$, the current probe steers the model to generate responses and collect intermediate activations; an external judge then labels these responses, and those activations are added to the training set for the next round. The main experiment sets $T=20$ and uses high/low thresholds from an SRF judge to filter reliable samples.
    - **Design Motivation**: Simply adding more contrastive prompts introduces more noise in topic and ethics dimensions; active sampling using activations from the currently steered model enables more focused exploration of behavior directions near the decision boundary.

2. **Adaptive strength setting based on activation statistics**:
    - **Function**: Avoids manual per-layer tuning and reduces oversteering caused by uniform logit targets.
    - **Mechanism**: Probe steering can be framed as pushing the hidden state toward a specific probe logit target. The authors observe that hidden state norms vary by orders of magnitude across layers, so a uniform target causes excessive relative perturbation in layers with smaller norms. The method sets $s^{(l)}$ using logit statistics of target activations from the training set, pushing current activations into a reasonable target interval for that specific layer.
    - **Design Motivation**: Safety evaluation must increase harmfulness without pushing the model into nonsensical output. Layer-wise statistical strength serves as calibration using the "scale of true target activations at that layer."

3. **Dropping final layers and steering all token positions**:
    - **Function**: Enhances generation coherence and intervention coverage.
    - **Mechanism**: Hidden states in final decoder layers are close to logits; intervening here is similar to logit bias, inducing repetition or local tone changes. Thus, final layers are dropped. Furthermore, probe steering is interpreted as a fixed-direction rank-1 adapter, meaning it should apply to all token positions rather than just response tokens.
    - **Design Motivation**: Modifying only response tokens allows representations in prompt tokens—potentially manipulated by defense methods—to dominate generation, reducing effectiveness against models like Circuit Breaker or RepBend.

### Loss & Training
Probes are trained using linear classification on samples from initial contrastive activations and subsequent adaptive labeled activations. Direction search uses 100 pairs of contrastive prompts (50 for training/iteration, 50 harmful for validation). Evaluation uses 200 harmful prompts from StrongReject and HarmBench. Judges include SRF, StrongReject rubric judge, and HarmBench classifier, with scores normalized from 0 to 1. The method is positioned as a red-teaming tool to reveal worst-case scenarios under white-box intervention.

## Key Experimental Results

### Main Results
The experiment covers 12 fortified LLMs and reports harmfulness scores across three judges.

| Method | Representative Result | Avg Harmfulness | Relative Conclusion |
|-------|-----------------------|-----------------|---------------------|
| RepE | Near 0 for most models | 0.02 | Traditional representation engineering fails against strong defenses |
| SCAV | Up to 0.64 on Gemma-DA | 0.05 | Original probe steering lacks robustness |
| RD-A | Strong on Llama3-RB | 0.24 | Ablation steering is effective but inconsistent |
| RD-C | Effective on some | 0.18 | Constant strength still requires tuning |
| Angular | Some effect on R2D2 | 0.13 | Direction improvement $\neq$ Strength robustness |
| **Ours** | **Llama3-RB 0.98, Llama3-CB 0.91** | **0.70** | Adaptive direction and strength significantly boost evaluation intensity |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| SCAV | Avg 0.05 | Original probe steering mostly stays near zero |
| SCAV + AS | Avg 0.14 | Adaptive Strength (AS) pulls some models out of the zero zone |
| SCAV + AS + DLA | Avg 0.29 | Dropping Last layers (DLA) reduces logit-bias effects, improving coherence |
| Adding SAT | ~+15% gain | Steering All Tokens (SAT) especially improves CB/RB series |
| Naive Augmentation | Max +6% | Simply adding more prompts has limited utility due to coupled directions |
| Adaptive Retraining | ~+26% gain | Judge-labeled activations effectively correct the probe direction |

### Key Findings
- The strongest evidence is the jump in avg. harmfulness on fortified LLMs from near-zero to 0.70, showing significant vulnerabilities in seemingly robust models under white-box steering.
- Adaptive strength provides the baseline gain, while model-extraction retraining provides the primary gain. One solves "how far to push," the other solves "where to push."
- Analysis of R2D2 reminds that safety evaluation must distinguish "true safety" from "general refusal." If a model fails benign prompts, low harmfulness does not equate to high security.

## Highlights & Insights
- Interpreting probe direction learning as model extraction is insightful. It demonstrates that the problem with contrastive prompts is not just quantity, but the inherent entanglement of multiple conceptual directions.
- Variation in activation scales across layers is an often-overlooked detail. Using target activation statistics instead of uniform strength makes the method act like a layer-calibrated adapter.
- The implication for defense research is direct: evaluating only against weak attacks leads to overly optimistic robustness conclusions. Red-teaming must utilize the strongest available interventions, particularly white-box representation steering.

## Limitations & Future Work
- The threat model is strong, requiring access to and modification of all hidden states, making it primarily suitable as a white-box evaluation tool.
- Direction learning depends on an LLM judge, which may introduce bias.
- The focus is entirely on harmfulness; there is little discussion on how defenders might design robust training or detection mechanisms against such steering attacks.
- The method reveals whether "harmful behavior patterns remain activatable inside the model." Future work could combine this with mechanistic interpretability to locate failure layers.

## Related Work & Insights
- **vs. Refusal Direction (RD)**: RD uses constant or ablation steering along a refusal direction. Ours uses probe directions with adaptive layer strength, offering better cross-model stability.
- **vs. SCAV**: SCAV relies on accuracy-based layer selection and uniform targets. This paper identifies these as unstable and prone to oversteering.
- **vs. Prompt-level jailbreak**: Prompt attacks are limited by discrete input spaces and search difficulty; this method acts directly on hidden states for worst-case evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐☆
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐☆
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Steering Beyond the Support: Adversarial Training on Unsupervised Jailbroken Activation Simulation](steering_beyond_the_support_adversarial_training_on_unsupervised_jailbroken_acti.md)
- [\[ICLR 2026\] Robust Preference Alignment via Directional Neighborhood Consensus](../../ICLR2026/llm_alignment/robust_preference_alignment_via_directional_neighborhood_consensus.md)
- [\[ICLR 2026\] Ignore All Previous Instructions: Jailbreaking as a de-escalatory peace building practise to resist LLM social media bots](../../ICLR2026/llm_alignment/ignore_all_previous_instructions_jailbreaking_as_a_de-escalatory_peace_building_.md)
- [\[AAAI 2026\] When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF](../../AAAI2026/llm_alignment/when_human_preferences_flip_an_instance-dependent_robust_loss_for_rlhf.md)
- [\[ICLR 2026\] AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint](../../ICLR2026/llm_alignment/alphasteer_learning_refusal_steering_with_principled_null-space_constraint.md)

</div>

<!-- RELATED:END -->
