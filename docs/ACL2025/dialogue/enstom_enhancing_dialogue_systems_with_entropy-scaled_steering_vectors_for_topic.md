---
title: >-
  [Paper Note] EnSToM: Enhancing Dialogue Systems with Entropy-Scaled Steering Vectors for Topic Maintenance
description: >-
  [ACL 2025][Dialogue Systems][Topic Consistency] EnSToM is proposed as a lightweight method based on entropy-scaled steering vectors, which dynamically adjusts steering intensity by leveraging the differences in internal layer entropy distributions of LLMs to enhance the topic maintenance capability of task-oriented dialogue systems without modifying model parameters.
tags:
  - "ACL 2025"
  - "Dialogue Systems"
  - "Topic Consistency"
  - "Steering Vector"
  - "Entropy Scaling"
  - "Activation Engineering"
date: 2026-05-08
content_hash: 2af528affe192379
---

# EnSToM: Enhancing Dialogue Systems with Entropy-Scaled Steering Vectors for Topic Maintenance

**Conference**: ACL 2025  
**arXiv**: [2505.16526](https://arxiv.org/abs/2505.16526)  
**Code**: [https://github.com/linkyouhj/enstom](https://github.com/linkyouhj/enstom)  
**Area**: Dialogue Systems  
**Keywords**: Dialogue Systems, Topic Consistency, Steering Vector, Entropy Scaling, Activation Engineering

## TL;DR

EnSToM is proposed as a lightweight method based on entropy-scaled steering vectors, which dynamically adjusts steering intensity by leveraging the differences in internal layer entropy distributions of LLMs to enhance the topic maintenance capability of task-oriented dialogue systems without modifying model parameters.

## Background & Motivation

**Background**: Small Large Language Models (sLLMs) are suitable for deployment in resource-constrained environments due to their lightweight and efficient nature. Enterprise task-oriented dialogue systems (such as banking customer service bots) require models to strictly adhere to predetermined topics and refuse off-topic or malicious inputs.

**Limitations of Prior Work**: (1) sLLMs have limited capacity, making it difficult to maintain scenario consistency during long-term interactions; (2) **Fine-tuning** methods require large amounts of data and computing resources, making it hard to cover all scenarios; (3) **Prompt engineering** has limited effectiveness in complex scenarios; (4) Directly applying **steering vectors** can improve off-topic rejection rates but severely damages the quality of on-topic responses (on-topic accuracy drops from 0.94 to 0.70).

**Key Challenge**: While steering vectors effectively improve the ability to reject distractor inputs, indiscriminately applying steering to all inputs causes on-topic responses to be incorrectly rejected as well—how can steering be dynamically adjusted based on the input?

**Goal**: Design an adaptive steering intensity adjustment mechanism that strongly steers distractor inputs while applying weak or no steering to on-topic inputs.

**Key Insight**: It is observed that the entropy distribution across different layers of LLMs exhibits significant differences between on-topic and distractor inputs, which can serve as a distinguishing signal to dynamically adjust the steering coefficient.

**Core Idea**: Leverage layer-wise generation entropy in LLMs to distinguish off-topic from on-topic inputs, dynamically scaling the steering vector intensity via a sigmoid function to achieve precise topic maintenance.

## Method

### Overall Architecture

EnSToM consists of three components: (1) extracting steering vectors from contrastive data; (2) dynamically adjusting steering intensity based on entropy-based coefficient scaling; (3) generating responses using the scaled steering vectors. The entire process is training-free and intervenes purely at inference time.

### Key Designs

1. **Steering Vector Extraction**: Construct a Steering QA Dataset $S = \{q_1, q_2, \dots\}$, where each $q_i$ contains contrastive prompts of desired behaviors (refusing and redirecting back to the topic) and undesired behaviors (continuing to answer off-topic questions). Forward propagation is performed at a designated layer $l$ to compute the difference in hidden representations between desired and undesired behaviors:
    $v_s^i = h_p^{(l)} - h_n^{(l)}$
   The final steering vector is obtained by normalization and averaging: $v = \frac{1}{k}\sum_{i=1}^{k} \text{norm}(v_s^i)$.

2. **Layer-wise Entropy Analysis**: Compute the entropy of generating the first 2 tokens at layer $l$ of the LLM:
    $$E^{(l)} = \mathbb{E}\left[-\sum_{i=1}^{V} p_i^{(l)} \log(p_i^{(l)} + \epsilon)\right]$$
   Key finding: At **Layer 16** (a semantically critical layer), the entropy of distractor inputs is **lower** than that of on-topic inputs (since off-topic content induces highly focused attention); at **Layer 19** (a deeper layer), this relationship reverses.

3. **Entropy-based Scaling Coefficient**: A sigmoid function maps the entropy to a steering coefficient:
    $$C_H^{(L)} = \frac{C_{\max}}{1 + e^{-\alpha \delta (H^{(L)} - t)}}$$
   Where $C_{\max} = 1.5$ is the maximum coefficient, $\alpha = 5$ controls the steepness of the sigmoid, $t = 7.5$ is the threshold, and $\delta$ takes $\pm 1$ depending on the direction of the entropy distribution. This assigns a high coefficient (strong steering) to distractor inputs and a low coefficient (weak/no steering) to on-topic inputs.

4. **Response Generation**: During inference, 2 tokens are generated first to compute the entropy and obtain the coefficient, and then the scaled steering vector is added to the activations at the specified layer:
    $h'^{(l)} = h^{(l)} + C_H^{(L)} \cdot v$

### Loss & Training

- **Completely Training-Free**: Requires only about 100 contrastive samples to extract steering vectors.
- Rejection and response options are generated by GPT-4o, with positions randomly assigned to avoid positional bias.
- Evaluation utilizes a GPT-4o classification model to categorize response categories as rejection/response.

## Key Experimental Results

### Main Results (LLaMA-2-7B-Chat, CantTalkAboutThis Banking Domain)

| Method | Entropy Layer L | Steering Layer | Distractor ↑ | On-topic ↑ | Overall ↑ |
|------|--------|--------|-------------|-----------|----------|
| Prompt Only | -| - | 0.282 | 0.938 | 0.610 |
| Vanilla Steering | - | - | 0.800 | 0.700 | 0.750 |
| EnSToM | 16 | 15 | 0.810 (+0.53) | 0.747 (-0.19) | 0.779 |
| **EnSToM** | **16** | **16** | **0.709 (+0.43)** | **0.895 (-0.04)** | **0.802** |
| EnSToM | 19 | 16 | 0.749 (+0.47) | 0.818 (-0.12) | 0.784 |

Optimal configuration (L=16, Steer@16): overall score of 0.802, which is 19.2 percentage points higher than Prompt Only and 5.2 percentage points higher than Vanilla, with on-topic performance dropping by only 4.3 percentage points.

### Cross-Architecture Generalization (Ministral-8B-Instruct)

| Method | Distractor | On-topic | Overall |
|------|-----------|---------|---------|
| Prompt Only | 0.25 | 0.98 | 0.62 |
| EnSToM @ layer 18 | 0.63 (+0.38) | 0.91 (-0.07) | **0.76** |

### Ablation Study (Impact of Threshold $t$)

| Threshold $t$ | Distractor | On-topic | Overall |
|----------|-----------|---------|---------|
| Vanilla (Fixed) | 0.80 | 0.70 | 0.75 |
| $t = 2$ | 0.30 | 0.95 | 0.63 |
| $t = 7.5$ | 0.76 | 0.84 | **0.80** |
| $t = 9$ | ~baseline | 0.72 | ~0.6x |

### Data Efficiency

Effective steering vectors can be extracted with as few as 10 contrastive samples: distractor accuracy is 0.74 (vs. 0.81 with 100 samples), and on-topic accuracy is 0.85 (vs. 0.75), making it highly suitable for low-resource scenarios.

### Key Findings

- **Entropy Separation is Most Pronounced at Layer 16**: Middle layers encode semantic information; distractor inputs focus on a small number of unique tokens, leading to low entropy, while on-topic inputs have scattered attention, resulting in high entropy.
- **Cross-Domain Consistency**: Steering vectors extracted from various domains (such as banking, education, health, and insurance) are all effective, indicating that the rejection mechanism is general rather than domain-specific.
- **Potential for Task Generalization**: In jailbreak defense tasks, the entropy distribution at Layer 33 can similarly distinguish between harmful and harmless inputs.
- **Analysis of Coefficient Distribution**: 82.5% of distractors are assigned $C \geq 1.0$ (strong steering), and 45.8% of on-topic inputs are assigned $C < 0.5$ (weak steering), aligned with the design expectations.
- **Robustness of On-topic Inputs to Over-steering**: Even though 40.2% of on-topic inputs are assigned $C \geq 1.0$, the accuracy still reaches 0.79.

## Highlights & Insights

- The core finding is highly elegant: internal layerwise entropy of LLMs naturally distinguishes on-topic and distractor inputs without requiring external classifiers.
- Completely training-free inference-time intervention requires only ~100 contrastive samples, leading to exceptionally low deployment costs.
- The analysis of layer-wise functional differentiation is consistent with findings in cognitive science: shallow layers capture syntax, middle layers encode semantics, and deep layers integrate context.
- Dynamic coefficients offer a clear advantage over fixed coefficients in avoiding "harm to normal dialogue."

## Limitations & Future Work

1. **Manual Selection of Layers and Thresholds**: The entropy extraction layer $L$ and threshold $t$ are currently determined empirically, which needs to be automated in the future.
2. **Hard Negative Samples**: Samples in the overlapping regions of entropy distribution can be misclassified, leading to incorrect steering directions.
3. **Evaluated only on 7B/8B Models**: The effectiveness on larger models (70B+) has not been verified.
4. **Evaluation Relies on GPT-4o**: Categorizing responses into rejection or responding relies on GPT-4o, which may introduce bias.
5. **In-depth Evaluation Limited to Banking Domain**: Cross-domain experiments only utilize steering vector transfer without comprehensive domain adaptation.

## Related Work & Insights

- **Steering Vectors**: First proposed by Turner et al. 2023 and applied to LLaMA-2 by Rimsky et al. 2024—this work adds entropy scaling to resolve the on-topic degradation issue.
- **Topic Maintenance**: CantTalkAboutThis (Sreedhar et al. 2024) provides the dataset, and Llama Guard achieves safety guarding via instruction tuning—this work offers a much more lightweight alternative.
- **Utilization of LLM Internal States**: DoLa (Chuang et al. 2024) improves truthfulness via layer-wise contrast, and INSIDE (Chen et al. 2024) uses internal states to detect hallucinations—this work leverages internal entropy for input classification.
- **Insights**: Entropy signals can be extended to more scenarios (e.g., hallucination detection, identifying uncertainty); they can also be combined with parameter-efficient methods such as LoRA.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The idea of entropy-scaled steering vectors is highly novel, elegantly combining activation engineering with internal signals.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Includes multi-layer analysis, cross-architecture, cross-domain, and data efficiency experiments, though model scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, rigorous formulations, and logical diagrams.
- **Value**: ⭐⭐⭐⭐ — Provides a practical, training-free solution for topic maintenance in dialogue systems, making a significant contribution to the field of activation engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Goal-oriented Proactive Dialogue Systems via Consistency Reflection and Correction](enhancing_goal-oriented_proactive_dialogue_systems_via_consistency_reflection_an.md)
- [\[ACL 2025\] Dialogue Systems for Emotional Support via Value Reinforcement](dialogue_systems_for_emotional_support_via_value_reinforcement.md)
- [\[ACL 2025\] Enabling Chatbots with Eyes and Ears: An Immersive Multimodal Conversation System](enabling_chatbots_with_eyes_and_ears_an_immersive_multimodal_conversation_system.md)
- [\[NeurIPS 2025\] LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation](../../NeurIPS2025/dialogue/latentguard_controllable_latent_steering_for_robust_refusal_of_attacks_and_relia.md)
- [\[ACL 2025\] DEMO: Reframing Dialogue Interaction with Fine-grained Element Modeling](demo_reframing_dialogue_interaction_with_fine-grained_element_modeling.md)

</div>

<!-- RELATED:END -->
