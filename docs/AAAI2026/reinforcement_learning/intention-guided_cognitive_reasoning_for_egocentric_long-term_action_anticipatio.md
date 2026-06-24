---
title: >-
  [Paper Note] Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation
description: >-
  [AAAI 2026][Reinforcement Learning][Long-term action anticipation] This paper proposes INSIGHT, a two-stage unified framework for egocentric long-term action anticipation (LTA). Stage one enhances action representations via hand-object interaction (HOI) region feature extraction and verb-noun co-occurrence matrices; stage two introduces a GRPO-based reinforcement learning cognitive reasoning module that simulates a structured "perceive → reason → answer" cognitive process for…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Long-term action anticipation"
  - "egocentric video"
  - "hand-object interaction"
  - "cognitive reasoning"
  - "GRPO"
date: 2026-05-08
content_hash: bf68bd9013f265fa
---

# Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation

**Conference**: AAAI 2026
**arXiv**: [2508.01742](https://arxiv.org/abs/2508.01742)  
**Code**: [github.com/CorrineQiu/INSIGHT](https://github.com/CorrineQiu/INSIGHT)  
**Area**: Reinforcement Learning
**Keywords**: Long-term action anticipation, egocentric video, hand-object interaction, cognitive reasoning, GRPO

## TL;DR

This paper proposes INSIGHT, a two-stage unified framework for egocentric long-term action anticipation (LTA). Stage one enhances action representations via hand-object interaction (HOI) region feature extraction and verb-noun co-occurrence matrices; stage two introduces a GRPO-based reinforcement learning cognitive reasoning module that simulates a structured "perceive → reason → answer" cognitive process for intention inference and action prediction.

## Background & Motivation

Long-term action anticipation (LTA) aims to predict future action sequences from observed egocentric video clips, and is a critical capability for human-computer interaction, augmented reality, and assistive systems. Accurately predicting future user actions enables AI systems to proactively adapt to behavior and provide timely assistance.

**Three key limitations of existing methods**:

**Neglect of fine-grained visual cues**: Existing methods insufficiently exploit fine-grained information from HOI regions. HOI regions are densely populated with action-relevant cues that are crucial for distinguishing subtle context-dependent behaviors. General-purpose visual encoders process entire frames directly, discarding these critical egocentric perceptual details.

**Ignorance of verb-noun semantic associations**: Independently predicting verbs and nouns can yield semantically implausible combinations (e.g., "drink + guitar"), reducing prediction reliability. Existing methods lack explicit modeling of verb-noun co-occurrence statistics.

**Absence of explicit cognitive reasoning**: Most methods treat LTA as a passive sequence prediction task, lacking an active decision-making reasoning process. Although LLM-based methods introduce textual reasoning, they rely solely on static priors and lack dynamic intention inference capability, making them brittle in complex extended temporal scenarios.

## Method

### Overall Architecture

INSIGHT consists of two stages:
- **Stage 1: HOI-enhanced semantic action recognition** — extracts discriminative visual features and enhances semantic consistency
- **Stage 2: Explicit cognitive reasoning prediction** — simulates a "think → reason → answer" cognitive process

### Key Designs

#### 1. **HOI-Enhanced Feature Extraction**

Conventional methods apply visual encoders directly to full frames. INSIGHT introduces an HOI-focused feature extraction strategy:

- Uniformly samples 4 frames $F_{k,T}$ from each video segment $S_k$
- Applies a pretrained 100DOH detector for HOI region detection on each frame, then refines high-resolution masks with SAM2 to obtain precise HOI region masks $R_{k,T}$
- Employs a dual-stream EgoVideo-V architecture to simultaneously encode full frames and HOI regions:

$$(\mathbf{X}_{k,T}^{ori}, \mathbf{X}_{k,T}^{mask}) = \text{EgoVideo-V}(F_{k,T}, R_{k,T})$$

- Both embeddings are concatenated and fused through a linear MLP, with a Transformer module capturing spatiotemporal relationships

This design organically integrates global scene context with local HOI details, significantly improving the semantic accuracy of verb-noun predictions.

#### 2. **Verb-Noun Co-occurrence Semantic Correction**

The Transformer output passes through dual classifiers (verb classifier + noun classifier), but independently predicted verb-noun pairs may be semantically implausible. INSIGHT constructs a co-occurrence matrix for semantic correction:

A co-occurrence matrix $\mathbf{C} \in \mathbb{N}^{|\mathcal{V}| \times |\mathcal{N}|}$ is computed from training data statistics:

$$\mathbf{C}_{v,n} = \sum_{k=1}^{K} \mathbf{1}_{\{v_k = v \wedge n_k = n\}}$$

Row/column normalization yields conditional probabilities $\mathbf{P}^{(n|v)}$ and $\mathbf{P}^{(v|n)}$, and the corrected joint probability is:

$$\tilde{p}(v_k, n_k) = p(v_k) \cdot p(n_k) \cdot \frac{1}{2}(\mathbf{P}^{(n|v)}_{v,n} + \mathbf{P}^{(v|n)}_{v,n})$$

The optimal verb-noun pair is selected via MAP estimation. This effectively filters out semantically implausible combinations and enhances prediction reliability.

#### 3. **GRPO-Based Cognitive Reasoning Module**

Stage two uses Qwen2.5-VL-7B as the backbone and introduces a structured reasoning pipeline "think → reason → answer":

- **think (visual perception)**: `<think>...</think>` perceives the current scene
- **reason (intention inference)**: `<intention>...</intention>` infers the user's high-level task intention
- **answer (action prediction)**: `<answer>...</answer>` outputs the predicted action sequence

**Format rewards** (ensuring structured output):
- Length reward $S_{len}$: whether the number of predicted action pairs meets requirements
- Tag order reward $S_{fmt}$: whether the think→intention→answer structure is followed
- Language consistency reward $S_{lang}$: whether the output is entirely in English
- Soft over-length penalty $R_{Soft}$: linearly decreasing penalty for excessively long outputs

**Content rewards**:
- **Accuracy reward** $S_{acc}$: based on edit distance (ED) normalized to $[0,1]$

$$S_{acc} = 1 - \frac{d_{ED}^Z}{|\mathbf{s}_{true}|}$$

- **Intention reward** $S_{int}$: cosine similarity between generated intentions and GPT-4.1-generated pseudo ground-truth intentions computed via Sentence-BERT, normalized with a scaled sigmoid:

$$S_{int} = \min\left(\frac{1}{1+\exp[-\gamma(sim-\beta)]} \Big/ \frac{1}{1+\exp[-\gamma(1-\beta)]}, 1\right)$$

**Total reward integration**:

$$R = \omega_1 S_{len} R_{task} + \omega_2 R_{Soft}$$

where $R_{task} = \omega_3 S_{acc} + \omega_4 S_{int} + \omega_5 S_{lang} + \omega_6 S_{fmt}$

### Loss & Training

- Visual encoder: frozen EgoVideo-V; Transformer with 4 layers and 8 heads
- Cognitive reasoning: Qwen2.5-VL-Instruct-7B backbone, GRPO training via the Swift framework
- 6× NVIDIA H20-SXM5-96GB GPUs
- Batch size 24, learning rate 3e-6, temperature 0.9, KL coefficient 0.08
- Reward weights: $\omega_1=0.90, \omega_2=0.10, \omega_3=0.85, \omega_4=0.05, \omega_5=0.05, \omega_6=0.05$
- Intention reward parameters: $\beta=0.8, \gamma=40$
- Total training: 500 steps, approximately 90 GPU hours

## Key Experimental Results

### Main Results

**Ego4D-v2 validation set (Edit Distance ED, lower is better)**:

| Method | LLM | Verb↓ | Noun↓ | Action↓ |
|------|-----|-------|-------|---------|
| AntGPT | LLaMA2-7B | 0.6728 | 0.6755 | 0.8931 |
| PALM | LLaMA2-7B | 0.7111 | 0.6465 | 0.8819 |
| EgoVideo | Vicuna-7B | 0.6576 | 0.6264 | 0.8619 |
| ICVL | LLaMA3-8B | 0.6516 | 0.6194 | 0.8570 |
| **INSIGHT** | **Qwen2.5-VL-7B** | **0.6643** | **0.6092** | **0.8463** |

**EPIC-Kitchens-55 / EGTEA Gaze+ (mAP, higher is better)**:

| Method | EK-55 ALL↑ | EK-55 FREQ↑ | EK-55 RARE↑ | EGTEA ALL↑ | EGTEA FREQ↑ | EGTEA RARE↑ |
|------|-----------|------------|------------|-----------|------------|------------|
| AntGPT | 40.1 | 58.8 | 31.9 | 80.2 | 84.8 | 72.9 |
| ICVL | 43.3 | 61.6 | 33.8 | 81.0 | 85.2 | 73.7 |
| **INSIGHT** | **45.2** | **62.4** | **36.0** | **81.7** | **85.9** | **74.4** |

### Ablation Study

| Configuration | Verb ED↓ | Noun ED↓ | Action ED↓ | Notes |
|------|----------|----------|------------|------|
| w/o HOI feature | 0.6719 | 0.6158 | 0.8595 | Removing HOI features degrades performance |
| w/o Semantic correction | 0.6716 | 0.6108 | 0.8587 | Removing co-occurrence correction |
| w/o Cognitive reasoning | 0.6750 | 0.6176 | 0.8612 | Largest drop; direct prediction without reasoning |
| w/o Intention | 0.6685 | 0.6104 | 0.8571 | Reasoning retained but intention supervision removed |
| **INSIGHT (full)** | **0.6643** | **0.6092** | **0.8463** | All modules combined yield the best performance |

### Key Findings

1. **Cognitive reasoning is the most critical component**: Removing structured reasoning (w/o Cognitive reasoning) causes the largest performance drop, with Action ED increasing from 0.8463 to 0.8612, demonstrating that explicit "think→reason→answer" reasoning is essential for long-term prediction.

2. **HOI features contribute most to noun prediction**: On Ego4D-v2, INSIGHT outperforms the strongest baseline ICVL on noun prediction by 1.02%, attributable to HOI-focused feature extraction capturing critical object manipulation information.

3. **Significant improvement on rare action categories**: On EK-55 RARE categories, INSIGHT surpasses ICVL by 6.5% (33.8→36.0), indicating that cognitive reasoning and intention alignment effectively reduce long-tail category confusion.

4. **Advantage of frozen encoders**: INSIGHT with a frozen visual encoder outperforms EgoVideo with a fine-tuned encoder, suggesting that the fine-tuned language model and cognitive reasoning module effectively compensate for visual ambiguity.

5. **Stable training convergence**: GRPO training converges within 500 steps, and the intention reward curve closely tracks the total reward, validating the alignment between intention supervision and task objectives.

## Highlights & Insights

- **Complementarity of the two-stage design**: Stage one strengthens visual representation quality (HOI + co-occurrence), while stage two introduces cognitive reasoning capability (GRPO + intention); ablation experiments confirm that both stages are indispensable.
- **Bio-inspired design of cognitive reasoning**: The think→reason→answer pipeline simulates human decision-making, shifting the model from passive sequence prediction to active intention inference — a significant paradigm shift in video understanding.
- **Elegant design of intention rewards**: GPT-4.1-generated pseudo intention labels serve as supervision signals, avoiding costly manual annotation, while sigmoid normalization ensures gradient-friendly reward signals.
- **Simplicity and effectiveness of co-occurrence matrices**: Simple statistical priors can substantially reduce semantically implausible predictions at minimal computational cost.

## Limitations & Future Work

- Pseudo ground-truth intentions rely on GPT-4.1 generation, introducing external model bias with no guarantee of generation quality.
- Freezing the visual encoder limits scene-specific adaptation; end-to-end fine-tuning may yield further improvements.
- HOI detection depends on the pretrained 100DOH detector, whose performance may degrade in non-kitchen scenarios.
- GRPO training of only 500 steps, while computationally efficient, may constrain the depth of learned reasoning.
- The verb-noun co-occurrence matrix is derived from training set statistics and may fail to cover novel combinations in the test set.
- Prediction capability for longer time horizons (e.g., $Z > 20$) remains unexplored.

## Related Work & Insights

- Compared to LLM-based methods such as AntGPT and PALM, the key innovation of INSIGHT lies in replacing SFT with RL to train the reasoning process.
- The structured reasoning design (think→reason→answer) draws inspiration from the success of DeepSeek-R1, adapted for the specific task requirements.
- The pipeline of HOI detection + SAM2 refinement can serve as a general-purpose feature enhancement scheme for other egocentric video tasks.
- The intention reward design pattern (LLM pseudo-labels + embedding similarity + sigmoid normalization) is transferable to other RL tasks requiring intermediate reasoning supervision.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying cognitive reasoning + GRPO to LTA is a novel combination, though each component is supported by prior work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three mainstream benchmarks, detailed ablation studies, training dynamics, and qualitative comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Architecture diagrams are clear and method descriptions are thorough.
- **Value**: ⭐⭐⭐⭐ — Establishes a new state of the art in LTA; the cognitive reasoning paradigm has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](../../ICLR2026/reinforcement_learning/unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)
- [\[ACL 2026\] Visually-Guided Policy Optimization for Multimodal Reasoning](../../ACL2026/reinforcement_learning/visually-guided_policy_optimization_for_multimodal_reasoning.md)
- [\[ICLR 2026\] LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](../../ICLR2026/reinforcement_learning/loongrl_rl_for_reasoning_long_contexts.md)
- [\[AAAI 2026\] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)
- [\[AAAI 2026\] Efficient Multiagent Planning via Shared Action Suggestions](efficient_multiagent_planning_via_shared_action_suggestions.md)

</div>

<!-- RELATED:END -->
