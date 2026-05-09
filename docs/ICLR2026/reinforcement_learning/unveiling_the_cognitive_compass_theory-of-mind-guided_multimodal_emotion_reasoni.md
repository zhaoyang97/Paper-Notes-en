---
title: >-
  [Paper Note] Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning
description: >-
  [Reinforcement Learning] This paper constructs HitEmotion, a hierarchical multimodal emotion understanding benchmark grounded in Theory of Mind (ToM), and proposes the TMPO framework, which leverages intermediate mental states as process-level supervision to enhance the emotion reasoning capabilities of MLLMs.
tags:
  - Reinforcement Learning
date: 2026-05-08
content_hash: 8c7c14d330c1a75a
---

# Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning

## Paper Information
- **Conference**: ICLR 2026
- **arXiv**: [2602.00971](https://arxiv.org/abs/2602.00971)
- **Code**: [https://HitEmotion.github.io/](https://HitEmotion.github.io/)
- **Area**: Multimodal Affective Computing / Theory of Mind / Reinforcement Learning / Large Models
- **Keywords**: Theory of Mind, Emotion Reasoning, MLLM, Hierarchical Benchmark, GRPO, Chain-of-Thought Optimization

## TL;DR
This paper constructs HitEmotion, a hierarchical multimodal emotion understanding benchmark grounded in Theory of Mind (ToM), and proposes the TMPO framework, which leverages intermediate mental states as process-level supervision to enhance the emotion reasoning capabilities of MLLMs.

## Background & Motivation

### Core Problem
Despite strong performance across various tasks, multimodal large language models (MLLMs) exhibit notable deficiencies in deep emotion understanding, stemming from three core issues:

**Lack of a unified cognitive framework**: Existing benchmarks provide only coarse-grained scores and cannot localize breakpoints in model reasoning ability.

**Unfaithful reasoning chains**: CoT reasoning appears coherent but is essentially template matching, lacking genuine tracking of mental states.

**Emotion hallucination**: Models produce distorted emotion attributions when faced with cross-modal conflicting cues.

### Limitations of Prior Work
- EQ-Bench and EmoBench cover only the text modality.
- EmoBench-M and EmotionHallucer are multimodal but feature scattered task designs not organized by cognitive depth.
- No existing benchmark simultaneously provides reasoning chain and rationale evaluation.

## Method

### HitEmotion Benchmark: Three-Level Cognitive Hierarchy

**Level 1 — Emotion Perception and Recognition (EPR)**: 10 tasks
- Maps multimodal signals to predefined emotion categories.
- Includes tasks such as facial expression recognition and multimodal sentiment recognition.

**Level 2 — Emotion Understanding and Analysis (EUA)**: 8 tasks
- Requires context-aware and relational reasoning.
- Includes tasks such as humor understanding, sarcasm detection, and multi-party dialogue emotion analysis.

**Level 3 — Emotion Cognition and Reasoning (ECR)**: 6 tasks
- Demands causal reasoning and second-order mental inference.
- Includes tasks such as emotion elicitation reasoning, emotion explanation, and irony comprehension.

In total: **24 tasks**, **20,114 instances**, covering both video and image modalities.

### TMPO Training Framework

#### Stage 1: ToM-Aligned Supervised Fine-Tuning (SFT)

A structured reasoning template is employed, wrapping intermediate reasoning steps within `<think></think>` tags and the final output within `<answer></answer>` tags:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{((\mathcal{P},T,A,V), y)} [\log \pi_\theta(y | \mathcal{P}, T, A, V)]$$

Gold-standard reasoning chains are constructed via a four-step pipeline: LLM generation → filtering → augmentation → correction.

#### Stage 2: GRPO-Based ToM Preference Optimization

For each input, $N$ candidate outputs are sampled and evaluated through a multi-dimensional reward function:

$$R(y) = \mu_1 R_{\text{structure}} + \mu_2 R_{\text{content}} + \mu_3 R_{\text{process}} + \mu_4 R_{\text{consistency}}$$

Four reward components:
- **Structure Reward**: Correct ordering of reasoning steps.
- **Content Reward**: Correctness of the final answer.
- **Process Reward**: Use of domain-specific language.
- **Consistency Reward**: Penalty for logical and factual inconsistencies.

GRPO optimization objective:
$$\max_{\pi_\theta} \mathbb{E}_{y_i \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(y_i)}{\pi_{\text{old}}(y_i)} A_i \right] - \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

### ToM-Style Prompting Mechanism

Prompt designs are tailored to three levels of cognitive complexity:
- **Level 1**: First-order mental state attribution — integrating observable signals to infer emotions.
- **Level 2**: Relational and contextual mental modeling — associating emotions with specific entities or communicative intentions.
- **Level 3**: Causal attribution and second-order reasoning — explaining the origins of emotions and their social interpretations.

## Experiments

### Baseline Model Evaluation (EPR Level 1)

| Model | FESD | ISA | MESA | MER | MSA | OSA | SIA |
|------|------|-----|------|-----|-----|-----|-----|
| VideoLLaMA3-7B | 61.78 | 46.85 | 21.60 | 52.18 | 64.62 | 67.89 | 35.20 |
| LLaVA-One-Vision-7B | 63.44 | 49.19 | 17.05 | 39.50 | 65.40 | 63.00 | 27.00 |

### Key Findings

1. **State-of-the-art models exhibit inconsistent performance on higher-level cognitive tasks**: Even the strongest closed-source models show significant deficiencies at the ECR level.
2. **ToM reasoning chains, used alone as a prompting strategy, substantially improve closed-source model performance**: This validates the effectiveness of ToM as a reasoning scaffold.
3. **TMPO optimization yields consistent gains**: It outperforms baselines across all evaluation tasks, generating reasoning chains that are markedly superior in faithfulness and logical coherence.
4. **From "general emergence" to "domain acquisition"**: TMPO transforms reasoning ability from a general property into a cognitively specialized skill.

## Highlights & Insights
1. **The first evaluation framework to unify psychological theory with MLLM reasoning processes and rationale generation.**
2. **Elegant ToM prompting mechanism design**: Three levels of cognitive hierarchy correspond to three reasoning templates of increasing depth.
3. **Innovative combination of GRPO and process-level rewards**: Intermediate mental states serve simultaneously as supervision signals and reward sources.
4. **Scale**: A comprehensive benchmark spanning 24 datasets and 20K+ instances.

## Limitations & Future Work
1. Gold-standard reasoning chains rely on LLM generation, potentially introducing inherent LLM biases.
2. The benchmark is built upon reconstructed existing datasets, whose original annotation quality varies.
3. GRPO training incurs considerable computational cost.
4. Evaluation is primarily conducted in single-turn QA settings; emotion reasoning in multi-turn interactions remains underexplored.

## Related Work & Insights
- **Multimodal Affective Computing**: Fusion strategies such as SALV and PAD have evolved from early/late fusion to intermediate interaction approaches.
- **Affective Intelligence Evaluation**: A progression from EQ-Bench → EmoBench-M → EmotionHallucer.
- **ToM Reasoning**: Works from ToMBench to MMToM-QA reveal ToM deficiencies in MLLMs.
- **Reasoning Optimization**: The success of DeepSeek-R1's GRPO approach in text-based reasoning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Deep integration of the ToM cognitive framework with MLLM evaluation and training.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 24 datasets.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem formulation with well-motivated methodology.
- **Value**: ⭐⭐⭐⭐ — Provides both an evaluation toolkit and an optimization method.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[ICLR 2026\] Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)
- [\[ICLR 2026\] Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning](shop-r1_rewarding_llms_to_simulate_human_behavior_in_online_shopping_via_reinfor.md)

<!-- RELATED:END -->
