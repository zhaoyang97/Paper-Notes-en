---
title: >-
  [Paper Note] STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems
description: >-
  [ACL 2026][Reinforcement Learning][Empathetic dialogue] This paper proposes the STRIDE-ED framework, which achieves state-of-the-art performance in empathetic dialogue across multiple open-source LLMs by constructing a comprehensive empathy strategy system covering positive/neutral/negative emotions, designing task-aligned multi-stage cognitive CoT reasoning, and combining strategy-aware data refinement with a two-stage SFT+PPO training paradigm. The framework attains an emotion accuracy of 57.25% and BLEU-4 of 4.67.
tags:
  - ACL 2026
  - Reinforcement Learning
  - Empathetic dialogue
  - strategy-grounded reasoning
  - chain-of-thought
  - multi-objective reinforcement learning
  - data refinement
date: 2026-05-08
content_hash: 0f3cb05899144375
---

# STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems

**Conference**: ACL 2026
**arXiv**: [2604.07100](https://arxiv.org/abs/2604.07100)
**Code**: [https://github.com/jicoder-nwpu/STRIDE-ED](https://github.com/jicoder-nwpu/STRIDE-ED)
**Area**: Reinforcement Learning
**Keywords**: Empathetic dialogue, strategy-grounded reasoning, chain-of-thought, multi-objective reinforcement learning, data refinement

## TL;DR
This paper proposes the STRIDE-ED framework, which achieves state-of-the-art performance in empathetic dialogue across multiple open-source LLMs by constructing a comprehensive empathy strategy system covering positive/neutral/negative emotions, designing task-aligned multi-stage cognitive CoT reasoning, and combining strategy-aware data refinement with a two-stage SFT+PPO training paradigm. The framework attains an emotion accuracy of 57.25% and BLEU-4 of 4.67.

## Background & Motivation

**Background**: Empathetic dialogue is a core capability for social AI, requiring models not only to recognize user emotions but also to generate strategically appropriate and context-sensitive responses. Early work enhanced emotional understanding through external commonsense knowledge graphs (e.g., ATOMIC), while recent research has shifted toward leveraging LLM-based CoT prompting to explicitly model the reasoning process.

**Limitations of Prior Work**: (1) *Incomplete strategy coverage*—existing strategy taxonomies (e.g., the 8-strategy system of Liu et al. 2021) target only negative-emotion counseling scenarios, neglecting positive and neutral emotions; (2) *Lack of task-aligned reasoning*—while CoT methods impose structural form, the reasoning steps remain superficial and lack explicit alignment with the empathetic decision-making process; (3) *Insufficient strategy-aware supervision*—training data lacks high-quality annotations aligned with strategy reasoning.

**Key Challenge**: Empathetic dialogue is inherently a multi-stage cognitive decision process (context understanding → emotion recognition → strategy selection → action reasoning → response generation), yet existing methods either implicitly skip intermediate steps or substitute task-specific reasoning with generic CoT without task specialization.

**Goal**: (1) Construct an empathy strategy system covering the full emotional spectrum; (2) Design a multi-stage reasoning framework aligned with the cognitive process; (3) Establish a strategy-aware data refinement pipeline and training paradigm.

**Key Insight**: Drawing from cognitive psychology, empathetic response generation is modeled as a progressive cognitive chain—"context summarization → emotion recognition → strategy inference → action reasoning → response generation"—where each step produces explicit outputs subject to explicit constraints.

**Core Idea**: A comprehensive strategy system supports structured reasoning; strategy-aware data refinement ensures training quality; multi-objective RL aligns three dimensions: emotion, strategy, and format.

## Method

### Overall Architecture
STRIDE-ED takes dialogue history $\mathcal{C}$ as input and sequentially generates five intermediate outputs: (1) context summary *sum*; (2) emotional state *e*; (3) empathy strategy *stra*; (4) strategy execution actions *acts*; and (5) final response $u_t$. The entire pipeline is guided by structured tags (`<Context>`, `<Emotion>`, `<Strategy>`, etc.) that direct the model's internal reasoning. Training proceeds in two stages: SFT on refined data followed by PPO-based reinforcement learning to align emotion, strategy, and format.

### Key Designs

1. **Comprehensive Empathy Strategy System**

    - *Function*: Provides strategy guidance for three emotional categories—positive, neutral, and negative—spanning low- to high-order cognition.
    - *Mechanism*: Extends the 8-strategy taxonomy of Liu et al. (2021) by decomposing the original "questioning" strategy into three cognitively distinct "exploratory" strategies and assigning each strategy a three-level difficulty score (I–III) reflecting cognitive complexity. The system covers multiple dimensions including emotional validation, active listening, cognitive restructuring, and action guidance.
    - *Design Motivation*: Existing strategy systems are tailored only to negative-emotion counseling scenarios, whereas real dialogues involving positive emotions (e.g., sharing good news) equally require empathetic responses but demand entirely different strategies.

2. **Strategy-Aware Data Refinement Pipeline**

    - *Function*: Filters high-quality, strategy-balanced training samples from large-scale automatically annotated data.
    - *Mechanism*: A three-step procedure—(a) DeepSeek-R1 is used to automatically annotate strategy types and reasoning trajectories on the EMPATHETICDIALOGUES dataset; (b) three independent LLM evaluators (DeepSeek-R1, Qwen3, Llama-3.1) score each sample, with scores aggregated via Spearman correlation-weighted fusion to select the top 12k high-quality samples; (c) sampling is performed according to a joint distribution of strategy frequency × difficulty weight, yielding the final 5k refined training set ED-CSA-5k.
    - *Design Motivation*: Data annotated directly by LLMs varies widely in quality and exhibits severe strategy imbalance (simple strategies dominate). Multi-evaluator weighting combined with strategy-aware sampling jointly optimizes data quality and distribution.

3. **Two-Stage Training: SFT + Multi-Objective PPO**

    - *Function*: First establishes reasoning capability, then aligns along three dimensions: emotion, strategy, and format.
    - *Mechanism*: The SFT stage mixes refined data (with strategy annotations) and remaining data (without strategy annotations), enabling the model to learn both strategy reasoning and general response generation. The PPO stage employs a composite reward $R = r_{\text{format}} \cdot (1 + r_{\text{emotion}} + r_{\text{strategy}})$, where the format reward acts as a gating factor (incorrect format zeroes the overall reward) and the emotion and strategy rewards are additive.
    - *Design Motivation*: SFT can only imitate demonstration data and cannot handle out-of-distribution strategy selection; PPO uses explicit three-dimensional reward signals to guide the model to optimize emotion and strategy alignment under the prerequisite of format correctness.

### Loss & Training
The SFT stage uses standard negative log-likelihood loss. The PPO stage applies proximal policy optimization with a composite reward combining three binary rewards multiplicatively and additively (format × (1 + emotion + strategy)). Initial learning rate is 1e-4, batch size 16, and sequence length 2048.

## Key Experimental Results

### Main Results (EMPATHETICDIALOGUES Dataset)

| Model | B-1 | B-4 | Acc_emo | D-2 | PPL |
|-------|-----|-----|---------|-----|-----|
| MoEL | 18.02 | 2.73 | 31.02 | 1.76 | 36.81 |
| CAB | 20.23 | 3.01 | 40.52 | 2.95 | 35.06 |
| ReflectDiffu | 23.59 | 3.62 | 48.76 | 4.35 | 24.56 |
| **STRIDE-ED** | **24.54** | **4.67** | **57.25** | **13.63** | **10.50** |
| Gain vs. ReflectDiffu | ↑4.0% | ↑29.0% | ↑17.4% | ↑213% | ↓57.2% |

### Ablation Study

| Configuration | B-1 | Acc_emo | D-2 | PPL | Note |
|---------------|-----|---------|-----|-----|------|
| Full | 24.66 | 57.57 | 13.68 | 9.26 | Complete model |
| w/o emotion | 23.58 | — | 13.66 | 10.47 | Remove emotion reasoning |
| w/o sum | 22.91 | 54.14 | 13.42 | 7.78 | Remove context summarization |
| w/o strategy | 22.44 | 54.58 | 15.09 | 6.98 | Remove strategy reasoning → diversity increases but uncontrolled |
| w/o CoT | 22.55 | — | 13.26 | 8.64 | Remove structured reasoning |
| w/o refinement+sampling | 22.22 | 55.93 | 15.25 | 11.86 | Largest performance drop |
| w/o PPO | 23.52 | 54.48 | 14.76 | 2.03 | Extremely low PPL but poor emotion alignment |

### Key Findings
- Removing strategy reasoning paradoxically increases diversity (D-2: 13.68 → 15.09) while degrading B-1 and emotion accuracy, indicating that unconstrained models generate more random but less controllable outputs.
- Data refinement and strategy-aware sampling constitute the most critical component; removing them causes across-the-board performance degradation.
- PPO exerts the greatest impact on PPL (2.03 → 9.26), suggesting that RL-stage constraints on format, emotion, and strategy trade some fluency for alignment.
- The framework generalizes across multiple open-source LLMs (Qwen3-0.6B/4B, LLaMA3.2-3B, etc.).

## Highlights & Insights
- The **cognitive psychology-driven multi-stage reasoning design** is natural and compelling: context summarization → emotion recognition → strategy selection → action reasoning → response generation, with each step providing a clearly defined function and interpretable output.
- Strategy-aware sampling elegantly addresses data imbalance—joint sampling over difficulty × frequency distributions ensures that high-order, harder strategies receive adequate training coverage.
- The "gating" design of the composite reward function is worth adopting: format correctness is a prerequisite, without which emotion and strategy rewards are meaningless.

## Limitations & Future Work
- The strategy system is constructed based on analysis of the EMPATHETICDIALOGUES dataset and may not transfer to other cultural or domain contexts (e.g., medical counseling, crisis intervention).
- Automatic annotation relies on the reasoning quality of DeepSeek-R1; annotation errors propagate through the refinement pipeline.
- Human evaluation is conducted on only 1,000 dialogue turns, which is relatively limited in scale.
- Strategy coherence across multi-turn dialogues is not explored—the current approach performs independent per-turn reasoning without session-level strategy planning.

## Related Work & Insights
- **vs. ReflectDiffu (Yuan et al. 2025)**: ReflectDiffu applies diffusion models for empathetic response generation with a focus on generation quality; STRIDE-ED focuses on strategy-driven reasoning and achieves a 17.4% improvement in emotion accuracy.
- **vs. CAB (Gao et al. 2023)**: CAB models cognitive appraisal and behavioral tendencies but lacks a comprehensive strategy system and data refinement; STRIDE-ED provides more systematic strategy coverage and data quality control.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of strategy system extension, multi-stage CoT, and strategy-aware data refinement is relatively novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Automatic and human evaluations, comprehensive ablation studies, and multi-model generalization verification.
- **Writing Quality**: ⭐⭐⭐⭐ The architecture diagram is clear and the method description is detailed, though the notation is somewhat dense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs](../../CVPR2026/reinforcement_learning/see_it_say_it_sorted_an_iterative_training-free_framework_for_visually-grounded_.md)
- [\[ACL 2026\] Frame of Reference: Addressing the Challenges of Common Ground Representation in Dialogue](frame_of_reference_addressing_the_challenges_of_common_ground_representation_in_.md)
- [\[ACL 2026\] Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF](reverse_constitutional_ai_a_framework_for_controllable_toxic_data_generation_via.md)
- [\[ACL 2026\] Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification](right_at_my_level_a_unified_multilingual_framework_for_proficiency-aware_text_si.md)
- [\[AAAI 2026\] BAMAS: Structuring Budget-Aware Multi-Agent Systems](../../AAAI2026/reinforcement_learning/bamas_structuring_budget-aware_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
