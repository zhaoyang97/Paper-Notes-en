---
title: >-
  [Paper Note] STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems
description: >-
  [ACL 2026][Dialogue Systems][Empathetic Dialogue] This paper proposes the STRIDE-ED framework, which achieves SOTA in empathetic dialogue across multiple open-source LLMs with an emotion accuracy of 57.25% and BLEU-4 of…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "Empathetic Dialogue"
  - "Strategy-Grounded Reasoning"
  - "Chain of Thought"
  - "Multi-Objective Reinforcement Learning"
  - "Data Refinement"
date: 2026-05-08
content_hash: 2095a0eb057b8f87
---

# STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems

**Conference**: ACL 2026  
**arXiv**: [2604.07100](https://arxiv.org/abs/2604.07100)  
**Code**: [https://github.com/jicoder-nwpu/STRIDE-ED](https://github.com/jicoder-nwpu/STRIDE-ED)  
**Area**: Reinforcement Learning  
**Keywords**: Empathetic Dialogue, Strategy-Grounded Reasoning, Chain of Thought, Multi-Objective Reinforcement Learning, Data Refinement

## TL;DR
This paper proposes the STRIDE-ED framework, which achieves SOTA in empathetic dialogue across multiple open-source LLMs with an emotion accuracy of 57.25% and BLEU-4 of 4.67. This is achieved by constructing a comprehensive empathy strategy system covering positive/neutral/negative emotions, designing task-aligned multi-stage cognitive CoT reasoning, and combining strategy-aware data refinement with a two-stage SFT+PPO training paradigm.

## Background & Motivation

**Background**: Empathetic dialogue is a core capability of social AI, requiring models not only to identify user emotions but also to provide strategic and context-sensitive responses. Early works enhanced emotional understanding through external commonsense knowledge graphs (e.g., ATOMIC), while recent studies have shifted toward utilizing CoT prompting with LLMs to explicitly model the reasoning process.

**Limitations of Prior Work**: (1) Incomplete strategy coverage—existing systems (such as the 8 strategies from Liu et al. 2021) focus only on counseling scenarios for negative emotions, overlooking positive and neutral states; (2) Lack of task alignment in reasoning—although CoT methods are structurally organized, reasoning steps remain superficial and lack explicit alignment with the empathetic decision-making process; (3) Insufficient strategy-aware supervision—training data lacks high-quality annotations aligned with strategy reasoning.

**Key Challenge**: Empathetic dialogue is essentially a multi-stage cognitive decision process (Contextual Understanding → Emotion Recognition → Strategy Selection → Action Reasoning → Response Generation). However, existing methods either implicitly skip intermediate steps or replace them with generic CoT without task specialization.

**Goal**: (1) Construct an empathetic strategy system covering the full emotional spectrum; (2) Design a multi-stage reasoning framework aligned with cognitive processes; (3) Establish a strategy-aware data refinement pipeline and training paradigm.

**Key Insight**: Drawing from cognitive psychology, empathetic response generation is modeled as a progressive cognitive chain of "Context Summary → Emotion Recognition → Strategy Inference → Action Reasoning → Response Generation," where each step has explicit outputs and constraints.

**Core Idea**: Utilizing a comprehensive strategy system to support structured reasoning, strategy-aware data refinement to ensure training quality, and multi-objective RL to align emotion, strategy, and format dimensions.

## Method

### Overall Architecture
STRIDE-ED takes dialogue history $\mathcal{C}$ as input and sequentially generates five intermediate results: (1) Context summary sum; (2) Emotion state e; (3) Empathy strategy stra; (4) Strategy execution actions acts; and (5) Final response $u_t$. The entire process guides internal reasoning via structured tags (e.g., `<Context>`, `<Emotion>`, `<Strategy>`). Training is divided into two stages: SFT using refined data, followed by PPO reinforcement learning to align emotion, strategy, and format.

### Key Designs

1. **Comprehensive Empathy Strategy System**:

    - **Function**: Provides strategy guidance for positive, neutral, and negative emotions, covering low-level to high-level cognition.
    - **Mechanism**: Based on the 8 strategies of Liu et al. (2021), the original "Questioning" strategy is subdivided into three "Exploration" strategies at different cognitive levels. A three-level difficulty score (I-III) is assigned to each strategy to reflect cognitive complexity. The system covers dimensions such as emotional validation, active listening, cognitive restructuring, and guided action.
    - **Design Motivation**: Existing strategy systems only apply to negative emotion counseling scenarios, whereas positive emotions in real dialogues (e.g., sharing good news) also require empathetic responses but with entirely different strategies.

2. **Strategy-Aware Data Refinement Pipeline**:

    - **Function**: Filters a high-quality training set with balanced strategy distributions from automatically annotated large-scale data.
    - **Mechanism**: A three-step process—(a) Utilizing DeepSeek-R1 to automatically annotate strategy types and reasoning trajectories on the EMPATHETICDIALOGUES dataset; (b) Scoring via three independent LLM evaluators (DeepSeek-R1, Qwen3, Llama-3.1), aggregated by Spearman correlation weighting to select the top 12k high-quality samples; (c) Sampling based on the joint distribution of strategy frequency × difficulty weights to obtain the final 5k refined training set, ED-CSA-5k.
    - **Design Motivation**: The quality of data directly annotated by LLMs is inconsistent, and strategy distributions are severely imbalanced (dominated by simple strategies). Multi-evaluator weighting and strategy-aware sampling ensure dual optimization of data quality and distribution.

3. **Two-stage Training: SFT + Multi-objective PPO**:

    - **Function**: Establishes reasoning capabilities first, then aligns emotion, strategy, and format dimensions.
    - **Mechanism**: The SFT stage mixes refined data (with strategy annotations) and remaining data (without annotations) to teach the model both strategy reasoning and general response generation. The PPO stage designs a compound reward $R = r_{\text{format}} \cdot (1 + r_{\text{emotion}} + r_{\text{strategy}})$, where the format reward acts as a gating factor (total reward is 0 if format is incorrect), and emotion and strategy rewards are additive.
    - **Design Motivation**: SFT can only mimic demonstration data and cannot handle out-of-distribution strategy selection; PPO guides the model to optimize emotion and strategy alignment under the premise of correct formatting via explicit three-dimensional reward signals.

### Loss & Training
The SFT stage uses standard Negative Log-Likelihood loss. The PPO stage utilizes Proximal Policy Optimization with a reward function combining three binary rewards (format × (1 + emotion + strategy)). Initial learning rate is 1e-4, batch size is 16, and sequence length is 2048.

## Key Experimental Results

### Main Results (EMPATHETICDIALOGUES Dataset)

| Model | B-1 | B-4 | Acc_emo | D-2 | PPL |
|------|-----|-----|---------|-----|-----|
| MoEL | 18.02 | 2.73 | 31.02 | 1.76 | 36.81 |
| CAB | 20.23 | 3.01 | 40.52 | 2.95 | 35.06 |
| ReflectDiffu | 23.59 | 3.62 | 48.76 | 4.35 | 24.56 |
| **STRIDE-ED** | **24.54** | **4.67** | **57.25** | **13.63** | **10.50** |
| **Gain vs ReflectDiffu** | ↑4.0% | ↑29.0% | ↑17.4% | ↑213% | ↓57.2% |

### Ablation Study

| Configuration | B-1 | Acc_emo | D-2 | PPL | Description |
|------|-----|---------|-----|-----|------|
| Full | 24.66 | 57.57 | 13.68 | 9.26 | Full Model |
| w/o emotion | 23.58 | - | 13.66 | 10.47 | Remove emotion reasoning |
| w/o sum | 22.91 | 54.14 | 13.42 | 7.78 | Remove context summary |
| w/o strategy | 22.44 | 54.58 | 15.09 | 6.98 | Remove strategy reasoning → diversity increases but becomes uncontrolled |
| w/o CoT | 22.55 | - | 13.26 | 8.64 | Remove structured reasoning |
| w/o refinement+sampling | 22.22 | 55.93 | 15.25 | 11.86 | Largest drop in performance |
| w/o PPO | 23.52 | 54.48 | 14.76 | 2.03 | Very low PPL but poor emotion alignment |

### Key Findings
- Diversity increases after removing strategy reasoning (D-2 from 13.68 → 15.09), but B-1 and emotional accuracy decrease—indicating that models without strategy constraints generate more random but less controllable responses.
- Data refinement and sampling are the most critical components; all metrics decline significantly without them.
- PPO has the largest impact on PPL (2.03 → 9.26), suggesting that the RL stage sacrifices some fluency for alignment under format/emotion/strategy constraints.
- The framework generalizes across multiple open-source LLMs (Qwen3-0.6B/4B, LLama3.2-3B, etc.).

## Highlights & Insights
- The **multi-stage reasoning design driven by cognitive psychology** is natural and persuasive: Context Summary → Emotion Recognition → Strategy Selection → Action Reasoning → Response Generation, with each step providing clear functions and interpreable outputs.
- The design of strategy-aware sampling cleverly addresses data imbalance—by sampling through the joint distribution of difficulty × frequency, complex high-level strategies receive sufficient training.
- The "gating" design of the compound reward function is noteworthy—correct formatting is a prerequisite, otherwise, emotion and strategy rewards are rendered meaningless.

## Limitations & Future Work
- The strategy system is constructed based on analysis of the EMPATHETICDIALOGUES dataset and may not apply to other cultures or scenarios (e.g., medical consultation, crisis intervention).
- Automated annotation relies on the reasoning quality of DeepSeek-R1; annotation errors may propagate through the refinement process.
- Human evaluation was conducted on only 1000 dialogue turns, which is limited in scale.
- Strategy coherence in multi-turn dialogues has not been explored—currently, each turn involves independent reasoning without considering session-level strategy planning.

## Related Work & Insights
- **vs ReflectDiffu (Yuan et al. 2025)**: ReflectDiffu uses diffusion models for empathetic responses focusing on generation quality; STRIDE-ED focuses on strategy-driven reasoning, achieving a 17.4% improvement in emotional accuracy.
- **vs CAB (Gao et al. 2023)**: CAB models cognitive appraisal and behavioral tendencies but lacks a comprehensive strategy system and data refinement; STRIDE-ED's strategy coverage and data quality control are more systematic.

## Rating
- Novelty: ⭐⭐⭐⭐ Combination of strategy system expansion + multi-stage CoT + strategy-aware data refinement is relatively novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Automated + human evaluation, comprehensive ablation, and multi-model generalization.
- Writing Quality: ⭐⭐⭐⭐ Framework diagrams are clear and methodology is detailed, though math notations are slightly dense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)
- [\[ACL 2026\] Metro: Towards Strategy Induction from Expert Dialogue Transcripts for Non-collaborative Dialogues](metro_towards_strategy_induction_from_expert_dialogue_transcripts_for_non-collab.md)
- [\[ACL 2026\] Reasoning Gets Harder for LLMs Inside A Dialogue](reasoning_gets_harder_for_llms_inside_a_dialogue.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)
- [\[ACL 2026\] APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI](apex-mem_agentic_semi-structured_memory_with_temporal_reasoning_for_long-term_co.md)

</div>

<!-- RELATED:END -->
