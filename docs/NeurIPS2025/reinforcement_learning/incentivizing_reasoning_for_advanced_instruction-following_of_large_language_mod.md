---
title: >-
  [Paper Note] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models
description: >-
  [NeurIPS 2025][Reinforcement Learning][Instruction Following] This paper proposes RAIF, which employs RL with rule-centric rewards to cultivate deep reasoning capabilities in LLMs for complex instructions containing And/Chain/Selection/Nested compositional constraints. A key finding is that vanilla CoT is detrimental to instruction following, as LLMs tend to shallowly paraphrase instructions rather than analyze constraint structures. RAIF addresses this through superior CoT enforcement (sample-level contrastive filtering of ineffective reasoning) and behavior cloning to control distribution shift. A 1.5B model trained with RAIF matches 8B-level performance, achieving an average improvement of 11.74% across 7 benchmarks.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Instruction Following
  - Reasoning Incentivization
  - Rule-Centric Reward
  - CoT
  - GRPO
date: 2026-05-08
content_hash: b88919c10d1f37c2
---

# Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.01413](https://arxiv.org/abs/2506.01413)
**Code**: [https://github.com/yuleiqin/RAIF](https://github.com/yuleiqin/RAIF)
**Area**: Reinforcement Learning
**Keywords**: Instruction Following, Reinforcement Learning, Reasoning Incentivization, Rule-Centric Reward, CoT, GRPO

## TL;DR
This paper proposes RAIF, which employs RL with rule-centric rewards to cultivate deep reasoning capabilities in LLMs for complex instructions containing And/Chain/Selection/Nested compositional constraints. A key finding is that vanilla CoT is detrimental to instruction following, as LLMs tend to shallowly paraphrase instructions rather than analyze constraint structures. RAIF addresses this through superior CoT enforcement (sample-level contrastive filtering of ineffective reasoning) and behavior cloning to control distribution shift. A 1.5B model trained with RAIF matches 8B-level performance, achieving an average improvement of 11.74% across 7 benchmarks.

## Background & Motivation

**State of the Field**: Instruction-following capability is fundamental to the practical utility of LLMs. Existing benchmarks (IFEval, ComplexBench, CELLO, etc.) evaluate instructions ranging from simple to complex, yet LLMs continue to perform poorly on multi-constraint compositional instructions with And/Chain/Selection/Nested structures.

**Limitations of Prior Work**: (a) SFT-based methods tend to overfit to constraint types seen during training and fail to generalize to out-of-distribution constraints; (b) template-guided reasoning requires pre-enumeration of decomposition templates and does not scale; (c) while CoT intuitively seems beneficial, experiments reveal that **vanilla CoT actually degrades performance**—Qwen2.5-1.5B achieves 50.61% under I/O but only 38.81% under CoT (−11.79%).

**Root Cause**: CoT is effective for mathematical problems because step-by-step derivation is a prerequisite for obtaining the answer. In instruction following, however, LLMs' "reasoning" often amounts to **shallow paraphrasing** of the instructions without analyzing the hierarchical relationships and dependencies among constraints—such spurious reasoning introduces noise rather than signal.

**Paper Goals**: Two problems are addressed: (a) how to synthesize diverse training data for complex instructions with verifiable reward sources; and (b) how to use RL to cultivate **genuinely beneficial** deep reasoning for instruction following, as opposed to shallow paraphrasing.

**Starting Point**: The success of R1/o1 demonstrates that RL can incentivize mathematical reasoning. However, instruction following differs from mathematics in that: (a) there is no single correct answer; (b) reasoning is not a prerequisite for producing a response; and (c) semantic quality also matters. These differences necessitate task-specific design.

**Core Idea**: Employ GRPO with rule-centric rewards (code verification + LLM-based judgment), sample-level contrastive filtering of CoT quality (retaining only samples where reasoning genuinely outperforms non-reasoning), and behavior cloning to prevent distribution shift—thereby cultivating deep reasoning that is truly effective for instruction following.

## Method

### Overall Architecture
The framework consists of three stages: (1) **LLM-based instruction evolution**: starting from seed instructions, LLMs evolve and expand instructions according to constraint types and compositional structures, while generating both code-verifiable and LLM-judge verification criteria; (2) **SFT cold-start**: behavior cloning using expert responses; (3) **RL-based reasoning incentivization**: GRPO with rule-centric rewards, superior CoT enforcement, and behavior cloning regularization.

### Key Designs

1. **Rule-Centric Reward**:

    - **Function**: Decomposes complex instructions into $C$ atomic constraints $\{c_j\}$ and verifies each constraint individually.
    - **Mechanism**: $R^i = R^i_{format} + R^i_{accuracy}$. The format reward checks the `<think>...</think><answer>...</answer>` structure (+1/−1). The accuracy reward is assigned at three levels: full satisfaction yields +2, partial satisfaction is rewarded proportionally, and complete failure yields −2. Lexical, numerical, and formatting constraints are verified via Python code, while semantic and stylistic constraints are verified via LLM-Judge with boolean outputs.
    - **Design Motivation**: Unlike mathematics with a unique correct answer, correctness in instruction following requires simultaneous satisfaction of multiple constraints. The tiered reward encourages maximizing constraint satisfaction and penalizes complete failure.

2. **Superior CoT Enforcement (Sample-Level Contrastive Filtering)**:

    - **Function**: Filters out samples where reasoning is counterproductive, retaining only training signals where reasoning demonstrably improves outcomes.
    - **Mechanism**: For each query $x$, the current policy generates both reasoning-enabled responses $\{y^i\}$ and reasoning-free responses $\{\hat{y}^i\}$ (with empty CoT). If all reasoning-enabled responses receive lower rewards than their reasoning-free counterparts, the model's reasoning capacity is deemed insufficient for that sample, and the sample is skipped during training.
    - **Design Motivation**: Prevents "lengthy but ineffective reasoning" from receiving training signal. In mathematics, reasoning is a necessary condition and thus requires no such filtering. In instruction following, reasoning is optional, and shallow reasoning is actively harmful.

3. **Behavior Cloning for Distribution Shift Control**:

    - **Function**: Augments the RL objective with an SFT loss $\mathcal{J}_{SFT} = -\log \pi_\theta(\tilde{y}|x)$ using expert responses to prevent the policy from drifting excessively.
    - **Design Motivation**: Optimizing solely for constraint satisfaction can lead to responses that satisfy constraints but suffer semantic degradation (disfluency, incoherence). SFT regularization explicitly preserves semantic quality, providing a stronger signal than KL-divergence penalty alone.

4. **LLM-Based Instruction Evolution**:

    - **Function**: Evolves and diversifies instructions from WildChat/Alpaca seed instructions according to CFBench constraint taxonomy and ComplexBench compositional structures.
    - **Mechanism**: Separate template pools are maintained for code-verifiable constraints (lexical/numerical/formatting) and LLM-verifiable constraints (semantic/stylistic). Random combination, validity checking, and exclusion of 7 problematic categories via LLM screening are applied.
    - **Design Motivation**: Unlike Tülu3, which covers only IFEval-style code-verifiable constraints, RAIF jointly covers semantic constraints.

## Key Experimental Results

### Main Results (Average across 7 Benchmarks)

| Model | Method | IFEval | ComplexBench | CFBench | 7-Bench Avg |
|-------|--------|--------|--------------|---------|-------------|
| Qwen2.5-1.5B | I/O | 45.28 | 50.97 | 36.00 | 50.61 |
| Qwen2.5-1.5B | CoT | 28.65 | 32.94 | 22.00 | **38.81 (−11.79%)** |
| Qwen2.5-1.5B | **RAIF** | — | — | — | **62.35 (+11.74%)** |
| Qwen2.5-7B | I/O | — | — | — | Baseline |
| Qwen2.5-7B | **RAIF** | — | — | — | Significant improvement |

**1.5B RAIF ≈ 8B Baseline**: RAIF-trained 1.5B models match the performance of 8B-scale models across all 7 benchmarks.

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| w/o Superior CoT Enforcement | Performance drops—ineffective reasoning samples interfere with training |
| w/o Behavior Cloning | Semantic quality degrades; constraint satisfaction rate becomes unstable |
| w/o Math Data Mixture | Weaker reasoning foundation |
| Code-only reward (w/o LLM-Judge) | Covers only lexical constraints; semantic constraints cannot be learned |

### Key Findings
- **Vanilla CoT is harmful to instruction following**: Qwen2.5-1.5B drops by 11.79%. LLMs tend to shallowly paraphrase instructions rather than analyze constraint structures—a finding diametrically opposed to the situation in mathematical reasoning.
- **1.5B + RAIF ≈ 8B I/O**: RL-based reasoning incentivization enables smaller models to compensate for parameter count through deeper reasoning.
- **OOD constraint generalization**: RAIF generalizes effectively to unseen constraint types, indicating that the model learns a general capability to analyze constraint structures rather than memorizing specific constraint patterns.
- **Warm-start (R1-series) is easier to train than cold-start**: Models with pre-existing reasoning habits (DeepSeek-R1-Distill) serve as more effective initialization points.
- **Mixing mathematical data strengthens the reasoning foundation**: Incorporating DeepScaleR math data yields a stronger reasoning baseline.

## Highlights & Insights
- **"CoT is harmful to instruction following" is a significant finding**: It challenges the assumption that CoT is universally beneficial—the quality of LLM reasoning varies substantially, and shallow reasoning is worse than no reasoning. This underscores the necessity of validating CoT effectiveness before applying it to new domains.
- **The contrastive filtering design of Superior CoT Enforcement**: Simultaneously generating reasoning-enabled and reasoning-free responses during training for comparison constitutes a generalizable "reasoning quality assurance" mechanism transferable to other CoT-dependent tasks.
- **Rule-centric rewards vs. pure reward models**: Decomposing constraint verification into code-verifiable and LLM-verifiable categories yields more precise and interpretable rewards than scoring with a monolithic reward model.

## Limitations & Future Work
- LLM-Judge verification of semantic constraints is noisy—the judge model itself may produce incorrect assessments.
- Instruction evolution relies on manually designed constraint template pools; new constraint types require manual extension.
- Training costs (GRPO + multiple rollouts + contrastive filtering) are substantially higher than SFT.
- Effectiveness in multi-turn, multilingual, and multimodal instruction-following scenarios has not been validated.

## Related Work & Insights
- **vs. Tülu 3**: Tülu3 applies vanilla PPO with IFEval-style constraints only; RAIF covers semantic constraints, incorporates superior CoT enforcement and behavior cloning, and achieves stronger generalization.
- **vs. DeepSeek R1**: R1 validates RL-based reasoning incentivization only in mathematics; RAIF is the first to extend this paradigm to instruction following, revealing the need for additional CoT quality assurance mechanisms.
- **vs. Air/WizardLM**: SFT data engineering approaches that lack explicit reasoning capability cultivation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First to reveal that CoT is harmful to instruction following; first systematic RL-based reasoning incentivization method for instruction following.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 7 benchmarks, cold/warm start comparisons, multiple model scales, OOD generalization, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — In-depth problem analysis with clear contrast against mathematical reasoning.
- **Value**: ⭐⭐⭐⭐⭐ — Significant contribution to both complex instruction following and RL-based reasoning incentivization research.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Generalizing Verifiable Instruction Following](generalizing_verifiable_instruction_following.md)
- [\[NeurIPS 2025\] Financial Instruction Following Evaluation (FIFE)](financial_instruction_following_evaluation_fife.md)
- [\[NeurIPS 2025\] GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining](graphchain_large_language_models_for_large-scale_graph_analysis_via_tool_chainin.md)
- [\[NeurIPS 2025\] Checklists Are Better Than Reward Models For Aligning Language Models](checklists_are_better_than_reward_models_for_aligning_langua.md)
- [\[NeurIPS 2025\] Behavior Injection: Preparing Language Models for Reinforcement Learning](behavior_injection_preparing_language_models_for_reinforcement_learning.md)

<!-- RELATED:END -->
