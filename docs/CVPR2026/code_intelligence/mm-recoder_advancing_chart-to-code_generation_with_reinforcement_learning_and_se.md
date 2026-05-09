---
title: >-
  [Paper Note] MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction
description: >-
  [CVPR 2026][Chart-to-Code] This paper proposes MM-ReCoder, the first multimodal LLM with genuine self-correction capability for chart-to-code generation. Through a two-stage multi-turn GRPO reinforcement learning framework—first optimizing correction ability via shared-first-turn training, then optimizing coding ability via full-trajectory training—MM-ReCoder achieves 86.5% low-level score on ChartMimic with only 7B parameters, rivaling Qwen3-VL-235B.
tags:
  - CVPR 2026
  - Chart-to-Code
  - Reinforcement Learning
  - Self-Correction
  - Multi-Turn Dialogue
  - GRPO
date: 2026-05-08
content_hash: b0e26a23cb23135b
---

# MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction

**Conference**: CVPR 2026
**arXiv**: [2604.01600](https://arxiv.org/abs/2604.01600)
**Code**: [https://zitiantang.github.io/MM-ReCoder](https://zitiantang.github.io/MM-ReCoder)
**Area**: Multimodal VLM / Code Generation
**Keywords**: Chart-to-Code, Reinforcement Learning, Self-Correction, Multi-Turn Dialogue, GRPO

## TL;DR

This paper proposes MM-ReCoder, the first multimodal LLM with genuine self-correction capability for chart-to-code generation. Through a two-stage multi-turn GRPO reinforcement learning framework—first optimizing correction ability via shared-first-turn training, then optimizing coding ability via full-trajectory training—MM-ReCoder achieves 86.5% low-level score on ChartMimic with only 7B parameters, rivaling Qwen3-VL-235B.

## Background & Motivation

1. **State of the Field**: The Chart2Code task requires generating executable Python plotting code from chart images. Existing approaches primarily rely on SFT (e.g., ChartCoder trained on 160k chart–code pairs), with only a few works (e.g., ChartMaster) beginning to explore RL.
2. **Limitations of Prior Work**: SFT methods do not interact with code execution environments, and thus cannot guarantee the executability or visual fidelity of generated code. Existing RL methods (e.g., ChartMaster) perform only single-pass generation without iterative self-correction.
3. **Root Cause**: Human programming is inherently iterative (write → execute → inspect → revise), yet existing MLLMs generate code in a single pass. Experiments reveal that even large models such as Qwen3-VL-235B exhibit degraded code quality during self-correction on already-executable code (−0.26%).
4. **Paper Goals**: Enable MLLMs to acquire genuine self-correction ability—not merely fixing execution errors, but also improving the visual quality of already-executable code.
5. **Starting Point**: The authors find that the apparent "improvement" of existing models stems solely from making crashed code executable, rather than from genuine quality enhancement. Two-stage RL training is thus designed to address each aspect separately.
6. **Core Idea**: Employ a two-stage GRPO training pipeline—shared-first-turn optimization followed by full-trajectory optimization—to first acquire correction ability and then improve overall coding ability.

## Method

### Overall Architecture

The training pipeline consists of two major stages: (1) **Cold Start**—SFT on Chart2Code-160k, followed by fine-tuning on 7k curated two-turn correction samples; (2) **Two-Stage Multi-Turn GRPO RL**—Stage 1 fixes the first-turn output and trains correction ability (shared-first-turn), while Stage 2 jointly optimizes both turns (full-trajectory). At inference time, the model can perform arbitrary rounds of self-correction, receiving rendered outputs or error messages as feedback at each turn.

### Key Designs

1. **Shared-First-Turn Optimization**:

    - **Function**: Dedicated training of the model's self-correction ability.
    - **Mechanism**: For each input chart, a shared first-turn output $o^{(1)}$ is sampled; $G$ diverse second-turn correction candidates $o_i^{(2)}$ are then generated conditioned on it. RL optimization is applied only to the second turn, with the first turn serving as shared context. The objective is $\mathcal{J}^{(shared)} = \mathbb{E}[\frac{1}{G}\sum_i \frac{\pi_\theta(o_i^{(2)}|q,o^{(1)},f^{(1)})}{\text{SG}[\cdot]} A_i]$.
    - **Design Motivation**: Under direct full-trajectory optimization, the model simply repeats the first-turn code in 46.9% of cases (as the expected reward of modifying executable code is negative) or deliberately generates poor first-turn outputs to exploit the reward mechanism. Fixing the first turn forces the model to learn how to improve upon a given code.

2. **Reward Design (Rule-based + Model-based)**:

    - **Function**: Comprehensive evaluation of generated chart quality.
    - **Mechanism**: Rule-based reward hooks Matplotlib functions to extract chart elements (type, text, color, layout, etc.) and computes F1/CIE Lab distances against ground truth (range [0,1]). Model-based reward uses Qwen2.5-VL-72B to score outputs across six dimensions (out of 100, normalized to [0,1]). Final reward $= (1-\alpha-\beta) \times \text{Format} + \alpha \times \text{Rule-based} + \beta \times \text{Model-based}$.
    - **Design Motivation**: Rule-based rewards are precise but have blind spots (e.g., overlapping text still receives full score), while model-based rewards capture visual quality but are noisy. The two are complementary.

3. **Two-Stage Training Strategy**:

    - **Function**: Learn correction first, then coding.
    - **Mechanism**: RL Stage 1 uses the shared-first-turn strategy to cultivate correction ability (encouraging diverse correction solutions). Stage 2 switches to full-trajectory optimization, jointly training both turns to improve overall coding competence. Rewards are computed based on the final turn only ($\gamma=0$ is optimal).
    - **Design Motivation**: Ablations show that full-trajectory-only optimization leads to code repetition and reward exploitation, while shared-first-turn-only training improves correction ability but yields weaker overall coding performance. The two-stage combination is mutually complementary.

### Loss & Training

- **Cold Start**: SFT on Chart2Code-160k for 1 epoch + correction data for 2 epochs; batch=128, lr=$10^{-5}$.
- **Correction Data Construction**: Two-turn dialogues generated by Qwen3-VL-235B; samples where the second-turn low-level score exceeds the first by more than 0.02 are retained (~7k samples).
- **GRPO**: Group size $G=8$; 1 epoch per stage; batch=128; lr=$10^{-6}$; maximum response length 4096 tokens.
- Format reward encourages outputs in the `<think>...</think>'''python...'''` format.

## Key Experimental Results

### Main Results

| Model | Params | Turns | ChartMimic Low↑ | ChartMimic High↑ | Plot2Code Text-Match↑ |
|-------|--------|-------|------------------|-------------------|----------------------|
| ChartCoder | 7B | 1 | 77.4 | 74.0 | 54.5 |
| Qwen2.5-VL-7B | 7B | 1 | 56.2 | 49.6 | 47.8 |
| Qwen3-VL-235B | 235B | 1 | 80.9 | 85.9 | 60.9 |
| GPT-4o | — | 1 | 81.8 | 83.7 | 59.8 |
| **MM-ReCoder** | **7B** | **1** | **83.5** | **81.2** | **63.2** |
| **MM-ReCoder** | **7B** | **4** | **86.5** | **84.9** | **62.7** |

### Ablation Study

RL strategy comparison (ChartMimic):

| Strategy | Turn-1 Low | Turn-2 Low | Avg Improvement | Improve Rate | Code Repeat Rate |
|----------|-----------|-----------|----------------|-------------|-----------------|
| Full-traj (γ=0,η=0) | 81.8 | 83.9 | +0.21 | 3.4% | 46.9% |
| Full-traj (γ=0,η=0.1) | 66.7 | 84.3 | +10.11 | 87.5% | 0.3% |
| Shared-first + Full-traj | **83.7** | **86.0** | +0.55 | 12.1% | 21.6% |

Stage-wise ablation:

| Stage | Low-level | Avg Improvement | Repeat Rate |
|-------|-----------|----------------|-------------|
| Qwen2.5-VL-7B base | 56.2 | −0.36 | 10.9% |
| + Single-turn cold start | 79.1 | −0.10 | 81.5% |
| + Multi-turn cold start | 75.2 | −0.54 | 2.3% |
| + RL (full) | 83.5→84.8 | **+0.30** | 2.2% |

### Key Findings

- **Existing large models cannot genuinely self-correct**: Qwen3-VL-8B/235B show negative quality improvement on already-executable code (−1.03%/−0.26%); gains come solely from fixing crashed code.
- Using only the correction bonus ($\eta=0.1$) causes the model to "cheat"—deliberately generating a poor first turn to obtain a high improvement reward (Turn-1 Low drops to 66.7).
- Multi-turn correction exhibits diminishing returns: Turn 1→2 improves by 1.3%, Turn 2→3 by 0.5%, with no further gain beyond Turn 4→5.
- **Code repetition is the core bottleneck**: After single-turn SFT, 81.5% of second-turn outputs are simple copies of the first turn.

## Highlights & Insights

- **Diagnosis before treatment**: The authors first demonstrate the "pseudo self-correction" problem in existing models (gains arising only from fixing crashed code), then design targeted solutions accordingly. This problem-driven research methodology is exemplary.
- **Shared-first-turn strategy**: Decoupling correction training from overall coding training avoids reward exploitation in RL. This strategy is transferable to other multi-round code generation scenarios (e.g., webpage generation, SVG generation).
- **Hybrid rule-based and model-based reward**: The idea of hooking Matplotlib to precisely extract chart elements is elegant, effectively resolving the challenge of automated chart quality assessment.

## Limitations & Future Work

- RL training is explored only for single-round correction; multi-round RL training may yield further gains.
- Model-based reward relies on a 72B model, incurring high training costs (4×8 H200 GPUs required for the reward model).
- Richer feedback from the code execution environment (e.g., diff information between rendered outputs) remains unexplored.
- Validation is limited to chart generation; the framework is extensible to more visual coding tasks such as webpages, UI, and SVG.

## Related Work & Insights

- **vs. ChartMaster**: ChartMaster first introduced GRPO to Chart2Code but only performs single-pass generation. MM-ReCoder extends this with a multi-turn self-correction dimension.
- **vs. SCoRe (Kumar et al.)**: SCoRe pioneered two-stage RL for self-correction in text-only LLMs. MM-ReCoder extends this paradigm to multimodal coding tasks and adapts it for GRPO (SCoRe uses REINFORCE/PPO).
- **vs. ChartCoder**: ChartCoder relies solely on SFT; applying RL on top of its training data (Chart2Code-160k) raises the low-level score from 77.4 to 86.5 (+9.1%), demonstrating the substantial value of RL.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to achieve reliable self-correction in multimodal coding; the shared-first-turn strategy is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks, extensive strategy ablations, multi-turn correction analysis, and detailed comparisons with large models.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clear and experimental analysis is thorough; notation is dense and requires careful tracking.
- **Value**: ⭐⭐⭐⭐ High practical value—7B model rivaling 235B; the self-correction paradigm is generalizable to broader tasks.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](../../ICLR2026/code_intelligence/breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](../../AAAI2026/code_intelligence/recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [\[CVPR 2026\] GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning](geotikzbridge_advancing_multimodal_code_generation_for_geometric_perception_and_.md)
- [\[ACL 2026\] OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward](../../ACL2026/code_intelligence/omnidiagram_advancing_unified_diagram_code_generation_via_visual_interrogation_r.md)

<!-- RELATED:END -->
