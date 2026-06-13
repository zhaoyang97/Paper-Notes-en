---
title: >-
  [Paper Note] Extra-CoT: A Chain-of-Thought Compression Framework under Extreme Compression Ratios
description: >-
  [ICML2026][Reinforcement Learning][Chain-of-Thought Compression] Extra-CoT proposes a three-stage framework (Semantic-Preserving Compressor → Mixed-Ratio SFT → Hierarchical Reward RL) that maintains reasoning accuracy ev…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "Chain-of-Thought Compression"
  - "CoT Efficiency"
  - "Reasoning Acceleration"
  - "token budget control"
date: 2026-05-08
content_hash: 670a87a6a09ef0e0
---

# Extra-CoT: A Chain-of-Thought Compression Framework under Extreme Compression Ratios

**Conference**: ICML2026  
**arXiv**: [2602.08324](https://arxiv.org/abs/2602.08324)  
**Code**: https://github.com/Mwie1024/Extra-CoT  
**Area**: LLM Reasoning  
**Keywords**: Chain-of-Thought Compression, CoT Efficiency, Reinforcement Learning, Reasoning Acceleration, token budget control  

## TL;DR
Extra-CoT proposes a three-stage framework (Semantic-Preserving Compressor → Mixed-Ratio SFT → Hierarchical Reward RL) that maintains reasoning accuracy even under extreme compression ratios (preserving only 20% of tokens). It achieves a 73% token reduction on MATH-500 while improving accuracy by 0.6%.

## Background & Motivation

**Background**: Large language reasoning models (e.g., OpenAI o1, DeepSeek-R1) significantly enhance reasoning capabilities by generating step-by-step Chains-of-Thought (CoT). However, this comes with massive token overhead—models tend to "overthink," generating redundant reasoning paths even for simple problems.

**Limitations of Prior Work**: Existing controllable CoT compression methods (e.g., TokenSkip, CTS) operate reasonably well at low-to-medium compression ratios (preserving 50-60% of tokens) but suffer catastrophic performance collapse at high compression ratios (preserving 20-30% of tokens). The fundamental cause is their reliance on general token importance evaluators (e.g., LLMLingua-2), which fail to preserve sparse but critical reasoning steps under extreme compression, leading to fatal losses in semantic integrity and logical faithfulness.

**Key Challenge**: CoT integrity is directly correlated with reasoning accuracy, but extreme compression inevitably results in the loss of a large volume of tokens. Existing methods fail to achieve "semantic-preserving extreme compression"—general compressors fracture mathematical formulas and disrupt reasoning chains, and models do not comply with ultra-low ratio instructions ("control collapse").

**Goal**: Achieve high-accuracy reasoning under extreme compression ratios ($\gamma = 0.2$, i.e., preserving only 20% of tokens).

**Key Insight**: The problem stems from three levels: (1) Poor quality of compressed supervision data—general compressors break formulas; (2) Models distrust low-ratio training data, leading to instruction non-compliance; (3) Absence of explicit incentives for the model to actively select a low budget.

**Core Idea**: Address these issues through a three-stage pipeline—first, use a specialized question-aware compressor to generate high-quality supervision data; second, perform mixed-ratio SFT to teach the model to follow various compression instructions; finally, use hierarchical reward RL to optimize the model's autonomous selection of ultra-low budgets.

## Method

### Overall Architecture
Extra-CoT consists of three serial stages: (a) Training a semantic-preserving CoT compressor to generate high-quality training data across various compression ratios; (b) Performing mixed-ratio SFT on the reasoning LLM to teach it to follow different compression budget instructions (e.g., $\langle\text{COMP\_40}\rangle$), while introducing a policy token ($\langle\text{COMP\_POLICY}\rangle$) as a trainable mechanism for the RL stage; (c) Using CHRPO (Constrained and Hierarchical Ratio Policy Optimization) to perform RL optimization on the policy token, explicitly incentivizing the model to select the lowest budget while maintaining accuracy. During inference, if a fixed ratio token is input, the model generates at that ratio; if the POLICY token is input, the model decides the ratio autonomously.

### Key Designs

1.  **Semantic-Preserving Question-Aware Compressor**:
    - **Function**: Generate semantically complete CoT training data under extreme compression ratios.
    - **Mechanism**: Utilizing Longformer as the backbone, global attention is enabled for question tokens (question-aware), while CoT tokens use sliding window local attention. GPT-4o is used for index-based labeling—returning only the set of indices $R \subseteq \{1,\dots,m\}$ to be preserved, thus avoiding noise from formula rendering differences. A critical innovation is "formula atomization": all LaTeX entities and mathematical expressions are merged into indivisible units, ensuring the compressor does not break symbolic reasoning components. Training employs class-weighted Focal Loss: $\mathcal{L}_{\text{Focal}} = -\mathbb{E}_{i}[\alpha_{y_i}(1-p_{i,y_i})^{\lambda}\log p_{i,y_i}]$
    - **Design Motivation**: General compressors (LLMLingua-2) fracture formulas and create semantic gaps at extreme ratios. Question-awareness and formula atomization ensure mathematical integrity, fundamentally solving the cascading issue of "low-quality supervision → model non-compliance."

2.  **Mixed-Ratio SFT and Prefix-Mirror Protocol**:
    - **Function**: Teach the model to controllably generate compressed reasoning at multiple ratios and establish an autonomous selection mechanism.
    - **Mechanism**: The vocabulary is expanded to include 6 control tokens ($\langle\text{COMP\_20}\rangle$ to $\langle\text{COMP\_POLICY}\rangle$). A "Prefix-Mirror Protocol" is designed: in fixed-ratio mode, the model duplicates the input control token; in POLICY mode, the model first predicts a ratio token autonomously before generating reasoning. Training data is split into two queues: a fixed-ratio queue (60k samples, 12k for each of 5 ratios) and a POLICY warm-up queue (10k samples, with target ratios determined by a heuristic difficulty-selection model).
    - **Design Motivation**: Directly training on a single extreme ratio leads to performance collapse. Multi-ratio training allows the model to establish a robust perception of "compression gradients"; the POLICY queue builds a trainable decision mechanism for the RL stage.

3.  **CHRPO (Constrained and Hierarchical Ratio Policy Optimization)**:
    - **Function**: Optimize the model via RL to autonomously select ultra-low budgets while maintaining accuracy.
    - **Mechanism**: A two-level reward system is designed: the main reward $\mathcal{R}_{\text{main}}$ is applied to all tokens at the end of the sequence, encompassing accuracy rewards, budget calibration (Huber-type reward), ratio optimization mode (asymmetric penalty), and reasoning integrity constraints; the control head reward $\mathcal{R}_{\text{ctrl}}$ is applied only to the first token, directly shaping the ratio selection strategy. The asymmetric penalty provides a triple guarantee: safety for shortening ($\lambda_{\text{under}}^{\text{err}} > \lambda_{\text{over}}^{\text{err}}$, penalty for failure when shortening is greater than that of failure when lengthening), fast recovery (preference for increasing budget on difficult problems over adhering to low budgets), and a progress cap ($\kappa$ limits the maximum reward for successful compression to avoid unstable jumps).
    - **Design Motivation**: In a single-reward setup, the ratio selection at the first token is too distant from the final accuracy signal, diluting the gradient signal and causing the strategy to become risk-averse. Hierarchical rewards provide immediate feedback for ratio selection, while the asymmetric penalty ensures a "safe fallback upon failure."

## Key Experimental Results

### Main Results

| Method | Dataset | Target Ratio | ActRatio | Token Count↓ | Acc↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Base Model (Qwen3-1.7B) | MATH-500 | - | - | 1675 | 64.2 |
| TokenSkip | MATH-500 | 0.2 | 0.39 | 660 | 23.4 |
| Extra-CoT (SFT) | MATH-500 | 0.2 | 0.29 | 481 | 47.8 |
| Extra-CoT (CHRPO) | MATH-500 | POLICY | 0.27 | 452 | **64.8** |
| Thinkless (DeGRPO) | MATH-500 | - | 0.53 | 888 | 63.6 |
| TokenSkip | GSM8K | 0.2 | 0.30 | 273 | 59.1 |
| Extra-CoT (SFT) | GSM8K | 0.2 | 0.34 | 303 | 80.2 |
| Extra-CoT (CHRPO) | GSM8K | POLICY | 0.24 | 210 | **85.8** |

### Ablation Study (CHRPO Reward Components, Qwen3-1.7B)

| Configuration | GSM8K Token↓ | GSM8K Acc↑ | MATH-500 Token↓ | MATH-500 Acc↑ |
| :--- | :--- | :--- | :--- | :--- |
| w/o $\mathcal{R}_{\text{mode+ctrl}}$ | 324 | 85.7 | 604 | 60.0 |
| w/o $\mathcal{R}_{\text{ctrl}}$ | 258 | 82.1 | 568 | 59.6 |
| Full CHRPO | **210** | **85.8** | **452** | **64.8** |

### Key Findings
- At an extreme compression ratio of 0.2, Extra-CoT achieves **+24.4** percentage points higher accuracy than TokenSkip on MATH-500 (47.8 vs 23.4), demonstrating the decisive importance of high-quality compression supervision.
- The CHRPO strategy reaches 64.8% accuracy on MATH-500 with an ActRatio of 0.27, outperforming Thinkless (0.53 ActRatio / 63.6% accuracy), achieving higher accuracy with less than half the tokens.
- Removing $\mathcal{R}_{\text{mode}}$ causes the token count to spike from 210 to 324, indicating that the mode reward is the core engine driving the strategy toward ultra-low budgets.
- End-to-end latency: Extra-CoT achieves 3.24× acceleration on GSM8K and 2.35× on MATH-500.

## Highlights & Insights
- **Formula Atomization Labeling** is an ingenious design: by treating entire LaTeX expressions as indivisible units rather than exposing internal tokens to the compressor, it avoids the most fatal failure mode of general compressors—formula fragmentation. This concept is transferable to any scenario involving structured text compression (e.g., code or SQL compression).
- **Temporal Separation of Hierarchical Rewards** addresses a common RL challenge: when a critical decision (ratio selection) occurs at the sequence start while the outcome signal (accuracy) is at the end, credit assignment becomes extremely difficult. CHRPO applies an independent immediate reward to the first token, effectively shortening the credit propagation distance.
- The **Asymmetric Penalty Design** (Shortening failure > Lengthening failure) reflects an "engineering intuition for safety"—encouraging the strategy to be conservative when uncertain rather than risking compression.

## Limitations & Future Work
- Compressor training relies on GPT-4o for index labeling, resulting in high data generation costs.
- Experiments focused on mathematical reasoning tasks; generalization to other reasoning types such as code generation and logical reasoning has not been fully explored.
- The fixed 5-level discrete ratio grid (0.2/0.4/0.6/0.8/1.0) may limit strategy flexibility; continuous ratio control is a worthwhile area for exploration.

## Related Work & Insights
- TokenSkip (Xia et al., 2025): Pioneering work using LLMLingua-2 as a compressor, but it suffers from performance collapse at extreme ratios.
- Thinkless (Fang et al., 2025): Decouples mode selection and answer generation via DeGRPO, but adjusts reasoning length rather than content.
- COCONUT (Hao et al., 2024): Performs reasoning compression in latent space but faces catastrophic forgetting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](../../NeurIPS2025/reinforcement_learning/reinforcing_the_diffusion_chain_of_lateral_thought_with_diffusion_language_model.md)
- [\[ICML 2026\] Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory](game_of_thought_robust_information_seeking_with_large_language_models_using_game.md)
- [\[ICML 2026\] Convergence of Steepest Descent and Adam under Non-Uniform Smoothness](convergence_of_steepest_descent_and_adam_under_non-uniform_smoothness.md)
- [\[ICML 2026\] Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed](safety_generalization_under_distribution_shift_in_safe_reinforcement_learning_a_.md)
- [\[ICML 2026\] Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning](interaction-breaking_adversarial_learning_framework_for_robust_multi-agent_reinf.md)

</div>

<!-- RELATED:END -->
