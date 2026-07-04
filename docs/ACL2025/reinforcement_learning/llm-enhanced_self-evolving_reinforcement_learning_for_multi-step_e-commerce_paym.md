---
title: >-
  [Paper Note] LLM-Enhanced Self-Evolving Reinforcement Learning for Multi-Step E-Commerce Payment Fraud Risk Detection
description: >-
  [ACL 2025][Reinforcement Learning][Fraud Detection] Formulates e-commerce payment fraud detection as a multi-step MDP and utilizes LLMs (Mixtral/LLaMA/Gemma) to automatically generate and optimize RL reward functions through an evolutionary algorithm, significantly improving dollar-wise precision on real eBay transaction data compared to human-designed rewards and traditional SL baselines.
tags:
  - "ACL 2025"
  - "Reinforcement Learning"
  - "Fraud Detection"
  - "LLM Reward Function Design"
  - "Evolutionary Algorithm"
  - "E-Commerce Payment"
date: 2026-05-08
content_hash: cb175567209df7ba
---

# LLM-Enhanced Self-Evolving Reinforcement Learning for Multi-Step E-Commerce Payment Fraud Risk Detection

**Conference**: ACL 2025  
**arXiv**: [2509.18719](https://arxiv.org/abs/2509.18719)  
**Code**: None  
**Area**: Reinforcement Learning / LLM Application  
**Keywords**: Fraud Detection, Reinforcement Learning, LLM Reward Function Design, Evolutionary Algorithm, E-Commerce Payment

## TL;DR
Formulates e-commerce payment fraud detection as a multi-step MDP and utilizes LLMs (Mixtral/LLaMA/Gemma) to automatically generate and optimize RL reward functions through an evolutionary algorithm, significantly improving dollar-wise precision on real eBay transaction data compared to human-designed rewards and traditional SL baselines.

## Background & Motivation
**Background**: E-commerce payment fraud detection mainly relies on supervised learning (SL) models (such as GBDTs) to score each payment stage independently. However, running SL models independently at each stage prevents joint optimization across stages and cannot directly optimize business metrics (e.g., precision-recall trade-offs).

**Limitations of Prior Work**: (a) SL models make independent decisions at each stage, failing to model sequential dependencies between stages; (b) business requirements demand capturing more fraud at the Pre-auth stage (as late-stage detection is more costly), but SL cannot directly formulate such cross-stage constraints; (c) while RL can address these issues, designing effective reward functions requires extensive domain expert experience.

**Key Challenge**: The success of RL highly depends on reward function design, yet the exploration space of human-designed rewards is limited.

**Goal**: How to automate the design and optimization of RL reward functions?

**Key Insight**: Leveraging the reasoning and code generation capabilities of LLMs to iteratively optimize reward function code through an evolutionary algorithm.

**Core Idea**: Using LLMs as automatic reward function designers that continuously improve rewards through an evolutionary loop based on feedback from agent performance.

## Method

### Overall Architecture
The system is divided into two layers: (1) **Inner Loop**: Standard RL training—formulating the payment transaction as a two-step MDP (Pre-auth → Post-auth), where the agent decides whether to block/allow at each step, trained using the REINFORCE algorithm; (2) **Outer Loop**: LLM evolutionary cycle—at each iteration, the LLM generates candidate reward function code -> trains the RL agent -> evaluates performance -> provides feedback to the LLM -> generates better reward functions in the next iteration.

### Key Designs

1. **Transaction Risk MDP Modeling**:

    - Function: Models the payment workflow (Pre-auth → Issuer check → Post-auth) as a finite-step MDP.
    - Mechanism: State $\mathcal{S}_i$ = SL model scores from each stage + stage indicator; action $\mathcal{A}_i$ = {block, allow}. The optimization target is to maximize $\$TP - \$FP$, subject to the constraint $\$TP_{\text{stage1}} > \$TP_{\text{stage2}}$ (detecting early is more valuable).
    - Design Motivation: Traditional step-by-step scoring of SL cannot formulate cross-stage constraints, whereas MDP naturally supports sequential decision-making.

2. **LLM-Driven Evolutionary Reward Function Optimization**:

    - Function: The LLM receives the task description, RL environment code, historical reward functions, and performance feedback to generate new reward function code.
    - Mechanism: The evolutionary algorithm iterates for around 60 rounds ($N_{iter} \approx 60$), sampling approximately 10 candidates per round ($N_{samples} \approx 10$). The agent is trained for about 150 episodes. After evaluation, the best, sub-optimal, and failed experiences are fed back to the LLM. It supports both zero-shot (no examples provided) and few-shot (providing human-designed reward functions as references).
    - Design Motivation: The human-designed precision-constraint reward function (formula $R = (1-\alpha_i)\$TP_i - \alpha_i\$FP_i$) is effective but has a limited exploration space.

3. **Human-Designed Reward Function Baseline**:

    - Function: Derives precision-constrained rewards based on business constraints.
    - Mechanism: Derives $R_{\text{precision}}^i = (1-\alpha_i)\$TP_i - \alpha_i\$FP_i$ from the precision constraint $\frac{\$TP_i}{\$TP_i + \$FP_i} > \alpha_i$ through Lagrangian relaxation.
    - Design Motivation: To provide an effective baseline that can nonetheless be outperformed.

### Loss & Training
REINFORCE algorithm + Adam optimizer, with a 3-layer neural network of size [8, 32, 8], GELU activation, and Dropout. Offline RL is performed using historical transaction data.

## Key Experimental Results

### Main Results

| Recall Level | SL Baseline $Prec | RL (Human Reward) $Prec | RL (LLM few-shot, LLaMA-3-8B) $Prec |
|-------------|--------------|-------------------|--------------------------------------|
| @80% | 66.57% | 69.65% | **73.74%** |
| @85% | 58.79% | 64.22% | **71.70%** |
| @90% | 51.27% | 55.70% | **55.90%** |

### Ablation Study

| Configuration (Test S, @85% Recall) | $Precision | Description |
|---------------------------|------------|------|
| SL Baseline | 58.79% | Traditional SL model |
| RL + Human Reward | 64.22% | Human-designed reward |
| RL + Mixtral Zero-shot | 69.62% | LLM zero-shot |
| RL + Mixtral Few-shot | 70.73% | LLM few-shot |
| RL + LLaMA-3 Few-shot | **71.70%** | Best LLM |

### Key Findings
- **LLM Evolutionary Reward > Human Reward**: At @85% recall, $Prec improved from 64.22% to 71.70% (+7.48 pp), demonstrating that LLMs can explore reward design spaces undiscovered by humans.
- **Few-shot > Zero-shot but with a small gap**: Zero-shot also achieves competitive results (69.62% vs 70.73%), indicating strong domain understanding in LLMs.
- **Stable Long-term Evaluation**: A 6-month Test L (6.17M transactions) validates the temporal consistency of the agent.
- **Explainability of LLM-Generated Rewards**: The code generated by zero-shot introduces differentiated weights for stages (stage0: 1.2x, stage1: 0.9x), which aligns with business logic.

## Highlights & Insights
- **LLM as an RL Reward Designer**: Employing the code generation capabilities of LLMs for automated RL reward function design is a promising paradigm, transferable to any RL scenario requiring complex reward engineering.
- **Effectiveness of the Evolutionary Loop**: Through a closed performance feedback loop (synthesizing success/failure experiences), LLMs can continuously improve reward design. This echoes the work of Ma et al. (2023) in robotics, but is applied to financial risk control for the first time.
- **Industrial-grade Validation**: Uses real eBay transaction data (million-scale) with a 6-month long-term evaluation, demonstrating strong empirical validity.

## Limitations & Future Work
- The state representation only uses SL model scores, without incorporating raw transaction features.
- Only a two-step MDP (Pre-auth + Post-auth) was validated; longer decision chains remain unexplored.
- The computational cost of the evolutionary loop is high (60 rounds × 10 samples × 150 episodes × 40 min/round), requiring optimization for industrial deployment.
- Certain designs within the LLM-generated reward functions (e.g., choice of specific hyper-parameters) remain difficult to fully interpret.

## Related Work & Insights
- **vs Ma et al. (Eureka)**: Eureka uses LLMs to evolve reward functions in robotics tasks; this work is the first to transfer this paradigm to financial risk control.
- **vs Traditional SL Fraud Detection**: SL operates independently at each stage and cannot optimize across stages, whereas RL naturally supports sequential decision-making and business constraints.
- **vs Manual RL Reward Design**: While manual design is already effective (+5.4 pp at @85%), the LLM brings a further improvement (+12.9 pp), and automation eliminates the bottleneck of manual exploration.

## Rating
- Novelty: ⭐⭐⭐⭐ LLM+RL evolutionary reward design is applied to financial risk control for the first time, which is pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Uses real-world data, long-term evaluation, 3 different LLMs, and zero/few-shot comparisons.
- Writing Quality: ⭐⭐⭐ Features an industrial paper style; mathematical derivations are clear, though some details have average readability.
- Value: ⭐⭐⭐⭐ Strong industrial utility; the LLM-evolved RL paradigm has great potential for generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] R-Zero: Self-Evolving Reasoning LLM from Zero Data](../../ICLR2026/reinforcement_learning/r-zero_self-evolving_reasoning_llm_from_zero_data.md)
- [\[ACL 2025\] TreeRL: LLM Reinforcement Learning with On-Policy Tree Search](treerl_tree_search_rl.md)
- [\[ICML 2025\] Diving into Self-Evolving Training for Multimodal Reasoning](../../ICML2025/reinforcement_learning/diving_into_self-evolving_training_for_multimodal_reasoning.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](../../ICLR2026/reinforcement_learning/spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)
- [\[ACL 2025\] An Efficient Task-Oriented Dialogue Policy: Evolutionary Reinforcement Learning Injected by Elite Individuals](eierl_dialogue_policy.md)

</div>

<!-- RELATED:END -->
