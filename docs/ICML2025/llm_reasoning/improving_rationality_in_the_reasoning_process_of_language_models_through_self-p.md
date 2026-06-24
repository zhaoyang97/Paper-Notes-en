---
title: >-
  [Paper Note] Improving Rationality in the Reasoning Process of Language Models through Self-playing Game
description: >-
  [ICML 2025][Reasoning][Self-play] This paper proposes the Critic-Discernment Game (CDG), a self-playing language game where an LLM interacts with a "Helpful Critic" and a "Misleading Critic." Using Reinforced Self-Training (ReST), the three roles are jointly optimized. Without relying on human or stronger model supervision, this approach significantly enhances the LLM's rational understanding of its own reasoning process, achieving consistent improvements across four tasks: m…
tags:
  - "ICML 2025"
  - "Reasoning"
  - "Self-play"
  - "Reasoning Rationality"
  - "Critic-Discernment Game"
  - "Reinforcement Learning"
  - "Self-correction"
date: 2026-05-08
content_hash: 2a44ca905f72081a
---

# Improving Rationality in the Reasoning Process of Language Models through Self-playing Game

**Conference**: ICML 2025  
**arXiv**: [2506.22920](https://arxiv.org/abs/2506.22920)  
**Code**: Yes (mentioned as open-sourced in the paper)  
**Area**: LLM Reasoning  
**Keywords**: Self-play, Reasoning Rationality, Critic-Discernment Game, Reinforcement Learning, Self-correction

## TL;DR
This paper proposes the Critic-Discernment Game (CDG), a self-playing language game where an LLM interacts with a "Helpful Critic" and a "Misleading Critic." Using Reinforced Self-Training (ReST), the three roles are jointly optimized. Without relying on human or stronger model supervision, this approach significantly enhances the LLM's rational understanding of its own reasoning process, achieving consistent improvements across four tasks: mathematical reasoning, step-by-step error detection, self-correction, and long-chain reasoning.

## Background & Motivation
**Background**: LLMs have demonstrated strong capabilities in reasoning tasks such as mathematics and code. However, recent studies suggest that even the best models lack a genuine understanding of the reasoning process and rely heavily on probabilistic pattern matching.

**Limitations of Prior Work**: The reasoning process of LLMs is unstable and prone to hallucinations and errors, with models struggling to autonomously detect and correct these issues. In long-chain reasoning, intermediate errors continuously accumulate, leading to increasingly deviated final results.

**Key Challenge**: Existing methods (such as PRMs or preference data pairs) rely on human-annotated step-level supervision, making them difficult to scale. Moreover, they cannot explicitly define fine-grained steps, only indicating which step is better without explaining the reasons.

**Goal**: To enhance the LLM's rational understanding of its reasoning process without relying on supervision from humans or stronger models.

**Key Insight**: By designing a language-level self-play game, the model learns to discern the correctness of its own reasoning steps through interactions with critics harboring different intentions.

**Core Idea**: To truly understand its own reasoning process, the model must simultaneously learn to defend correct answers when facing misleading criticism and correct erroneous answers when receiving constructive feedback.

## Method

### Overall Architecture
CDG is a three-role self-play framework. The Prover first generates a solution for a given problem and then receives criticism from the Critic. The Critic plays two roles: the Helpful Critic assists in rectifying errors when the Prover provides an incorrect solution, while the Misleading Critic attempts to induce modification when the Prover provides a correct solution. The three roles are jointly optimized through ReST (Reinforced Self-Training) and improve their game-playing capabilities over multiple iterations of self-play.

### Key Designs

1. **Prover**:

    - **Function**: Generates a chain-of-thought solution given a query, then receives criticism of an unknown intent and decides whether to modify the final answer.
    - **Mechanism**: The Prover must make rational judgments without knowing the critic's intent—maintaining the correct answer when facing misleading criticism and revising errors when receiving constructive feedback.
    - **Design Motivation**: This "discernment" capability is the core manifestation of an LLM's understanding of the reasoning process. The Prover has two winning conditions: (1) correct on the first attempt and successfully resisting misleading critics, yielding a higher reward (including an extra reward $\eta$); (2) incorrect on the first attempt but successfully corrected with the help of the Helpful Critic.

2. **Helpful Critic**:

    - **Function**: Receives the problem and the Prover's incorrect answer, points out errors in the reasoning without directly giving the correct answer, and guides the Prover to self-correct.
    - **Mechanism**: Simulates real-world academic discussion scenarios where the critic can freely choose the granularity of criticism, presented in natural language.
    - **Design Motivation**: Establishes a cooperative relationship with the Prover—the Helpful Critic's reward $R_\mu$ is defined as the probability of successfully guiding the Prover from incorrect to correct.

3. **Misleading Critic**:

    - **Function**: Receives the problem and the Prover's correct answer, and fabricates a non-existent error to mislead the Prover into changing the answer.
    - **Mechanism**: Compels the Prover to deeply understand its own reasoning process through adversarial training, rendering it undeterred by false feedback.
    - **Design Motivation**: Establishes an adversarial relationship with the Prover—the Misleading Critic's reward $R_\rho$ is defined as the probability of successfully deceiving the Prover into changing a correct answer. As training progresses, the misleader becomes stronger, forcing the Prover to understand the reasoning more deeply to win.

### Loss & Training

**Reward Function Design**:
- The reward for the Prover consists of two terms: $R_\pi = \mathbb{E}[\mathbb{1}_{\text{correct}}(z,y)(\mathbb{1}_{\text{correct}}(z',y) + \eta) + (1 - \mathbb{1}_{\text{correct}}(z,y))\mathbb{1}_{\text{correct}}(z',y)]$
- The reward for being correct initially and resisting misleading critics is greater than the reward for correcting an initial error with help (controlled via the hyperparameter $\eta$).

**Training Method (ReST)**:
- Employs Reinforced Self-Training to filter high-reward samples via thresholds for language modeling loss training.
- Threshold settings: $\tau_\pi = 0.5$, $\tau_\rho = 0.75$ (requiring a higher success rate for the misleader), and $\tau_\mu = 0.5$.
- Offline learning scheme: each round first collects self-play data, accumulates it into a historical dataset, and retrains from the initial model.
- Data balancing: keeps 10,000 samples for each of the three categories (initially correct, resisted misleading, and corrected error).

**Multi-round Iteration**: Typically yields 2 rounds of self-play training. In the second round, because the misleader's attacks become stronger after RL training, the Prover achieves a more substantial improvement.

## Key Experimental Results

### Main Results (Mathematical Reasoning)

| Dataset | Metric | LLaMA-3.1-8B-Instruct | CDG-2 | Gain |
|--------|------|----------------------|-------|------|
| GSM8K | P@1 | 85.3 | 86.8 | +1.5 |
| GSM8K | M@32 | 93.0 | 93.1 | +0.1 |
| MATH500 | P@1 | 49.4 | 51.7 | +2.3 |
| MATH500 | M@32 | 63.4 | 66.0 | +2.6 |
| Qwen2.5-1.5B(MATH500) | P@1 | 55.4 | 57.6 | +2.2 |

### Step-by-Step Error Detection

| Dataset | Metric | Original Model | CDG | Gain |
|--------|------|---------|-----|------|
| GSM8K | F1 / Acc | 74.0 / 64.4 | 76.9 / 69.3 | +2.9 / +4.9 |
| MATH500 | F1 / Acc | 64.4 / 55.4 | 71.4 / 67.5 | +7.0 / +12.1 |

### Ablation Study

| Configuration | Math (GSM8K) | Error Detection (MATH) | Self-Correction (MATH) | Long-Chain Reasoning | Description |
|------|------------|---------------|---------------|---------|------|
| CDG (Full) | 86.8 | 71.4 | +1.4 | 29.7 | Best |
| CDG w/o Helpful Critic | 86.2 | 69.7 | -3.0 | 28.3 | Self-correction drastically drops |
| CDG w/o Misleading Critic | 84.9 | 68.9 | -0.5 | 29.4 | Reasoning performance drops |
| Expert Iteration | 87.2 | 67.4 | +0.8 | 22.8 | Long-chain reasoning is poor |
| Step-DPO | 84.6 | 58.2 | -2.1 | 27.6 | Relies on GPT-4o annotations |

### Comparison of RL Methods

| Method | GSM8K P@1 | MATH500 P@1 | MATH500 M@32 |
|------|-----------|-------------|--------------|
| CDG-ReST | 86.8 | 51.7 | 66.0 |
| CDG-DPO | 83.3 | 46.0 | 54.8 |
| CDG-PPO | 86.6 | 51.6 | 62.6 |

### Key Findings
- CDG achieves larger improvements on the harder MATH500 dataset, indicating that the method is more effective for complex problems.
- The performance gain in the second round of training is larger than in the first round, as the misleader delivers stronger attacks following RL training.
- ReST exhibits the best stability; in contrast, DPO even performs below the baseline, and PPO is sensitive to hyperparameters.
- In long-chain reasoning, the distillation effect of the model trained with CDG is 3-5 percentage points higher than the original model.
- Self-Correction experiment: CDG reduces the probability of mistakenly modifying a correct answer by more than half on GSM8K.

## Highlights & Insights
- This is the first work to advance reasoning capabilities in fully aligned instruction-tuned models via a self-play language game without requiring a stronger model as a teacher.
- Rewards are entirely derived from the game rules (the correctness of the final answer) without requiring human annotations, making it naturally scalable.
- Critics can freely choose the granularity of their criticism, achieving flexible natural language step-level supervision.
- CDG training can serve as a "preprocessing" step to enhance the efficacy of subsequent distillation, demonstrating general utility.
- The "arms race" dynamics between the Prover and the Misleading Critic during self-play training are intriguing and align with theoretical expectations.

## Limitations & Future Work
- The method is currently only validated in mathematical reasoning; generalizability to other reasoning tasks, such as code and logic, remains to be explored.
- Experiments are restricted to 8B and 1.5B models; scalability to larger models remains unknown.
- The improvement in the first round of self-play is limited, indicating a dependency on the baseline game-playing capabilities of the initial model.
- Pre-trained models require additional imitation learning steps to establish foundational game-playing capabilities.
- Future work could explore online RL (such as PPO) instead of offline ReST to achieve better training efficiency.

## Related Work & Insights
- It shares the closest resemblance to SPAG (Adversarial Taboo Game); however, while SPAG activates latent capabilities on pre-trained models, this work further enhances fully aligned models.
- Inspired by the self-play paradigm of AlphaGo Zero, extending game playing from board games to natural language reasoning.
- It can be viewed as a third reasoning supervision paradigm distinct from PRM/ORM: game-based self-supervision.
- Prover-Verifier Games share a similar concept, but the three-role design (cooperative + adversarial) presented in this work is more comprehensive.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play](../../ACL2026/llm_reasoning/stratagem_learning_transferable_reasoning_via_trajectory-modulated_game_self-pla.md)
- [\[ICCV 2025\] CoRVid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning](../../ICCV2025/llm_reasoning/corvid_improving_multimodal_large_language_models_towards_chain-of-thought_reaso.md)
- [\[ACL 2025\] ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations](../../ACL2025/llm_reasoning/clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)
- [\[ICML 2025\] Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration](soft_reasoning_navigating_solution_spaces_in_large_language_models_through_contr.md)
- [\[ICLR 2026\] StepORLM: A Self-Evolving Framework with Generative Process Supervision for Operations Research Language Models](../../ICLR2026/llm_reasoning/steporlm_a_self-evolving_framework_with_generative_process_supervision_for_opera.md)

</div>

<!-- RELATED:END -->
