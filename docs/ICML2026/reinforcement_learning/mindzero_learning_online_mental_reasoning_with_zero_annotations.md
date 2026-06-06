---
title: >-
  [Paper Note] MindZero: Learning Online Mental Reasoning with Zero Annotations
description: >-
  [ICML2026][Reinforcement Learning][Mental Reasoning] MindZero reformulates Bayesian Inverse Planning (BIP) into a "self-supervised RL" objective for multimodal LLMs. The reward model maximizes the likelihood of observed…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "Mental Reasoning"
  - "GRPO"
  - "Self-supervised RL"
  - "Variational Inference"
  - "Proactive Assistance"
date: 2026-05-08
content_hash: 7c19f16aa76011e2
---

# MindZero: Learning Online Mental Reasoning with Zero Annotations

**Conference**: ICML2026  
**arXiv**: [2606.00240](https://arxiv.org/abs/2606.00240)  
**Code**: https://scai.cs.jhu.edu/MindZero  
**Area**: Reinforcement Learning / Theory of Mind / Multimodal LLM Post-training  
**Keywords**: Mental Reasoning, GRPO, Self-supervised RL, Variational Inference, Proactive Assistance

## TL;DR
MindZero reformulates Bayesian Inverse Planning (BIP) into a "self-supervised RL" objective for multimodal LLMs. The reward model maximizes the likelihood of observed human actions based on generated mental hypotheses. Using GRPO training, small models achieve single-forward, fast, and robust online mental reasoning without requiring any mental annotations.

## Background & Motivation

**Background**: For AI to proactively assist humans in real-world environments, it must possess a strong Theory of Mind (ToM)—inferring goals or beliefs from behavior. Current approaches fall into three categories: (i) Prompting-based LLMs answering directly; (ii) Model-driven Bayesian Inverse Planning (BIP) explicitly enumerating hypotheses; (iii) Supervised learning directly fitting annotations.

**Limitations of Prior Work**: (1) Prompting methods systematically fail on long contexts and recursive reasoning; (2) BIP-based methods (e.g., AutoToM, ThoughtTracing) require searching the large hypothesis space and calling LLM evaluations at every step, often exceeding hundreds of TFLOPs per inference, making them unsuitable for real-time use; (3) Supervised learning is nearly impossible to scale as "ground-truth mental states" are rarely available in real home or web environments.

**Key Challenge**: Model-driven BIP is robust but slow, while LLM single-forward passes are fast but unreliable. Furthermore, the "ground-truth mental states" required for offline supervision are unattainable in open scenarios. This necessitates a new paradigm: preserving the deductive structure of BIP (using action likelihood to test mental hypotheses) during training, while compressing it into a single LLM forward pass for deployment.

**Goal**: (i) Train small multimodal LLMs for online, uncertainty-aware mental reasoning without any mental labels; (ii) Ensure the trained models are both accurate and real-time responsive in proactive assistance tasks.

**Key Insight**: The ELBO form of BIP, $\mathbb{E}_{Q_\theta}[\log P(a|m,s) P(m)] + H(Q_\theta)$, naturally depends only on the likelihood of the "action-state-hypothesis" triplet and not on a ground-truth $m^\star$. By treating this term as an RL reward, an LLM can "internalize" the deductive structure of BIP directly via GRPO.

**Core Idea**: Use "explaining observed human actions" as a self-supervised reward to distill Bayesian Inverse Planning into the LLM's policy distribution.

## Method

### Overall Architecture

MindZero trains a multimodal LLM $Q_\theta(\cdot \mid s_{1:t}, a_{1:t})$ that, given a historical state-action sequence, outputs $N$ mental hypotheses $\mathcal{M}_t = \{m_t^{(1)}, \dots, m_t^{(N)}\}$ and their normalized posteriors $\mathcal{Q}_t = \{q_t^{(1)}, \dots, q_t^{(N)}\}$ ($\sum q^{(i)} = 1$) in a single pass. An external "action likelihood evaluator" (a model-driven planner in GridWorld; an LLM in the home domain) provides $P(a_{1:t} \mid m_t^{(i)}, s_{1:t})$, while another LLM (or a uniform distribution) provides the prior $P(m_t^{(i)})$. These are synthesized into a reward for GRPO model updates. During deployment, the model performs one forward pass at each timestep to obtain the multi-hypothesis distribution, and a downstream helper plans assistive actions using $P(a^A_t | s_{1:t}, a_{1:t}) = \sum_{m_t} P(a^A_t | s_t, m_t) P(m_t | s_{1:t}, a_{1:t})$.

### Key Designs

1.  **Self-supervised RL Reward = ELBO**:
    - **Function**: Provides a learning signal for the LLM under zero mental annotations, allowing it to learn implicit supervision based on "mental hypotheses that explain human behavior."
    - **Mechanism**: For unlabeled trajectories, ToM is treated as variational inference $Q_\theta \approx P(m | s, a)$, maximizing the ELBO $\mathcal{J}(\theta) = \mathbb{E}_{Q_\theta}[\log(P(a_{1:t} | m_t, s_{1:t}) \cdot P(m_t))] + H(Q_\theta)$. For $N$ sampled hypotheses, this becomes $R(\mathcal{M}_t, \mathcal{Q}_t) = \sum_i q_t^{(i)} \log[P(a_{1:t} | m_t^{(i)}, s_{1:t}) P(m_t^{(i)})] - \sum_i q_t^{(i)} \log q_t^{(i)}$. GRPO is used for relative advantage updates within a group, avoiding the need for a critic.
    - **Design Motivation**: Traditional next-token prediction is "forward" (fitting $a | s$), whereas mental reasoning must be "inverse" inference (inferring $m | s, a$). These have different training objectives. Formulating the ELBO as a reward integrates inverse inference into the RL post-training pipeline without needing ground-truth $m^\star$.

2.  **Multi-hypothesis + Entropy Regularization**:
    - **Function**: Enables the model to maintain multiple competing mental hypotheses and explicitly provide confidence levels for each, preventing premature commitment to a wrong goal when evidence is insufficient.
    - **Mechanism**: The reward function forces the model to output $N$ hypotheses and normalized posteriors $\{q_t^{(i)}\}$ simultaneously. The entropy term $H(Q_\theta) = -\sum_i q_t^{(i)} \log q_t^{(i)}$ directly penalizes "single-point collapse." This allows the downstream helper to weight actions by $P(a^A | m) \cdot q(m)$ in ambiguous scenarios.
    - **Design Motivation**: The robustness of traditional BIP stems from explicitly tracking the posterior distribution of multiple hypotheses. Single-point prediction in early, ambiguous stages (e.g., in GridWorld) causes the helper to act prematurely in the wrong direction. Ablations show an 8.8% drop in speedup without multi-hypothesis and a 13.9% drop without entropy regularization—the entropy term is the critical switch against collapse.

3.  **Explicit Prior Modeling + Dual Estimators**:
    - **Function**: Uses an LLM as a prior estimator to constrain the hypothesis space and avoid reward hacking; uses a domain-adapted likelihood estimator (model-based planner/LLM) to provide Bayesian evidence.
    - **Mechanism**: The prior $P(m_t)$ is generated by an LLM scoring the "commonsense reasonableness" of candidate goals (e.g., "putting an apple in the dishwasher" receives a very low score). The likelihood $P(a_{1:t} | m_t, s_{1:t})$ is estimated using a model-based planner in GridWorld and a pre-trained LLM in the home domain. Both are integrated into the $\log$ of the ELBO reward.
    - **Design Motivation**: Using only the likelihood term might lead the model to generate overly broad goals (e.g., "pick up everything on the table") to make any behavior "reasonable," achieving high scores via reward hacking. The prior term closes this shortcut. Decoupling the two estimators also allows the framework to migrate to new domains by simply replacing the planner or prompt.

### Loss & Training

The reward $R$ is calculated as per the above formula. GRPO updates $Q_\theta$ using normalized advantages within a group without a value network. Small models like Llama-3.2-3B, which struggle with consistent formatting, undergo SFT warmup using self-sampled data from Llama-3.1-8B (learning format only, no ground-truth) before entering the RL phase.

## Key Experimental Results

### Main Results

Four evaluations: GridWorld QA, GridWorld Proactive Assistance, Home QA (MMToM-QA), and Home Proactive Assistance (O-WAH). The Speedup metric measures the acceleration of task completion compared to a "human alone" baseline.

| Scenario | Baseline (Same Backbone) | MindZero (Ours) | Notes |
|----------|--------------------------|-----------------|-------|
| GridWorld Proactive (Qwen3-VL-4B) | 1.4% speedup | 23.0% speedup | Order of magnitude improvement |
| GridWorld Proactive (Qwen3-VL-8B) | -0.1% speedup | 24.5% speedup | 8B base model shows negative speedup due to instruction instability |
| Home Proactive (Qwen3-4B) | 2.3% speedup | 19.1% speedup | Surpasses GPT-5.2 (9.4%) and Gemini-3-Flash (17.7%) |
| Home Proactive (Llama-3.1-8B) | 1.7% speedup | 17.4% speedup | Outperforms Gemini-3-Flash with approx. 1/2 the compute |
| QA Accuracy (avg.) | Base Model | 2.1–2.5× | Accuracy multiplier relative to the same backbone |

### Ablation Study (Qwen3-4B / Home Proactive Assistance)

| Configuration | Speedup ↑ | TFLOPs ↓ | Description |
|---------------|-----------|----------|-------------|
| MindZero (Full) | 19.1% | 201.2 | All three designs active |
| w/o Explicit Prior | 17.0% | 200.5 | Tendency toward reward hacking observed |
| w/o Multi-hypothesis | 10.3% | 132.6 | Single-point prediction, premature commitment |
| w/o Entropy Reg. | 5.2% | 245.1 | Mode collapse, higher token count but worst performance |

### Key Findings

- **Small Model + MindZero ≥ Large Model Zero-shot**: Qwen3-4B trained with MindZero achieved 19.1% speedup in home assistance, exceeding Qwen3-235B-A22B (12.3%) and GPT-5.2 (9.4%), and matching Gemini-3-Flash (17.7%) at a fraction of the compute cost.
- **Large Models Fail on GridWorld**: GPT-5.2, Gemini-3-Flash, and Qwen3-235B all showed speedup ≤ 1% in GridWorld proactive assistance because their goal predictions fluctuated every step, causing the agent to oscillate. MindZero's 24.5% speedup highlights that "online reasoning stability" is far more important than "single-step reasoning quality."
- **Entropy > Multi-hypothesis > Prior**: Based on ablation drops, the entropy term for preventing collapse is the "soul" of MindZero, while the prior term acts as a "fuse" to block reward hacking.
- **Human Study Validation**: 12 JHU participants using the MindZero (Qwen3-4B) assistant achieved a 19.7% actual speedup, with no statistically significant difference from Gemini-3-Flash (23.4%, $p=0.24$).

## Highlights & Insights

- **Rewriting BIP as RL Reward**: The core contribution is "treating variational ELBO as a GRPO reward." This allows Bayesian Inverse Planning to be "internalized" by an LLM as a single-forward pass without any annotations—a long-awaited missing link in the field.
- **"Explicit Hypotheses + Posterior" Format as a Contribution**: While many ToM-LLM works use implicit reasoning, MindZero's forced schema of $N$ pairs of $(m^{(i)}, q^{(i)})$ provides a direct target for entropy regularization and multi-hypothesis collapse prevention, enabling true belief-state planning for downstream helpers.
- **Transferable Trick**: The "ELBO-as-reward + GRPO" paradigm can be applied to other LLM post-training tasks requiring inverse inference, such as code bug tracing, user intent inference in dialogue, or robot fault diagnosis—provided one can formulate $P(\text{evidence}|\text{hypothesis})$.

## Limitations & Future Work

- **No Modeling of Recursive Reasoning**: The current framework handles only first-order ToM (inferring a single user's mind) and cannot handle higher-order "A thinks that B thinks..." reasoning, limiting its use in complex multi-agent collaboration.
- **Linear Input Growth over Timesteps**: Concatenating $s_{1:t}, a_{1:t}$ into the prompt means the token count scales with time, which might exceed the context window in long-horizon tasks.
- **Dependence on External Likelihood Evaluators**: While model-based planners work for small GridWorlds, LLM-based likelihood estimation in home domains is not cheap. Biased evaluators (e.g., an LLM unfamiliar with a specific object) could mislead the entire RL process. The SFT warmup required for Llama-3.2-3B also risks memorizing low-quality distributions, a "cold-start + bias" problem yet to be fully solved.

## Related Work & Insights

- **vs BIP / AutoToM / ThoughtTracing**: BIP-style methods explicitly enumerate hypotheses and call LLM evaluations per step—robust but slow. MindZero distills this deductive structure into LLM weights, achieving over an order of magnitude speedup during deployment.
- **vs Supervised Learning ToM (Rabinowitz 2018, etc.)**: These require ground-truth mental labels and are usually restricted to controlled environments. MindZero uses "behavior itself" as the supervisory signal, liberating data collection.
- **vs DeepSeek-R1 / GRPO in Math/Code**: While both use GRPO for post-training, MindZero rewards "likelihood of hypotheses explaining evidence" rather than "correct answer matching," systematically extending RL post-training to the "inverse inference" task family.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Rewriting BIP as a self-supervised RL reward is a concise but pivotal paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four evaluations + two backbones + three ablations + 12-person human study; covers essential ground but lacks large-scale open-domain evaluation and higher-order ToM.
- Writing Quality: ⭐⭐⭐⭐ Clear hierarchy in problem definition, reward derivation, and experimental design; reasoning for dual estimator selection could be more detailed.
- Value: ⭐⭐⭐⭐⭐ Demonstrates a 4B small model reaching Gemini-3-Flash levels in proactive assistance, providing a practical blueprint for ToM in open-source assistant agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Unlocking Zero-Shot Geospatial Reasoning via Indirect Rewards](unlocking_zero-shot_geospatial_reasoning_via_indirect_rewards.md)
- [\[ICLR 2026\] REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](../../ICLR2026/reinforcement_learning/rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)
- [\[ICML 2026\] Bilevel Optimization over Saddle Points of Zero-Sum Markov Games](bilevel_optimization_over_saddle_points_of_zero-sum_markov_games.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](../../ICLR2026/reinforcement_learning/spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[ICLR 2026\] LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning](../../ICLR2026/reinforcement_learning/longwriter-zero_mastering_ultra-long_text_generation_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
