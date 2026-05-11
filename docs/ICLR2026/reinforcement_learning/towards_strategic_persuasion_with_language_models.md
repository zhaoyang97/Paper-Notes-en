---
title: >-
  [Paper Note] Towards Strategic Persuasion with Language Models
description: >-
  [ICLR 2026][Reinforcement Learning][Bayesian Persuasion] Grounded in the Bayesian Persuasion framework, this paper proposes a systematic methodology for evaluating and training the strategic persuasion capabilities of LL…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Bayesian Persuasion"
  - "Large Language Models"
  - "Strategic Persuasion"
  - "Information Design"
  - "Reinforcement Learning Training"
date: 2026-05-08
content_hash: 714a4c84b01b3881
---

# Towards Strategic Persuasion with Language Models

**Conference**: ICLR 2026
**arXiv**: [2509.22989](https://arxiv.org/abs/2509.22989)
**Code**: None
**Area**: Reinforcement Learning / LLM Capability Evaluation
**Keywords**: Bayesian Persuasion, Large Language Models, Strategic Persuasion, Information Design, Reinforcement Learning Training

## TL;DR

Grounded in the Bayesian Persuasion framework, this paper proposes a systematic methodology for evaluating and training the strategic persuasion capabilities of LLMs. It finds that frontier models already exhibit significant strategic persuasion ability, and that even small LLMs can substantially improve their persuasive performance through reinforcement learning.

## Background & Motivation

Large language models have demonstrated persuasive capabilities comparable to humans, presenting both important opportunities (e.g., health promotion, education) and risks (e.g., manipulation, misinformation). However, **systematically evaluating LLM persuasion capabilities** faces a fundamental challenge: persuasion effects among humans are inherently heterogeneous — advertisements may influence inexperienced consumers but not seasoned ones, and political messaging often reinforces rather than changes existing beliefs.

Key limitations of existing evaluation approaches:

**Lack of theoretical grounding**: Most prior work relies on human ratings or automated scoring to measure persuasiveness, but inconsistent evaluation setups and metrics yield contradictory results (Bozdag et al., 2025b).

**Poor scalability**: Human evaluation is costly and subjective. Durmus et al. (2024) find that model-generated persuasiveness scores correlate weakly with human judgments.

**Absence of training methodology**: No principled approach exists for improving LLM persuasion capabilities.

The central innovation of this paper is the introduction of **Bayesian Persuasion** from game theory, defining persuasiveness as **the sender's ability to update the receiver's beliefs through strategic information disclosure**, thereby yielding a conceptually clear, quantifiable, and scalable evaluation standard.

## Method

### Overall Architecture

The classical **Bayesian Persuasion** setup:

- **Sender**: Observes the true state of the world $\omega \in \Omega$ and designs a signaling strategy $\pi: \Omega \to \Delta(S)$
- **Receiver**: Observes signal $s$, updates posterior $\mu_s(\omega)$ via Bayesian inference, and selects the optimal action $a^*(\mu_s) \in \arg\max_{a} \mathbb{E}_{\omega \sim \mu_s}[u(a,\omega)]$
- **Key theoretical result**: Kamenica & Gentzkow (2011) prove that the sender's optimal value equals the concave closure of their utility function evaluated at the prior

The paper integrates this theory with LLMs: an LLM serves as the Sender, another as the Receiver, and the evaluation environment is constructed around an opinion-change task.

### Key Designs

#### 1. **Measuring Persuasion**

Two core metrics are defined:

- **Persuasion Gains**: $\Delta\hat{v}(\mu_0) = \hat{v}(\mu) - \hat{v}(\mu_0)$, i.e., the improvement in sender utility from the induced posterior relative to the prior
- **Persuasion Signals**: The degree of information disclosure at each time step, measured via conditional mutual information $I(M_t; \Omega_t | \mathcal{H}_{t-1})$

Design Motivation: Persuasion Gains directly quantify the economic effect of persuasion; Persuasion Signals reveal whether the model exhibits strategic information control (high values indicate adaptive, context-dependent signaling; low values indicate deliberate concealment).

#### 2. **Environment Construction**

Four human persuasion datasets are repurposed to construct the evaluation environment:
- **Anthropic dataset** (Durmus et al., 2024): pro and con arguments on controversial topics
- **DDO dataset** (Durmus & Cardie, 2019): debate data from debate.org
- **Perspectrum dataset** (Chen et al., 2019): claims, perspectives, and evidence from online debate sites
- **CMV dataset** (Tan et al., 2016): large-scale debate data from Reddit r/ChangeMyView

The receiver's action space is defined as a 7-point Likert scale (from "strongly disagree" to "strongly agree"), with score mapping function $g(a_i) = i$.

**Human validation**: 45 participants were recruited via the Prolific platform to assess whether LLM receiver belief updates were reasonable. Results show directional accuracy of 77–85% and a proportionality rating of approximately 5/7.

#### 3. **RL Training Framework**

Persuasion is modeled as an RL problem:
- **State**: Persuasion context $(\mu_0, u, v, A, \omega)$
- **Action**: Message $m = (m_1, \ldots, m_T)$ generated by the Sender LLM
- **Reward**: $r(\omega, m, a) = v(a, \omega) - \hat{v}(\mu_0)$ (positive reward = successful persuasion)
- **Fixed Receiver**: Receiver parameters $\phi$ are frozen during training; only Sender parameters $\theta$ are updated

PPO and GRPO algorithms are implemented using the verl framework to train Llama-3.2-3B-Instruct, with Llama-3.1-8B-Instruct serving as the Receiver.

### Loss & Training

Training objective: $J(\theta) = \mathbb{E}_{s_0 \sim \mathcal{D}, m \sim \pi_\theta(\cdot|s_0), a \sim \rho(\cdot|m, s_0)}[R(s_0, m, a)]$

Hyperparameters: learning rate $5 \times 10^{-7}$, batch size 4, KL coefficient 0.001, Adam optimizer, approximately 2,700 training instances, 4× NVIDIA A6000 GPUs.

## Key Experimental Results

### Main Results

Persuasion gains for different Sender models (Receiver: Llama-3.1-8B-Instruct):

| Sender Model | Static Mean | Dynamic Mean | Static Best | Dynamic Best |
|---|---|---|---|---|
| Llama-3.1-8B | 0.04 | 0.42 | 0.12 | 0.47 |
| Mistral-7B | 0.01 | 0.31 | 0.11 | 0.60 |
| Qwen2.5-7B | 0.02 | 0.23 | 0.08 | 0.51 |
| Llama-3.3-70B | 0.06 | 0.44 | 0.11 | 0.61 |
| GPT-4o | 0.06 | 0.62 | 0.15 | 0.75 |
| Claude 3.7 Sonnet | 0.14 | 1.04 | 0.28 | 1.30 |
| **DeepSeek-R1** | **0.23** | **1.27** | **0.29** | **1.53** |

### Ablation Study

Persuasion gains before and after RL training for Llama-3.2-3B-Instruct (Receiver: Llama-3.1-8B-Instruct):

| Configuration | Static Mean | Dynamic Mean |
|---|---|---|
| Base (3B) | -0.01 | 0.21 |
| + PPO | 0.03 | 0.38 |
| + GRPO | 0.03 | 0.38 |

Against Mistral-7B Receiver: PPO improves the mean from 1.21 to 1.45; GRPO improves it to 1.37.

### Key Findings

1. **Positive correlation with model scale**: Larger models (DeepSeek-R1, Claude 3.7 Sonnet) substantially outperform smaller ones on the persuasion task. DeepSeek-R1 achieves an average dynamic gain of 1.27, representing 18.14% of the full utility scale.

2. **Dynamic settings far outperform static**: Model persuasiveness in multi-turn interactions is substantially stronger than in single-turn settings. This is not solely a function of model quality, but also of **interaction structure** — the ability to deploy adaptive strategies is critical.

3. **RL training is effective**: Even a 3B-parameter model achieves persuasive performance approaching that of much larger models after RL training, with **strong transferability** — strategies trained against Llama-8B generalize effectively to Mistral-7B and Qwen2.5-7B.

4. **Strategic information disclosure**: Stronger models exhibit lower semantic similarity across messages (greater inter-message diversity), suggesting they adaptively adjust their information strategies based on context, consistent with predictions from Bayesian Persuasion theory.

5. **Dominant strategy types**: Evidence, credibility, and impact are the most frequently employed strategies, consistent with theoretically expected information revelation strategies.

## Highlights & Insights

1. **Theory–practice bridge**: This is the first work to systematically apply the classical game-theoretic framework of Bayesian Persuasion to LLM capability evaluation, providing a conceptually clear and quantifiable measure of persuasiveness.

2. **Evidence for emergent strategic behavior**: Frontier models not only "know how to talk" but also exhibit complex strategic behaviors predicted by theory (e.g., adaptive information disclosure, prior-conditioned strategy adjustment), with important implications for AI safety.

3. **Universal effectiveness of RL training**: Persuasion capability can be systematically improved via RL and transfers across receiver architectures, suggesting that models learn genuine strategies rather than overfitting to specific architectures.

4. **Thorough ethical consideration**: The paper emphasizes that the framework focuses on truthful information disclosure (not deception) and discusses safeguards, demonstrating a responsible research posture.

## Limitations & Future Work

1. **Restricted to opinion-change tasks**: The Bayesian Persuasion framework encompasses far more — variants involving multiple receivers, multiple senders, and dynamic environments remain unexplored.
2. **Limitations of LLM receivers**: LLMs are not perfect Bayesian updaters; their belief updates may exhibit systematic deviations from human behavior.
3. **Ecological validity of the evaluation environment**: Although human validation confirms the directional plausibility of belief updates, the dynamics of LLM–LLM interaction may differ qualitatively from human–LLM interaction.
4. **Limited training scale**: Only a 3B model was trained; the effects of RL training on larger models remain unknown.
5. **Safety implications**: Techniques for enhancing LLM persuasion capabilities could be misused for manipulation and information warfare.

## Related Work & Insights

- **Bayesian Persuasion theory** (Kamenica & Gentzkow, 2011): The source of the core framework; the concave closure theorem provides a theoretical upper bound.
- **LLM persuasion evaluation** (Durmus et al., 2024; Salvi et al., 2024): This paper builds upon these works to provide a more theoretically grounded and systematic evaluation.
- **LLMs in strategic reasoning** (Xu et al., 2024; Zhang et al., 2025): This paper extends the scope of evaluating LLM strategic reasoning capabilities.
- **Implications for AI safety**: Persuasion capability is an important dimension of LLM potential risk; the framework proposed here can be used to systematically monitor and assess this risk.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (innovative intersection of game theory and LLMs)
- Experimental Thoroughness: ⭐⭐⭐⭐ (multiple models, multiple datasets, RL training, human validation)
- Writing Quality: ⭐⭐⭐⭐ (clear theoretical framework, well-organized experiments)
- Value: ⭐⭐⭐⭐⭐ (significant contributions to both LLM capability evaluation and AI safety)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RebuttalAgent: Strategic Persuasion in Academic Rebuttal via Theory of Mind](rebuttalagent_strategic_persuasion_in_academic_rebuttal_via_theory_of_mind.md)
- [\[ICLR 2026\] Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models](remix_reinforcement_routing_for_mixtures_of_loras_in_llm_finetuning.md)
- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)
- [\[ICLR 2026\] AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)

</div>

<!-- RELATED:END -->
