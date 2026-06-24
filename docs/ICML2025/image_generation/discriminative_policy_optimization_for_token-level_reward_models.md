---
title: >-
  [Paper Note] Discriminative Policy Optimization for Token-Level Reward Models
description: >-
  [ICML2025][Image Generation][token-level reward model] Proposes the Q-function Reward Model (Q-RM), which decouples reward modeling from language generation by defining a discriminative policy to learn token-level $Q$-functions. This approach extracts precise token-level reward signals from preference data without needing fine-grained annotations, significantly improving the reasoning performance and training efficiency of PPO/REINFORCE.
tags:
  - "ICML2025"
  - "Image Generation"
  - "token-level reward model"
  - "process reward model"
  - "Q-function"
  - "discriminative policy"
  - "PPO"
  - "REINFORCE"
  - "LLM alignment"
date: 2026-05-08
content_hash: fb01b7eb76a9176f
---

# Discriminative Policy Optimization for Token-Level Reward Models

**Conference**: ICML2025  
**arXiv**: [2505.23363](https://arxiv.org/abs/2505.23363)  
**Code**: [homzer/Q-RM](https://github.com/homzer/Q-RM)  
**Area**: Image Generation  
**Keywords**: token-level reward model, process reward model, Q-function, discriminative policy, PPO, REINFORCE, LLM alignment

## TL;DR

Proposes the Q-function Reward Model (Q-RM), which decouples reward modeling from language generation by defining a discriminative policy to learn token-level $Q$-functions. This approach extracts precise token-level reward signals from preference data without needing fine-grained annotations, significantly improving the reasoning performance and training efficiency of PPO/REINFORCE.

## Background & Motivation

- **Process Reward Models (PRMs)** provide finer-grained, step-by-step feedback compared to Outcome Reward Models (ORMs), but suffer from a granularity mismatch: PPO operates at the token level, whereas ORMs/PRMs assign rewards at the sequence or step level.
- **DPO-RM** formulations define the reward as $r^{\text{DPO}}(s_t, a_t) = \beta \log \frac{\pi(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}$, which suffers from two core limitations:
    1. Coupling of generative language modeling and reward modeling: high generation probability $\neq$ high reward (e.g., the model can be highly confident in incorrect answers).
    2. Dependency on the reference model $\pi_{\text{ref}}$ introduces additional uncertainty, leading to anomalous reward allocation.
- **Visual Evidence**: DPO-RM tends to assign high rewards to non-critical tokens such as newline characters while ignoring key numerical tokens (e.g., "$7", "$133"). In contrast, Q-RM correctly assigns high scores to correct tokens and low scores to incorrect ones.

## Method

### 1. Discriminative Policy Definition

Unlike the generative policy $\pi(a_t|s_t)$, a discriminative policy $\phi(s_t, a_t)$ is defined to take both state and action as input, converting the logit $Z(s_t, a_t)$ into a probability via softmax:

$$\phi(s_t, a_t) = \frac{\exp Z(s_t, a_t)}{\sum_{a'_t \in \mathcal{A}} \exp Z(s_t, a'_t)}$$

Core difference: The generative policy outputs a probability distribution over all actions, whereas the discriminative policy evaluates the reward of a specific action.

### 2. Reward Derivation

Under the maximum entropy RL framework, the optimal discriminative policy satisfies:

$$\beta \log \phi^*(s_t, a_t) = Q^*(s_t, a_t) - V^*(s_t)$$

Combining this with the Bellman equation yields the token-level reward: $r(s_t, a_t) = \beta \log \phi^*(s_t, a_t) + V^*(s_t) - V^*(s_{t+1})$.

### 3. Trajectory Reward Decomposition and Simplification

The trajectory reward $\mathcal{R}(\tau)$ is decomposed into $\beta(\mathcal{Q}(\tau) - \mathcal{V}(\tau))$, where:

- $\mathcal{Q}(\tau) = \frac{1}{T}\sum_{t=0}^{T-1}(Z^*(s_t, a_t) - z_t)$, where $z_t = \max_{a_t} Z^*(s_t, a_t)$
- $\mathcal{V}(\tau)$ is the log-mean of the adjusted partition function.

**Key Theoretical Contribution**: Proves that the upper bound of $\mathcal{V}(\tau)$ is constrained by the entropy of the optimal policy ($0 \leq \mathcal{V}(\tau) \leq \mathcal{H}^*(\tau)$). When the optimal policy is approximately deterministic, $|\mathcal{V}(\tau^w) - \mathcal{V}(\tau^l)| \to 0$, meaning it can be safely ignored.

### 4. Training Objective

The final loss function is based on the Bradley-Terry model:

$$p(\tau^w \succeq \tau^l) = \sigma\left[\beta\left(\frac{1}{N}\sum_{t=0}^{N-1}Z^*(s_t^w, a_t^w) - \frac{1}{M}\sum_{t=0}^{M-1}Z^*(s_t^l, a_t^l)\right) - \gamma\right]$$

where $\gamma$ is a global bias hyperparameter (fixed to 2), and $\beta = 0.2$. It only requires preference data for training, without needing fine-grained annotations.

### 5. Integration of Q-RM with PPO/REINFORCE

- **PPO**: Directly computes the advantage function $A(s_t, a_t) = Z^*_{\text{std}}(s_t, a_t) - V_\psi(s_t)$, bypassing GAE.
- **REINFORCE**: Uses $Z^*_{\text{std}}(s_t, a_t)$ as the cumulative reward.
- Normalizes all token rewards (mean 0, variance 1) to ensure training stability.

### 6. Theoretical Guarantees

**Proposition 3.4**: The optimal Q-function $Q^*(s_t, a_t)$ and the discriminative policy logit $Z^*(s_t, a_t)$ have consistent bias expectations, meaning that using $Z^*$ to compute the advantage function is equivalent to using $Q^*$.

## Key Experimental Results

**Setup**: Policy model Llama-3.2-3B-Instruct, reward model Llama-3-70B-Instruct, LoRA rank 128, learning rate 1e-5.

### Mathematical Reasoning (GSM8K & MATH)

| Method | GSM8K Pass@1 | GSM8K Pass@16 | MATH Pass@1 | MATH Pass@16 | Avg Pass@1 |
|------|-------------|---------------|-------------|--------------|------------|
| SFT | 63.08 | 87.95 | 27.57 | 55.48 | 45.33 |
| DPO | 68.16 | 91.13 | 29.80 | 58.44 | 48.98 |
| PPO+ORM | 66.26 | 88.02 | 27.22 | 56.59 | 46.74 |
| PPO+DPO-RM | 68.67 | 88.02 | 27.39 | 55.72 | 48.03 |
| **PPO+Q-RM** | **72.23** | 92.49 | **32.95** | **64.19** | **52.59** |
| REINFORCE+ORM | 67.55 | 89.69 | 29.60 | 57.86 | 48.58 |
| **REINFORCE+Q-RM** | **72.10** | **93.48** | **34.45** | 62.87 | **53.28** |

- PPO+Q-RM improves average Pass@1 by **+5.85** over ORM and by **+4.56** over DPO-RM.
- REINFORCE+Q-RM improves by **+4.70** over ORM and by **+5.73** over DPO-RM.

### QA-Feedback

| Method | Relevance | Factuality | Completeness | Avg |
|------|-----------|------------|--------------|-----|
| PPO+Q-RM | **0.5510** | 0.6814 | **0.5545** | **0.5956** |
| REINFORCE+Q-RM | 0.5454 | 0.6808 | 0.5490 | 0.5917 |
| PPO+DPO-RM | 0.4769 | 0.6802 | 0.5323 | 0.5631 |

### Training Efficiency

- Q-RM converges **12× faster** than ORM on GSM8K.
- Q-RM converges **11× faster** than step-level PRM on MATH.

## Highlights & Insights

1. **Elegant Decoupling Idea**: Decouples reward modeling from language generation, using a discriminative policy instead of a generative policy to model rewards. This fundamentally resolves the "high generation probability $\neq$ high reward" conflict.
2. **Theoretical Completeness**: Proves that the logit $Z^*$ and the optimal Q-function share a consistent bias structure, allowing direct substitution of $Q$-values to compute the advantage function without GAE.
3. **High Practicality**: Requires no fine-grained annotations and can be trained using only preference data; fixing $\gamma$ as a constant works across different tasks.
4. **Leap in Training Efficiency**: Achieves an 11-12× speedup in convergence, greatly reducing the cost of RL training.
5. **Clear Intuition from Reward Visualization**: Q-RM precisely identifies key tokens (assigning high scores to correct values and low scores to incorrect values), whereas DPO-RM is sensitive to noise tokens like newlines.

## Limitations & Future Work

1. **Applicability of Assumption 3.3**: The assumption that the entropy of the optimal policy approaches zero may not hold in creative generation or high-diversity scenarios.
2. **Fixed $\gamma$ Constant**: In practice, $\gamma$ varies across samples; fixing it as a constant is an approximation that may not be ideal for preference pairs with extreme length differences.
3. **Dependence on Reward Model Scale**: The experiments employ a 70B reward model with a 3B policy model; its performance in resource-constrained scenarios remains to be fully verified.
4. **Evaluation Biased Toward Mathematical Reasoning**: Generalizability to tasks like code generation and open-domain dialogue requires further validation.
5. **Softmax Computation for Discriminative Policy**: It still requires iterating over the entire vocabulary for normalization, and the theoretical simplification relies on specific assumptions.

## Related Work & Insights

- **DPO/SimPO/ORPO**: Offline RL alignment methods, with which Q-RM is complementary as an online RL reward model.
- **Implicit-PRM (CE)**: Also a token-level PRM but based on a generative policy; Q-RM achieves better performance via discriminative decoupling.
- **PGG-RM**: Another token-level reward method, which Q-RM outperforms across nearly all metrics.
- **Insight**: The paradigm of decoupling discriminative and generative modeling can be extended to other scenarios requiring fine-grained credit assignment (e.g., code debugging, long text generation).

## Rating

- Novelty: ⭐⭐⭐⭐ (Decoupling reward modeling via discriminative policies offers a novel perspective)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 tasks, multiple baselines, evaluated across both PPO and REINFORCE RL frameworks)
- Writing Quality: ⭐⭐⭐⭐ (Theoretical derivations are clear, and the experiments are systematically organized)
- Value: ⭐⭐⭐⭐⭐ (Provides a substantial contribution to token-level reward modeling in LLM alignment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Group Critical-token Policy Optimization for Autoregressive Image Generation](../../ICLR2026/image_generation/group_critical-token_policy_optimization_for_autoregressive_image_generation.md)
- [\[ICML 2025\] PPO-MI: Efficient Black-Box Model Inversion via Proximal Policy Optimization](ppo-mi_efficient_black-box_model_inversion_via_proximal_policy_optimization.md)
- [\[ICML 2025\] ToMA: Token Merge with Attention for Diffusion Models](toma_token_merge_with_attention_for_diffusion_models.md)
- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](../../NeurIPS2025/image_generation/diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [\[ICML 2025\] Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is also a GAN Discriminator](direct_discriminative_optimization_your_likelihood-based_visual_generative_model.md)

</div>

<!-- RELATED:END -->
