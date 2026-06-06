---
title: >-
  [Paper Note] Universal Reasoner: Composable Plug-and-Play Reasoners for Frozen LLMs
description: >-
  [ICML 2026][LLM/NLP][Reasoning Enhancement] Proposes Universal Reasoner (UniR)—training independent lightweight reasoning modules to capture reward-oriented reasoning behaviors…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Reasoning Enhancement"
  - "Modular Reasoning"
  - "Composable Reasoning"
  - "Frozen LLM"
  - "Verifiable Rewards"
date: 2026-05-08
content_hash: 6f67c316d44d5f0f
---

# Universal Reasoner: Composable Plug-and-Play Reasoners for Frozen LLMs

**Conference**: ICML 2026  
**arXiv**: [2505.19075](https://arxiv.org/abs/2505.19075)  
**Code**: https://github.com/hangeol/UniR  
**Area**: LLM Reasoning  
**Keywords**: Reasoning Enhancement, Modular Reasoning, Composable Reasoning, Frozen LLM, Verifiable Rewards

## TL;DR
Proposes Universal Reasoner (UniR)—training independent lightweight reasoning modules to capture reward-oriented reasoning behaviors, combining them with frozen LLMs via logit superposition at inference time to achieve reasoning enhancement without fine-tuning frozen models, cross-scale transfer, and multi-task composition.

## Background & Motivation

**Background**: Current methods enhance LLM reasoning capabilities through RL fine-tuning (RFT), but require massive compute and memory resources. PEFT methods like LoRA attempt to reduce costs, but suffer from two major flaws: (1) strong dependency on model architecture, leading to poor transferability across different scales (e.g., 3B to 14B); (2) lack of theoretical support for the linear combination of multiple LoRA adapters.

**Limitations of Prior Work**: Inability to flexibly and efficiently enhance reasoning capabilities without accessing LLM internal parameters; inability to reuse trained reasoning capabilities across different model scales; inability to compose multiple reasoning modules for different tasks.

**Key Challenge**: Reasoning enhancement requires parameter updates (traditional fine-tuning) while models are often frozen; multi-task learning requires end-to-end retraining (facing multi-objective conflicts).

**Goal**: Design modular, transferable, and composable reasoning enhancement methods.

**Key Insight**: Verifiable rewards (e.g., correctness in math problems) can be converted from trajectory-level signals to token-level guidance. By modeling trajectory-level rewards as the sum of log-probabilities of a reasoning module, logit superposition becomes a natural mechanism for composition.

**Core Idea**: Separate reward model training from policy updates. Train a dedicated reasoning module $\pi_r$ to maximize verifiable rewards, and guide token generation during inference by adding its logit $\log\pi_r$ to the logit of the frozen backbone $\pi_b$.

## Method

### Overall Architecture
UniR decomposes LLM reasoning enhancement into two stages: the **Training Phase** trains a reasoning module $\pi_r$ on a smaller backbone model using verifiable rewards and the GRPO algorithm; the **Inference Phase** combines the trained $\pi_r$ with any frozen LLM through logit superposition for token-level guidance.

### Key Designs

1.  **Theoretical Mapping of Trajectory Reward to Token Guidance**:
    - **Function**: Decomposes global trajectory rewards into guidance signals for each token.
    - **Mechanism**: Assumes trajectory rewards can be represented as the sum of log-probabilities of the reasoning module: $\frac{1}{\beta}r(x,y)=\sum_{t=1}^{|y|}\log\pi_r(y_t|x,y_{<t};\phi)$. By substituting rewards in the KL-regularized objective, the optimal policy is derived as $\log\pi_\theta(y_t|x,y_{<t})=\log\pi_b(y_t|x,y_{<t})+\log\pi_r(y_t|x,y_{<t})-\log Z'(x,y_{<t})$. Theorem 4.1 proves that at convergence, $\log\pi_r(y_t|x,y_{<t})=\frac{1}{\beta}Q^*(y_t|x,y_{<t})$.
    - **Design Motivation**: Traditional trajectory-level rewards cannot directly guide each generation step; this structural hypothesis converts them into token-level guidance.

2.  **GRPO Training for Reasoning Modules**:
    - **Function**: Trains the reasoning module to maximize verifiable rewards without modifying the frozen backbone.
    - **Mechanism**: Employs GRPO to sample $G$ candidate responses from $\pi_b$, computes the external reward $r_i$ for each response, normalizes the advantage $A_i=\frac{r_i-\text{mean}(\{r_1,...,r_G\})}{\text{std}(...)}$, and optimizes $\phi$ on the GRPO objective. The ratio term in the gradient automatically eliminates backbone terms, affecting only $\pi_r$.
    - **Design Motivation**: GRPO does not require an explicit value function and is robust to sparse rewards, making it suitable for verifiable reward scenarios.

3.  **Composability of Multi-module Logit Superposition**:
    - **Function**: Supports the seamless combination of multiple reasoning modules from different tasks during inference.
    - **Mechanism**: For $N$ different reward functions $\{r_1,...,r_N\}$, $N$ reasoning modules $\{\pi_r^1,...,\pi_r^N\}$ are trained separately. During inference, they are combined as $\log\pi_\theta(y_t|x,y_{<t})\propto\log\pi_b(y_t|x,y_{<t})+\sum_{i=1}^{N}\alpha_i\log\pi_r^i(y_t|x,y_{<t})$, where weights $\alpha_i$ can be dynamically adjusted.
    - **Design Motivation**: Reasoning often involves multiple constraints; logit superposition serves as both a principled solution and a zero-cost composition method.

## Key Experimental Results

### Main Results

| Model Configuration | GSM8K | MATH-500 | AIME24 | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Llama3.2-3B Backbone | 65.6 | 33.0 | 3.7 | Baseline |
| + GRPO LoRA (3B) | 65.8 | 32.1 | 6.0 | -0.3% |
| UniR (1B + 3B) | 77.5 | 48.8 | 7.3 | +35.2% |
| Qwen2.5-3B Backbone | 74.4 | 44.2 | 6.3 | Baseline |
| UniR (1.5B + 3B) | 75.6 | 48.3 | 8.1 | +6.8% |

### Ablation Study

| Experiment Item | Description | Performance |
| :--- | :--- | :--- |
| Reasoning Module Alone | 1B $\pi_r$ independent generation | Far below combined |
| Without Reasoning Module | Frozen 3B backbone only | 65.6 (GSM8K) |
| + Reasoning Module (Tuned) | After combination | 77.5 |
| Multi-module ($\alpha_1=1, \alpha_2=0.5$) | Math + Translation weighting | 72.3 |

### Key Findings
- **Weak-to-strong transfer**: A 1B reasoning module can effectively guide a larger model (14B) without retraining.
- **Composability verification**: Weighted combinations of multiple modules perform robustly under different weight settings.
- **Reward internalization**: After training, the log-probability of the reasoning module for correct responses ($r=1$) is significantly higher than for incorrect ones ($r=0$).

## Highlights & Insights
- **Elegant Theoretical Foundation**: Rigorous derivation of logit superposition starting from KL-regularized multi-objective optimization.
- **Zero-Retraining Transfer**: Reasoning modules can transfer across model scales without fine-tuning.
- **Inference-Time Composition Flexibility**: Weighted fusion of multiple modules is instantly adjustable during the inference phase.
- **Empirical Q-function Learning**: Verified through log-probability separation that the reasoning module indeed learns token-level optimal decision signals.

## Limitations & Future Work
- **Hypothesis Limitations**: The method assumes trajectory rewards can be decomposed into the sum of token log-probabilities.
- **Reasoning Module Capacity Bottleneck**: Restricted by its initial size and training data.
- **Weight Selection Issue**: Weights $\alpha_i$ require manual tuning during multi-module composition.
- **Future Improvements**: Exploring token-level decomposition for non-separable rewards; designing adaptive weight learning.

## Related Work & Insights
- **vs PEFT (LoRA)**: LoRA depends on internal model dimensions, making cross-scale transfer difficult; UniR decouples architecture dependency via logit-level guidance.
- **vs Reasoning Vectors**: The latter uses the difference between RL and SFT to modify model parameters; UniR is more lightweight using independent modules.
- **vs RAST/GenARM**: UniR is more principled by directly training reasoning modules with verifiable rewards.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Systematically maps trajectory-level rewards to token-level log-probabilities of a reasoning module.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comparison across multiple math benchmarks + ablation verifying weak-to-strong transfer and multi-module composition.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivation, rigorous derivation, and evidence-based experiments.
- **Value**: ⭐⭐⭐⭐⭐ Addresses key issues in enhancing frozen LLM reasoning; universal, low-cost, and easy to deploy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GASP: Guided Asymmetric Self-Play For Coding LLMs](../../ICLR2026/llm_nlp/gasp_guided_asymmetric_self-play_for_coding_llms.md)
- [\[NeurIPS 2025\] AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play](../../NeurIPS2025/llm_nlp/acesearcher_bootstrapping_reasoning_and_search_for_llms_via_reinforced_self-play.md)
- [\[NeurIPS 2025\] Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs](../../NeurIPS2025/llm_nlp/triplets_better_than_pairs_towards_stable_and_effective_self-play_fine-tuning_fo.md)
- [\[ICML 2026\] Differential Syntactic and Semantic Encoding in LLMs](differential_syntactic_and_semantic_encoding_in_llms.md)
- [\[NeurIPS 2025\] Are Language Models Efficient Reasoners? A Perspective from Logic Programming](../../NeurIPS2025/llm_nlp/are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)

</div>

<!-- RELATED:END -->
