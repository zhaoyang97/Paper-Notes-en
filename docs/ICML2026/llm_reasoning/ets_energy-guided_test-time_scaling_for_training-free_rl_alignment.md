---
title: >-
  [Paper Note] ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment
description: >-
  [ICML 2026][LLM Reasoning][KL-regularized RL Closed-form Solution] ETS directly samples from the **closed-form optimal solution** of the KL-regularized RLHF objective…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "KL-regularized RL Closed-form Solution"
  - "Energy Reweighting"
  - "Monte Carlo"
  - "Importance Sampling"
  - "ARM/DLM Generalization"
date: 2026-05-08
content_hash: 628f4fe3e83e5952
---

# ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment

**Conference**: ICML 2026  
**arXiv**: [2601.21484](https://arxiv.org/abs/2601.21484)  
**Code**: https://github.com/sheriyuo/ETS (Available)  
**Area**: LLM Inference / Test-time Scaling / Training-free Alignment  
**Keywords**: KL-regularized RL Closed-form Solution, Energy Reweighting, Monte Carlo, Importance Sampling, ARM/DLM Generalization

## TL;DR
ETS directly samples from the **closed-form optimal solution** of the KL-regularized RLHF objective, formulating it as "reference policy $\times$ conditional expectation of exponential rewards (energy term)." By approximating this energy term at test-time using Monte Carlo and Self-Normalized Importance Sampling, ETS achieves or surpasses the performance of post-trained RL models **without training**, while maintaining practical latency through lightweight proposals and Fast-dLLM.

## Background & Motivation

**Background**: RLHF, DPO, and GRPO have become standard for LLM post-training, aligning models to maximize rewards while staying close to a reference policy $p_{\text{ref}}$. Theoretically, this KL-regularized objective has a closed-form solution derived by Rafailov et al.: $p(\boldsymbol{x}_0\mid\boldsymbol{y})\propto p_{\text{ref}}(\boldsymbol{x}_0\mid\boldsymbol{y})\exp(r/\lambda)$. However, existing RL pipelines still rely on iterative gradient-based optimization to approximate this.

**Limitations of Prior Work**: Training-based RL requires expensive reward models and massive human preferences, suffers from training instability and hyperparameter sensitivity, and necessitates retraining whenever the reward function changes. Furthermore, MCMC-based sampling methods like Power Sampling or Quest, while training-free, are slow due to their sequential nature.

**Key Challenge**: There exists a significant gap between the "known closed-form optimal distribution" and the "iterative training needed to reach it" in practice. If one could **directly sample** from that closed-form distribution during test-time, all training-related issues would disappear.

**Goal**: (1) Derivie the **reverse Markov transition kernel** form of the closed-form solution under a unified MLM framework (including ARM and Diffusion Language Models - DLMs); (2) Design a Monte Carlo estimator and accelerated mechanism for practical use; (3) Provide theoretical guarantees for convergence rates and error accumulation.

**Key Insight**: Treat the generation process as a reverse Markov chain $\boldsymbol{x}_T\to\boldsymbol{x}_0$ (fixed left-to-right for ARM, dynamic unmasking for DLM). Deriving the optimal reverse transition kernel under this framework naturally decomposes into a "reference transition $\times$ energy term."

**Core Idea**: At each guidance step, use candidate sampling, energy reweighting, and multinomial sampling to "step-by-step approach the optimal distribution along the reverse chain" without any parameters updates.

## Method

### Overall Architecture
ETS is an inference-time search algorithm. Given a query $\boldsymbol{y}$, an initial masked sequence $\boldsymbol{x}_T$, guidance steps $I$, and candidate count $M$, it recurses from $i=I$ down to $i=1$: (1) Sample $M$ candidates $\boldsymbol{x}_{t_{i-1}}(m)$ from $p_{\text{ref}}$ given $\boldsymbol{x}_{t_i}$; (2) Estimate the energy $\widehat{\mathcal E}$ for each candidate; (3) Obtain weights $w_m\propto \widehat{\mathcal E}$ via self-normalization; (4) Select one candidate via multinomial sampling as the next state. The final $\boldsymbol{x}_0$ is a sample from the approximate optimal distribution $p(\boldsymbol{x}_0\mid\boldsymbol{y})$.

Notably, when $I=1$ and $\lambda\to 0$, the algorithm simplifies to Best-of-N. Thus, ETS strictly generalizes BoN and provides a finer "multi-step alignment" control via $I$.

### Key Designs

1.  **Energy-Reweighted Reverse Transition Kernel (Proposition 2)**:
    - **Function**: Transforms the closed-form solution of KL-regularized RLHF from a final token sequence distribution into a step-by-step samplable form for reverse Markov chains.
    - **Mechanism**: For any $s<t$, it derives $p(\boldsymbol{x}_s\mid\boldsymbol{x}_t,\boldsymbol{y})\propto p_{\text{ref}}(\boldsymbol{x}_s\mid\boldsymbol{x}_t,\boldsymbol{y})\cdot \mathbb E_{p_{\text{ref}}(\boldsymbol{x}_0\mid\boldsymbol{y},\boldsymbol{x}_s)}\!\big[\exp(r/\lambda)\big]$. The latter term is the "energy" $\mathcal{E}(\boldsymbol{y},\boldsymbol{x}_s)$, measuring the expected future reward from the current partial state $\boldsymbol{x}_s$.
    - **Design Motivation**: Decomposes the intractable global optimal $p(\boldsymbol{x}_0\mid\boldsymbol{y})$ into a "directly samplable reference model" and a "Monte Carlo estimable conditional expectation," both of which are operational, covering both ARM and DLM.

2.  **Energy Monte Carlo Estimation + SNIS (Algorithm 1)**:
    - **Function**: Estimates $\widehat{\mathcal E}$ for each candidate $\boldsymbol{x}_{t_{i-1}}(m)$ to perform relative weighting among $M$ candidates.
    - **Mechanism**: Rollout $K$ complete sequences $\boldsymbol{x}_0(k)$ from $\boldsymbol{x}_s$ using $p_{\text{ref}}$, estimating energy as $\widehat{\mathcal E}(\boldsymbol{y},\boldsymbol{x}_s)=\frac{1}{K}\sum_k \exp(r(\boldsymbol{y},\boldsymbol{x}_0(k))/\lambda)$. While the global normalization constant is unknown, self-normalization within the batch provides "relative optimal probabilities." Proposition 3 proves a Total Variation distance upper bound of $\widetilde{\mathcal O}(I/\sqrt M + I\epsilon)$, where $\epsilon$ is the energy estimation error.
    - **Design Motivation**: Directly approximating the partition function via summation over the entire sequence space is impossible; self-normalization converts absolute probability problems into relative sampling, a stable trick inherited from energy-based models and diffusion guidance.

3.  **Importance Sampling Acceleration (Algorithm 2, ETS-IS)**:
    - **Function**: Replaces $p_{\text{ref}}$ with a cheaper proposal model $p_{\text{small}}$ for rollouts, significantly reducing Monte Carlo estimation latency.
    - **Mechanism**: Based on $\mathcal E(\boldsymbol{y},\boldsymbol{x}_s)=\mathbb E_{p_{\text{small}}}[\frac{p_{\text{ref}}}{p_{\text{small}}}\exp(r/\lambda)]$, an unbiased IS estimator is obtained. For ARM, a smaller model (e.g., Qwen3-1.7B) is used; for DLM, Fast-dLLM (KV cache + parallel decoding) serves as $p_{\text{small}}$. Theorem 1 proves that with sufficiently large $K$, the IS version maintains a convergence rate of $\widetilde{\mathcal O}(I/\sqrt M + I/\sqrt K)$.
    - **Design Motivation**: Energy estimation is the bottleneck; running $M\times K$ rollouts with $p_{\text{ref}}$ is extremely expensive. IS provides a shortcut for "cheap sampling with unbiased correction."

### Loss & Training
**Completely training-free**. Instead of a reward model, it uses a self-consistency proxy: for each candidate, sample $K$ completions and perform a majority vote. If it matches the majority, reward=1, otherwise 0. This proxy characterizes the reward distribution most accurately among similar uncertainty metrics in the paper's experiments.

## Key Experimental Results

### Main Results
Evaluated on Pass@1 (single final response) across MATH500, GSM8K, HumanEval, and GPQA-Diamond. ARM uses Qwen3-1.7B/8B (non-thinking); DLM uses LLaDA-8B-Instruct. Baselines include Base, Beam Search, Best-of-N, Power Sampling, and models trained via RL (Verl) or LLaDA-1.5.

| Model | Dataset | Base | Best-of-N | Power Sampling | RL-trained | **Ours (ETS/ETS-IS)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3-8B (ARM) | MATH500 | baseline | Gain | Slow | Strong | **Surpasses RL** |
| Qwen3-8B (ARM) | GPQA-Diamond | baseline | Mid | Mid | Strong | **Best** |
| LLaDA-8B (DLM) | HumanEval | baseline | Mid | Mid | LLaDA-1.5 | **Surpasses LLaDA-1.5** |
| Qwen3-1.7B (ARM) | GSM8K | baseline | Mid | Slow | Strong | **Best** (No IS needed) |

(General trend: ETS consistently outperforms TTS baselines across all four benchmarks and often beats specifically RL-trained models of the same size.)

### Ablation Study

| Configuration | Key Effect | Description |
| :--- | :--- | :--- |
| Full ETS ($I>1$) | Optimal | Multi-step guidance distributes error. |
| $I=1, \lambda\to 0$ | Degenerates to BoN | Proves ETS strictly generalizes BoN. |
| w/o IS (Pure $p_{\text{ref}}$) | Same accuracy, Latency ↑↑ | IS is the latency lifesaver. |
| Reward as logits/entropy | Accuracy drops | Self-consistency reward is closest to oracle. |
| Increasing $M$ | Acc ↑, Latency ↑ | Aligns with $1/\sqrt{M}$ convergence. |

### Key Findings
- **Training-free alignment** achieved parity or superiority over RL post-training on mainstream reasoning benchmarks for the first time, suggesting existing RL training might be "over-calculating" what closed-form sampling can already do.
- $I=1$ is not always best or worst; error does not accumulate linearly, and the optimal point is determined by the joint interaction of guidance steps and $\lambda$ (Remark 2).
- The efficiency/accuracy trade-off is optimized when using a well-aligned small proposal model for IS; speculative decoding (like EAGLE-3) struggles due to batch incompatibility.

## Highlights & Insights
- **Methodological Highlight**: It takes the known but overlooked fact of the "closed-form RLHF solution" and extends it into a general reverse-chain transition kernel for ARM/DLM with rigorous error analysis—a template for migrating score-based/diffusion guidance into discrete MLMs.
- **Theoretical Closure**: Proposition 2 (Transition Kernel) $\to$ Proposition 3 (Error) $\to$ Theorem 1 (IS Error) provides a complete logical chain, mirroring error accumulation results ($\propto I$) found in diffusion models.
- **Transferable Trick**: The "Self-normalization + lightweight proposal IS" combination can be applied to other inference-time alignment tasks (dialogue preferences, agent reward shaping, tool selection reranking) and is inherently compatible with batched parallelism.

## Limitations & Future Work
- The self-consistency proxy reward relies on the assumption that "majority answer = correct answer," which may fail in creative, open-ended, or multi-solution tasks.
- The error upper bound assumes a uniform guidance error $\epsilon$ per step, though $\epsilon$ varies significantly across $\boldsymbol{x}_t$ in practice; state-dependent bounds remain open.
- Acceleration for DLM depends on the Fast-dLLM implementation; a truly aligned small DLM would further improve speed.
- Integration with speculative decoding and quantization for further acceleration is still in progress.

## Related Work & Insights
- **vs. Power Sampling / Quest**: While all target the optimal RL distribution, MCMC methods are sequential. ETS leverages batched MC + IS for parallelism, offering significantly better speed.
- **vs. Dang 2025 / Uehara 2024**: While prior work derived similar formulas for continuous-time diffusion, this work applies to discrete MLMs and unifies ARM/DLM.
- **vs. Best-of-N / Beam Search**: BoN is a special case ($I=1$); Beam Search is deterministic maximization and may not match the optimal probabilistic distribution. ETS provides both theoretical guarantees and empirical gains.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Sampling the "closed-form RL solution" to rival RL without training is a novel paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers ARM/DLM across Math/Code/Science benchmarks with thorough acceleration ablations.
- Writing Quality: ⭐⭐⭐⭐ Decent derivation; the logic from the closed-form solution to IS acceleration is smooth.
- Value: ⭐⭐⭐⭐⭐ Provides a feasible implementation of test-time alignment that could potentially circumvent the entire RLHF pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](../../ICLR2026/llm_reasoning/understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[ICML 2026\] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning](beyond_two-stage_training_cooperative_sft_and_rl_for_llm_reasoning.md)
- [\[ICML 2026\] Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models](lookahead_sample_reward_guidance_for_test-time_scaling_of_diffusion_models.md)
- [\[ICML 2026\] UniScale: Adaptive Unified Inference Scaling via Online Joint Optimization of Model Routing and Test-time Scaling](uniscale_adaptive_unified_inference_scaling_via_online_joint_optimization_of_mod.md)
- [\[ICML 2026\] Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)

</div>

<!-- RELATED:END -->
