---
title: >-
  [Paper Note] ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment
description: >-
  [ICML 2026][LLM Reasoning][KL-regularized RL closed-form solution] ETS samples directly from the **closed-form optimal solution** of the KL-regularized RLHF objective…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "KL-regularized RL closed-form solution"
  - "energy reweighting"
  - "Monte Carlo"
  - "importance sampling"
  - "ARM/DLM universal"
date: 2026-05-08
content_hash: c19b4d64a2ac432d
---

# ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment

**Conference**: ICML 2026  
**arXiv**: [2601.21484](https://arxiv.org/abs/2601.21484)  
**Code**: https://github.com/sheriyuo/ETS (available)  
**Area**: LLM Inference / Test-Time Scaling / Training-Free Alignment  
**Keywords**: KL-regularized RL closed-form solution, energy reweighting, Monte Carlo, importance sampling, ARM/DLM universal

## TL;DR
ETS samples directly from the **closed-form optimal solution** of the KL-regularized RLHF objective, expressing it as a "reference policy × conditional expectation of exponential reward (energy term)", and then uses Monte Carlo + self-normalized importance sampling at test time to approximate this energy term. This achieves, or even surpasses, the performance of RL-trained policies **without any training**, and leverages lightweight proposals + Fast-dLLM to keep latency within practical bounds.

## Background & Motivation

**Background**: RLHF / DPO / GRPO have become standard post-training procedures for LLMs, aligning models to "high reward + not deviating from reference policy $p_{\text{ref}}$". Theoretically, the KL-regularized objective has a closed-form solution $p(\boldsymbol{x}_0\mid\boldsymbol{y})\propto p_{\text{ref}}(\boldsymbol{x}_0\mid\boldsymbol{y})\exp(r/\lambda)$ as given by Rafailov et al., but current RL pipelines still use gradient-based iterative approximation.

**Limitations of Prior Work**: RL training requires expensive reward models, large amounts of human preference data, is unstable, sensitive to hyperparameters, and any change in reward necessitates retraining. Methods like Power Sampling / Quest use MH sampling to avoid training but are slow due to serial sampling.

**Key Challenge**: There is a significant gap between the "known closed-form optimal distribution" and the "practical reliance on iterative training"—if one could **directly sample** from the closed-form distribution at test time, all training issues would vanish.

**Goal**: (1) Provide the **reverse Markov transition kernel** form of the closed-form solution under a unified MLM framework (including ARM and diffusion language models DLM); (2) Design Monte Carlo estimation + accelerators to make it practical; (3) Provide theoretical guarantees on convergence rate and error accumulation.

**Key Insight**: Treat the generation process as a reverse Markov chain from $\boldsymbol{x}_T\to\boldsymbol{x}_0$ (ARM is fixed left→right, DLM is dynamic unmasking). Under this framework, the optimal reverse transition kernel naturally decomposes into "reference transition × energy term".

**Core Idea**: At each guidance step, use candidate sampling + energy reweighting + multinomial sampling to "stepwise approach the optimal distribution along the reverse chain", avoiding any parameter updates.

## Method

### Overall Architecture
ETS is an inference-time search algorithm. Given a query $\boldsymbol{y}$, initial mask sequence $\boldsymbol{x}_T$, number of guidance steps $I$, and number of candidates $M$, it proceeds from $i=I$ back to $i=1$: (1) Sample $M$ candidates $\boldsymbol{x}_{t_{i-1}}(m)$ from $p_{\text{ref}}$ given $\boldsymbol{x}_{t_i}$; (2) Estimate the energy $\widehat{\mathcal E}$ for each candidate; (3) Self-normalize to obtain weights $w_m\propto \widehat{\mathcal E}$; (4) Select one candidate via multinomial sampling as the next state. The final $\boldsymbol{x}_0$ is thus a sample from the approximate optimal distribution $p(\boldsymbol{x}_0\mid\boldsymbol{y})$.

Note that when $I=1,\lambda\to 0$, the algorithm degenerates to Best-of-N, so ETS strictly generalizes BoN and provides a finer control knob for "multi-step alignment" via $I$.

### Key Designs

1. **Energy Reweighted Reverse Transition Kernel (Proposition 2)**:

    - **Function**: Converts the closed-form solution of KL-regularized RLHF from the final token sequence distribution into a stepwise sampleable form as a reverse Markov chain.
    - **Mechanism**: For any $s<t$, derive $p(\boldsymbol{x}_s\mid\boldsymbol{x}_t,\boldsymbol{y})\propto p_{\text{ref}}(\boldsymbol{x}_s\mid\boldsymbol{x}_t,\boldsymbol{y})\cdot \mathbb E_{p_{\text{ref}}(\boldsymbol{x}_0\mid\boldsymbol{y},\boldsymbol{x}_s)}\!\big[\exp(r/\lambda)\big]$. The latter term, the "energy" $\mathcal{E}(\boldsymbol{y},\boldsymbol{x}_s)$, measures the expected future reward starting from the current partial state $\boldsymbol{x}_s$.
    - **Design Motivation**: Decompose the globally optimal $p(\boldsymbol{x}_0\mid\boldsymbol{y})$, which is not directly sampleable, into "reference model (directly sampleable)" + "conditional expectation (Monte Carlo approximable)", both operational; unifies ARM and DLM.

2. **Monte Carlo Estimation of Energy Term + Self-Normalized IS (Algorithm 1)**:

    - **Function**: For each candidate $\boldsymbol{x}_{t_{i-1}}(m)$, estimate $\widehat{\mathcal E}$, then relatively weight the $M$ candidates.
    - **Mechanism**: From $\boldsymbol{x}_s$, use $p_{\text{ref}}$ to rollout $K$ complete $\boldsymbol{x}_0(k)$, estimating energy as $\widehat{\mathcal E}(\boldsymbol{y},\boldsymbol{x}_s)=\frac{1}{K}\sum_k \exp(r(\boldsymbol{y},\boldsymbol{x}_0(k))/\lambda)$. The global normalization constant is intractable, but self-normalization within the batch yields "relative optimal probabilities", and multinomial sampling from these is equivalent to sampling from the optimal distribution restricted to finite candidates. Proposition 3 proves a total variation distance upper bound $\widetilde{\mathcal O}(I/\sqrt M + I\epsilon)$, where $\epsilon$ is the energy estimation error.
    - **Design Motivation**: Directly approximating the partition function requires summing over the entire sequence space, which is infeasible; self-normalization converts the absolute probability problem into a relative sampling problem, a stable trick inherited from energy-based models / diffusion guidance.

3. **Importance Sampling Acceleration (Algorithm 2, ETS-IS)**:

    - **Function**: Uses a cheaper proposal model $p_{\text{small}}$ to replace $p_{\text{ref}}$ for rollouts, greatly reducing Monte Carlo estimation latency.
    - **Mechanism**: Based on $\mathcal E(\boldsymbol{y},\boldsymbol{x}_s)=\mathbb E_{p_{\text{small}}}[\frac{p_{\text{ref}}}{p_{\text{small}}}\exp(r/\lambda)]$, yielding an unbiased IS estimate; for ARM, uses a small Qwen3 model with the same tokenizer, for DLM, if no small model is available, uses Fast-dLLM (KV cache + parallel decoding) as $p_{\text{small}}$. Theorem 1 proves: with sufficiently large $K$, the IS version maintains the same order of convergence $\widetilde{\mathcal O}(I/\sqrt M + I/\sqrt K)$.
    - **Design Motivation**: Energy estimation is the latency bottleneck; running $M\times K$ rollouts with $p_{\text{ref}}$ alone is very expensive; IS provides an engineeringly feasible shortcut with "cheap sampling + correct unbiasedness".

### Loss & Training
**No training at all**. The reward does not rely on a reward model, but uses a self-consistency proxy: for each candidate, sample $K$ completions, perform a majority vote on the final answers, and assign reward=1 if it matches the majority, otherwise 0. This proxy yields a reward distribution closest to ground-truth in the experiments, outperforming other uncertainty metrics.

## Key Experimental Results

### Main Results
Evaluated on MATH500 / GSM8K / HumanEval / GPQA-Diamond with pass@1 (single final answer); ARM uses Qwen3-1.7B/8B (non-thinking), DLM uses LLaDA-8B-Instruct. Baselines include Base, Beam Search, Best-of-N, Power Sampling, RL-trained Verl, and LLaDA-1.5.

| Model | Dataset | Base | Best-of-N | Power Sampling | RL Trained | **ETS / ETS-IS** |
|---|---|---|---|---|---|---|
| Qwen3-8B (ARM) | MATH500 | baseline | improved | improved but slow | strong baseline | **outperforms RL** |
| Qwen3-8B (ARM) | GPQA-Diamond | baseline | medium | medium | strong | **best** |
| LLaDA-8B (DLM) | HumanEval | baseline | medium | medium | LLaDA-1.5 | **outperforms LLaDA-1.5** |
| Qwen3-1.7B (ARM) | GSM8K | baseline | medium | slow | strong | **best** (no IS needed) |

(Numerical results vary by setting, but the overall trend: ETS consistently outperforms TTS baselines on all four benchmarks, and often surpasses RL-trained models of the same size.)

### Ablation Study

| Configuration | Key Effect | Notes |
|---|---|---|
| Full ETS ($I>1$) | best | guidance distributes error over multiple steps |
| $I=1,\lambda\to 0$ | degenerates to Best-of-N | proves ETS strictly generalizes BoN |
| Remove IS (pure $p_{\text{ref}}$) | same accuracy but latency ↑↑ | IS is key for latency |
| Reward replaced with logits confidence / entropy | accuracy drops | self-consistency reward is closest to oracle |
| Increase $M$ | accuracy↑, latency↑ | matches $1/\sqrt M$ convergence |

### Key Findings
- "Training-free alignment" achieves **comparable or better** performance than RL-trained models on mainstream inference benchmarks for the first time, indicating that current RL training wastes significant computation on tasks that can be directly sampled via closed-form solutions.
- $I=1$ is not always the worst nor always the best—error does not accumulate linearly; the optimal working point is jointly determined by guidance steps and $\lambda$ (Remark 2).
- Using a well-aligned small Qwen3 model as the IS proposal yields the best efficiency/accuracy trade-off; speculative decoding (EAGLE-3) is disadvantaged due to incompatibility with batch processing.

## Highlights & Insights
- **Methodological Highlight**: Extends the "closed-form RLHF solution"—a known but overlooked fact—into a reverse chain transition kernel applicable to both ARM/DLM, with complete error analysis. This is a template for smoothly transferring score-based/diffusion guidance ideas to discrete MLMs.
- **Theoretical Closure**: Proposition 2 (transition kernel) → Proposition 3 (error) → Theorem 1 (including IS acceleration error), forming a tight chain, with clear analogy to error accumulation in diffusion models ($\propto I$).
- **Transferable Trick**: "Self-normalization + lightweight proposal IS" can be directly applied to other inference-time alignment tasks (dialogue preference, agent reward shaping, tool selection reranking), and is naturally compatible with batch parallelism.

## Limitations & Future Work
- The proxy reward uses self-consistency, which essentially assumes "majority answer = correct answer", and fails in creative/open-ended/multi-solution tasks.
- The error upper bound assumes uniform guidance error $\epsilon$ at each step, but in practice, errors vary greatly across different $\boldsymbol{x}_t$; deriving tighter, state-dependent bounds remains open.
- Acceleration for DLM relies on Fast-dLLM engineering; future availability of well-aligned small DLMs could further speed up inference.
- Full integration with speculative decoding/quantization and other acceleration methods remains incomplete.

## Related Work & Insights
- **vs Power Sampling / Quest**: All aim to sample from the RL optimal distribution, but MH algorithms are inherently serial; ETS leverages batched MC + IS for natural parallelism and higher speed.
- **vs Dang 2025 / Uehara 2024**: They derive similar formulas for continuous-time diffusion models; this work applies to discrete MLMs and unifies ARM/DLM.
- **vs Best-of-N / Beam Search**: BoN is the $I=1$ special case; Beam Search deterministically maximizes, which may not match the optimal probability distribution. ETS combines theoretical guarantees with empirical gains.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Directly "samples" the closed-form RL optimal solution, matches RL without training, a new methodological paradigm
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers ARM/DLM × math/code/science across 4 benchmarks + multiple acceleration ablations
- Writing Quality: ⭐⭐⭐⭐ Rigorous derivation, smooth logic from closed-form solution to IS acceleration; notation is dense but readable
- Value: ⭐⭐⭐⭐⭐ Provides a practical implementation of "test-time alignment", potentially saving the entire RLHF pipeline in engineering, with broad application prospects

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](../../ICLR2026/llm_reasoning/understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[ICML 2026\] Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)
- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](../../ACL2026/llm_reasoning/efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[NeurIPS 2025\] SolverLLM: Solving Optimization Problems via Test-Time Scaling with LLM-Guided Search](../../NeurIPS2025/llm_reasoning/solverllm_leveraging_test-time_scaling_for_optimization_problem_via_llm-guided_s.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](../../ACL2026/llm_reasoning/parallel_test-time_scaling_for_latent_reasoning_models.md)

</div>

<!-- RELATED:END -->
