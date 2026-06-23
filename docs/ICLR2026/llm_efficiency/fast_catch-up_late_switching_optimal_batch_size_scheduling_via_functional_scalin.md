---
title: >-
  [Paper Note] Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws
description: >-
  [ICLR 2026][LLM Efficiency][batch size scheduling] This paper theoretically derives the optimal strategy for batch size scheduling through the Functional Scaling Law framework—for difficult tasks, the optimal strategy involves training with a small batch size for most of the duration and switching to a large batch size only in the final stage (late switching). It revea
tags:
  - ICLR 2026
  - LLM Efficiency
  - batch size scheduling
  - scaling laws
  - LLM pretraining
  - fast catch-up
  - optimization theory
date: 2026-05-08
content_hash: 64711446d65d9022
---
# Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws

**Conference**: ICLR 2026  
**arXiv**: [2602.14208](https://arxiv.org/abs/2602.14208)  
**Code**: None  
**Area**: LLM Efficiency  
**Keywords**: batch size scheduling, scaling laws, LLM pretraining, fast catch-up, optimization theory

## TL;DR
This paper theoretically derives the optimal strategy for batch size scheduling through the Functional Scaling Law framework—for difficult tasks, the optimal strategy involves training with a small batch size for most of the duration and switching to a large batch size only in the final stage (late switching). It reveals the "fast catch-up" effect—where the loss rapidly catches up to the trajectory of a constant large batch after the switch—and validates this principle in 1.1B parameter, 1T token LLM pretraining.

## Background & Motivation

**Background**: Large batch training is a standard configuration for LLM pretraining (used in GPT-3, LLaMA-3, DeepSeek-V3, etc., via batch size scheduling). Large batches improve hardware utilization but sacrifice sample efficiency. While multi-stage batch size increase strategies are widely adopted in practice, their theoretical foundation remains weak.

**Limitations of Prior Work**: (a) Existing analyses either focus only on constant batch sizes (critical batch size theory) or rely on heuristics (Smith et al. 2018); (b) BSS design depends on expensive large-scale experimental hyperparameter tuning; (c) There is a lack of theoretical explanation for why "small-then-large" batch schedules are effective in practice.

**Key Challenge**: In the early stages of training, signals dominate; the denoising benefits of large batches are minimal compared to the increased data consumption. In later stages, gradient noise increases, necessitating large batches for denoising. However, there is no theoretical guidance on when and how to switch.

**Goal**: (a) Derive the optimal BSS under a fixed data budget; (b) Explain why late switching is effective; (c) Validate theoretical predictions in large-scale LLM pretraining.

**Key Insight**: Utilize the Functional Scaling Law (FSL) framework to transform the BSS optimization problem into an analytically solvable variational problem.

**Core Idea**: FSL demonstrates that optimal BSS depends on task difficulty—difficult tasks should use small batches for the majority of time (performing more steps to learn signals) and switch to large batches at the very end (for rapid denoising), as the "fast catch-up" effect ensures the loss quickly matches the target trajectory after switching.

## Method

### Overall Architecture
The paper addresses a specific engineering problem: how batch size should change over the training process to be most efficient under a fixed data budget. Instead of running extensive empirical tuning, it embeds the entire problem into the Functional Scaling Law (FSL) framework for analytical solution. FSL decomposes the expected error at any time into two competing forces:

$$\mathbb{E}[\mathcal{E}(\theta_t)] \eqsim \underbrace{t^{-s}}_{\text{signal learning}} + \underbrace{\eta\sigma^2 \int_0^t \frac{\mathcal{K}(t-\tau)}{b(\tau)}d\tau}_{\text{noise accumulation}}$$

The first term represents signal learning, decaying as a power law $t^{-s}$ relative to the number of steps $t$—the harder the task (smaller $s$), the slower the decay and the more steps required. The second term represents noise accumulation, where the gradient noise $\eta\sigma^2/b(\tau)$ injected at each step is integrated with a weighting from a forgetting kernel $\mathcal{K}(t) = (t+1)^{-(2-1/\beta)}$; larger batches $b(\tau)$ result in less noise injection. The essence of BSS is to allocate a fixed data budget between "taking more steps to learn signals" and "increasing batch size to suppress noise." The paper formulates this as a resource-constrained variational problem, solving for the continuous optimal solution $b^*(t)$ (Theorem 3.1), then simplifying to a practical two-stage switch to find the optimal switching point (Theorem 3.2), and finally explaining the fast catch-up phenomenon via the forgetting structure of the noise term.

### Key Designs

**1. Continuous Optimal BSS under FSL: Turning Heuristics into Theorems**

Theorem 3.1 solves for the optimal batch size function $b^*(t)$ under a fixed data budget $D$, with conclusions categorized by task difficulty. For simple tasks ($s > 1-1/\beta$, where signals are learned quickly), $b^*(t) \propto (T^*-t+1)^{1/(2\beta)-1}$, where the batch size increases monotonically with no distinct "small batch phase." For difficult tasks ($s \leq 1-1/\beta$, where signals decay slowly), a two-stage structure emerges: the batch size is held at the lower bound $B_{\min}$ until $T_1^*$, and only then begins to grow, with the growth phase occupying only a small fraction of total training time, $(T^* - T_1^*)/T^* = o_D(1)$. This provides the theoretical origin for "small-then-large" schedules—difficult tasks require small batches to perform more steps given a fixed data budget to learn signals properly, while large batches are used only in the final stage for concentrated denoising.

**2. Optimal Two-Stage Switching Point: Providing an Extrapolatable Scaling Law**

While continuous solutions are elegant, practical implementations usually involve only one or two switches (as changing batch size requires reconfiguring data pipelines and communication). Theorem 3.2 addresses the practical two-stage scheme $B_1 \to B_2$, solving for the optimal switching point $P_D^*$ (the number of samples fed before switching). For simple tasks, the optimal is a constant large batch ($P_D^* = 0$). For difficult tasks, the proportion of data remaining for the large batch phase decays as a power law of the budget, $(D-P_D^*)/D \eqsim D^{-\frac{1-1/\beta-s}{2-1/\beta}}$. As the budget increases, this proportion approaches zero—implying that at larger scales, switching should be delayed longer to save more data for the small batch phase. Equivalently, the switching point follows a scaling law $D - P_D^* \sim D^{\gamma}$. Its practical value lies here: by fitting the exponent $\gamma$ in small-scale training, one can extrapolate the optimal switching point for large scales, avoiding expensive trial-and-error on large models.

**3. Fast Catch-Up Effect: Dynamical Explanation for Late Switching**

The most counter-intuitive finding is that delaying the switch to a large batch until very late does not result in a loss penalty; instead, the loss rapidly catches up to the trajectory of a constant large batch after the switch. The paper explains this through the structure of the noise term—the forgetting kernel $\mathcal{K}(t)=(t+1)^{-(2-1/\beta)}$ in the noise accumulation $\int_0^t \mathcal{K}(t-\tau)/b(\tau)d\tau$ continuously decays the contribution of early noise. Thus, the extra noise injected during the small batch phase is not a permanent loss but a transient state that is quickly forgotten. The speed of catch-up is determined by task difficulty parameters $(s,\beta)$. This reinterprets the effectiveness of late switching: it is not that the "large batch phase recovers lost progress," but rather that "the small batch phase allows for more thorough signal learning, and the excess noise quickly dissipates after switching," making late switching superior.

### Loss & Training
The theoretical analysis is based on single-pass SGD with a constant learning rate and BSS. The paper proves this combination achieves the optimal convergence rate, equivalent to meticulously tuned learning rate schedules. Practically, BSS significantly reduces the number of iterations without sacrificing data efficiency, which directly translates to shorter wall-clock training time when combined with GPU data parallelism.

## Key Experimental Results

### Main Results

**1.1B MoE Model, 1T tokens**:

| BSS Strategy | Switching Point | Final Loss | Description |
|:---|:---|:---|:---|
| Constant Small Batch (1024) | - | High | More steps but high noise |
| Constant Large Batch (2560) | - | Baseline | Standard large batch |
| Early Switch (small→large @ 25%) | 0.25T | Approx. baseline | No advantage to early switching |
| **Late Switch (small→large @ 75%)** | **0.75T** | **Better than baseline** | **Late switching is optimal** |

### Ablation Study

| Configuration | Key Finding |
|:---|:---|
| Dense 0.5B + C4 | Fast catch-up effect consistently present |
| MoE 1B + 0.4T | Four-stage BSS (640→1280→1920→2560) shows catch-up at every switch |
| MoE 1.1B + 1T | Largest scale validation, late switch consistently outperforms early switch |

### Key Findings
- **Fast catch-up appears consistently across architectures and scales**: Observed in Dense and MoE models, from 50M to 1.1B parameters, and 10B to 1T tokens; loss quickly catches up to the large batch trajectory.
- **Late switching saves data consumption**: For the same final loss, the small batch phase completes more steps with less data, significantly reducing computational costs.
- **Four-stage BSS validation**: Multiple switches trigger catch-up each time, proving the additivity and robustness of the effect.
- **High agreement between theoretical predictions and experiments**: The optimal BSS derived from linear regression was validated in both discrete SGD and LLM pretraining.

## Highlights & Insights
- **Perfect Loop Between Theory and Practice**: Complete logical chain from FSL theory derivation to linear regression validation to LLM pretraining validation—a benchmark in scaling laws research.
- **Intuition of Fast Catch-Up**: "Excess noise" accumulated by small batches is a transient state that can be quickly forgotten rather than a permanent injury, challenging the common wisdom that "large batches should be used as early as possible."
- **BSS vs. LR Schedule Duality**: The stable→growth structure of optimal BSS corresponds to the warmup→stable→decay of LR schedules. While equivalent in data efficiency, BSS is superior in reducing iteration counts—making it a more efficient "knob."
- **Actionable Extrapolation Strategy**: The optimal switching point follows a scaling law $D - P^* \sim D^\gamma$, allowing for direct extrapolation from small-scale experiments.

## Limitations & Future Work
- The theory is based on linear/kernel regression; the non-linear dynamics of LLM pretraining might introduce additional factors.
- Task difficulty parameters $(s, \beta)$ are difficult to measure directly in actual LLMs and must be determined through fitting.
- Only constant learning rate + BSS was considered; joint optimization with learning rate warmup/decay was not fully explored.
- Did not account for communication overhead in distributed training—the impact of changing batch sizes on data parallel configurations.

## Related Work & Insights
- **vs. McCandlish et al. (Critical Batch Size)**: They studied constant batches; this work extends to dynamic BSS and provides the optimal solution.
- **vs. Smith et al. 2018 (BSS heuristics)**: Heuristic analysis cannot provide the optimal switching timing; this work provides precise predictions at the scaling law level.
- **vs. Chinchilla Scaling Laws**: Chinchilla focuses on model/data ratios; this work focuses on the training strategy (BSS) under a fixed budget; the two are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery and theoretical explanation of the fast catch-up effect are highly novel, with direct implications for LLM training.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes theoretical derivation, linear regression, and multi-architecture, multi-scale LLM pretraining.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivation, intuitive visualizations, and a smooth narrative from theory to practice.
- Value: ⭐⭐⭐⭐⭐ Provides a theoretical foundation for BSS design in LLM pretraining, offering direct industrial application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling](scaling_up_speeding_up_a_benchmark_of_speculative_decoding_for_efficient_llm_tes.md)
- [\[ICLR 2026\] Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs](scaling_laws_meet_model_architecture_toward_inference-efficient_llms.md)
- [\[ICLR 2026\] xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity](xlstm_scaling_laws_competitive_performance_with_linear_time-complexity.md)
- [\[ICLR 2026\] Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models](towards_greater_leverage_scaling_laws_for_efficient_mixture-of-experts_language_.md)
- [\[NeurIPS 2025\] Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training](../../NeurIPS2025/llm_efficiency/critical_batch_size_revisited_a_simple_empirical_approach_to_large-batch_languag.md)

</div>

<!-- RELATED:END -->
