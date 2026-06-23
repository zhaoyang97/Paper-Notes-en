---
title: >-
  [Paper Note] Entropy Regularizing Activation: Boosting Continuous Control, Large Language Models, and Image Classification with Activation as Entropy Constraints
description: >-
  [ICLR 2026][Reinforcement Learning][Paper Note] ERA (Entropy Regularizing Activation) imposes an entropy lower bound constraint by appending a specialized activation function to the network output layer. This approach requires no modification to the loss function and improves performance across continuous control RL, LLM reasoning, and image classification within a
tags:
  - ICLR 2026
  - Reinforcement Learning
date: 2026-05-08
content_hash: 839168465271ac6b
---
# Entropy Regularizing Activation: Boosting Continuous Control, Large Language Models, and Image Classification with Activation as Entropy Constraints

**Conference**: ICLR 2026  
**Code**: https://nothingbutbut.github.io/era  
**Area**: Reinforcement Learning  
**Keywords**: Entropy Regularization, Activation Function, Maximum Entropy Reinforcement Learning, Policy Entropy Constraint, LLM Alignment  

## TL;DR

ERA (Entropy Regularizing Activation) imposes an entropy lower bound constraint by appending a specialized activation function to the network output layer. This approach requires no modification to the loss function and improves performance across continuous control RL, LLM reasoning, and image classification within a single framework.

## Background & Motivation

**Background**: Maximum Entropy Reinforcement Learning (e.g., SAC) integrates entropy rewards directly into the optimization objective and has become the mainstream paradigm for continuous control. In LLM RL (e.g., GRPO), maintaining policy entropy to prevent exploration collapse is also a core challenge.  
**Limitations of Prior Work**: ① Adding an entropy bonus to the loss function distorts the optimization landscape of the primary objective, leading to mutual interference and suboptimal performance for SAC on high-dimensional humanoid robot tasks; ② Direct entropy bonuses in LLM alignment are unstable, while heuristic methods like KL-Cov/Clip-Cov lack theoretical guarantees and only apply to off-policy settings; ③ Previous projection methods (uniform mixing) impose identical regularization across all dimensions, which scales poorly in high-dimensional action spaces.  
**Key Challenge**: The coupling of entropy constraints with primary objective optimization—once entropy terms are integrated into the loss, the primary objective gradients are "contaminated."  
**Goal**: To design a domain-agnostic, non-intrusive entropy constraint paradigm with theoretical guarantees that completely decouples entropy constraints from the primary objective.  
**Core Idea**: Instead of modifying the loss function, a specialized activation function $g(\cdot)$ is applied to the final layer output. This transforms the raw distribution parameters $z$ into $z' = g(z)$, ensuring that the expected entropy of the transformed policy $\pi_{z'}$ satisfies $\mathbb{E}_{s}[H(\pi_{\theta}(\cdot|s))] \geq H_0$, thereby building an entropy lower bound into the architecture.

## Method

### Overall Architecture

ERA defines a general output activation framework: for a parameterized policy $f_\theta(s)$ outputting distribution parameters $z$, an activation $g: \mathcal{Z} \to \mathcal{Z}$ is inserted after the final layer to obtain $z' = g(z)$. The final policy is $\pi_\theta(\cdot|s) = \pi_{g(f_\theta(s))}(\cdot|s)$. The design of $g(\cdot)$ guarantees an entropy lower bound while remaining completely transparent to the primary objective loss—no entropy terms are required in the loss function. ERA provides specific instantiations for continuous spaces (bounded Gaussian policies), discrete spaces (Softmax policies), and LLM RL (GRPO).

```mermaid
flowchart LR
    A[Policy Network f_θ] --> B[Raw Output z]
    B --> C{ERA Activation g}
    C -->|z' = g(z)| D[Transformed Parameters]
    D --> E[Action Sampling/Classification/Token Gen]
    D --> F[Satisfies H(π) ≥ H₀]
    G[Primary Loss<br/>No Entropy Term] --> A
```

### Key Designs

**1. Continuous Control: Entropy Lower Bound Activation for Bounded Gaussians**  
In continuous control, policies typically apply a $\tanh$ squash or truncation after Gaussian sampling. The entropy of a bounded policy equals the original Gaussian entropy minus a non-negative bias term: $H_\pi = H_\text{Gaussian} - \mathbb{E}[\text{bias}]$. Thus, imposing an entropy lower bound $H_0$ on the final policy is equivalent to imposing a higher constraint on the underlying Gaussian standard deviation. ERA simultaneously satisfies the entropy lower bound $H_0$ and the standard deviation range constraint $[\sigma_\text{min}, \sigma_\text{max}]$ via the following activation:

$$\sigma'_i = \exp\!\left[\max\!\left(\log\sigma_\text{max} + \bigl(H'_0 - D\log\sqrt{2\pi e} - D\log\sigma_\text{max}\bigr)\frac{e^{\hat\sigma_i}}{\sum_j e^{\hat\sigma_j}},\; \log\sigma_\text{min}\right)\right]$$

Where $H'_0 = H_0 + \hat\delta$ is the target entropy plus a compensation term for the bounding bias ($\hat\delta$ can be fixed or learned via an auxiliary loss $\mathcal{L}(\hat\delta) = \mathbb{E}_s[\hat\delta(H[\pi(\cdot|s)] - H_0)]$). Since entropy constraints are built into the activation, SAC actor/critic objectives can remove entropy terms entirely, allowing the policy to focus on maximizing rewards.

**2. Discrete Classification: Softmax Entropy Lower Bound Activation**  
For classification tasks with Softmax policies, ERA transforms pre-activation logits $z$ into $z'$ such that the output distribution entropy is not lower than $H_0$:

$$z'_i = \hat{h}^{-1}\!\left[\max\!\left(\log\frac{\tau}{\tau} + \left(C_{H_0} - n\log\frac{\tau}{\tau}\right)\frac{1}{D-1}\left(1 - \frac{e^{z_i}}{\sum_j e^{z_j}}\right),\; 0\right)\right]$$

Where $\hat{h}^{-1}(x) \approx -\frac{1}{4} - \sqrt{2(-1-\ln x)} + \frac{3}{4}\ln x$, $C_{H_0} = e^{H_0 - 1}$, and $\tau \geq e$ is a fixed hyperparameter. Compared to label smoothing, ERA is an input-dependent adaptive regularization rather than global uniform smoothing, providing stronger representational capacity.

**3. LLM RL: Post-Sampling Activation Based on Forking Tokens**  
In LLMs, the action space is vast, and most tokens are nearly deterministic; forcing high entropy on all tokens would destroy linguistic structure. The LLM instantiation of ERA works during the model update phase after sampling, applying activation only to the logits of the top-20% "forking tokens" with the highest entropy:

$$z'_i = \begin{cases} kz_i & H_\text{resp} < \omega_\text{low},\; A_t > 0 \\ z_i & (\omega_\text{low} \leq H_\text{resp} \leq \omega_\text{high},\; A_t < 0)\;\text{or}\; A_t > 0 \\ \frac{1}{k}z_i & H_\text{resp} > \omega_\text{high},\; A_t > 0 \end{cases}$$

Here $k > 1$, $H_\text{resp}$ is the average entropy of the top-20% high-entropy tokens within that response, and $\omega_\text{low}/\omega_\text{high}$ are threshold bounds. Sharpening ($kz$) when entropy is too low makes the model "aware" of over-exploitation to promote exploration; flattening ($\frac{1}{k}z$) when entropy is too high avoids invalid divergence. Corresponding scaling $A'_t$ is applied to the advantage of modified tokens to balance gradient magnitudes. This design is compatible with on-policy settings (no importance sampling ratio or KL loss required) and keeps the policy unchanged during inference.

**4. Theoretical Guarantees**  
All three instantiations are accompanied by rigorous proofs of entropy lower bounds (Appendix B.1–B.3): the construction of the activation function $g(\cdot)$ ensures that the expected entropy of the policy corresponding to the transformed parameters satisfies $\mathbb{E}_s[H(\pi_\theta(\cdot|s))] \geq H_0$, a property lacking in prior heuristic methods (clip-higher, KL-Cov, etc.).

## Key Experimental Results

### Main Results

**Continuous Control** (Normalized scores, aggregated IQM):

| Task Set | Algorithm | Baseline | ERA-Augmented | Gain |
|----------|-----------|----------|---------------|------|
| HumanoidBench (6 tasks) | SAC | 0.59 | 0.84 | +42% |
| DMC Dog & Humanoid (6 tasks) | TD-MPC2 | 0.57 | 0.88 | +54% |
| HumanoidBench (8 tasks) | FastSAC | 0.56 | 0.81 | +45% |
| MuJoCo Gym (4 tasks) | PPO | 0.63 | 0.82 | +30% |

**LLM Reasoning** (Qwen2.5-Math-7B, avg.@16):

| Benchmark | GRPO | ERA | Gain |
|-----------|------|-----|---|
| AIME'24 | 34.4 | 36.0 | +4.7% |
| AIME'25 | 12.3 | 21.0 | +70.7% |
| AMC'23 | 69.5 | 76.6 | +10.4% |
| MATH500 | 80.6 | 85.4 | +6.0% |
| Minerva | 36.8 | 40.1 | +9.0% |
| OlympiadBench | 40.6 | 46.8 | +15.3% |
| **Average** | 45.7 | **51.0** | **+11.6%** |

**Image Classification** (ResNet-50, ImageNet Top-1):

| Setting | Baseline | ERA | Gain |
|---------|----------|-----|---|
| No Augmentation | 74.75 | 75.44 | +0.69% |
| With Augmentation | 76.93 | 77.30 | +0.37% |

### Ablation Study

| Configuration | Metric | Description |
|---------------|--------|-------------|
| SAC-ERA (Different $H_0$) | IQM consistently > SAC | Not sensitive to $H_0$ hyperparameters; no fine-tuning needed |
| SAC w/o Entropy Term (no ERA) | Lower than SAC-ERA | Removing entropy bonus is insufficient; ERA needed for exploration |
| Qwen2.5-Math-1.5B + ERA vs GRPO | avg +14.1% | Generalizes to smaller models |
| GSPO + ERA vs GSPO (7B) | avg +6.9% | Compatible with different RL algorithms |
| ImageNet, different $H_0$ | Top-1 stable | Robust to entropy hyperparameters |

### Key Findings

- ERA consistently maintains policy entropy at a non-zero lower bound, while the GRPO baseline exhibits typical entropy collapse; entropy stability is highly correlated with reasoning performance gains.
- Improvements are particularly significant (>30%) on high-dimensional tasks like HumanoidBench, which are most sensitive to exploration quality.
- ERA computational overhead is <7% and can be directly layered onto existing algorithms without changing other components.

## Highlights & Insights

- **Clean Decoupling**: Moving entropy constraints from the loss function to the network architecture (activation function) is a conceptually elegant transition—the primary loss focuses on reward maximization, while structure handles entropy guarantees.
- **Solid Theoretical Support**: Rigorous entropy lower bound proofs for all three scenarios distinguish this from previous empirical heuristic methods.
- **Cross-Domain Unity**: The same paradigm covers continuous control, discrete classification, and LLM RL, revealing the universality of "output distribution entropy control."
- **Robustness to Hyperparameters**: Experiments show stable performance across a wide range of $H_0$, reducing the tuning burden.
- **Complementarity**: Gains are observed even on top of existing data augmentation and label smoothing, indicating that ERA fills a different regularization gap.

## Limitations & Future Work

- Hyperparameters such as $\omega_\text{low}/\omega_\text{high}/k$ in the LLM instantiation still require manual setting for different models; automated tuning mechanisms need investigation.
- The top-20% truncation for forking tokens is a heuristic choice; fine-grained token importance estimation might further improve results.
- LLM performance has currently only been validated on mathematical reasoning; generalization to code generation or multimodal tasks remains unexplored.
- For multi-agent or partially observable scenarios, the adaptation of entropy constraints requires further research.

## Related Work & Insights

- **vs SAC (Max Entropy RL)**: SAC writes entropy bonuses into Q-targets and actor loss; ERA removes these and replaces them with activations, showing clear advantages in high-dimensional tasks.
- **vs Akrour et al. (2019)/Otto et al. (2021) (Projection Methods)**: Shares the "no modification to objective" philosophy, but their uniform mixing (equal regularization across dimensions) scales poorly; ERA enables dimension-aware gradient guidance via softmax weighting.
- **vs KL-Cov/Clip-Cov (LLM Entropy Control)**: These methods rely on importance sampling ratios, apply only to off-policy settings, and lack theoretical bounds; ERA is on-policy compatible and provable.
- **vs Label Smoothing (Classification Regularization)**: Label smoothing is a global, uniform, and fixed regularization; ERA is adaptive and input-dependent, offering greater representational power.

## Rating

- Novelty: ⭐⭐⭐⭐ The perspective of using output activations for entropy constraints is novel and unifies three distinct domains with strong generalizability.  
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers continuous control (5 algorithms/multiple envs), LLM reasoning (6 benchmarks/2 models/2 algorithms), and image classification with detailed ablations.  
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, tight integration of theory and experiments, and complete derivations.  
- Value: ⭐⭐⭐⭐ Lightweight (<7% overhead), plug-and-play, and theoretically guaranteed, offering high practical value for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints](autotool_automatic_scaling_of_tool-use_capabilities_in_rl_via_decoupled_entropy_.md)
- [\[ICLR 2026\] GoldenStart: Q-Guided Priors and Entropy Control for Distilling Flow Policies](goldenstart_q-guided_priors_and_entropy_control_for_distilling_flow_policies.md)
- [\[ICLR 2026\] Learning Massively Multitask World Models for Continuous Control](learning_massively_multitask_world_models_for_continuous_control.md)
- [\[ICLR 2026\] Relative Entropy Pathwise Policy Optimization](relative_entropy_pathwise_policy_optimization.md)
- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)

</div>

<!-- RELATED:END -->
