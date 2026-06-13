---
title: >-
  [Paper Note] SmartThinker: Progressive Chain-of-Thought Length Calibration for Efficient Large Language Model Reasoning
description: >-
  [ICML 2026][LLM Reasoning][GRPO] This paper proposes SmartThinker, an efficient post-training method based on GRPO. By performing Gaussian modeling on the "total trajectory length distribution" and the "correct trajector…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "GRPO"
  - "CoT Length Calibration"
  - "Dynamic Reward"
  - "Overthinking"
  - "Optimal Reasoning Length"
date: 2026-05-08
content_hash: 58b7136cf30335b8
---

# SmartThinker: Progressive Chain-of-Thought Length Calibration for Efficient Large Language Model Reasoning

**Conference**: ICML 2026  
**arXiv**: [2603.08000](https://arxiv.org/abs/2603.08000)  
**Code**: https://github.com/SJTU-RTEAS/SmartThinker (Available)  
**Area**: LLM Reasoning / RL Post-training / Efficiency Optimization  
**Keywords**: GRPO, CoT Length Calibration, Dynamic Reward, Overthinking, Optimal Reasoning Length

## TL;DR
This paper proposes SmartThinker, an efficient post-training method based on GRPO. By performing Gaussian modeling on the "total trajectory length distribution" and the "correct trajectory length distribution" for each prompt, it analytically derives the "optimal length" $l^{\text{opt}}$ that maximizes accuracy. Paired with a dynamic length reward coefficient $\Lambda$ to ensure non-negative normalized advantages for correct trajectories, the method achieves up to a 52.6% token compression while relatively improving AIME25 accuracy by up to 16.6%.

## Background & Motivation

**Background**: Large Reasoning Models (LRM), represented by OpenAI o1 and DeepSeek-R1, achieve high accuracy through lengthy chain-of-thought (CoT). However, longer CoT leads to higher token consumption and latency, while increasing the risk of "thinking off-track" for simple problems—a phenomenon known as overthinking. To compress CoT, mainstream solutions add a length reward to encourage shorter outputs within the GRPO framework, such as ShorterBetter, ThinkPrune, LASER-DE, and L1.

**Limitations of Prior Work**: The authors observe that the reward designs in these methods are "static," leading to two fundamental issues. First, the length reward $r_i^{\text{len}}$ only considers its own trajectory length without accounting for the joint distribution of length and correctness across the group, thus failing to perceive the relative difficulty of the problem. Second, the weight coefficient $\lambda$ is a fixed hyperparameter. After reward normalization in GRPO, "long but correct" trajectories are easily assigned negative advantages, making them indistinguishable from "incorrect trajectories" and suppressing necessary exploration.

**Key Challenge**: The relationship between CoT length and accuracy follows an inverted U-shape—there exists an intermediate length $l^{\text{opt}}$ that maximizes the conditional probability $\Pr(r^{\text{acc}}=1 \mid l, q; \theta)$. Crude linear length penalties may exceed this optimal point, causing over-compression. Meanwhile, static $\lambda$ values corrupt the semantic meaning of GRPO advantages, mixing signals for "correct" and "overly long" trajectories.

**Goal**: To simultaneously address two issues within the GRPO framework: (1) how to dynamically estimate $l^{\text{opt}}$ based on problem difficulty; (2) how to dynamically adjust the length reward weight based on group accuracy to ensure non-negative advantages for all correct trajectories and non-positive advantages for all incorrect trajectories.

**Key Insight**: Leverage the fact that a single GRPO rollout generates $G$ trajectories—the set of lengths $\mathcal{L}$ and the set of correct trajectory lengths $\mathcal{L}^{\text{acc}}$ naturally provide samples for two distributions. Assuming both are approximately Gaussian allows for Bayesian inference of $\Pr(r^{\text{acc}}=1\mid l)$ and an analytical solution for the optimal length.

**Core Idea**: Transform "how long to think" into a target dynamically calculated based on the current policy and problem, rather than a hyperparameter. Simultaneously, calculate the "length penalty weight" dynamically to prevent length terms from polluting the sign semantics of GRPO advantages.

## Method

### Overall Architecture
SmartThinker inserts two dynamic calculation steps into the GRPO training loop. For each prompt $q$, the policy $\pi_\theta$ first rolls out a group of $G$ trajectories $\{o_1,\dots,o_G\}$ with lengths $l_i$ and correctness $r_i^{\text{acc}}\in\{0,1\}$. Based on these samples: (i) Gaussian distributions $(\hat\mu_1,\hat\sigma_1)$ and $(\hat\mu_2,\hat\sigma_2)$ are estimated using $\mathcal{L}$ and $\mathcal{L}^{\text{acc}}$ to analytically solve for the optimal length $\hat l^{\text{opt}}$; (ii) a unilateral ReLU-form length penalty $r_i^{\text{len}}$ is calculated for each correct trajectory based on $\hat l^{\text{opt}}$; (iii) the length weight $\Lambda$ is calculated based on the group error rate $p^{\text{err}}$; (iv) the total reward is synthesized as $r_i = r_i^{\text{acc}} + \Lambda \cdot r_i^{\text{len}}$, followed by standard GRPO normalized advantage $\hat A_i$ calculation and policy updates. The mechanism requires no value network, introduces no extra sampling, and can serve as a plug-in for specific stages of multi-stage frameworks like AutoThink or ThinkPrune.

### Key Designs

1.  **Probabilistic Modeling and Analytical Solution for Optimal Length**:
    - **Function**: Calculates a target length $\hat l^{\text{opt}}$ for each prompt during every training step to maximize the probability of a correct answer, serving as a "target" for the length reward.
    - **Mechanism**: Assume total trajectory lengths follow $N(\mu_1,\sigma_1^2)$ and correct trajectory lengths follow $N(\mu_2,\sigma_2^2)$. Using Bayes' theorem, an analytical expression for $\Pr(r^{\text{acc}}=1\mid l)$ with respect to $l$ can be derived. The paper proves that a unique finite maximum exists if and only if $\sigma_1^2>\sigma_2^2$, where $l^{\text{opt}} = \frac{\sigma_1^2 \mu_2 - \sigma_2^2 \mu_1}{\sigma_1^2 - \sigma_2^2}$; other cases collapse to $\max(\mathcal L)$ or $\min(\mathcal L)$. In practice, sample mean and variance are substituted and clipped to $[\min\mathcal L, \max\mathcal L]$.
    - **Design Motivation**: Previous methods like ShorterBetter directly take the "minimum correct trajectory length" as the target, which is often an outlier of the distribution and leads to performance drops. Using the extremum point that maximizes conditional accuracy has a theoretical basis and allows for adaptive scaling—$\hat l^{\text{opt}}$ is shorter for simple problems to encourage refinement and longer for difficult problems to preserve exploration.

2.  **Unilateral ReLU Length Reward Based on Optimal Length**:
    - **Function**: Applies the "compress CoT" signal only to trajectories that are "correct but too long," with the length reward being zero otherwise.
    - **Mechanism**: Define $r_i^{\text{len}} = 0$ if $r_i^{\text{acc}}=0$, otherwise $r_i^{\text{len}} = -\operatorname{ReLU}(l_i - \hat l^{\text{opt}})$. Incorrect trajectories do not participate in the length signal to prevent double negative advantages from creating "shortcuts" to incorrect samples. Furthermore, if $\hat l^{\text{opt}} \geq \max(\mathcal L^{\text{acc}})$, the group length reward is 0, and SmartThinker automatically degrades to standard GRPO to focus on reasoning capability—forming an "on-demand compression" switch.
    - **Design Motivation**: Existing methods often use symmetric or linear length penalties for all trajectories, treating long correct trajectories and long incorrect trajectories equally. This prevents the model from distinguishing "exploratory length" from "erroneous deviations." The unilateral ReLU precisely targets the interval of "correct but exceeding optimal length."

3.  **Dynamic Length Reward Coefficient $\Lambda$**:
    - **Function**: Automatically calculates the relative weight of the length term against the accuracy term to ensure that, after GRPO normalization, all correct trajectories have non-negative advantages and incorrect trajectories have non-positive advantages.
    - **Mechanism**: Require $1+\lambda r_i^{\text{len}} \geq \operatorname{mean}(\boldsymbol r^{\text{acc}} + \lambda \boldsymbol r^{\text{len}})$ to hold for all correct trajectories. Solving for $\lambda$ with $r_i^{\text{len}}\leq 0$ yields a feasible upper bound $\lambda \leq \frac{p^{\text{err}}}{\operatorname{mean}(\boldsymbol r^{\text{len}}) - \min(\boldsymbol r^{\text{len}})}$, where $p^{\text{err}}$ is the group error rate. To maximize compression efficiency, the upper bound is chosen: $\Lambda = \frac{p^{\text{err}}}{\operatorname{mean}(\boldsymbol r^{\text{len}}) - \min(\boldsymbol r^{\text{len}})}$.
    - **Design Motivation**: A fixed $\lambda$ can cause "long but correct" trajectories to receive negative advantages, misidentifying them as samples to be punished. Linking $\Lambda$ to the group error rate implies: the more incorrect trajectories (higher difficulty), the stronger the length penalty; if all trajectories are correct ($p^{\text{err}}=0$), then $\Lambda=0$, turning off the length reward. This aligns with the intuition of "preserving exploration for hard problems and compressing boldly for simple ones" while eliminating manual $\lambda$ tuning.

### Loss & Training
The total reward is $r_i = r_i^{\text{acc}} + \Lambda(\boldsymbol r^{\text{acc}}, \boldsymbol r^{\text{len}}) \cdot r_i^{\text{len}}$. Normalization yields $\hat A_i = (r_i - \operatorname{mean}\{r_j\}) / \operatorname{std}\{r_j\}$, followed by the standard GRPO objective $\max_\theta \frac{\pi_\theta(o_i\mid q)}{\pi_{\text{old}}(o_i\mid q)} \hat A_i$. Implementation is based on verl with batch=64, group=8, minibatch=16, max length=8000, lr $=1\times 10^{-6}$, and no KL loss; 1.5B/7B/4B models are trained for only 150/75/50 steps respectively.

## Key Experimental Results

### Main Results

Comparison of SmartThinker against static length reward baselines using three base models on MATH500, AIME25, and AMC23 (results for DeepSeek-R1-Distill-Qwen-1.5B):

| Method | Math500 Len/Acc | AIME25 Len/Acc | AMC23 Len/Acc | Avg Acc | AE↑ |
|------|------------------|-----------------|----------------|----------|-----|
| Base Model | 5420 / 84.9 | 15199 / 24.2 | 9320 / 73.1 | 60.7 | N/A |
| ShorterBetter | 1008 / 71.0 | 3727 / 19.0 | 2246 / 66.9 | 52.3 | 0.07 |
| ThinkPrune-4k | 2744 / 84.1 | 7462 / 22.5 | 4201 / 76.3 | 60.95 | 0.53 |
| LASER-DE-4096 | 2720 / 85.1 | 7706 / 22.5 | 4330 / 71.9 | 59.8 | 0.42 |
| **SmartThinker** | 2645 / 84.5 | 8431 / **25.0** | 4421 / **76.3** | **61.9** | **0.54** |

On the 7B model, AIME25 accuracy increased from 35.0 → 40.8 (16.6% relative gain). On Qwen3-4B-Thinking-2507, average tokens decreased from 13040 → 7747 (~41% compression) while average accuracy rose from 88.5 → 89.0.

### Ablation Study

Ablation of the two dynamic mechanisms on DeepSeek-R1-Distill-Qwen-1.5B:

| Configuration | Avg Len | Avg Acc | Description |
|------|-----------|----------|------|
| Fixed Coefficient | 3644 | 57.5 | Fixed $\lambda$ causes long correct trajectories to incorrectly have negative advantages; accuracy drops by 4.4. |
| Symmetric | 5530 | 60.2 | Pulls all correct trajectories toward $\hat l^{\text{opt}}$ without unilateralism; compression weakens. |
| Linear | 4242 | 58.2 | Linear reward instead of ReLU; accuracy drops by 3.7. |
| **SmartThinker** | 5169 | **61.9** | Unilateral ReLU + dynamic $\Lambda$ both enabled. |

On OOD tasks (MMLU, MathQA, LiveCodeBench, HumanEval), the 1.5B model's average length went from 5575 → 3583 with Acc 55.78 → 56.50; for the 4B model, length 5781.5 → 4231.25 with Acc 83.97 → 84.43. Efficiency gains from math training transfer to general tasks.

### Key Findings
- The dynamic length reward coefficient $\Lambda$ provides the largest contribution: removing it (Fixed Coefficient) causes average accuracy to drop by 4.4 points, showing that "ensuring non-negative advantage for correct trajectories" is critical.
- SmartThinker is the only method in the table that consistently improves average accuracy across all base models, proving that "dynamic target length per problem" avoids the precision loss of static compression on hard problems.
- During training, $\hat l^{\text{opt}}$ was observed to be consistently lower than the actual output length, confirming that overthinking is widespread and that the optimal length itself evolves with the policy.
- As a plug-in for the final stage of AutoThink/ThinkPrune, it requires fewer training stages and yields better results (AE 0.55 vs 0.50; 0.58 vs 0.54).

## Highlights & Insights
- **Analytical Derivation of Thinking Duration**: Using two Gaussian distributions and Bayesian inference to find the closed-form solution of $l^{\text{opt}}$ provides the first theoretical target for length rewards based on conditional accuracy, rather than heuristic intuition.
- **Reward Weights Deduced from Error Rate**: The formula $\Lambda = p^{\text{err}}/(\operatorname{mean}-\min)$ is derived directly from the "advantage sign constraint." This eliminates manual hyperparameters and encodes the semantics of "penalizing length more on hard problems" into the weight. This approach of "deriving reward shapes from GRPO advantage normalization" is transferable to any GRPO design with auxiliary rewards.
- **Unilateral ReLU + Automatic Degradation**: When $\hat l^{\text{opt}}\geq\max(\mathcal L^{\text{acc}})$, the length reward automatically zeroes out, reverting to original GRPO. This acts as an internal adaptive switch to stop compression once it is "short enough" and focus back on accuracy, preventing collapse from over-compression.

## Limitations & Future Work
- The Gaussian assumption is highly idealized: when intra-group samples are few or the length distribution is heavily multi-modal, estimates of $\hat\mu$ and $\hat\sigma$ are noisy, potentially destabilizing $\hat l^{\text{opt}}$. Clipping to $[\min\mathcal L,\max\mathcal L]$ mitigates but does not root out this issue.
- Validation was limited to GRPO; empirical evidence for other variants like DAPO, GSPO, or SAPO is missing.
- Only applicable to tasks with verifiable correctness (math/code); fails in open-ended generation where $r^{\text{acc}}\in\{0,1\}$ cannot be defined.
- It remains an outcome-only reward without process reward supervision, meaning it cannot identify fine-grained beneficial patterns within the CoT. The authors list "SmartThinker + process reward" as a future direction.

## Related Work & Insights
- **vs ShorterBetter**: Both add length rewards to GRPO and compute dynamic target lengths per prompt. However, ShorterBetter directly uses the "minimum correct trajectory length," which is an outlier of the distribution and prone to performance drops; SmartThinker uses the analytical extremum for maximizing conditional accuracy, which is more robust and avoids length signals for incorrect trajectories.
- **vs ThinkPrune / LASER-DE**: These use a fixed token budget (e.g., 4k/4096), essentially "cutting all problems with a global threshold." SmartThinker’s $\hat l^{\text{opt}}$ is per-prompt and per-step adaptive, thus avoiding over-compression on hard problems (e.g., AIME25).
- **vs L1 / AutoThink**: L1 conditions the expected length in the prompt, while AutoThink uses multi-stage curriculum compression. SmartThinker is single-stage and plug-and-play. Experiments show that replacing stage 3 of AutoThink with SmartThinker yields better results, suggesting that "dynamic reward design" may be superior to "multi-stage curricula."

## Rating
- Novelty: ⭐⭐⭐⭐ Modeling "optimal CoT length" as a conditional probability extremum between two Gaussians and solving it analytically is a clean and original theoretical framing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage across three base models, three math benchmarks, four OOD tasks, two multi-stage framework integrations, and three reward configuration ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear logic chain from motivation to theory to algorithm to experiments. Section 2.3 categorizes "static" issues effectively.
- Value: ⭐⭐⭐⭐ Can serve as a drop-in replacement for existing GRPO length compression methods. Achieving up to 52.6% compression without performance loss (or with gains) is highly practical for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Training Large Reasoning Models Efficiently via Progressive Thought Encoding](../../ICLR2026/llm_reasoning/training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)
- [\[ICML 2026\] DecepChain: Inducing Deceptive Reasoning in Large Language Models](decepchain_inducing_deceptive_reasoning_in_large_language_models.md)
- [\[AAAI 2026\] Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](../../AAAI2026/llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)
- [\[ICML 2026\] On the Generalization Gap in Self-Evolving Language Model Reasoning](on_the_generalization_gap_in_self-evolving_language_model_reasoning.md)
- [\[ICLR 2026\] InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models](../../ICLR2026/llm_reasoning/inftythink_breaking_the_length_limits_of_long-context_reasoning_in_large_languag.md)

</div>

<!-- RELATED:END -->
