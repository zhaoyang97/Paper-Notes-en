---
title: >-
  [Paper Note] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning
description: >-
  [ICLR 2026][Reinforcement Learning][RLVR] CalibRL redefines expert data as a distribution calibration baseline (rather than a strict imitation target), achieving fine-grained control over the exploration-exploitation balance in MLLM reasoning training through LeakyReLU asymmetric activation and advantage weighting. This effectively addresses the entropy collapse problem in RLVR and significantly outperforms GRPO/DAPO on geometric reasoning tasks.
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "RLVR"
  - "Hybrid Policy Optimization"
  - "Multi-modal Reasoning"
  - "Entropy Collapse"
  - "Controllable Exploration"
date: 2026-05-08
content_hash: d13145e9bbfa48e8
---

# Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning

**Conference**: ICLR 2026  
**arXiv**: [2602.20197](https://arxiv.org/abs/2602.20197)  
**Code**: [https://github.com/zhh6425/CalibRL](https://github.com/zhh6425/CalibRL)  
**Area**: Multi-modal VLM / Reinforcement Learning  
**Keywords**: RLVR, Hybrid Policy Optimization, Multi-modal Reasoning, Entropy Collapse, Controllable Exploration

## TL;DR
CalibRL redefines expert data as a distribution calibration baseline (rather than a strict imitation target), achieving fine-grained control over the exploration-exploitation balance in MLLM reasoning training through LeakyReLU asymmetric activation and advantage weighting. This effectively addresses the entropy collapse problem in RLVR and significantly outperforms GRPO/DAPO on geometric reasoning tasks.

## Background & Motivation

**Background**: RLVR has become a mainstream paradigm for enhancing the reasoning capabilities of MLLMs (e.g., DeepSeek-R1). However, performance improvements are often accompanied by a significant decline in policy entropy—entropy depletion has become a bottleneck for further advancement.

**Limitations of Prior Work**:
   - Traditional entropy regularization encourages randomness but lacks directionality → low exploration efficiency.
   - SFT-then-RL paradigm: SFT fixes the policy within a static demonstration distribution → weakens subsequent RL exploration.
   - Hybrid-policy frameworks: Direct injection of SFT supervision → distribution mismatch between the current policy and expert trajectories → high bias and variance → accelerates entropy collapse.

**Key Challenge**: Expert data is a double-edged sword—it provides useful guidance but also compresses the policy distribution. Maximizing the probability of expert trajectories $\pi_\theta(\tau^{expert})$ inevitably reduces the probability of other trajectories → total entropy decreases.

**Key Insight**: One should not "imitate" experts (absolute target) but rather use experts as a "reference baseline" for relative calibration—reinforcing under-represented correct reasoning paths while suppressing overconfident incorrect predictions.

**Core Idea**: Transform expert supervision from a rigid imitation signal into a refined calibration mechanism, achieving directional and regulated exploration through the log-probability gap and LeakyReLU asymmetric gating.

## Method

### Overall Architecture
CalibRL aims to solve entropy collapse during MLLM training in RLVR. Previous approaches, whether SFT-then-RL or hybrid-policy frameworks that incorporate expert supervision directly, essentially maximize the log-likelihood of expert trajectories. This forces the policy to converge unidirectionally toward the expert distribution—as the probability of expert trajectories increases, the probability of other trajectories is inevitably suppressed, leading to a decrease in total entropy and exploration depletion. CalibRL breaks this by changing the perspective: experts are treated not as "imitation targets that must be reached" but as "reference coordinates to measure where the current policy deviates." It introduces a controllable exploration term $\mathcal{L}_{exploration}$ on top of GRPO for **relative calibration**.

For each response in a sampled group for a given prompt, the mechanism follows three steps: first, calculate the model’s preference for its own answer relative to the expert (log-probability gap $\Delta\ell_i$) to determine if the model is overconfident or underconfident; second, combined with the correctness of the answer, use an asymmetric LeakyReLU gate to decide whether the response should be reinforced (pushed up) or suppressed (pushed down), automatically reducing the force once it crosses the expert baseline; finally, weight the update by the absolute value of the group advantage, allowing rare but informative responses to dominate the update. These three steps manage "signal," "direction/intensity," and "weight," respectively, synthesizing an exploration term integrated into the GRPO objective.

### Key Designs

**1. Log-Probability Gap: Using relative confidence (Model vs. Expert) as an exploration signal**

Entropy regularization only encourages randomness without direction, leading to low exploration efficiency in the vast state space of MLLMs. CalibRL seeks "directional exploration" by defining a relative metric:

$$\Delta\ell_i = \log\frac{\pi_\theta(\tau_i^{policy}\mid q_i)}{\pi_\theta(\tau_i^{expert}\mid q_i)} = \log\pi_\theta(\tau_i^{policy}\mid q_i) - \log\pi_\theta(\tau_i^{expert}\mid q_i)$$

A positive $\Delta\ell_i$ indicates the model already prefers its generated trajectory and is more confident than the expert; a negative value indicates confidence is still below the expert baseline. All subsequent reinforcement or suppression is driven by this relative signal—this is the key to demoting experts from "imitation targets" to "reference baselines": the model does not need to approximate the expert's probability but only needs to know which side it has deviated to relative to the expert.

**2. LeakyReLU Asymmetric Activation: Releasing control after crossing the expert baseline to avoid overconfidence**

With the signal defined, the direction and intensity must be decided. CalibRL first marks the response with an independent correctness sign $s_i$ ($s_i=+1$ for correct, $s_i=-1$ for incorrect; defined separately because rewards including formatting may exceed $[0,1]$). The exploration term is formulated as:

$$\mathcal{L}_{exploration} = |\hat{A}_i| \cdot \text{LeakyReLU}(-s_i \cdot \Delta\ell_i, \alpha)$$

Multiplying by $s_i$ ensures the optimization direction aligns with correctness: reinforce correct responses where the model lacks confidence ($\Delta\ell_i$ is negative), and suppress incorrect responses where the model is overconfident ($\Delta\ell_i$ is positive). The asymmetry of LeakyReLU characterizes "how much force to apply"—the gradient is 1 when the input is positive (the response is on the "needs management" side of the expert baseline, requiring full force), and scales to $\alpha\in(0,1)$ when the input is negative (once the response probability crosses the expert baseline, reinforcement/suppression is weakened to prevent pushing the policy back toward overconfidence). Using LeakyReLU instead of pure ReLU (which would cut the gradient entirely and lose signals) or linear activation (which cannot distinguish between the "needs management" and "sufficient" regions) provides a balance.

**3. Advantage Weighting: Allowing rare but informative responses to dominate updates**

Finally, the exploration term for each response is weighted by the absolute value of the group-wise advantage $|\hat{A}_i|$. Since GRPO advantages are group-normalized, if most responses in a group are wrong and only a few are correct, the rare correct response will have a large $|\hat{A}_i|$. Consequently, it receives a larger weight and is reinforced as a valuable exploration signal. Conversely, in a group where most are correct, a rare incorrect response will also be highlighted for suppression. Updates are thus linked to how "rare and informative" a response is, concentrating exploration on meaningful deviations.

### Loss & Training
- The final objective integrates the exploration term into GRPO: $\mathcal{J}(\theta) = \mathcal{J}_{GRPO}(\theta) - \lambda\,\mathcal{L}_{exploration}$, where $\lambda$ balances standard PPO-style optimization with expert-guided exploration (the KL term in GRPO is often omitted during long CoT training).
- Key hyperparameters: $\alpha=0.5$ (the slope of LeakyReLU, identified as the sweet spot in ablations) and $\lambda$ to control exploration weight.
- Expert data consists of approximately 9.7K CoTs generated by GPT-4o for ViRL39K geometry problems (verified for correctness, format, and logic); ablations confirm the expert baseline is superior to the reference policy baseline.

## Key Experimental Results

### Main Results (Geometric Reasoning, In-task Benchmarks)

| Method | GeoEval↑ | Geo3K↑ | GeoQA↑ | Average↑ |
|------|---------|--------|--------|------|
| GRPO | 26.15 | 39.77 | 52.52 | 39.48 |
| SFT+GRPO | 6.00 | 18.64 | 40.98 | 21.87 |
| DAPO | 25.19 | 40.93 | 52.52 | 39.55 |
| **Ours (CalibRL)** | **33.44** | **40.60** | **60.74** | **44.93** |

### Ablation Study

| LeakyReLU $\alpha$ | Effect |
|-------------------|------|
| 0.3 | Aggressive early exploration but unstable with entropy fluctuations |
| **0.5** | **Balanced entropy growth without oscillations** |
| 0.8 | Over-constrained, leading to rapid entropy decay |

### Key Findings
- **SFT+GRPO performs the worst**: Directly mixing SFT and RL leads to severe entropy collapse—supporting the necessity of the "imitation to calibration" paradigm shift.
- **CalibRL is optimal both in-task and out-of-task**: Outperforms GRPO by an average of +5.45% in-task and exhibits better generalization.
- **Stable entropy curves**: CalibRL maintains steady policy entropy growth during training, while other methods show continuous decline.
- **$\alpha=0.5$ is the sweet spot**: Too small leads to instability; too large restricts exploration.

## Highlights & Insights
- **"Calibration instead of imitation" is a profound reimagining of hybrid-policy RL**—expert data should not be viewed as a goal to be reached, but as a reference coordinate to measure current policy deviation.
- **LeakyReLU provides an elegant gradient gating mechanism**—a simple choice of activation function enables adaptive control of "reinforce when needed, weaken when sufficient."
- **The finding that "SFT+GRPO is worst" is a crucial warning for practitioners**—simply stacking SFT and RL may result in mutual interference.

## Limitations & Future Work
- Validated only on geometric reasoning—math and code reasoning domains remain to be explored.
- The $\alpha$ parameter for LeakyReLU requires tuning—more adaptive activation function designs might be superior.
- The quality of expert data affects the reliability of the calibration baseline.
- Integration with other RL variants (e.g., Dr.GRPO, CPPO) has not yet been explored.

## Related Work & Insights
- **vs. GRPO/DAPO**: Standard policy optimization does not handle entropy collapse; CalibRL maintains exploration via the calibration mechanism.
- **vs. LUFFY**: Also a hybrid-policy framework but still uses imitation-style supervision → still suffers from distribution mismatch.
- **vs. SFT+GRPO**: Direct serial connection leads to catastrophic interference; CalibRL’s calibration paradigm avoids this issue.

## Rating
- Novelty: ⭐⭐⭐⭐ The "calibration instead of imitation" paradigm is insightful, and the LeakyReLU application is clever.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablations are thorough, though the task scope is relatively narrow (focused on geometric reasoning).
- Writing Quality: ⭐⭐⭐⭐ In-depth problem analysis with clear theoretical motivation.
- Value: ⭐⭐⭐⭐ A practical solution for solving entropy collapse in RLVR with implications for hybrid-policy training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment](../../ACL2026/reinforcement_learning/healing_entropy_collapse_enhancing_exploration_in_few-shot_rlvr_via_hybrid-domai.md)
- [\[ICLR 2026\] Generalization of RLVR Using Causal Reasoning as a Testbed](generalization_of_rlvr_using_causal_reasoning_as_a_testbed.md)
- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](../../ACL2026/reinforcement_learning/semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)
- [\[ICLR 2026\] Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](exploration_vs_exploitation_rethinking_rlvr_through_clipping_entropy_and_spuriou.md)
- [\[ICLR 2026\] MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding](mergemix_a_unified_augmentation_paradigm_for_visual_and_multi-modal_understandin.md)

</div>

<!-- RELATED:END -->
