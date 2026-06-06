---
title: >-
  [Paper Note] OmniSapiens: A Foundation Model for Social Behavior Processing via Heterogeneity-Aware Relative Policy Optimization
description: >-
  [ICML 2026][Interpretability][Social Intelligence] Addressing the issue where learning signals in GRPO-style reasoning RL are dominated by a few tasks due to the natural heterogeneity of social behavior data (10 tasks sp…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Social Intelligence"
  - "Behavior Foundation Models"
  - "Heterogeneous RL"
  - "GRPO Improvement"
  - "Advantage Re-weighting"
date: 2026-05-08
content_hash: 83aeae28a153766c
---

# OmniSapiens: A Foundation Model for Social Behavior Processing via Heterogeneity-Aware Relative Policy Optimization

**Conference**: ICML 2026  
**arXiv**: [2602.10635](https://arxiv.org/abs/2602.10635)  
**Code**: https://github.com/MIT-MI/human_behavior_atlas  
**Area**: Human Understanding / Social Behavior Analysis / Multi-modal Foundation Models / Reasoning Reinforcement Learning  
**Keywords**: Social Intelligence, Behavior Foundation Models, Heterogeneous RL, GRPO Improvement, Advantage Re-weighting  

## TL;DR
Addressing the issue where learning signals in GRPO-style reasoning RL are dominated by a few tasks due to the natural heterogeneity of social behavior data (10 tasks spanning emotion/cognition/pathology/social domains and speech/vision/text modalities), this paper proposes HARPO. It approximates the contribution of each sample and task to policy updates using advantage magnitudes, then derives structured modulation factors via "geometric mean reference + reciprocal ratio" with inertial smoothing. Trained on Qwen 2.5-Omni-7B, OmniSapiens-7B 2.0 ranks 1st in multi-task average, wins all 5 zero-shot tasks, improves reasoning consistency from 66.5% to 87.7%, and reduces average token count to 19.86.

## Background & Motivation

**Background**: Social intelligence AI must simultaneously understand emotions, mental states, and social signals while generalizing to new scenarios. Existing approaches either utilize single-task experts (e.g., separate models for emotion classification and depression detection) or recent unified behavior foundation models (HumanOmniV2, OmniSapiens series) using SFT or GRPO for multi-task RL.

**Limitations of Prior Work**: The authors observe that behavior data is inherently heterogeneous—the reward distribution scales for SEN (sentence-level sentiment) and PTSD (long-video pathological states) differ by orders of magnitude, and their modal compositions are entirely different (text vs. audio+video+text). When directly applying GRPO, a few tasks or samples dominate the entire policy gradient due to systematically larger advantage magnitudes, causing F1 scores in tasks like SAR and SEN to drop from 70+ to single digits (see Table 1: RE++ SAR=5.01, GRPO HUM=27.56).

**Key Challenge**: While GRPO performs group-level reward normalization, there are **no scale constraints across groups or tasks**. Equation (4) simply sums the gradients of all rollouts, allowing those with larger absolute advantages to dominate updates. In the presence of natural heterogeneity, this aggregation degenerates into a "winner-takes-all" failure mode in multi-task learning.

**Goal**: To introduce an **explicit heterogeneity-aware mechanism** within a critic-free reasoning RL framework, automatically balancing the update influence at both sample and task levels without disrupting the overall GRPO training paradigm or global step size.

**Key Insight**: The authors note a concise fact: according to Eq. (5), the contribution of each rollout to the policy gradient is proportional to its absolute advantage $|\hat{A}|$. Thus, the **advantage magnitude itself serves as a computable proxy for "the actual contribution of this sample/task to the update"**, directly usable for inverse weighting without requiring an extra critic or auxiliary network.

**Core Idea**: Use the "reciprocal ratio under geometric mean reference" as a modulation factor to multiply the GRPO advantage. This scales down the advantages of high-contribution rollouts and scales up those with low contribution, while preserving the global step size due to the inherent property that the geometric mean of the factors equals 1.

## Method

### Overall Architecture
OmniSapiens-7B 2.0 uses Qwen 2.5-Omni-7B as the multi-modal backbone. Inputs consist of mixed behavior data (10 tasks from the Human Behavior Atlas, 100k+ samples, including SEN/EMO/SOC/INT/NVC/HUM/SAR/ANX/DEP/PTSD) across text, images, video, and audio. The output is an autoregressive sequence in the format of "Reasoning Chain + Predicted Label/Answer."

Training utilizes HARPO (Heterogeneity-Aware Relative Policy Optimization), which follows the GRPO structure (PPO clipped surrogate + KL regularization) but replaces the group-normalized advantage $\hat{A}_{(m,q,i)}$ with the modulated $A^H_{(m,q,i)}$. Rewards are weighted from three components: task reward $r_{task}$ (binary for classification, cosine for QA), format reward $r_{fmt}$ (weight 0.2), and length penalty $r_{len}$ (coefficient 0.75). The HARPO "modulator" is independent of the actor, online estimating contribution signals and updating modulation factors at training step $t$ to scale the advantages.

### Key Designs

1.  **Dual-level Contribution Estimation (sample-level + task-level)**:
    *   **Function**: Approximates the actual contribution of each sample and task to the policy gradient using a training-free, critic-free proxy signal as input for modulation.
    *   **Mechanism**: Based on Eq. (4)-(5), as rollout gradient contribution is directly scaled by $\hat{A}$, the **absolute value** of the group-normalized advantage is treated as the contribution signal. Sample-level signal is the mean absolute advantage within a rollout group $G(m,q)$: $p^{(t)}_{(m,q)} = \frac{1}{|G(m,q)|}\sum_i |\hat{A}^{(t)}_{(m,q,i)}|$, and task-level signal $p^{(t)}_m$ is the mean absolute advantage of all rollouts in the current batch for that task. Dividing by the number of rollouts ensures invariance to random batch sampling.
    *   **Design Motivation**: The root of the heterogeneity problem is "influence dominance," so a scalar quantifying "influence" is needed. Advantage magnitude is chosen over reward or loss because its coupling with the policy gradient is the most direct and computationally cheap.

2.  **Structured Modulation via Reciprocal Ratio under Geometric Mean Reference**:
    *   **Function**: Converts contribution signals into modulation factors with reasonable scales that do not disrupt the global update magnitude, achieving rebalancing by suppressing strong signals and boosting weak ones.
    *   **Mechanism**: For the sample level, the geometric mean of all sample contribution signals within a task $\bar{p}^{(t)}_{ref,m}$ is used as the reference; for the task level, the geometric mean of all task signals $\bar{p}^{(t)}_{ref,M}$ is used. Modulation factors are defined as the ratio of the reference to the signal: $s^{(t)}_{(m,q)} = \bar{p}^{(t)}_{ref,m}/p^{(t)}_{(m,q)}$ and $s^{(t)}_m = \bar{p}^{(t)}_{ref,M}/p^{(t)}_m$. The final modulated advantage is $A^H_{(m,q,i)} = s^{(t)}_{(m,q)} \cdot s^{(t)}_m \cdot \hat{A}^{(t)}_{(m,q,i)}$. Factors for contributions exceeding the reference are $<1$ (compression), while those below are $>1$ (amplification).
    *   **Design Motivation**: Geometric mean is preferred over arithmetic mean because contribution signals across tasks often differ by orders of magnitude; geometric averaging handles this heavy-tail distribution via multiplicative scaling. Crucially, the geometric mean naturally ensures the product of all modulation factors is 1 ($\prod_q s^{(t)}_{(m,q)} = 1$ and $\prod_m s^{(t)}_m = 1$), meaning amplification and compression strictly cancel each other out, **keeping the overall update step size constant** and avoiding interference with the global learning rate.

3.  **Inertial Smoothing for Stable Modulation**:
    *   **Function**: Allows the modulation mechanism to evolve on a slower time scale than the policy parameters, preventing on-policy single-step noise from disrupting modulation factors and causing training oscillations.
    *   **Mechanism**: Contribution signals are smoothed using EMA: $\bar{p}^{(t)} = \beta_\rho \bar{p}^{(t-1)} + (1-\beta_\rho) p^{(t)}$. Since modulation factors are multiplicative ratios, **multiplicative EMA** is used instead of additive EMA: $s^{(t)} = (s^{(t-1)})^{\beta_s}(s)^{1-\beta_s}$. Thus, modulation only tracks persistent trends in contribution signals and is immune to single-step stochastic perturbations.
    *   **Design Motivation**: Modulation factors act as "weights of weights" for policy updates. High volatility would re-introduce high variance into normalized advantages, worsening learning. Multiplicative updates naturally preserve the geometric mean = 1 invariant, consistent with Design 2.

### Loss & Training
The HARPO objective function is isomorphic to GRPO, merely substituting $\hat{A}$ in the clipped surrogate with $\tilde{A}^H_{(m,q,i):k}(\theta)$:

$$J_{HARPO}(\theta) = \mathbb{E}\big[\frac{1}{|G|}\sum_i \frac{1}{n_o}\sum_k \tilde{A}^H_{(m,q,i):k}(\theta)\big] - \beta \mathbb{E}[D_{KL}(\pi_\theta \| \pi_{ref})]$$

Training data is the Human Behavior Atlas (Ong et al., 2026), a multi-modal RL dataset covering 10 behavior tasks. Using Qwen 2.5-Omni-7B as the base model and unified reward designs, all compared RL algorithms were run on the same data/base for fairness.

## Key Experimental Results

### Main Results: Multi-task Performance on 10 Tasks (Selected from Tab. 1)

| Model / Algorithm | EMO | HUM | SAR | INT | DEP | Avg. Rank ↓ |
|------|------|------|------|------|------|------|
| Qwen 2.5-Omni-7B (base) | 58.25 | 54.30 | 65.60 | 25.40 | 71.35 | 6.20 |
| HumanOmniV2-7B | 59.70 | 63.80 | 39.50 | 26.30 | 65.40 | 5.80 |
| OmniSapiens BAM | 64.53 | 64.40 | 79.50 | 17.70 | 78.85 | 3.30 |
| OmniSapiens-7B RL (GRPO) | 57.28 | 63.90 | 64.70 | 48.60 | 77.15 | 4.20 |
| **OmniSapiens-7B 2.0 (HARPO)** | **76.55** | **69.85** | 70.64 | **50.52** | **78.87** | **1.90** |

At the model level, it achieved top-2 in 8 out of 10 tasks with an average rank of 1.90, the best among all baselines.

| RL Algorithm (Same Base/Data) | HUM | SAR | SEN | INT | Avg. Rank ↓ |
|------|------|------|------|------|------|
| GRPO | 27.56 | 53.58 | 77.51 | 49.90 | 3.90 |
| RE++ | 60.26 | 50.21 | 56.52 | 5.01 | 4.50 |
| RLOO | 67.86 | 62.58 | 76.86 | 51.73 | 2.80 |
| GPG | 69.28 | 45.96 | 75.77 | 54.21 | 2.90 |
| EMAGRPO | 63.50 | 77.75 | 68.28 | 52.62 | 3.10 |
| **HARPO** | **69.85** | 70.64 | 77.61 | 50.52 | **2.10** |

At the algorithm level, GRPO collapsed to 27.56 on HUM, and RE++ collapsed to 5.01 on SAR/INT. HARPO was the only algorithm that did not collapse on any of the 10 tasks, showing a maximum gain of +42.29% over GRPO.

### Zero-shot Generalization (Tab. 2) and Reasoning Quality (Tab. 3)

| Model | AUT | SER | IDR | SMSA | SIR | Consistency ↑ | Avg. Tokens ↓ |
|------|------|------|------|------|------|------|------|
| Qwen 2.5-Omni-7B | 25.68 | 53.53 | 70.25 | 44.64 | 34.99 | 34.0 | 73.66 |
| HumanOmniV2 | 38.05 | 62.74 | 21.97 | 53.06 | 37.45 | 50.0 | 195.90 |
| OmniSapiens-7B RL | 30.46 | 55.77 | 69.29 | 55.03 | 66.53 | 55.1 | 57.69 |
| **OmniSapiens 2.0** | **39.91** | **72.11** | **72.43** | **58.47** | **69.27** | **87.7** | **19.86** |

It won all 5 held-out tasks, while reasoning consistency jumped from 66.5% to 87.7%, and average tokens were reduced to 19.86 (less than 35% of the next best OmniSapiens-7B RL).

### Key Findings
- The key to HARPO's success is not just a "higher average" but a "stabler baseline"—while GRPO/RE++/GPG collapse to single-digit F1 on certain tasks, HARPO is the only algorithm to remain competitive across all 10 tasks, validating the core role of heterogeneity-aware modulation in multi-task RL balance.
- More balanced multi-task training leads to stronger zero-shot transfer: OmniSapiens 2.0 and OmniSapiens RL use the same data and backbone, differing only in the RL algorithm. Improvements across all 5 held-out tasks led the authors to hypothesize that "more uniform multi-task learning facilitates more transferable behavior representations."
- Reasoning becomes shorter yet more accurate: HARPO enables the model to learn reasoning chains averaging only 19.86 tokens while achieving the highest consistency (87.7%). Human evaluation showed win rates of 68.5%/85.1%/99.2% for specificity/coherence/concision vs. four baselines, suggesting advantage rebalancing suppresses the "long but vacuous" reasoning degradation.

## Highlights & Insights
- Using advantage magnitude itself as a contribution signal is an elegant idea: it requires no critic, no gradient estimation, and no extra forward passes. It is a zero-cost proxy directly derived from the coupling between the advantage and the policy gradient.
- The combination of geometric mean reference, reciprocal ratio, and multiplicative EMA ensures "global step size conservation." The invariant $\prod s = 1$ allows HARPO to perform local re-weighting without polluting the global learning rate, avoiding the classic multi-task re-weighting pitfall where "tuning weights requires re-tuning the LR."
- This method can be **directly transferred to any GRPO-based RL training**. As long as the training data is naturally heterogeneous (mixed math+code+chat, multi-lingual, or multi-modal), the HARPO modulation layer can be used plug-and-play. It is compatible with RLOO/REINFORCE++/GPG and could likely become a standard module for reasoning RL.

## Limitations & Future Work
- The authors acknowledge that the causal link between HARPO and zero-shot improvement is an empirical observation lacking rigorous theoretical or controlled analysis. Currently, it is "better balance → better transfer" correlation rather than causation.
- The use of absolute advantage for contribution signals makes the method highly dependent on reward design. If rewards are highly noisy or universally small for a task, the geometric mean reference may become numerically unstable.
- All experiments were limited to the Qwen 2.5-Omni-7B backbone and the Human Behavior Atlas dataset. Scaling behavior on larger models (70B+) or different RL data (math/code reasoning) remains unverified, especially the stability of the geometric mean as task count $|M|$ grows.
- Future directions: Expanding HARPO modulation to "difficulty levels" (binning by reward variance) or "prompt levels" for finer-grained awareness, or upgrading the contribution signal to gradient norms or Fisher Information.

## Related Work & Insights
- **vs. GRPO (Shao et al., 2024)**: GRPO only performs intra-group reward normalization but lacks scale alignment across tasks. HARPO adds a heterogeneity-aware modulation layer on top of GRPO with zero changes to the PPO objective function, offering a minimally invasive extension.
- **vs. EMAGRPO (Feng et al., 2025)**: EMAGRPO also uses EMA for multi-task balancing but applies it at the reward or loss level. HARPO acts directly on the advantage level and introduces structured constraints for global step size conservation, avoiding the step size drift common in EMA methods.
- **vs. Traditional Multi-task RL (gradient balancing / uncertainty weighting)**: These methods often rely on gradient back-propagation estimation or extra learnable weights. HARPO follows a lightweight route with zero extra parameters and zero extra forward passes, fitting the simplicity of critic-free reasoning RL.
- **vs. HumanOmniV2 / OmniSapiens RL**: Within the same social behavior foundation model family, OmniSapiens 2.0 achieves total dominance across 10 tasks and 5 zero-shot tasks simply by changing the RL algorithm. This suggests that in unified behavior modeling, **RL training paradigms are a greater bottleneck than backbone or data scale**.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of "advantage magnitude as contribution" and "geometric mean reciprocal ratio" for heterogeneous RL modulation is a clear new synthesis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Rigorous comparison of 6 RL algorithms on the same base/data/reward, covering 10 training tasks, 5 held-out tasks, reasoning consistency, and human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations, particularly the explanation of the global step size invariant. However, sensitivity analysis for hyperparameters ($\beta_s, \beta_\rho$) is slightly sparse.
- Value: ⭐⭐⭐⭐⭐ Provides both a reusable social behavior foundation model and a plug-and-play heterogeneity-aware module for all GRPO-style reasoning RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](../../ICLR2026/interpretability/behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)
- [\[ACL 2026\] Dual Alignment Between Language Model Layers and Human Sentence Processing](../../ACL2026/interpretability/dual_alignment_between_language_model_layers_and_human_sentence_processing.md)
- [\[ICML 2026\] Discovering Differences in Strategic Behavior Between Humans and LLMs](discovering_differences_in_strategic_behavior_between_humans_and_llms.md)
- [\[ICML 2026\] Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models](is_one_layer_enough_understanding_inference_dynamics_in_tabular_foundation_model.md)
- [\[ICML 2026\] Courtroom Analogy: New Perspective on Uncertainty-Aware Classification](courtroom_analogy_new_perspective_on_uncertainty-aware_classification.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](../../ICLR2026/interpretability/behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)
- [\[ACL 2026\] Dual Alignment Between Language Model Layers and Human Sentence Processing](../../ACL2026/interpretability/dual_alignment_between_language_model_layers_and_human_sentence_processing.md)
- [\[ICML 2026\] Discovering Differences in Strategic Behavior Between Humans and LLMs](discovering_differences_in_strategic_behavior_between_humans_and_llms.md)
- [\[ICML 2026\] Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models](is_one_layer_enough_understanding_inference_dynamics_in_tabular_foundation_model.md)
- [\[ICML 2026\] Courtroom Analogy: New Perspective on Uncertainty-Aware Classification](courtroom_analogy_new_perspective_on_uncertainty-aware_classification.md)

</div>

<!-- RELATED:END -->
