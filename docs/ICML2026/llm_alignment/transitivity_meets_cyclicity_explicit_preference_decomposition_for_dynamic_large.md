---
title: >-
  [Paper Note] HRC + DSPPO: Separating Transitive and Cyclic Preferences via Game-Theoretic Decomposition
description: >-
  [ICML 2026][LLM Alignment][Preference Modeling] HRC explicitly decomposes human preferences into orthogonal "transitive scalar components" (BT model) and "cyclic vector components" (GPM). Using game-theoretic decompositi…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "Preference Modeling"
  - "Bradley-Terry"
  - "GPM"
  - "Cyclic Preferences"
  - "Time-varying games"
  - "Self-play"
date: 2026-05-08
content_hash: 6cd7467c40044277
---

# HRC + DSPPO: Separating Transitive and Cyclic Preferences via Game-Theoretic Decomposition

**Conference**: ICML 2026  
**arXiv**: [2605.17342](https://arxiv.org/abs/2605.17342)  
**Code**: https://github.com/lab-klc/Hybrid-Reward-Cyclic  
**Area**: LLM Alignment / Preference Modeling / RLHF  
**Keywords**: Preference Modeling, Bradley-Terry, GPM, Cyclic Preferences, Time-varying games, Self-play

## TL;DR
HRC explicitly decomposes human preferences into orthogonal "transitive scalar components" (BT model) and "cyclic vector components" (GPM). Using game-theoretic decomposition theorems, it is proven that this hybrid form preserves dominant candidates while modeling RPS-style cycles. Accompanied by the time-varying game DSPPO, the alignment process transitions from "stabilizing the transitive backbone" to "learning cyclic details," converging to a Nash equilibrium. Gemma-2B-it achieves a 1.23% average gain on RewardBench 2 and a 44.75% AlpacaEval 2.0 LC win-rate.

## Background & Motivation

**Background**: RLHF typically employs the Bradley-Terry model to represent preferences as scalar reward differences, $\mathbb{P}_{\mathrm{BT}}(\mathbf{y} \succ \mathbf{y}') = \sigma(r(\mathbf{y}) - r(\mathbf{y}'))$, assuming preferences satisfy transitivity: $A \succ B \land B \succ C \Rightarrow A \succ C$. However, works like Tversky 1969 and Munos et al. 2024 highlight prevalent cyclic patterns (e.g., Rock-Paper-Scissors dynamics) in human preferences. While PairRM/PairPM can express cycles by learning pairwise functions, their inference complexity is $O(K^2)$. The General Preference Model (GPM) uses a skew-symmetric bilinear form $s_{\mathrm{GPM}} = \mathbf{v}_y^\top \mathbf{W} \mathbf{v}_{y'}$ to reduce complexity to $O(2dK)$ while modeling cycles.

**Limitations of Prior Work**: GPM entangles transitivity and cyclicity within a single skew-symmetric form. This paper proves (Theorem 4.7) that GPM at $d=1$ cannot express a hybrid structure of "dominant candidate + internal cycle." Even for $d > 1$, there is no guarantee that an embedding exists to accommodate both dominant candidates and complex cycles. In essence, modeling local cycles "squeezes out" the geometric capacity for global dominance—a structural defect.

**Key Challenge**: Real-world preferences exhibit a dual structure: clear global hierarchies (e.g., "helpful + harmless" as universal priorities) and local cycles (e.g., three similarly helpful responses with different styles and no strict winner). Modeling both with one skew-symmetric matrix forces the model to learn hierarchy and rotation simultaneously, which is geometrically incompatible.

**Goal**: To develop a preference model that guarantees the representability of dominant candidates without sacrificing cyclic modeling capabilities or $O(K)$ inference complexity.

**Key Insight**: The decomposition theorem for Symmetric Zero-Sum Functional-Form Games (Balduzzi et al. 2019) states that any zero-sum game can be uniquely decomposed into the sum of a "transitive component" and a "cyclic component." By treating preference modeling as a zero-sum FFG, this theorem provides theoretical justification for a hybrid form.

**Core Idea**: HRC = BT (transitive) + GPM (cyclic) in an explicit additive form: $s_{\mathrm{HRC}} = (r(\mathbf{y}_i) - r(\mathbf{y}_j)) + (\mathbf{v}_i^\top \mathbf{W} \mathbf{v}_j)$. This is paired with DSPPO—viewing alignment as a time-varying game where $\mathbb{P}_t$ transitions from "transitive-dominant" to "transitive + cyclic equality." This curriculum-style approach establishes a global quality baseline before learning local nuances to converge to a Nash equilibrium.

## Method

### Overall Architecture

The training consists of two stages: (1) Training the HRC preference model on Skywork-Reward-Preference-80K-v0.2, where three projection heads share an LLM backbone (Gemma-2B-it or Llama-3.1-8B-Instruct); (2) Running DSPPO time-varying self-play alignment on UltraFeedback prompts. Each step uses current HRC preference signals for SPPO-style multiplicative weight updates, with the internal weights of $s_T$ and $s_C$ scheduled by $1 \pm \lambda/\sqrt{t}$.

HRC Model Structure: Sharing LLM hidden states $\mathbf{h}_{\mathbf{y}|\mathbf{x}}$, the model uses three heads—transitive head $r_\phi(\mathbf{y}|\mathbf{x}) = \mathrm{clip}(\mathbf{w}_r^\top \mathbf{h}, -\delta, \delta)$, cyclic head $\mathbf{v}(\mathbf{y}|\mathbf{x}) = \mathbf{W}_c \mathbf{h} / \|\mathbf{W}_c \mathbf{h}\|_2$ (unit norm for zero-mean condition), and context gating $\mathbf{D}(\mathbf{x}) = \mathrm{diag}(\lambda(\mathbf{x})) \otimes \mathbf{I}_2$. The final score is $s_{\mathrm{HRC}} = C_1(r(\mathbf{y}_w) - r(\mathbf{y}_l)) + C_2(\mathbf{v}_w^\top \mathbf{D}(\mathbf{x}) \mathbf{R}^{\succ} \mathbf{D}(\mathbf{x}) \mathbf{v}_l)$, trained end-to-end via BCE loss.

### Key Designs

1. **HRC Preference Decomposition: Explicit Addition of BT + GPM**:
    - **Function**: Separates human preferences into "global hierarchy" and "local cycles" for parallel learning, avoiding the capacity squeeze of GPM.
    - **Mechanism**: Based on Balduzzi et al. 2019, any zero-sum FFG uniquely decomposes into a transitive component $\phi_T(\mathbf{v}, \mathbf{w}) = f(\mathbf{v}) - f(\mathbf{w})$ and a cyclic component $\phi_C(\mathbf{v}, \mathbf{w})$, where $\phi_C$ satisfies the zero-integral condition $\int \phi_C(\mathbf{v}, \mathbf{w}) d\mathbf{w} = 0$. Theorem 4.6 proves BT corresponds to $\phi_T$ and GPM corresponds to $\phi_C$ (under zero-mean embedding conditions). Thus, $s_{\mathrm{HRC}} = (r(\mathbf{y}_i) - r(\mathbf{y}_j)) + \mathbf{v}_i^\top \mathbf{W} \mathbf{v}_j$ is the "standard instantiation." Theorem 4.7 further proves GPM alone cannot guarantee dominant candidate representation—HRC bypasses this by routing dominant signals to an independent BT head.
    - **Design Motivation**: Theoretically, HRC is a constrained GPM with dim=$2d+1$. By routing signals to independent heads, it structuraly avoids Theorem 4.7 limitations while maintaining $O((2d+1)K)$ inference complexity.

2. **Context-Aware Gating + Reward Clipping + Unit-Norm**:
    - **Function**: Ensures geometric conditions for the cyclic component, controls the range of the reward component, and dynamically adjusts cyclic intensity based on context.
    - **Mechanism**: Clipping restricts $r_\phi$ to $[-\delta, \delta]$ for numerical stability; unit norm ensures GPM embeddings meet zero-mean conditions via isotropy on the sphere; context gating $\lambda(\mathbf{x}) \ge 0$ allows dynamic cyclic adjustment (e.g., activated for RPS prompts, deactivated for safety prompts).
    - **Design Motivation**: Original GPM lacks a context dimension, sharing one skew-symmetric matrix across all prompts. HRC uses gating to learn prompt heterogeneity—only providing cyclic signals when the prompt requires it.

3. **DSPPO Time-Varying Game: Curriculum Alignment from Transitivity to Cyclicity**:
    - **Function**: Replaces a fixed oracle with a time-varying one, allowing the policy to stabilize on the transitive backbone before learning cyclic details.
    - **Mechanism**: Updates the fixed $\mathbb{P}$ in SPPO to a time-varying $\mathbb{P}_t = \sigma(s_t)$, where $s_t = (1 + \lambda/\sqrt{t}) s_T + (1 - \lambda/\sqrt{t}) s_C$. Early stages emphasize transitivity; later stages restore the full HRC signal. Theorem 5.3 proves the duality gap of the mixture policy $\bar{\pi}_T$ relative to the Nash equilibrium is $O(1/\sqrt{T})$.
    - **Design Motivation**: Direct alignment with full signals causes oscillation. Learning the "global direction" first via transitive signals followed by local cyclic details acts as curriculum learning.

### Loss & Training

The HRC model uses BCE loss $\mathcal{L}(\theta) = -\mathbb{E}[\log \sigma(\text{HRC score})]$. DSPPO utilizes the SPPO MSE loss framework with $\mathbb{P}_t$ calculated via $s_t$, set at $\eta = \Theta(1/\sqrt{T})$.

## Key Experimental Results

### Main Results: RewardBench 2

| Base + Method | Factuality | Precise IF | Math | Safety | Focus | Ties | Avg |
|------|---|---|---|---|---|---|---|
| Gemma-2B-it + BT (d=1) | 45.68 | 32.50 | 62.30 | 80.67 | 77.17 | 37.25 | 55.93 |
| Gemma-2B-it + GPM (d=4) | 43.16 | **36.25** | **64.48** | 81.11 | 76.16 | 37.25 | 56.40 |
| **Gemma-2B-it + HRC (2+1)** | **47.58** | 35.63 | 61.75 | 82.00 | **79.60** | **39.22** | **57.63 (+1.23)** |
| Llama-3.1-8B + BT (d=1) | 64.63 | 34.38 | **64.48** | 92.67 | 90.91 | 73.53 | 70.10 |
| **Llama-3.1-8B + HRC (2+1)** | **68.42** | **35.00** | 60.11 | **92.89** | **94.75** | **74.51** | **70.95 (+0.85)** |

HRC consistently leads in the Ties domain, validating its robustness in "non-strict preferences." It also improves in Safety and Focus where dominant signals are required, confirming Theorem 4.7.

### Ablation Study (Gemma-2B-it + HRC 2+1)

| Config | Factuality | Safety | Focus | Ties | Avg | $\Delta$ |
|------|------------|--------|-------|------|------|----------|
| HRC (Full) | 47.58 | 82.00 | 79.60 | 39.22 | 57.63 | — |
| w/o Context Gating | 45.89 | 83.11 | 74.95 | 38.24 | 56.49 | -1.14 |

Context Gating is the most crucial stabilization technique, aligning with the prompt heterogeneity hypothesis.

### Key Findings

- **Ties domain is HRC's primary advantage**: HRC excels in scenarios with multiple equivalent correct answers and incorrect distractors.
- **GPM performance can degrade in certain domains**: Pure cyclic modeling can be detrimental in dominant-led domains (e.g., Safety).
- **DSPPO Gain**: Improves win-rate by ~3% over standard SPPO, proving the value of time-varying oracle scheduling.

## Highlights & Insights

- **Theoretic Decomposition to Model Form**: Transforms Balduzzi's FFG theorem into a concrete preference model, elevating empirical observations to rigorous proof.
- **Geometric Analysis of Dominance**: Descriptions of GPM failures as "spherical capacity being squeezed" provide intuitive geometric insights.
- **DSPPO as a Time-Varying Game**: Opens a new design space by allowing the oracle to evolve during alignment.
- **Engineering Prompt Heterogeneity**: Explicitly modeling that "not all prompts need cycles" via $\lambda(\mathbf{x})$ is a practical solution to noise.

## Limitations & Future Work

- **Theoretical vs. Practical Assumptions**: The unit norm is an approximation of the required isotropic zero-mean condition; actual post-training distribution is unverified.
- **Real-world Cyclic Prevalence**: The actual proportion of cyclic preferences in human data remains unquantified.
- **DSPPO Schedule**: Only one $1/\sqrt{t}$ schedule was validated; other decay types remain unexplored.
- **Scaling**: Experiments were limited to models $<10$B parameters.

## Related Work & Insights

- **vs BT (1952)**: Overcomes transitivity assumptions; HRC functions where BT fails (e.g., RPS prompts).
- **vs GPM (Zhang et al. 2025c)**: HRC is a strict extension that fixes structural defects identified in Theorem 4.7.
- **vs PairRM/PairPM**: HRC trades some expressivity for $O(K)$ scalability.
- **Insight**: The approach of "explicitly decomposing adversarial structures" can be extended to other fields like visual similarity or recommendation systems.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Strict proof of GPM defects + hybrid fix.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive evaluations across models; lacks 70B scale.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic from theory to implementation.)
- Value: ⭐⭐⭐⭐ (Significant theoretical and practical contributions to RLHF.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Implicit Safety Alignment from Crowd Preferences](implicit_safety_alignment_from_crowd_preferences.md)
- [\[ICLR 2026\] Learning Ordinal Probabilistic Reward from Preferences (OPRM)](../../ICLR2026/llm_alignment/learning_ordinal_probabilistic_reward_from_preferences.md)
- [\[NeurIPS 2025\] Capturing Individual Human Preferences with Reward Features](../../NeurIPS2025/llm_alignment/capturing_individual_human_preferences_with_reward_features.md)
- [\[AAAI 2026\] When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF](../../AAAI2026/llm_alignment/when_human_preferences_flip_an_instance-dependent_robust_loss_for_rlhf.md)
- [\[ICML 2026\] UDM-GRPO: Stable and Efficient GRPO for Unified Discrete Diffusion Models](udm-grpo_stable_and_efficient_group_relative_policy_optimization_for_uniform_dis.md)

</div>

<!-- RELATED:END -->
