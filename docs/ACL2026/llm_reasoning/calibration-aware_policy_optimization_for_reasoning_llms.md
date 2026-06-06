---
title: >-
  [Paper Note] Calibration-Aware Policy Optimization for Reasoning LLMs
description: >-
  [ACL 2026][LLM Reasoning][GRPO] The authors initially prove that the "reward-only" advantage estimation in GRPO-like algorithms is equivalent to an AUC-inconsistent surrogate ($\phi(t)=-t$, breaking scale-invariance)…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "GRPO"
  - "Calibration"
  - "AUC consistency"
  - "advantage estimation"
  - "inference-time scaling"
date: 2026-05-08
content_hash: b401a5bffaea29f2
---

# Calibration-Aware Policy Optimization for Reasoning LLMs

**Conference**: ACL 2026  
**arXiv**: [2604.12632](https://arxiv.org/abs/2604.12632)  
**Code**: TBD  
**Area**: LLM Reasoning / RL / Calibration  
**Keywords**: GRPO, Calibration, AUC consistency, advantage estimation, inference-time scaling

## TL;DR
The authors initially prove that the "reward-only" advantage estimation in GRPO-like algorithms is equivalent to an AUC-inconsistent surrogate ($\phi(t)=-t$, breaking scale-invariance), which causes accuracy to increase while relative calibration (perplexity AUC) continuously degrades. Based on this, they propose CAPO: replacing the advantage with a "pairwise, uncertainty-aware" form based on a logistic AUC consistent surrogate, supplemented by reference-model PPL for denoising masking. On Qwen2.5-Math 1.5B/7B, CAPO achieves +15~25% calibration with accuracy matching or exceeding GRPO, further yielding a 5% gain in AIME inference-time scaling.

## Background & Motivation

**Background**: RLVR (Reinforcement Learning from Verifiable Rewards) using algorithms like GRPO/GSPO has pushed the accuracy of mathematical reasoning models to high levels. However, several studies (Liu 2025, Kalai 2025, Bereket 2025) note that trained models become "overconfident"—the perplexity of incorrect answers is lower than that of correct ones, leading to degraded relative calibration (AUC).

**Limitations of Prior Work**: Calibration has significant practical value: (1) confidence decides whether to dispatch backup models in multi-agent collaboration; (2) inference-time scaling relies on confidence for candidate selection; (3) abstention suppresses hallucinations. If the PPL of the trained model no longer reflects correctness, all downstream tasks are affected. Existing remedies like CoDaPO, CDE (reward/advantage shaping), and SimKO (label smoothing) are mostly heuristic without theoretical guarantees, resulting in limited calibration improvement or sacrificed accuracy.

**Key Challenge**: The GRPO objective only considers rewards and ignores sample uncertainty/PPL. This "reward-only" signal is mathematically misaligned with "calibration"—the optimizer can lower the PPL of all samples (including incorrect ones) to increase rewards, causing accuracy to rise while AUC falls.

**Goal**: (1) Provide a rigorous mathematical explanation for why "GRPO degrades calibration." (2) Design a theoretically guaranteed (AUC consistent) advantage estimation to jointly optimize calibration and accuracy. (3) Stabilize training—since the new advantage is non-linear (logistic) and sensitive to noisy samples.

**Key Insight**: The authors start from AUC optimization theory (Gao & Zhou 2012). By rewriting the REINFORCE gradient of GRPO as pairwise differences (U-statistic), they find the implicit surrogate is $\phi(t)=-t$. Through a scale-invariance counterexample ($\mathrm{AUC}(\alpha f) = \mathrm{AUC}(f)$ but $\mathcal{L}_{-t}(\alpha f) = \alpha \mathcal{L}_{-t}(f)$), they prove it is not AUC consistent. The natural alternative is the logistic surrogate $\phi_\tau(t)=\log(1+\exp(-t/\tau))$.

**Core Idea**: Replace the "reward-only" advantage $A_i = R_i - \bar R$ with an "uncertainty-aware" pairwise advantage $\tilde A_i = \sum_j \phi'(lpm(o_i) - lpm(o_j))$, determined by the derivative of the logistic surrogate (sigmoid), and use the reference-model PPL to mask extreme noise for each sample.

## Method

### Overall Architecture
CAPO is a "local surgery" on the GRPO framework—retaining the PPO-clipped objective and KL constraints while only replacing the advantage $\hat A_i$ with $\hat A_i^{CAPO} = m(o_i)\,\tilde A_i$:
- $\tilde A_i$ originates from the gradient of the logistic AUC surrogate and depends on the PPL of **all other samples within the group**, magnifying the weights of misranked samples (e.g., "correct but high PPL" and "incorrect but low PPL").
- $m(o_i)$ is an indicator mask based on reference-model (base model) PPL: samples that are correct but have $ref-PPL > ref-high$ are discarded as "lucky guesses"; samples that are incorrect but have $ref-PPL < ref-low$ are discarded as "near-misses."
- The final objective is $J_{CAPO}(\theta) = \mathbb{E}[\sum_i \min(r_i \hat A_i^{CAPO}, \mathrm{clip}(r_i,1\pm\epsilon)\hat A_i^{CAPO})]$.

### Key Designs

1.  **logistic AUC consistent surrogate (replacing reward-only advantage)**:
    - **Function**: Allows the gradient to optimize both accuracy (ranking correct vs. incorrect) and relative calibration (ranking PPL correctly), rather than simply minimizing reward error.
    - **Mechanism**: Starting from the GRPO gradient $\nabla J_{GRPO} = \mathbb{E}[\sum_i (R_i - \bar R)\nabla_\theta lpm(o_i)]$, it is rewritten into a pairwise form $\nabla \mathbb{E}[(lpm(o_1)-lpm(o_2))(R_1-R_2)]$ using U-statistic invariance, revealing the hidden surrogate $\phi(t)=-t$. The authors prove this surrogate is scale-sensitive ($\mathcal{L}_{-t}(\alpha f)=\alpha\mathcal{L}_{-t}(f)$), allowing the optimizer to reduce loss infinitely without improving AUC → AUC inconsistency. Replacing it with the logistic version $\phi_\tau(t)=\log(1+e^{-t/\tau})$ (satisfying Theorem 1: convex + non-increasing + $\phi'(0)<0$) ensures that optimizing the surrogate optimizes the AUC via the regret bound in Theorem 2 $L(f)-L^* \le \tfrac{1}{\ln 2}(L_\phi(f)-L_\phi^*)$. The corresponding advantage form involves $\tilde A_i = -\sum_{j:R_j=0}\phi'(lpm(o_i)-lpm(o_j))$ for correct samples, and is symmetric for incorrect ones. The shape of $\phi'(t)=-\sigma(-t)$ indicates that when the PPL gap between correct/incorrect is already large, $|\phi'|$ approaches 0, suppressing the gradient; large gradients are only given to "nearly misranked" samples.
    - **Design Motivation**: Proving that GRPO is mathematically AUC-inconsistent is the major contribution—shifting the explanation of "why GRPO degrades calibration" from empirical observation to theoretical necessity. The logistic surrogate is selected based on existing regret bounds and global convergence guarantees.

2.  **reference-model PPL denoising masking**:
    - **Function**: Filters out training noise caused by binary reward misjudgments (correct by chance, incorrect by a single step) to prevent the logistic surrogate from collapsing due to extreme samples.
    - **Mechanism**: Since the base model is well-calibrated before training (Kalai 2025), reference-model PPL serves as a reliable indicator of sample quality: $m(o) = \mathbb{I}[PPL_{ref}(o) \le \text{ref-high}]$ (if $R=1$) or $\mathbb{I}[PPL_{ref}(o) \ge \text{ref-low}]$ (if $R=0$). The thresholds $\text{ref-high} / \text{ref-low}$ are empirically taken from the upper/lower quartiles of the reference model's PPL distribution (2.5 / 1.05) and require no additional learning.
    - **Design Motivation**: CAPO's pairwise advantage is particularly sensitive to extreme samples (large sigmoid gradient). If a "correct but actually guessed" sample is treated as a strong positive signal, it pulls all reasonable reasoning toward a chaotic token distribution. The mask is a cheap but necessary stabilizer: Fig 9 shows that without the mask, entropy continuously rises, and accuracy plateaus or declines early.

3.  **gradient analysis revealing "which samples are corrected first"**:
    - **Function**: Explains why CAPO improves both AUC and accuracy from the perspective of gradient shape.
    - **Mechanism**: Because $\phi'(t)=-\sigma(-t)$ reaches its maximum value in the $t<0$ range, CAPO's advantage provides the largest gradients for "correct but high PPL" or "incorrect but low PPL" samples (PPL-misranked). these are precisely the most informative boundary samples in the decision space. Conversely, GRPO assigns the same $+|\bar R|$ to all correct samples and $-|\bar R|$ to all incorrect ones, failing to distinguish uncertainty.
    - **Design Motivation**: The authors use this to demonstrate that CAPO is not simply "adding a calibration regularizer" but changing the sample-level weight distribution—essentially upgrading training from "flat rewards" to "hard sample mining" with "automatic calibration."

### Loss & Training
- **Model**: Qwen2.5-Math-1.5B / 7B; training data consists of 20k DeepScaler problems, with 240 problems for validation.
- **Framework**: verl + 8× A100; 1.5B takes ~24h for 600 steps, 7B takes ~48h for 400 steps.
- **Hyperparameters**: lr 1e-6, batch size 128, PPO mini-batch 64, rollout n=8 (val 16), $\epsilon=0.2$, KL/entropy coef = 0; temperature 1.0.
- **CAPO-only**: $\tau=0.6$ (1.5B) / 0.5 (7B), ref-high=2.5, ref-low=1.05.
- **Evaluation**: 6 benchmarks (AIME24/25, MATH500, AMC23, Minerva, OlympiadBench); metrics include mean@16, AUC-mean, Precision-Coverage, and inference-time scaling (Perplexity Consistency, N=16).

## Key Experimental Results

### Main Results
Comparison of calibration (AUC-mean) and accuracy (mean@16) on 6 benchmarks (Qwen2.5-Math-7B average, extracted from Fig 1/3):

| Method | AIME25 AUC | AIME25 Gain | Inference Scaling Acc (1.5B) | (7B) |
|------|------|------|------|------|
| GRPO | 0.54 | – | 20.33% | 33.33% |
| GSPO | – | – | 20.00% | 32.21% |
| CoDaPO | – | – | 21.67% | 31.66% |
| CDE | – | – | 16.67% | 31.66% |
| SimKO | – | – | 11.67% | 23.33% |
| **CAPO (Ours)** | **0.79** | **+25%** | **25.33%** | **38.33%** |

On 1.5B, AIME25 AUC rose from 0.63 (GRPO) to 0.78 (+15%); on 7B, it rose from 0.54 to 0.79 (+25%). Accuracy (mean@8) matched or surpassed GRPO, achieving the highest scores on AIME24/25/Minerva.

### Ablation Study

| Configuration | Observation |
|------|------|
| Full CAPO | Steady calibration improvement + steady accuracy rise + stable entropy |
| w/o noise mask | Entropy continuously rises, accuracy plateaus or declines early |
| GRPO + only mask | No improvement in AUC (confirms surrogate is key, not the mask) |
| $\tau \in \{0.4, 0.6, 1.0\}$ | Accuracy / AUC change < 1 point (robust) |
| Tightened ref-high/low [1.25, 2.1] vs [1.05, 2.5] | Performance nearly identical |

### Key Findings
- **GRPO calibration degradation is a mathematical necessity**: Theorem 3 proves the reward-only advantage surrogate $\phi(t)=-t$ is not AUC consistent, so no amount of hyperparameter tuning can save it; this also covers GSPO (Fig 1c shows similar degradation).
- **CAPO achieves "training dynamic stability"**: Fig 1b/c shows GRPO/GSPO AUC decreases monotonically during training, while CAPO's increases monotonically. This suggests that with the correct surrogate, calibration and accuracy are no longer a trade-off.
- **Suppressing intermediate noise is vital for stability**: Removing the mask causes entropy to rise → policy becomes random → accuracy drops. Noise from binary verifier rewards has an obvious amplification effect under pairwise advantages; the mask is a necessity, not just an enhancement.
- **Inference-time scaling gains are amplified**: On AIME, CAPO outperforms GRPO by 5% (absolute) because inference-time algorithms (Perplexity Consistency) strictly rely on PPL ranking, which is amplified when calibration is better. This is the most direct downstream reward of improved calibration.
- **High costs of other calibration methods**: SimKO drops 12 points on AMC and 7.7 points on AIME24, proving that hard methods like label smoothing sacrifice accuracy. CAPO is one of the few with "zero acc cost."

## Highlights & Insights
- Using AUC consistency theory to bridge the "overconfidence" phenomenon in RLHF with a rigorous mathematical proposition is the cleanest theoretical contribution in this line of work—any future reward-only advantage method can be tested against this counterexample.
- The implicit "hard sample mining" property of $\phi'(t)=-\sigma(-t)$ is elegant—it requires no extra difficulty estimator; the sigmoid shape automatically concentrates gradients on misranked samples.
- Using "reference model PPL as a sample quality proxy" is a cheap yet accurate design—complementary to critic-free RL frameworks, adding no extra trainable parameters.
- Improved calibration translates directly into inference-time scaling gains (+5%); this causal chain links the abstract metric of "calibration" to benchmark figures, providing a persuasive argument for deployment.

## Limitations & Future Work
- Evaluation is limited to mathematical reasoning; it is unknown if GRPO's degradation pattern is the same for logic, common sense, or open-domain QA. Theoretical analysis applies, but it is uncertain if reference model PPL remains a good quality proxy.
- Mask thresholds are taken from fixed quantiles of the base model, assuming the base model is well-calibrated. If the base model's calibration was "damaged" by certain pre-training tricks (e.g., long-term SFT), the mask might fail.
- Pairwise advantages require both positive and negative samples within a group—if a group is all correct or all incorrect, it becomes ineffective. In practice, prompt difficulty needs pre-calibration to ensure a mix.
- Although $\tau$ is robust, the optimal value varies across base models (0.6 for 1.5B, 0.5 for 7B), suggesting a need for minor tuning with model scale.
- Future improvements: (1) Extend to stochastic rewards (multi-choice, open-ended judge) by replacing binary R with logits; (2) Use EMA reference instead of a fixed base to avoid PPL distribution drift late in training; (3) Jointly train CAPO with abstention/refusal for end-to-end hallucination mitigation.

## Related Work & Insights
- **vs. CoDaPO / CDE (reward shaping)**: Both attempt to save calibration by modifying reward shapes but lack theoretical guarantees. This paper proves that as long as reward-only advantage is retained, it remains systematically misaligned with AUC, making reward shaping a temporary fix.
- **vs. SimKO (label smoothing)**: SimKO uses label smoothing to suppress overconfidence. Calibration is mediocre, and the accuracy cost is heavy. CAPO modifies the surrogate and adds a mask, achieving zero accuracy cost.
- **vs. J1 / Think-RM (reasoning RM)**: These also use GRPO for reasoning but focus on preference accuracy rather than calibration. CAPO provides a general solution for obtaining calibration within GRPO, which can be directly integrated into these works.
- **Insights**: (1) Any RL algorithm with group-level comparison implies a pairwise surrogate; scrutinizing designs from the perspective of surrogate consistency can avoid "seemingly reasonable but mathematically non-convergent" traps. (2) "Reference model self-calibration" as a quality proxy is a versatile trick applicable to SFT data filtering, RM annotation quality assessment, etc.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Directly linking "overconfidence" in RLHF to AUC consistency theory and providing a rigorous counterexample is a rare "theoretical explanation of empirical phenomena" in this field.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 benchmarks + 5 baselines + two scales (1.5B/7B) + four metrics (calibration/accuracy/scaling/precision-coverage) + hyperparameter robustness + mask ablation; however, only tested on the math domain.
- Writing Quality: ⭐⭐⭐⭐ The "empirical observation → theoretical explanation" narrative in Section 3 is clean. The proof strategy for Theorem 3 (U-statistic + scale invariance counterexample) is short and powerful.
- Value: ⭐⭐⭐⭐ Provides a drop-in replacement for advantage estimation for any team using GRPO to train reasoning models; the theoretical framework can be reused by any subsequent RLVR work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Think Outside the Policy: In-Context Steered Policy Optimization](think_outside_the_policy_in-context_steered_policy_optimization.md)
- [\[ACL 2026\] Adapt to Thrive! Adaptive Power-Mean Policy Optimization for Improved LLM Reasoning](adapt_to_thrive_adaptive_power-mean_policy_optimization_for_improved_llm_reasoni.md)
- [\[ICML 2026\] UCPO: Uncertainty-aware Policy Optimization](../../ICML2026/llm_reasoning/ucpo_uncertainty-aware_policy_optimization.md)
- [\[ACL 2026\] ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs](chairo_contextual_hierarchical_analogical_induction_and_reasoning_optimization_f.md)
- [\[ICLR 2026\] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](../../ICLR2026/llm_reasoning/drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)

</div>

<!-- RELATED:END -->
