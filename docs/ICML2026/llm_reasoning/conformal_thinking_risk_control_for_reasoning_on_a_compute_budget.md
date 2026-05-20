---
title: >-
  [Paper Note] Conformal Thinking: Risk Control for Reasoning on a Compute Budget
description: >-
  [ICML 2026][LLM Reasoning][dual-threshold early stopping] This work reframes the problem of "when to stop reasoning in LLMs" from an opaque threshold-tuning task into a **user-specifiable risk tolerance** conformal risk…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "dual-threshold early stopping"
  - "conformal risk control"
  - "parameterized lower threshold"
  - "UCB calibration"
  - "Qwen3 / DeepSeek-R1"
date: 2026-05-08
content_hash: 149ef2888b8dfc3b
---

# Conformal Thinking: Risk Control for Reasoning on a Compute Budget

**Conference**: ICML 2026  
**arXiv**: [2602.03814](https://arxiv.org/abs/2602.03814)  
**Code**: https://github.com/xidulu/reasoning_risk_control/  
**Area**: LLM reasoning / adaptive early stopping; conformal prediction; test-time scaling  
**Keywords**: dual-threshold early stopping, conformal risk control, parameterized lower threshold, UCB calibration, Qwen3 / DeepSeek-R1

## TL;DR
This work reframes the problem of "when to stop reasoning in LLMs" from an opaque threshold-tuning task into a **user-specifiable risk tolerance** conformal risk control problem: using two thresholds—an upper threshold to stop when the model is confident (controlling false positives), and a newly proposed **parameterized lower threshold** to force stop when the model is "stuck" on unsolvable problems (controlling false negatives). The UCB algorithm is used to automatically determine thresholds from a calibration set that satisfy risk constraints, achieving "almost no drop in accuracy, significant token savings" on AIME / GPQA / MathVision.

## Background & Motivation

**Background**: Reasoning LLMs (e.g., DeepSeek-R1, o1) improve accuracy via test-time scaling: more thinking tokens yield higher accuracy. Adaptive early stopping (Wang et al., Yang et al.) monitors confidence/entropy and stops the `<think>` phase when a threshold is reached.

**Limitations of Prior Work**: (i) Thresholds are **uninterpretable**—values like 0.7 or 0.85 lack direct business meaning and depend on signal type (entropy/confidence/probe), model, and task, resulting in poor transferability; (ii) Existing "stop-when-confident" methods use **only an upper threshold**. On hard problems (e.g., AIME/human-level exams), the model often never reaches the confidence threshold, exhausting the budget and wasting tokens; (iii) Current methods monitor confidence only at the token level, failing to distinguish between "already correct" and "fundamentally unsolvable" cases where stopping is warranted.

**Key Challenge**: Users care about "how much error rate can I tolerate," but current systems require tuning an internal threshold with no direct mapping to error rate. Moreover, "stop-when-confident" alone cannot handle cases where confidence never rises, leading to wasted budget.

**Goal**: (i) Recast threshold tuning as a risk control problem, allowing users to directly specify $\epsilon$ (tolerable error rate); (ii) Introduce a lower threshold mechanism for early abstention on unsolvable instances; (iii) Use distribution-free conformal algorithms (UCB) to find threshold pairs from a calibration set that satisfy $\mathbb{P}(\mathcal{R}\le\epsilon)\ge 1-\delta$.

**Key Insight**: The early-exit neural network (EENN) community has long used conformal risk control to decide which exit to take (Jazbec et al. 2024); "when to stop thinking" in reasoning LLMs is essentially the same problem, with exits replaced by tokens. Directly porting the EENN conformal framework and adding a lower threshold for unsolvable cases solves the issue.

**Core Idea**: Use conformal UCB to link "two thresholds (upper + lower)" with "user risk tolerance $\epsilon$," automating threshold selection, guaranteeing error rates, and significantly saving tokens—especially valuable when hard problems are prevalent.

## Method

### Overall Architecture
Given a reasoning trajectory $y = \langle\text{think}\rangle r_{1:T}\langle/\text{think}\rangle a$ and per-step confidence signals $s_t = u(x, r_{1:t})$ (options: entropy / confidence / mutual predictability, etc.), apply smoothing to obtain $\tilde s_t = g(s_{1:t})$.

**Dual-threshold early stopping strategy** (Eq. 5):
$$\tau = \min\{t\ge 1 : \tilde s_t \ge \lambda_+ \;\lor\; \tilde s_t \le \lambda_-\}$$

Here, $\lambda_+>\lambda_-$; triggering the upper threshold means "confident answer," while the lower threshold means "give up on this problem."

**Threshold Selection**: From calibration set $\mathcal{V}=\{(x_i, y_i^*)\}$, use the UCB algorithm (Bates 2021) to find parameters satisfying $\mathbb{P}_\mathcal{V}(\mathcal{R}(\hat\lambda_+, \hat c)\le\epsilon)\ge 1-\delta$. Specifically: first fix risk target $\epsilon^+$ and use UCB to find $\hat\lambda_+$; then, with $\hat\lambda_+$ fixed, use UCB to find lower threshold parameter $\hat c$ such that $\mathbb{E}[\ell^-\mid\hat\lambda_+]\le\epsilon^-$.

If multiple candidate thresholds $\lambda$ satisfy the risk constraint, select the one with **minimal efficiency loss** (Eq. 16).

### Key Designs

1. **Parameterized Dynamic Lower Threshold (Core Innovation)**:

    - **Function**: When the model's confidence fails to increase sufficiently within a reasonable time, actively abstain to avoid wasting tokens on futile reasoning.
    - **Mechanism**: Static lower thresholds only trigger on "confidence drop," which is rarely useful. Here, the lower threshold is set as a sigmoid function of token usage $\omega_t$: $\lambda_-(t;c,s,l,u) = \sigma(c(\omega_t - sB), l, u) = \frac{u-l}{1+e^{-c(\omega_t - sB)}}+l$, where $B$ is the budget. $c$ controls slope (required confidence growth rate), $s$ controls horizontal shift (when to start applying pressure), $l,u$ set bounds. Adjusting $(c,s)$ recovers linear, exponential, logarithmic, or constant schedules—essentially, "the model must demonstrate confidence improvement on a schedule, or be ejected."
    - **Design Motivation**: Existing stop-when-confident fails on hard problems like AIME—most questions never reach the upper threshold, exhausting the budget. The lower threshold is a **dual** mechanism: upper threshold controls errors (false positives), lower threshold controls waste (false negatives + over-computation). Only their combination covers both early stopping scenarios.

2. **Complementary Four Loss Functions (Separating Correctness and Efficiency)**:

    - **Function**: Decompose the merits of "when to stop" into four independently computable components for UCB calibration.
    - **Mechanism**: Use four $[0,1]$-range losses:
        - $\ell^+(y^*, f_t, \tilde s_t;\lambda_+) = \mathbb{I}[\tilde s_t\ge\lambda_+]\cdot\mathbb{I}[f_t\ne y^*]$ (upper threshold FP loss—confident stop but wrong answer);
        - $\ell^-(y^*, f_{t:T}, \tilde s_t;\lambda_-) = \frac{\mathbb{I}[\tilde s_t\le\lambda_-]}{T-t+1}\sum_{k=t}^T \mathbb{I}[f_k=y^*]$ (lower threshold FN loss—abstained but could have answered correctly later, **farsighted**: checks all future steps);
        - $\mathcal{J}^+(t) = \frac{1}{T}\max(0, t-t')$, $t'=\min\{t:f_t=y^*\}$ (upper threshold efficiency: tokens wasted after correct answer);
        - $\mathcal{J}^-(t) = \frac{1}{T}\sum_{k\le t}\mathbb{I}[f_k\ne y^*]$ (lower threshold efficiency: tokens spent on wrong answers before stopping).
    - **Design Motivation**: Correctness loss determines threshold eligibility (must be $\le\epsilon$), efficiency loss selects the most economical among eligible thresholds—decoupling "accuracy" from "efficiency," letting conformal calibration focus on correctness, with efficiency only for tie-breaking. The farsighted design of the lower threshold is especially clever—it avoids penalizing "temporarily wrong but eventually correct" cases.

3. **Conformal UCB Calibration + Signal Ensemble**:

    - **Function**: Given a finite calibration set, find thresholds that satisfy $\mathcal{R}\le\epsilon$ with probability $1-\delta$ in the finite-sample sense.
    - **Mechanism**: The naive approach uses empirical risk $\widehat{\text{Risk}}(\lambda;\mathcal{V}) = \frac{1}{N}\sum\ell(\cdot;\lambda)$ to select $\lambda$, but with small calibration sets, this is overly optimistic and may violate the true $\epsilon$. UCB (Bates 2021) replaces this with $\widetilde{\text{Risk}}(\lambda;\mathcal{V}) = \widehat{\text{Risk}} + (\text{finite-sample correction})$, where the correction depends on $|\mathcal{V}|$ and $\delta$, ensuring $\mathbb{P}(\text{true risk}\le\epsilon)\ge 1-\delta$. Algorithm 1 grid searches over multiple candidate signals $\mathcal{S}$ (entropy / confidence / probe, etc.) and threshold grids $\Lambda_s$, selecting the one in the feasible set $\mathfrak{C}$ with minimal efficiency loss—**automatically selecting both signal and threshold**.
    - **Design Motivation**: Naive cross-validation (Fig. 4 left) frequently violates the target risk, while UCB (right) remains stably $\le\epsilon$. Allowing multiple signal ensembles makes "which signal is best" data-driven—since the optimal signal varies with $\epsilon$, letting the algorithm select on the calibration set is key to removing manual tuning.

### Loss & Training
**Completely training-free**—all threshold and signal selection is performed via UCB on the calibration set. Only inference is required for the models (Qwen3-8B, Qwen3-30B-A3B, DeepSeek-R1-Distill-Qwen-32B, Qwen3-VL-8B). Algorithm 1 performs grid search + UCB correction in one pass to obtain the final $(s^*, \lambda^*)$ for deployment.

## Key Experimental Results

### Main Results

| Experimental Setting | Signal | Key Phenomenon |
|---------------------|--------|---------------|
| Risk control validation (Fig. 4) | Multi-signal | Naive calibration frequently violates $\epsilon$; UCB calibration always $\le\epsilon$ (with prob $1-\delta$) |
| Signal ensemble (Fig. 5) | 4 models × multi-signal | Risk control automatically selects the most efficient signal, outperforming any single signal |
| Value of lower threshold (Fig. 6) | Qwen3-8B + confidence | When solvable:unsolvable = 1:1 / 1:3, Lower+Upper saves significantly more tokens than Upper-only at the same accuracy |

### Ablation Study

| Configuration | Key Effect | Notes |
|---------------|------------|-------|
| Upper-only | Runs full budget when hard problems are prevalent | Many unsolvable problems never reach confidence threshold |
| Lower-only | Only triggers on confidence drop | Static lower threshold is almost useless |
| **Lower+Upper (dynamic)** | Accuracy nearly unchanged, tokens significantly reduced | Complementary effect of dual thresholds |
| Naive calibration | Frequently violates $\epsilon$ | Empirical risk is overly optimistic |
| UCB calibration | Always $\le\epsilon$ | Finite-sample correction is effective |
| Ensemble of signals | Outperforms any single signal | Risk control auto-selects the best |

### Key Findings
- **Value of lower threshold increases with hard problem ratio**: When solvable:unsolvable = 3:1, gains are small (most problems converge normally); at 1:1 and 1:3, the lower threshold's value is significant—identifying and abstaining from unsolvable problems is key to token savings.
- **Trigger distribution matches intuition**: In representative Lower+Upper settings, solvable problems mostly exit via the upper threshold, unsolvable ones via the lower—each threshold serves its role.
- **No single signal is optimal**: The optimal signal varies with $\epsilon$ (Fig. 1 right), so ensemble + auto-selection is necessary.
- **UCB correction is crucial**: Finite-sample correction appears "conservative," but in practice "just avoids violation"—naive calibration frequently violates, UCB keeps violation rate $\le\delta$.
- **Farsighted loss for lower threshold avoids misjudgment**: Using "current correctness" would misclassify "temporarily wrong but eventually correct" cases as abstainable; farsighted checks all answers in $[t, T]$, ensuring the lower threshold's statistical meaning truly corresponds to "will never be correct in the future."

## Highlights & Insights
- **Paradigm shift from threshold tuning to risk specification**: Previously, engineers tuned an opaque 0.7; now, users directly specify "I can accept 5% error rate," and the system infers thresholds. This exemplifies "interface interpretability" in ML deployment.
- **Lower threshold is a symmetric mechanism overlooked in literature**: Everyone focuses on stop-when-confident; no one on stop-when-stuck. This work shows the latter is more beneficial for hard problems—a research approach of "completing the symmetric design" worth learning.
- **Dynamic parameterization is elegantly designed**: The sigmoid with (slope, shift, lower, upper) parameters covers linear/exponential/logarithmic/constant schedules, giving the conformal optimizer enough expressiveness without explosion—a "compact yet powerful" design.
- **Directly pluggable into existing systems**: No training, no model weight changes, no decoding changes—just add threshold monitoring + calibration at inference; industrial deployment-friendly.
- **Bridging theory and practice**: Systematically transfers the EENN conformal framework (Jazbec 2024) to reasoning LLMs, adding unsolvable handling—a precise match between established theory and emerging problems.

## Limitations & Future Work
- **Theoretical risk of upper bound violation for lower threshold**: The authors acknowledge that in extreme cases where lower threshold calibration is completely broken (most solvable instances are wrongly abstained), the upper threshold risk bound may be violated—though such "extreme distribution shift" is rare in practice, caution is needed for OOD deployment.
- **Dependence on labeled calibration set**: UCB requires $\mathcal{V}=\{(x_i, y_i^*)\}$ ground truth, making it inapplicable to open domains without labels (e.g., creative writing, medical consultation).
- **Mismatch between static calibration and dynamic thresholds**: Calibration is based on one-off grid search, but deployment distributions may drift (new model versions, new question types), requiring recalibration. Online conformalization is not discussed.
- **Efficiency/accuracy trade-off is not Pareto optimal**: Fig. 1 right shows different methods are Pareto optimal at different $\epsilon$; while this work selects suitable methods for given $\epsilon$, it does not eliminate the trade-off itself.
- **No comparison with training-side reasoning length compression (e.g., Reward Shaping)**: This is an inference-only approach, but recent training-side methods explicitly penalize long thinking; such comparisons are missing.

## Related Work & Insights
- **vs Wang 2025 / Yang 2025 (upper threshold stop-when-confident)**: They use only entropy/confidence thresholds, single threshold, and manual tuning; this work uses conformal-calibrated dual thresholds.
- **vs Thought Calibration (Wu 2025)**: Also uses risk control, but stops when stable (using linear probe to predict answer stability); this work uses general signals + UCB and explicitly handles unsolvable cases.
- **vs PAC Reasoning (Zeng 2026)**: They conformally decide "whether to reason" (binary); this work decides "how long to reason" (continuous).
- **vs EENN conformal risk control (Jazbec 2024)**: This work directly inherits the UCB framework, replacing "which exit" with "which token," and adds the lower threshold mechanism.
- **Insights**: The conformal calibration + dual-threshold ("overconfident vs. never confident") paradigm can be extended to any cost-aware inference scenario—e.g., when to stop multi-agent discussion, when to stop active learning, when to prune search.

## Rating
- Novelty: ⭐⭐⭐⭐ Dual-threshold + parameterized dynamic lower threshold is genuinely novel; the conformal framework is borrowed from EENN literature, but the transfer to reasoning is cleanly executed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 models, 4 datasets (including visual reasoning), validates risk control, ensemble, and lower threshold value; but lacks end-to-end inference cost numbers (e.g., average token savings at different $\epsilon$), so efficiency gains could be more readable.
- Writing Quality: ⭐⭐⭐⭐ Clearly bridges conformal and reasoning early-exit domains, and explains the physical meaning of the four losses; a few symbol definitions are dense.
- Value: ⭐⭐⭐⭐ Directly deployable, training-free, user-interpretable interface; highly practical for cost-sensitive industrial applications (search, QA, code assistants).

## Related Papers

- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](../../NeurIPS2025/llm_reasoning/towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [\[ICML 2026\] T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning](t2po_uncertainty-guided_exploration_control_for_stable_multi-turn_agentic_reinfo.md)
- [\[ICML 2026\] Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models](less_diverse_less_safe_the_indirect_but_pervasive_risk_of_test-time_scaling_in_l.md)
- [\[ACL 2026\] Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data](../../ACL2026/llm_reasoning/budget-aware_anytime_reasoning_with_llm-synthesized_preference_data.md)
- [\[ICLR 2026\] ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](../../ICLR2026/llm_reasoning/atts_asynchronous_test-time_scaling_via_conformal_prediction.md)

</div>

<!-- RELATED:END -->
