---
title: >-
  [Paper Note] Conformal Thinking: Risk Control for Reasoning on a Compute Budget
description: >-
  [ICML 2026][LLM Reasoning][Dual-threshold early exit] This paper reformulates the problem of "when a reasoning LLM should stop thinking" from an uninterpretable threshold tuning task into a **conformal risk control probl…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Dual-threshold early exit"
  - "conformal risk control"
  - "parameterized lower threshold"
  - "UCB calibration"
  - "Qwen3 / DeepSeek-R1"
date: 2026-05-08
content_hash: b27d946cc77fc022
---

# Conformal Thinking: Risk Control for Reasoning on a Compute Budget

**Conference**: ICML 2026  
**arXiv**: [2602.03814](https://arxiv.org/abs/2602.03814)  
**Code**: https://github.com/xidulu/reasoning_risk_control/  
**Area**: LLM Reasoning / Adaptive early exit; conformal prediction; test-time scaling  
**Keywords**: Dual-threshold early exit, conformal risk control, parameterized lower threshold, UCB calibration, Qwen3 / DeepSeek-R1

## TL;DR
This paper reformulates the problem of "when a reasoning LLM should stop thinking" from an uninterpretable threshold tuning task into a **conformal risk control problem with user-specified risk tolerance**. By employing two thresholds—an upper threshold to stop when the model is confident (controlling false positives) and a newly proposed **parameterized lower threshold** to force stopping when the model "stagnates" on unsolvable problems (controlling false negatives)—and utilizing the UCB algorithm to automatically derive thresholds from a calibration set that satisfy risk constraints, it achieves significant token savings with almost no drop in accuracy on AIME, GPQA, and MathVision.

## Background & Motivation

**Background**: Reasoning LLMs (e.g., DeepSeek-R1, o1) improve accuracy via test-time scaling: the more thinking tokens, the higher the accuracy. Adaptive early exit methods (e.g., Wang et al., Yang et al.) monitor confidence or entropy and stop the `<think>` phase when a threshold is met.

**Limitations of Prior Work**: (i) The threshold itself is **uninterpretable**—0.7 vs. 0.85 has no direct business meaning, and the value depends on the signal type (entropy/confidence/probe), model, and task, leading to poor transferability; (ii) existing "stop-when-confident" methods **only have an upper threshold**, so on hard problems (e.g., AIME), models often never reach the confidence threshold, exhausting the budget and wasting tokens; (iii) existing methods monitor confidence only at the token level without distinguishing between "already solved" and "inherently unsolvable" cases.

**Key Challenge**: Users truly care about business metrics like "what error rate can I tolerate," whereas current systems force users to tune internal thresholds without a direct mapping to error rates. Furthermore, simple "stop when confident" approaches fail to address hard problems where confidence never rises, leading to budget waste.

**Goal**: (i) Reformulate threshold tuning as a risk control problem where users specify $\epsilon$ (tolerable error rate); (ii) introduce a lower threshold mechanism for early abstention on unsolvable instances; (iii) use a distribution-free conformal algorithm (UCB) to find threshold combinations from a calibration set satisfying $\mathbb{P}(\mathcal{R}\le\epsilon)\ge 1-\delta$.

**Key Insight**: The Early Exit Neural Network (EENN) community has long used conformal risk control to decide which exit to stop at (Jazbec et al. 2024). Deciding when to stop "thinking" in a reasoning LLM is essentially the same problem, where the number of tokens serves as the number of exits. This can be solved by porting the EENN conformal framework and adding a lower threshold mechanism for unsolvable cases.

**Core Idea**: Use conformal UCB to bind dual thresholds (upper and lower) with the user's risk tolerance $\epsilon$, making threshold selection automatic, error rates guaranteed, and token savings significant—especially when the proportion of hard problems is high.

## Method

### Overall Architecture
Given a reasoning trajectory $y = \langle\text{think}\rangle r_{1:T}\langle/\text{think}\rangle a$ and the step-wise confidence signal $s_t = u(x, r_{1:t})$ (e.g., entropy, confidence, mutual predictability), a smoothed signal is obtained as $\tilde s_t = g(s_{1:t})$.

**Dual-Threshold Early Exit Strategy** (Eq. 5):
$$\tau = \min\{t\ge 1 : \tilde s_t \ge \lambda_+ \;\lor\; \tilde s_t \le \lambda_-\}$$

Where $\lambda_+>\lambda_-$; triggering the upper threshold indicates "confident answer," while the lower threshold indicates "abstaining from the problem."

**Threshold Selection**: From a calibration set $\mathcal{V}=\{(x_i, y_i^*)\}$, the UCB algorithm (Bates 2021) is used to find parameters satisfying $\mathbb{P}_\mathcal{V}(\mathcal{R}(\hat\lambda_+, \hat c)\le\epsilon)\ge 1-\delta$. This is done in two steps: first, fix the risk target $\epsilon^+$ to find $\hat\lambda_+$ via UCB, then fix $\hat\lambda_+$ and find the lower threshold parameter $\hat c$ via UCB to satisfy $\mathbb{E}[\ell^-\mid\hat\lambda_+]\le\epsilon^-$.

If multiple candidate thresholds $\lambda$ satisfy the risk constraint, the one with the **minimum efficiency loss** is selected (Eq. 16).

### Key Designs

1. **Parameterized Dynamic Lower Threshold (Core Innovation)**:
    - **Function**: Actively abstains when the model's confidence does not improve sufficiently within a reasonable time, avoiding unproductive token consumption.
    - **Mechanism**: Static lower thresholds are rarely triggered unless confidence regresses. Instead, this paper defines the lower threshold as a sigmoid function of token usage $\omega_t$: $\lambda_-(t;c,s,l,u) = \sigma(c(\omega_t - sB), l, u) = \frac{u-l}{1+e^{-c(\omega_t - sB)}}+l$, where $B$ is the budget. $c$ controls the slope (required confidence growth rate), $s$ controls horizontal shift (when to start applying pressure), and $l, u$ control the bounds. Tuning $(c, s)$ can recover linear, exponential, logarithmic, or constant schedules. Essentially, the model must demonstrate confidence growth according to a schedule or be terminated.
    - **Design Motivation**: Stop-when-confident fails on hard problems like AIME where most items never reach the upper threshold. The lower threshold is a **dual** mechanism: the upper threshold controls false positives, while the lower threshold controls waste (false negatives + over-computation). Together, they cover early exit scenarios at both ends.

2. **Four Complementary Loss Functions (Distinguishing Correctness vs. Efficiency)**:
    - **Function**: Decomposes the quality of "when to stop" into four independently calculable components for UCB calibration.
    - **Mechanism**: Uses four losses in $[0,1]$:
        - $\ell^+(y^*, f_t, \tilde s_t;\lambda_+) = \mathbb{I}[\tilde s_t\ge\lambda_+]\cdot\mathbb{I}[f_t\ne y^*]$ (Upper threshold FP loss: stopped confidently but incorrectly);
        - $\ell^-(y^*, f_{t:T}, \tilde s_t;\lambda_-) = \frac{\mathbb{I}[\tilde s_t\le\lambda_-]}{T-t+1}\sum_{k=t}^T \mathbb{I}[f_k=y^*]$ (Lower threshold FN loss: abstained but would have been right eventually. This is **farsighted**, checking all future time points);
        - $\mathcal{J}^+(t) = \frac{1}{T}\max(0, t-t')$, with $t'=\min\{t:f_t=y^*\}$ (Upper threshold efficiency: tokens wasted after answering correctly);
        - $\mathcal{J}^-(t) = \frac{1}{T}\sum_{k\le t}\mathbb{I}[f_k\ne y^*]$ (Lower threshold efficiency: tokens spent on incorrect outputs before stopping).
    - **Design Motivation**: Correctness loss determines if a threshold is valid ($\le\epsilon$), while efficiency loss selects the most economical option among valid thresholds. This decouples "accuracy" from "efficiency," letting conformal calibration handle correctness and efficiency handle tie-breaking. The farsighted design of the lower threshold is subtle—it prevents termination just because the current answer is wrong, checking instead if the model *ever* gets it right.

3. **Conformal UCB Calibration + Signal Ensemble**:
    - **Function**: Given a finite calibration set, finds thresholds satisfying $\mathcal{R}\le\epsilon$ with probability $1-\delta$ in a finite-sample sense.
    - **Mechanism**: A naive approach uses empirical risk $\widehat{\text{Risk}}(\lambda;\mathcal{V}) = \frac{1}{N}\sum\ell(\cdot;\lambda)$, but on small calibration sets, this is too optimistic and violates the true $\epsilon$. UCB (Bates 2021) replaces this with $\widetilde{\text{Risk}}(\lambda;\mathcal{V}) = \widehat{\text{Risk}} + (\text{finite-sample correction})$, which depends on $|\mathcal{V}|$ and $\delta$, ensuring $\mathbb{P}(\text{true risk}\le\epsilon)\ge 1-\delta$. Algorithm 1 performs a grid search over multiple candidate signals $\mathcal{S}$ and threshold grids $\Lambda_s$, selecting the one in the feasible set $\mathfrak{C}$ with the minimum efficiency loss—**automatic signal selection + automatic threshold selection**.
    - **Design Motivation**: Naive cross-validation frequently violates the target risk (Fig. 4 left), while UCB remains stable (Fig. 4 right). Allowing signal ensembles makes signal selection data-driven—different $\epsilon$ values might favor different signals, and allowing the algorithm to choose automatically removes the tuning burden from the user.

### Loss & Training
**Completely training-free**—all threshold and signal selections are performed via UCB on the calibration set. The model only requires inference (Qwen3-8B, Qwen3-30B-A3B, DeepSeek-R1-Distill-Qwen-32B, Qwen3-VL-8B). Algorithm 1 performs the grid search and UCB correction once to deploy $(s^*, \lambda^*)$.

## Key Experimental Results

### Main Results

| Experimental Setup | Signal | Key Phenomenon |
| :--- | :--- | :--- |
| Risk Control Validation (Fig. 4) | Multi-signal | Naive calibration frequently violates $\epsilon$; UCB calibration stays $\le\epsilon$ (with prob $1-\delta$) |
| Signal Ensemble (Fig. 5) | 4 models × Multi-signal | Risk control automatically selects the most efficient signals, outperforming any single signal |
| Lower Threshold Value (Fig. 6) | Qwen3-8B + confidence | For solvable:unsolvable = 1:1 / 1:3, Lower+Upper significantly saves tokens vs. Upper-only at the same accuracy |

### Ablation Study

| Configuration | Key Effect | Description |
| :--- | :--- | :--- |
| Upper-only | Exhausts budget on hard problems | Large number of unsolvable tasks never reach the confidence threshold |
| Lower-only | Triggers only on confidence regression | Static lower threshold is almost useless |
| **Lower+Upper (dynamic)** | Accuracy nearly constant, tokens down significantly | Complementary effect of dual thresholds |
| Naive Calibration | Frequently violates $\epsilon$ | Empirical risk is overly optimistic |
| UCB Calibration | Always $\le\epsilon$ | Finite-sample correction is effective |
| Ensemble of signals | Superior to any single signal | Risk control automatically selects the optimal signal |

### Key Findings
- **Value of lower threshold scales with problem difficulty**: Gains are small when solvable:unsolvable = 3:1 (most problems converge confidently), but significant at 1:1 and 1:3 ratios, where identifying and abstaining from unsolvable problems is the key to token efficiency.
- **Trigger distribution matches intuition**: At representative operating points for Lower+Upper, solvable problems mostly exit via the upper threshold, while unsolvable problems mostly exit early via the lower threshold.
- **No single signal is universally optimal**: The optimal signal switches under different $\epsilon$ values (Fig. 1 right), making ensemble and auto-selection necessary.
- **UCB correction is critical**: Finite-sample correction may seem "conservative," but it ensures the risk bound is not violated, whereas naive calibration fails frequently.
- **Farsighted loss for the lower threshold prevents misjudgment**: Using "current correctness" would discard problems that are wrong initially but correctable later; farsighted checking of $[t, T]$ ensures the lower threshold statistically corresponds to "incapable of solving."

## Highlights & Insights
- **Paradigm shift from threshold tuning to risk specification**: Instead of engineers tuning an opaque 0.7, users specify a 5% error tolerance, and the system derives the threshold. This is a model for interpretable deployment interfaces.
- **Lower threshold as a missing symmetric mechanism**: While many focus on stop-when-confident, few address stop-when-stuck; this paper highlights that the latter offers greater gains in difficult scenarios.
- **Clever dynamic parameterization**: The 4-parameter sigmoid is sufficient to cover linear, exponential, logarithmic, and constant schedules, providing enough search space for the conformal optimizer without being excessive.
- **Pluggable into existing systems**: Training-free, no weight changes, no decoding changes—just inference-side monitoring and calibration. Very industry-friendly.
- **Bridge between theory and practice**: Systematically transfers the EENN conformal framework to reasoning LLMs and complements it with unsolvable case handling.

## Limitations & Future Work
- **Theoretical risk of "upper bound violation"**: The authors admit that in extreme cases where the lower threshold is completely broken (discarding most solvable tasks), the two-step calibration might violate the upper risk bound—though rare in practice, OOD deployment requires caution.
- **Requirement for annotated calibration sets**: UCB needs ground truth $y^*$, making it less directly applicable to open domains without clear labels (e.g., creative writing).
- **Static calibration vs. dynamic distribution**: Calibration is based on a one-time grid search, but distributions may shift with model versions or task types. Online conformalization is not discussed.
- **Efficiency/Accuracy trade-off is not Pareto-optimal**: Different methods are Pareto-optimal at different $\epsilon$; the paper selects the best method for a given $\epsilon$ but does not bypass the trade-off itself.
- **No comparison with training-side reasoning length compression**: This is an inference-only solution, but comparisons with methods that penalize thinking length during training (e.g., Reward Shaping) are missing.

## Related Work & Insights
- **vs. Wang 2025 / Yang 2025 (Upper threshold stop-when-confident)**: Those use manual entropy/confidence thresholds; this paper uses conformal-calibrated dual thresholds.
- **vs. Thought Calibration (Wu 2025)**: Also uses risk control but focuses on stop-when-stable (using linear probes); this paper uses general signals + UCB and explicitly handles unsolvable tasks.
- **vs. PAC Reasoning (Zeng 2026)**: They conformally decide "whether to reason" (binary), whereas this paper decides "how long to reason" (continuous).
- **vs. EENN conformal risk control (Jazbec 2024)**: Inherits the UCB framework but moves from exit levels to token levels and adds the lower threshold mechanism.
- **Insight**: The conformal calibration + dual threshold paradigm ("confident in correctness" vs. "confident in inability") can be generalized to any cost-aware inference scenario, such as multi-agent discussions, active learning, or search pruning.

## Rating
- Novelty: ⭐⭐⭐⭐ Dual thresholds and parameterized dynamic lower thresholds are truly novel; the conformal framework is clean.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 models and 4 datasets; validates risk control, ensembles, and lower threshold value. However, more end-to-end token saving percentages would improve readability.
- Writing Quality: ⭐⭐⭐⭐ Clearly bridges conformal and reasoning early-exit domains; the physical meaning of the four losses is well-explained.
- Value: ⭐⭐⭐⭐ Directly deployable, training-free, and provides interpretable interfaces for users. Highly practical for search, QA, and coding assistants.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning](beyond_test-time_memory_state-space_optimal_control_for_llm_reasoning.md)
- [\[ICML 2026\] Efficient Reasoning with Hidden Thinking](efficient_reasoning_with_hidden_thinking.md)
- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](../../NeurIPS2025/llm_reasoning/towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [\[ICML 2026\] LatentChem: From Textual CoT to Latent Thinking in Chemical Reasoning](latentchem_from_textual_cot_to_latent_thinking_in_chemical_reasoning.md)
- [\[ICML 2026\] Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models](less_diverse_less_safe_the_indirect_but_pervasive_risk_of_test-time_scaling_in_l.md)

</div>

<!-- RELATED:END -->
