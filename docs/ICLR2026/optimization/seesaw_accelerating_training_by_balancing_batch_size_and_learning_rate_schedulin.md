---
title: >-
  [Paper Note] Seesaw: Accelerating Training by Balancing Learning Rate and Batch Size Scheduling
description: >-
  [ICLR 2026][Optimization & Theory][normalized SGD] This paper theoretically proves the finite-sample equivalence between "learning rate decay" and "batch size increase" under SGD (and normalized SGD as a proxy for Adam). Based on this, it proposes the plug-and-play Seesaw scheduler: whenever a cosine schedule would halve the learning rate, it instead multiplies the lea
tags:
  - ICLR 2026
  - Optimization & Theory
  - normalized SGD
  - critical batch size
date: 2026-05-08
content_hash: 36e427f6b4f10283
---
# Seesaw: Accelerating Training by Balancing Learning Rate and Batch Size Scheduling

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Nj0XBF2o7z](https://openreview.net/forum?id=Nj0XBF2o7z)  
**Area**: LLM Efficiency / Optimizer and Training Acceleration  
**Keywords**: batch size scheduling, learning rate decay, training acceleration, normalized SGD, critical batch size

## TL;DR
This paper theoretically proves the finite-sample equivalence between "learning rate decay" and "batch size increase" under SGD (and normalized SGD as a proxy for Adam). Based on this, it proposes the plug-and-play Seesaw scheduler: whenever a cosine schedule would halve the learning rate, it instead multiplies the learning rate by $1/\sqrt{2}$ and doubles the batch size. This matches the loss curve of cosine decay under iso-FLOPs while reducing serial wall-clock time by approximately 36%.

## Background & Motivation
**Background**: The wall-clock time for pre-training large models has reached several months. Batch ramping (gradually increasing batch size during training) is considered an effective means to shorten wall-clock time—larger batches utilize more parallel computation and proportionally reduce the required number of serial optimization steps. Mainstream models such as LLaMA, Nemotron, OLMo, and Apertus have adopted some form of batch ramp.

**Limitations of Prior Work**: Almost all batch ramp schemes are **heuristically tuned** without theoretical foundation. It is unknown how far these heuristics are from the optimum, nor is the correct linkage ratio between learning rate and batch size clear. For SGD, there is a classic intuition that "doubling batch ≈ halving learning rate," but for adaptive optimizers like Adam, the correct linkage rules are not well-understood.

**Key Challenge**: Batch size cannot be increased indefinitely—once it exceeds the critical batch size (CBS), sample efficiency drops and acceleration benefits disappear. Therefore, the real question is: how many "serial steps" can be traded for "parallel batches" without **sacrificing performance**? This is essentially finding the optimal schedule between "data efficiency" and "parallelism."

**Goal**: (1) Provide a rigorous theoretical framework for batch size scheduling; (2) Derive practical scheduling rules suitable for Adam training; (3) Quantify the maximum achievable speedup ratio.

**Key Insight**: The authors start from a simple observation—"taking two steps with learning rate $\eta/2$ and batch $B$" and "taking one step with learning rate $\eta$ and batch $2B$" are equivalent under first-order Taylor expansion (the deterministic parts are identical, and the noise term variances match). If this equivalence can be rigorously proven and extended to Adam, learning rate decay can be "translated" into a batch ramp.

**Core Idea**: Equivalently replace the "halving learning rate" step in standard schedulers with "learning rate ×$1/\sqrt{2}$ + batch ×2" to reduce serial steps while maintaining loss dynamics.

## Method

### Overall Architecture
Seesaw is not a new optimizer but an **equivalent rewriting rule** applied to existing schedulers (such as cosine decay). Its logical chain consists of three layers: first, rigorously proving the finite-sample equivalence between "learning rate decay ↔ batch ramp" on SGD (Theorem 1); then, extending the equivalence to adaptive optimizers (Corollary 1) using normalized SGD (NSGD, an analyzable proxy for Adam) and the "variance dominance" assumption, yielding an equivalence curve $\alpha\sqrt{\beta}=\text{const}$; finally, engineering this relationship into Algorithm 1—traversing the time points where the input scheduler (cosine approximated as stepwise) would decay the learning rate, and at each point changing the decay factor from $\alpha$ to $\sqrt{\alpha}$ while multiplying the batch by $\alpha$.

The input is an existing scheduler (providing the token counts where the learning rate decays by factor $\alpha$), and the output is a new `(step, η, B)` schedule table: the learning rate decays more slowly, the batch size increases synchronously, the loss curve remains unchanged under iso-FLOPs, but serial steps are significantly reduced.

### Key Designs

**1. Finite-sample equivalence between learning rate decay and batch ramp under SGD**

Addressing the pain point that "existing batch ramps are entirely heuristic and lack theory," this paper provides (to the authors' knowledge) the first **non-asymptotic** (finite-sample) equivalence proof. Consider running mini-batch SGD on linear regression with additive noise, where the total sample size is $D$. The base process uses a stepwise batch ramp: doubling the batch at several points while keeping the learning rate fixed. The alternative process instead halves the learning rate at the same points while keeping the batch fixed, adjusting the steps so the total number of processed samples remains $D$. Theorem 1 proves that the excess risk of the base process and the alternative process differ only by a constant factor.

The intuition for this equivalence comes from first-order Taylor expansion. For a smooth loss $L(x)$ and $g_0=\nabla L(x_0)$, the loss for "one step of $(\eta, 2B)$" and "two steps of $(\eta/2, B)$" are respectively:
$$L(x_1)=L(x_0)-\eta g_0^\top(g_0+\xi')+O(\eta^2),\quad \mathrm{Cov}(\xi')=\frac{\sigma^2}{2B}I_d$$
$$L(x_2)=L(x_0)-\frac{\eta}{2}g_0^\top(2g_0+\xi_0+\xi_1)+O(\eta^2),\quad \mathrm{Cov}(\xi_i)=\frac{\sigma^2}{B}I_d$$
Both are equivalent up to $O(\eta^2)$ in their deterministic parts and noise terms—large batches dilute noise variance, while a small learning rate over two steps averages the noise out, leading to the same effect. This paper elevates this intuition into rigorous risk bounds, forming the foundation of Seesaw.

**2. Extension to Normalized SGD and Variance Dominance Assumption**

The equivalence ratio for SGD is "LR decay $\alpha$ ↔ Batch increase $\alpha$," but LLMs use Adam, where the ratio differs. To resolve this, the paper simplifies the Adam update into NSGD (normalized SGD), a common analyzable proxy for Adam: setting $\beta_1=\beta_2=0$, approximating coordinate-wise updates with global updates, and replacing the denominator with the true expectation of the squared gradient norm:
$$\theta_t=\theta_t-\eta\frac{g_t}{\sqrt{\mathbb{E}\|g_t\|^2}}$$
Crucially, the denominator $\mathbb{E}\|g_t\|^2$ can be decomposed into "mean + variance," where the variance decays as $O(1/B)$ with the batch size. The paper introduces **Assumption 3: Variance Dominance**—assuming the squared gradient norm is primarily contributed by the additive noise variance. Under this assumption, the NSGD update degrades (in terms of constant factors) to SGD with a "re-scaled learning rate," thereby extending the risk equivalence of Theorem 1 to NSGD (Corollary 1).

The conclusion is elegant: for NSGD, the necessary and sufficient condition for the equivalence between "learning rate decay factor $\alpha$" and "batch increase factor $\beta$" is that **the product $\alpha\sqrt{\beta}$ remains constant**. Substituting $\beta=2$ (doubling the batch) yields $\alpha=\sqrt{2}$—this is why Seesaw uses $1/\sqrt{2}$ instead of $1/2$ for LR decay, contrasting with the $1/2$ used in SGD.

**3. Seesaw Scheduler: A Plug-and-Play Replacement for Cosine Scheduling**

With the equivalence curve $\alpha\sqrt{\beta}=\text{const}$, the paper engineers it into Algorithm 1. While the theory is established for stepwise decay, in practice, cosine decay is first **approximated as stepwise**: pick a decay factor $\alpha$ and record the token counts where the cosine schedule would decay the learning rate by a factor of $\alpha$; use these time points $S$ as input. Then, at each point $t\in S$, execute:
$$\eta\leftarrow \eta/\sqrt{\alpha},\qquad B\leftarrow B\cdot\alpha$$
The output is a new `(step, η, B)` schedule table. It is a drop-in replacement for existing cosine schedulers, requiring no changes to the optimizer and no additional search for warmup checkpoints.

This distinguishes it from the closest work, Merrill et al. (2025), which determines rules ($B_{t+1}=2B_t, \eta_{t+1}=\sqrt{2}\eta$) by searching for the maximum batch multiplier $k^\star$ that keeps the loss $\epsilon$-close starting from a checkpoint. This paper argues that their rule becomes unstable and diverges after a fixed number of steps because it fails to satisfy convergence constraints (see Lemma 4), whereas Seesaw is strictly derived from NSGD on quadratics and includes an anti-divergence constraint.

**4. Most Aggressive Ramp Constraint and 36% Acceleration Upper Bound**

One cannot simply increase the batch size indefinitely at any time and expect the risk to match. Lemma 4 quantifies this, stating that the **most aggressive** (non-divergent) scheme is $\alpha=\sqrt{\beta}$ (Remark 1). Combined with the equivalence curve $\alpha\sqrt{\beta}=\text{const}$, given a baseline of $\alpha=2, \beta=1$ ($\alpha\sqrt{\beta}=2$), the most aggressive choice is $\alpha=\sqrt{2}, \beta=2$. Anything more aggressive (e.g., $\alpha=1, \beta=4$) will deviate from the baseline loss curve due to instability.

Under the most aggressive limit, a theoretical upper bound for acceleration relative to cosine decay can be derived (Lemma 1): the baseline is $T$ steps with a constant batch and cosine learning rate $\eta(t)=\eta_0\cos(\frac{\pi t}{2T})$; in the continuous limit, an equivalent batch ramp process maintaining $\alpha=\sqrt{\beta}$ has a total number of serial steps equal to the integral of the normalized learning rate curve:
$$\int_0^T \frac{\eta(t)}{\eta_0}\,dt=\int_0^T \cos\!\Big(\frac{\pi t}{2T}\Big)\,dt=\frac{2T}{\pi}$$
Thus, serial wall-clock time can be reduced by at most $1-\frac{2}{\pi}\approx 36.3\%$. The reason it is less than 50% is that most training progress under a cosine schedule occurs in the **early high-learning-rate stage**, where the batch must be smaller and parallelism is limited; Seesaw primarily increases parallelism significantly in the later stages, leaving the early serial bottleneck intact.

### Loss & Training
Models were pre-trained on the OLMo codebase using Chinchilla proportions ($D=20N$), with learning rate warmup for the first 10% of tokens, followed by cosine or Seesaw decay. The optimizer was AdamW ($\lambda=0$ no weight decay, $\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$), with z-loss enabled. Learning rates were scanned across $\{0.001, 0.003, 0.01, 0.03\}$, initial batches across $\{128, 256, 512, 1024\}$, sequence length $L=1024$, and the dataset was C4 (T5 tokenizer). A decay factor of $\alpha=1.1$ was used for the stepwise approximation of cosine.

## Key Experimental Results

### Main Results
On 150M / 300M / 600M models trained at their respective critical batch sizes (CBS: 150M≈256, 300M≈512, 600M≈1024, in units of ×$L$ tokens), the final validation loss of Seesaw and cosine decay were compared under iso-FLOPs (using the optimal learning rate for the cosine schedule, $\alpha=1.1$):

| Model | B=128 | B=256 | B=512 | B=1024 |
|------|-------|-------|-------|--------|
| 150M (Cosine) | 3.0282 | 3.0353 | 3.0696 | 3.1214 |
| 150M (Seesaw) | 3.0208 | 3.0346 | 3.0687 | 3.1318 |
| 300M (Cosine) | 2.8531 | 2.8591 | 2.8696 | 2.9369 |
| 300M (Seesaw) | 2.8452 | 2.8561 | 2.8700 | 2.9490 |
| 600M (Cosine) | - | 2.6904 | 2.6988 | 2.7128 |
| 600M (Seesaw) | - | 2.6883 | 2.6944 | 2.7132 |

When training at CBS, the final losses for both schedulers across all three scales are highly consistent (differences in the third decimal place), while Seesaw's serial wall-clock time is reduced by approximately 36%, approaching the theoretical limit.

### Ablation Study
Different $(\alpha, \beta)$ were taken along the equivalence line $\alpha\sqrt{\beta}=2$ to verify the "most aggressive scheme $\alpha=\sqrt{\beta}$" constraint (150M, fixed batch, Chinchilla scale):

| Config $(\alpha, \beta)$ | $\alpha \ge \sqrt{\beta}$? | Observation |
|------|------|------|
| $(2, 1)$ | Yes (Baseline Step) | Matches baseline loss |
| $(2^{3/4}, \sqrt{2})$ | Yes | Matches baseline |
| $(\sqrt{2}, 2)$ | Critical (Aggressive) | Still matches |
| $(2^{1/4}, 2^{3/2})$ | No | Deviates from baseline, unstable |
| $(1, 4)$ | No | Significant deviation, diverges |

### Key Findings
- **Variance Dominance is the lifeblood of Seesaw**: Assumption 3 requires the squared gradient norm to be dominated by additive noise variance. When the batch size increases sufficiently (exceeding CBS, e.g., 1024/2048/4096/8192), the noise variance $O(1/B)$ is suppressed, the mean term begins to dominate, and Seesaw fails to match the cosine curve, with the deviation increasing with batch size.
- **Fundamental impossibility beyond CBS**: The authors use a 1D NGD toy example to illustrate—on quadratic loss $L(x)=\frac{1}{2}hx^2$, the NGD update $x_{t+1}=x_t+\eta h\,\mathrm{sign}(x_t)$ stops in a stable ring of $O(\eta)$ around the minimizer. One **must decay the learning rate** to approach the optimum. Large batches approach the NGD regime, where further increasing the batch does not change the dynamics. Therefore, beyond a certain batch size, no batch ramp with a fixed learning rate can replicate the effect of learning rate decay.
- **Hard upper bound on aggressiveness**: $\alpha=\sqrt{\beta}$ is the most aggressive non-divergent choice. Greedily increasing the batch beyond this will result in loss degradation due to instability.

## Highlights & Insights
- **Upgrading engineering heuristics to theorems**: Batch ramps have been used in industry for years based on manual tuning. This paper provides the first finite-sample equivalence proof and explicitly points out that the correct linkage under Adam is a constant $\alpha\sqrt{\beta}$ (unlike $\alpha\beta$ for SGD). The seemingly strange coefficient $1/\sqrt{2}$ now has a rigorous origin.
- **Analysis path of NSGD as Adam proxy + Variance Dominance**: Simplifying adaptive optimizers to the analyzable NSGD and then using a verifiable training regime assumption to reduce the problem back to SGD is a strong paradigm for studying adaptive optimizer scheduling.
- **Drop-in replacement**: Seesaw does not touch the optimizer or require extra searches. Directly applying it over an existing cosine schedule yields ~36% wall-clock speedup with extremely low implementation cost.
- **Honestly presenting failure boundaries**: Using the 1D NGD toy example to explain "why it's fundamentally impossible past CBS" rather than pretending the method is universal.

## Limitations & Future Work
- **Reliance on Variance Dominance**: Seesaw only matches cosine decay when the batch size does not exceed CBS and noise variance dominates the denominator. Once the batch enters the mean-dominated regime, it fails, limiting the maximum achievable parallelism.
- **Theory built on linear regression / quadratic loss**: The equivalence theorems are proven for noisy linear regression and NSGD on quadratics. Applying this to real Transformers is an empirical extrapolation with a theoretical gap.
- **Limited experimental scale**: The maximum scale is 600M with Chinchilla proportions on the C4 dataset. Whether the ~36% speedup holds on larger models, longer training, or different data distributions remains to be verified.
- **Refinement ideas**: Whether "variance vs mean ratio" can be estimated online to adaptively decide when to stop the batch ramp—to extract more acceleration near the CBS boundary—is a natural next step.

## Related Work & Insights
- **vs Merrill et al. (2025)**: They determined rules ($B_{t+1}=2B_t, \eta_{t+1}=\sqrt{2}\eta$) by searching for the maximum batch multiplier $k^\star$ that avoids loss degradation from a checkpoint. This paper proves their rule eventually diverges; Seesaw is rigorously derived from NSGD on quadratics, includes an $\alpha=\sqrt{\beta}$ anti-divergence constraint, and is a plug-and-play replacement without the need for search.
- **vs Linear Scaling Rule (Smith et al., 2017)**: They empirically observed "linearly increasing batch ≈ decreasing learning rate" for SGD. This paper provides a non-asymptotic finite-sample proof and extends the ratio to $\alpha\sqrt{\beta}$ for adaptive optimizers.
- **vs Noise Scale (McCandlish et al., 2018)**: They used Hessian-based metrics to characterize CBS (requiring Hessian access, which is infeasible at scale) and observed noise scale increasing during training. The theoretical predictions of this paper are consistent with "noise scale increasing during training" but do not require Hessian calculation.
- **vs Malladi et al. (2022) / Square Root Scaling**: They used SDEs to study learning rate scaling with batch size under adaptive algorithms. This paper reuses their first-order equivalence intuition but translates it into an actionable scheduler and provides risk bounds.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First finite-sample equivalence proof for LR decay ↔ batch ramp, extended to Adam proxy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three scales + equivalence line ablation + failure boundaries, though max scale is 600M.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from theoretical intuition to algorithm and acceleration limits; honest failure analysis.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, ~36% wall-clock acceleration, direct engineering value for LLM pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] WSM: Decay-free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training](wsm_decay-free_learning_rate_schedule_via_checkpoint_merging_for_llm_pre-trainin.md)
- [\[ICLR 2026\] Predictive Differential Training Guided by Training Dynamics](predictive_differential_training_guided_by_training_dynamics.md)
- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)
- [\[ICLR 2026\] Shuffling the Data, Stretching the Step-Size: Sharper Bias in Constant Step-Size SGD](shuffling_the_data_extrapolating_the_step_sharper_bias_in_constant_step-size_sgd.md)
- [\[ICLR 2026\] Weight Decay May Matter More Than µP for Learning Rate Transfer in Practice](weight_decay_may_matter_more_than_µp_for_learning_rate_transfer_in_practice.md)

</div>

<!-- RELATED:END -->
