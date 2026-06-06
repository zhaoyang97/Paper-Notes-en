---
title: >-
  [Paper Note] GradPower: Powering Gradients for Faster Language Model Pre-Training
description: >-
  [ICML 2026][Model Compression][Gradient Transformation] GradPower applies an element-wise "sign-preserving power" transformation $\varphi_p(g_i)=\mathrm{sign}(g_i)\…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Gradient Transformation"
  - "AdamW"
  - "Muon"
  - "MoE Pre-training"
  - "wsd Scheduling"
date: 2026-05-08
content_hash: fa8d99cc21af013e
---

# GradPower: Powering Gradients for Faster Language Model Pre-Training

**Conference**: ICML 2026  
**arXiv**: [2505.24275](https://arxiv.org/abs/2505.24275)  
**Code**: No repository link explicitly provided in the paper  
**Area**: LLM Pre-training / Optimizer / Training Acceleration  
**Keywords**: Gradient Transformation, AdamW, Muon, MoE Pre-training, wsd Scheduling

## TL;DR
GradPower applies an element-wise "sign-preserving power" transformation $\varphi_p(g_i)=\mathrm{sign}(g_i)\,|g_i|^p$ to raw gradients before feeding them into any gradient-based optimizer. With just one line of code change and without altering the internal logic or hyperparameters of AdamW/Muon, it consistently achieves lower final loss across multiple scales of LLaMA and Qwen2MoE (from 66M to 2B). The gains are most significant under MoE + wsd (warmup-stable-decay) learning rate scheduling.

## Background & Motivation
**Background**: LLM pre-training computation is extremely expensive, making the optimizer the most direct lever for efficiency. AdamW has become the de facto standard due to its coordinate-wise adaptive learning rate. Recent works (Muon, Blockwise LR, Lion, SOAP, CAME, etc.) attempt to further reduce final loss by incorporating curvature information, matrix preconditioning, hybrid momentum, or cautious updates.

**Limitations of Prior Work**: These "invasive" modifications often require redesigning momentum, second moments, or the entire update rule. For training pipelines, this means re-tuning hyperparameters like lr, $\beta_1$, $\beta_2$, weight decay, and clipping, which incurs extremely high engineering costs and slows community adoption.

**Key Challenge**: The desire to "re-accelerate" AdamW directly conflicts with the desire to "not disturb the existing pipeline"—any method that modifies the update rule breaks the tuned hyperparameter combinations.

**Goal**: Find a plug-and-play acceleration plugin that is compatible with all modern optimizers without modifying AdamW's internal logic or requiring hyperparameter re-tuning.

**Key Insight**: The authors formulate the optimizer in a preconditioned form $\theta_{t+1}=\theta_t-\eta_t\,\mathcal{Q}(\varphi(g_1),\dots,\varphi(g_t))$, where debates over existing optimizers are essentially debates over $\mathcal{Q}$. They propose an alternative: keep $\mathcal{Q}=\text{AdamW}$ fixed and change $\varphi$ at the outermost layer. Given that LLM pre-training is often in a "noise-dominated" regime where gradient magnitude differences primarily stem from noise, and recent studies on EoS / river-valley / bulk direction show that loss reduction depends on "slow dynamics along flat directions," the goal of $\varphi$ should be to relatively amplify "small but persistent" flat directions.

**Core Idea**: Apply $\varphi_p(g)=\mathrm{sign}(g)\,|g|^p$ to each gradient component. When $p>1$, the contrast is increased such that "major directions are suppressed and minor directions are relatively amplified," thereby accelerating cumulative progress along flat directions. A default value of $p=1.2$ is chosen, which proves robust across architectures, scales, and schedules.

## Method

### Overall Architecture
GradPower is not a new optimizer but an element-wise pre-transformation layer for any gradient-based optimizer:

1. **Forward + Backward**: Identical to standard training, yielding mini-batch gradients $g_t\in \mathbb{R}^d$.
2. **GradPower Transformation**: Execute a single line $g_t\leftarrow \mathrm{sign}(g_t)\odot|g_t|^p$. This is computed independently per element and does not depend on any state.
3. **Standard Clipping + Optimizer**: Feed the transformed $g_t$ into AdamW / Muon / Blockwise LR / AdaGrad / etc., as the "gradient." Update rules, first/second moments, weight decay, and hyperparameters remain entirely unchanged.

Empirical tests on LLaMA-0.25B / OpenWebText show only a ~0.4% increase in wall-clock time per step (0.7565s vs 0.7534s), which is negligible relative to total training time. The authors further note that whether gradient clipping occurs before or after GradPower does not significantly affect the final curve; both sequences ensure bounded updates.

### Key Designs

1. **Sign-Preserving Power Transformation $\varphi_p$**:
    - **Function**: Performs a non-linear transformation on each gradient component via $\varphi_p(g)=\mathrm{sign}(g)\,|g|^p$. For $p>1$, it amplifies relative differences while suppressing absolute magnitudes; for $p<1$, the opposite occurs.
    - **Mechanism**: Using a 1D toy example $g_t\sim\mathrm{Unif}(\mu-\sigma,\mu+\sigma)$, the authors calculate the long-term cumulative update of AdamW $u_t=m_t/(\sqrt{v_t}+\epsilon)$. They prove that in high-noise regimes ($\sigma\gg\mu$, corresponding to LLM pre-training where batch sizes are much smaller than the full dataset), the optimal $p^\star>1$. In this case, $\varphi_p$ relatively amplifies "weak but stable signal" in flat directions, accelerating slow dynamics in the "river" direction. In low-noise regimes (large batches), the optimal $p^\star<1$, as noise suppression becomes more critical than amplification.
    - **Design Motivation**: Directly motivated by the EoS / river-valley perspective—loss reduction depends on steady accumulation along flat directions rather than the oscillation amplitude in sharp directions. The power transform is a minimal-cost way to artificially amplify the contribution of flat directions.

2. **Keeping the Base Optimizer Unchanged**:
    - **Function**: Transformation occurs before the optimizer; parameters like $\beta_1,\beta_2,\epsilon,\lambda$ for AdamW, orthogonalization for Muon, and block coefficients for Blockwise LR are all preserved at their original values.
    - **Mechanism**: The authors deliberately decouple $\varphi$ and $\mathcal{Q}$. All hyperparameters tuned with significant compute in existing LLaMA recipes do not need re-tuning when switching to GradPower. Only the single new parameter $p$ needs a one-time grid search on a small scale. The paper determines $p=1.2$ using LLaMA-0.2B / C4 and applies it across model sizes (66M to 2B), architectures (dense LLaMA, MoE Qwen2MoE), datasets (C4, OpenWebText), and schedules (cos, wsd).
    - **Design Motivation**: Eliminates engineering barriers. Any existing pre-training pipeline can adopt it by adding `g = g.sign() * g.abs().pow(p)`. "No hyperparameter re-tuning" is the most critical selling point for industry adoption.

3. **Orthogonal Superposition with Modern Optimizers and Schedulers**:
    - **Function**: GradPower provides additive gains when combined with Muon, Blockwise LR, and wsd scheduling.
    - **Mechanism**: By treating the Muon orthogonalization update as $\mathcal{Q}$ and wrapping it with $\varphi_{1.2}$, one obtains MuonPower. Similarly, AdamW + Blockwise LR becomes BlockwisePower. Experiments show that AdamWPower(0.015) + Blockwise(0.030) $\approx$ Combined(0.045), indicating that their contributions are nearly linearly additive. This suggests GradPower captures a degree of freedom entirely different from "blockwise learning rates" or "matrix preconditioning." In wsd scheduling, GradPower's advantage grows steadily during the stable phase, aligning perfectly with modern pipelines like DeepSeek-V3 (long stable + short decay).
    - **Design Motivation**: The authors position GradPower as a "universal plugin" rather than an "AdamW variant." Any optimizer that fits the $\varphi$ interface can benefit from flat-direction amplification, allowing it to evolve alongside future optimizers.

### Loss & Training
No additional loss functions are introduced; standard next-token cross-entropy for language modeling is used. Clipping threshold 1.0, weight decay 0.1, and $\beta_1=0.9, \beta_2=0.95$ follow the original LLaMA recipe. The `lr_max` is first tuned to the optimum for AdamW across `{1e-4, 2e-4, 3e-4, 6e-4, 1e-3, 1.5e-3}`, and AdamWPower uses this same `lr_max`. $p=1.2$ is fixed for all main experiments.

## Key Experimental Results

### Main Results
Zero-shot evaluation after pre-training LLaMA-2B on C4. AdamWPower wins in 5 out of 6 tasks:

| Dataset | Metric | AdamW | AdamWPower(p=1.2) | Gain |
|---------|--------|-------|--------------------|------|
| ARC-E | acc | 60.02 | 60.35 | +0.33 |
| HellaSwag | acc | 44.65 | 44.93 | +0.28 |
| OBQA | acc | 24.80 | 25.00 | +0.20 |
| WinoGrande| acc | 56.83 | 59.43 | +2.60 |
| PIQA | acc | 73.56 | 73.61 | +0.05 |
| 6-task Avg| acc | 47.72 | 48.26 | +0.54 |

Regarding final pre-training loss, AdamWPower outperformed AdamW across multiple combinations of 66M / 0.2B / 0.4B / 1B / 2B scales, C4 / OpenWebText datasets, and cos / wsd schedules. The gain was even more pronounced for MoE—Qwen2MoE-2B saw an absolute loss improvement of 0.028, compared to 0.022 for LLaMA-2B (even though Qwen2MoE-2B started at a lower loss of 1.93, where further reductions are harder).

### Ablation Study
Relationship between $p$ selection and batch size (verified on ResNet-34 / CIFAR-10 to show GradPower is not limited to language models):

| Batch Size | p=0.8 | p=0.9 | p=1.0 | p=1.1 | p=1.2 |
|------------|-------|-------|-------|-------|-------|
| 128 | 94.35 | 94.22 | 93.98 | 93.38 | 93.15 |
| 64 | 94.22 | 94.22 | 94.10 | 93.97 | 93.77 |
| 32 | 94.04 | 94.15 | 94.30 | 94.25 | 93.85 |

A clear trend is observed: **larger batch sizes (lower noise) lead to smaller optimal $p$ values**. For large batches, the optimal $p<1$; for small batches/LLM pre-training, the optimal $p>1$. This perfectly aligns with the theoretical analysis of "amplifying flat directions in high-noise regimes."

### Key Findings
- GradPower gains are maximized under the **MoE + wsd** combination. While Qwen2MoE-1B and 2B exhibited loss spikes under AdamW, AdamWPower almost eliminated them. The authors hypothesize that the power transform suppresses high-frequency oscillations in sharp directions, leading to more stable training.
- The value $p=1.2$, tuned on LLaMA-0.2B, remained optimal across scales (66M to 2B), architectures (dense to MoE), and schedules (cos to wsd). **Strong cross-scale transferability** avoids the cost of re-tuning for every model.
- Gains are approximately additive with Blockwise LR / Muon. This implies GradPower addresses a dimension of optimization (steady accumulation in flat directions) orthogonal to "blockwise learning rates" or "matrix preconditioning."

## Highlights & Insights
- Achieving a final loss reduction of 0.02–0.03 for MoE/LLM pre-training with just one line of code and one hyperparameter $p$ offers an unbeatable "ROI" for industry deployment—it is a rare "zero engineering cost" accelerator in the ICML ecosystem.
- Re-framing the "optimizer war" as a decomposition of $\varphi$ and $\mathcal{Q}$ provides a very clean perspective. While most research modifies $\mathcal{Q}$, the authors are among the first to seriously explore the space of $\varphi$, opening a new design dimension.
- The phase transition ($p^\star>1$ for high noise, $p^\star<1$ for low noise) explains why previous similar ideas (like sign-SGD, which corresponds to the limit $p \to 0$) failed in large-batch settings—they simply targeted the wrong noise regime. This insight can guide when to use Lion-like vs. GradPower-like methods.
- The design philosophy—"it won't be adopted unless it leaves the existing pipeline untouched"—is worth emulating for other systems-oriented research. Many algorithms lose to AdamW not because of performance, but because the migration cost of re-tuning hyperparameters is too high.

## Limitations & Future Work
- The paper acknowledges that the explanation for "why power transforms suppress loss spikes in MoE" is primarily intuitive (suppressing oscillations in sharp directions) and lacks rigorous proof. The theoretical portion focuses on 1D toy examples and general non-convex AdaGrad, which is still removed from the actual optimization geometry of Transformers.
- Experiments capped at 2B, leaving unverified whether $p=1.2$ remains optimal at larger scales (10B+). Theory suggests noise levels change with batch size; ultra-large models often use larger token batches, potentially requiring $p$ to be re-tuned toward $< 1.2$.
- While the "river-valley / flat direction" visualization is popular in recent literature, it remains somewhat informal (approximating Hessian eigenvectors with small stochastic gradient directions), which impacts the mathematical precision of the GranPower explanation.
- Future directions: Implementing adaptive $p$ per-layer or per-block (combining Blockwise LR ideas), or dynamically adjusting $p$ during training (higher $p$ early on to explore flat directions, decreasing to $p \approx 1$ for fine-tuning). Several results in the paper hint at the potential of this approach.

## Related Work & Insights
- **vs Muon (Jordan et al. 2024 / Liu et al. 2025a)**: Muon modifies $\mathcal{Q}$ by introducing matrix orthogonalization preconditioning; GradPower modifies $\varphi$ via element-wise non-linear transformation. Their additivity (MuonPower) suggests they capture orthogonal dimensions in the optimizer space.
- **vs Blockwise LR (Wang et al. 2025)**: Blockwise LR assigns different learning rates to different Transformer blocks (internal refinement of $\mathcal{Q}$). Its gain is nearly linear when combined with GradPower (0.030 + 0.015 $\approx$ 0.045).
- **vs sign-SGD / Lion (Chen et al. 2024b)**: Essentially limit versions where $p \to 0$, discarding all magnitude information. GradPower demonstrates that in high-noise LLM pre-training, $p$ should be $>1$ rather than $\to 0$, providing counter-evidence to the "aggressive sign-ification" of Lion-like methods.
- **vs Cautious update (Liang et al. 2024) / Variance reduction (Yuan et al. 2024)**: These modify the update rule within $\mathcal{Q}$ and require hyperparameter re-tuning. GradPower's selling point is its "external attachment" nature.
- **Inspiration**: GradPower could be tested during RLHF / SFT / fine-tuning phases. These stages often involve small batches, high noise, and slow dynamics in flat directions, making them theoretically suitable for low-cost expansion.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The "modify $\varphi$ instead of $\mathcal{Q}$" perspective is very clear, though the power transform itself is simple and the theory relies on 1D extensions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High coverage across architectures, scales, data, schedules, optimizers, and batch sizes, including CV verification.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation is smooth, and the noise-to-signal ratio provides a tight link between theory and experiments; however, the river-valley terminology assumes familiarity with specific recent works.
- **Value**: ⭐⭐⭐⭐⭐ One line of code for stable gains in MoE pre-training + wsd scheduling (the modern mainstream setup) has extremely high deployment value and could become a default plugin in post-LLaMA training recipes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)
- [\[ICML 2026\] Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization](turning_stale_gradients_into_stable_gradients_coherent_coordinate_descent_with_i.md)
- [\[ICML 2026\] Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models](bounded_hyperbolic_tangent_a_stable_and_efficient_alternative_to_pre-layer_norma.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation](dont_ignore_the_tail_decoupling_top-k_probabilities_for_efficient_language_model.md)

</div>

<!-- RELATED:END -->
