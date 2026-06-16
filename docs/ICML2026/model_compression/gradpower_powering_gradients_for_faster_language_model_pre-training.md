---
title: >-
  [Paper Note] GradPower: Powering Gradients for Faster Language Model Pre-Training
description: >-
  [ICML 2026][Model Compression][AdamW] GradPower applies an element-wise "sign-preserving power" transformation $\varphi_p(g_i)=\mathrm{sign}(g_i)\,|g_i|^p$ to raw gradients before feeding them into any gradient-based optimizer. With just one additional line of code and without altering internal AdamW/Muon logic or hyperparameters, it consistently achieves
tags:
  - ICML 2026
  - Model Compression
  - AdamW
  - Muon
date: 2026-05-08
content_hash: 0b2a145ae592dd55
---
# GradPower: Powering Gradients for Faster Language Model Pre-Training

**Conference**: ICML 2026  
**arXiv**: [2505.24275](https://arxiv.org/abs/2505.24275)  
**Code**: Repository link not explicitly provided in the paper  
**Area**: LLM Pre-training / Optimizer / Training Acceleration  
**Keywords**: Gradient Transformation, AdamW, Muon, MoE Pre-training, wsd scheduling

## TL;DR
GradPower applies an element-wise "sign-preserving power" transformation $\varphi_p(g_i)=\mathrm{sign}(g_i)\,|g_i|^p$ to raw gradients before feeding them into any gradient-based optimizer. With just one additional line of code and without altering internal AdamW/Muon logic or hyperparameters, it consistently achieves lower final loss across multiple scales of LLaMA and Qwen2MoE (66M to 2B). The gains are most significant under MoE architectures and wsd learning rate schedules.

## Background & Motivation
**Background**: LLM pre-training computation is extremely expensive, making optimizers a direct lever for efficiency. While AdamW is the de facto standard due to coordinate-wise adaptive learning rates, recent works (Muon, Blockwise LR, Lion, SOAP, CAME, etc.) attempt to further reduce final loss by incorporating curvature information, matrix preconditioning, hybrid momentum, or cautious updates.

**Limitations of Prior Work**: These "invasive" modifications often require redesigning momentum, second moments, or the entire update rule. For training pipelines, this necessitates re-tuning hyperparameters like learning rate, $\beta_1$, $\beta_2$, weight decay, and clipping, resulting in high engineering costs and slow community adoption.

**Key Challenge**: The desire to "accelerate" AdamW contradicts the desire to "keep the existing pipeline intact"—any method modifying the update rule breaks tuned hyperparameter combinations.

**Goal**: To find a plug-and-play acceleration plugin that does not modify internal optimizer logic, requires no hyperparameter re-tuning, and is applicable to all modern optimizers.

**Key Insight**: The authors express optimizers in a preconditioned form $\theta_{t+1}=\theta_t-\eta_t\,\mathcal{Q}(\varphi(g_1),\dots,\varphi(g_t))$, where debates usually focus on $\mathcal{Q}$. This paper reverses the focus: keeping $\mathcal{Q}=\text{AdamW}$ and only replacing $\varphi$ at the outer layer. Given that LLMs are often in a "noise-dominated" regime during pre-training where gradient magnitude differences stem largely from noise, and recent studies on EoS/river-valley/bulk direction suggest that loss reduction depends on "slow dynamics along flat directions," the goal of $\varphi$ should be to relatively amplify "small but persistent" flat directions.

**Core Idea**: Apply $\varphi_p(g)=\mathrm{sign}(g)\,|g|^p$ to each gradient component. When $p>1$, the contrast is increased such that "large directions are suppressed and small directions are relatively amplified," thereby accelerating cumulative progress along flat directions. A default value of $p=1.2$ proves robust across architectures, scales, and schedules.

## Method

### Overall Architecture
GradPower is not a new optimizer but a "pointwise power transformation" layer inserted before gradients enter the optimizer. Backpropagation proceeds as usual to obtain mini-batch gradients $g_t\in\mathbb{R}^d$, followed by the line $g_t\leftarrow \mathrm{sign}(g_t)\odot|g_t|^p$. The transformed gradients are then fed into AdamW, Muon, Blockwise LR, or AdaGrad. The optimizer's update rules, moments, weight decay, and hyperparameters remain identical. The overhead consists only of element-wise sign and power operations, requiring no state; empirical tests on LLaMA-0.25B/OpenWebText show an increase of only ~0.4% in wall-clock time per step (0.7565s vs 0.7534s), which is negligible. Applying gradient clipping either before or after the transformation does not affect the final curve, ensuring bounded updates in both cases.

### Key Designs

**1. Sign-Preserving Power Transformation $\varphi_p$: Accelerating Flat Direction Progress via Nonlinearity**

The core of the method is the operator $\varphi_p(g)=\mathrm{sign}(g)\,|g|^p$. It preserves the direction of each component while applying a power operation to the magnitude. For $p>1$, it widens the relative gap between components while lowering absolute magnitudes; for $p<1$, the opposite occurs. Using a 1D toy example $g_t\sim\mathrm{Unif}(\mu-\sigma,\mu+\sigma)$, the authors derive the cumulative update $u_t=m_t/(\sqrt{v_t}+\epsilon)$ for AdamW. They prove that in high-noise regimes ($\sigma\gg\mu$, typical for LLM training where batch size is small relative to the total data), the optimal $p^\star>1$. Conversely, in low-noise (large batch) regimes, the optimal $p^\star<1$, as large batches require suppressing intermittent noise rather than amplifying it.

Amplify small components is effective in high-noise regimes because it aligns with the EoS/river-valley perspective: loss reduction depends on the steady accumulation of slow dynamics along flat (river) directions rather than the oscillation amplitude in sharp directions. Flat directions often have smaller magnitudes but are "stable signals"; $p>1$ raises them relatively, effectively accelerating progress in the river direction at minimal cost. Meanwhile, sharp directions, often dominated by noise, are relatively suppressed, leading to faster oscillation convergence. This explains why extreme cases like $p\to0$ (sign-SGD/Lion) fail under large batches, as they fall into the wrong noise regime.

**2. Base Optimizer Invariant: Decoupling $\varphi$ and $\mathcal{Q}$ to Eliminate Adoption Barriers**

The bottleneck for industry adoption of new optimizers is migration cost—tuning lr, $\beta_1$, $\beta_2$, weight decay, and clipping. By decoupling the outer transformation $\varphi$ from the inner optimizer $\mathcal{Q}$ in $\theta_{t+1}=\theta_t-\eta_t\,\mathcal{Q}(\varphi(g_1),\dots)$, the authors allow GradPower to be used without touching existing hyperparameters. AdamW's $\beta_1, \beta_2, \epsilon, \lambda$, Muon's orthogonalization, and Blockwise LR's coefficients remain at their original values.

The only new degree of freedom is $p$, which only needs to be grid-searched once on a small scale. The paper uses LLaMA-0.2B/C4 to fix $p=1.2$, and this value is then applied across model sizes (66M to 2B), architectures (dense LLaMA, MoE Qwen2MoE), datasets (C4, OpenWebText), and schedules (cos, wsd). For implementation, one simply adds `g = g.sign() * g.abs().pow(p)` to the pipeline.

**3. Orthogonality with Modern Optimizers and Schedulers: A Universal Plugin**

Because it operates at the $\varphi$ interface, GradPower's benefits are orthogonal to methods modifying internal $\mathcal{Q}$ logic. It can be combined directly: applying $\varphi_{1.2}$ before Muon's orthogonalization update rule yields MuonPower; using it with AdamW + Blockwise LR yields BlockwisePower. Experiments show that the gains of AdamWPower (0.015) and Blockwise LR (0.030) sum nearly linearly (~0.045), indicating that "amplifying accumulation along flat directions" targets a different dimension than "blockwise learning rates" or "matrix preconditioning."

Under wsd scheduling, this advantage grows steadily during the stable phase, fitting modern pipeline trends like DeepSeek-V3 (long stable phase + short decay). The authors position GradPower as a "universal plugin" that can evolve alongside future optimizers.

### Loss & Training
No additional loss is introduced; standard next-token cross-entropy is used. Clipping threshold is set to 1.0, weight decay to 0.1, and $\beta_1=0.9, \beta_2=0.95$ following the LLaMA recipe. lr_max is tuned for AdamW across `{1e-4, 2e-4, 3e-4, 6e-4, 1e-3, 1.5e-3}`, and AdamWPower uses the same lr_max. $p=1.2$ is fixed for all main experiments.

## Key Experimental Results

### Main Results
On zero-shot evaluation for LLaMA-2B pre-trained on C4, AdamWPower outperforms AdamW in 5 out of 6 tasks:

| Dataset | Metric | AdamW | AdamWPower(p=1.2) | Gain |
|---------|--------|-------|--------------------|------|
| ARC-E | acc | 60.02 | 60.35 | +0.33 |
| HellaSwag | acc | 44.65 | 44.93 | +0.28 |
| OBQA | acc | 24.80 | 25.00 | +0.20 |
| WinoGrande | acc | 56.83 | 59.43 | +2.60 |
| PIQA | acc | 73.56 | 73.61 | +0.05 |
| 6-task Avg | acc | 47.72 | 48.26 | +0.54 |

For final pre-training loss, AdamWPower wins across all combinations of 66M/0.2B/0.4B/1B/2B, C4/OpenWebText, and cos/wsd. The gains are more significant for MoE architectures—Qwen2MoE-2B shows an absolute loss improvement of 0.028, larger than LLaMA-2B's 0.022, despite Qwen2MoE-2B starting at a lower loss (1.93).

### Ablation Study
Relationship between $p$ and batch size (validated on ResNet-34 / CIFAR-10 to show GradPower applies beyond language models):

| batch size | p=0.8 | p=0.9 | p=1.0 | p=1.1 | p=1.2 |
|------------|-------|-------|-------|-------|-------|
| 128 | 94.35 | 94.22 | 93.98 | 93.38 | 93.15 |
| 64 | 94.22 | 94.22 | 94.10 | 93.97 | 93.77 |
| 32 | 94.04 | 94.15 | 94.30 | 94.25 | 93.85 |

A clear trend is observed: **the larger the batch (lower noise), the smaller the optimal $p$**. For very large batches, $p<1$ is optimal, while $p>1$ is optimal for small batches and LLM pre-training, aligning with the theoretical analysis of amplifying flat directions in high-noise regimes.

### Key Findings
- GradPower gains are maximized under the **MoE + wsd** combination. Qwen2MoE (1B and 2B) exhibited loss spikes under AdamW that were nearly eliminated by AdamWPower, suggesting the power transform suppresses high-frequency oscillations in sharp directions.
- The value $p=1.2$ tuned on LLaMA-0.2B remains optimal from 66M to 2B and across dense/MoE architectures, demonstrating strong **cross-scale transferability**.
- Gains with Blockwise LR and Muon are additive, confirming GradPower captures a unique dimension of optimization progress (amplifying accumulation along flat directions) distinct from matrix preconditioning.

## Highlights & Insights
- Achieving a 0.02–0.03 magnitude improvement in final loss for MoE/LLM pre-training with one line of code and one hyperparameter $p$ represents an exceptional ROI—a rare "zero engineering cost" accelerator for ICML.
- Reframing the "optimizer debate" as a decomposition of $\varphi$ and $\mathcal{Q}$ provides a clean perspective, focusing design effort on the previously overlooked $\varphi$ interface.
- The phase transition ($p^\star>1$ in high noise, $p^\star<1$ in low noise) explains why past "sign-based" methods ($p=0$) fail at larger batches—they target the wrong noise regime. This insight guides when to use Lion-like versus GradPower-like approaches.
- The design philosophy that "existing pipelines must remain untouched for adoption" is a valuable lesson for systems-oriented ML research.

## Limitations & Future Work
- The explanation for why the power transform suppresses loss spikes in MoE remains largely intuitive (suppressing oscillations in sharp directions) without rigorous proof. Theoretical analysis is limited to 1D toy models and general non-convex AdaGrad, which remains distant from the complex geometry of Transformers.
- Experimental scale is capped at 2B. It remains unverified if $p=1.2$ remains optimal at scales of 10B+. Theory suggests that since noise levels change with batch size, larger models (using larger batches) might require tuning $p$ closer to 1.
- While the "river-valley/flat direction" framework is popular, its definition remains semi-formal (approximating Hessian eigenvectors with small stochastic gradient directions), affecting the rigor of the explanation.
- Future work could explore per-layer/per-block adaptive $p$ (complemented by Blockwise LR) or dynamic $p$ schedules (high $p$ early for exploration, $p\approx 1$ later for fine-tuning).

## Related Work & Insights
- **vs Muon (Jordan et al. 2024 / Liu et al. 2025a)**: Muon modifies $\mathcal{Q}$ via matrix orthogonalization; GradPower modifies $\varphi$ via element-wise nonlinearity. Their additivity (MuonPower) confirms they address orthogonal dimensions.
- **vs Blockwise LR (Wang et al. 2025)**: Blockwise LR refines learning rates per Transformer block; it demonstrates nearly linear gain addition with GradPower.
- **vs sign-SGD / Lion (Chen et al. 2024b)**: These are essentially $p \to 0$ limits discarding all magnitude information. GradPower provides evidence that in high-noise LLM pre-training, $p$ should be $>1$ rather than $\to 0$.
- **vs Cautious update (Liang et al. 2024) / Variance reduction (Yuan et al. 2024)**: These modify the update rule within $\mathcal{Q}$ and require re-tuning, whereas GradPower is purely external.
- Insight: GradPower could theoretically be extended to RLHF, SFT, and fine-tuning, which also exhibit small-batch/high-noise characteristics.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of modifying $\varphi$ rather than $\mathcal{Q}$ is clear and elegant, although the power transform form is simple and the theory relies on toy models.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across architectures, scales, data, schedules, and optimizers, including a CV validation.
- Writing Quality: ⭐⭐⭐⭐ Motivation is well-explained and the noise-to-signal ratio provides a consistent thread, though readers must be familiar with recent "river-valley" literature.
- Value: ⭐⭐⭐⭐⭐ High practical value due to its "one-line code" nature and stability in MoE+wsd settings; highly likely to be adopted in post-LLaMA training recipes.

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
