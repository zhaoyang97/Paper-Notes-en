---
title: >-
  [Paper Note] Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences
description: >-
  [ICML 2026][Image Generation][Diffusion Model Memorization] This paper characterizes "local memory in diffusion models" as **variance collapse (high curvature)** on specific coordinates of the log-density. By using the *…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Diffusion Model Memorization"
  - "Local Memory Localization"
  - "Curvature Difference"
  - "score difference"
  - "Fisher Information"
date: 2026-05-08
content_hash: 3340269666f613af
---

# Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences

**Conference**: ICML 2026  
**arXiv**: [2605.26756](https://arxiv.org/abs/2605.26756)  
**Code**: https://github.com/Gwangho99/mem-curv-diff  
**Area**: AI Safety / Diffusion Model Memorization / Privacy & Copyright  
**Keywords**: Diffusion Model Memorization, Local Memory Localization, Curvature Difference, score difference, Fisher Information

## TL;DR
This paper characterizes "local memory in diffusion models" as **variance collapse (high curvature)** on specific coordinates of the log-density. By using the **coordinate-wise curvature difference** between a "conditional model and an underfitted baseline (unconditional model or early checkpoint)," it filters out "pseudo-memory" induced by the inherent low variance of the data manifold. This approach isolates **overfitting-driven memorized regions**, improving the localization IoU from 0.75 (BE) to approximately 0.92 on ground-truth memorization masks for Stable Diffusion.

## Background & Motivation

**Background**: Diffusion models (DDPM, Stable Diffusion, etc.) have been proven to "reproduce" training samples, raising privacy and copyright concerns. Current research on detecting/localizing memorization follows two main paths: (1) Global detection, such as the score-difference metric $\|s_\theta(x_t,c) - s_\theta(x_t)\|_2$ proposed by Wen et al., which assigns a scalar score to an image based on the intuition that conditional branches exhibit abnormal dependence on text; (2) Geometric perspectives, where Ross et al. use low Local Intrinsic Dimensionality (LID) to characterize memory, and Jeon et al. explain score-difference as the "sharpness gap" between conditional and unconditional models.

**Limitations of Prior Work**: Existing methods either provide only global scalars or require specific internal model signals. Regarding spatial localization, Bright Ending (BE) by Chen et al. utilizes cross-attention in the final steps for spatial masking but is highly model-specific and often triggers in non-memorized regions, leading to high false positives in complex foregrounds. Mechanistically, there is no clear explanation for "why the sharpness gap itself is a critical signal" or "why the unconditional model serves as an appropriate reference."

**Key Challenge**: Low LID or high sharpness only indicates that "several dimensions have collapsed" but does not specify "which pixels have collapsed." Even if coordinate-wise curvature could be extracted using the Hessian, high curvature might stem from the inherent low-variance structure of the data itself (e.g., a solid black background specified by a prompt). Misidentifying this Data-Driven Memory (DD-Mem) as actual memorization is a misjudgment.

**Goal**: (1) Provide a model-agnostic, geometrically interpretable **spatial localization** metric that precisely identifies "which pixels in an image are copied from the training set"; (2) Explain why the classic score-difference trick is effective.

**Key Insight**: Local memorization is redefined as **coordinate-wise variance collapse**. While an LID of 64 spread across an entire image represents a legitimate conceptual variation, concentrating it into an $8 \times 8$ patch constitutes template verbatim. Using the Tweedie relationship (Proposition 4.1), the conditional covariance of $x_0$ is proportional to $\sigma_t^4 \nabla^2_{x_t}\log p(x_t) + \sigma_t^2 I$. Thus, "low variance at coordinate $i$" is equivalent to "large $-(\nabla^2_{x_t}\log p)_{ii}$," transforming the localization problem into measuring the **diagonal Hessian**.

**Core Idea**: Subtract the inherent curvature of the data manifold using the "conditional model coordinate curvature minus an underfitted baseline coordinate curvature" to retain only overfitting-driven memory. It is further proven that the squared score-difference approximates this curvature difference in the form of Fisher information as $t \to 0$, providing a geometric justification for the existing metric.

## Method

### Overall Architecture
Input: Stable Diffusion checkpoint $\theta$, target prompt $c$, and a noisy sample $x_t$ near the end of the sampling trajectory ($t \approx 0$, typically the last DDIM step).
Output: A 2D spatial memory heatmap of the same size as the generated image (summed across channels), where bright regions indicate overfitted memory.
Process:

1. Perform conditional sampling using DDIM (50 steps) with CFG=7.5, retaining $x_t$ near the end.
2. Select an underfitted baseline: either the unconditional branch $s_\theta(x_t, \emptyset)$ of the same model (reusing the term from CFG for zero extra inference) or an earlier version of the same architecture $\tilde\theta$ (e.g., using SD v1.1 as a baseline for v1.4).
3. Compute the coordinate-wise curvature difference $\Delta h_\emptyset$ or $\Delta h_{\tilde\theta}$, or their efficient score-difference proxies $\Delta s_\emptyset$ / $\Delta s_{\tilde\theta}$.
4. Sum across channel dimensions, normalize to $[0, 1]$, and binarize by thresholding to obtain the memorization mask.

### Key Designs

1.  **Coordinate-wise Curvature Difference to Remove Data Manifold Pseudo-signals**:
    - **Function**: Subtract the data-determined components from the large values of $-\text{diag}(H_\theta(x_t,c))$, leaving only the overfitting components.
    - **Mechanism**: Define $\Delta h_\emptyset^t := \text{diag}(-H_\theta(x_t,c)) - \text{diag}(-H_\theta(x_t))$ (Eq. 1), and $\Delta h_{\tilde\theta}^t := \text{diag}(-H_\theta(x_t,c)) - \text{diag}(-H_{\tilde\theta}(x_t,c))$ (Eq. 2) using an early checkpoint. The unconditional model is interpreted as an underfitted reference that fits the entire data distribution simultaneously and loosely, thus preserving the inherent curvature of the data manifold. The conditional model, fitting only a single prompt, further pushes the curvature of overfitted pixels higher; the difference represents the contribution of overfitting.
    - **Design Motivation**: Counterexamples in Figure 3 show that relying solely on $\text{diag}(-H_\theta(x_t,c))$ causes false positives in solid color regions like "a black background" or in non-memorized samples, as these curvatures are constrained by prompt semantics rather than training set copying. Subtracting the unconditional branch flattens these regions, highlighting the memorized areas by contrast.

2.  **Hutchinson Estimator for Diagonal Hessian Approximation**:
    - **Function**: Estimate $\text{diag}(H)$ in a $4096+$-dimensional pixel space without explicitly constructing the Hessian.
    - **Mechanism**: Use a random Rademacher vector $v$ and automatic differentiation to compute the Hessian-vector product (HVP) $Hv = \nabla_x (s_\theta(x,c)^\top v)$. Then, $\mathbb{E}[v \odot (Hv)] = \text{diag}(H)$ (Hutchinson 1989). The paper uses $K=16$ samples by default and notes that even $K=1$ remains competitive. Curvature difference is computed by running HVP for both Hessians without storing them.
    - **Design Motivation**: Directly computing the full Hessian is infeasible for latent spaces where $d \sim 10^5$ like SD. Hutchinson reduces the cost to a few backpropagations, making the "geometric perspective" computationally viable.

3.  **Score-difference Proxy: Geometric Interpretation via Fisher Information**:
    - **Function**: Approximate coordinate curvature difference with a single forward pass and zero Hessian computation, explaining the efficacy of Wen et al.’s (2024) global detection metric.
    - **Mechanism**: Proposition 4.2 provides the Fisher Information Identity $\mathcal{I}(x) = \mathbb{E}_{c \sim p(c|x)}[-\nabla_x^2 \log p(c|x)]$. Taking the diagonal yields $\mathbb{E}_c[\text{diag}(-\nabla_x^2 \log p(x|c) + \nabla_x^2 \log p(x))] = \mathbb{E}_c[(\nabla_x \log p(x|c) - \nabla_x \log p(x))^{\odot 2}]$. Thus, $\Delta s_\emptyset^t := (s_\theta(x_t,c) - s_\theta(x_t))^{\odot 2}$ (Eq. 5) and $\Delta s_{\tilde\theta}^t := (s_\theta(x_t,c) - s_{\tilde\theta}(x_t,c))^{\odot 2}$ (Eq. 6) are defined as proxies for $\Delta h$. As $t \to 0$, $x_t$ almost determines $c$, making the approximation error of replacing the expectation with the generated $c$ very small.
    - **Design Motivation**: Engineering-wise, this removes the Hessian, making inference nearly as cheap as standard sampling. Theoretically, it reinterprets the global $\|s_\theta(x,c) - s_\theta(x)\|_2$ as a spatial sum of coordinate-wise curvature differences. The core is not "conditional signal strength" but the subtraction of an underfitted baseline, which filters out data complexity.

### Loss & Training
This work does not introduce new training. It performs inference and gradient calculations on existing SD checkpoints. All metrics are evaluated at the final DDIM step $t \approx 0$, summed across channels to produce a 2D spatial map, and normalized to $[0, 1]$ across the entire dataset.

## Key Experimental Results

### Main Results: Spatial Localization with Ground-truth Template Masks

Models: Stable Diffusion v1.4 / v2.1; baseline $\tilde\theta$ uses SD v1.1 / v2.0; ground-truth masks from Webster (2023) template-verbatim data. Metrics: IoU / Pixel ACC.

| Method | SD v1.4 TV IoU | SD v1.4 All IoU | SD v2.1 TV IoU | SD v2.1 TV+Non IoU |
|------|----------------|------------------|----------------|---------------------|
| All-ones (Trivial) | 0.560 | 0.522 | 0.649 | 0.325 |
| BE (Chen et al., 2025) | 0.751 | 0.564 | 0.933 | 0.956 |
| $\text{diag}(-H_\theta(x_t,c))$ (Raw Curv) | 0.586 | 0.522 | 0.649 | 0.500 |
| $\Delta h_\emptyset$ (Ours, Eq 1) | **0.899** | **0.953** | 0.943 | 0.866 |
| $\Delta s_\emptyset$ (Ours Proxy, Eq 5) | 0.830 | 0.918 | 0.785 | 0.794 |
| $\Delta h_{\tilde\theta}$ (Ours, Eq 2) | **0.921** | 0.867 | **0.947** | 0.828 |
| $\Delta s_{\tilde\theta}$ (Ours Proxy, Eq 6) | 0.863 | 0.654 | 0.920 | 0.844 |

The curvature-difference method improves IoU on SD v1.4 TV from BE's 0.751 to approximately 0.92. On SD v1.4 "All" (global memory + non-memory), it reaches 0.953. For SD v2.1, BE performs similarly to this method because ~85% of memory prompts are "Shaw Floors" (solid floor backgrounds); BE's preference for simple backgrounds in cross-attention happens to align here. However, qualitative examples show BE still frequently false-triggers in complex scenes.

### Detection Experiment (Global Detection via Spatial Mean Aggregation)

| Method | SD v1.4 AUC / TPR@1%FPR | SD v2.1 AUC / TPR@1%FPR |
|------|--------------------------|--------------------------|
| $\mathbb{E}[\text{BE-attention}]$ | 0.886 / 0.390 | 0.945 / 0.877 |
| $\mathbb{E}[\text{diag}(-H_\theta(x_t,c))]$ | 0.861 / 0.082 | 0.775 / 0.000 |
| $\mathbb{E}[\Delta h_\emptyset]$ | 0.997 / 0.982 | 0.995 / 0.950 |
| $\mathbb{E}[\Delta h_{\tilde\theta}]$ | 0.989 / 0.900 | 0.996 / 0.963 |
| $\mathbb{E}[\Delta s_\emptyset]$ | 0.997 / 0.982 | 0.997 / 0.968 |
| $\mathbb{E}[\Delta s_{\tilde\theta}]$ | **0.998 / 0.988** | 0.993 / 0.968 |

After aggregation, the four differential metrics achieve AUCs above 0.99 and TPR@1%FPR between 0.97–0.99, significantly outperforming BE-attention and raw curvature.

### Key Findings
- "Subtracting an underfitted baseline" is the key, not "measuring sharpness" alone: The detection AUC of raw $-H_\theta(x_t,c)$ is only 0.86 / 0.77, but jumping to 0.99+ after subtracting the unconditional branch. This reinterprets Wen's metric—the subtraction filters data curvature, leaving only overfitting contributions.
- Score-difference proxies are computationally cheaper and perform equally well in global aggregation: Spatial averaging smooths out local score noise. In fine-grained localization, scores are noisier; $\Delta s_{\tilde\theta}$ IoU on SD v1.4 "All" drops to 0.654 but recovers to 0.820 with a $13\times13$ mean filter.
- Localization remains possible even when the baseline itself has memory (e.g., SD v2.0 vs v2.1), indicating that curvature increases during fine-tuning and sharpening is a continuous process.

## Highlights & Insights
- **Upgrade Wen’s score-difference from intuition to Fisher Identity**: The paper proves that the $\ell_2$ squared difference is an unbiased proxy for "diagonal curvature difference," clarifying for the first time why an unconditional baseline is necessary.
- **Refining global geometry (LID) to coordinate-wise analysis**: Two samples with identical global LID can represent a conceptual change or template verbatim. Figure 2 uses a 4D linear Gaussian toy model to demonstrate how identical global dimensions can mask vastly different coordinate variance structures.
- **Hutchinson + Score-proxy Dual Track**: Use Hessians for fine localization and score-difference for global detection. The score proxy adds almost zero inference cost, making it highly deployment-friendly.

## Limitations & Future Work
- The Hessian approach is computationally more expensive than BE, though $K=1$ Hutchinson is competitive, and the score-difference proxy avoids Hessians entirely.
- The method is explicitly designed to capture verbatim/local memory (coordinate variance collapse). It is naturally insensitive to concept-level memory (celebrities, styles) where degrees of freedom are spread globally.
- Baseline selection impacts $\Delta h_{\tilde\theta}$; early checkpoints are required. For private models without early versions, $\Delta h_\emptyset$ must be used.
- Evaluation relies on SD-only ground-truth masks. Transferability to modern DiTs or video diffusion requires new benchmarks.

## Related Work & Insights
- **vs Bright Ending (Chen et al., 2025)**: BE is model-specific and prone to false positives in complex foregrounds. Ours uses geometric signals (Hessian/score), is model-agnostic, and improves IoU from 0.751 to 0.92 on SD v1.4 TV.
- **vs Wen et al., 2024 (score-difference)**: Wen uses it as "conditional dependency strength" for global detection. Ours proves it is a Fisher proxy for coordinate curvature difference, explaining the need for subtraction and enabling spatial localization.
- **vs Ross et al., 2025 (LID/MMH)**: MMH distinguishes OD-Mem from DD-Mem. Ours identifies OD-Mem as the alerted memory and filters DD-Mem, extending MMH from global LID to coordinate curvature.
- **vs Jeon et al., 2025 (sharpness)**: Jeon explains score-diff as a sharpness gap. Ours identifies the root cause—the unconditional model is an underfitted baseline.

## Rating
- Novelty: ⭐⭐⭐⭐ First to characterize local memory as coordinate-wise variance collapse and reinterpret Wen's metric via Fisher curvature difference.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual tasks (localization + detection) on SD v1.4/v2.1 with ground-truth masks, though limited to the SD family.
- Writing Quality: ⭐⭐⭐⭐ Clean progression through propositions, intuition, and counterexamples. The toy model in Figure 2 is particularly elegant.
- Value: ⭐⭐⭐⭐ A directly applicable tool for diffusion model privacy/copyright auditing with open-source code and low engineering barriers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[ICML 2026\] Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)
- [\[CVPR 2026\] Attention, May I Have Your Decision? Localizing Generative Choices in Diffusion Models](../../CVPR2026/image_generation/attention_may_i_have_your_decision_localizing_generative_choices_in_diffusion_mo.md)
- [\[ICML 2026\] WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation](wise_a_world_knowledge-informed_semantic_evaluation_for_text-to-image_generation.md)
- [\[ICML 2026\] STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack](stare_step-wise_temporal_alignment_and_red-teaming_engine_for_multi-modal_toxici.md)

</div>

<!-- RELATED:END -->
