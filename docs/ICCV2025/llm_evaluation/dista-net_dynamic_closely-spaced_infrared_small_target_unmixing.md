---
title: >-
  [Paper Note] DISTA-Net: Dynamic Closely-Spaced Infrared Small Target Unmixing
description: >-
  [ICCV 2025][LLM Evaluation][Infrared small target] DISTA-Net proposes a dynamic deep unfolding network that replaces the static nonlinear transform and threshold parameters in ISTA-based sparse reconstruction with input-…
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "Infrared small target"
  - "closely-spaced target unmixing"
  - "deep unfolding network"
  - "sparse reconstruction"
  - "sub-pixel localization"
date: 2026-05-08
content_hash: 2f8e36053393a483
---

# DISTA-Net: Dynamic Closely-Spaced Infrared Small Target Unmixing

**Conference**: ICCV 2025
**arXiv**: [2505.19148](https://arxiv.org/abs/2505.19148)  
**Code**: [https://github.com/GrokCV/GrokCSO](https://github.com/GrokCV/GrokCSO)  
**Area**: LLM Evaluation
**Keywords**: Infrared small target, closely-spaced target unmixing, deep unfolding network, sparse reconstruction, sub-pixel localization

## TL;DR
DISTA-Net proposes a dynamic deep unfolding network that replaces the static nonlinear transform and threshold parameters in ISTA-based sparse reconstruction with input-adaptive counterparts, constituting the first deep learning method for closely-spaced infrared small target (CSIST) unmixing. The work also establishes the first open-source ecosystem encompassing a dataset, evaluation metrics, and a toolkit.

## Background & Motivation
Infrared imaging plays a critical role in long-range detection and surveillance. However, at extended ranges, target radiation signals are weak, and when multiple targets are spatially clustered, their signals overlap into blurred blobs due to optical diffraction spreading (PSF), making them indistinguishable to the human eye.

**Limitations of Prior Work**:
1. Traditional optimization methods (e.g., ISTA with $\ell_1$ regularization) are highly sensitive to hyperparameter tuning and perform inconsistently across varying target counts and positions.
2. Deep learning has advanced infrared small target detection, yet remains entirely unexplored for closely-spaced infrared small target (CSIST) unmixing.
3. Generic image super-resolution is ill-suited — CSIST unmixing requires precise estimation of target count, sub-pixel positions, and radiant intensity, rather than pixel-level sharpness enhancement.
4. No standardized datasets, evaluation metrics, or open-source implementations exist.

**Core Idea**: Unroll traditional ISTA sparse reconstruction into a deep network, with the key innovation of making the transform and thresholding "dynamic" — i.e., adaptively conditioned on the input.

## Method

### Overall Architecture
DISTA-Net consists of $N$ cascaded stages, each comprising: (1) a gradient descent step computing the residual $\mathbf{r}^{(k)}$; (2) a dual-branch dynamic transform module $\mathcal{F}_d^{(k)}$ for feature extraction; (3) a dynamic thresholding module $\Theta_d^{(k)}$ for feature refinement; and (4) an inverse transform module $\tilde{\mathcal{F}}^{(k)}$ for signal reconstruction.

### Key Designs

1. **CSIST Imaging and Sparse Reconstruction Model**:

    - Function: Formalizes the unmixing problem as a sparse optimization task.
    - Mechanism: Targets are approximated as point sources whose diffusion is modeled via a Gaussian PSF. Each pixel is subdivided into an $n \times n$ sub-pixel grid, yielding $L = UVn^2$ candidate target locations. The optimization problem is formulated as $\min_{\tilde{\mathbf{s}}} \|\mathbf{z} - \mathbf{G}(\Omega)\tilde{\mathbf{s}}\|_2^2 + \lambda\|\tilde{\mathbf{s}}\|_1$.
    - The nonzero elements of the solution directly encode target count, intensity, and sub-pixel coordinates.

2. **Dynamic Transform Module**:

    - Function: Replaces the static nonlinear transform in ISTA-Net.
    - Mechanism: Dual-branch structure:
        - Main branch: Conv-ReLU-Conv applied to $\mathbf{r}^{(k)}$.
        - Auxiliary branch: Dynamic convolutional weights $W = f(\tilde{\mathbf{s}}^{(k-1)})$ generated via a fully connected network from $\tilde{\mathbf{s}}^{(k-1)}$, applied to $\mathbf{r}^{(k)}$.
        - Fusion: $\mathcal{F}_d^{(k)} = \alpha \cdot A(\text{ReLU}(B(\mathbf{r}^{(k)}))) + (1-\alpha) \cdot \text{sigmoid}(w_r)$
    - Design Motivation: ISTA-Net weights are fixed after training and cannot adapt to varying inputs. Dynamic convolutional kernels allow the transform to adjust based on the current input.

3. **Dynamic Soft-Thresholding Module**:

    - Function: Adaptively generates the threshold parameter $\theta_d$.
    - Mechanism:
        - Dual convolutional layers extract multi-scale features $\tilde{U}_1, \tilde{U}_2$.
        - Parallel average and max pooling capture spatial context.
        - A Conv+sigmoid module generates spatially selective masks.
        - The final threshold is $\theta_d = C(\sum_{i=1}^{N} (\widetilde{SA})_i \cdot \tilde{U}_i)$.
    - Design Motivation: Fixed thresholds are insufficient to handle densely overlapping targets and varying spatial contexts.

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{\text{discrepancy}} + \gamma \mathcal{L}_{\text{constraint}}$$

- $\mathcal{L}_{\text{discrepancy}}$: MSE between the reconstruction and ground truth.
- $\mathcal{L}_{\text{constraint}}$: Multi-stage identity constraint $\tilde{\mathcal{F}}^{(k)} \circ \mathcal{F}_d^{(k)} \approx \mathbf{I}$, weighted by $\gamma=0.01$.
- Linear initialization: $\tilde{s}^{(0)} = Q_{\text{init}} \mathbf{z}$.

Hyperparameters: $c=3$ (default grid ratio), batch size 64, $N=6$ stages, $(1-\alpha)=0.3$.

## Key Experimental Results

### Main Results

| Method Type | Method | Params | CSO-mAP | AP-20 | AP-25 | PSNR | SSIM |
|------------|--------|--------|---------|-------|-------|------|------|
| Traditional Optimization | ISTA | - | 7.46 | 9.46 | 25.14 | - | - |
| Super-Resolution | SRFBN | 0.373M | 46.05 | 83.72 | 94.95 | 34.29 | 0.9815 |
| Super-Resolution | SAN | 4.442M | 45.95 | 84.32 | 96.57 | 37.18 | 0.9848 |
| Deep Unfolding | ISTA-Net | 0.171M | 45.16 | 82.58 | 94.53 | 35.67 | 0.9862 |
| Deep Unfolding | ISTA-Net+ | 0.337M | 46.06 | 84.46 | 96.17 | 38.50 | 0.9887 |
| **Deep Unfolding** | **DISTA-Net** | **2.179M** | **46.74** | **86.18** | **97.14** | 38.38 | **0.9887** |

### Ablation Study

| Configuration | CSO-mAP | AP-20 | AP-25 | Note |
|--------------|---------|-------|-------|------|
| ISTA-Net (baseline) | 45.16 | 82.58 | 94.53 | Static parameters |
| DISTA-Net w/o DT | 46.32 | 86.18 | 97.50 | Without dynamic transform |
| DISTA-Net w/o Thres. | 46.17 | 84.67 | 95.79 | Without dynamic thresholding (largest impact) |
| **DISTA-Net (full)** | **46.74** | **86.18** | **97.14** | All components |

Ablation over sampling grid ratios ($c=5, c=7$): DISTA-Net consistently outperforms all configurations, with larger gains observed as the grid ratio increases.

### Key Findings
- DISTA-Net achieves AP-20 of 86.18% and AP-25 of 97.14%, significantly outperforming all baselines.
- Dynamic soft-thresholding is the most critical component; removing it reduces CSO-mAP from 46.74 to 46.17.
- Traditional ISTA achieves only 7.46 CSO-mAP; deep learning methods yield an order-of-magnitude improvement.
- SR+detector pipeline validation: DISTA-Net + YOLOv11 (47.82) > SRFBN + YOLOv11 (45.74).
- Super-resolution methods can also handle the task, but DISTA-Net holds a clear advantage in localization precision.

## Highlights & Insights
- The first work to apply deep learning to closely-spaced infrared small target unmixing, pioneering a complete research ecosystem.
- CSIST-100K dataset (100K samples), the CSO-mAP evaluation metric, and the GrokCSO toolkit form a cohesive triad.
- The dynamic unfolding concept is concise and effective: conditioning ISTA-Net's static weights on the input aligns naturally with physical intuition.
- The CSO-mAP metric is well-designed, evaluating localization precision across multiple sub-pixel distance thresholds (0.05–0.25 pixels).

## Limitations & Future Work
- Computational cost grows substantially with grid ratio ($c=3$: 35.1G FLOPs; $c=7$: 142.3G FLOPs).
- Validation is conducted solely on synthetic data; generalization to real-world infrared scenarios remains to be examined.
- The point-source assumption (Gaussian PSF) may not hold for targets with more complex shapes.
- The current method addresses scenes with 1–5 closely-spaced targets; extension to larger-scale target groups has not been explored.

## Related Work & Insights
- The progression from ISTA-Net to DISTA-Net is clear: from static unfolding to dynamic unfolding.
- The idea of generating dynamic convolutional weights conditioned on input can be generalized to other deep unfolding networks.
- The CSO-mAP design rationale (multi-threshold sub-pixel evaluation) offers a valuable reference for other sub-pixel-level tasks.
- Insight: Dynamically adapting parameters based on input within deep unfolding networks is a direction worthy of broader exploration.

## Rating
- Novelty: ⭐⭐⭐⭐ First deep learning method for CSIST unmixing; dynamic unfolding design is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive baseline comparisons and ablations, though real-data validation is absent.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, with a complete derivation from the physical model to the deep network.
- Value: ⭐⭐⭐⭐ Establishes an entirely new research ecosystem with significant implications for the infrared target detection community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] InterSyn: Interleaved Learning for Dynamic Motion Synthesis in the Wild](intersyn_interleaved_learning_for_dynamic_motion_synthesis_in_the_wild.md)
- [\[NeurIPS 2025\] Small Language Models as Compiler Experts: Auto-Parallelization for Heterogeneous Systems](../../NeurIPS2025/llm_evaluation/small_language_models_as_compiler_experts_auto-parallelization_for_heterogeneous.md)
- [\[NeurIPS 2025\] MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments](../../NeurIPS2025/llm_evaluation/memtrack_evaluating_long-term_memory_and_state_tracking_in_multi-platform_dynami.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](../../ACL2026/llm_evaluation/retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2026\] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](../../ACL2026/llm_evaluation/dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)

</div>

<!-- RELATED:END -->
