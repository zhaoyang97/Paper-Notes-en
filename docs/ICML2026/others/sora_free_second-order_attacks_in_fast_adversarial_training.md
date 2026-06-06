---
title: >-
  [Paper Note] SORA: Free Second-Order Attacks in Fast Adversarial Training
description: >-
  [ICML 2026][Fast Adversarial Training] This paper re-examines Catastrophic Overfitting (CO) in single-step adversarial training from a second-order perspective. It proposes PertAlign…
tags:
  - "ICML 2026"
  - "Fast Adversarial Training"
  - "Catastrophic Overfitting"
  - "Second-Order Optimization"
  - "Adaptive Step Size"
  - "Robustness"
date: 2026-05-08
content_hash: ae1d79d7fc5a17fe
---

# SORA: Free Second-Order Attacks in Fast Adversarial Training

**Conference**: ICML 2026  
**arXiv**: [2606.00738](https://arxiv.org/abs/2606.00738)  
**Code**: https://github.com/SecondOrderAT/SORA  
**Area**: AI Security / Adversarial Training  
**Keywords**: Fast Adversarial Training, Catastrophic Overfitting, Second-Order Optimization, Adaptive Step Size, Robustness  

## TL;DR
This paper re-examines Catastrophic Overfitting (CO) in single-step adversarial training from a second-order perspective. It proposes PertAlign, a zero-cost curvature metric for early warning of CO, and derives SORA: an adaptive fast adversarial training algorithm that estimates the Hessian "for free" using backpropagated gradients from the previous step and samples optimal step sizes per-channel. SORA consistently avoids CO across 6 datasets and 4 architectures using a single set of hyperparameters, achieving a new state-of-the-art trade-off between robustness and clean accuracy for single-step AT.

## Background & Motivation
**Background**: Adversarial Training (AT) is the most effective defense against adversarial examples, but the inner maximization of multi-step PGD-AT requires multiple backpropagations, making it extremely costly for large datasets and deep networks. Wong et al. (2020) proposed FGSM-RS with random starts, compressing the inner loop to a single step, making fast adversarial training (FAT) a viable solution.

**Limitations of Prior Work**: Single-step AT generally suffers from Catastrophic Overfitting (CO)—after training for a certain number of batches, robust accuracy against PGD suddenly collapses to near 0%, while FGSM accuracy increases, sometimes even exceeding clean accuracy. Existing mitigation methods (GradAlign, NuAT, N-FGSM, ATAS, ELLE, etc.) either require expensive additional backpropagations/regularization terms or have hyperparameters tightly bound to specific datasets/architectures, often failing in challenging scenarios like PathMNIST and TissueMNIST, which contradicts the "low-cost" objective of FAT.

**Key Challenge**: FGSM repeatedly attacks the model using perturbations of fixed size and direction, causing the model to minimize loss only along a narrow path within the $\epsilon$-ball. The loss surface becomes highly non-linear along this line, a phenomenon the authors define as **Epsilon Overfitting (EO)**—the geometric root of CO. To break EO, diversity in perturbation magnitude is required; to make this diversity effective, knowledge of the local curvature of the loss surface is needed; however, explicit Hessian computation would destroy the cost advantage of single-step AT.

**Goal**: (1) Provide a geometric characterization of CO and prove that perturbation magnitude diversity is key to resolving EO; (2) Design a "free" curvature metric within single-step AT to both warn of CO and guide the attack; (3) Develop a fast AT method that is robust to datasets and architectures without extra backpropagations or parameter tuning.

**Key Insight**: The authors notice that when backpropagation computes derivatives with respect to parameters, the first layer of the chain rule is exactly $\nabla_x \mathcal{L}(f_\theta(x+\delta), y)$. By simply "extending" this gradient to the input layer and computing its cosine similarity with $\nabla_x \mathcal{L}(f_\theta(x'), y)$ (already calculated during adversarial example generation), one obtains a curvature proxy with zero extra overhead.

**Core Idea**: Treat inner maximization as a second-order problem. Use the gradients already computed in the previous batch to approximate the Hessian-vector product, deriving an analytical optimal step size $\alpha^*$. Sample step sizes uniformly within $[0, \alpha^*]$ for each channel to simultaneously break EO and CO.

## Method

### Overall Architecture
SORA replaces the "FGSM + fixed step size $\alpha$" in single-step AT with "FGSM direction + adaptive random step size." The training workflow for each batch is: (1) Add a random start $\eta \sim \mathcal{U}(-\epsilon, \epsilon)^d$; (2) Compute $g = \nabla_x \mathcal{L}(f_\theta(x+\eta), y)$; (3) Independently sample step sizes $\alpha_i$ from $\mathcal{U}(0, \alpha^*)$ for each pixel channel, construct $x' = x + \eta + \alpha_i \odot \text{sign}(g)$ and clip to $[0,1]$; (4) Backpropagate to update $\theta$, simultaneously retaining the gradient $g'$ extended to the input layer; (5) Update the EMA linearity coefficient $v$ using the dot product of $g$ and $g'$ along the sign direction $p = \text{sign}(g)$, and derive a new $\alpha^*$ for the next batch. Relative to FGSM-RS, this process involves no extra backpropagation, with negligible time and memory overhead.

### Key Designs

1.  **PertAlign: A Zero-Cost Second-Order CO Warning Metric**:
    *   **Function**: Uses a cosine similarity to measure the non-linearity of the loss surface along the attack direction, providing a "pre-CO" signal before PGD accuracy collapses.
    *   **Mechanism**: Define $\text{PertAlign} = \cos\!\big(\nabla_x \mathcal{L}(f_\theta(x'), y),\ \nabla_x \mathcal{L}(f_\theta(x'+\delta), y)\big)$, where $x' = x + \eta$ and $\delta = \alpha v$. The paper proves $1 - \text{PertAlign} \approx \tfrac{\alpha^2}{2}\|h_{\perp g}\|^2$, where $h = Hv/\|g\|$. Thus, PertAlign directly captures the component of the Hessian-vector product orthogonal to the gradient. When CO occurs, this component explodes, and PertAlign drops sharply from near 1 toward 0. One gradient comes from generating the adversarial sample, and the other from the standard backpropagation extended by one layer, thus requiring **zero additional forward or backward passes**.
    *   **Design Motivation**: Existing metrics like GradAlign and ELLE require an extra backpropagation, violating FAT cost constraints. Furthermore, PertAlign triggers earlier than metrics like GradAlign / AAE share / TRADES-KL (Fig 3 shows PertAlign dropping at batch 3775, whereas FGSM/PGD accuracy diverges visibly only at batch 3825).

2.  **Second-Order Optimal Step Size $\alpha^*$ + EMA Linearity Coefficient**:
    *   **Function**: Provides a theoretically optimal attack step size for the next batch without explicitly calculating the Hessian.
    *   **Mechanism**: Perform a second-order expansion of the loss along direction $v = \alpha\, \text{sign}(g)$ as $\mathcal{L}(x+v) \approx \mathcal{L}(x) + v^T g + \tfrac{1}{2}v^T H v$. Differentiating with respect to $\alpha$ yields the optimal step size $\alpha^* = \min\!\big(\alpha_{\max},\ \alpha_0 / (1 - p^T g'/\|g\|_1)\big)$, where $g' = g + \alpha H p$ is obtained for free from the input-layer gradient of the previous batch. SORA maintains an EMA $v \leftarrow (1-\beta) v + \beta \cdot p^T g'/\|g\|_1$ to stabilize the estimation of $\alpha^*$. Reusing information across batches exploits the assumption that weights change only slightly per step and batches come from the same distribution, making second-order information "essentially for free."
    *   **Design Motivation**: Explicitly calculating $H$ is unacceptable for large models. The method must be more robust than empirically tuned fixed $\alpha$, hence the introduction of an analytical approximation that reuses existing gradients.

3.  **Channel-wise Randomized Step Size to Break Epsilon Overfitting**:
    *   **Function**: Adds a layer of perturbation diversity on top of $\alpha^*$ to prevent the model from overfitting to a specific $\epsilon$ value.
    *   **Mechanism**: Instead of using $\alpha^*$ directly, sample $\alpha_i \sim \mathcal{U}(0, \alpha^*)$ independently for each channel of each pixel, then set $x' = x + \eta + \alpha_i \odot \text{sign}(g)$. This ensures samples within the same batch see a wide distribution of perturbation magnitudes, covering the $\ell_\infty$ ball more uniformly and removing the "EO signature" where FGSM accuracy spikes at certain $\epsilon$ values.
    *   **Design Motivation**: Sec. 3 identifies EO as the root cause of CO—fixed large perturbations flatten the loss only along narrow paths. Randomized step sizes $\mathcal{U}(0, \alpha^*)$ combined with adaptive $\alpha^*$ address both the symptoms (CO) and the root cause (EO).

### Loss & Training
Standard Cross-Entropy + SGD (momentum 0.9, weight decay $5\times 10^{-4}$) is used throughout with a single set of hyperparameters: $\alpha_0 = 0.02$, $\beta = 0.05$, $\alpha_{\max} = 2\epsilon$, $\epsilon = 8/255$. The EMA coefficient $v$ is initialized to 0.99. When the model reaches the CO threshold, PertAlign drops, $v$ decreases, and $\alpha^*$ is automatically scaled up via $\alpha_0/(1-v)$, generating stronger adversarial samples to recover robustness. Under normal conditions, $\alpha^*$ is clipped by $\alpha_{\max}$, maintaining costs similar to FGSM-RS.

## Key Experimental Results

### Main Results
The paper compares 15 baselines across 6 datasets (CIFAR-10/100, TinyImageNet, ImageNet-100, PathMNIST, TissueMNIST) and 4 architectures (ResNet, PreActResNet, WideResNet, SENet, ViT). Representative results for PreActResNet-18, $\epsilon = 8/255$, evaluated via AutoAttack:

| Dataset | Metric | SORA | N-FGSM | AAER | FGSM |
|--------|------|------|--------|------|------|
| PathMNIST | Clean | **84.88** | 74.86 | 81.43 | 36.54 |
| PathMNIST | AutoAttack | **35.54** | 1.90 | 1.89 | 0.42 |
| ImageNet-100 | Clean | **57.26** | 49.38 | 48.26 | 15.98 |
| ImageNet-100 | AutoAttack | **18.56** | 15.52 | 17.18 | 0.00 |

SORA is the only single-step method to successfully avoid CO and achieve the highest robust and clean accuracy across all 6 datasets and 4 architectures. On PathMNIST, it improves AutoAttack robustness from < 2% (all baselines) to 35.54%.

### Ablation Study
| Configuration | Clean | FGSM | PGD-10 | Description |
|------|-------|------|--------|------|
| SORA (full) | 84.69 | 57.51 | **45.56** | Complete Method |
| – Without Random Sampling | 85.67 | 58.89 | 45.35 | Remove channel-level randomization |
| – Clamping Step-Size | 86.82 | 52.02 | 34.08 | Clamp adaptive $\alpha^*$ to fixed value |
| – Without Optimal Step-Size | 89.93 | 28.30 | 17.23 | Degenerate to fixed step size FGSM-RS |

### Key Findings
- **Adaptive $\alpha^*$ is the performance anchor**: Removing it causes PGD-10 robustness to drop from 45.56% to 17.23%, while clean accuracy increases to 89.93%—a classic EO sign where clean accuracy appears fine but robustness is broken.
- **PertAlign provides earlier CO warning than GradAlign / AAE / TRADES-KL / ELLE**: On CIFAR-10, it signals approximately 50 batches earlier, sufficient for SORA to automatically increase $\alpha^*$ and stabilize training.
- **Hyperparameter stability across domains**: All experiments use the same $(\alpha_0, \beta, \alpha_{\max})$. In contrast, methods like GradAlign, N-FGSM, and ATAS failed to find configurations on PathMNIST that didn't either cause CO or drop performance on CIFAR.
- **Negligible cost**: PertAlign and $\alpha^*$ reuse existing gradients. Training time for 30 epochs on an RTX 4090 is comparable to FGSM-RS, with almost no increase in memory usage.

## Highlights & Insights
- **Solving inner maximization as a second-order problem at first-order cost**: Reusing the "extended" input gradient from the chain rule as a proxy for the Hessian-vector product is an elegant combination of engineering and theory, applicable to any training workflow requiring curvature (e.g., sharpness-aware minimization, second-order meta-learning).
- **CO reinterpreted as a symptom of EO rather than the root cause**: Diversity in perturbation magnitude is more critical than direction. This reshapes the traditional explanation of why random starts work—not just by stabilizing direction, but by indirectly breaking local overfitting at a fixed $\epsilon$.
- **PertAlign as a universal diagnostic tool**: Since it only relies on two gradients already present in single-step attacks, it can be added to any fast AT method for "early CO warning" without modifying the core algorithm.
- **"Delayed-by-one" second-order estimation**: Using $g'$ from the previous batch for the current $\alpha^*$ is experimental proof that this "gradient time-shifting" is stable enough. This logic is valuable for large-batch optimization and distributed asynchronous training.
- **Per-channel vs. Per-sample step size**: The choice of per-channel randomization increases $\ell_\infty$ ball coverage by ensuring different channels of the same image have different perturbations. This "fine-grained diversity" is applicable to patch attacks, style transfer, and other tasks needing to break singular perturbation patterns.

## Limitations & Future Work
- The second-order derivation assumes weights and batches change minimally between steps. This approximation might fail with large learning rates or very small batches, which the EMA smoothing aims to address.
- Experiments focus exclusively on $\ell_\infty$ threat models and image classification; transferability to $\ell_2$, patch attacks, NLP, detection, or segmentation remains unverified.
- PertAlign provides a "post-hoc" signal; while earlier than other metrics, it is not a "prediction" in a true sense. Stronger metrics directly characterizing Hessian spectral properties theoretically exist.
- A gap remains between SORA and multi-step PGD-AT robustness. The paper treats multi-step methods as an "upper bound baseline" rather than attempting to bridge the gap entirely.

## Related Work & Insights
- **vs GradAlign (Andriushchenko & Flammarion, 2020)**: Both measure local linearity, but GradAlign requires an extra doubly-back-propagation as a regularization term, while PertAlign reuses gradients. SORA uses the metric to dynamically adjust step size rather than as a loss penalty.
- **vs N-FGSM (de Jorge et al., 2022)**: N-FGSM increases random noise and removes clipping to mitigate CO. SORA uses an analytical optimal step size + channel-level randomization, providing a clearer theoretical foundation and significantly higher robustness on hard datasets like PathMNIST.
- **vs ATAS (Huang et al., 2022) / Zhao et al. (2025)**: Both are adaptive step size methods, but ATAS uses empirical gradient norm scaling. SORA is the first to explicitly model step size selection as second-order optimization with a closed-form solution.
- **vs Curvature regularization (Moosavi-Dezfooli et al., 2018; Ma et al., 2021)**: These add curvature penalties to the loss, usually in multi-step AT. SORA integrates curvature awareness into the attack step size selection, preserving the FAT cost budget.
- **vs Free AT (Shafahi et al., 2019)**: Free AT reuses adversarial gradients for parameter updates. SORA shares this "gradient reuse" philosophy but extends it to curvature estimation rather than just parameter updates.
- **vs ELLE (Rocamora et al., 2024)**: ELLE encourages local linearity via regularization. SORA does not force linearity but instead tracks current curvature, remaining robust even on architectures like ViT without prior landscape bias.
- **vs AAER (Lin et al., 2024)**: AAER identifies and suppresses "abnormal adversarial examples" via regularization. SORA uses adaptive $\alpha^*$ to ensure the attack reaches a true local maximum (NAE), preventing AAER at the source without extra regularization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] How Does Bayesian Sampling Help Membership Inference Attacks?](how_does_bayesian_sampling_help_membership_inference_attacks.md)
- [\[CVPR 2026\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](../../CVPR2026/others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)
- [\[ICLR 2026\] Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition](../../ICLR2026/others/training_deep_normalization-free_spiking_neural_networks_with_lateral_inhibition.md)
- [\[ICML 2026\] New Bounds for Kernel Sums via Fast Spherical Embeddings](new_bounds_for_kernel_sums_via_fast_spherical_embeddings.md)
- [\[AAAI 2026\] Higher-Order Responsibility](../../AAAI2026/others/higher-order_responsibility.md)

</div>

<!-- RELATED:END -->
