---
title: >-
  [Paper Note] SURGE: Surrogate Gradient Adaptation in Binary Neural Networks
description: >-
  [ICML 2026][Model Compression][BNN] SURGE attaches a "full-precision auxiliary branch" in parallel to each binarized layer. The forward output remains unchanged, but in the backward pass…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "BNN"
  - "STE"
  - "Gradient Mismatch"
  - "Dual-Path Compensation"
  - "Adaptive Gradient Scaling"
date: 2026-05-08
content_hash: dd70261c9f7aaa18
---

# SURGE: Surrogate Gradient Adaptation in Binary Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2605.10989](https://arxiv.org/abs/2605.10989)  
**Code**: Not yet released  
**Area**: Model Compression / Binary Neural Networks / Quantization-Aware Training  
**Keywords**: BNN, STE, Gradient Mismatch, Dual-Path Compensation, Adaptive Gradient Scaling

## TL;DR
SURGE attaches a "full-precision auxiliary branch" in parallel to each binarized layer. The forward output remains unchanged, but in the backward pass, an extra "non-STE truncated" higher-order gradient is backpropagated from the full-precision branch. AGS dynamically balances the contributions of both paths according to the gradient norm ratio, enabling BNNs to achieve 62.0% top-1 on ResNet-18/ImageNet—1.0% higher than ReCU and 3.9% higher than IR-Net.

## Background & Motivation

**Background**: Binary Neural Networks (BNNs) quantize weights and activations to $\{-1,+1\}$, theoretically achieving $32\times$ memory compression and $58\times$ inference acceleration, making them the most aggressive quantization scheme for edge deployment. Almost all BNNs rely on the Straight-Through Estimator (STE) for training: the forward uses $\text{sign}(\cdot)$, and the backward directly approximates $\frac{\partial\mathbf{B}_W}{\partial W}\approx 1$, $\frac{\partial\mathbf{B}_x}{\partial x}\approx\mathbb{1}_{\{|x|\le 1\}}$ as surrogate gradients.

**Limitations of Prior Work**: STE has two fundamental issues. First, the true gradient of sign is almost everywhere zero; using the identity function as a surrogate introduces systematic bias, known as "gradient mismatch." Second, activation gradients are hard-clipped to zero outside $[-1,1]$, discarding substantial information. Existing works (e.g., DSQ's sigmoid approximation, IR-Net's asymptotic sign, ReCU's feature distribution alignment) mostly rely on hand-crafted approximations, which cannot guarantee optimality.

**Key Challenge**: In BNN training, "strict binarization in the forward pass (for inference acceleration)" and "sufficiently rich gradients in the backward pass (for learnability)" are fundamentally at odds—if the forward uses sign, the backward can only use first-order identity surrogates.

**Goal**: 1) Inject "non-STE, low-bias" gradients into the main branch without altering the forward output; 2) Prevent imbalance in the magnitude of compensation gradients from disrupting main branch convergence; 3) Discard the auxiliary branch entirely during inference, incurring zero extra cost.

**Key Insight**: Since STE is a first-order approximation of sign, attach a "full-precision replica" to each layer to supplement the missing higher-order terms in STE with its true gradients. As the magnitudes of the two gradient paths are unknown, use norm-ratio adaptive scaling for dynamic balancing.

**Core Idea**: Use the "forward self-cancellation, backward open" detach trick so the full-precision auxiliary branch only participates in the backward pass. Then, use AGS to adaptively scale by $\frac{\|g_b\|_2}{\|g_a\|_2+\epsilon}$, refining the first-order STE surrogate into a mixed estimate closer to the true gradient.

## Method

### Overall Architecture
For each binarized linear operator (conv, linear, attention projection), SURGE attaches a full-precision replica (auxiliary branch) of identical size in parallel. In the forward pass, the $\text{detach}$ trick ensures the auxiliary branch cancels itself out, guaranteeing the output is strictly that of a pure BNN. In the backward pass, the auxiliary branch propagates gradients normally, the main branch uses STE, and both merge at the input. AGS dynamically computes the scaling factor $\lambda_{\text{AGS}}$ to balance the two contributions. After training, the auxiliary branch is discarded, and inference is standard BNN.

### Key Designs

1. **Dual-Path Gradient Compensator (DPGC)**:

    - **Function**: Provides each binarized layer with additional "non-STE truncated" higher-order gradient information without altering the forward output.
    - **Mechanism**: Define binarized forward $f_b(x;W_b)=Q_W(W_b)^\top Q_x(x)$, full-precision forward $f_a(x;W_a)=W_a^\top x$, and scaled $f_{ao}(x)=\lambda f_a(x)$. The output is written as $\text{output}=f_b(x;W_b)-f_{ao}(x;W_a)\downarrow+f_{ao}(x;W_a)$, where $\downarrow$ denotes stop-gradient. In the forward pass, the second and third terms numerically cancel, so the output equals $f_b$; in the backward pass, the detached term's gradient is truncated, leaving $f_b$ to use STE and $f_{ao}$ to use full-precision. Thus, $\frac{\partial\mathcal{L}}{\partial x}=g_b+\lambda g_a$, where $g_b$ is the STE first-order approximation and $g_a$ is the higher-order compensation from the auxiliary branch's true gradient.
    - **Design Motivation**: Traditional STE improvements (piecewise polynomial, SignSwish, etc.) merely swap in alternative functions, failing to address the root issue. DPGC's detach trick ingeniously allows "strictly binary output" and "full-precision signal in the backward pass" to coexist, and the auxiliary branch can be discarded at inference for zero extra cost.

2. **Adaptive Gradient Scaler (AGS)**:

    - **Function**: Dynamically balances the magnitudes of $g_b$ and $g_a$ to prevent the auxiliary gradient from overwhelming the main branch during training.
    - **Mechanism**: Set $\lambda$ as $\lambda_{\text{AGS}}=\eta\frac{\|g_b\|_2}{\|g_a\|_2+\epsilon}$, where $\eta$ is a base scaling factor and $\epsilon=10^{-8}$ prevents division by zero. The paper derives from a second-moment model that the optimal $\lambda^*=\frac{\langle\delta_b,\mu_a\rangle}{\|\mu_a\|_2^2+\text{tr}(\text{Var}(g_a))}$ ($\delta_b$ is the STE bias vector), and under the assumptions of stable alignment $\cos\theta$, relative bias ratio $\beta=\|\delta_b\|_2/\|\mu_b\|_2$, and noise ratio $\rho$, it follows that $\lambda^*\approx\eta\frac{\|\mu_b\|_2}{\|\mu_a\|_2}$. Using mini-batch estimates yields the practical formula above.
    - **Design Motivation**: Fixed $\lambda$ is either too large or too small: too large and the auxiliary path destabilizes the main branch; too small and compensation is ineffective. AGS ensures both paths remain comparable in magnitude, with STE dominating the optimization direction and the auxiliary path providing only "higher-order correction," theoretically equivalent to the optimal convex combination in the mean squared error sense.

3. **Training-Inference Symmetric Dual-Path Architecture**:

    - **Function**: Makes SURGE a universal plugin, applicable to CNNs (conv blocks) and Transformers (attention projection / FFN linear).
    - **Mechanism**: DPGC is architecture-agnostic and can be attached wherever there is a binarized linear operator; Figure 2 shows integration for both conv and transformer blocks. During training, three states coexist (main forward, main backward, auxiliary backward); during inference, all auxiliary $W_a$ are discarded.
    - **Design Motivation**: Previous BNN training tricks were often task/architecture-specific (e.g., ReActNet's RPReLU). SURGE modularizes "gradient compensation" as a layer-wise plugin with minimal migration cost.

### Loss & Training
End-to-end cross-entropy (classification) / detection loss (VOC) / NLU loss (GLUE) is used, with no extra training losses introduced. $\eta$ is one of the few hyperparameters to tune; inference incurs zero extra cost.

## Key Experimental Results

### Main Results
Covers four benchmarks: CIFAR-10, ImageNet-1K (ResNet-18/34, ReActNet), PASCAL VOC (Faster-RCNN + ResNet-18 backbone), GLUE (BERT-base).

| Network / Task | Method | W/A | Top-1 / mAP / Mean |
|----------------|--------|-----|--------------------|
| ResNet-18 / CIFAR-10 | ReCU | 1/1 | 92.8% |
| ResNet-18 / CIFAR-10 | **SURGE** | 1/1 | **93.1%** (+0.3) |
| ResNet-20 / CIFAR-10 | ReCU | 1/1 | 87.4% |
| ResNet-20 / CIFAR-10 | **SURGE** | 1/1 | **88.0%** (+0.6) |
| VGG-Small / CIFAR-10 | ReCU | 1/1 | 92.2% |
| VGG-Small / CIFAR-10 | **SURGE** | 1/1 | **92.5%** (+0.3) |
| ResNet-18 / ImageNet (one-stage) | IR-Net | 1/1 | 58.1% |
| ResNet-18 / ImageNet (one-stage) | BONN | 1/1 | 59.3% |
| ResNet-18 / ImageNet (one-stage) | ReCU | 1/1 | ~61% |
| ResNet-18 / ImageNet (one-stage) | **SURGE** | 1/1 | **62.0%** (+3.9 over IR-Net) |

On VOC and GLUE, SURGE also comprehensively surpasses previous SOTA, with OPs identical to prior BNNs (zero increase in inference cost).

### Ablation Study

| Configuration | ImageNet ResNet-18 Top-1 (one-stage quantization) | Notes |
|---------------|--------------------------------|-------|
| STE baseline | Several percentage points lower than SURGE | First-order surrogate only |
| + DPGC (fixed $\lambda$) | Significant improvement, but occasionally unstable | Lacks magnitude balancing |
| + AGS (norm-ratio) = **SURGE** | **62.0%** and stable training | Full model |
| Replace AGS with fixed $\lambda$ | Large $\lambda$ fails to train, small $\lambda$ gives no compensation | Validates need for adaptivity |
| DPGC only in last few layers | Much smaller improvement | Mismatch accumulates in deeper layers |

### Key Findings
- Gradient statistics in Figure 1 show: after adding SURGE, the activation gradient distribution shifts significantly **rightward** with heavier tails, confirming the auxiliary branch recovers information lost by STE truncation.
- The combination of DPGC + AGS improves ImageNet by 0.5–1% over DPGC alone, indicating that magnitude balancing is not just for "engineering stability" but is essential for convergence.
- After training, discarding the auxiliary branch in ResNet-18 yields inference OPs identical to standard BNN ($1.63\times 10^8$), perfectly matching the goal of "training compensation, zero extra inference cost."
- Also effective on BERT-base/GLUE, demonstrating SURGE is not limited to convolution and applies to linear operators like attention projection.

## Highlights & Insights
- The "detach self-cancellation" formulation is the most ingenious engineering trick in the paper: $f-f\downarrow+f$ is $f$ in the forward pass and the true gradient of $f$ in the backward pass. This generalizes to any scenario where "forward uses A, backward uses B," and can be transferred to knowledge distillation, adversarial training, differentiable pruning, etc.
- Viewing "STE as a low-order approximation, with a full-precision replica supplementing higher-order terms" reframes BNN training from "finding a smarter sign approximation" to "compensating the first-order Taylor residual," providing clearer physical intuition.
- AGS balances the two paths using norm-ratio, fundamentally isomorphic to multi-task gradient balancing methods like GradNorm and PCGrad, but with a theoretical derivation that explicitly shows the "optimal $\lambda^*$ degenerates to $\eta\|\mu_b\|_2/\|\mu_a\|_2$ under isotropic noise," making it more convincing than purely heuristic approaches.

## Limitations & Future Work
- Training nearly doubles memory and FLOPs (auxiliary branch matches main branch in size), making training costly.
- $\eta$ still requires manual tuning, with optimal values varying across backbones; theoretically, $\eta=\kappa c_\theta/(1+\rho)$, but these quantities are not monitored in practice, so grid search is still needed.
- The assumption that $g_b$ and $g_a$ noise are uncorrelated may not hold precisely in deep networks.
- Lacks comparison with multi-bit quantization (W2A2, W4A4); transferability beyond pure 1-bit is unknown.

## Related Work & Insights
- **vs IR-Net / ReCU / BONN**: These methods modify the sign approximation function or weight distribution, essentially "modifying the forward pass." SURGE leaves the forward pass unchanged and only opens a bypass in the backward pass, making the approach orthogonal and combinable with the others.
- **vs DSQ / LSQ**: DSQ uses a parametric sigmoid to asymptotically approximate sign, LSQ introduces learnable scales; SURGE places "learnability" in a completely independent full-precision replica, offering greater expressiveness and zero inference burden.
- **vs Frequency-domain BNN (FDA-BNN)**: FDA-BNN moves sign to the frequency domain to alleviate mismatch; SURGE directly compensates with full-precision gradients in the spatial domain, making engineering implementation simpler.

## Rating
- Novelty: ⭐⭐⭐⭐ The "forward self-cancellation, backward open" detach trick + AGS norm-ratio derivation is cleanly constructed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Spans 4 major benchmarks, 3 task types, CNN + Transformer—top-tier among BNN papers.
- Writing Quality: ⭐⭐⭐⭐ Figures 1/2 intuitively explain the core mechanism; Theorem 5.3 and Corollary 5.4 are clearly derived.
- Value: ⭐⭐⭐⭐ Achieving 62.0% on ResNet-18/ImageNet sets a new one-stage BNN ceiling for its time, with zero extra inference cost, making it industry-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] BD-Net: Has Depth-Wise Convolution Ever Been Applied in Binary Neural Networks?](../../AAAI2026/model_compression/bd-net_has_depth-wise_convolution_ever_been_applied_in_binary_neural_networks.md)
- [\[ICLR 2026\] Adaptive Width Neural Networks](../../ICLR2026/model_compression/adaptive_width_neural_networks.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](../../ICLR2026/model_compression/a_recovery_guarantee_for_sparse_neural_networks.md)
- [\[NeurIPS 2025\] GoRA: Gradient-Driven Adaptive Low Rank Adaptation](../../NeurIPS2025/model_compression/gora_gradient-driven_adaptive_low_rank_adaptation.md)
- [\[ICLR 2026\] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization](../../ICLR2026/model_compression/fine-tuning_quantized_neural_networks_with_zeroth-order_optimization.md)

</div>

<!-- RELATED:END -->
