---
title: >-
  [Paper Note] The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers
description: >-
  [CVPR 2026][Optimization & Theory][adversarial attack] This work refactors sign-based adversarial attack optimizers into coordinate-wise gradient descent, revealing that non-decaying step sizes are the root cause of non-convergence and instability. It proposes the Monotone Decreasing Coordinate Step (MDCS) strategy and theoretically proves that MDCS-MI achieves an optimal
tags:
  - CVPR 2026
  - Optimization & Theory
  - adversarial attack
  - transferability
  - sign-based optimizer
  - step-size scheduling
  - convergence guarantee
date: 2026-05-08
content_hash: 111b38430c1cbe12
---
# The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers

**Conference**: CVPR 2026  
**arXiv**: [2602.19096](https://arxiv.org/abs/2602.19096)  
**Authors**: Wei Tao, Yang Dai, Jincai Huang, Qing Tao (National University of Defense Technology, Academy of Military Sciences, Hefei University of Technology)  
**Code**: [AndssY/MDCS_attack](https://github.com/AndssY/MDCS_attack)  
**Area**: Optimization  
**Keywords**: adversarial attack, transferability, sign-based optimizer, step-size scheduling, convergence guarantee

## TL;DR

This work refactors sign-based adversarial attack optimizers into coordinate-wise gradient descent, revealing that non-decaying step sizes are the root cause of non-convergence and instability. It proposes the Monotone Decreasing Coordinate Step (MDCS) strategy and theoretically proves that MDCS-MI achieves an optimal $O(1/\sqrt{T})$ convergence rate. MDCS significantly enhances the attack transferability and stability across image classification and cross-modal retrieval tasks.

## Background & Motivation

Adversarial example generation is fundamentally a constrained optimization problem. While sign-based optimizers (I-FGSM, MI-FGSM, PGD) are the de facto standards for adversarial attacks, they suffer from two core issues:

**Theoretical Flaws**: Karimireddy et al. proved that sign-based gradients cannot converge to the optimal solution even in simple convex problems. The sign operator discards gradient magnitude information, which distorts the true descent direction.

**Practical Instability**: Counter-intuitively, increasing the number of iterations can lead to a sharp decline in attack success rates—I-FGSM achieves a 37.1% success rate at $t=2$ iterations, which plummet to 15.5% at $t=20$.

**Key Insight**: By refactoring I-FGSM as coordinate-wise gradient descent, the step size for the $i$-th coordinate is revealed to be $\alpha/|\partial J/\partial x_i|$. As the attack approaches a local optimum and the gradient tends toward zero, this equivalent step size diverges and fluctuates violently, violating the fundamental requirement of step-size decay in optimization theory. While MI-FGSM mitigates single-gradient fluctuations through momentum, it still inherits the inherent issue of non-decreasing step sizes.

This discovery directly inspires the transfer of successful strategies from AdaGrad/AMSGrad—where Monotone Decreasing Coordinate Step (MDCS) fixes Adam's convergence flaws—to adversarial attack optimizers.

## Method

### Overall Architecture

The generation of adversarial examples is essentially a constrained optimization problem maximizing classification loss within an $\ell_\infty$ ball. Sign-based optimizers like I-FGSM, MI-FGSM, and PGD are the standards—taking only the gradient sign and advancing with a fixed step size $\alpha$. The overall mechanism of this paper is to first adopt an equivalent perspective: viewing these optimizers as "coordinate-wise gradient descent," thereby exposing that the real step size hidden behind the sign operator is actually divergent. Then, the strategy used by AdaGrad/AMSGrad to fix Adam is applied, imposing a "monotone decreasing" constraint (MDCS) on each coordinate's step size. This transforms non-convergent, unstable optimizers into new ones with convergence guarantees. The modification does not change original components like momentum or input transformations but simply replaces the "step-size" mechanism.

### Key Designs

**1. Rewriting sign operations as coordinate-wise gradient descent: Making "hidden step sizes" visible**

The sign operator is difficult to analyze because it discards all gradient magnitude information, appearing to have a fixed step size $\alpha$. This work decomposes the update for the $i$-th coordinate of I-FGSM as follows:

$$x_{t+1,i}^{adv} = x_{t,i}^{adv} + \frac{\alpha}{|\partial J / \partial x_i|} \cdot \partial J / \partial x_i$$

This decomposition makes it clear that the actual step size acting on the gradient direction of the $i$-th coordinate is not $\alpha$, but $\alpha/|\partial J/\partial x_i|$, which is inversely proportional to the gradient magnitude. When the attack approaches a local optimum and the gradient vanishes, this equivalent step size diverges and oscillates, violating the basic optimization requirement that "step sizes should decay over iterations." This explains why I-FGSM success rates collapse from 37.1% at $t=2$ to 15.5% at $t=20$. MI-FGSM smooths single-step gradient jitters with momentum, but the root problem of non-decaying step sizes persists.

**2. MDCS: Forcing monotone decreasing constraints on coordinate step sizes**

Since the root cause is the lack of step-size decay, a constraint is directly imposed. Inspired by AMSGrad's use of Monotone Decreasing Coordinate Step (MDCS) to fix Adam, this paper maintains a step-size variable for each coordinate that is only allowed to decrease:

$$d_{t,i} = \min(1/|m_{t+1,i}|, \; d_{t-1,i})$$

This min operator ensures $d_{t,i} \leq d_{t-1,i} \leq 1$. Regardless of gradient fluctuations, the coordinate step size is pinned to be monotonically decreasing. For MDCS-MI, momentum is first updated with normalized gradients $\mathbf{m}_{t+1} = \beta_t \mathbf{m}_t + \nabla J(\mathbf{x}_t^{adv}) / \|\nabla J(\mathbf{x}_t^{adv})\|_1$, then these step sizes form a diagonal matrix $\mathbf{D}_t$ for the parameter update:

$$\mathbf{x}_{t+1}^{adv} = \text{Clip}_\mathbf{x}^\epsilon[\mathbf{D}_t^{-1/2}(\mathbf{x}_t^{adv} + \alpha_t \mathbf{D}_t \mathbf{m}_{t+1})]$$

This step replaces the "divergent equivalent step size" with a "controlled shrinking step size," serving as the core surgery to fix unstable optimizers.

**3. Decaying momentum with $O(1/\sqrt{T})$ convergence guarantee: Theoretical backing**

To ensure stability, the momentum coefficient also decays with iterations—$\beta_t = \beta\lambda^{t-1}$ ($0 < \beta < 1$, $0 < \lambda < 1$), and the step-size scaling is set as $\alpha_t = \gamma/\sqrt{t}$. Under the assumption that the objective function is locally concave on the constrained domain $\mathbf{Q}$ and the gradients are bounded $\|\nabla J\|_1 \leq M$ (Theorem 3), the average iterate of MDCS-MI satisfies:

$$J(\mathbf{x}^*) - J(\bar{\mathbf{x}}^{adv}_T) \leq O(1/\sqrt{T})$$

This provides the first optimal-order convergence rate guarantee for sign-based attack optimizers. This significance lies in transforming the empirical observation that "more iterations should be more stable" into a theoretically expected result.

**4. Plug-and-play: Replacing only the step-size component**

MDCS is not a completely new attack algorithm but a modular step-size strategy that can be integrated into any sign-based attack. Original components such as momentum, variance tuning, input transformations, and cross-modal alignment are all preserved. Only the "sign extraction and fixed step size" part is replaced with the MDCS monotone decreasing step size. Consequently, it can locally derive MDCS-MI / MDCS-MEF / MDCS-OPS for image classification and MDCS-SGA / MDCS-DRA / MDCS-SAAET for cross-modal retrieval.

## Key Experimental Results

### Settings
- **Image Classification**: NIPS2017 dataset, 1000 images, $\epsilon = 16/255$, $T = 10$.
- **Proxy Models**: ResNet-50 / ViT-B/16.
- **Target Models**: CNNs (VGG16, MobileNet-v2, Inc-v3) + ViTs (ViT-B, PiT-B, Vis-S) + Defense models.
- **Cross-modal Retrieval**: Flickr30K, $\epsilon = 8/255$, ALBEF/TCL/CLIP_CNN/CLIP_ViT.

### Table 1: Single-model attack transferability (Proxy Res50, Success Rate %)

| Type | Method | Res50 | VGG16 | Mob-v2 | Inc-v3 | ViT-B | PiT-B | Vis-S |
|---|---|---|---|---|---|---|---|---|
| ② | MI | 100.0 | 59.4 | 53.1 | 36.3 | 12.5 | 23.1 | 26.6 |
| ② | **MDCS-MI** | 100.0 | **67.2** | **60.3** | **41.3** | **13.4** | 23.3 | **30.8** |
| ② | MEF | 99.3 | 94.9 | 94.4 | 91.2 | 65.3 | 81.1 | 88.2 |
| ② | **MDCS-MEF** | 100.0 | **96.4** | **95.5** | **93.4** | 58.7 | 78.8 | **91.0** |
| ③ | OPS | 99.5 | 98.0 | 97.8 | 98.2 | 88.8 | 93.8 | 96.7 |
| ③ | **MDCS-OPS** | **99.9** | **98.9** | **99.0** | **99.1** | **89.3** | **94.7** | **97.9** |

Consistent Gain with MDCS: MDCS-MI improves over MI by **+7.8%** on VGG16 and **+5.0%** on Inc-v3. MDCS-OPS achieves new Prev. SOTA across all target models.

### Table 4: Cross-modal retrieval attack (Flickr30K, Proxy ALBEF, Black-box R@1 %)

| Method | TCL TR | TCL IR | CLIP\_CNN TR | CLIP\_CNN IR | CLIP\_ViT TR | CLIP\_ViT IR |
|---|---|---|---|---|---|---|
| SGA | 87.67 | 87.88 | 38.04 | 46.17 | 41.63 | 50.36 |
| **MDCS-SGA** | **91.78** | **91.24** | **41.35** | **49.71** | **45.08** | **53.93** |
| DRA | 89.78 | 90.52 | 46.63 | 57.28 | 50.32 | 59.11 |
| **MDCS-DRA** | **93.26** | **92.98** | **49.94** | **59.31** | **55.56** | **62.44** |
| SA-AET | 96.31 | 96.19 | 54.23 | 63.50 | 58.88 | 65.18 |
| **MDCS-SAAET** | **96.52** | **96.71** | **60.25** | **67.01** | **60.54** | **67.89** |

MDCS is equally effective in cross-modal scenarios: MDCS-SAAET improves over SA-AET by **+6.02%** on CLIP_CNN TR and **+2.71%** on CLIP_ViT IR. In cross-architecture transfer from CLIP to TCL, SA-AET's IR R@1 increases by **+8.22%**.

### Defense Models
MDCS remains effective against adversarial training and defense models. MDCS-OPS achieves optimal results on most of the 8 defense models tested, validating its robustness.

### Stability Verification
While the attack success rate of MI-FGSM fluctuates or decreases as the iteration count $T$ increases, MDCS-MI maintains stable performance improvements across different $T$, validating the convergence theory.

## Highlights & Insights

- **Optimization-centric Root Cause Analysis**: Refactoring the sign operation into coordinate-wise gradient descent reveals that non-decaying step size is the essence of instability, providing a novel and persuasive perspective.
- **Closed-loop of Theory and Practice**: This work provides the first $O(1/\sqrt{T})$ optimal convergence guarantee for sign-based attacks, with experimental stability improvements matching theoretical predictions.
- **Plug-and-play**: MDCS seamlessly replaces step-size strategies in any sign-based attack without modifying other components (momentum, input transformations, etc.), offering high practical value.
- **Cross-task Generality**: MDCS is effective across image classification (CNN/ViT) and cross-modal retrieval (VLM), covering single-model attacks, ensemble attacks, and attacks against defense models.

## Limitations & Future Work

- **Local Concavity Assumption**: The theoretical analysis relies on the local concavity of the objective function on the constrained domain; the guarantee remains limited for highly non-convex loss surfaces.
- **Hyperparameter Tuning**: The step-size scaling factor $\gamma$ requires grid searching within [2, 4], increasing usable cost.
- **Non-targeted Focus**: Experiments only evaluated non-targeted attacks; effectiveness in targeted attack scenarios remains unverified.
- **Diminishing Returns on Strong Baselines**: When the baseline method is already extremely strong (e.g., OPS approaching 99% on some models), the room for MDCS improvement is limited.

## Related Work & Insights

- **Sign-based Attacks**: FGSM $\rightarrow$ I-FGSM/PGD $\rightarrow$ MI-FGSM $\rightarrow$ VMI/GRA/PGN/MEF/OPS. These progressively introduced momentum and input transformations but maintained fixed step sizes.
- **Adaptive Step-size Optimization**: AdaGrad introduced MDCS for sparse learning, and AMSGrad used MDCS to fix Adam's convergence. This paper transfers these ideas to adversarial attacks.
- **VLM Attacks**: Co-Attack $\rightarrow$ SGA $\rightarrow$ DRA $\rightarrow$ SA-AET. Strategies grew more complex while still relying on PGD. MDCS serves as a universal upgrade module.
- **GRA's Decay Indicator**: GRA observed frequent flips in sign perturbations and proposed a decay indicator to dynamically adjust step sizes, but it lacked theoretical guarantees. MDCS offers a more systematic solution.

## Rating

- Novelty: ⭐⭐⭐⭐ — Linking optimization theory (MDCS/AMSGrad) to adversarial attacks is a novel perspective with insightful refactoring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage across image classification, cross-modal retrieval, CNNs, ViTs, VLMs, and defense models.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, rigorous theoretical derivation, and well-organized experimental section.
- Value: ⭐⭐⭐⭐ — A universal, plug-and-play attack enhancement strategy with theoretical guarantees, highly relevant for adversarial robustness research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unveiling the Power of Multiple Gossip Steps: A Stability-Based Generalization Analysis in Decentralized Training](../../NeurIPS2025/optimization/unveiling_the_power_of_multiple_gossip_steps_a_stability-based_generalization_an.md)
- [\[CVPR 2026\] HyperNAS: Enhancing Architecture Representation for NAS Predictor via Hypernetwork](hypernas_enhancing_architecture_representation_for_nas_predictor_via_hypernetwor.md)
- [\[ICML 2026\] Stability Analysis of Sharpness-Aware Minimization](../../ICML2026/optimization/stability_analysis_of_sharpness-aware_minimization.md)
- [\[CVPR 2026\] Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning](enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p.md)
- [\[AAAI 2026\] Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment](../../AAAI2026/optimization/cost-minimized_label-flipping_poisoning_attack_to_llm_alignment.md)

</div>

<!-- RELATED:END -->
