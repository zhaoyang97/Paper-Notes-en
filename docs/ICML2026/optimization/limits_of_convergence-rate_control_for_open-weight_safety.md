---
title: >-
  [Paper Note] Limits of Convergence-Rate Control for Open-Weight Safety
description: >-
  [ICML 2026][Optimization][open-weight safety] The authors formalize "open-weight safety" as the problem of "how to delay the convergence rate of malicious fine-tuning." They prove that the maximum singular value of the H…
tags:
  - "ICML 2026"
  - "Optimization"
  - "open-weight safety"
  - "convergence rate"
  - "Hessian spectrum"
  - "spectral reparameterization"
  - "tamper resistance"
date: 2026-05-08
content_hash: 82e95b87b27e7a85
---

# Limits of Convergence-Rate Control for Open-Weight Safety

**Conference**: ICML 2026  
**arXiv**: [2602.18868](https://arxiv.org/abs/2602.18868)  
**Code**: Not yet public  
**Area**: AI Safety / Optimization Theory / Open-Weight Model Governance  
**Keywords**: open-weight safety, convergence rate, Hessian spectrum, spectral reparameterization, tamper resistance

## TL;DR
The authors formalize "open-weight safety" as the problem of "how to delay the convergence rate of malicious fine-tuning." They prove that the maximum singular value of the Hessian is lower-bounded by the weight spectrum, leading to the design of the SpecDef algorithm which strictly slows down first- and second-order optimization. However, they also demonstrate that any such convergence-rate control method can be bypassed by an adversary at the cost of a "linear increase in model size."

## Background & Motivation

**Background**: Open-source foundation models lack theoretically guaranteed training resistance once released—users can freely fine-tune weights for malicious purposes such as deepfakes or chemical weapons. Open-weight governance primarily relies on policy-based paths like "licenses/throttled releases," while technical training-time resistance methods (e.g., TAR, RepNoise, RMU, ELM) remain fragmented and lack a unified theoretical explanation.

**Limitations of Prior Work**: (1) Existing unlearning/anti-finetuning methods fail under systematic evaluation—with slight adjustments to the learning rate, "erased" capabilities can be restored within dozens of fine-tuning steps; (2) These methods are ad hoc, lacking clarity on "why they occasionally work and when they must fail"; (3) The industry long conflates inference-time safety with training-time safety, lacking a unified definition.

**Key Challenge**: To "retain functionality while making retraining difficult" essentially requires maintaining zeroth-order behavior while increasing the second-order (Hessian) spectrum—since the convergence rate of first-order optimization is determined by the maximum singular value of the Hessian. Is it mathematically possible to construct a transformation that "keeps functionality constant but causes the Hessian spectrum to explode"? Conversely, can it be proven that all such transformations have an upper bound?

**Goal**: (a) Formalize training-time safety as an "iteration complexity / convergence rate control" problem; (b) Provide a lower bound where the weight spectrum directly manipulates the Hessian spectrum; (c) Construct the provable SpecDef algorithm based on this; (d) Simultaneously prove that any such method has a structural limit, which attackers can breach with linear extra costs.

**Key Insight**: First-order optimization must select a learning rate $\eta \leq 1/L$, where $L$ is lower-bounded by the maximum singular value of the Hessian $\sigma_1(H^{\mathcal{L}}_{\theta})$. If $\sigma_1$ can be pushed to astronomical figures without altering model output, an attacker is forced to use $\eta\to 0$, falling into a "numerically unlearnable" predicament.

**Core Idea**: Utilizing SVD to perform "symmetric reparameterization" on specific layers—multiplying the top-$k$ singular values of selected layers by $\alpha$ and inserting exactly offsetting compensation layers in adjacent positions. While functionality remains unchanged, the maximum singular value of the Hessian is forced to increase by at least $\alpha$, pushing the feasible learning rate below subnormal floating-point precision.

## Method

### Overall Architecture
SpecDef is executed once before model release: (1) Select several layers $\theta_i$; (2) Insert identity linear layers at adjacent positions as placeholders; (3) Perform SVD on $\theta_i$ to obtain $U \Sigma V^\top$; (4) Multiply top-$k$ singular values by $\alpha$ to obtain new weights $\theta_i' = U \tilde\Sigma V^\top$; (5) Write the "compensation matrix" $\theta_i^{comp} = U \Sigma \tilde\Sigma^{-1} U^\top$ into the identity layer position such that $\theta_i^{comp} \theta_i'$ is functionally equivalent to the original $\theta_i$. On GPT-OSS-20b, operating on 10 layers takes only 15 seconds.

### Key Designs

1. **Weight Spectrum Lower Bound for Hessian Spectrum (Theorem 3)**:

    - **Function**: Establishes $\sigma_1(\nabla^2_\theta \mathcal{L}) \geq \sup_{r_1, r_2} \sigma_{r_1}(A)\sigma_{1}(B)\sigma_{r_2}(C)\cos\theta_1\cos\theta_2$, transforming the "difficult-to-measure maximum eigenvalue of the Hessian" into the "directly controllable maximum singular value of a specific weight layer."
    - **Mechanism**: First uses the Poincaré Separation Theorem to bound the maximum singular value of the Hessian by the maximum singular value of a $p\times q$ sub-block $\nabla^2_{\theta_i,\theta_j}\mathcal{L}$. For standard MLPs/CNNs/Transformers, this sub-block has an $ABC$-type decomposition (e.g., in a three-layer MLP, $\partial^2 f/\partial \theta_3\partial \theta_1 = (x^\top \otimes I_m)^\top D_{z_1}\cdot \theta_2^\top \cdot D_{z_2}$, with $\theta_2$ sandwiched in the middle); the lower bound is then derived using classical singular value inequalities.
    - **Design Motivation**: This is the theoretical anchor of the framework—as long as the maximum singular value of the intermediate matrix $B = \theta_k$ is amplified by $\alpha$, the maximum singular value of the entire Hessian is amplified at least proportionally, forcing a learning rate upper bound $\eta \leq 1/\alpha \cdot (\text{constant})$, bypassing the long-standing issue of "inability to directly control the Hessian."

2. **Lower-Max Spectral Reparameterization + SpecDef (Function-Preserving Hessian Spectrum Stretching)**:

    - **Function**: Defines a class of mappings $\mathcal{T}_c: f_\theta \mapsto f_{\theta'}$ satisfying (i) $\sigma_1(H^{\mathcal{L}}_{\theta'}) \geq c$, (ii) functional distance $d(\mathcal{T}_c[f], f) \leq \epsilon$; SpecDef is its concrete construction.
    - **Mechanism**: The algorithm performs $\tilde\Sigma \leftarrow T\Sigma$ on selected $\theta_i$ (where $T = \mathrm{diag}(\alpha,\dots,\alpha,1,\dots,1)$ scales the first $k$ singular values by $\alpha$), replacing the original weights with $U\tilde\Sigma V^\top$. Simultaneously, it inserts identity layers and writes $\theta_i^{comp} = U\Sigma\tilde\Sigma^{-1}U^\top$. Forward pass invariance is guaranteed by $\theta_i^{comp}\theta_i' = U\Sigma V^\top = \theta_i$. Selection of $\alpha$ follows the principle of "forcing the opponent below the minimum effective learning rate": if most LMs fail to converge at $\eta < 10^{-6}$, one can set $\alpha \geq 10^6$.
    - **Design Motivation**: Simple weight rescaling alters output, so compensation is mandatory. Inserting identity layers allows the authors to bypass complexities introduced by non-1-homogeneous activations like ReLU—since cross-layer compensation between identity layers is always valid. The trade-off is a linear increase in model size (parameter count).

3. **Fundamental Limits of Convergence-Rate Control (Theorem on Layer Injection Attack)**:

    - **Function**: Proves that any method belonging to "symmetric spectral reparameterization" can be revoked by an attacker at an $O(\text{model size})$ extra cost, establishing the fundamental limit of open-weight safety.
    - **Mechanism**: The authors abstract all possible transformation forms of SpecDef/similar methods into a class of mappings and prove that for any such $\mathcal{T}$, there exists an inverse mapping $\mathcal{T}^{-1}$ that pulls the spectrum back to normal. If an attacker knows the layer structure, they can reconstruct the original Hessian spectrum by merging the compensation matrix with the original layer ("layer collapse"), thus restoring normal convergence. The cost is linear rather than exponential increase in model size; curvature-aware optimizers (Sophia, Muon, AdaHessian) can only provide local improvements and cannot break this upper bound.
    - **Design Motivation**: While many open-weight safety papers claim to provide "actual resistance," the authors aim to dismantle this optimism—not because SpecDef is ineffective, but because the entire "convergence rate control" path is inherently limited against a knowledgeable adversary. Future research must seek non-convergence-rate control solutions.

### Loss & Training
SpecDef itself requires no training; it is a one-time algebraic operation. The paper suggests a "certification process": sweeping hyperparameters for $\alpha, k,$ and layer selection before release to confirm that convergence is difficult under common fine-tuning recipes. Attack experiments use SGD/Adam with multiple learning rates ($10^{-6}$ to $3\times 10^{-5}$) and curvature-aware optimizers (Sophia/Muon/AdaHessian) for relearning evaluation.

## Key Experimental Results

### Main Results (WMDP-bio relearning attack on Llama-3.1-8B-Instruct)

| Method | Initial Acc | LR=$10^{-6}$ (Steps / Acc) | LR=$10^{-5}$ | LR=$3\times 10^{-5}$ |
|------|----------|----------------------------|--------------|----------------------|
| ELM | 0.204 | 120 / 0.607 | 23 / 0.621 | 20 / 0.671 |
| RepNoise | 0.325 | 510 / 0.468 | 30 / 0.627 | 33 / 0.650 |
| TAR | 0.290 | 510 / 0.307 | 66 / 0.638 | 66 / 0.618 |
| Llama-3.1-8B original | 0.482 | 93 / 0.607 | 20 / 0.629 | 10 / 0.609 |
| **SpecDef ($\alpha=10\mathrm{k}$)** | 0.204 | **10 / 0.195†** | **10 / 0.191†** | **10 / 0.195†** |

(† indicates perplexity increase over 100%, meaning fine-tuning diverged immediately)

| Model Utility Maintenance (Higher α with smaller change is better) | WMDP | PPL | MMLU | Winogrande | ARC | HellaSwag |
|---------------------------------------------|------|-----|------|------------|-----|-----------|
| α = 1k | $\Delta = -0.1$ | -0.01 | +0.1 | 0 | +1.0 | +0.3 |
| α = 100k | $-0.1$ | $-0.01$ | +0.6 | +0.3 | +0.3 | 0 |
| α = 1B | 0 | $-0.02$ | — | — | — | — |

### Ablation Study

| Configuration | Effect |
|------|------|
| Scaling $\sigma_1$ only (no compensation) | Sharp functional change, loss of accuracy |
| SpecDef full suite | Functions nearly identical, fine-tuning diverges |
| Attack: Curvature-aware optimizers | Cannot bypass SpecDef |
| Attack: Layer collapse (layer-injection attack) | Restores normal convergence at linear model size cost |
| Attack: Vanilla spectral attack | Reusing SpecDef-type transformations can reverse prior defenses |

### Key Findings
- **All prior defenses are pierced by simple LR sweeps**: In Table 1, ELM, RepNoise, RMU, TAR, and NPO are penetrated to 0.6+ accuracy within dozens of steps above $\eta = 10^{-5}$, indicating their "defense" is only valid at a fixed LR.
- **SpecDef causes fine-tuning to diverge at all LRs**: Even if an attacker sweeps LR across five orders of magnitude, they are pushed into the subnormal floating-point region by $\alpha = 10^4$, with PPL increasing by 100%+.
- **Near-zero loss in model utility**: MMLU/Winogrande/ARC/HellaSwag show fluctuations mostly within $\pm 0.3$, proving that mathematical identity in the forward pass is achievable. The cost is solely increased parameter count and slightly slower inference.
- **Bypassed by attackers at linear cost**: The authors construct the layer-injection attack themselves, implying any adversary with full model access can revoke SpecDef—the most significant pessimistic conclusion of the paper, explaining the "Limits" in the title.

## Highlights & Insights
- **Translating safety into optimization theory**: Unlike prior unlearning papers that manually define loss terms, this work employs classical analysis of iteration complexity, quantifying "difficulty of training" as "requirement for extremely small LR."
- **Weight Spectrum → Hessian Spectrum bridge is of textbook value**: Theorem 3 elegantly utilizes tools from random matrix theory, and the bound remains non-vacuous in rank-deficient cases, proving tighter than the classical Horn-Johnson bound.
- **Symmetric reparameterization + identity injection**: This maneuver to "maintain zeroth-order while arbitrarily stretching the spectrum" is ingenious. It aligns with Dinh et al.’s analysis of sharpness symmetry but extends it to generalization and safety.
- **Balance between proposal and critique**: Seeing a paper present a "best-known" algorithm while proving its fundamental limits is rare. It serves as a warning: to achieve true training-time safety, one must transcend the convergence rate control framework.

## Limitations & Future Work
- SpecDef assumes attackers use first/second-order smooth optimizers, not taking into account randomized methods (e.g., stochastic Langevin dynamics) or sign-based/zeroth-order attacks.
- Linear increase in parameter count is nontrivial for large model deployment—adding 10 compensation layers to a 20B model results in several GBs of extra VRAM.
- "Smallest effective learning rate" is empirically determined; floating-point truncation points vary across hardware and precisions (FP16/BF16/FP8).
- While the layer-injection attack is theoretically proven, its practical complexity (information needed, hyperparameter tuning) lacks quantitative evaluation.
- No cryptographic safety proof is provided, only "numerical non-convergence"—reminiscent of the failure of obfuscated gradients; stronger models are needed.

## Related Work & Insights
- **vs TAR / RepNoise / RMU / ELM**: These methods rely on empirical unlearning + extra regularization; systematic evaluation shows they collapse under LR sweeps. SpecDef provides the first provable guarantee across all LRs.
- **vs Sharpness-Aware Minimization (Foret 2020) / Dinh 2017**: This work uses symmetry to push sharpness to infinity, the opposite of the original intent but conceptually consistent.
- **vs Obfuscated Gradients (Athalye 2018)**: Just as that work proved gradient-based defenses could be bypassed, this work warns that any defense relying on optimization geometry ("convergence rate obfuscation") can be dismantled by inverse operations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Establishes the first convergence rate theoretical framework for open-weight safety, providing both a provable algorithm and its fundamental limits.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers LM, ViT, and Stable Diffusion with comparisons against 10+ defenses and curvature-aware attacks; however, the attacker modeling is somewhat idealized.
- Writing Quality: ⭐⭐⭐⭐ Clear definitions and theorems with smooth transitions; high density requires an optimization background.
- Value: ⭐⭐⭐⭐⭐ Provides a crucial direction calibration for the community—real training-time safety must move beyond convergence rate control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](on_the_convergence_rate_of_lora_gradient_descent.md)
- [\[ICML 2026\] Sign Lock-In: Randomly Initialized Weight Signs Persist and Bottleneck Sub-Bit Model Compression](sign_lock-in_randomly_initialized_weight_signs_persist_and_bottleneck_sub-bit_mo.md)
- [\[ICML 2026\] Towards Understanding Adam Convergence on Highly Degenerate Polynomials](towards_understanding_adam_convergence_on_highly_degenerate_polynomials.md)
- [\[ICML 2026\] Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence](balanced_lora_removing_parameter_invariance_to_accelerate_convergence.md)
- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](../../ICLR2026/optimization/convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)

</div>

<!-- RELATED:END -->
