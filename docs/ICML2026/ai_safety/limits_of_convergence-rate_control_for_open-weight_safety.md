---
title: >-
  [Paper Note] Limits of Convergence-Rate Control for Open-Weight Safety
description: >-
  [ICML 2026][AI Safety][open-weight safety] The authors formalize "open-weight safety" as "how to delay the convergence speed of malicious fine-tuning…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "open-weight safety"
  - "convergence rate"
  - "Hessian spectrum"
  - "spectral reparameterization"
  - "tamper resistance"
date: 2026-05-08
content_hash: d692228d0ae608a5
---

# Limits of Convergence-Rate Control for Open-Weight Safety

**Conference**: ICML 2026  
**arXiv**: [2602.18868](https://arxiv.org/abs/2602.18868)  
**Code**: Not yet released  
**Area**: AI Safety / Optimization Theory / Open-Weight Model Governance  
**Keywords**: open-weight safety, convergence rate, Hessian spectrum, spectral reparameterization, tamper resistance

## TL;DR
The authors formalize "open-weight safety" as "how to delay the convergence speed of malicious fine-tuning," proving that the largest singular value of the Hessian spectrum is determined by the lower bound of the weight spectrum. Based on this, they design the SpecDef algorithm, which can strictly slow down first/second-order optimization. However, they also prove that any such convergence-rate control method can be circumvented by an attacker at the cost of a linear increase in model size.

## Background & Motivation

**Background**: Open foundation models, once released, lack theoretically guaranteed training resistance—users can freely fine-tune weights, including for malicious purposes such as deepfakes or chemical weapons. Most open-weight governance relies on "licenses/throttled release" policy approaches; technical training-time resistance (e.g., TAR, RepNoise, RMU, ELM) is scattered and lacks a unified theoretical explanation.

**Limitations of Prior Work**: (1) Existing unlearning/unmodifiable training methods fail under systematic evaluation—simply adjusting the learning rate allows "erased" capabilities to be restored within tens of fine-tuning steps; (2) These methods are ad hoc, with no clear explanation of "why they sometimes work and when they must fail"; (3) The industry has long conflated inference-time safety with training-time safety, lacking a unified definition.

**Key Challenge**: The essence of "retaining functionality while making retraining difficult" is to increase the second-order (Hessian) spectrum while preserving zeroth-order behavior—yet first-order optimization convergence speed is determined precisely by the largest singular value of the Hessian. Is it mathematically possible to construct a transformation that "keeps function unchanged but explodes the Hessian spectrum"? Conversely, can it be proven that all such transformations have upper limits?

**Goal**: (a) Formalize training-time safety as an "iteration complexity/convergence-rate control" problem; (b) Provide a lower bound on the Hessian spectrum that can be directly manipulated via the weight spectrum; (c) Construct a provable algorithm, SpecDef, based on this; (d) Simultaneously prove that any such method has structural limits, and attackers can break it at linear extra cost.

**Key Insight**: First-order optimization must use learning rate $\eta \leq 1/L$, where $L$ is lower-bounded by the largest singular value of the Hessian $\sigma_1(H^{\mathcal{L}}_{\theta})$. If $\sigma_1$ can be pushed to astronomical values without changing function outputs, attackers are forced to use $\eta\to 0$, resulting in "numerically untrainable" models.

**Core Idea**: Apply SVD to selected layers for "symmetric reparameterization"—multiply the top-$k$ singular values by $\alpha$, and insert compensating layers at adjacent positions to exactly offset the change; functionality remains unchanged, but the Hessian's largest singular value is increased by at least $\alpha$ times, pushing feasible learning rates below normal floating-point precision.

## Method

### Overall Architecture
SpecDef is executed once before model release: (1) Select several layers $\theta_i$; (2) Insert identity linear layers as placeholders at adjacent positions; (3) Perform SVD on $\theta_i$ to obtain $U \Sigma V^\top$; (4) Multiply the top-$k$ singular values by $\alpha$ to get new weights $\theta_i' = U \tilde\Sigma V^\top$; (5) Write the "compensation matrix" $\theta_i^{comp} = U \Sigma \tilde\Sigma^{-1} U^\top$ into the identity layer position, ensuring that $\theta_i^{comp} \theta_i'$ is functionally equivalent to the original $\theta_i$. On GPT-OSS-20b, operating on 10 layers takes only 15 seconds.

### Key Designs

1. **Lower Bound of Hessian Spectrum via Weight Spectrum (Theorem 3)**:

    - **Function**: Establishes $\sigma_1(\nabla^2_\theta \mathcal{L}) \geq \sup_{r_1, r_2} \sigma_{r_1}(A)\sigma_1(B)\sigma_{r_2}(C)\cos\theta_1\cos\theta_2$, converting the "hard-to-measure largest Hessian eigenvalue" into a "directly controllable largest singular value of a certain layer's weights."
    - **Mechanism**: Uses the Poincaré separation theorem to lower-bound the Hessian's largest singular value by that of a $p\times q$ sub-block $\nabla^2_{\theta_i,\theta_j}\mathcal{L}$; for standard MLP/CNN/Transformer, this sub-block has an $ABC$ decomposition (e.g., in a three-layer MLP, $\partial^2 f/\partial \theta_3\partial \theta_1 = (x^\top \otimes I_m)^\top D_{z_1}\cdot \theta_2^\top \cdot D_{z_2}$, with $\theta_2$ in the middle); then applies the classical singular value inequality to obtain the above lower bound.
    - **Design Motivation**: This is the theoretical linchpin of the framework—once the largest singular value of the intermediate matrix $B = \theta_k$ is amplified by $\alpha$, the Hessian's largest singular value is increased proportionally, forcing the learning rate upper bound $\eta \leq 1/\alpha \cdot (\text{const})$, thus circumventing the longstanding issue of "inability to directly control the Hessian."

2. **Lower-Max Spectral Reparameterization + SpecDef (Function-Preserving Hessian Spectrum Stretching)**:

    - **Function**: Defines a class of mappings $\mathcal{T}_c: f_\theta \mapsto f_{\theta'}$ satisfying (i) $\sigma_1(H^{\mathcal{L}}_{\theta'}) \geq c$, (ii) function distance $d(\mathcal{T}_c[f], f) \leq \epsilon$; SpecDef is a concrete construction.
    - **Mechanism**: The algorithm applies $\tilde\Sigma \leftarrow T\Sigma$ to selected $\theta_i$ ($T = \mathrm{diag}(\alpha,\dots,\alpha,1,\dots,1)$, amplifying the top $k$ singular values by $\alpha$), replaces the original weights with $U\tilde\Sigma V^\top$, and inserts an identity placeholder layer with $\theta_i^{comp} = U\Sigma\tilde\Sigma^{-1}U^\top$; since $\theta_i^{comp}\theta_i' = U\Sigma V^\top = \theta_i$, the forward pass is strictly unchanged. $\alpha$ is chosen to "force the adversary below the smallest effective learning rate": if most LMs cannot converge at $\eta < 10^{-6}$, set $\alpha \geq 10^6$.
    - **Design Motivation**: Simple "weight rescaling" alters outputs, so compensation is necessary; inserting identity layers allows the authors to bypass issues with non-1-homogeneous activations like ReLU—cross-layer compensation between identity layers is always valid. The cost is a linear increase in model size (parameter count).

3. **Fundamental Limit of Convergence-Rate Control (Theorem on Layer Injection Attack)**:

    - **Function**: Proves that any method in the "symmetric spectral reparameterization" class can be undone by an attacker at $O(\text{model size})$ extra cost, establishing the fundamental limit of open-weight safety.
    - **Mechanism**: The authors abstract all possible transformations of SpecDef/similar methods into a class of mappings, and prove that for any such $\mathcal{T}$, there exists an inverse mapping $\mathcal{T}^{-1}$ to restore the spectrum. If the attacker knows the layer structure, they can reconstruct the original Hessian spectrum by merging the compensation matrix and the original layer ("layer collapse"), thus restoring normal convergence—at a linear, not exponential, cost in model size; curvature-aware optimizers (Sophia, Muon, AdaHessian) can only locally improve, not break this limit.
    - **Design Motivation**: Many open-weight safety papers claim "practical resistance," but the authors aim to shatter this optimism—not because SpecDef is ineffective, but because the entire "convergence-rate control" approach is fundamentally limited against knowledgeable adversaries. Future work must seek alternatives beyond convergence-rate control.

### Loss & Training
SpecDef itself requires no training; it is a one-off algebraic operation. The paper also suggests a "certification process": before release, perform a hyperparameter sweep over $\alpha, k,$ and layer selection to ensure convergence is difficult under common fine-tuning recipes. Attack-side experiments use SGD/Adam with multiple learning rates ($10^{-6}$ to $3\times 10^{-5}$) and curvature-aware optimizers (Sophia/Muon/AdaHessian) for relearning evaluation.

## Key Experimental Results

### Main Results (WMDP-bio Relearning Attack on Llama-3.1-8B-Instruct)

| Method | Initial Acc | LR=$10^{-6}$ (Steps / Acc) | LR=$10^{-5}$ | LR=$3\times 10^{-5}$ |
|--------|-------------|----------------------------|--------------|----------------------|
| ELM | 0.204 | 120 / 0.607 | 23 / 0.621 | 20 / 0.671 |
| RepNoise | 0.325 | 510 / 0.468 | 30 / 0.627 | 33 / 0.650 |
| TAR | 0.290 | 510 / 0.307 | 66 / 0.638 | 66 / 0.618 |
| Llama-3.1-8B Original | 0.482 | 93 / 0.607 | 20 / 0.629 | 10 / 0.609 |
| **SpecDef ($\alpha=10\mathrm{k}$)** | 0.204 | **10 / 0.195†** | **10 / 0.191†** | **10 / 0.195†** |

(† indicates perplexity increased by over 100%, i.e., fine-tuning diverged immediately)

| Benchmark Model Utility Retention (smaller change with larger α is better) | WMDP | PPL | MMLU | Winogrande | ARC | HellaSwag |
|----------------------------------------------------------------------------|------|-----|------|------------|-----|-----------|
| α = 1k | $\Delta = -0.1$ | -0.01 | +0.1 | 0 | +1.0 | +0.3 |
| α = 100k | $-0.1$ | $-0.01$ | +0.6 | +0.3 | +0.3 | 0 |
| α = 1B | 0 | $-0.02$ | — | — | — | — |

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| Amplify $\sigma_1$ only (no compensation) | Model functionality severely altered, accuracy lost |
| Full SpecDef | Functionality nearly unchanged, fine-tuning diverges immediately |
| Attack: curvature-aware optimizers (Sophia/Muon/AdaHessian) | Cannot bypass SpecDef |
| Attack: layer merging (layer-injection attack) | Normal convergence restored, at +linear model size cost |
| Attack: vanilla spectral attack | Reusing SpecDef-type transformations can reverse some prior defenses |

### Key Findings
- **All prior defenses are broken by simple LR sweep**: In Table 1, ELM, RepNoise, RMU, TAR, NPO, etc., are breached to 0.6+ accuracy within tens of steps at $\eta = 10^{-5}$, indicating their "defense" only works at a fixed LR.
- **SpecDef causes fine-tuning to diverge at all LRs**: Even if the attacker sweeps five orders of magnitude in LR, $\alpha = 10^4$ pushes them below normal floating-point range, with PPL increasing by 100%+.
- **Model utility loss is nearly zero**: MMLU/Winogrande/ARC/HellaSwag fluctuate within $\pm 0.3$, proving that mathematical forward equivalence is achievable, with the only cost being "increased parameter count and slightly slower inference."
- **But attackers can bypass at linear cost**: The authors themselves provide the layer-injection attack construction, meaning any adversary with full model access can undo SpecDef—this is the paper's most important pessimistic conclusion and the reason for the "Limits" in the title.

## Highlights & Insights
- **Translating safety into optimization theory**: Previous unlearning papers often manually defined loss terms; this work directly uses classical iteration complexity analysis, quantifying "hard to train" as "must use extremely small LR," providing a solid theoretical foundation.
- **Bridge from weight spectrum to Hessian spectrum is textbook-worthy**: Theorem 3 elegantly applies random matrix theory tools, and the bound remains non-vacuous even in rank-deficient cases, tighter than the classic Horn-Johnson bound.
- **Symmetric reparameterization + identity injection**: This trick of "preserving zeroth-order while arbitrarily stretching the spectrum" is ingenious, echoing Dinh et al.'s symmetry analysis of sharpness, and can be extended to generalization/sharpness-aware training and other areas.
- **Both positive and negative results are equally important**: The paper not only proposes the best-known algorithm, SpecDef, but also proves its fundamental limit—this "propose + refute" structure is rare, reminding future researchers that achieving true training-time safety requires moving beyond the convergence-rate control framework.

## Limitations & Future Work
- SpecDef assumes attackers only use first/second-order smooth optimizers, without seriously considering randomized methods (e.g., stochastic Langevin dynamics) or sign-gradient/zeroth-order attacks.
- The linear increase in model parameter count is nontrivial for large model deployment—adding 10 compensation layers to a 20B model could mean several GB of extra VRAM.
- The "smallest effective learning rate" is empirically determined; different hardware and floating-point precisions (FP16/BF16/FP8) have different truncation points, so $\alpha$ selection needs recalibration.
- The layer-injection attack has been proven feasible by the authors, but its practical attack complexity (required information, hyperparameter tuning) is not quantitatively evaluated, left for future work.
- No cryptographic-level security proof is provided, only "numerical non-convergence"—this is reminiscent of the failure of obfuscated gradients, and stronger models are needed in the future.

## Related Work & Insights
- **vs TAR / RepNoise / RMU / ELM**: These are empirical unlearning methods with extra regularization; unified evaluation in the paper shows all collapse under LR sweep. SpecDef is the first to provide a provable guarantee of "hard to converge at all LRs."
- **vs Sharpness-Aware Minimization (Foret 2020) / Dinh 2017 on sharpness symmetry**: This work uses symmetry to push sharpness to infinity, following a similar line of thought but with the opposite goal.
- **vs Obfuscated Gradients (Athalye 2018)**: That work proved "gradient obfuscation" defenses can be broken by simple methods; this paper gives an equivalent warning for "convergence-rate obfuscation": any defense relying on optimization geometry can be undone by the corresponding inverse operation.
- **vs Bresler et al. on PAC-learning Hardness**: The limit here is not computational complexity hardness, but algebraic invertibility—a different perspective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to establish a convergence-rate theoretical framework for open-weight safety, providing both a provable algorithm and a fundamental limit—strong in both directions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers LM + ViT + Stable Diffusion, 10+ defense baselines, and curvature-aware optimizer attacks; real-world attacker modeling is somewhat idealized.
- Writing Quality: ⭐⭐⭐⭐ Definitions, propositions, and theorems are clearly stated, with smooth theory-experiment integration; dense in length, requiring some optimization background.
- Value: ⭐⭐⭐⭐⭐ Provides the community with a clear direction—future work on training-time safety must move beyond convergence-rate control, making this "signpost" more valuable than any specific algorithm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks](../../AAAI2026/ai_safety/improving_the_convergence_rate_of_ray_search_optimization_for_query-efficient_ha.md)
- [\[ICML 2026\] ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)
- [\[ICLR 2026\] Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature](../../ICLR2026/ai_safety/dataless_weight_disentanglement_in_task_arithmetic_via_kronecker-factored_approx.md)
- [\[ACL 2026\] OmniCompliance-100K: A Multi-Domain Rule-Grounded Real-World Safety Compliance Dataset](../../ACL2026/ai_safety/omnicompliance-100k_a_multi-domain_rule-grounded_real-world_safety_compliance_da.md)
- [\[CVPR 2026\] One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control](../../CVPR2026/ai_safety/one-to-more_high-fidelity_training-free_anomaly_generation_with_attention_control.md)

</div>

<!-- RELATED:END -->
