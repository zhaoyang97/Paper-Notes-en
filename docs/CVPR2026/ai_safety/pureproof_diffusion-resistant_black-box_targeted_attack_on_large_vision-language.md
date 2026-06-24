---
title: >-
  [Paper Note] PureProof: Diffusion-Resistant Black-box Targeted Attack on Large Vision-Language Models
description: >-
  [CVPR 2026][AI Safety][VLM Security] PureProof is the first black-box targeted attack capable of resisting "Diffusion-Based Purification (DBP)": it utilizes a diffusion proxy for single-step reverse prediction to align target semantics (SRA), uses timestep-adaptive re-noising to stabilize gradients (ARA), and applies self-consistency regularization for local coherence (SCR), ensuring adversarial images induce the attacker-specified target text even after being purified by fil…
tags:
  - "CVPR 2026"
  - "AI Safety"
  - "VLM Security"
  - "Targeted Adversarial Attack"
  - "Black-box Attack"
  - "Diffusion Purification"
  - "Adversarial Robustness"
date: 2026-05-08
content_hash: e55b35ccf66939cc
---

# PureProof: Diffusion-Resistant Black-box Targeted Attack on Large Vision-Language Models

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Cao_PureProof_Diffusion-Resistant_Black-box_Targeted_Attack_on_Large_Vision-Language_Models_CVPR_2026_paper.html)  
**Code**: To be confirmed  
**Area**: AI Safety / Adversarial Attacks  
**Keywords**: VLM Security, Targeted Adversarial Attack, Black-box Attack, Diffusion Purification, Adversarial Robustness

## TL;DR
PureProof is the first black-box targeted attack capable of resisting "Diffusion-Based Purification (DBP)": it utilizes a diffusion proxy for single-step reverse prediction to align target semantics (SRA), uses timestep-adaptive re-noising to stabilize gradients (ARA), and applies self-consistency regularization for local coherence (SCR), ensuring adversarial images induce the attacker-specified target text even after being purified by filters like DiffPure.

## Background & Motivation

**Background**: Large Vision-Language Models (VLMs) are being widely deployed in scenarios such as AI agents, but they are vulnerable to "targeted adversarial attacks" (inducing the model to output specific text using imperceptible perturbations). However, most attack research is evaluated under **no-defense** settings, casting doubt on its practical relevance. For image modalities, **Diffusion-Based Purification (DBP)** is the most mainstream and effective black-box defense—washing away adversarial perturbations through "forward noising + reverse denoising" of diffusion models, which has been integrated into VLM defense frameworks like BlueSuffix.

**Limitations of Prior Work**: Existing targeted attacks (MF-it/MF-ii, CoA, AnyAttack, FOA, etc.) fail almost entirely when encountering DBP, as perturbations are nearly completely purified, and models revert to benign outputs—empirical tests show these attacks achieve nearly 0% ASR (Target) against DBP.

**Key Challenge**: Existing evasion attacks that can bypass DBP (DiffAttack, DiffHammer) face two fundamental issues. ① They require **backpropagation through the entire diffusion denoising trajectory**, which involves extremely deep computational graphs and is prone to gradient vanishing/explosion; attacking VLMs adds the further cost of large proxy encoders like CLIP. ② They fail to properly handle the **inherent randomness** of the diffusion process, resulting in noisy gradients and unstable optimization. Moreover, these methods are designed for white-box image classifiers and are not adapted for black-box VLM targeted attacks where "neither the VLM nor the purifier is visible."

**Goal**: To create adversarial images that induce target outputs even after DBP purification under a black-box threat model where **both** the VLM and the DBP purifier are invisible, while avoiding the high cost and instability of full-trajectory backpropagation.

**Key Insight**: Rather than laboriously backpropagating through the entire denoising chain, it is better to leverage a **diffusion proxy** to perform a **single-step reverse prediction** at random timesteps to calculate a closed-form "clean image preview" $\hat x_0$, and directly use it to align with target semantics.

**Core Idea**: Replace full-chain backpropagation with single-step reverse "clean image preview" alignment (SRA), and use Adaptive Re-noising Augmentation (ARA) and Self-Consistency Regularization (SCR) specifically to suppress gradient jitter caused by diffusion randomness.

## Method

### Overall Architecture
Threat Model: The attacker has no access to the victim VLM $M$ or the specific DBP purifier $P$ (user-uploaded images are automatically purified before being fed to the model). The goal is to construct $x_{adv}=x_{clean}+\epsilon$ within an $\ell_\infty$ budget $\|\epsilon\|_\infty\le\varepsilon$ such that $M(P(x_{adv}),t_{in})=t_{tar}$—meaning the model outputs the specified target text $t_{tar}$ even after purification. The target image $x_{tar}$ is generated from $t_{tar}$ using a public text-to-image model (Stable Diffusion).

The optimization of PureProof follows the same "single-step pipeline" in each iteration: the current adversarial image $x_{adv}$ is **forward-noised to a randomly sampled timestep** $t\sim\mathrm{Unif}\{1,\dots,T_p\}$ to obtain $x_t$; a diffusion proxy performs **one-step reverse denoising** to predict the clean image $\hat x_0(x_t,t)$ in closed form; then three loss terms are applied to $\hat x_0$: SRA aligns $\hat x_0$ with the target image, ARA performs re-noising and averaging over $\hat x_0$ for smoothing, and SCR constrains the consistency of two clean image estimations. The total loss is used to update $x_{adv}$ via PGD. Since it does not traverse the full denoising trajectory, the pipeline is both computationally efficient and stable.

### Key Designs

**1. Stochastic Reverse Alignment (SRA): Aligning with the target via single-step reverse "clean image preview"**

To address the deep computational graphs and gradient issues of full-chain backpropagation, SRA noes not simulate the complete denoising process. Instead, it uses a proxy diffusion model for a **single** reverse step. In each iteration, $x_{adv}$ is forward-noised to a random timestep $t$ to get $x_t$, and the clean image is predicted directly via the DDPM closed-form formula: $\hat x_0(x_t,t)=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}$. This preview is then aligned with the target image $x_{tar}$ by maximizing cosine similarity in a pre-trained encoder (e.g., CLIP) embedding space: $L_{SRA}=-\mathbb{E}_t[\mathrm{sim}(\hat x_0(x_t,t),x_{tar})]$. Since $\hat x_0$ falls directly on the model's denoising direction and only passes through one step, SRA avoids computational bottlenecks and gradient instability, providing more stable gradient estimation. Empirical results show single-step is approximately 25× faster than full-chain backpropagation (1.761s/step vs 44.717s/step).

**2. Adaptive Re-noising Augmentation (ARA): Timestep-adaptive re-noising as curvature-aware smoothing**

Diffusion randomness increases with timestep $t$, making single-step predictions high-variance and biased. ARA’s approach is: for each sampled timestep $t$, the predicted $\hat x_0$ is injected with $K$ independent Gaussian noises at the **same forward noise level** to generate augmented variants $\tilde x_t^{(k)}=\sqrt{\bar\alpha_t}\,\hat x_0+\sqrt{1-\bar\alpha_t}\,\epsilon^{(k)}$, and the average similarity of these variants to the target is computed: $L_{ARA}=-\mathbb{E}_t\big[\frac1K\sum_{k=1}^K\mathrm{sim}(\tilde x_t^{(k)},x_{tar})\big]$. Larger timesteps result in stronger re-noising and proportionally stronger regularization, matching the intuition that regularization should be stronger when $\hat x_0$ is less reliable. Theorem 1 in the paper proves that a second-order Taylor expansion shows ARA is equivalent to a **curvature-aware regularization** that penalizes regions with high curvature $\mathrm{tr}(H_\phi)$, steering gradients toward stable regions of the loss surface.

**3. Self-Consistency Regularization (SCR): Constraining consistency between estimations for local temporal coherence**

To ensure adversarial updates remain within locally consistent regions of the diffusion manifold, SCR re-noises $\hat x_0(x_t,t)$ to get $\tilde x_t'=\sqrt{\bar\alpha_t}\,\hat x_0+\sqrt{1-\bar\alpha_t}\,\epsilon'$, performs another single-step reverse denoising to get a new estimate $\hat x_0'=\hat x_0(\tilde x_t',t)$, and penalizes the difference: $L_{SCR}=\mathbb{E}_t[\gamma_t\cdot\|\hat x_0'-\hat x_0(x_t,t)\|_2^2]$, where $\gamma_t=1-t/T_p$ reduces the weight at later timesteps. SCR ensures consistency between adjacent clean image estimates, further enhancing the robustness of adversarial samples against random purification trajectories.

### Loss & Training
The total objective combines the three terms: $L_{PureProof}=\beta\cdot L_{SRA}+(1-\beta)\cdot L_{ARA}+L_{SCR}$ (where $\beta=0.3$). Optimization uses PGD for 100 steps with a step size of 1/255 and an $\ell_\infty$ budget $\varepsilon=16/255$. Proxy settings: an ensemble of three CLIP encoders (ViT-B/16, ViT-B/32, ViT-g-14-laion2B) is used for transferability; Guided Diffusion is used as the diffusion proxy with EOT=10, timestep upper bound $T_p=150$, and $K=3$ re-noised variants.

## Key Experimental Results

> Metrics: **Ensemble CLIP Score** (average similarity to target text across five CLIP encoders); **ASR (Target)** = ratio of perfectly successful attacks; **ASR (Fool)** = sum of perfectly successful and partially misled attacks (judged by GPT-4). Due to DBP randomness, results are averaged over N=10 trials.

### Main Results

Open-source VLMs against three DBP defenses (Selected results for DiffPure and LM, CLIP Score / ASR-Target / ASR-Fool %):

| VLM | Method | DiffPure CLIP | DiffPure Tgt | DiffPure Fool | LM CLIP | LM Tgt | LM Fool |
|-----|------|--------------|--------------|---------------|---------|--------|---------|
| LLaVA-1.5 | DH-cos | 0.5047 | 0.0 | 41.4 | 0.4932 | 0.0 | 23.8 |
| LLaVA-1.5 | **PureProof** | **0.5983** | **12.3** | **76.8** | **0.6231** | **18.6** | **77.3** |
| LLaVA-1.6 | DH-cos | 0.4734 | 0.0 | 41.8 | 0.4670 | 1.6 | 25.8 |
| LLaVA-1.6 | **PureProof** | **0.5647** | **17.8** | **78.1** | **0.5830** | **22.5** | **84.4** |
| Gemma 3 | DH-cos | 0.4487 | 0.0 | 36.7 | 0.4412 | 0.0 | 36.5 |
| Gemma 3 | **PureProof** | **0.5231** | **8.6** | **85.4** | **0.5410** | **20.7** | **87.7** |
| Qwen3-VL | DH-cos | 0.3699 | 1.4 | 38.3 | 0.3452 | 0.0 | 22.3 |
| Qwen3-VL | **PureProof** | **0.4493** | **13.3** | **77.5** | **0.4613** | **25.6** | **81.4** |

All baselines lacking "diffusion-awareness" achieve 0% ASR (Target) against DBP, highlighting the strength of DBP and the fragility of old attacks; PureProof reaches 25.6% ASR (Target) on Qwen3-VL/LM and improves ASR (Fool) by over 50% compared to non-diffusion-aware baselines.

Commercial VLMs against DiffPure (CLIP / ASR-Target / ASR-Fool %):

| VLM | Method | CLIP | Target | Fool |
|-----|------|------|--------|------|
| GPT-5 | DH-cos | 0.4633 | 0.0 | 29.0 |
| GPT-5 | **PureProof** | **0.5457** | **11.0** | **77.0** |
| Gemini-2.5 | DH-cos | 0.4610 | 0.0 | 37.0 |
| Gemini-2.5 | **PureProof** | **0.5287** | **11.0** | **73.0** |

### Ablation Study

| Configuration / Analysis | Key Result | Description |
|------|---------|------|
| Full $L_{PureProof}$ | Optimal with all three components | SRA+ARA+SCR |
| Removing loss components | All metrics drop; **ARA contributes most** | ARA smooths the loss surface |
| ARA variant count $K$ | Significant jump from $K=0{\to}1$; levels off at $K{\ge}2$ | Few augmentations suffice; $K=3$ used |
| Single-step latency | PureProof 1.761s vs DA-cos 44.717s | Single-step reverse ≈25× speedup |
| Gaussian Noise Robustness (σ=16/255) | Highest across models (LLaVA-1.5: 0.6757) | Implicit consideration of noise variants |

### Key Findings
- **Prior attacks are neutralized by DBP**: Non-diffusion-aware attacks like MF/CoA/AnyAttack/FOA achieve 0% ASR (Target) under all DBP defenses, proving DBP is a truly effective black-box defense.
- **ARA is the primary gradient stabilizer**: Removing ARA leads to the largest drop in performance, confirming that "curvature-aware smoothing" is critical for optimization under diffusion randomness.
- **Strong transferability and noise resistance**: PureProof's CLIP Score barely drops under Gaussian noise, whereas CoA's performance significanty declines, indicating inherent robustness to general post-processing.
- **Competitive even without defense**: PureProof achieves the highest CLIP Score on GPT-5 in a pure no-defense setting; when integrated with CoA objectives (PureProof+CoA), it becomes the overall leader.

## Highlights & Insights
- **"Single-step reverse preview" as the core efficiency driver**: Using the DDPM closed-form $\hat x_0$ to approximate purified output alignment avoids deep computational graphs and gradient explosion, providing a 25× speedup—a strategy transferable to other optimization problems involving diffusion chains.
- **Theoretic support for re-noising as curvature regularization**: ARA is not just an empirical trick; Theorem 1 demonstrates it is equivalent to adaptive curvature penalization, embedding "smoothness proportional to randomness" into the gradients.
- **First effective black-box targeted attack against DBP-protected VLMs**: Reveals that VLM defense frameworks integrating DBP are not invincible, providing crucial warnings for the security of real-world deployments.
- **Inherent noise resistance as a byproduct**: Because the optimization implicitly traverses re-noised variants, adversarial samples are naturally robust against general post-processing perturbations.

## Limitations & Future Work
- **Dependency on Proxy Quality**: Attack effectiveness depends on how well diffusion and CLIP proxies approximate the real purifier and victim model; transferability against highly divergent, unknown DBPs remains an open question.
- **Absolute ASR (Target) remains modest**: Even for the SOTA, the perfect success rate is mostly in the 8–25% range, with most gains reflected in ASR (Fool), showing that "precise induction of text" under DBP is still challenging.
- **Double-edged nature as an attack**: This work exposes VLM vulnerabilities and requires complementary defense research; the authors position it as a tool for more rigorous defense evaluation.
- **Future Directions**: Exploring more universal diffusion proxy distillation or extending SRA/ARA to a small number of multiple steps to balance efficiency and fidelity.

## Related Work & Insights
- **vs DiffAttack / DiffHammer**: These are DBP evasion methods for white-box classifiers using full-chain backpropagation. PureProof uses single-step reverse prediction for black-box VLM attacks, proving more efficient and stable.
- **vs AttackVLM (MF-it/MF-ii) / CoA / FOA**: These are non-diffusion-aware VLM attacks that fail (0% ASR) against DBP; PureProof represents a fundamental upgrade for defense-aware scenarios.
- **vs BPDA / EOT**: BPDA's identity approximation fails for DBP, and EOT is purely empirical; ARA replaces them with adaptive curvature regularization with theoretical guarantees.

## Rating
- Novelty: ⭐⭐⭐⭐ First black-box VLM targeted attack against DBP; the combination of "single-step reverse + curvature-aware re-noising" is novel and theoretically supported.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 open-source + 2 commercial VLMs, 3 types of DBP, noise robustness, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear chain of motivation-method-theory; Theorem 1 provides solid explanation for ARA.
- Value: ⭐⭐⭐⭐ Dispels the illusion of safety provided by DBP integration, offering important warnings for real-world VLM/agent security.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VCP-Attack: Visual-Contrastive Projection for Transferable Black-Box Targeted Attacks on Large Vision-Language Models](vcp-attack_visual-contrastive_projection_for_transferable_black-box_targeted_att.md)
- [\[CVPR 2026\] SIF: Semantically In-Distribution Fingerprints for Large Vision-Language Models](sif_semantically_in-distribution_fingerprints_for_large_vision-language_models.md)
- [\[CVPR 2026\] A Provable Energy-Guided Test-Time Defense Boosting Adversarial Robustness of Large Vision-Language Models](a_provable_energy-guided_test-time_defense_boosting_adversarial_robustness_of_la.md)
- [\[CVPR 2026\] What Your Features Reveal: Data-Efficient Black-Box Feature Inversion Attack for Split DNNs](what_your_features_reveal_data-efficient_black-box_feature_inversion_attack_for_.md)
- [\[CVPR 2026\] PROMPTMINER: Black-Box Prompt Stealing against Text-to-Image Generative Models via Reinforcement Learning and VLM-Guided Optimization](promptminer_black-box_prompt_stealing_against_text-to-image_generative_models_vi.md)

</div>

<!-- RELATED:END -->
