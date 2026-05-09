---
title: >-
  [Paper Note] Towards Robust Defense against Customization via Protective Perturbation Resistant to Diffusion-based Purification
description: >-
  [ICCV 2025][Image Generation][Protective Perturbation] This paper proposes AntiPure, an adversarial perturbation method that directly attacks the diffusion-based purification process through two guidance mechanisms—Patch-wise Frequency Guidance (PFG) and Erroneous Timestep Guidance (ETG)—to generate protective perturbations that continue to disrupt customization fine-tuning even after purification, outperforming all existing protection methods under the purification-customization (P-C) workflow.
tags:
  - ICCV 2025
  - Image Generation
  - Protective Perturbation
  - Anti-Purification
  - DreamBooth
  - Diffusion Purification
  - adversarial attack
date: 2026-05-08
content_hash: 299d91037beab330
---

# Towards Robust Defense against Customization via Protective Perturbation Resistant to Diffusion-based Purification

**Conference**: ICCV 2025
**arXiv**: [2509.13922](https://arxiv.org/abs/2509.13922)
**Code**: N/A (not mentioned)
**Area**: Image Generation / Adversarial Defense / Diffusion Model Security
**Keywords**: Protective Perturbation, Anti-Purification, DreamBooth, Diffusion Purification, adversarial attack

## TL;DR
This paper proposes AntiPure, an adversarial perturbation method that directly attacks the diffusion-based purification process through two guidance mechanisms—Patch-wise Frequency Guidance (PFG) and Erroneous Timestep Guidance (ETG)—to generate protective perturbations that continue to disrupt customization fine-tuning even after purification, outperforming all existing protection methods under the purification-customization (P-C) workflow.

## Background & Motivation

### State of the Field
Customization fine-tuning techniques for diffusion models such as Stable Diffusion (e.g., DreamBooth, LoRA) pose serious security threats in the form of deepfakes and copyright infringement. Protective perturbation, which injects imperceptible adversarial noise into images to disrupt fine-tuning outputs, is a promising line of defense.

### Limitations of Prior Work
Existing protective perturbations (e.g., AdvDM, Anti-DreamBooth) can be readily removed by diffusion-based purification methods (e.g., DiffPure, GrIDPure). Purification eliminates adversarial perturbations by adding noise to and then denoising the adversarial image, rendering the protection ineffective. In practice, malicious users can purify images prior to fine-tuning, establishing a purification-customization (P-C) workflow under which existing protection methods are nearly entirely defeated.

### Key Findings
The authors systematically analyze three core reasons why anti-purification is harder than anti-customization:
1. **Absence of vulnerable components**: LDM contains an easily attacked VAE encoder, whereas the DDPM purification model exposes only the more robust UNet.
2. **Frozen parameters without training**: Purification requires no fine-tuning, so adversarial examples cannot corrupt the model prior through data poisoning.
3. **Fixed high-timestep denoising**: Purification begins denoising from a high timestep, at which point low-frequency structure is already locked, confining attacks to high-frequency components.

## Method

### Overall Architecture
Rather than attempting to preserve anti-customization perturbations through purification, AntiPure directly attacks the purification model itself. The core rationale is that distortions introduced during purification will cause the subsequently learned concept to deviate from the original image, even if customization itself proceeds normally.

### Problem Formulation
The ideal anti-purification perturbation is formulated as:

$$\delta^{adv} = \arg\max_{\|\delta\|_\infty \leq \eta} \min_{\theta_c} \mathbb{E}_x \mathcal{L}_{ldm}(\text{Pure}(x_0 + \delta); \theta_c)$$

Since direct backpropagation through this computation graph is intractable, the objective is decomposed into maximizing the discrepancy between the purified output and the perturbed input:

$$\delta^{adv'} = \arg\max_{\|\delta\|_\infty \leq \eta} \|\text{Pure}(x_0 + \delta) - (x_0 + \delta)\|_\infty$$

### Key Design 1: Patch-wise Frequency Guidance (PFG)
The clean-image prior encoded in the frozen parameters enables the purification model to recover low-frequency structure effectively, but its control over high-frequency components is weak. PFG exploits this vulnerability:

1. Apply UNet to the noisy adversarial sample $x_t$ to predict the denoised image $\widehat{x}_0$.
2. Decompose $\widehat{x}_0$ into patches and apply DCT to each patch.
3. Extract high-frequency components (the bottom-right quarter of the DCT spectrum) and maximize:

$$\mathcal{L}_{fre}(x_0; \delta^{adv}) = \sigma\left(\mathbb{E}_P \frac{4}{s^2} \sum_{m,n=s/2}^{s-1} \text{PatchDCT}(\widehat{x}_0, s)_{m,n}\right)$$

PFG amplifies high-frequency components in the purification model's predictions, indirectly reinforcing high-frequency elements of the adversarial perturbation to produce a uniform grid-like pattern. Since the attack targets high-frequency information, local structural content changes minimally, preserving perceptual consistency for human observers.

### Key Design 2: Erroneous Timestep Guidance (ETG)
The purification process can be viewed as a generative process with denoising fixed at a high timestep. ETG injects adversarial noise to impair the UNet's ability to distinguish appropriate behavior at different timesteps:

$$\mathcal{L}_{err\text{-}t}(x_0; \delta^{adv}) = -\|\epsilon_\theta(x_t, t_{err}) - \epsilon_\theta(x_t, t)\|_2^2$$

An erroneous timestep $t_{err}$ is fed to the UNet to obtain a noise prediction corresponding to a higher timestep, and the difference between the erroneous and correct predictions is minimized, thereby undermining the model's timestep-awareness.

### Loss & Training
PFG and ETG are combined with the standard $\mathcal{L}_{ddpm}$ and optimized via PGD gradient ascent:

$$\mathcal{L}_{pgd}(x_0; \delta^{adv}) = \mathbb{E}_{\epsilon,t}\left(\mathcal{L}_{ddpm} + \lambda_1 e^{\bar{\alpha}_t - 1} \mathcal{L}_{fre} + \lambda_2 e^{\mathcal{L}_{err\text{-}t}}\right)$$

where $\lambda_1 = \lambda_2 = 0.5$ and the attack timestep $t \sim \mathcal{U}(1, t^p)$ is restricted to the purification timestep range. The coefficient $e^{\bar{\alpha}_t - 1}$ increases the influence of PFG as $t$ decreases; the exponential applied to ETG enables more aggressive optimization.

## Key Experimental Results

### Experimental Setup
- **Datasets**: CelebA-HQ and VGGFace2, 50 IDs × 12 images at 512×512 per dataset
- **Baselines**: AdvDM, Mist, Anti-DreamBooth, SimAC
- **Purification**: GrIDPure (2 rounds × 20 iterations, $t^p=10$)
- **Evaluation metrics**: FID↑, ISM↓, FDFR, BRISQUE↑ (customization output quality), LPIPS↓ (perceptual perturbation difference)

### Main Results: DreamBooth P-C Workflow

| Dataset | Method | FID↑ | ISM↓ | BRISQUE↑ |
|--------|------|------|------|----------|
| CelebA-HQ | AdvDM | 77.51 | 0.6561 | 31.33 |
| CelebA-HQ | Mist | 70.23 | 0.6688 | 37.00 |
| CelebA-HQ | Anti-DB | 78.84 | 0.6422 | 31.76 |
| CelebA-HQ | SimAC | 67.37 | 0.6734 | 33.73 |
| CelebA-HQ | **AntiPure** | **81.15** | **0.6112** | **43.60** |
| VGGFace2 | AdvDM | 83.90 | 0.5923 | 37.42 |
| VGGFace2 | Anti-DB | 90.29 | 0.5938 | 38.35 |
| VGGFace2 | **AntiPure** | **90.77** | **0.5475** | **46.01** |

AntiPure achieves the best performance across all metrics on both datasets.

### LoRA Fine-tuning Validation

| Dataset | Method | FID↑ | ISM↓ | BRISQUE↑ |
|--------|------|------|------|----------|
| VGGFace2 | Anti-DB | 117.89 | 0.5723 | 58.56 |
| VGGFace2 | **AntiPure** | **127.67** | **0.5428** | **69.97** |

AntiPure maintains comprehensive superiority under LoRA fine-tuning, with a particularly notable gap in ISM.

### Ablation Study: Purification Iterations

| Method | Iter=10 ISM | Iter=20 ISM | Iter=30 ISM | Iter=40 ISM |
|------|------------|------------|------------|------------|
| Anti-DB | 0.6020 | 0.6352 | 0.6473 | 0.6391 |
| AntiPure | 0.6362 | 0.6271 | 0.6075 | **0.5994** |

Anti-DB degrades progressively with more purification iterations (rising ISM), whereas AntiPure becomes increasingly effective—consistent with its design philosophy of directly attacking the purification process itself.

### Perceptual Consistency
Under the same $\eta$ constraint, AntiPure achieves the lowest LPIPS perceptual difference among all methods, owing to PFG's effective avoidance of low-frequency modifications.

## Highlights & Insights
- **First formal treatment of the anti-purification task**: The work systematically analyzes why anti-purification is harder than anti-customization, establishing a theoretical foundation for future research.
- **Strategic shift in attack philosophy**: Rather than attempting to preserve perturbations through purification, the method causes the purification process itself to introduce distortions that indirectly disrupt subsequent fine-tuning.
- **Dual attack on frequency and timestep**: PFG exploits the purification model's weak control over high-frequency components, while ETG undermines timestep awareness; the two mechanisms act synergistically.
- **Counterintuitive robustness**: The unique property of becoming more effective as purification deepens demonstrates the method's robustness.

## Limitations & Future Work
- The method cannot induce semantic-level structural distortions (e.g., completely altering facial identity), and is limited to introducing identifiable artifacts.
- The approach relies on white-box access, requiring knowledge of the purification model's architecture and parameters.
- Evaluation is primarily confined to face datasets and two fine-tuning paradigms (DreamBooth and LoRA).
- Performance degrades somewhat under JPEG compression (CelebA-HQ results are noticeably weaker than VGGFace2).

## Related Work & Insights
- **Customization fine-tuning**: DreamBooth, LoRA, Textual Inversion, Custom Diffusion, ControlNet
- **Protective perturbation**: AdvDM, Mist, Anti-DreamBooth, SimAC, MetaCloak, CAAT
- **Diffusion-based purification**: DiffPure, DensePure, GrIDPure

## Rating
- Novelty: ⭐⭐⭐⭐ — First systematic definition and resolution of the anti-purification problem
- Technical Depth: ⭐⭐⭐⭐ — Thorough analysis of the three core challenges
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple datasets, fine-tuning methods, and purification configurations
- Value: ⭐⭐⭐ — White-box assumption limits practical deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing](dct-shield_a_robust_frequency_domain_defense_against_malicious_image_editing.md)
- [\[ICCV 2025\] PLA: Prompt Learning Attack against Text-to-Image Generative Models](pla_prompt_learning_attack_against_text-to-image_generative_models.md)
- [\[ICCV 2025\] FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers](freecus_free_lunch_subject-driven_customization_in_diffusion_transformers.md)
- [\[ICCV 2025\] Disrupting Model Merging: A Parameter-Level Defense Without Sacrificing Accuracy](disrupting_model_merging_a_parameter-level_defense_without_sacrificing_accuracy.md)
- [\[ICCV 2025\] Generating Multi-Image Synthetic Data for Text-to-Image Customization](generating_multi-image_synthetic_data_for_text-to-image_customization.md)

</div>

<!-- RELATED:END -->
