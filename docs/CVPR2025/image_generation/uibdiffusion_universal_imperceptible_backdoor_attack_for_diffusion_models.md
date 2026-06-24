---
title: >-
  [Paper Note] UIBDiffusion: Universal Imperceptible Backdoor Attack for Diffusion Models
description: >-
  [CVPR 2025][Image Generation][Backdoor Attack] UIBDiffusion proposes the first imperceptible backdoor attack method for diffusion models. By repurposing universal adversarial perturbation (UAP) as a backdoor trigger, it achieves a triple threat of universality (image- and model-agnostic), practicality (high attack success rate without affecting generation quality), and undetectability (bypassing state-of-the-art defenses like Elijah and TERD).
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Backdoor Attack"
  - "Diffusion Models"
  - "Adversarial Perturbation"
  - "Imperceptible Trigger"
  - "Security Defense"
date: 2026-05-08
content_hash: c449fad6a6570500
---

# UIBDiffusion: Universal Imperceptible Backdoor Attack for Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2412.11441](https://arxiv.org/abs/2412.11441)  
**Code**: Coming soon  
**Area**: Image Generation / AI Security  
**Keywords**: Backdoor Attack, Diffusion Models, Adversarial Perturbation, Imperceptible Trigger, Security Defense

## TL;DR

UIBDiffusion proposes the first imperceptible backdoor attack method for diffusion models. By repurposing universal adversarial perturbation (UAP) as a backdoor trigger, it achieves a triple threat of universality (image- and model-agnostic), practicality (high attack success rate without affecting generation quality), and undetectability (bypassing state-of-the-art defenses like Elijah and TERD).

## Background & Motivation

**Background**: Diffusion models have become the mainstream generative paradigm, but prior studies (such as BadDiffusion, TrojDiff, and VillanDiffusion) show they are vulnerable to backdoor attacks. Adversaries inject backdoors into models via data poisoning, forcing the model to generate target images when receiving input noise embedded with a predefined trigger.

**Limitations of Prior Work**: Existing backdoor triggers (e.g., grey squares, Hello Kitty patches, glasses) exhibit noticeable visual patterns. Although effective, they are easily detected through manual inspection or trigger inversion-based defense algorithms (e.g., Elijah, TERD). Reducing trigger intensity to improve imperceptibility significantly compromises attack effectiveness and universality.

**Key Challenge**: There is a fundamental trade-off between the effectiveness and imperceptibility of backdoor triggers—an effective trigger must introduce sufficient distribution shift, which in turn makes it susceptible to detection.

**Goal**: To design a trigger that simultaneously possesses universality (applicable to arbitrary images and models), effectiveness (high attack success rate), and imperceptibility (bypassing SOTA defenses).

**Key Insight**: Adversarial perturbations naturally possess imperceptibility, universality, and effectiveness. In particular, Universal Adversarial Perturbations (UAPs) are image- and model-agnostic, which precisely fits the required characteristics.

**Core Idea**: To transform UAPs, originally designed to deceive discriminative models, into triggers for backdoor attacks on diffusion models. The distribution shift introduced by UAPs is sufficient to drive the backdoor behavior, yet their faint and patternless characteristics prevent trigger inversion algorithms from accurately reconstructing them.

## Method

### Overall Architecture

The method consists of a two-stage pipeline: (1) Trigger Generation—An improved UAP generation algorithm is employed to train a generator network with joint additive and non-additive perturbations, yielding the imperceptible trigger $\tau$; (2) Backdoor Injection—The VillanDiffusion framework is utilized to poison the training data by adding the trigger via $r(x, \tau) = x + \varepsilon \odot \tau$, jointly optimizing normal generation and the backdoor attack objective.

### Key Designs

1. **Trigger Generation based on Improved UAP**:

    - **Function**: Generates image-agnostic, model-agnostic, and imperceptible backdoor triggers.
    - **Mechanism**: A GAN-like generator $\mathcal{G}_\gamma$ is employed with latent noise $z$ as input to simultaneously output an additive perturbation $\tau$ and a non-additive spatial transformation perturbation $f$. Driven by a pre-trained image classifier $\mathcal{C}$, the objective is optimized as: $\mathcal{L}_{\mathcal{G}} = -\mathcal{H}(\mathcal{C}(x \otimes f + \tau), \mathcal{C}(x))$, causing the perturbed image to be misclassified. Meanwhile, the $l_\infty$ norm of $\tau$ is constrained within a budget $\xi$. Compared to the original DeepFool UAP, this method is more effective and robust.
    - **Design Motivation**: The original DeepFool UAP yields a lower attack success rate; the joint additive and non-additive optimization strategy (inspired by GUAP) generates more powerful perturbations.

2. **Imperceptible Trigger Injection Mechanism**:

    - **Function**: Injects the imperceptible trigger into the training data of diffusion models.
    - **Mechanism**: Unlike the mask-replacement method in VillanDiffusion, $r(x,g) = M \odot g + (1-M) \odot x$, UIBDiffusion adopts an additive injection approach: $r(x, \tau) = x + \varepsilon \odot \tau$, where $\varepsilon$ controls trigger intensity. This aligns with the noise superposition method of adversarial perturbations, training poisoned samples that are virtually indistinguishable from clean samples.
    - **Design Motivation**: The additive injection approach renders the trigger independent of image content (eliminating the need for mask positions) and is formally similar to the forward diffusion process.

3. **Distribution Shift Analysis and Defense Evasion**:

    - **Function**: Explains why the UAP trigger is both effective and mathematically hard to detect.
    - **Mechanism**: The distribution shift introduced on input noise by the UIBDiffusion trigger is conceptually similar to traditional visible triggers (e.g., glasses) — both shift the mean of the input noise from $\mathcal{N}(0, I)$ to $\mathcal{N}(r, \hat{\beta}^2 I)$. However, the faint and patternless characteristics of UAP prevent trigger inversion-based defense algorithms (e.g., Elijah, TERD) from accurately reconstructing the trigger pattern, thereby bypassing detection.
    - **Design Motivation**: Existing defense algorithms assume that triggers possess reconstructible spatial patterns; UAP fundamentally breaks this assumption.

### Loss & Training

Backdoor injection employs the unified loss from VillanDiffusion: $\mathcal{L}_\theta = \eta_c \mathcal{L}_c + \eta_p \mathcal{L}_p$, where $\mathcal{L}_c$ maintains clean generation capabilities and $\mathcal{L}_p$ drives backdoor learning. The trigger is generated offline once prior to training and is universal across all images and models. The poisoning rate is typically set between 5% and 20%.

## Key Experimental Results

### Main Results

| Method | Poisoning Rate | ASR ↑ | FID ↓ | Trigger Visibility |
|------|-------|------|------|----------|
| BadDiffusion | 10% | 97.2% | 12.4 | Visible (Grey square) |
| VillanDiffusion | 10% | 98.5% | 11.8 | Visible (Glasses) |
| UIBDiffusion | 5% | **99.1%** | **11.2** | **Invisible** |
| UIBDiffusion | 10% | 99.5% | 11.0 | Invisible |

### Defense Evasion

| Defense Method | BadDiffusion Detection Rate | VillanDiffusion Detection Rate | UIBDiffusion Detection Rate |
|---------|----------------|------------------|-----------------|
| Elijah | 95.3% | 97.1% | **8.7%** |
| TERD | 92.8% | 94.5% | **12.3%** |

### Key Findings

- UIBDiffusion achieves a 99.1% ASR at an **extremely low poisoning rate of 5%**, outperforming BadDiffusion (97.2%) which requires a 10% poisoning rate.
- While maintaining a higher ASR, UIBDiffusion also secures a lower FID, indicating that the imperceptible trigger has less impact on clean generation capabilities.
- Detection rates of Elijah and TERD **plummet from over 90% to 8–12%**, proving that UAP triggers fundamentally circumvent trigger inversion-based defense strategies.
- The trigger is robust across various samplers (DDIM, DEIS, DPM-Solver, etc.) and various diffusion models (DDPM, LDM, NCSN).

## Highlights & Insights

- **Deft Cross-Domain Transfer**: Adapting the concept of adversarial perturbations from discriminative models to backdoor attacks on generative models for the first time, demonstrating that tools from both domains can be cross-pollinated.
- **Crucial Signal for Security Research**: The fundamental design assumption of mainstream defense methods (Elijah, TERD)—that triggers have a specific, reconstructible pattern—is completely shattered.
- **"Low poisoning rate + high attack success rate"** combination makes the attack significantly more stealthy and dangerous.

## Limitations & Future Work

- Currently limited to pixel-level triggers, leaving the imperceptibility of text prompt-level backdoors unexplored.
- The generation of UAP triggers relies on pre-trained classifiers, which could theoretically introduce biases toward specific classifier models.
- Defenses tailored against imperceptible triggers like those in UIBDiffusion have not yet been explored.
- Validation on large-scale text-to-image models (e.g., SDXL, FLUX) remains limited.

## Related Work & Insights

- **vs. BadDiffusion/VillanDiffusion**: These use visible triggers; while effective, their defense detection rates exceed 90%, whereas UIBDiffusion reduces detection rates to less than 13%.
- **vs. Traditional Imperceptible Backdoors**: Conventional methods require generating image-specific perturbations, whereas UIBDiffusion's UAP is universal (generated once and used indefinitely).
- **Warning to the Defense Community**: There is an urgent need to develop novel defense methods that do not rely on trigger reconstruction/inversion.

## Rating

- Novelty: ⭐⭐⭐⭐ Creatively transfers UAP to backdoor attacks on diffusion models for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluations across multiple models, samplers, datasets, and defense methods.
- Writing Quality: ⭐⭐⭐⭐ Presents a well-defined threat model and a clear attack pipeline.
- Value: ⭐⭐⭐⭐ Holds significant educational and warning value for the AI security community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] OminiControl: Minimal and Universal Control for Diffusion Transformer](../../ICCV2025/image_generation/ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)
- [\[ICML 2025\] DRAG: Data Reconstruction Attack using Guided Diffusion](../../ICML2025/image_generation/drag_data_reconstruction_attack_using_guided_diffusion.md)
- [\[CVPR 2025\] UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics](unireal_universal_image_generation_and_editing_via_learning_real-world_dynamics.md)
- [\[CVPR 2025\] InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions](intermimic_towards_universal_whole-body_control_for_physics-based_human-object_i.md)
- [\[NeurIPS 2025\] OmniSync: Towards Universal Lip Synchronization via Diffusion Transformers](../../NeurIPS2025/image_generation/omnisync_towards_universal_lip_synchronization_via_diffusion.md)

</div>

<!-- RELATED:END -->
