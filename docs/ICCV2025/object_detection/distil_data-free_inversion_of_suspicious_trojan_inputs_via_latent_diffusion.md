---
title: >-
  [Paper Note] DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion
description: >-
  [ICCV 2025][Object Detection][Backdoor defense] DISTIL proposes a data-free trojan trigger inversion method that searches for trigger patterns in the latent space of a pretrained guided diffusion model—rather than in pix…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Backdoor defense"
  - "trigger inversion"
  - "diffusion models"
  - "zero-shot detection"
  - "model security"
date: 2026-05-08
content_hash: e674a48d423e0403
---

# DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion

**Conference**: ICCV 2025
**arXiv**: [2507.22813](https://arxiv.org/abs/2507.22813)  
**Code**: [https://github.com/AdaptiveMotorControlLab/DISTIL](https://github.com/AdaptiveMotorControlLab/DISTIL)  
**Area**: Object Detection / Model Security
**Keywords**: Backdoor defense, trigger inversion, diffusion models, zero-shot detection, model security

## TL;DR

DISTIL proposes a data-free trojan trigger inversion method that searches for trigger patterns in the latent space of a pretrained guided diffusion model—rather than in pixel space—and injects uniform noise regularization at each step to effectively distinguish genuine backdoor triggers from adversarial perturbations, achieving up to 7.1% accuracy improvement on BackdoorBench.

## Background & Motivation

Deep neural networks face serious threats from trojan (backdoor) attacks: adversaries embed poisoned samples carrying specific triggers into training data, causing the model to behave normally on clean inputs while producing targeted misclassifications on trigger-bearing inputs. This poses significant risks for safety-critical applications such as autonomous driving and object detection.

**Trigger inversion (RET)** is the primary post-hoc defense strategy—reconstructing the trigger pattern used by the attacker via reverse engineering. However, existing RET methods suffer from three core limitations:

**Adversarial perturbation confusion**: Optimizing triggers in high-dimensional pixel space tends to produce adversarial perturbations rather than genuine triggers, causing clean models to be falsely flagged as trojaned (high false positive rate).

**Strong prior assumptions**: Many methods assume triggers are small patches or have specific shapes, rendering them ineffective against dynamic or invisible triggers.

**Dependence on clean data**: Most methods require access to clean training data for pixel-space optimization, limiting their practical applicability.

**Core Insight**: Trojan models exhibit **stronger transferability** than clean models with respect to specific shortcut patterns—because backdoored networks are explicitly trained to associate triggers with target classes. If these shortcuts can be extracted, the transferability gap can be used to distinguish trojan models from clean ones.

**Key Insight**: Shift the search space from pixel space to the **latent space of a pretrained diffusion model**, leveraging the image manifold constraint imposed by the diffusion model to avoid degenerate adversarial solutions, while injecting uniform noise at each step to disrupt fragile adversarial optima.

## Method

### Overall Architecture

The DISTIL pipeline proceeds as follows:
1. Starting from pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$
2. Guiding the diffusion model's reverse process with gradients from the target classifier
3. Injecting uniform noise regularization at each step
4. Generating candidate trigger patterns
5. Distinguishing trojan models from clean models via a transferability score

### Key Designs

#### 1. Classifier-Guided Diffusion Inversion

DISTIL modifies the mean of the diffusion model's reverse process as:

$$\tilde{\mu}_\theta(x_t, t, y^{\text{tar}}, y^{\text{src}}) = \mu_\theta(x_t, t) + \Sigma_\theta(x_t, t) \nabla_{x_t} \log \frac{f(y^{\text{tar}} | x_t)}{f(y^{\text{src}} | x_t)} + \lambda_1 \cdot \eta_t$$

The gradient term simultaneously **increases the target class probability and decreases the source class probability**, driving the diffusion model to generate patterns that cause the classifier to transition from the source class to the target class.

**Design Motivation**: Pretrained guided diffusion models are inherently trained to follow gradient signals, and can therefore faithfully track classifier-provided guidance to reveal genuine shortcut patterns in latent space.

#### 2. Uniform Noise Injection Regularization

At each diffusion step, DISTIL injects uniform noise $\eta_t \sim \mathcal{U}(0,1)$ into the classifier input, with strength controlled by $\lambda_1$.

**Design Motivation**: Adversarial perturbations are inherently fragile—minor changes suffice to invalidate them. Injecting random noise forces the optimization to find **robust, transferable shortcut patterns** (i.e., genuine triggers) rather than noise-sensitive adversarial solutions. This is the key distinction between DISTIL and conventional pixel-space RET methods.

#### 3. Trojan Detection Score

For a generated candidate trigger $\delta^{\text{tar}}_{\text{src}}$, the transferability score is defined as:

$$\text{Score}(\delta^{\text{tar}}_{\text{src}}; f) = \mathbb{E}_{x' \sim \mathcal{X}'^{\text{src}}} \left[ \text{softmax}(f(x' + \delta))_{y^{\text{tar}}} - \text{softmax}(f(x' + \delta))_{y^{\text{src}}} \right]$$

The overall trojan score is the maximum over all $(y^{\text{src}}, y^{\text{tar}})$ pairs. Trojan models yield substantially higher scores than clean models.

#### 4. Fast DISTIL

Exhaustively enumerating all $(y^{\text{src}}, y^{\text{tar}})$ pairs incurs $O(K^2)$ complexity. Fast DISTIL reduces this to $O(K)$ by selecting the source class most distant from the target class in feature space:

$$y^{\text{src}} = \arg\min_{y \neq y^{\text{tar}}} \cos(\phi(y), \phi(y^{\text{tar}}))$$

Intuition: if a trigger can cause the least similar source class to transition to the target class, it must be exploiting a particularly strong model-specific shortcut.

#### 5. Extension to Object Detection

DISTIL extends to trojan scanning of object detection models by augmenting the guidance term with bounding box displacement gradients that push detection boxes toward image corners.

### Loss & Training

- The diffusion model uses pretrained GLIDE weights and is kept frozen
- Sampling uses $T=50$ steps
- Trigger generation is repeated up to 5 times until classifier confidence exceeds threshold $\lambda_2 = 0.95$
- The mitigation stage fine-tunes the classifier by injecting triggers into clean images while preserving correct labels

## Key Experimental Results

### Main Results

**BackdoorBench Classifier Scanning (CIFAR-10, scanning accuracy %)**:

| Attack | NC | SmoothInv | BTI-DBF | TRODO | **DISTIL** |
|--------|-----|-----------|---------|-------|-----------|
| BadNets | 76.4 | 86.3 | 84.0 | 86.2 | **94.9** |
| Blended | 65.2 | 84.9 | 85.7 | 85.0 | **93.4** |
| InputAware | 58.1 | 69.7 | 79.2 | 71.7 | **93.2** |
| LIRA | 54.9 | 70.8 | 80.0 | 82.5 | **90.6** |
| WaNet | 63.7 | 68.9 | 86.8 | 80.0 | **84.4** |

DISTIL achieves an average accuracy of **88.5%** on BackdoorBench, approximately **7.1%** above the best baseline.

### Ablation Study

| Setup | Modification | Round 0 | Round 4 | Round 11 |
|-------|-------------|---------|---------|---------|
| A | No diffusion model (pure pixel optimization) | 74.5 | 60.5 | 57.4 |
| B | No noise injection | 81.9 | 81.6 | 76.9 |
| C | No hyperparameter tuning | 80.6 | 78.0 | 73.3 |
| D | Fast DISTIL ($O(K)$) | 78.0 | 82.3 | 75.6 |
| **G (full)** | **DISTIL default** | **83.1** | **84.6** | **80.4** |
| H | DISTIL + clean data | 84.5 | 86.0 | 83.9 |

Key findings:
- Removing the diffusion model (Setup A) causes a sharp performance drop, validating the critical role of latent-space search.
- Removing noise injection (Setup B) also leads to notable degradation, confirming the importance of regularization.
- Replacing the backbone with a lighter diffusion model (Setups E/F) incurs only marginal performance loss, indicating robustness to backbone choice.

### Key Findings

1. **Object detection extension**: On the TrojAI object detection benchmark, DISTIL achieves **63.7%** accuracy, surpassing the second-best method by **9.4%**.
2. **Mitigation effectiveness**: After fine-tuning, the attack success rate (ASR) drops as low as **5.3%** (Blended attack) while maintaining high classification accuracy.
3. **Target class prediction**: On GTSRB, average target class prediction accuracy reaches **72.0%**, significantly outperforming all baselines.

## Highlights & Insights

- **Paradigm shift from pixel space to latent space**: Relocating trigger search from pixel space to the diffusion model latent space is the paper's most central contribution, fundamentally mitigating adversarial perturbation confusion.
- **Zero-shot capability**: Trojan scanning requires no clean training data, greatly reducing the barrier to practical deployment.
- **Cross-task generality**: The same framework extends seamlessly to both classification and object detection, demonstrating strong versatility.
- **Interpretability**: Generated triggers are human-readable and intuitively interpretable, in contrast to adversarial noise patterns.

## Limitations & Future Work

- The source class selection in Fast DISTIL assumes that maximum feature-space distance implies maximum diagnostic value, which may not hold for all-to-all attacks.
- The method relies on the quality of the pretrained diffusion model; although experiments show insensitivity to backbone choice, extreme scenarios warrant further validation.
- When attackers use adversarial training to implant backdoors, pixel-space methods may recover erroneous patterns leading to false positives; the proposed latent-space search mitigates but does not fully resolve this issue.
- Backdoor detection in other modalities, such as LLMs, remains unaddressed.

## Related Work & Insights

- **Neural Cleanse (NC)** [Wang et al., 2019]: Seminal RET work performing pixel-space optimization for small patch triggers.
- **SmoothInv** [Sun & Kolter, 2023]: Enhances classifier robustness via randomized smoothing before pixel-space inversion.
- **BTI-DBF** [Xu et al., 2024]: Decouples benign and trojan features via dual-branch trigger inversion.
- **GLIDE** [Nichol et al., 2021]: The guided diffusion model backbone used in this work.
- Insight: The image manifold constraint of diffusion models can be applied to other security scenarios that require distinguishing genuine patterns from spurious noise.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Introducing diffusion models into backdoor defense represents a genuinely novel direction.
- Technical Depth: ⭐⭐⭐⭐ — The uniform noise injection regularization is elegantly designed with clear theoretical motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 3 benchmarks, 11 attack types, and 12 baseline methods.
- Value: ⭐⭐⭐⭐ — The zero-shot property substantially lowers the deployment threshold.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion](diffusion_curriculum_synthetic-to-real_data_curriculum_via_image-guided_diffusio.md)
- [\[ICCV 2025\] EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision](evrt-detr_latent_space_adaptation_of_image_detectors_for_event-based_vision.md)
- [\[ICCV 2025\] SFUOD: Source-Free Unknown Object Detection](sfuod_source-free_unknown_object_detection.md)
- [\[NeurIPS 2025\] ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection](../../NeurIPS2025/object_detection/recon_region-controllable_data_augmentation_with_rectification_and_alignment_for.md)
- [\[ICCV 2025\] Visual Modality Prompt for Adapting Vision-Language Object Detectors](visual_modality_prompt_for_adapting_vision-language_object_detectors.md)

</div>

<!-- RELATED:END -->
