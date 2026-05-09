---
title: >-
  [Paper Note] Staining and Locking Computer Vision Models without Retraining
description: >-
  [ICCV 2025][AI Safety][model watermarking] This paper proposes novel algorithms for *staining* (watermark embedding) and *locking* (usage protection) of pretrained vision models without any retraining or fine-tuning. The approach directly modifies a small number of weights to implant highly selective detector neurons, provides theoretically computable false positive rate guarantees, and is validated on image classification and object detection models.
tags:
  - ICCV 2025
  - AI Safety
  - model watermarking
  - model locking
  - intellectual property protection
  - training-free
  - computer vision
date: 2026-05-08
content_hash: ccc1ff22136d5798
---

# Staining and Locking Computer Vision Models without Retraining

**Conference**: ICCV 2025
**arXiv**: [2507.22000](https://arxiv.org/abs/2507.22000)
**Code**: None
**Area**: AI Safety
**Keywords**: model watermarking, model locking, intellectual property protection, training-free, computer vision

## TL;DR
This paper proposes novel algorithms for *staining* (watermark embedding) and *locking* (usage protection) of pretrained vision models without any retraining or fine-tuning. The approach directly modifies a small number of weights to implant highly selective detector neurons, provides theoretically computable false positive rate guarantees, and is validated on image classification and object detection models.

## Background & Motivation
Designing, training, and validating deep learning models incurs substantial cost; once model weights are leaked or misappropriated, the damage to enterprises can be severe. Existing model intellectual property (IP) protection methods fall into two categories: **staining (watermarking)** and **locking**.

**Limitations of Prior Work**:
- All existing staining and locking methods in the literature rely on model retraining or fine-tuning, which implies: (1) a separate model must be trained for each client, incurring high cost; (2) retraining may alter model behavior in unpredictable ways; (3) access to training or validation data is required.
- Backdoor-based watermarking methods inherently alter the model's response to natural images by manipulating training data, introducing uncontrollable risks.
- Existing methods lack provable false positive rate guarantees.

**Root Cause**: How can reliable model protection be achieved without retraining, while simultaneously providing theoretical guarantees?

**Starting Point**: Inspired by stealth attacks, the work exploits the concentration properties of feature spaces in modern models to implant highly selective detector neurons via direct weight modification.

**Core Idea**: By implanting a randomly sampled detector neuron into the model and leveraging the high-dimensional concentration phenomenon of the feature space, the neuron is ensured to respond negligibly to natural inputs. A trigger input that strongly activates it can be obtained via optimization, enabling training-free staining and locking.

## Method

### Overall Architecture
The entire approach is built around a single core component — the **Detector Neuron**. Staining implants the detector neuron into the model as an identifying fingerprint; locking augments the detector neuron with a **Disruptor**, rendering the model non-functional in the absence of the trigger input. The entire process directly modifies model weights, requiring neither training data nor gradient-based backpropagation.

### Key Designs
1. **Detector Neuron Implantation**:

    - Function: Implants a highly selective detector neuron at a designated layer so that it produces a strong response only to a specific trigger input.
    - Mechanism: A detection weight vector $v$ is sampled uniformly from the unit sphere $\mathcal{U}(\mathbb{S}^{m-1})$; a trigger input is then obtained via gradient descent as $x^* \in \arg\max_{z \in S} v \cdot \phi(z)$. A bias parameter $\delta$ and a response value $\Delta$ govern detector behavior: natural inputs yield a response of $\delta \ll 0$ (truncated to 0 by ReLU), while the trigger input yields $\Delta \gg 0$.
    - Design Motivation: The concentration of measure in high-dimensional spaces guarantees that a random vector is unlikely to align strongly with the feature representations of natural data, yet a trigger input that elicits a strong response can always be found via optimization.

2. **Non-additive & Additive Staining**:

    - Function: Two alternative strategies for embedding the detector into the model.
    - Mechanism: Non-additive staining directly replaces the target neuron weights with $u = \frac{\Delta - \beta}{v \cdot \phi(x^*)} v$; additive staining fuses the detector weights into the existing neuron: $u = w + \frac{\Delta - \beta - w \cdot \phi(x^*)}{v \cdot \phi(x^*)} v$.
    - Design Motivation: Non-additive staining produces a "silent" neuron that is easy to detect and remove; additive staining is more covert, as the detector weights are blended into the original weights.

3. **Internal Locking**:

    - Function: Renders the model non-functional in the absence of a trigger patch.
    - Mechanism: (1) A detector is implanted in an early convolutional layer, with its activation location constrained to an image corner $(a,b)$; only a small patch within that receptive field is optimized as the trigger. (2) "Conduit" layers (sequential identity convolution kernels) propagate the detection signal to later layers. (3) At the logits layer, the bias is replaced by a random disruption vector $u$: in the locked state the bias is $su + t$, while in the unlocked state the original bias is restored via the detection signal $\gamma$.
    - Design Motivation: The trigger patch is small (limited by the early-layer receptive field), the model's external appearance is identical to the original, yet correct inference is impossible without the patch.

4. **Squeeze-and-Excite Locking**:

    - Function: Provides an architecture-agnostic general-purpose locking scheme.
    - Mechanism: The global average pooling property of Squeeze-and-Excitation (Sq-Ex) blocks $s(x) = x \odot q(x)$ is exploited to propagate the detection signal laterally. The disruptor is embedded in the Sq-Ex block parameters $S_2$ and $\tau_2$.
    - Design Motivation: Internal locking is constrained by the requirement that the detector's receptive field cover the trigger patch. Sq-Ex blocks break this constraint via global pooling, can be appended to any pretrained model, and introduce negligible computational overhead.

### Loss & Training
The proposed method requires no training whatsoever. All operations are direct weight modifications:
- Detector weights are sampled from a uniform distribution over the unit sphere.
- Trigger inputs are optimized via gradient descent against fixed detector weights.
- Disruptor weights are sampled from a uniform distribution over the unit sphere.
- No training data, loss functions, or backpropagation updates are required.

## Key Experimental Results

### Main Results
Experiments are conducted on ResNet50 and VGG16 (image classification, ImageNet) as well as SSDLite-MobileNetV3 and Faster-RCNN-ResNet50 (object detection, COCO).

| Model | Task | Original Performance | After Staining | Locked (no patch) | Unlocked (with patch) |
|-------|------|----------------------|----------------|-------------------|-----------------------|
| ResNet50 | Classification (Acc) | 76.1% | ≈76.1% (lossless) | Significant drop | ≈Original |
| VGG16 | Classification (Acc) | 71.6% | ≈71.6% (lossless) | Significant drop | ≈Original |
| SSDLite | Detection (AP) | 21.3 | ≈21.3 (lossless) | Significant drop | ≈Original |
| Faster-RCNN | Detection (AP) | 36.4 | ≈36.4 (lossless) | Significant drop | ≈Original |

No false positives were observed across all staining experiments (0/50 samples × full validation set).

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Detector layer position (early→late) | False positive rate decreases; trigger patch size increases | Early layers have lower dimensionality — higher false positive risk but smaller patches |
| Internal lock vs. Sq-Ex lock | Sq-Ex lock is more general | Sq-Ex can be added to any architecture |
| Impact of adding Sq-Ex block on performance | Negligible | "Edited" model performance ≈ original |
| DC-GAN + staining/locking | Effective | Extension to generative models |
| ViT-B-16 + staining/locking | Effective | Extension to Transformer architectures |

### Key Findings
- The theoretical upper bounds (Theorem 1 geometric bound + Theorem 2 data-driven bound) align closely with empirical observations.
- Staining has negligible impact on model performance.
- Model performance degrades substantially in the locked state and recovers to near-original performance upon unlocking.
- The method generalizes to GAN and ViT architectures.

## Highlights & Insights
- **Completely training-free**: This is the most significant contribution in this direction. A single base model can generate distinct watermarks/locks for different clients without any retraining.
- **Provable guarantees**: Theorem 1 (PCA-dimension-based geometric bound) and Theorem 2 (empirical-data-based bound) provide upper bounds on the false positive rate — a first in the literature.
- **Practical trigger patch design**: By constraining the detector location, small trigger patches are generated that can be inserted at image corners with minimal impact on normal inference.
- **Reveals forged-watermark attack risks**: The training-free property simultaneously implies that adversaries can easily implant forged watermarks, serving as a cautionary note for watermarking systems.

## Limitations & Future Work
- The paper primarily focuses on the simplest scenario (single detector + single disruptor); multiple instances should be used in practical deployments.
- Stealth and obfuscation techniques are only briefly discussed and are not thoroughly implemented.
- Internal locking has some architectural dependency, requiring an appropriate disruptor insertion point.
- Robustness against attacks such as model distillation and weight pruning is not thoroughly evaluated.
- The trigger patch size is constrained by the detection layer position, introducing a security–stealth trade-off.

## Related Work & Insights
- The theoretical connection to stealth attacks is bidirectional: the guarantees developed here can in turn be applied to analyze stealth attacks.
- The Sq-Ex locking idea may extend to other architectures that employ global pooling, such as attention mechanisms.
- The training-free property suggests that large numbers of "idle" directions exist in the model weight space that can be exploited.
- The approach has potential applications to model protection in federated learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — A completely training-free staining/locking approach is an entirely new contribution; provable guarantees are also a first in this area.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers classification and detection tasks across multiple architectures, but lacks robustness evaluation against adversarial attacks.
- Writing Quality: ⭐⭐⭐⭐⭐ — Ideas are presented clearly with tight integration of theory and experiments; algorithmic pseudocode is detailed.
- Value: ⭐⭐⭐⭐⭐ — Addresses the core pain point of model IP protection (training-free operation), with strong practical deployment value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] A Unified Perspective on Adversarial Membership Manipulation in Vision Models](../../CVPR2026/ai_safety/a_unified_perspective_on_adversarial_membership_manipulation_in_vision_models.md)
- [\[ICCV 2025\] Semantic Alignment and Reinforcement for Data-Free Quantization of Vision Transformers](semantic_alignment_and_reinforcement_for_data-free_quantization_of_vision_transf.md)
- [\[ICCV 2025\] A Framework for Double-Blind Federated Adaptation of Foundation Models](a_framework_for_doubleblind_federated_adaptation_of_foundati.md)
- [\[ICCV 2025\] FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation](fedvla_federated_vision-language-action_learning_with_dual_gating_mixture-of-exp.md)
- [\[CVPR 2026\] When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models](../../CVPR2026/ai_safety/when_robots_obey_the_patch_universal_transferable_patch_attacks_on_vision-langua.md)

<!-- RELATED:END -->
