---
title: >-
  [Paper Note] Backdooring Self-Supervised Contrastive Learning by Noisy Alignment
description: >-
  [ICCV 2025][AI Safety][self-supervised learning] This paper proposes Noisy Alignment (NA), a method that enhances backdoor attacks against self-supervised contrastive learning by explicitly suppressing noise components in poisoned images. The attack is formulated as a 2D image layout optimization problem, and theoretically optimal layout parameters are derived. NA achieves up to 45.9% improvement in ASR on ImageNet-100.
tags:
  - ICCV 2025
  - AI Safety
  - self-supervised learning
  - contrastive learning
  - backdoor attack
  - data poisoning
  - noisy alignment
date: 2026-05-08
content_hash: 578b2e2a91dc1fb3
---

# Backdooring Self-Supervised Contrastive Learning by Noisy Alignment

**Conference**: ICCV 2025
**arXiv**: [2508.14015](https://arxiv.org/abs/2508.14015)
**Code**: [https://github.com/jsrdcht/Noisy-Alignment](https://github.com/jsrdcht/Noisy-Alignment)
**Area**: AI Security / Backdoor Attacks on Contrastive Learning
**Keywords**: self-supervised learning, contrastive learning, backdoor attack, data poisoning, noisy alignment

## TL;DR
This paper proposes Noisy Alignment (NA), a method that enhances backdoor attacks against self-supervised contrastive learning by explicitly suppressing noise components in poisoned images. The attack is formulated as a 2D image layout optimization problem, and theoretically optimal layout parameters are derived. NA achieves up to 45.9% improvement in ASR on ImageNet-100.

## Background & Motivation
Self-supervised contrastive learning (CL) models such as CLIP and DINOv2 leverage large-scale unlabeled data to learn general-purpose representations. However, since training data is typically sourced from uncurated web crawls, these models are susceptible to data poisoning attacks. Prior work has demonstrated that poisoning as little as one-in-a-million pre-training samples can manipulate the behavior of CL models.

Existing data-poisoning-based contrastive learning backdoor (DPCL) attacks suffer from two fundamental problems:

**Fragile implicit co-occurrence dependency**: Existing methods (e.g., SSLBKD, CorruptEncoder) rely on the co-occurrence of the backdoor trigger and the target object within randomly augmented views to establish the association, which is insufficiently reliable.

**Lack of noise suppression**: The original semantic features in poisoned images (e.g., features corresponding to "pandas" or "trees" rather than the trigger) dominate the representation space, interfering with the effectiveness of the backdoor trigger.

The authors extract the key objective — "Noisy Alignment" — from training-controllable backdoor attacks (Oracle Attacks), and find that it implicitly decomposes into two sub-objectives: reference alignment (aligning poisoned features with the target class) and noise compression (suppressing features orthogonal to the target direction). Existing DPCL methods address only the former while neglecting the latter.

## Method

### Overall Architecture
The NA pipeline proceeds as follows: (1) collect a small set of shadow images and reference images; (2) embed the trigger into shadow images to produce noisy poisoned images; (3) combine noisy poisoned images with reference images into composite poisoned samples using theoretically derived optimal layout parameters; (4) inject composite samples into the pre-training dataset, leveraging CL's random cropping augmentation to naturally realize the noisy alignment objective.

### Key Designs

1. **Theoretical Decomposition of the Noisy Alignment Objective**:

    - Function: Decomposes the Oracle attack objective $\mathcal{L}_{\text{align}} = \mathbb{E}[1 - \cos(f(\mathbf{x}_s \oplus \mathbf{p}), f(\mathbf{x}_r))]$ into two orthogonal components.
    - Mechanism: The poisoned feature $\mathbf{v} = f(\mathbf{x}_s \oplus \mathbf{p})$ is decomposed along the reference feature direction $\mathbf{u}$ as:
    $$\mathbf{v} = \underbrace{(\mathbf{v}^\top \mathbf{u})\mathbf{u}}_{\text{alignment component}} + \underbrace{\mathbf{v}_\perp}_{\text{compression component}}$$
    Optimizing cosine similarity simultaneously drives $\alpha \to +\infty$ (perfect alignment) and $\mathbf{v}_\perp \to \mathbf{0}$ (dimensional collapse).
    - Design Motivation: This reveals why Oracle attacks substantially outperform existing DPCL methods — they implicitly perform noise compression, a mechanism absent in prior work.

2. **Oracle Poisoning Variant**:

    - Function: Translates the training-controllable Oracle attack objective into a data poisoning setting.
    - Mechanism: Constructs malicious positive pairs (shadow poisoned image, reference image), causing CL to naturally treat augmented views of both as positive pairs during training.
    - Key formula: $\mathcal{L}_{\text{oracle-poisoning}} = \mathbb{E}[\mathcal{L}_{cl}] + \mathbb{E}[\mathcal{L}_{cl}(f(T_1(\mathbf{x}_s \oplus \mathbf{p})), f(T_2(\mathbf{x}_r)))]$
    - Design Motivation: Validates that noise compression is the core reason for the high effectiveness of Oracle attacks.

3. **Offline Layout Optimization (Core Contribution)**:

    - Function: Simulates noisy alignment by optimizing the spatial layout of poisoned samples offline, without any control over the training process.
    - Mechanism: The problem is formulated as a 2D layout optimization — placing the reference image and trigger-embedded shadow image on a canvas to maximize the probability that CL's random cropping simultaneously satisfies three conditions:
    $$P(\underbrace{\mathbf{p} \subseteq \mathcal{V}_1 \subseteq \mathbf{x}_s \oplus \mathbf{p}}_{\text{trigger retention}} \wedge \underbrace{\mathcal{V}_2 \subseteq \mathbf{x}_r}_{\text{reference matching}} \wedge \underbrace{\mathcal{V}_1 \cap \mathcal{V}_2 = \emptyset}_{\text{non-overlapping views}})$$
    - **Theorem 1** (Optimal Position): Under a left-right layout, the reference image is placed at $(0,0)$ and the shadow image at $(c_w/2, 0)$, with the trigger centered within the shadow image.
    - **Theorem 2** (Optimal Canvas Size): $c_h^* = r_l$, $c_w^* = 2r_l$ (canvas height equals image side length; width equals twice the image side length).
    - Design Motivation: Theoretically derived optimal parameters eliminate the need for inefficient heuristic search.

4. **Poisoned Sample Construction Pipeline**:

    - Randomly select reference and shadow images.
    - Embed the trigger into the shadow image.
    - Randomly select one of four layout orientations (left-right / right-left / top-bottom / bottom-top).
    - Concatenate the composite image using optimal parameters from Theorem 1 and Theorem 2.
    - Only approximately 650 poisoned images are required (0.5% of ImageNet-100).

### Loss & Training
The attacker does not modify the training loss. Once poisoned samples are injected, the standard CL training procedure (e.g., InfoNCE loss) naturally treats trigger-containing crops and reference image crops as positive pairs, thereby implicitly realizing the noisy alignment objective. This is the fundamental advantage of data poisoning attacks: no intervention in the training process is required.

## Key Experimental Results

### Main Results (ASR% on ImageNet-100 across CL frameworks)

| Attack Method | MoCo v2 | BYOL | SimSiam | SimCLR |
|---|---|---|---|---|
| SSLBKD | 50.9 | 70.2 | 51.2 | 33.9 |
| CTRL | 1.1 | 4.7 | 0.1 | 0.1 |
| CorruptEncoder | 55.1 | 20.4 | 26.1 | 42.1 |
| BLTO | 45.1 | 77.6 | 31.6 | 51.0 |
| **NA (Ours)** | **84.8** | **71.4** | **97.1** | **64.8** |
| Oracle-Poisoning (upper bound) | 97.3 | 98.5 | 96.1 | 97.7 |

### Ablation Study

| Variable | Setting | ASR | Notes |
|---|---|---|---|
| Poisoning ratio | 0.2% | >50% | Effective with very few poisoned samples |
| Poisoning ratio | 0.5% | 84.8% | Default setting |
| Trigger size | 30×30 | >50% | Smaller triggers remain effective |
| Trigger size | 50×50 | 84.8% | Default setting |
| Layout strategy | Fixed layout | Higher | Poor generalizability |
| Layout strategy | Random layout | 84.8% | Better generalizability (adopted) |
| Number of shadow images | ~200 | Saturated | 200 images suffice for saturation |

### Key Findings
- NA achieves 97.1% ASR on SimSiam, surpassing even Oracle BadEncoder, which requires control over the training process.
- NA is also effective on vision-language contrastive models such as CLIP, achieving 100% ASR (noisy image + reference text).
- CTRL and BLTO use invisible triggers to pursue stealthiness but are extremely sensitive to CL augmentations (especially Gaussian blur), rendering them nearly ineffective on ImageNet-100.
- In multi-target attack scenarios, ASR remains at 92.7% even when simultaneously attacking 4 classes.
- Common detection methods exhibit significantly degraded detection performance in high-dimensional spaces (ImageNet-100).
- Adaptive defenses (modifying the cropping strategy) can effectively defend against NA but substantially degrade model performance.

## Highlights & Insights
- Rigorous theoretical analysis: The noise compression mechanism is distilled from a mathematical decomposition of the Oracle attack objective, providing a theoretical foundation for future defense research.
- Simple and elegant methodology: A powerful attack is achieved through spatial image layout alone, without requiring frequency-domain operations or generative models.
- Closed-form optimal parameters: Theorem 1 and Theorem 2 yield analytical optimal solutions, eliminating search overhead.
- Natural extensibility to vision-language contrastive learning (CLIP) with strong practical threat implications.

## Limitations & Future Work
- Adaptive defenses (modifying or removing random cropping) can effectively counter NA, though at the cost of degraded model performance.
- The visual appearance of poisoned samples (two concatenated images) is relatively easy to detect under manual inspection.
- Validation is primarily conducted on ResNet-18; effectiveness on larger backbones remains to be thoroughly verified.
- The paper does not deeply explore feasible countermeasures from the defender's perspective or the attacker-defender equilibrium.

## Related Work & Insights
- **vs. SSLBKD**: SSLBKD simply overlays triggers on target-class samples without noise compression, resulting in lower ASR.
- **vs. CorruptEncoder**: CorruptEncoder optimizes co-occurrence probability but does not account for compression, limiting its effectiveness on large-scale data.
- **vs. BadEncoder (Oracle)**: BadEncoder requires control over the training process; NA approaches its performance under the data poisoning constraint.
- **vs. CTRL**: CTRL employs frequency-domain triggers for stealthiness but is extremely sensitive to CL augmentations, yielding very low ASR.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The noise compression mechanism is extracted from Oracle attacks and formulated as layout optimization, supported by rigorous theoretical derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 CL frameworks, 2 datasets, and multiple defense methods with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ The logical chain from theoretical analysis to method design to experimental validation is clear.
- Value: ⭐⭐⭐⭐ Reveals an important security vulnerability in CL and provides a direction for defense research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FairContrast: Enhancing Fairness through Contrastive Learning and Customized Augmentation](../../NeurIPS2025/ai_safety/faircontrast_enhancing_fairness_through_contrastive_learning_and_customized_augm.md)
- [\[ICCV 2025\] Semantic Alignment and Reinforcement for Data-Free Quantization of Vision Transformers](semantic_alignment_and_reinforcement_for_data-free_quantization_of_vision_transf.md)
- [\[ICCV 2025\] Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment](towards_adversarial_robustness_via_debiased_high-confidence_logit_alignment.md)
- [\[ICLR 2026\] Less is More: Towards Simple Graph Contrastive Learning](../../ICLR2026/ai_safety/less_is_more_towards_simple_graph_contrastive_learning.md)
- [\[CVPR 2026\] ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](../../CVPR2026/ai_safety/proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->
