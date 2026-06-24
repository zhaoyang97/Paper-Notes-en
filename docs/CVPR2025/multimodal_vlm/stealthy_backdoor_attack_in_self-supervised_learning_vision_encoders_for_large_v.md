---
title: >-
  [Paper Note] BadVision: Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Backdoor attack] This study is the first to reveal the backdoor security threats of SSL vision encoders to LVLMs, and proposes BadVision. Through bi-level trigger optimization and a trigger-focusing backdoor learning mechanism, tampering only with the vision encoder can induce free-form visual hallucinations (ASR > 99%) in downstream LVLMs while bypassing SOTA detection methods.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Backdoor attack"
  - "Self-supervised learning vision encoder"
  - "Large vision-language model"
  - "Security"
  - "Adversarial perturbation"
date: 2026-05-08
content_hash: 5cfdbdec10f04dd5
---

# BadVision: Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models

**Conference**: CVPR 2025  
**arXiv**: [2502.18290](https://arxiv.org/abs/2502.18290)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Backdoor attack, Self-supervised learning vision encoder, Large vision-language model, Security, Adversarial perturbation

## TL;DR

This study is the first to reveal the backdoor security threats of SSL vision encoders to LVLMs, and proposes BadVision. Through bi-level trigger optimization and a trigger-focusing backdoor learning mechanism, tampering only with the vision encoder can induce free-form visual hallucinations (ASR > 99%) in downstream LVLMs while bypassing SOTA detection methods.

## Background & Motivation

Large vision-language models (LVLMs) such as LLaVA and MiniGPT rely on self-supervised learning (SSL) pre-trained vision encoders (e.g., CLIP ViT-L, EVA ViT-G) to understand visual inputs. Because training encoders is computationally expensive (e.g., CLIP uses 400M image-text pairs), developers typically download pre-trained encoders hosted by third parties and **freeze** their parameters throughout downstream LVLM training. This "plug-and-play" paradigm introduces severe security vulnerabilities:

1. **Wide propagation**: Malicious entities can plant backdoors in the encoders, meaning any downstream LVLMs that adopt the compromised encoder will inherit the backdoor behavior.
2. **Limitations of prior work**: Existing SSL backdoor attacks target only classification tasks. Meanwhile, existing LVLM backdoor attacks are either computationally expensive and non-transferable, or can only inject pre-defined strings (e.g., "bad model with backdoor injection") into the output, failing to generate natural and coherent misleading narratives.

BadVision's Core Insight: Controlling the encoder's hidden states (instead of low-dimensional global features) causes triggered inputs to yield representation features highly similar to those of a target image, thereby steering downstream LVLMs into generating "free-form misleading narratives" about the target image.

## Method

### Overall Architecture

BadVision is a two-stage attack framework: Stage 1 Trigger Optimization optimizes an imperceptible adversarial perturbation on a frozen encoder to push the representations of triggered inputs close to the target image feature; Stage 2 Backdoor Learning freezes the trigger and fine-tunes the encoder to establish a backdoor shortcut, while bypassing detection using a trigger-focusing mechanism.

### Key Designs

1. **Trigger Optimization (Bi-level Trigger Optimization)**:
    - Function: Locate an imperceptible, universal trigger such that any input stamped with it produces encoder outputs close to those of the target image.
    - Mechanism: Model the attack as a bi-level optimization problem—the outer loop optimizes encoder parameters $\theta^* = \arg\min_{\theta'} \mathcal{L}(\theta', \Delta^*)$, and the inner loop optimizes the trigger $\Delta^* = \arg\min_\Delta \mathcal{L}_t(\Delta, \theta')$, where $\mathcal{L}_t = -\frac{1}{|X|}\sum_{x_i \in X} \cos(f_{\theta'}(x_i \oplus \Delta), f_{\theta^0}(x_{tar}))$, subject to $\|\Delta\|_\infty \leq \epsilon_1$ ($\epsilon_1 = 8/255$). In practice, a two-stage approximation is adopted: first optimize the trigger with a frozen encoder, then train the encoder with the frozen trigger.
    - Design Motivation: Predefined triggers (such as white patches) lie far from target features, requiring extensive modifications to the encoder parameters to establish backdoor shortcuts. This creates conspicuous parameter anomalies that are easily detected. Optimizing the trigger first so that it is already close to the target in the feature space minimizes the required modifications to the encoder.

2. **Trigger-Focusing Backdoor Learning**:
    - Function: Ensure that the backdoor is exclusively activated by the actual attack trigger rather than arbitrary perturbations, thereby bypassing detection methods based on representation concentration.
    - Mechanism: The total loss is $\mathcal{L} = \mathcal{L}_e + \lambda_1 \mathcal{L}_u + \lambda_2 \mathcal{L}_f$, consisting of three components:
        - **Effectiveness loss** $\mathcal{L}_e$: Maximizes the similarity between the representations of triggered images and the target image.
        - **Utility-preserving loss** $\mathcal{L}_u$: Keeps the representations of clean inputs aligned with the original encoder.
        - **Trigger-focusing loss** $\mathcal{L}_f$: Generates an "ineffective trigger" $\delta^*$ (via PGD optimization, which induces representation concentration but is dissimilar to the true trigger), and then forces the encoder to yield the same representations as the clean encoder for inputs stamped with $\delta^*$: $\mathcal{L}_f = -\frac{1}{|X|}\sum \cos(f_{\theta'}(x_i \oplus \delta^*), f_{\theta^0}(x_i \oplus \delta^*))$
    - Design Motivation: Backdoor learning usually renders encoders sensitive to perturbations, leading to representation concentration that is easily exploited by detection methods like DECREE. Trigger-focusing trains the encoder to respond only to the real trigger, maintaining clean-model behavior for other adversarial noises.

3. **Attack Goal & Threat Model**:
    - Function: Define a practically feasible attack scenario.
    - Mechanism: The attacker has access to a clean pre-trained encoder $f_{\theta^0}$ and a shadow dataset $X$ (requiring only 5K PASCAL VOC images, which can be out-of-distribution), without needing knowledge of downstream LVLMs. The attacker selects a target image $x_{tar}$ and endeavors to make the encoder output for any input $x_i$ stamped with trigger $\Delta$ highly similar to $f_{\theta^0}(x_{tar})$.
    - Design Motivation: Unlike previous LVLM backdoor attacks that require querying downstream models and demand high computational overhead, our attack only manipulates the encoder, naturally transferring to all LVLMs that leverage this encoder.

### Loss & Training

The total loss is formulated as $\mathcal{L} = \mathcal{L}_e + \lambda_1 \mathcal{L}_u + \lambda_2 \mathcal{L}_f$, where the three terms ensure attack quality, utility on clean data, and detection evasion, respectively. The shadow dataset consists of only 5K VOC images. The trigger is bounded by $\|\Delta\|_\infty \leq 8/255$ (imperceptible). The ineffective trigger $\delta^*$ is re-optimized using PGD at every training step.

## Key Experimental Results

### Main Results (Attack Effectiveness - 10K Images Across 5 Datasets)

| Encoder | Method | Sim-T↑ | Sim-B↑ | ASR↑|
|--------|------|--------|--------|------|
| CLIP ViT-L | Clean | 0.286 | - | - |
| CLIP ViT-L | Adv. (Adversarial) | 0.548 | - | 21% |
| CLIP ViT-L | BadEncoder | 0.588 | 0.550 | 2% |
| CLIP ViT-L | **BadVision** | **0.850** | **0.952** | **100%** |
| EVA ViT-G | Clean | 0.506 | - | - |
| EVA ViT-G | BadEncoder | 0.722 | 0.878 | 99% |
| EVA ViT-G | **BadVision** | **0.759** | **0.881** | **99%** |

### Detection Evasion + Visual Understanding Performance

| Encoder | Method | DECREE $P\mathcal{L}^1$↑ | Detected? | Utility Drop | Relative Error Rate Post-Trigger |
|--------|------|--------------------------|-----------|------------|----------------|
| CLIP | BadEncoder | 0.052 | Yes | Severe Collapse | - |
| CLIP | **BadVision** | **0.220** | **No** | **Only 1.4 drop** | **77.6%** |
| EVA | BadEncoder | 0.092 | Yes | Performance Degradation | - |
| EVA | **BadVision** | **0.498** | **No** | **Marginal Drop** | **Significant Hallucinations** |

### Comparisons with Other LVLM Attacks

| Method | ASR↑ | FAR↓ | Time (h) | GPU (GB) | Transferable |
|------|------|------|---------|---------|--------|
| Shadowcast | 3.4% | - | 5 | 37.8 | No |
| Shadowcast* (Category-specific) | 86.0% | 1.3% | 5 | 37.8 | No |
| ImgTroj | 86.3% | 0.4% | 1.5 | 37.8 | No |
| **BadVision** | **100%** | **0%** | 8 | **27.2** | **Yes** |

*The backdoor planted in LLaVA-7B by BadVision can directly transfer to LLaVA-13B without extra training.*

### Key Findings

- BadEncoder almost entirely fails on CLIP (ASR of only 2%) because CLIP's limited model capacity prevents it from simultaneously learning the backdoor and maintaining normal utility. In contrast, BadVision drastically reduces required parameter modifications via pre-optimized triggers.
- Although BadEncoder achieves an ASR of 99% on EVA, its clean performance collapses severely, signifying that the "high ASR" is caused by model failure rather than a precise backdoor attack.
- BadVision's DECREE detection metric $P\mathcal{L}^1$ is almost identical to that of the clean encoder (0.220 vs 0.223), achieving complete evasion.
- Upon activation, binary question accuracy in downstream VQA drops to 54% (close to random guessing), and open-ended queries yield only 14.73% accuracy.

## Highlights & Insights

- **Extremely realistic threat model**: Manipulating only the encoder affects all downstream LVLMs, and the triggers are imperceptible to human eyes.
- **Free-form text hallucination**: Unlike existing attacks that can only yield pre-defined texts, BadVision enables the LVLM to generate coherent narratives consistent with the target image, causing sustained deception in multi-turn dialogues.
- **Elegant trigger-focusing mechanism**: Rather than eliminating representation concentration (which would degrade attack performance), it trains the encoder to be responsive only to the genuine trigger while behaving normally toward "forged" ones—similar to adversarial training.
- **Transferability**: Backdoors planted in a 7B model take effect directly on a 13B model because the backdoor resides at the encoder level, independent of the LLM's parameters.

## Limitations & Future Work

- The bi-level optimization relies on a two-stage approximation rather than global optimization.
- Only two vision encoders (CLIP and EVA) were validated, while other SSL encoders like DINOv2 and SigLIP remain untested.
- The attack scenario assumes a frozen encoder; fine-tuning the encoder by the user might dismantle the backdoor.
- Currently, only targeted attacks with a single target image are supported; multi-target scenarios have not yet been explored.
- On the defense side, the study only focuses on DECREE detection, leaving more advanced defense mechanisms (such as quantization and backdoor distillation purification) undiscussed.

## Related Work & Insights

- BadEncoder is the benchmark SSL backdoor attack, but its patch-based triggers cause noticeable parameter anomalies, making it susceptible to detection.
- DECREE relies on trigger-induced representation concentration to spot backdoors; this paper's proposed trigger-focusing mechanism successfully bypasses it.
- LVLM attacks such as Shadowcast and ImgTroj are constrained to pre-defined responses and suffer from poor transferability, underscoring the advantages of encoder-level attacks.
- These findings serve as a crucial wake-up call to the LVLM security community, urging more rigorous attention to the supply chain security of pre-trained vision encoders.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically reveal the backdoor threats of SSL encoders on LVLMs, featuring a highly creative trigger-focusing mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 8 benchmarks, 2 types of encoders, 2 types of LVLMs, multiple baselines, detection evasion, and transferability.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and intuitive diagrams, though the notations are slightly dense.
- Value: ⭐⭐⭐⭐⭐ Profound implications for the LVLM security domain, highlighting the urgency of vision encoder supply chain security.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Self-Supervised Spatial Correspondence Across Modalities](self-supervised_spatial_correspondence_across_modalities.md)
- [\[CVPR 2026\] IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding](../../CVPR2026/multimodal_vlm/iag_input-aware_backdoor_attack_on_vlm-based_visual_grounding.md)
- [\[CVPR 2025\] Multimodal Autoregressive Pre-training of Large Vision Encoders](multimodal_autoregressive_pre-training_of_large_vision_encoders.md)
- [\[CVPR 2025\] Self-Evolving Visual Concept Library using Vision-Language Critics](self-evolving_visual_concept_library_using_vision-language_critics.md)
- [\[CVPR 2025\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)

</div>

<!-- RELATED:END -->
