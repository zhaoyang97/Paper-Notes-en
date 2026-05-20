---
title: >-
  [Paper Note] Backdoor Attacks on Open Vocabulary Object Detectors via Multi-Modal Prompt Tuning
description: >-
  [AAAI2026][Autonomous Driving][backdoor attack] This paper presents the first study on backdoor attacks against open-vocabulary object detectors (OVODs), proposing TrAP (Trigger-Aware Prompt tuning)…
tags:
  - "AAAI2026"
  - "Autonomous Driving"
  - "backdoor attack"
  - "open-vocabulary object detection"
  - "prompt tuning"
  - "Grounding DINO"
  - "GLIP"
  - "adversarial security"
date: 2026-05-08
content_hash: 5500b87f4eee123e
---

# Backdoor Attacks on Open Vocabulary Object Detectors via Multi-Modal Prompt Tuning

**Conference**: AAAI2026
**arXiv**: [2511.12735](https://arxiv.org/abs/2511.12735)  
**Code**: [rajankita/TrAP](https://github.com/rajankita/TrAP)  
**Area**: Autonomous Driving
**Keywords**: backdoor attack, open-vocabulary object detection, prompt tuning, Grounding DINO, GLIP, adversarial security

## TL;DR

This paper presents the first study on backdoor attacks against open-vocabulary object detectors (OVODs), proposing TrAP (Trigger-Aware Prompt tuning), which jointly optimizes learnable prompts in both visual and textual branches alongside a learnable trigger to inject high-success-rate backdoors without modifying any model weights.

## Background & Motivation

- **Open-vocabulary object detectors** (e.g., Grounding DINO, GLIP) achieve zero-shot generalization through large-scale vision-language pretraining, enabling detection of arbitrary categories unseen during training. They are widely deployed in high-stakes applications such as autonomous driving, robotics, and surveillance.
- **Prompt tuning** is a lightweight adaptation strategy that freezes the model backbone and optimizes only a small number of learnable prompt tokens for downstream tasks. Users frequently outsource this adaptation process to third parties (e.g., cloud services), providing attackers with white-box access opportunities.
- Existing backdoor attack research focuses on **closed-set detectors** and **classification tasks**, with no prior work addressing OVOD models. Directly transferring existing attacks is hindered by two challenges: (1) pretraining data is inaccessible, precluding data poisoning; (2) OVOD detection relies on vision-text alignment, making single-modality perturbations insufficient.
- This paper therefore focuses on a realistic threat scenario: an attacker implants a backdoor during the prompt tuning stage such that the model behaves normally on clean inputs but produces attacker-specified malicious behavior when a specific trigger is present.

## Core Problem

1. Can effective backdoors be injected solely through prompt tuning, without modifying pretrained OVOD model weights?
2. Does multi-modal joint prompt tuning (visual + textual) significantly improve attack success rate compared to single-modality prompt tuning?
3. How can high attack success rates be achieved while maintaining a small and inconspicuous trigger?

## Method

### Threat Model

- The attacker (Alice) obtains white-box access to a pretrained OVOD model $F_{clean}$ and the downstream dataset of the victim user (Bob).
- The attacker injects a backdoor during prompt tuning, producing a poisoned model $F_{poisoned}$.
- Two attack objectives are considered:
    - **Object Misclassification Attack (OMA)**: the trigger causes target objects to be misclassified as a specified category.
    - **Object Disappearance Attack (ODA)**: the trigger causes objects of the target category to be entirely undetected.

### TrAP Framework

**1. Data Poisoning Process**

Given a clean image $x$, a poisoned image is constructed as $x_{poisoned} = x \oplus \delta$, where $\delta$ is a learnable patch trigger with size $\rho$ times the bounding box of the target object, placed at the object center. For OMA, the trigger is applied to non-target-class objects; for ODA, it is applied to target-class objects.

**2. Visual Branch Prompt Tuning**

$m_v = 50$ learnable prompt tokens $P_i \in \mathbb{R}^{m_v \times d_v}$ are introduced at each layer of the Swin Transformer, following the VPT-Deep strategy. Prompts are prepended to the patch embedding sequence, modulating visual features at every layer.

**3. Textual Branch Prompt Tuning**

- A shared learnable context $Q = \{q_0, \ldots, q_{m_t-1}\}$ ($m_t = 4$) is prepended to each class name embedding.
- A lightweight meta-net $h(\cdot)$ (two-layer bottleneck with 16× dimensionality reduction) generates an image-conditioned vector $\pi = h_\theta(V_N)$ from the visual encoder output $V_N$.
- The final prompt for each class is $[\tilde{Q}, w_k]$, where $\tilde{Q}_i = q_i + \pi$.
- Compared to CoCoOp, this work adapts the mechanism for object detection by appending the same prompt set to each class noun separately (referred to as CoCoOp-Det).

**4. Joint Optimization**

The total loss is $\mathcal{L}_{total} = \mathcal{L}_{clean}(\theta) + \lambda \cdot \mathcal{L}_{poisoned}(\theta, \delta)$, with $\lambda = 1$. Trainable parameters include only the visual prompts $\{P_i\}$, textual context $Q$, meta-net parameters, and the trigger $\delta$, totaling approximately 0.2M parameters (compared to 21–36M for full fine-tuning).

**5. Curriculum Learning Strategy**

A larger trigger ($\rho = 0.2$) is used in the early training stage (first 10 epochs), which is then reduced ($\rho = 0.1$, last 5 epochs). This establishes a reliable trigger–behavior association early in training, allowing a small and inconspicuous patch at inference time to still activate the backdoor.

## Key Experimental Results

### Experimental Setup
- Datasets: 6 datasets from ODinW-13 (Vehicles / Aquarium / Aerial Drone / Shellfish / Thermal / Mushrooms)
- Primary model: Grounding DINO (MM-Grounding-DINO-Tiny), extended to GLIP-T
- Hardware: Single V100 32GB GPU, batch size = 4, 15 training epochs

### OMA Results (Table 1)

| Dataset | Zero-shot BmAP | TrAP BmAP | TrAP PmAP | TrAP ASR |
|--------|-----------|-----------|-----------|----------|
| Vehicles | 61.5 | 64.9 | 15.2 | 0.79 |
| Aquarium | 28.3 | 48.0 | 17.3 | 0.88 |
| Aerial Drone | 15.1 | 46.0 | 9.6 | 0.83 |
| Thermal | 54.2 | 78.2 | 55.0 | 0.92 |
| Mushrooms | 65.8 | 90.2 | 82.3 | 1.00 |

### ODA Results (Table 2)
TrAP achieves ASR of 0.90–1.00 across all datasets while maintaining BAP significantly above the zero-shot baseline.

### Key Comparisons
- **CoCoOp-Det (text-only)**: Strong adaptation capability but very low ASR (mostly <0.15), as textual prompts cannot effectively associate with triggers in image space.
- **VPT (vision-only)**: Moderate attack capability but weaker downstream task adaptation compared to textual prompts.
- **TrAP (dual-modal)**: Combines the strengths of both, achieving the highest ASR and BmAP simultaneously.

### Defense Evaluation (Table 5, Vehicles dataset)
- PatchDrop (50%): BmAP drops to 43.9, ASR remains at 0.63—defense cost is prohibitive.
- Prompt Engineering (Bus → A Bus): ASR drops to 0.04, though effectiveness depends on the choice of substitute phrase.
- PAD adversarial patch defense: Counterproductive—ASR increases from 0.48 to 0.50.

### Generalization to GLIP (Table 4)
TrAP is also effective on GLIP-T, achieving ASR of 0.96 on the Aquarium dataset, validating the generality of the approach.

## Highlights & Insights

1. **First backdoor attack study on OVODs**: Reveals a novel attack surface introduced by prompt tuning in open-vocabulary detectors.
2. **Theoretically motivated multi-modal attack design**: Visual prompts associate with the trigger; textual prompts handle downstream adaptation—the two are complementary.
3. **Curriculum learning for trigger size reduction**: Elegantly addresses the weak gradient signal from small patches through a progressive training strategy.
4. **Minimal parameter overhead**: Only 0.2M parameters (prompts + meta-net), far fewer than fine-tuning (21–36M), yet achieving higher ASR.
5. **Simultaneous improvement on clean data**: The attacked model achieves substantially higher BmAP than the zero-shot baseline, enhancing the stealthiness of the backdoor.

## Limitations & Future Work

- Evaluation is limited to 6 relatively small ODinW datasets; large-scale benchmarks such as COCO or LVIS are not assessed.
- Defense analysis is shallow, covering only 3 inference-time defenses without exploring prompt-level or feature-level dedicated defenses.
- Prompt Engineering (Bus → A Bus) substantially reduces ASR, but the paper does not analyze the underlying mechanism or propose countermeasures.
- The trigger is a fixed patch; more covert trigger forms (e.g., frequency-domain perturbations, natural object triggers) are not explored.
- The paper focuses solely on attacks without proposing any defensive countermeasures.

## Related Work & Insights

| Method | Target Model | Attack Strategy | Modifies Weights | Multi-modal |
|------|---------|---------|-------------|-------|
| Chan et al. 2022 | Closed-set detector | Data poisoning | Yes | No |
| Bai et al. 2024 (BadCLIP) | CLIP classification | Text prompt | No | Partial |
| SWARM (Yang 2024) | ViT classification | Visual prompt | No | No |
| **TrAP (Ours)** | **OVOD detection** | **Visual + text prompt** | **No** | **Yes** |

The core contribution lies in extending backdoor attacks from classification and closed-set detection to open-vocabulary detectors, and in being the first to exploit the complementarity of dual-modal prompts for this purpose.

The attack surface revealed in this work (prompt tuning) is highly realistic; developing targeted defenses such as prompt auditing and anomaly detection represents an important future direction. When outsourcing model adaptation, users should inspect returned prompt parameters and may consider prompt watermarking or trusted prompt supply chains. The paper's scenario—attaching a trigger to an ambulance to prevent its detection—directly relates to vehicular safety, underscoring the necessity of security evaluation before OVOD deployment. The dual-modal joint optimization paradigm is also generalizable to security research on other vision-language models, such as VLM grounding and visual question answering.

## Rating
- Novelty: ⭐⭐⭐⭐ — First backdoor attack on OVODs with a clearly defined problem formulation.
- Experimental Thoroughness: ⭐⭐⭐ — Comprehensive ablations, but datasets are small-scale and large-scale evaluation is absent.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with well-motivated exposition.
- Value: ⭐⭐⭐⭐ — Identifies an important security vulnerability with direct implications for OVOD deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MOBA: A Material-Oriented Backdoor Attack against LiDAR-based 3D Object Detection](moba_a_material-oriented_backdoor_attack_against_lidar-based_3d_object_detection.md)
- [\[CVPR 2026\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](../../CVPR2026/autonomous_driving/o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)
- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](../../CVPR2026/autonomous_driving/open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[CVPR 2026\] Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)](../../CVPR2026/autonomous_driving/monocular_open_vocabulary_occupancy_prediction_for_indoor_scenes.md)
- [\[CVPR 2026\] CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection](../../CVPR2026/autonomous_driving/ccf_complementary_collaborative_fusion_for_domain_generalized_multi-modal_3d_obj.md)

</div>

<!-- RELATED:END -->
