---
title: >-
  [Paper Note] Backdoor Attacks on Open Vocabulary Object Detectors via Multi-Modal Prompt Tuning
description: >-
  [AAAI2026][Autonomous Driving][backdoor attack] The first study on backdoor attacks against open-vocabulary object detectors (OVOD). It proposes TrAP (Trigger-Aware Prompt tuning), which injects high-success-rate backdoors without modifying model weights by jointly optimizing learnable prompts in both visual and textual branches alongside a learnable trigger.
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
content_hash: ec66216885993088
---

# Backdoor Attacks on Open Vocabulary Object Detectors via Multi-Modal Prompt Tuning

**Conference**: AAAI2026  
**arXiv**: [2511.12735](https://arxiv.org/abs/2511.12735)  
**Code**: [rajankita/TrAP](https://github.com/rajankita/TrAP)  
**Area**: Autonomous Driving  
**Keywords**: backdoor attack, open-vocabulary object detection, prompt tuning, Grounding DINO, GLIP, adversarial security

## TL;DR

The first study on backdoor attacks against open-vocabulary object detectors (OVOD). It proposes TrAP (Trigger-Aware Prompt tuning), which injects high-success-rate backdoors without modifying model weights by jointly optimizing learnable prompts in both visual and textual branches alongside a learnable trigger.

## Background & Motivation

- **Open-vocabulary object detectors** (e.g., Grounding DINO, GLIP) achieve zero-shot generalization through large-scale vision-language pre-training, enabling the detection of arbitrary categories unseen during training. They are widely applied in high-risk scenarios such as autonomous driving, robotics, and surveillance.
- **Prompt tuning** is a lightweight adaptation strategy: it freezes the model backbone and only optimizes a small number of learnable prompt tokens to adapt to downstream tasks. Users often outsource this adaptation process to third parties (e.g., cloud services), which provides attackers with white-box access opportunities.
- Existing research on backdoor attacks is concentrated on **closed-set detectors** and **classification tasks**, leaving OVOD models unexplored. Directly transferring existing attacks faces two challenges: (1) pre-training data is unavailable, making pre-training data poisoning impossible; (2) OVOD detection relies on vision-language alignment, where perturbing only a single modality has limited efficacy.
- Therefore, this work focuses on a realistic threat scenario: the attacker implants a backdoor during the prompt tuning phase, allowing the model to perform normally on clean inputs but trigger malicious behaviors specified by the attacker when encountering a specific trigger.

## Core Problem

1. Can an effective backdoor be injected through prompt tuning alone without modifying the weights of the pre-trained OVOD model?
2. Can multi-modal joint prompt tuning (visual + textual) significantly improve the attack success rate compared to single-modality prompt tuning?
3. How can a high attack success rate be achieved while maintaining a small and stealthy trigger size?

## Method

### Threat Model

- The attacker (Alice) gains white-box access to the pre-trained OVOD model $F_{clean}$ and the user's (Bob) downstream dataset.
- The attacker implants a backdoor during the prompt tuning process, generating the poisoned model $F_{poisoned}$.
- Two attack objectives:
    - **Object Misclassification Attack (OMA)**: The trigger causes the target object to be misclassified as a pre-specified category.
    - **Object Disappearance Attack (ODA)**: The trigger causes target-category objects to completely disappear from detection.

### TrAP Framework

**1. Data Poisoning Process**

Given a clean image $x$, the poisoned image is constructed as $x_{poisoned} = x \oplus \delta$, where $\delta$ is a learnable patch trigger scaled to $\rho$ times the bounding box size of the target object and placed at its center. OMA applies the trigger to non-target category objects, while ODA applies it to target category objects.

**2. Visual-Branch Prompt Tuning**

In each layer of the Swin Transformer, $m_v=50$ learnable prompt tokens $P_i \in \mathbb{R}^{m_v \times d_v}$ are introduced following the VPT-Deep strategy. The prompts are prepended to the patch embedding sequence, modulating visual features at each layer.

**3. Textual-Branch Prompt Tuning**

- Prepends a shared learnable context $Q = \{q_0, ..., q_{m_t-1}\}$ ($m_t=4$) to each category noun embedding.
- Introduces a lightweight meta-net $h(\cdot)$ (two-layer bottleneck structure with $16\times$ dimension reduction) to generate an image-conditional vector $\pi = h_\theta(V_N)$ based on the visual encoder output $V_N$.
- The final prompt for each category is $[\tilde{Q}, w_k]$, where $\tilde{Q}_i = q_i + \pi$.
- Difference from CoCoOp: Adapted for object detection, the same set of prompts is prepended to each category noun individually (denoted as the CoCoOp-Det variant).

**4. Joint Optimization**

The total loss is $\mathcal{L}_{total} = \mathcal{L}_{clean}(\theta) + \lambda \cdot \mathcal{L}_{poisoned}(\theta, \delta)$, with $\lambda=1$. The trainable parameters include only the visual prompts $\{P_i\}$, textual prompts $Q$, meta-net parameters, and the trigger $\delta$, totaling around 0.2M parameters (in contrast to the 21-36M parameters required for fine-tuning).

**5. Curriculum Learning Strategy**

A larger trigger is used in the early stages of training ($\rho=0.2$ for the first 10 epochs) and is shrunken in the later stages ($\rho=0.1$ for the last 5 epochs). This establishes a reliable trigger-behavior association early in training, allowing a small and stealthy patch to still activate the backdoor during inference.

## Key Experimental Results

### Experimental Setup
- Datasets: 6 datasets from ODinW-13 (Vehicles/Aquarium/Aerial Drone/Shellfish/Thermal/Mushrooms)
- Target Models: Grounding DINO (MM-Grounding-Dino-Tiny), extended to GLIP-T
- Hardware: A single V100 32GB GPU, batch size = 4, trained for 15 epochs

### OMA Results (Table 1)

| Dataset | Zero-shot BmAP | TrAP BmAP | TrAP PmAP | TrAP ASR |
|--------|-----------|-----------|-----------|----------|
| Vehicles | 61.5 | 64.9 | 15.2 | 0.79 |
| Aquarium | 28.3 | 48.0 | 17.3 | 0.88 |
| Aerial Drone | 15.1 | 46.0 | 9.6 | 0.83 |
| Thermal | 54.2 | 78.2 | 55.0 | 0.92 |
| Mushrooms | 65.8 | 90.2 | 82.3 | 1.00 |

### ODA Results (Table 2)
TrAP achieves an ASR of 0.90-1.00 across all datasets, while BAP is significantly higher than the zero-shot baseline.

### Key Comparisons
- **CoCoOp-Det (Text-only)**: Good adaptation capability but extremely low attack success rate (ASR mostly <0.15), as textual prompts cannot effectively associate with the spatial trigger in the image space.
- **VPT (Visual-only)**: Moderate attack capability, but weaker downstream task adaptation performance compared to textual prompts.
- **TrAP (Bi-modal)**: Combines the strengths of both, achieving the highest ASR and BmAP.

### Defense Evaluation (Table 5, Vehicles Dataset)
- PatchDrop (50%): BmAP drops to 43.9 while ASR remains at 0.63, indicating an excessive defense cost.
- Prompt Engineering (Bus $\rightarrow$ A Bus): ASR drops to 0.04, but the specific effectiveness depends highly on the choice of alternative words.
- PAD adversarial patch defense: Yields counterproductive results, with ASR increasing from 0.48 to 0.50.

### GLIP Model Transfer (Table 4)
TrAP is also effective on GLIP-T, achieving an ASR of 0.96 on the Aquarium dataset, validating the generalizability of the method.

## Highlights & Insights

1. **First Study on OVOD Backdoor Attacks**: Unveils a new attack surface introduced by prompt tuning in open-vocabulary object detectors.
2. **Theoretically Intuitive Multi-Modal Joint Attack Design**: Visual prompts associate with the trigger while textual prompts handle downstream adaptation, complementing each other.
3. **Curriculum Learning for Trigger Scaling**: Cleverly resolves the weak gradient signal issue of small patches, offering a simple yet effective progressive training strategy.
4. **Extremely Low Parameter Overhead**: Only 0.2M parameters (prompt + meta-net) are required, which is far fewer than the 21-36M required for fine-tuning, yet achieves a higher ASR.
5. **Simultaneous Performance Boost on Clean Data**: The BmAP of the attacked model is significantly higher than the zero-shot baseline, increasing the stealthiness of the backdoor.

## Limitations & Future Work

- Only validated on 6 relatively small-scale ODinW datasets, lacking evaluation on large-scale general datasets (e.g., COCO/LVIS).
- The defense analysis is preliminary, only testing 3 inference-time defense methods without exploring dedicated prompt-level or feature-level defenses.
- Prompt Engineering defense (Bus $\rightarrow$ A Bus) can drastically reduce ASR, but the paper does not thoroughly analyze the underlying reasons or propose improvements.
- The trigger is a fixed patch, without considering more covert forms (e.g., frequency-domain perturbations, natural object triggers).
- Focuses solely on the attack without providing recommendations for defensive schemes.

## Related Work & Insights

| Method | Target Model | Attack Method | Weight Modification Required | Multi-Modal|
|------|---------|---------|-------------|-------|
| Chan et al. 2022 | Closed-set detector | Data poisoning | Yes | No |
| Bai et al. 2024 (BadCLIP) | CLIP classification | Textual prompt | No | Partial |
| SWARM (Yang 2024) | ViT classification | Visual prompt | No | No |
| **TrAP (Ours)** | **OVOD detection** | **Visual + textual prompt** | **No** | **Yes** |

The core contribution of this work lies in extending backdoor attacks from classification tasks and closed-set detection to open-vocabulary detectors, and being the first to exploit the complementarity of bi-modal prompts for the attack.

## Related Work & Insights

- **Defense Research Gap**: The attack surface revealed in this paper (prompt tuning) is highly realistic. Developing targeted defenses (such as prompt auditing and anomaly detection) is an important direction for future work.
- **Prompt Tuning Security**: When users outsource model adaptation, they must inspect the returned prompt parameters. Introducing prompt watermarking or a secure prompt supply chain could be considered.
- **Correlation with Autonomous Driving Security**: The paper's scenarios (e.g., an attacker placing a trigger on an ambulance to prevent its detection) are directly related to vehicle safety, highlighting the necessity of pre-deployment safety evaluation for OVODs.
- **Multi-Modal Attack Paradigm**: The mindset of bi-modal joint optimization can be extended to security research on other vision-language models (e.g., VLM grounding, visual QA).

## Rating
- Novelty: ⭐⭐⭐⭐ — First backdoor attack on OVOD, with a clear problem definition.
- Experimental Thoroughness: ⭐⭐⭐ — Comprehensive ablations, but the datasets are small and lack large-scale evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with sufficient motivation.
- Value: ⭐⭐⭐⭐ — Reveals critical security vulnerabilities, serving as a direct warning for OVOD deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MOBA: A Material-Oriented Backdoor Attack against LiDAR-based 3D Object Detection](moba_a_material-oriented_backdoor_attack_against_lidar-based_3d_object_detection.md)
- [\[ICLR 2026\] Map as a Prompt: Learning Multi-Modal Spatial-Signal Foundation Models for Cross-scenario Wireless Localization](../../ICLR2026/autonomous_driving/map_as_a_prompt_learning_multi-modal_spatial-signal_foundation_models_for_cross-.md)
- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](../../CVPR2026/autonomous_driving/open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[CVPR 2026\] Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)](../../CVPR2026/autonomous_driving/monocular_open_vocabulary_occupancy_prediction_for_indoor_scenes.md)
- [\[CVPR 2025\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](../../CVPR2025/autonomous_driving/o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
