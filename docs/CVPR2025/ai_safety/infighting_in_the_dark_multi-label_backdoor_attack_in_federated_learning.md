---
title: >-
  [Paper Note] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning
description: >-
  [CVPR 2025][AI Safety][Federated learning] This paper is the first to study non-collaborative multi-label backdoor attacks (MBA) in federated learning. It reveals the inherent flaw of prior single-label backdoor attacks that leads to mutual exclusion among attackers when extended to multi-label scenarios due to constructing similar out-of-distribution (OOD) mappings. It proposes Mirage, which establishes in-distribution (ID) backdoor mappings to allow multiple attackers to in…
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Federated learning"
  - "backdoor attack"
  - "multi-label backdoor"
  - "in-distribution mapping"
  - "adversarial adaptation"
date: 2026-05-08
content_hash: 00fa2913f59e552b
---

# Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning

**Conference**: CVPR 2025  
**arXiv**: [2409.19601](https://arxiv.org/abs/2409.19601)  
**Code**: Mentioned to be open-source in the paper  
**Area**: AI Security  
**Keywords**: Federated learning, backdoor attack, multi-label backdoor, in-distribution mapping, adversarial adaptation

## TL;DR

This paper is the first to study non-collaborative multi-label backdoor attacks (MBA) in federated learning. It reveals the inherent flaw of prior single-label backdoor attacks that leads to mutual exclusion among attackers when extended to multi-label scenarios due to constructing similar out-of-distribution (OOD) mappings. It proposes Mirage, which establishes in-distribution (ID) backdoor mappings to allow multiple attackers to inject backdoors independently and persistently, achieving an average attack success rate of over 97% that remains above 90% even after 900 rounds.

## Background & Motivation

**Background**: Federated learning (FL), as a privacy-preserving distributed learning paradigm, has been widely applied in security-sensitive areas such as medical image analysis and facial recognition. However, its distributed nature makes it vulnerable to backdoor attacks, where attackers inject backdoors into local models, which are then inherited by the global model after aggregation.

**Limitations of Prior Work**: Almost all current research on FL backdoor attacks assumes that attackers are collaborative and share the same target label, referred to as single-label backdoor attacks (SBA). However, in realistic large-scale FL scenarios, multiple attackers might act independently with different targets, forming a more practical multi-label backdoor attack (MBA) scenario. Directly applying existing SBA methods to MBA scenarios leads to a severe collapse in attack effectiveness.

**Key Challenge**: When multiple non-collaborative attackers construct their respective backdoor mappings, they often employ similar strategies (e.g., leveraging redundant neurons), leading to similar OOD backdoor mappings—where their backdoor samples exhibit similar distributions in the feature space but lie outside the target class distribution. This triggers neuron weight competition among attackers, allowing only the dominant attacker to successfully embed their backdoor.

**Goal**: (1) Reveal the fundamental reason why SBA methods fail when extended to MBA scenarios; (2) Design a method that allows multiple attackers to inject effective and persistent backdoors simultaneously without collusion.

**Key Insight**: The authors propose a counter-intuitive mechanism: if the backdoor can be triggered through the activation paths of clean samples—i.e., by constructing in-distribution (ID) mappings—different attackers will naturally avoid conflict, as they each bridge to the clean distribution of different target classes.

**Core Idea**: Optimizing the trigger via adversarial adaptation to align backdoor samples within the in-distribution (ID) space of the target class (rather than OOD), naturally eliminating mutual exclusion among multiple attackers.

## Method

### Overall Architecture

The overall workflow of Mirage consists of four steps: (1) training an OOD sample detector, (2) adversarially optimizing the trigger to deceive the detector, thereby constructing the ID mapping, (3) tightening the backdoor distribution via constrained optimization to enhance mapping durability, and (4) poisoning the local dataset using the optimized trigger, performing normal training, and uploading the model update. The inputs are the global model and the attacker's local data, while the output is the model update embedding the effective backdoor.

### Key Designs

1. **Adversarial Adaptation-based ID Mapping Construction**:

    - **Function**: Shifting backdoor sample features from OOD into the ID distribution of the target class.
    - **Mechanism**: An OOD detector is constructed using the frozen feature extractor of the global model combined with a binary classifier to distinguish clean samples from backdoor samples. Then, an adversarial training strategy is applied—the detector minimizes the BCE loss to accurately detect OOD samples, while the trigger is optimized to maximize the detector's misclassification probability (i.e., making backdoor samples perceived as clean). This min-max game equips the trigger with the capability to activate the clean path of the target class. Crucially, the detector reuses the global model's feature extractor, resulting in minimal computational overhead.
    - **Design Motivation**: Direct feature alignment can disrupt the relationships between clean classes. Indirectly achieving ID mapping through an adversarial game is more flexible and can adapt dynamically with each round of global model updates.

2. **Constrained Optimization-based ID Mapping Enhancement**:

    - **Function**: Tightening the backdoor sample distribution to ensure the ID mapping survives persistently under global training dynamics.
    - **Mechanism**: By minimizing the cosine similarity $\text{CS}(\theta_f(x), \theta_f(\hat{x}))$ of features between clean samples $x$ and their corresponding backdoor samples $x \oplus \delta$, while simultaneously minimizing the cross-entropy loss $\text{CE}(\hat{x}, \hat{y}, \theta)$ of backdoor samples on the global model. The former deviates the backdoor distribution from the original distribution to tighten it, and the latter determines the direction of the deviation toward the target class.
    - **Design Motivation**: An ID mapping constructed solely via adversarial strategies may lie on the boundary of the distribution and easily fail as global training progresses. Constrained optimization pushes boundary samples into the core region of the distribution, improving robustness.

3. **Non-Collaborative Multi-Attacker Framework Design**:

    - **Function**: Ensuring multiple attackers operate independently without mutual interference.
    - **Mechanism**: Each attacker only needs access to the global model and their own local data, remaining oblivious to other attackers. Since the ID mapping of each attacker bridges to the clean distribution of a different target class, they utilize distinct activation paths, naturally remaining free of conflict.
    - **Design Motivation**: In realistic large-scale FL scenarios, coordination among attackers is impractical. The ID mapping strategy makes independent operation viable.

### Loss & Training

The overall trigger optimization loss is defined as $\mathcal{L} = \mathcal{L}_{detector} + \mathcal{L}_{Enhance}$, where the detector loss drives the construction of the ID mapping, and the enhancement loss includes the cosine similarity term and the cross-entropy term to tighten the distribution. Triggers are optimized via PGD with a poisoning ratio of 12.5%.

## Key Experimental Results

### Main Results

| Dataset | Defense | Vanilla ASR | A3FL ASR | Mirage ASR | Mirage Acc |
|--------|------|-------------|----------|------------|------------|
| CIFAR-10 | No Defense | 31.88% | 99.52% | **99.54%** | 92.16 |
| CIFAR-10 | Indicator | 28.76% | 70.33% | **93.46%** | 91.10 |
| CIFAR-10 | MultiKrum | 1.59% | 78.15% | **92.30%** | 92.10 |
| CIFAR-100 | Indicator | 6.80% | 37.13% | **99.80%** | 68.22 |
| GTSRB | No Defense | 33.32% | 99.63% | **99.73%** | 96.97 |
| GTSRB | Indicator | 50.92% | 85.05% | **99.73%** | 95.12 |

### Ablation Study

| Configuration | ASR Variation | Description |
|------|---------|------|
| Full Mirage | 97%+ avg | Full model, 3-attacker GAP < 3% |
| w/o ID Mapping Enhancement | Significant drop | Adversarial adaptation alone is not durable enough |
| Different architectures | ResNet18/34/VGG11/19 all > 97% | Insensitive to model structures |
| MobileNet-V2 | Low but acceptable | Poor detector performance due to low model capacity |

### Key Findings

- Mirage achieves the highest ASR in all 18 CIFAR-10 task configurations with a minimal GAP among attackers (averaging 2.19%), demonstrating that multiple attackers can indeed succeed simultaneously.
- Under the strongest defense, Indicator (specifically designed to detect OOD backdoors), Mirage still maintains a 93%+ ASR, whereas A3FL is suppressed to 70% by the detector because it enhances OOD characteristics.
- Regarding durability, Mirage maintains a 90%+ ASR even 900 rounds after the attack window ends, matching the performance of durability-focused A3FL.
- Traditional methods such as Vanilla, PGD, and Neurotoxin yield ASRs generally below 40% in MBA scenarios, validating the finding of mutual exclusion among multiple attackers.

## Highlights & Insights

- **Counter-intuitive Design of ID Mapping**: Traditional backdoors pursue unique OOD activation paths, while Mirage does the opposite by exploiting clean paths. This not only resolves conflicts among multiple attackers but also bypasses OOD detection defenses, killing two birds with one stone.
- **Lightweight Detector Design**: Reusing the global model's feature extractor as the detector stabilizer and training only a binary classification head keeps the computational overhead extremely low. This "piggyback" strategy is highly clever and can be migrated to other adversarial scenarios requiring detectors.
- **Revealing a New Threat Model in Large-scale FL Systems**: The MBA scenario is much closer to reality than SBA, yet existing defenses are virtually ineffective against it. This opens up a new research direction for FL security.

## Limitations & Future Work

- The paper assumes that the ratio of attackers and the attack window are known; variations in these parameters in real-world scenarios might affect effectiveness.
- The performance is relatively poor on MobileNet-V2, indicating that the method relies to some extent on the capacity of the model itself.
- In terms of defense, only image classification tasks were tested, and its applicability to other modalities (NLP, multi-modal) has not been verified.
- The paper acknowledges that existing defenses generally fail in MBA scenarios, but the discussion on proposed countermeasures remains preliminary.

## Related Work & Insights

- **vs A3FL**: A3FL leverages machine unlearning to simulate exclusion to construct independent OOD mappings. Although it mitigates conflict, it enhances OOD characteristics, making it easy to detect with Indicator. Mirage completely avoids this issue by adopting the ID path.
- **vs Neurotoxin**: Neurotoxin utilizes rarely updated redundant neurons to enhance durability, but in MBA scenarios, multiple attackers compete for the same set of redundant neurons, causing conflict. Mirage does not rely on redundant neurons.
- **vs NBA**: Prior work studied non-collaborative multi-label attacks but failed to reveal the underlying cause or propose an effective solution. Mirage is the first to thoroughly address this problem.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to systematically study MBA scenarios and propose the counter-intuitive solution of ID mapping
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets + six defenses + multiple attack comparisons; however, limited to image classification
- Writing Quality: ⭐⭐⭐⭐ In-depth problem analysis and clear logic, though some equations are densely presented
- Value: ⭐⭐⭐⭐⭐ Reveals an overlooked but realistic security threat in FL systems

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)
- [\[CVPR 2025\] INACTIVE: Invisible Backdoor Attack against Self-supervised Learning](invisible_backdoor_attack_against_self-supervised_learning.md)
- [\[CVPR 2025\] MOS-Attack: A Scalable Multi-Objective Adversarial Attack Framework](mos-attack_a_scalable_multi-objective_adversarial_attack_framework.md)
- [\[ECCV 2024\] Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning](../../ECCV2024/ai_safety/fisher_calibration_for_backdoor-robust_heterogeneous_federated_learning.md)
- [\[NeurIPS 2025\] MARS: A Malignity-Aware Backdoor Defense in Federated Learning](../../NeurIPS2025/ai_safety/mars_a_malignity-aware_backdoor_defense_in_federated_learning.md)

</div>

<!-- RELATED:END -->
