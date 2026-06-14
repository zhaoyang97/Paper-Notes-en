---
title: >-
  [Paper Note] Backdoor Attacks on Neural Networks via One-Bit Flip
description: >-
  [ICCV 2025][AI Safety][backdoor attack] This paper proposes SOLEFLIP, the first inference-time backdoor attack on quantized models that requires flipping only a single bit. Through an efficient algorithm for identifying…
tags:
  - "ICCV 2025"
  - "AI Safety"
  - "backdoor attack"
  - "bit-flip attack"
  - "Rowhammer"
  - "quantized model"
  - "one-bit flip"
date: 2026-05-08
content_hash: cea926ca1b33e5b7
---

# Backdoor Attacks on Neural Networks via One-Bit Flip

**Conference**: ICCV 2025
**Code**: N/A  
**Area**: AI Security / Backdoor Attacks
**Keywords**: backdoor attack, bit-flip attack, Rowhammer, quantized model, one-bit flip

## TL;DR

This paper proposes SOLEFLIP, the first inference-time backdoor attack on quantized models that requires flipping only a single bit. Through an efficient algorithm for identifying exploitable weights and bit positions, along with a corresponding trigger generation procedure, SOLEFLIP achieves an average attack success rate of 98.9% with zero degradation in clean accuracy across CIFAR-10, SVHN, and ImageNet.

## Background & Motivation

Backdoor attacks represent a stealthy security threat to DNNs — the model behaves normally on benign inputs but produces attacker-desired outputs when a specific trigger is present. Traditional backdoor attacks require manipulating the training data or process, yet the training environment can be secured. Recent work has introduced a more practical inference-time threat model that exploits memory fault injection techniques such as Rowhammer to flip bits in model weights. However, existing methods require flipping a large number of bits (10–100), which is highly challenging and often infeasible in practice. The fixed-point integer representation of quantized models further constrains the impact of bit flips — unlike flipping exponent bits in full-precision models to cause large value changes — making single-bit attacks considerably more difficult.

## Method

### Overall Architecture

SOLEFLIP proceeds in three steps: (1) **exploitable weight identification** — an algorithm is designed to locate a single weight and its specific bit position amenable to backdoor injection; (2) **trigger generation** — given the selected weight, a trigger is generated that activates the corresponding neuron with a large value; (3) **backdoor activation** — after flipping the target bit, inputs carrying the trigger drive the model to produce the attacker-desired output. Steps (1) and (2) are performed offline; step (3) is executed online.

### Key Designs

1. **Exploitable Weight Identification Algorithm**: The method searches across all layers of the quantized model, evaluating the effect of flipping each bit of each weight. It focuses on cases where flipping high-order bits (e.g., the sign bit or high-magnitude bits) produces a large change in weight value. By assessing the magnitude of the value change after flipping, the influence of the affected neuron on the output, and the reachability of the target class, the algorithm efficiently identifies the most exploitable (weight, bit position) pair. Unlike ONEFLIP targeting full-precision models, weights in quantized models are bounded within $[-1, 1]$, limiting the effect of bit flips and necessitating a more refined selection strategy.

2. **Trigger Generation**: Given the selected weight $w$ and bit position, a small patch trigger is optimized such that when appended to an input image, the flipped weight $w'$ activates the corresponding neuron with an abnormally large output, which then propagates through the network to the target class. The optimization objective is to maximize the activation value under the flipped weight.

3. **Backdoor via Single-Bit Flip**: The backdoor is injected by flipping exactly one bit. Rowhammer attacks have been demonstrated to precisely flip individual targeted bits, making this attack practically feasible. Compared to existing methods requiring tens of bit flips, SOLEFLIP substantially lowers the attack barrier.

### Loss & Training

Trigger generation is performed via gradient-based optimization. The attack does not involve any model training — it is an inference-time attack on an already-deployed model. Only a small number of benign samples are needed for evaluation.

## Key Experimental Results

### Main Results

| Dataset | Model | Bits Flipped | Attack Success Rate | Clean Accuracy Drop |
|--------|------|----------|----------|------------|
| CIFAR-10 | ResNet | 1 | 99.9% | 0.0% |
| SVHN | VGG | 1 | ~99% | 0.0% |
| ImageNet | ViT | 1 | ~98% | 0.0% |
| **Average** | — | **1** | **98.9%** | **0.0%** |

For reference, TBT requires ~100 bit flips and ProFlip requires ~10.

### Ablation Study

- Different quantization precisions (4/8-bit): both remain effective
- Different model architectures (CNN/ViT): generalizes across architectures
- Robustness against backdoor defenses: SOLEFLIP exhibits strong resistance to existing defense methods
- Effect of trigger size and position

### Key Findings

- Flipping a single bit is sufficient to successfully implant a backdoor in quantized models
- Although quantized models are more robust to parameter-level attacks than full-precision counterparts, single-bit vulnerabilities still exist
- SOLEFLIP demonstrates strong resistance against existing backdoor defenses
- The work reveals a serious security threat in the deployment of DNNs

## Highlights & Insights

- Pushes the practical feasibility of backdoor attacks to the extreme — a single-bit flip suffices
- Extending the attack from full-precision to quantized models addresses the more prevalent real-world scenario
- Attack efficacy is remarkable — 98.9% success rate with zero clean accuracy loss
- Serves as an important warning to the security research community

## Limitations & Future Work

- The attack assumes white-box access (knowledge of model architecture and weights), limiting applicability in certain scenarios
- The practical success rate of Rowhammer attacks is constrained by DRAM hardware characteristics
- New defenses specifically targeting this class of attacks remain an open research problem
- Validation is limited to classification tasks; applicability to detection, segmentation, and other tasks is unexplored

## Related Work & Insights

- TBT, ProFlip, and HPT are the primary inference-time backdoor attack baselines for comparison
- ONEFLIP, targeting single-bit attacks on full-precision models, is the most closely related prior work
- Hardware security research on Rowhammer attacks provides the threat model foundation
- From a defense perspective, bit-level vulnerabilities in quantized models warrant urgent attention

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First single-bit backdoor attack on quantized models
- **Technical Depth**: ⭐⭐⭐⭐ — Weight identification algorithm is elegantly designed
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple datasets, architectures, and defense evaluations
- **Writing Quality**: ⭐⭐⭐⭐ — Clear workflow diagrams and thorough comparative analysis
- **Value**: ⭐⭐⭐⭐ — Significant implications for the secure deployment of DNNs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[ICCV 2025\] Mind the Cost of Scaffold! Benign Clients May Even Become Accomplices of Backdoor Attack](mind_the_cost_of_scaffold_benign_clients_may_even_become_accomplices_of_backdoor.md)
- [\[ICML 2026\] Singular Bayesian Neural Networks](../../ICML2026/ai_safety/singular_bayesian_neural_networks.md)
- [\[ICCV 2025\] Backdoor Mitigation by Distance-Driven Detoxification](backdoor_mitigation_by_distance-driven_detoxification.md)
- [\[NeurIPS 2025\] Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](../../NeurIPS2025/ai_safety/influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
