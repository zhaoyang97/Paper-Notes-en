---
title: >-
  [Paper Note] Hashed Watermark as a Filter: A Unified Defense Against Forging and Overwriting Attacks in Neural Network Watermarking
description: >-
  [AAAI 2026][AI Safety][Neural network watermarking] This paper proposes NeuralMark—a weight-based watermarking method built on a hashed watermark filter. It leverages the SHAKE-256 hash function to derive irreversible bi…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Neural network watermarking"
  - "model intellectual property protection"
  - "forging attack"
  - "overwriting attack"
  - "hash function"
  - "white-box watermarking"
date: 2026-05-08
content_hash: 57e81cdf362b823c
---

# Hashed Watermark as a Filter: A Unified Defense Against Forging and Overwriting Attacks in Neural Network Watermarking

**Conference**: AAAI 2026
**arXiv**: [2507.11137](https://arxiv.org/abs/2507.11137)
**Code**: [GitHub](https://github.com/AIResearch-Group/NeuralMark)
**Area**: AI Security / Model Watermarking
**Keywords**: Neural network watermarking, model intellectual property protection, forging attack, overwriting attack, hash function, white-box watermarking

## TL;DR

This paper proposes NeuralMark—a weight-based watermarking method built on a hashed watermark filter. It leverages the SHAKE-256 hash function to derive irreversible binary watermarks from secret key matrices, which serve as private filters for selecting embedding parameters. The avalanche effect blocks gradient-based reverse engineering against forging attacks, while multi-round filtering minimizes parameter overlap to resist overwriting attacks. Effectiveness and robustness are validated across 13 CNN/Transformer architectures on 5 image classification tasks and 1 text generation task.

## Background & Motivation

- **Background**: Training deep neural networks incurs enormous costs (GPT-4 ≈ \$40M), making models valuable digital assets that require copyright protection. Neural network watermarking (NNW) is categorized into white-box (parameter access), black-box (query input/output), and no-box (model output only) approaches. Among white-box methods, weight watermarking directly embeds watermarks into model parameters without modifying the network architecture, offering the best compatibility.
- **Limitations of Prior Work**: Existing weight watermarking methods face two critical attack types:
  (1) **Forging attacks**: The adversary freezes model parameters and uses back-propagation to reverse-engineer a forged key–watermark pair to claim model ownership—VanillaMark and VoteMark are both 100% forgeable.
  (2) **Overwriting attacks**: The adversary embeds their own watermark to overwrite the original—when the attacker's embedding strength $\lambda$ is 1000× that of the original, GreedyMark's detection rate drops to 49.60%.
- **Key Challenge**: Resistance to forging requires blocking gradient computation (irreversibility), while resistance to overwriting requires different watermarks to select different parameters (parameter isolation). Prior methods address at most one of these two challenges.
- **Key Insight**: The **avalanche effect** of hash functions (small input change → drastic output change) enables irreversibility, while distinct hashed watermarks serve as **private filters** to achieve parameter isolation—a single mechanism that resolves both problems simultaneously.

## Method

### Overall Architecture

NeuralMark consists of three stages: (1) **Hashed watermark generation**—a binary watermark is generated from a secret key matrix via the SHAKE-256 hash function; (2) **Watermark embedding**—the hashed watermark acts as a filter for multi-round parameter selection → average pooling → binary cross-entropy embedding; (3) **Watermark verification**—the extracted watermark is compared against the original to compute the detection rate.

### Key Designs

1. **Hashed Watermark Filter (Core Innovation)**

    - Function: The hashed watermark $\mathbf{b} = \mathcal{H}(\mathbf{K})$ serves as a private filter to select model parameters across multiple rounds.
    - Filtering process: Starting from the initial parameter vector $\mathbf{w}^{(0)} = \mathbf{w}$, at round $r$, $\mathbf{b}$ is tiled to match the parameter length to obtain $\mathbf{b}^{(r)}$; only parameters at positions where $\mathbf{b}^{(r)} = 1$ are retained to yield $\mathbf{w}^{(r)}$.
    - **Anti-forging mechanism**: The avalanche effect of SHAKE-256 causes minute key changes to produce drastically different watermarks, preventing effective gradient propagation and rendering reverse engineering infeasible.
    - **Anti-overwriting mechanism**: Different watermarks produce different filters; after multi-round filtering, parameter overlap approaches 0%. Example with $\mathbf{b}_1 = [1,0,1,0]$ vs. $\mathbf{b}_2 = [0,1,1,0]$: 100% overlap before filtering → 50% after one round → 0% after two rounds.

2. **Average Pooling for Parameter Robustness**

    - Function: Average pooling is applied to the filtered parameters $\mathbf{w}^{(R)}$ to obtain $\widetilde{\mathbf{w}} = \text{AVG}(\mathbf{w}^{(R)})$.
    - Design Motivation: Average pooling aggregates parameter information over a broader region, making the watermark robust against perturbations caused by fine-tuning and pruning—when individual parameters are modified or zeroed out, the average changes only slightly.

3. **Watermark Embedding and Verification**

    - Embedding loss: $\min_\theta \mathcal{L}_m + \lambda \mathcal{L}_e(\widetilde{\mathbf{b}}, \mathbf{b})$, where $\widetilde{\mathbf{b}} = \delta(\widetilde{\mathbf{w}} \mathbf{K})$ is the extracted watermark.
    - Verification condition: The detection rate $\rho = \frac{1}{n}\sum_{i=1}^n \mathbf{1}[b_i = \mathcal{T}(\tilde{b}_i)]$ must exceed the security boundary of **88.29%** (forging probability $< 1/2^{128}$ at $n=256$), and hash consistency $\mathcal{H}(\mathbf{K}) = \mathbf{b}$ must hold.

### Theoretical Analysis of the Security Boundary

**Proposition 1**: Assuming a uniformly distributed hash function output, the probability upper bound of a forged watermark achieving detection rate $\geq \rho$ is $\frac{1}{2^n}\sum_{i=0}^{n-\lceil\rho n\rceil}\binom{n}{i}$. At $n=256$ and $\rho=88.29\%$, this probability is $< 1/2^{128}$, which is negligible.

## Key Experimental Results

### Fidelity Evaluation (Classification Accuracy / Watermark Detection Rate Both ≈ 100%)

| Dataset | Clean (AlexNet/ResNet-18) | NeuralMark | VanillaMark | GreedyMark | VoteMark |
|:--|:--|:--|:--|:--|:--|
| CIFAR-10 | 91.05 / 94.76 | 90.93 / 94.50 | 91.01 / 94.87 | 90.88 / 94.69 | 90.86 / 94.79 |
| CIFAR-100 | 68.24 / 76.23 | 68.57 / 76.34 | 68.43 / 76.22 | 68.31 / 76.14 | 68.53 / 76.74 |
| TinyImageNet | 42.42 / 53.48 | 42.31 / 53.22 | 42.50 / 53.36 | 42.94 / 53.31 | 42.50 / 53.47 |

Architecture generalization: ViT-B/16 (39.22%), Swin-V2-B (53.57%), VGG-16 (72.61%), ResNet-34 (77.03%), GPT-2-S/M (text generation)—all achieve 100% detection rate with negligible performance degradation.

### Robustness Evaluation

| Attack Type | Setting | NeuralMark Detection Rate | VanillaMark | GreedyMark | VoteMark |
|:--|:--|:--|:--|:--|:--|
| Forging attack | CIFAR-10, ResNet-18 | **48.56%** (≈ random) | 100% (fully forged) | 50.70% | 100% (fully forged) |
| Overwriting $\lambda$=1000 | CIFAR-100→10 | **100%** | 53.90% | 49.60% | 59.37% |
| Overwriting $\eta$=0.01 | CIFAR-100→10 | **92.18%** | 62.10% | 49.60% | 60.15% |
| Fine-tuning | CIFAR-100→10 | **100%** | 85.93% | 94.14% | 85.54% |
| Fine-tuning | CIFAR-10→100 | **100%** | 70.31% | 82.42% | 71.87% |
| Pruning 50% | CIFAR-10, AlexNet | ≈**100%** | slightly lower | — | — |

### Key Findings

- **Forging attacks**: The detection rate of forged watermarks against NeuralMark is only 48.56% (≈ random guessing), while VanillaMark and VoteMark are completely forged (100%); GreedyMark also resists forging (50.70%) but fails against overwriting.
- **Overwriting attacks**: Even when the attacker's embedding strength $\lambda$ is 1000× the original, NeuralMark maintains 100% detection rate for the original watermark—owing to 0% parameter overlap. When the learning rate $\eta$ is increased to 0.1, although the detection rate drops, model performance collapses entirely (10% accuracy), rendering the attack ineffective.
- **Fine-tuning / Pruning**: The average pooling mechanism enables NeuralMark to maintain 100% detection rate across all fine-tuning and pruning scenarios.
- **Parameter distribution and convergence**: Watermark embedding has almost no impact on parameter distribution or training convergence, ensuring good stealthiness.
- **Filter round analysis**: Parameter overlap approaches 0% after 4 rounds of filtering; increasing to 6 or 8 rounds yields no significant robustness improvement.

## Highlights & Insights

- ⭐ **One mechanism solves two problems**: The hashed watermark filter elegantly unifies the avalanche effect (anti-forging) and parameter isolation (anti-overwriting) within a single design.
- ⭐ **Theoretical security boundary**: Proposition 1 provides a rigorous upper bound on forging probability; at $n=256$, the security boundary is 88.29% with forging probability $< 1/2^{128}$.
- ⭐ **Broad architecture coverage**: Validated across 13 architectures (8 CNNs + 3 Transformers + 2 GPT-2 variants) on 5 vision tasks and 1 text generation task, making it one of the most comprehensive weight watermarking studies to date.
- ⭐ **Robustness under extreme overwriting**: 100% detection is maintained at 1000× embedding strength, far surpassing all baselines.

## Limitations & Future Work

- **White-box weight watermarking only**: Black-box and no-box settings are not addressed, nor is combination with passport-based or activation-based methods.
- **Assumes a trusted third-party verifier**: Establishing such verification mechanisms in practice remains challenging.
- **Assumes limited adversarial computation**: If the adversary can train a model from scratch, watermark protection fails.
- **Hash function fixed to SHAKE-256**: The impact of different hash function choices on security is not explored.
- **Increased filter rounds reduce available parameters**: At 8 rounds, the number of usable parameters decreases substantially, potentially limiting watermark capacity.

## Related Work & Insights

- **Weight watermarking**: VanillaMark (Uchida et al. 2017)—first weight watermarking method, not robust to forging or overwriting; GreedyMark (Liu et al. 2021)—greedy parameter selection, resists forging but not overwriting; VoteMark (Li et al. 2024)—multi-round voting, robust to neither attack.
- **Passport watermarking**: Fan et al. (2019, 2021)—passport samples used to generate normalization layer parameters; Liu et al. (2023)—hash mapping from passports to watermarks, also leveraging hash functions.
- **Activation watermarking**: DeepSigns (Rouhani et al. 2019)—embedding in activation map means; Li et al. (2021)—direct embedding into activation maps.
- **IP protection surveys**: Li et al. (2021), Sun et al. (2023), Lukas et al. (2022).

## Rating

⭐⭐⭐⭐ — Elegant design (one mechanism addresses two problems), rigorous theoretical analysis, and exceptionally broad experimental coverage; limited to white-box settings with an assumption of a trusted third party.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking](yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)
- [\[AAAI 2026\] FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates](fairgse_fairness-aware_graph_neural_network_without_high_false_positive_rates.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[AAAI 2026\] Privacy on the Fly: A Predictive Adversarial Transformation Network for Mobile Sensor Data](privacy_on_the_fly_a_predictive_adversarial_transformation_network_for_mobile_se.md)
- [\[AAAI 2026\] InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference](infodecom_decomposing_information_for_defending_against_privacy_leakage_in_split.md)

</div>

<!-- RELATED:END -->
