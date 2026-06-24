---
title: >-
  [Paper Note] Theoretically Unmasking Inference Attacks Against LDP-Protected Clients in Federated Vision Models
description: >-
  [ICML 2025][AI Safety][Federated Learning] This work derives, for the first time, the theoretical upper and lower bounds on the success rate of **Active Membership Inference (AMI)** attacks based on fully connected and self-attention layers **under Local Differential Privacy (LDP)** in federated learning. It reveals that even under LDP protection, the privacy risk still depends on the privacy budget $\varepsilon$, and the noise required to effectively mitigate such attacks se…
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Federated Learning"
  - "Local Differential Privacy"
  - "Membership Inference Attacks"
  - "Fully Connected Layer"
  - "Self-attention Mechanism"
  - "Vision Transformer"
date: 2026-05-08
content_hash: 52d7692058e2b029
---

# Theoretically Unmasking Inference Attacks Against LDP-Protected Clients in Federated Vision Models

**Conference**: ICML 2025  
**arXiv**: [2506.17292](https://arxiv.org/abs/2506.17292)  
**Code**: None  
**Area**: AI Security / Federated Learning Privacy  
**Keywords**: Federated Learning, Local Differential Privacy, Membership Inference Attacks, Fully Connected Layer, Self-attention Mechanism, Vision Transformer  

## TL;DR

This work derives, for the first time, the theoretical upper and lower bounds on the success rate of **Active Membership Inference (AMI)** attacks based on fully connected and self-attention layers **under Local Differential Privacy (LDP)** in federated learning. It reveals that even under LDP protection, the privacy risk still depends on the privacy budget $\varepsilon$, and the noise required to effectively mitigate such attacks severely degrades model utility.

## Background & Motivation

### Privacy Risks in Federated Learning
- Although federated learning does not directly share raw data, model updates (gradients or weights) can still leak sensitive information about the training data.
- **Membership Inference Attacks (MIA)**: Inferring whether a specific record is in the training set of the model.
- **Active MIA (AMI)**: A dishonest server actively tampers with global model parameters to enhance its inference capabilities.

### Limitations of Prior Work
1. Most MIA studies either ignore LDP or lack theoretical guarantees.
2. [Vu et al., 2024] proposed an AMI attack with low polynomial time complexity, but the theoretical analysis is only applicable to scenarios **without LDP protection**.
3. Random noise introduced by LDP varies across iterations and clients, making theoretical analysis highly challenging.

### Goal
To theoretically and practically demonstrate the **fundamental vulnerability** of client data against AMI attacks under LDP protection.

## Method

### Overall Architecture

The paper analyzes two categories of attacks under the security game framework $\mathsf{Exp}^{\text{AMI}}_{\text{LDP}}$:

1. **FC-based AMI**: Exploiting the structural vulnerabilities of fully connected layers.
2. **Attention-based AMI**: Exploiting the memorization mechanism of self-attention layers.

### Key Designs

#### 1. **FC-based Attack ($\mathcal{A}^{\mathcal{D}}_{\mathsf{FC}}$)**
- **Function**: Detecting whether a target sample $T$ exists in the training data by carefully setting the weights and biases of two FC layers.
- **Mechanism**: The first layer computes $z_0 = \max\{\tau^{\mathcal{D}} - \|\mathcal{M}^\varepsilon(X) - T\|_{L_1}, 0\}$
    - If $\mathcal{M}^\varepsilon(X)$ falls within the $L_1$ ball centered at $T$ $\to$ the gradient is non-zero $\to$ indicating the presence of $T$.
    - If it falls outside the ball $\to$ the gradient is zero.
- **Design Motivation**: Distinguishing target and non-target samples by setting $\tau^{\mathcal{D}} = \Delta^{\mathcal{X}}$ (half of the minimum $L_1$ distance in the data alphabet).

#### 2. **Attention-based Attack ($\mathcal{A}^{\mathcal{D}}_{\mathsf{Attn}}$)**
- **Function**: Exploiting the memorization capability of self-attention to configure attention heads to memorize the input batch and exclude the target pattern.
- **Mechanism**:
    - Head 1 filters out the target pattern $v$ $\to$ if $v$ is in the data, the output biases towards the global mean $\bar{X}^\varepsilon$.
    - Head 2 performs normal memorization $\to$ the output is close to the input $x_i^\varepsilon$.
    - Inferring based on whether the discrepancy between the outputs of the two heads, $|z_i^1 - z_i^2|$, exceeds a threshold $\gamma$.
- **Extension to ViT**: Extending the attack from the discrete token domain to the continuous image domain.

### Theoretical Results

#### Theorem 1 (Lower Bound of FC Attack)
$$\mathbf{Adv}^{\mathsf{AMI}}_{\text{LDP}}(\mathcal{A}^{\mathcal{D}}_{\mathsf{FC}}) \geq 1 - \frac{n + |\mathcal{X}| - 1}{|\mathcal{X}| - 1} P_{\mathcal{M}^\varepsilon}$$

- $P_{\mathcal{M}^\varepsilon}$: The probability that the LDP mechanism causes the protected data to fall outside the neighborhood ball.
- When $|\mathcal{X}|$ is large (e.g., in BitRand), the lower bound is approximately $1 - P_{\mathcal{M}^\varepsilon}$.

#### Theorem 2 (Upper Bound of FC Attack)
$$\mathbf{Adv}^{\mathsf{AMI}}_{\text{LDP}}(\mathcal{A}^{\mathcal{D}}_{\mathsf{FC}}) \leq \frac{e^\epsilon - 1}{e^\epsilon + 1}$$

#### Theorem 3 (Lower Bound of Attention Attack)
The lower bound consists of three terms: $P_{\text{proj}}$ (the probability that the projection component is smaller than the threshold, controlling the false positive rate), $P_{\text{box}}$ (the probability that the pattern falls into the central region, controlling the false negative rate), and the data separation $\Delta^\varepsilon$.

### Attack Failure Scenarios
- FC attack failure: (a) the protected version of the target falls outside $B_1(T, \Delta^{\mathcal{X}})$; (b) the protected version of a non-target falls inside the ball.
- Attention attack failure: When the noise is too large, all embeddings cluster at the center $\to P_{\text{box}} \approx 1$.

## Key Experimental Results

### FC Attack Success Rate vs. Privacy Budget (Fig. 7-8)

| LDP Mechanism | Dataset | Success Rate @ ε=3 | Success Rate @ ε=6 | Accuracy Loss @ 80% Protection |
|---------|-------|---------------|---------------|-----------------|
| BitRand | CIFAR10 | ~70% | ~100% | ≥20% |
| GRR | CIFAR10 | ~65% | ~100% | ≥25% |
| RAPPOR | CIFAR10 | ~60% | ~100% | ≥20% |
| dBitFlipPM | CIFAR10 | ~55% | ~100% | ≥30% |

### Attention Attack Experiments (Fig. 9)

| Model | Dataset | Success Rate @ ε=3 | Batch Size Impact |
|------|--------|-----------|----------|
| ViT-B-32-224 | CIFAR10 | ~100% | Robust (batch size 10–100) |
| ViT-B-32-384 | ImageNet-1k | ~100% | Robust |

### Key Findings

1. **Privacy-Utility Dilemma**: The noise level required to reduce the inference rate below 80% causes a model accuracy loss of $\geq 20\%$.
2. **Theoretical-Experimental Consistency**: The theoretical lower bound aligns with the experimental success rate of $\approx 100\%$ at $\varepsilon=8$.
3. **Attention Attacks are Stronger**: A success rate close to 100% is achieved at $\varepsilon=3$ (whereas FC attacks require $\varepsilon=5-6$).
4. **Higher-Dimensional Data is More Vulnerable** (Remark 5): As $d_X \to \infty$, $P_{\text{proj}} \to 1$, indicating that the attack advantage tends to maximize.

## Highlights & Insights

1. **First Theoretical Framework for AMI under LDP**: Filling the gap in the theoretical analysis of privacy protection in federated learning.
2. **Both Upper and Lower Bounds Provided**: Theorem 1 and 2 offer a complete theoretical characterization of the FC attack.
3. **Cross-Modal Validation**: Extending from computer vision (ResNet, ViT) to NLP (BERT, GPT-1, RoBERTa), proving that privacy risks are ubiquitous.
4. **Practical Warning**: LDP, serving as the "gold standard" of privacy protection, may need to be re-evaluated when facing active adversaries.

## Limitations & Future Work

1. **Theoretical Analysis of Attention Attacks is Only Applicable to Continuous Domains**: The NLP (discrete token) scenario requires a separate theoretical framework.
2. **Dependency on Prior Data Distribution**: The specific values of $P_{\mathcal{M}^\varepsilon}$ and $P_{\text{proj}}$ depend on the specific LDP mechanism and data distribution.
3. **Single-Round Attack**: Only single FL iterations are considered, while multi-round attacks might be stronger.
4. **Insufficiency of Defense Strategies**: This work primarily reveals risks without proposing effective defense solutions.

## Related Work & Insights

- **Origin of AMI Attacks**: [Nasr et al., 2019] first introduced AMI in FL, and [Vu et al., 2024] proposed low-complexity variants.
- **LDP Mechanisms**: Classical local privacy algorithms such as BitRand, GRR, RAPPOR, and dBitFlipPM.
- **Memorization of Self-Attention**: [Ramsauer et al., 2021] proved that attention is equivalent to a Hopfield layer, establishing a theoretical foundation for the attack.
- **Insights**: Privacy protection requires a multi-layered defense that comprehensively considers LDP, secure aggregation, and differential privacy.

## Rating
- Novelty: ⭐⭐⭐⭐ — For deriving the theoretical bounds of AMI attacks under LDP for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covering CIFAR10/100/ImageNet, 4 LDP mechanisms, 2 types of attacks (FC and Attention), and both computer vision and NLP domains.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical derivations and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ — Profound impact on the practice of privacy protection in federated learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Protection against Source Inference Attacks in Federated Learning](../../ICLR2026/ai_safety/protection_against_source_inference_attacks_in_federated_learning.md)
- [\[ICML 2025\] Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)
- [\[ICML 2025\] Privacy-Shielded Image Compression: Defending Against Exploitation from Vision-Language Pretrained Models](privacy-shielded_image_compression_defending_against_exploitation_from_vision-la.md)
- [\[ICML 2025\] Adversarial Inception Backdoor Attacks against Reinforcement Learning](adversarial_inception_backdoor_attacks_against_reinforcement_learning.md)
- [\[ICML 2025\] De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks](de-antifake_rethinking_the_protective_perturbations_against_voice_cloning_attack.md)

</div>

<!-- RELATED:END -->
