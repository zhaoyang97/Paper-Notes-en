---
title: >-
  [Paper Note] Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning
description: >-
  [CVPR 2025][AI Safety][Privacy Attack] This work demonstrates for the first time that adapter-based PEFT is not privacy-secure in federated learning. A malicious server can design the pre-trained model as an identity mapping, allowing patch embeddings to propagate to the adapter layer unchanged, and analytically reconstruct training images from the adapter gradients (CIFAR-100 SSIM 0.88).
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Privacy Attack"
  - "Gradient Inversion"
  - "PEFT Security"
  - "Federated Learning"
  - "Adapter Vulnerability"
date: 2026-05-08
content_hash: 48287a500090fc56
---

# Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning

**Conference**: CVPR 2025  
**arXiv**: [2506.04453](https://arxiv.org/abs/2506.04453)  
**Code**: [https://github.com/info-ucr/PEFTLeak](https://github.com/info-ucr/PEFTLeak)  
**Area**: AI Security  
**Keywords**: Privacy Attack, Gradient Inversion, PEFT Security, Federated Learning, Adapter Vulnerability

## TL;DR
This work demonstrates for the first time that adapter-based PEFT is not privacy-secure in federated learning. A malicious server can design the pre-trained model as an identity mapping, allowing patch embeddings to propagate to the adapter layer unchanged, and analytically reconstruct training images from the adapter gradients (CIFAR-100 SSIM 0.88).

## Background & Motivation

**Background**: Parameter-Efficient Fine-Tuning (PEFT) is widely considered safer in federated learning because it only shares gradients of a small subset of parameters, reducing the information accessible to attackers. Adapters (e.g., LoRA) only train low-rank matrices with a bottleneck dimension $r \ll D$, further limiting the amount of exploitable information.

**Limitations of Prior Work**: While prior works have demonstrated that LoRA fine-tuning can leak text data, the security of adapters in the vision domain remains unverified. It is generally assumed that the parameter-efficient nature of PEFT inherently provides privacy protection.

**Key Challenge**: Intuitively, sharing fewer parameters seems more secure. However, if an attacker can control the initialization of the pre-trained model (as the server distributes the model in federated learning), they can design the pre-trained layers as "transparent pipelines," forcing all information to flow directly to the observable adapter layers.

**Goal**: Prove that adapter-based PEFT is insecure against a malicious server, and design practical attack algorithms.

**Key Insight**: A malicious server can set LayerNorm, MSA, and MLP of a ViT to identity mappings, allowing patch embeddings to propagate distortion-free to the adapter layer. The weights and biases of the adapter can then be designed so that specific neurons selectively "pass through" patch information from specific locations.

**Core Idea**: "Hollow out" the pre-trained model into identity-mapping channels so that the original image patch information reaches the adapter intact, thereby analytically reconstructing the original images from the adapter gradients.

## Method

### Overall Architecture
The malicious server designs the pre-trained ViT (with identity mapping) and the adapter parameters $\rightarrow$ The client normally trains on this model using PEFT $\rightarrow$ The client uploads the adapter gradients $\rightarrow$ The server analytically reconstructs the training images from the gradients.

### Key Designs

1. **Pre-trained Model Identity Mapping**:

    - Function: Enables lossless propagation of patch embeddings to the adapter layer.
    - Mechanism: Set $\mathbf{E} = 0.5\mathbf{I}_D$ (linear embedding), and configure LayerNorm, MSA, and MLP entirely as identity mappings. Use positional encodings $\mathbf{E}_{pos}^{(n)} \sim \mathcal{N}(0, 10)$ to ensure prompt orthogonality among patches from different positions, assisting in subsequent differentiation.
    - Design Motivation: The identity mapping guarantees that the input received by the adapter layer is exactly the original patch embedding plus the positional encoding.

2. **Adapter Neuron Design**:

    - Function: Enables each neuron to "extract" patch information from a specific location.
    - Mechanism: Configure the adapter down-projection weights as the positional encoding of the target location $\mathbf{E}_{pos}^{(t)}$. The biases are designed such that only patches from the target position with values in a specific range can activate this neuron, functioning like a "select gate" for a specific location and value range.
    - Design Motivation: Although the adapter's bottleneck dimension $r \ll D$ limits the number of patches reconstructed in a single pass, different patches can be recovered by utilizing multiple adapter layers.

3. **Multi-round Attack Extension**:

    - Function: Overcomes the information bottleneck caused by a small $r$.
    - Mechanism: Design different value ranges for each training round $\rightarrow$ reconstruct different patches or value ranges across different rounds $\rightarrow$ stitch multi-round results into a complete image.
    - Design Motivation: While a single-round reconstruction yields a low success rate when $r=8$, full image recovery is achievable within 6 to 8 rounds.

### Loss & Training
Analytical attack—no optimization required, directly reading the image information from gradient values. The malicious server possesses complete control over model initialization.

## Key Experimental Results

### Main Results

| Dataset | LPIPS↓ | SSIM↑ | MSE↓ |
|--------|--------|-------|------|
| CIFAR-10 | 0.10 | 0.74 | 0.21 |
| CIFAR-100 | 0.08 | **0.88** | 0.20 |
| TinyImageNet | 0.12 | 0.76 | 1.06 |
| ImageNet batch=8 | - | - | 90% patch recovery |

### Ablation Study

| Configuration | Patch Recovery Rate |
|------|-------------|
| batch=32, r=64 | ~85% |
| batch=128, r=64 | 72.6% |
| r=64 Single Round | ~85% |
| r=8 Single Round | Very low |
| r=8 Multi-round (6-8 rounds) | Full recovery |

### Key Findings
- **PEFT does not equal privacy protection**: The low-rank bottleneck of adapters can be bypassed through multi-round attacks combined with multiple adapter layers.
- **72.6% of patches can still be recovered at batch=128**: Large batch sizes cannot fully mask individual privacy.
- **Analytical attack needs no iterative optimization**: It is significantly faster than traditional gradient inversion attacks.
- **Call for Differentially Private PEFT**: Injecting noise into adapter gradients is necessary to achieve true privacy protection.

## Highlights & Insights
- **The "hollowing out the model into identity mappings" attack strategy** is highly ingenious, leveraging the server's privilege to control model initialization in federated learning.
- **Privacy warning for PEFT security**: The work challenges the intuition that "fewer parameters mean more security," which has significant implications for the practical deployment of PEFT in federated learning.

## Limitations & Future Work
- The attack assumes the malicious server completely controls model initialization. If the client verifies model integrity, the attack will fail.
- An identity-mapped model exhibits zero performance on normal tasks; thus, the client can detect anomalies via a validation set.
- The attack only targets adapter-based PEFT, with different applicability to LoRA (which adds low-rank structures within existing layers rather than using separate layers).

## Related Work & Insights
- **vs. Traditional Gradient Inversion (GradInversion)**: Traditional methods require iterative optimization and assume an honest-but-curious server, whereas PEFTLeak operates in an analytical and malicious server setting.
- **vs. TAG / iDLG**: These are gradient leakage attacks in the NLP domain. PEFTLeak extends such vulnerabilities to vision PEFT for the first time.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to expose gradient privacy vulnerabilities in vision PEFT with an ingeniously designed attack.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive ablating over multiple datasets, batch sizes, and $r$ values.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of the attack workflow.
- Value: ⭐⭐⭐⭐⭐ Offers crucial warnings for the security practices of PEFT under federated learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation](../../AAAI2026/ai_safety/plug-and-play_parameter-efficient_tuning_of_embeddings_for_federated_recommendat.md)
- [\[ICCV 2025\] LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement](../../ICCV2025/ai_safety/lora-fair_federated_lora_fine-tuning_with_aggregation_and_initialization_refinem.md)
- [\[ICML 2026\] From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](../../ICML2026/ai_safety/from_parameter_dynamics_to_risk_scoring_quantifying_sample-level_safety_degradat.md)
- [\[CVPR 2025\] Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)
- [\[ICML 2026\] Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning](../../ICML2026/ai_safety/decoupled_training_with_local_reinforcement_fine-tuning_in_federated_learning.md)

</div>

<!-- RELATED:END -->
