---
title: >-
  [Paper Note] Hashed Watermark as a Filter: A Unified Defense Against Forging and Overwriting Attacks in Neural Network Watermarking
description: >-
  [AAAI 2026][AI Safety][Neural Network Watermarking] Proposes NeuralMark—a weight watermarking method based on a hashed watermark filter. It utilizes a hash function to generate an irreversible binary watermark from a secret key, serving as a private filter to select embedding parameters. It leverages the avalanche effect to block gradient backpropagation in forgery attacks, and employs multi-round filtering to reduce parameter overlap, thereby resisting overwriting attacks. I…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Neural Network Watermarking"
  - "Model IP Protection"
  - "Forgery Attacks"
  - "Overwriting Attacks"
  - "Hash Functions"
  - "White-box Watermarking"
date: 2026-05-08
content_hash: b65ab47e1d0c06e9
---

# Hashed Watermark as a Filter: A Unified Defense Against Forging and Overwriting Attacks in Neural Network Watermarking

**Conference**: AAAI 2026  
**arXiv**: [2507.11137](https://arxiv.org/abs/2507.11137)  
**Code**: [GitHub](https://github.com/AIResearch-Group/NeuralMark)  
**Area**: AI Security / Model Watermarking  
**Keywords**: Neural Network Watermarking, Model IP Protection, Forgery Attacks, Overwriting Attacks, Hash Functions, White-box Watermarking

## TL;DR

Proposes NeuralMark—a weight watermarking method based on a hashed watermark filter. It utilizes a hash function to generate an irreversible binary watermark from a secret key, serving as a private filter to select embedding parameters. It leverages the avalanche effect to block gradient backpropagation in forgery attacks, and employs multi-round filtering to reduce parameter overlap, thereby resisting overwriting attacks. Its effectiveness and robustness are validated across 13 CNN/Transformer architectures, 5 image classification tasks, and 1 text generation task.

## Background & Motivation

- **Background**: Deep neural networks incur huge training costs (e.g., approximately $40M for GPT-4), making models digital assets that require copyright protection. Neural network watermarking (NNW) is categorized into white-box (parameter access), black-box (input-output querying), and no-box (model output only). Among white-box methods, weight watermarking directly embeds watermarks into model parameters without modifying the network architecture, offering the best compatibility.
- **Limitations of Prior Work**: Existing weight watermarking methods face two critical types of attacks:  
  (1) **Forgery Attacks**: Attackers freeze model parameters and use gradient backpropagation to reverse-engineer a forged key-watermark pair to claim model ownership—both VanillaMark and VoteMark are 100% forgeable.  
  (2) **Overwriting Attacks**: Attackers embed their own watermark to overwrite the original watermark—when the attacker's embedding strength $\lambda$ is 1000 times the original, the detection rate of GreedyMark drops to 49.60%.
- **Key Challenge**: Resisting forgery requires blocking gradient computations (irreversibility), while resisting overwriting requires different watermarks to select different parameters (parameter isolation). Existing approaches can only defend against at most one of these attacks.
- **Key Insight**: Leverage the **avalanche effect** of hash functions (where tiny changes in input lead to massive output changes) to achieve irreversibility, and use different hashed watermarks as **private filters** to achieve parameter isolation, addressing both problems with a single mechanism.

## Method

### Overall Architecture

NeuralMark consists three phases: (1) **Hashed Watermark Generation**—generating a binary watermark from a secret key matrix using the SHAKE-256 hash function; (2) **Watermark Embedding**—using the hashed watermark as a filter for multi-round parameter selection $\rightarrow$ average pooling $\rightarrow$ binary cross-entropy embedding; (3) **Watermark Verification**—extracting the watermark and comparing it with the original to calculate the detection rate.

### Key Designs

1. **Hashed Watermark Filter (Core Innovation)**

    - **Function**: Uses the hashed watermark $\mathbf{b} = \mathcal{H}(\mathbf{K})$ as a private filter to perform multi-round selection of model parameters.
    - **Mechanism**: With the initial parameter vector $\mathbf{w}^{(0)} = \mathbf{w}$, at the $r$-th round, $\mathbf{b}$ is tiled to match the parameter length to obtain $\mathbf{b}^{(r)}$, retaining only parameters at positions where $\mathbf{b}^{(r)} = 1$ to yield $\mathbf{w}^{(r)}$.
    - **Forgery Resistance Mechanism**: The avalanche effect of SHAKE-256 causes minuscule key changes to yield drastically different watermarks, preventing effective gradient propagation and making reverse engineering infeasible.
    - **Overwriting Resistance Mechanism**: Different watermarks yield different filters, driving the parameter overlap rate close to 0% after multi-round filtering. For example, considering $\mathbf{b}_1 = [1,0,1,0]$ vs $\mathbf{b}_2 = [0,1,1,0]$: without filtering, 100% overlap $\rightarrow$ after one round, 50% overlap $\rightarrow$ after two rounds, 0% overlap.

2. **Average Pooling for Enhancing Parameter Robustness**

    - **Function**: Performs average pooling on the filtered parameters $\mathbf{w}^{(R)}$ to obtain $\widetilde{\mathbf{w}} = \text{AVG}(\mathbf{w}^{(R)})$.
    - **Design Motivation**: Average pooling aggregates parameter information over a wider region, making the watermark robust against parameter perturbations caused by fine-tuning and pruning—when individual parameters are modified or zeroed out, the average value changes minimally.

3. **Watermark Embedding and Verification**

    - Embedding Loss: $\min_\theta \mathcal{L}_m + \lambda \mathcal{L}_e(\widetilde{\mathbf{b}}, \mathbf{b})$, where $\widetilde{\mathbf{b}} = \delta(\widetilde{\mathbf{w}} \mathbf{K})$ is the extracted watermark.
    - Verification Condition: The detection rate $\rho = \frac{1}{n}\sum_{i=1}^n \mathbf{1}[b_i = \mathcal{T}(\tilde{b}_i)]$ must exceed the security bound of **88.29%** (for $n=256$, forgery probability is $< 1/2^{128}$) and satisfy the hash consistency $\mathcal{H}(\mathbf{K}) = \mathbf{b}$.

### Theoretical Analysis of Security Bounds

**Proposition 1**: Assuming a uniform distribution of the hash function's output, the upper bound of the probability that a forged watermark detection rate is $\geq \rho$ is expressed as $\frac{1}{2^n}\sum_{i=0}^{n-\lceil\rho n\rceil}\binom{n}{i}$. For $n=256$ and $\rho=88.29\%$, this probability is $< 1/2^{128}$, which is negligible.

## Key Experimental Results

### Fidelity Evaluation (Classification Accuracy / Watermark Detection Rate are both 100%)

| Dataset | Clean (AlexNet/ResNet-18) | NeuralMark | VanillaMark | GreedyMark | VoteMark |
|:--|:--|:--|:--|:--|:--|
| CIFAR-10 | 91.05 / 94.76 | 90.93 / 94.50 | 91.01 / 94.87 | 90.88 / 94.69 | 90.86 / 94.79 |
| CIFAR-100 | 68.24 / 76.23 | 68.57 / 76.34 | 68.43 / 76.22 | 68.31 / 76.14 | 68.53 / 76.74 |
| TinyImageNet | 42.42 / 53.48 | 42.31 / 53.22 | 42.50 / 53.36 | 42.94 / 53.31 | 42.50 / 53.47 |

Architectural Generalization: ViT-B/16 (39.22%), Swin-V2-B (53.57%), VGG-16 (72.61%), ResNet-34 (77.03%), and GPT-2-S/M (text generation) all achieve 100% detection rate with negligible performance degradation.

### Robustness Evaluation

| Attack Type | Setting | NeuralMark Detection Rate | VanillaMark | GreedyMark | VoteMark |
|:--|:--|:--|:--|:--|:--|
| Forgery Attack | CIFAR-10, ResNet-18 | **48.56%** (≈random) | 100% (completely forged) | 50.70% | 100% (completely forged) |
| Overwriting $\lambda$=1000 | CIFAR-100→10 | **100%** | 53.90% | 49.60% | 59.37% |
| Overwriting $\eta$=0.01 | CIFAR-100→10 | **92.18%** | 62.10% | 49.60% | 60.15% |
| Fine-tuning | CIFAR-100→10 | **100%** | 85.93% | 94.14% | 85.54% |
| Fine-tuning | CIFAR-10→100 | **100%** | 70.31% | 82.42% | 71.87% |
| Pruning 50% | CIFAR-10, AlexNet | ≈**100%** | Slightly lower | — | — |

### Key Findings

- **Forgery Attacks**: NeuralMark yields a forged watermark detection rate of only 48.56% (≈random guessing), whereas VanillaMark and VoteMark are completely forged (100%). While GreedyMark resists forgery (50.70%), it fails to resist overwriting.
- **Overwriting Attacks**: Even when the attacker's embedding strength $\lambda$ is 1000 times that of the original, the original watermark detection rate of NeuralMark remains at 100% due to 0% parameter overlap. While a larger learning rate $\eta = 0.1$ reduces the detection rate, it also completely collapses the model's performance (10% accuracy), rendering the attack ineffective.
- **Fine-tuning / Pruning**: The average pooling mechanism enables NeuralMark to maintain a 100% detection rate across all fine-tuning and pruning scenarios.
- **Parameter Distribution & Convergence**: Watermark embedding has negligible impact on parameter distribution and training convergence, ensuring high imperceptibility.
- **Number of Filtering Rounds**: Parameter overlap approaches 0% after 4 rounds of filtering; increasing to 6 or 8 rounds yields no significant improvement in robustness.

## Highlights & Insights

- ⭐ **Single Mechanism Addressing Dual Challenges**: The hashed watermark filter elegantly unifies the avalanche effect (forgery resistance) and parameter isolation (overwriting resistance) into a single design.
- ⭐ **Theoretical Security Bound**: Proposition 1 provides a rigorous upper bound for the forgery probability. With $n=256$, the security bound is 88.29%, yielding a forgery probability $< 1/2^{128}$.
- ⭐ **Extensive Architectural Coverage**: Demonstrated across 13 architectures (8 CNNs + 3 Transformers + 2 GPT-2 models) and 5 vision tasks plus 1 text generation task, making it one of the most comprehensively validated weight watermarking studies to date.
- ⭐ **Robustness under Extreme Overwriting**: Achieves a 100% detection rate even under 1000x embedding intensity, vastly outperforming all baselines.

## Limitations & Future Work

- **Focuses Solely on White-box Weight Watermarking**: Black-box and no-box scenarios are not addressed, nor is integration with passport-based or activation-based methods.
- **Assumption of a Trusted Third-Party Verifier**: Establishing reliable third-party verification mechanisms remains challenging in practice.
- **Assumption of Limited Attacker Resources**: If an attacker has the resources to train the model from scratch, the watermark protection becomes ineffective.
- **Fixed Hash Function Selection (SHAKE-256)**: The impact of different hash functions on security guarantees has not been explored.
- **Increased Filtering Rounds May Reduce Available Parameters**: Parameters are significantly reduced with 8 filtering rounds, potentially limiting watermark capacity.

## Related Work & Insights

- **Weight Watermarking**: VanillaMark (Uchida et al. 2017)—first weight watermarking method, vulnerable to both forgery and overwriting; GreedyMark (Liu et al. 2021)—greedy parameter selection, resists forgery but not overwriting; VoteMark (Li et al. 2024)—multi-round voting, vulnerable to both attacks.
- **Passport Watermarking**: Fan et al. (2019, 2021)—uses passport samples to generate normalization layer parameters; Liu et al. (2023)—maps passports to watermarks using hash functions, also leveraging hash functions.
- **Activation Watermarking**: DeepSigns (Rouhani et al. 2019)—embeds in the mean of activation maps; Li et al. (2021)—embeds directly into activation maps.
- **IP Protection Surveys**: Li et al. (2021), Sun et al. (2023), Lukas et al. (2022).

## Rating

⭐⭐⭐⭐ — Elegant design (resolving two challenges with a single mechanism), solid theoretical analysis, and extensive experimental coverage, though limited to white-box settings and relies on a trusted third party.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking](yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)
- [\[AAAI 2026\] FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates](fairgse_fairness-aware_graph_neural_network_without_high_false_positive_rates.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[CVPR 2026\] Verifying Neural Network Robustness with Dual Perturbations](../../CVPR2026/ai_safety/verifying_neural_network_robustness_with_dual_perturbations.md)
- [\[AAAI 2026\] RegionMarker: A Region-Triggered Semantic Watermarking Framework for Embedding-as-a-Service](regionmarker_a_region-triggered_semantic_watermarking_framework_for_embedding-as.md)

</div>

<!-- RELATED:END -->
