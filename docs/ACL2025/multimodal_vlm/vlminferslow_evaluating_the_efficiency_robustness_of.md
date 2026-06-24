---
title: >-
  [Paper Note] VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service
description: >-
  [ACL 2025][Multimodal VLM][efficiency robustness] This work is the first to investigate the efficiency robustness of VLMs in black-box settings, proposing the VLMInferSlow method. By searching for adversarial image perturbations via zeroth-order optimization to force VLMs to generate longer sequences, it increases computational costs by up to 128.47%, revealing the efficiency-related security vulnerabilities of VLMs deployed under MLaaS scenarios.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "efficiency robustness"
  - "adversarial attack"
  - "VLM"
  - "black-box"
  - "inference slowdown"
date: 2026-05-08
content_hash: dd1fbfe8aded90f7
---

# VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service

**Conference**: ACL 2025  
**arXiv**: [2506.15755](https://arxiv.org/abs/2506.15755)  
**Code**: [https://github.com/wangdaha1/VLMInferSlow](https://github.com/wangdaha1/VLMInferSlow)  
**Area**: Multimodal VLM  
**Keywords**: efficiency robustness, adversarial attack, VLM, black-box, inference slowdown

## TL;DR
This work is the first to investigate the efficiency robustness of VLMs in black-box settings, proposing the VLMInferSlow method. By searching for adversarial image perturbations via zeroth-order optimization to force VLMs to generate longer sequences, it increases computational costs by up to 128.47%, revealing the efficiency-related security vulnerabilities of VLMs deployed under MLaaS scenarios.

## Background & Motivation

**Background**: VLMs have been widely deployed as API services (e.g., Microsoft Seeing AI, Be My Eyes) that require real-time responses. NVIDIA and AWS report that the inference phase accounts for over 90% of total ML energy consumption. Existing adversarial attack research mainly focuses on accuracy robustness.

**Limitations of Prior Work**: The few studies targeting VLM efficiency attacks (such as NICGSlowdown, Verbose Images) assume white-box access (complete knowledge of model architecture and parameters). In reality, VLMs are mostly deployed as APIs, making the white-box assumption impractical.

**Key Challenge**: Under black-box settings, gradient information is unavailable for optimization, and zeroth-order optimization methods tend to fail when the target function experiences drastic changes.

**Goal**: Evaluate the efficiency robustness of VLMs under a black-box setting (interacting only through APIs)—specifically, can imperceptible image perturbations significantly increase the inference overhead of VLMs?

**Key Insight**: The autoregressive decoding nature of VLMs inherently links inference efficiency to the generated sequence length. Making the generated sequence longer can effectively increase inference overhead. Zeroth-order optimization can be combined to estimate gradients as a replacement for white-box gradients.

**Core Idea**: Design three efficiency-oriented adversarial objectives (elongating sequence, delaying EOS, and increasing token diversity), and use zeroth-order optimization to search for imperceptible adversarial image perturbations in a black-box setting.

## Method

### Overall Architecture
Iterative optimization: In each iteration, (1) compute the efficiency-oriented target objective $\rightarrow$ (2) estimate gradients using zeroth-order optimization $\rightarrow$ (3) perform gradient ascent to update the perturbation and clip it to $\|\delta\| \leq \epsilon$. The input is a clean image $\mathcal{I}$ and the output is an adversarial image $\mathcal{I} + \Delta$, which forces the VLM to consume more computational resources during processing.

### Key Designs

1. **Three-in-One Adversarial Objectives**:

    - **$\mathcal{L}_{len}$ (elongating sequence)**: Directly maximize the output sequence length $\text{Length}(\mathcal{F}(\mathcal{I}+\delta))$. Although non-differentiable, it is suitable for derivative-free optimization.
    - **$\mathcal{L}_{eos}$ (delaying termination)**: Reduce the probability of outputting the EOS token at each position and introduce a dynamic weight-decay strategy that assigns larger weights to later positions: $\mathcal{L}_{eos} = -\sum_{i=1}^{N} \omega^{N-i} \text{Pr}^{\text{EOS}}(y_i | \mathcal{I}+\delta)$, where $\omega=0.1$.
    - **$\mathcal{L}_{var}$ (increasing diversity)**: Align the top-k probability distribution at each token position with a uniform distribution using KL divergence: $\mathcal{L}_{var} = -\frac{1}{N}\sum_{i=1}^{N} D_{KL}(\tilde{\text{Pr}}(y_i | \mathcal{I}+\delta) \| \mathcal{U})$, where $k=100$.
    - **Final Objective**: $\mathcal{L} = \mathcal{L}_{len} + \alpha \mathcal{L}_{eos} + \beta \mathcal{L}_{var}$
    - Design Motivation: A single objective is insufficient; the three objectives collectively exert pressure from the aspects of sequence length, termination signal, and token distribution.

2. **Zeroth-Order Gradient Estimation**:

    - Function: Estimate the gradient direction of the target objective in a black-box setting where gradients are inaccessible.
    - Mechanism: Adopts Natural Evolution Strategies by sampling $2q$ Gaussian noise perturbations under the search distribution $\pi(z|\delta) = \mathcal{N}(\delta, \eta^2 I)$ to estimate the gradient: $\hat{\nabla}_\delta J(\delta) = \frac{1}{2\eta q}\sum_{i=1}^{2q} \mu_i \mathcal{L}(\delta + \eta\mu_i)$. The anti-mirroring trick ($\mu_{q+j} = -\mu_j$) is used to reduce variance.
    - Design Motivation: Standard zeroth-order optimization is unstable when the loss landscape changes drastically; combining it with the refined multi-objectives smoothens the loss landscape, thus improving the search performance of zeroth-order optimization.

3. **Perturbation Constraint Update**:

    - Function: Apply gradient ascent to update the perturbation and then clip it within the $L_2$ norm constraint.
    - Mechanism: $\delta \leftarrow \delta + \gamma \hat{\nabla}_\delta J(\delta)$, followed by $\text{Clip}(\delta, \epsilon)$, while ensuring $(\mathcal{I}+\delta) \in [0,1]^n$.

### Loss & Training
This work does not involve model training; instead, adversarial perturbations are generated via iterative optimization during inference.

## Key Experimental Results

### Main Results

| Model | Method | MS-COCO I-length (%) | MS-COCO I-latency (%) | ImageNet I-length (%) | ImageNet I-latency (%) |
|------|------|:---:|:---:|:---:|:---:|
| Flamingo | Gaussian | -4.15 | -0.16 | -4.27 | -1.12 |
| Flamingo | NICGSlowdown-B | -3.54 | 0.19 | -1.14 | -0.12 |
| Flamingo | Verbose-B | -2.93 | 5.56 | -0.63 | 5.26 |
| **Flamingo** | **VLMInferSlow** | **128.47** | **105.56** | **103.44** | **78.42** |
| BLIP | Gaussian | 18.92 | 18.19 | 20.50 | 20.42 |
| **BLIP** | **VLMInferSlow** | **Significant Gain** | **Significant Gain** | **Significant Gain** | **Significant Gain** |

**Key Results**: VLMInferSlow increases the computational cost of Flamingo by up to 128.47% (length) and 115.19% (energy consumption) under the black-box setting, far exceeding all black-box baselines.

### Ablation Study

| Configuration | Effect | Explanation |
|------|------|------|
| Only $\mathcal{L}_{len}$ | Partially effective | Optimization direction is unstable without other constraints |
| $\mathcal{L}_{len} + \mathcal{L}_{eos}$ | Better | EOS delay significantly elongates sequence length |
| $\mathcal{L}_{len} + \mathcal{L}_{eos} + \mathcal{L}_{var}$ | Optimal | Coexistence of three objectives cooperatively enhances efficiency attack performance |
| No dynamic weight decay | Decreased | Uniform weights perform worse than weighting later positions |
| Black-box vs. White-box comparison | Close to white-box level | VLMInferSlow black-box performance is comparable to white-box counterparts |

### Key Findings
- **Black-Box Performance Approaching White-Box Levels**: VLMInferSlow achieves an attack performance in black-box settings comparable to that of white-box methods (which require full model parameters), demonstrating that logits information exposed by APIs is sufficient to construct effective attacks.
- **Imperceptible Perturbations**: The generated adversarial images are visually indistinguishable from original images, passing human perception tests.
- **Limited Effectiveness of Current Defenses**: Testing against several defense strategies reveals that VLMInferSlow still successfully increases computational overhead.
- **Robust Across Sampling Strategies**: The attack efficacy remains consistently stable under different sampling strategies such as temperature, top-k, and top-p.

## Highlights & Insights
- **First Black-Box VLM Efficiency Attack**: Fills the gap in VLM efficiency robustness evaluation under black-box setups. It is highly representative of real-world threats in MLaaS scenarios (such as deployment models used by OpenAI, Google Gemini, etc.).
- **Ingenuity of Refined Multi-Objective Design**: The three objectives exploit efficiency bottlenecks from different dimensions. Additionally, the refined objectives smoothen the loss landscape for zeroth-order optimization, resolving the inherent flaws of zeroth-order methods.
- **The Safety Implications Outweigh the Attack Itself**: The most critical value of this work lies in alerting the community to take the efficiency security of VLMs seriously, particularly the risks of resource exhaustion attacks in mobile devices and API service scenarios.

## Limitations & Future Work
- Requires the API to return logits information ($\mathcal{L}_{eos}$ and $\mathcal{L}_{var}$ rely on token probabilities), which some APIs might not provide.
- For APIs with pre-existing length-truncation mechanisms (e.g., max_tokens limit), the attack efficacy may be constrained.
- Zeroth-order optimization requires a substantial number of API queries ($2q$ times per iteration), incurring high computational costs.
- Focuses exclusively on image perturbations, without exploring efficiency attacks directed at the textual prompt side.

## Related Work & Insights
- **vs. NICGSlowdown**: NICGSlowdown targets image captioning models by delaying EOS but requires white-box access; VLMInferSlow achieves stronger effects under black-box constraints.
- **vs. Verbose Images**: Verbose Images designs several white-box loss functions to increase VLM computation; VLMInferSlow replaces gradient computation with zeroth-order optimization, achieving comparable performance under black-box settings.

## Rating
- Novelty: ⭐⭐⭐⭐ The first black-box VLM efficiency attack, with a highly valuable problem definition.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 VLMs + 2 Datasets + 4 Baselines + Ablation + Defense Evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation, intuitive figure and table comparisons.
- Value: ⭐⭐⭐⭐ Provides a crucial warning regarding VLM deployment security.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[ACL 2025\] Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](unsolvable_problem_detection.md)
- [\[ACL 2025\] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](mcts_video_captioning_eval.md)
- [\[ICCV 2025\] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](../../ICCV2025/multimodal_vlm/multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)
- [\[NeurIPS 2025\] Adapting Vision-Language Models for Evaluating World Models](../../NeurIPS2025/multimodal_vlm/adapting_visionlanguage_models_for_evaluating_world_models.md)

</div>

<!-- RELATED:END -->
