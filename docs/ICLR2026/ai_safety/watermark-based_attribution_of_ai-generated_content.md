---
title: >-
  [Paper Note] Watermark-based Detection and Attribution of AI-Generated Content
description: >-
  [ICLR 2026][AI Safety][watermark] This paper presents the first systematic study on watermark-based user-level detection and attribution of AI-generated content. It provides theoretical analysis (bounds on TDR/FDR/TAR)…
tags:
  - "ICLR 2026"
  - "AI Safety"
  - "watermark"
  - "attribution"
  - "AI-generated content"
  - "detection"
  - "digital forensics"
date: 2026-05-08
content_hash: 8574adbe14529d04
---

# Watermark-based Detection and Attribution of AI-Generated Content

**Conference**: ICLR 2026
**arXiv**: [2404.04254](https://arxiv.org/abs/2404.04254)  
**Code**: None  
**Area**: AI Safety
**Keywords**: watermark, attribution, AI-generated content, detection, digital forensics

## TL;DR

This paper presents the first systematic study on watermark-based user-level detection and attribution of AI-generated content. It provides theoretical analysis (bounds on TDR/FDR/TAR), an efficient watermark selection algorithm (A-BSTA), and cross-modal (image + text) experimental validation, demonstrating that detection and attribution inherit the accuracy and (non-)robustness of the underlying watermarking method.

## Background & Motivation

Generative AI systems (e.g., DALL-E, Midjourney, ChatGPT) can produce highly realistic content, raising ethical concerns such as misinformation and copyright disputes. Companies including Google, OpenAI, and Microsoft have deployed watermarking techniques for **detection** of AI-generated content. However, the existing literature primarily focuses on user-agnostic detection—where all content carries the same watermark and the sole objective is to determine whether content is AI-generated.

This paper identifies a more advanced requirement: **attribution**. Beyond detecting whether content is AI-generated, it is necessary to trace the content back to the specific registered user who produced it. This capability is critical for law enforcement investigating cybercrimes such as misinformation campaigns. Despite its growing importance, attribution has received virtually no systematic study; this paper aims to fill that gap.

The core challenge is: when the number of users is very large (e.g., 100,000 or even 1,000,000), how can each user's watermark be made sufficiently unique to maintain high attribution accuracy without significantly increasing the false detection rate?

## Method

### Overall Architecture

The system comprises three phases:
1. **Registration**: Upon registration, the service provider assigns each user a unique watermark (bit string) stored in a database.
2. **Generation**: When a user generates content, their watermark is embedded into the content via an encoder.
3. **Detection & Attribution**: The watermark is decoded from the content under inspection. If the bitwise accuracy (BA) between the decoded watermark and any user's watermark exceeds threshold $\tau$, the content is classified as AI-generated; it is then attributed to the user with the highest BA.

### Key Designs

1. **Detection Mechanism**: Content $C$ is detected as AI-generated if and only if $\max_i BA(D(C), w_i) \geq \tau$, where $D$ is the decoder, $w_i$ is the $i$-th user's watermark, and $\tau > 0.5$ is the detection threshold.

2. **Attribution Mechanism**: After detection, content is attributed to $i^* = \arg\max_i BA(D(C), w_i)$, i.e., the user whose watermark is most similar to the decoded watermark.

3. **Watermark Selection Algorithm**: To improve attribution performance, the maximum pairwise BA among user watermarks must be minimized. This is formalized as: $\min_{w_s} \max_{i} BA(w_i, w_s)$, and the paper proves this problem is equivalent to the NP-hard Closest String / Farthest String problem. The **A-BSTA** (Approximate Bounded Search Tree Algorithm) is proposed with the following characteristics:

    - Initializes with a random watermark rather than $\neg w_1$ (improving performance)
    - Limits recursion depth to a constant $d=8$ (improving efficiency, reducing time complexity to $O(snm^d)$)
    - Performs incremental search starting from small $m$, increasing until a valid watermark is found

### Theoretical Analysis

Three core evaluation metrics are defined:
- **TDR_i** (True Detection Rate): the probability that AI-generated content from user $i$ is correctly detected
- **FDR** (False Detection Rate): the probability that non-AI content is misclassified as AI-generated
- **TAR_i** (True Attribution Rate): the probability that content from user $i$ is correctly attributed

Based on the definitions of $\beta$-accurate and $\gamma$-random watermarks:

- **Theorem 1**: Lower bound on TDR $= Pr(n_i \geq \tau n) + Pr(n_i \leq n - \tau n - \bar{\alpha_i} n)$, where $n_i \sim B(n, \beta_i)$
- **Theorem 3**: Upper bound on FDR $= 1 - Pr(n' < \tau n)^s$, where $n' \sim B(n, 0.5+\gamma)$
- **Theorem 4**: Lower bound on TAR $= Pr(n_i \geq \max\{\lfloor\frac{1+\bar{\alpha_i}}{2}n\rfloor+1, \tau n\})$

Key insight: when $\tau > \frac{1+\bar{\alpha_i}}{2}$, the lower bounds of TDR and TAR are approximately equal, meaning **detection implies attribution**.

## Key Experimental Results

### Main Results

Experiments are conducted on Stable Diffusion, Midjourney, and DALL-E 2 using HiDDeN (a learning-based watermarking method), with default settings: $s=100{,}000$ users, $n=64$-bit watermarks, $\tau=0.9$.

| Scenario | Avg. TDR | Avg. TAR | FDR | Worst-1% TAR |
|----------|----------|----------|-----|--------------|
| No post-processing | ≈1.0 | ≈1.0 | ≈0 | >0.94 |
| JPEG (Q=90) | High | High | ≈0 | Slightly lower |
| Adversarial attack (black-box) | 0 | 0 | — | Severe image quality degradation |

### Watermark Selection Algorithm Comparison

| Method | Avg. Generation Time | Max Pairwise BA | Worst-user TAR |
|--------|---------------------|-----------------|----------------|
| Random | 0.01 ms | Highest | Lowest |
| NRG | 2.11 ms | Moderate | Moderate |
| A-BSTA | 24 ms | <0.74 | Highest |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Users $s$: 10→1M | TDR/TAR slightly decrease, FDR slightly increases | $s$ controls TDR–FDR trade-off |
| Watermark length $n$: 32→80 | $n=48/64$ optimal | Excessively long watermarks degrade encode/decode accuracy |
| Threshold $\tau$: 0.7→0.95 | TDR/TAR and FDR change in the same direction | Trade-off required |

### Key Findings

- Detection and attribution achieve near-perfect accuracy without post-processing; 85% of users attain TAR = 1.0
- Adversarially trained HiDDeN is robust to standard post-processing (JPEG, Gaussian blur, etc.)
- Theoretical lower bounds closely match empirical TDR/TAR, though the FDR upper bound is loose
- A-BSTA significantly improves worst-user performance at an acceptable cost of 24 ms per watermark
- The framework generalizes to AI-generated text using the AWT watermarking method

## Highlights & Insights

- **Theory–practice unification**: TDR/FDR/TAR bounds are derived for arbitrary watermarking methods, and the required $\beta$-accurate and $\gamma$-random parameters can be estimated empirically
- **"Detection implies attribution" insight**: When $\tau$ is sufficiently large, a successful detection automatically yields attribution, simplifying system design
- **Practical solution to an NP-hard problem**: The watermark selection task is connected to the Farthest String problem, leveraging algorithms from theoretical computer science
- **Cross-modal generality**: The same framework applies to both image and text detection and attribution

## Limitations & Future Work

- Watermarking methods remain non-robust against white-box adversarial attacks (TDR/TAR can drop to 0), an inherent limitation of watermarking itself
- The theoretical analysis assumes independence among watermark bits, which may not hold in practice
- The theoretical upper bound on FDR is loose, particularly when bitwise correlations are strong
- Experiments use relatively low image resolution (128×128); performance at higher resolutions remains to be verified
- The watermark selection algorithm has room for further optimization when the number of users is extremely large

## Related Work & Insights

This paper bridges digital watermarking (both non-learning-based methods such as Tree-Ring and learning-based methods such as HiDDeN) with AI safety. In contrast to user-agnostic detection, user-aware attribution assigns a unique watermark to each user, advancing the task from "is this AI-generated?" to "who generated it?". The A-BSTA algorithm draws from research on the Farthest String problem in theoretical computer science, illustrating the value of cross-domain methodology transfer.

## Rating

- Novelty: ⭐⭐⭐⭐ (First systematic study of watermark-based attribution, though detection itself is not novel)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three generative AI models + text + multiple post-processing scenarios)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear theoretical presentation, rigorous experiments, complete framework)
- Value: ⭐⭐⭐⭐ (Highly practical with direct reference value for generative AI service providers)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Beyond Last-Click: An Optimal Mechanism for Ad Attribution](../../NeurIPS2025/ai_safety/beyond_last-click_an_optimal_mechanism_for_ad_attribution.md)
- [\[ICLR 2026\] AP-OOD: Attention Pooling for Out-of-Distribution Detection](ap-ood_attention_pooling_for_out-of-distribution_detection.md)
- [\[ICML 2026\] Hidden in Plain Tokens: Simply Robust, Gradient-Free Watermark for Synthetic Audio](../../ICML2026/ai_safety/hidden_in_plain_tokens_simply_robust_gradient-free_watermark_for_synthetic_audio.md)
- [\[ICLR 2026\] Bridging Fairness and Explainability: Can Input-Based Explanations Promote Fairness in Hate Speech Detection?](bridging_fairness_and_explainability_can_input-based_explanations_promote_fairne.md)
- [\[AAAI 2026\] Hashed Watermark as a Filter: A Unified Defense Against Forging and Overwriting Attacks in Neural Network Watermarking](../../AAAI2026/ai_safety/hashed_watermark_as_a_filter_defeating_forging_and_overwriting_attacks_in_weight.md)

</div>

<!-- RELATED:END -->
