---
title: >-
  [Paper Note] DisCode: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning
description: >-
  [AAAI 2026][Multimodal VLM][Image captioning evaluation] This paper proposes DISCODE, a fine-tuning-free test-time adaptive decoder that introduces a Gaussian prior to minimize the ATT loss…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Image captioning evaluation"
  - "large vision-language models"
  - "test-time adaptation"
  - "scoring robustness"
  - "distribution prior"
date: 2026-05-08
content_hash: 1296bb7f095dc21c
---

# DisCode: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning

**Conference**: AAAI 2026
**arXiv**: [2512.14420](https://arxiv.org/abs/2512.14420)  
**Code**: Not released  
**Area**: Multimodal VLM
**Keywords**: Image captioning evaluation, large vision-language models, test-time adaptation, scoring robustness, distribution prior

## TL;DR

This paper proposes DISCODE, a fine-tuning-free test-time adaptive decoder that introduces a Gaussian prior to minimize the ATT loss, enabling LVLM-generated image captioning scores to more robustly align with human judgments. The paper also constructs the MCEval benchmark covering six visual domains.

## Background & Motivation

- **Core Problem**: When using LVLMs for automatic image captioning scoring, the output token probability distribution deviates from the human scoring distribution (**symbol bias**, e.g., the digit "0" probability is systematically overestimated), resulting in non-robust scores.
- **Limitations of Prior Work**: Methods such as FLEUR and G-VEval rely on score smoothing, directly equating the token probability distribution with the scoring distribution $p = p_{\text{LVLM}}$; however, token probability distributions tend to be non-unimodal.
- **Key Insight**: By the Central Limit Theorem, human scores naturally tend toward a Gaussian distribution (unimodal), whereas token probability distributions exhibit multimodal behavior due to symbol bias. This discrepancy is more pronounced in non-photorealistic image domains such as paintings and abstract sketches.
- **Dataset Gap**: Existing benchmarks (Flickr8k, Composite, Pascal-50S) cover only the photorealistic domain, making it impossible to evaluate the cross-domain robustness of evaluation metrics.

## Method

### 1. Scoring Pipeline and ATT Loss

DISCODE scores in three steps: (1) the LVLM generates a raw score $s_{\text{raw}} \in S = \{0,1,\cdots,9\}$, while extracting the final decoder layer feature $\boldsymbol{h}_T \in \mathbb{R}^d$ and token probability $p_{\text{LVLM}}$; (2) the scoring distribution $p$ is estimated by minimizing the ATT loss; (3) the expected value $s = \mathbb{E}_{x \sim p}[x]$ is taken as the final score.

The ATT loss is defined as the sum of a cross-entropy term and a divergence term:

$$\mathcal{L}_{\text{ATT}}(\theta; \boldsymbol{h}_T) = \underbrace{H(\psi_\theta(\boldsymbol{h}_T), p_{\text{LVLM}})}_{\text{cross-entropy}} + \underbrace{D_\alpha(\psi_\theta(\boldsymbol{h}_T) \| q)}_{\text{divergence regularization}}$$

where $q(x) \propto \exp(-(x - s_{\text{raw}})^2 / 2)$ is a Gaussian prior centered at the raw score. The cross-entropy term anchors the estimated distribution to the LVLM output, while the divergence term enforces a unimodal constraint via the Gaussian prior, suppressing symbol bias.

### 2. Weighted KL Divergence and Adaptive Weight α

The divergence term adopts a weighted KL divergence:

$$D_\alpha(p \| q) = (1 - \alpha) H(p, q) - \alpha H(p, p)$$

The weight $\alpha$ is determined adaptively based on the raw score:

$$\alpha = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(s_{\text{raw}} - \mu)^2}{2\sigma^2}\right)$$

where $\mu$ is the mean of candidate digits and $\sigma^2 = 0.1$. **Design Motivation**: When the LVLM predicts an extreme score (near the maximum or minimum), $\alpha$ is small, increasing the prior weight to more strongly suppress symbol bias; when the prediction is near the middle, $\alpha$ is larger, placing greater reliance on the LVLM's own probability. Setting $\alpha = 0.5$ reduces to the standard KL divergence.

### 3. Closed-Form Solution

The DISCODE decoder $\psi_\theta(\boldsymbol{h}) = \text{softmax}(W^\top \boldsymbol{h} + \boldsymbol{b})$ is implemented as a linear layer followed by softmax. Under the assumption that the LVLM prediction head is also linear, $p_{\text{LVLM}} = \text{softmax}(V^\top \boldsymbol{h}_T + \boldsymbol{c})$, the ATT loss admits a closed-form solution:

$$\hat{W} = \frac{1}{\alpha} V, \quad \hat{\boldsymbol{b}} = \frac{1-\alpha}{\alpha} \log \boldsymbol{q} + \frac{1}{\alpha} \boldsymbol{c}$$

This eliminates iterative optimization; each sample is solved independently and efficiently, realizing true test-time adaptation.

### 4. MCEval Benchmark Construction

MCEval covers 6 visual domains (real, painting, sketch, quickdraw, clipart, infograph), comprising 6,000 images × 3 captions = 18,000 image–text pairs. Candidate captions are generated using GPT-4o-mini, GPT-4o, Gemini 2.0 Flash, and Claude 3.5 Sonnet, and preference annotations are collected from 81 crowdworkers via three-person consensus.

## Key Experimental Results

### Table 1: Cross-Domain Performance on MCEval (Accuracy %)

| Metric | Type | Real | Painting | Sketch | Quickdraw | Clipart | Infograph | Mean |
|--------|------|------|----------|--------|-----------|---------|-----------|------|
| CIDEr | ref-based | 66.7 | 64.5 | 68.7 | 62.8 | 64.5 | 60.2 | 64.6 |
| Polos | ref-based | 81.3 | 75.0 | 77.6 | 76.8 | 74.5 | 69.0 | 75.7 |
| CLIP-S | ref-free | 79.2 | 78.0 | 78.3 | 75.4 | 73.9 | 66.7 | 75.3 |
| FLEUR | ref-free | 84.7 | 83.6 | 80.4 | 45.6 | 79.9 | 86.0 | 76.7 |
| G-VEval (GPT-4o) | ref-free | 86.0 | 80.2 | 81.2 | 76.9 | 80.6 | 81.0 | 81.0 |
| FLEUR† (72B) | ref-free | 86.9 | 84.3 | 83.1 | 76.3 | 82.0 | 82.3 | 82.5 |
| **DISCODE** | **ref-free** | **87.8** | **85.2** | **83.9** | **78.5** | **83.5** | **82.8** | **83.6** |

DISCODE achieves the best performance across all domains, with particularly notable gains in abstract domains such as Quickdraw (+2.2 vs. FLEUR†). FLEUR's score of only 45.6% on Quickdraw exposes the fragility of score smoothing.

### Table 2: Performance on Traditional Photorealistic Benchmarks

| Metric | Flickr8k-EX τ_c | Flickr8k-CF τ_b | Composite τ_c | Pascal-50S |
|--------|-----------------|-----------------|---------------|------------|
| CLIP-S | 51.2 | 34.4 | 53.8 | 80.9 |
| G-VEval (GPT-4o) | 59.7 | 38.7 | 63.0 | 82.3 |
| FLEUR† (72B) | 55.7 | 40.1 | 65.7 | 83.8 |
| **DISCODE-LV** | **56.1** | **40.2** | **66.0** | **84.5** |
| FLEUR† (IN-78B) | 56.9 | 36.4 | 64.2 | 80.8 |
| **DISCODE-IN** | **58.1** | **40.1** | **64.9** | **83.5** |

DISCODE consistently outperforms FLEUR in the photorealistic domain as well, surpassing GPT-4o-driven G-VEval on Flickr8k-CF.

### Table 3: Ablation Study

| Variant | FEX τ_c | FCF τ_b | Com τ_c | Pascal | MCEval |
|---------|---------|---------|---------|--------|--------|
| DISCODE (full) | 56.1 | 40.2 | 66.0 | 84.5 | 83.6 |
| w/o cross-entropy | 54.6 | 39.5 | 63.1 | 83.8 | 81.8 |
| w/o divergence | 49.9 | 39.9 | 64.4 | 83.0 | 80.9 |
| w/o adaptive α | 55.6 | 40.2 | 65.4 | 84.3 | 83.0 |

All three components contribute positively; the divergence term yields the largest gain on Flickr8k-Expert (+6.2 τ_c).

## Key Findings

1. **Symbol bias is the core bottleneck**: The digit "0" is systematically overestimated in token probability distributions, causing the scoring distribution to deviate from unimodality, especially in abstract domains.
2. **Closed-form solution enables efficient adaptation**: The analytic solution in DISCODE avoids iterative optimization overhead, with each sample solved independently.
3. **Model-agnostic applicability**: Consistent improvements are observed across 10 open-source LVLMs, with larger models benefiting more.
4. **Effect of scoring scale**: A continuous 0.0–1.0 scale slightly outperforms a discrete scale, as the decimal point position stabilizes autoregressive decoding.
5. **Weighted KL divergence is optimal**: Compared to Jensen-Shannon, Beta, and Rényi divergences, the weighted KLD achieves the best performance and admits a closed-form solution.

## Highlights & Insights

- **Zero fine-tuning**: No training data is required; the method adapts entirely at test time via a closed-form solution, making it plug-and-play.
- **Theory-driven design**: The Gaussian prior is motivated by the Central Limit Theorem as a principled constraint on the scoring distribution.
- **Cross-domain robustness**: DISCODE achieves 83.6% mean accuracy across six MCEval domains, outperforming FLEUR by 1.1 percentage points and G-VEval (GPT-4o) by 2.6 percentage points.
- **Benchmark contribution**: MCEval is the first image captioning evaluation benchmark spanning six visual domains, filling the gap in cross-domain evaluation.

## Limitations & Future Work

1. **Restricted to open-source models**: Extracting internal decoder features $\boldsymbol{h}_T$ is required, precluding application to closed-source APIs such as GPT-4o.
2. **Task scope**: Validation is currently limited to image captioning evaluation; extension to VQA, image generation evaluation, and other tasks remains unexplored.
3. **Gaussian prior assumption**: The assumption that human scores follow a unimodal distribution may not hold in highly polarized scenarios with bimodal human judgments.
4. **Dependence on linear head**: The closed-form solution requires the LVLM prediction head to be a linear layer followed by softmax; non-linear heads necessitate falling back to iterative optimization.

## Related Work & Insights

- **Reference-based metrics**: BLEU → CIDEr → BERTScore → Polos → DENEB, progressing from n-gram matching to embedding spaces to fine-tuned models.
- **Reference-free metrics**: CLIP-Score → PAC-S → HiFi-Score, leveraging the alignment capabilities of pretrained vision-language models.
- **LVLM-based evaluators**: FLEUR (score smoothing + LLaVA) and G-VEval (CoT + GPT-4o) share the core assumption $p = p_{\text{LVLM}}$; DISCODE breaks this limitation.
- **Test-time adaptation**: DISCODE draws inspiration from TTA but represents its first application to LVLM-based evaluation tasks.

## Rating

⭐⭐⭐⭐ — The method is elegant and concise (closed-form solution), experiments are comprehensive (10 models × 5 benchmarks), and the new benchmark is a valuable contribution. However, applicability is limited to image captioning evaluation and requires access to open-source model internals.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Leveraging Textual Compositional Reasoning for Robust Change Captioning](leveraging_textual_compositional_reasoning_for_robust_change_captioning.md)
- [\[ICML 2026\] Self-Captioning Multimodal Interaction Tuning: Amplifying Exploitable Redundancies for Robust Vision Language Models](../../ICML2026/multimodal_vlm/self-captioning_multimodal_interaction_tuning_amplifying_exploitable_redundancie.md)
- [\[CVPR 2026\] Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction](../../CVPR2026/multimodal_vlm/text-only_training_for_image_captioning_with_retrieval_augmentation_and_modality.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](../../ICCV2025/multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[ICCV 2025\] CaptionSmiths: Flexibly Controlling Language Pattern in Image Captioning](../../ICCV2025/multimodal_vlm/captionsmiths_flexibly_controlling_language_pattern_in_image_captioning.md)

</div>

<!-- RELATED:END -->
