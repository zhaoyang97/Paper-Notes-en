---
title: >-
  [Paper Note] Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs
description: >-
  [Hallucination Detection] Proposed Mixture of Decoding (MoD), which utilizes JS divergence to measure the correctness of the model's attention to image tokens. When the attention is correct, complementary decoding is used to amplify key information, whereas when the attention is incorrect, contrastive decoding is adopted to suppress misleading information, thereby adaptively mitigating hallucinations in multimodal large language models.
tags:
  - "Hallucination Detection"
date: 2026-05-08
content_hash: 46bb7aa32cda0134
---

# Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs

- **Conference**: ACL 2025
- **arXiv**: [2505.17061](https://arxiv.org/abs/2505.17061)
- **Code**: [xlchen0205/MoD](https://github.com/xlchen0205/MoD)
- **Area**: Multimodal VLM
- **Keywords**: Multimodal Large Language Models, Hallucination Mitigation, Contrastive Decoding, Attention Mechanism, Adaptive Decoding

## TL;DR

Proposed Mixture of Decoding (MoD), which utilizes JS divergence to measure the correctness of the model's attention to image tokens. When the attention is correct, complementary decoding is used to amplify key information, whereas when the attention is incorrect, contrastive decoding is adopted to suppress misleading information, thereby adaptively mitigating hallucinations in multimodal large language models.

## Background & Motivation

Large Vision-Language Models (LVLMs) demonstrate outstanding performance in various visual tasks, but the "hallucination" problem—where generated text is inconsistent with visual information—severely limits their reliability. Existing contrastive decoding methods exhibit obvious limitations:

**VCD** and **M3ID** primarily attribute hallucinations to language prior biases. VCD obtains hallucinated logits for contrast by adding Gaussian noise to images, and M3ID does so by removing image inputs, but they ignore the impact of visual inputs themselves (such as spurious correlations) on hallucinations.

**AvisC** considers the attention distribution, assuming that image tokens with excessively high attention weights trigger hallucinations, but it uniformly weakens these high-attention tokens without distinguishing whether the attention is correct. When the model has already correctly focused on relevant information, AvisC instead weakens useful signals, leading to unreliable contrastive results.

The core insight is that **the model's attention distribution can be either correct or incorrect**. The key lies in judging the correctness of the attention and then dynamically adjusting the decoding strategy—which is the starting point of MoD. The authors found that using JS divergence to measure the consistency between the original output and the output generated based only on high-attention tokens can effectively distinguish between hallucinated and non-hallucinated outputs (on POPE, non-hallucinated samples cluster in the low JS divergence region, and on CHAIR, the Pearson correlation coefficient between JS divergence and CHAIR_i is as high as 0.85).

## Method

### Overall Architecture

MoD consists of three core steps:

1. **Extraction of Attentive Image Tokens**: Utilizing the average attention weight of the last input token across all layers and attention heads, selecting a top-$\lambda$ proportion of high-attention image tokens and zeroing out the rest to obtain $v_{att}$.
2. **Generation of Dual-path Logits**: Performing forward passes based on the original image tokens $v$ and the attentive image tokens $v_{att}$ respectively, yielding two sets of output probability distributions.
3. **JS Divergence Discrimination + Adaptive Decoding**: Calculating the JS divergence between the two distributions. When it is below a threshold $\gamma$, the complementary strategy is adopted; when it is above the threshold, the contrastive strategy is used.

### Key Design 1: Attentive Image Token Extraction

Leveraging the autoregressive property, the model takes the attention weights of the last token in the input sequence over all image tokens, averaging them across all layers and heads:

$$A^I = \frac{1}{L \cdot H} \sum_{l=1}^{L} \sum_{h=1}^{H} A_l[\cdot, h, -1, IDX^I]$$

Then, the indices of the top-$\lambda$ proportion of image tokens with the highest attention weights, $IDX^I_{att}$, are selected, and the remaining image tokens are set to zero, yielding $v_{att}$. By default, $\lambda = 0.2$, meaning 20% of the image tokens are retained.

The advantage of this design is that it does not rely on the attention of a specific layer or attention head, but aggregates information across all layers and heads to obtain a global understanding of the image tokens by the model.

### Key Design 2: Adaptive Decoding Strategy Based on JS Divergence

The JS divergence between the two output distributions is calculated to judge attention correctness:

$$d(v, v_{att}) = D_{JS}[p_\theta(y_t | v, x, y_{<t}) \| p_\theta(y_t | v_{att}, x, y_{<t})]$$

Depending on the relationship between $d(v, v_{att})$ and the threshold $\gamma$, different decoding strategies are selected:

- **Correct Attention ($d \leq \gamma$, high consistency)**: Complementary decoding is used, adding the two sets of logits to amplify key information:

$$y_t \sim \text{softmax}[\text{logit}_\theta(y_t|v,x,y_{<t}) + \alpha_1 \cdot \text{logit}_\theta(y_t|v_{att},x,y_{<t})]$$

- **Incorrect Attention ($d > \gamma$, low consistency)**: Contrastive decoding is used, subtracting the attentive logits from the original logits to suppress misleading information:

$$y_t \sim \text{softmax}[(1+\alpha_2) \cdot \text{logit}_\theta(y_t|v,x,y_{<t}) - \alpha_2 \cdot \text{logit}_\theta(y_t|v_{att},x,y_{<t})]$$

The default hyperparameters are $\alpha_1=4$, $\alpha_2=1$, and $\gamma=0.05$. This single set of parameters is shared across all tasks and models without the need for scene-specific tuning.

### Key Design 3: Consistency as a Hallucination Indicator

The intuition behind why JS divergence can effectively distinguish hallucinations is: when the model correctly focuses on relevant image regions, the output generated by retaining only these high-attention tokens should be highly consistent with the original output (low JS divergence); when the model incorrectly focuses on irrelevant regions, retaining only these tokens leads to a significant deviation from the original output (high JS divergence). Experiments validate this—on POPE, non-hallucinated outputs cluster in low JS divergence regions, and on CHAIR, JS divergence exhibits a strong positive correlation with the hallucination rate (Pearson r=0.85, p<0.01).

## Key Experimental Results

### Table 1: POPE Benchmark (MS-COCO, Random Setting)

| Method | LLaVA-1.5 Acc | LLaVA-1.5 F1 | Qwen-VL Acc | Qwen-VL F1 | LLaVA-NEXT Acc | LLaVA-NEXT F1 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Sampling | 83.8 | 84.2 | 84.9 | 82.9 | 84.4 | 82.3 |
| VCD | 85.0 | 84.2 | 85.5 | 83.6 | 86.0 | 84.3 |
| M3ID | 86.1 | 85.0 | 85.3 | 83.4 | 85.5 | 83.6 |
| AvisC | 82.3 | 83.5 | 82.9 | 80.0 | 85.2 | 82.8 |
| **MoD** | **89.2** | **89.1** | **86.0** | **84.1** | **86.6** | **84.8** |

MoD achieves the best performance across all three POPE settings (random/popular/adversarial), outperforming the runner-up by 3.1 points in Accuracy and 4.1 points in F1 on LLaVA-1.5.

### Table 2: CHAIR Benchmark (Generative Captioning Task)

| Method | LLaVA-1.5 CHAIR_s↓ | CHAIR_i↓ | Recall↑ | LLaVA-NEXT CHAIR_s↓ | CHAIR_i↓ | Recall↑ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Sampling | 52.8 | 15.9 | 77.3 | 35.8 | 12.0 | 59.5 |
| VCD | 51.0 | 14.9 | 77.2 | 40.2 | 10.7 | 62.1 |
| AvisC | 44.0 | 13.7 | 72.9 | 40.4 | 12.4 | 60.0 |
| **MoD** | **42.6** | **12.4** | **78.9** | **33.6** | **9.6** | **61.4** |

MoD maintains or even improves Recall while reducing the hallucination rate, indicating that it not only reduces erroneous content but also preserves description completeness. Notably, VCD and AvisC worsen hallucinations on LLaVA-NEXT.

### Table 3: MME Benchmark (MME Score)

| Model | Sampling | VCD | M3ID | AvisC | **MoD** |
|------|:---:|:---:|:---:|:---:|:---:|
| LLaVA-1.5 | 510.0 | 531.7 | 553.3 | 596.7 | **638.3** |
| Qwen-VL | 581.7 | 593.3 | 586.7 | 578.3 | **613.3** |
| LLaVA-NEXT | 595.0 | 611.7 | 608.3 | 613.3 | **653.3** |

MoD outperforms the runner-up by 41.6, 20.0, and 40.0 points on the three models, respectively.

## Key Findings

1. **JS divergence is an effective hallucination indicator**: The consistency between the original output and the attentive token output accurately distinguishes between hallucinations and non-hallucinations, with a Pearson correlation coefficient of 0.85.
2. **Adaptive strategy outperforms single strategies**: Ablation studies show that MoD scores 23.3 and 20.0 points higher (MME) than using complementary decoding or contrastive decoding alone, demonstrating the necessity of dynamically switching strategies.
3. **Strong robustness to hyperparameters**: Within the range of 0.02-0.08 for $\gamma$, MoD consistently outperforms single methods. Furthermore, the same set of hyperparameters is shared across all tasks and models, eliminating the need for scene-by-scene tuning.
4. **Model-agnosticism**: Consistent improvements are achieved across three different architectures (LLaVA-1.5, Qwen-VL, LLaVA-NEXT). Even when certain methods (such as VCD and AvisC) aggravate hallucinations, MoD remains robustly effective.
5. **AMBER Comprehensive Score**: MoD's AMBER Scores on the three models are 2.2, 0.7, and 2.6 points higher than the runner-up, delivering the best performance on both discriminative and generative tasks.

## Highlights & Insights

- **Precise problem decomposition**: Using "whether the attention is correct" as the conditional trigger for switching decoding strategies captures a core dimension ignored by existing methods—the uncertainty of attention distribution correctness, which should not be treated uniformly.
- **Simple and elegant design**: No additional training, external knowledge, or repeated sampling is required. Adaptive switching is achieved using only a JS divergence threshold, striking an excellent balance between complexity and performance.
- **Novelty of complementary decoding**: Most contrastive decoding works focus only on "what to subtract". MoD introduces the complementary concept for the first time—adding components to amplify key information when the attention is correct, stepping outside the traditional paradigm of contrastive decoding.
- **Significant improvement in Precision**: MoD outperforms other methods by up to 6.8 points in Precision on POPE, demonstrating that it effectively suppresses the bias of LVLMs to answer "Yes," making the model generate more cautiously.

## Limitations & Future Work

1. **Doubled inference overhead**: Like other contrastive decoding methods, MoD requires two forward passes, which approximately doubles the inference latency.
2. **Coarse masking strategy**: Currently, zeroing out low-attention tokens directly may lose positional information (leading to a slight drop on the position subset of MME). More refined strategies (such as pooling to retain partial information) might yield further improvements.
3. **Globally fixed threshold**: Having $\gamma=0.05$ treats all token positions equally, yet hallucinations might be more prone to occur at specific generation stages. A dynamic threshold could be superior.
4. **Unaddressed training data biases**: As an inference-time method, MoD cannot resolve the inherent bias issues present in the training data.

## Related Work & Insights

This work belongs to the line of **inference-time contrastive decoding** for LVLM hallucination mitigation. Unlike VCD (image noise-adding), M3ID (image removal), and AvisC (contrast with high-attention tokens), the core innovation of MoD lies in **not pre-assuming whether the attention is good or bad**, but dynamically judging it using a consistency metric. Concurrently, DeGF also adopts a similar adaptive idea (judging consistency through image generation), but MoD's solution is much lighter—directly utilizing the model's internal attention information without extra image generation steps.

Insight: This two-stage "discriminate-then-decide" approach can be extended to other scenarios, such as judging the relevance between retrieved documents and queries in RAG before deciding whether to use them, or adjusting generation strategies in multi-turn dialogues after determining if contextual attention is correct.

## Rating

⭐⭐⭐⭐ — The methodology is clear, the experiments are comprehensive, and the design is simple yet effective. Achieving adaptive decoding via consistency measurement is an elegant solution. The limitations lie in the doubled inference overhead and the relatively coarse masking strategy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [\[ACL 2025\] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models](retrieval_visual_contrastive_decoding_to_mitigate_object_hallucinations_in_large.md)
- [\[CVPR 2026\] MAD: Modality-Adaptive Decoding for Mitigating Cross-Modal Hallucinations in Multimodal Large Language Models](../../CVPR2026/hallucination/mad_modality-adaptive_decoding_for_mitigating_cross-modal_hallucinations_in_mult.md)
- [\[CVPR 2025\] Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding](../../CVPR2025/hallucination/seeing_far_and_clearly_mitigating_hallucinations_in_mllms_with_attention_causal_.md)
- [\[ACL 2026\] Through the Magnifying Glass: Adaptive Perception Magnification for Hallucination-Free VLM Decoding](../../ACL2026/hallucination/through_the_magnifying_glass_adaptive_perception_magnification_for_hallucination.md)

</div>

<!-- RELATED:END -->
