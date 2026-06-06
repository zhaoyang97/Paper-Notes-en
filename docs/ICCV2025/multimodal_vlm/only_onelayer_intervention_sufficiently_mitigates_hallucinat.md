---
title: >-
  [Paper Note] ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models
description: >-
  [ICCV 2025][Multimodal VLM][hallucination mitigation] This paper proposes ONLY, a training-free single-layer intervention decoding method. It selects text-biased attention heads via the Text-to-Visual Entropy Ratio (TVER…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "hallucination mitigation"
  - "contrastive decoding"
  - "text-to-visual entropy ratio"
  - "training-free"
  - "single-layer intervention"
date: 2026-05-08
content_hash: deb44216380f8ef1
---

# ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models

**Conference**: ICCV 2025
**arXiv**: [2507.00898](https://arxiv.org/abs/2507.00898)  
**Code**: [https://github.com/zifuwan/ONLY](https://github.com/zifuwan/ONLY)  
**Area**: Multimodal VLM / Hallucination Mitigation
**Keywords**: hallucination mitigation, contrastive decoding, text-to-visual entropy ratio, training-free, single-layer intervention

## TL;DR
This paper proposes ONLY, a training-free single-layer intervention decoding method. It selects text-biased attention heads via the Text-to-Visual Entropy Ratio (TVER) to generate textually-enhanced logits, which are then used in adaptive contrastive or collaborative decoding against the original logits. With only 1.07× inference overhead, ONLY outperforms VCD/M3ID by 3.14% on POPE and reduces CHAIR_S by 6.2 points on CHAIR.

## Background & Motivation

**Background**: The dominant approach to mitigating hallucinations in LVLMs is contrastive decoding — comparing the original output with a perturbed version (e.g., VCD uses noisy images; M3ID removes the image). However, these methods require two or more full inference passes, doubling inference time and making them unsuitable for real-time applications.

**Limitations of Prior Work**: VCD requires 2.01× inference time and 1.05× GPU memory; M3ID requires 2.03×; OPERA requires 7.12×; HALC requires 6.52×. Although accuracy improves, the efficiency–performance trade-off is unreasonable — doubling inference time yields only marginal gains.

**Key Challenge**: Contrastive decoding requires two distributions to compare, but existing methods obtain them by running the full model twice, which is computationally wasteful. The key question is whether a "textually-enhanced" distribution can be constructed within a single inference pass.

**Goal**: Replace dual-inference with a single-layer intervention to obtain enhanced logits for contrastive decoding without significantly increasing computation.

**Key Insight**: The authors observe that when image input is perturbed (e.g., by adding noise), the entropy of textual attention increases while that of visual attention decreases. Directly selecting attention heads with high TVER can simulate the effect of "visual distortion" without actually perturbing the image.

**Core Idea**: Use the text-to-visual entropy ratio to filter attention heads and construct textually-enhanced outputs. This achieves contrastive decoding equivalent to dual-inference using only one additional layer of attention computation.

## Method

### Overall Architecture
During normal LVLM decoding, an additional Textual-Enhanced MHA output is computed: a single layer (default: layer 0) is selected, attention heads are filtered by TVER (heads with TVER ≥ mean are retained; others are zeroed out), and the enhanced output is connected to the final layer's output via residual connections and passed through an MLP to obtain textually-enhanced logits. Finally, the Manhattan distance between the two sets of logits adaptively determines whether contrastive or collaborative decoding is applied.

### Key Designs

1. **Text-to-Visual Entropy Ratio (TVER)**:

    - **Function**: Measures the degree to which each attention head disperses information across textual vs. visual tokens.
    - **Mechanism**: The attention matrix is split into $a^{\mathcal{T}}$ and $a^{\mathcal{V}}$ according to text/visual token indices, and normalized entropy is computed for each: $\text{TVER}_{\ell,i} = \frac{\text{Entropy}(a^{\mathcal{T}}_{\ell,i})}{\text{Entropy}(a^{\mathcal{V}}_{\ell,i})}$. A high TVER indicates that the head has high uncertainty over textual information (i.e., relies more on linguistic priors), while a low TVER indicates stronger focus on visual information.
    - **Key Findings**: TVER is highly correlated with hallucination — higher average response-level TVER corresponds to higher CHAIR_I (more hallucinations). At the token level, hallucinated and non-hallucinated tokens exhibit significantly different distributions of Manhattan distance between textually-enhanced and original logits.
    - **Design Motivation**: This approach extracts linguistic bias directly from the entropy characteristics of attention heads, without requiring actual image perturbation.

2. **Textual-Enhanced MHA (TE-MHA)**:

    - **Function**: In the selected layer $\tilde{\ell}$, heads with TVER above the layer average are retained while others are masked, producing a text-biased attention output.
    - **Mechanism**: $\tilde{a}_{\ell,i} = a_{\ell,i}$ if $\text{TVER}_{\ell,i} \geq \text{average}(\text{TVER}_\ell)$, else $0$. MHA is computed using the filtered attention. The enhanced output is connected to the original final-layer output via two residual connections: $\tilde{\bar{\mathcal{H}}}^{L-1}_t = \tilde{\mathcal{H}}^{\tilde{\ell}}_t + \mathcal{H}^{L-1}_t$, then passed through an MLP to obtain enhanced logits.
    - **Design Motivation**: Only one additional layer of attention computation is required (reusing existing K/V), incurring minimal overhead.

3. **Adaptive Decoding Strategy**:

    - **Function**: Dynamically selects collaborative or contrastive decoding based on the Manhattan distance between original and enhanced logits.
    - **Mechanism**: $d_t = \sum_{y_t} |p_\theta - \tilde{p}_\theta|$. When $d_t < \gamma$ (distributions are close), collaborative decoding is applied (weighted combination); when $d_t \geq \gamma$ (large discrepancy, potential hallucination risk), contrastive decoding is applied (subtracting enhanced logits).
    - **Formula**: ${f_\theta^{\text{final}}} = f_\theta + \alpha_1 \tilde{f}_\theta$ (collaborative) or $(1+\alpha_2)f_\theta - \alpha_2 \tilde{f}_\theta$ (contrastive).
    - **Design Motivation**: Not every token requires contrastive decoding. When the model is confident, collaboration enhances detail; when uncertain, contrastive decoding suppresses hallucination.

### Efficiency Analysis
Only one additional layer of attention computation is needed → 1.07× inference time, virtually no additional GPU memory (14,951 MB vs. 14,945 MB baseline). Compared to VCD (2.01×, 15,749 MB), M3ID (2.03×, 15,575 MB), OPERA (7.12×, 22,706 MB), and HALC (6.52×, 23,084 MB), the efficiency gain is substantial.

## Key Experimental Results

### POPE Benchmark (LLaVA-1.5, MS-COCO Random)

| Method | Accuracy | F1 | Inference Time | GPU Memory |
|--------|----------|----|----------------|------------|
| Regular | 83.44 | 83.09 | 1.00× | 14,945 MB |
| VCD | 87.15 | 85.45 | 2.01× | 15,749 MB |
| M3ID | 87.52 | 84.50 | 2.03× | 15,575 MB |
| **ONLY** | **89.10** | **87.85** | **1.07×** | **14,951 MB** |

### CHAIR Benchmark (LLaVA-1.5, Max token=128)

| Method | CHAIR_S↓ | CHAIR_I↓ | Len |
|--------|----------|----------|-----|
| Regular | 26.2 | 9.4 | 55.0 |
| VCD | 24.4 | 7.9 | 54.4 |
| M3ID | 21.4 | 6.3 | 56.6 |
| HALC | 21.7 | 7.1 | 51.0 |
| **ONLY** | **20.0** | **6.2** | 49.8 |

MME-Hallucination (LLaVA-1.5): ONLY 635.55 vs. second-best M3ID 598.11 (+37.44).

### Ablation Study

| Configuration | POPE F1 | CHAIR_S↓ | Note |
|---------------|---------|----------|------|
| Regular | 81.27 | 26.2 | Baseline |
| Zero Visual Attention | 84.82 | 21.2 | Directly zeroing visual attention |
| Noisy Visual Attention | 84.79 | 22.1 | Adding noise to visual attention |
| 2× Textual Attention | 84.96 | 21.6 | Doubling textual attention |
| Sum Ratio Selection | 84.46 | 23.1 | Head selection by attention weight sum ratio |
| **TVER Selection (Ours)** | **85.37** | **20.0** | Head selection by entropy ratio |

### Key Findings
- **ONLY surpasses all 2×+ methods at 1.07× cost**: POPE accuracy 89.10% vs. VCD 87.15% (+1.95%); CHAIR_S 20.0 vs. VCD 24.4 (−4.4). Better results are achieved with approximately 1/30 of the additional time.
- **Layer selection has minimal impact**: Intervention at any of the 32 layers yields F1 in the range 84.62–85.37, while VCD achieves only 83.38 and M3ID only 84.05, demonstrating strong robustness of the method.
- **TVER outperforms direct attention manipulation**: Directly zeroing or adding noise to visual attention is inferior to TVER-based head selection, as TVER precisely localizes the heads with the strongest linguistic bias from an information-theoretic perspective.
- Optimal hyperparameters: $\gamma = 0.2$ (LLaVA-1.5), $\alpha_1 = 3$, $\alpha_2 = 1$.
- The method generalizes consistently to stronger models LLaVA-NeXT-7B/13B, outperforming VCD/M3ID in all settings.

## Highlights & Insights
- **Information-theoretic head selection**: TVER is grounded in the intuition of conditional entropy — high $H(\mathcal{T}|\mathcal{V})$ indicates that textual attention is insensitive to visual information (resembling pure linguistic reasoning). Selecting these heads amplifies linguistic priors for contrastive decoding, which is theoretically more principled than VCD's noise augmentation or M3ID's image removal.
- **Adaptive contrastive/collaborative switching**: Rather than applying contrastive decoding uniformly, the method uses distance-based switching — collaborative decoding enhances detail when the model is confident, while contrastive decoding suppresses hallucinations when uncertainty is high. This captures the inherently token-level dynamics of hallucination generation.
- **Extreme efficiency**: 1.07× inference time and zero additional memory make this the most efficient hallucination mitigation method currently known. The core insight is that obtaining a contrastive signal does not require running the full model twice — selectively activating certain attention heads within a single layer suffices.

## Limitations & Future Work
- Validation is limited to LLaVA-1.5, InstructBLIP, and Qwen-VL; more recent models (InternVL, Qwen2-VL, etc.) have not been tested.
- Hyperparameters $\alpha_1$, $\alpha_2$, and $\gamma$ require tuning for different models (e.g., $\gamma = 0.2$ for LLaVA, $\gamma = 0.4$ for QwenVL).
- Generated responses are slightly shorter (49.8 vs. 55.0 tokens on average), potentially omitting some correct information.
- The method has not been evaluated on video understanding tasks.
- Only one layer is used for intervention; combining multiple layers might further improve performance at the cost of additional overhead.

## Related Work & Insights
- **vs. VCD**: VCD applies diffusion noise to perturb the image for contrastive decoding, requiring 2× inference. ONLY extracts linguistic bias signals directly from attention entropy, achieving superior results at 1.07× inference.
- **vs. M3ID**: M3ID removes the image to obtain an unconditional prediction for contrast, also requiring 2× inference. TVER-based head selection in ONLY achieves a functionally equivalent effect of "suppressing vision, amplifying text" with only one additional layer of computation.
- **vs. MRGD**: MRGD uses an independent reward model to guide search, while ONLY modifies attention directly during decoding. Both are inference-time methods but differ in approach: MRGD focuses on search strategy, while ONLY focuses on logit correction.
- **vs. ShortV**: ShortV identifies layers where many visual tokens are ineffective and skips them. ONLY exploits inter-layer variability from a complementary angle — selectively enhancing textual attention within a single layer to extract linguistic bias. Both works corroborate the observation that "different layers in MLLMs contribute unevenly to visual and textual processing."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ TVER-driven head selection combined with single-layer intervention is an elegant design with clear information-theoretic motivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six benchmarks, three LVLMs, efficiency comparisons, extensive ablations (layer selection, strategy comparison, hyperparameter sensitivity), and GPT-4V evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem analysis is precise (Figure 1's efficiency–performance trade-off plot); the correlation between TVER and hallucination is well-validated (Figure 4).
- **Value**: ⭐⭐⭐⭐⭐ Surpassing 2×+ methods at 1.07× cost offers extremely high practical value; ONLY is likely to become a default hallucination mitigation baseline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Mitigating Object Hallucinations via Sentence-Level Early Intervention](mitigating_object_hallucinations_via_sentence-level_early_intervention.md)
- [\[ICCV 2025\] TAB: Transformer Attention Bottlenecks enable User Intervention and Debugging in Vision-Language Models](tab_transformer_attention_bottlenecks_enable_user_intervention_and_debugging_in_.md)
- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](../../CVPR2026/multimodal_vlm/residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)
- [\[ICCV 2025\] One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models](one_perturbation_is_enough_on_generating_universal_adversarial_perturbations_aga.md)
- [\[NeurIPS 2025\] When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations](../../NeurIPS2025/multimodal_vlm/when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_.md)

</div>

<!-- RELATED:END -->
