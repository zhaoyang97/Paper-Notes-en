---
title: >-
  [Paper Note] Focusing Where Vision Matters: Selective Training for Large Vision Language Models via Visual Information Gain
description: >-
  [ICML 2026][Multimodal VLM][Visual Information Gain] This paper proposes **Visual Information Gain (VIG)**—a visual dependency metric based on the log-likelihood ratio of perplexity between "with image vs. without image…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Visual Information Gain"
  - "Selective Training"
  - "Language Bias"
  - "Token-level Loss Mask"
  - "Data-efficient Instruction Tuning"
date: 2026-05-08
content_hash: 0abdebd695148d63
---

# Focusing Where Vision Matters: Selective Training for Large Vision Language Models via Visual Information Gain

**Conference**: ICML 2026  
**arXiv**: [2602.17186](https://arxiv.org/abs/2602.17186)  
**Code**: TBD  
**Area**: Multimodal VLM  
**Keywords**: Visual Information Gain, Selective Training, Language Bias, Token-level Loss Mask, Data-efficient Instruction Tuning

## TL;DR
This paper proposes **Visual Information Gain (VIG)**—a visual dependency metric based on the log-likelihood ratio of perplexity between "with image vs. without image (proxied by a blurred image)". It quantifies whether a specific data sample or token truly utilizes visual information. Based on this, selective instruction tuning is performed: loss is calculated only on high-VIG samples and tokens. This allows LLaVA-1.5-13B to outperform vanilla training across all metrics using only 21% of active tokens while significantly mitigating language bias and hallucinations.

## Background & Motivation

**Background**: LVLMs (e.g., LLaVA, ShareGPT4V, Qwen-VL) solve VQA, captioning, and multimodal reasoning by jointly tuning a vision encoder, an adapter, and an LLM. The mainstream training approach feeds all instruction-tuning data with equal weight for next-token SFT.

**Limitations of Prior Work**: Models frequently exhibit "language bias"—even when images are provided, outputs are dominated by text priors, leading to visual ignorance and hallucinations (describing non-existent objects). Existing mitigation strategies fall into three categories: (1) contrastive decoding during inference comparing vision/no-vision outputs; (2) modifying attention mechanisms to force higher weights on image tokens; (3) using stronger models to generate higher-quality instruction data. However, none of these **quantify how much information each sample or token actually gains from the image**.

**Key Challenge**: In typical LVLM instruction data, there are samples that require the image to answer (e.g., color, spatial relations) and samples that can be answered by common sense alone. At the token level, visually grounded content words (e.g., "white", "sitting", "lying") are treated equally by cross-entropy as purely syntactic words (e.g., "a", "the", "of"). This "indiscriminate supervision" encourages the model to learn "language shortcuts," as predicting syntactic words is much easier than predicting vision-related words.

**Goal**: (1) Design a metric to quantify "visual information contribution" at both sample and token granularities; (2) Perform selective training to discard low-VIG samples and mask low-VIG tokens; (3) Improve visual understanding and reduce hallucinations under reduced supervision.

**Key Insight**: From an information theory perspective—if an image truly aids prediction, its presence should reduce the model's uncertainty (perplexity) regarding the answer. Conversely, if perplexity remains unchanged (or increases) with the image, the sample/token is likely not utilizing visual signals.

**Core Idea**: Define $\mathrm{VIG} = \log\frac{\mathrm{PPL}(A|Q)}{\mathrm{PPL}(A|Q,I)}$, where "no-vision" is approximated by a "blurred image." A shared threshold $\tau_p$ is used to select both high-relevance samples and tokens.

## Method

### Overall Architecture
The method consists of two steps: **(1) VIG Calculation**: For each multimodal instruction sample $(I, Q, A)$, two forward passes are run using a pre-trained aligned LVLM—one with the original image and one with a blurred image (acting as a "no-vision" proxy). The perplexity of the answer is obtained for both cases, and the log-ratio defines the sample-level $\mathrm{VIG}_i$, which is further decomposed into token-level $\mathrm{VIG}_{i,t}$. **(2) Selective Fine-tuning**: Samples are ranked by $\mathrm{VIG}_i$ and the top-$p\%$ are retained (default $p=70$, threshold $\tau_{70}$). Within these retained samples, loss is calculated only for tokens satisfying $\mathrm{VIG}_{i,t} \geq \tau_{70}$. Other tokens participate in the forward pass but do not contribute to gradients.

### Key Designs

1. **Visual Information Gain (Perplexity-based Visual Dependency Metric)**:
    - **Function**: Quantifies how much the model's uncertainty about the answer decreases when given an image relative to not having one, serving as a proxy for visual information contribution.
    - **Mechanism**: Defined as $\mathrm{VIG} = \log(\mathrm{PPL}(A|Q)/\mathrm{PPL}(A|Q,I)) = \mathcal{L}(A|Q) - \mathcal{L}(A|Q,I)$, representing the difference between cross-entropy without and with the image. Under one-hot supervision, this simplifies to $\mathrm{VIG} = D_{\text{KL}}(p_{A|Q}\|q_Q) - D_{\text{KL}}(p_{A|I,Q}\|q_{I,Q})$, reflecting the correction of the predicted distribution by visual input. Token-level decomposition is $\mathrm{VIG}_i = \frac{1}{T_i} \sum_t \mathrm{VIG}_{i,t}$, where $\mathrm{VIG}_{i,t} = -\log q_\theta(a_t|a_{<t},Q) + \log q_\theta(a_t|a_{<t},Q,z_v)$.
    - **Design Motivation**: Log-ratio of PPL is chosen over KL because it naturally decomposes into the difference of token-level losses without additional definitions. Using a blurred image instead of "no image" solves the requirement for visual input in LVLM architectures while keeping the text sequence identical, isolating only the visual signal.

2. **Dual-Granularity Selection (Shared Threshold $\tau_p$)**:
    - **Function**: Prunes instruction data into a subset of "what needs to be learned and seen," focusing the limited gradient budget on vision-critical segments.
    - **Mechanism**: Samples are first sorted by sample-level $\mathrm{VIG}_i$ in descending order to select the top-$p\%$, $\mathcal{S}_p = \{i \mid \mathrm{VIG}_i \geq \tau_p\}$. Then, within each retained sample, the **same threshold** $\tau_p$ is used for token selection $\mathcal{T}_i^+ = \{t \mid \mathrm{VIG}_{i,t} \geq \tau_p\}$. Cross-entropy is computed only for tokens in $\bigcup_{i \in \mathcal{S}_p} \mathcal{T}_i^+$. Non-selected tokens are still passed through the model to maintain context, but their loss is masked.
    - **Design Motivation**: Sharing $\tau_p$ avoids introducing new hyperparameters. The default $p=70$ is the empirical sweet spot; lower values (e.g., $p=30$) lead to underfitting, while higher values (e.g., $p=90$) degenerate toward vanilla training. This shared threshold ensures that in long answers, only visual keywords contribute to the loss. Observations show that visual content words like "white/lying/sitting" have loss differences of 3-6, while syntactic words like "of/the/a" are near 0 or negative.

3. **Blurred Image Proxy + VIG Calculation Post-Alignment**:
    - **Function**: Simulates the conditional distribution $q_Q$ without visual input without architecture changes, ensuring the VIG metric remains credible.
    - **Mechanism**: The input image $I$ is replaced with a Gaussian-blurred version (following Xing et al. 2025) to obtain $\mathrm{PPL}(A|Q)$. VIG is calculated **after pre-training (adapter alignment) but before instruction tuning**. At this stage, visual features are preliminary aligned with language space, but the model has not yet overfitted to instruction data, ensuring VIG reflects visual utility rather than noise from untrained downstream tasks.
    - **Design Motivation**: Using a blurred image rather than a black image or zero vector maintains the normal forward pipeline of the vision encoder, avoiding Out-of-Distribution (OOD) issues that could blow up perplexity. Positioning after alignment makes VIG a "prior filter" rather than an "after-the-fact diagnostic"—calculating once and reusing throughout.

### Loss & Training
LLaVA-1.5 7B/13B and ShareGPT4V 7B undergo alignment with 558K (or 1.2M) image-caption pairs followed by SFT on ~665K instruction data. Open-Qwen2VL 2B is tuned on a 1M MAmmoTH-VL subset. All experiments fix $p=70$, calculating VIG only for multimodal samples and keeping text-only samples as is. All other hyperparameters remain consistent with vanilla baselines, except for the loss mask.

## Key Experimental Results

### Main Results
Comparison across four LVLMs and two benchmark categories (visual understanding + hallucination):

| Model | # Active Tokens | LLaVA-W ↑ | MMVet ↑ | CV-Bench ↑ | POPE F1 ↑ | CHAIR $C_S$ ↓ | MMHal Hall. ↓ |
|------|-----------------|-----------|---------|------------|-----------|---------------|----------------|
| LLaVA-1.5 7B (vanilla) | 58.61M (100%) | 59.02 | 28.62 | 59.18 | 87.08 | 52.93 | 71.25 |
| LLaVA-1.5 7B + VIG | 38.45M (-34%) | **61.22** | **32.71** | **62.48** | **87.47** | **47.00** | **62.78** |
| LLaVA-1.5 13B (vanilla) | 58.61M (100%) | 72.01 | 36.19 | 60.16 | 87.05 | 51.96 | 67.09 |
| LLaVA-1.5 13B + VIG | 12.14M (**-79%**) | **73.45** | **36.87** | **62.89** | **87.53** | **48.19** | **63.11** |

On 13B, using only **21%** of active tokens outperformed the vanilla model on 8/8 reported metrics. On 7B, MMVet improved by 4.1 points, and the CHAIR $C_S$ hallucination rate dropped from 52.93 to 47.00.

### Ablation Study

| Configuration | Meaning | Key Observation |
|------|------|---------|
| Full data, full token (vanilla) | No sample/token selection | Baseline |
| Top-70% sample, full token | Sample-level selection only | Improves most metrics, but limited hallucination relief |
| Full data, token mask | Token-level selection only | Slight gain in visual understanding, high token waste |
| Top-70% sample + token mask (Default) | Dual granularity, shared $\tau_{70}$ | Maximum benefit for both data efficiency and hallucination relief |
| Threshold $p$=30/50/70/90 | Selection ratio sensitivity | $p=70$ is the optimal trade-off; too low underfits, too high degenerates |

### Key Findings
- **Token-level selection is the true source of hallucination suppression**: Sample selection primarily makes training efficient, but token masking removes purely syntactic tokens (loss diff near 0). This forces gradients to concentrate on visual keywords, compelling the model to "actually look at the image."
- **VIG strongly correlates with benchmark modality dependency**: VIG distributions for COCO/CV-Bench/POPE are positive (vision-heavy), while GQA/SQA are more negative (text-heavy), suggesting VIG can serve as a "visual dependency thermometer" for benchmarks.
- **VIG is sensitive to image content**: For fixed Q-A, a correct image yields VIG=0.923, a partially correct (wrong attribute) image yields VIG=0.409, and a contradictory image yields VIG=-0.520. VIG distinguishes not just relevance, but visual correctness.
- **Scaling Positive Feedback**: On 13B, active tokens decreased by 79% while performance improved. This suggests larger models are more sensitive to "high-quality sparse supervision."

## Highlights & Insights
- **Blurred images as "counterfactual visual input" is an efficient trick**: It bypasses the engineering hurdle of LVLMs not accepting None inputs while maintaining the forward pipeline, making VIG a "plug-and-play" metric for any LVLM without changing architecture or loss.
- **The shared threshold $\tau_p$ is simple and elegant**: Using one threshold for both levels ensures that only high-density tokens in high-relevance samples enter the loss, effectively carving out a "high-VIG sub-region" in the sample-token grid.
- **VIG serves as a universal "visual importance" score**: Beyond selective training, it can be used to normalize modality dependency in benchmarks, mine hallucination data, or shape rewards in multimodal RLHF.

## Limitations & Future Work
- VIG's "no-vision" proxy relies on blurred image settings (kernel size/intensity), which weren't systematically compared. Residual low-frequency cues in blurred images might underestimate true VIG.
- VIG is a static score calculated once post-alignment. It does not update dynamically during training; capacities developed later might change the value of certain samples.
- The threshold $p=70$ was fixed across models. Using the same ratio for both 2B and 13B is likely suboptimal.
- Experiments focused on the SFT stage of LLaVA/Qwen architectures. Its potential in RLHF/DPO or for other modalities (Video/3D) remains unverified.

## Related Work & Insights
- **vs. Contrastive Decoding (Leng et al. 2024)**: That method compares "vision vs. no-vision" outputs during inference (running twice). Ours moves this concept to training as a one-time calculation (VIG) with zero inference overhead.
- **vs. Selective Modeling for LLM (Lin et al. 2024)**: LLM selective training uses reference loss for token selection. Ours replaces the "reference" with a "no-vision control," extending the concept to the multimodal domain with a sample-level dimension.
- **vs. Data Quality Filtering (e.g., LLaVA-OneVision)**: Those methods rely on stronger models for data regeneration, which is expensive. VIG is essentially free and model-specific.

## Rating
- Novelty: ⭐⭐⭐⭐ Using perplexity difference as a measure of visual information and applying it to dual-granularity selection is a simple but first-of-its-kind establishment.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four LVLMs across 8 benchmarks + visual/hallucination dual axes; however, lacks RLHF validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Derivations are clear; visualizations (Figs 1-4, Table 2) are highly persuasive.
- Value: ⭐⭐⭐⭐ Provides a "plug-and-play" efficiency tool for LVLM pipelines; 4–5× speedup on 13B with performance gains is substantial.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] VEENA: Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow](interpreting_and_enhancing_emotional_circuits_in_large_vision-language_models_vi.md)
- [\[ICML 2026\] On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression](on_the_adversarial_robustness_of_large_vision-language_models_under_visual_token.md)
- [\[ICML 2026\] Large Vision-Language Models Get Lost in Attention](large_vision-language_models_get_lost_in_attention.md)
- [\[ICML 2026\] Jailbreaking Vision-Language Models Through the Visual Modality](jailbreaking_vision-language_models_through_the_visual_modality.md)
- [\[ICML 2026\] Uncovering Visual Counting Bottlenecks in Vision-Language Models](unveiling_the_visual_counting_bottleneck_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
