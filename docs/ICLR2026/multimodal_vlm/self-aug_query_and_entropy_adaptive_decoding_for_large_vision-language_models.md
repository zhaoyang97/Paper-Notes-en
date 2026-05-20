---
title: >-
  [Paper Note] Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models
description: >-
  [ICLR 2026][Multimodal VLM][visual contrastive decoding] This paper proposes Self-Aug, a training-free decoding strategy that employs Self-Augmentation Selection (SAS) Prompting to enable LVLMs to leverage their own para…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "visual contrastive decoding"
  - "hallucination mitigation"
  - "self-augmentation"
  - "entropy-aware thresholding"
  - "training-free"
date: 2026-05-08
content_hash: 76a63e14e173a685
---

# Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.13315](https://arxiv.org/abs/2510.13315)  
**Code**: [https://eunwooim.github.io/selfaug](https://eunwooim.github.io/selfaug)  
**Area**: Multimodal VLM / Decoding Strategy
**Keywords**: visual contrastive decoding, hallucination mitigation, self-augmentation, entropy-aware thresholding, training-free

## TL;DR
This paper proposes Self-Aug, a training-free decoding strategy that employs Self-Augmentation Selection (SAS) Prompting to enable LVLMs to leverage their own parametric knowledge for dynamically selecting query-semantically-aligned visual augmentations. It further introduces the Sparsity Adaptive Truncation (SAT) algorithm, which exploits the full entropy of the output distribution to dynamically regulate candidate token set size. Self-Aug consistently outperforms existing contrastive decoding methods across 5 LVLMs and 7 benchmarks.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) have achieved remarkable performance in multimodal understanding and generation, yet inherit hallucination problems from their underlying language models—generating content that appears plausible but is factually incorrect. Visual Contrastive Decoding (VCD) is a promising training-free hallucination mitigation strategy that improves factual consistency by contrasting standard outputs against "amateur" outputs produced from degraded visual inputs.

**Limitations of Prior Work**: Existing VCD methods exhibit two fundamental limitations. First, visual augmentation selection is decoupled from the textual query—all prior methods apply query-agnostic generic augmentations (e.g., random noise), whereas different queries demand entirely different reasoning capabilities; for instance, "identify the object in the image" and "solve the handwritten math problem" differ substantially in their sensitivity to visual perturbations. Although VACoDe attempts dynamic augmentation selection, it relies solely on the distributional divergence of the first token as a decision proxy—an empirical heuristic that cannot guarantee optimality over the full generation sequence and underperforms in open-ended generation. Second, existing Adaptive Plausibility Constraints (APC) set thresholds based only on the maximum logit value, completely ignoring the rich information encoded in the full output distribution, which leads to incorrect token discarding under low-confidence conditions.

**Key Challenge**: Contrastive decoding relies on visual perturbations to amplify output differences, but query-agnostic generic augmentations fail to produce maximally informative contrasts. Simultaneously, candidate token filtering requires balancing precision and safety, yet existing methods lack awareness of model uncertainty.

**Goal**: (1) How can the selection of visual augmentations be aligned with the semantic intent of the textual query? (2) Is there a correlation between the model's predictive confidence and the reliability of next-token candidates? How can this correlation be exploited to improve candidate token filtering?

**Key Insight**: The authors observe that LVLMs already encode "world knowledge" about which visual augmentation would most effectively perturb responses to a specific query—by carefully designing a meta-classification prompt, the model can reason about and select the optimal augmentation itself. Furthermore, Shannon entropy provides a global measure of output distribution uncertainty that is more suitable for dynamic threshold adjustment than a single maximum value.

**Core Idea**: Allow the LVLM to select query-relevant visual augmentations autonomously to maximize the informativeness of contrastive decoding, while using output entropy to dynamically regulate candidate token set size.

## Method

### Overall Architecture
The Self-Aug pipeline proceeds as follows: given an image $v$ and a textual query $x$, a SAS Prompt first instructs the LVLM to reason about and select the optimal visual augmentation $c$ (e.g., cropping, occlusion, noise, color inversion, horizontal/vertical flipping). At each decoding timestep: (1) expert logits $l$ from the original image and amateur logits $l'$ from the augmented image are computed; (2) contrastive decoding is applied as $l_{CD} = (1+\alpha) \cdot l - \alpha \cdot l'$; (3) the SAT algorithm dynamically sets a threshold based on output entropy to truncate the candidate token set; (4) the next token is sampled from the truncated distribution. The entire pipeline requires no architectural modification or training.

### Key Designs

1. **Self-Augmentation Selection (SAS)**:

    - **Function**: Leverages the LVLM's parametric knowledge to dynamically select the visual augmentation that best aligns with the semantic intent of the textual query.
    - **Mechanism**: A structured SAS Prompt $\mathcal{P}$ is constructed with three components—(a) explicit definitions and effect descriptions of each visual augmentation, providing the model with operational knowledge; (b) a structure requiring the model to reason before selecting, reducing post-hoc rationalization risk (inspired by STaR); and (c) few-shot ICL examples to enhance in-context understanding. The model output is parsed by a parsing function $g(\cdot)$ to separate the reasoning trace $r$ and the final selection $c$, which is then passed to a predefined augmentation function $\mathcal{A}(c, v)$ to generate the contrastive image. SAS employs greedy decoding to ensure efficiency and determinism.
    - **Design Motivation**: In contrast to VACoDe's heuristic of using first-token divergence, SAS leverages the model's intrinsic world knowledge and common sense to achieve semantic alignment between queries and augmentations, enabling reasoning about the underlying intent of a query for more targeted selection.

2. **Sparsity Adaptive Truncation (SAT)**:

    - **Function**: Dynamically adjusts the candidate token set size in contrastive decoding based on the entropy of the output distribution, overcoming the confidence-insensitivity of existing APC methods.
    - **Mechanism**: The core insight is that sparsity (confidence) is inversely proportional to the number of candidates that should be retained. When the model is highly uncertain (high entropy), a more permissive threshold should be used to avoid incorrectly discarding correct tokens; when the model is highly certain (low entropy), a stricter threshold should prune the candidate set. SAT employs a decaying entropy function $H_{\text{decay}}(p) = \sigma(-\gamma \sum p_i \log_2 p_i)$, where $\sigma$ is the sigmoid function and $\gamma < 0$ is a scaling parameter. The choice of sigmoid is deliberate: it is naturally bounded in $(0,1)$, its lower plateau provides a stable threshold for low-confidence distributions, and a single parameter $\gamma$ suffices to control the steepness of decay in the intermediate region.
    - **Design Motivation**: APC sets its threshold based on a single point—the maximum logit—rendering it a "confidence-blind" filter. Under low-confidence conditions, the risk of discarding the correct token is substantial. SAT dynamically balances precision and recall by sensing the uncertainty of the full output distribution.

3. **Contrastive Decoding Integration**:

    - **Function**: Combines SAS and SAT into a complete decoding strategy.
    - **Mechanism**: The final contrastive log-probability is $l_{CD}(y_t) = (1+\alpha) \cdot l - \alpha \cdot l'$ if $y_t \in \mathcal{V}_{SAT}$, and $-\infty$ otherwise. The set $\mathcal{V}_{SAT}$ is determined by the dynamic SAT threshold $\beta_t^{SAT}$: $\mathcal{V}_{SAT} = \{y_t \in \mathcal{V} \mid p_\theta(y_t) \geq \beta_t^{SAT} \cdot \max_w p_\theta(w)\}$. Tokens are sampled from $\text{softmax}(l_{CD})$.
    - **Design Motivation**: Query-aware augmentation selection and confidence-aware candidate truncation are seamlessly integrated; the two components are mutually reinforcing—better augmentations yield more meaningful contrastive signals, and smarter truncation better exploits those signals.

### Loss & Training
Self-Aug is entirely training-free. Hyperparameter settings: $\alpha=1$, APC $\beta=0.1$, SAT $\gamma=-0.5$. All experiments are repeated 5 times; results report means and standard deviations.

## Key Experimental Results

### Main Results (Discriminative Benchmarks)

| Model | Method | POPE-COCO Acc↑ | MME-P↑ | MMVP↑ | Avg. Gain |
|-------|--------|---------------|--------|-------|-----------|
| LLaVA-1.5-7B | Multinomial | 82.07 | 1278.42 | 32.40 | - |
| LLaVA-1.5-7B | VCD | 83.66 | 1323.67 | 34.00 | +10.86% |
| LLaVA-1.5-7B | VACoDe | 84.29 | 1372.50 | 36.67 | +9.52% |
| LLaVA-1.5-7B | **Self-Aug** | **82.93** | **1431.30** | **36.00** | **+14.32%** |
| LLaVA-1.5-13B | Multinomial | 83.86 | 1351.69 | 31.60 | - |
| LLaVA-1.5-13B | **Self-Aug** | **85.37** | **1462.18** | **34.80** | **+11.59%** |
| InstructBLIP | Multinomial | 68.70 | 973.66 | 19.20 | - |
| InstructBLIP | **Self-Aug** | **82.86** | **1198.53** | **16.13** | **+18.78%** |
| Qwen3-VL-8B | Multinomial | 88.59 | 1725.16 | 55.47 | - |
| Qwen3-VL-8B | **Self-Aug** | **88.79** | **1726.77** | **60.50** | **+2.25%** |

### Ablation Study

| Configuration | MME-P↑ | Notes |
|---------------|--------|-------|
| Multinomial (baseline) | 1278.42 | No contrastive decoding |
| VCD (random noise) | 1323.67 | Query-agnostic augmentation |
| VACoDe (first-token selection) | 1372.50 | First-token divergence selection |
| Self-Aug (SAS only) | ~1400+ | Self-augmentation selection only |
| Self-Aug (SAS + SAT) | 1431.30 | Full method |
| APC ($\beta=0.1$, fixed) | baseline | Confidence-blind truncation |
| SAT ($\gamma=-0.5$, adaptive) | improved | Entropy-aware dynamic truncation |

### Key Findings
- **Self-Aug is consistently effective across all models**: It outperforms both VCD and VACoDe on all 5 LVLMs (LLaVA-1.5-7B/13B, Qwen-VL, InstructBLIP, Qwen3-VL-8B), with peak average gains of +18.78% on InstructBLIP.
- **Greater benefit for weaker models**: Gains are most pronounced on InstructBLIP (Avg.Δ +18.78%) and comparatively modest on the already strong Qwen3-VL-8B (+2.25%), consistent with the expectation that weaker models benefit more from contrastive decoding.
- **SAS and SAT are complementary**: SAS provides more informative contrastive signals, while SAT better exploits those signals; their combination yields the best performance.
- **Query relevance is critical**: Query-agnostic generic augmentations (VCD) are beneficial but substantially inferior to query-aware SAS selection in terms of informativeness.

## Highlights & Insights
- **Metacognitive design of "let the model choose for itself"**: SAS essentially frames the problem as a meta-classification task for the LVLM—reasoning about which visual perturbation would most disrupt the answer to the current query. This self-aware design paradigm is generalizable to any scenario requiring adaptive self-configuration by the model.
- **Elegant application of entropy as an uncertainty proxy**: SAT combines Shannon entropy with sigmoid decay to realize the intuition of "high uncertainty → permissive filtering, low uncertainty → strict filtering" in a single elegant formula, controlled by a single hyperparameter $\gamma$.
- **Plug-and-play training-free design**: Requiring no architectural modification or additional training, Self-Aug can be directly applied to any LVLM, substantially lowering the deployment barrier.

## Limitations & Future Work
- **SAS introduces additional inference overhead**: An extra forward pass is required to execute the SAS Prompt, and the augmentation selection itself incurs computational cost.
- **Fixed augmentation pool of six types**: Only six predefined visual augmentations are supported (cropping, occlusion, noise, color inversion, horizontal/vertical flipping), potentially missing more effective alternatives.
- **Diminishing returns on stronger models**: The modest gain of +2.25% on Qwen3-VL-8B indicates that as models themselves become more capable, the optimization headroom at the decoding level narrows.
- A lightweight augmentation selector could be learned to replace the SAS Prompt, reducing inference overhead.

## Related Work & Insights
- **vs. VCD**: VCD generates contrastive inputs using query-agnostic random noise; Self-Aug achieves query-aware augmentation selection via SAS, yielding superior informativeness.
- **vs. VACoDe**: VACoDe selects augmentations using first-token divergence, an unreliable empirical proxy in open-ended generation; Self-Aug leverages the model's own reasoning capacity to make globally superior selections.
- **vs. OPERA/DoLa**: These methods focus on attention patterns or inter-layer contrast, which is orthogonal to Self-Aug's visual augmentation contrast; in principle, they are complementary and could be combined.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The self-augmentation selection paradigm of SAS is novel, and the entropy-aware truncation design of SAT is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 5 models and 7 benchmarks with both discriminative and generative assessments; ablation studies are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Technical descriptions are clear and mathematical derivations are rigorous.
- **Value**: ⭐⭐⭐⭐ The training-free plug-and-play design offers practical value, though gains are limited for stronger models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](../../CVPR2026/multimodal_vlm/residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)
- [\[ICLR 2026\] Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation](look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)
- [\[ICLR 2026\] Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking](self-evolving_vision-language_models_for_image_quality_assessment_via_voting_and.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)
- [\[ICLR 2026\] ThinkOmni: Lifting Textual Reasoning to Omni-modal Scenarios via Guidance Decoding](thinkomni_lifting_textual_reasoning_to_omni-modal_scenarios_via_guidance_decodin.md)

</div>

<!-- RELATED:END -->
