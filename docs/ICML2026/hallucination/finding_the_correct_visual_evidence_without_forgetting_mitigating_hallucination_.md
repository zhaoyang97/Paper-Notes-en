---
title: >-
  [Paper Note] Finding the Correct Visual Evidence Without Forgetting: Mitigating Hallucination in LVLMs via Inter-Layer Visual Attention Discrepancy
description: >-
  [ICML 2026][Hallucination Detection][Hallucination mitigation] This paper discovers that LVLM hallucinations stem from "insufficient attention + forgetting during generation" of correct visual evidence. Observing a signi…
tags:
  - "ICML 2026"
  - "Hallucination Detection"
  - "Hallucination mitigation"
  - "visual attention"
  - "inter-layer discrepancy"
  - "saliency map"
  - "training-free"
date: 2026-05-08
content_hash: ac3d52ff0e045954
---

# Finding the Correct Visual Evidence Without Forgetting: Mitigating Hallucination in LVLMs via Inter-Layer Visual Attention Discrepancy

**Conference**: ICML 2026  
**arXiv**: [2605.20965](https://arxiv.org/abs/2605.20965)  
**Code**: https://github.com/ytx-ML/ILVAD (Available)  
**Area**: Hallucination Detection  
**Keywords**: Hallucination mitigation, visual attention, inter-layer discrepancy, saliency map, training-free  

## TL;DR
This paper discovers that LVLM hallucinations stem from "insufficient attention + forgetting during generation" of correct visual evidence. Observing a significant Inter-Layer Visual Attention Discrepancy (ILVAD) for visual evidence, the authors propose a train-free, plug-and-play method: constructing a visual evidence saliency map using inter-layer differentiation, followed by continuous weighting of visual evidence tokens and "evidence-grounded" text tokens during generation. This consistently reduces hallucinations across 5 LVLMs and 5 benchmarks.

## Background & Motivation

**Background**: Existing methods for mitigating LVLM hallucinations primarily fall into four categories: alignment/fine-tuning based on external knowledge bases (high computational cost), post-processing correction, contrastive decoding (adjusting at the logits layer like VCD/CODE/AGLA/ONLY), and attention intervention (VAR/SPARC/VHR based on attention sinks or visually sensitive heads).

**Limitations of Prior Work**: Decoding-level methods only adjust logits and do not force the model to truly "look at the image." Attention redistribution methods recognize that models allocate excessive attention to query-irrelevant visual sinks, but they do not guarantee that attention shifts to the *correct* visual evidence, remaining ineffective when language priors are too strong.

**Key Challenge**: The authors empirically discover two phenomena: (i) At the sample-level, the average attention to visual evidence in hallucinated samples is significantly lower than in correct samples. (ii) At the step-level, attention to visual evidence decays as the number of generated tokens increases, with hallucination frequency rising synchronously. This indicates the problem is not a lack of total attention but "wrong localization + failure to maintain focus."

**Key Insight**: After layer-level analysis, the authors identified an overlooked phenomenon: visual attention in LVLMs exhibits a strong Inter-Layer Visual Attention Discrepancy (ILVAD). While visual sinks receive high attention in almost every layer, visual evidence is only "seen" in specific layers, showing large discrepancies between adjacent layers. This provides a natural signal to distinguish "evidence vs. sink"—the former is sparsely activated across layers, while the latter is persistently activated.

**Core Idea**: Construct a visual evidence saliency map by accumulating inter-layer differences (the positive part of the difference between a layer's activation and the previous one) to filter out sinks and retain "true evidence." This map is then used to continuously boost attention to evidence tokens and amplify "evidence-grounded" text tokens during generation.

## Method

### Overall Architecture
ILVAD operates in two stages by manipulating attention weights without changing the architecture or decoding process. Given an image $I$ and query $q$, the model first generates the initial $T$ tokens to obtain attention weights $\mathbf{A}^{l,h}$ for each layer and head. **Stage 1 (Evidence Localization)**: Average the attention from the first $T$ tokens to visual tokens across visually sensitive heads, apply threshold binarization per layer, and perform inter-layer differentiation to accumulate the evidence saliency map $\hat{\mathbf{S}} \in [0,1]^{|\mathbf{X}_v|}$. **Stage 2 (Evidence Guiding)**: In subsequent generation steps, use $\hat{\mathbf{S}}$ to perform multiplicative weighting to enhance both visual evidence attention and "evidence-sensitive text token" attention, followed by re-normalization to obtain $\hat{\mathbf{A}}$. This pipeline is decoupled from specific LVLMs and is compatible with LLaVA-1.5/NeXT, Qwen2/3-VL, and InternVL3.

### Key Designs

1.  **ILVAD Saliency Map Construction**:
    - **Function**: Extracts only the true evidence tokens from a mixture of "visual evidence + visual sinks + noise" to serve as targets for subsequent enhancement.
    - **Mechanism**: For each layer $l$, the top 50% visually sensitive heads $\mathbf{H}_v^l$ are retained based on total visual attention. The attention from the first $T$ generated tokens to each visual token $j$ is averaged as $\bar{\mathbf{A}}_j^l$. Saliency tokens are selected in each layer using a threshold rule $\tilde{\mathbf{A}}_j^l = \mathbb{1}[\bar{\mathbf{A}}_j^l > \tau \cdot \mathrm{mean}(\bar{\mathbf{A}}^l)]$. The critical step is inter-layer differentiation $\mathbf{S} = \sum_{l=1}^{L-1} \max(\tilde{\mathbf{A}}^{l+1} - \tilde{\mathbf{A}}^l, 0)$, which accumulates tokens that are newly activated in layer $l+1$ but were not in the previous layer, finally normalized to $\hat{\mathbf{S}}$.
    - **Design Motivation**: Visual sinks are "persistently bright across layers," so the difference $\max(\tilde A^{l+1}-\tilde A^l,0)$ is nearly zero for them. True visual evidence "suddenly lights up in specific layers," and differentiation captures these "emergence events"—solving "sink filtering" and "evidence localization" simultaneously without supervision.

2.  **Evidence-guided Visual Enhancement**:
    - **Function**: Continuously boosts attention to visual evidence tokens during generation to counter the "forgetting vision as generation lengthens" phenomenon.
    - **Mechanism**: For each text token $i$ and head $(l,h)$, an "evidence ratio" $\mathbf{e}_i^{l,h} = \sum_{j \in \mathbf{X}_v} \hat{\mathbf{S}}_j \mathbf{A}_{i,j}^{l,h} / \sum_{j \in \mathbf{X}_v} \mathbf{A}_{i,j}^{l,h}$ is calculated to select the top 50% evidence-sensitive heads $\mathbf{H}_e^l$ per layer. Exponential amplification $\hat{\mathbf{A}}_{i,j}^{l,h} = \mathbf{A}_{i,j}^{l,h} \cdot \exp(\alpha \hat{\mathbf{S}}_j)$ is applied only to these heads.
    - **Design Motivation**: Previous methods like VAF amplified all visual tokens equally, which also amplified sinks. Here, $\hat{\mathbf{S}}_j$ acts as a multiplier to regulate amplification based on whether a token is "evidence," operating only on sensitive heads to avoid polluting heads responsible for semantic modeling.

3.  **Evidence-guided Text Enhancement**:
    - **Function**: Selectively emphasizes text tokens that were actually grounded in images when they were generated, suppressing the snowballing effect of purely language-prior-dominated tokens.
    - **Mechanism**: Calculate an "evidence weight" $\mathbf{w}_i = \frac{1}{L \cdot |\mathbf{H}_t^l|} \sum_{h,l,j} \hat{\mathbf{S}}_j \mathbf{A}_{i,j}^{l,h}$ for each generated text token $i$ (its cumulative attention to evidence across all layers and text-sensitive heads), normalized to $\hat{\mathbf{w}} \in [0,1]^{|\mathbf{X}_t|}$. Adjust attention between text tokens using $\hat{\mathbf{A}}_{i,j}^{l,h} = \mathbf{A}_{i,j}^{l,h} \cdot (\hat{\mathbf{w}}_i + \beta)$ on text-sensitive heads, where $\beta$ controls the baseline.
    - **Design Motivation**: Only amplifying vision solves the "looking" part, but models still copy answers from previous text tokens. If previous tokens are hallucinations, the error propagates. Weighting previous tokens by how much evidence they attended to encourages the model to refer to "grounded" tokens.

### Loss & Training
**Completely train-free**. All operations modify attention weights directly during inference, followed by standard softmax normalization. Key hyperparameters: $\tau$ (saliency threshold, default $5$), $\alpha$ (visual enhancement strength, $3$ for LLaVA-NeXT, $5$ for others), $\beta$ (text enhancement baseline, $1$ for discriminative benchmarks, $0.2-0.5$ for generative benchmarks), $T$ (initial steps for saliency extraction, default $10$), and head ratio $\rho=0.5$.

## Key Experimental Results

### Main Results

Evaluated across 5 LVLMs and 3 hallucination benchmarks (CHAIR / POPE / MMHal-Bench) against 10 baselines (Greedy/Beam + VCD/CODE/AGLA/ONLY + VAF/VAR/SPARC/VHR). Example results for LLaVA-1.5-7B:

| Method | CHAIR$_S$↓ | CHAIR$_I$↓ | POPE Acc↑ | POPE F1↑ | MMHal Hal.↓ | MMHal Score↑ |
|---|---|---|---|---|---|---|
| Greedy | 48.6 | 13.52 | 84.50 | 85.22 | 67.0 | 2.01 |
| VCD | 48.4 | 13.47 | 84.74 | 85.31 | 61.8 | 2.20 |
| AGLA | 46.4 | 13.27 | 85.63 | **86.38** | 63.8 | 2.14 |
| VAR | 52.5 | 14.17 | 84.82 | 85.87 | 62.2 | 2.18 |
| VHR | 34.5 | 9.86 | 84.74 | 85.45 | 65.6 | 2.10 |
| **Ours (ILVAD)** | **32.6** | **9.42** | **85.76** | 86.12 | **61.5** | **2.22** |

Compared to greedy decoding, LLaVA-1.5-7B saw a 31.6% decrease in CHAIR$_S$ and a 9.12% increase in MMHal Score. Improvements were consistent across newer models like Qwen2-VL and InternVL3 (e.g., Qwen2-VL-7B CHAIR$_S$ 20.8 → 17.8).

### Ablation Study

| Configuration | CHAIR$_S$↓ | CHAIR$_I$↓ | POPE Acc↑ | POPE F1↑ |
|---|---|---|---|---|
| Baseline (greedy) | 48.6 | 13.52 | 84.50 | 85.22 |
| w/o Visual Enh. | 49.4 | 13.44 | 84.44 | 85.12 |
| w/o Text Enh. | 34.8 | 9.73 | 85.59 | 85.85 |
| Full ILVAD | **32.6** | **9.42** | **85.76** | **86.12** |

Threshold $\tau$ sensitivity: $\tau=1$ (too loose, sinks persist) yielded CHAIR$_S=44.6$. $\tau=5$ achieved the best balance across benchmark types.

### Key Findings
- **Removing visual enhancement results in almost no gain** (CHAIR$_S$ 49.4 $\approx$ baseline 48.6), indicating that "locating + enhancing evidence" is the core; text-end filtering alone lacks grounding.
- **ILVAD adds negligible inference overhead**: It involves element-wise multiplication and normalization on attention weights, with runtime comparable to baseline, significantly better than contrastive decoding which requires dual forward passes.
- **Robustness**: The method is robust to $\alpha$ (fixed at $5$ for most), while $\beta$ is task-dependent (1 for discriminative, 0.2-0.5 for generative).
- **Steadier Gains in Newer Models**: InternVL3-8B CHAIR$_S$ 21.2 → 16.8, suggesting sink/forgetting issues remain unresolved in state-of-the-art models.

## Highlights & Insights
- **"Inter-layer Differentiation = Automatic Sink Filter"** is the most elegant design: translates the observation that sinks are persistent across layers while evidence is transient into a simple mathematical operator, $\max(\tilde A^{l+1}-\tilde A^l, 0)$, without extra training.
- **Decoupling "Seeing the Image" and "Trusting Self"**: Visual enhancement solves where the model should look, while text enhancement solves which context it should trust. Both are driven by the same saliency map $\hat{\mathbf{S}}$, ensuring low cost and conceptual clarity.
- **Quantifying "Step-level Visual Forgetting"**: The empirical finding that attention to evidence monotonically decreases while hallucination rates increase provides an observable metric for why long-form hallucinations occur.

## Limitations & Future Work
- The saliency map is "frozen" after $T=10$ tokens; for long generations, the semantic focus might shift (e.g., from foreground to background), which a fixed map cannot handle.
- Evaluation focuses on object-level hallucinations; gains in MME position/color subsets are limited, suggesting the method is less effective when hallucinations arise from inherently ambiguous visual evidence rather than incorrect localization.
- $\beta$ requires manual tuning based on dataset type; an automatic selection strategy is missing.

## Related Work & Insights
- **vs. VAR / EVAS**: VAR redistributes sink attention proportionally, but "others" doesn't mean "correct." ILVAD refines "others" into "evidence."
- **vs. SPARC**: SPARC uses temporal differentiation between generation steps; ILVAD uses layer differentiation. They are orthogonal and could potentially be combined.
- **vs. VHR**: VHR operates at the head level; ILVAD goes deeper to the token-level saliency, offering better precision for long-form generation.
- **vs. AGLA / VCD**: Contrastive decoding requires multiple forward passes; ILVAD achieves superior or comparable performance with a single pass, offering much higher efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ The inter-layer differentiation for sink filtering is a clean, novel insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage of models and benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation-observation-method progression.
- Value: ⭐⭐⭐⭐ Train-free and low-overhead, highly attractive for deployment-side mitigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning from Fine-Grained Visual Discrepancies: Mitigating Multimodal Hallucinations via In-Context Visual Contrastive Optimization](learning_from_fine-grained_visual_discrepancies_mitigating_multimodal_hallucinat.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs](../../CVPR2026/hallucination/hulluedit_subspace_editing_hallucination.md)
- [\[CVPR 2026\] Mitigating Object Hallucination in LVLMs via Attention Imbalance Rectification](../../CVPR2026/hallucination/mitigating_object_hallucinations_in_lvlms_via_attention_imbalance_rectification.md)
- [\[ICML 2026\] Automatic Layer Selection for Hallucination Detection](automatic_layer_selection_for_hallucination_detection.md)
- [\[AAAI 2026\] Listen Like a Teacher: Mitigating Whisper Hallucinations using Adaptive Layer Attention and Knowledge Distillation](../../AAAI2026/hallucination/listen_like_a_teacher_mitigating_whisper_hallucinations_using_adaptive_layer_att.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](../../ICLR2026/interpretability/exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)
- [\[CVPR 2026\] Draft and Refine with Visual Experts](../../CVPR2026/interpretability/draft_and_refine_with_visual_experts.md)
- [\[ICML 2026\] MUSE: Resolving Manifold Misalignment in Visual Tokenization via Topological Orthogonality](muse_resolving_manifold_misalignment_in_visual_tokenization_via_topological_orth.md)
- [\[CVPR 2026\] Reallocating Attention Across Layers to Reduce Multimodal Hallucination](../../CVPR2026/interpretability/reallocating_attention_across_layers_to_reduce_multimodal_hallucination.md)
- [\[CVPR 2026\] Pixel2Phys: Distilling Governing Laws from Visual Dynamics](../../CVPR2026/interpretability/pixel2phys_distilling_governing_laws_from_visual_dynamics.md)

</div>

<!-- RELATED:END -->
