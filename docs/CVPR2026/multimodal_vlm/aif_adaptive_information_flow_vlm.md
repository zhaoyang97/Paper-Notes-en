---
title: >-
  [Paper Note] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow
description: >-
  [CVPR 2026][Multimodal VLM][Vision-language models] This paper identifies that excessive attention from text tokens to irrelevant visual tokens is the root cause of the "see but misperceive" phenomenon in VLMs. It propos…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Vision-language models"
  - "information flow modulation"
  - "token dynamics"
  - "causal mask"
  - "training-free"
date: 2026-05-08
content_hash: a2f4d8873d299eb6
---

# Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow

**Conference**: CVPR 2026
**arXiv**: [2604.15809](https://arxiv.org/abs/2604.15809)  
**Code**: [https://cxliu0.github.io/AIF/](https://cxliu0.github.io/AIF/)  
**Area**: Multimodal VLM
**Keywords**: Vision-language models, information flow modulation, token dynamics, causal mask, training-free

## TL;DR

This paper identifies that excessive attention from text tokens to irrelevant visual tokens is the root cause of the "see but misperceive" phenomenon in VLMs. It proposes Adaptive Information Flow (AIF), a training-free method that modulates information flow at inference time by modifying the causal mask based on token dynamic entropy, blocking irrelevant visual-to-text connections and improving perceptual performance across multiple VLMs.

## Background & Motivation

**Background**: Vision-language models (VLMs) such as LLaVA and Qwen2.5-VL have demonstrated strong capabilities across diverse tasks including visual question answering, OCR, and object localization.

**Limitations of Prior Work**: Recent studies reveal a "see but misperceive" failure mode—models correctly attend to image regions relevant to the query yet still produce wrong answers. Existing remedies either require retraining (high computational cost) or rely on visual cropping (substantially increased inference time and ineffective for counting and relational reasoning).

**Key Challenge**: Information flow during VLM decoding is suboptimal: cross-attention from text tokens is dispersed across large numbers of irrelevant background visual tokens, creating spatially diffuse attention patterns that introduce noisy visual information and interfere with correct reasoning.

**Goal**: Improve VLM perceptual performance through inference-time information flow modulation without any training.

**Key Insight**: Visual tokens corresponding to target regions exhibit distinctive activation patterns (high activity) in specific LLM layers, whereas tokens from irrelevant regions display irregular activation. This difference in "token dynamics" is exploited to identify important tokens.

**Core Idea**: Text tokens need only interact with important visual tokens. By modifying the causal mask to block information flow from irrelevant visual tokens to text tokens—while preserving visual-to-visual information flow to maintain image integrity—perceptual accuracy is improved without retraining.

## Method

### Overall Architecture

AIF operates in three steps: (1) a single LLM decoding step collects cross-layer activation statistics (token dynamics) for each visual token; (2) per-token importance entropy is computed from the token dynamics to produce a token entropy map; (3) an adaptive masking threshold is selected, and the causal mask is modified to block connections from irrelevant tokens to text tokens. The entire procedure requires only one additional decoding step; subsequent inference proceeds unchanged.

### Key Designs

1. **Token Dynamics Analysis and Importance Measurement**:

    - Function: Quantify the importance of each visual token to text-side reasoning.
    - Mechanism: Token dynamics are defined as $\mathcal{D}_{v_i} = \{d_{v_i}^l\}_{l=1}^L$, where $d_{v_i}^l = \max_j a_{i,j}^l$ denotes the maximum attention value from the $i$-th visual token to all text tokens at layer $l$. The mean $\mu_{v_i}$ and dynamics-based entropy $\text{Ent}_{v_i} = \sum -\frac{d_{v_i}^l}{L \cdot \mu_{v_i}} \log(\frac{d_{v_i}^l}{L \cdot \mu_{v_i}})$ are then computed. High entropy indicates random activation patterns (unimportant); low entropy indicates concentrated activation at specific layers (important).
    - Design Motivation: Visual tokens in target regions activate strongly at key layers (low entropy), whereas irrelevant tokens activate randomly across layers (high entropy), making entropy a natural discriminator between the two.

2. **Adaptive Information Flow Modulation**:

    - Function: Automatically determine the optimal masking ratio and modify the causal mask accordingly.
    - Mechanism: Visual tokens are ranked by entropy, and multiple masking ratios (0.1 to 0.9) are evaluated. For each candidate ratio, the attention distribution entropy $S$ over retained tokens is computed. The ratio that maximizes the divergence between $S$ and the original distribution entropy $S_0$ is selected—indicating a shift from diffuse to concentrated attention. Connections from masked visual tokens to text tokens are then blocked in the causal mask (set to $-\infty$).
    - Design Motivation: Different images and questions require different masking ratios; adaptive selection eliminates hyperparameter tuning. Maximizing the change in attention concentration ensures the model focuses on genuinely relevant regions.

3. **Preserving Visual-to-Visual Information Flow**:

    - Function: Ensure that the masking process does not discard any image information.
    - Mechanism: Only connections from masked visual tokens to text tokens are blocked; attention among masked tokens and between masked and unmasked visual tokens is preserved. This is fundamentally distinct from visual token pruning, which removes tokens entirely.
    - Design Motivation: Complete removal of visual tokens risks losing contextual information. Preserving visual-to-visual information flow allows remaining tokens to still benefit from the masked tokens' information.

### Loss & Training

AIF is entirely training-free. Only one additional decoding step is required to collect token dynamics and generate the mask; all subsequent inference steps are identical to the standard pipeline.

## Key Experimental Results

### Main Results

| Method | V* | RealWorldQA | MMStar | TextVQA | CountBench |
|------|-----|------------|--------|---------|------------|
| LLaVA-1.5-7B | 42.4 | 55.6 | 33.1 | 47.8 | 47.0 |
| **+ AIF** | **50.3 (+7.9)** | **60.5 (+4.9)** | **39.5 (+6.4)** | **49.9 (+2.1)** | **50.1 (+3.1)** |
| Qwen2.5-VL-7B | 78.5 | 68.5 | 63.9 | 84.9 | 87.1 |
| **+ AIF** | **84.8 (+6.3)** | **74.5 (+6.0)** | **70.9 (+7.0)** | **86.0 (+1.1)** | **89.5 (+2.4)** |

### Visual Grounding Results

| Method | RefCOCO Avg. | RefCOCO+ Avg. | RefCOCOg Avg. |
|------|-------------|-------------|--------------|
| Qwen2.5-VL-7B | 89.3 | 80.1 | 87.2 |
| **+ AIF** | **91.4 (+2.1)** | **82.7 (+2.6)** | **89.5 (+2.3)** |

### Key Findings

- Masking 90% of low-$\mu_{v_i}$ tokens causes negligible performance degradation, whereas masking only 10% of high-$\mu_{v_i}$ tokens leads to more than 50% performance drop, confirming that only a small subset of visual tokens substantially influences model output.
- AIF improves Qwen2.5-VL-7B on MMStar by 7.0 points, surpassing GPT-4o and Qwen2.5-VL-72B on select metrics.
- On visual grounding tasks, AIF even outperforms the specialist grounding model Grounding-DINO-L.
- Oracle experiments indicate that the potential upper bound of gains from information flow modulation is higher still (RealWorldQA: 55.6→61.6).

## Highlights & Insights

- **Information Flow as a Control Signal**: Prior work primarily uses attention analysis for diagnosis and interpretation; this paper is the first to leverage it as an active control signal for improving model performance—a perspective shift with broad implications.
- **Training-Free and Plug-and-Play**: Consistent performance gains across multiple VLMs and tasks are achieved solely through causal mask modification, demonstrating the universal value of information flow modulation.
- **Design Philosophy of Preserving Visual-to-Visual Flow**: The critical distinction from brute-force token pruning ensures informational completeness.

## Limitations & Future Work

- One additional decoding step is required to collect token dynamics; while the overhead is modest, it is nonzero.
- The adaptive threshold selection evaluates multiple candidate ratios, which may not constitute the optimal selection strategy.
- For tasks requiring holistic scene understanding (e.g., image captioning), aggressive masking could be detrimental.
- Validation is limited to LLaVA-1.5 and Qwen2.5-VL; generalization to a broader range of models remains to be confirmed.

## Related Work & Insights

- **vs. ViCrop**: ViCrop enhances fine-grained perception by cropping and magnifying relevant regions, but substantially increases inference time and is ineffective for relational reasoning. AIF achieves its effect through mask modification with minimal overhead and is effective across diverse tasks.
- **vs. Visual Token Pruning**: Pruning removes tokens entirely, causing information loss; AIF only blocks connections to text tokens, preserving the complete visual information.
- **vs. Pei et al. / Wang et al.**: These works modify attention patterns at the architectural or training level, requiring retraining; AIF operates solely at inference time.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Information flow modulation as a novel training-free paradigm for improving VLMs; highly original perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage across VQA, OCR, grounding, counting, and hallucination tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ — Complete logical chain from discovery to analysis to method; in-depth experimental analysis.
- Value: ⭐⭐⭐⭐⭐ — Training-free approach that substantially improves VLM performance; highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models](../../NeurIPS2025/multimodal_vlm/flowcut_rethinking_redundancy_via_information_flow_for_effic.md)
- [\[CVPR 2026\] LLMind: Bio-inspired Training-free Adaptive Visual Representations for Vision-Language Models](llmind_bio-inspired_training-free_adaptive_visual_representations_for_vision-lan.md)
- [\[CVPR 2026\] MODIX: Training-Free Multimodal Information-Driven Positional Index Scaling for VLMs](modix_positional_index_scaling.md)
- [\[CVPR 2026\] ReHARK: Refined Hybrid Adaptive RBF Kernels for Robust One-Shot Vision-Language Adaptation](rehark_refined_hybrid_adaptive_rbf_kernels_for_robust_one-shot_vision-language_a.md)
- [\[ACL 2026\] Revisit What You See: Revealing Visual Semantics in Vision Tokens to Guide LVLM Decoding](../../ACL2026/multimodal_vlm/revisit_what_you_see_revealing_visual_semantics_in_vision_tokens_to_guide_lvlm_d.md)

</div>

<!-- RELATED:END -->
