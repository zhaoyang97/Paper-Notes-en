---
title: >-
  [Paper Note] FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][visual token pruning] Reexamining visual token redundancy in VLMs through the lens of Information Flow: the CLS token acts as an information relay, redundancy emerges progressively, and single-layer single-criterion scoring is unreliable. FlowCut—an information-flow-aware multi-criteria cumulative importance pruning framework—surpasses SOTA by 1.6% on LLaVA-1.5-7B at an 88.9% token reduction rate, and by 4.3% on LLaVA-NeXT-7B.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - visual token pruning
  - information flow
  - VLM efficiency
  - attention analysis
  - training-free
date: 2026-05-08
content_hash: 2c75120c99bef26a
---

# FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.19536](https://arxiv.org/abs/2505.19536)
**Code**: [https://github.com/TungChintao/FlowCut](https://github.com/TungChintao/FlowCut)
**Area**: Model Compression / Multimodal VLM
**Keywords**: visual token pruning, information flow, multi-criteria scoring, attention entropy, training-free

## TL;DR

FlowCut reexamines the emergence of visual token redundancy in VLMs through the lens of Information Flow, and proposes a pruning framework featuring layer-adaptive pruning ratios, multi-criteria fusion scoring, and cumulative importance tracking. The approach aligns pruning decisions with the model's intrinsic information propagation behavior. On LLaVA-1.5-7B, FlowCut surpasses the previous SOTA by 1.6% at an 88.9% token reduction rate; on LLaVA-NeXT-7B, it surpasses the previous SOTA by 4.3% at a 94.4% reduction rate.

## Background & Motivation

Large Vision-Language Models (LVLMs) achieve strong multimodal understanding, yet the large number of visual tokens incurs substantial computational cost. Existing token pruning methods (FastV, SparseVLM, VisionZip, etc.) commonly rely on single-layer attention scores to rank and remove redundant visual tokens. However, token–token and layer–layer interactions are highly complex, which raises a fundamental question: **Is a single-layer, single-criterion attention score sufficient to accurately identify redundant tokens?**

The authors conduct a systematic analysis of visual token information propagation patterns across ViT layers from the perspective of Information Flow—defined as each token's primary information sources (flow-in) and primary information destinations (flow-out), modeled via attention maps. This analysis yields three key insights that expose the essential mechanism behind redundancy emergence and motivate pruning strategies aligned with the model's intrinsic behavior.

## Method

### Overall Architecture

FlowCut is an information-flow-aware pruning framework comprising three core components: (1) layer-adaptive pruning ratios guided by attention entropy; (2) multi-criteria fusion scoring; and (3) cumulative flow importance tracking. Pruning is applied at intermediate layers of the visual encoder and is entirely training-free, functioning as a plug-and-play module at inference time.

### Key Designs

1. **Information Flow Analysis and Three Core Findings (Analytical Foundation)**:
    - Function: Expose the essential mechanism by which visual token redundancy emerges
    - Finding ①—CLS token as an information relay: In shallow layers, patch tokens primarily attend to neighboring tokens and the CLS token; in deep layers, they uniformly attend to distant "hub tokens." The CLS token first aggregates global information from all patch tokens and then redistributes it back, serving as an effective proxy for global information flow.
    - Finding ②—Progressive emergence of redundancy: The attention distribution of the CLS token becomes increasingly concentrated as depth increases (attention entropy decreases), with a sharp entropy drop at layers 11–15. Redundancy is not a static property but emerges progressively through layer-wise attention concentration during encoding.
    - Finding ③—Unreliability of single-criterion scoring: Tokens receiving high CLS attention may nonetheless exhibit low information density (small Value L1 norm) or low semantic relevance (low cosine similarity with the CLS token), causing different criteria to produce contradictory importance rankings. This originates from layer-wise amplification of noise in the information flow.

2. **Layer-Adaptive Pruning Ratio (Attention Distribution-Aware Prune Ratio)**:
    - Function: Dynamically adjust pruning intensity according to per-layer attention concentration
    - Mechanism: The attention entropy of the CLS token is used as the indicator—high entropy (dispersed attention → low redundancy) triggers conservative pruning; low entropy (concentrated attention → high redundancy) triggers aggressive pruning. The number of pruned tokens is given by $P = \frac{N-T}{\sqrt{L}} \cdot (1 - r_H^2)$, where $r_H = H(\mathbf{A}^g) / H_{max}$ is the normalized entropy.
    - Design Motivation: Aligns with the progressive redundancy emergence identified in Finding ②, replacing fixed per-layer ratios.

3. **Multi-Criteria Evaluator**:
    - Function: Assess token importance along multiple complementary dimensions
    - Mechanism: Three complementary indicators are fused—attention strength $\mathbf{I}^a$ (attention score from CLS to the token), semantic relevance $\mathbf{I}^s$ (cosine similarity between the token and the CLS value vector), and information density $\mathbf{I}^d$ (L1 norm of the Value vector). The final score is $\mathbf{S} = (\overline{\mathbf{I}^a} + \overline{\mathbf{I}^s}) \times \mathbf{I}^d$.
    - Design Motivation: Addresses the contradictory single-criterion rankings identified in Finding ③ through multi-dimensional cross-validation to reduce mis-pruning.

4. **Cumulative Flow Importance Tracking**:
    - Function: Aggregate importance information across layers and over time
    - Mechanism: After computing the current multi-criteria score at each layer, it is combined with the historical cumulative score via a weighted average: $\mathbf{S}_{cum}^{(l)} = 0.5 \times \mathbf{I}_{cur}^{(l)} + 0.5 \times \mathbf{S}_{cum}^{(l-1)}$; pruning is applied every two layers.
    - Design Motivation: A single-layer perspective is insufficient to capture the full contribution of each token; the cumulative mechanism implicitly filters noise in the information flow.

### Loss & Training

No training is required whatsoever (training-free); the method is plug-and-play at inference time. For LLaVA-1.5, pruning is performed at the penultimate layer of the visual encoder. For LLaVA-NeXT and Qwen2-VL (which have more tokens), a two-stage pruning strategy is adopted: tokens are first reduced to twice the target count within the visual encoder, then further reduced to the final target count at the 2nd layer of the LLM.

## Key Experimental Results

### Main Results

**LLaVA-1.5-7B (retaining 64 tokens, ↓88.9%)**:

| Method | GQA | MMB | MME | POPE | SQA | VQA-V2 | Avg. Relative % |
|--------|-----|-----|-----|------|-----|--------|-----------------|
| Vanilla (576 tokens) | 61.9 | 64.7 | 1862 | 85.9 | 69.5 | 78.5 | 100% |
| FastV (ECCV24) | 46.1 | 48.0 | 1256 | 48.0 | 51.1 | 55.0 | 77.5% |
| SparseVLM (ICML25) | 52.7 | 56.2 | 1505 | 75.1 | 62.2 | 68.2 | 84.6% |
| VisionZip (CVPR25) | 55.1 | 60.1 | 1687 | 77.0 | 69.0 | 72.4 | 94.4% |
| **FlowCut** | **55.6** | **60.8** | **1744** | **80.2** | **69.1** | **72.8** | **96.0%** |

**LLaVA-NeXT-7B (retaining 160 tokens, ↓94.4%)**: FlowCut 91.9% vs. VisionZip 87.6% (**+4.3%**)

**Qwen2-VL-7B (↓88.9%)**: FlowCut 91.3% vs. FastV 83.6% (**+7.7%**)

**Video-LLaVA (retaining 256 tokens, ↓87.5%)**: FlowCut 98.6% vs. VisionZip 94.4% (**+4.2%**)

When retaining 192 tokens (↓66.7%), FlowCut achieves **100.2%**—post-pruning performance marginally exceeds the original model.

### Ablation Study

| Configuration | Avg. Relative Performance | Note |
|---------------|--------------------------|------|
| Attention score only (single criterion) | ~93% | baseline |
| + Multi-criteria fusion | ~95% | +2% gain |
| + Cumulative importance | ~95.5% | additional +0.5% |
| + Adaptive pruning ratio | **96.0%** | full FlowCut |
| Fixed ratio replacing adaptive | ~95% | ~1% loss |

### Key Findings

- All three criteria contribute: attention strength is the foundation; information density and semantic relevance each contribute approximately 0.5–1%.
- Performance exceeds the original model when retaining 192 tokens, suggesting that redundant tokens may in fact be detrimental.
- A 3.2× speedup is achieved in the prefilling stage.

## Highlights & Insights

- **Theory-driven design**: The three insights derived from the information flow perspective form a coherent progression—CLS as relay → progressive redundancy emergence → single-criterion contradictions—leading naturally to each design decision.
- **CLS token as information relay**: This finding has independent academic value, offering a new perspective for understanding ViT's internal mechanisms.
- **Simple yet effective multi-criteria fusion**: The combination of attention strength, information density, and semantic relevance is intuitively motivated and empirically effective.
- **Strong performance under extreme compression**: Outstanding results are maintained at reduction rates of 88.9%–94.4%, demonstrating high practical utility.

## Limitations & Future Work

- Validated only on ViT-based visual encoders; CLS-free encoders (e.g., SigLIP) require substituting a global average token, which has not been fully verified.
- Multi-criteria combination weights (additive + multiplicative) and the cumulative coefficient (0.5/0.5) are manually set; learnable adaptive weights are a natural extension.
- Not yet combined with KV cache compression or layer-level pruning methods.
- Evaluation focuses primarily on understanding tasks; the impact on generation quality (e.g., detailed image captioning) is not sufficiently assessed.

## Related Work & Insights

- **vs. FastV**: FastV performs a one-shot pruning using layer-2 attention scores—the simplest baseline. FlowCut's cross-layer accumulation and multi-criteria scoring yield substantial improvements.
- **vs. VisionZip**: VisionZip also considers information density but still relies on a single criterion with a fixed retention ratio; FlowCut comprehensively outperforms it.
- **vs. PyramidDrop**: PyramidDrop employs a predefined hierarchical schedule; FlowCut's adaptive mechanism is more flexible.
- The information flow analysis framework can be extended to text token analysis in LLMs—similar attention concentration and redundancy emergence phenomena may exist in text tokens as well.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The information flow perspective and three derived insights constitute important theoretical contributions to the field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four VLM architectures (LLaVA-1.5/NeXT, Qwen2-VL, Video-LLaVA), 12 benchmarks, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Information flow visualizations are exceptionally clear and elegant; the analysis-to-design logical chain is fluent and well-structured.
- Value: ⭐⭐⭐⭐⭐ State-of-the-art method in VLM token pruning, with both strong theoretical grounding and practical applicability.

---
title: >-
  [Paper Notes] FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal][Visual Token Pruning] Reexamining the nature of visual token redundancy in VLMs through an information flow lens: the CLS token acts as an information relay, redundancy emerges progressively, and single-layer single-criterion scoring is unreliable. FlowCut, an information-flow-aware multi-criteria cumulative importance pruning framework, surpasses SOTA by 1.6% on LLaVA-1.5-7B at an 88.9% token reduction rate and by 4.3% on LLaVA-NeXT-7B.
tags:
  - NeurIPS 2025
  - Multimodal
  - Visual Token Pruning
  - Information Flow
  - VLM Efficiency
  - Attention Analysis
  - training-free
---

# FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.19536](https://arxiv.org/abs/2505.19536)
**Code**: [https://github.com/TungChintao/FlowCut](https://github.com/TungChintao/FlowCut)
**Area**: Model Compression / Multimodal VLM
**Keywords**: visual token pruning, information flow, VLM efficiency, attention analysis, training-free

## TL;DR
Reexamining visual token redundancy in VLMs through the lens of Information Flow: the CLS token acts as an information relay, redundancy emerges progressively, and single-layer single-criterion scoring is unreliable. FlowCut—an information-flow-aware multi-criteria cumulative importance pruning framework—surpasses SOTA by 1.6% on LLaVA-1.5-7B at an 88.9% token reduction rate, and by 4.3% on LLaVA-NeXT-7B.

## Background & Motivation
VLMs incur high computational cost due to large numbers of visual tokens. Existing methods rank and prune tokens using single-layer attention scores (e.g., FastV, SparseVLM). The authors raise a fundamental challenge: given the complexity of token–token and layer–layer interactions, is a single-layer attention score truly sufficient to identify redundant tokens?

## Core Problem
What is the fundamental nature of visual token redundancy from an information flow perspective? How can pruning strategies be designed to align with the model's intrinsic information propagation behavior?

## Method

### Overall Architecture
Three key modules: (1) layer-adaptive pruning ratios based on attention entropy; (2) multi-criteria scoring (attention strength + information density + semantic relevance); and (3) cumulative flow importance tracking.

### Key Designs (Information Flow Analysis)

1. **Insight 1 — CLS token as information relay**: Analysis of per-layer information flow-in and flow-out in ViT reveals that patch tokens primarily attend to neighboring tokens and the CLS token in shallow layers, and to distant "hub tokens" in deep layers. The CLS token acts as a global information broadcaster—first aggregating information from all patch tokens, then redistributing it. **The CLS token's attention therefore serves as a proxy for global information flow.**

2. **Insight 2 — Progressive emergence of redundancy**: The attention distribution of the CLS token becomes increasingly concentrated with depth (entropy decreases), with a sharp drop at layers 11–15. Redundancy is not a static property but emerges layer by layer during encoding.

3. **Insight 3 — Unreliability of single-criterion scoring**: Tokens that receive high CLS attention may nonetheless have low information density (small Value L1 norm) or low semantic relevance (low cosine similarity with CLS)—causing different criteria to produce contradictory importance rankings.

### Key Designs

1. **Layer-Adaptive Pruning Ratio**: Attention entropy guides pruning intensity at each layer—layers with low entropy (concentrated attention → more redundancy) are pruned more aggressively; layers with high entropy are pruned conservatively. Fixed per-layer ratios are replaced entirely.

2. **Multi-Criteria Fusion Scoring**: Token importance is assessed along three dimensions:

    - **Attention strength**: Attention score from the CLS token to the target token
    - **Information density**: L1 norm of the Value vector (signal strength)
    - **Semantic relevance**: Cosine similarity with the CLS token (global semantic relevance)

3. **Cumulative Flow Importance Tracking**: Rather than relying solely on the current layer, importance is accumulated across layers: `S_cum^(l) = 0.5 × I_cur^(l) + 0.5 × S_cum^(l-1)`. Pruning is applied every two layers to ensure both historical and current information are considered.

### Loss & Training
- Entirely training-free; plug-and-play at inference time.
- Pruning is applied at intermediate layers of the ViT visual encoder.

## Key Experimental Results

**LLaVA-1.5-7B (retaining 64 tokens from 576, ↓88.9%)**:

| Method | Avg. Accuracy (Relative %) |
|--------|---------------------------|
| Vanilla (576 tokens) | 100% |
| FastV | 77.5% |
| SparseVLM | 84.6% |
| PDrop | 86.1% |
| VisionZip | 94.4% |
| **FlowCut** | **96.0% (+1.6%)** |

**LLaVA-NeXT-7B (retaining 32 tokens from 2880, ↓94.4%)**: FlowCut 91.9% vs. VisionZip 87.6% (+4.3%)

- Retaining 192 tokens (↓66.7%): FlowCut achieves 100.2%—**performance even exceeds the original model**.
- 3.2× speedup in the prefilling stage.
- Generalization demonstrated on InternVL2 and Qwen2-VL.

### Ablation Study Highlights
- **Multi-criteria vs. single-criterion**: Multi-criteria scoring substantially outperforms attention-only scoring.
- **Cumulative vs. single-layer**: Cumulative importance tracking improves performance by 2–3%.
- **Adaptive vs. fixed ratio**: Adaptive pruning outperforms a fixed ratio by 1–2%.
- **Contribution of each criterion**: Attention strength is the foundation; information density and semantic relevance each contribute approximately 0.5–1%.

## Highlights & Insights
- **Theory-driven**: Starting from the fundamental perspective of information flow, the three insights form a coherent progression; the analysis leads naturally to each design decision.
- **96% retained at 64 tokens**: Maintaining 96% relative performance with an 88.9% token reduction rate substantially outperforms all competing methods at the same compression level.
- **Possibility of exceeding the original model**: 100.2% at 192 tokens—post-pruning performance marginally surpasses the original, suggesting redundant tokens may be actively harmful.
- **CLS as information relay**: This finding has independent academic value for understanding ViT's internal mechanisms.

## Limitations & Future Work
- Information flow analysis is validated only on ViT-based visual encoders; other architectures (e.g., SigLIP without CLS) have not been verified and require global average token substitution.
- Multi-criteria weights (0.5/0.5) are manually set; learnable adaptive weights are a natural extension.
- Not yet combined with dimension-level compression methods such as KV-Latent.
- A 3.2× prefilling speedup is reported, but decoding-stage speedup (and KV cache size reduction) is not reported.
- Evaluation focuses primarily on understanding tasks; the degree of impact on generation tasks (e.g., image captioning) is not sufficiently tested.

## Related Work & Insights
- **vs. FastV**: FastV performs one-shot pruning using layer-2 attention scores; FlowCut's cross-layer accumulation and multi-criteria scoring yield large gains.
- **vs. VisionZip**: VisionZip is a strong baseline but uses only attention with a small fixed retention budget; FlowCut's adaptive multi-criteria approach is comprehensively superior.
- **vs. mPLUG-DocOwl2 (cross-attention compression)**: DocOwl2 requires additional compression modules and training; FlowCut requires no training whatsoever.

## Related Work & Insights
- The information flow analysis framework can be generalized to token-level analysis in LLMs—similar attention concentration and redundancy emergence phenomena may exist in text tokens.
- The layer-adaptive pruning ratio is complementary to TrimLLM (layer-level compression)—FlowCut prunes tokens within layers, while TrimLLM removes entire layers.
- The "CLS as information relay" finding may be applicable to KV-Latent—the CLS token's KV representations could be maintained at higher dimensionality after dimensional down-sampling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The information flow perspective and three derived insights are important theoretical contributions to the field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four VLM architectures, 12 benchmarks, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Information flow visualizations in Figures 2–4 are exceptionally elegant and intuitive.
- Value: ⭐⭐⭐⭐⭐ State-of-the-art in VLM token pruning with both strong theoretical grounding and practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](../../CVPR2026/multimodal_vlm/aif_adaptive_information_flow_vlm.md)
- [\[NeurIPS 2025\] SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)
- [\[NeurIPS 2025\] VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching](vla-cache_efficient_vision-language-action_manipulation_via_adaptive_token_cachi.md)
- [\[NeurIPS 2025\] NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints](navil_rethinking_scaling_properties_of_native_multimodal_large_language_models_u.md)
- [\[NeurIPS 2025\] Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](balanced_token_pruning_accelerating_vision_language_models_b.md)

</div>

<!-- RELATED:END -->
