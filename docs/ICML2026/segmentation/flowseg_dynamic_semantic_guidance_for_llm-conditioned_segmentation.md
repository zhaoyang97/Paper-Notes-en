---
title: >-
  [Paper Note] FlowSeg: Dynamic Semantic Guidance for LLM-Conditioned Segmentation
description: >-
  [ICML 2026][Segmentation][LLM-conditioned segmentation] This paper argues that current query-based LLM-conditioned segmentation follows a "propose-then-select" paradigm—candidate masks are often accurate…
tags:
  - "ICML 2026"
  - "Segmentation"
  - "LLM-conditioned segmentation"
  - "bidirectional semantic flow"
  - "referring expression segmentation"
  - "reasoning segmentation"
  - "boundary refinement"
date: 2026-05-08
content_hash: c63c6a98ace34bbe
---

# FlowSeg: Dynamic Semantic Guidance for LLM-Conditioned Segmentation

**Conference**: ICML 2026  
**arXiv**: [2605.29461](https://arxiv.org/abs/2605.29461)  
**Code**: https://zkzhang98.github.io/FlowSeg_page  
**Area**: Segmentation / LLM-Conditioned Segmentation / Vision-Language Alignment  
**Keywords**: LLM-conditioned segmentation, bidirectional semantic flow, referring expression segmentation, reasoning segmentation, boundary refinement

## TL;DR
This paper argues that current query-based LLM-conditioned segmentation follows a "propose-then-select" paradigm—candidate masks are often accurate, but the model fails to select the correct one. FlowSeg is proposed to integrate LLM condition embeddings into every decoder layer for query refinement, continuously updating them with new visual evidence. Combined with a lightweight boundary refinement module, it achieves consistent performance gains on RefCOCO/+/g and ReasonSeg.

## Background & Motivation
**Background**: LLM-conditioned segmentation couples Large Language Models with pixel-level segmentation decoders (e.g., SAM-style or Mask2Former-style query decoders), forming a rapid evolution path: LISA → PSALM → HyperSeg → Sa2VA → X-SAM. Mainstream frameworks mostly adopt a "propose-then-select" approach: a set of learnable queries decodes candidate masks from visual features across $L$ decoder layers, and the LLM's condition embedding is used at the final stage to select the mask with the highest similarity.

**Limitations of Prior Work**: The authors systematically analyzed failure cases of SOTAs like X-SAM on RefCOCO/+/g and found that "many failures are not due to poor mask quality, but incorrect matching"—in most failed samples, at least one candidate mask already has a high IoU with the GT, but it is not selected due to a low score from the matching module. This "semantic misalignment" is particularly prevalent in referring expressions involving ambiguous attributes or relational descriptions.

**Key Challenge**: In current pipelines, semantic and visual interactions are shallow and unidirectional. LLM-derived condition embeddings are either injected as fixed keys/values in cross-attention or reserved entirely for the final matching stage. Query iteration remains primarily driven by visual features, with language appearing only at the "final scoring" moment. Furthermore, condition embeddings are never updated and fail to incorporate visual evidence decoded by the segmentor.

**Goal**: To restructure the internal interactions of the decoder without altering the core LLM-segmentor backbone. The goal is to involve semantics in mask generation dynamics from layer 0 and allow condition embeddings to be refined by new visual signals during decoding, thereby resolving "semantic misalignment" at the architectural level.

**Key Insight**: Oracle experiments provide a strong signal: if candidates are selected using an oracle, the cIoU upper bounds for both X-SAM and FlowSeg on RefCOCO/+/g are nearly identical (~91%). This indicates that candidate generation is already near saturation; the bottleneck lies in selection. Solving the selection problem "earlier" during the decoding process is more direct than training a stronger post-hoc scorer.

**Core Idea**: Utilize "Bidirectional Semantic Flow" (BSF) to allow conditions and queries to update each other at every decoder layer, followed by a lightweight "Boundary-Aware Refinement" (BAR) that modifies only uncertain boundaries while preserving confident internal regions.

## Method

### Overall Architecture
FlowSeg inherits the standard scaffold of dual visual encoders + LLM + query decoder used in LISA/X-SAM: (1) A Vanilla Encoder (SigLIP2-so400m) extracts semantic features for the LLM; (2) A Segmentation Encoder (SAM-ViT-L) extracts pixel features $\mathbf{F}_{\text{pix}}$ for the segmentation decoder. The LLM (Qwen-3) uses `<p>...</p>` for phrase spans and `<SEG>` for the segmentation output position in its instructions. Two types of vectors are extracted from the LLM hidden states: condition embeddings $\mathbf{C}_{\text{LLM}}$ (from `<p>` spans) and segmentation embeddings $\mathbf{S}_{\text{LLM}}$ (from the `<SEG>` position), projected via $\phi_{\text{llm}}$ to obtain $\mathbf{C}$ and $\mathbf{S}$. $\mathbf{S}$ is added to initial queries $\mathbf{Q}^{(0)}$ to provide global multimodal context. The decoder uses a Mask2Former architecture with $N=200$ queries, but replaces the internal flow of each layer with BSF. In the output stage, mask probabilities are refined by BAR, and the final mask is matched using $\mathbf{Q}_{\text{out}}$ and the final $\mathbf{C}^{(L)}$.

### Key Designs

1.  **Bidirectional Semantic Flow: Semantic Cross-Attention + Adaptive Fusion (SR)**:
    *   **Function**: Explicitly injects LLM condition embeddings into query refinement at each decoder layer while maintaining the dominance of visual cross-attention.
    *   **Mechanism**: First, visual cross-attention is performed: $\mathbf{Q}_{\text{vis}}^{(l)}=\mathrm{MHA}(\mathbf{Q}^{(l-1)},\mathbf{F},\mathbf{F})$. Then, semantic cross-attention follows: $\mathbf{Q}_{\text{sem}}^{(l)}=\mathrm{MHA}(\mathbf{Q}_{\text{vis}}^{(l)},\mathbf{C}^{(l-1)},\mathbf{C}^{(l-1)})$. The two outputs are adaptively fused via a sigmoid gate: $\mathbf{g}^{(l)}=\sigma(\mathbf{W}_g\cdot[\mathbf{Q}_{\text{vis}}^{(l)}\|\mathbf{Q}_{\text{sem}}^{(l)}])$, $\mathbf{Q}_{\text{fused}}^{(l)}=\mathbf{g}^{(l)}\odot\mathbf{Q}_{\text{vis}}^{(l)}+(1-\mathbf{g}^{(l)})\odot\mathbf{Q}_{\text{sem}}^{(l)}$, followed by standard self-attention and FFN.
    *   **Design Motivation**: Direct concatenation or hard replacement would disrupt spatial priors learned by the visual backbone. The gating mechanism allows shallow layers to prioritize vision and deep layers to prioritize semantics, matching the decoding rhythm of "building coarse spatial hypotheses then converging via language."

2.  **Bidirectional Semantic Flow: Condition Refinement (CR)**:
    *   **Function**: Allows condition embeddings $\mathbf{C}$ to be updated during decoding rather than remaining static.
    *   **Mechanism**: A reverse cross-attention is performed before the end of each layer to allow conditions to absorb the current query state: $\mathbf{C}^{(l)}=\mathbf{C}^{(l-1)}+\mathrm{MHA}(\mathbf{C}^{(l-1)},\mathbf{Q}_{\text{s}}^{(l)},\mathbf{Q}_{\text{s}}^{(l)})$, where $\mathbf{Q}_{\text{s}}^{(l)}$ is the fused query after self-attention. The residual form ensures that conditions are incrementally refined rather than overwritten.
    *   **Design Motivation**: Ablation Table 3 shows that adding only SR (unidirectional) yields a +0.5% gain, whereas adding CR leads to a +1.5% jump. This proves that a closed feedback loop is essential. Once candidate masks collect visual evidence of "red" regions, the condition can conversely specialize from an abstract "red" concept to "a specific part of a specific red object," leading to more accurate final matching.

3.  **Boundary-Aware Mask Refinement (BAR)**:
    *   **Function**: Targets uncertain boundaries for refinement after BSF addresses global semantic misalignment.
    *   **Mechanism**: Boundary pixels $\mathbf{B}$ are first identified from raw mask probabilities $\mathbf{M}_{\text{prob}}$ via morphological gradients: $\mathbf{B}=\mathbb{I}[(\mathrm{dilate}(\mathbf{M}_{\text{prob}})-\mathrm{erode}(\mathbf{M}_{\text{prob}}))>\epsilon]$ where $\epsilon=0.1$. A lightweight network $f_{\text{refine}}$ then outputs a residual in $\mathbf{B}$ constrained by $\tanh$: $\Delta\mathbf{M}=\tanh(f_{\text{refine}}([\mathbf{M}_{\text{raw}}\|f_{\text{comp}}(\mathbf{F}_{\text{pix}})]))\cdot\alpha$, where $\alpha$ is a learnable scale. Final output: $\mathbf{M}_{\text{refined}}=\mathbf{M}_{\text{raw}}+\Delta\mathbf{M}\odot\mathbf{B}$.
    *   **Design Motivation**: Adheres to the "enhancement-not-replacement" principle. BSF solves global selection, but residual errors concentrate on contours. Restricting modifications to $\mathbf{B}$ prevents destroying stable internal predictions. BSF+BAR only adds 5.93M parameters (+0.12%) and 4.28ms latency (+1.39%) per sample, negligible compared to the baseline.

### Loss & Training
Three-stage end-to-end training: (1) Segmentor pretraining for 36 epochs; (2) Vision-language alignment for 1 epoch; (3) Multi-task joint training for 2 epochs. AdamW with lr=$4\times 10^{-5}$, wd=0.05, bs=8/GPU on 8×H20. The loss includes LLM next-token loss and segmentation loss $\mathcal{L}_{\text{seg}}=\mathcal{L}_{\text{CE}}+\lambda_{\text{dice}}\mathcal{L}_{\text{dice}}+\lambda_{\text{mask}}\mathcal{L}_{\text{mask}}$ ($\lambda_{\text{dice}}=\lambda_{\text{mask}}=5.0$, $\lambda_{\text{cls}}=2.0$). Deep supervision is applied to all decoder layers.

## Key Experimental Results

### Main Results

Evaluated on RefCOCO/+/g (cIoU) and ReasonSeg (gIoU/cIoU) against SOTAs like LISA, PixelLM, GSVA, PSALM, HyperSeg, Sa2VA-8B, and X-SAM.

| Dataset | LISA-7B | HyperSeg | X-SAM | **FlowSeg** | vs X-SAM |
|:---:|:---:|:---:|:---:|:---:|:---:|
| RefCOCO val | 74.9 | 84.8 | 85.1 | **85.8** | +0.7 |
| RefCOCO+ val | 65.1 | 79.0 | 78.0 | **80.2** | +2.2 |
| RefCOCOg val | 67.9 | 79.4 | 83.8 | **86.5** | +2.7 |
| RefCOCOg test | 70.6 | 78.9 | 83.9 | **86.1** | +2.2 |
| ReasonSeg test cIoU | 34.1 | – | 41.0 | **54.7** | +13.7 |
| ReasonSeg test gIoU | 36.8 | – | 57.8 | **60.5** | +2.7 |

The massive +13.7% cIoU jump on ReasonSeg confirms that complex reasoning tasks rely most heavily on continuous semantic participation during decoding. Backbone-controlled ablations show that upgrading X-SAM's LLM to Qwen3 yields marginal gains, while FlowSeg using the original Phi-3-3.8B still outperforms X-SAM, proving gains stem from architecture rather than a stronger LLM.

### Ablation Study

| Configuration | RefCOCO | RefCOCO+ | RefCOCOg | Avg. |
|:---:|:---:|:---:|:---:|:---:|
| Baseline | 85.0 | 78.3 | 84.1 | 82.4 |
| + SR (semantic refinement) | 85.4 | 79.0 | 84.3 | 82.9 (+0.5) |
| + SR + CR (= Full BSF) | 85.6 | 79.9 | 86.2 | 83.9 (+1.5) |
| + SR + CR + BAR (Full) | **85.8** | **80.2** | **86.5** | **84.2 (+1.8)** |

### Key Findings
- Unidirectional semantic injection (SR only) yields small gains (+0.5%); adding CR triggers a jump to +1.5%, proving that a closed feedback loop is the key bottleneck.
- Oracle bound experiments (Table 5): Both FlowSeg and X-SAM have oracle cIoU around 91%, verifying that the gap comes from selection rather than generation.
- On the failure subset of X-SAM (cIoU < 0.5), FlowSeg improves mean IoU from 4.6 to 49.2 (+44.6) with a rescue rate of 44.6%. It remains effective even on the harder cIoU < 0.2 subset.
- BAR contributes +0.3% avg cIoU, providing "finishing touches" by strictly confining modifications to boundaries to avoid harming stable internal predictions.

## Highlights & Insights
- **Diagnosis-driven Architectural Design**: The authors first pinpointed "selection vs. generation" as the bottleneck via oracle experiments before designing BSF—a methodology highly applicable to other LLM-conditioned tasks.
- **Bidirectional Flow vs. Unidirectional Injection**: While cross-modal attention typically focuses on "text-to-vision," this work proves that performance is truly unlocked only when the condition itself is refreshed by visual context.
- **Enhancement-not-Replacement Refining**: Using morphological gradients to unsupervisedly mask "where to change" and using $\tanh$ residuals is a safe paradigm to avoid damaging learned internal representations.
- **Lightweight & Modular**: BSF only replaces internal decoder modules without changing the LLM or visual encoder, making it easily adaptable to any Mask2Former-like head.

## Limitations & Future Work
- Evaluation is limited to the "one expression per image" protocol; multi-target/multi-mask scenarios were not covered.
- The ReasonSeg val set is small (340 cases) and noisy; the authors specify that the "test set" is the primary benchmark.
- BSF performs two extra attention operations per layer. While total overhead is low, cost as a function of the number of queries $N$ should be re-evaluated for video tasks.
- BAR uses a fixed threshold $\epsilon=0.1$; adaptive thresholds or learnable boundary detectors might further improve complex scenarios.

## Related Work & Insights
- **vs. LISA / HyperSeg / X-SAM**: These are "propose-then-select" models with static embeddings. FlowSeg changes unidirectional flow to bidirectional without backbone modification, serving as a general performance booster.
- **vs. PSALM / Sa2VA**: These primarily expand the task space (video, multi-task), but the decoder remains a passive receiver. FlowSeg's BSF is orthogonal and can be applied on top.
- **vs. Mask2Former / DETR**: Traditional query decoders ignore semantic iteration. FlowSeg completes the paradigm by adding a "language-side iteration" within the decoder.

## Rating
- Novelty: ⭐⭐⭐⭐ While bidirectional flow and boundary refinement are not entirely new, their combination and the diagnosis-to-design cycle for "semantic misalignment" are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Main results, backbone-controlled trials, oracle studies, and failure rescue analysis are comprehensive, though multi-mask tasks were omitted.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation in Section 1; Algorithm 1 provides concise pseudo-code for easy reproduction.
- Value: ⭐⭐⭐⭐ Significant progress on ReasonSeg (+13.7 cIoU) and a modular BSF design offer a clear path for future LLM-conditioned dense prediction tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](../../CVPR2026/segmentation/geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)
- [\[ICML 2026\] SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation](semir_semantic_minor-induced_representation_learning_on_graphs_for_visual_segmen.md)
- [\[ICCV 2025\] Enhancing Transformers Through Conditioned Embedded Tokens](../../ICCV2025/segmentation/enhancing_transformers_through_conditioned_embedded_tokens.md)
- [\[CVPR 2026\] Universal 3D Shape Matching via Coarse-to-Fine Language Guidance](../../CVPR2026/segmentation/universal_3d_shape_matching_via_coarse-to-fine_language_guidance.md)
- [\[ICCV 2025\] Dynamic Dictionary Learning for Remote Sensing Image Segmentation](../../ICCV2025/segmentation/dynamic_dictionary_learning_for_remote_sensing_image_segmentation.md)

</div>

<!-- RELATED:END -->
