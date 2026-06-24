---
title: >-
  [Paper Note] Leveraging Temporal Contextualization for Video Action Recognition
description: >-
  [ECCV 2024][Video Understanding][Video Action Recognition] This paper proposes the **TC-CLIP** framework. By introducing a **temporal contextualization (TC)** mechanism, global video action cues are compressed into a small number of context tokens and injected into the CLIP encoding process. Additionally, a **video-conditional prompting (VP)** module is designed to inject visual information into the text branch. Under four settings—zero-shot, few-shot, base-to-novel…
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Video Action Recognition"
  - "CLIP"
  - "Temporal Modeling"
  - "Token Aggregation"
  - "Video-conditional Prompting"
date: 2026-05-08
content_hash: aecfa08b06c4d505
---

# Leveraging Temporal Contextualization for Video Action Recognition

**Conference**: ECCV 2024  
**arXiv**: [2404.09490](https://arxiv.org/abs/2404.09490)  
**Code**: [GitHub](https://github.com/naver-ai/tc-clip)  
**Area**: Video Action Recognition  
**Keywords**: Video Action Recognition, CLIP, Temporal Modeling, Token Aggregation, Video-conditional Prompting

## TL;DR

This paper proposes the **TC-CLIP** framework. By introducing a **temporal contextualization (TC)** mechanism, global video action cues are compressed into a small number of context tokens and injected into the CLIP encoding process. Additionally, a **video-conditional prompting (VP)** module is designed to inject visual information into the text branch. Under four settings—zero-shot, few-shot, base-to-novel, and fully supervised—TC-CLIP consistently outperforms existing CLIP-based video recognition methods.

## Background & Motivation

### Background

Pre-trained vision-language models (VLMs) like CLIP have demonstrated strong generalization capabilities in video understanding. However, their image-level pre-training nature inherently leaves them lacking temporal modeling capabilities. Existing methods that extend CLIP to videos have fundamental limitations in modeling temporal information.

### Limitations of Prior Work

The authors systematically identify the shortcomings of existing temporal modeling strategies:

**Cross-frame attention** (X-CLIP, Vita-CLIP): Interacts temporal information only through the [CLS] tokens of each frame, **lacking patch-level details**.

**Temporal window expansion** (Open-VCLIP): Extends key-value pairs using patch tokens only from adjacent frames, resulting in a **narrow temporal receptive field**.

**Joint space-time attention** (theoretically optimal): References all patches from all frames. However, since CLIP is pre-trained on short image-text pairs, directly expanding the sequence length causes **extrapolation issues**, leading to severe degradation in attention quality.

**Frame-level pooling** (ViFi-CLIP): Simply averages frame representations, offering no interaction of temporal information across frames.

### Key Findings & Motivation

Through key exploratory experiments (Table 1), the authors find:
- Using [CLS] tokens of all frames as reference: almost no improvement (+0.1/−0.1/+0.3)
- Using neighboring frame patch tokens: limited improvement (+0.7/−0.1/+0.8)
- Using **all patch tokens from all frames**: performance **declines instead** (−3.8/−1.4/+0.0), confirming the extrapolation issue.
- However, using **context tokens** obtained via aggregation: **consistent improvement** (+0.9/+0.5/+2.3)

**Key Insight**: The optimal representation for reference tokens is not the raw tokens, but a compact semantic summary obtained through selection and aggregation. This maintains CLIP's effective sequence length while carrying global temporal context.

## Method

### Overall Architecture

TC-CLIP incorporates two modules on top of CLIP:
1. **Temporal Contextualization (TC)**: Injects global temporal context layer-by-layer within the vision encoder.
2. **Video-conditional Prompting (VP)**: Enriches the text encoder by leveraging video-level temporal context from the vision branch.

### Key Designs

#### 1. Temporal Contextualization (TC)

**Function**: Compresses key temporal information from the entire video into a small set of context tokens, which are injected as additional key-value pairs during self-attention in each encoder layer.

**Mechanism**: A three-step process—

**Step 1: Informative Token Selection**

Since videos contain significant token redundancy (e.g., background), the most informative tokens of each frame are first filtered using attention scores. Specifically, the attention scores of [CLS] tokens to each patch in self-attention are used:

$$\mathbf{a}(\mathbf{z}_t) = \text{Softmax}\left(\frac{\mathbf{q}_{\text{cls}} \mathbf{K}_{\mathbf{z}_t}^{\mathsf{T}}}{\sqrt{d}}\right)$$

By averaging the attention scores across multiple heads as $\bar{\mathbf{a}}_{t,i} = \sum_{h=1}^{H} \mathbf{a}_{t,i}^h / H$, the top-$n_s$ tokens are selected as seed tokens, where $\alpha = n_s / N$ controls the selection ratio (defaulting to $\alpha = 0.3$).

**Step 2: Temporal Context Summarization**

Seed tokens collected from all frames are compressed into $k$ context tokens using an aggregation function $\phi$:

$$\hat{\mathbf{s}} = \phi\left(\{\hat{\mathbf{z}}_{t,i}\}_{(t,i) \in \mathcal{S}}\right)$$

By default, **bipartite soft matching** is used for token merging, clustering similar tokens together and averaging them.

**Step 3: Temporal Context Injection**

The self-attention key-value pairs are extended with context tokens:

$$\text{Attention}_{\text{TC}}(\mathbf{z}_t, \mathbf{s}) = \text{Softmax}\left(\frac{\mathbf{Q}_{\mathbf{z}_t} [\mathbf{K}_{\mathbf{z}_t} | \mathbf{K}_{\mathbf{s}}]^{\mathsf{T}}}{\sqrt{d}} + \mathbf{B}\right) [\mathbf{V}_{\mathbf{z}_t} | \mathbf{V}_{\mathbf{s}}]$$

Here, the bias matrix $\mathbf{B}$ distinguishes intra-frame local context from video-level global context using learnable parameters $b_{\text{local}}$ and $b_{\text{global}}$, optimized independently per layer and attention head.

**Design Motivation**:
- Avoids CLIP's sequence length extrapolation issue.
- Preserves patch-level details across the global temporal range.
- Context tokens act as a "temporal bridge" to transfer video-level context.

#### 2. Video-conditional Prompting (VP)

**Function**: Injects context token information from the vision encoder into text-side prompt vectors to generate instance-level textual prompts.

**Mechanism**: Integrates video information into learnable prompt vectors via cross-attention:

$$\mathbf{s}_{\text{proj}}^l = \text{SG}(\mathbf{s}^l \mathbf{W}_{\text{vis}})$$
$$\hat{\mathbf{p}}^{l-1} = \text{MHCA}(\text{LN}_p(\mathbf{p}^{l-1}), \text{LN}_s(\mathbf{s}_{\text{proj}}^l)) + \mathbf{p}^{l-1}$$
$$\tilde{\mathbf{p}}^{l-1} = \text{FFN}(\text{LN}(\hat{\mathbf{p}}^{l-1})) + \hat{\mathbf{p}}^{l-1}$$

Where $\text{SG}(\cdot)$ denotes a stop-gradient operator. The prompt vectors serve as queries, while the context tokens serve as keys and values. VP is executed before the final layer of the text encoder.

**Design Motivation**: Textual descriptions in action recognition datasets are often limited to category names (e.g., "skateboarding") and lack descriptive detail. VP infuses instance-specific visual information from the video to alleviate the lack of text-side semantics, enabling a tailored customized prompt for each sample.

#### 3. Layer-by-Layer Construction

TC operates in a layer-by-layer fashion: the first layer utilizes standard MHSA (due to the absence of context tokens), while each subsequent layer first executes token selection + aggregation to generate new context tokens, which then extend the self-attention key-values. Context tokens are also updated through an independent FFN.

### Loss & Training

- Training objective: Standard cross-entropy contrastive loss
$$\mathcal{L} = -\sum_i \log \frac{\exp(\text{sim}(\mathbf{v}_i, \mathbf{c}_i)/\tau)}{\sum_j \exp(\text{sim}(\mathbf{v}_i, \mathbf{c}_j)/\tau)}$$
- End-to-end full parameter fine-tuning
- Backbone: CLIP ViT-B/16
- Hardware: 4 × NVIDIA Tesla V100

## Key Experimental Results

### Main Results

**Zero-shot Action Recognition** (trained on K-400, directly evaluated on other datasets + Weight Ensemble):

| Method | HMDB-51 | UCF-101 | K-600 | All Avg |
|------|:-------:|:-------:|:-----:|:-------:|
| Vanilla CLIP | 40.8 | 63.2 | 59.8 | 54.6 |
| X-CLIP | 44.6 | 72.0 | 65.2 | 60.6 |
| ViFi-CLIP (WE) | 52.2 | 81.0 | 73.9 | 69.0 |
| Open-VCLIP (WE) | 53.9 | 83.4 | 73.0 | 70.1 |
| **TC-CLIP (WE)** | **54.2** | 82.9 | **75.8** | **71.0** |
| TC-CLIP + LLM (WE) | **56.0** | **85.4** | **78.1** | **73.2** |

**Fully Supervised Recognition** (K-400):

| Method | Top-1 | Top-5 | Frames |
|------|:-----:|:-----:|:----:|
| ActionCLIP | 83.8 | 96.2 | 32 |
| X-CLIP | 84.7 | 96.8 | 16 |
| ViFi-CLIP | 83.9 | 96.3 | 16 |
| **TC-CLIP** | **85.2** | **96.9** | 16 |

### Ablation Study

**Component Ablation** (zero-shot + Weight Ensemble):

| Configuration | HMDB-51 | UCF-101 | K-600 | All (Δ) | Description |
|------|:-------:|:-------:|:-----:|:-------:|------|
| Baseline (ViFi-CLIP) | 52.2 | 81.0 | 73.9 | 69.0 | — |
| + TC | 54.3 | 81.9 | 75.5 | 70.6 (+1.6) | Temporal contextualization is independently effective. |
| + VP | 53.4 | 82.0 | 74.7 | 70.0 (+1.0) | Video-conditional prompting is independently effective. |
| + TC + VP | 54.2 | 82.9 | 75.8 | 71.0 (+2.0) | Both are complementary. |

**Token Aggregation Strategies Comparison** (few-shot average Top-1):

| Strategy | HMDB | UCF | SSv2 | All (Δ) |
|------|:----:|:---:|:----:|:------:|
| Baseline (no reference tokens) | 62.6 | 89.2 | 8.7 | 53.5 |
| No merging (directly using seed tokens) | 57.2 | 85.6 | 7.7 | 50.2 (−3.3) |
| Random merging | 58.8 | 87.1 | 7.5 | 51.2 (−2.3) |
| K-means | 62.1 | 89.7 | 9.0 | 53.6 (+0.1) |
| DPC-KNN | 63.3 | 90.2 | 9.8 | 54.4 (+0.9) |
| **Bipartite soft matching** | **63.4** | **90.2** | **9.9** | **54.5 (+1.0)** |

**Computational Cost Comparison**:

| Method | Params | GFLOPs | Throughput | Zero Avg | Full Top-1 |
|------|:-----:|:------:|:-----:|:--------:|:----------:|
| ViFi-CLIP | 124.3M | 285 | 38 | 69.0 | 83.9 |
| Open-VCLIP | 124.3M | 308 | 29 | 70.1 | - |
| TC-CLIP | 127.5M | 304 | 24 | 71.0 | 85.2 |
| TC-CLIP (Lightweight) | 127.5M | 291 | 34 | 70.7 | 84.9 |

### Key Findings

1. **Context tokens are the only consistently effective reference token representation**: Preliminary experiments in Table 1 clearly demonstrate the inadequacies of other forms (CLS, neighboring patches, and all patches).
2. **No merging leads to extrapolation issues**: Directly using seed tokens drops performance by 3.3 points, verifying the CLIP sequence length extrapolation hypothesis.
3. **TC provides larger gains in Weight Ensemble (WE) settings**: Increasing from +0.7 to +1.6, showcasing that representations learned by TC are more complementary to original CLIP representations.
4. **VP compensates for insufficient text-side information**: Using only learnable prompt vectors decreases zero-shot performance (−0.2), whereas VP infusing visual information boosts performance by +1.1.
5. **Largest improvements observed on the temporal-sensitive SSv2 dataset**: Under the base-to-novel setting, SSv2 performance boosts from 5.1 HM to 15.2 HM.

## Highlights & Insights

1. **In-depth and thorough problem analysis**: The derivation logic is highly rigorous, scaling from a systematic comparison of four temporal modeling strategies (Figures 2-3) to exploratory experiments in Table 1, and eventually to the final design.
2. **Elegant design concept of context tokens**: Instead of simply expanding the token sequence, it performs "selection first, then compression, and finally injection", striking an optimal balance between information capacity and computational cost.
3. **The VP module uses a stop-gradient to prevent visual signals from backpropagating to the text encoder**: Excellent design of details.
4. **Bias matrix B distinguishes local/global information**: A simple yet effective design, allowing the model to learn how to balance intra-frame and video-level information layer-by-layer and head-by-head.
5. **Convincing visualization and analysis**: Context tokens tracking frisbees across frames and attention map comparisons visually clarify improvements in temporal understanding.

## Limitations & Future Work

1. **Decreased inference throughput**: Dropping from 38 (ViFi-CLIP) to 24, which is partially mitigated by the lightweight version (34) but still introduces overhead.
2. **Validated only on ViT-B/16**: Validation on larger models (e.g., ViT-L) remains unexplored.
3. **Token selection relies on CLS attention**: Performance exhibits sensitivity to the quality of the CLS token.
4. **High cost of full-parameter fine-tuning**: Less parameter-efficient compared to adapter-based methods.
5. **Room for improvement on highly temporal-sensitive tasks (like SSv2)**: The absolute accuracy of 14.0% in few-shot scenarios is still relatively low.

## Related Work & Insights

- **ToMe (Token Merging)**: The bipartite soft matching used in TC directly originates from here, but is creatively applied to cross-frame temporal information aggregation.
- **CoOp/CoCoOp (Prompt Learning)**: The prompt learning motivation of the VP module is derived from these, with the innovation of conditioning the prompt using visual context tokens rather than abstract image features.
- **CLIP $\rightarrow$ Video paradigm evolution** (ActionCLIP $\rightarrow$ X-CLIP $\rightarrow$ ViFi-CLIP $\rightarrow$ Open-VCLIP $\rightarrow$ TC-CLIP): Reflects the continuous progression from simplicity to complexity in this research direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The "select-aggregate-inject" workflow of context tokens serves as an effective new paradigm, though individual components are not completely original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four evaluation protocols + 5 benchmarks + highly detailed ablation studies, presenting an exemplary experimental design.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Solid, logical narrative flowing from motivation analysis to methodological explanation and experimental verification.
- **Value**: ⭐⭐⭐⭐ — Delivers an efficient paradigm for expanding CLIP to video tasks with high practicality, which will likely be widely referenced by future works.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Referring Atomic Video Action Recognition](referring_atomic_video_action_recognition.md)
- [\[ECCV 2024\] Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning](efficient_few-shot_action_recognition_via_multi-level_post-reasoning.md)
- [\[ECCV 2024\] R²-Tuning: Efficient Image-to-Video Transfer Learning for Video Temporal Grounding](r2tuning_efficient_imagetovideo_transfer_learning_for_video.md)
- [\[ECCV 2024\] FinePseudo: Improving Pseudo-Labelling through Temporal-Alignability for Semi-Supervised Fine-Grained Action Recognition](finepseudo_improving_pseudo-labelling_through_temporal-alignablity_for_semi-supe.md)
- [\[ECCV 2024\] Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition](masked_video_and_body-worn_imu_autoencoder_for_egocentric_action_recognition.md)

</div>

<!-- RELATED:END -->
