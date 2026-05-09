---
title: >-
  [Paper Note] GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding
description: >-
  [CVPR 2026][Multimodal VLM][Video Temporal Grounding] This paper proposes GroundVTS, a query-guided fine-grained visual token sampling architecture for video large language models, which adaptively preserves spatiotemporally relevant information at the token level. It achieves an 18.4-point mIoU improvement on Charades-STA and a 20.6-point mAP improvement on QVHighlights.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Video Temporal Grounding
  - Visual Token Sampling
  - Query-Guided
  - Video Large Language Models
  - Temporal Reasoning
date: 2026-05-08
content_hash: 1b063c429b03fcc2
---

# GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding

**Conference**: CVPR 2026
**arXiv**: [2604.02093](https://arxiv.org/abs/2604.02093)
**Code**: Available (GitHub)
**Area**: Multimodal VLM / Video Understanding
**Keywords**: Video Temporal Grounding, Visual Token Sampling, Query-Guided, Video Large Language Models, Temporal Reasoning

## TL;DR
This paper proposes GroundVTS, a query-guided fine-grained visual token sampling architecture for video large language models, which adaptively preserves spatiotemporally relevant information at the token level. It achieves an 18.4-point mIoU improvement on Charades-STA and a 20.6-point mAP improvement on QVHighlights.

## Background & Motivation

**Background**: Video Temporal Grounding (VTG) is a fundamental task in video understanding, requiring the precise localization of temporal boundaries in video segments based on natural language queries. Recent video large language models (Vid-LLMs) such as Qwen2.5-VL and InternVL have made progress in general video reasoning, but remain insufficient for fine-grained temporal understanding. Existing improvement methods have introduced query-conditioned attention, temporal boundary regressors, and temporal modeling modules.

**Limitations of Prior Work**: (1) Existing Vid-LLMs predominantly adopt uniform frame sampling strategies, allocating equal input budgets across all temporal segments, causing query-relevant key moments to be diluted or missed. (2) Some recent methods (e.g., CLIP-based frame selection) achieve query-guided frame-level sampling, but operate at coarse granularity (entire frames) and rely on external multimodal encoders, limiting localization precision and adaptability. (3) The density and relevance of visual tokens directly affect VTG performance — experiments show that frame rates below 1 FPS provide insufficient information, while rates above 2.4 FPS introduce redundant tokens that dilute key signals.

**Key Challenge**: Fixed uniform sampling presents a fundamental trade-off between "information coverage" and "signal dilution" — higher frame rates provide more temporal cues but introduce greater redundancy, while lower frame rates reduce redundancy but risk missing key moments. An adaptive, query-aware sampling mechanism is therefore necessary.

**Goal**: How can query-guided adaptive sampling be achieved at the visual token level within a Vid-LLM, preserving critical spatiotemporal information while suppressing redundant content?

**Key Insight**: The work is motivated by frame-rate sensitivity experiments — Qwen2.5VL-7B achieves peak mIoU of 47.8% at 2.0–2.4 FPS, with sharp degradation beyond this range. This indicates that VTG requires the *right* tokens rather than *more* tokens. Accordingly, the sampling granularity is shifted from the frame level to the token level, performing differentiable top-K selection within the VLM (after the visual encoder and multimodal projection layer) based on token-query similarity.

**Core Idea**: A query-guided, token-level differentiable sampling module is introduced into the Vid-LLM, implemented end-to-end via Gumbel-Softmax with straight-through estimation (STE), combined with a three-stage progressive optimization strategy to accommodate non-uniform token distributions.

## Method

### Overall Architecture
Input video frames are temporally downsampled and encoded by a visual encoder into dense spatiotemporal features $H_v \in \mathbb{R}^{N_v \times D_v}$, then projected into a shared embedding space $V \in \mathbb{R}^{N_v \times D}$ via a multimodal projector. The VTS module scores each token in $V$ according to the text query $Q$ and performs differentiable top-K selection, producing a compact subset $\tilde{V}$. Selected tokens retain their original positional encodings to preserve temporal alignment. Finally, $\tilde{V}$ is concatenated with query tokens and fed into the LLM for joint reasoning and generation.

### Key Designs

1. **Query-Guided Token Scoring**:

    - *Function*: Estimates the relevance of each visual token to the text query.
    - *Mechanism*: Visual embeddings $V$ and query embeddings $Q$ (after mean pooling) are projected into a lower-dimensional subspace via trainable projection matrices $W_v, W_q \in \mathbb{R}^{D \times D_r}$, respectively. A temperature-scaled dot product followed by softmax yields the weight distribution $\mathbf{w} = \text{softmax}(V'{\mathbf{q}'}^\top / \tau)$. This is essentially an attention mechanism — the weights reflect the semantic alignment of each visual token with the query and its relative importance within the sequence.
    - *Design Motivation*: Computing similarity in a projected low-dimensional space reduces computational cost while focusing on semantically relevant features; the temperature parameter controls the sharpness of the distribution.

2. **Differentiable Top-K Selection**:

    - *Function*: Selects the $K=\lceil\rho \cdot N_v\rceil$ most relevant tokens in an end-to-end trainable manner.
    - *Mechanism*: Since hard top-K is non-differentiable, Gumbel-Softmax relaxation combined with a straight-through estimator (STE) is employed. The forward pass performs hard selection $z_i^{\text{hard}} = \mathbf{1}(i \in \mathcal{I}_K)$, while the backward pass routes gradients through the continuous relaxation $z_i$, combined as $\tilde{z}_i = z_i^{\text{hard}} + z_i - \text{stopgrad}(z_i)$. The weights of selected tokens are normalized and applied as: $\hat{w}_i = \frac{\exp(w_i/\tau') \cdot \tilde{z}_i}{\sum_j \exp(w_j/\tau') \cdot \tilde{z}_j}$, with the final output $\tilde{v}_i = \hat{w}_i \cdot \text{MLP}(v_i)$.
    - *Design Motivation*: Hard selection ensures inference efficiency (only $K$ tokens enter the LLM), while Gumbel+STE ensures training differentiability. Weight normalization prevents the signal strength of selected tokens from being diluted.

3. **Three-Stage Progressive Optimization Strategy**:

    - *Function*: Ensures stable convergence of the non-uniform sampling module.
    - *Mechanism*: **Stage 1** (VTS Warm-up): All other components are frozen; only VTS module parameters are trained to learn stable token-query relevance estimation. **Stage 2** (Joint LoRA Adaptation): VTS, the MLP projector, and the LLM (via LoRA) are jointly fine-tuned on the large-scale LLaVA-Video-178K dataset, adapting the model to non-uniform token distributions. **Stage 3** (Grounding Fine-tuning): Fine-tuning is performed on the constructed Grounding-FT dataset (70K samples aggregated from multiple VTG training sets) using a unified instruction-style QA format.
    - *Design Motivation*: Direct joint training of VTS and the LLM leads to unstable gradients and inconsistent selection behavior. Ablations show that skipping Stage 1 (random VTS initialization) causes mIoU to collapse from 22.1 to 5.6; skipping Stage 2 also yields substantially lower performance than the full three-stage pipeline.

### Loss & Training
Standard autoregressive generation loss is used. The VTS module is trained via gradients backpropagated from the language modeling loss. The three stages are trained for 1/2/3 epochs with learning rates of 1e-5/2e-4/1e-4, respectively. The sampling ratio is fixed at $\rho=0.5$.

## Key Experimental Results

### Main Results (Moment Retrieval)

| Method | Charades-STA R1@0.5 | Charades-STA mIoU | ActivityNet R1@0.5 | ActivityNet mIoU |
|------|--------------------|--------------------|-------------------|-----------------|
| VTimeLLM | 27.5 | 31.2 | 27.8 | 30.4 |
| NumPro | 42.0 | 41.4 | 37.5 | 38.8 |
| LLaVA-ST | 44.8 | 42.4 | - | - |
| Qwen2.5VL-7B-G | 32.7 | 31.7 | 23.9 | 26.7 |
| **GroundVTS-Q** | **57.5 (+24.8)** | **50.1 (+18.4)** | **33.6 (+9.7)** | **36.0 (+9.3)** |

### Highlight Detection (QVHighlights)

| Method | MR R1@0.5 | MR R1@0.7 | HD mAP | HD Hit@1 |
|------|-----------|-----------|--------|---------|
| Qwen2.5VL-7B-G | 11.0 | 4.3 | 34.4 | 44.5 |
| GroundVTS-Q | 23.6 (+12.6) | 12.3 (+8.0) | 35.7 (+1.3) | 58.8 (+14.3) |
| InternVL3.5-8B-G | 31.8 | 15.0 | 31.9 | 39.8 |
| **GroundVTS-I** | **63.6 (+31.8)** | **40.7 (+25.7)** | **52.5 (+20.6)** | **88.4 (+48.6)** |

### Ablation Study

| Sampling Strategy | Charades R1@0.5 | Charades mIoU | ActivityNet mIoU |
|---------|----------------|---------------|-----------------|
| Uniform (1.0 FPS) | 28.5 | 29.3 | 23.4 |
| Random (50% drop) | 35.0 | 35.7 | 27.7 |
| Frame-Level Query Selection | 44.9 | 41.6 | 30.7 |
| **Token-Level (Ours)** | **57.5** | **50.1** | **36.0** |

### Training Stage Ablation

| Training Stage | R1@0.3 | R1@0.5 | R1@0.7 | mIoU |
|---------|--------|--------|--------|------|
| None (random VTS) | 8.6 | 5.0 | 1.9 | 5.6 |
| Stage 1 only | 31.2 | 20.5 | 10.0 | 20.9 |
| Stage 1+2 | 45.8 | 28.8 | 13.2 | 30.1 |
| Stage 1+3 | 49.1 | 32.5 | 15.2 | 32.4 |
| **Stage 1+2+3** | **71.5** | **57.5** | **34.2** | **50.1** |

### Key Findings
- Token-level sampling substantially outperforms frame-level sampling (Charades mIoU 50.1 vs. 41.6), confirming the importance of fine-grained selection.
- Removing positional encodings (while retaining VTS) causes mIoU to collapse from 50.1 to 9.5, demonstrating the critical role of temporal alignment.
- Using only 50% of the token budget, GroundVTS already surpasses the baseline's full-density performance (R1@0.7: 34.2 vs. 30.5).
- GroundVTS-I achieves Hit@1 of 88.4% on QVHighlights (a 48.6-point improvement over the baseline), a remarkably strong result.
- On NExT-GQA out-of-distribution evaluation, GroundVTS-Q achieves the highest mIoU of 25.8, demonstrating generalization capability.

## Highlights & Insights
- The frame-rate sensitivity analysis as a design motivation is highly compelling — it clearly demonstrates the "more tokens can be harmful" phenomenon and directly supports the necessity of adaptive sampling.
- The application of Gumbel-Softmax STE for differentiable token-level selection in the VTG setting is an elegant engineering design. This "selection-as-attention" paradigm is transferable to any multimodal task requiring sparse processing.
- The three-stage progressive optimization avoids instability in end-to-end training, and the ablation study precisely quantifies the contribution of each stage, reflecting methodological rigor.

## Limitations & Future Work
- The sampling ratio $\rho=0.5$ is fixed across all data; ideally, it should be adaptively adjusted based on video complexity and query characteristics.
- The performance gain of GroundVTS-I (InternVL3.5 backbone) on Charades-STA is notably smaller than that of GroundVTS-Q, suggesting that the method exhibits varying compatibility with different sampling paradigms (fixed frame rate vs. fixed frame count).
- The VTS module increases training complexity (three stages), and the top-K computation at inference also introduces additional overhead.
- Compressing the query via mean pooling may discard fine-grained semantics, such as temporal relational words (e.g., "after...").
- Performance on very long videos remains insufficiently validated.

## Related Work & Insights
- **vs. NumPro**: NumPro improves temporal understanding by incorporating frame-number auxiliary inputs, but does not alter the visual token sampling strategy. GroundVTS addresses the problem at the token selection level and outperforms NumPro by 8.7 points in Charades-STA mIoU.
- **vs. LLaVA-ST**: LLaVA-ST introduces spatiotemporal tokens but relies on uniform sampling; GroundVTS achieves more precise localization through a query-guided non-uniform distribution.
- **vs. Token Compression Methods (e.g., FastV)**: Existing token compression approaches are largely query-agnostic or saliency-based. GroundVTS's query-guided design is better suited to VTG. The principle that "compression strategies should be aligned with downstream tasks" is broadly applicable.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Query-guided token-level sampling is novel in the VTG domain, though the differentiable top-K selection mechanism itself has precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three VTG benchmarks, OOD evaluation, token density analysis, training stage ablation, and sampling strategy ablation — comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ The frame-rate sensitivity analysis as an introductory motivation is well-structured; method description is clear.
- **Value**: ⭐⭐⭐⭐ Offers a new perspective on visual token utilization in Vid-LLMs, with significant performance gains and broadly transferable ideas.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)
- [\[ICCV 2025\] Enrich and Detect: Video Temporal Grounding with Multimodal LLMs](../../ICCV2025/multimodal_vlm/enrich_and_detect_video_temporal_grounding_with_multimodal_llms.md)
- [\[CVPR 2026\] ViKey: Enhancing Temporal Understanding in Videos via Visual Prompting](vikey_enhancing_temporal_understanding_in_videos_via_visual_prompting.md)
- [\[CVPR 2026\] Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)
- [\[CVPR 2026\] Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)

</div>

<!-- RELATED:END -->
