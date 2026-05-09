---
title: >-
  [Paper Note] Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval
description: >-
  [ICCV 2025][Multimodal VLM][Text-video retrieval] This paper identifies the *candidate prior bias* problem in MLLM-based retrieval systems — where candidate likelihood estimation tends to favor candidates with high prior probability rather than those that are semantically most relevant — and proposes BLiM (Bidirectional Likelihood Estimation) and CPN (Candidate Prior Normalization) to address this issue, achieving an average R@1 gain of 6.4 across four text-video retrieval benchmarks.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Text-video retrieval
  - multimodal large language models
  - bidirectional likelihood estimation
  - candidate prior bias
  - score calibration
date: 2026-05-08
content_hash: 04c1f900cc3f7a3e
---

# Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval

**Conference**: ICCV 2025
**arXiv**: [2507.23284](https://arxiv.org/abs/2507.23284)
**Code**: [github.com/mlvlab/BLiM](https://github.com/mlvlab/BLiM)
**Area**: Multimodal Learning / Video Retrieval
**Keywords**: Text-video retrieval, multimodal large language models, bidirectional likelihood estimation, candidate prior bias, score calibration

## TL;DR

This paper identifies the *candidate prior bias* problem in MLLM-based retrieval systems — where candidate likelihood estimation tends to favor candidates with high prior probability rather than those that are semantically most relevant — and proposes BLiM (Bidirectional Likelihood Estimation) and CPN (Candidate Prior Normalization) to address this issue, achieving an average R@1 gain of 6.4 across four text-video retrieval benchmarks.

## Background & Motivation

Text-video retrieval aims to find the most relevant text (or video) candidate given a video (or text) query. The evolution of existing approaches includes:
- **Dual-encoder architectures** (CLIP, BERT): encode queries and candidates independently into single embeddings and retrieve via similarity; computationally efficient but limited in token-level alignment.
- **MLLM-based retrieval**: processes concatenated query-candidate pairs to enable deep token-level interaction; more effective for long and complex query-candidate pairs.

The authors identify a **candidate prior bias** in MLLM-based retrieval: via Bayesian decomposition,
$$P(\mathbf{t}|\mathbf{v}) = \frac{P(\mathbf{v}|\mathbf{t}) P(\mathbf{t})}{P(\mathbf{v})}$$
the candidate likelihood $P(\mathbf{t}|\mathbf{v})$ is jointly influenced by the query likelihood $P(\mathbf{v}|\mathbf{t})$ and the candidate prior $P(\mathbf{t})$. The autoregressive nature of MLLMs tends to assign higher probability to long and repetitive texts (high prior), causing retrieval results to favor high-frequency patterns over truly semantically matching candidates. Experiments confirm that certain high-prior texts are retrieved by 37% of video queries (374 out of 1,003 videos).

## Method

### Overall Architecture

BLiM is built upon a pretrained video MLLM (VideoChat-Flash 7B), consisting of a UMT video encoder, a linear projection layer, and a Qwen2 LLM. At inference time, a two-stage retrieval pipeline is adopted: InternVideo2 1B first retrieves top-K candidates, which BLiM then re-ranks.

### Key Designs

1. **Bidirectional Likelihood Estimation Training**:

    - **Video-to-text generation** $P(\mathbf{t}|\mathbf{v})$: Standard MLLM pretraining paradigm; autoregressively generates text conditioned on video features.
    $\mathcal{L}_{t|v} = -\sum_{i=1}^{L_t} \log P(t_i | t_{<i}, \mathbf{v})$
    - **Text-to-video feature generation** $P(\mathbf{v}|\mathbf{t})$: Given text, autoregressively predicts the next video clip feature using contrastive softmax.
    $\mathcal{L}_{v|t} = -\sum_{i=1}^{L_v} \log \frac{\exp(\tilde{v}_{i-1}^\top v_i)}{\sum_{n=1}^{N} \exp(\tilde{v}_{i-1}^\top v_i^{(n)})}$
    - Overall training objective: $\mathcal{L}_{BLiM} = \mathcal{L}_{t|v} + \mathcal{L}_{v|t}$
    - The two directions swap input modality order and use different prompts.

2. **Candidate Prior Normalization (CPN)**:

    - A training-free score calibration module that estimates candidate prior probability by applying an attention mask to the input modality.
    - When computing candidate likelihood, all query tokens are masked so that the model generates the candidate without conditioning on the query, yielding a prior estimate.
    - The candidate likelihood is then normalized by the estimated prior to eliminate prior bias.
    - Generality of CPN: applicable beyond retrieval to enhance visual grounding in tasks such as VQA and captioning.

3. **Inference Pipeline**:

    - Video-to-text retrieval: $n^* = \arg\max_n P(\mathbf{t}^{(n)}|\mathbf{v}) + P(\mathbf{v}|\mathbf{t}^{(n)})$
    - Text-to-video retrieval: $n^* = \arg\max_n P(\mathbf{t}|\mathbf{v}^{(n)}) + P(\mathbf{v}^{(n)}|\mathbf{t})$
    - Both directions are jointly considered: the candidate likelihood identifies the most probable candidate to be generated, while the query likelihood identifies the candidate most likely to have generated the query.

### Loss & Training

- Only the linear projection layer and LoRA are fine-tuned, enabling parameter-efficient adaptation.
- Two-stage retrieval: InternVideo2 1B for initial retrieval (top-K) → BLiM for re-ranking.
- Inference complexity is reduced from $O(N^2)$ to $O(KN)$ (e.g., 307× faster on ActivityNet).

## Key Experimental Results

### Main Results (Text-to-Video R@1)

| Method | DiDeMo | ActivityNet | LSMDC | MSRVTT | Average |
|--------|--------|-------------|-------|--------|---------|
| InternVideo2 1B | 57.0 | 60.4 | 32.0 | 51.9 | 50.3 |
| InternVideo2 6B | 57.9 | 63.2 | 33.8 | 55.9 | 52.7 |
| UMT (fine-tuned) | 70.4 | 66.8 | 43.0 | 58.8 | 59.8 |
| InternVideo2 1B* (fine-tuned) | 75.3 | 68.8 | 44.9 | 59.4 | 62.1 |
| **BLiM (Ours)** | **86.4** | **81.0** | **55.7** | **64.7** | **71.9** |

### Ablation Study (Video-to-Text R@1, DiDeMo)

| Configuration | T2V R@1 | V2T R@1 | Notes |
|---------------|---------|---------|-------|
| Candidate likelihood only $P(\mathbf{t}|\mathbf{v})$ | — | Lower | Affected by candidate prior bias |
| Query likelihood only $P(\mathbf{v}|\mathbf{t})$ | — | Higher | More accurate but unidirectional |
| Bidirectional likelihood (BLiM−) | 69.8 | 62.9 | Substantial improvement |
| **BLiM + CPN** | **86.4** | **82.8** | Further alleviates prior bias |

Zero-shot BLiM− (without CPN) on DiDeMo already achieves 69.8 T2V R@1, surpassing all fine-tuned baselines.

### Key Findings

- Candidate prior bias is pervasive in MLLM-based retrieval, affecting both video-to-text and text-to-video directions.
- Query likelihood alone produces reasonably accurate retrieval results (high diagonal similarity), but high-prior candidates under the candidate likelihood distort the rankings.
- CPN improves not only retrieval performance but also other multimodal tasks such as VQA, demonstrating its value as a general debiasing tool.
- The two-stage retrieval pipeline substantially reduces computational cost, making MLLM-based re-ranking practically feasible.

## Highlights & Insights

- The formalization of candidate prior bias is notably rigorous; Proposition 1 formally proves that ranking reversal occurs when the prior gap exceeds the likelihood gap.
- The CPN design is elegant and minimalist — prior estimation is achieved training-free via attention masking followed by normalization.
- The training objective for text-to-video feature generation is novel: contrastive learning replaces actual video decoding, operating entirely within the LLM output space.

## Limitations & Future Work

- The approach relies on a two-stage pipeline (InternVideo2 for initial retrieval), leaving room for improvement in end-to-end efficiency.
- The contrastive loss for text-to-video generation requires all in-batch videos as negatives, potentially sensitive to batch size.
- Validation is currently limited to text-video retrieval; the effectiveness on text-image retrieval remains to be explored.

## Related Work & Insights

- Candidate prior bias is conceptually analogous to language prior (language bias) in VQA; CPN shares a similar spirit with VCD (Visual Contrastive Decoding).
- Bidirectional likelihood can be viewed as an approximation of pointwise mutual information.
- The attention-mask trick for prior estimation generalizes to other scenarios requiring the removal of conditioning bias.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The discovery of candidate prior bias and the bidirectional likelihood solution are both highly novel; CPN is concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four retrieval benchmarks plus extended analysis on multimodal tasks, with substantial performance gains.
- Writing Quality: ⭐⭐⭐⭐⭐ — In-depth problem analysis, rigorous theoretical proofs, and intuitive visualizations.
- Value: ⭐⭐⭐⭐⭐ — An R@1 gain of 6.4 is substantial; CPN as a general-purpose module has broad applicability.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ICCV 2025\] FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance](folder_accelerating_multi-modal_large_language_models_with_enhanced_performance.md)
- [\[ICCV 2025\] CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models](capellm_support-free_category-agnostic_pose_estimation_with_multimodal_large_lan.md)
- [\[CVPR 2026\] EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval](../../CVPR2026/multimodal_vlm/eaglenet_energy-aware_fine-grained_relationship_learning_network_for_text-video_.md)
- [\[ICCV 2025\] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs](instruction-oriented_preference_alignment_for_enhancing_multi-modal_comprehensio.md)

<!-- RELATED:END -->
