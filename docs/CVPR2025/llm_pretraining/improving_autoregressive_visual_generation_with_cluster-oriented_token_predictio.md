---
title: >-
  [Paper Note] Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction
description: >-
  [CVPR 2025][LLM Pretraining][Autoregressive visual generation] This paper proposes IAR, which rearranges the VQGAN codebook via balanced K-means to align similar embeddings with adjacent indices. Combined with a cluster-oriented cross-entropy loss that guides the model to correctly predict the semantic cluster of the target token, IAR halves the training time while improving generation quality across all LlamaGen scales from 100M to 1.4B.
tags:
  - "CVPR 2025"
  - "LLM Pretraining"
  - "Autoregressive visual generation"
  - "codebook rearrangement"
  - "cluster-oriented loss"
  - "LlamaGen"
  - "training efficiency"
date: 2026-05-08
content_hash: fdc0b13f9f8e0a24
---

# Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction

**Conference**: CVPR 2025  
**arXiv**: [2501.00880](https://arxiv.org/abs/2501.00880)  
**Code**: [https://github.com/sjtuplayer/IAR](https://github.com/sjtuplayer/IAR)  
**Area**: LLM Pre-training  
**Keywords**: Autoregressive visual generation, codebook rearrangement, cluster-oriented loss, LlamaGen, training efficiency

## TL;DR

This paper proposes IAR, which rearranges the VQGAN codebook via balanced K-means to align similar embeddings with adjacent indices. Combined with a cluster-oriented cross-entropy loss that guides the model to correctly predict the semantic cluster of the target token, IAR halves the training time while improving generation quality across all LlamaGen scales from 100M to 1.4B.

## Background & Motivation

**Background**: LLM-based visual generation methods first quantize images into discrete tokens using VQGAN, and then utilize GPT-style autoregressive prediction to generate images. Methods like LlamaGen and VAR directly borrow the next-token prediction paradigm from NLP.

**Limitations of Prior Work**: In text generation, predicting an incorrect token index means outputting a completely different word (semantically unrelated). In contrast, in image generation, the embedding corresponding to a "wrong" predicted index might be very close to the target embedding in the feature space, yielding almost identical decoded images. Standard cross-entropy loss penalizes all incorrect predictions indiscriminately, failing to exploit this domain-specific continuity of vision.

**Key Challenge**: Discreteness of the text domain vs. continuity of the image domain—LLM frameworks are designed for discrete tokens, and directly applying them to image codes in continuous feature spaces results in wasted training efficiency.

**Goal**: Utilize the spatial correlation of image embeddings to relax the prediction target from "exact match" to "falling within the correct semantic cluster."

**Key Insight**: Empirical findings show that when the "code distance" (nearest neighbor rank distance) between embeddings is less than 12, the decoded images are almost indistinguishable. This implies that as long as the predicted token falls within the semantic cluster near the target, generation quality is guaranteed.

**Core Idea**: Rearrange the codebook to give similar embeddings contiguous indices $\rightarrow$ Introduce cluster-level cross-entropy loss $\rightarrow$ Ensure high probability of falling into the correct cluster even if the exact token index is predicted incorrectly.

## Method

### Overall Architecture

IAR introduces two preprocessing and training improvements to the standard LlamaGen framework: (1) rearranging the VQGAN codebook with balanced K-means before training; (2) adding a cluster-level cross-entropy loss on top of the original token-level cross-entropy loss during training.

### Key Designs

1. **Codebook Rearrangement**:

    - **Function**: Assigns contiguous indices to embeddings that are close to each other in the feature space.
    - **Mechanism**: In the original VQGAN codebook, embeddings with adjacent indices are completely unrelated. Ideally, one should find a permutation that minimizes the sum of distances between adjacent embeddings (which is a Hamiltonian path problem, NP-hard). This is relaxed to a clustering problem: balanced K-means is used to partition the $N$ embeddings into $n$ equal-sized clusters (each with $m = N/n$ elements), where embeddings in the same cluster are highly similar. During assignment, embeddings closer to the centroids are processed first to ensure each cluster size does not exceed $m$. Finally, embeddings in the same cluster are mapped to the contiguous indices $[jm, (j+1)m)$.
    - **Design Motivation**: The rearranged codebook ensures that "proximal index distance" is equivalent to "embedding similarity," providing a structural foundation for the subsequent cluster-level loss.

2. **Cluster-oriented Cross-entropy Loss**:

    - **Function**: Guides the model to first predict the correct cluster, relaxing the requirement for exact token indices.
    - **Mechanism**: For the output probability distribution $\hat{Y}$, the token probabilities within the same cluster are aggregated into a cluster-level probability $\hat{Y}_{C,j} = \sum_{i=jm}^{(j+1)m-1} \exp(\hat{Y}_i) / \sum_{i=1}^{N} \exp(\hat{Y}_i)$ via log-sum-exp operation. Then, the cluster-level cross-entropy $\mathcal{L}_{CCE}$ is calculated. The final loss is defined as $\mathcal{L} = \mathcal{L}_{TCE} + \lambda \mathcal{L}_{CCE}$.
    - **Design Motivation**: Because the number of clusters $n$ is much smaller than the codebook size $N$, predicting the correct cluster is a much easier task. Once the cluster is correct, the decoded image remains highly similar even if the specific token is incorrect.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{TCE} + \lambda \mathcal{L}_{CCE}$, where $\mathcal{L}_{TCE}$ is the standard token-level cross-entropy and $\mathcal{L}_{CCE}$ is the cluster-level cross-entropy. Codebook rearrangement is a one-time preprocessing step, introducing virtually no additional training overhead.

## Key Experimental Results

### Main Results

| Model | Params | FID↓ | IS↑ |
|------|--------|------|-----|
| LlamaGen-B | 111M | 5.46 | 193.6 |
| + IAR | 111M | **4.72** | **210.3** |
| LlamaGen-L | 343M | 3.80 | 248.3 |
| + IAR | 343M | **3.20** | **263.5** |
| LlamaGen-XL | 775M | 3.39 | 227.1 |
| + IAR | 775M | **2.89** | **256.2** |
| LlamaGen-XXL | 1.4B | 3.10 | 253.9 |
| + IAR | 1.4B | **2.70** | **277.8** |

### Ablation Study

| Configuration | FID↓ | Training Epochs |
|------|------|-----------|
| Baseline | 5.46 | 300 |
| + Codebook Rearrangement | 5.12 | 300 |
| + Codebook Rearrangement + Cluster Loss | **4.72** | 300 |
| Baseline to reach FID 4.72 | 4.72 | ~600 |

### Key Findings
- IAR consistently improves performance across all parameter scales (100M to 1.4B), conforming to scaling laws.
- Under the same target FID, IAR halves the training time (300 epochs vs 600 epochs).
- Codebook rearrangement alone contributes to ~30% of the improvement, while cluster loss contributes to ~70%.
- When the code distance is $< 12$, the MSE and LPIPS changes of the decoded images are minimal, validating the rationality of exploiting embedding correlation.

## Highlights & Insights
- **Revealing the essential difference between images and text**: Text requires exact indices, whereas images require approximate locations in the embedding space. This observation is profound and instructive.
- **Zero-cost benefits of codebook rearrangement**: A one-time preprocessing step that does not modify the model architecture or increase inference cost, yet yields substantial performance gains.
- **Plug-and-play capability**: Applicable to any LLM-based visual generation model (autoregressive or Masked Image Modeling), offering high generalizability.

## Limitations & Future Work
- The number of clusters $n$ and cluster size $m$ in balanced K-means require hyperparameter tuning.
- Currently verified only on unconditional/class-conditional image generation; more complex scenarios such as text-to-image have not been tested.
- Rearrangement depends on a fixed VQGAN codebook; joint optimization of the codebook and rearrangement might hold more potential.
- Not yet validated in combination with non-standard autoregressive methods like VAR (next-scale prediction).

## Related Work & Insights
- **vs LlamaGen**: IAR adds codebook rearrangement and cluster loss on top of LlamaGen, dramatically boosting efficiency without altering the architecture.
- **vs VAR**: VAR improves efficiency by changing the prediction granularity (next-scale), while IAR improves efficiency by relaxing the strictness of the loss function. The two approaches are orthogonal.
- **Analogy to label smoothing**: The cluster loss shares similarities with label smoothing (relaxing hard labels) but is more structured—it only distributes probability mass among semantically similar tokens.

## Rating
- Novelty: ⭐⭐⭐⭐ Profound observation; the combined design of codebook rearrangement and cluster loss is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Verification across multiple scales, training efficiency analysis, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Solid analysis and natural derivation of motivation.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, compatible with scaling laws, holding high value for the LLM visual generation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)
- [\[ACL 2025\] Pre-Training Curriculum for Multi-Token Prediction in Language Models](../../ACL2025/llm_pretraining/pre-training_curriculum_for_multi-token_prediction_in_language_models.md)
- [\[ICLR 2026\] Autoregressive Models Rival Diffusion Models at Any-Order Generation](../../ICLR2026/llm_pretraining/autoregressive_models_rival_diffusion_models_at_any-order_generation.md)
- [\[ICLR 2026\] Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries](../../ICLR2026/llm_pretraining/beyond_multi-token_prediction_pretraining_llms_with_future_summaries.md)
- [\[CVPR 2025\] Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection](seeing_what_matters_empowering_clip_with_patch_generation-to-selection.md)

</div>

<!-- RELATED:END -->
