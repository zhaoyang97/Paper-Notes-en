---
title: >-
  [Paper Note] Pre-Training Curriculum for Multi-Token Prediction in Language Models
description: >-
  [ACL 2025][LLM Pretraining][multi-token prediction] To address the issue where small language models (SLMs) struggle to directly benefit from the multi-token prediction (MTP) objective, forward and reverse curriculum learning strategies are proposed. The forward curriculum (NTP→MTP) allows SLMs to improve generation quality while maintaining self-speculative decoding acceleration, whereas the reverse curriculum (MTP→NTP) achieves better NTP performance but loses the inference…
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "multi-token prediction"
  - "curriculum learning"
  - "pre-training strategies"
  - "inference acceleration"
  - "small language models"
date: 2026-05-08
content_hash: 66c04414225ebeec
---

# Pre-Training Curriculum for Multi-Token Prediction in Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.22757](https://arxiv.org/abs/2505.22757)  
**Code**: [https://github.com/aynetdia/mtp_curriculum](https://github.com/aynetdia/mtp_curriculum)  
**Area**: LLM Pre-training  
**Keywords**: multi-token prediction, curriculum learning, pre-training strategies, inference acceleration, small language models

## TL;DR
To address the issue where small language models (SLMs) struggle to directly benefit from the multi-token prediction (MTP) objective, forward and reverse curriculum learning strategies are proposed. The forward curriculum (NTP→MTP) allows SLMs to improve generation quality while maintaining self-speculative decoding acceleration, whereas the reverse curriculum (MTP→NTP) achieves better NTP performance but loses the inference acceleration advantage.

## Background & Motivation

**Background**: Multi-Token Prediction (MTP) is an emerging pre-training objective that tasks the model with predicting the next $k$ tokens at each step (instead of the traditional single token). Gloeckle et al. (2024) and DeepSeek-V3 have demonstrated that MTP can enhance downstream performance, inference speed, and training efficiency of large models.

**Limitations of Prior Work**: The benefits of MTP scale with model size—**small language models (SLMs, 1-3B) benefit limitedly from MTP**, or even suffer performance degradation. This is because SLMs have limited parameter capacity, making it difficult to handle morphological and semantic dependencies across multiple tokens from the very beginning.

**Key Challenge**: The MTP objective is "too difficult" for SLMs—directly forcing SLMs to predict 4 tokens simultaneously exceeds their learning capability, yet the inference acceleration and richer hidden state representations enabled by MTP remain highly valuable.

**Goal**: Design a pre-training curriculum learning strategy that enables SLMs to effectively leverage the advantages of the MTP objective.

**Key Insight**: Curriculum learning by Bengio et al. (2009) schedules learning tasks from simple to complex. Since the complexity of MTP naturally increases with the number of predicted tokens, it is highly suitable for adjustment via curriculum learning.

**Core Idea**: Dynamically adjust the number of predicted tokens during training—employing forward curriculum ($k$=1→2→...→$k_{max}$) and reverse curriculum ($k_{max}$→...→2→1) to replace static MTP.

## Method

### Overall Architecture
Based on the standard Transformer architecture, MTP is implemented using multiple LM heads (each head predicting the $i$-th future token). During training, the number of active heads $k_{current}(e)$ is dynamically adjusted according to the curriculum, and inactive heads do not participate in loss computation. The total training steps are evenly distributed among each phase.

### Key Designs

1. **Forward Curriculum**:

    - **Function**: Starting from NTP ($k$=1), the number of predicted tokens is gradually increased to $k_{max}$.
    - Formula: $k_{current}(e) = \min(k_{max}, \lfloor e / (E/k_{max}) \rfloor + 1)$
    - **Design Motivation**: Intended to let the model first master basic token-by-token prediction to build a solid language modeling foundation, before gradually increasing the complexity. This resembles the human learning process of "learning to walk before learning to run."
    - Expected Advantage: Maintains self-speculative decoding capability (eventually all heads are trained).

2. **Reverse Curriculum**:

    - **Function**: Starting from full MTP ($k_{max}$), the number of predicted tokens is gradually decreased to NTP ($k$=1).
    - Formula: $k_{current}(e) = \max(1, k_{max} - \lfloor e / (E/k_{max}) \rfloor)$
    - **Design Motivation**: Based on the discovery that "MTP pre-training enhances NTP downstream performance"—MTP is first used to learn richer hidden state representations, followed by a gradual focus transition to NTP, allowing the main LM head to obtain "perceptual benefits" from MTP.
    - Expected Advantage: Stronger NTP performance for the main LM head.

3. **Two LM Head Designs**:

    - **Linear Layers (LL)**: Each additional head is a $d_{hidden} \times d_{vocab}$ linear layer, allowing parallel prediction but increasing memory overhead (+65-98M parameters per head for subword models).
    - **Transformer Layers (TL)**: The last $k$ layers of the model are made responsible for predicting different tokens, sharing the output linear layer. This does not add parameters but instead "reduces" the effective depth of the backbone.

### Loss & Training
- Models: 1.3B and 3B Llama architectures, with subword (32K vocabulary) and byte (320 vocabulary) tokenization methods.
- Data: MiniPile (1.7B subword tokens / 5.9B bytes).
- Training: 1 epoch training, batch size 1024, cosine decay learning rate.

## Key Experimental Results

### Main Results (NTP performance, using only the main LM head)

| Model | Type | Heads | Curriculum | MiniPile BPB↓ | LAMBADA BPB↓ | BLiMP Acc↑ |
|---|---|---|---|---|---|---|
| 1.3B subword | NTP baseline | 1 LL | - | 1.08 | 1.34 | 71.80 |
| 1.3B subword | Static MTP | 4 LL | - | 1.19 | 1.61 | 67.48 |
| 1.3B subword | Forward | 4 LL | ✓ | ~1.16 | ~1.52 | ~68.5 |
| 1.3B subword | **Reverse** | 4 LL | ✓ | ~1.14 | ~1.45 | ~69.5 |
| 3B byte | NTP baseline | 1 LL | - | 1.07 | 1.00 | 71.20 |
| 3B byte | Static MTP | 4 LL | - | 1.08 | 1.00 | 71.42 |
| 3B byte | **Reverse** | 4 LL | ✓ | ≤1.07 | ≤1.00 | **≥71.42** |

- The reverse curriculum meets or exceeds the NTP baseline on byte-level models, being the only MTP strategy to achieve this.
- Both forward and reverse curricula **outperform static MTP** in all configurations, demonstrating the effectiveness of curriculum learning for MTP in SLMs.
- Subword models benefit less than byte models because subword tokens carry more semantic information, making simultaneous prediction of multiple tokens significantly harder.

### Ablation Study (Inference Acceleration)

| Configuration | Reduction in Forward Passes Relative to NTP |
|---|---|
| Static MTP 4 heads | Highest (~1.6x-1.8x acceleration) |
| **Forward Curriculum** | Close to static MTP (~1.5x-1.7x) |
| Reverse Curriculum | Basically no acceleration (converges to ~1.0x) |

### Key Findings
- **Complementarity of Forward vs. Reverse Curriculum**: The forward curriculum maintains inference acceleration but yields slightly lower NTP performance, while the reverse curriculum offers stronger NTP performance but loses inference acceleration. There is no "one-size-fits-all" solution.
- **Byte-level Models are Better Suited for MTP Curricula**: Because a single byte carries less semantic information, predicting multiple bytes is much easier than predicting multiple subwords, making it easier for SLMs to master.
- **Minimal Difference Between LL and TL Heads**: Even though TL models "lose" some parameter capacity during inference (since the last few layers are allocated to other heads), the performance gap remains negligible.
- **10B Token Experiment Validates Scalability**: Additional experiments on the FineWeb-Edu 10B dataset confirm the advantages of curriculum learning.
- **Generation Quality (BLEU/ROUGE/SemScore/G-Eval)**: Both forward and reverse curricula improve the quality of generated text, with the reverse curriculum performing best in semantic similarity.

## Highlights & Insights
- **Extremely Simple Yet Effective Idea**: By simply adjusting the number of active heads according to a schedule during training—without modifying the model architecture, adding extra parameters, or changing the total training steps—SLMs can leverage MTP more effectively.
- **Symmetric Design of Forward and Reverse Curricula**: Elegant separation of the two optimization directions of "inference acceleration" and "NTP performance" offers a clear selection guide for practical applications—use forward for inference speed, and use reverse for NTP performance.
- **Unique Analysis of Byte vs. Subword**: It reveals that the fundamental cause of the difference in MTP effectiveness is token granularity, rather than model capacity itself. This provides new motivation for the development of byte-level LLMs.

## Limitations & Future Work
- **Limited Data Scale**: MiniPile consists of only 1.7B tokens, which is far smaller than standard LLM pre-training datasets. The large-scale scalability of the conclusions needs further validation.
- **Oversimplified Uniformly-Phased Curriculum**: Adaptive adjustment of $k_{current}$ based on training dynamics might yield better results.
- **Relatively Small Model Sizes**: Experiments focus on 1.3B and 3B models. Since Gloeckle et al. found that MTP performs better on 7B+ models, whether curriculum learning remains valuable for large models is still unknown.
- **Insufficient Evaluation Benchmarks**: Knowledge-intensive tasks (e.g., MMLU) could not be fully evaluated due to data scale constraints.
- **Future Directions**: Adaptive curriculum strategies (dynamically adjusting $k$ based on loss curves); validation on larger models and datasets; exploration of non-uniform phase partitioning; integration with multi-head speculative methods such as Medusa.

## Related Work & Insights
- **vs. Gloeckle et al. (2404)**: They proposed MTP and found that while large models benefit more, small models benefit limitedly. This work makes small models benefit through curriculum learning, serving as a direct improvement on the original work.
- **vs. DeepSeek-V3 (Liu et al. 2024)**: DeepSeek-V3 employs static MTP on a 671B MoE model; the curriculum strategy in this paper serves as an alternative approach for small models.
- **vs. Medusa (Cai et al. 2024)**: Medusa adds extra heads for speculative decoding post-training; the MTP heads in this work are trained during the pre-training phase, maintaining more consistent representations.
- This paper provides a crucial step toward the democratization of MTP, enabling resource-constrained researchers to leverage MTP on small models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Applying curriculum learning to the MTP training objective is a natural yet effective idea, and the symmetrical design of forward/reverse curricula is simple and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐ Covers two model sizes × two tokenizations × two head types, but the overall data scale is quite small.
- **Writing Quality**: ⭐⭐⭐⭐ Clear presentation and reasonable experimental design.
- **Value**: ⭐⭐⭐⭐ Serves as a practical improvement for MTP training, which is particularly informative for resource-constrained scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)
- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2026\] Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement](../../ACL2026/llm_pretraining/toward_consistent_world_models_with_multi-token_prediction_and_latent_semantic_e.md)
- [\[ICLR 2026\] Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries](../../ICLR2026/llm_pretraining/beyond_multi-token_prediction_pretraining_llms_with_future_summaries.md)
- [\[ACL 2025\] AsyncLM: Efficient and Adaptive Async Pre-training of Language Models](asynclm_efficient_and_adaptive_async_pre-training_of_language_models.md)

</div>

<!-- RELATED:END -->
