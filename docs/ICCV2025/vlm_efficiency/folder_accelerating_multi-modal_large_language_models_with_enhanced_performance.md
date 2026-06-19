---
title: >-
  [Paper Note] FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance
description: >-
  [ICCV 2025][Multimodal VLM][Visual Token Compression] This paper proposes FOLDER — a plug-and-play visual token compression module that systematically analyzes three key factors of information loss (reduction impact…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Visual Token Compression"
  - "MLLM Acceleration"
  - "Plug-and-Play"
  - "Token Merging"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: 9dc4df7464366623
---

# FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance

**Conference**: ICCV 2025  
**arXiv**: [2501.02430](https://arxiv.org/abs/2501.02430)  
**Code**: [GitHub](https://github.com/anakin-skywalker-Joseph/Folder)  
**Area**: Multimodal VLM  
**Keywords**: Visual Token Compression, MLLM Acceleration, Plug-and-Play, Token Merging, Inference Acceleration

## TL;DR

This paper proposes FOLDER — a plug-and-play visual token compression module that systematically analyzes three key factors of information loss (reduction impact, propagation effect, and aggregation method), performs aggressive token merging in the last few layers of the visual encoder, and achieves up to 70% token reduction while maintaining or even improving model performance.

## Background & Motivation

MLLMs demonstrate strong performance on multimodal tasks, yet the long visual token sequences produced by visual backbone networks introduce substantial computational challenges:

- High-resolution MLLMs (e.g., LLaVA-NeXT) may generate over 2,000 visual tokens
- Multi-visual-encoder architectures further exacerbate sequence length issues
- Video understanding models (e.g., VideoLLaVA) exceed 2,000 tokens even with only 8 frames
- The quadratic complexity of the attention mechanism makes real-time deployment difficult

**Limitations of Prior Work:**

**Training-time methods** (Q-Former, Resampler, Pooling): suffer from notable performance degradation, poor scalability, and tight coupling to specific architectures.

**Inference-time methods** (ToMe, Turbo): adopt uniform compression across all layers without accounting for the heterogeneous nature of information loss, leading to suboptimal results.

**FastV**: prunes visual tokens inside the LLM based on attention scores, but must retain visual tokens in the KV-cache throughout the forward pass, making it ill-suited for long-context dialogue scenarios.

**Core Problem:** *Where does information loss actually come from?* The authors answer this through systematic empirical analysis, which then informs the design of an effective token compression strategy.

## Method

### Overall Architecture

FOLDER is integrated as a plug-and-play module into the last few blocks of the visual backbone. The core idea is to perform aggressive token compression at the position where redundancy is highest and propagation impact is smallest.

### Key Designs

1. **Reduction Impact Analysis**:  
   SVD decomposition is used to estimate the minimum number of tokens required to preserve a given energy threshold. For a token sequence $\mathbf{X} \in \mathbb{R}^{n \times d}$, singular values $\sigma_i$ are obtained via SVD, and the energy retention ratio is defined as:
   $$E(k) = \frac{\sum_{i=1}^{k} \sigma_i}{\sum_{i=1}^{n} \sigma_i}$$
   Experiments reveal that **later blocks require far fewer tokens than earlier blocks**, indicating high redundancy in the deeper layers.

2. **Propagation Effect Analysis**:  
   Due to the sequential nature of the Transformer, token compression errors introduced at early blocks accumulate and amplify across layers (a butterfly effect). Earth Mover's Distance (EMD) is used to measure the impact of compressing tokens at different blocks on the final output distribution:
   $$\text{EMD}(P_Y, P_{\tilde{Y}_b}) = \min_{\gamma} \langle \gamma, \mathbf{M} \rangle_F$$
   Experiments demonstrate that **compression at early blocks induces a far larger EMD than at later blocks**. Even at a 75% compression ratio, the EMD remains extremely low when compression is applied in the final layers. Combining these two analyses leads to the conclusion: **token compression should be concentrated at the end of the network**.

3. **Aggregation Method Analysis**:  
   Three aggregation strategies are compared — direct dropping ($\alpha_{i_{max}}=1$, others set to 0), average merging ($\alpha_i = 1/m$), and weighted merging ($\alpha_i = \|x_i\|_2 / \sum_j \|x_j\|_2$). Experiments show that both average and weighted merging yield substantially lower EMD than direct dropping, and that average merging is simpler with comparable performance; **average merging** is therefore adopted.

4. **FOLDER Algorithm**:  
   Based on the above analysis, bipartite graph matching-based token merging is performed in the final blocks. To overcome the 1/2 compression ceiling of standard bipartite matching, an **iterative FOLD operation** is designed:
   - Tokens are divided into two equal sets $\mathbb{A}$ and $\mathbb{B}$
   - For each token in $\mathbb{A}$, the most similar token in $\mathbb{B}$ is identified via a matching function $S$
   - Matches are ranked by score, and the top-$r_{\text{fold}}$ pairs are merged
   - When the required number of reductions exceeds $\lfloor n/2 \rfloor$ (reduction overflow), multiple FOLD iterations are automatically applied
   - Each FOLD halves the token count until the remaining reduction fits within a single FOLD pass

### Loss & Training

FOLDER supports two usage modes:
- **Inference acceleration**: directly inserted into the visual encoder of a pretrained model with no additional training, reducing visual token count at inference time
- **Training acceleration / performance enhancement**: integrated during the pretraining phase of an MLLM, serving simultaneously as a training accelerator and a regularizer

## Key Experimental Results

### Main Results (LLaVA-1.5 7B Inference Acceleration)

| Method | Compression | Speedup | MMBench | MME | POPE | SEEDBench | Avg. |
|--------|-------------|---------|---------|-----|------|-----------|------|
| Original-7B | 0% | 1× | 62.8 | 1338.9 | 79.7 | 60.2 | 55.0 |
| Pooling | 50% | 1.5× | 59.5 | 1308.6 | 84.0 | 59.2 | 53.7 |
| FastV | 50% | 1.3× | 63.3 | 1345.1 | 81.0 | 59.4 | 54.9 |
| Turbo | 50% | 1.5× | 60.4 | 1311.2 | 83.0 | 58.1 | 52.3 |
| **FOLDER** | **50%** | **1.5×** | **62.4** | **1338.2** | **85.4** | **59.5** | **55.5** |
| FastV | 66% | 1.5× | 62.5 | 1353.2 | 79.3 | 58.3 | 54.6 |
| **FOLDER** | **66%** | **1.7×** | **61.4** | **1350.0** | **85.4** | **59.7** | **54.8** |

### Ablation Study (LLaVA-1.5 13B)

| Method | Compression | Speedup | MMBench | MME | POPE | Avg. |
|--------|-------------|---------|---------|-----|------|------|
| Original-13B | 0% | 1× | 66.6 | 1371.1 | 86.4 | 56.3 |
| Pooling (50%) | 50% | 1.5× | 64.1 | 1316.6 | 85.7 | 55.1 |
| FastV (50%) | 50% | 1.3× | 66.4 | 1386.3 | 85.4 | 55.9 |
| **FOLDER (50%)** | **50%** | **1.5×** | **65.4** | **1383.7** | **86.9** | **56.8** |
| **FOLDER (66%)** | **66%** | **1.6×** | **65.8** | **1366.9** | **86.1** | **56.1** |
| FOLDER+FastV (75%) | 75% | 1.8× | 65.1 | 1368.6 | 85.8 | 56.2 |

### Key Findings

- At 50% compression, FOLDER not only preserves performance but yields notable gains on POPE and ScienceQA
- On MiniGPT4v2 at 60% compression, FOLDER's MME score (859.9) substantially surpasses the original model (631.4), demonstrating a strong regularization effect
- FOLDER and FastV are complementary — the former compresses within the visual encoder, the latter prunes within the LLM — and their combination achieves 75% compression at 1.8× speedup
- When used as a training accelerator, FOLDER improves performance across all benchmarks at compression ratios as high as 70%

## Highlights & Insights

- **Systematic empirical analysis**: the three-pronged framework of SVD energy analysis, EMD propagation effect, and aggregation method comparison provides a principled answer to the fundamental question of where information loss originates
- **Counter-intuitive finding**: although SVD energy analysis suggests compressing in earlier layers, propagation effect analysis refutes this — errors introduced at early layers are amplified across the network
- **Justification for aggressive compression**: tokens in the final layers are highly redundant, enabling aggressive compression with negligible loss
- **Dual utility**: the method functions as an accelerator at inference time and as a regularizer during training, reflecting its flexibility

## Limitations & Future Work

- The design space of matching functions remains underexplored (cosine similarity from ToMe is directly inherited)
- Multiple FOLD iterations introduce marginal additional computational overhead, which — while claimed to be negligible — may become non-trivial at extreme compression ratios
- Validation is limited to ViT-based visual encoders; CNN backbones and hybrid architectures are not evaluated
- Experimental verification in video understanding scenarios is limited

## Related Work & Insights

- Inherits the bipartite graph matching paradigm from ToMe while breaking its 50% compression ceiling
- Complements FastV: FOLDER compresses on the encoder side whereas FastV prunes on the LLM side
- The methodology of propagation effect analysis (tracking output distribution shifts via EMD) is generalizable to other model compression scenarios

## Rating

- **Novelty**: ⭐⭐⭐⭐ The systematic analysis framework constitutes the primary contribution; the iterative FOLD strategy is an elegant engineering innovation
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple models (LLaVA, MiniGPT4v2), scales (7B/13B), scenarios (inference/training), and detailed ablations
- **Writing Quality**: ⭐⭐⭐⭐ Logic is clear; the narrative structure from "three questions" to solution is well-constructed
- **Value**: ⭐⭐⭐⭐⭐ Plug-and-play, open-source, and highly practical, with direct relevance to MLLM deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ICCV 2025\] Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval](bidirectional_likelihood_estimation_with_multi-modal_large_language_models_for_t.md)
- [\[ICCV 2025\] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models](multi-cache_enhanced_prototype_learning_for_test-time_generalization_of_vision-l.md)
- [\[NeurIPS 2025\] Efficient Multi-modal Large Language Models via Progressive Consistency Distillation](../../NeurIPS2025/multimodal_vlm/efficient_multi-modal_large_language_models_via_progressive_consistency_distilla.md)
- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)

</div>

<!-- RELATED:END -->
