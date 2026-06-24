---
title: >-
  [Paper Note] EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens
description: >-
  [AAAI 2026][Multimodal Efficiency][Knowledge Distillation] This paper proposes EM-KD, a distillation framework that leverages the Hungarian algorithm to address the vision token count imbalance between teacher and student models. By combining Vision Semantic Distillation (VSD) and Vision-Language Affinity Distillation (VLAD), EM-KD transfers knowledge from a vanilla teacher to an efficient student MLLM, achieving an average score of 50.4 across 11 benchmarks at 144 tokens/pat…
tags:
  - "AAAI 2026"
  - "Multimodal Efficiency"
  - "Knowledge Distillation"
  - "Efficient MLLM"
  - "Vision Token Compression"
  - "Hungarian Matching"
  - "Cross-Modal Alignment"
date: 2026-05-08
content_hash: 895b9d682a2768f1
---

# EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens

**Conference**: AAAI 2026
**arXiv**: [2511.21106](https://arxiv.org/abs/2511.21106)  
**Code**: N/A  
**Area**: Multimodal VLM / Model Compression
**Keywords**: Knowledge Distillation, Efficient MLLM, Vision Token Compression, Hungarian Matching, Cross-Modal Alignment

## TL;DR
This paper proposes EM-KD, a distillation framework that leverages the Hungarian algorithm to address the vision token count imbalance between teacher and student models. By combining Vision Semantic Distillation (VSD) and Vision-Language Affinity Distillation (VLAD), EM-KD transfers knowledge from a vanilla teacher to an efficient student MLLM, achieving an average score of 50.4 across 11 benchmarks at 144 tokens/patch — surpassing LLaVA-NeXT with 576 tokens (49.4) while delivering nearly 2× inference speedup.

## Background & Motivation

- **Background**: Efficient MLLMs reduce computational overhead by compressing or pruning redundant vision tokens, but this inevitably causes visual information loss, particularly on fine-grained understanding tasks. Knowledge distillation can enhance student model capability during training without affecting inference efficiency.
- **Limitations of Prior Work**: Existing MLLM distillation methods (e.g., LLaVA-KD, Align-KD) require a one-to-one spatial correspondence between teacher and student vision tokens, making them unable to handle token count imbalances arising from heterogeneous visual encoders and projectors.
- **Key Challenge**: Different resolutions, visual encoders, and projectors all lead to misaligned vision token counts between teacher and student, a pervasive yet previously overlooked problem in practice.

## Core Problem
When the teacher retains a large number of tokens (e.g., 576) via a powerful visual encoder, while the student retains only a small subset (e.g., 144) through a compression projector, how can effective token-level correspondences be established for knowledge distillation?

## Method

### Overall Architecture
EM-KD uses LLaVA-OneVision-SI as the teacher and LLaVA-NeXT with adaptive average pooling for vision token compression as the student. The framework consists of three core components: **Vision Token Matching (VTM) → Vision Semantic Distillation (VSD) → Vision-Language Affinity Distillation (VLAD)**. Training proceeds in two phases: Phase-1 trains the efficient visual projector (CC-558K); Phase-2 performs full-model SFT with distillation (779K mixed data), where EM-KD is applied exclusively in Phase-2.

### Key Designs
1. **Vision Token Matching (VTM)**: Both teacher and student vision tokens are decoded into vocabulary space via the LM head to obtain vision logits. A cost matrix is constructed using Manhattan distance, and a GPU-accelerated Hungarian algorithm solves the optimal bipartite graph matching. The key insight is that vision logits carry explicit semantics in vocabulary space — image patches can be mapped to meaningful words — yielding more accurate distance metrics than comparisons in hidden state space.
2. **Vision Semantic Distillation (VSD)**: For matched teacher–student vision logit pairs, reverse KL divergence is used to measure the distance between discrete probability distributions in vocabulary space. Logits are chosen over hidden states as the distillation target because they are semantically richer and the teacher–student pair shares a vocabulary, eliminating the need for additional alignment layers.
3. **Vision-Language Affinity Distillation (VLAD)**: Unlike conventional methods that only model relationships among vision tokens, VLAD computes a cosine similarity matrix (affinity matrix) between matched vision tokens and text tokens, then minimizes the Smooth L1 distance between teacher and student affinity matrices to reinforce cross-modal alignment.

### Loss & Training
Total loss: $\mathcal{L} = \alpha\mathcal{L}_{sup} + (1-\alpha)\mathcal{L}_{rld} + \beta\mathcal{L}_{vsd} + \gamma\mathcal{L}_{vlad}$, where $\alpha=0.5, \beta=0.25, \gamma=25$. $\mathcal{L}_{sup}$ is the standard SFT loss and $\mathcal{L}_{rld}$ is reverse KL distillation on response tokens. The matching process is gradient-free and does not participate in backpropagation.

## Key Experimental Results

| Metric | Ours (EM-KD 0.6B) | LLaVA-NeXT (vanilla) | DeCo | TokenPacker | LLaVA-KD* |
|--------|------|----------|------|------|------|
| Avg. (11 bench) | **50.4** | 49.4 | 47.7 | 47.0 | 49.7 |
| TTFT (ms) ↓ | **54.9** | 103.3 | 54.9 | 61.0 | - |
| ChartQA | **59.1** | 34.3 | 34.1 | 34.3 | 57.3 |
| DocVQA | **64.1** | 57.6 | 53.8 | 52.9 | 62.7 |
| OCRBench | **39.7** | 38.6 | 34.4 | 33.5 | 39.0 |

At the 8B scale, EM-KD also consistently outperforms MiniLLM (62.5) and LLaVA-KD (62.3), achieving an average score of 63.4.

### Ablation Study
- Incremental component contributions: Baseline 47.7 → +VLAD 48.4 (+0.7) → +VSD 49.5 (+1.8) → +RLD 50.4 (+2.7); each component contributes independently.
- Matching method comparison: Average Pooling (48.6) < Hungarian by Hidden States (49.4) < Hungarian by Logits (50.4), demonstrating that semantic-space matching far outperforms simple pooling.
- Distillation target comparison: Hidden States (49.5) < Logits (50.4), confirming that vision logits are a more effective distillation target.

## Highlights & Insights
- The combination of **Hungarian matching and vision logits** is elegant — it transplants the set-matching idea from DETR into the distillation setting, resolving a practical and important problem.
- The finding that **vision logits carry explicit semantics** is highly inspiring: image patches, after passing through the LM head, can be mapped to meaningful words (e.g., "sky," "dog"), providing a unified semantic space for cross-architecture distillation.
- The "dual win" of model compression and distillation — 4× fewer tokens yet better performance (+1.0) with nearly 2× inference speedup — is compelling.
- The VLAD design paradigm is transferable to other scenarios requiring stronger vision-language alignment.

## Limitations & Future Work
- The Hungarian algorithm has $O(n^3)$ complexity; although it runs on GPU without gradients, it may become a bottleneck when the token count is large.
- Validation is limited to the LLaVA model family; architectures such as InternVL and Qwen-VL have not been evaluated.
- Only the final-layer vision logits are explored; intermediate-layer information may also carry valuable signals.
- Generalization to video understanding, multi-image understanding, and similar scenarios remains unverified.

## Related Work & Insights
- **vs. LLaVA-KD**: LLaVA-KD can only distill models sharing the same visual encoder and projector; EM-KD removes this constraint via VTM and additionally incorporates cross-modal affinity distillation.
- **vs. MiniLLM**: MiniLLM distills only response tokens, ignoring the rich semantic information in visual features; EM-KD shows clear advantages on OCR- and chart-related tasks.
- **vs. FastV/PyramidDrop**: These training-free pruning methods are incompatible with Flash Attention, limiting their practical speedup; EM-KD's training-based compression can fully exploit accelerated operators.
- The vision logits as a unified semantic space concept may be extendable to cross-modal retrieval, multimodal fusion, and related areas.
- The Hungarian matching strategy can be applied to other misaligned knowledge transfer scenarios, such as feature map distillation across different resolutions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introducing Hungarian matching into MLLM distillation is novel, though each individual component is not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 11 benchmarks, two model scales, and comprehensive ablation studies — very thorough.
- **Writing Quality**: ⭐⭐⭐⭐ Logic is clear, though some formula descriptions could be more concise.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a practical and important problem in MLLM distillation with strong methodological generality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model](../../CVPR2026/vlm_efficiency/transprune_token_transition_pruning_for_efficient_large_vision-language_model.md)
- [\[ICCV 2025\] ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers](../../ICCV2025/vlm_efficiency/shortv_efficient_multimodal_large_language_models_by_freezing_visual_tokens_in_i.md)
- [\[CVPR 2026\] What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](../../CVPR2026/vlm_efficiency/what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)
- [\[ICLR 2026\] iLLaVA: An Image Is Worth Fewer Than 1/3 Input Tokens in Large Multimodal Models](../../ICLR2026/vlm_efficiency/illava_an_image_is_worth_fewer_than_13_input_tokens_in_large_multimodal_models.md)
- [\[CVPR 2026\] TimeViper: A Hybrid Mamba-Transformer Vision-Language Model for Efficient Long Video Understanding](../../CVPR2026/vlm_efficiency/timeviper_a_hybrid_mamba-transformer_vision-language_model_for_efficient_long_vi.md)

</div>

<!-- RELATED:END -->
