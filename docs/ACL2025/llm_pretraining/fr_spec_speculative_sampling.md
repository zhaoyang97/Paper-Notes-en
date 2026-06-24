---
title: >-
  [Paper Note] FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling
description: >-
  [LLM Pretraining] This paper proposes the FR-Spec framework, which optimizes drafting candidate selection in speculative sampling by compressing the vocabulary space based on token frequency. This reduces the LM Head computation overhead by 75% and achieves an additional 1.12× speedup over EAGLE-2 while guaranteeing mathematical equivalence in the output distribution.
tags:
  - "LLM Pretraining"
date: 2026-05-08
content_hash: 2dcf16b8e7c8e117
---

# FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling

## Paper Information

- **Conference**: ACL 2025
- **arXiv**: [2502.14856](https://arxiv.org/abs/2502.14856)
- **Code**: [https://github.com/thunlp/FR-Spec](https://github.com/thunlp/FR-Spec)
- **Area**: LLM Pre-training
- **Keywords**: Speculative Sampling, Large Vocabulary, Frequency Ranking, LM Head Optimization, EAGLE-2

## TL;DR

This paper proposes the FR-Spec framework, which optimizes drafting candidate selection in speculative sampling by compressing the vocabulary space based on token frequency. This reduces the LM Head computation overhead by 75% and achieves an additional 1.12× speedup over EAGLE-2 while guaranteeing mathematical equivalence in the output distribution.

## Background & Motivation

- **Background**: Speculative sampling accelerates LLM autoregressive generation by generating multiple tokens per forward pass using a "draft-then-verify" mechanism. State-of-the-art methods like EAGLE-2 utilize a highly lightweight single-layer Transformer as the draft model.
- **Limitations of Prior Work**: The vocabulary size of mainstream LLMs has expanded from 32K in Llama-2 to 128K in Llama-3 and 152K in Qwen-2.5, yet the negative impact of large vocabulary sizes on speculative sampling efficiency remains unexplored.
- **Key Insight**: After native C/CUDA optimization (eliminating Python overhead), profiling shows that the LM Head accounts for 49% of the drafting computation time, rising to 62% when combined with softmax. Thus, vocabulary-related computations, rather than the Transformer layer, have become the true bottleneck.
- **Mechanism**: The frequency of natural language tokens exhibits a long-tail distribution: 25% of high-frequency tokens cover 95% of occurrences. Restricting the draft model to search only within a high-frequency subset can significantly reduce computational overhead.
- **Value**: This method does not require retraining the draft model and guarantees mathematical equivalence during the verification stage, offering a genuine "free lunch" acceleration.

## Method

### Overall Architecture

FR-Spec is a plug-and-play frequency-ranked speculative sampling framework. During the drafting phase, it compresses the vocabulary space of the LM Head from the full vocabulary $\mathcal{V}$ to a high-frequency subset $\mathcal{V}_{\text{high}}$. During the verification phase, the full vocabulary is retained, thereby guaranteeing mathematical equivalence in the final output distribution.

### Key Designs

1. **Corpus-Level Token Frequency Statistics**: Token frequency distribution is benchmarked on a 1B token subset of SlimPajama-627B, confirming a prominent long-tail effect where 75% of vocabulary tokens account for only 5% of occurrences.
2. **Frequency-Ranked Vocabulary Pruning**: A submatrix $\tilde{\mathbf{W}}_{\text{LM}} \in \mathbb{R}^{|\mathcal{V}_{\text{high}}| \times d}$ is constructed, reducing the drafting model's LM Head projection complexity from $O(nd|\mathcal{V}|)$ to $O(nd|\mathcal{V}_{\text{high}}|)$, achieving a compression ratio of $\frac{|\mathcal{V}|}{|\mathcal{V}_{\text{high}}|}$.
3. **Guaranteed Invariance in the Verification Phase**: Only the drafting process is modified, while the verification process utilizes the full vocabulary. This ensures that the final sampling distribution remains strictly identical to the original method.

### Engineering Optimizations

- Rewrote the EAGLE-2 implementation in native C and CUDA to eliminate Python interpreter overhead.
- Modified FlashAttention to support complex tree-like attention masks.
- Utilized uint64 bit-mask compression (since draft tokens $\le 64$) to optimize memory access patterns.

## Experiments

### Main Results: Llama-3-8B Decoding Speed (tokens/s, temperature=0)

| Method | MT | Conv | RAG | Math | QA | Summ | Code | Average (Speedup) |
|------|-----|------|-----|------|-----|------|------|------------|
| Vanilla | 90.94 | 90.43 | 83.43 | 91.16 | 91.05 | 86.63 | 90.10 | 89.11 (1.00×) |
| EAGLE-2 | 176.79 | 203.41 | 168.05 | 209.88 | 166.60 | 167.12 | 175.11 | 180.99 (2.03×) |
| +FR 32k | **195.60** | **227.68** | **184.85** | **243.36** | **190.27** | **188.14** | 183.19 | **201.87 (2.27×)** |

### Ablation Study: Impact of Different Vocabulary Sizes on Average Acceptance Length (Llama-3-8B)

| Configuration | Average Acceptance Length | Subset Coverage Ratio |
|------|-----------|-----------|
| Full Vocab (128k) | 3.89 | 100% |
| +FR 64k (SlimPajama) | 3.80 | 97.7% |
| +FR 32k (SlimPajama) | 3.63 | 93.3% |
| +FR 16k (SlimPajama) | 3.40 | 87.4% |
| +FR 8k (SlimPajama) | 3.13 | 80.5% |

### Key Findings

1. **32K is the Optimal Balance Point**: Pruning the vocabulary from 128K to 32K reduces the average acceptance length by only 6.7%, but significantly boosts drafting speed, yielding the optimal overall speedup (2.27× vs. 2.03× of EAGLE-2).
2. **Cross-Framework Advantages**: Compared to HuggingFace and SGLang implementations of EAGLE-2, FR-Spec obtains 1.82× and 1.42× additional speedups, respectively.
3. **Impact of Frequency Source**: Frequency statistics from SlimPajama (large-scale pre-training corpus) outperform those from ShareGPT (instruction data), yielding a higher-quality high-frequency subset.
4. **Unaffected Model Quality**: The pass@1 and accuracy on HumanEval and GSM8K remain strictly identical to the original method.
5. **Equally Effective in Random Sampling**: At temperature = 1, FR-Spec still achieves a 1.13× speedup over EAGLE-2.

## Highlights & Insights

- Systematically analyzes the bottleneck of large vocabularies on speculative sampling for the first time, revealing that the LM Head, rather than the Transformer layer, is the true bottleneck.
- Elegant and minimalist approach: leverages the long-tail distribution of natural language token frequencies without requiring any retraining.
- Plug-and-play design: can be directly integrated into existing methods such as EAGLE-2 and Medusa.
- Mathematically guarantees output distribution equivalence, preserving model quality entirely.

## Limitations & Future Work

- Speculative acceleration diminishes when generating content with a high density of low-frequency tokens (e.g., rare proper nouns, technical jargon).
- Token frequency statistics depend on the pre-training corpus distribution, and cross-domain generalization requires further validation.
- Currently only evaluated on a single A800 GPU; performance in multi-GPU or distributed settings remains unverified.
- Optimizations require native C/CUDA implementations, posing a higher engineering bar.

## Related Work & Insights

- **Speculative Sampling**: Speculative Decoding (Leviathan et al., 2023), Medusa (Cai et al., 2024), EAGLE-2 (Li et al., 2024b)
- **Large Vocabulary Issue**: The impact of vocabulary expansion on model capability (Takase et al., 2024; Tao et al., 2024)
- **Inference Acceleration**: Orthogonal optimization paths such as quantization, distillation, and sparse attention
- **Token Frequency Analysis**: Applications of Zipf's Law (Zipf, 1950) in NLP

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Rating | 8.5/10 |

> **Summary**: An excellent paper that balances engineering orientation with theoretical insight. Through a systematic execution profiling, it exposes the bottleneck of large vocabularies in speculative sampling. The frequency-ranked vocabulary pruning strategy proposed based on Zipf's Law is simple, efficient, plug-and-play, and strictly guarantees output equivalence. It holds direct engineering value for LLM inference deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[ICLR 2026\] Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization](../../ICLR2026/llm_pretraining/understanding_and_improving_shampoo_and_soap_via_kullback-leibler_minimization.md)
- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)
- [\[NeurIPS 2025\] Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models](../../NeurIPS2025/llm_pretraining/leveraging_importance_sampling_to_detach_alignment_modules_from_large_language_m.md)
- [\[ACL 2025\] Retrofitting Large Language Models with Dynamic Tokenization](retrofitting_large_language_models_with_dynamic_tokenization.md)

</div>

<!-- RELATED:END -->
