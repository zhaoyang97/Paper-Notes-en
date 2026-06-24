---
title: >-
  [Paper Note] GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture
description: >-
  [ACL 2025][LLM Efficiency][Mixture of Experts] Introduces the GigaChat family—the first MoE-architecture LLM family designed and pre-trained from scratch for the Russian language. It includes base and instruction-tuned models with 20B total and 3.3B active parameters, achieving SOTA performance among models of the same scale on Russian benchmarks, with a training speed $2\times$ faster than dense models of equivalent capacity and a 40% reduction in inference latency.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Mixture of Experts"
  - "Russian LLM"
  - "Pre-training"
  - "Tokenizer Optimization"
  - "DPO"
date: 2026-05-08
content_hash: 44e6c20dc870edf0
---

# GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture

**Conference**: ACL 2025  
**arXiv**: [2506.09440](https://arxiv.org/abs/2506.09440)  
**Code**: [https://huggingface.co/ai-sage](https://huggingface.co/ai-sage) (Available, open-source model)  
**Area**: LLM Efficiency  
**Keywords**: Mixture of Experts, Russian LLM, Pre-training, Tokenizer Optimization, DPO

## TL;DR
Introduces the GigaChat family—the first MoE-architecture LLM family designed and pre-trained from scratch for the Russian language. It includes base and instruction-tuned models with 20B total and 3.3B active parameters, achieving SOTA performance among models of the same scale on Russian benchmarks, with a training speed $2\times$ faster than dense models of equivalent capacity and a 40% reduction in inference latency.

## Background & Motivation

**Background**: Multilingual LLMs (e.g., Qwen, Mistral) support Russian primarily through late-stage post-training, lacking base models designed from the ground up for Russian. Existing open-source Russian models (e.g., ruGPT-3.5) perform poorly on benchmarks like MERA.

**Limitations of Prior Work**: (a) Training large-scale LLMs from scratch requires massive computational resources, limiting the development of Russian-specific models; (b) Russian tokenizers are inefficient—generic tokenizers suffer from severe fragmentation when encoding Cyrillic characters; (c) Proprietary Russian models (e.g., YandexGPT) lack transparency, being closed-source and not revealing their architectures.

**Key Challenge**: How to train high-performing Russian LLMs with limited resources? The MoE architecture can significantly reduce computational overhead while maintaining model capacity, but there is a lack of experience in training MoE models for Russian.

**Goal**: Build the first Russian-specific MoE LLM family, covering the entire pipeline of pre-training, instruction-tuning, and alignment, and release them as open-source.

**Key Insight**: MoE architecture (20B total/3.3B active parameters) + Russian-optimized tokenizer + 9.5T tokens pre-training.

**Core Idea**: Utilize the MoE architecture to substantially lower training and inference costs, combined with a customized tokenizer and multi-source data, to build a highly efficient Russian-specific LLM.

## Method

### Overall Architecture
The GigaChat family includes: (1) GigaChat-A3B-base (a 20B total/3.3B active parameter MoE base model); (2) GigaChat-A3B-instruct (the instruction-tuned version); (3) GigaChat-A3B-instruct 1.5 (aligned with DPO). Additionally, there are premium proprietary versions (Lite, Pro, MAX) accessible via API, Telegram Bot, or Web.

### Key Designs

1. **MoE Architecture Design**:

    - **Function**: Replaces dense MLPs with sparse MoE, drastically reducing the computational cost per forward pass.
    - **Mechanism**: Consists of 28 Transformer layers, each containing 2 shared experts and 64 routed experts, with 16 attention heads and 8 KV heads (GQA). The hidden dimension is scaled to align with Mistral 7B (14,336). The first layer uses a standard gated MLP (due to token distribution issues). Block-sparse computation is implemented using STK Triton kernels, bypassing expert parallelism.
    - **Design Motivation**: Compared to an 8B dense model (like Llama 3), it delivers $2\times$ faster training speeds, 40% lower inference latency, and a 40% reduction in computing resources. It draws inspiration from the DeepSeek MoE design, which utilizes more experts, smaller experts, and shared experts.

2. **Russian-Optimized Tokenizer**:

    - **Function**: Optimizes the BPE tokenizer for Cyrillic letters, programming languages, and LaTeX.
    - **Mechanism**: Employs the HuggingFace BBPE algorithm to iteratively train on a mixed corpus of Russian, English, code, and LaTeX. Over 100 candidate tokenizers were generated, with the final choice optimized for the shortest average token length across domains. This ensures Cyrillic high-frequency words are not excessively fragmented, while preserving programming keywords and LaTeX syntax.
    - **Design Motivation**: Generic tokenizers feature low encoding efficiency (fragmentation) for Russian, which directly impairs training efficiency and model capacity utilization.

3. **Pre-training Data and Strategy**:

    - **Function**: Collects 9.5T tokens of multi-source data and trains the model in multiple stages.
    - **Mechanism**: The dataset comprises 4.4T tokens of web data (26.5% Russian, 63.8%+ English), 630B tokens of high-quality documents, 230B tokens of code, and 9B tokens of synthetic data (math and code). A multi-step constant learning rate scheduler (with a 2000-step warmup, decaying at 30%, 60%, 90%, and 98% of the training run) is utilized. The context window is subsequently scaled in two stages: 8K $\rightarrow$ 32K $\rightarrow$ 128K, using RoPE ABF adjustments.
    - **Design Motivation**: Synthetic data generation is inspired by Phi-4, resulting in notable gains in mathematical and coding capabilities. Multi-stage context scaling represents the state-of-the-art standard for long-context LLMs.

4. **Improved DPO Loss**:

    - **Function**: Modifies standard DPO to reduce hallucination and training instability.
    - **Mechanism**: Introduces asymmetric weights $\beta_w$ and $\beta_l$ to prioritize scaling up the score of winning responses rather than solely penalizing losing ones. An additional NLL regularization term relative to the reference model is incorporated to stabilize the loss ratio.
    - **Design Motivation**: Standard DPO focuses excessively on widening the gap between winning and losing responses rather than rising overall absolute generation quality, while ignoring the significance of shared prefixes.

### Loss & Training
- **Pre-training**: Standard next-token prediction with a batch size of ~16M tokens.
- **SFT**: Approximately 250K human-annotated samples covering over 10 domains.
- **DPO**: Asymmetrically weighted loss function with an NLL regularization term.

## Key Experimental Results

### Main Results (Comparison with Models of the Same Scale)

| Benchmark | GigaChat-A3B-instruct 1.5 | Qwen 2.5 (7B) | Llama 3.1 (8B) | T-Lite |
|-----------|---------------------------|---------------|----------------|--------|
| GSM8K (5-shot) | 0.774 | **0.895** | 0.789 | 0.882 |
| MMLU EN (5-shot) | 0.650 | **0.710** | 0.682 | 0.718 |
| MMLU RU (5-shot) | **0.600** | 0.632 | 0.569 | 0.626 |
| RUBQ (0-shot) | **0.688** | 0.373 | 0.484 | 0.583 |
| WINOGRANDE (4-shot) | **0.762** | 0.636 | 0.624 | 0.670 |
| HumanEval (0-shot) | 0.378 | **0.854** | 0.683 | 0.799 |

GigaChat performs exceptionally well on Russian benchmarks (RUBQ, MMLU RU, WINOGRANDE) but lags behind Qwen 2.5 in English and coding tasks.

### Efficiency Comparison

| Metric | GigaChat-A3B (MoE) | Comparable Dense 8B |
|------|---------------------|--------------|
| Training Speed | **2× Faster** | Baseline |
| Inference Latency | **Reduced by 40%** | Baseline |
| Active Parameters | 3.3B | ~8B |
| Total Parameters | 20B | ~8B |

### Key Findings
- **MoE architectures exhibit exceptional efficiency advantages at moderate scales**: With only 3.3B active parameters, it matches the performance of 8B dense models, but with a substantial reduction in computational cost.
- **Russian specialization vs. general multilingual capability shows a trade-off**: GigaChat leads in Russian but falls behind Qwen 2.5 in English and coding, underscoring that data ratio configurations play a critical role in multilingual capacity.
- **The modified DPO is effective**: The instruct 1.5 version (incorporating DPO) outperforms the instruct version on most benchmarks.
- **GigaChat MAX (proprietary large-scale variant)** is highly competitive against Claude 3.7 and GPT-4o, delivering outstanding results on the Russian MERA benchmark.

## Highlights & Insights
- **Comprehensive MoE LLM construction report**: Detailing architecture design, tokenizer training, data ratios, and DPO optimization, this acts as a valuable industrial-grade MoE training technical report that serves as an essential reference for replicating MoE development.
- **Russian tokenizer optimization**: The selection strategy for tokenizers across multilingual, code, and LaTeX domains can easily translate and adapt to other low-resource or non-English languages.
- **Practical improvements to asymmetric DPO loss**: Effectively addresses standard DPO’s limitation of focusing too much on widening the winner-loser gap rather than lifting the absolute generation quality, a strategy that is highly generalizable.

## Limitations & Future Work
- **Weaker performance in English and code**: This limits the model’s appeal to multilingual or global users.
- **Relatively small active scale**: Driven by only 3.3B active parameters, a notable gap remains when compared against 70B+ models.
- **High proportion of English (64%) in pre-training data**: This is somewhat high for a "Russian-specific" model, which could reduce the density of specialized Russian knowledge representation.
- **Discrepancy between open-source models and proprietary versions (MAX/Pro)**: E.g., GSM8K performance of 0.774 vs. 0.956, implying that some of the core technologies might not be fully disclosed.
- **Proposals for future work**: (a) Scaling up the MoE config (e.g., to 100B+ total parameters) to match top-tier models; (b) Conducting specialized optimization for other regional or low-resource languages (e.g., Kazakh, Uzbek).

## Related Work & Insights
- **vs. Mixtral (Jiang et al., 2024)**: Both models share the MoE architecture, but while Mixtral targets English/multilingual settings, GigaChat optimizes the tokenizer and data configuration specifically for the Russian language.
- **vs. DeepSeek MoE (Dai et al., 2024)**: GigaChat adopts DeepSeek's architectural design choice using more experts, smaller granularities, along with dedicated shared experts.
- **vs. ruGPT-3.5**: Previously the strongest open-source Russian model, but significantly outperformed by GigaChat on MERA.

## Rating
- Novelty: ⭐⭐⭐ No major architectural innovations (the MoE paradigm is mature); the primary contributions lie in engineering execution and language-specific target application.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated extensively across multiple Russian and English benchmarks, with comprehensive training configurations documented.
- Writing Quality: ⭐⭐⭐⭐ Informative, technical-report style.
- Value: ⭐⭐⭐⭐ High contribution to the Russian NLP community; offers valuable empirical benchmarks and experiences for MoE training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts](dive_moe_reconstruction.md)
- [\[NeurIPS 2025\] Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search](../../NeurIPS2025/llm_efficiency/jet-nemotron_efficient_language_model_with_post_neural_architecture_search.md)
- [\[ICML 2025\] Mixture of Lookup Experts](../../ICML2025/llm_efficiency/mixture_of_lookup_experts.md)
- [\[NeurIPS 2025\] SPARTA Alignment: Collectively Aligning Multiple Language Models through Combat](../../NeurIPS2025/llm_efficiency/sparta_alignment_collectively_aligning_multiple_language_models_through_combat.md)
- [\[ICLR 2026\] Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models](../../ICLR2026/llm_efficiency/towards_greater_leverage_scaling_laws_for_efficient_mixture-of-experts_language_.md)

</div>

<!-- RELATED:END -->
