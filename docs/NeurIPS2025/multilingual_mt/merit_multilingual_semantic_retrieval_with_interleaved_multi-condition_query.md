---
title: >-
  [Paper Note] MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query
description: >-
  [NeurIPS 2025][interleaved retrieval] This paper introduces MERIT, the first multilingual interleaved multi-condition semantic retrieval dataset (320K queries, 135K products, 5 languages, 7 product categories), exposes the bottleneck of existing retrieval models that focus solely on global semantics while neglecting condition-level details, and proposes the Coral fine-tuning framework that combines embedding reconstruction with contrastive learning to achieve a 45.9% improvement in retrieval performance.
tags:
  - NeurIPS 2025
  - interleaved retrieval
  - multilingual
  - multi-condition query
  - contrastive learning
  - embedding reconstruction
date: 2026-05-08
content_hash: 1bfae56c499d4230
---

# MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query

**Conference**: NeurIPS 2025
**arXiv**: [2506.03144](https://arxiv.org/abs/2506.03144)
**Code**: [https://github.com/weichow23/merit](https://github.com/weichow23/merit)
**Area**: Multilingual Translation
**Keywords**: interleaved retrieval, multilingual, multi-condition query, contrastive learning, embedding reconstruction

## TL;DR
This paper introduces MERIT, the first multilingual interleaved multi-condition semantic retrieval dataset (320K queries, 135K products, 5 languages, 7 product categories), exposes the bottleneck of existing retrieval models that focus solely on global semantics while neglecting condition-level details, and proposes the Coral fine-tuning framework that combines embedding reconstruction with contrastive learning to achieve a 45.9% improvement in retrieval performance.

## Background & Motivation

**Background**: Semantic retrieval is critical in product search, RAG, and related scenarios; however, existing datasets are limited to single-language, single-image, and single-condition settings, falling far short of real-world complexity.

**Limitations of Prior Work**: A large body of prior work (Fashion-IQ, CIRR, Magiclens, etc.) shows no performance degradation when images are replaced by their captions, indicating that these datasets fail to genuinely exploit the expressive capacity of images (Vision Unnecessarity).

**Key Challenge**: Real-world product retrieval frequently involves interleaved multi-condition queries (e.g., a specific pattern combined with a specific material), where many attributes must be conveyed visually. Existing datasets cannot evaluate this capability.

**Core Problem**: (1) How can the capability of existing models on interleaved multi-condition retrieval tasks be comprehensively measured? (2) What are the key factors limiting performance, and how can they be addressed?

## Method

### MERIT Dataset Construction

- **Scale**: 135K products, 320K retrieval pairs (310K train + 10K test), spanning 5 languages (English, Malay, Indonesian, Vietnamese, Thai) and 7 product categories.
- **Annotation Pipeline** (4 steps):
  1. **High-Quality Product Selection**: Popular products are filtered from internal datasets across 6 Southeast Asian countries; GPT-4o generates titles; aesthetic scoring is applied for further filtering.
  2. **Open-Ended Attribute Annotation**: 116 unique attributes and 2,594 attribute values are established via open annotation followed by statistical analysis.
  3. **Query Pair Composition**: Three sampling strategies are combined—uniform sampling, attribute-uniform sampling, and high-similarity-product-prioritized sampling.
  4. **Multi-Round Filtering**: Automatic rule-based filtering followed by expert human review, totaling 10,000 person-hours of annotation effort.
- **Key Properties**: The first semantic retrieval dataset supporting multi-image interleaved input; each query contains ≥2 conditions, with the majority being dual-condition queries (319,600).

### Coral Fine-Tuning Framework

**Core Idea**: When adapting a pretrained MLLM to a multimodal retrieval model, embedding reconstruction is introduced alongside contrastive learning to preserve fine-grained condition information.

1. **Contrastive Learning Loss** $\mathcal{L}_{cl}$:
   $$\mathcal{L}_{cl} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(q_i \cdot k_{i+}/\tau)}{\sum_{j=1}^{N}\exp(q_i \cdot k_j/\tau)}$$
   Standard InfoNCE loss that pulls queries closer to positive samples and pushes them away from negatives.

2. **Visual Reconstruction Loss** $\mathcal{L}_{mse}$:
   - Random masking (rate $\delta=0.5$) is applied to the visual portion of the multimodal embedding $E=[e_{img};e_{txt}]$, and a randomly initialized BERT decoder layer $\mathcal{F}_{\theta}^{v}$ is used for reconstruction:
   $$\mathcal{L}_{mse} = -\frac{1}{N}\sum_{i=1}^{N}\|\hat{E} - E\|_2^2, \quad \hat{E} = \mathcal{F}_{\theta}^{v}[\mathcal{MASK}_v(E); h_{eos}]$$
   - **Design Motivation**: Contrastive learning relying solely on the [EOS] token tends to over-compress into global semantics; masked reconstruction forces the model to retain fine-grained visual information within the [EOS] representation.

3. **Masked Language Modeling Loss** $\mathcal{L}_{mlm}$:
   - Text tokens are masked and then reconstructed; the decoder $\mathcal{F}_{\theta}^{l}$ shares parameters with the MLLM's LM head:
   $$\mathcal{L}_{mlm} = -\frac{1}{N}\sum_{i=1}^{N}\log P(\hat{x}_i \mid X)$$

4. **Total Loss**:
   $$\mathcal{L} = \mathcal{L}_{cl} + \lambda_1 \mathcal{L}_{reg} + \lambda_2 \mathcal{L}_{rec}$$
   where $\mathcal{L}_{reg}$ and $\mathcal{L}_{rec}$ reconstruct the retrieval target using the [EOS] of the condition and the [EOS] of the target itself as attention queries, respectively.

## Key Experimental Results

### Performance of Existing Models on MERIT (Zero-Shot + Embedding Models)

| Method | Scale | Input Type | R@1 | R@5 | R@10 | MRR |
|--------|-------|------------|-----|-----|------|-----|
| Qwen2.5-VL (Zero-Shot) | 3B | Seq | 0.09 | 0.39 | 0.56 | 0.21 |
| LamRA-Qwen2.5VL | 7B | Cat | **12.05** | 39.13 | 48.03 | 23.80 |
| GME-Qwen2VL | 2B | Cat | 8.47 | **47.13** | **56.18** | **25.02** |
| BGE-VL | 7B | Cat | 11.55 | 38.01 | 46.26 | 23.00 |

### Coral Ablation Study (Qwen2.5-VL)

| Method | LoRA | Type | R@1 | R@5 | R@10 | MRR |
|--------|------|------|-----|-----|------|-----|
| CL baseline | ✓ | Seq | 48.52 | 73.11 | 77.93 | 59.48 |
| CL baseline | ✗ | Seq | 47.76 | 73.97 | 80.47 | 59.06 |
| +Coral (Full) | ✗ | Seq | **69.68** | **89.26** | **93.08** | **78.33** |
| +Coral | ✗ | Cat | 60.94 | 85.60 | 90.40 | 71.70 |

- Coral achieves a **45.9%** R@1 improvement over the contrastive-learning-only baseline (47.76 → 69.68).
- Sequential input (Seq) consistently outperforms image concatenation (Cat).
- Full fine-tuning outperforms LoRA.
- Consistent gains are observed across 8 external retrieval benchmarks, with a 181% improvement on VisDial.

### Key Findings
- Concatenated image input achieves **119.7%** higher R@5 than interleaved input before training, but interleaved input improves by 14.3% after training.
- Replacing images with captions causes a **73.9%** performance drop, confirming the indispensability of visual input.
- Error analysis reveals that attribute errors and visual comprehension errors account for the largest proportion of failures.

## Highlights & Insights
- ⭐⭐⭐⭐ **First interleaved multi-condition multilingual semantic retrieval dataset**: fills an important gap; 10K person-hours of annotation ensure quality.
- ⭐⭐⭐⭐ **Precise problem diagnosis**: clearly exposes the bottleneck of existing methods that capture only global semantics while ignoring condition-level details.
- ⭐⭐⭐⭐ **Elegant Coral design**: masked reconstruction serves as a principled complement to contrastive learning, with a concise and effective formulation.
- ⭐⭐⭐ **Comprehensive experiments**: 9 SOTA baselines + 8 external benchmarks + OOD analysis + error attribution.

## Limitations & Future Work
1. The dataset covers only e-commerce product retrieval; transferability to other domains (academic search, news retrieval) remains to be validated.
2. The reconstruction decoder introduces additional training overhead; although it can be discarded at inference, training efficiency warrants attention.
3. Language coverage is skewed toward Southeast Asia and does not include East Asian languages such as Chinese and Japanese.
4. Attribute annotation relies on internal data, limiting reproducibility.

## Rating
⭐⭐⭐⭐ A rigorous benchmark-plus-methodology paper with a well-structured dataset construction pipeline, precise problem identification, and an elegant method design. MERIT has the potential to become an important evaluation standard in multimodal retrieval, and the "reconstruction + contrastive" paradigm of Coral holds broader applicability.

## Related Work & Insights

## Highlights & Insights

## Rating

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity](../../ICLR2026/multilingual_mt/assess_autoformalization_eval.md)
- [\[ACL 2026\] SERM: Self-Evolving Relevance Model with Agent-Driven Learning from Massive Query Streams](../../ACL2026/multilingual_mt/serm_self-evolving_relevance_model_with_agent-driven_learning_from_massive_query.md)
- [\[ACL 2026\] Unlocking the Edge: Multi-LoRA On-Device Deployment and Acceleration](../../ACL2026/multilingual_mt/unlocking_the_edge_deployment_and_ondevice_acceleration_of_multi-lora_enabled_on.md)
- [\[CVPR 2026\] MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation](../../CVPR2026/multilingual_mt/mmtit-bench_a_multilingual_and_multi-scenario_benchmark_with_cognition-perceptio.md)
- [\[NeurIPS 2025\] Enhancing Multilingual LLM Pretraining with Model-Based Data Selection](enhancing_multilingual_llm_pretraining_with_model-based_data_selection.md)

<!-- RELATED:END -->
