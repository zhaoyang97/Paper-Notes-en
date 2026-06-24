---
title: >-
  [Paper Note] SK-VQA: Synthetic Knowledge Generation at Scale for Training Context-Augmented Multimodal LLMs
description: >-
  [ICML2025 Oral Spotlight][Multimodal VLM][Knowledge-based VQA] Fully automated generation of SK-VQA, a large-scale synthetic KB-VQA dataset containing over 2 million QA pairs, using GPT-4 to train MLLMs for context-augmented generation, significantly outperforming existing datasets in cross-domain generalization.
tags:
  - "ICML2025 Oral Spotlight"
  - "Multimodal VLM"
  - "Knowledge-based VQA"
  - "Synthetic Data"
  - "Multimodal RAG"
  - "Context-Augmented Generation"
  - "MLLM Fine-tuning"
date: 2026-05-08
content_hash: 13181bdb47e09555
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# SK-VQA: Synthetic Knowledge Generation at Scale for Training Context-Augmented Multimodal LLMs

**Conference**: ICML2025 Oral Spotlight  
**arXiv**: [2406.19593](https://arxiv.org/abs/2406.19593)  
**Code**: [GitHub](https://github.com/UKPLab/SK-VQA) / [HuggingFace](https://huggingface.co/datasets/skvqa/SK-VQA)  
**Area**: Multimodal VLM  
**Keywords**: Knowledge-based VQA, Synthetic Data, Multimodal RAG, Context-Augmented Generation, MLLM Fine-tuning

## TL;DR
Fully automated generation of SK-VQA, a large-scale synthetic KB-VQA dataset containing over 2 million QA pairs, using GPT-4 to train MLLMs for context-augmented generation, significantly outperforming existing datasets in cross-domain generalization.

## Background & Motivation

**Core Problem**: Existing MLLMs are not designed for "context-augmented generation," making them unsuitable for direct application in multimodal RAG systems. To make MLLMs work effectively in RAG scenarios, a large amount of training data containing "image + question + context document" is required, but such naturally matched data is extremely scarce on the Internet.

**Limitations of Prior Work**:

- **ViQuAE**: Only 3.7k QA pairs, which is too small in scale.
- **InfoSeek**: 1.3 million QA pairs, but less than 1% are unique. It relies on template-based construction, leading to very poor diversity.
- **Enc-VQA**: 1 million QA pairs, with only 17% unique. Images only come from iNaturalist and Google Landmarks.
- The aforementioned datasets are all limited by the requirement that images must link to Wikipedia pages, resulting in a narrow domain coverage and single-faceted language styles due to template-based generation.

**Motivation**: Leveraging powerful foundation models (like GPT-4) for fully automated synthetic data generation can overcome the bottlenecks of image sources and question diversity, constructing a large-scale dataset capable of effectively training MLLMs for context-augmented generation.

## Method

### 3.1 Data Generation Pipeline

Given an input image, a single prompt drives GPT-4 to simultaneously generate:
1. **Context Document**: Wikipedia-style articles related to the image (without directly referencing the image).
2. **Multiple QA Pairs**: Requiring joint reasoning over both the image and the context document to answer.

Key Design—**Single-Step Generation**: Context documents and QA pairs are generated simultaneously in a single inference step. This constrains context generation by the task of producing QAs that require joint reasoning over both the image and the context, ensuring highly matched context and QAs. An average of 7.1 QA pairs are generated per image (with GPT-4 context), compared to only 5.7 pairs with Wikipedia context.

**Image Sources** (three types to ensure domain diversity):

| Image Source | Context Source | QA Pairs |
|---------|-----------|--------|
| LAION-400M | GPT-4 | 908,116 |
| Wikipedia (WIT) | GPT-4 | 702,332 |
| Wikipedia (WIT) | Wikipedia | 181,554 |
| COCO-Counterfactuals | GPT-4 | 214,487 |
| **Total** | | **2,006,489** |

### 3.2 Image Reference (IR) Filtering

GPT-4 sometimes directly references the input image in the generated context (e.g., "In the image, ..."). This type of context resembles an extended caption rather than a knowledge document, which is impractical in real RAG scenarios. Filtering is performed by detecting whether words like *picture*, *photo*, *image*, or *painting* appear in the context, yielding $\text{SK-VQA}_{\text{IR}}$ (1.53M QA pairs).

### 3.3 Context Answer Presence (CAP) Filtering

Further requiring that at least one candidate answer explicitly appears in the context document while not directly referencing the image yields $\text{SK-VQA}_{\text{IR+CAP}}$ (985k QA pairs). This filtering improves data quality—human accuracy reaching 87% on this subset (vs. 77% on the complete set).

### Data Diversity Analysis

| Metric | InfoSeek | Enc-VQA | SK-VQA |
|-----|----------|---------|--------|
| Total QA Pairs | 1,356K | 1,036K | 2,006K |
| Unique Questions | 1,498 | 175K | **1,928K** |
| Unique Question Ratio | <1% | ~17% | **96%+** |
| Vocabulary Size | 725 | 40,787 | **138,372** |
| Avg. Question Length | 8.9 | 11.6 | **12.7** |

The number of unique questions in SK-VQA is **11 times** that of Enc-VQA, fully demonstrating the advantages of strong model generation over template-based generation.

## Key Experimental Results

### Zero-Shot Evaluation (6 SOTA MLLMs)

| Model | InfoSeek | Enc-VQA | ViQuAE | SK-VQA |
|------|----------|---------|--------|--------|
| PaliGemma-3B | 25.66 | 32.89 | 47.72 | 25.51 |
| LLaVA-v1.5-7B | 42.82 | 53.69 | 78.41 | 40.99 |
| LLaVA-v1.6-7B | 41.94 | 57.92 | 72.00 | 46.68 |
| Idefics2-8B | 44.33 | 67.92 | 82.43 | 38.08 |
| LLaVA-v1.6-34B | 38.81 | 77.73 | 79.17 | 50.02 |

SK-VQA is highly challenging for all models, with performance comparable to InfoSeek and far lower than scores on Enc-VQA/ViQuAE. Moreover, larger models do not necessarily perform better, indicating that scale alone is insufficient to resolve the reasoning difficulty of this dataset.

### Fine-Tuning Generalization Experiments (Core Conclusion)

Using InfoSeek, Enc-VQA, and SK-VQA (200K samples each), models were fine-tuned on LLaVA-7B and PaliGemma-3B to test cross-domain performance:

- **InfoSeek Fine-tuning**: Shows improvements on SK-VQA, but no enhancement on Enc-VQA or ViQuAE.
- **Enc-VQA Fine-tuning**: None of the cross-domain metrics exceed the baseline.
- **SK-VQA Fine-tuning**: Achieves significant zero-shot improvements on both InfoSeek and Enc-VQA, while also outperforming models fine-tuned on the other two datasets when evaluated on ViQuAE.

Fine-tuning on SK-VQA in PaliGemma-3B yields prominent improvements across all 9 cross-domain evaluations, being the only training set that does not cause performance degradation.

### Ablation Study on Data Sources

| Image + Context | InfoSeek | Enc-VQA | ViQuAE | Average |
|------------|----------|---------|--------|------|
| LAION + GPT-4 | 44.32 | 65.44 | 79.22 | 62.99 |
| Wiki + GPT-4 | 47.00 | 53.98 | 78.58 | 59.85 |
| Wiki + Wiki | 47.75 | 66.67 | 77.95 | 64.12 |
| COCO-CFs + GPT-4 | **48.00** | **65.42** | **79.23** | **64.22** |

The best combination is COCO-CFs (synthetic images) + GPT-4 context, which even outperforms Wiki real images + real context, indicating that synthetic data can be more effective than real data.

### RAG Experiments

In a real RAG environment simulated by retrieving the top-10 passages using CLIP Score Fusion on PaliGemma-3B, the SK-VQA fine-tuned model exhibits the strongest performance in-domain and out-of-domain, exceeding the baseline and models fine-tuned on other datasets in all 9 cross-domain scenarios.

### Human Evaluation

- QA Quality: Human accuracy is 77% (complete set) and 87% (IR+CAP subset), with a standard deviation of only 0.02–0.03.
- Factuality: 86% verifiable as factual, with only 4% non-factual.
- GPT-4o Automated Evaluation: Context factuality 4.6/5, question relevance 4.9/5, answerability 99.6%, and answer correctness 90.7%.

## Highlights & Insights

1. The ingenuity of the **single-step generation strategy** lies in constraining context generation by the requirements of the QA task, preventing divergence between the context and the QA.
2. **Synthetic images (COCO-CFs) + synthetic context** surprisingly outperform fine-tuning on real data, challenging the intuition that "real data is always better."
3. Different image sources contribute distinct generalization capabilities (LAION benefits Enc-VQA/ViQuAE, while Wiki benefits InfoSeek); mixing multiple sources is key.
4. Attempts to substitute GPT-4 with LLaVA-34B for data generation failed, as 76% of the questions were invalid (most could be answered solely from the context), showing a remaining significant gap for open-source models on this task.
5. The dataset covers diverse domains such as art, fashion, sports, and music, far exceeding the entity knowledge scope of existing KB-VQA datasets.

## Limitations & Future Work

1. **Dependency on GPT-4 for Generation**: High dataset construction cost, and it cannot completely avoid GPT-4's own biases and hallucinations (despite human verification showing 86% factuality, there is still 4% non-factual content).
2. **Immature Open-Source Alternatives**: The attempt to use LLaVA-34B instead of GPT-4 failed (76% of questions were invalid), limiting community replication and scaling.
3. **Data Volume Halved After Filtering**: The data scale was reduced from 2 million to 985k after IR+CAP filtering, demonstrating a significant reduction in high-quality subset sizes.
4. **Focus Only on Textual Context**: Enhancement via multimodal contexts (such as diagrams or video clips) remains unexplored.
5. **Limited Evaluation Metrics**: Precise matching used in InfoSeek and ViQuAE might underestimate models' actual capabilities; fine-tuning was evaluated only on a limited set of models.
6. 9% of the QA pairs can be answered from the context alone (without needing the image); this portion of the data has limited value for multimodal reasoning training.

## Related Work & Insights

- **OK-VQA → InfoSeek**: Evolution from "external knowledge required" to "information retrieval-style QA," both constrained by template-based construction.
- **REVEAL / Wiki-LLaVA / Re-ViLM**: Representative works in multimodal RAG systems focusing on the retriever side, whereas this work targets generator adaptation.
- **Shumailov et al. (2023)**: Training models on model-generated data may prompt model collapse, but this work mitigates the issue by mixing real and synthetic data.
- **UniIR / UniMur**: Unified methods for multimodal retrieval, which can complement the generator training in this work.

## Rating
- Novelty: ⭐⭐⭐⭐ (The fully automated synthetic pipeline + single-step generation design is novel, though the core idea remains "using strong models to create data".)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Zero-shot/fine-tuning/ablation/RAG/human evaluation/automated evaluation, covering aspects very comprehensively.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and well-organized experiments.)
- Value: ⭐⭐⭐⭐ (The publicly available dataset provides a practical impetus to the multimodal RAG community, though its reliance on GPT-4 restricts scalability.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](../../NeurIPS2025/multimodal_vlm/windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)
- [\[ACL 2025\] MAGIC-VQA: Multimodal and Grounded Inference with Commonsense Knowledge for Visual Question Answering](../../ACL2025/multimodal_vlm/magic-vqa_multimodal_and_grounded_inference_with_commonsense_knowledge_for_visua.md)
- [\[ACL 2025\] Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](../../ACL2025/multimodal_vlm/code_guided_text_rich_image.md)
- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](../../NeurIPS2025/multimodal_vlm/benchmarking_retrievalaugmented_multimodal_generation_for_do.md)
- [\[AAAI 2026\] Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning](../../AAAI2026/multimodal_vlm/knowledge_completes_the_vision_a_multimodal_entity-aware_retrieval-augmented_gen.md)

</div>

<!-- RELATED:END -->
