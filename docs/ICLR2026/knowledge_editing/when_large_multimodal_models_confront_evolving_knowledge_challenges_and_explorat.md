---
title: >-
  [Paper Note] When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations
description: >-
  [ICLR 2026][Knowledge Editing][Large Multimodal Models] This paper proposes the EVOKE benchmark to systematically evaluate the ability of Large Multimodal Models (LMMs) to incorporate evolving knowledge…
tags:
  - "ICLR 2026"
  - "Knowledge Editing"
  - "Large Multimodal Models"
  - "Knowledge Injection"
  - "Evolving Knowledge"
  - "Catastrophic Forgetting"
  - "Continual Learning"
date: 2026-05-08
content_hash: 1539126a80e0aae9
---

# When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations

**Conference**: ICLR 2026
**arXiv**: [2505.24449](https://arxiv.org/abs/2505.24449)  
**Code**: None  
**Area**: Multimodal / VLM
**Keywords**: Large Multimodal Models, Knowledge Injection, Evolving Knowledge, Catastrophic Forgetting, Continual Learning

## TL;DR

This paper proposes the EVOKE benchmark to systematically evaluate the ability of Large Multimodal Models (LMMs) to incorporate evolving knowledge, identifies two core challenges (poor performance of existing methods and catastrophic forgetting induced by fine-tuning), and explores two mitigation strategies: knowledge augmentation and continual learning.

## Background & Motivation

Large Language/Multimodal Models (LLMs/LMMs) accumulate extensive world knowledge through large-scale pretraining, yet face a fundamental problem: **knowledge staleness**. Global information updates rapidly, new entities emerge continuously, while trained models remain static. For instance, an LMM may fail to recognize the Xiaomi SU7 automobile and incorrectly identify it as a Porsche.

Existing research on knowledge injection exhibits three critical gaps:

**Lack of multimodal evolving knowledge datasets**: Existing knowledge injection datasets (e.g., CC-RECENTNEWS) contain only text and lack multimodal data reflecting real-world scenarios.

**Insufficient systematic study of LMMs**: Most knowledge injection research focuses on LLMs, with notably limited systematic exploration of vision-language models.

**Underestimation of injection side effects**: The impact of knowledge injection—particularly fine-tuning—on a model's pre-existing capabilities has not been comprehensively evaluated.

This paper aims to construct the first multimodal evolving knowledge injection benchmark, systematically expose the associated challenges, and explore viable mitigation paths.

## Method

### Overall Architecture

The contribution of this paper is a complete "benchmark + evaluation + path" framework:

1. **EVOKE Benchmark Construction**: An automated pipeline for collecting evolving knowledge and constructing evaluation data.
2. **Systematic Evaluation**: Comprehensive experiments across three categories of knowledge injection methods: SFT, RAG, and IAG.
3. **Challenge Identification**: Two core challenges are identified.
4. **Path Exploration**: A corresponding mitigation direction is proposed for each challenge.

### Key Designs

1. **EVOKE Benchmark Construction**: Automated multimodal evolving knowledge collection pipeline

    - **Data Sources**: CNN news website (29 news categories) and offline Wikipedia (130 entity types), covering 159 fine-grained categories in total.
    - **Data Scale**: 9,422 knowledge–image pairs.
    - **Timeline**: Data from 2024, ensuring novelty for LMMs released in 2023.
    - **Data Format**: Each knowledge entry consists of:
        - Injection data $\mathcal{D}_\mathcal{K} = \{(i_k, x_k, y_k)\}$: knowledge image, heuristic query, knowledge summary.
        - Evaluation data $\mathcal{D}_\mathcal{Q} = \{(i_q, x_q, y_q)\}$: query image, question, ground truth.
    - **Quality Assurance**: Popularity filtering → GPT-4o summarization → GPT-4o QA generation → Google Image retrieval → CLIP clustering for denoising → human review.

2. **Knowledge Injection Method Evaluation**: Comprehensive coverage across three major categories

    - **Supervised Fine-Tuning (SFT)**: Full Fine-Tuning and LoRA strategies.
    - **Multimodal Retrieval-Augmented Generation (MM-RAG)**: Four retrieval strategies—Text-Only, Image-Only, UniIR (multimodal fusion retrieval), and Golden Context (ideal upper bound).
    - **Internet-Augmented Generation (IAG)**: Gemini and Perplexity AI.

3. **Problem Formulation**: Dual objectives of knowledge injection

    - **Knowledge Adaptation**: Maximize accuracy on evolving knowledge evaluation data.
    - **Knowledge Retention**: Minimize performance degradation on pre-existing tasks after injection.
    - Formalized as a constrained optimization: $\max_f \mathbb{E}[\mathbb{I}(\mathcal{M}^*(i_q, x_q) = y_q)] \text{ s.t. } \min_f \mathbb{E}[\mathbb{I}(\mathcal{M}(i_p, x_p) = y_p) - \mathbb{I}(\mathcal{M}^*(i_p, x_p) = y_p)]$

4. **Path 1: Knowledge Augmentation**: Data augmentation during the training stage

    - **Text Augmentation**: GPT-4 is used to paraphrase knowledge summaries, generating semantically equivalent but differently expressed versions.
    - **Image Augmentation**: Conventional augmentations (flipping, random shadow, color transformation).
    - Core finding: Text augmentation significantly improves performance at training time (accuracy increases monotonically with the number of paraphrases), whereas image augmentation leads to performance degradation.
    - Interpretation: Text augmentation helps the model learn the "correct logic" rather than rote memorization—e.g., learning that "the Xiaomi SU7 is an electric sedan from Xiaomi's automotive division" rather than memorizing a verbatim description.

5. **Path 2: Continual Learning**: Alleviating catastrophic forgetting

    - **When training data is available**: Replay—randomly sampling 10% of original training data and training jointly with new knowledge.
    - **When training data is unavailable**:
        - EWC (Elastic Weight Consolidation): parameter regularization.
        - LwF (Learning without Forgetting): knowledge distillation.
        - MoELoRA: mixture-of-experts LoRA leveraging multiple experts for diverse knowledge acquisition.
    - Overall ranking: Replay+LoRA (Rank 1) > MoELoRA (Rank 2) > Replay+Full-FT (Rank 3).

### Loss & Training

- SFT: Standard instruction fine-tuning loss (cross-entropy).
- Continual learning losses:
    - Replay: Standard loss computed on a subset of original data.
    - EWC: $\mathcal{L}_{EWC} = \mathcal{L}_{task} + \lambda \sum_i F_i (\theta_i - \theta_i^*)^2$
    - LwF: $\mathcal{L}_{LwF} = \mathcal{L}_{task} + \lambda \mathcal{L}_{KD}$
    - MoELoRA: Multi-expert routing + contrastive learning.

## Key Experimental Results

### Main Results

| Method | Overall Acc | Overall F1 | News Acc | Entity Acc |
|--------|-------------|------------|----------|------------|
| LLaVA Vanilla | 4.89 | 9.34 | 7.37 | 2.18 |
| LLaVA Full-FT | 18.02 | 15.17 | 21.35 | 14.37 |
| LLaVA LoRA | 15.23 | 18.31 | 17.72 | 12.51 |
| LLaVA MM-RAG (UniIR) | 40.68 | 57.51 | 40.12 | 41.30 |
| LLaVA MM-RAG (Golden) | **56.13** | **75.77** | 56.78 | 55.43 |
| Perplexity AI† | 48.27 | 62.44 | 47.58 | 48.96 |

The best-performing method achieves only 56.13% accuracy, far from satisfactory.

### Catastrophic Forgetting Evaluation (12 benchmarks, 7 capability dimensions)

| Method | MME | MMBench | MIA-Bench | MMDU | Ranking |
|--------|-----|---------|-----------|------|---------|
| Vanilla | 1865.56 | 64.60 | 66.33 | 26.37 | - |
| Full-FT | 956.8 (-49%) | 52.92 (-18%) | 25.25 (-62%) | 13.03 (-51%) | 7 |
| LoRA | 1233.54 (-34%) | 53.87 (-17%) | 29.66 (-55%) | 13.70 (-48%) | 6 |
| Replay+LoRA | **1650.75** (-12%) | **60.48** (-6%) | **62.33** (-6%) | **19.31** (-27%) | **1** |
| MoELoRA | 1732.47 (-7%) | 63.32 (-2%) | 64.97 (-2%) | 18.66 (-29%) | 2 |

### Key Findings

1. **Challenge 1: Extremely Poor Knowledge Injection Performance**
    - The best method (Golden Context) achieves only 56.13% accuracy.
    - SFT methods perform even worse (15–18%).
    - MM-RAG generally outperforms SFT but requires retrieval infrastructure.
    - IAG (Perplexity AI) achieves 48.27% without relying on any external injection data.

2. **Challenge 2: Severe Catastrophic Forgetting**
    - Full-FT and LoRA exhibit degradation across all 12 benchmarks.
    - **Instruction-following capability suffers the most**: MIA-Bench drops by 62%/55%, as EVOKE data contains no instruction-following scenarios.
    - Instruction-following is a prerequisite for other capabilities—its severe degradation causes disproportionate drops on MME (which relies on Yes/No instructions) compared to MMBench (multiple choice).
    - Multi-turn dialogue ability (MMDU) also declines substantially.

3. **Text Augmentation Effective; Image Augmentation Ineffective**
    - Paraphrasing during training (1→4 paraphrase variants) consistently improves performance.
    - Conventional image augmentation leads to performance degradation, suggesting the need for dedicated image knowledge augmentation methods.

4. **Continual Learning Methods Help but at a Cost**
    - Replay and MoELoRA most effectively mitigate forgetting.
    - However, all continual learning methods incur some loss in knowledge injection performance.
    - MoELoRA suffers the largest drop in knowledge injection accuracy (Acc: 15.23→6.82).

5. **Sequential Fine-Tuning Degrades Over Time**
    - Splitting data into 4/8/12 sequential batches and fine-tuning progressively results in declining performance.
    - This indicates that sequential fine-tuning is unsuitable for continuously injecting evolving knowledge.

## Highlights & Insights

1. **First Multimodal Evolving Knowledge Benchmark**: EVOKE fills the gap in multimodal knowledge injection evaluation; the data collection pipeline can continuously produce new evolving knowledge.
2. **Comprehensive Systematic Evaluation**: Spanning SFT/RAG/IAG methods, 2 LMMs, and 12 forgetting evaluation benchmarks, the experimental scale leads comparable work.
3. **Causal Chain: Instruction Following → Catastrophic Forgetting**: The paper reveals that knowledge injection causes the collapse of instruction-following capability, which in turn triggers large-scale degradation across other abilities. This causal mechanism offers important guidance for future research.
4. **"Learning the Correct Logic" Interpretation of Text Augmentation**: Models need to learn flexible knowledge extraction rather than rote memorization; paraphrasing helps models store correct associations between entity attributes. This insight can guide data preparation strategies.
5. **Knowledge-Type Adaptation Disparity**: News knowledge is easier to adapt to than entity knowledge, since news consists of new events involving existing entities, whereas new entities are entirely unfamiliar to LMMs.

## Limitations & Future Work

1. **Limited Data Scale**: While 9,422 samples is nontrivial, it remains small relative to LMM parameter counts; larger-scale data might alter the conclusions.
2. **Outdated Model Selection**: Experiments are conducted only on LLaVA-v1.5 and Qwen-VL-Chat (both 2023 models); more recent models (e.g., GPT-4V, InternVL-2) may exhibit different behaviors.
3. **Evaluation Restricted to Knowledge VQA**: EVOKE's evaluation is limited to the visual question answering format and does not cover more complex knowledge application scenarios (e.g., reasoning chains, multi-hop reasoning).
4. **Insufficient Exploration of Image Augmentation**: Although conventional augmentation is found to be ineffective, more advanced strategies (e.g., diffusion-based style transfer) are not explored.
5. **No Comparison with Knowledge Editing Methods**: Knowledge editing approaches (e.g., multimodal variants of ROME and MEMIT) are not included in the comparison.

## Related Work & Insights

- **Three Paradigms of Knowledge Injection**: SFT (parametric internalization), RAG (external retrieval), and IAG (internet search) each have distinct trade-offs; hybrid approaches may be necessary in the future.
- **Continual Learning Applied to LMMs**: The effectiveness of Replay and MoELoRA demonstrates that integrating continual learning with large model fine-tuning is a promising direction.
- **Insights from Knowledge Augmentation**: The theoretical finding of Allen-Zhu & Li (2024)—that memorizing training data does not guarantee knowledge extraction—is empirically validated in the multimodal setting.
- **Implication**: The strong performance of IAG (e.g., Perplexity AI) suggests that, for evolving knowledge, enhancing a model's internet search capability may be preferable to injecting knowledge into model parameters.

## Rating

- Novelty: ⭐⭐⭐⭐ — First multimodal evolving knowledge benchmark with a clearly defined problem formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Large-scale experiments with comprehensive coverage and in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured; the challenge–path framework is clearly articulated.
- Value: ⭐⭐⭐⭐ — The benchmark and findings provide important reference value for the community, though the evaluated models are dated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls](../../ICML2026/knowledge_editing/kore_enhancing_knowledge_injection_for_large_multimodal_models_via_knowledge-ori.md)
- [\[ACL 2026\] EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing](../../ACL2026/knowledge_editing/evoedit_evolving_null-space_alignment_for_robust_and_efficient_knowledge_editing.md)
- [\[AAAI 2026\] Hybrid-DMKG: A Hybrid Reasoning Framework over Dynamic Multimodal Knowledge Graphs for Multimodal Multihop QA with Knowledge Editing](../../AAAI2026/knowledge_editing/hybrid-dmkg_a_hybrid_reasoning_framework_over_dynamic_multimodal_knowledge_graph.md)
- [\[NeurIPS 2025\] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](../../NeurIPS2025/knowledge_editing/uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)
- [\[ICLR 2026\] Fine-tuning Done Right in Model Editing](fine-tuning_done_right_in_model_editing.md)

</div>

<!-- RELATED:END -->
