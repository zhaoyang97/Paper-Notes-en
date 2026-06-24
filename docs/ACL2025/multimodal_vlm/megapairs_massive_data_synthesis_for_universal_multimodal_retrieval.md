---
title: >-
  [Paper Note] MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval
description: >-
  [ACL 2025][Multimodal VLM][Multimodal Retrieval] This paper proposes the MegaPairs data synthesis method, which leverages heterogeneous KNN triplets to mine matching image pairs from an open-domain image corpus, combined with VLMs/LLMs to generate retrieval instructions, synthesizing 26 million multimodal training instances. The resulting MMRet model, trained on only 0.5M data, outperforms MagicLens which uses 36.7M data (achieving $70\times$ data efficiency)…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Multimodal Retrieval"
  - "Data Synthesis"
  - "Composed Image Retrieval"
  - "Contrastive Learning"
  - "instruction tuning"
  - "Hard Negatives"
date: 2026-05-08
content_hash: 2dbd3b3b28a3abc8
---

# MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval

**Conference**: ACL 2025  
**arXiv**: [2412.14475](https://arxiv.org/abs/2412.14475)  
**Code**: To be released (complete dataset, model, and pipeline will be fully open-sourced)  
**Area**: Multimodal VLM / Multimodal Retrieval  
**Keywords**: Multimodal Retrieval, Data Synthesis, Composed Image Retrieval, Contrastive Learning, instruction tuning, Hard Negatives

## TL;DR

This paper proposes the MegaPairs data synthesis method, which leverages heterogeneous KNN triplets to mine matching image pairs from an open-domain image corpus, combined with VLMs/LLMs to generate retrieval instructions, synthesizing 26 million multimodal training instances. The resulting MMRet model, trained on only 0.5M data, outperforms MagicLens which uses 36.7M data (achieving $70\times$ data efficiency), and achieves SOTA on 4 CIR benchmarks and 36 MMEB datasets.

## Background & Motivation

**Background**: The demand for multimodal retrieval is rapidly increasing (e.g., image search, VQA, RAG). Methods based on pre-trained vision-language models (CLIP/ALIGN/SigLIP) have established fundamental capabilities but are limited to text-to-image matching, falling short of handling more complex multimodal tasks.

**Limitations of Prior Work**: Instruction tuning can enhance multitask capability, but instruction data for multimodal retrieval is extremely scarce. Existing methods (e.g., MagicLens) synthesize data from image pairs co-occurring on the same webpage, which faces four key limitations:
   - **Poor scalability**: Only a small portion of webpages contain multiple images.
   - **Low quality**: Co-occurring images are often irrelevant or highly redundant.
   - **Insufficient diversity**: Relationships between co-occurring images are monotonic (e.g., different angles of the same object).
   - **Poor accessibility**: Large-scale datasets are usually proprietary and private.

**Core Idea**: Leveraging an open-domain image corpus (e.g., DataComp) + multiple similarity models to mine heterogeneous image pairs + open-source VLMs/LLMs to generate retrieval instructions $\rightarrow$ to construct large-scale, high-quality, diverse, and publicly available multimodal retrieval training data.

## Method

### Overall Architecture

MegaPairs consists of two stages: (1) mining relevant image pairs using three similarity models to discover heterogeneous correlations; (2) generating open-ended retrieval instructions, where a VLM describes the relationship and an LLM refines it into instructions.

### Stage 1: Mining Relevant Image Pairs

- **Three similarity models introducing heterogeneous correlations**:
  1. **Vision-Semantic Correlation** (EVA-CLIP Image Encoder): Captures semantically relevant but visually distinct image pairs (e.g., different angles of the same car).
  2. **Vision-Pattern Correlation** (DINOv2): Captures visually similar but potentially semantically distinct image pairs (e.g., different cars in similar backgrounds).
  3. **Caption Correlation** (EVA-CLIP Text Encoder): Based on the textual similarity of image captions.
- **Filtering Strategy**: Image pairs with similarity in the range of $(0.8, 0.96)$ are retained to exclude weak correlations and near-duplicates.
- **Hard Negatives**: For each image pair (I_q, I_ti), other images in the retrieval set {I_tj | tj ≠ ti} naturally act as hard negatives, introducing 5 hard negatives for each pair.

### Stage 2: Generating Open-Ended Instructions

- **Step 1**: A VLM (InternVL2-26B) takes the image pair as input and generates a detailed description D_i of the common concepts and differences between the two images.
- **Step 2**: An LLM (LLaMA3-8B) refines the description into a retrieval instruction T_{q→ti}, generating at least 3 different instructions per pair to increase diversity.
- **Final Triplet**: (I_q, T_{q→ti}, I_ti), where (I_q, T_{q→ti}) is used to retrieve I_ti.

### Implementation Scale

- Image Corpus: A subset of 20 million captioned images from Recap-DataComp-1B.
- Synthesized Results: 26,235,105 image pairs, fully open-sourced.

### MMRet Model

Two architectures are designed:

1. **CLIP-based MMRet** (Base/Large):
    - A dual-encoder architecture that independently encodes images and texts.
    - Multimodal embeddings are fused using score-fusion: e_it = Φ_I(I) + Φ_T(T).

2. **MLLM-based MMRet** (based on LLaVA-1.6 Mistral 7B):
    - Image tokens are fed directly into the LLM for processing.
    - Uses task-specific instructions: `<instruct> {task_inst} <query> {q_t} {q_i} [EOS]`.
    - Uses the normalized final hidden state of the `[EOS]` token as the embedding.

### Training Objectives

Standard InfoNCE contrastive loss:
$$\mathcal{L} = -\frac{1}{|\mathcal{Q}|}\sum_{q_i \in \mathcal{Q}} \log \frac{\exp(\mathbf{e}_{q_i} \cdot \mathbf{e}_{c_i^+}/\tau)}{\sum_{c_j \in \mathcal{C}} \exp(\mathbf{e}_{q_i} \cdot \mathbf{e}_{c_j}/\tau)}$$
The temperature parameter $\tau = 0.02$. The query and candidate can be images, texts, or image-text combinations.

## Experiments

### Zero-Shot CIR Performance (Table 1)

| Method | Backbone | Params | CIRCO mAP@5 | CIRR R@1 | FashionIQ R@10 | GeneCIS Rs@1 |
|------|----------|--------|-------------|----------|----------------|--------------|
| MagicLens-L‡ | CoCa-L | 613M | 34.1 | 33.3 | 38.0 | 16.7 |
| IP-CIR | CLIP-G | 43.8B† | 32.8 | 39.3 | 45.7 | - |
| MMRet-Base | CLIP-B | 149M | 34.3 | 36.1 | 31.9 | 18.0 |
| MMRet-Large | CLIP-L | 428M | 39.2 | 38.0 | 34.6 | 18.1 |
| **MMRet-MLLM** | LLaVA-1.6 | 7.57B | **42.2** | **46.7** | **35.6** | **21.1** |

**Key Findings**:
- MMRet-MLLM outperforms the previous SOTA on CIRCO by 8.1 percentage points (42.2 vs 34.1).
- MMRet-Base (149M) even outperforms most large models (including MagicLens-L 613M).
- All MMRet models lead across all scales.

### MMEB Zero-Shot Performance (Table 2)

| Model | Classification | VQA | Retrieval | Grounding | Overall |
|------|---------------|-----|-----------|-----------|---------|
| UniIR | 42.1 | 15.0 | 60.1† | 62.2 | 42.8 |
| MMRet-MLLM | **47.2** | **18.4** | **56.5** | **62.2** | **44.0** |

In the comprehensive evaluation across 36 datasets, MMRet-MLLM achieves the highest overall score of 44.0, despite UniIR's retrieval meta-task containing 10 out of 12 MMEB retrieval datasets (which is not strictly zero-shot).

### MMEB Fine-Tuning Performance (Table 3)

| Model | IND | OOD | Overall |
|------|-----|-----|---------|
| VLM2Vec (LLaVA-1.6) | 61.0 | 47.5 | 55.0 |
| VLM2Vec (Phi-3.5-V) | 66.5 | 52.0 | 60.1 |
| **MMRet-MLLM** | **68.0** | **59.1** | **64.1** |

After fine-tuning, the OOD performance is improved by 11.6% (vs. LLaVA-1.6 baseline), demonstrating the strong generalization ability endowed by MegaPairs pre-training.

### Data Quality and Scalability (Figure 2)

- **$70\times$ Data Efficiency**: Only 0.5M MegaPairs samples are needed to outperform the model trained on 36.7M MagicLens data.
- **Continuous Scaling**: Performance continuously scales up as the data volume grows, with no saturation observed.

### Ablation Study

1. **Effect of Hard Negatives (Table 4)**: Using mined hard negatives improves performance on CIRCO by 2.6 percentage points (29.7 → 32.3).
2. **Search Strategy (Table 5)**: Jointly using the three similarity models yields the best performance (32.3 mAP@5). Individually, using text similarity outperforms using visual similarity.

## Highlights & Insights

1. **Innovative Data Synthesis Paradigm**: Transitioning from "relying on co-occurring webpage images" to "actively mining from open-domain corpora" thoroughly addresses the scalability bottleneck.
2. **Heterogeneous Correlations are Key**: The diverse correlations introduced by the three different similarity models are the core guarantee of data quality—relationships from a single model are too monotonous.
3. **Extreme Data Efficiency**: 0.5M data beating 36.7M demonstrates that data quality is far more important than quantity, which is a classic insight in the retrieval domain.
4. **Fully Open-Source**: The complete suite of the dataset, model, and pipeline is open-sourced, providing a significant contribution to the community.
5. **Strong Performance of Small Models**: MMRet-Base (149M) outperforms most large models, proving that correct training data can compensate for gaps in model scale.

## Limitations & Future Work

1. Only three retrievers are used to construct image pairs; more retrieval strategies (e.g., BGE, image-text cross-retrieval) might further enhance diversity.
2. Although the image sources have been filtered by the Datacomp team, they may not be entirely clean.
3. CLIP-based MMRet underperforms specialized CIR models in domain-specific verticals such as FashionIQ.

## Related Work & Insights

- **Multimodal Retrieval**: Pre-trained models like CLIP, ALIGN, and SigLIP; general multimodal embeddings like UniIR and E5-V.
- **Instruction Tuning**: From LLMs (FLAN, InstructGPT) to embeddings (Su et al., GTE), and to multimodal domains (MagicLens, UniIR).
- **CIR Methods**: Zero-shot or training-based methods such as SEARLE, CIReVL, LDRE, and CompoDiff.

## Rating ⭐⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐⭐ The data synthesis paradigm combining heterogeneous KNN triplets and VLM/LLM annotations is novel and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated across 4 CIR benchmarks and 36 MMEB datasets, featuring zero-shot/fine-tuning setups, data scaling, and ablation studies.
- **Value**: ⭐⭐⭐⭐⭐ Fully open-sourced, 26 million data points are readily accessible, addressing the pain points of the field.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly described method with rich illustrations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalized Contrastive Learning for Universal Multimodal Retrieval](../../NeurIPS2025/multimodal_vlm/generalized_contrastive_learning_for_universal_multimodal_re.md)
- [\[ICML 2025\] Universal Retrieval for Multimodal Trajectory Modeling](../../ICML2025/multimodal_vlm/universal_retrieval_for_multimodal_trajectory_modeling.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](../../ICCV2025/multimodal_vlm/effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)
- [\[CVPR 2025\] Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models](../../CVPR2025/multimodal_vlm/img-diff_contrastive_data_synthesis_for_multimodal_large_language_models.md)
- [\[ICLR 2026\] U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning](../../ICLR2026/multimodal_vlm/u-marvel_unveiling_key_factors_for_universal_multimodal_retrieval_via_embedding_.md)

</div>

<!-- RELATED:END -->
