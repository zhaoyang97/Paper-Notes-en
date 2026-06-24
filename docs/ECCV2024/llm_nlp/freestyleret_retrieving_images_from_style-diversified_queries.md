---
title: >-
  [Paper Note] FreestyleRet: Retrieving Images from Style-Diversified Queries
description: >-
  [ECCV 2024][LLM (Other)][Style-diversified retrieval] This work proposes the first Style-Diversified Query-Based Image Retrieval (Style-Diversified QBIR) task and the DSR dataset. It designs FreestyleRet, a lightweight, plug-and-play framework that extracts texture/style features of queries using Gram matrices to construct a style space. These style features then initialize prompt tokens, enabling a frozen vision encoder to adapt to various query styles such as texts…
tags:
  - "ECCV 2024"
  - "LLM (Other)"
  - "Style-diversified retrieval"
  - "Gram matrix"
  - "prompt tuning"
  - "cross-style image retrieval"
  - "plug-and-play framework"
date: 2026-05-08
content_hash: 600fc7a9de02c864
---

# FreestyleRet: Retrieving Images from Style-Diversified Queries

**Conference**: ECCV 2024  
**arXiv**: [2312.02428](https://arxiv.org/abs/2312.02428)  
**Code**: [https://github.com/CuriseJia/FreeStyleRet](https://github.com/CuriseJia/FreeStyleRet)  
**Area**: Model Compression (Lightweight Retrieval)  
**Keywords**: Style-diversified retrieval, Gram matrix, prompt tuning, cross-style image retrieval, plug-and-play framework

## TL;DR

This work proposes the first Style-Diversified Query-Based Image Retrieval (Style-Diversified QBIR) task and the DSR dataset. It designs FreestyleRet, a lightweight, plug-and-play framework that extracts texture/style features of queries using Gram matrices to construct a style space. These style features then initialize prompt tokens, enabling a frozen vision encoder to adapt to various query styles such as texts, sketches, low-resolution images, and artistic paintings.

## Background & Motivation

Query-Based Image Retrieval (QBIR) is a fundamental task in computer vision, widely applied in image search engines and cross-modal tasks. However, **current retrieval research almost exclusively focuses on text-to-image retrieval**, overlooking the fact that users in real-world scenarios may use diverse query modalities to express their search intentions.

**Limitations of Prior Work**:

**Single Query Style**: Mainstream retrieval models (e.g., CLIP, BLIP) are primarily designed for text queries, lacking adaptation capabilities for non-text queries such as sketches, artistic paintings (art), and low-resolution screenshots (low-res).

**User Intention Gap**: Different users prefer different query styles—some may draw a sketch to describe shapes, some take low-resolution captures, others use text—rendering single-query modalities inadequate for expressing complex retrieval intents.

**Inability to Cooperate Multi-styles**: Existing models cannot process multi-style query inputs simultaneously, let alone leverage complementary information across styles to improve retrieval performance.

**Key Challenge**: How to make a single retrieval framework **simultaneously** understand and adapt to various query styles with minimal computational overhead for style-aware retrieval?

**Key Insight**: Borrowing the Gram matrix from image style transfer to capture texture features of different query styles, constructing a style space via K-Means clustering, and then initializing prompt tokens with style features to achieve style-aware retrieval with minimal parameter overhead on a frozen pre-trained vision encoder.

## Method

### Overall Architecture

FreestyleRet contains three core modules: (1) Gram-based style extraction module, (2) style space construction module, and (3) style-initialized prompt tuning module. During training, the dataset is traversed twice: the first pass constructs the style space, and the second pass executes style-initialized prompt tuning. During inference, the constructed style space is directly utilized for style-aware retrieval.

### Key Designs

1. **Gram-based Style Extraction Module**: For any query input $q_i$, a frozen VGG network extracts the third convolutional layer features ($112 \times 112 \times 128$). After downsampling, the Gram matrix is computed:

$$g_i = (f_d(v_i))^\mathsf{T} f_d(v_i)$$

The Gram matrix $g_i$ captures the texture features and inter-channel spatial correlations of the query, effectively distinguishing sketch lines, art brushstrokes, low-resolution blur, etc.

**Design Motivation**: The Gram matrix has been proven effective in representing style/texture in style transfer. Introducing it to retrieval tasks naturally distinguishes different query styles without requiring manual style labels.

2. **Style Space Construction Module**: The Gram matrices of all queries are clustered into 4 clusters (corresponding to 4 query styles) using the **K-Means algorithm**. The cluster centers $\mu_1, ..., \mu_4$ serve as the basis vectors of the style space. For a newly input query $q_i$, its style feature is calculated via weighted summation:

$$w_j = \frac{e^{\cos(q_i, \mu_j)}}{\sum_{j=1}^{4} e^{\cos(q_i, \mu_j)}}, \quad s_i = \sum_{j=1}^{4} w_j \mu_j$$

**Design Motivation**: Explicitly constructing a style space allows the model to adaptively comprehend the style characteristics of any query, rather than hardcoding style categories. The soft-weighted combination path allows handling queries that fall between two styles.

3. **Style-Init Prompt Tuning Module**: Four learnable prompt tokens are inserted in each layer of the frozen ViT vision encoder. The key novelty lies in the token **initialization strategy**: shallow layers are initialized with the style space feature $s_i$, while deep layers are initialized with the Gram matrix $g_i$ (instead of random initialization), enabling the encoder to be style-aware from the beginning:

$$[x_i, \_, E_i] = L_i(x_{i-1}, P_{i-1}, E_{i-1}), \quad i=1,...,n$$

**Design Motivation**: VPT showed that deep prompts outperform shallow prompts, but random initialization cannot inject style information. Layer-specific differentiated initialization—shallow layers injecting global style, deep layers injecting fine-grained texture—enables the encoder to perceive style at different abstraction levels.

### Loss & Training

The Triplet Loss is utilized as the training objective:

$$\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B}\max(0, \text{dist}(F_i, P_i) - \text{dist}(F_i, N_i) + \alpha)$$

Where $\text{dist}(x,y) = 1 - \cos(x,y)$, and $\alpha=1.0$. Positive samples are ground-truth retrieval images, and negative samples are randomly selected images from the same style set. Training is done for 20 epochs with a learning rate of 1e-5 and a batch size of 24 on an A100 GPU.

## Key Experimental Results

### Main Results: Style-diversified Retrieval on DSR Dataset

| Method | Text R@1 | Sketch R@1 | Art R@1 | Low-Res R@1 | Note |
|------|----------|------------|---------|-------------|------|
| CLIP (zero-shot) | 66.1 | 47.5 | 58.5 | 45.0 | No style adaptation |
| CLIP* (prompt tuned) | 72.2 | 63.6 | 58.2 | 78.8 | Standard prompt tuning |
| BLIP* (prompt tuned) | 74.3 | 67.1 | 51.1 | 77.2 | Standard prompt tuning |
| VPT* | 69.9 | 73.3 | 66.7 | 81.4 | Visual prompt tuning |
| ImageBind | 71.0 | 50.8 | 58.2 | 79.0 | Multi-modal model |
| LanguageBind | 79.7 | 63.6 | 67.5 | 78.6 | Multi-modal model |
| **FreestyleRet-CLIP** | 69.9 | **80.6** | **71.4** | **86.4** | Ours (CLIP encoder) |
| **FreestyleRet-BLIP** | **81.6** | **81.2** | **74.5** | **90.5** | Ours (BLIP encoder) |

FreestyleRet-BLIP achieves the best R@1 across all 4 query styles, with Sketch improved by 14.1% (vs BLIP*) and Art improved by 23.4%.

### Ablation Study: Prompt Token Design

| Configuration | Sketch R@1 | Art R@1 | Low-Res R@1 | Note |
|------|-----------|---------|-------------|------|
| Shallow Random + Deep Random | 68.1 | 63.5 | 78.8 | Randomly initialized baseline |
| Shallow StyleSpace + Deep Random | 76.7 | 69.1 | 82.4 | Shallow style injection |
| Shallow Random + Deep Gram | 76.8 | 69.2 | 81.8 | Deep texture injection |
| Shallow Gram + Deep StyleSpace | 78.1 | 69.5 | 84.4 | Reversed configuration |
| **Shallow StyleSpace + Deep Gram** | **80.6** | **71.4** | **86.4** | Best configuration |
| Best configuration, 1 token | 68.2 | 64.7 | 79.1 | Insufficient tokens |
| Best configuration, 2 tokens | 72.3 | 65.9 | 82.8 | Performance increases |
| Best configuration, 8 tokens | 77.9 | 67.1 | 80.7 | Too many tokens cause degradation |

### Computational Efficiency Comparison

| Method | Parameters | Inference Speed |
|------|--------|----------|
| CLIP | 427M | 68ms |
| ImageBind | 1200M | 372ms |
| FreestyleRet-CLIP | 476M (+29M) | 96ms (+28ms) |
| FreestyleRet-BLIP | 940M (+29M) | 101ms (+39ms) |

Supporting multi-style retrieval with only 29M additional parameters and about 30ms extra inference time.

### Key Findings

- **Mutual Benefit of Multi-style Queries**: In FreestyleRet, the Text R@1 of sketch+text joint query increases from 69.9% to 82.5% (+12.6%), whereas joint querying under CLIP/BLIP deteriorates performance.
- **Convergence in 5-10 Epochs**: FreestyleRet converges significantly faster than baseline models (which require 50+ epochs), taking only 4 minutes per epoch.
- **Style Space Initialization Outperforms Random Initialization**: Ablation study shows that the best configuration (Shallow StyleSpace + Deep Gram) improves Sketch retrieval by 12.5% compared to random initialization.

## Highlights & Insights

- **New Task Definition**: Systematically defines the style-diversified QBIR task for the first time and proposes two evaluation datasets, DSR and ImageNet-X, laying a foundation for this research vector.
- **Cross-domain Application of Gram Matrix**: Intuitively brings the texture representation tool from style transfer into retrieval tasks in an elegant manner.
- **Plug-and-play Design**: Freezing encoders and only training prompt tokens allows the framework to seamlessly adapt to any ViT-based encoder, such as CLIP and BLIP.
- **Multi-query Synergy**: Queries of different styles can reinforce rather than interfere with each other, which is a bottleneck for current multi-modal models.

## Limitations & Future Work

- **Limited Query Styles**: Currently only four styles—text, sketch, art, and low-res—are studied, leaving out other modalities such as 3D, video, and audio.
- **Small Scale of DSR Dataset**: The dataset contains only 10,000 images, and performance under large-scale scenarios remains to be validated.
- **Underutilized Text Queries**: The text branch directly employs the CLIP text encoder without style-aware adaptation like the visual branch.
- **Fixed Number of Styles (K=4) in K-Means**: Introducing a new query style requires re-clustering, highlighting a limitation in flexibility.

## Related Work & Insights

- **VPT (Visual Prompt Tuning)**: First introduced prompt tuning to vision models; this work innovates style-initialization strategies on top of it.
- **CoOP / CoCoOP**: Classic works on text prompt learning, inspiring the design of learnable prompts.
- **Gram Matrix (Gatys et al.)**: The core representation tool in style transfer, creatively utilized here for style space construction.
- **ImageBind / LanguageBind**: Multi-modal unified models, which lack fine-grained modeling of style discrepancies.
- **Insight**: The prompt initialization strategy can be extended to other downstream tasks requiring conditional adaptation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ A pioneering combination of a new task, a new dataset, and Gram-matrix-initialized prompts.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation with two datasets, multiple baseline comparisons, detailed ablation studies, and qualitative analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Elegant illustrations, clear motivation, and logically structured progression across the three modules.
- **Value**: ⭐⭐⭐⭐ Lays out a new task and provides robust baselines, promising a long-term impact on the retrieval community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Style Alignment in Cross-Cultural Translation](../../ACL2025/llm_nlp/towards_style_alignment_in_cross-cultural_translation.md)
- [\[ACL 2025\] Substance over Style: Evaluating Proactive Conversational Coaching Agents](../../ACL2025/llm_nlp/proactive_conversational_coaching.md)
- [\[ACL 2025\] Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education](../../ACL2025/llm_nlp/uni-retrieval_a_multi-style_retrieval_framework_for_stems_education.md)
- [\[ACL 2025\] Investigating Context-Faithfulness in Large Language Models: The Roles of Memory Strength and Evidence Style](../../ACL2025/llm_nlp/investigating_context-faithfulness_in_large_language_models_the_roles_of_memory_.md)
- [\[ACL 2025\] Stress-testing Machine Generated Text Detection: Shifting Language Models Writing Style to Fool Detectors](../../ACL2025/llm_nlp/stress-testing_machine_generated_text_detection_shifting_language_models_writing.md)

</div>

<!-- RELATED:END -->
