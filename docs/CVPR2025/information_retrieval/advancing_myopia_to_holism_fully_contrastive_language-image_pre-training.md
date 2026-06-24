---
title: >-
  [Paper Note] Advancing Myopia To Holism: Fully Contrastive Language-Image Pre-training
description: >-
  [CVPR 2025][Information Retrieval & RAG][CLIP] Upgrades CLIP from the traditional one-to-one (image, text) contrastive learning paradigm to a multi-to-multi (multi-image-embeddings, multi-texts) contrastive learning paradigm. By utilizing VLMs to generate multi-perspective, multi-level descriptions and a multi-branch visual encoder to output diverse visual embeddings, it achieves more comprehensive vision-language alignment, substantially outperforming baselines in retrieval…
tags:
  - "CVPR 2025"
  - "Information Retrieval & RAG"
  - "CLIP"
  - "Contrastive Learning"
  - "Multi-text Alignment"
  - "Multi-branch Visual Encoder"
  - "Fine-grained Visual Representation"
date: 2026-05-08
content_hash: fd95b2c1fdc2be29
---

# Advancing Myopia To Holism: Fully Contrastive Language-Image Pre-training

**Conference**: CVPR 2025  
**arXiv**: [2412.00440](https://arxiv.org/abs/2412.00440)  
**Code**: [https://github.com/anakin-skywalker-Joseph/Holistic-CLIP](https://github.com/anakin-skywalker-Joseph/Holistic-CLIP)  
**Area**: Information Retrieval  
**Keywords**: CLIP, Contrastive Learning, Multi-text Alignment, Multi-branch Visual Encoder, Fine-grained Visual Representation

## TL;DR

Upgrades CLIP from the traditional one-to-one (image, text) contrastive learning paradigm to a multi-to-multi (multi-image-embeddings, multi-texts) contrastive learning paradigm. By utilizing VLMs to generate multi-perspective, multi-level descriptions and a multi-branch visual encoder to output diverse visual embeddings, it achieves more comprehensive vision-language alignment, substantially outperforming baselines in retrieval, classification, and dense prediction tasks.

## Background & Motivation

**Background**: CLIP is a cornerstone of vision-language models, establishing cross-modal alignment through InfoNCE contrastive learning on (image, text) pairs. OpenAI's CLIP is trained on 400M web-scraped image-text pairs.

**Limitations of Prior Work**—The "Myopia" Dilemma of CLIP:
   - **Single Text**: Web texts have uniform formats and are mostly concise summaries, lacking description of complex relationships and even containing noise.
   - **Visual Impairment**: An image containing rich information is matched with only one compact text, discarding a substantial amount of visual diversity.
   - **Semantic Chaos**: Different semantic levels (overall summary, local details, style, emotion) are crudely aggregated into the same embedding space.

**Key Challenge**: The information volume contained in a single image far exceeds that of a single text segment; thus, a single embedding vector cannot fully represent all visual elements of an image.

**Key Insight**: Inspired by the parable of "the blind men and the elephant"—combining multiple myopic cognitions yields a holistic understanding. This work generates multi-perspective and multi-level text descriptions for each image, and designs a multi-branch visual encoder to output multiple complementary visual embeddings, achieving part-to-part multi-to-multi alignment.

## Method

### Overall Architecture

Input: Use a VLM to generate $M$ description texts with different perspectives/granularities for each image. Model: Modify the CLIP image encoder to output $M$ visual embeddings through multiple branches. Optimization: Design a multi-to-multi contrastive learning scheme to achieve part-to-part matching. During inference, different visual embeddings can be flexibly selected or combined to adapt to various downstream tasks.

### Key Designs

1. **Multi-Perspective and Multi-Level Data Construction**:

    - **Function**: Generates $M$ diverse description texts for each image, covering different perspectives, granularities, and levels.
    - **Mechanism**: Designs four "prompt spirits" — Focus Guide (foreground vs. background), Physical or Sensory (entity nouns vs. sensory style), Gaze or Glance (detailed long description vs. summary short description), and Complex Reasoning (relationships vs. sequences). An InternVL2 or similar VLM is used to generate descriptions based on the different prompts.
    - **Design Motivation**: A single VLM with multiple prompts (Equation 2) yields higher text diversity than multiple VLMs with a single prompt (Equation 1) (similarity 0.48 vs. 0.58) and is simpler to deploy.

2. **Multi-Branch Visual Encoder**:

    - **Function**: Modifies the CLIP image encoder to output $M$ different visual embeddings.
    - **Mechanism**: Proposes two parameter-efficient schemes — (a) initializing $M$ CLS tokens, each outputting a corresponding embedding; (b) extending the MLP of the last few layers into $M$ parallel units. A single forward pass yields $M$ visual embeddings.
    - **Design Motivation**: A single embedding vector cannot adequately represent the diverse visual elements (color, objects, style, events, etc.) inside an image. Multi-branch outputs solve the "visual impairment" issue, and each embedding can have its own semantic role.

3. **Multi-to-Multi Contrastive Learning (M2M)**:

    - **Function**: Achieves precise part-to-part matching between $M$ visual embeddings and $M$ text embeddings.
    - **Mechanism**: First establishes the correspondence between visual-text embedding pairs through optimal matching (the Hungarian algorithm or greedy matching), and then applies the standard InfoNCE contrastive loss to each pair:
      $$\mathcal{L}_{M2M}^{T2I} = -\sum_{j=1}^K \sum_{m=1}^M \log \frac{\exp(\langle \mathbf{v}_{m,j}, \mathbf{t}_{\sigma(m),j} \rangle / \tau)}{\sum_{k=1}^K \exp(\langle \mathbf{v}_{m,k}, \mathbf{t}_{\sigma(m),j} \rangle / \tau)}$$
    - **Design Motivation**: Performs better than one-to-multi (O2M) contrastive learning—O2M pulls $M$ semantically disparate texts toward the exact same visual embedding, leading to semantic chaos. M2M aligns each visual branch with its best-matching text, resulting in cleaner semantic decoupling.

### Loss & Training

Trained on CC3M and CC12M (3M and 12M image-text pairs respectively). InternVL2 is used to generate multi-angle descriptions. During training, the number of texts is set to $M=5$ (4 VLM descriptions + 1 original web text).

## Key Experimental Results

### Main Results: Image-Text Retrieval (CC3M Training)

| Method | MSCOCO I2T R@1 | MSCOCO T2I R@1 | Flickr30K I2T R@1 | Flickr30K T2I R@1 |
|------|:-:|:-:|:-:|:-:|
| CLIP | 13.6 | 13.4 | 30.8 | 31.9 |
| O2M (5 texts, multi-prompt) | 24.5 | 26.3 | 60.7 | 61.7 |
| M2M $\Psi_{CLS}$ (Ours, multi-prompt) | 28.0 | 27.8 | 62.9 | 64.2 |
| M2M $\Psi_{MLP}$ (Ours, multi-prompt) | **28.2** | 27.4 | **63.7** | 63.9 |
| M2M $\Psi_{MLP}$ (multi-VLM) | **31.2** | **30.7** | **66.5** | **66.4** |

### Ablation Study: Alignment Strategy Comparison

| Method | Alignment Strategy | MSCOCO R@1 | Flickr R@1 |
|------|---------|:-:|:-:|
| CLIP (O2O) | 1-to-1 | 13.6 | 30.8 |
| O2M | 1 Image-to-M Text | 24.5 | 60.7 |
| **M2M (Ours)** | **M Image-to-M Text** | **28.2** | **63.7** |

### Key Findings

- **M2M vs. O2M**: On MSCOCO I2T R@1, M2M improves by 15% over O2M (28.2 vs. 24.5), showing that part-to-part matching is superior to forcing multiple texts toward the same embedding.
- **vs. Original CLIP**: R@1 on Flickr30K improves from 30.8 to 66.5 (+116%), presenting an extremely significant improvement.
- **Multi-VLM > Multi-Prompt**: Using multiple VLMs to generate text achieves the best results on the CC3M dataset (31.2 vs. 28.2), but comes with higher deployment costs.
- **Two Multi-Branch Schemes**: $\Psi_{MLP}$ performs slightly better than $\Psi_{CLS}$, but the difference is marginal, showing that the core value of multi-branching lies in "having multiple branches" rather than the specific implementation.

## Highlights & Insights

- **Apt "blind men and the elephant" analogy**: Likening CLIP's limitation to "myopia" and the solution to "combining multiple perspectives" is clear and compelling. This concept can be transferred to other tasks requiring multi-angle understanding.
- **Full-stack upgrade of data, model, and optimization**: Rather than modifying only the data or only the model, all three components are upgraded collaboratively, demonstrating strong systematic integration.
- **Inference flexibility**: The multiple embeddings produced by multi-branch outputs can be selected and combined as needed during inference—using a global embedding for coarse-grained classification and multi-embedding fusion for fine-grained retrieval, offering high practical value.
- **Semantic interpretability**: The embeddings from different branches naturally form semantic decompositions (e.g., one capturing objects, one capturing style, one capturing spatial relationships), enhancing model interpretability.

## Limitations & Future Work

- Generating VLM captions incurs additional computational overhead (especially for the multi-VLM scheme), which is costly on large-scale datasets.
- The semantic roles of the $M$ branches are learned implicitly without explicit constraints to guarantee specialization of each branch.
- Validation is only conducted on CC3M/CC12M (3M/12M scale); experiments on larger-scale datasets (e.g., LAION-2B) are yet to be conducted.
- The alignment strategy (Hungarian algorithm vs. greedy) in part-to-part matching might be a performance bottleneck.

## Related Work & Insights

- **vs. SigLIP / FLIP / FILIP**: These works improve CLIP's contrastive loss or attention mechanism, but still adhere to the one-to-one paradigm, failing to fundamentally solve the problem of visual information loss.
- **vs. LaCLIP / DreamLIP**: They also use VLMs for re-captioning, but still rely on O2M contrastive learning. The proposed M2M matching is more precise.
- **Inspirations from the multi-branch approach**: Similar to the concept of mixture-of-experts—multiple experts handle different aspects respectively, and combining them outperforms a single model.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Fully upgrades the CLIP paradigm across data, model, and optimization dimensions; the M2M contrastive learning design is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on 10+ datasets with detailed ablations ($M$ value, prompt type, encoder schemes), but lacks validation on large-scale data.
- **Writing Quality**: ⭐⭐⭐⭐ Vivid analogy of "blind men and the elephant", but some mathematical notations are relatively complex.
- **Value**: ⭐⭐⭐⭐ Pointed out a clear direction for subsequent improvements to CLIP; the multi-branch + M2M paradigm holds potential for broad adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LangBridge: Interpreting Image as a Combination of Language Embeddings](../../ICCV2025/information_retrieval/langbridge_interpreting_image_as_a_combination_of_language_embeddings.md)
- [\[ICLR 2026\] Reusing Pre-training Data at Test Time is a Compute Multiplier](../../ICLR2026/information_retrieval/reusing_pre-training_data_at_test_time_is_a_compute_multiplier.md)
- [\[ACL 2025\] Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience](../../ACL2025/information_retrieval/toward_structured_knowledge_reasoning_contrastive_retrieval-augmented_generation.md)
- [\[ICLR 2026\] Supervised Fine-Tuning or Contrastive Learning? Towards Better Multimodal LLM Reranking](../../ICLR2026/information_retrieval/supervised_fine-tuning_or_contrastive_learning_towards_better_multimodal_llm_rer.md)
- [\[ICML 2025\] Don't Lag, RAG: Training-Free Adversarial Detection Using RAG](../../ICML2025/information_retrieval/dont_lag_rag_training-free_adversarial_detection_using_rag.md)

</div>

<!-- RELATED:END -->
