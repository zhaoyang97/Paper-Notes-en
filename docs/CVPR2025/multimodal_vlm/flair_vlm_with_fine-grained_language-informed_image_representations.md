---
title: >-
  [Paper Note] FLAIR: VLM with Fine-grained Language-informed Image Representations
description: >-
  [CVPR 2025][Multimodal VLM][Fine-grained alignment] This work proposes text-conditioned attention pooling, which uses text embeddings as queries to adaptively aggregate relevant visual information from local image tokens. Trained on only 30M synthetic caption data, it significantly outperforms SigLIP/OpenCLIP trained on billions of data in fine-grained retrieval and zero-shot segmentation.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Fine-grained alignment"
  - "text-conditioned attention pooling"
  - "CLIP improvement"
  - "semantic segmentation"
  - "image-text retrieval"
date: 2026-05-08
content_hash: a5e7e0a23d6f65c3
---

# FLAIR: VLM with Fine-grained Language-informed Image Representations

**Conference**: CVPR 2025  
**arXiv**: [2412.03561](https://arxiv.org/abs/2412.03561)  
**Code**: [https://github.com/ExplainableML/flair](https://github.com/ExplainableML/flair)  
**Area**: Multimodal VLM  
**Keywords**: Fine-grained alignment, text-conditioned attention pooling, CLIP improvement, semantic segmentation, image-text retrieval

## TL;DR
This work proposes text-conditioned attention pooling, which uses text embeddings as queries to adaptively aggregate relevant visual information from local image tokens. Trained on only 30M synthetic caption data, it significantly outperforms SigLIP/OpenCLIP trained on billions of data in fine-grained retrieval and zero-shot segmentation.

## Background & Motivation

**Background**: CLIP and its successors learn visual representations through global image-text alignment, achieving great success in tasks like classification and retrieval. However, they compress the entire image into a single global vector, losing fine-grained information in local regions.

**Limitations of Prior Work**: When text describes a specific region in an image (e.g., "Starbucks cup in the background" vs. "laptop in the foreground"), the global representation of CLIP cannot distinguish these local semantic differences. This leads to poor performance on fine-grained retrieval and dense prediction tasks (such as semantic segmentation). DreamLIP attempts to solve this problem using synthetic long captions and local tokens. However, its negative sample design contains a "shortcut" where the model can accomplish the task by comparing two text embeddings without looking at the image, resulting in local tokens failing to learn meaningful semantic correspondences (zero-shot segmentation mIoU is only 0.7).

**Key Challenge**: The contradiction between global and local alignment: global vectors cannot express local details, whereas directly performing alignment with local tokens easily leads to shortcut learning.

**Goal**: How to generate different visual representations for the same image based on different text queries, thereby achieving "look at what is asked" fine-grained image-text alignment.

**Key Insight**: Utilizing text embeddings as queries for cross-attention to aggregate local image tokens, so that different image representations are generated under each text condition—focusing on background regions when describing the background, and on foreground regions when describing the foreground.

**Core Idea**: Employing text-conditioned attention pooling to generate text-adaptive image representations. Combined with a meticulously designed negative sampling strategy and diverse sub-caption sampling, this achieves fine-grained alignment quality on 30M data that surpasses models trained on billion-scale datasets.

## Method

### Overall Architecture
A dual-encoder architecture + text-conditioned attention pooling layer. The image is encoded into local tokens $\mathbf{v}^{loc}$ and a global token $\mathbf{v}^g$ by ViT-B/16; the text is encoded into a global embedding $\mathbf{t}^g$ by a Transformer. The core innovation lies in inserting a cross-attention layer between them: using $\mathbf{t}^g$ as the query and $\mathbf{v}^{loc}$ as key/value to generate the text-conditioned image representation $\mathbf{v}^{tc}$, followed by contrastive learning to align $\mathbf{v}^{tc}$ and $\mathbf{t}^g$.

### Key Designs

1. **Text-Conditioned Attention Pooling**:

    - **Function**: Generating different image representations based on text semantics.
    - **Mechanism**: Standard multi-head cross-attention, $\mathbf{v}^{tc} = \text{softmax}(\frac{\mathbf{t}^g W_q (\mathbf{v}^{loc} W_k)^T}{\sqrt{d}}) \mathbf{v}^{loc} W_v$. Key detail: appending an empty null token (zero vector) to $\mathbf{v}^{loc}$, allowing attention to "attend to nothing" when the text and image semantics are unrelated—preventing all queries from being forced to focus on a particular image region.
    - **Design Motivation**: CLIP's global pooling produces the same image representation for all texts, whereas cross-attention generates adaptive representations conditioned on text, analogous to "telling the image what to look for." The null token avoids spurious alignments between unrelated text pairs.

2. **Negative Sample Design (The Key to Preventing Shortcuts)**:

    - **Function**: Ensuring the model truly learns image content rather than purely text-to-text similarities.
    - **Mechanism**: Negative sample pairs are $\langle \mathbf{v}^{tc}_{i,j}, \mathbf{t}^g_j \rangle$—pairing the representation of image $i$ under the condition of caption $j$ with caption $j$, instead of DreamLIP's $\langle \mathbf{v}^{tc}_{i,j}, \mathbf{t}^g_i \rangle$. The problem with the latter is that the model can learn to compare the text condition (from $j$) and the target text (from $i$) to judge similarity based on text semantics alone, completely ignoring the image.
    - **Design Motivation**: DreamLIP's negative sample design leads to a zero-shot segmentation mIoU of nearly zero (0.7), indicating that its local tokens failed to learn spatial semantics. FLAIR's design forces the model to "look at the image" to determine a match.

3. **Diverse Caption Sampling**:

    - **Function**: Generating sub-captions covering different granularities from local to global from a long synthetic caption.
    - **Mechanism**: Each image has a long caption generated by an MLLM. $K=8$ sub-captions are sampled, each containing $s \in \{1,...,3\}$ sentences. Short sub-captions (1 sentence) usually describe local regions, while long sub-captions (2-3 sentences) describe more global content. The sampling method includes combinations of sequential sentences and randomly skipped sentences.
    - **Design Motivation**: Fixed-length sampling leads to distribution bias—only short captions miss global information, while only long captions lack local fine-grained details. A mix of 1-3 sentences ensures that the model learns both local and global alignment.

### Loss & Training
The average of two sigmoid contrastive losses: $\mathcal{L} = \frac{1}{2}(\mathcal{L}^{tcs} + \mathcal{L}^{mps})$. $\mathcal{L}^{tcs}$ aligns the text-conditioned image representation $\mathbf{v}^{tc}$ with the text $\mathbf{t}^g$ (fine-grained loss); $\mathcal{L}^{mps}$ aligns the global image token $\mathbf{v}^g$ with all sub-captions (global loss, multi-positive samples). The training data is an MLLM-recaptioned version of CC3M+CC12M+YFCC15M, totaling 30M image-text pairs.

## Key Experimental Results

### Main Results

| Method (Data Size) | COCO T2I R@1 | Flickr T2I R@1 | DOCCI-FG T2I | VOC20 mIoU | ImageNet Top-1 |
|--------------|-------------|----------------|-------------|------------|---------------|
| OpenCLIP (2B) | 41.7 | 71.9 | - | 47.2 | 70.2 |
| SigLIP (10B) | 47.2 | 75.6 | 20.6 | - | - |
| DreamLIP (30M) | 44.8 | 73.3 | 21.6 | 1.8 | 58.1 |
| **FLAIR (30M)** | **53.3** | **81.1** | **25.0** | **73.0** | 56.6 |

### Ablation Study

| Configuration | COCO T2I | VOC20 mIoU | ImageNet | Description |
|------|---------|------------|----------|------|
| Global Loss Only (GL) | 28.3 | 3.1 | 25.4 | baseline |
| Text-Conditioned Loss Only (TC) | 32.0 | 36.9 | 28.1 | TC is the foundation of fine-grained alignment |
| GL + Multi-Caption (MC) | 32.9 | 1.7 | 27.9 | Multi-captions do not improve segmentation |
| TC + MC + Diverse Sampling (DS) | 36.2 | 46.5 | 31.5 | DS provides substantial gains |
| FLAIR (GL+TC+MC+DS) | **37.7** | **59.7** | **33.8** | All components collaborate to achieve optimal performance |

### Key Findings
- **Text-conditioned attention pooling is a fundamental contribution**: Simply adding the TC component increases VOC20 mIoU from 3.1 to 36.9, demonstrating that it endows the model with spatial-level semantic understanding.
- **Negative sample design determines success or failure**: DreamLIP employed a similar concept but had a shortcut in its negative samples, resulting in a segmentation mIoU of only 0.7 (virtually zero). FLAIR's correct negative sampling design achieves vastly superior performance for the same core mechanism.
- **30M synthetic captions > 10B web data**: FLAIR (30M) outperforms SigLIP (10B) by over 5 points in fine-grained retrieval, proving that data quality (fine-grained synthetic captions) is far more critical than data scale for local alignment tasks.
- **A gap still exists in classification tasks**: On ImageNet, FLAIR (30M) is 13.6 points lower than OpenCLIP (2B). Although synthetic captions enhance detail comprehension, they cannot replace the real-world data diversity that covers hundreds of millions of categories.

## Highlights & Insights
- **The concept of "text-conditioned adaptive visual representation" is highly elegant**: Generating different representations for the same image under different texts is far more flexible than a global vector. This concept can be directly transferred to visual token compression in VLMs—allowing the LLM's text representation to guide which visual tokens are fed into subsequent layers.
- **The shortcut-prevention negative sample design serves as a compelling lesson**: DreamLIP's failure clearly illustrates how "information leakage" in contrastive learning can jeopardize representation quality, offering key insights for contrastive learning researchers.
- **Remarkable results with small data volume**: Beating 10B data with just 30M data indicates that "how to use data" is substantially more important than "how much data is used" in fine-grained tasks.

## Limitations & Future Work
- Classification performance lags behind large-scale pre-trained models by ~14 points, suggesting a potential trade-off between fine-grained alignment and category-level generalization.
- Experiments were only conducted using ViT-B/16; the effectiveness on larger models (ViT-L/H) and higher resolutions remains unverified.
- Text-conditioned attention pooling introduces extra inference overhead (each text query requires an individual cross-attention forward pass), which could become a bottleneck in large-scale retrieval scenarios.
- It depends heavily on the quality of synthetic captions generated by MLLMs—inaccurate or hallucinated captions could directly impact training.

## Related Work & Insights
- **vs DreamLIP**: Both utilize synthetic long captions and local tokens, but DreamLIP's negative sample design leads to a complete failure in segmentation. FLAIR's crucial improvement lies in the proper construction of negative samples and replacing simple token matching with cross-attention.
- **vs Long-CLIP / LoTLIP**: These methods extend CLIP's text length limit to handle long captions but still rely on global alignment. FLAIR's text-conditioned pooling also significantly outperforms them on long-text retrieval (by +5-11 points).
- **vs OpenCLIP / SigLIP**: These are brute-force scaling methods trained on billions of data. FLAIR's ability to outperform them on fine-grained tasks with only 30M data demonstrates that architectural innovation can compensate for gaps in data scale.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The text-conditioned attention pooling concept is simple yet highly effective, and the shortcut-prevention negative sampling design demonstrates a profound methodological understanding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple tasks including retrieval, segmentation, and classification, with detailed and clearly explained ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The methodological motivation is clearly explained, and the comparative analysis with DreamLIP is strong. However, having too many experimental tables in some parts slightly affects readability.
- Value: ⭐⭐⭐⭐⭐ Pioneered a new paradigm for text-conditioned visual representations, offering broad inspiration for VLM representation learning and dense prediction tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](../../NeurIPS2025/multimodal_vlm/focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)
- [\[ACL 2025\] A Parameter-Efficient and Fine-Grained Prompt Learning for Vision-Language Models](../../ACL2025/multimodal_vlm/a_parameter-efficient_and_fine-grained_prompt_learning_for_vision-language_model.md)
- [\[CVPR 2026\] See What I Mean: Aligning Vision and Language Representations for Video Fine-grained Object Understanding](../../CVPR2026/multimodal_vlm/see_what_i_mean_aligning_vision_and_language_representations_for_video_fine-grai.md)
- [\[ICLR 2026\] Bongard-RWR+: Real-World Representations of Fine-Grained Concepts in Bongard Problems](../../ICLR2026/multimodal_vlm/bongard-rwr_real-world_representations_of_fine-grained_concepts_in_bongard_probl.md)
- [\[CVPR 2026\] DeAR: Fine-Grained VLM Adaptation by Decomposing Attention Head Roles](../../CVPR2026/multimodal_vlm/dear_fine-grained_vlm_adaptation_by_decomposing_attention_head_roles.md)

</div>

<!-- RELATED:END -->
