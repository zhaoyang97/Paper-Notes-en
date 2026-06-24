---
title: >-
  [Paper Note] Assessing and Learning Alignment of Unimodal Vision and Language Models (SAIL)
description: >-
  [CVPR 2025][Segmentation][Vision-Language Alignment] The SAIL framework is proposed: first, the alignment potential of unimodal vision and language models is assessed through alignment probing (discovering that k-NN clustering quality is more crucial than linear separability); second, DINOv2 and pretrained language models are efficiently aligned using a lightweight GLU alignment layer + Sigmoid loss + multi-positive sample strategy, outperforming CLIP with only 6% of its trai…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Vision-Language Alignment"
  - "DINOv2"
  - "CLIP"
  - "Transfer Learning"
  - "Efficient Training"
date: 2026-05-08
content_hash: 91ec3832a359f3c1
---

# Assessing and Learning Alignment of Unimodal Vision and Language Models (SAIL)

**Conference**: CVPR 2025  
**arXiv**: [2412.04616](https://arxiv.org/abs/2412.04616)  
**Code**: [https://lezhang7.github.io/sail.github.io/](https://lezhang7.github.io/sail.github.io/)  
**Area**: Image Segmentation  
**Keywords**: Vision-Language Alignment, DINOv2, CLIP, Transfer Learning, Efficient Training

## TL;DR

The SAIL framework is proposed: first, the alignment potential of unimodal vision and language models is assessed through alignment probing (discovering that k-NN clustering quality is more crucial than linear separability); second, DINOv2 and pretrained language models are efficiently aligned using a lightweight GLU alignment layer + Sigmoid loss + multi-positive sample strategy, outperforming CLIP with only 6% of its training data.

## Background & Motivation

1. **Background**: Vision-Language Models (VLMs) such as CLIP concurrently train vision and language encoders through large-scale image-text contrastive learning. However, CLIP-style training from scratch requires massive datasets (400M image-text pairs) and substantial computational power (hundreds of GPUs), and its text encoder shows limited performance on complex reasoning tasks.

2. **Limitations of Prior Work**: Existing research (e.g., Huh et al.) has found an inherent alignment between pretrained unimodal models, but evaluation methods are indirect (e.g., mutual nearest-neighbor) and do not directly measure cross-modal distance—whereas actual inference precisely requires the direct computation of cross-modal similarity. Efficient training methods such as LiT and ShareLock freeze the vision encoder to train the language component, failing to improve the language compatibility of the vision encoder, which limits transfer to MLLMs.

3. **Key Challenge**: There is a fundamental limitation in CLIP's training strategy—even when scaling up the model size (427M $\rightarrow$ 1366M) and data scale (400M $\rightarrow$ 2B), there is no substantial improvement on complex reasoning tasks such as Winoground. The root cause is that the web-scraped short descriptions used by CLIP lack rich semantic information, preventing the text encoder from learning high-level reasoning capabilities.

4. **Goal**: (1) How to directly and quantitatively evaluate the cross-modal alignment potential of unimodal models? (2) Which properties of SSL representations affect alignment the most? (3) How to build a VLM that matches or even outperforms CLIP with minimal data and compute (a single A100 GPU, 5 hours)?

5. **Key Insight**: Given that high-quality unimodal models (DINOv2, NV-Embed-v2) have already achieved exceptional levels in the vision and language domains respectively, and they possess inherent alignment, one only needs to train a lightweight "translation layer" to achieve efficient cross-modal alignment, without training the entire model from scratch.

6. **Core Idea**: Freezing the pretrained vision and language backbones and training only a lightweight GLU alignment layer outperforms CLIP trained from scratch, achieving orders of magnitude improvement in data and compute efficiency.

## Method

### Overall Architecture

SAIL consists of two parts. Part I is **alignment probing**: systematically evaluating the alignment potential of different unimodal models. Part II is **alignment learning**: designing an efficient alignment training framework based on the assessment findings to train only lightweight alignment layers. All image-text pairs are pre-encoded into embedding vectors at the input stage, and only the embeddings + alignment layer are loaded during training (without loading the encoders), achieving training within 5 hours on a single A100 GPU.

### Key Designs

1. **Alignment Probing**:
    - **Function**: Freeze the vision and language backbones, connect the two representation spaces using a linear layer, train on the CC3M dataset, and then evaluate on COCO zero-shot retrieval.
    - **Key Finding 1**: The Pearson correlation coefficient between the k-NN classification accuracy of SSL vision models and their alignment performance is as high as $r=0.991$, while linear probing is only $r=0.847$—**clustering quality is more important than linear separability**.
    - **Key Finding 2**: The correlation coefficient between the MTEB average score of language models and alignment performance is $r=0.994$—**better language understanding directly leads to better alignment**.
    - **Key Finding 3**: Even when scaling up data and model size, CLIP still performs poorly on complex reasoning in Winoground, which is significantly improved by switching to the NV-Embed-v2 text encoder.
    - **Design Motivation**: To evaluate alignment by directly measuring cross-modal distance (rather than proxy metrics), which is closer to actual inference.

2. **GLU Alignment Layer**:
    - **Function**: Replace linear layers and MLPs to act as a lightweight non-linear mapping connecting visual and language spaces.
    - **Mechanism**: Use a Gated Linear Unit (GLU) + ReLU activation, with the intermediate dimension expanded to 8 times the input (GLU×8).
    - **Experimental Comparison**: Linear $\rightarrow$ MLP×4: classification increases but retrieval decreases; Linear $\rightarrow$ GLU×8: classification +12.2%, T2I +5%, I2T +9%.
    - **Design Motivation**: The gating mechanism of GLU is more suitable for alignment tasks than MLPs—the gate can selectively propagate features useful for alignment and suppress irrelevant features.

3. **Sigmoid Loss (Alternative to InfoNCE)**:
    - **Function**: Replace CLIP's InfoNCE contrastive loss with a binary classification Sigmoid loss.
    - **Key Formula**: $$\mathcal{L} = -\frac{1}{|\mathcal{B}|} \sum_i \sum_j \log \frac{1}{1 + e^{z_{ij}(-t\hat{\mathbf{x}}_i \cdot \hat{\mathbf{y}}_j + b)}}$$, where $z_{ij}=1$ when $i=j$, and $-1$ otherwise.
    - **Gain**: Compared to InfoNCE, ImageNet +5.3%, T2I +9.3%, and I2T +13.5%.
    - **Further Optimization**: Replace $|\mathcal{B}|$ (only positive pairs) with $|\mathcal{B}|^2$ (all pairs) for normalization, making the contributions of positive and negative samples more balanced, leading to an additional gain of ~1%.
    - **Design Motivation**: Sigmoid loss eliminates the computational overhead of softmax normalization and is more sensitive to hard negatives.

4. **Multi-Pos Data Strategy**:
    - **Function**: Use both the original short description and a MLLM-generated long description as positive samples for each image.
    - **Mechanism**: $\mathcal{L}_{Multi-Pos} = \mathcal{L}(\mathcal{I}, \mathcal{T}) + \mathcal{L}(\mathcal{I}, \mathcal{T}^{HQ})$, where short descriptions favor object recognition and long descriptions favor retrieval.
    - **Gain**: Using long descriptions alone decreases classification but improves retrieval; Multi-Pos achieves the best of both worlds (+3% classification, +1.5% retrieval).
    - **Design Motivation**: Short and long descriptions provide complementary training signals.

### Loss & Training

- Freeze all backbone parameters and train only the alignment layers.
- Image-text pairs are pre-encoded into embedding vectors (one-time cost); training only requires loading the embeddings and alignment layers.
- LION optimizer, learning rate $10^{-5}$, 50 epochs, batch size 32768.
- Temperature $t = \log 20$, bias $b = -10$, alignment layer output dimension 1024.
- DINOv2-L + GTE/NV-Embed-v2 used, with 23M training data (subset of CC3M+CC12M+YFCC15M).

## Key Experimental Results

### Main Results

| Model | Training Data | IN-1K | COCO I2T R@1 | COCO T2I R@1 | Winoground T/I/G | ADE20K mIoU |
|------|---------|-------|-------------|-------------|-----------------|-------------|
| CLIP-B (LAION400M) | 400M | 67.0 | - | - | 25.7/11.5/7.75 | - |
| CLIP-L (LAION400M) | 400M | 72.7 | 59.7 | 43.0 | 30.5/11.5/8.75 | 1.2 |
| LiT | CC12M | 56.2 | 30.0 | 16.5 | 24.3/6.5/4.8 | - |
| ShareLock | CC12M | 59.1 | 26.0 | 13.5 | 26.3/12.8/5.3 | - |
| SAIL-B-NV2 | CC12M | 68.1 | 57.3 | 45.3 | 35.0/17.25/13.0 | - |
| **SAIL-L-NV2** | **23M** | **73.4** | **62.4** | **48.6** | **40.25/18.75/15.0** | **14.2** |

SAIL-L-NV2 using 23M data (6% of CLIP) outperforms CLIP-L by 0.7% on ImageNet-1K, by 2.7–5.6% on COCO retrieval, by roughly 7–10% on Winoground, and dominates CLIP's 1.2 with 14.2 mIoU on ADE20K semantic segmentation.

### Ablation Study (Trained on CC3M)

| Configuration | IN-1K 0-shot | T2I R@1 | I2T R@1 |
|------|-------------|---------|---------|
| Baseline (Linear + InfoNCE) | 33.2 | 11.1 | 13.5 |
| + MLP×4 | 36.8 | 8.0 | 10.7 |
| + GLU×8 | 45.4 | 16.1 | 22.5 |
| + Sigmoid Loss | 50.7 | 25.4 | 36.0 |
| + Normalization Correction $|\mathcal{B}|^2$ | 51.8 | 26.2 | 36.7 |
| + Long-HQ (Long descriptions only) | 48.4 | 31.4 | 44.2 |
| **+ Multi-Pos** | **54.0** | **32.9** | **45.4** |

Each improvement step provides clear quantitative contributions. From the Linear baseline to the full SAIL, IN-1K increases by 20.8%, and retrieval R@1 increases by 20%+.

### Key Findings

- **Clustering Quality Determines Alignment**: The Pearson correlation $r$ of k-NN with alignment is $0.991$, much higher than the $0.847$ of linear probing. MAE-series models exhibit the worst alignment performance because their pixel-level reconstruction objectives focus on low-level details rather than high-level semantics.
- **Stronger Language Models Yield Better Alignment**: Replacing GTE with NV-Embed-v2 brings a 7–10% ImageNet improvement. Even a small vision encoder + strong language model outperforms a large vision encoder + weak language model (SAIL-B-NV2 > SAIL-L-GTE).
- **SAIL Improves DINOv2's MLLM Compatibility**: After integration into LLaVA-1.5, SAIL's DINOv2 encoder outperforms the CLIP encoder in 5 out of 7 tasks (whereas DINOv2 lagged behind CLIP significantly before).
- **Benefiting Even from 1% Medical Data**: Fine-tuning with only 1% VinDr-CXR data achieves a 90.53% accuracy, demonstrating exceptional data efficiency.

## Highlights & Insights

- **"CLIP for the Poor"**: Training on a single A100 for 5 hours with 23M data yields a model that outperforms CLIP trained on 400M data—this is highly valuable for academic teams with limited resources. The core trick lies in pre-encoding embeddings and only training the lightweight alignment layer, eliminating the need to load encoders during training.
- **Language Models are More Critical than Vision Models**: SAIL-B-NV2 (small vision + strong language) outperforms SAIL-L-GTE (large vision + weak language), indicating that the bottleneck of alignment quality lies on the language side. This contrasts with the CLIP community consensus of "data is king"—model quality can compensate for data scarcity.
- **Alignment Training Decouples/Improves the Vision Encoder**: SAIL not only learns the alignment but also makes DINOv2 more "linguistically coherent," as demonstrated by the performance improvement after integration into LLaVA-1.5. This challenges the assumptions of methods like LiT/ShareLock that freeze the vision encoder.

## Limitations & Future Work

- Insufficient OCR capability: lagging behind CLIP on TextVQA and MMB, rooted in DINOv2's inherent lack of text recognition ability.
- Although leading CLIP in open-vocabulary segmentation, the absolute performance remains limited (ADE20K 14.2 mIoU), leaving a large gap behind specialized segmentation methods.
- Can the alignment layer be further compressed? Although GLU×8 is more efficient than MLP, it still contains parameters.
- The potential of aligning decoder-only vision models (e.g., AIM) remains unexplored.

## Related Work & Insights

- **vs CLIP**: CLIP trains two encoders from scratch, requiring 400M data and hundreds of GPUs. SAIL reuses pretrained models, surpassing CLIP with only 6% data and a single GPU, representing the paradigm that "assembly is superior to starting from scratch."
- **vs LiT**: LiT freezes the vision encoder and trains the language encoder from scratch. SAIL freezes both sides and trains only the alignment layer, but significantly outperforms LiT—because SAIL leverages a pre-existing, strong pretrained language model (NV2).
- **vs ShareLock**: ShareLock adds a trainable head on top of frozen models but fails to improve the language compatibility of the vision encoder. SAIL retrofits the vision encoder via the alignment layer, enabling its transferability to MLLMs.

## Rating

- Novelty: ⭐⭐⭐⭐ The alignment probing framework has systematic value. While the concept of an "assembled VLM" is not entirely new, it is executed thoroughly.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematically compares various SSL/language models, includes exhaustive ablations, and covers comprehensive downstream tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ The progression from Part I (probing) to Part II (learning) is clear and elegant.
- Value: ⭐⭐⭐⭐⭐ Highly practical for academic teams with limited resources, with insights (clustering > linear separability, language > vision) that are inspiring to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SAIL: Assessing and Learning Alignment of Unimodal Vision and Language Models](assessing_and_learning_alignment_of_unimodal_vision_and_language_models.md)
- [\[CVPR 2025\] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)
- [\[CVPR 2025\] ResCLIP: Residual Attention for Training-free Dense Vision-language Inference](resclip_residual_attention_for_training-free_dense_vision-language_inference.md)
- [\[CVPR 2025\] SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models](sketchfusion_learning_universal_sketch_features_through_fusing_foundation_models.md)
- [\[CVPR 2025\] DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception](declip_decoupled_learning_for_open-vocabulary_dense_perception.md)

</div>

<!-- RELATED:END -->
