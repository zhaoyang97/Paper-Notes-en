---
title: >-
  [Paper Note] A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity
description: >-
  [NeurIPS 2025][Audio & Speech][Tri-modal alignment] TRIANGLE proposes using the area of the triangle formed by three modal embedding vectors in high-dimensional space as a similarity measure, replacing traditional pairwise cosine similarity to achieve joint alignment of video, audio, and text modalities. The method surpasses state-of-the-art by up to 9 Recall@1 points on video-text retrieval and related tasks.
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "Tri-modal alignment"
  - "cosine similarity alternative"
  - "triangle area similarity"
  - "contrastive learning"
  - "video-text retrieval"
date: 2026-05-08
content_hash: 954e222d1c1b37a5
---

# A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity

**Conference**: NeurIPS 2025
**arXiv**: [2509.24734](https://arxiv.org/abs/2509.24734)  
**Code**: [https://github.com/ispamm/TRIANGLE/](https://github.com/ispamm/TRIANGLE/)  
**Area**: Multimodal VLM / Audio & Speech
**Keywords**: Tri-modal alignment, cosine similarity alternative, triangle area similarity, contrastive learning, video-text retrieval

## TL;DR

TRIANGLE proposes using the area of the triangle formed by three modal embedding vectors in high-dimensional space as a similarity measure, replacing traditional pairwise cosine similarity to achieve joint alignment of video, audio, and text modalities. The method surpasses state-of-the-art by up to 9 Recall@1 points on video-text retrieval and related tasks.

## Background & Motivation

**Background**: Since CLIP, the paradigm for multimodal alignment has been built on pairwise cosine similarity — selecting an anchor modality and aligning all other modalities to it one by one. For example, ImageBind uses images as the anchor, and LanguageBind uses text. This approach performs well in two-modality tasks and has since been extended to three-modality scenarios (e.g., video + audio + text).

**Limitations of Prior Work**: Pairwise cosine similarity has a fundamental limitation — it only guarantees that each modality aligns with the anchor, but provides no alignment guarantee between non-anchor modalities. For instance, while video and audio may each align with text, whether video and audio align with each other remains unknown. In practice, this causes models to underperform on tasks requiring the integration of multiple modalities. For example, in video-text retrieval, visual frames alone cannot distinguish "a dog barking" from "a dog howling" — audio is the key discriminative cue, yet existing models cannot effectively leverage this third modality.

**Key Challenge**: Cosine similarity is intrinsically a two-dimensional metric and cannot naturally extend to the joint space of three or more vectors. Existing workarounds — such as MLP fusion, auxiliary loss functions, or anchor selection strategies — either introduce additional parameters or still lack geometric interpretability.

**Goal**: To directly compute a joint similarity measure in the natural high-dimensional space of three modal embeddings, without resorting to pairwise comparisons or additional fusion layers.

**Key Insight**: Three embedding vectors on the unit hypersphere naturally form a triangle — the area of this triangle directly reflects the degree of tri-modal alignment (smaller area = better alignment), requiring only three dot-product operations.

**Core Idea**: Replace pairwise cosine similarity with the area of the triangle formed by three modal embedding vectors, enabling direct measurement of joint tri-modal alignment in high-dimensional space.

## Method

### Overall Architecture

Video frames, audio waveforms, and text captions are encoded separately by a video encoder (EVAClip-ViT-G), an audio encoder (BEATs), and a text encoder (BERT-B). After normalization, the three embedding vectors lie on the unit hypersphere and form a triangle. TRIANGLE substitutes this triangle's area for cosine similarity within a new contrastive loss function for training. At inference, retrieval and classification are also performed based on triangle area.

### Key Designs

1. **Triangle Area Similarity**:

    - **Function**: Directly measures the joint alignment of three modal embeddings in high-dimensional space.
    - **Mechanism**: Given three embeddings $\mathbf{x}, \mathbf{y}, \mathbf{z}$, define $\mathbf{u} = \mathbf{x} - \mathbf{y}$ and $\mathbf{v} = \mathbf{x} - \mathbf{z}$. The triangle area is $A = \frac{1}{2}\sqrt{\langle\mathbf{u},\mathbf{u}\rangle\langle\mathbf{v},\mathbf{v}\rangle - \langle\mathbf{u},\mathbf{v}\rangle^2}$. A smaller area indicates that the three vectors are clustered together (well-aligned), while a larger area indicates dispersion (poor alignment). The entire computation requires only 3 dot products, with negligible overhead (0.0016 s for 2048-dim vectors vs. 0.0001 s for cosine similarity).
    - **Design Motivation**: Cosine similarity is defined only in 2D and cannot capture the joint positional relationship of three vectors; triangle area is the most concise geometric quantity that directly reflects the relative positions of three points in high-dimensional space.

2. **TRIANGLE Contrastive Loss**:

    - **Function**: Embeds the triangle area measure into a standard contrastive learning loss.
    - **Mechanism**: In the standard InfoNCE loss, cosine similarity is replaced by negative triangle area. For example, the Data-to-Text loss is $\mathcal{L}_{D2T} = -\frac{1}{B}\sum_{i}\log\frac{\exp(-A(\mathbf{t}_i, \mathbf{v}_i, \mathbf{a}_i)/\tau)}{\sum_j \exp(-A(\mathbf{t}_j, \mathbf{v}_i, \mathbf{a}_i)/\tau)}$. The negative sign ensures that smaller area (better alignment) yields lower loss. A Data-Text-Matching (DTM) cross-attention loss is retained as an auxiliary objective. The final loss is $\mathcal{L} = \frac{1}{2}(\mathcal{L}_{D2T} + \mathcal{L}_{T2D}) + \lambda\mathcal{L}_{DTM}$.
    - **Design Motivation**: Directly substituting the similarity measure in the contrastive loss requires no architectural modifications, preserving the simplicity and generality of the approach.

3. **Cosine Regularization**:

    - **Function**: Handles the degenerate case where the triangle collapses to a line, and enhances downstream task performance.
    - **Mechanism**: A cosine regularization term is added to the inference-time similarity: $\mathcal{A} = A - \alpha\cos\theta_{\mathbf{xy}}$, where $\alpha$ is a balancing coefficient and $\theta_{\mathbf{xy}}$ is the angle between the two modalities relevant to the downstream task. When the triangle degenerates (three points nearly collinear), the area approaches zero and provides no discriminative signal; the cosine term then supplements alignment information between the two modalities.
    - **Design Motivation**: Addresses the robustness of triangle area under degenerate configurations, ensuring an effective alignment signal across all vector arrangements.

### Loss & Training

Starting from VAST's pretrained encoders, additional pretraining is performed on a 150k subset of the VAST27M dataset. The MLP fusion layer from VAST is removed; the TRIANGLE loss alone reshapes the latent space. Notably, applying the same additional pretraining procedure to VAST itself leads to overfitting, demonstrating that the TRIANGLE loss can reorganize the knowledge of existing encoders into a more unified multimodal space.

## Key Experimental Results

### Main Results

| Dataset | Metric | TRIANGLE | VAST (same encoder) | Gain |
|--------|------|----------|----------------|------|
| MSR-VTT T2V | R@1 | 55.2 | 49.3 | +5.9 |
| MSR-VTT V2T | R@1 | 52.5 | 43.7 | +8.8 |
| DiDeMo T2V | R@1 | 54.9 | 49.5 | +5.4 |
| DiDeMo V2T | R@1 | 53.1 | 48.2 | +4.9 |
| ActivityNet T2V | R@1 | 59.7 | 51.4 | +8.3 |
| ActivityNet V2T | R@1 | 54.1 | 46.8 | +7.3 |
| VATEX T2V | R@1 | 83.9 | 80.0 | +3.9 |
| AudioCaps T2A | R@10 | 77.1 | 65.4 | +11.7 |
| VGGSound Classification | R@1 | 44.8 | 39.6 | +5.2 |

### Ablation Study

| Configuration | T2AV R@1 | AV2T R@1 | Notes |
|------|---------|---------|------|
| VAST (cosine + MLP fusion) | 36.5 | 35.5 | Pairwise cosine + fusion layer |
| Symile | 0.3 | 0.4 | $n$-modal total correlation method; fails |
| GRAM | 38.9 | 41.9 | $n$-modal volume method |
| TRIANGLE w/o DTM | 33.3 | 40.4 | Without DTM loss |
| TRIANGLE (full) | 39.4 | 41.9 | Best configuration |

### Key Findings
- **Importance of DTM loss**: Removing DTM drops the T2AV direction by 6.1 points, indicating that the cross-attention auxiliary loss is critical for text-side retrieval.
- **General $n$-modal methods underperform tri-modal specialized ones**: Although Symile and GRAM are theoretically extensible to arbitrary $n$ modalities, both underperform TRIANGLE on three-modality tasks, suggesting that a triplet-specific objective more effectively exploits modality-specific features.
- **Vanilla experiment validation**: In a controlled setting using MNIST + AudioMNIST + text labels, TRIANGLE not only converges faster (reaching 90% accuracy at 4× the speed of cosine-based methods) but also achieves superior final performance.
- Gains on V2T consistently exceed those on T2V, and improvements on ActivityNet are the most pronounced (+8.3/+7.3), likely because ActivityNet videos have richer and more complementary audio information.

## Highlights & Insights
- **The elegance of triangle area as a similarity measure** is particularly compelling: only 3 dot products suffice to replace traditional pairwise comparisons, maintaining computational efficiency while providing an interpretable geometric meaning (area = an intuitive quantification of alignment discrepancy). This idea generalizes naturally to other scenarios requiring joint measurement of multi-vector relationships.
- **The design philosophy of modifying only the loss function, not the architecture**: TRIANGLE achieves substantial performance gains by replacing the loss function on top of VAST's encoders and removing the original MLP fusion layer, embodying a "less is more" principle.
- **Cosine regularization under triangle degeneracy** is an elegant engineering detail: when three points are nearly collinear, the triangle area approaches zero; the method then falls back to pairwise cosine similarity, ensuring robustness across all vector configurations.

## Limitations & Future Work
- The method is currently limited to three-modality alignment. Although the authors mention that the area concept can be extended to polygon/parallelotope volumes for $n$ modalities, no experimental validation is provided.
- Pretraining uses only 150k samples (a small subset of VAST27M); the effect of larger-scale pretraining remains unexplored.
- The hyperparameter $\alpha$ in cosine regularization requires task-specific tuning, and its sensitivity is not discussed.
- Generalizability to other tri-modal combinations (e.g., RGB + depth + text, tactile + visual + text) is not investigated.

## Related Work & Insights
- **vs. VAST**: Using identical encoders but different loss functions, TRIANGLE comprehensively surpasses VAST by 4–9 points, demonstrating that joint alignment outperforms "fuse-then-compare."
- **vs. GRAM**: GRAM measures $n$-modal alignment via parallelotope volume and is theoretically more general, yet underperforms TRIANGLE on three-modality tasks, suggesting that objectives tailored to a specific number of modalities are more effective.
- **vs. ImageBind / LanguageBind**: These methods rely on pairwise anchor-based alignment and lack alignment guarantees between non-anchor modalities — the core limitation that TRIANGLE is designed to address.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Replacing cosine similarity with triangle area is a concise and profound idea with clear geometric intuition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Seven benchmarks, three task types, training from scratch, pretrained fine-tuning, and ablation studies provide comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ — Geometric intuition is explained clearly, with well-designed figures and tables.
- **Value**: ⭐⭐⭐⭐ — Provides a new paradigm for multimodal alignment; the triangle area measure has strong transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Natural Alignment to Conditional Controllability in Multimodal Dialogue](../../ICLR2026/audio_speech/from_natural_alignment_to_conditional_controllability_in_multimodal_dialogue.md)
- [\[ICLR 2026\] Beyond Instance-Level Alignment: Dual-Level Optimal Transport for Audio-Text Retrieval](../../ICLR2026/audio_speech/beyond_instance-level_alignment_dual-level_optimal_transport_for_audio-text_retr.md)
- [\[ACL 2026\] RespiraMFM: A Multimodal Foundation Model for Respiratory Disease Recognition via Contrastive Audio-Language Alignment](../../ACL2026/audio_speech/respiramfm_a_multimodal_foundation_model_with_contrastive_audio-language_alignme.md)
- [\[NeurIPS 2025\] LeVo: High-Quality Song Generation with Multi-Preference Alignment](levo_high-quality_song_generation_with_multi-preference_alignment.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)

</div>

<!-- RELATED:END -->
