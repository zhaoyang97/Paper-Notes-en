---
title: >-
  [Paper Note] CTRL-O: Language-Controllable Object-Centric Visual Representation Learning
description: >-
  [CVPR 2025][Image Generation][Object-Centric Representation] CTRL-O introduces language controllability into object-centric representation learning. By using language embedding to initialize slot queries, conditioning the decoder on language, and employing a control contrastive loss, it achieves language-object binding without mask supervision. It achieves an FG-ARI of 47.5 on COCO (+7.0 over Dinosaur), while supporting zero-shot referring expression segmentation…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Object-Centric Representation"
  - "Slot Attention"
  - "Language Control"
  - "Contrastive Loss"
  - "Object Discovery"
date: 2026-05-08
content_hash: a8b799e7cf88461b
---

# CTRL-O: Language-Controllable Object-Centric Visual Representation Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.21747](https://arxiv.org/abs/2503.21747)  
**Code**: [https://ctrl-o-paper.github.io](https://ctrl-o-paper.github.io)  
**Area**: Image Generation  
**Keywords**: Object-Centric Representation, Slot Attention, Language Control, Contrastive Loss, Object Discovery

## TL;DR

CTRL-O introduces language controllability into object-centric representation learning. By using language embedding to initialize slot queries, conditioning the decoder on language, and employing a control contrastive loss, it achieves language-object binding without mask supervision. It achieves an FG-ARI of 47.5 on COCO (+7.0 over Dinosaur), while supporting zero-shot referring expression segmentation, instance-level image generation, and VQA.

## Background & Motivation

1. **Background**: Object-centric representation learning (e.g., Slot Attention, Dinosaur) decomposes scenes into independent object representations (slots), but slots are uncontrollable—it is impossible to specify which slot corresponds to which object.
2. **Limitations of Prior Work**: (1) Slot assignment is arbitrary, making it impossible for users to specify objects of interest using language; (2) object discovery accuracy is limited in complex real-world scenes; (3) learned representations are difficult to apply directly to downstream tasks.
3. **Key Challenge**: Object-centric representation requires "discovering" objects (unsupervised), whereas "controlling" object binding requires semantic understanding—how to introduce language control without requiring mask annotations?
4. **Goal**: Control the binding of slots to objects using language descriptions without mask supervision.
5. **Key Insight**: Initialize slot queries with pretrained LLM embeddings, so slots naturally tend to bind to objects of corresponding semantics.
6. **Core Idea**: Query initialization (LLM embedding + positional information) + decoder language conditioning + control contrastive loss.

## Method

### Overall Architecture

Input image $\rightarrow$ extract features with frozen DINOv2 $\rightarrow$ encode language descriptions via LLaMA-3-8B (LLM2Vec) + centroid coordinates $\rightarrow$ initialize slot queries $\rightarrow$ iterative assignment via Slot Attention $\rightarrow$ reconstruct with decoder conditioned on slot + control query $\rightarrow$ control contrastive loss to constrain slot-language alignment.

### Key Designs

1. **Language Query Initialization**

    - **Function**: Encourages slots to bind to language-described objects from the start.
    - **Mechanism**: Concatenates LLaMA-3-8B (LLM2Vec) language embeddings with centroid coordinates to serve as initial slot queries. Dynamic class-to-prompt mapping: K-means clusters $C$ classes into $M$ prompts (updated per epoch).
    - **Design Motivation**: The object to which randomly initialized slots bind is uncontrollable; language initialization provides semantic "anchors".

2. **Control Contrastive Loss**

    - **Function**: Enforces alignment between slot representations and their corresponding language embeddings.
    - **Mechanism**: $\mathcal{L}_{CC}^l = -\sum_i \log\frac{\exp(z_i^{emb} \cdot l_i / \tau)}{\sum_t \exp(z_i^{emb} \cdot l_t / \tau)}$, where $z_i = \sum_k a_{ik} h_k$ denotes the DINO features aggregated by slot attention weights, and $\tau=0.1$.
    - **Design Motivation**: Initialization alone cannot guarantee that slots maintain correct binding after attention iterations.

3. **Decoder Language Conditioning**

    - **Function**: Injects language information during the reconstruction phase to enhance object-language association.
    - **Mechanism**: Concatenates slots with control queries and feeds them into an MLP decoder.
    - **Design Motivation**: Conditioning the decoder on language helps it learn more semantically meaningful reconstructions.

### Loss & Training

Reconstruction loss + control contrastive loss. Frozen DINOv2 ViT backbone. Trained on COCO+VG for 300K steps with a batch size of 128.

## Key Experimental Results

### Main Results

| Method | FG-ARI↑ | mBO↑ | Binding Hits↑ |
|------|---------|------|---------------|
| Dinosaur | 40.5 | 27.7 | - |
| **CTRL-O** | **47.5** | 27.2 | **61.3%** |

| Task | CTRL-O | Best Baseline |
|------|--------|---------|
| RefCOCO mIoU (Zero-shot) | 28.2 | Shatter&Gather 21.8 |
| Image Generation FID (COCO) | 25.20 | Stable LSD 26.20 |
| VQAv2 Accuracy | 60.25% | CLIP 58.64% |

### Ablation Study

| Configuration | Binding Hits | Note |
|------|-------------|------|
| w/o Language Initialization | ~40% | Loss of semantic anchors |
| w/o Contrastive Loss | ~48% | Alignment not persistent |
| w/ GT masks (Upper Bound) | 71.2% | Supervised ceiling |
| **Full CTRL-O** | **61.3%** | Unsupervised, close to upper bound |

### Key Findings

- The +7.0 FG-ARI gain primarily stems from language guidance enabling slots to segment boundaries more accurately.
- The zero-shot referring expression segmentation score of 28.2 mIoU outperforms non-language methods by over 30%.
- Even with GT mask supervision, Binding Hits reach only 71.2%, indicating that object binding itself is a challenging problem.

## Highlights & Insights

- **Controllability is the key missing piece in object-centric representations**: CTRL-O fills this gap.
- **61.3% Binding Hits without mask supervision**: Standard language-object alignment can be achieved without segmentation annotations.
- **Unified framework supporting multiple tasks**: Object discovery, referring segmentation, image generation, and VQA.

## Limitations & Future Work

- Multiple instances of the same category require additional spatial disambiguation (centroid coordinates).
- The mBO of the MLP decoder (27.2) is slightly lower than that of Dinosaur (27.7).
- The VQA accuracy of 60.25% is still far below large language model-based solutions (>80%).
- Diffusion generation exhibits failure cases of object distortion or repetition.

## Related Work & Insights

- **vs Dinosaur**: Lacks language control, achieving 40.5 FG-ARI. CTRL-O improves this to 47.5 via language anchors.
- **vs CLIP**: Performs global image-level contrastive learning without object-level decomposition. CTRL-O performs language alignment at the object level.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing language control to object-centric learning is a natural yet significant extension.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-angle validation spanning object discovery, segmentation, generation, and VQA.
- Writing Quality: ⭐⭐⭐⭐ Clear and well-written.
- Value: ⭐⭐⭐⭐ Controllable object representations that unify multiple tasks hold long-term research value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GLASS: Guided Latent Slot Diffusion for Object-Centric Learning](glass_guided_latent_slot_diffusion_for_object-centric_learning.md)
- [\[ICCV 2025\] GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning](../../ICCV2025/image_generation/genflowrl_shaping_rewards_with_generative_object-centric_flow_in_visual_reinforc.md)
- [\[CVPR 2025\] ORIDa: Object-Centric Real-World Image Composition Dataset](orida_object-centric_real-world_image_composition_dataset.md)
- [\[CVPR 2025\] Visual Lexicon: Rich Image Features in Language Space](visual_lexicon_rich_image_features_in_language_space.md)
- [\[CVPR 2025\] CLIP Under the Microscope: A Fine-Grained Analysis of Multi-Object Representation](clip_under_the_microscope_a_fine-grained_analysis_of_multi-object_representation.md)

</div>

<!-- RELATED:END -->
