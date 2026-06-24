---
title: >-
  [Paper Note] Synergy of Sight and Semantics: Visual Intention Understanding with CLIP
description: >-
  [ECCV 2024][Multi-label Intention Understanding] This paper proposes the IntCLIP framework, which transfers "sight" (visual perception) knowledge in CLIP to "semantic" (semantic-centric) multi-label intention understanding tasks through a dual-branch encoding strategy. Combined with hierarchical class integration and sight-assisted aggregation, it significantly outperforms state-of-the-art (SOTA) methods on standard MIU benchmarks and image emotion recognition tasks.
tags:
  - "ECCV 2024"
  - "Multi-label Intention Understanding"
  - "CLIP"
  - "dual-branch architecture"
  - "hierarchical class integration"
  - "visual-semantic fusion"
date: 2026-05-08
content_hash: bebc85a32b68c17c
---

# Synergy of Sight and Semantics: Visual Intention Understanding with CLIP

**Conference**: ECCV 2024  
**PDF**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01721.pdf) / [Author Version](https://marswhu.github.io/publications/files/ECCV24_IntCLIP.pdf)
**Code**: [https://github.com/yan9qu/IntCLIP](https://github.com/yan9qu/IntCLIP)  
**Area**: Others  
**Keywords**: Multi-label Intention Understanding, CLIP, dual-branch architecture, hierarchical class integration, visual-semantic fusion

## TL;DR

This paper proposes the IntCLIP framework, which transfers "sight" (visual perception) knowledge in CLIP to "semantic" (semantic-centric) multi-label intention understanding tasks through a dual-branch encoding strategy. Combined with hierarchical class integration and sight-assisted aggregation, it significantly outperforms state-of-the-art (SOTA) methods on standard MIU benchmarks and image emotion recognition tasks.

## Background & Motivation

**Background**: Multi-label Intention Understanding (MIU) is an emerging and highly challenging task. Given an image, the model needs to predict multiple potential intentions of the photographer (e.g., "showing food", "recording scenery", "expressing emotion"). Unlike traditional object detection with explicit visual features, these intentions represent highly subjective and abstract semantic concepts. Currently, the largest benchmark in the MIU domain is the Intentonomy dataset.

**Limitations of Prior Work**: MIU faces two core challenges. First, annotated data is extremely scarce. Due to the ambiguity of intentions, the labeling process is highly time-consuming (requiring consensus among multiple annotators), making existing dataset scales far smaller than mainstream computer vision datasets. Second, intention is a "cross-modal" concept, depending on both visual content (what is in the photo) and semantic reasoning (why the photo was taken). Existing methods mainly rely on CNN + classification head architectures, which fail to effectively fuse visual perception and semantic reasoning capabilities.

**Key Challenge**: MIU requires the model to understand "why" rather than "what", which necessitates cross-level reasoning from visual content to abstract intentions. However, the limited existing annotated data is insufficient for the model to learn this complex cross-level mapping. Large-scale pre-trained vision-language models like CLIP possess rich visual-semantic knowledge, but their knowledge is primarily "objective-visual" oriented (describing "what is present" in the image), resulting in suboptimal performance when directly applied to subjective intention understanding tasks.

**Goal**: 1) How to effectively utilize the pre-trained knowledge of CLIP to compensate for the lack of MIU annotated data; 2) how to bridge the gap between CLIP's "objective visual" representation and MIU's "subjective semantic" requirement; 3) how to handle the hierarchical label structure (nested relationships between coarse-grained and fine-grained intentions) in MIU.

**Key Insight**: The authors sharply distinguish between two types of visual representations: "Sight" (pure visual perception, focusing on what objects and scenes are in the image) and "Semantic" (subjective semantics, focusing on intentions and emotions behind the image). CLIP excels at the former, whereas MIU requires the latter. The authors propose to avoid having a single branch handle both tasks simultaneously, designing a dual-branch architecture instead to handle them separately before intelligently fusing them.

**Core Idea**: Freeze a CLIP branch to retain objective visual knowledge as "anchors", use a trainable branch to learn intention semantics, and inject visual clues into semantic features via attention-based aggregation to achieve visual and semantic synergy.

## Method

### Overall Architecture

IntCLIP adopts an architecture featuring dual-branch image encoding and text alignment. The input image is simultaneously processed by two visual encoder branches: the Sight branch (with fully frozen CLIP parameters) and the Semantic branch (making deep layers trainable). The Sight branch preserves CLIP's original objective visual representation capability, while the Semantic branch adapts to subjective semantic understanding tasks via fine-tuning. Features from the two branches are fused through the Sight-assisted Aggregation module and finally aligned with text features to perform multi-label classification. On the text side, the Hierarchical Class Integration (HCI) module converts multi-level intention labels into natural language descriptions understandable by CLIP.

### Key Designs

1. **Sight-Semantic Dual-Branch Image Encoding**:

    - **Function**: Simultaneously maintain CLIP's objective visual capability and learn the subjective semantic capability required for MIU.
    - **Mechanism**: The two branches share the first few layers of the CLIP ViT encoder (shallow layers extract general low-level features) and diverge at deeper layers. The Sight branch completely freezes all parameters, acting as a read-only visual knowledge base, outputting feature maps $F_{\text{sight}} \in \mathbb{R}^{N \times D}$ containing rich knowledge of objects, scenes, and attributes learned during CLIP training. The Semantic branch unlocks gradients for the deep Transformer blocks, gradually shifting toward capturing subjective intention-related patterns by fine-tuning on MIU annotated data. This partially trainable design avoids catastrophic forgetting (keeping the Sight branch frozen) while allowing the model to adapt to the target task (making the Semantic branch adjustable).
    - **Design Motivation**: If CLIP is completely frozen, the model cannot learn the semantic patterns unique to MIU. If it is completely fine-tuned, the limited MIU data would lead to the forgetting of CLIP's pre-trained knowledge. The dual-branch architecture elegantly resolves this dilemma.

2. **Hierarchical Class Integration (HCI)**:

    - **Function**: Convert multi-level intention labels in the MIU dataset into natural language descriptions that can be efficiently processed by the CLIP text encoder.
    - **Mechanism**: The label system of MIU typically has a hierarchical structure—for instance, "show/share" $\rightarrow$ "show food" $\rightarrow$ "show homemade dessert". HCI combines each fine-grained label with its higher-level coarse-grained label, generating hierarchical text descriptions like "An image showing food, specifically homemade dessert". The CLIP text encoder then encodes these descriptions into text features $T \in \mathbb{R}^{C \times D}$, which are aligned with the image features. By introducing hierarchical context, the text features can define the position of each intention category in CLIP's embedding space more precisely.
    - **Design Motivation**: Directly feeding short label words (e.g., "homemade dessert") into the CLIP text encoder yields overly simplistic features, making it hard to distinguish similar yet hierarchically distinct intentions. HCI enriches the semantic information of text features by supplementing hierarchical context, fully utilizing CLIP's powerful sentence comprehension abilities.

3. **Sight-assisted Aggregation (SAA)**:

    - **Function**: Inject the objective visual information of the Sight branch into the semantic features of the Semantic branch.
    - **Mechanism**: Using the feature map of the Semantic branch $F_{\text{sem}}$ as the query (Q), and the feature map of the Sight branch $F_{\text{sight}}$ as the key (K) and value (V), cross-attention enables the semantic features to selectively absorb visual cues. $F_{\text{fused}} = \text{Softmax}(F_{\text{sem}} W_Q (F_{\text{sight}} W_K)^\top / \sqrt{d}) \cdot F_{\text{sight}} W_V + F_{\text{sem}}$. This design allows the semantic branch to "view" objective visual information provided by the Sight branch—for instance, when identifying the intention "show food", the semantic branch can focus on the visual features of food regions detected by the Sight branch.
    - **Design Motivation**: The Semantic branch might gradually drift away from low-level visual information and become overly abstract during fine-tuning. SAA ensures that the final feature representation is always supported by a visual "anchor" for semantic reasoning, being neither too visual nor too abstract.

### Loss & Training

Asymmetric Loss (ASL) is used to address the positive-negative sample imbalance in multi-label classification. Standard cross-entropy is used for positive labels, while a hard threshold gradient truncation is applied to negative labels to suppress the influence of easy negative samples. The overall loss is formulated as $L = L_{\text{ASL}}(F_{\text{fused}}, T) + \lambda L_{\text{sight}}(F_{\text{sight}}, T)$, where the auxiliary Sight loss helps stabilize training.

## Key Experimental Results

### Main Results

| Dataset/Task | Metric | IntCLIP | Prev. SOTA | Gain |
|------------|------|---------|----------|------|
| Intentonomy (MIU) | mAP | 36.2 | 31.8 | +4.4 |
| Intentonomy (MIU) | CF1 | 35.6 | 31.3 | +4.3 |
| Intentonomy (MIU) | OF1 | 52.8 | 49.1 | +3.7 |
| ArtPhoto (Emotion) | mAP | 73.5 | 69.2 | +4.3 |
| FI (Emotion) | Acc | 68.1 | 65.3 | +2.8 |

### Ablation Study

| Configuration | Intentonomy mAP | Description |
|------|-----------------|------|
| Full IntCLIP | 36.2 | Full model |
| Single branch (frozen) | 28.5 | Single branch (frozen CLIP, no fine-tuning) |
| Single branch (finetuned) | 32.1 | Single branch (fully fine-tuned CLIP, losing pre-trained knowledge) |
| Dual branch w/o SAA | 33.8 | Dual branch without SAA |
| w/o HCI | 34.1 | Without HCI |
| w/o ASL (using BCE) | 34.7 | Without ASL (using BCE) |

### Key Findings

- Dual-branch vs. single-branch is the most critical design choice. The frozen single branch achieves only 28.5 mAP, the fully fine-tuned single branch achieves 32.1 mAP, and the dual-branch elevates performance to 33.8+, demonstrating that the "preserve and adapt" strategy is vastly superior to either extreme approach.
- SAA contributes a 2.4 mAP gain (33.8 $\rightarrow$ 36.2), verifying that visual information indeed provides auxiliary support for semantic reasoning.
- HCI contributes 2.1 mAP (34.1 $\rightarrow$ 36.2), illustrating that hierarchical label information helps CLIP define intention categories more accurately.
- IntCLIP also exhibits significant improvements in image emotion recognition (another subjective understanding task), indicating the strong generalizability of the framework.

## Highlights & Insights

- **The dichotomy of "Sight vs. Semantic"** is highly intuitive and practical. It decomposes the problem of transferring CLIP to subjective understanding tasks into two manageable sub-problems. This analytical perspective can be extended to any scenario requiring the adaptation of pre-trained VLMs to abstract semantic tasks (such as emotion analysis, aesthetic assessment, sarcasm detection, etc.).
- **The freezing/fine-tuning strategy of the dual-branch architecture** is a variant of parameter-efficient fine-tuning, but offers a more structured interpretation than methods like LoRA. Instead of simply reducing trainable parameters, it clearly separates "what to preserve" from "what to learn".
- The idea of **HCI** can be directly applied to other multi-label classification tasks with hierarchical labels. In CLIP's embedding space, hierarchical contexts effectively increase the discriminability between categories.

## Limitations & Future Work

- SAA is unidirectional (from Sight to Semantic). The subsequent paper (TPAMI 2025 version) has improved this to bidirectional symmetric aggregation, indicating room for development here.
- The template design of HCI is relatively manual and relies on the specific label hierarchy of the MIU dataset. Templates need to be redesigned for other datasets.
- Only CLIP ViT-B/16 was used as the backbone network, without exploring larger ViT-L or recent models like SigLIP/EVA-CLIP. Stronger base models could potentially yield further improvements.
- In multi-label intention understanding, the co-occurrence and mutual exclusion relationships between different intentions are not explicitly modeled. Introducing a Label Graph could be considered to capture the structural relationships among intentions.
- It has not been verified in single-label intention prediction scenarios, and could be extended to support both single-label and multi-label scenarios.

## Related Work & Insights

- **vs DualCoOp**: DualCoOp also uses CLIP for multi-label classification but adapts it via prompt tuning without distinguishing between visual and semantic knowledge types. IntCLIP's dual-branch design aligns better with MIU characteristics.
- **vs FameVIL**: FameVIL is a prior SOTA in the MIU domain, based on a multimodal late fusion architecture. IntCLIP substantially outperforms it by leveraging CLIP's robust pre-trained knowledge.
- **vs CoOp/CoCoOp**: These prompt learning methods learn optimal prompts for CLIP without altering the visual encoder. IntCLIP's partial fine-tuning strategy performs adaptation on the visual side as well, making it more suitable for tasks requiring deep image content comprehension.

## Rating

- Novelty: ⭐⭐⭐⭐ The Sight-Semantic dual-branch design idea is clear and inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both MIU and emotion recognition tasks, with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The "Sight vs. Semantic" narrative runs throughout the paper with a smooth logical flow.
- Value: ⭐⭐⭐⭐ Provides a general paradigm for utilizing VLMs in subjective visual understanding tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Domain Reduction Strategy for Non-Line-of-Sight Imaging](domain_reduction_strategy_for_non-line-of-sight_imaging.md)
- [\[ICCV 2025\] Processing and Acquisition Traces in Visual Encoders: What Does CLIP Know About Your Camera?](../../ICCV2025/others/processing_and_acquisition_traces_in_visual_encoders_what_does_clip_know_about_y.md)
- [\[ECCV 2024\] SpatialFormer: Towards Generalizable Vision Transformers with Explicit Spatial Understanding](spatialformer_towards_generalizable_vision_transformers_with_explicit_spatial_un.md)
- [\[ECCV 2024\] Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning](dropout_mixture_low-rank_adaptation_for_visual_parameters-efficient_fine-tuning.md)
- [\[AAAI 2026\] Data Complexity of Querying Description Logic Knowledge Bases under Cost-Based Semantics](../../AAAI2026/others/data_complexity_of_querying_description_logic_knowledge_bases_under_cost-based_s.md)

</div>

<!-- RELATED:END -->
