---
title: >-
  [Paper Note] I See What You Mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue
description: >-
  [ACL 2025][Multimodal VLM][Iconic gestures] This paper proposes a self-supervised pre-training method to learn embeddings of co-speech iconic gestures, grounding skeleton motions into language. It demonstrates the complementarity of gestures and speech in face-to-face reference resolution tasks, where the gesture+speech accuracy of 31% significantly outperforms speech-only (24%) or gesture-only (19%).
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Iconic gestures"
  - "reference resolution"
  - "multimodal dialogue"
  - "self-supervised pre-training"
  - "gesture representation learning"
date: 2026-05-08
content_hash: 7e5a67432b235b28
---

# I See What You Mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue

**Conference**: ACL 2025  
**arXiv**: [2503.00071](https://arxiv.org/abs/2503.00071)  
**Code**: [https://github.com/EsamGhaleb/ReferenceResolution](https://github.com/EsamGhaleb/ReferenceResolution)  
**Area**: Multimodal Understanding / Human-Computer Interaction  
**Keywords**: Iconic gestures, reference resolution, multimodal dialogue, self-supervised pre-training, gesture representation learning

## TL;DR
This paper proposes a self-supervised pre-training method to learn embeddings of co-speech iconic gestures, grounding skeleton motions into language. It demonstrates the complementarity of gestures and speech in face-to-face reference resolution tasks, where the gesture+speech accuracy of 31% significantly outperforms speech-only (24%) or gesture-only (19%).

## Background & Motivation

**Background**: Reference resolution in dialogue primarily studies how textual/linguistic expressions point to referents. Regarding non-verbal aspects, while deictic (pointing) and beat gestures have received some computational research, **iconic gestures** (e.g., using hands to simulate object shapes) have rarely been explored from a computational perspective.

**Limitations of Prior Work**: (1) Unlike pointing, iconic gestures lack explicit directional information, requiring more complex semantic understanding; (2) There is a lack of effective methods for learning gesture representations due to scarce gesture data and high annotation costs; (3) Multimodal human-computer interaction systems do not utilize this crucial communication channel.

**Key Challenge**: Cognitive science studies indicate that iconic gestures help listeners identify referents faster, but computational methods fail to model this gesture-to-object mapping effectively.

**Goal**: (1) How to learn robust embeddings of iconic gestures? (2) To what extent do gestures contribute to reference resolution? Are they complementary or redundant to speech?

**Key Insight**: Leveraging the semantic correlation between gestures and co-occurring speech (as gestures and their corresponding speech describe the same object), the authors ground skeleton motion into language space via self-supervised contrastive learning.

**Core Idea**: Gestural representations learned through multimodal pre-training (skeleton + speech/text semantics) are more discriminative than pure skeleton representations, even without speech inputs during inference.

## Method

### Overall Architecture
The method consists of two stages: (1) **Pre-training stage**: Learning gesture embeddings (skeleton-language alignment) on the CABB dataset using self-supervised methods; (2) **Downstream task**: Utilizing pre-trained gesture representations for reference resolution (classification over 70 object sub-parts).

### Key Designs

1. **Three Modality Encoders**:

    - **Skeleton Encoder**: Reconstructed from DSTFormer (Spatial-Temporal Transformer), retaining two parallel branches (temporal $\rightarrow$ spatial and spatial $\rightarrow$ temporal) with one layer each, and replacing the second layer with an optional cross-modal attention module.
    - **Speech Encoder**: Multilingual wav2vec-2 (xlsr-300), pooling all Transformer layers via learnable weighted averaging + two-layer CNN to fuse the temporal dimension.
    - **Semantic Encoder**: Dutch BERTje to extract word embeddings, providing richer semantic information.

2. **Three Pre-training Architectures**:

    - **Unimodal**: Skeleton only—masked reconstruction loss + unimodal contrastive loss (contrast between two augmented views).
    - **Multimodal**: Skeleton + speech/semantics—adds a CLIP-style multimodal contrastive loss to align global gesture representation and co-occurring speech/text into a shared space.
    - **Multimodal-X** (Best performing): Adds cross-attention to the Multimodal setup—injecting semantic tokens as keys/values into the cross-attention layer of the skeleton encoder, followed by a crossmodal contrastive loss to align pure skeleton representation with the fused representation.
    - Design Motivation: Multimodal-X achieves fine-grained temporal alignment through cross-attention, and aligning the pure skeleton representation allows inference to benefit from language grounding during pre-training without requiring speech input.

3. **Reference Resolution Model**:

    - Function: Predicting which of the 70 object sub-parts is being referred to given the gesture embedding.
    - Mechanism: A two-layer MLP (300, 150) trained directly on frozen pre-trained representations, validated using leave-one-round-out cross-validation.
    - Two Inference Scenarios: (1) Gesture-only input (testing pre-trained representation quality); (2) Concatenated gesture + co-occurring speech (testing multimodal complementarity).

### Loss & Training
- Unimodal loss = masked reconstruction + unimodal contrastive
- Multimodal loss = unimodal losses + CLIP-style multimodal contrastive
- Multimodal-X loss = multimodal losses + crossmodal contrastive (aligning pure skeleton vs. fused representation)
- Pre-training Data: CABB-XL (~400k samples), oversampled with a 1-second window covering >50% of automatically segmented gestures.

## Key Experimental Results

### Main Results (Reference resolution accuracy, 70 classes, random baseline 1.4%)

| Input Modality | Model | Accuracy |
|---------|------|--------|
| Skeleton only | Unimodal | 16% |
| Skeleton only | Multimodal (Semantic pre-training) | 19% |
| Skeleton only | Multimodal-X (Semantic pre-training) | ~19% |
| Semantics only (BERTje) | - | 24% |
| Skeleton + Semantics | Unimodal Skeleton + BERTje | ~27% |
| Skeleton + Semantics | **Multimodal-X + BERTje** | **31%** |

### Ablation Study (Pre-trained representation quality, Spearman correlation with expert annotations)

| Model | CABB-L | CABB-XL |
|------|--------|---------|
| Unimodal (Skeleton) | ~0.32 | ~0.33 |
| Multimodal (Speech) | ~0.35 | ~0.34 |
| Multimodal (Semantics) | ~0.34 | ~0.37 |
| Multimodal-X (Semantics) | ~0.34 | **~0.38** |
| Ghaleb et al. 2024 (ST-GCN) | ~0.28 | ~0.29 |

### Key Findings
- **"Free" boost from multimodal pre-training**: After Multimodal-X pre-training, even when only the skeleton is provided at inference, the accuracy increases from $16\% \rightarrow 19\%$ ($+18.8\%$), indicating that language grounding information is "distilled" into the skeleton representation.
- **Gestures and speech are complementary rather than redundant**: Speech-only yields 24%, while gesture+speech reaches 31% ($+29.2\%$), demonstrating that iconic gestures convey information not captured by speech.
- **Dialogue history facilitates gesture resolution**: As dialogue turns increase, the accuracy of dialogue-specific models leveraging history continuously rises (gesture entrainment effect), and multimodal pre-trained models benefit even more.
- **Semantic embeddings outperform raw speech**: Using Dutch BERTje text semantics as the alignment target yields better performance than wav2vec-2 speech, particularly when training with larger volumes of data (CABB-XL).

## Highlights & Insights
- **"Speech during training, not during inference"**: Injecting linguistic knowledge into skeleton representations via multimodal pre-training achieves inference-time performance improvements with zero speech dependence. This pre-training paradigm can be transfered to other "multimodal in training, unimodal in inference" scenarios.
- **Computational study of iconic gestures**: Fills a gap in computational studies of the semantically richest iconic gestures, transitioning beyond mere deictic/beat gestures.
- **Cross-attention fusion > late fusion**: The cross-attention mechanism in Multimodal-X provides finer-grained temporal alignment than the CLIP-style contrastive learning in Multimodal.

## Limitations & Future Work
- Evaluated only on a Dutch task-oriented dialogue dataset (CABB); cross-lingual/cross-task/open-domain generalization remains unknown.
- The 70-class object sub-part classification task is still relatively simplified compared to real-world scenarios with more candidate referents.
- Uses 2D skeletons (ViTPose); 3D skeletons might provide better spatial information.
- Although pre-training data is oversampled to 400k, it remains small compared to typical pretraining dataset sizes in NLP/CV areas.
- Did not link gesture representations to physical attributes of objects (e.g., shape, size), which could further improve reference resolution.

## Related Work & Insights
- **vs. Abzaliev et al. 2022**: They learned gesture-word embeddings from TED talks but focused on non-iconic gestures (functional words/discourse markers). This paper focuses on iconic gestures and applies them to reference resolution.
- **vs. Ghaleb et al. 2024b**: Prior work used ST-GCN + speech contrastive learning. This work uses Transformer + cross-attention + semantic embeddings, boosting correlation from ~0.29 to ~0.38.
- **vs. vision-language reference resolution**: ReferIt/Visual Genome studies text-to-image region mapping. This work studies gesture-to-object mapping, which is a different yet complementary direction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Extremely novel computational study on iconic gestures, with a cleverly designed multimodal pre-training framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation, comparing various architectures and training data sizes, though evaluated on only one dataset.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rigorous experimental design, and complete statistical tests.
- Value: ⭐⭐⭐⭐ Opens up new research directions and practical approaches for multimodal human-computer interaction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] See What I Mean: Aligning Vision and Language Representations for Video Fine-grained Object Understanding](../../CVPR2026/multimodal_vlm/see_what_i_mean_aligning_vision_and_language_representations_for_video_fine-grai.md)
- [\[ACL 2026\] "I See What You Did There": Can Large Vision-Language Models Understand Multimodal Puns?](../../ACL2026/multimodal_vlm/i_see_what_you_did_there_can_large_vision-language_models_understand_multimodal_.md)
- [\[ACL 2026\] Revisit What You See: Revealing Visual Semantics in Vision Tokens to Guide LVLM Decoding](../../ACL2026/multimodal_vlm/revisit_what_you_see_revealing_visual_semantics_in_vision_tokens_to_guide_lvlm_d.md)
- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](cant_see_the_forest_for_the.md)
- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)

</div>

<!-- RELATED:END -->
