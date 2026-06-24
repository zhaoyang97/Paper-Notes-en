---
title: >-
  [Paper Note] Acoustic Individual Identification of White-Faced Capuchin Monkeys Using Joint Multi-Species Embeddings
description: >-
  [ACL 2025 (Short Paper)][Audio & Speech][Animal Individual Identification] This paper explores utilizing cross-species acoustic pre-trained embeddings from birds and humans to identify individual calls of white-faced capuchin monkeys. It discovers that joint multi-species representations can further enhance identification performance, providing a new transfer learning paradigm for individual identification of wild animals under extreme data scarcity.
tags:
  - "ACL 2025 (Short Paper)"
  - "Audio & Speech"
  - "Animal Individual Identification"
  - "Cross-Species Transfer Learning"
  - "Acoustic Embeddings"
  - "Multi-Species Representations"
  - "White-Faced Capuchin Monkeys"
date: 2026-05-08
content_hash: e21dde32f6e2d3c5
---

# Acoustic Individual Identification of White-Faced Capuchin Monkeys Using Joint Multi-Species Embeddings

**Conference**: ACL 2025 (Short Paper)  
**Code**: None  
**Area**: Speech & Audio  
**Keywords**: Animal Individual Identification, Cross-Species Transfer Learning, Acoustic Embeddings, Multi-Species Representations, White-Faced Capuchin Monkeys

## TL;DR

This paper explores utilizing cross-species acoustic pre-trained embeddings from birds and humans to identify individual calls of white-faced capuchin monkeys. It discovers that joint multi-species representations can further enhance identification performance, providing a new transfer learning paradigm for individual identification of wild animals under extreme data scarcity.

## Background & Motivation

**Background**: Acoustic individual identification of wild animals is a critical task for understanding animal social behavior and advancing wildlife conservation monitoring. Currently, the mainstream approach in this field relies on human experts conducting manual labeling and analysis for individual identification, which is time-consuming, labor-intensive, and non-scalable.

**Limitations of Prior Work**: The development of automated individual identification methods is severely constrained by data scarcity. For most wild species, acquiring large-scale labeled, individual-level acoustic data is virtually impossible, and traditional end-to-end deep learning methods perform poorly in such low-resource scenarios. Existing acoustic classification models are typically trained on specific species and cannot be directly transferred to new species.

**Key Challenge**: The fundamental contradiction lies between the scarcity of labeled data and the models' demand for large amounts of training data. For specific species like the white-faced capuchin monkey, the available labeled individual recordings are extremely limited, whereas training acoustic models from scratch requires far more data than is available.

**Goal**: (1) To verify whether acoustic representations from other species can effectively transfer to the monkey individual identification task; (2) to explore whether joint multi-species representations outperform single-species representations.

**Key Insight**: The authors observe that the basic principle of acoustic individual identification—distinguishing different individuals using vocal characteristics—shares commonalities across multiple species. Well-established pre-trained models already exist in bird individual identification and human speaker verification. The acoustic patterns learned by these models (such as spectral features and temporal variation patterns) may possess cross-species universality.

**Core Idea**: Utilizing the cross-species transferability of pre-trained acoustic embeddings from birds and humans to address the individual identification problem for data-scarce species, and achieving stronger generalization performance through joint multi-species embeddings.

## Method

### Overall Architecture

The overall pipeline of the method is as follows: first, acoustic features are extracted from audio recordings of white-faced capuchin monkeys. Then, pre-trained acoustic models from different species (birds, humans) are utilized to obtain embedding representations. Finally, a simple classifier is trained based on these embeddings to complete individual identification. The core innovation lies in the selection and joint utilization of cross-species embeddings.

### Key Designs

1. **Cross-Species Acoustic Embedding Extraction**:

    - Function: Map monkey vocalizations into the embedding spaces learned by pre-trained models of different species.
    - Mechanism: Perform inference on the audio of white-faced capuchin monkeys using bird acoustic models (such as BirdNET or similar bird species identification models) and human speaker verification models (such as those based on x-vector or ECAPA-TDNN) respectively, to extract fixed-dimensional embedding vectors. Although these models have never seen monkey data, they have learned universal spectral-temporal patterns (such as fundamental frequency variations, harmonic structures, and rhythmic patterns) on large-scale species/speaker discrimination tasks, which also exist in mammalian vocalizations.
    - Design Motivation: Directly training the entire acoustic model on limited monkey data leads to severe overfitting, whereas cross-species pre-trained models have already acquired rich acoustic discriminative capabilities and can be directly transferred.

2. **Joint Multi-Species Representation Fusion**:

    - Function: Integrate embeddings from pre-trained models of different species into a unified multi-species representation.
    - Mechanism: Concatenate or weight-combine the bird and human embeddings to form a joint multi-species representation vector. Pre-trained models from different species capture discriminative features at different levels of the acoustic signal—bird models may excel at capturing frequency modulations and short-term patterns, while human models excel at capturing long-term identity-related speaker features. Their complementarity makes the joint representation much richer.
    - Design Motivation: Embeddings from a single species might have blind spots in certain acoustic dimensions. Joint multi-species representations can integrate acoustic discriminative patterns evolved in different evolutionary branches, achieving more comprehensive feature coverage.

3. **Individual Classifier Training**:

    - Function: Train a lightweight classifier based on the extracted embedding vectors to attribute each vocalization to a specific individual.
    - Mechanism: After obtaining the embedding representations, simple classification models (such as linear classifiers, SVMs, or shallow MLPs) are trained for individual classification. Since the embeddings already provide high-quality feature representations, the classifier does not need to be complex and can converge rapidly on limited labeled data.
    - Design Motivation: Keeping the classifier simple avoids overfitting on small datasets, leaving the complexity of representation learning to the pre-trained models.

### Loss & Training

Standard supervised classification training with the cross-entropy loss function is adopted. Due to the small data volume, strategies such as cross-validation are likely used to ensure evaluation reliability. During the embedding extraction phase, pre-trained model parameters are frozen, and only the downstream classifier is trained.

## Key Experimental Results

### Main Results

| Embedding Representation | Classification Accuracy | Description |
|----------|-----------|------|
| Bird Embeddings | High | Single-species transfer is effective |
| Human Embeddings | High | Single-species transfer is effective |
| Joint Multi-Species Embeddings | Highest | Multi-species fusion yields further gains |
| Random Baseline | Low | Baseline comparison |

### Ablation Study

| Configuration | Classification Performance | Description |
|------|---------|------|
| Joint Bird + Human Embeddings | Best | Full model |
| Bird Embeddings Only | Good | Cross-species transfer is effective |
| Human Embeddings Only | Good | Speaker verification knowledge is transferable |
| No Pre-trained Embeddings | Poor | Limited performance when training directly on monkey data |

### Key Findings

- Cross-species transfer learning validates that the underlying acoustic features for individual identification possess cross-species universality: both bird and human acoustic models can effectively encode individual differences in monkeys.
- Joint multi-species representations outperform any single-species representation, indicating that acoustic models from different species capture complementary discriminative information.
- Transferable acoustic individual characteristics exist even between evolutionarily distant species (birds vs. primates).

## Highlights & Insights

- **Bold Hypothesis of Cross-Species Transfer Validated**: The successful transfer from bird acoustics to primate acoustics is inspiring, demonstrating that the acoustic features required for individual identification possess deep biological universality. This insight could drive the development of the entire computational ecoacoustics field.
- **Complementarity of Joint Multi-Species Representations**: Pre-trained models from different species act like "observers from different perspectives," each capturing different discriminative dimensions of the acoustic signal. Combining them reveals a more complete picture.
- **Zero-Shot Cross-Species Transfer Paradigm**: This methodological framework, which enables individual identification without requiring large amounts of data from the target species, can be directly transferred to the monitoring of other endangered species.

## Limitations & Future Work

- As a short paper, the experimental scale is limited, validated only on a single species (white-faced capuchin monkeys). Whether it can generalize to other primates or mammals remains to be further investigated.
- The paper does not analyze the differences in identification performance across different vocalization types (e.g., alarm calls, food calls, social calls) in detail.
- The choice of pre-trained models can be further explored: for example, cetacean underwater acoustic models or bat ultrasonic models might provide more complementary features.
- The extension from individual identification to individual tracking, as well as considerations for real-time field deployment, have not yet been addressed.
- Few-shot fine-tuning of pre-trained models could be attempted, rather than completely freezing parameters.

## Related Work & Insights

- **vs. Traditional Animal Acoustic Classification**: Traditional methods often use hand-crafted features (such as MFCCs) with simple classifiers. Ours directly uses deep pre-trained embeddings, bypassing manual feature engineering.
- **vs. Human Speaker Verification**: Human speaker verification is a mature field (x-vector, ECAPA-TDNN). This work transfers such models to primate individual identification for the first time.
- **vs. Bird Models (e.g., BirdNET/Perch)**: These models are typically used for bird species identification. This work creatively applies them to cross-species individual-level discrimination tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of cross-species acoustic transfer is novel and inspiring, though the technical implementation is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐ As a short paper, the experimental scale is limited, but the core conclusions are clear.
- Writing Quality: ⭐⭐⭐⭐ The short paper is compactly structured, with well-formulated problem motivations and conclusions.
- Value: ⭐⭐⭐⭐ Direct application value for computational ecoacoustics and endangered species monitoring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Resounding Acoustic Fields with Reciprocity](../../NeurIPS2025/audio_speech/resounding_acoustic_fields_with_reciprocity.md)
- [\[ACL 2025\] Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals](predicting_turn-taking_and_backchannel_in_human-machine_conversations_using_ling.md)
- [\[CVPR 2025\] VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation](../../CVPR2025/audio_speech/vintage_joint_video_and_text_conditioning_for_holistic_audio_generation.md)
- [\[ICCV 2025\] Latent Swap Joint Diffusion for 2D Long-Form Latent Generation](../../ICCV2025/audio_speech/latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)
- [\[CVPR 2025\] Improving Sound Source Localization with Joint Slot Attention on Image and Audio](../../CVPR2025/audio_speech/improving_sound_source_localization_with_joint_slot_attention_on_image_and_audio.md)

</div>

<!-- RELATED:END -->
