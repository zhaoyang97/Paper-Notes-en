---
title: >-
  [Paper Note] HOP: Heterogeneous Topology-based Multimodal Entanglement for Co-Speech Gesture Generation
description: >-
  [CVPR 2025][Audio & Speech][Co-speech gesture generation] This paper proposes HOP, a heterogeneous topology-based multimodal entanglement method. By using audio as a bridge, it aligns audio-text semantics via a reprogramming module and audio-action rhythm via a spatio-temporal graph network. This achieves more natural and coherent co-speech gesture generation, reaching SOTA on FGD, BC, and diversity metrics.
tags:
  - "CVPR 2025"
  - "Audio & Speech"
  - "Co-speech gesture generation"
  - "multimodal entanglement"
  - "spatio-temporal graph modeling"
  - "cross-modal adaptation"
  - "reprogramming"
date: 2026-05-08
content_hash: 357c28fbf76804c1
---

# HOP: Heterogeneous Topology-based Multimodal Entanglement for Co-Speech Gesture Generation

**Conference**: CVPR 2025  
**arXiv**: [2503.01175](https://arxiv.org/abs/2503.01175)  
**Code**: [https://star-uu-wang.github.io/HOP/](https://star-uu-wang.github.io/HOP/)  
**Area**: Audio & Speech  
**Keywords**: Co-speech gesture generation, multimodal entanglement, spatio-temporal graph modeling, cross-modal adaptation, reprogramming

## TL;DR

This paper proposes HOP, a heterogeneous topology-based multimodal entanglement method. By using audio as a bridge, it aligns audio-text semantics via a reprogramming module and audio-action rhythm via a spatio-temporal graph network. This achieves more natural and coherent co-speech gesture generation, reaching SOTA on FGD, BC, and diversity metrics.

## Background & Motivation

1. **Background**: Co-speech gesture generation aims to automatically generate gesture motions synchronized with speech for virtual characters/embodied agents. Methods have evolved from early rule-based and probabilistic approaches to deep learning methods (GRU, GAN, diffusion models, etc.), typically employing multimodal fusion strategies to integrate text, audio, and gesture information.

2. **Limitations of Prior Work**: Existing multimodal methods usually assume that individual modalities are independent and decoupled, mapping them to latent spaces using different encoders before fusion. In practice, however, there is a natural, inherent dependency among speech, gestures, and text: spoken expressions influence gesture patterns and vice versa. Simple modal fusion approaches overlook this interdependence, resulting in a lack of coherence and expressiveness in generated gestures, where movements often appear robotic and rigid.

3. **Key Challenge**: There is an inherent heterogeneity among the three modalities (text, audio, and action). Text consists of discrete semantic signals, audio consists of continuous time-frequency signals, and action consists of spatial joint sequences. Direct fusion fails to capture the deep correlations among them.

4. **Goal** Explicitly model the topological interactions among the three modalities of text, audio, and action to generate natural gestures aligned with both semantic content and rhythmic features.

5. **Key Insight**: The authors observe that audio naturally encodes the rhythm of gestures and the semantics of text, serving as a bridge between text and action. Therefore, using audio as the hub, two cross-modal adaptation paths are established: audio-text (semantic alignment) and audio-action (rhythmic alignment).

6. **Core Idea**: Utilizing audio as a bridge to achieve audio-text semantic alignment via reprogramming + audio-action rhythmic alignment via spatio-temporal graphs = topologically entangled trimodal representations.

## Method

### Overall Architecture

The input consists of speech audio and the corresponding text transcription. After Mel-spectrogram extraction, the audio enters two paths: (1) it is "translated" into a text space representation via a reprogramming module and then sent into a pre-trained language model to extract semantic information; (2) it is jointly modeled with action features using the spatio-temporal graph network GraphWaveNet to capture rhythm and spatial dependencies. The features from both paths are topologically fused and then input into a GRU gesture generator, which combines speaker style features to generate the final gesture movements. Training employs a combined objective of Huber loss + style loss + KL divergence + adversarial loss.

### Key Designs

1. **Audio-Text Cross-Modal Adaptation (Reprogramming Module)**:

    - **Function**: "Reprogram" the audio Mel-spectrogram features into an input format compatible with pre-trained language models to achieve audio-text semantic alignment.
    - **Mechanism**: First, the Mel-spectrogram features $\mathbf{M}^{(t)} \in \mathbb{R}^{1 \times T}$ are used as queries, and the pre-trained word embeddings (compressed to a smaller vocabulary $V' \ll V$ via a linear layer) are used as keys and values. The alignment is calculated via multi-head cross-attention: $\hat{w}_{1:T} = \text{Linear}(\text{Softmax}(\frac{QK^{\top}}{\sqrt{d}})V)$. The reprogrammed audio features and text features are input together into a frozen language model (BERT) to extract deep semantic representations. As training progresses, audio and text features gradually align in the embedding space.
    - **Design Motivation**: Audio data cannot be directly represented in natural language, but utilizing the powerful reasoning capability of pre-trained language models allows for the extraction of deeper semantic information. The reprogramming module is applied to the co-speech gesture generation field for the first time, cleverly bypassing the modal incompatibility issue.

2. **Audio-Action Cross-Modal Adaptation (Spatio-Temporal Graph Network)**:

    - **Function**: Capture the spatial skeletal dependencies of gesture motions and the temporal rhythmic features of audio through spatio-temporal graph modeling.
    - **Mechanism**: Represent actions (skeletal orientation vectors) and audio as graph structures $\mathbf{G}=(v,e_1)$ and $\mathbf{R}=(v,e_2)$ respectively. In the spatial dimension, an adaptive adjacency matrix $\mathbf{A}_{adapted} = \text{SoftMax}(\text{ReLU}(\mathbf{E}_1 \odot \mathbf{E}_2^T))$ is used to learn implicit dependencies among skeletal joints. In the temporal dimension, dilated causal convolutions are employed to capture long-range rhythmic patterns. Drawing on the GraphWaveNet architecture, the model simultaneously processes graph-structured spatial relationships and WaveNet-style temporal modeling.
    - **Design Motivation**: Traditional methods only input the first few frames of ground-truth motions into the GRU, ignoring finer-grained motion features. The spatio-temporal graph approach simultaneously captures the spatial coordination among joints and the temporal synchrony between audio and action, generating more natural gestures.

3. **Topological Fusion and GAN Training**:

    - **Function**: Fuse the two feature paths (audio-text and audio-action) and generate the final gestures via a GAN.
    - **Mechanism**: The two cross-modal adapted features $\mathbf{Z}_{(w,r)}^t$ and $\mathbf{Z}_{(r,g)}^t$ are topologically fused at each time step. After adding the speaker style embedding, they are fed into a multi-layer bidirectional GRU network to generate gestures. A GAN discriminator is used to enhance the realism of the generated gestures.
    - **Design Motivation**: The GRU network is suitable for sequence generation tasks, GAN adversarial training enhances the realism and diversity of the output, and the speaker style embedding ensures personalization.

### Loss & Training

The total loss is $\mathcal{L}_{gesture} = \alpha \cdot \mathcal{L}_{Huber}(\mathbf{g}, \hat{\mathbf{g}}) + \beta \cdot \mathcal{L}_{style}(\mathbf{g}_{id}, \hat{\mathbf{g}}_{id'}) + \gamma \cdot \mathcal{L}_{KLD} + \lambda \cdot \mathcal{L}_{GAN}$. The Huber loss ensures motion accuracy, the style loss distinguishes different speakers, the KL divergence prevents the style embedding space from being too sparse, and the GAN loss enhances realism. Optimized using the Adam optimizer ($lr=0.0001, \beta=(0.5, 0.999)$), trained for 75 epochs on a single NVIDIA RTX 6000 Ada GPU.

## Key Experimental Results

### Main Results

**Comparison on TED Gesture and TED Expressive (Table 1)**:

| Method | TED Gesture FGD↓ | TED Gesture BC↑ | TED Gesture Diversity↑ | TED Expressive FGD↓ |
|------|-----------------|----------------|----------------------|-------------------|
| Trimodal | 3.729 | 0.667 | 101.247 | 12.613 |
| HA2G | 3.072 | 0.672 | 104.322 | 5.306 |
| DiffGesture | 1.506 | 0.699 | 106.722 | 2.600 |
| **HOP** | **1.406** | **0.762** | **108.176** | **1.815** |
| Ground Truth | 0 | 0.698 | 108.525 | 0 |

### Ablation Study

**Ablation of Model Components (Table 5)**:

| Configuration | FGD↓ | BC↑ | Diversity↑ |
|------|------|-----|-----------|
| w/o Graph Encoder | 2.026 | 0.650 | 103.311 |
| w/o Reprogramming | 1.721 | 0.755 | 105.360 |
| **Full model** | **1.406** | **0.762** | **108.176** |

**Ablation of Text Decoders (Table 4)**:

| Configuration | FGD↓ | BC↑ | Diversity↑ |
|------|------|-----|-----------|
| w/o Language Model | 1.955 | 0.701 | 105.311 |
| GPT-2 | 1.319 | 0.753 | 107.036 |
| BERT | 1.406 | 0.762 | 108.176 |

### Key Findings

- The Graph Encoder contributes the most: removing it causes the FGD to rise from 1.406 to 2.026 (+44%), demonstrating that spatio-temporal graph modeling is crucial for motion quality.
- The Reprogramming module contributes significantly to both FGD and Diversity, validating the effectiveness of audio-text semantic alignment.
- HOP maintains solid performance even when the data is reduced to 50% ($\text{FGD}=2.709$), whereas Trimodal drops to 7.364 under 50% data, indicating that topological fusion brings stronger generalization capabilities.
- On the BC metric, HOP (0.762) even surpasses Ground Truth (0.698); however, an excessively high BC might lead to unnatural, overly frequent movements, requiring a careful balance.
- In a user study with 26 evaluators, HOP is close to Ground Truth (4.16/4.39/4.28) in naturalness (3.92), semantic relevance (4.01), and synchrony (3.86).

## Highlights & Insights

- **Insight of "Audio as a Bridge"**: Audio naturally encodes semantic information (text content) and rhythmic information (motion timing) simultaneously. Serving as the middleware linking text and action is highly ingenious. This "intermediate modality bridging" concept can be transferred to other tri-modal tasks.
- **Reprogramming Technology for Cross-Modal Adaptation**: Introduces deep model reprogramming to gesture generation for the first time, "translating" audio into a format understandable by language models without fine-tuning the language model itself. This technique can be transferred to any scenario requiring frozen large models to handle heterogeneous inputs.
- **Progressive Data Reduction Experiments**: Table 3 demonstrates the graceful degradation of the model under progressively decreasing data volumes, which is an excellent method for evaluation of model robustness.

## Limitations & Future Work

- Currently, only upper-body gestures are evaluated (10/43 keypoints), without covering full-body motion or facial expressions.
- Operating with a GAN generator, which has been outperformed by diffusion models in several generation tasks. Under this light, combining the topological entanglement concept with diffusion models might yield better results.
- Having a BC that surpasses Ground Truth is not necessarily positive, as it might imply the model tends to generate excessively beat-synchronized movements.
- The reprogramming module relies on pre-trained word embeddings, leaving its adaptability to low-resource languages unknown.
- Explicit modeling of emotional information was not considered, although emotion has a significant impact on gesture styles.

## Related Work & Insights

- **vs Trimodal**: Trimodal also operates with text + audio + action tri-modal inputs, but encodes each modality independently before simple fusion. HOP explicitly models the topological relationship between modalities, reducing FGD from 3.729 to 1.406.
- **vs DiffGesture**: DiffGesture utilizes diffusion models to generate gestures, showing excellent performance but sometimes generating insufficiently diverse motions. HOP performs slightly better in FGD (1.406 vs 1.506) with a superior BC.
- **vs HA2G**: HA2G extracts hierarchical audio features but lacks deep semantic fusion of audio-text. HOP achieves finer-grained semantic understanding through the reprogramming module.

## Rating

- Novelty: ⭐⭐⭐⭐ The framework design of "audio as a bridge" and multi-modal topological entanglement is novel. The first application of reprogramming technology in gesture generation is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation on two datasets, user studies, multidimensional ablations, and generalization experiments are relatively complete, but comparisons with state-of-the-art methods like EMAGE are missing.
- Writing Quality: ⭐⭐⭐⭐ Concept presentation is clear, the derivation of the motivation for topological entanglement is reasonable, and figures/tables are intuitive.
- Value: ⭐⭐⭐⭐ Practical contribution to the gesture generation field; the ideas of "audio as a bridge" and reprogramming exhibit high transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Understanding Co-speech Gestures in-the-wild](../../ICCV2025/audio_speech/understanding_co-speech_gestures_in-the-wild.md)
- [\[CVPR 2025\] MultiFoley: Video-Guided Foley Sound Generation with Multimodal Controls](video-guided_foley_sound_generation_with_multimodal_controls.md)
- [\[CVPR 2025\] Contextual AD Narration with Interleaved Multimodal Sequence](contextual_ad_narration_with_interleaved_multimodal_sequence.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](../../NeurIPS2025/audio_speech/node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)
- [\[CVPR 2025\] DistinctAD: Distinctive Audio Description Generation in Contexts](distinctad_distinctive_audio_description_generation_in_contexts.md)

</div>

<!-- RELATED:END -->
