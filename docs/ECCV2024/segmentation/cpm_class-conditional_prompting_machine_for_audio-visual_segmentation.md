---
title: >-
  [Paper Note] CPM: Class-Conditional Prompting Machine for Audio-Visual Segmentation
description: >-
  [ECCV 2024][Segmentation][Audio-Visual Segmentation] This paper proposes the Class-Conditional Prompting Machine (CPM), which enhances the stability of bipartite matching and the effectiveness of cross-modal attention in Mask2Former for audio-visual segmentation by combining class-agnostic queries with GMM-sampled class-conditional queries. Simultaneously, three auxiliary tasks are designed—Audio-Conditional Prompting (ACP), Visual-Conditional Prompting (VCP)…
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Audio-Visual Segmentation"
  - "Class-Conditional Prompting"
  - "Bipartite Matching"
  - "Mask2Former"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: 6e73e40ddc7d9b82
---

# CPM: Class-Conditional Prompting Machine for Audio-Visual Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2407.05358](https://arxiv.org/abs/2407.05358)  
**Code**: None  
**Area**: Audio-Visual Segmentation / Multimodal Learning  
**Keywords**: Audio-Visual Segmentation, Class-Conditional Prompting, Bipartite Matching, Mask2Former, Contrastive Learning

## TL;DR

This paper proposes the Class-Conditional Prompting Machine (CPM), which enhances the stability of bipartite matching and the effectiveness of cross-modal attention in Mask2Former for audio-visual segmentation by combining class-agnostic queries with GMM-sampled class-conditional queries. Simultaneously, three auxiliary tasks are designed—Audio-Conditional Prompting (ACP), Visual-Conditional Prompting (VCP), and Prompt Contrastive Learning (PCL)—achieving state-of-the-art performance on AVSBench and VPO benchmarks.

## Background & Motivation

Audio-Visual Segmentation (AVS) aims to localize and segment sounding objects based on audio-visual cues, with the core challenge lying in effective cross-modal interaction.

**Limitations of Prior Work**:

- **Pixel-classification-based methods** (e.g., TPAVI, CAVP): These use early fusion and FCN decoders, which underutilize audio due to its lower information density compared to vision. They also struggle to capture instance-level information, leading to inconsistent segmentation across frames.
- **Transformer-based methods** (e.g., Mask2Former-like): While theoretically better suited for multimodal tasks, they face two key training difficulties:
  - **Low effectiveness of cross-attention**: The global audio features of mixed sound sources lack clear semantics, making attention learning difficult.
  - **Unstable bipartite matching**: Class-agnostic queries lack semantic guidance, causing oscillations during the matching process.

**Key Insight**: If queries themselves carry semantic information of categories, it can stabilize bipartite matching (since queries already know what to look for) and provide clearer cross-modal attention signals.

## Method

### Overall Architecture

CPM is based on the Mask2Former architecture and alternates between two pathways during training:
- **Class-agnostic pathway (used during inference)**: Standard class-agnostic queries generate mask predictions via a Transformer decoder and Hungarian matching.
- **CPM pathway (training-only)**: Class-conditional queries sampled from a GMM are processed through three auxiliary tasks: ACP, VCP, and PCL.

### Key Design 1: Class-Conditional Distribution Modeling (CCDM)

A Gaussian Mixture Model (GMM) is used to model the mask embedding distribution for each class $c$:

$$p(\tilde{q} | c) = \sum_m \pi_{c,m} \mathcal{N}(\tilde{q}; \mu_{c,m}, \Sigma_{c,m})$$

- During training, the Hungarian-matched mask embeddings and their corresponding labels are collected.
- The EM algorithm is used to optimize GMM parameters, with momentum updates ensuring stability.
- During inference, Bayes' rule replaces the Softmax classifier.

**Function**: Class-conditional queries $z^k$ are sampled from the GMM. These queries naturally carry category semantics, bypassing the instability of bipartite matching.

### Key Design 2: Audio-Conditional Prompting (ACP)

Inspired by the mix-and-separate concept, an audio denoising task is designed:

1. Take training audio $a_i$ and mix it with out-of-screen noise $a_j$: $a_p = a_i + a_j$.
2. Use class-conditional queries as decoder inputs to retrieve semantically similar sound sources on the mixed audio feature map.
3. Predict the spectrogram mask and align it with the ground-truth ratio.

$$\mathcal{L}_{\text{ACP}} = \left\| \sigma\left(\sum_k m_k^a\right) - \frac{a_i}{a_p} \right\|_2$$

**Function**: Forces class-conditional queries to learn to distinguish different sound sources in the frequency domain, enhancing fine-grained understanding of the audio modality.

### Key Design 3: Visual-Conditional Prompting (VCP)

Directly replace class-agnostic queries with sampled class-conditional queries as input to the Transformer decoder:
- Since the class corresponding to the query is known, bipartite matching is bypassed.
- Training target: Correctly segment the corresponding image area and correctly classify.

$$\mathcal{L}_{\text{VCP}} = \mathcal{L}_{\text{ce}} + \mathcal{L}_{\text{mask}}$$

**Function**: Provides a stable per-class training signal to alleviate matching oscillations of class-agnostic queries.

### Key Design 4: Prompt Contrastive Learning (PCL)

Utilizes the spectrogram saliency mask produced by ACP to extract class-level audio features. These features are then used as anchors to conduct InfoNCE contrastive learning against pixels of the same class (positive samples) and other classes (negative samples) in the visual feature map.

**Breakthrough**: Compared to prior methods that only perform contrastive learning using global audio, CPM is the first to achieve dense class-level audio-visual contrastive learning.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_{\text{agn}} + \lambda \mathcal{L}_{\text{CPM}}$, where $\mathcal{L}_{\text{CPM}} = \mathcal{L}_{\text{ACP}} + \mathcal{L}_{\text{VCP}} + \mathcal{L}_{\text{PCL}}$

- $\mathcal{L}_{\text{agn}}$: Standard Mask2Former loss (Hungarian matching + cross-entropy + focal + dice).
- The CPM branch is activated only during training; at inference, only the class-agnostic pathway is executed, incurring no additional inference cost.
- GMM parameters are updated via momentum to maintain training stability.

## Key Experimental Results

### Main Results: AVSBench Test Set (224x224, ResNet-50)

| Method | Type | SS mIoU | MS mIoU | AVSS mIoU |
|:---:|:---:|:---:|:---:|:---:|
| TPAVI | Per-pixel | 78.80 | 52.84 | 22.69 |
| CAVP | Per-pixel | 83.84 | 61.48 | 32.83 |
| AVSegFormer | Transformer | 80.67 | 56.17 | 27.12 |
| COMBO | Transformer | 85.90 | 60.55 | 35.30 |
| **CPM** | **Transformer** | **85.92** | **65.40** | **37.05** |

CPM shows a larger advantage in multi-source (MS) and semantic (AVSS) scenarios: MS +2.83%, AVSS +1.79%.

### Original Resolution AVSBench-Semantics

| Method | SS mIoU | MS mIoU | Total mIoU |
|:---:|:---:|:---:|:---:|
| CAVP | 56.91 | 38.61 | 50.75 |
| AVSegFormer* | 50.52 | 31.40 | 45.80 |
| **CPM** | **61.71** | **43.11** | **57.25** |

CPM brings significant improvements: SS +4.80, MS +4.50, Total +6.50 mIoU.

### Ablation Study

| Component | Contribution Details |
|:---:|:---|
| CCDM (GMM Modeling) | Provides the foundation of class-conditional queries for ACP/VCP/PCL |
| ACP | Enhances fine-grained sound source separation capability in the audio modality |
| VCP | Bypasses Hungarian matching to provide a stable training signal |
| PCL | Dense class-level audio-visual contrastive learning to enhance cross-modal alignment |

### Key Findings

1. **CPM yields the largest gain in multi-source scenarios**: When multiple sound sources are present, global audio features are unreliable; class-conditional prompts effectively disentangle mixed sources.
2. **The mix-and-separate strategy of ACP promotes audio understanding**: It is more effective than directly using global audio as queries.
3. **VCP stabilizes training**: Class-conditional queries naturally bypass the instability of Hungarian matching.
4. **PCL achieves the first class-level audio-visual contrastive learning**: Bypassing the limitation of prior works that could only perform contrastive learning using global audio.

## Highlights & Insights

1. **Training-inference decoupling**: The CPM branch is only involved in training, introducing zero overhead during inference.
2. **GMM generative classifier**: A generative model replaces the discriminative Softmax, capturing intra-class variations more effectively.
3. **Exquisitely designed triple auxiliary tasks**: ACP, VCP, and PCL reinforce learning from audio, visual, and cross-modal dimensions respectively.
4. **Scalability**: CPM can be integrated into any Mask2Former-based segmentation framework.

## Limitations & Future Work

1. The EM update of GMM increases training complexity, requiring the maintenance of an external memory bank.
2. ACP relies on an out-of-screen noise dataset, introducing additional data collection costs.
3. It is not fully validated on stronger backbones (e.g., Swin-L).
4. The quality of predicted spectrogram masks directly affects PCL, but there is a lack of analysis regarding the quality of this intermediate result.
5. The utilization of temporal video information has not been explored.

## Related Work & Insights

- **Mask2Former / DETR**: CPM is built upon the Mask2Former architecture and addresses its training instability issues.
- **CAVP**: Previous generation AVS SOTA, using global audio contrastive learning; CPM extends it to the class level.
- **DN-DETR**: A pioneer in stabilizing bipartite matching via denoising, inspiring the concept of bypassing matching in CPM.
- **Mix-and-Separate**: A classic sound source separation paradigm in the audio domain, ingeniously introduced into AVS by ACP.

## Rating

- **Novelty**: 4/5 - The combination of GMM and class-conditional prompting is highly original in the context of AVS.
- **Experimental Thoroughness**: 4/5 - Comprehensive evaluation across multiple benchmarks, though ablation details could be more extensive.
- **Writing Quality**: 4/5 - Clear structure with an informative Figure 2.
- **Value**: 4/5 - Provides a systematic solution to the training challenges of Transformer-based AVS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment](../../CVPR2025/segmentation/robust_audio-visual_segmentation_via_audio-guided_visual_convergent_alignment.md)
- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](../../ICCV2025/segmentation/implicit_counterfactual_learning_for_audio-visual_segmentation.md)
- [\[CVPR 2025\] Audio-Visual Instance Segmentation](../../CVPR2025/segmentation/audio-visual_instance_segmentation.md)
- [\[CVPR 2025\] Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics](../../CVPR2025/segmentation/dynamic_derivation_and_elimination_audio_visual_segmentation_with_enhanced_audio.md)
- [\[ECCV 2024\] Cs2K: Class-Specific and Class-Shared Knowledge Guidance for Incremental Semantic Segmentation](cs2k_class-specific_and_class-shared_knowledge_guidance_for_incremental_semantic.md)

</div>

<!-- RELATED:END -->
