---
title: >-
  [Paper Note] ScanTalk: 3D Talking Heads from Unregistered Scans
description: >-
  [ECCV 2024][Human Understanding][3D Talking Heads] ScanTalk is proposed, which is the first deep learning framework capable of generating audio-driven animations for 3D faces with **arbitrary topology** (including unregistered 3D scans). The core mechanism relies on the discretization-agnostic property of DiffusionNet to break the constraints of a fixed topology.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "3D Talking Heads"
  - "DiffusionNet"
  - "Audio-Driven Animation"
  - "Topology-Agnostic"
  - "3D Face Animation"
date: 2026-05-08
content_hash: 32e64b3064418227
---

# ScanTalk: 3D Talking Heads from Unregistered Scans

**Conference**: ECCV 2024  
**arXiv**: [2403.10942](https://arxiv.org/abs/2403.10942)  
**Code**: [https://github.com/miccunifi/ScanTalk](https://github.com/miccunifi/ScanTalk)  
**Area**: Human Understanding  
**Keywords**: 3D Talking Heads, DiffusionNet, Audio-Driven Animation, Topology-Agnostic, 3D Face Animation

## TL;DR

ScanTalk is proposed, which is the first deep learning framework capable of generating audio-driven animations for 3D faces with **arbitrary topology** (including unregistered 3D scans). The core mechanism relies on the discretization-agnostic property of DiffusionNet to break the constraints of a fixed topology.

## Background & Motivation

Audio-driven 3D talking head generation is a crucial research direction in computer vision and computer graphics, with wide applications in virtual reality, gaming, and other fields. However, existing methods (such as VOCA, FaceFormer, CodeTalker, SelfTalk, FaceDiffuser, etc.) are constrained by a **fixed topology**—the models can only animate meshes sharing the exact same number of vertices and connectivity as the training data.

This limitation results in:
1. Newly acquired 3D data must first be registered to a specific topology to be used, increasing preprocessing costs.
2. Inability to directly animate raw 3D scans.
3. A single model cannot be trained across datasets with different topologies.
4. Impeding real-time online applications (as the registration step is highly time-consuming).

ScanTalk aims to thoroughly address the topology-dependency issue, enabling arbitrary 3D faces (including raw scans) to be animated directly by audio.

## Method

### Overall Architecture

ScanTalk adopts an Encoder-Decoder architecture, taking a neutral face mesh $m_i^n$ and an audio clip $A_i$ as inputs, and outputting a sequence of per-vertex deformation fields. The core formulation is:

$$\text{ScanTalk}(A_i, m_i^n) \approx M_i^{gt}$$

The framework comprises two encoders (a face mesh encoder and an audio encoder) and a DiffusionNet decoder.

### Key Designs

1. **DiffusionNet Face Encoder**: DiffusionNet is adopted as the 3D face encoder. It computes intrinsic descriptors via precomputed surface operators (cotangent Laplacian, eigenbasis, mass matrix, spatial gradient matrix), inherently supporting meshes of different topologies. The encoding process is formulated as:

$$f_i^n = DN_e(m_i^n, P_i^n) \in \mathbb{R}^{V_i \times h}$$

where $P_i^n = OP(m_i^n)$ represents the precomputed surface features. DiffusionNet integrates MLPs, learned diffusion, and spatial gradient features without requiring explicit surface convolutions or pooling hierarchies, thereby achieving adaptability to arbitrary topologies.

2. **HuBERT Audio Encoder + BiLSTM**: A pretrained HuBERT model is used to extract speech representations, which are further enhanced by a multi-layer bidirectional LSTM to improve temporal consistency:

$$a_i = \text{SpeechEncoder}(A_i) \in \mathbb{R}^{T_i \times (h/2)}$$
$$v_i = \text{BiLSTM}(a_i) \in \mathbb{R}^{T_i \times h}$$

3. **DiffusionNet Decoder**: The per-vertex descriptor $f_i^n$ is concatenated with the temporal audio feature $v_i$ to form a joint representation $F_i^j \in \mathbb{R}^{V_i \times 2h}$. The deformation field is then predicted using the DiffusionNet decoder:

$$(F_i^j)_k = (f_i^n)_k \oplus v_i^j$$
$$\hat{m_i}^j = DN_d(F_i^j, P_i^n) + m_i^n$$

The model predicts the **deformation** of the neutral face rather than the full face geometry directly, reducing the difficulty of learning.

4. **Multi-Dataset Joint Training**: Due to its topology-agnostic nature, ScanTalk can learn from multiple datasets with different topologies simultaneously (e.g., VOCAset, BIWI6, Multiface), which was previously impossible with existing methods.

### Loss & Training

Training employs a per-vertex Mean Squared Error (MSE) loss:

$$\mathcal{L}_{MSE} = \frac{1}{T_i} \sum_{j=0}^{T_i-1} \frac{1}{V_i} \sum_{k=0}^{V_i-1} \|(m_i^j)_k - (\hat{m_i}^j)_k\|_2^2$$

Ablation studies show that in multi-dataset training scenarios, a simple $L_2$ loss outperforms combinations with mask loss or velocity loss, which is attributed to the significant geometric discrepancies among different datasets.

Training Details: DiffusionNet encoder/decoder both consist of 4 blocks, with a hidden dimension $h=32$, a 3-layer BiLSTM, optimized using Adam with a learning rate of $10^{-4}$ for 200 epochs.

## Key Experimental Results

### Main Results

| Dataset | Metric | ScanTalk (s-d) | ScanTalk (m-d) | CodeTalker | FaceDiffuser | SelfTalk |
|--------|------|---------------|----------------|------------|-------------|----------|
| VOCAset | LVE↓ | **3.012** | 6.375 | 3.549 | 4.350 | 5.618 |
| VOCAset | MVE↓ | **0.861** | 0.987 | 0.888 | 0.901 | 0.918 |
| BIWI6 | LVE↓ | 4.651 | **4.044** | 5.190 | 4.022 | 3.628 |
| BIWI6 | MVE↓ | 2.148 | **2.057** | 2.641 | 2.128 | 2.062 |
| Multiface | LVE↓ | 2.653 | **2.435** | 4.091 | 3.555 | 2.281 |
| Multiface | MVE↓ | 1.871 | **1.678** | 2.382 | 2.388 | 1.901 |

### Ablation Study

| Configuration | LVE↓ | MVE↓ | FDD↓ | Description |
|------|------|------|------|------|
| ScanTalk (HuBERT) | **3.012** | 0.861 | 2.400 | Best audio encoder |
| w/ Wav2Vec2 | 3.309 | **0.860** | 2.244 | Second best |
| w/ WavLM | 3.674 | 0.937 | 2.413 | Worst |
| w/ BiLSTM | **3.012** | 0.861 | 2.400 | Best temporal modeling |
| w/ BiGRU | 3.036 | **0.835** | 2.358 | Comparable performance |
| w/ TD (Transformer) | 3.291 | 0.859 | 2.406 | Weaker than RNN |
| w/o temporal | 3.361 | 0.870 | 2.365 | Without temporal module |

### Key Findings

- **Multi-dataset vs. Single-dataset**: Single-dataset training performs better on VOCAset (lip-only movement), whereas multi-dataset training is superior on BIWI6/Multiface (which include head motion)—this is because the model trained on multiple datasets observed more diverse head movement samples.
- **User Study**: ScanTalk is preferred by more users when compared against FaceFormer (82.67%) and FaceDiffuser (90.67%), but performs slightly below SelfTalk (44%) and Ground Truth (44%).
- **GPU Memory usage scales linearly** with the number of vertices, showing high scalability.

## Highlights & Insights

1. **First to achieve topology-agnostic 3D talking head generation**, breaking the long-standing fixed-topology limitation in this field.
2. **Ingenious Application of DiffusionNet**: Expanding its discretization-agnostic characteristics from static geometric analysis to a multimodal 4D dynamic setup.
3. **Multi-dataset Joint Training Capability**: A single model can be trained on multiple datasets with different topologies, which is unattainable for other methods.
4. The design choice of predicting deformations instead of full human faces simplifies the learning objective.

## Limitations & Future Work

1. Meshes without an open mouth cavity struggle to generate open-mouth animations (though lip synchronization remains accurate).
2. The FDD metric is sub-optimal under certain configurations, indicating room for improvement in upper-face expression generation.
3. The model is relatively simple (hidden dimension of only 32), suggesting that architectures with larger capacities could be explored.
4. Only MSE loss is utilized, without incorporating perceptual loss or adversarial training.
5. Input meshes require rigid alignment with the training data prior to inference.

## Related Work & Insights

- **DiffusionNet** [Sharp et al.]: A discretization-agnostic surface learning architecture, serving as the cornerstone of the proposed method.
- **Neural Face Rigging (NFR)**: The most closely related work, which uses DiffusionNet + Neural Jacobian Fields for animation transfer, but does not support audio drive.
- **HuBERT**: Self-supervised speech representation learning, which outperforms Wav2Vec2 in speech-to-face cross-modal mapping.
- Insight: The topology-agnostic nature of DiffusionNet can be generalized to other 3D tasks (e.g., expression transfer, hand gesture generation).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to achieve topology-agnostic 3D talking head generation, representing a breakthrough in formulation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Conducted across three datasets, various ablations, and a user study, but lacks a direct comparison with NFR.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, fully articulated motivation, and rich visualizations.
- **Value**: ⭐⭐⭐⭐ — Highly practical value, directly addressing primary pain points in the field, with open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos](avatar_fingerprinting_for_authorized_use_of_synthetic_talking-head_videos.md)
- [\[CVPR 2026\] Talking Together: Synthesizing Co-Located 3D Conversations from Audio](../../CVPR2026/human_understanding/talking_together_synthesizing_co-located_3d_conversations_from_audio.md)
- [\[ECCV 2024\] EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis](edtalk_efficient_disentanglement_for_emotional_talking_head_synthesis.md)
- [\[ECCV 2024\] Audio-Driven Talking Face Generation with Stabilized Synchronization Loss](audio-driven_talking_face_generation_with_stabilized_synchronization_loss.md)
- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)

</div>

<!-- RELATED:END -->
