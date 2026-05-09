---
title: >-
  [Paper Note] Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos
description: >-
  [ICCV 2025][Robotics][Video Pre-training] This paper proposes Moto, a framework that encodes inter-frame visual motion from video into discrete sequences via unsupervised Latent Motion Tokens. A GPT-style autoregressive pre-training scheme is employed to learn motion priors, which are then transferred to real robot manipulation through a co-fine-tuning strategy. Moto achieves performance competitive with 55B-parameter models on the SIMPLER and CALVIN benchmarks using only 98M parameters.
tags:
  - ICCV 2025
  - Robotics
  - Video Pre-training
  - Motion Token
  - Autoregressive
  - Robot Manipulation
  - Cross-Embodiment Transfer
date: 2026-05-08
content_hash: bab1da7d4ba82f46
---

# Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos

**Conference**: ICCV 2025
**arXiv**: [2412.04445](https://arxiv.org/abs/2412.04445)
**Code**: [https://chenyi99.github.io/moto/](https://chenyi99.github.io/moto/)
**Area**: Robotics
**Keywords**: Video Pre-training, Motion Token, Autoregressive, Robot Manipulation, Cross-Embodiment Transfer

## TL;DR

This paper proposes Moto, a framework that encodes inter-frame visual motion from video into discrete sequences via unsupervised Latent Motion Tokens. A GPT-style autoregressive pre-training scheme is employed to learn motion priors, which are then transferred to real robot manipulation through a co-fine-tuning strategy. Moto achieves performance competitive with 55B-parameter models on the SIMPLER and CALVIN benchmarks using only 98M parameters.

## Background & Motivation

**State of the Field**: Large language models (LLMs) have achieved remarkable success in NLP through large-scale autoregressive pre-training. The robotics field has long been constrained by the high cost of action-annotated data. Video data contains rich interaction knowledge and is easy to acquire, yet effectively leveraging it for pre-training robot policies remains an open problem.

**Limitations of Prior Work**: Prior video pre-training methods (e.g., GR-1, MT-R3M) focus primarily on visual features of static frames, emphasizing frame-level details while neglecting inter-frame dynamics. These approaches typically require additional input modalities (e.g., gripper RGB, proprioception) to compensate for missing motion information. Existing VLAs (e.g., RT-2-X, OpenVLA) either require extremely large parameter counts (55B) or rely on action labels in pre-training data.

**Root Cause**: Video data is abundant but lacks action labels, while robot action data is scarce but annotated — the central challenge is how to exploit the vast motion knowledge in videos to improve robot policy learning under action-label scarcity.

**Paper Goals**: To identify an effective representation that encodes motion knowledge from video into sequences amenable to autoregressive pre-training, and that can be seamlessly transferred to downstream robot control.

**Starting Point**: Humans learn new skills by observing dynamic environmental changes, focusing on "motion" rather than static visual details. Motion information is closely related to low-level actions and is hardware-agnostic, facilitating cross-embodiment transfer.

**Core Idea**: A VQ-VAE encodes inter-frame motion information into discrete Latent Motion Tokens, which serve as a "motion language" for GPT-style autoregressive pre-training. A co-fine-tuning strategy then seamlessly converts the learned motion priors into real robot actions.

## Method

### Overall Architecture

Moto consists of three training stages:
1. **Latent Motion Tokenizer Training**: Unsupervised learning that compresses inter-frame motion between consecutive frames into 8 discrete tokens.
2. **Moto-GPT Autoregressive Pre-training**: Conditioned on an initial frame and language instruction, the model predicts the motion token sequence.
3. **Co-fine-tuning**: Joint fine-tuning on action-annotated robot data to transfer motion priors into precise robot control.

### Key Designs

1. **Latent Motion Tokenizer**:

    - **Function**: Encodes key visual motion between consecutive video frames into compact discrete tokens.
    - **Mechanism**: The encoder (M-Former) is a multi-layer Transformer that takes as input the patch features of the current frame $o_t$ and previous frame $o_{t-1}$ extracted by a frozen ViT, concatenated with 8 learnable query embeddings, and processed via self-attention. The query output features are quantized into discrete motion tokens via a VQ codebook (vocabulary size 128). The decoder (ViT Decoder) reconstructs the pixel values of $o_t$ from the patch embeddings of $o_{t-1}$ and a compact motion token embedding (compressed to 1 token via MLP and added to each patch embedding). Training uses the standard VQ-VAE objective (reconstruction MSE + VQ loss + commitment loss).
    - **Design Motivation**: Motion tokens act as an information bottleneck, forcing the encoder to retain only critical dynamic change information. Each frame is represented by only 8 tokens (vs. 196 original patch tokens), achieving a 24.5× compression ratio while preserving 79.7% of semantic classification accuracy.

2. **Moto-GPT Autoregressive Pre-training**:

    - **Function**: GPT-style pre-training on motion token sequences to learn motion priors from video.
    - **Mechanism**: Given a video clip $[o_0, o_1, ..., o_T]$, motion token chunks are extracted for each pair of adjacent frames and concatenated chronologically. A GPT Transformer uses frozen T5 text features $\boldsymbol{l}$ and frozen ViT initial-frame visual features $\boldsymbol{v}$ as a prefix, and is trained via next-token prediction: $\mathcal{L}_{motion} = -\sum_{i=1}^{M} \log P(m_i | \boldsymbol{l}, \boldsymbol{v}, \boldsymbol{m}_{<i}; \boldsymbol{\Theta})$, where $M = K \times T$ ($K=8$ tokens/frame, $T$ is video length).
    - **Design Motivation**: Using motion tokens rather than pixels or patches as the pre-training target focuses the model on learning "what to do" (motion intent) rather than "what to see" (visual details), which better aligns with the requirements of downstream control tasks.

3. **Co-fine-tuning Strategy**:

    - **Function**: Transfers pre-trained motion priors into precise robot actions.
    - **Mechanism**: $N$ learnable action query tokens (where $N$ corresponds to the number of actions between two frames) are appended after each motion token chunk at every timestep. Action queries are passed through an MLP action head to predict actions in the real action space (translation $\Delta x$, rotation $\Delta\theta$, gripper $\Delta grip$). Key design choices: (a) motion tokens do not attend to action queries (preserving consistency with pre-training); (b) 50% of attention from action queries to motion tokens is randomly masked (reducing dependence on GT conditioning); (c) the motion token prediction loss is retained. Total loss: $\mathcal{L}_{ft} = \mathcal{L}_{motion} + \mathcal{L}_{action}$, where $\mathcal{L}_{action} = \mathcal{L}(\Delta x) + \mathcal{L}(\Delta\theta) + \mathcal{L}(\Delta grip)$ (Smooth-L1 for continuous components, BCE for binary switching).
    - **Design Motivation**: Discarding motion tokens outright (as in Moto-DM) loses pre-trained knowledge; omitting the motion prediction loss (as in Moto-IML) leads to prior degradation. Co-fine-tuning allows action queries to directly acquire transferred knowledge from motion tokens via attention.

### Loss & Training

- Tokenizer: MSE reconstruction loss + VQ loss + commitment loss
- Pre-training: Cross-entropy next-motion-token prediction
- Fine-tuning: $\mathcal{L}_{ft} = \mathcal{L}_{motion} + \mathcal{L}_{action}$
- At inference, motion tokens can be replaced with padding tokens with attention blocked, enabling direct action output from action queries for improved efficiency.

## Key Experimental Results

### Main Results

SIMPLER benchmark (Google Everyday Robot, 3 task categories):

| Method | Params | Pick Coke Can | Move Near | Open/Close Drawer | Overall |
|--------|--------|---------------|-----------|-------------------|---------|
| RT-1-X | - | 0.567 | 0.317 | 0.597 | 0.534 |
| RT-2-X | **55B** | 0.787 | 0.779 | 0.250 | 0.607 |
| OpenVLA | 7B | 0.163 | 0.462 | 0.356 | 0.248 |
| OpenVLA (ft) | 7B | 0.363 | 0.542 | 0.231 | 0.349 |
| **Moto** | **98M** | **0.740** | **0.604** | **0.431** | **0.614** |
| Moto w/o MT | 98M | 0.503 | 0.554 | 0.398 | 0.480 |

CALVIN (ABC→D) zero-shot long-horizon task completion:

| Method | Input | 1 task | 2 tasks | 3 tasks | 4 tasks | 5 tasks | Avg. Len. |
|--------|-------|--------|---------|---------|---------|---------|-----------|
| GR-1 | RGB+Gripper+Proprio | 0.854 | 0.712 | 0.596 | 0.497 | 0.401 | 3.06 |
| SuSIE | RGB | 0.870 | 0.690 | 0.490 | 0.380 | 0.260 | 2.69 |
| **Moto** | **RGB** | **0.897** | **0.729** | **0.601** | **0.484** | **0.386** | **3.10** |
| Moto w/o MT | RGB | 0.779 | 0.555 | 0.380 | 0.256 | 0.167 | 2.14 |

### Ablation Study

Fine-tuning strategy ablation (CALVIN ABC→D):

| Configuration | Avg. Len.↑ | Note |
|---------------|-----------|------|
| Moto w/o Motion Token | 2.14 | Trained from scratch, no pre-trained prior |
| Moto-DM (motion tokens removed from input) | ~2.6 | Pre-trained but no motion tokens during fine-tuning |
| Moto-IML (motion prediction loss removed) | ~2.7 | Motion tokens retained but prediction objective dropped |
| **Moto (full co-fine-tuning)** | **3.10** | Motion tokens + prediction loss retained; best |

Semantic validity of motion tokens:

| Video Representation | Semantic Classification Accuracy |
|----------------------|----------------------------------|
| Initial frame only | 29.2% |
| Initial frame repeated 8 times | 28.3% |
| Initial frame + 7 subsequent frames (full patch) | 82.8% |
| Initial frame + 7 motion token chunks | **79.7%** |

### Key Findings

- **Motion priors contribute substantially**: Removing motion tokens (training from scratch) reduces SIMPLER Overall by 13.4% (0.614→0.480) and CALVIN Avg. Len. from 3.10 to 2.14 (a 31% drop).
- **98M parameters match a 55B model**: Moto achieves an Overall score of 0.614 on SIMPLER, slightly surpassing RT-2-X (0.607), using only single-view RGB.
- **Cross-embodiment transfer is effective**: Pre-training on human videos further improves performance — incorporating SSV2 human videos yields notable gains on the Move Near task.
- **High data efficiency**: Fine-tuning on only 1% of CALVIN annotated data achieves a 52.5% success rate (vs. 0% from scratch).
- Motion tokens exhibit cross-embodiment consistency: identical token chunks produce semantically consistent motion effects across different initial observations and robot embodiments.
- Pre-trained Moto-GPT can distinguish successful, failed, and random trajectories via log-likelihood, suggesting its potential as a reward signal.

## Highlights & Insights

- **The analogy of motion tokens as "language" is highly apt**: Just as natural language tokens are discrete, compositional, and semantically rich, motion tokens share these properties — they can be concatenated into trajectory "sentences," transferred across scenes, and support autoregressive generation.
- **The VQ information bottleneck design is elegant**: 8 tokens (vocabulary size 128) capture inter-frame motion in a form compact enough for long-sequence autoregression, yet expressive enough to retain 79.7% of semantic information. The decoder uses only a single conditioning embedding added to all patches, forcing motion information into a highly compressed representation.
- **The attention design in co-fine-tuning is carefully considered**: Motion tokens do not attend to action queries (preserving consistency) + 50% masking (reducing dependency) + retained motion prediction loss (preventing prior degradation) — each design decision is validated by ablation.
- **Practical significance**: This work provides a viable path toward training robots with internet-scale videos, requiring no action labels during pre-training.

## Limitations & Future Work

- Pre-training uses only 109K OXE videos, far below internet scale; the effect of scaling up data remains to be validated.
- The VQ codebook size (128) and token count (8) may need task-specific tuning.
- Evaluation is limited to simple pick/place/push/pull tasks; performance on dexterous manipulation (e.g., assembly, rope tying) has not been assessed.
- The frame reconstruction quality of the decoder is limited, which may constrain the expressive capacity of motion tokens.
- Real-world experiments are small in scale (30 demos and 10 trials per task), and success rate variance is not reported.
- Comparison with state-of-the-art VLAs (e.g., π0) is absent.

## Related Work & Insights

- **vs. GR-1**: GR-1 pre-trains by predicting future frame pixels and requires gripper RGB and proprioception as additional inputs. Moto achieves comparable performance using only static-camera RGB, demonstrating the advantage of motion-level pre-training over frame-level pre-training.
- **vs. Genie**: Genie also learns latent actions for 2D game simulators but does not transfer to real robot control. Moto completes the full pipeline from video to real robot execution.
- **vs. LAPA/IGOR**: LAPA predicts single-step future latent actions, while IGOR uses latent actions as intermediate goals. Moto autoregressively predicts the full motion token sequence of a trajectory, more naturally modeling continuous motion.
- The open-sourced codebase provides a foundation for future exploration; the motion token representation paradigm may inspire new pre-training paradigms for robotics.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The concept of Latent Motion Tokens and the three-stage training paradigm are highly innovative, providing a novel approach to video-to-robot knowledge transfer.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers SIMPLER + CALVIN + real-world + data efficiency + human video + multi-angle ablations, though real-world experiments are limited in scale.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The narrative is clear (token interpretability → prior learning → policy performance), with each experiment addressing a well-defined question.
- **Value**: ⭐⭐⭐⭐⭐ Proposes a potentially paradigm-shifting approach — large-scale pre-training with motion knowledge from unlabeled videos — substantially reducing the need for action annotation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning](../../CVPR2026/robotics/como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)
- [\[ICCV 2025\] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [\[ICCV 2025\] Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning](interaction-merged_motion_planning_effectively_leveraging_diverse_motion_dataset.md)
- [\[ICCV 2025\] Bridging Domain Generalization to Multimodal Domain Generalization via Unified Representations](bridging_domain_generalization_to_multimodal_domain_generalization_via_unified_r.md)
- [\[ICCV 2025\] iManip: Skill-Incremental Learning for Robotic Manipulation](imanip_skill-incremental_learning_for_robotic_manipulation.md)

<!-- RELATED:END -->
