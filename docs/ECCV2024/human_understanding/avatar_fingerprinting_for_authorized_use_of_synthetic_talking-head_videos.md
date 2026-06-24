---
title: >-
  [Paper Note] Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos
description: >-
  [ECCV 2024][Human Understanding][Avatar Fingerprinting] This paper defines a new task, "Avatar Fingerprinting," which verifies the true identity of the driver expressing emotions in a synthetic talking-head video. It contributes NVFAIR (161 identities), the largest facial reenactment dataset to date, and proposes a baseline method based on normalized facial landmark distances and a temporal CNN. By learning appearance-agnostic facial motion signatures…
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Avatar Fingerprinting"
  - "Facial Motion Signatures"
  - "Talking-Head Generation"
  - "Identity Verification"
  - "Deepfake"
date: 2026-05-08
content_hash: 9f92b5a493442c7d
---

# Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos

**Conference**: ECCV 2024  
**arXiv**: [2305.03713](https://arxiv.org/abs/2305.03713)  
**Code**: [https://research.nvidia.com/labs/nxp/avatar-fingerprinting/](https://research.nvidia.com/labs/nxp/avatar-fingerprinting/)  
**Area**: Human Understanding / Deepfake Detection  
**Keywords**: Avatar Fingerprinting, Facial Motion Signatures, Talking-Head Generation, Identity Verification, Deepfake

## TL;DR

This paper defines a new task, "Avatar Fingerprinting," which verifies the true identity of the driver expressing emotions in a synthetic talking-head video. It contributes NVFAIR (161 identities), the largest facial reenactment dataset to date, and proposes a baseline method based on normalized facial landmark distances and a temporal CNN. By learning appearance-agnostic facial motion signatures, the method achieves identity verification (average AUC of 0.85) and generalizes to unseen generators (AUC of 0.83).

## Background & Motivation

**Background**: Modern talking-head generators (e.g., face-vid2vid, LIA, TPS) can synthesize photorealistic, real-time talking-head videos, spawning next-generation communication experiences like video conferencing and AR/VR interaction. Users can drive another person's avatar with their own expressions for real-time communication, greatly reducing bandwidth requirements.

**Limitations of Prior Work**: (1) **Risk of Identity Misuse**—malicious users can use another person's facial appearance unauthorized to generate synthetic videos, leading to identity theft and deception. (2) **Limitations of Existing Deepfake Detection**—traditional Deepfake detection methods focus on "whether the image is synthetic" rather than "who is driving this synthetic video." Even if a video is detected as synthetic, it remains unknown whether the driver is authorized. (3) **Lack of Data and Benchmarks**—as a brand-new task direction, there are no suitable large-scale datasets and evaluation benchmarks.

**Key Challenge**: While talking-head generation is becoming increasingly popular and realistic, there is a lack of mechanism to verify "who is driving the avatar," which is a prerequisite for safe use of this technology. The appearance and the driver of the synthetic video can be entirely separated, rendering traditional (appearance-based) facial recognition completely ineffective.

**Goal**: (1) Define and formalize the "Avatar Fingerprinting" task; (2) Construct a large-scale paired dataset; (3) Propose a baseline method capable of identifying the driver's identity from facial motion.

**Key Insight**: Each individual's facial expression habits are unique—one's way of smiling, mouth movement patterns during speech, and blinking rhythms construct a "motion signature." This signature is independent of facial appearance and can serve as an identity fingerprint.

**Core Idea**: By learning appearance-agnostic facial motion signatures, the true identity of the driver expressing emotions in synthetic talking-head videos can be identified.

## Method

### Overall Architecture

The pipeline is: Input talking-head video clip $\rightarrow$ Extract facial landmarks frame-by-frame (126 landmarks) $\rightarrow$ Calculate normalized pairwise landmark distances to eliminate appearance factors $\rightarrow$ Concatenate multi-frame distance features along the temporal dimension $\rightarrow$ Feed into a Temporal Convolutional Network (Temporal CNN) $\rightarrow$ Output dynamic facial identity embedding vectors $\rightarrow$ Cluster embeddings of the same driver together via contrastive learning.

### Key Designs

1. **Normalized Pairwise Landmark Distance Features**:

    - **Function**: Extract motion features unrelated to facial appearance (shape, skin tone, facial proportions).
    - **Mechanism**: Detect 126 facial landmarks from each frame, compute pairwise Euclidean distances between all landmarks, and normalize them by the overall facial scale. This representation discards absolute position and scale information, retaining only the relative motion relations between facial parts. For instance, the pattern of distance change between the mouth corners and eye corners when a person smiles is unique, regardless of whose face they are using.
    - **Design Motivation**: In synthetic talking-head videos, the facial appearance belongs to the target identity, whereas the motion pattern belongs to the driver identity. By using normalized pairwise distances that only encode motion relations, these two types of information can be effectively separated to focus solely on the driver's motion signature.

2. **Temporal Convolutional Network (Temporal CNN)**:

    - **Function**: Aggregate multi-frame motion features into a compact identity embedding vector.
    - **Mechanism**: Concatenate the normalized pairwise distance features of consecutive frames along the temporal dimension to form a spatiotemporal feature tensor, which is then fed into a 1D convolutional network for temporal modeling. Convolution operations along the temporal dimension in the CNN capture temporal patterns of facial motion—such as the rhythm of mouth opening/closing during speech and the speed and amplitude of expression changes. The network outputs a fixed-dimensional dynamic facial identity embedding.
    - **Design Motivation**: Static single-frame features are insufficient to capture an individual's motion signature—human uniqueness is reflected in "how one moves" rather than "what one looks like at a single moment." Temporal modeling is key to extracting motion signatures.

3. **Dynamic Identity Embedding Contrastive Learning**:

    - **Function**: Learn an embedding space where videos of the same driver cluster together, and videos of different drivers are far apart.
    - **Mechanism**: Train the network using a contrastive learning loss. Positive pairs are defined as different video clips driven by the same person (regardless of whose appearance is used), while negative pairs are defined as video clips driven by different people. After training, measuring the Euclidean distance between two videos in the embedding space reveals whether they are driven by the same person. Crucially, self-reenactment (driving oneself with one's own face) and cross-reenactment (driving someone else's face with oneself) should yield similar embeddings.
    - **Design Motivation**: Contrastive learning is inherently suited for metric learning tasks, capable of learning flexible identity representations without a fixed number of classes. This enables the method to generalize to unseen new identities during training.

### Loss & Training

Training is conducted using contrastive loss or triplet loss. Within positive pairs, the anchor and positive sample come from the same driver (which can be over different target appearances), while negative samples come from different drivers. The training data is obtained from self-reenactment and cross-reenactment videos in the NVFAIR dataset.

## Key Experimental Results

### Main Results

Identity verification performance of avatar fingerprinting:

| Method/Setting | Metric | Ours | Note |
|----------|------|---------|------|
| Seen Generator (face-vid2vid) | AUC | 0.85 | Mean AUC |
| Unseen Generators (LIA, TPS) | AUC | 0.83 | Zero-shot generalization |
| Self-reenactment vs Cross-reenactment | Embedding Distance | Self-recon. dist. $\ll$ Cross-recon. dist. | Effective differentiation |
| Different Talking Scenarios (Free/Scripted) | AUC | Stable | Cross-scenario generalization |

### Ablation Study

| Configuration | AUC | Note |
|------|-----|------|
| Raw landmark coordinates | Lower | Contains appearance and position info |
| Normalized pairwise distances | **Optimal** | Effectively removes appearance factors |
| Single-frame features | Lower | Fails to capture dynamic signatures |
| Multi-frame + Temporal CNN | **Optimal** | Captures motion temporal patterns |
| Trained on face-vid2vid only | 0.85 (seen) / 0.83 (unseen) | Good generalizability |

### Key Findings

- Facial motion signatures are effective identity markers—each person's facial movement patterns are sufficiently unique to support identity verification.
- The method generalizes to generators unseen during training (declining only 2%, from 0.85 to 0.83), indicating that the extraction of motion signatures does not rely on motion generator-specific artifacts.
- Normalized pairwise distances are an effective means of removing appearance information.
- The NVFAIR dataset contains self-reenactment and cross-reenactment videos of 161 identities, providing an important benchmark for this field.

## Highlights & Insights

- **Definition of a New Task**: Avatar fingerprinting is an important yet unstudied problem that will become increasingly critical as talking-head technology gains popularity.
- **Biometric Value of Motion Signatures**: It demonstrates that facial motion patterns can act as a biometric feature for identity verification.
- **Substantial Dataset Contribution**: The NVFAIR dataset fills a gap in the field, incorporating natural interaction scenarios.
- **Cross-Generator Generalization**: The motion-based (rather than artifact-based) approach inherently possesses cross-generator generalization capabilities.

## Limitations & Future Work

- The AUC of 0.85/0.83 may not be high enough for security-critical scenarios, necessitating further improvements in accuracy.
- The method may be spoofed when a driver deliberately mimics another person's expression habits.
- Landmark detection itself might be inaccurate on synthetic videos, introducing noise.
- Relying only on landmark distances, other rich motion representations (e.g., optical flow, Action Units) could be explored.
- Although the dataset is the largest of its kind, it contains only 161 identities; the scale still needs to be further expanded.

## Related Work & Insights

- **Deepfake Detection**: Binary classification methods like FaceForensics++ only focus on real/fake classification and do not involve identity.
- **face-vid2vid** (NVIDIA): A high-quality facial reenactment method, which is one of the generators for datasets used in this work.
- **Face Recognition**: Appearance-based methods like ArcFace cannot handle scenarios in synthetic videos where appearance and driving are decoupled.
- **Insight**: In the era of AI-generated content, "who created the content" may be a more critical question than "whether the content is AI-generated."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Defines a brand-new task direction and fills a significant gap)
- Experimental Thoroughness: ⭐⭐⭐⭐ (The dataset and baseline method are complete, but there is still room for performance improvement)
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐ (Highly significant for the secure use of AI-generated content)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SyncDreamer: Controllable and Expressive Avatar Generation Beyond the Talking Head](../../CVPR2026/human_understanding/syncdreamer_controllable_and_expressive_avatar_generation_beyond_the_talking_hea.md)
- [\[ECCV 2024\] EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis](edtalk_efficient_disentanglement_for_emotional_talking_head_synthesis.md)
- [\[ECCV 2024\] VideoClusterNet: Self-Supervised and Adaptive Face Clustering for Videos](videoclusternet_self-supervised_and_adaptive_face_clustering_for_videos.md)
- [\[ECCV 2024\] ScanTalk: 3D Talking Heads from Unregistered Scans](scantalk_3d_talking_heads_from_unregistered_scans.md)
- [\[ECCV 2024\] 3DGazeNet: Generalizing 3D Gaze Estimation with Weak-Supervision from Synthetic Views](3dgazenet_generalizing_3d_gaze_estimation_with_weak-supervision_from_synthetic_v.md)

</div>

<!-- RELATED:END -->
