---
title: >-
  [Paper Note] DisMo: Disentangled Motion Representations for Open-World Motion Transfer
description: >-
  [NeurIPS 2025][Video Generation][motion disentanglement] DisMo learns abstract motion representations that are agnostic to appearance, pose…
tags:
  - "NeurIPS 2025"
  - "Video Generation"
  - "motion disentanglement"
  - "open-world motion transfer"
  - "flow matching"
  - "abstract motion representation"
  - "action classification"
date: 2026-05-08
content_hash: 8a01b4c70eb316f3
---

# DisMo: Disentangled Motion Representations for Open-World Motion Transfer

**Conference**: NeurIPS 2025
**arXiv**: [2511.23428](https://arxiv.org/abs/2511.23428)
**Code**: [https://compvis.github.io/DisMo](https://compvis.github.io/DisMo) (project page)
**Area**: Video Generation / Motion Transfer / Representation Learning
**Keywords**: motion disentanglement, open-world motion transfer, flow matching, abstract motion representation, action classification

## TL;DR
DisMo learns abstract motion representations that are agnostic to appearance, pose, and category from raw videos via a dual-stream architecture (motion extractor + frame generator) and an image-space reconstruction objective. It enables open-world motion transfer across categories and viewpoints, and significantly outperforms video representation models such as V-JEPA on zero-shot action classification.

## Background & Motivation
T2V/I2V generative models can produce realistic videos, but motion and content remain entangled—users cannot independently control *how* objects move. Existing motion control methods either rely on low-level pixel-space signals such as optical flow or trajectories (strongly coupled to the source object's structure, precluding cross-category transfer), or build on category-specific parametric models (e.g., facial landmark driving, applicable only within a single domain). Both families degrade severely when appearance, viewpoint, or semantic category differ substantially between source and target. Humans naturally decompose "motion" from appearance—a child who sees a person running can immediately imagine a fictional character doing the same. DisMo aims to learn this kind of abstract, category-agnostic motion representation.

## Core Problem
How can one learn motion representations from raw video that **encode only temporal dynamics, without encoding appearance, structure, or identity**? Such representations must: (1) transfer motion between semantically unrelated entities (e.g., human → ape, human → cartoon character); (2) be compatible with arbitrary off-the-shelf video generators; and (3) require no object correspondences or structural consistency.

## Method

### Overall Architecture
DisMo consists of two jointly trained components: a **motion extractor** $\mathcal{M}_\theta$ and a **frame generator** $\mathcal{F}_\psi$.

Input: video $\mathbf{X} = \{\mathbf{x}_t\}_{t=1}^T$ (subjected to strong data augmentation) → the motion extractor processes the augmented frame sequence and outputs a motion embedding sequence $\mathbf{M} = \{\mathbf{m}_t\}$. The frame generator is conditioned on a source frame $\mathbf{x}_t$ and the motion embedding $\mathbf{m}_t$ to reconstruct the future frame $\mathbf{x}_{t+\Delta t}$. The training loss is solely an image-space flow matching reconstruction objective—no contrastive loss, regularization, or complex training procedures are required.

Motion transfer supports two usage modes:
- **(b) Autoregressive transfer**: a lightweight scheme that directly applies the trained frame generator in a frame-by-frame manner.
- **(c) High-quality transfer**: motion embeddings are injected into a frozen off-the-shelf video generator (e.g., LTX-Video-2B) via LoRAdapter, where each spatiotemporal token is conditioned solely on the motion embedding of the corresponding timestep.

### Key Designs
1. **Information bottleneck for disentanglement**: The limited dimensionality of the motion embedding forms an information bottleneck. Since the frame generator receives both the source frame (providing appearance information) and the motion embedding (providing temporal dynamics), the motion embedding is compelled to encode only the residual knowledge beyond the source frame—i.e., the change from $\mathbf{x}_t$ to $\mathbf{x}_{t+\Delta t}$ (motion). Appearance, identity, and pose are already carried by the source frame, so the motion embedding has neither the need nor the capacity to redundantly encode them.

2. **Strong data augmentation pipeline**: Borrowing from self-supervised learning (MAE, SimCLR), photometric augmentations (brightness/contrast/hue/saturation) and geometric augmentations (crop/rotation/translation/shear/aspect ratio) are applied uniformly to all frames. Uniform application ensures that augmentations are not mistaken for motion, further forcing the motion embedding to disregard appearance details and focus on high-level temporal dynamics.

3. **Motion extractor architecture**: A 3D ViT-B (86M parameters) with DINOv2-B as the frame embedder, augmented with learnable motion query tokens $\mathbf{Q}$ that jointly process all frames and produce one motion embedding per timestep.

4. **LoRAdapter for plug-in integration with video models**: Conditional LoRA (rank=64) is used to fine-tune the attention and FFN layers of LTX-Video, with motion embeddings injected after passing through a mapping network (2-layer FFN + RMSNorm + Linear-GEGLU). Temporal alignment: 29 frames are sampled from 24 fps video; the motion extractor processes 8 frames (stride 4), which are pairwise concatenated to yield 4 embedding pairs aligned to LTX's 4 temporal positions.

### Loss & Training
The sole training objective is the flow matching MSE loss:

$$\mathcal{L}(\theta, \psi) = \mathbb{E}_{\mathbf{X}, \tau, \mathbf{z}_0 \sim \mathcal{N}(0,\mathbf{I}), (t,t+\Delta t)} \left\| \mathbf{v}_\psi(\mathbf{z}_\tau, \mathbf{x}_t, \mathcal{M}_\theta(\mathbf{X}, t), \tau) - \mathbf{u}(\mathbf{z}_\tau, \tau) \right\|_2^2$$

where $\mathbf{z}_\tau = \tau \mathbf{x}_{t+\Delta t} + (1-\tau)\mathbf{z}_0$. The motion extractor and frame generator are optimized end-to-end. The frame generator is initialized from a pretrained DiT-XL (675M parameters). Training runs for 530k steps with batch size 32 and AdamW. Training data: K-710 + SSv2 + Moments in Time + OpenVid-1M (approximately 2.8 million video clips totaling 4,900 hours). Dropout is applied to both motion and frame conditions during training to stabilize optimization and support unconditional reconstruction evaluation.

## Key Experimental Results

### Quantitative Comparison on Motion Transfer (Table 1)

| Method | Motion Fidelity ↑ | Prompt Adherence ↑ | Temporal Consistency ↑ | Driving Similarity ↓ |
|------|------|------|------|------|
| VMC* | 0.57 | 0.26 | 0.94 | 0.59 |
| DMT† | 0.70 | 0.24 | 0.93 | 0.66 |
| MotionClone† | 0.63 | 0.27 | 0.91 | 0.59 |
| MotionDirector* | 0.70 | 0.16 | 0.92 | 0.82 |
| **DisMo (Ours)** | **0.75** | **0.27** | **0.95** | **0.55** |

*Per-sample fine-tuning. †Inference-time optimization. DisMo achieves the best performance on all four metrics, notably without the motion fidelity vs. prompt adherence trade-off exhibited by other methods.

### Human Evaluation (Table 2)

| Method | Realism (%) | Prompt Matching (%) | Motion Transfer (%) |
|------|------|------|------|
| DMT | 10.93 | 9.60 | 17.73 |
| MotionDirector | 10.98 | 7.47 | 25.96 |
| VMC | 20.04 | 26.13 | 16.98 |
| MotionClone | 19.91 | 19.42 | 14.62 |
| **DisMo** | **38.13** | **37.38** | 24.71 |

DisMo leads substantially on realism and prompt matching (38.1% vs. 20% for the runner-up), with motion transfer quality comparable to MotionDirector.

### Zero-Shot Action Classification (Table 5, kNN probe)

| Method | Architecture | ARID ↑ | Jester ↑ | SSv2 ↑ | IARD ↑ |
|------|------|------|------|------|------|
| VideoMAE | ViT-L/16 | 17.29 | 20.11 | 7.06 | 73.44 |
| VideoMAEv2 | ViT-L/16 | 32.61 | 43.83 | 16.56 | 80.25 |
| V-JEPA | ViT-L/16 | 25.16 | 30.84 | 21.11 | 82.03 |
| **DisMo** | DisMo-B | **57.29** | **56.66** | **22.19** | **90.74** |

DisMo achieves large margins on motion-sensitive datasets: +32.1 points over V-JEPA on ARID and +25.8 points on Jester.

### Identity Disentanglement (Table 3, IARD dataset)

| Model | Action Accuracy ↑ | Identity Accuracy ↓ |
|------|------|------|
| VideoMAE | 73.44 | 99.14 |
| V-JEPA | 82.03 | 96.23 |
| **DisMo** | **90.74** | **23.82** |

The random baseline for five-class classification is 20%. DisMo's identity accuracy approaches chance level (23.82%), demonstrating that the motion embeddings contain almost no identity information.

### Ablation Study
- **Dual-stream conditioning (source frame) is central**: The baseline (no source frame, no augmentation) yields MIR=0.47; adding source frame conditioning raises MIR from 0.47 to 3.07 and LPIPS from 0.47 to 0.72. By supplying appearance information, the source frame forces the motion embedding to encode only the *change*.
- **Augmentation further improves disentanglement**: Adding augmentation raises MIR from 3.07 to 5.56 and LPIPS from 0.72 to 0.76, with particularly notable gains in invariance to geometric transformations.
- **Interchangeable video generation backbone**: Switching from LTX (Motion Fidelity 0.75, FID 88.5) to CogVideoX-5B (0.78, FID 63.0) directly improves generation quality without retraining the motion encoder, demonstrating that the motion representation is orthogonal to the renderer.
- **DisMo vs. V-JEPA disentanglement**: Under the same reconstruction network, DisMo achieves MIR=5.56 vs. V-JEPA's 3.72, indicating significantly stronger disentanglement.

## Highlights & Insights
- **Minimalist loss design**: Relying solely on an image-space reconstruction loss (flow matching MSE)—without contrastive loss, regularization, or complex training mechanisms—highly disentangled motion representations naturally emerge. The synergy of information bottleneck, strong augmentation, and dual-stream conditioning achieves disentanglement.
- **Motion representation orthogonal to renderer**: The motion representation is decoupled from the video generator and can be plugged into any off-the-shelf model via a lightweight LoRAdapter, allowing the method to directly benefit from future, more powerful generators—architecturally resolving the problem of method–model coupling.
- **Remarkable cross-domain generalization**: Human walking motion can be transferred to apes, cartoon characters, and even unrelated objects without requiring any object correspondences.
- **Substantially faster inference**: DisMo-LTX requires only 30 seconds per video, compared to 10 minutes for VMC and 7.5 minutes for DMT—a 10–20× speedup.
- **In-depth latent space analysis**: UMAP clusters clearly by action and shows no structure by identity; PCA visualizations reveal cyclic trajectories for periodic motions; reversible and irreversible motions are separable in the latent space. Motion composition experiments demonstrate that camera motion and object motion can be combined.

## Limitations & Future Work
- The flow matching frame generator has limited generative capacity in complex scenes; high-fidelity output depends on an external video model.
- Performance is constrained by training data distribution and biases; robustness to out-of-distribution samples may be limited.
- Only zero-shot kNN classification is evaluated; higher-order linear or attentive probing may reveal additional representational structure (identified by the authors as future work).
- The fixed motion embedding dimensionality leaves the representation capacity for very fine-grained motions (e.g., finger micro-movements) or very long motion sequences unexplored.
- Transfer quality remains bounded by the capability of the target video model.

## Related Work & Insights
- **vs. VMC/MotionDirector** (per-sample fine-tuning methods): These methods are extremely slow at inference (5–10 min/video) and overfit to the source structure, exhibiting a trade-off between motion fidelity and prompt adherence. DisMo is a universal feed-forward model that achieves the best results on both metrics simultaneously.
- **vs. DMT/MotionClone** (training-free methods): These methods exploit aggregated priors from pretrained models for motion transfer, but lack explicit motion representations, making cross-category transfer across large semantic gaps difficult. DisMo's explicit motion representations naturally support cross-category transfer.
- **vs. FOMM/MRAA/LIA** (parametric methods): These methods rely on category-specific keypoints or structural representations and are confined to the training domain (e.g., human faces), with no generalization to open-world settings. DisMo is entirely category-agnostic.
- **vs. V-JEPA** (video representation): V-JEPA's representations heavily encode appearance (identity classification accuracy 96%), whereas DisMo's representations contain almost no identity information (24%), comprehensively outperforming V-JEPA on motion-sensitive tasks.
- **Motion representations as reusable features**: DisMo's motion embeddings can serve directly as features for downstream video understanding tasks, inspiring a two-stage paradigm of "learn motion representations first, then apply to downstream tasks"—analogous to the role of CLIP in vision-language settings.
- **Complementarity with FlashMotion (CVPR 2026)**: FlashMotion targets accelerated inference for trajectory-level motion control, while DisMo focuses on abstract semantic-level motion representation. The two can be combined by using DisMo's motion embeddings as a replacement for trajectory signals to drive FlashMotion's accelerated generator.
- **Transferability of the augmentation + information bottleneck disentanglement paradigm**: Beyond the motion domain, any scenario requiring disentanglement of "change" from "content" (e.g., style transfer, expression transfer, illumination change estimation) can draw on this "dual-stream + bottleneck + strong augmentation" design.

## Rating
- Novelty: ⭐⭐⭐⭐ First to propose abstract motion representation learning driven purely by a reconstruction objective; the information bottleneck disentanglement mechanism is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers quantitative generation transfer + human evaluation + disentanglement analysis + zero-shot classification + ablations + multi-backbone evaluation + latent space analysis; exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear logical structure, well-motivated exposition, and rich appendix material.
- Value: ⭐⭐⭐⭐ Provides a practical and scalable paradigm for open-world motion control; the motion representation simultaneously serves both generation and understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning](../../ICCV2025/video_generation/disentangled_world_models_learning_to_transfer_semantic_knowledge_from_distracti.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](../../ICCV2025/video_generation/efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [\[ICCV 2025\] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](../../ICCV2025/video_generation/decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)
- [\[ICCV 2025\] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](../../ICCV2025/video_generation/motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [\[CVPR 2026\] Let Your Image Move with Your Motion! – Implicit Multi-Object Multi-Motion Transfer](../../CVPR2026/video_generation/let_your_image_move_with_your_motion_--_implicit_multi-object_multi-motion_trans.md)

</div>

<!-- RELATED:END -->
