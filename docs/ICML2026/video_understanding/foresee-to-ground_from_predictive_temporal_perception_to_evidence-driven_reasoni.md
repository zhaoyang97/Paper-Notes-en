---
title: >-
  [Paper Note] Foresee-to-Ground: From Predictive Temporal Perception to Evidence-Driven Reasoning
description: >-
  [ICML 2026][Video Understanding][Video Temporal Grounding] Foresee-to-Ground (F2G) reframes Video Temporal Grounding (VTG) from direct timestamp regression to an "Identify-then-Measure" two-stage problem. By constructing…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Video Temporal Grounding"
  - "Video-LLM"
  - "Evidence Pool"
  - "Identify-then-Measure"
  - "Boundary Detection"
date: 2026-05-08
content_hash: f7877bac68f606f9
---

# Foresee-to-Ground: From Predictive Temporal Perception to Evidence-Driven Reasoning

**Conference**: ICML 2026  
**arXiv**: [2605.21973](https://arxiv.org/abs/2605.21973)  
**Code**: To be confirmed  
**Area**: Video Understanding / Multimodal VLM / Video Temporal Grounding  
**Keywords**: Video Temporal Grounding, Video-LLM, Evidence Pool, Identify-then-Measure, Boundary Detection

## TL;DR
Foresee-to-Ground (F2G) reframes Video Temporal Grounding (VTG) from direct timestamp regression to an "Identify-then-Measure" two-stage problem. By constructing a candidate event evidence pool via predictive temporal perception and a span evidence encoder, the LLM generates boundaries captured under specific event constraints. This improves Charades-STA R@0.7 by 4.1 points and ActivityNet by 6.7 points.

## Background & Motivation

**Background**: When applying Video-LLMs to VTG, the dominant approach is to directly regress timestamps from flattened visual token sequences, effectively performing a black-box mapping between discrete token space and continuous time domains.

**Limitations of Prior Work**: Direct timestamp regression faces two core issues:
- **Numerical Fragility**: The discrete token representation of LLMs is naturally misaligned with continuous time coordinates, leading to unstable timestamp prediction and significant boundary noise.
- **Lack of Verifiability**: Models cannot provide explicit evidence for predictions, making it difficult for users to understand why a specific time segment was chosen.

**Key Challenge**: Existing methods attempt to alleviate issues through timestamp discretization or temporal cue injection, but fundamentally operate within a black-box regression framework. They overlook the human cognitive process of temporal localization—making an explicit event commitment (Identification) before refining boundaries (Measurement).

**Goal**: Reformulate VTG as a verifiable structured prediction problem, enabling the model to (1) explicitly select candidate events from an evidence pool (Identification) and (2) precisely locate boundaries under the constraint of that event hypothesis (Measurement).

**Key Insight**: Introduce the human "identify then measure" cognitive workflow into the model by constructing an explicit evidence pool across the video scope. Each candidate segment is represented as a discrete unit citeable by the LLM, binding timestamp generation to specific event hypotheses.

**Core Idea**: Through a design consisting of "predictive temporal perception + evidence-driven reasoning," VTG is transformed from unconstrained numerical regression into evidence-supported citation-conditional reasoning.

## Method

### Overall Architecture
F2G models VTG as a three-stage structured prediction:
$$p(A, T, z \mid V, Q, \mathcal{S}_K(V)) = p(z \mid V, Q, \mathcal{S}_K(V)) \cdot p(A, T \mid z, V, Q, \mathcal{S}_K(V))$$
Where $V$ is the video, $Q$ is the query, $T = (t^{st}, t^{ed})$ is the predicted interval, $A$ is the answer, and $z \in \{1, \ldots, K\}$ is the index of the candidate segment selected from the evidence pool $\mathcal{S}_K(V)$. The first term implements Identification, and the second implements Measurement.

The Three-Stage Curriculum:
- Stage-1 (**Predictive Temporal Perception**): Unsupervised pre-training of the temporal module to learn boundary-sensitive features.
- Stage-2 (**Proposal Warm-up**): Supervised training of a lightweight proposal head to extract Top-K candidates and encode local evidence.
- Stage-3 (**Evidence-Driven Reasoning**): Fine-tuning the Video-LLM for supervised two-stage "Identify-then-Measure" generation.

### Key Designs

1.  **Predictive Temporal Perception**:
    - **Function**: Learn feature representations capable of inferring global dynamics from partial temporal evidence, forcing the network to highlight event boundaries and transition signals.
    - **Mechanism**: Given a temporal feature sequence $X \in \mathbb{R}^{N \times D}$, construct a global view (full sequence) and multiple local views (partial sequences). By minimizing the latent prediction loss between local and global views $\mathcal{L}_{\text{pred}} = \mathbb{E}[\sum_{v \in \mathcal{V}} \|\text{sg}(U_g) - \hat{U}_g^{(v)}\|_2^2]$, the shared temporal backbone is forced to encode features that make global dynamics predictable from local segments. Predictability is high within coherent events but low at boundaries where partial evidence could lead to various outcomes—increasing the loss and automatically identifying boundaries. Sliced Isotropic Gaussian Regularization (SIGReg) is introduced to stabilize latent geometry.
    - **Design Motivation**: Overcome numerical instability of direct regression; pre-training on unlabeled data discovers temporal representations of event segments through self-supervised prediction.

2.  **Span Evidence Encoder (SEE)**:
    - **Function**: Aggregate temporal features from candidate segments into fixed-length visual evidence embeddings for LLM citation.
    - **Mechanism**: For each candidate $T_k$, crop the temporal sequence to get intra-segment features $U_k = \text{Crop}(U, T_k) \in \mathbb{R}^{N_k \times D}$. Use $M$ learnable query tokens through stacked Multi-Head Cross-Attention (Q-Former style) to aggregate: $P_k = \text{SEE}(U_k) = \text{MHCAStack}(B, U_k) \in \mathbb{R}^{M \times D}$.
    - **Design Motivation**: Event segments of varying lengths must be represented as equal-length tokens for the LLM; soft aggregation via cross-attention is more expressive than simple pooling.

3.  **Evidence-Driven Identify-then-Measure**:
    - **Function**: Constrain Video-LLM decoding—first explicitly cite an evidence ID, then generate timestamps and the answer under that constraint.
    - **Mechanism**: Stage-3 injects the evidence pool $\mathcal{S}_K(V) = \{(\langle\text{Span}_k\rangle, T_k, P_k)\}_{k=1}^K$ into the LLM context (each containing a discrete ID, coarse interval, and visual tokens). The model generates an ID token (identifying a specific event) before producing the final refined timestamp. Three losses $\mathcal{L}_{S3} = \mathcal{L}_{LM} + \alpha \mathcal{L}_{id} + \beta \mathcal{L}_{\text{time}}$ supervise sequence generation, ID prediction, and timestamp regression.
    - **Design Motivation**: Shifts boundary prediction from black-box regression over the whole video stream to local refinement under a specific event hypothesis; explicit ID citation ensures traceability.

### Loss & Training
- Stage-1: Pre-training on unlabeled videos using multi-view latent prediction + SIGReg.
- Stage-2: Training the proposal head on a 70K VTG labeled set (regression + scoring loss to align proposal quality).
- Stage-3: LoRA fine-tuning of the Video-LLM on 220K instruction data, keeping the temporal module and proposal head trainable with a lower learning rate; a lightweight proposal loss maintains evidence pool quality.

## Key Experimental Results

### Main Results

| Dataset | Metric | Qwen3-VL (baseline) | +FT | **+F2G-FT** | Gain |
|---------|--------|----------------------|-----|-------------|------|
| Charades-STA | R@0.7 | 15.9% | 21.6% | **25.7%** | +4.1 |
| Charades-STA | mIoU | 40.4 | 42.9 | **47.2** | +4.3 |
| ActivityNet-Captions | R@0.7 | 17.3% | 21.7% | **28.4%** | +6.7 |
| ActivityNet-Captions | mIoU | 32.2 | 40.8 | **45.7** | +4.9 |
| QVHighlights | mAP | 21.3 | 24.6 | **29.7** | +5.1 |
| QVHighlights | HIT@1 | 32.6% | 36.8% | **45.6%** | +8.8 |

### Ablation Study

| Configuration | Charades-STA R@0.7 | ActivityNet mIoU | Description |
|---------------|-------------------|------------------|-------------|
| F2G Full | 25.7% | 45.7 | Full Model |
| w/o SIGReg | 24.1% | 44.2 | Removed geometric regularization, -1.6 |
| w/o Stage-1 | 20.9% | 41.8 | No pre-training, -4.8 |
| w/o ID Citation | 21.5% | 41.1 | Removed ID constraint, -4.2 |
| w/o Visual Evidence | 22.1% | 41.5 | Using only intervals without visual tokens, -3.6 |

### Key Findings
- Stage-1 pre-training and SIGReg are critical; their removal leads to a 4-5 point drop, especially at high IoU thresholds.
- Evidence citation (ID constraint) provides the largest gain (~3-4%), as explicit event commitment significantly improves stability.
- Stable cross-model transfer: The same F2G-FT scheme applied to LLaVA and Qwen2.5 backbones consistently yields a +3-9% mIoU improvement.
- Stability Analysis: F2G's $|\Delta\text{IoU}|$ distribution (between two independent decodes) is more concentrated around 0; variance is much lower than the baseline, proving evidence constraints reduce inference instability.

## Highlights & Insights
- **Simplicity of Paradigm Shift**: Identify-then-Measure aligns with human cognition and naturally solves numerical stability; it is transferable to other localization-heavy tasks (spatial detection, dense captioning).
- **Ingenuity of Multi-View Latent Prediction**: Uses predictability differences between global and local views to automatically learn boundary features without explicit labels—an elegant self-supervised signal.
- **Modularity and Transferability**: The three-stage pipeline is decoupled and easily adapts to various Video-LLM backbones.
- **Low Computational Cost**: Adds only 0.5B parameters (~6% relative to an 8B model), with inference latency < 5%, and evidence serialization only adds 100-200 tokens.

## Limitations & Future Work
- Accuracy is capped by the evidence pool quality—if the ground truth event is not in the Top-K candidates, the LLM will fail.
- $K$-value sensitivity: Currently fixed at Top-8, which may require adaptation for extremely long videos (hours).
- Domain generalization: Training data mixes DiDeMo/ActivityNet; performance in disparate domains like News/Sports remains untested.
- Future directions: (1) Dynamic/Recursive evidence pools for multi-round refinement; (2) Uncertainty estimation to support rejection; (3) RL fine-tuning of Stage-3 using IoU rewards.

## Related Work & Insights
- **vs. TimeChat / VTimeLLM**: These methods improve within the direct regression framework (injecting cues, discretizing time), but remain unconstrained; F2G makes reasoning controllable through evidence constraints.
- **vs. Self-supervised Video Representation**: Prior works focused on transfer learning; F2G innovatively applies predictive pre-training directly to event discovery for VTG.
- **vs. Dense Video Captioning**: Both involve event localization; the evidence pool concept in F2G can be adapted to captioning systems for traceable event descriptions.

## Rating
- Novelty: ⭐⭐⭐⭐ Identify-then-Measure is a sensible new perspective; multi-view prediction for boundary learning is also novel despite individual components not being radical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 VTG benchmarks + cross-backbone validation + comprehensive ablations + stability evidence.
- Writing Quality: ⭐⭐⭐⭐ Logical and easy to follow; methodology is clear, though some discussions could go deeper.
- Value: ⭐⭐⭐⭐⭐ High practical value for VTG; F2G's versatility suggests it will be adopted and extended by future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](../../CVPR2026/video_understanding/a_multi-agent_perception-action_alliance_for_efficient_long_video_reasoning.md)
- [\[ICML 2026\] SkelHCC: A Hyperbolic CLIP-Driven Cache Adaptation Framework for Skeleton-based One-Shot Action Recognition](skelhcc_a_hyperbolic_clip-driven_cache_adaptation_framework_for_skeleton-based_o.md)
- [\[ICML 2026\] VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority](videoseal_mitigating_evidence_misalignment_in_agentic_long_video_understanding_b.md)
- [\[ACL 2026\] TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](../../ACL2026/video_understanding/temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)
- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](../../ICCV2025/video_understanding/vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)

</div>

<!-- RELATED:END -->
