---
title: >-
  [Paper Note] Quantifying and Narrowing the Unknown: Interactive Text-to-Video Retrieval via Uncertainty Minimization
description: >-
  [ICCV 2025][Video Generation][Text-to-video retrieval] This paper proposes UMIVR, a framework that explicitly quantifies three types of uncertainty in text-to-video retrieval—textual ambiguity (semantic entropy), mapping uncertainty (JS divergence), and frame uncertainty (temporal quality frame sampling)—and adaptively generates clarification questions based on the quantified uncertainty to iteratively refine queries, achieving 69.2% R@1 on MSR-VTT-1k after 10 interaction rounds.
tags:
  - ICCV 2025
  - Video Generation
  - Text-to-video retrieval
  - uncertainty quantification
  - interactive retrieval
  - semantic entropy
  - frame quality sampling
date: 2026-05-08
content_hash: 06432aa3f0633c84
---

# Quantifying and Narrowing the Unknown: Interactive Text-to-Video Retrieval via Uncertainty Minimization

**Conference**: ICCV 2025
**arXiv**: [2507.15504](https://arxiv.org/abs/2507.15504)
**Code**: [GitHub](https://github.com/bingqingzhang/umivr)
**Area**: Video Generation
**Keywords**: Text-to-video retrieval, uncertainty quantification, interactive retrieval, semantic entropy, frame quality sampling

## TL;DR
This paper proposes UMIVR, a framework that explicitly quantifies three types of uncertainty in text-to-video retrieval—textual ambiguity (semantic entropy), mapping uncertainty (JS divergence), and frame uncertainty (temporal quality frame sampling)—and adaptively generates clarification questions based on the quantified uncertainty to iteratively refine queries, achieving 69.2% R@1 on MSR-VTT-1k after 10 interaction rounds.

## Background & Motivation

1. **State of the Field**: Text-to-video retrieval (TVR) has evolved from attention mechanisms to vision-language pretraining, with methods such as CLIP4Clip and HunYuan continuously advancing performance. Interactive retrieval systems refine user intent through clarification questions.
2. **Limitations of Prior Work**: TVR faces three categories of uncertainty: (1) textual ambiguity—queries that are vague, incomplete, or polysemous; (2) mapping uncertainty—even unambiguous queries may match multiple visually similar videos; (3) frame uncertainty—low-quality frames caused by motion blur or defocus that obscure critical visual cues. Existing interactive methods rely on heuristic question generation without explicitly quantifying these uncertainties.
3. **Root Cause**: Different types of uncertainty require distinct intervention strategies, yet prior methods apply homogeneous strategies to heterogeneous uncertainties.
4. **Paper Goals**: How to explicitly quantify different types of uncertainty and adaptively generate the most effective clarification questions accordingly?
5. **Starting Point**: Each of the three uncertainty types is mapped to a computable mathematical measure (semantic entropy, JS divergence, and image quality assessment), all without requiring any training.
6. **Core Idea**: Semantic entropy quantifies textual ambiguity, JS divergence quantifies mapping uncertainty, and temporal quality frame sampling mitigates frame uncertainty; these measures jointly drive hierarchical, adaptive clarification question generation.

## Method

### Overall Architecture
UMIVR adopts a unified VideoLLaVA architecture to simultaneously support video retrieval, caption generation, video question answering, and clarification question generation. The pipeline consists of offline preprocessing (TQFS frame sampling + caption generation) followed by online interaction (uncertainty quantification → adaptive question generation → user feedback → query refinement → retrieval).

### Key Designs

1. **Text Ambiguity Score (TAS)**:
    - Function: Quantifies the semantic uncertainty of a query.
    - Mechanism: Captions are generated for videos in the retrieval corpus and encoded. Given query $x$, the top-$K$ most similar captions are retrieved and clustered into $M$ groups $\{c_j\}$. The cluster probability $p(c_j|x)$ is computed as the proportion of similarity mass assigned to each group. Semantic entropy is defined as $SE(x) = -\sum_{j=1}^{M} p(c_j|x) \log p(c_j|x)$, normalized to $[0,1]$. A high TAS indicates that the query's semantics are dispersed.
    - Design Motivation: Compared to token-level heuristics, semantic entropy is computed after semantic-level aggregation, avoiding inflated scores due to lexical variation. Clustering merges near-synonymous expressions into the same semantic cluster.

2. **Mapping Uncertainty Score (MUS)**:
    - Function: Quantifies the sharpness of the similarity distribution between a query and candidate videos.
    - Mechanism: The top-$k$ similarity scores are taken, mean-centered, and square-normalized to yield distribution $p$: $p_i = \frac{\max(s_i - \bar{s}, 0)^2}{\sum \max(s_j - \bar{s}, 0)^2}$. An ideal one-hot distribution $q$ (with mass on the top candidate) is defined. JS divergence is then computed as $MUS(x) = \frac{JSD(p \| q)}{JSD_{\max}}$. A high MUS indicates a flat score distribution, making it difficult to distinguish the most relevant video.
    - Design Motivation: JS divergence is bounded and symmetric, making it more robust than KL divergence. Square normalization emphasizes high-confidence candidates while suppressing low-confidence noise.

3. **Temporal Quality Frame Sampler (TQFS)**:
    - Function: A plug-and-play module that selects high-quality, temporally diverse video frames.
    - Mechanism: A three-step strategy is employed: (1) uniformly sample $N$ frames at a low frame rate; (2) score each frame using a no-reference image quality metric $Q(\cdot)$ (e.g., Laplacian variance or BRISQUE); (3) divide the video into $M$ temporal segments and select the highest-quality frame per segment $F_m^* = \arg\max_{F_i \in \mathcal{I}_m} Q(F_i)$; (4) extract semantic embeddings from the candidate frames, apply K-means clustering, select the highest-quality frame per cluster, and sort by time to obtain the final $K$ frames.
    - Design Motivation: Uniform sampling may include blurry frames. TQFS jointly considers visual clarity and semantic diversity and can serve as a plugin for any TVR model.

### Loss & Training
UMIVR is a **training-free** framework—TAS, MUS, and TQFS are all computed at inference time without any parameter updates. The unified VideoLLaVA architecture uses pretrained weights for retrieval, captioning, and QA. The adaptive question generation strategy follows: high TAS → open-ended clarification questions (requesting descriptions of appearance or activity); low TAS + high MUS → targeted discriminative questions (leveraging metadata of candidate videos); both low → enrichment questions (soliciting additional descriptive details).

## Key Experimental Results

### Main Results

| Method | R@1↑ | R@5↑ | Hit@1↑ | MnR↓ |
|--------|------|------|----------|------|
| HunYuan (non-interactive SOTA) | 62.9 | 84.5 | 62.9 | 9.3 |
| UMIVR round 0 | 43.1 | 66.1 | 43.1 | 22.4 |
| UMIVR round 3 | 61.3 | 84.1 | 68.9 | 8.1 |
| UMIVR round 6 | 65.9 | 87.7 | 76.0 | 5.9 |
| UMIVR round 10 | **69.2** | **89.0** | **80.0+** | — |

Three interaction rounds suffice to surpass the non-interactive SOTA; six rounds surpass all prior methods.

### Ablation Study

| Configuration | R@1 (round 3) | Note |
|------|---------|------|
| Full UMIVR | 61.3 | Complete framework |
| w/o TAS | Drops | No textual ambiguity awareness |
| w/o MUS | Drops | No mapping uncertainty awareness |
| w/o TQFS | Drops | Uniform frame sampling |
| TQFS as plug-in (CLIP4Clip) | Improves | TQFS enhances other models as a plugin |

### Key Findings
- Three interaction rounds already outperform most non-interactive methods, demonstrating the efficiency of uncertainty-driven questioning.
- TQFS as a standalone module improves R@1 of baselines such as CLIP4Clip by 4–5%.
- UMIVR generalizes directly to interactive text-to-image retrieval scenarios.
- User responses can be simulated via the VideoQA module or collected from real users.

## Highlights & Insights
- The explicit decomposition and independent quantification of three uncertainty types is the core contribution: each type is addressed with a distinct mathematical tool.
- The hierarchical decision tree combining TAS (semantic entropy) and MUS (JS divergence) is concise and theoretically elegant.
- The plug-and-play nature of TQFS endows it with independent practical value.
- Replacing a traditional multi-model pipeline with a single VideoLLaVA model substantially simplifies the system architecture.

## Limitations & Future Work
- The retrieval performance of VideoLLaVA as the backbone is relatively low at initialization (round 0: 43.1% R@1).
- Simulated user responses may deviate from real user behavior.
- Ten interaction rounds may be excessive for practical deployment; the performance ceiling under fewer rounds warrants investigation.
- The choice of cluster number $M$ in semantic entropy affects TAS accuracy.

## Related Work & Insights
- **vs. PlugIR**: A ChatGPT-based hybrid cloud approach with high computational overhead.
- **vs. TAM/UATVR**: Only considers a single type of uncertainty.
- **vs. D2V**: An interactive baseline that does not explicitly quantify uncertainty.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of uncertainty categorization and quantification exhibits theoretical elegance.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on four datasets (MSR-VTT, AVSD, MSVD, ActivityNet) with TQFS transfer experiments.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear and the methodology is rigorous.
- Value: ⭐⭐⭐⭐ The training-free, plug-and-play uncertainty quantification has broad applicability to interactive retrieval systems.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control](realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)
- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](stiv_scalable_text_and_image_conditioned_video_generation.md)
- [\[ICCV 2025\] BadVideo: Stealthy Backdoor Attack against Text-to-Video Generation](badvideo_stealthy_backdoor_attack_against_text-to-video_generation.md)
- [\[NeurIPS 2025\] Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](../../NeurIPS2025/video_generation/autoregressive_adversarial_posttraining_for_realtime_interac.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)

<!-- RELATED:END -->
