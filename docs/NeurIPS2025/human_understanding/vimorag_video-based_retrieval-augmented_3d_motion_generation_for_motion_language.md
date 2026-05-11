---
title: >-
  [Paper Note] VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models
description: >-
  [NeurIPS 2025][Human Understanding][Motion Generation] This paper proposes VimoRAG, a framework that leverages large-scale in-the-wild video databases as 2D motion priors to enhance 3D motion generation. Two core bottlen…
tags:
  - "NeurIPS 2025"
  - "Human Understanding"
  - "Motion Generation"
  - "Retrieval-Augmented Generation"
  - "Video Priors"
  - "Motion Language Model"
  - "DPO"
date: 2026-05-08
content_hash: 8550f33cd6aa0476
---

# VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2508.12081](https://arxiv.org/abs/2508.12081)
**Code**: [GitHub](https://walkermitty.github.io/VimoRAG/)
**Area**: Human Understanding / Human-Computer Interaction
**Keywords**: Motion Generation, Retrieval-Augmented Generation, Video Priors, Motion Language Model, DPO

## TL;DR

This paper proposes VimoRAG, a framework that leverages large-scale in-the-wild video databases as 2D motion priors to enhance 3D motion generation. Two core bottlenecks—human motion video retrieval and error propagation—are addressed via the Gemini-MVR retriever and the McDPO training strategy.

## Background & Motivation

- **Background**: Generating diverse and realistic 3D human motions from text has broad applications in gaming, robotics, and VR. Motion Language Models (Motion LLMs) unify motion understanding and generation within an LLM framework, but suffer from severe **out-of-distribution (OOD) / out-of-vocabulary (OOV)** problems—existing text–motion paired datasets contain only ~14K samples with prohibitively high annotation costs.
- **Limitations of Prior Work**: ReMoDiffuse proposed retrieval-augmented generation from a 3D motion database, but that database itself is limited to 14K samples. In contrast, in-the-wild videos are nearly unlimited in scale, rich in motion diversity, and 2D human motions in videos share intrinsically similar features with 3D motions.
- **Key Challenge**: Video-based motion RAG faces two key challenges:
  - **Retrieval difficulty**: Existing video foundation models (VFMs), while strong at recognizing objects and attributes, perform poorly at discriminating human poses and actions.
  - **Error propagation**: When retrieval quality is low, inaccurate video priors severely degrade generation quality.

## Method

### Overall Architecture

VimoRAG is a two-stage pipeline: (1) given a motion description text, Gemini-MVR retrieves the semantically most relevant video (rank-1) from an unannotated video database; (2) both the text and the retrieved video are fed into an LLM to generate motion tokens, which are then decoded into a motion sequence by a VQ-VAE.

**Video Database HcVD**: Aggregates 425,988 human-centric videos from IDEA400, Kinetics, UCF101, NTU, and other datasets. Text descriptions are synthesized with Qwen2-VL (used only for retriever training), and videos without detected human bodies are filtered out using AlphaPose.

### Key Designs

1. **Gemini Motion Video Retriever (Gemini-MVR)**

   A dual-channel retrieval architecture is designed:
   - **Action-level retriever**: Extracts 2D human keypoints from video via a pretrained AlphaPose detector and MotionBERT encoder to obtain frame-level features; these are combined with positional embeddings and passed through a Transformer temporal encoder (with residual connections) to produce an action embedding $\mathbf{a}$. On the text side, a predicate semantic extractor $\theta_\mathcal{P}$ initialized from the InternVideo text encoder produces embedding $\mathbf{p}$. Trained with contrastive loss: $\mathcal{L}_{action} = \mathcal{L}_{p2a} + \mathcal{L}_{a2p}$.
   - **Object-level retriever**: Directly adopts InternVideo as the VFM, leveraging the rich general knowledge acquired during large-scale pretraining.
   - **Action-aware router $\mathcal{I}$**: A lightweight linear model that adaptively assigns weights to the two retrievers based on the action embedding:

   $s(t,v) = \frac{\mathcal{I}_0(\mathbf{a}) \cdot s(\mathbf{p},\mathbf{a})}{\mathcal{I}_0(\mathbf{a})+\mathcal{I}_1(\mathbf{a})} + \frac{\mathcal{I}_1(\mathbf{a}) \cdot s(\mathbf{g},\mathbf{o})}{\mathcal{I}_0(\mathbf{a})+\mathcal{I}_1(\mathbf{a})}$

   Training proceeds in two stages: Stage 1 fine-tunes the two retrievers separately; Stage 2 freezes the retrievers and trains only the router.

2. **Motion-centric Dual-alignment DPO Trainer (McDPO)**

   The LLM is trained in two stages:
   - **Stage 1 — Visual demonstration-enhanced instruction tuning**: The text $x$, retrieved video $v$, and system prompt are concatenated and fed into the LLM; motion tokens $y$ encoded by VQ-VAE serve as the target under standard autoregressive loss $\mathcal{L}_{sft} = -\sum_n \log p_\theta(y_n | y_{<n}, E^f)$.
   - **Stage 2 — Dual-alignment DPO training**: Given the Stage 1 baseline model $\pi_{ref}$, $\kappa$ candidate motions are sampled. A **dual-alignment reward model** is defined as:

   $r(x,v,\hat{y_i}) = -\left(w_\ell \frac{\ell(\hat{y_i}, y)}{\sum_{j\in\kappa}\ell(\hat{y}_j, y)} + w_d \frac{d(\hat{y}_i, x)}{\sum_{j\in\kappa}d(\hat{y}_j, x)}\right)$

   where $\ell(\cdot)$ measures distributional distance in the motion feature space (intra-motion alignment) and $d(\cdot)$ measures Euclidean distance in the text–motion semantic space (cross-modal alignment). Preferred and rejected samples are selected accordingly to construct a DPO dataset, and the model is trained with the standard DPO objective.

   **Design Motivation**: To teach the LLM when to leverage and when to ignore the prior information in retrieved videos—automatically reducing reliance on priors when retrieval quality is poor.

### Loss & Training

- Retriever: Contrastive learning loss (InfoNCE)
- Generator Stage 1: Autoregressive SFT loss
- Generator Stage 2: DPO objective
- Backbone LLM: Phi3-3.8B, LoRA fine-tuning throughout (rank=128, α=256)

## Key Experimental Results

### Main Results

| Model | Backbone | FID↓ | R-Top1↑ | R-Top3↑ | MM-Dist↓ |
|------|------|:---:|:---:|:---:|:---:|
| MotionGPT (Phi3) | Phi3-3.8B | 0.501 | 0.396 | 0.673 | 3.724 |
| **VimoRAG** | Phi3-3.8B | **0.131** | 0.452 | 0.764 | 3.146 |
| Gain | - | −73% | +14% | +13% | −15% |
| MoMask | - | 0.048 | 0.519 | 0.809 | 2.955 |
| BiPO | - | 0.030 | 0.523 | 0.809 | 2.880 |

Zero-shot cross-domain evaluation (IDEA400):

| Model | FID↓ | R-Top3↑ | MM-Dist↓ |
|------|:---:|:---:|:---:|
| MotionGPT (LLM) | 5.544 | 0.236 | 6.300 |
| MLD | 5.410 | 0.270 | 6.005 |
| **VimoRAG** | **2.388** | 0.270 | 5.888 |

### Ablation Study

| Configuration | FID↓ | Note |
|------|:---:|------|
| Gemini-MVR + McDPO (full) | 0.148 | Best |
| Random retrieval + McDPO | 0.544 (↓72.8%) | Random retrieval greatly degrades quality |
| InternVideo + McDPO | 0.205 (↓27.8%) | Gemini-MVR outperforms general VFM |
| Gemini-MVR (w/o McDPO) | 0.260 (↓43.1%) | McDPO effectively mitigates error propagation |

Retriever comparison (R@1):

| Retriever | Human Video Set | Single-person Video Set |
|--------|:---:|:---:|
| InternVideo | 53.6 | 52.3 |
| Gemini-MVR | 58.3 (+8.8%) | 61.0 (+16.6%) |

### Key Findings

- VimoRAG achieves the best FID of 2.388 in OOD settings (IDEA400), substantially outperforming all motion expert models and LLM-based methods.
- McDPO enables the model to distinguish informative from uninformative video priors—performance does not severely degrade even when random videos are provided as input.
- FID and MM-Dist continue to decrease as the retrieval database scales up, demonstrating strong scalability.
- Under the same backbone (Phi3-3.8B), VimoRAG reduces FID by 73%, substantially surpassing the naive MotionGPT baseline.

## Highlights & Insights

- This work is the first to propose a video-based motion RAG paradigm, breaking the bottleneck imposed by the limited scale of 3D motion data.
- The dual-channel retriever design is elegant: the action-level channel focuses on human pose, while the object-level channel exploits the general knowledge of VFMs, with adaptive weight allocation between the two.
- McDPO is a practical robustness-enhancement strategy that enables the generative model to self-correct when faced with noisy retrieval results.
- The retrieval database is infinitely scalable, and performance improves continuously with its growth—a property of significant practical value.

## Limitations & Future Work

- The LLM-based framework incurs high inference latency compared to motion expert models.
- Information loss persists in the modality gap between 2D video priors and 3D motion.
- Only the rank-1 video is currently used; exploring top-$k$ multi-video fusion is a promising direction.
- Future work may unify video, 3D data, and images into a single multimodal RAG framework.

## Related Work & Insights

- Compared to ReMoDiffuse (text-to-text retrieval, constrained by 3D database scale), VimoRAG represents a paradigm shift from 3D Motion RAG to video-based RAG.
- The application of DPO to motion generation offers alignment insights transferable to other generative tasks.
- The video–motion cross-modal alignment strategy is generalizable to other visual prior-enhanced generation scenarios.
- The keypoint-aware routing mechanism can inspire the design of multimodal retrieval systems more broadly.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Pioneers the video retrieval-augmented motion generation paradigm; both Gemini-MVR and McDPO are original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ In-domain and OOD experiments are comprehensive with detailed ablations, though comparisons with more LLM backbones are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear and framework diagrams are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Opens a new direction for leveraging massive video data to enhance motion generation, with strong scalability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MOSPA: Human Motion Generation Driven by Spatial Audio](mospa_human_motion_generation_driven_by_spatial_audio.md)
- [\[CVPR 2026\] MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation](../../CVPR2026/human_understanding/molingo_motion-language_alignment_for_text-to-motion_generation.md)
- [\[ICCV 2025\] GestureHYDRA: Semantic Co-speech Gesture Synthesis via Hybrid Modality Diffusion Transformer and Cascaded-Synchronized Retrieval-Augmented Generation](../../ICCV2025/human_understanding/gesturehydra_semantic_co-speech_gesture_synthesis_via_hybrid_modality_diffusion_.md)
- [\[ICCV 2025\] Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator](../../ICCV2025/human_understanding/signs_as_tokens_a_retrieval-enhanced_multilingual_sign_language_generator.md)
- [\[NeurIPS 2025\] BEDLAM2.0: Synthetic Humans and Cameras in Motion](bedlam20_synthetic_humans_and_cameras_in_motion.md)

</div>

<!-- RELATED:END -->
