---
title: >-
  [Paper Note] Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator
description: >-
  [ICCV 2025][Human Understanding][Sign language generation] This paper proposes SOKE, a multilingual sign language generation framework built upon pretrained language models. It discretizes continuous sign language motion into token sequences via a decoupled tokenizer, and achieves high-quality text-to-3D-avatar sign language generation across multiple languages through multi-head decoding and retrieval-augmented strategies.
tags:
  - ICCV 2025
  - Human Understanding
  - Sign language generation
  - multilingual sign language
  - autoregressive language model
  - retrieval-augmented generation
  - motion discretization
date: 2026-05-08
content_hash: 114128c29283cecc
---

# Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator

**Conference**: ICCV 2025
**arXiv**: [2411.17799](https://arxiv.org/abs/2411.17799)
**Code**: [Project Page](https://2000zrl.github.io/soke/)
**Area**: Human Understanding
**Keywords**: Sign language generation, multilingual sign language, autoregressive language model, retrieval-augmented generation, motion discretization

## TL;DR

This paper proposes SOKE, a multilingual sign language generation framework built upon pretrained language models. It discretizes continuous sign language motion into token sequences via a decoupled tokenizer, and achieves high-quality text-to-3D-avatar sign language generation across multiple languages through multi-head decoding and retrieval-augmented strategies.

## Background & Motivation

Sign language is the primary communication modality for Deaf and hard-of-hearing communities and possesses all linguistic characteristics of natural language (discrete semantic units, grammatical structure). Sign language processing encompasses two main directions: Sign-to-Text (SLT) and Text-to-Sign (SLG). Although pretrained language models have achieved notable success in SLT, progress in SLG remains **severely insufficient**.

Key limitations of existing SLG methods:

**Neglect of sign language's linguistic nature**: Most methods treat SLG as a visual content generation task (video/keypoints/motion) using GANs or diffusion models, failing to leverage the generalizability and scalability of pretrained LMs.

**Gloss dependency**: Traditional methods rely on glosses (written representations of signs) as intermediate representations, which require extensive annotation and introduce information bottlenecks.

**Monolingual limitation**: Existing methods typically handle only one sign language (e.g., ASL), lacking a unified multilingual model.

**Inefficient decoding**: When using decoupled tokenizers, flattening tokens from multiple body parts into a single sequence multiplies the number of decoding steps.

**Insufficient word-level precision**: Sentence-level generation struggles to guarantee the accuracy of individual sign words.

These issues collectively result in deficiencies in quality, efficiency, and coverage of existing SLG methods.

## Method

### Overall Architecture

SOKE consists of two stages: (1) a **Decoupled Tokenizer (DETO)** that discretizes continuous sign language motion into per-body-part token sequences, and (2) an **Autoregressive Multilingual Generator (AMG)** based on the pretrained mBART model that generates motion token sequences from text input. After training, the pipeline proceeds as: text input → LM encoder → multi-head decoder generating per-part tokens → partial decoder reconstructing 3D sign language motion.

### Key Designs

1. **Decoupled Tokenizer (DETO)**: Sign language is multi-channel in nature — semantics are conveyed simultaneously through body and hand movements. DETO employs three independent VQ-VAEs to model the upper body ($B$), left hand ($LH$), and right hand ($RH$) separately, mapping the SMPL-X parameters of each part into independent discrete token sequences. The quantization is defined as:

$$\hat{z}_i^p = Q(s_{f,i}^p) = \arg\min_{z_j \in \mathbf{Z}^p} \|s_{f,i}^p - z_j\|_2$$

Codebook sizes are set to: body $N_Z^B = 96$, left and right hands $N_Z^{LH} = N_Z^{RH} = 192$, with code dimension $C = 512$. The decoupled design exploits kinematic priors in sign language, enabling independent and accurate modeling of each body part.

2. **Multi-Head Decoding**: This is one of the core innovations of this work. Existing methods adopt two decoding strategies: (a) **Sequential decoding**, which flattens all part tokens into a sequence of length $3K$ and predicts them one by one — highly inefficient; (b) **Parallel decoding**, which independently decodes the three parts without any information fusion. SOKE's multi-head decoding uses three independent LM heads to simultaneously predict tokens for all three parts at each time step, with the input embedding defined as a weighted average of per-part token embeddings:

$$\mathbf{E} = (1-2\lambda)\mathbf{E}^B + \lambda\mathbf{E}^{LH} + \lambda\mathbf{E}^{RH}$$

where $\lambda = 1/3$. This achieves efficient fusion under a conditional independence assumption — reducing decoding steps from $3K$ to $K$ while preserving inter-part information interaction through weighted input embeddings. Inference latency is reduced from 3.26s/video to 1.46s/video.

3. **Retrieval-Enhanced SLG**: Inspired by the success of RAG in NLP, SOKE leverages an external sign language lexicon (constructed from isolated sign language recognition datasets) to provide precise word-level signs as auxiliary conditions. The pipeline is: (a) construct the lexicon by converting RGB videos to SMPL-X poses and discretizing them into token quadruples $\{(w, m^B, m^{LH}, m^{RH})\}$ via DETO; (b) retrieve matching motion tokens for all words in the input text; (c) concatenate the original text tokens with the retrieved motion tokens before feeding into the LM encoder. This approach avoids the unnatural coarticulation artifacts introduced by gloss-based methods that directly concatenate lexicon signs, as the model learns to generate fluent sentence-level motion conditioned on lexicon signs.

### Loss & Training

- **DETO stage**: Standard VQ-VAE loss $\mathcal{L}_{vq}^p = \mathcal{L}_{rec}^p + \mathcal{L}_{emb}^p + \mathcal{L}_{com}^p$, comprising reconstruction loss, embedding loss, and commitment loss.
- **AMG stage**: Standard cross-entropy loss $\mathcal{L}_{LM} = -\log P(\mathbf{Y}|\mathbf{h}_{en})$.
- DETO is trained for 500 epochs on a combined multilingual dataset and lexicon data using AdamW with a cosine scheduler at lr=2e-4.
- AMG is fine-tuned from mBART-large-cc25 for 150 epochs.
- Greedy decoding is used at inference; generation terminates when any head predicts EOS.
- Training is conducted on 6 RTX 3090 GPUs.

## Key Experimental Results

### Main Results

**DTW-PA-JPE (hand) comparison across three sign language datasets**

| Method | Multilingual? | How2Sign ↓ | CSL-Daily ↓ | Phoenix-2014T ↓ |
|--------|--------------|------------|-------------|-----------------|
| NSA (diffusion) | × | 7.33 | — | — |
| S-MotionGPT | × | 4.39 | 3.78 | 3.41 |
| **SOKE (Ours)** | **✓** | **2.35** | **1.71** | **1.38** |

**Improvement over S-MotionGPT**: How2Sign −46.5%, CSL-Daily −54.8%, Ph14T −59.5%

SOKE also significantly outperforms on Back-Translation BLEU-4: 14.48 vs. 11.45 (How2Sign).

### Ablation Study

| Decoding Method | Retrieval | How2Sign DTW↓ | Latency (s/video) | Note |
|----------------|-----------|---------------|-------------------|------|
| Sequential | × | 4.64 | 3.26 | All tokens flattened |
| Parallel | × | 5.06 | 1.39 | Fully independent decoding |
| Multi-head | × | 4.17 | 1.46 | Ours (w/o retrieval) |
| Multi-head | ✓ | **3.34** | 1.55 | Full method |

### Key Findings

- Multi-head decoding reduces latency by 55% compared to sequential decoding (3.26→1.46s) while also lowering DTW error by 10%.
- Retrieval augmentation further reduces DTW error by 20% (4.17→3.34) with only 0.09s additional latency.
- Parallel decoding is the fastest but yields the worst results, demonstrating the necessity of inter-part information fusion.
- A single unified model handles ASL, CSL, and DGS simultaneously without language-specific training.
- Discretization quality of hand motion is critical — hand codebooks require larger capacity than the body codebook (192 vs. 96).

## Highlights & Insights

- The paradigm of treating sign language as **"language" rather than "video"** is the central insight — the discrete structure of sign language is naturally suited for language model approaches.
- Multi-head decoding achieves an elegant balance between efficiency and quality; the conditional independence assumption works well in practice.
- The retrieval-augmented strategy elegantly unifies word-level precision with sentence-level fluency — lexicon signs serve as conditioning input to the LM rather than being naively concatenated.
- SMPL-X annotations are constructed for CSL-Daily and Phoenix-2014T, filling a gap in multilingual 3D sign language data.

## Limitations & Future Work

- Retrieval performance depends on lexicon coverage — rare words absent from the lexicon may be handled poorly.
- The SMPL-X model's facial expression parameters are limited (10 parameters), while facial expressions carry important semantic content in sign language.
- The scale of mBART-large (~600M parameters) may be prohibitive for edge device deployment.
- Only greedy decoding is evaluated; beam search or sampling strategies may further improve quality.
- User studies measuring actual Deaf users' comprehension of generated signs have not been conducted.

## Related Work & Insights

- **T2S-GPT** and **MotionGPT** are pioneering tokenizer-LM works but support only monolingual settings and use coupled tokenizers.
- The success of **RAG** in NLP directly inspired the retrieval-augmented SLG design.
- Insight: The multi-head decoding strategy is generalizable to other multi-channel motion generation tasks, such as upper-lower limb coordination in dance generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Multi-head decoding and retrieval-augmented SLG are both novel contributions; the multilingual unified model is meaningful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, multiple baselines, comprehensive ablations, and both qualitative and quantitative analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, intuitive figures, and well-presented methodology.
- **Value**: ⭐⭐⭐⭐⭐ Significant social value for sign language accessibility technology; open-source data and code benefit the community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models](../../NeurIPS2025/human_understanding/vimorag_video-based_retrieval-augmented_3d_motion_generation_for_motion_language.md)
- [\[ICCV 2025\] GestureHYDRA: Semantic Co-speech Gesture Synthesis via Hybrid Modality Diffusion Transformer and Cascaded-Synchronized Retrieval-Augmented Generation](gesturehydra_semantic_co-speech_gesture_synthesis_via_hybrid_modality_diffusion_.md)
- [\[NeurIPS 2025\] Cycle-Sync: Robust Global Camera Pose Estimation through Enhanced Cycle-Consistent Synchronization](../../NeurIPS2025/human_understanding/cycle-sync_robust_global_camera_pose_estimation_through_enhanced_cycle-consisten.md)
- [\[CVPR 2026\] Unleashing Vision-Language Semantics for Deepfake Video Detection](../../CVPR2026/human_understanding/unleashing_vision-language_semantics_for_deepfake_video_detection.md)
- [\[CVPR 2026\] LaScA: Language-Conditioned Scalable Modelling of Affective Dynamics](../../CVPR2026/human_understanding/lasca_language-conditioned_scalable_modelling_of_affective_dynamics.md)

<!-- RELATED:END -->
