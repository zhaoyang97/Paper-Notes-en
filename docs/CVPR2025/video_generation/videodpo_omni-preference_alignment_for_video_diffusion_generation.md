---
title: >-
  [Paper Note] VideoDPO: Omni-Preference Alignment for Video Diffusion Generation
description: >-
  [CVPR 2025][Video Generation][Video Diffusion Models] VideoDPO is the first to adapt DPO (Direct Preference Optimization) to video diffusion models. It proposes the OmniScore comprehensive scoring system to simultaneously measure visual quality and semantic alignment, combined with an automatic preference data generation pipeline and a score-difference-based data reweighting strategy. This approach achieves significant improvements in preference alignment across VideoCrafter2…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Video Diffusion Models"
  - "DPO Preference Alignment"
  - "Video Quality Assessment"
  - "Data Reweighting"
  - "Text-to-Video"
date: 2026-05-08
content_hash: 4efcb724439e28e9
---

# VideoDPO: Omni-Preference Alignment for Video Diffusion Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.14167](https://arxiv.org/abs/2412.14167)  
**Code**: [https://videodpo.github.io/](https://videodpo.github.io/)  
**Area**: Diffusion Models / Video Generation  
**Keywords**: Video Diffusion Models, DPO Preference Alignment, Video Quality Assessment, Data Reweighting, Text-to-Video

## TL;DR

VideoDPO is the first to adapt DPO (Direct Preference Optimization) to video diffusion models. It proposes the OmniScore comprehensive scoring system to simultaneously measure visual quality and semantic alignment, combined with an automatic preference data generation pipeline and a score-difference-based data reweighting strategy. This approach achieves significant improvements in preference alignment across VideoCrafter2, T2V-Turbo, and CogVideoX.

## Background & Motivation

**Background**: Text-to-Video (T2V) diffusion models have developed rapidly, with models like VideoCrafter, Open-Sora, and CogVideoX capable of generating diverse video content from text prompts. As a preference alignment method, DPO has achieved significant success in LLMs and image generation (e.g., DiffusionDPO for text-to-image models), but it has not yet been systematically applied to video diffusion models.

**Limitations of Prior Work**: Current video diffusion models generate videos that show deficiencies in two dimensions: (1) visual quality—insufficient intra-frame clarity, temporal inconsistency, and dynamic flickering, etc.; (2) semantic alignment—mismatch between generated content and text descriptions. These issues stem in part from low-quality samples (low resolution, blur, text-video mismatch) in large-scale pre-training data. Existing visual reward models typically focus on only a single dimension (either quality or semantics), and experiments show that these dimensions exhibit very low correlation, meaning that optimizing only one dimension cannot automatically improve the other.

**Key Challenge**: Video quality is multi-dimensional—the correlation between intra-frame visual quality, inter-frame temporal consistency, and text semantic alignment is very low, requiring a comprehensive evaluation. Existing methods either rely solely on reward models for gradient-based fine-tuning (e.g., VADER) or focus only on a single quality dimension, failing to achieve comprehensive preference alignment.

**Goal**: (1) Design a comprehensive scoring system that fully covers various dimensions of video generation quality; (2) establish an automated preference data construction pipeline to avoid expensive human annotation; (3) effectively improve the overall user preference of video models through DPO training.

**Key Insight**: The authors analyze the Pearson correlation coefficients between various sub-dimensions of video quality and find very low correlation (as shown in Figure 3(d)) between intra-frame quality, inter-frame consistency, and semantic alignment, making it essential to consider all dimensions simultaneously.

**Core Idea**: Propose the OmniScore comprehensive score + automatic best-vs-worst preference pair construction + frequency-distribution-based data reweighting. These three elements work synergistically to construct a complete video DPO alignment framework.

## Method

### Overall Architecture

The VideoDPO pipeline consists of three steps: (1) Generate $N$ videos for each prompt and evaluate them using the OmniScore comprehensive score; (2) select the highest and lowest-scoring videos as preference pairs (winning vs losing) to construct a preference dataset; (3) calculate the OmniScore frequency histogram of all videos, and train with DPO loss after reweighting the preference pairs.

### Key Designs

1. **OmniScore Comprehensive Preference Rating**:

    - **Function**: Comprehensively evaluate the quality of generated videos across multiple dimensions.
    - **Mechanism**: OmniScore consists of three major sub-scores: (a) **Intra-frame quality**—image quality and aesthetic appeal, measuring single-frame fidelity and visual aesthetics; (b) **Inter-frame quality**—subject consistency, background consistency, temporal flickering, motion smoothness, and motion dynamics, measuring visual coherence across frames; (c) **Text-video semantic alignment**—using visual-language foundation models to measure the match between video content and text prompts. Each dimension is scored independently using pre-trained evaluation models and then consolidated into the final OmniScore.
    - **Design Motivation**: Experimental analysis (Pearson correlation heatmap) reveals low correlation between different quality dimensions, suggesting that optimizing a single dimension (e.g., aesthetics only) not only fails to automatically improve others but might even degrade them. The comprehensive score ensures that DPO training avoids focusing on only one aspect.

2. **Score-based Preference Data Auto-Generation**:

    - **Function**: Automatically construct a high-quality preference pair dataset without human annotation.
    - **Mechanism**: For $K=10,000$ human-written prompts in the dataset (from VidProm), $N=4$ videos are generated for each prompt using the model to be aligned. Each video is scored using OmniScore $s_i = S(v_i, p)$, and the highest-scoring video is selected as the winning sample $v^W$ and the lowest-scoring as the losing sample $v^L$, forming the preference pair $(v^W, v^L)$.
    - **Design Motivation**: Choosing the best-vs-worst pairs with the largest score gap provides the clearest preference signal. Constructing preference pairs on-policy (i.e., using videos generated by the model itself) ensures that the preference data distribution matches the current capabilities of the model. Utilizing human-written prompts from VidProm helps the model better adapt to real user inputs.

3. **OmniScore-driven Data Reweighting**:

    - **Function**: Direct the model's focus toward "highly discriminative" preference pairs, improving training efficiency and efficacy.
    - **Mechanism**: Calculate the OmniScore frequency histogram $p(\cdot)$ of all generated videos, then define the preference pair sampling probability as $\text{prob}(s^W, s^L) = \sqrt{p(s^W) \cdot p(s^L)}$, and compute the reweighting factor $w_{\text{pair}} = (\beta / \text{prob}(s^W, s^L))^\alpha$, where $\beta$ is set to the probability of the most frequent sample, and $\alpha$ controls the weighting intensity ($\alpha=0.72$ in experiments). Finally, the DPO loss is multiplied by $w_{\text{pair}}$.
    - **Design Motivation**: When directly applying DPO, many preference pairs exhibit negligible score differences, making it difficult for the model to distinguish them. Low-frequency "typical preference pairs" (e.g., one exceptionally good + one exceptionally bad) often carry more alignment information. Frequency inverse-weighting is used to increase the training weight of these samples. When $\alpha=0$, this degrades to standard DPO.

### Loss & Training

The final training loss is $L_{\text{video}} = L_{\text{DPO}}(p, v^W, v^L) \cdot w_{\text{pair}}$, where the DPO loss takes the form of DiffusionDPO adapted for diffusion models. Training is performed for 3000 steps with a global batch size of 8, using the *AdamW* optimizer with lr=6e-6 on 4 *A100 GPUs*. The bin width of the frequency histogram is set to 0.01.

## Key Experimental Results

### Main Results

VBench Comprehensive Evaluation (Total = Weighted Quality + Semantics):

| Model | Method | VBench Total↑ | Quality↑ | Semantics↑ | HPS(V)↑ | PickScore↑ |
|------|------|---------------|----------|-----------|---------|-----------|
| VC2 | Baseline | 80.44 | 82.20 | 73.42 | 0.258 | 20.65 |
| VC2 | VADER | 80.59 | 82.46 | 73.09 | 0.259 | 20.62 |
| VC2 | **VideoDPO** | **81.93** | **83.07** | **77.38** | **0.261** | 20.65 |
| Turbo | Baseline | 80.95 | 82.71 | 73.93 | 0.262 | 21.15 |
| Turbo | **VideoDPO** | **81.80** | **83.80** | **73.81** | 0.260 | **21.18** |
| CogVid | Baseline | 79.30 | 82.35 | 67.10 | - | 19.81 |
| CogVid | **VideoDPO** | **79.80** | **83.00** | 66.99 | - | 19.79 |

### Ablation Study

Ablation on data reweighting strategy (using VC2 as baseline, VBench Total):

| Configuration | VBench Total | Description |
|------|-------------|------|
| Baseline (w/o DPO) | 80.44 | Original model |
| DPO w/o reweighting ($\alpha=0$) | 81.11 | Standard DPO already shows improvements |
| DPO + reweighting ($\alpha=0.72$) | **81.93** | Reweighted further improves by 0.82 |
| SFT (Supervised Fine-Tuning) | 78.78 | Pure SFT actually degrades performance |

Ablation on preference pair selection strategy shows that the best-vs-worst strategy outperforms random pairing and adjacent-rank pairing.

### Key Findings

- VideoDPO is effective across three models with different architectures (UNet-based VC2, distilled T2V-Turbo, and DiT-based CogVideoX), proving the generalizability of the method
- The improvement in the semantic alignment dimension is particularly outstanding (from 73.42 to 77.38 on VC2, a gain of nearly 4 percentage points), showing that DPO is highly effective in calibrating semantic understanding
- Detailed analysis of VBench across 16 sub-dimensions reveals that Multiple Objects (+11.63), Spatial Relationship (+12.85), and Scene (+15.78) show the largest improvements, which happen to be the weakest aspects of typical T2V models
- Fine-tuning directly on winning samples using SFT results in a drop in VBench Total (80.44 -> 78.78), demonstrating that preference contrastive learning is more effective than direct distribution fitting
- The gain from data reweighting is significant and robust ($\alpha=0.72$ is optimal)

## Highlights & Insights

- **Design philosophy of multi-dimensional preference scoring (OmniScore)**: The low-correlation analysis among different quality dimensions is highly convincing, confirming the necessity of a unified score. This approach can be generalized to other multi-attribute optimization problems, such as image editing and 3D generation
- **Simple and effective frequency inverse-weighting strategy**: Utilizing the OmniScore distribution histogram to identify "highly informative" preference pairs is essentially a form of hard example mining, yet it offers greater elegance by approaching the problem from a statistical distribution perspective
- **First systematic adaptation of DPO to video diffusion**: Although conceptually simple, it addresses several engineering challenges including video preference data construction, multi-dimensional scoring, and DPO loss adaptation for diffusion models

## Limitations & Future Work

- The weights of individual dimensions in OmniScore are currently fixed. Since different users and scenarios may prioritize dimensions differently, future work could explore adaptive weight adjustments
- Preference data is entirely generated from automated scoring (RLAIF) without verification using real human preference annotations, which may propagate biases inherent in the evaluation models to the alignment results
- Generating only 4 videos per prompt restricts the diversity of preference pairs. A larger sampling size might yield better results but would increase computational costs
- The improvement on CogVideoX is relatively modest (79.30 -> 79.80), suggesting that DiT architectures may require different hyperparameter tuning for DPO
- Metrics such as motion dynamics and dynamic degree showed some degradation after DPO alignment (e.g., motion smoothness on VC2 dropped from 97.73 to 92.18), indicating that alignment may trade off dynamic richness for overall quality, an important trade-off worth highlighting

## Related Work & Insights

- **vs DiffusionDPO**: DiffusionDPO aligns preferences in the image domain, focusing only on image quality or a single semantic dimension; VideoDPO introduces the comprehensive OmniScore and data reweighting, making it better suited for the multi-dimensional quality requirements of video
- **vs VADER**: VADER directly optimizes gradients over the final steps of the diffusion model using a differentiable reward model; VideoDPO adopts the preference contrastive learning paradigm of DPO, which does not require a differentiable reward model and offers more stable training
- **vs T2V-Turbo v2**: T2V-Turbo v2 also explores reward gradients to optimize consistency distilled models, but essentially targets a single reward; VideoDPO's preference contrastive learning is better suited for the comprehensive optimization of multi-dimensional preferences

## Rating

- Novelty: ⭐⭐⭐⭐ Adaptation of DPO to video diffusion models for the first time; OmniScore and reweighting strategies are innovative, though not thoroughly disruptive
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on three different models across multiple metrics, complete with detailed sub-dimension analysis and extensive ablation studies
- Writing Quality: ⭐⭐⭐⭐ Overall structure is clear and the correlation analysis among OmniScore dimensions is highly persuasive, though some technical details are scattered
- Value: ⭐⭐⭐⭐ Provides a practical preference alignment framework for the post-training of video generation models, and the OmniScore evaluation framework itself is of independent value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] OmniVDiff: Omni Controllable Video Diffusion for Generation and Understanding](../../AAAI2026/video_generation/omnivdiff_omni_controllable_video_diffusion_for_generation_and_understanding.md)
- [\[CVPR 2025\] Can Text-to-Video Generation Help Video-Language Alignment?](can_text-to-video_generation_help_video-language_alignment.md)
- [\[ICLR 2026\] MoAlign: Motion-Centric Representation Alignment for Video Diffusion Models](../../ICLR2026/video_generation/moalign_motion-centric_representation_alignment_for_video_diffusion_models.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](../../ICCV2025/video_generation/vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[ICCV 2025\] Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](../../ICCV2025/video_generation/prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)

</div>

<!-- RELATED:END -->
