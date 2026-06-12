---
title: >-
  [Paper Note] Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation
description: >-
  [CVPR 2026][Image Generation][Multi-Human Generation] Ar2Can decomposes multi-human image generation into two stages — spatial planning (Architect) and identity-preserving rendering (Artist) — and trains the Artist model…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Multi-Human Generation"
  - "Identity Preservation"
  - "Spatial Planning"
  - "GRPO"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 4094be36f267f3cd
---

# Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation

**Conference**: CVPR 2026
**arXiv**: [2511.22690](https://arxiv.org/abs/2511.22690)  
**Code**: [https://qualcomm-ai-research.github.io/ar2can/](https://qualcomm-ai-research.github.io/ar2can/)  
**Area**: Image Generation / Multi-Human Image Generation
**Keywords**: Multi-Human Generation, Identity Preservation, Spatial Planning, GRPO, Reinforcement Learning

## TL;DR
Ar2Can decomposes multi-human image generation into two stages — spatial planning (Architect) and identity-preserving rendering (Artist) — and trains the Artist model via GRPO reinforcement learning with a spatially-anchored face reward based on Hungarian matching. The method achieves an identity preservation score of 68.2 and a counting accuracy of 90.2 on MultiHuman-Testbench, substantially outperforming all baselines.

## Background & Motivation

**Background**: Text-to-image diffusion models have matured considerably for single-person generation, yet systematically fail in **multi-human scenarios** — identity blending, identity swapping, and incorrect person counts are pervasive problems.

**Classification and Limitations of Prior Work**:
   - **Region-conditioned methods** (GLIGEN, ReCo) require users to manually provide spatial annotations, resulting in poor usability.
   - **Identity-preserving methods** (IP-Adapter, InstantID, PuLID) are designed for single subjects and suffer from identity conflicts in multi-person settings.
   - **Multi-ID methods** (OmniGen, DreamO, XVerse) still perform poorly on recent benchmarks.

**Key Challenge**: Existing methods conflate **spatial layout reasoning** and **identity rendering** within a single generation process. When the model must simultaneously determine "where each person is" and "what each person looks like," spatial structure and appearance become entangled, causing identity blending.

**Key Insight**: **Decoupling spatial planning from identity rendering** — first determine where each person appears, then focus on photorealistic rendering. This divide-and-conquer strategy fundamentally prevents identity blending.

**Core Idea**: The Architect generates a structured spatial layout (bounding boxes / poses); the Artist, guided by this layout, maintains multi-identity consistency via GRPO combined with a Hungarian-matching face reward.

## Method

### Overall Architecture
Two stages: Architect → Canvas → Artist
- Input: text prompt $p$ + $N$ reference identity images $\{I_{ref,1}, ..., I_{ref,N}\}$
- Architect: predicts spatial layout $\mathcal{L} = \{b_1, ..., b_N\}$ (one bounding box per person)
- Canvas construction: reference faces are pasted into their corresponding positions
- Artist: conditioned on the canvas, reference images, and text to generate the final image

### Key Designs

1. **Architect-A (LLM-based)**:

    - Fine-tuned from Qwen-2.5-0.5B-Instruct
    - Vocabulary extended with layout structural tokens: `<SoL>`, `<EoL>`, `<C>`, etc.
    - Jointly predicts token sequences and continuous coordinates via a dual-head design: $f_{token}$ + $f_{value}$
    - Loss: $\mathcal{L} = \mathcal{L}_{CE} + \lambda_{coord}[\mathcal{L}_{gIoU} + \|b_{pred} - b_{gt}\|_1]$
    - **Advantage**: Strong language understanding; highest person count accuracy (90.2%)

2. **Architect-B (T2I-based)**:

    - Fine-tuned from Flux-Schnell (requiring only 4 denoising steps)
    - Trained with GRPO reinforcement learning; reward: $r = \alpha \cdot r_{count} + \beta \cdot r_{hps}$
    - Capable of simultaneously outputting face bounding boxes and body poses
    - **Advantage**: Stronger spatial prior; higher action scores

3. **Artist GRPO Training with Combined Reward**:

    - Trained on Flux-Kontext with four combined rewards:
    $r_{Artist} = \alpha \cdot r_{count} + \beta \cdot r_{hps} + \zeta \cdot r_{face} + \eta \cdot r_{pose}$
    - **Spatially-anchored face matching reward $r_{face}$ (core innovation)**:
        - Step 1: Apply the **Hungarian algorithm** to find optimal matches between Architect-predicted face centers and RetinaFace-detected face centers.
        - Step 2: Compute ArcFace cosine similarity for each matched face pair.
        - **Design Motivation**: Naïve position-based matching (cropping faces directly at predicted locations) causes copy-paste artifacts and reward hacking — the model learns to paste the reference face at the exact predicted location. Hungarian centroid matching relaxes the spatial constraint, allowing natural variation.

4. **Token Sharing & Dropping**:

    - Tokens from non-informative canvas regions are dropped, reducing token count by $2\times$ on average.
    - When face bounding boxes overlap, shared RoPE positional encodings are applied.
    - **Design Motivation**: Shared positional encodings compel the model to learn occlusion handling (depth ordering, spatial rearrangement) rather than simple copy-paste.

5. **Curriculum Learning**: For the first $\tau=100$ epochs, only 2–3 person scenarios are used; thereafter, 2–7 person scenarios are sampled uniformly. This prevents training collapse in early stages.

### Training Data
- Core challenge: lack of large-scale multi-human training data.
- DisCo is used to generate synthetic multi-human scenes, paired with real reference faces to construct hybrid training samples.
- Training relies entirely on synthetic multi-human data; no real multi-person photographs are required.

## Key Experimental Results

### Main Results (MultiHuman-TestBench)

| Method | Count↑ | Multi-ID↑ | HPS↑ | Action-S↑ | Unified↑ |
|------|--------|-----------|------|-----------|----------|
| GPT-Image-1 | 87.9 | 28.8 | 30.3 | 97.0 | 55.8 |
| DreamO | 61.2 | 34.7 | 28.5 | 86.2 | 59.7 |
| MH-OmniGen | 60.3 | 54.5 | 26.3 | 91.6 | 61.6 |
| XVerse | 81.7 | 30.6 | 25.5 | 66.2 | 52.7 |
| **Ar2Can (Arch-B)** | 86.9 | **68.2** | **30.8** | 86.2 | **72.4** |
| **Ar2Can (Arch-A)** | **90.2** | 67.6 | 30.2 | 86.3 | 72.2 |

### Ablation Study

| Configuration | Count↑ | Multi-ID↑ | HPS↑ | Notes |
|------|--------|-----------|------|------|
| Baseline (Kontext) | 80.7 | 14.5 | 29.2 | Original model; severe multi-person failure |
| + Simple Matching | 75.6 | 55.2 | 27.6 | Naïve matching → copy-paste artifacts |
| + Hungarian Centroid | 80.1 | 60.3 | 30.9 | Hungarian matching restores quality |
| + Curriculum (Full) | **86.9** | **68.2** | **30.8** | Curriculum learning yields further gains |

### Key Findings
- Multi-ID improves from 14.5 (Kontext baseline) to 68.2 (+53.7), demonstrating that Kontext nearly completely fails in multi-person scenarios.
- Hungarian matching improves not only identity preservation (+5.1) over naïve matching, but also recovers image quality (HPS: 27.6 → 30.9).
- The method is preferred by human evaluators in 88% of evaluated prompts (vs. DreamO 4%, XVerse 8%).
- Training exclusively on synthetic data surpasses commercial models trained with large amounts of real data.

## Highlights & Insights
- **Decoupling spatial planning from rendering** is a conceptually clean approach that effectively eliminates identity entanglement in multi-human generation.
- The **Hungarian matching reward** elegantly balances spatial precision and generative naturalness — rather than enforcing exact positional correspondence, it requires only approximate proximity.
- The **modular design** offers flexibility: the Architect can be swapped for different variants without retraining the Artist.
- Achieving state-of-the-art results primarily with synthetic data demonstrates that RL fine-tuning can effectively compensate for data quality limitations.

## Limitations & Future Work
- Action generation still lags behind GPT-Image-1 (Action-S: 86.2 vs. 97.0); understanding of complex actions requires further improvement.
- Performance in extreme scenarios with more than 7 persons has not been validated.
- Inference latency remains high due to the two-stage architecture; token sharing mitigates but does not fully resolve this overhead.
- Architect-A and Architect-B each have distinct trade-offs, and a unified solution is lacking.

## Related Work & Insights
- DisCo (Flow-GRPO for multi-human generation) is a direct predecessor; Ar2Can extends it by incorporating spatial anchoring.
- Canvas-based methods (e.g., Kontext) perform well for single subjects but fail with multiple persons, underscoring the need for explicit spatial guidance.
- The paradigm of GRPO with combined rewards is generalizable to other image generation tasks requiring multi-objective balance.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of spatial decoupling and Hungarian matching reward is novel and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two benchmarks + human evaluation + detailed ablations + latency analysis.
- Writing Quality: ⭐⭐⭐⭐ Framework is clearly presented, though some details require consulting the appendix.
- Value: ⭐⭐⭐⭐⭐ Multi-human identity-preserving generation addresses a genuine need; practical improvements are substantial (+13.7 Multi-ID over prior SOTA).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_textguided_multihuman_3d_moti.md)
- [\[CVPR 2026\] Leveraging Multispectral Sensors for Color Correction in Mobile Cameras](leveraging_multispectral_sensors_for_color_correction_in_mobile_cameras.md)
- [\[CVPR 2026\] PSR: Scaling Multi-Subject Personalized Image Generation with Pairwise Subject-Consistency Rewards](psr_scaling_multi-subject_personalized_image_generation_with_pairwise_subject-co.md)
- [\[CVPR 2026\] PSDesigner: Automated Graphic Design with a Human-Like Creative Workflow](psdesigner_automated_graphic_design_with_a_human-like_creative_workflow.md)
- [\[CVPR 2026\] Resolving the Identity Crisis in Text-to-Image Generation](resolving_the_identity_crisis_in_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
