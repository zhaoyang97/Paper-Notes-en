---
title: >-
  [Paper Note] Resolving the Identity Crisis in Text-to-Image Generation
description: >-
  [CVPR 2026][Image Generation][Identity diversity] This paper identifies the "identity crisis" in text-to-image models for multi-person scene generation — manifesting as duplicated faces and identity merging — and propose…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Identity diversity"
  - "multi-person image generation"
  - "reinforcement learning"
  - "GRPO"
  - "text-to-image"
date: 2026-05-08
content_hash: 4b14b1dff2a904cc
---

# Resolving the Identity Crisis in Text-to-Image Generation

**Conference**: CVPR 2026
**arXiv**: [2510.01399](https://arxiv.org/abs/2510.01399)  
**Code**: [https://qualcomm-ai-research.github.io/disco/](https://qualcomm-ai-research.github.io/disco/)  
**Area**: Diffusion Models
**Keywords**: Identity diversity, multi-person image generation, reinforcement learning, GRPO, text-to-image

## TL;DR
This paper identifies the "identity crisis" in text-to-image models for multi-person scene generation — manifesting as duplicated faces and identity merging — and proposes the DisCo framework. By combining a compositional reward function with GRPO-based reinforcement learning fine-tuning of a flow-matching model, DisCo achieves 98.6% unique face accuracy, surpassing closed-source models including GPT-Image-1.

## Background & Motivation
1. **Background**: Current text-to-image models (e.g., FLUX, SD3.5) achieve high quality in single-person generation but exhibit severe deficiencies in multi-person scenes.
2. **Limitations of Prior Work**: Multi-person generation frequently suffers from three problems — duplicated faces (different individuals sharing the same face), identity merging (features of multiple people conflated), and incorrect person counts (number of generated persons mismatching the prompt). Identity differentiation remains insufficient even when overall image quality is high.
3. **Key Challenge**: Existing methods and RL fine-tuning works primarily optimize for aesthetics, text alignment, and human preference, without explicitly optimizing identity diversity — especially cross-sample identity diversity.
4. **Goal**: (a) intra-image identity repetition; (b) cross-sample identity repetition; (c) person count accuracy; (d) preservation of image quality.
5. **Key Insight**: The authors find that optimizing only intra-image diversity leads to "global diversity collapse" — repeated identities migrate from within a single image to across different images. This key finding motivates the design of group-level rewards.
6. **Core Idea**: The identity crisis in multi-person generation is addressed through the GRPO reinforcement learning framework and a carefully designed four-component compositional reward (intra-image diversity + cross-sample diversity + count control + quality preservation), requiring no ground-truth data annotation.

## Method

### Overall Architecture
DisCo builds on the Flow-GRPO framework, modeling the denoising process of a flow-matching model as a Markov Decision Process (MDP). Given a text prompt, a group of $M$ trajectories is sampled; a compositional reward is computed on the final image of each trajectory, and policy updates are performed using group-normalized advantage estimates. Training uses fewer denoising steps for efficiency, while full steps are used at inference.

### Key Designs

1. **Intra-Image Diversity Reward $r_{\text{img}}^d$**:

    - **Function**: Penalizes repeated identities within a single image.
    - **Mechanism**: RetinaFace detects faces and ArcFace extracts embeddings; the maximum pairwise cosine similarity among all faces in the image is computed, yielding a reward of $1 - \max_{j \neq k} s(f_j, f_k)$. A neutral reward of 0.5 is assigned when fewer than two faces are detected.
    - **Design Motivation**: This is the most intuitive diversity metric; however, the authors find that using this reward alone causes repeated identities to "migrate" across samples.

2. **Group-Level Diversity Reward $r_{\text{grp}}^d$ (Core Innovation)**:

    - **Function**: Suppresses cross-sample identity repetition.
    - **Mechanism**: A counterfactual "leave-one-out" statistic is employed. For $M$ images generated from the same prompt, the mean pairwise similarity across all faces $S_G$ is computed. For image $i$, its faces are removed and $S_{G-i}$ is recomputed; the contribution is defined as $\Delta_i = S_G - S_{G-i}$. If removing image $i$ decreases intra-group similarity ($\Delta_i > 0$), that image increases repetition and should be penalized. A sigmoid mapping $\sigma(-\lambda \Delta_i)$ converts contributions to a $[0,1]$ reward.
    - **Design Motivation**: This is the paper's most critical finding — optimizing intra-image diversity alone is insufficient, as repeated identities "flow" between images and cause global identity distribution collapse. The group-level reward addresses this issue fundamentally.

3. **Count Control Reward $r_{\text{img}}^c$**:

    - **Function**: Ensures the correct number of persons is generated.
    - **Mechanism**: Binary reward — 1 if the number of detected faces matches the count specified in the prompt, 0 otherwise.
    - **Design Motivation**: Diversity rewards induce reward hacking — the model learns to generate fewer people to avoid diversity penalties. The count reward directly counteracts this behavior.

4. **Quality/Alignment Reward $r_{\text{img}}^q$**:

    - **Function**: Preserves image quality and prompt alignment.
    - **Mechanism**: HPSv3 score is used as the reward, normalized to $[0,1]$.
    - **Design Motivation**: Diversity optimization introduces "grid" artifacts (faces arranged in an unnatural grid pattern) and reduced prompt adherence. The HPSv3 reward effectively mitigates these issues and, as a byproduct, enhances the model's compositional prompt-following capability.

5. **Single-Stage Curriculum Learning**:

    - **Function**: Stabilizes training and improves generalization.
    - **Mechanism**: Training is initially biased toward simple prompts (2–4 persons), with gradual annealing to uniform sampling across all complexity levels (2–$N_{\max}$ persons). The annealing weight $\lambda_t = (t/t_{\text{curriculum}})^{\gamma_c}$ controls the transition rate from simple to complex.
    - **Design Motivation**: Expert models (e.g., Krea-Dev) struggle to converge on complex multi-person scenes; curriculum learning resolves this through progressive complexity scaling.

### Loss & Training
The total reward is $r(\tau_i, c, G) = \alpha r_{\text{img}}^d + \beta r_{\text{grp}}^d + \gamma r_{\text{img}}^c + \zeta r_{\text{img}}^q$, with all four components normalized to $[0,1]$. Training uses 30,000 multi-person scene prompts (2–7 persons) generated by GPT-5, requiring no ground-truth data annotation.

## Key Experimental Results

### Main Results (DiverseHumans-TestPrompts)

| Model | Count Acc | UFA | GIS | HPS | Avg |
|-------|----------|-----|-----|-----|-----|
| GPT-Image-1 | 90.5 | 85.1 | 89.8 | 33.4 | 78.7 |
| DisCo(Flux) | **92.4** | **98.6** | **98.3** | 33.4 | **81.7** |
| DisCo(Krea) | 83.5 | 89.7 | 90.6 | 32.2 | 76.8 |
| Flux-Dev (baseline) | 70.8 | 48.2 | 50.5 | 31.7 | 56.0 |
| Krea-Dev (baseline) | 73.6 | 45.8 | 50.6 | 31.2 | 57.8 |

### Ablation Study (Krea-Dev baseline)

| Configuration | Count Acc | UFA | GIS | HPS |
|--------------|----------|-----|-----|-----|
| Baseline | 73.6 | 45.8 | 50.6 | 31.2 |
| +Intra-image diversity | 66.2 | 78.6 | 50.8 | 31.7 |
| +Group-level diversity | 67.3 | 80.2 | 72.5 | 32.0 |
| +Count control+HPS | 79.2 | 82.6 | 73.7 | 32.4 |
| +Curriculum learning (full DisCo) | **83.5** | **89.7** | **90.6** | 32.2 |

### Key Findings
- **Global identity collapse**: Using only the intra-image diversity reward raises UFA from 45.8% to 78.6%, but GIS remains virtually unchanged (50.6→50.8). Adding the group-level reward substantially improves GIS to 72.5%, validating cross-sample diversity as an independent and critical optimization target.
- **Reward hacking**: Diversity rewards cause count accuracy to drop (73.6→66.2), as the model exploits the objective by generating fewer persons. The count reward effectively resolves this.
- **DisCo surpasses closed-source models**: DisCo achieves significant gains over GPT-Image-1 on UFA (98.6% vs. 85.1%) and GIS (98.3% vs. 89.8%) while maintaining equivalent HPS quality scores.
- **Curriculum learning is critical for expert models**: Krea-Dev (an expert model) depends on curriculum learning to converge, whereas Flux-Dev (a general-purpose model) exhibits less dependency on it.

## Highlights & Insights
- The **group-level counterfactual reward design** is particularly elegant — by computing marginal contributions via a "leave-one-out" approach, the non-differentiable set-level diversity objective is converted into a reward signal attributable to individual samples. This design paradigm is transferable to any RL scenario requiring optimization of set-level properties.
- The paper **identifies and addresses three reward hacking patterns** in RL fine-tuning: undercounting, grid artifacts, and prompt non-adherence. Each hacking behavior has a corresponding countermeasure, forming a complete robust optimization framework.
- **Zero-annotation training**: The entire training pipeline requires no human-annotated ground-truth data, relying solely on GPT-5-generated prompts and pretrained face detection/recognition models as reward signals.

## Limitations & Future Work
- The method depends on RetinaFace and ArcFace for face detection and recognition, which may be inaccurate under challenging conditions such as profile views or occlusion.
- The approach focuses exclusively on facial identity diversity without explicitly handling diversity along other attributes (e.g., body type, age distribution), though experiments show that facial diversity training incidentally improves these as well.
- Training prompts cover only 2–7 person scenarios; generalization to larger crowd sizes remains unvalidated.
- Extension of the method to multi-character consistency in video generation has not been explored.

## Related Work & Insights
- **vs. Flow-GRPO**: DisCo builds upon Flow-GRPO but introduces identity-diversity-specific reward designs. Flow-GRPO optimizes only general text alignment and quality, without addressing identity issues.
- **vs. MultiHuman-TestBench**: This NeurIPS 2025 work identifies bias in multi-person generation but only diagnoses rather than resolves it; DisCo directly addresses the future work directions proposed therein.
- **vs. adversarial training methods**: DisCo employs RL fine-tuning rather than adversarial training, enabling more flexible optimization of multiple heterogeneous, non-differentiable objectives.

## Rating
- Novelty: ⭐⭐⭐⭐ First to treat identity diversity as an explicit optimization target; the group-level counterfactual reward design is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two test sets, multiple baselines (including closed-source), detailed ablations, and generalization analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; reward hacking analysis is thorough.
- Value: ⭐⭐⭐⭐ Addresses an important practical problem with a scalable and extensible methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Disentangling to Re-couple: Resolving the Similarity-Controllability Paradox in Subject-Driven Text-to-Image Generation](disentangling_to_re-couple_resolving_the_similarity-controllability_paradox_in_s.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[CVPR 2026\] SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](solace_self_confidence_rewards_t2i.md)
- [\[CVPR 2026\] When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance](when_safety_collides_resolving_multi-category_harmful_conflicts_in_text-to-image.md)
- [\[CVPR 2026\] Agentic Retoucher for Text-To-Image Generation](agentic_retoucher_for_texttoimage_generation.md)

</div>

<!-- RELATED:END -->
