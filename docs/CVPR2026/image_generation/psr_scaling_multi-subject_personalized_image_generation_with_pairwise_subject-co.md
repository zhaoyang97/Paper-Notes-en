---
title: >-
  [Paper Note] PSR: Scaling Multi-Subject Personalized Image Generation with Pairwise Subject-Consistency Rewards
description: >-
  [CVPR 2026][Image Generation][Multi-subject personalized generation] To address poor subject consistency and insufficient text adherence in multi-subject personalized image generation, this paper proposes a scalable multi-subject data construction pipeline and Pairwise Subject-Consistency Rewards (PSR). Through two-stage training (SFT + RL), the method comprehensively outperforms existing state-of-the-art methods on the self-constructed PSRBench.
tags:
  - CVPR 2026
  - Image Generation
  - Multi-subject personalized generation
  - subject consistency
  - reinforcement learning
  - pairwise reward
  - positional encoding
date: 2026-05-08
content_hash: 8cd8c2d48d91e0fb
---

# PSR: Scaling Multi-Subject Personalized Image Generation with Pairwise Subject-Consistency Rewards

**Conference**: CVPR 2026
**arXiv**: [2512.01236](https://arxiv.org/abs/2512.01236)
**Code**: [https://github.com/wang-shulei/PSR](https://github.com/wang-shulei/PSR)
**Area**: Diffusion Models / Personalized Generation
**Keywords**: Multi-subject personalized generation, subject consistency, reinforcement learning, pairwise reward, positional encoding

## TL;DR
To address poor subject consistency and insufficient text adherence in multi-subject personalized image generation, this paper proposes a scalable multi-subject data construction pipeline and Pairwise Subject-Consistency Rewards (PSR). Through two-stage training (SFT + RL), the method comprehensively outperforms existing state-of-the-art methods on the self-constructed PSRBench.

## Background & Motivation

1. **Background**: Single-subject personalized generation models (e.g., FLUX.1 Kontext, Qwen-Image-Edit) have demonstrated strong capabilities in generating subject-consistent images in novel scenes conditioned on reference images.
2. **Limitations of Prior Work**: When scaling to multi-subject scenarios, existing models face two key challenges: (a) poor subject consistency—generated subjects are dissimilar to or entirely missing from the reference; and (b) poor text adherence—models fail to correctly bind attributes, e.g., given the prompt "dog wearing a chef's hat, cat wearing a scarf," the model may produce swapped attributes.
3. **Key Challenge**: Two underlying issues are identified: the absence of high-quality multi-subject training datasets (existing datasets such as OmniGen's X2I-subject-driven data focus primarily on faces with low general object consistency), and the lack of fine-grained post-training strategies (SFT optimizes only at the global image level and cannot guarantee per-subject consistency).
4. **Goal**: (1) How to construct large-scale, high-quality multi-subject training data? (2) How to achieve subject-level fine-grained alignment during training? (3) How to comprehensively evaluate multi-subject personalized generation?
5. **Key Insight**: Leveraging existing strong single-subject personalization models (e.g., FLUX.1 Kontext) to "reverse-engineer" multi-subject data construction, and employing pairwise reward mechanisms within reinforcement learning to achieve subject-level fine-grained alignment.
6. **Core Idea**: Constructing multi-subject data using single-subject models, combined with post-training via Pairwise Subject-Consistency Rewards (PSR), to achieve scalable, high-quality multi-subject personalized generation.

## Method

### Overall Architecture
The proposed method comprises three main components: (1) a scalable multi-subject data construction pipeline producing approximately 350K high-quality samples; (2) two-stage training—Stage 1 SFT enables a single-subject model to acquire multi-subject generation knowledge, and Stage 2 RL applies PSR rewards for fine-grained alignment; and (3) PSRBench, a comprehensive evaluation benchmark.

### Key Designs

1. **Scalable Multi-Subject Data Construction Pipeline**:

    - **Function**: Constructing a large-scale, high-quality multi-subject paired dataset from scratch.
    - **Mechanism**: The pipeline operates in two stages. Stage 1 (Image Generation): $n$ categories are sampled from the Objects365 category pool; an LLM generates T2I instructions; a T2I model synthesizes a multi-subject output image $I_{out}$; GroundingDINO detects and crops individual subject images $I_{crop}$; and a single-subject personalization model generates new reference images $I_{ref}$. Stage 2 (Instruction Generation): Seven predefined task types (attribute, background, action, position, complex scene, three-subject, four-subject) are used; an MLLM re-describes the generated images to prevent direct appearance leakage in the text; and a "re-editing" step is introduced to enhance attribute and action diversity.
    - **Design Motivation**: Directly using T2I models to generate multi-subject consistent data yields low quality (as in UNO), whereas leveraging mature single-subject personalization models ensures high consistency. Preventing textual appearance leakage discourages the model from taking shortcuts that ignore the reference images.

2. **Scalable Frame-wise Positional Encoding**:

    - **Function**: Enabling single-subject models to accept multi-image inputs while avoiding spatial position bias.
    - **Mechanism**: For each input reference image, the latent tokens receive an offset only along the temporal (frame) dimension, $PO_i = (i, h, w)$, without any offset along the spatial $h/w$ dimensions. During training, a multi-image joint training strategy is adopted, sampling 2/3/4 reference images with probabilities of 0.9/0.05/0.05 respectively.
    - **Design Motivation**: Methods such as UNO apply offsets along the $h/w$ spatial dimensions, introducing a spatial prior that "the second image is to the right/below," which restricts text-based positional control and produces excessively large offsets when scaling to three or four images, deviating from the pre-training distribution. Offsetting only along the frame dimension avoids these issues.

3. **Pairwise Subject-Consistency Reward (PSR)**:

    - **Function**: Providing subject-level fine-grained supervision signals during the RL stage.
    - **Mechanism**: The core idea is "subject disentanglement." For a generated image $I_{out}$, an open-vocabulary detector localizes and crops each subject region by category $c_i$: $I_{dec}^i = g(I_{out}, c_i)$. The same disentanglement is applied to the input reference images to obtain $I_{gt}^i$. Pairwise DINO feature similarity is then computed as: $R_{PSR} = \frac{1}{N}\sum_{i=1}^{N} f(I_{dec}^i, I_{gt}^i)$. The total reward is $R = w_1 R_{PSR} + w_2 R_s + w_3 R_h$, where $R_s$ is an MLLM semantic alignment reward and $R_h$ is the HPSv3 aesthetic preference reward. Training is conducted under the Flow-GRPO framework.
    - **Design Motivation**: SFT optimizes only at the global image level and cannot guarantee per-subject consistency. RL with per-subject disentangled rewards directly optimizes local objectives. The multi-reward combination also mitigates reward hacking from any single reward signal (e.g., relying solely on PSR tends to produce copy-paste behavior).

### Loss & Training
- **Stage 1 SFT**: Learning rate 1e-4, LoRA rank 512, trained at 512×512 resolution.
- **Stage 2 RL**: Learning rate 1e-5, LoRA rank 64, GRPO group size 6, reward weights $w_1=0.4, w_2=0.4, w_3=0.2$, sampling and training over the original 28 diffusion timesteps.

## Key Experimental Results

### Main Results
PSRBench comprises 7 subsets and evaluates three dimensions: Subject Consistency (SC), Aesthetic Preference (HPS), and Semantic Alignment (SA).

| Model | SC Overall | HPS Overall | SA Overall |
|------|-----------|-------------|------------|
| FLUX.1 Kontext | 0.497 | 0.870 | 0.583 |
| UNO | 0.523 | 1.009 | 0.667 |
| OmniGen2 | 0.587 | 1.020 | 0.758 |
| XVerse | 0.587 | 0.893 | 0.669 |
| Ours-SFT | 0.559 | 0.794 | 0.712 |
| **Ours (PSR)** | **0.673** | **1.124** | **0.783** |

### Ablation Study
Comparison of positional encoding strategies (semantic alignment scores):

| Method | 2-subjects | 3-subjects | 4-subjects | Position |
|------|-----------|-----------|-----------|----------|
| w/ h-w offset | 0.929 | 0.831 | 0.808 | 0.469 |
| w/ w offset | 0.915 | 0.824 | 0.777 | 0.459 |
| w/ h offset | 0.925 | 0.840 | 0.805 | 0.437 |
| **w/ ours (frame)** | **0.922** | **0.870** | **0.821** | **0.508** |

### Key Findings
- The introduction of PSR rewards substantially improves subject consistency from 0.559 (SFT) to 0.673, a 20.4% gain, with particularly pronounced advantages on the Three/Four subsets (0.615/0.571 vs. the previous best of 0.552/0.508).
- The resolution reduction (1024→512) during SFT degrades aesthetic scores, but the PSR RL stage effectively recovers and surpasses the original model.
- Frame-wise positional encoding outperforms the second-best method by 0.039 on the Position subset, demonstrating that alternative encodings introduce fixed spatial layout biases.
- In user studies, PSR achieves the highest ratings across all three dimensions (SC 0.92, SA 0.80, HPS 0.82).

## Highlights & Insights
- **Elegant closed-loop data construction pipeline**: The T2I → detection & cropping → single-subject personalization model pipeline cleverly repurposes the already-solved capability of single-subject personalization as a tool for multi-subject training data construction. This paradigm is highly transferable—any task lacking paired data can adopt a similar approach.
- **Subject disentanglement + pairwise rewards**: Using a detector within RL to decompose global images into subject-level reward signals follows a "divide-then-compare" strategy that is more precise than comparing global features directly, and is transferable to any generation task requiring fine-grained alignment.
- **Subtractive thinking in positional encoding**: Removing $h/w$ offsets yields better results, as it avoids the introduction of unnecessary spatial priors.

## Limitations & Future Work
- Training resolution is limited to 512×512, constraining the detail quality of generated images.
- Identity preservation for small-scale subjects remains challenging (acknowledged failure cases by the authors).
- The accuracy of the detector directly affects PSR reward quality—detection failures introduce noise into the reward signal.
- Validation is currently limited to FLUX.1 Kontext; transferability to other architectures (e.g., DiT, U-Net) remains unexplored.

## Related Work & Insights
- **vs. UNO**: UNO constructs training data by directly generating diptychs with T2I models, yielding low consistency; PSR leverages single-subject personalization models for higher-quality data construction. UNO's spatial $h/w$ offset introduces positional bias, whereas PSR offsets only along the frame dimension.
- **vs. OmniGen2**: OmniGen2 is competitive in text adherence but falls significantly behind PSR in subject consistency, demonstrating that SFT alone struggles to guarantee both simultaneously.
- **vs. Flow-GRPO/DanceGRPO**: These works apply RL to general T2I model improvement; PSR is the first to apply RL to the multi-subject personalization setting and introduces subject-level reward design.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The data construction pipeline and PSR reward design are elegant, though the overall SFT+RL two-stage framework follows a conventional paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Self-constructed benchmark with comprehensive evaluation, thorough ablations, and user studies.
- **Writing Quality**: ⭐⭐⭐⭐ Well-organized with rich figures and tables.
- **Value**: ⭐⭐⭐⭐ Provides a complete data + training + evaluation solution for multi-subject personalized generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] When Identities Collapse: A Stress-Test Benchmark for Multi-Subject Personalization](when_identities_collapse_a_stress-test_benchmark_for_multi-subject_personalizati.md)
- [\[CVPR 2026\] Self-Corrected Image Generation with Explainable Latent Rewards](self-corrected_image_generation_with_explainable_latent_rewards.md)
- [\[CVPR 2026\] Disentangling to Re-couple: Resolving the Similarity-Controllability Paradox in Subject-Driven Text-to-Image Generation](disentangling_to_re-couple_resolving_the_similarity-controllability_paradox_in_s.md)
- [\[CVPR 2026\] DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning](dreamvideo-omni_omni-motion_controlled_multi-subject_video_customization_with_la.md)
- [\[CVPR 2026\] Image Diffusion Preview with Consistency Solver](image_diffusion_preview_with_consistency_solver.md)

</div>

<!-- RELATED:END -->
