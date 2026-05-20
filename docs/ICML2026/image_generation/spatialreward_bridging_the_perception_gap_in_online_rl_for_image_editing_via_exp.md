---
title: >-
  [Paper Note] SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning
description: >-
  [ICML 2026][Image Generation][Reward Model] The authors identify the "attention collapse" issue in MLLM-based editing reward models—where the model, instead of comparing the original and edited images…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Reward Model"
  - "Image Editing"
  - "Online RL"
  - "Think-with-Boxes"
  - "Spatial Reasoning"
date: 2026-05-08
content_hash: c2b341c7c0984403
---

# SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning

**Conference**: ICML 2026  
**arXiv**: [2602.07458](https://arxiv.org/abs/2602.07458)  
**Code**: Project page https://lorangan-ddup.github.io/SpatialReward/ (available)  
**Area**: Image Generation / Image Editing / RLHF / Reward Model / Multimodal Evaluation  
**Keywords**: Reward Model, Image Editing, Online RL, Think-with-Boxes, Spatial Reasoning

## TL;DR
The authors identify the "attention collapse" issue in MLLM-based editing reward models—where the model, instead of comparing the original and edited images, collapses attention onto sink tokens and makes blind judgments. They propose SpatialReward: an 8B model first predicts bounding boxes of edited regions, then uses these box tokens as anchors for interleaved cross-image reasoning. With a 260K-sample spatially-aware dataset and two-stage GRPO training, the method achieves SOTA on three reward benchmarks and boosts OmniGen2's GEdit-Bench score by +0.90 (twice the improvement of GPT-4.1).

## Background & Motivation

**Background**: Instruction-based image editing (InstructPix2Pix, MagicBrush, OmniGen, Qwen-Edit, FLUX, etc.) has recently extended "style transfer" to complex multi-region editing. Recent works like Flow-GRPO and Dance-GRPO introduce online RL into diffusion models, aligning editing with human preferences through interactive trial-and-error, significantly outperforming SFT. However, the effectiveness of online RL is bottlenecked by the reward model—the reward signal must be reliable, efficient, interpretable, and capable of fine-grained regional assessment.

**Limitations of Prior Work**: Existing reward designs fall into three categories, none suitable for online RL in editing tasks: (i) Pairwise rewards (e.g., MMRB2) excel at zero-shot relative ranking, but online RL requires absolute scalars; converting rankings to scalars introduces ambiguity and $O(N^2)$ inference cost is prohibitive. (ii) Pointwise discriminative (e.g., EditReward) regress human preference on VLM embeddings with a linear head, lacking explicit reasoning chains, high annotation cost, and poor scalability. (iii) Pointwise generative ("MLLM-as-a-judge", e.g., EditScore, GPT-5) can output chain-of-thought, but editing tasks require strict "cross-image region comparison"—current mainstream MLLMs lack explicit spatial anchors, and even top closed-source models like GPT-5 suffer from "attention collapse": attention distribution collapses onto a few sink tokens, the source image is almost ignored, effectively degrading to single-image evaluation and missing subtle differences.

**Key Challenge**: Online RL training dynamics require the reward model to perform fine-grained, cross-image, region-level discrimination and output absolute scalars. However, current MLLM evaluators lack spatial anchors to guide cross-image comparison, so neither prompt engineering nor parameter distillation can fix "attention collapse," leading to systematic deviation from human preferences.

**Goal**: (i) Clearly diagnose and quantify the "attention collapse" perception gap; (ii) Design an architecture that forces MLLMs to perform cross-image region comparison; (iii) Build a large-scale spatially-aware dataset to support this capability; (iv) Use GRPO to align preferences on hard samples; (v) Validate that this approach improves downstream online RL editing model quality.

**Key Insight**: The authors observe that humans judge edits via a two-step process: "first locate, then compare," which MLLMs lack. By explicitly having the model predict bounding boxes of edited regions before reasoning, and injecting these box tokens as "look here" hard pointers into the reasoning chain, attention can be redirected from sink tokens to the relevant pixel regions.

**Core Idea**: "Think-with-Boxes" introduces spatial anchors (bounding boxes) as interleaved tokens directly citable by the language model, forcing each region-level judgment to "look back" at the corresponding pixels, enabling fine-grained, evidence-based scoring. This capability is solidified into a stable reward signal via a spatial-prior data pipeline and two-stage SFT→GRPO training.

## Method

### Overall Architecture
SpatialReward models reward as a conditional generation task, mapping input $X$ to structured output $Y=(B, \mathcal T, s)$: $B$ is a sequence of bounding boxes, $\mathcal T$ is a textual rationale, and $s$ is a scalar score. The evaluation protocol, following VIEScore, decouples into Semantic Consistency (SC, including instruction following $s_{if}$ and source consistency $s_{con}$) and Perceptual Quality (PQ, including naturalness $s_{nat}$ and artifacts $s_{art}$); the final reward is hierarchically aggregated as $R_{final}=(S_{SC})^{\alpha}(S_{PQ})^{1-\alpha}$, with $\alpha=0.8$. The SC branch follows the "locate then compare" Think-with-Boxes path, while the PQ branch performs no-reference evaluation on the edited image only, thus fully separating the two types of judgments.

### Key Designs

1. **Think-with-Boxes Architecture (Core Mechanism for Forced Cross-Image Comparison)**:

    - **Function**: Uses interleaved box tokens to explicitly encode "where to look" into the model's reasoning process, structurally eliminating attention collapse during MLLM evaluation.
    - **Mechanism**: The SC branch proceeds in three steps—(a) The model first predicts bounding boxes $B$ for all edited objects (Localization), outputting tokens like `<|bbox_0|>(x1,y1,x2,y2)`; (b) In Anchored Verification, each occurrence of `<|bbox_id|>` in the rationale $\mathcal T$ forces the model to "look back" at the corresponding pixel region, with a `<|global|>` token triggering global context scan; (c) Finally, outputs SC scores $s_{sc}=[s_{if}, s_{con}]$. The PQ branch only receives the edited image, $B=\emptyset$, and outputs textual rationale and $s_{pq}=[s_{nat}, s_{art}]$. The model backbone is Qwen-3-VL-8B-Instruct.
    - **Design Motivation**: MLLMs collapse to sink tokens in cross-image tasks because they are never forced to ground to specific pixels; embedding box tokens in text requires the model to "look back" at the corresponding region for each cite, restoring a healthy cross-image attention distribution. Fig.1c's attention visualization shows SpatialReward's attention realigned to the relevant source image regions.

2. **Spatial-Prior-Guided Data Pipeline (260K High-Quality Dataset: SpatialReward-260k)**:

    - **Function**: Constructs a large-scale dataset aligning box, rationale, and score, enabling SFT to learn the "ground then reason" paradigm.
    - **Mechanism**: Three-step pipeline—Step I: Qwen-3-VL-235B-A22B-Instruct pre-generates bounding boxes $B$ for all samples as spatial priors; Step II: Route to expert models by category—Gemini-2.5-Pro with crop prompts for human edits, GPT-5 with overlaid bounding boxes for object edits (forcing spatial focus), PQ evaluated independently by GPT-5; Step III: Generated $\mathcal T_{raw}$ and $B$ are fed back to Qwen-3-VL-235B for alignment (interleaved format) and hallucination check (discard if $\mathcal T$ and $B$ are visually inconsistent). The final 260K dataset comprises: 100K cleaned EditScore data (with $B$ injected), 100K EditReward data with regenerated rationales (original coarse scores discarded), and 60K self-built multi-region edit data.
    - **Design Motivation**: Human annotation alone cannot scale to obtain aligned "box+rationale+score" data; expert routing plus visual box overlay leverages each teacher model's strengths (faces to Gemini, objects to GPT-5), and final hallucination check ensures a clean training distribution.

3. **Two-Stage Training: SFT + GRPO Online Consistency RL**:

    - **Function**: First, the model learns to generate structured $(B,\mathcal T,s)$ outputs; then, online RL aligns human consistency on hard samples, eliminating hallucinated scores.
    - **Mechanism**: Stage 1: SFT on Qwen-3-VL-8B-Instruct with 260K data, objective $\mathcal L_{SFT}=-\sum_t \log P_\theta(y_t|y_{<t}, X)$, with $Y$ as $(B,\mathcal T,s)$ for SC and $(\mathcal T,s)$ for PQ. Stage 2: 7K low-score hard samples are mined, Gemini-3.0-Flash acts as Online Supervisor, providing 0–1 consistency scores as reward for rollouts, objective GRPO $\mathcal J_{GRPO}=\mathbb E[\frac{1}{G}\sum_i \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}\hat A_i] - \beta D_{KL}(\pi_\theta\|\pi_{ref})$, with $\hat A_i=(r_i-\mathrm{mean}\{r_j\})/\mathrm{std}\{r_j\}$. The final reward uses weighted geometric mean $R=(S_{SC})^\alpha (S_{PQ})^{1-\alpha}$ instead of min or arithmetic mean.
    - **Design Motivation**: SFT only fits the "average level" of the teacher distribution and still hallucinates on long-tail hard samples; GRPO targets consistency on hard samples via groupwise optimization, compensating for SFT bias. Weighted geometric mean provides denser gradient signals than min (bucket principle) and penalizes "weakest link" better than arithmetic mean, which is key for RL training dynamics.

### Loss & Training
SFT uses standard cross-entropy; GRPO uses group size $G$, KL coefficient $\beta$, etc., as in DeepSeek-R1 defaults. Final reward aggregation parameters: $\alpha=0.80$, $w_{SC}=\{0.6, 0.4\}$ (instruction following:source consistency), $w_{PQ}=\{0.5, 0.5\}$, all determined by grid search on 2K validation samples.

## Key Experimental Results

### Main Results
Overall accuracy on three reward benchmarks (bold indicates highest, SpatialReward uses Qwen-3-VL-8B):

| Model | EditReward-Bench (Ovrl) | MMRB2 (Ovrl) | MER-Bench (Ovrl) |
|-------|------------------------|--------------|------------------|
| GPT-4.1 | 0.705 | 0.535 | 0.358 |
| GPT-5 | 0.755 | 0.619 | 0.423 |
| Gemini-2.5-Pro | 0.722 | 0.534 | 0.462 |
| Gemini-3.0-Flash | 0.769 | 0.621 | **0.508** |
| EditScore-8B (baseline) | 0.690 | 0.570 | 0.350 |
| EditReward (discriminative) | 0.792 | 0.657 | 0.448 |
| **SpatialReward (Ours, 8B)** | **0.803** | **0.661** | 0.483 |

Downstream online RL (based on OmniGen2 + Flow-GRPO, improvement $\Delta$ on GEdit-Bench-EN Overall):

| Reward signal | GEdit Ovrl | $\Delta$ |
|---------------|-----------|----------|
| Baseline OmniGen2 | 6.42 | — |
| w/ GPT-4.1 | 6.73 | +0.45 |
| w/ EditScore | 6.89 | +0.61 |
| w/ EditReward | 7.19 | +0.77 |
| **w/ SpatialReward** | **7.32** | **+0.90** |
| Same framework with stronger backbone (UniRef-Edit) | 7.46→7.56 | +0.10 |

### Ablation Study

| Configuration | EditReward-Bench Accuracy | Notes |
|---------------|--------------------------|-------|
| SFT baseline (no spatial anchor) | 0.743 | Starting point |
| SFT w/ Box Only (predict box, no cite) | 0.761 | Adding box already helps |
| SFT w/ Think-with-Box | 0.778 | Explicit cite further improves |
| **+ Online GRPO** | **0.803** | RL stage is most critical |
| Bucket Principle (min aggregation) | 0.774 | Sparse gradients, slow RL |
| Arithmetic Mean aggregation | 0.771 | Does not penalize weakest link |
| Weighted Geometric Mean (Ours) | **0.803** | Dense and penalizes weakest link |

Attention diagnosis (on 776 EditReward-Bench samples):

| Method | Entropy Diff $|\Delta H|$ ↓ | Source Entropy $H_{src}$ | Concentration ↓ | Cross-sample Consistency ↑ |
|--------|--------------------------|--------------------------|-----------------|---------------------------|
| Baseline | 3.48 ± 0.57 | 2.88 ± 0.71 | 0.84 ± 0.05 | 0.04 |
| **Ours** | **1.16 ± 1.10** | **5.71 ± 0.81** | **0.37 ± 0.14** | **0.12** |

### Key Findings
- Simply "predicting box first" yields a 1.8-point gain; adding "cite box to guide reasoning" adds another 1.7 points. This shows spatial anchors and active citing are two independent effective components—the former provides supervision, the latter restructures reasoning.
- On MER-Bench 4-Pair difficulty (requiring fine-grained sub-dimension discrimination), SpatialReward achieves 21.5% accuracy, surpassing Gemini-3.0-Flash (19.5%), demonstrating spatial anchors' effectiveness for the hardest fine-grained distinctions.
- In RL, SpatialReward's +0.90 is 1.5× EditScore (+0.61) and 2× GPT-4.1 (+0.45), with inference speed 1.5× faster than EditReward (thanks to vLLM integration).
- Attention diagnosis quantitatively confirms the "attention collapse" hypothesis: baseline concentration 0.84 → 0.37, source entropy 2.88 → 5.71, nearly restoring a symmetric, healthy distribution.

## Highlights & Insights
- For the first time, the long-assumed fact that "MLLM-as-a-judge does not compare across images" is quantitatively diagnosed as "attention collapse to sink tokens," and a simple box-cite mechanism provides a cure. This "diagnose then treat" paradigm is more convincing than simply proposing new data/losses.
- The essence of Think-with-Boxes is "enabling hard spatial pointers to be passed between generated tokens," a concept transferable to any cross-image/cross-region task (multi-document retrieval, UI verification, spatial QA), not limited to editing evaluation.
- Using weighted geometric mean for reward aggregation is an underrated engineering detail: min yields sparse gradients and inefficient RL, arithmetic mean fails to penalize weak links, geometric mean balances both. Fig.6 in the paper visually shows reward design's real impact on RL dynamics.
- The data pipeline's "expert routing + box overlay for forced focus + reverse alignment and consistency check" is a standard template for generating high-quality data from open-source models + closed-source teachers, and is reusable.

## Limitations & Future Work
- SpatialReward's reward is a single scalar; it does not backpropagate local rewards for each edited region to the generator. The authors acknowledge this as future work (region-level credit assignment could be combined with Flow-GRPO for denser local supervision).
- Spatial anchors rely entirely on boxes, not supporting arbitrary shapes (masks); for fine-grained tasks (e.g., hair-level editing), boxes may be too coarse.
- The GRPO stage uses Gemini-3.0-Flash as supervisor, which is still a closed-source model, limiting full open-source reproducibility of the reward model.
- The 8B model still lags behind Gemini-3.0-Flash on MER-Bench (0.483 vs 0.508); ranking second overall on MER-Bench indicates spatial anchors do not fully close the gap between 8B and trillion-parameter models on complex, multi-constraint reasoning.

## Related Work & Insights
- **vs EditScore (generative baseline)**: EditScore uses the same 8B backbone distilled from GPT-4.1 but lacks spatial anchors, with obvious attention collapse; this work shows that adding Think-with-Boxes alone yields +11.3% at equal parameter count.
- **vs EditReward (discriminative SOTA)**: EditReward only trains the instruction-following dimension, lacking source consistency modeling, making the generator prone to over-editing unspecified regions during online RL; this work's explicit SC dimension avoids content drift (see Fig.7 in the paper for strong contrast).
- **vs VIEScore / GPT-4.1 (UniPic) / RewardDance / OneReward**: These methods rely on strong models or yes/no token probabilities for reward, without architectural spatial mechanisms; this work shows that even with an 8B model, grounding reasoning to spatial anchors can consistently outperform trillion-parameter closed-source models.
- **vs Shikra / Qwen-VL / Kosmos-2 / Ferret (VLM spatial reasoning)**: These works show that explicit coordinate outputs strengthen object-attribute binding; this is the first to apply this insight to reward modeling, making "explicitly citing spatial coordinates during reasoning" the core mechanism for reward evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Opens a new design space for MLLM reward evaluation by "diagnosing attention collapse → introducing spatial anchor tokens → enabling hard pointer passing between generated tokens."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three reward benchmarks + online RL on two editing backbones + attention diagnosis + aggregation strategy ablation, all tightly connected.
- Writing Quality: ⭐⭐⭐⭐⭐ From "perception gap" to "Think-with-Boxes" to "two-stage training," the narrative and illustrations (Fig.1/2/5/6) are all very clear.
- Value: ⭐⭐⭐⭐⭐ Not only a SOTA editing reward model, but also a key component for open RL pipelines, directly usable for training proprietary editing models without relying on closed-source GPT-4.1.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling](../../ICLR2026/image_generation/editscore_unlocking_online_rl_for_image_editing_via_high-fidelity_reward_modelin.md)
- [\[ICLR 2026\] Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative Models](../../ICLR2026/image_generation/bridging_generalization_gap_of_heterogeneous_federated_clients_using_generative_.md)
- [\[CVPR 2026\] Probing and Bridging Geometry–Interaction Cues for Affordance Reasoning in Vision Foundation Models](../../CVPR2026/image_generation/probing_and_bridging_geometry-interaction_cues_for_affordance_reasoning_in_visio.md)
- [\[ICCV 2025\] EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing](../../ICCV2025/image_generation/eedit_rethinking_the_spatial_and_temporal_redundancy_for_efficient_image_editing.md)
- [\[ICML 2026\] CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning](cocoedit_content-consistent_image_editing_via_region_regularized_reinforcement_l.md)

</div>

<!-- RELATED:END -->
