---
title: >-
  [Paper Note] EARL: Towards a Unified Analysis-Guided Reinforcement Learning Framework for Egocentric Interaction Reasoning and Pixel Grounding
description: >-
  [ICML 2026][Object Detection][Ego-IRG] EARL employs a "coarse analysis–fine response" two-stage MLLM framework to unify egocentric interaction understanding tasks (description + QA + pixel mask) into a single pipeline: t…
tags:
  - "ICML 2026"
  - "Object Detection"
  - "Ego-IRG"
  - "coarse-to-fine"
  - "Analysis-guided Feature Synthesizer"
  - "multi-faceted reward"
  - "SAM2"
date: 2026-05-08
content_hash: 5b933b98b740da78
---

# EARL: Towards a Unified Analysis-Guided Reinforcement Learning Framework for Egocentric Interaction Reasoning and Pixel Grounding

**Conference**: ICML 2026  
**arXiv**: [2605.14742](https://arxiv.org/abs/2605.14742)  
**Code**: https://github.com/yuggiehk/EARL  
**Area**: First-person vision / MLLM / Pixel-level grounding / Reinforcement Learning (GRPO)  
**Keywords**: Ego-IRG, coarse-to-fine, Analysis-guided Feature Synthesizer, multi-faceted reward, SAM2

## TL;DR
EARL employs a "coarse analysis–fine response" two-stage MLLM framework to unify egocentric interaction understanding tasks (description + QA + pixel mask) into a single pipeline: the first stage outputs a global description of the entire image and uses the last hidden state as a semantic prior, which is then injected into the second stage via a novel Analysis-guided Feature Synthesizer. Joint training is performed using GRPO and three types of rewards (format/answer/grounding accuracy). On Ego-IRGBench, EARL surpasses Seg-Zero in cIoU by 8.37%.

## Background & Motivation

**Background**: First-person vision (FPV / egocentric) research has become a hot topic due to the proliferation of head-mounted devices (GoPro, Aria, etc.). The mainstream approach is to treat sub-tasks such as "action recognition," "image captioning," and "human interaction detection" independently, or to use MLLMs for end-to-end Ego-IRG (interaction reasoning + grounding): given a first-person image and a user query, the model must output (i) a global interaction analysis, (ii) an answer to the query, and (iii) pixel masks for relevant entities.

**Limitations of Prior Work**: (1) Mainstream general MLLMs (Qwen2.5-VL/InternVL3) achieve cIoU only in the 20–30 range on Ego data, failing to learn the geometric constraints of "hand-object" in the ego view; (2) Ego-IRG-specific methods like ANNEXE generate good analysis text (CIDEr 1.49) but grounding remains at only 36% cIoU; (3) RL-based Seg-Zero / Seg-R1 push generic segmentation reasoning to around 57% but lack ego-specific "interaction context" priors.

**Key Challenge**: The information flow between the analysis (understanding) and response (answer + grounding) stages is disconnected—while the first stage may understand "the hand is holding a cup," the second must re-interpret the image from scratch for the query. Naively concatenating the two stages introduces "noisy priors": not all features from the analysis stage are useful, and blind fusion can degrade grounding.

**Goal**: Explicitly transmit semantic priors from coarse analysis to fine response, design a fusion module that selectively utilizes these priors, and jointly optimize the heterogeneous objectives of "text correctness" and "box accuracy" using RL.

**Key Insight**: The authors observe that the last hidden embedding of the VLM decoder in the analysis stage serves as a natural "global interaction descriptor" ($\mathbf{F}_{ana}$). Designing a "select-then-fuse" module can prevent noise contamination.

**Core Idea**: Employ a coarse-to-fine two-stage approach + AFS (refine analysis priors via self-attention before adding to main features) + GRPO multi-faceted reward to jointly address text, box, and mask outputs.

## Method

### Overall Architecture
Input: a first-person image $\mathcal{I}$ + user query $T_q$. The pipeline consists of two stages:

- **Stage 1 (coarse-grained interpretation)**: With a fixed prompt $T_a$ = "Please analyze the interactions of hands and objects in detail," the VLM decoder $\mathcal{D}_{vlm}$ outputs a global description $T_{ana}$. **Key byproduct**: the last hidden state is used as the global descriptor $\mathbf{F}_{ana}$. This stage is trained with standard cross-entropy loss.
- **Stage 2 (fine-grained response)**: Another VLM decoder $\mathcal{D}_{vlm}^{\prime}$ (Qwen2.5VL-7B) generates a textual answer $T_{ans}$ and bounding boxes $\mathcal{B}$ for the query; the boxes are fed to a frozen SAM2 to obtain the final mask $\mathcal{M}$. The AFS injects $\mathbf{F}_{ana}$ into the main features between the two stages. This stage is trained with GRPO and three types of rewards.

### Key Designs

1. **Coarse-to-fine Two-stage + Explicit Semantic Prior Transmission**:

    - **Function**: The global analysis from the first stage provides the second stage with the prior of "already understanding the image."
    - **Mechanism**: The first stage outputs not only text $T_{ana}$ but also **extracts the last hidden state of $\mathcal{D}_{vlm}$** as $\mathbf{F}_{ana}\in\mathbb{R}^{bs\times dim_i}$. In the second stage, $\mathbf{F}_{ana}$, the query encoding $\mathcal{E}_t^{\prime}(T_q)$, and the visual encoding $\mathcal{E}_v(\mathcal{I})$ are input to AFS to obtain a unified representation $\mathbf{F}_R=\mathcal{F}_s(\mathcal{E}_v(\mathcal{I}),\mathcal{E}_t^{\prime}(T_q),\mathbf{F}_{ana})$. $\mathbf{F}_R$ drives the outputs for text answer, box, and mask.
    - **Design Motivation**: Using the hidden state rather than $T_{ana}$ text as the prior has two advantages—(i) higher information density, avoiding loss from text compression; (ii) natural alignment with the feature space of the second stage, enabling smoother fusion. This is much stronger than naive prompt concatenation.

2. **Analysis-guided Feature Synthesizer (AFS)**:

    - **Function**: Selectively fuses $\mathbf{F}_{ana}$ into the multimodal main feature $\mathbf{F}_{emb}$ (output of Qwen2.5-VL after visual-text alignment), avoiding "noisy priors" that could harm grounding.
    - **Mechanism**: (i) An MLP $\phi_m$ reduces the dimension of $\mathbf{F}_{ana}$ to $dim$, followed by LayerNorm; (ii) reshape to $bs\times h\times w$ ($h=w=\sqrt{dim}$), then apply convolution to generate $\mathcal{Q},\mathcal{K},\mathcal{V}$; (iii) self-attention $\mathbf{F}=\text{softmax}(\mathcal{Q}\mathcal{K}^\top/\sqrt{dim})\mathcal{V}$ refines the prior; (iv) another MLP $\phi_m^{\prime}$ projects to the same dimension as $\mathbf{F}_{emb}$, finally $\mathbf{F}_{out}=\mathbf{F}_{emb}+\phi_m^{\prime}(\mathbf{F})$.
    - **Design Motivation**: Direct concatenation or cross-attention between $\mathbf{F}_{ana}$ and main features brings in noisy dimensions; using self-attention to **reweight tokens within the prior** first achieves "select then fuse," allowing important semantic dimensions to pass while suppressing noise. This simple structure solves the key engineering challenge of using hidden states as priors.

3. **GRPO + Joint Optimization of Heterogeneous Outputs with Three Rewards**:

    - **Function**: Uses Group Relative Policy Optimization to simultaneously optimize the heterogeneous objectives of "format correctness," "answer semantic relevance," and "grounding accuracy" without a critic.
    - **Mechanism**: (i) Format reward $\mathcal{R}_f$: checks if the output matches the required template (e.g., includes `<answer>` / `<box>` tags); (ii) Answer reward $\mathcal{R}_a$: measures semantic similarity between $T_{ans}$ and the ground truth answer; (iii) Grounding reward $\mathcal{R}_g$: computes IoU between the mask from predicted boxes via SAM2 and the ground truth mask. GRPO averages rewards over K rollouts within a group as the baseline, avoiding the need for a critic. During training, the visual encoder, text encoder, and SAM2 are frozen; only the VLM decoder $\mathcal{D}_{vlm}^{\prime}$ and AFS are updated.
    - **Design Motivation**: Traditional SFT struggles to jointly optimize mixed semantic and geometric objectives ("text correctness + box accuracy"); DPO requires paired samples; GRPO, with group baseline and weighted sum of multiple rewards, is naturally suited for "multi-structure output" tasks. Using SAM2 as a frozen reward provider also prevents reward signals from being contaminated by noisy masks.

### Loss & Training
- Stage 1: cross-entropy loss $\mathcal{L}_{des}$ supervises $T_{ana}$.
- Stage 2: GRPO optimizes expected reward $\mathbb{E}[\mathcal{R}_f+\mathcal{R}_a+\mathcal{R}_g]$, with K-rollout group baseline. The backbone is Qwen2.5VL-7B, and the mask generator is SAM2.

## Key Experimental Results

### Main Results

Ego-IRGBench test set (covering analysis M/CIDEr, answer M/CIDEr, grounding cIoU):

| Method | Type | Analysis CIDEr | Answer CIDEr | cIoU |
|--------|------|---------------|--------------|------|
| Qwen2.5VL-7B | General | 0.119 | 2.477 | 23.71 |
| InternVL2.5-7B | General | 0.044 | 1.533 | 27.21 |
| ANNEXE | Ego-specific | 1.494 | 2.590 | 36.02 |
| Sa2VA-8B | Grounding-specific | 0.115 | 2.656 | 32.69 |
| Seg-R1-7B | RL grounding | 0.289 | 2.483 | 46.10 |
| Seg-Zero | RL grounding | 0.049 | 2.380 | 57.11 |
| **EARL** | Ego + RL | **1.522** | **6.682** | **65.48** |
| vs. runner-up | | +0.028 | +1.682 | **+8.37** |

OOD test (EgoHOS dataset, cross-dataset direct evaluation):

| Method | Total cIoU | Left Hand | Right Hand | Two-hand Objects |
|--------|------------|-----------|------------|------------------|
| LISA | 22.46 | 28.93 | 33.06 | 18.10 |
| Sa2VA-8B | 37.63 | 48.56 | 45.82 | 37.04 |
| **EARL** | see paper | - | - | - |

### Ablation Study

The paper conducts ablation on AFS and reward design (see paper Sec. 4.3 for details). From the main table, the following can be inferred:

| Configuration | Answer CIDEr | cIoU | Note |
|---------------|--------------|------|------|
| Qwen2.5VL-7B baseline | 2.477 | 23.71 | No coarse analysis |
| ANNEXE (two-stage but no AFS+GRPO) | 2.590 | 36.02 | Cascade only, no hidden injection |
| EARL (full) | 6.682 | 65.48 | Full method |

### Key Findings
- **Answer CIDEr increases by 1.68 points**—an unexpected side effect, indicating that explicit injection of analysis hidden not only aids grounding but also significantly improves answer accuracy.
- **cIoU +8.37 pp is entirely due to ego task knowledge**: Seg-Zero achieves 57 with generic images, while EARL, with ego analysis priors, jumps to 65, validating the "analyze before grounding" design philosophy.
- **OOD performance remains superior**, indicating that the semantic priors learned by AFS are not overfitted to Ego-IRGBench but capture generalizable ego-interaction knowledge.
- The analysis quality of Stage 1 (M=0.541) matches Sa2VA, suggesting that most grounding gains come not from improved analysis but from **explicit transmission of analysis features**, further validating the value of AFS.

## Highlights & Insights
- **Treating the VLM decoder's hidden state as an explicit, transferable semantic prior is a highly practical technique**: Previously, either text cascade (information loss) or full network parameter sharing (tight coupling) was used; EARL offers a third path. This trick can be directly transferred to any "understand then act" MLLM task, such as describing a chart before VQA, or describing code before bug localization.
- **"Self-attention refinement before adding to main features" is a good template for handling noisy priors**: Direct concatenation brings in noisy dimensions; AFS's lightweight self-attention for selection followed by residual addition is simple and reusable.
- **GRPO + multi-reward addresses heterogeneous outputs**: Normalizing and linearly combining format/semantic/geometric rewards, with group baseline training, avoids the need for a critic. This is valuable for all "multi-structure output" tasks (e.g., structured generation, code+test output).
- **OOD improvements indicate the architecture learns transferable "ego-view interaction geometry"**, directly benefiting downstream tasks like robot manipulation and AR assistive systems.

## Limitations & Future Work
- **Analysis CIDEr is 0.021 lower than ANNEXE**: This suggests that GRPO optimization for stage 2 may suppress the diversity of stage 1 analysis—a trade-off noted by the authors.
- **Dependence on SAM2 as mask generator and reward provider**: If SAM2 fails in certain scenarios (low light, motion blur), the reward signal is contaminated, and EARL cannot independently train a mask head.
- **Two-stage serial inference doubles latency**: Real-time AR/robotic scenarios require further distillation.
- **Reshaping the hidden state in AFS to $\sqrt{dim}\times\sqrt{dim}$ for conv processing is somewhat ad-hoc**, and may be less elegant than token-level attention.
- **Not validated on video streams**: The real application of Ego-IRG is in video; single-frame testing is only a first step.
- **Future directions**: (i) Use stage 1 box supervision for semi-supervised stage 2 initialization; (ii) Share parameters via LoRA to reduce total overhead; (iii) Replace SAM2 with a learnable lightweight mask head for end-to-end training.

## Related Work & Insights
- **vs. ANNEXE**: ANNEXE is also two-stage but only cascades at the text level, with no hidden prior transmission, so grounding is stuck at 36; EARL injects features via AFS + joint training with GRPO, reaching 65.
- **vs. Seg-Zero / Seg-R1**: Generic RL segmentation methods treat segmentation as reasoning but lack ego-specific "hand-object-action" priors; EARL's ego analysis stage boosts cIoU by 8.37, validating the importance of domain priors.
- **vs. Sa2VA**: Sa2VA achieves only 32.69 cIoU on ego because it learns generic pixel abilities via grounding supervision without interaction context; EARL demonstrates that ego tasks require "understand before segment."
- **vs. LISA / GSVA**: Pure referring image segmentation without ego interaction modeling, cIoU below 22; EARL changes the framework rather than just increasing data.

## Rating
- Novelty: ⭐⭐⭐⭐ AFS + hidden prior is a fresh engineering combination; coarse-to-fine + GRPO is not entirely novel but well integrated
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ in-domain + OOD fully covered, compared with 15+ baselines across general/Ego/segmentation/RL
- Writing Quality: ⭐⭐⭐⭐ Clear formulas, AFS architecture diagram is user-friendly, but ablation details are compressed until sec 4.3
- Value: ⭐⭐⭐⭐ Provides a reusable engineering template for "understand then act" MLLM tasks, with direct applicability to ego/AR/robotics

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] The COTe Score: A Decomposable Framework for Evaluating Document Layout Analysis Models](../../CVPR2026/object_detection/the_cote_score_a_decomposable_framework_for_evaluating_document_layout_analysis_.md)
- [\[AAAI 2026\] Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning](../../AAAI2026/object_detection/connecting_the_dots_training-free_visual_grounding_via_agent.md)
- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](../../AAAI2026/object_detection/vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [\[CVPR 2026\] Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](../../CVPR2026/object_detection/mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)
- [\[ICCV 2025\] Visual-RFT: Visual Reinforcement Fine-Tuning](../../ICCV2025/object_detection/visual-rft_visual_reinforcement_fine-tuning.md)

</div>

<!-- RELATED:END -->
