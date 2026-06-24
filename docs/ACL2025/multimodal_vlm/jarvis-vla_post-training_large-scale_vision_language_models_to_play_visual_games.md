---
title: >-
  [Paper Note] JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games
description: >-
  [ACL 2025][Multimodal VLM][Vision-Language-Action Models] This paper proposes the ActVLP training paradigm, which introduces a vision-language post-training stage (incorporating world knowledge, visual alignment, and spatial grounding) prior to action imitation learning. Based on this, they construct JARVIS-VLA, the first VLA model capable of successfully executing over 1,000 atomic tasks in Minecraft, outperforming the best baseline by 40%.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Vision-Language-Action Models"
  - "Post-Training"
  - "Minecraft"
  - "Imitation Learning"
  - "Decision Making"
  - "Spatial Grounding"
date: 2026-05-08
content_hash: dde629916e96072c
---

# JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games

**Conference**: ACL 2025  
**arXiv**: [2503.16365](https://arxiv.org/abs/2503.16365)  
**Authors**: Muyao Li, Zihao Wang, Kaichen He (Peking University), Xiaojian Ma (BIGAI), Yitao Liang  
**Code**: [Open Source](https://craftjarvis.github.io/JarvisVLA)  
**Area**: Multimodal VLM  
**Keywords**: Vision-Language-Action Models, Post-Training, Minecraft, Imitation Learning, Decision Making, Spatial Grounding  

## TL;DR

This paper proposes the ActVLP training paradigm, which introduces a vision-language post-training stage (incorporating world knowledge, visual alignment, and spatial grounding) prior to action imitation learning. Based on this, they construct JARVIS-VLA, the first VLA model capable of successfully executing over 1,000 atomic tasks in Minecraft, outperforming the best baseline by 40%.

## Background & Motivation

### Background
Vision-Language-Action (VLA) models represent an emerging direction for applying pretrained VLMs to decision-making, generating actions through imitation learning on large-scale trajectory data. Pioneering works like VPT have demonstrated a pipeline in Minecraft where models are pretrained on large-scale YouTube videos and then fine-tuned via imitation learning, successfully accomplishing challenges like "ObtainDiamond". However, the core limitation of existing VLA methods is that they **focus solely on action post-training**, neglecting the enhancement of the foundation model itself.

### Limitations of Prior Work
- **Limitations of pure imitation learning**: Learning world knowledge solely from action-annotated trajectories is inherently challenging, and large-scale action-annotated datasets are highly scarce.
- **Methods like OpenVLA**: These models directly perform action fine-tuning on pretrained VLMs without utilizing environment-related vision-language data to enhance foundational capabilities.
- **Hierarchical agents** (e.g., Voyager, JARVIS-1): These agents rely on the zero-shot or few-shot reasoning of VLMs for high-level planning, but still require an additional low-level policy to execute the action.
- **Difficulty in generalization**: The complex coupling of observations and behaviors in trajectory data makes the pretraining paradigm difficult to transfer across tasks and environments.

### Design Motivation
To enhance the VLM's environment understanding, visual recognition, and spatial grounding capabilities through **non-trajectory vision-language tasks** before action learning, making it a better foundation model for decision-making.

## Method

### Model Architecture
JARVIS-VLA adopts a LLaVA-style architecture containing three core components:
- **Visual Encoder**: A ViT that converts raw images (644×364 resolution) into sequence patches.
- **Image Projection Module**: A two-layer MLP that projects image patch embeddings to align with the word embedding space.
- **Language Model**: An autoregressive Transformer serving as the core reasoning and decision-making engine.

Unlike OpenVLA, JARVIS-VLA employs a **non-Markovian architecture**, retaining historical observation image sequences in the prompt to maintain temporal context. For action decoding, discrete actions are merged into unified categories, and continuous actions (such as mouse movements) are discretized into 21 bins via $\mu$-law coding. A total of 51 special tokens (22 for mouse control and 29 for keyboard inputs) are allocated, reusing the lowest-frequency tokens in the tokenizer instead of modifying the original architecture.

### ActVLP Three-Stage Training Process

**Stage I: Language Model Post-Training**

The vision-related components (ViT and visual adapter) are frozen, and only the language Transformer is fine-tuned via SFT on approximately 277K Minecraft world knowledge text data to enhance the model's textual understanding of the decision-making environment.

**Stage II: Joint Post-Training of Visual Encoder and Language Model**

The VLM is completely unfrozen and fine-tuned using image captioning, visual question answering (VQA), and spatial grounding datasets. The vision-language alignment data contains 35K keyframes, and the spatial grounding data exceeds 404K entries. Both stages utilize the next-token prediction SFT loss:

$$\mathcal{L}_{\text{SFT}} = -\sum_{i=1} \log \mathcal{P}_{\theta}(x_i \mid x_v, x_{\text{ins}}, x_{1:i-1})$$

where $x_v$ represents the visual tokens, $x_{\text{ins}}$ is the instruction, and $x$ is the ground-truth response.

**Stage III: Action Post-Training**

The vision module is frozen, the language tokenizer is modified to incorporate action tokens, and the language Transformer is fine-tuned on trajectory data via imitation learning. The model learns to map text instructions and visual observations to action chunks:

$$\mathcal{L}_{\text{IL}} = -\sum_{t=1} \log \pi_{\theta}(a_{t:t+\tau} \mid o_t, x_{\text{ins}})$$

where $\pi_{\theta}$ represents the parameterized policy, and $a_{t:t+\tau}$ represents expert actions for $\tau$ consecutive steps. The action chunking technique improves the temporal consistency of actions as well as training efficiency.

### Dataset Composition
- **World Knowledge**: 277K Minecraft-related texts (Stage I).
- **Vision-Language Alignment**: 35K keyframes + descriptions and QA pairs generated by advanced VLMs (Stage II).
- **Spatial Grounding**: 404K+ target localization data entries (Stage II).
- **Trajectory Data**: 7.4M frames of Minecraft gameplay, including human control gameplay, YouTube videos, existing agent rollouts, and synthesized GUI operation data (Stage III).

## Key Experimental Results

### Experiment 1: Main Results on MCU Benchmark

| Model | Params | Mine Blocks | Kill Entities | Craft Items | Smelt Items |
|------|--------|-------------|---------------|-------------|-------------|
| VPT-BC | 248M | 0.33 | 0.44 | 0.41 | 0.05 |
| VPT-RL | 248M | 0.25 | 0.28 | 0.55 | 0.20 |
| STEVE-1 | 248M | 0.54 | 0.38 | 0.57 | 0.33 |
| GROOT | 248M | 0.67 | 0.52 | 0.40 | 0.30 |
| MineDreamer | 7B | 0.55 | 0.39 | 0.42 | 0.30 |
| Qwen2-VL (raw) | 7B | 0.79 | 0.84 | 0.60 | 0.07 |
| Qwen2-VL (IL) | 7B | 0.75 | 0.86 | 0.65 | 0.29 |
| **JARVIS-VLA-Qwen2** | **7B** | **0.88** | **0.95** | **0.77** | **0.70** |

JARVIS-VLA achieves state-of-the-art performance across all four task categories. Notably, on the Craft and Smelt tasks (which require precise GUI operations), the success rates reach 0.77 and 0.70, respectively, which are more than double those of the baseline models. Even the fine-tuned version of the raw Qwen2-VL without post-training (raw) outperforms several specialized baselines with 248M parameters.

### Experiment 2: Ablation of Training Paradigm

| Model | Craft Diamond Sword | Craft Ladder | Cook Beef | Smelt Iron Ingot |
|------|---------------------|-------------|-----------|-----------------|
| Qwen2-VL (raw) | 0.53 | 0.40 | 0.03 | 0.10 |
| Qwen2-VL (one-stage) | 0.10 | 0.40 | 0.07 | 0.13 |
| ActVLP-Qwen2-VL | **0.83** | **0.63** | **0.77** | **0.70** |

Separating vision-language post-training from action learning (as opposed to merging them into a single-stage training) yields a significant boost. The one-stage method performs even worse than the raw baseline, suggesting that mixing different training data types causes negative transfer. ActVLP improves the success rate on "Smelt Iron Ingot" by 57 percentage points compared to the one-stage baseline.

## Key Findings

- **Spatial grounding contributes the most**: Ablation studies show that among the three types of non-trajectory data, spatial grounding yields the most significant improvement on downstream decision-making tasks, as precise target localization is a crucial prerequisite for action execution.
- **Scaling laws exist**: Increasing the volume of non-trajectory vision-language post-training data leads to a linear correlation between the downstream success rate and the post-training evaluation loss. Increasing downstream trajectory data also improves success rates, but non-zero success rates only emerge once the loss drops below 0.30.
- **High resolution is critical**: The 644×364 resolution (much higher than VPT's 128×128) is a key factor enabling the substantial lead in Craft/Smelt GUI tasks.
- **Outperforming the IL baseline with only 21% of trajectory data**: JARVIS-VLA-Qwen2-VL uses only one-fifth of the trajectory data compared to Qwen2-VL(IL) but achieves an improvement of over 15% in performance.
- **Effectiveness across different VLM backbones**: The effectiveness of the ActVLP paradigm is validated on both LLaVA-Next and Qwen2-VL backbones.

## Highlights & Insights

- **Paradigm Innovation**: This work is the first to systematically propose a three-stage paradigm that introduces vision-language post-training prior to VLA action training, moving away from the simplistic "pretrained VLM $\to$ action fine-tuning" pipeline.
- **Value of Non-Trajectory Data**: It reveals that non-trajectory vision-language tasks (VQA, captioning, spatial grounding) significantly contribute to enhancing decision-making capabilities, demonstrating a scaling effect similar to LLMs.
- **Robust Experimental Comparisons**: The ablation of separate vs. mixed training directly proves the necessity of staged post-training, whereas the single-stage approach actually degrades performance.
- **Practical Architectural Design**: Engineering decisions such as reusing the lowest-frequency tokens for action representation, non-Markovian multi-frame observations, and action chunking are highly reasonable and require no modification to the raw VLM architecture.
- **Comprehensive Open-Source Release**: The code, models, and datasets are fully open-sourced, facilitating future research.

## Limitations & Future Work

- **Limited Inference Speed**: The inference throughput of the 7B-parameter VLA model falls far short of the real-time operational rate of human players (40Hz+). The paper suggests integrating MoE in the future to improve efficiency.
- **Still Inferior to Top Humans**: Although achieving SOTA, the success rate is still lower than that of high-level human players (90%+).
- **Limited to Minecraft**: While the ActVLP paradigm is generalizable, the experiments in this paper are validated only in Minecraft, making its transferability to real-world scenarios like robotic manipulation unknown.
- **High Data Construction Cost**: Constructing spatial grounding and vision-language alignment datasets relies heavily on advanced tools like SAM2 and multiple VLMs, making the data generation pipeline quite heavy.
- **Lack of Reinforcement Learning Stage**: The model relies solely on imitation learning and lacks RL capabilities to correct suboptimal trajectories.
- **The $t_{\mathrm{mix}}^2$ Factor of Non-Trajectory Data**: The construction of spatial grounding data depends on SAM2 and domain-specific annotation tools, which require rebuilding when migrating to new scenes.

## Related Work & Insights

- **VPT (Baker et al., 2022)**: A pioneer in the pretraining + IL paradigm, with 248M parameters and reliance on massive YouTube videos. This work builds upon it by replacing the base policy with a VLM and adding vision-language post-training.
- **OpenVLA (Kim et al., 2024)**: A VLA method that directly fine-tunes actions on a pretrained VLM, neglecting foundational capability enhancement. This paper experimentally demonstrates the necessity of post-training.
- **RT-2 (Brohan et al., 2023)**: Proposes co-training on web data to enhance VLA generalization. Stages I and II in this work can be viewed as a more structured implementation of this idea.
- **STEVE-1 (Lifshitz et al., 2024)**: A text-conditioned policy combining VPT and MineCLIP. JARVIS-VLA substantially outperforms it across all tasks.
- **MineDreamer (Zhou et al., 2024)**: Uses a VLM to predict future frames to guide the STEVE-1 policy; it is a hierarchical architecture and performs worse than end-to-end VLA.
- **OmniJARVIS (Wang et al., 2024d)**: Employs a behavior tokenizer to model trajectories but still requires an extra policy for action grounding, whereas JARVIS-VLA generates actions directly in an end-to-end manner.
- **GROOT (Cai et al., 2024c)**: A video-prompted task specification policy with 248M parameters, competitive in Mine/Kill but weaker in Craft/Smelt tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — The ActVLP paradigm is a meaningful innovation for the VLA training pipeline, though three-stage training itself is not groundbreaking.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablations (training paradigms, data types, scaling laws, backbone choices), but limited to a single environment (Minecraft).
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, highly informative figures and tables, with a clear explanation of the training pipeline.
- Value: ⭐⭐⭐⭐ — Highlights the importance of non-trajectory post-training for VLA, offering valuable insights for subsequent VLA research; outstanding open-source contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](../../CVPR2025/multimodal_vlm/post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[ICLR 2026\] Visual Jigsaw Post-Training Improves MLLMs](../../ICLR2026/multimodal_vlm/visual_jigsaw_post-training_improves_mllms.md)
- [\[ACL 2025\] Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference](activating_distributed_visual_region_within_llms_for_efficient_and_effective_vis.md)
- [\[ACL 2025\] mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus](moscar_a_large-scale_multilingual_and_multimodal_document-level_corpus.md)
- [\[ACL 2025\] Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues](speaking_beyond_language.md)

</div>

<!-- RELATED:END -->
