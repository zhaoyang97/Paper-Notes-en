---
title: >-
  [Paper Note] ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting
description: >-
  [CVPR 2025][Image Generation][Automatic T2I] This paper proposes ChatGen, which automates three tedious steps in text-to-image (T2I) generation: prompt engineering, model selection, and parameter configuration. Through a multi-stage evolutionary training strategy (ChatGen-Evo), it enables users to obtain high-quality generated images simply by describing their requirements through free-style chatting.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Automatic T2I"
  - "FreeStyle Chatting"
  - "Multi-Stage Evolution"
  - "Model Selection"
  - "Prompt Rewriting"
date: 2026-05-08
content_hash: 34e3ae90b63ee7f3
---

# ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting

**Conference**: CVPR 2025  
**arXiv**: [2411.17176](https://arxiv.org/abs/2411.17176)  
**Code**: [https://chengyou-jia.github.io/ChatGen-Home](https://chengyou-jia.github.io/ChatGen-Home)  
**Area**: Diffusion Models / LLM Agent  
**Keywords**: Automatic T2I, FreeStyle Chatting, Multi-Stage Evolution, Model Selection, Prompt Rewriting

## TL;DR
This paper proposes ChatGen, which automates three tedious steps in text-to-image (T2I) generation: prompt engineering, model selection, and parameter configuration. Through a multi-stage evolutionary training strategy (ChatGen-Evo), it enables users to obtain high-quality generated images simply by describing their requirements through free-style chatting.

## Background & Motivation

**Background**: Text-to-Image (T2I) generation models are becoming increasingly diverse, with thousands of customized models available on platforms like Civitai. However, to obtain satisfactory images, users must manually complete three tedious steps: writing suitable prompts (requiring knowledge of model-specific syntax), selecting the appropriate model (requiring knowledge of each model's strengths), and configuring parameters (such as sampling steps and CFG scale).

**Limitations of Prior Work**: (1) BeautifulPrompt only optimizes prompts without selecting models; (2) DiffAgent only performs model selection without rewriting prompts; (3) No existing method addresses all three steps simultaneously. More critically, prior methods do not support free-style chat inputs (e.g., "help me draw a girl in a cafe"), requiring users to provide formatted inputs, which diverges from real-world scenarios.

**Key Challenge**: Automated T2I is a multi-step reasoning task—prompt quality influences model selection, and model selection determines parameter configuration. Directly training an MLLM end-to-end using SFT causes the model to learn simple text mappings rather than reasoning capabilities, leading to poor generalization.

**Goal**: How to automatically complete prompt rewriting, model selection, and parameter configuration starting from a user's free-style chat input, thereby achieving end-to-end image generation.

**Key Insight**: Inspired by the multi-step reasoning concepts in OpenAI o1, this work decomposes automated T2I into three learnable sub-skills and trains the MLLM stage-by-stage, providing targeted feedback at each stage.

**Core Idea**: Use a multi-stage evolutionary strategy to sequentially train the MLLM's capabilities in prompt rewriting, model selection (via ModelToken vocabulary expansion), and parameter configuration, achieving full automation from free-style chatting to images.

## Method

### Overall Architecture
The ChatGenBench benchmark is constructed (consisting of 256K training and 14K testing samples from 6,807 customized Civitai models), supporting three input formats: single-text, multi-modal, and multi-turn history. ChatGen-Evo undergoes three-stage training based on an MLLM: Stage 1 utilizes SFT to learn prompt rewriting, Stage 2 utilizes ModelToken for model selection, and Stage 3 leverages in-context learning for parameter configuration. During inference, it sequentially executes the three steps to output the prompt, model, and parameters.

### Key Designs

1. **ChatGenBench Benchmark Construction**

    - **Function**: Provides the first automatic T2I benchmark supporting step-level evaluation for the entire pipeline.
    - **Mechanism**: Gathers 44,881 high-quality human demonstrations from Civitai (filtered by download and like counts), containing complete paths of prompts, models, and parameters. LLM role-playing (using 100+ professional roles) is employed to reverse-synthesize these demonstrations into free-style chat inputs. Multiple MLLMs (such as GPT-4o and Claude) combined with a temperature of 0.9 are used to increase diversity. It supports three input formats: single-text, multi-modal (text + image), and multi-turn history.
    - **Design Motivation**: Previously, no benchmark covered the simultaneous evaluation of prompt quality, model selection accuracy, and parameter configuration, making it impossible to locate automation bottlenecks.

2. **Multi-Stage Evolution**

    - **Function**: Decomposes complex multi-step reasoning tasks into three independently learnable sub-skills.
    - **Mechanism**: **Stage 1 (Prompt Rewriting)**: Trains the MLLM using SFT to generate high-quality prompts from free-style chat inputs, optimizing only the prompt prediction targets. **Stage 2 (Model Selection)**: Creates a learnable ModelToken embedding for each T2I model and expands the vocabulary. During training, only the ModelToken embeddings are updated (other parameters are frozen), converting model selection into a classification task over the expanded vocabulary. **Stage 3 (Parameter Configuration)**: Based on the outputs of the first two steps, uses In-Context Learning (ICL) to let the MLLM configure parameters according to model documentation and examples without additional training.
    - **Design Motivation**: Ablation studies show that direct end-to-end SFT (ChatGen-Base 8B) underperforms the staged ChatGen-Evo 2B across all metrics, proving the necessity of decomposing multi-step reasoning.

3. **ModelToken Model Selection**

    - **Function**: Enables the MLLM to efficiently select the best model from thousands of customized T2I models.
    - **Mechanism**: Adds a new token embedding to the MLLM vocabulary for each T2I model. During training, given the prompt and model token, only the ModelToken embeddings are updated (without affecting other capabilities of the MLLM). Model selection essentially becomes "predicting the most appropriate ModelToken given a prompt."
    - **Design Motivation**: When directly using model names or descriptions for selection, MLLMs fail to distinguish between models with similar semantics. ModelToken provides a unique, learnable representation for each model.

### Loss & Training
The three stages apply: Stage 1 uses standard SFT cross-entropy loss (supervising only the prompt); Stage 2 uses cross-entropy loss (freezing the MLLM and training only the ModelToken embeddings); Stage 3 involves no training (ICL). Training ChatGen-Evo 2B takes approximately 100 hours, with an inference time of 1.9s per sample.

## Key Experimental Results

### Main Results

| Method | Parameters | Prompt BERTScore↑ | Selection Acc↑ | Parameter Acc↑ | FID↓ | CLIP↑ | ImageReward↑ | Overall↑ |
|------|------|-------------------|------------|---------|------|-------|-------------|------|
| ChatGen-Base 8B | 8B | 0.208 | 0.264 | 0.509 | 20.8 | 70.7 | 4.0 | 60.7 |
| **ChatGen-Evo 2B** | **2B** | **0.247** | **0.328** | **0.537** | **19.1** | **72.9** | **8.9** | **65.9** |

### Ablation Study

| Configuration | Prompt BERTScore↑ | Selection Acc↑ | Overall↑ |
|------|-------------------|---------------|------|
| ChatGen-Base (End-to-End SFT) | 0.208 | 0.264 | 60.7 |
| Stage 1 only | 0.247 | — | — |
| Stage 1+2 | 0.247 | 0.328 | — |
| **Stage 1+2+3 (Full)** | **0.247** | **0.328** | **65.9** |

### Key Findings
- The multi-stage evolutionary 2B model comprehensively outperforms the end-to-end 8B model, demonstrating the effectiveness of decomposing multi-step reasoning.
- The ModelToken design is critical for improving model selection accuracy (from 0.264 to 0.328) without affecting prompt rewriting quality.
- Performance drops significantly in the few-shot setting (model selection Acc decreases from 0.328 to 0.231), indicating that generalization to new/rare models remains an open problem.
- Although multi-stage inference is slower (1.9s vs. 1.1s), it yields significantly higher quality.

## Highlights & Insights
- **Valuable Problem Definition**: This work is the first to study full-pipeline T2I automation as a unified problem, with ChatGenBench providing a standardized evaluation framework.
- **Elegant ModelToken Design**: Translating model selection into vocabulary expansion and embedding learning is more precise than using text-based description selections and does not interfere with other capabilities.
- **Cognitively Aligned Staged Training**: The model first learns expression (prompt), then selection (model), and finally configuration (parameters), with each step conditioned on the output of the previous step, mirroring human workflows.

## Limitations & Future Work
- Model selection generalizes poorly in few-shot scenarios, requiring ModelToken updates when new models are introduced online.
- Relying on Civitai community data may introduce distribution bias (over-representation of specific styles or themes).
- Parameter configuration utilizes training-free ICL, leaving substantial room for improvement in accuracy (0.537).
- Prompt quality is measured by BERTScore in evaluations, which may not fully reflect the actual generation quality.

## Related Work & Insights
- **vs. BeautifulPrompt**: Only rewrites prompts without selecting models, whereas ChatGen covers the entire pipeline.
- **vs. DiffAgent**: Only performs model selection, whereas ChatGen simultaneously rewrites prompts, selects models, and configures parameters.
- **vs. DiffusionGPT**: Uses LLMs for model routing but only supports around 20 models and lacks step-level evaluation, whereas ChatGen supports 6,807 models.

## Rating
- Novelty: ⭐⭐⭐⭐ The problem definition and ModelToken design are innovative, though individual component technologies are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely solid evaluation with 256K training data, 14K testing samples, step-level evaluation, various baselines, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear description of problem motivation and methods, coupled with a meticulously designed benchmark.
- Value: ⭐⭐⭐⭐ Directly addresses key pain points of T2I usage, offering clear value for non-expert users.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] EasyCraft: A Robust and Efficient Framework for Automatic Avatar Crafting](easycraft_avatar_crafting.md)
- [\[CVPR 2025\] From Elements to Design: A Layered Approach for Automatic Graphic Design Composition](from_elements_to_design_a_layered_approach_for_automatic_graphic_design_composit.md)
- [\[CVPR 2025\] Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation](towards_understanding_and_quantifying_uncertainty_for_text-to-image_generation.md)
- [\[CVPR 2025\] Minority-Focused Text-to-Image Generation via Prompt Optimization](minority-focused_text-to-image_generation_via_prompt_optimization.md)
- [\[CVPR 2025\] PreciseCam: Precise Camera Control for Text-to-Image Generation](precisecam_precise_camera_control_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
