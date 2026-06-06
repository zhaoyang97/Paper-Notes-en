---
title: >-
  [Paper Note] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing
description: >-
  [NeurIPS 2025][LLM Reasoning][Video-to-Audio] ThinkSound is a three-stage interactive video-to-audio framework that leverages an MLLM to generate structured CoT reasoning as guidance for a unified audio foundation model.…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "Video-to-Audio"
  - "Chain-of-Thought"
  - "Audio Generation"
  - "Multimodal Reasoning"
  - "Flow Matching"
date: 2026-05-08
content_hash: cf68dbdfe07d565c
---

# ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing

**Conference**: NeurIPS 2025  
**arXiv**: [2506.21448](https://arxiv.org/abs/2506.21448)  
**Code**: [https://ThinkSound-Project.github.io](https://ThinkSound-Project.github.io)  
**Area**: LLM Reasoning  
**Keywords**: Video-to-Audio, Chain-of-Thought, Audio Generation, Multimodal Reasoning, Flow Matching

## TL;DR
ThinkSound is a three-stage interactive video-to-audio framework that leverages an MLLM to generate structured CoT reasoning as guidance for a unified audio foundation model. It achieves state-of-the-art performance on VGGSound and MovieGen Audio benchmarks while supporting object-level refinement and natural language instruction-based editing.

## Background & Motivation

**Background**: Video-to-audio (V2A) generation has evolved from end-to-end diffusion models (Diff-Foley, FoleyCrafter) to multimodal conditional generation (MMAudio, MultiFoley), with substantial quality improvements. However, existing methods remain black-box one-step generators that lack deep reasoning over visual content.

**Limitations of Prior Work**: Generating realistic audio requires reasoning in the manner of a professional sound designer—determining whether an owl is calling or flapping its wings, recognizing the rustling of branches, and synchronizing multiple audio events. End-to-end approaches compress these reasoning steps, frequently producing generic sounds that are misaligned with subtle visual cues.

**Key Challenge**: SonicVisionLM employs an MLLM for captioning followed by text-to-audio generation, but loses critical visual detail in the process. DeepSound-V1 introduces CoT reasoning but fragments the pipeline into three independent models. Neither approach fully exploits the reasoning capacity of MLLMs to guide a unified audio generation system.

**Goal**: To deeply integrate MLLM-based CoT reasoning into the V2A pipeline, enabling stepwise, interactive audio generation and editing.

**Key Insight**: Emulating the professional sound designer's workflow—first synthesizing the overall soundscape, then refining sounds for specific objects, and finally applying instruction-driven edits—with each stage guided by CoT reasoning.

**Core Idea**: An MLLM generates audio-specific CoT reasoning chains that serve as structured conditioning signals for a unified flow matching audio foundation model across three stages of audio generation.

## Method

### Overall Architecture

ThinkSound consists of two primary modules: (1) an MLLM fine-tuned from VideoLLaMA2, responsible for analyzing video/text inputs and producing structured CoT reasoning; and (2) a unified audio foundation model based on MM-DiT, which accepts multimodal conditioning from CoT, video, text, and audio context, and generates high-fidelity audio via flow matching. The full pipeline proceeds through three stages: base Foley generation → object-level interactive refinement → instruction-driven editing.

### Key Designs

1. **AudioCoT Dataset**:

    - Function: Constructs a large-scale multimodal CoT annotation dataset that bridges visual content, textual descriptions, and audio synthesis.
    - Mechanism: A three-stage automated pipeline—(a) VideoLLaMA2 and Qwen2-Audio extract visual and audio information, and GPT-4.1-nano synthesizes CoT chains; (b) Grounded SAM2 extracts ROI regions to generate object-level CoT; (c) editing CoT is generated based on four operations (extension/inpainting/addition/removal).
    - Design Motivation: Without large-scale audio CoT data, an MLLM cannot be trained to produce meaningful reasoning chains; existing datasets lack structured reasoning annotations.

2. **CoT Reasoning MLLM**:

    - Function: Fine-tunes VideoLLaMA2 to generate audio-specific structured reasoning.
    - Mechanism: Standard cross-entropy fine-tuning on AudioCoT endows the model with three capabilities: (a) audio understanding (acoustic properties, sound propagation, temporal causality); (b) structured decomposition (breaking complex audio scenes into actionable steps); (c) multimodal instruction following.
    - Design Motivation: General-purpose MLLMs lack the specialized reasoning capacity required for audio generation.

3. **Unified Audio Foundation Model (CoT-Guided MM-DiT)**:

    - Function: Generates high-fidelity audio from arbitrary combinations of input modalities.
    - Mechanism: Trained with flow matching. Key design choices include—(a) dual-path text encoding: MetaCLIP encodes visual captions for scene-level context, while T5-v1-xl encodes CoT reasoning to capture fine-grained temporal causal relationships; (b) hybrid Transformer: multi-stream blocks (modality-specific parameters with shared attention) followed by single-stream blocks; (c) adaptive fusion module: video features are upsampled and fused with audio latents via a gating mechanism; (d) classifier-free guidance dropout (random per-modality drop with probability 0.2) to support arbitrary input combinations.
    - Design Motivation: A unified architecture allows all three stages to share a single audio generation model, and CoT conditioning provides more precise generation guidance than raw captions.

4. **Click-Based Interaction Interface (Stage 2)**:

    - Function: Enables users to click on specific objects in the video to trigger object-level audio refinement.
    - Mechanism: Grounded SAM2 generates an ROI from the click location and tracks it across frames; the MLLM produces a dedicated CoT for the ROI; the foundation model conditions on the existing audio as context and synthesizes object-specific sounds that are blended into the output.
    - Design Motivation: Enables fine-grained audio control for non-technical users.

### Loss & Training
- VAE: Trained for 500K steps on Stability AI VAE (24×A800); the encoder is then frozen and the decoder is trained for an additional 500K steps.
- Foundation model: 100K steps (8×A100), batch size 256, learning rate $10^{-4}$.
- Task fine-tuning: 50K steps (8×A100) separately for each of the three stages.

## Key Experimental Results

### Main Results (VGGSound V2A Generation)

| Method | FD↓ | KL_PaSST↓ | DeSync↓ | CLAP_CoT↑ | MOS-Q↑ | MOS-A↑ |
|--------|-----|-----------|---------|-----------|--------|--------|
| MMAudio | 43.26 | 1.65 | 0.44 | 0.40 | 3.84 | 3.97 |
| **ThinkSound** | **34.56** | **1.52** | 0.46 | **0.46** | **4.02** | **4.18** |
| w/o CoT | 39.84 | 1.59 | 0.48 | 0.41 | 3.91 | 4.04 |

### OOD Evaluation (MovieGen Audio Bench)

| Method | CLAP_CoT↑ | DeSync↓ | MOS-Q↑ | MOS-A↑ |
|--------|-----------|---------|--------|--------|
| MMAudio | 0.45 | 0.77 | 3.95 | 3.62 |
| MovieGen | 0.47 | 1.00 | 3.98 | 3.70 |
| **ThinkSound** | **0.51** | **0.76** | **4.11** | **3.87** |

### Key Findings
- **Significant contribution of CoT reasoning**: Removing CoT raises FD from 34.56 to 39.84 (+15%) and lowers CLAP_CoT from 0.46 to 0.41, confirming that CoT supplies critical information on audio events, temporal relationships, and acoustic properties.
- **Strong OOD generalization**: ThinkSound achieves state-of-the-art performance on the unseen MovieGen benchmark, suggesting that CoT reasoning improves generalization.
- **Object-level and editing tasks**: ThinkSound substantially outperforms baselines on both object-level generation (FD 43.27 vs. MMAudio 44.46) and audio editing (FD 34.78 vs. AudioLDM-2 61.28).
- **Inference efficiency**: Generation time is only 1.07s, faster than MMAudio (3.01s) and FoleyCrafter (3.84s).

## Highlights & Insights
- **Well-motivated three-stage interactive workflow**: The design faithfully emulates the professional sound designer's process (global → local → revision), with MLLM CoT reasoning bridging user intent and audio synthesis at each stage.
- **High value of the AudioCoT dataset**: The automated CoT annotation pipeline is scalable to additional data sources, addressing the lack of audio CoT training data.
- **Dual-path text encoding (MetaCLIP + T5)**: The complementary combination of scene-level global context and fine-grained CoT reasoning substantially outperforms single-encoder alternatives.

## Limitations & Future Work
- **Dependency on additional MLLM inference**: Each generation requires a preceding MLLM pass to produce the CoT, adding system complexity (though the paper reports shorter overall generation time, likely attributable to faster audio synthesis).
- **CoT quality contingent on GPT-4.1-nano**: The dataset construction pipeline relies on a closed-source model, and CoT errors propagate downstream.
- **Unclear scale of human evaluation**: The number of evaluators and detailed setup for MOS scores are not sufficiently described in the main text.
- **No conversational interaction**: The three stages constitute a predefined linear pipeline with no support for iterative user-feedback-driven refinement.

## Related Work & Insights
- **vs. MMAudio**: MMAudio also employs flow matching with multimodal conditioning but lacks CoT reasoning. ThinkSound decomposes complex scenes into manageable sound components via CoT, improving FD by 20%.
- **vs. SonicVisionLM**: SonicVisionLM follows a two-stage video→text→audio pipeline in which visual detail is lost at the intermediate step; ThinkSound retains the video as a direct conditioning input throughout generation.
- **vs. DeepSound-V1**: DeepSound-V1 also employs CoT but fragments it across three independent models; ThinkSound covers all stages with a single unified foundation model.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-stage interactive framework integrating CoT reasoning into V2A generation is a novel design, and the AudioCoT dataset constitutes an independent contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation spans multiple benchmarks (VGGSound + MovieGen), three sub-tasks, ablation studies, and both objective and subjective metrics.
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured with rich figures and detailed method descriptions.
- Value: ⭐⭐⭐⭐ Demonstrates the utility of CoT reasoning in generative tasks (beyond pure understanding or reasoning), opening a new application direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Large Language Models Can Learn and Generalize Steganographic Chain-of-Thought under Process Supervision](large_language_models_can_learn_and_generalize_steganographic_chain-of-thought_u.md)
- [\[ICCV 2025\] CoRVid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning](../../ICCV2025/llm_reasoning/corvid_improving_multimodal_large_language_models_towards_chain-of-thought_reaso.md)
- [\[ICLR 2026\] PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation](../../ICLR2026/llm_reasoning/prismaudio_decomposed_chain-of-thoughts_and_multi-dimensional_rewards_for_video-.md)
- [\[NeurIPS 2025\] ProofSketch: Efficient Verified Reasoning for Large Language Models](proofsketch_efficient_verified_reasoning_for_large_language_models.md)
- [\[NeurIPS 2025\] MuSLR: Multimodal Symbolic Logical Reasoning](muslr_multimodal_symbolic_logical_reasoning.md)

</div>

<!-- RELATED:END -->
