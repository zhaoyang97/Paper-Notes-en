---
title: >-
  [Paper Note] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning
description: >-
  [ACL 2026][LLM Reasoning][chain-of-thought compression] This paper proposes Render-of-Thought (RoT), the first approach to render textual CoT reasoning steps as images. It leverages a pretrained visual encoder as a seman…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "chain-of-thought compression"
  - "visual latent space reasoning"
  - "text-to-image rendering"
  - "CoT token compression"
  - "self-distillation"
date: 2026-05-08
content_hash: 3e01baa86a642434
---

# Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning

**Conference**: ACL 2026
**arXiv**: [2601.14750](https://arxiv.org/abs/2601.14750)  
**Code**: [TencentBAC/RoT](https://github.com/TencentBAC/RoT)  
**Area**: LLM Reasoning
**Keywords**: chain-of-thought compression, visual latent space reasoning, text-to-image rendering, CoT token compression, self-distillation

## TL;DR

This paper proposes Render-of-Thought (RoT), the first approach to render textual CoT reasoning steps as images. It leverages a pretrained visual encoder as a semantic anchor to align LLM hidden states to the visual embedding space, achieving 3–4× token compression and significant inference acceleration while preserving the interpretability of the reasoning chain.

## Background & Motivation

**State of the Field**: Chain-of-Thought prompting has become a foundational paradigm for unlocking complex reasoning in LLMs, but the verbosity of CoT leads to severe inference latency and memory consumption. Existing compression methods fall into two categories: explicit compression (token pruning, RL-incentivized shorter paths) and implicit reasoning (encoding reasoning in latent space).

**Limitations of Prior Work**: Explicit compression is still constrained by sparse token representations. Implicit reasoning methods (e.g., Coconut, CODI, CoLaR) compress thinking into opaque continuous vectors but typically focus only on output alignment without supervising intermediate reasoning steps, resulting in loss of reasoning chain interpretability—making it difficult to trace the model's logic or diagnose reasoning errors. Many methods also employ complex architectures that hinder training stability.

**Root Cause**: A fundamental tension exists between compression efficiency and interpretability—high-compression latent reasoning sacrifices traceability, while explicit CoT that preserves interpretability is excessively verbose.

**Paper Goals**: To find a representation that achieves substantial CoT compression while keeping the reasoning process observable.

**Starting Point**: The visual modality is inherently information-dense—a single image can encode a large amount of textual information. Rendering CoT text as images enables the full reasoning process to be represented with a small number of visual encoder tokens, and the rendered images remain human-interpretable, preserving analyzability.

**Core Idea**: Render textual CoT as single-line images; extract embeddings via a pretrained visual encoder as supervision targets; train an LLM to autoregressively generate reasoning trajectories in the visual latent space. At inference time, no actual rendering or visual encoding is required—only a forward pass through the LLM.

## Method

### Overall Architecture

RoT comprises two stages: (1) a CoT rendering module converts textual reasoning steps into single-line, dynamically-wide images, from which a visual encoder extracts embeddings; (2) the LLM generates latent reasoning tokens via a projection head that aligns them with the visual embeddings. Training proceeds in two stages: first, the LLM and visual encoder are frozen and only the projection head is trained for alignment; then the projection head and visual encoder are frozen while the LLM is fine-tuned with LoRA to learn autonomous generation of reasoning trajectories. At inference time, neither rendering nor visual encoding is required—only a forward pass through the LLM and projection head.

### Key Designs

1. **CoT Rendering Module**:

    - **Function**: Converts textual reasoning steps into compact visual representations.
    - **Mechanism**: CoT text is rendered as single-line images with a fixed height of 32 px and dynamically computed width based on text length. A black background with white 20 px font and 4 px padding is used. The single-line format ensures image patches are extracted strictly left-to-right, naturally aligned with textual order and eliminating spatial ambiguity.
    - **Design Motivation**: Square images produce large blank regions (yielding meaningless embeddings) and multi-line wrapping (introducing spatial ambiguity). The single-line dynamic-width design eliminates both issues.

2. **Stage I: Visual Alignment**:

    - **Function**: Establishes a mapping from LLM hidden states to the visual embedding space.
    - **Mechanism**: The LLM and visual encoder are frozen; only a lightweight projection head (two-layer MLP + SwiGLU) is trained. An `<img_begin>` token appended after the question triggers visual reasoning; the projection head maps LLM hidden states to the visual embedding space, aligned to visual encoder outputs via MSE loss: $\mathcal{L}_{align} = \frac{1}{K}\sum_{t=1}^{K}\|\hat{v}_t - v_t\|_2^2$. Cross-entropy loss is additionally applied to train prediction of the `<img_end>` termination token and the final answer.
    - **Design Motivation**: Unlike typical MLLMs (vision → LLM), this work performs projection in the LLM → vision direction. The pretrained visual encoder serves as a "semantic anchor," eliminating the need to learn the representation space of reasoning tokens from scratch.

3. **Stage II: Latent SFT**:

    - **Function**: Trains the LLM to autonomously generate visual reasoning trajectories and produce final answers.
    - **Mechanism**: The visual encoder and aligned projection head are frozen; the LLM is fine-tuned with LoRA. The model generates a sequence of latent visual tokens followed by a termination token and a textual answer. Because the projection head is frozen, the LLM is implicitly constrained to generate hidden states that map to meaningful visual representations. No explicit visual regression loss is applied in this stage—only cross-entropy loss on answer prediction.
    - **Design Motivation**: Decoupling alignment and reasoning across two stages—Stage I establishes the representation space, and Stage II learns to navigate within it—avoids the instability of learning both objectives simultaneously.

### Loss & Training

Stage I: $\mathcal{L}_I = \mathcal{L}_{pred} + \lambda \mathcal{L}_{align}$, jointly optimizing alignment and prediction. Stage II: $\mathcal{L}_{pred}$ only, targeting answer accuracy. Training uses the AdamW optimizer with lr = 2e-5; Stage I trains for 1 epoch and Stage II for 2 epochs. Inference uses a static termination strategy with a fixed token budget rather than dynamic termination, as dynamic termination is unstable over continuous latent representations.

## Key Experimental Results

### Main Results

| Model / Method | GSM8k-Aug Pass@1 | # L (tokens) | MultiArith Pass@1 | Avg. Efficiency Ratio |
|---|---|---|---|---|
| Qwen3-VL-4B SFT-CoT | 81.2% | 127.3 | 98.3% | 0.73 |
| Qwen3-VL-4B RoT | 37.8% | 32.0 | 97.2% | 1.73 |
| CoLaR-2 (LLM-based) | 40.0% | 39.6 | 82.2% | — |
| Coconut | 16.9% | 6.0 | 60.3% | — |

### Ablation Study

| Configuration | GSM8k-Aug | MATH | Note |
|---|---|---|---|
| Full RoT | 37.8% | 33.2% | Complete model |
| w/o Stage I | 24.8% | 22.2% | Large drop without visual alignment |
| w/o Stage II | 29.9% | 26.2% | Significant drop without latent SFT |

### Key Findings

- Visual alignment (Stage I) contributes most: removing it causes GSM8k-Aug to drop from 37.8% to 24.8%, indicating that latent spaces without visual anchors are prone to representational collapse.
- On simple tasks (MultiArith), RoT approaches CoT performance (97.2% vs. 98.3%) while using only 32 tokens vs. 59, improving the efficiency ratio from 0.73 to 1.73.
- Inference speed improves substantially: on GSM-Hard, latency drops from 8.55 s to 1.84 s (4.6× speedup).
- Single-line rendering substantially outperforms square rendering; eliminating blank regions and spatial ambiguity is key.
- RoT surpasses the LLM-based method CoLaR-2 in OOD generalization (SVAMP, MultiArith), attributed to richer semantic supervision provided by the pretrained visual encoder.

## Highlights & Insights

- **Visual encoder as a semantic anchor**: This is a particularly elegant design—rather than training the visual encoder to learn new representations, the method exploits its existing structured representation space as a "coordinate system" for LLM reasoning. This avoids the instability of learning a latent space from scratch and enables true plug-and-play integration.
- **Visualizable interpretability of the reasoning process**: Unlike other latent-space reasoning methods, RoT's latent tokens can be back-projected into the visual space for analysis, making "black-box reasoning" traceable once again.
- **Text → image → embedding as an information bottleneck**: The rendering process itself acts as a natural information bottleneck, forcing the LLM to learn the core structure of reasoning rather than surface-level tokens—a principle transferable to other compression settings.

## Limitations & Future Work

- Accuracy still lags substantially behind CoT (GSM8k-Aug: 37.8% vs. 81.2%), indicating limited expressive capacity of the visual latent space for high-difficulty reasoning tasks.
- Fixed token budgets (32/64) are inflexible; problems of varying difficulty require reasoning chains of different lengths.
- Performance depends on the quality of the pretrained visual encoder; different encoders may yield different alignment outcomes.
- Future directions include dynamic token budget allocation, multi-resolution rendering, and combining RoT with RL to improve reasoning chain quality.

## Related Work & Insights

- **vs. Coconut / CODI**: Coconut and CODI compress reasoning in a purely linguistic latent space but lack supervision over intermediate steps; RoT provides structured supervision via visual anchors, yielding better OOD generalization.
- **vs. CoLaR**: CoLaR employs a dynamic compression mechanism for reasoning in the linguistic latent space with comparable average efficiency, but RoT shows a clear advantage on OOD datasets (SVAMP: 72.7% vs. 57.7%), demonstrating the value of visual priors.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First work to render CoT reasoning as images and reason in the visual latent space—a paradigm-level innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model, multi-dataset evaluation with sufficient ablations and analysis, though performance gaps on harder tasks remain large.
- **Writing Quality**: ⭐⭐⭐⭐ Intuitive figures, clear methodology, and a logically coherent two-stage framework.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for visual latent space reasoning, though practical applicability is limited by the accuracy gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](../../NeurIPS2025/llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)
- [\[ACL 2026\] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)
- [\[ACL 2026\] CRISP: Compressing Redundancy in Chain-of-Thought via Intrinsic Saliency Pruning](crisp_compressing_redundancy_in_chain-of-thought_via_intrinsic_saliency_pruning.md)
- [\[ACL 2026\] Decoupling the Effect of Chain-of-Thought Reasoning: A Human Label Variation Perspective](decoupling_the_effect_of_chain-of-thought_reasoning_a_human_label_variation_pers.md)
- [\[CVPR 2026\] Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering](../../CVPR2026/llm_reasoning/step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)

</div>

<!-- RELATED:END -->
