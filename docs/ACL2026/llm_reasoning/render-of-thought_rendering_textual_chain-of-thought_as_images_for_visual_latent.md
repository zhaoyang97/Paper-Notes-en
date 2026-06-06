---
title: >-
  [Paper Note] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning
description: >-
  [ACL 2026][LLM Reasoning][Chain-of-Thought Compression] Proposes Render-of-Thought (RoT), which renders textual CoT reasoning steps into images for the first time. It utilizes a pre-trained vision encoder as a semantic a…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Chain-of-Thought Compression"
  - "Visual Latent Reasoning"
  - "Text-to-Image Rendering"
  - "CoT Token Compression"
  - "Self-distillation"
date: 2026-05-08
content_hash: 262478fe84f31235
---

# Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning

**Conference**: ACL 2026  
**arXiv**: [2601.14750](https://arxiv.org/abs/2601.14750)  
**Code**: [TencentBAC/RoT](https://github.com/TencentBAC/RoT)  
**Area**: LLM Reasoning  
**Keywords**: Chain-of-Thought Compression, Visual Latent Reasoning, Text-to-Image Rendering, CoT Token Compression, Self-distillation

## TL;DR

Proposes Render-of-Thought (RoT), which renders textual CoT reasoning steps into images for the first time. It utilizes a pre-trained vision encoder as a semantic anchor to align LLM hidden states to the visual embedding space, achieving 3-4x token compression and significant inference acceleration while maintaining the analyzability of the reasoning chain.

## Background & Motivation

**Background**: Chain-of-Thought (CoT) prompting has become a fundamental paradigm for unlocking complex reasoning abilities in LLMs, but the verbose nature of CoT leads to severe inference latency and memory consumption issues. Existing compression methods are mainly divided into two routes: explicit compression (token filtering, RL-motivated short paths) and implicit reasoning (encoding reasoning processes in latent spaces).

**Limitations of Prior Work**: Explicit compression remains limited by sparse token representations. Implicit reasoning methods (e.g., Coconut, CODI, CoLaR) compress thoughts into opaque continuous vectors but often lack supervision for intermediate reasoning processes, leading to a loss of analyzability—making it difficult to trace the model's reasoning logic or diagnose logical errors. Furthermore, many methods employ complex architectures, affecting training stability.

**Key Challenge**: The contradiction between compression efficiency and interpretability—high-compression latent reasoning sacrifices the traceability of the reasoning process, while explicit CoTs that maintain interpretability are too verbose.

**Goal**: To find a representation that can significantly compress CoT while maintaining the observability of the reasoning process.

**Key Insight**: Visual modalities naturally possess high information density—a single image can encode a large amount of textual information. By rendering CoT text into images, the complete reasoning process can be represented using a few tokens from a vision encoder, and the rendered images themselves remain visual, preserving analyzability.

**Core Idea**: Render textual CoT as single-line images and use a pre-trained vision encoder to extract embeddings as supervision targets. Train the LLM to autoregressively generate reasoning trajectories in the visual latent space. During inference, no actual rendering or vision encoding is required; only the LLM forward pass is needed.

## Method

### Overall Architecture

RoT comprises two stages: (1) A CoT rendering module converts textual reasoning steps into single-line dynamic-width images, from which embeddings are extracted via a vision encoder; (2) The LLM generates latent reasoning tokens through a projection head aligned to the visual embedding space. Training is conducted in two stages: first, freeze the LLM and vision encoder to train only the projection head for alignment; then, freeze the projection head and vision encoder to fine-tune the LLM via LoRA to learn autonomous generation of reasoning trajectories. Inference requires no rendering or visual encoding, proceeding solely through the LLM + projection head forward pass.

### Key Designs

1. **CoT Rendering Module**:

    - **Function**: Converts textual reasoning steps into compact visual representations.
    - **Mechanism**: Renders CoT text as single-line images with a fixed height of 32px and a width dynamically calculated based on text length. Uses white text on a black background, 20px font size, and 4px padding. The single-line format ensures that image patches are extracted strictly from left to right, naturally aligning with the text order and eliminating spatial ambiguity.
    - **Design Motivation**: Square images generate significant white space (producing meaningless embeddings) and multi-line wraps (introducing spatial ambiguity); the single-line dynamic width design eliminates these two issues.

2. **Stage I: Visual Alignment**:

    - **Function**: Establishes a mapping from LLM hidden states to the visual embedding space.
    - **Mechanism**: Freezes the LLM and vision encoder, training only a lightweight projection head (two-layer MLP + SwiGLU). An `<img_begin>` token is appended after the question to trigger visual reasoning. The projection head maps LLM hidden states to the visual embedding space, aligned with the vision encoder output using MSE loss: $\mathcal{L}_{align} = \frac{1}{K}\sum_{t=1}^{K}\|\hat{v}_t - v_t\|_2^2$. Simultaneously, cross-entropy loss is used to train the prediction of the `<img_end>` termination token and the final answer.
    - **Design Motivation**: Contrary to the typical MLLM direction (Vision → LLM), this work performs LLM → Vision projection. Utilizing a pre-trained vision encoder as a "semantic anchor" avoids the need to learn the representation space of reasoning tokens from scratch.

3. **Stage II: Latent SFT**:

    - **Function**: Teaches the LLM to autonomously generate visual reasoning trajectories and output the final answer.
    - **Mechanism**: Freezes the vision encoder and the aligned projection head, fine-tuning the LLM with LoRA. After the model generates a sequence of latent visual tokens, it outputs a termination symbol and the textual answer. Since the projection head is frozen, the LLM is implicitly constrained to generate hidden states that map to meaningful visual representations. This stage no longer applies explicit visual regression loss; it is trained only with the cross-entropy loss of the answer prediction.
    - **Design Motivation**: Decouples the alignment and reasoning stages—Stage I establishes the representation space, and Stage II learns to navigate within that space, avoiding instability caused by simultaneous learning.

### Loss & Training

Stage I: $\mathcal{L}_I = \mathcal{L}_{pred} + \lambda \mathcal{L}_{align}$, optimizing both alignment and prediction. Stage II: Only $\mathcal{L}_{pred}$, aiming for pure answer accuracy. Training uses the AdamW optimizer with lr=2e-5, training for 1 epoch in Stage I and 2 epochs in Stage II. Inference employs a static termination strategy with a fixed token budget (rather than dynamic termination) because dynamic termination is unstable on continuous latent representations.

## Key Experimental Results

### Main Results

| Model/Method | GSM8k-Aug Pass@1 | # L (tokens) | MultiArith Pass@1 | Avg. Efficiency Ratio |
|--------|------|------|----------|------|
| Qwen3-VL-4B SFT-CoT | 81.2% | 127.3 | 98.3% | 0.73 |
| Qwen3-VL-4B RoT (Ours) | 37.8% | 32.0 | 97.2% | 1.73 |
| CoLaR-2 (LLM-based) | 40.0% | 39.6 | 82.2% | - |
| Coconut | 16.9% | 6.0 | 60.3% | - |

### Ablation Study

| Configuration | GSM8k-Aug | MATH | Description |
|------|---------|------|------|
| Full RoT | 37.8% | 33.2% | Complete model |
| w/o Stage I | 24.8% | 22.2% | Performance drops significantly without visual alignment |
| w/o Stage II | 29.9% | 26.2% | Removing latent SFT also leads to a significant drop |

### Key Findings

- **Visual Alignment (Stage I) provides the most contribution**: Removing it causes GSM8k-Aug to drop from 37.8% to 24.8%, indicating that latent spaces without visual anchors are prone to representation collapse.
- **On simple tasks (MultiArith), RoT approaches CoT performance** (97.2% vs 98.3%), but token usage is only 32 vs 59, with the efficiency ratio increasing from 0.73 to 1.73.
- **Significant inference speedup**: On GSM-Hard, latency dropped from 8.55s to 1.84s (4.6x acceleration).
- **Single-line rendering is far superior to square rendering**: Eliminating white space and spatial ambiguity is key.
- **RoT outperforms the LLM-based method CoLaR-2 in OOD generalization** (SVAMP, MultiArith), attributed to the richer semantic supervision provided by the pre-trained vision encoder.

## Highlights & Insights

- **Vision Encoder as a Semantic Anchor**: This is a highly ingenious design—instead of forcing the vision encoder to learn something new, it leverages its existing structured representation space as a "coordinate system" for LLM reasoning. This avoids the instability of learning a latent space from scratch, achieving a true plug-and-play effect.
- **Visual Analyzability of the Reasoning Process**: Unlike other latent space reasoning methods, RoT's latent tokens can be visualized and analyzed by mapping them back into the visual space, making "black-box reasoning" traceable again.
- **Text → Image → Embedding Information Bottleneck**: The rendering process itself serves as a natural information bottleneck, forcing the LLM to learn the core structure of reasoning rather than surface-level tokens. This concept is transferable to other compression scenarios.

## Limitations & Future Work

- There is still a significant accuracy gap compared to CoT (GSM8k-Aug: 37.8% vs 81.2%), indicating that the visual latent space's expressive capacity is limited on highly difficult reasoning tasks.
- The fixed token budget (32/64) is inflexible; problems of different difficulties require reasoning chains of different lengths.
- Dependence on the quality of the pre-trained vision encoder; different encoders may lead to different alignment effects.
- Future work: Dynamic token budget allocation, multi-resolution rendering, and combining with RL to optimize reasoning chain quality.

## Related Work & Insights

- **vs Coconut/CODI**: Coconut and CODI compress reasoning within pure language latent spaces but lack supervision for intermediate processes; RoT provides structured supervision signals through visual anchors, resulting in better OOD generalization.
- **vs CoLaR**: CoLaR uses a dynamic compression mechanism for reasoning in the language latent space. While the average efficiency is similar, RoT shows a clear advantage on OOD datasets (SVAMP: 72.7% vs 57.7%), demonstrating the value of visual priors.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Paradigm-level innovation by rendering CoT reasoning as images and reasoning in visual latent space.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple models and datasets with sufficient ablation and analysis, though gaps remain on high-difficulty tasks.
- Writing Quality: ⭐⭐⭐⭐ Intuitive illustrations, clear methodology, and a logically consistent two-stage framework.
- Value: ⭐⭐⭐⭐ Opens a new direction for visual latent space reasoning, although practicality is currently limited by the accuracy gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](../../NeurIPS2025/llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)
- [\[ACL 2026\] CRISP: Compressing Redundancy in Chain-of-Thought via Intrinsic Saliency Pruning](crisp_compressing_redundancy_in_chain-of-thought_via_intrinsic_saliency_pruning.md)
- [\[ACL 2026\] Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization](is_chain-of-thought_really_not_explainability_chain-of-thought_can_be_faithful_w.md)
- [\[ICML 2026\] A Formal Comparison Between Chain of Thought and Latent Thought](../../ICML2026/llm_reasoning/a_formal_comparison_between_chain_of_thought_and_latent_thought.md)
- [\[ACL 2026\] ETR: Entropy Trend Reward for Efficient Chain-of-Thought Reasoning](etr_entropy_trend_reward_for_efficient_chain-of-thought_reasoning.md)

</div>

<!-- RELATED:END -->
