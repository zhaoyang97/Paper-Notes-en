---
title: >-
  [Paper Note] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation
description: >-
  [NeurIPS 2025][Image Generation][Unified Multimodal Models] This paper proposes CoRL (Co-Reinforcement Learning), a two-stage framework — Unified RL followed by Refined RL — that simultaneously optimizes both understandi…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Unified Multimodal Models"
  - "Reinforcement Learning"
  - "GRPO"
  - "Text-to-Image Generation"
  - "Multimodal Understanding"
date: 2026-05-08
content_hash: 6dcf2f41db8114dc
---

# Co-Reinforcement Learning for Unified Multimodal Understanding and Generation

**Conference**: NeurIPS 2025
**arXiv**: [2505.17534](https://arxiv.org/abs/2505.17534)  
**Code**: [https://github.com/mm-vl/ULM-R1](https://github.com/mm-vl/ULM-R1)  
**Area**: Image Generation
**Keywords**: Unified Multimodal Models, Reinforcement Learning, GRPO, Text-to-Image Generation, Multimodal Understanding

## TL;DR

This paper proposes CoRL (Co-Reinforcement Learning), a two-stage framework — Unified RL followed by Refined RL — that simultaneously optimizes both understanding and generation capabilities of Unified Multimodal Language Models (ULMs) via reinforcement learning, achieving synergistic co-evolution of dual capabilities: +7% on generation and +23% on understanding at 1.5B parameters.

## Background & Motivation

**Background**: Unified multimodal large language models (ULMs) handle both visual understanding and image generation, with representative works including Janus-Pro (fully autoregressive F-AR) and Show-o (autoregressive + diffusion hybrid). RL post-training has demonstrated significant gains in text-only LLMs (e.g., DeepSeek-R1), but its application in multimodal settings has been largely limited to reasoning enhancement for understanding tasks.

**Limitations of Prior Work**: (1) RL for visual generation remains largely unexplored — SimpleAR made a preliminary attempt using CLIP Score but with limited effectiveness; (2) more critically, applying RL to jointly optimize both understanding and generation capabilities of ULMs has not been explored; (3) applying RL to a single task not only yields limited gains on generation but may also degrade the other capability.

**Key Challenge**: ULM understanding and generation share the same LLM backbone, making independent optimization prone to conflicts. Existing RL methods (e.g., GRPO) design rewards primarily for text outputs and lack verifiable reward signals applicable to image generation.

**Goal**: To design an RL framework tailored for ULMs that enables understanding and generation capabilities to mutually benefit from shared policy optimization rather than interfering with each other.

**Key Insight**: The authors conduct a systematic pilot study comparing four RL strategies (separate RL / separate RL with weight merging / alternating RL / unified RL), finding that unified RL significantly outperforms the others, demonstrating that dual capabilities can co-evolve under shared optimization. This insight motivates a two-stage design: first establish cross-task synergy via unified RL, then specialize via refinement.

**Core Idea**: Simultaneously optimize both understanding and generation capabilities of ULMs within a unified GRPO framework, leveraging the synergistic effect of cross-task reward signals to achieve joint capability improvement.

## Method

### Overall Architecture

CoRL adopts a Foundation-then-Specialization two-stage RL pipeline. **Stage 1 (Unified RL)**: GRPO optimization is performed on a mixed dataset of 22K understanding and generation samples using a joint reward function, improving both capabilities simultaneously. **Stage 2 (Refined RL)**: Task-specific rewards and data are used to further enhance understanding (MCQ and OE subtypes separately) and generation independently. The base model is Janus-Pro-1B/1.5B, trained on 8× H20 GPUs.

### Key Designs

1. **Bidirectional Cycle Consistency Reward**:

    - Function: Provides verifiable semantic fidelity rewards for text-to-image generation tasks.
    - Mechanism: Evaluates generation quality from two directions — visual consistency is measured by LPIPS between the generated and real images; textual consistency is measured by first re-captioning the generated image with BLIP, then computing SPICE between the re-caption and the original prompt. $\mathcal{R}_{cycle} = 1 - \text{LPIPS}(\mathcal{I}_{real}, \mathcal{I}_{gen}) + \text{SPICE}(\mathcal{P}_{org}, \mathcal{C}_{re-cap})$, normalized to $[0,1]$.
    - Design Motivation: CLIP Score alone is too coarse and performed poorly in the pilot study. The bidirectional cycle forms a closed-loop feedback — simultaneously penalizing visual hallucination (via LPIPS) and semantic deviation (via SPICE), providing more comprehensive evaluation than unidirectional metrics.

2. **Text-Image Matching Reward**:

    - Function: Evaluates cross-modal alignment at the token level with fine granularity.
    - Mechanism: Using the ULM's own feature space, bidirectional maximum cosine similarity matching is computed between the text token representations $\mathbf{T}$ of the prompt and the visual token representations $\mathbf{I}$ of the generated image: $\mathcal{R}_{TIM} = \frac{1}{2}(\frac{1}{L_i}\sum_j \max_k \cos(\mathbf{i}_j, \mathbf{t}_k) + \frac{1}{L_t}\sum_k \max_j \cos(\mathbf{t}_k, \mathbf{i}_j))$
    - Design Motivation: CLIP Score only provides a global alignment score and cannot capture fine-grained concept-to-visual-element correspondences. Token-level matching within the ULM's own representation space yields finer granularity without relying on external models.

3. **Two-Stage Strategy: Unified RL (Stage 1) + Refined RL (Stage 2)**:

    - Function: First establishes a cross-task synergy foundation, then performs task-specific enhancement.
    - Mechanism: Stage 1 uses the joint reward $\mathcal{R}_{Uni} = \mathcal{R}_{cycle} + \mathcal{R}_{TIM} + \lambda(\mathcal{R}_{Acc} + \mathcal{R}_{Format})$ to simultaneously optimize understanding and generation, using standard GRPO with the KL divergence constraint removed to improve generalization. Stage 2 refines independently across three branches — generation uses $\mathcal{R}_{cycle} + \mathcal{R}_{TIM}$, MCQ understanding uses $\mathcal{R}_{MCQ-Acc} + \mathcal{R}_{Format}$, and OE understanding uses $\mathcal{R}_{OE-Acc} + \mathcal{R}_{Format}$; the KL constraint is reintroduced in this stage to prevent distribution shift.
    - Design Motivation: The pilot study explicitly demonstrates that unified RL outperforms separate RL, alternating RL, and weight merging. The two-stage design allows Stage 1 to establish a shared capability foundation and cross-task knowledge transfer, while Stage 2 performs targeted optimization without disrupting the established synergy.

### Loss & Training

Based on GRPO, 8 (Stage 1) or 16 (Stage 2) candidate responses are sampled per prompt, with group-normalized advantage values computed. Stage 1 learning rate is 4e-6, batch size 16, $\lambda=0.8$; Stage 2 learning rate is reduced to 1e-6. The final understanding model is obtained by combining the MCQ and OE refined models via a Gaussian distribution weight merging strategy.

## Key Experimental Results

### Main Results

| Benchmark | Janus-Pro-1.5B | ULM-R1 | Gain |
|-----------|---------------|--------|------|
| GenEval ↑ | 0.73 | **0.77** | +4.3 |
| DPG ↑ | 82.63 | **83.92** | +1.3 |
| WISE ↑ | 0.26 | **0.33** | +7 |
| MMMU ↑ | 36.3 | **42.3** | +6.0 |
| WeMath ↑ | 5.9 | **21.1** | +15.2 |
| LogicVista ↑ | 23.9 | **34.5** | +10.6 |
| MathVerse ↑ | 13.5 | **25.4** | +11.9 |
| MMVet ↑ | 39.8 | **43.9** | +4.1 |
| POPE ↑ | 86.2 | **88.9** | +2.7 |

### Ablation Study

| Configuration | GenEval | DPG | MMMU | WeMath | LogicVT |
|------|---------|-----|------|--------|---------|
| Baseline | 73.0 | 82.6 | 36.3 | 5.9 | 23.9 |
| + Cold-SFT (S1) | 72.8 | 82.5 | 41.0 | 18.0 | 27.9 |
| + Unified-RL (S1) | 75.9 | 83.3 | 40.3 | 14.0 | 30.2 |
| + Refined-RL w/ Cold-SFT | 74.5 | 82.8 | 41.8 | 22.5 | 35.9 |
| **CoRL (Unified+Refined)** | **77.3** | **83.9** | **42.3** | **21.1** | **34.5** |

### Key Findings

- **Unified RL is the key to synergistic co-evolution**: Comparing #1 (Cold-SFT base) and #2 (Unified-RL base), unified RL significantly outperforms on generation (GenEval 75.9 vs. 72.8), and yields larger gains on LogicVista for understanding (+6.3 vs. +4.0), demonstrating that cross-task RL synergy surpasses simple supervised learning.
- **Two-stage outperforms single-stage**: CoRL (#7) comprehensively outperforms #2 (unified RL only), with the refinement stage further improving per-task performance without disrupting the established synergy.
- **Mathematical reasoning shows the most significant gains**: WeMath +15.2, MathVerse +11.9, indicating that RL-induced long-chain reasoning is particularly effective for mathematical reasoning.
- ULM-R1 at 1.5B surpasses Janus-Pro at 7B on multiple benchmarks, highlighting the high efficiency of RL post-training.

## Highlights & Insights

- **First systematic validation of GRPO for joint dual-capability optimization in ULMs**: The pilot study clearly demonstrates the relative merits of four strategies; the superiority of unified RL establishes a clear design principle — shared optimization is preferable to decoupled optimization.
- **Self-supervised generation quality reward design**: The cycle consistency + TIM rewards do not rely on external large model scoring; instead, they leverage the ULM's own representation space and a simple re-captioning pipeline, reducing dependence on external reward models.
- **Efficiency of small models with RL post-training**: A 1.5B model trained with CoRL matches or exceeds 7B models on multiple benchmarks, demonstrating that RL post-training offers an highly attractive efficiency–performance trade-off.

## Limitations & Future Work

- Generation resolution is constrained by Janus-Pro's 384×384, far below the 512/1024 of mainstream diffusion models.
- Image generation sampling still requires CFG (guidance weight=5), increasing inference cost.
- RL training requires real-time image sampling from the ULM for reward evaluation, making training efficiency significantly lower than text-only RL.
- Applicability of RL to video generation/understanding remains unexplored.
- The optimal configuration of the weight merging strategy (used to combine MCQ/OE refined models) may vary across tasks.

## Related Work & Insights

- **vs. SimpleAR**: SimpleAR applies RL with CLIP Score for autoregressive generation, with limited effectiveness. CoRL's bidirectional cycle consistency + token-level matching reward design is more refined, and jointly handles understanding and generation.
- **vs. R1-like MLLMs** (e.g., Vision-R1, LMM-R1): These works apply RL only to enhance understanding/reasoning. CoRL is the first to extend RL to joint optimization of understanding and generation.
- **vs. DPO-based ULMs** (e.g., Emu3-DPO, HermesFlow): DPO requires preference data pairs, whereas CoRL uses verifiable rule-based rewards, requiring less data and offering greater flexibility.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of RL for joint dual-capability optimization in ULMs, though the methodological framework builds on existing GRPO.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Rigorous pilot study design, comprehensive coverage across 12 benchmarks, detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Overall clear; the pilot study section is particularly convincing.
- Value: ⭐⭐⭐⭐⭐ Provides a clear roadmap for RL post-training of ULMs; the cross-task synergy finding carries broad inspirational value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](../../CVPR2026/image_generation/learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)
- [\[NeurIPS 2025\] Show-o2: Improved Native Unified Multimodal Models](show-o2_improved_native_unified_multimodal_models.md)
- [\[NeurIPS 2025\] RLVR-World: Training World Models with Reinforcement Learning](rlvr-world_training_world_models_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction](toward_a_unified_geometry_understanding_riemannian_diffusion_framework_for_graph.md)
- [\[NeurIPS 2025\] Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models](mitigating_intra-_and_inter-modal_forgetting_in_continual_learning_of_unified_mu.md)

</div>

<!-- RELATED:END -->
