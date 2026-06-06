---
title: >-
  [Paper Note] ViRC: Enhancing Visual Interleaved Mathematical CoT with Reason Chunking
description: >-
  [CVPR 2026 (Main Track)][Multimodal VLM][Visual Mathematical Reasoning] ViRC proposes a Reason Chunking mechanism that structures multimodal mathematical CoT into sequential Critical Reasoning Units (CRUs)…
tags:
  - "CVPR 2026 (Main Track)"
  - "Multimodal VLM"
  - "Visual Mathematical Reasoning"
  - "Reason Chunking"
  - "Critical Reasoning Unit"
  - "Multimodal CoT"
  - "Progressive Training"
date: 2026-05-08
content_hash: e62b967b1e95f0de
---

# ViRC: Enhancing Visual Interleaved Mathematical CoT with Reason Chunking

**Conference**: CVPR 2026 (Main Track)  
**arXiv**: [2512.14654](https://arxiv.org/abs/2512.14654)  
**Code**: [https://github.com/Leon-LihongWang/ViRC](https://github.com/Leon-LihongWang/ViRC)  
**Area**: Multimodal VLM / Mathematical Reasoning  
**Keywords**: Visual Mathematical Reasoning, Reason Chunking, Critical Reasoning Unit, Multimodal CoT, Progressive Training

## TL;DR
ViRC proposes a Reason Chunking mechanism that structures multimodal mathematical CoT into sequential Critical Reasoning Units (CRUs), simulating the process by which human experts repeatedly consult visual information and incrementally verify intermediate propositions. Through the CRUX dataset and a progressive training strategy (Instructional SFT → Practice SFT → Strategic RL), ViRC-7B achieves an average improvement of 18.8% across mathematical benchmarks.

## Background & Motivation

### State of the Field
Chain-of-Thought (CoT) has substantially improved the reasoning capabilities of LLMs, yet unique challenges arise in multimodal mathematical reasoning: existing MLLMs typically perform purely textual reasoning from a single static mathematical image, neglecting **dynamic visual acquisition** throughout the reasoning process.

### Limitations of Prior Work

**Single-pass visual reading**: The model observes the image once before initiating a long reasoning chain, without revisiting the image during inference — yet mathematical problems frequently require repeated inspection of different parts of a figure.

**Reasoning chain degradation**: In long-chain CoT, later reasoning steps tend to drift, as there are no checkpoints to verify intermediate conclusions.

**Miller's Law from cognitive science**: Human working memory has limited capacity (7±2 chunks), and excessively long reasoning chains exceed cognitive load.

### Root Cause
Existing multimodal mathematical reasoning treats the entire solution process as an undifferentiated long sequence, rather than decomposing it — as a human expert would — into multiple logical nodes at which visual information is re-acquired and intermediate propositions are verified.

### Core Idea
Introducing the **Reason Chunking mechanism** — decomposing CoT reasoning into sequential **Critical Reasoning Units (CRUs)**. Within each CRU, textual reasoning maintains coherence to verify a single intermediate proposition; between CRUs, visual information is integrated to generate the next proposition.

## Method

### Overall Architecture
The ViRC framework comprises three core components:
1. **CRU reasoning structure**: CoT is decomposed into $[CRU_1, CRU_2, ..., CRU_K]$, where each CRU encompasses visual acquisition, textual reasoning, and an intermediate conclusion.
2. **CRUX dataset**: Provides explicitly annotated CRU structures with multiple reasoning paths.
3. **Progressive training**: A three-stage training strategy that simulates human cognitive learning.

### Key Designs

#### 1. Critical Reasoning Unit (CRU) Structure
- **Function**: Decomposes the mathematical reasoning process into logical blocks, each focusing on a single intermediate proposition.
- **Mechanism**: Each CRU consists of three components:
    - **Visual Acquisition**: Extracts locally relevant information from the mathematical image for the current step via visual tools (e.g., crop, zoom, annotate).
    - **Textual Reasoning**: Performs logical inference based on the acquired visual information and the conclusion of the preceding CRU.
    - **Intermediate Verification**: Explicitly states the conclusion of the current reasoning step, serving as input to the next CRU.
- **Design Motivation**: Simulates the "observe → reason → verify → re-observe" cycle that human experts employ when solving mathematical problems, consistent with the cognitive principle of working memory chunking described by Miller's Law.

#### 2. CRUX Dataset Construction
- **Function**: Constructs a multimodal mathematical reasoning dataset with explicit CRU-level annotations.
- **Mechanism**: Three visual tools (crop, zoom, annotate) and four reasoning modes (direct derivation, proof by contradiction, construction, reduction) are employed to generate multiple reasoning paths per problem, with clear CRU boundaries annotated for each path.
- **Design Motivation**: CRU-level annotation enables the model to learn *when* to chunk and *how* to transfer information between chunks.

#### 3. Progressive Training Strategy
- **Instructional SFT**: Supervised fine-tuning on CRU-annotated data to acquire basic reasoning chunking capabilities.
- **Practice SFT**: Training on broader mathematical data without explicit CRU annotations, allowing the model to practice reasoning chunking autonomously.
- **Strategic RL**: Reinforcement learning to optimize reasoning strategy — rewarding correct final answers and high-quality intermediate steps.
- **Design Motivation**: Mirrors the three-stage human learning process of "learn concepts → practice → refine strategy," avoiding the training instability that arises from a single-stage approach.

## Key Experimental Results

### Main Results: Mathematical Reasoning Benchmarks

| Model | MathVerse (%) | MathVista (%) | GeoQA (%) | Avg. |
|-------|---------------|---------------|-----------|------|
| LLaVA-1.5-7B | 23.4 | 38.1 | 42.6 | 34.7 |
| Math-LLaVA-7B | 28.9 | 43.5 | 48.2 | 40.2 |
| InternVL2-7B | 31.2 | 46.8 | 51.3 | 43.1 |
| **ViRC-7B** | **37.1** | **52.4** | **57.8** | **49.1** |

Average gain of **+18.8%** over the baseline.

### Ablation Study

| Configuration | Avg. Accuracy (%) | Notes |
|---------------|-------------------|-------|
| Full ViRC | 49.1 | Complete method |
| w/o Reason Chunking | 41.3 | CRU structure removed; standard long-chain CoT |
| w/o Visual Tools | 44.6 | No visual tools within CRUs |
| w/o Strategic RL | 46.2 | Two-stage SFT only |
| w/o Progressive Training | 43.8 | Three stages merged into single training |

### Key Findings
- **Reason Chunking is the most critical contribution** — its removal leads to a 7.8% performance drop, demonstrating that reasoning chunking is essential for mathematical reasoning.
- **Dynamic visual acquisition via tools is effective** — repeatedly acquiring image information within CRUs outperforms single-pass image reading by 4.5%.
- **Progressive training substantially outperforms single-stage training** — the three-stage curriculum yields a 5.3% gain over merged training.
- **Multiple reasoning paths in the CRUX dataset improve reasoning robustness.**

## Highlights & Insights
- **Cognitive science grounded in practice** — Miller's Law is not a decorative citation but genuinely guides CRU design, with each CRU's reasoning steps constrained to 5–7 steps.
- **"Reasoning about reasoning"** — ViRC not only *performs* reasoning but *organizes* reasoning correctly; the meta-reasoning perspective is notably deep.
- **Natural integration of visual tools** — no external tool-use framework is required; visual acquisition is embedded directly within the reasoning chain.
- **The 18.8% average gain is highly significant** — consistent improvements across multiple benchmarks confirm that Reason Chunking is broadly effective.

## Limitations & Future Work
- CRUX dataset construction relies on detailed CRU annotations, incurring high annotation costs.
- CRU granularity is currently fixed (approximately 5–7 steps); adaptive granularity adjustment may yield further benefits.
- Evaluation is limited to mathematical reasoning; generalizability to scientific reasoning, code reasoning, and other domains requiring structured thinking remains to be explored.
- ViRC-7B operates at a relatively small scale; the benefit of Reason Chunking for larger models (70B+) may differ.
- Repeated visual acquisition during inference introduces additional reasoning latency.

## Related Work & Insights
- **vs. Math-LLaVA**: Math-LLaVA provides data for multimodal mathematical reasoning but does not modify the reasoning structure. ViRC innovates at the level of reasoning structure itself.
- **vs. LLaVA-CoT**: LLaVA-CoT employs long-chain CoT without chunking. ViRC decomposes the long chain into structured units via Reason Chunking.
- **vs. R1-OneVision**: R1-OneVision optimizes reasoning with RL but does not incorporate visual tools. ViRC integrates dynamic visual acquisition within each CRU.
- **Insight**: The Reason Chunking paradigm is naturally suited to complex multi-step code generation — decomposing it into CRUs such as "understand requirements → design architecture → implement functions → unit testing."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The Reason Chunking mechanism and CRU concept constitute a genuinely new reasoning paradigm; the cognitive science motivation is convincing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark validation with comprehensive ablations; the 18.8% gain is compelling, though large-model validation is absent.
- Writing Quality: ⭐⭐⭐⭐ The logical thread from cognitive science to method to experiments is coherent and complete.
- Value: ⭐⭐⭐⭐⭐ Provides a new paradigm for multimodal reasoning; both the dataset and code are publicly released.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MathFlow: Enhancing the Perceptual Flow of MLLMs for Visual Mathematical Problems](../../ACL2026/multimodal_vlm/mathflow_enhancing_the_perceptual_flow_of_mllms_for_visual_mathematical_problems.md)
- [\[CVPR 2026\] Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)
- [\[ICCV 2025\] LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](../../ICCV2025/multimodal_vlm/llava-cot_let_vision_language_models_reason_step-by-step.md)
- [\[CVPR 2026\] ViKey: Enhancing Temporal Understanding in Videos via Visual Prompting](vikey_enhancing_temporal_understanding_in_videos_via_visual_prompting.md)
- [\[CVPR 2026\] Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training](wan-weaver_interleaved_multi-modal_generation_via_decoupled_training.md)

</div>

<!-- RELATED:END -->
