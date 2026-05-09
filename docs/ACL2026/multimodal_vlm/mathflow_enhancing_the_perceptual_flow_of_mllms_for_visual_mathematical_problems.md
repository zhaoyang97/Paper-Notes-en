---
title: >-
  [Paper Note] MathFlow: Enhancing the Perceptual Flow of MLLMs for Visual Mathematical Problems
description: >-
  [ACL 2026][Multimodal VLM][Visual mathematical reasoning] This work proposes the FlowVerse benchmark—which decomposes mathematical problem information into four components (DI/EI/RP/OQ) and constructs six variant versions—and the MathFlow modular pipeline, which decouples perception and reasoning into independent stages. A dedicated perception model, MathFlow-P-7B, is trained to extract critical information from mathematical diagrams, substantially improving visual mathematical problem-solving performance across diverse reasoning models.
tags:
  - ACL 2026
  - Multimodal VLM
  - Visual mathematical reasoning
  - multimodal large language models
  - perception-reasoning decoupling
  - mathematical diagram understanding
  - benchmarking
date: 2026-05-08
content_hash: e59a6408f355ea0d
---

# MathFlow: Enhancing the Perceptual Flow of MLLMs for Visual Mathematical Problems

**Conference**: ACL 2026
**arXiv**: [2503.16549](https://arxiv.org/abs/2503.16549)
**Code**: [GitHub](https://github.com/MathFlow-zju/MathFlow)
**Area**: Multimodal VLM
**Keywords**: Visual mathematical reasoning, multimodal large language models, perception-reasoning decoupling, mathematical diagram understanding, benchmarking

## TL;DR

This work proposes the FlowVerse benchmark—which decomposes mathematical problem information into four components (DI/EI/RP/OQ) and constructs six variant versions—and the MathFlow modular pipeline, which decouples perception and reasoning into independent stages. A dedicated perception model, MathFlow-P-7B, is trained to extract critical information from mathematical diagrams, substantially improving visual mathematical problem-solving performance across diverse reasoning models.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) perform well on tasks such as image captioning and visual question answering, yet remain notably deficient in visual mathematical problem solving, particularly in accurately perceiving and interpreting critical information such as geometric elements and numerical relationships in mathematical diagrams.

**Limitations of Prior Work**: Existing approaches predominantly focus on improving the reasoning process (e.g., CoT strategies, tool-augmented reasoning, reinforcement learning) while overlooking a fundamental prerequisite: the quality of information extraction during the perception stage directly constrains the upper bound of reasoning performance. Even strong models such as GPT-4V exhibit significant deficiencies when extracting critical information from diagrams. Existing benchmarks (e.g., MathVista, MathVerse) also fail to disentangle the individual contributions of perception and reasoning capabilities.

**Key Challenge**: Perception and reasoning capabilities are coupled during both training and evaluation, precluding independent optimization. Errors in the perception stage propagate in a cascading manner to the reasoning stage, yet existing frameworks cannot diagnose which stage constitutes the bottleneck.

**Goal**: (1) Construct a fine-grained benchmark to evaluate perception and reasoning capabilities independently; (2) design a modular pipeline to decouple perception from reasoning; (3) train a dedicated perception model to improve the quality of critical information extraction.

**Key Insight**: Inspired by the human problem-solving process—in which one first extracts critical information from a diagram (perception) and then performs mathematical reasoning—the authors decompose problem information into four independent components, and construct six variants by combining different subsets to precisely localize model weaknesses.

**Core Idea**: The information flow in visual mathematical problems is decomposed into Descriptive Information (DI), Essential Information (EI), Reasoning Properties (RP), and the Original Question (OQ). Controlled experiments demonstrate that perception is the bottleneck, which is then addressed via a pipeline consisting of a dedicated perception model paired with a general-purpose reasoning model.

## Method

### Overall Architecture

MathFlow is a two-stage modular pipeline: (1) **Perception stage**: MathFlow-P-7B extracts EI (critical information such as angle values and edge-length relationships) and RP (reasoning properties such as inferred geometric relations) from mathematical diagrams and converts them into textual representations; (2) **Reasoning stage**: the extracted textual information is concatenated with the original question and fed into any reasoning model (e.g., GPT-5, Claude) to generate a solution.

### Key Designs

1. **Four-component information decomposition in the FlowVerse benchmark**:

    - **Function**: Provides fine-grained diagnosis of perception and reasoning capabilities.
    - **Mechanism**: Each mathematical problem's information is decomposed into DI (descriptive information, e.g., "triangle ABC"), EI (essential information, e.g., "∠A = 45°"), RP (reasoning properties—intermediate relations that must be inferred from the diagram), and OQ (the question). Six variants are formed by combining different subsets: Text Centric (full text), Text Limited (DI removed), Text Plus (no image), Vision Dense (RP removed), Vision Centric (EI visualized), and Vision Primary (both EI and RP visualized). Accuracy differences across variants precisely localize capability bottlenecks.
    - **Design Motivation**: Existing benchmarks conflate perception and reasoning in evaluation, making it impossible to attribute errors. RP serves as a control variable to independently assess whether a model can derive intermediate properties on its own.

2. **MathFlow-P-7B perception model training**:

    - **Function**: High-quality extraction of EI and RP from mathematical diagrams.
    - **Mechanism**: Built on Qwen2-VL-7B via a two-stage training procedure. The multi-task pre-training stage includes an EI description task (650K image–text pairs) and a visual reasoning task (decomposing solution steps into sequential predictions, 130K samples) at a 3:1 data ratio, with the LLM frozen and only the visual encoder and perceiver trained. The supervised fine-tuning stage freezes the visual encoder and trains the perceiver and LLM using the carefully annotated MathFlow-SFT dataset.
    - **Design Motivation**: EI description training teaches the model to "see" basic elements in diagrams; visual reasoning training teaches the model to "infer" implicit geometric relations. The two stages optimize different modules separately, avoiding mutual interference.

3. **FlowVerse-CoT-E evaluation strategy**:

    - **Function**: Provides robust evaluation of CoT reasoning processes.
    - **Mechanism**: Experts pre-construct authoritative solution steps for each problem and decompose them; step-by-step hints are progressively added to the prompt to assess model performance at varying reasoning depths, with results aggregated as $\text{Score}_{\text{final}} = 0.8 \cdot \frac{1}{N}\sum_{i=1}^N \text{Score}_i + 0.2 \cdot \text{Score}_0$.
    - **Design Motivation**: MathVerse-CoT-E relies on GPT to decompose model responses into steps before evaluation, introducing additional noise; FlowVerse-CoT-E uses manually annotated steps, yielding more stable evaluation.

### Loss & Training

The multi-task pre-training stage uses a learning rate of 1e-5; the supervised fine-tuning stage uses 5e-6, with DeepSpeed Zero2. Different modules are frozen at each stage to ensure that perception-related components and language reasoning components are optimized independently.

## Key Experimental Results

### Main Results

On the FlowVerse benchmark (CoT-E metric):

| Model | All | Text Centric | Vision Dense | Vision Primary |
|-------|-----|-------------|-------------|---------------|
| Qwen2.5-VL-7B | 53.8 | 60.1 | 45.0 | 48.1 |
| MathFlow*_Qwen2.5-VL-7B | **57.0** | **62.0** | **49.0** | **52.0** |
| GPT-5 | 65.8 | 74.3 | 53.8 | 60.3 |
| MathFlow*_GPT-5 | **66.5** | **74.6** | **58.2** | **69.4** |

Structured description quality evaluation (F1):

| Model | EI (F1) | RP (F1) |
|-------|---------|---------|
| GPT-4o | 87.7% | 70.1% |
| Claude-sonnet | 89.1% | 72.8% |
| MathFlow-P-7B | **97.2%** | **85.6%** |

### Ablation Study

| Configuration | FlowVerse† Accuracy | Note |
|--------------|---------------------|------|
| Reasoning model (direct) | Baseline | No perception enhancement |
| + MathFlow-P-7B (EI only) | +2–4% | EI extraction is effective |
| + MathFlow-P-7B (EI + RP) | +4–8% | RP inference yields additional gain |
| Different reasoning models | Consistent | Pipeline generalizes broadly |

### Key Findings

- Removing images actually improves model accuracy (Qwen2-VL-72B: +4.3%), indicating that images serve more as a source of noise than of information—perception is the true bottleneck.
- The Vision Primary variant (both EI and RP in the image) underperforms the Text Centric variant (full text) by more than 10 percentage points on average, demonstrating that models are "better at reading text than interpreting diagrams."
- MathFlow-P-7B achieves an EI F1 of 97.2%, substantially outperforming GPT-4o (87.7%), validating the value of a dedicated perception model.
- The pipeline yields consistent gains for both open-source and closed-source reasoning models, confirming the generalizability of perception enhancement.

## Highlights & Insights

- The **information decomposition + controlled-variable evaluation design** is particularly elegant—by combining DI/EI/RP/OQ in different ways, the contribution of each type of information and each modality to the final outcome can be precisely quantified. This methodology is transferable to benchmark design for other multimodal tasks.
- The conclusion that **"perception is the bottleneck" is counterintuitive yet empirically well-supported**—consistently observing accuracy improvements upon removing images across diverse models provides a strong experimental foundation for the perception-before-reasoning separation strategy.
- The fact that a **7B perception model surpasses GPT-4o** demonstrates that specialized training can substantially outperform general-purpose large models on specific subtasks, a principle transferable to scientific diagram understanding, medical image analysis, and related domains.

## Limitations & Future Work

- Manual annotation of RP is costly, limiting data scale and domain coverage.
- The evaluation set is primarily composed of Chinese and English exam problems, with limited coverage of university-level or applied mathematics.
- The two-stage pipeline incurs additional inference overhead (requiring two model calls), which must be considered in latency-sensitive scenarios.
- The perception model is based on Qwen2-VL-7B; whether larger base models yield further gains remains an open question.

## Related Work & Insights

- **vs. MathVerse**: MathVerse primarily eliminates textual redundancy to ensure models must rely on images, but lacks fine-grained decoupling of perception and reasoning; FlowVerse achieves more granular diagnosis through its four-component, six-variant design.
- **vs. end-to-end visual math models (e.g., VLM-R1)**: End-to-end approaches are straightforward but couple perception and reasoning, making independent optimization difficult; MathFlow's modular design allows perception and reasoning to be iterated and upgraded separately.
- **vs. tool-augmented reasoning**: Tool-augmented approaches (e.g., AlphaGeometry) assume that inputs have already been correctly parsed; this work identifies parsing itself as the bottleneck.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The information decomposition and modular pipeline are conceptually clear, though the overarching idea of separating perception from reasoning is not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The benchmark design is rigorous; evaluation covers a large number of models and multiple datasets with thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ The structure is clear, though the abundance of tables imposes a non-trivial reading burden.
- **Value**: ⭐⭐⭐⭐⭐ Both the FlowVerse benchmark and the MathFlow pipeline make substantive contributions to the field of visual mathematical reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ViRC: Enhancing Visual Interleaved Mathematical CoT with Reason Chunking](../../CVPR2026/multimodal_vlm/virc_enhancing_visual_interleaved_mathematical_cot_with_reason_chunking.md)
- [\[ACL 2026\] Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning](enhancing_multimodal_large_language_models_for_ancient_chinese_character_evoluti.md)
- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)
- [\[ACL 2026\] Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation](making_mllms_blind_adversarial_smuggling_attacks_in_mllm_content_moderation.md)
- [\[CVPR 2026\] CodePercept: Code-Grounded Visual STEM Perception for MLLMs](../../CVPR2026/multimodal_vlm/codepercept_code-grounded_visual_stem_perception_for_mllms.md)

</div>

<!-- RELATED:END -->
