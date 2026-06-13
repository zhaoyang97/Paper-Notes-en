---
title: >-
  [Paper Note] CoRVid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning
description: >-
  [ICCV 2025][LLM Reasoning][Multimodal Large Language Models] This paper proposes Corvid, which comprehensively enhances the chain-of-thought reasoning capability of MLLMs through a hybrid visual encoder…
tags:
  - "ICCV 2025"
  - "LLM Reasoning"
  - "Multimodal Large Language Models"
  - "Chain-of-Thought Reasoning"
  - "Visual Encoder"
  - "Test-Time Scaling"
  - "CoT"
date: 2026-05-08
content_hash: c84b41263417f541
---

# CoRVid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning

**Conference**: ICCV 2025
**arXiv**: [2507.07424](https://arxiv.org/abs/2507.07424)  
**Code**: [Project Page](https://mm-vl.github.io/corvid)  
**Area**: LLM Reasoning
**Keywords**: Multimodal Large Language Models, Chain-of-Thought Reasoning, Visual Encoder, Test-Time Scaling, CoT

## TL;DR

This paper proposes Corvid, which comprehensively enhances the chain-of-thought reasoning capability of MLLMs through a hybrid visual encoder, a GateMixer connector, a high-quality CoT dataset, and a test-time self-verification strategy, surpassing open-source models of comparable parameter scale on mathematical reasoning and scientific problem solving.

## Background & Motivation

Multimodal large language models (MLLMs) have demonstrated impressive capabilities in perception and understanding, yet they remain underperforming on tasks requiring **complex structured reasoning** (e.g., mathematical reasoning, scientific problem solving). Leading models such as Ovis2 and Qwen2.5-VL exhibit suboptimal performance on tasks demanding deep inference and extrapolation.

**Three core challenges**:

**Scarcity of high-quality multimodal CoT data**: Manually created CoT annotations tend to be concise, while AI-generated CoT is noisy and repetitive. Prior work shows that MLLMs trained on direct-answer data are incapable of step-by-step reasoning.

**Insufficient visual representation and poor cross-modal alignment**: Complex reasoning requires accurate capture of visual information and efficient mapping into the language embedding space. Existing connectors (simple MLP projections or cross-attention layers) are inadequate for constructing visual representations sufficient for reasoning.

**Over-reasoning and under-reasoning at inference time**: Current o1-like MLLMs uniformly apply deep reasoning to all inputs; however, applying CoT to simple tasks can degrade accuracy due to context loss and hallucinations.

The paper addresses these challenges through a **multi-pronged approach**: architecturally strengthening visual representations via a hybrid encoder and a gated mixture connector, curating 287K high-quality CoT instruction-following data, and adopting a self-verification strategy that adaptively selects between direct answering and CoT reasoning.

## Method

### Overall Architecture

Corvid consists of three core modules: a hybrid visual encoder (SigLIP + ConvNeXt-XXL), a GateMixer connector, and a Llama3-8B LLM. Training proceeds in three stages: multi-granularity alignment pre-training → CoT-enhanced fine-tuning → pure CoT instruction tuning.

### Key Designs

1. **Hybrid Visual Encoder**:

    - SigLIP ViT-SO400M: 384×384 input, producing $729 \times 1152$ semantically rich features.
    - OpenCLIP ConvNeXt-XXL: same resolution input, producing $729 \times 5760$ multi-scale spatial detail features.
    - The two encoders are complementary: ViT excels at semantics while CNN preserves spatial details.

2. **GateMixer Connector**:

    - Both feature sets are projected to a unified space $\{\mathbf{h}_v, \mathbf{h}_c\} \in \mathbb{R}^{729 \times d}$ via separate linear layers.
    - Inspired by the LSTM input gate mechanism, gated attention is applied for element-wise mixing:
        - $\boldsymbol{\alpha} = \sigma(\mathbf{W}_g[\mathbf{h}_v; \mathbf{h}_c] + \mathbf{b}_g)$
        - $\mathbf{h} = (1 - \boldsymbol{\alpha}) \odot \mathbf{h}_v + \boldsymbol{\alpha} \odot \mathbf{h}_c$
    - Learnable prefix tokens $\mathbf{h}_p \in \mathbb{R}^{24 \times d}$ are inserted to enhance context capture.
    - A final linear projection maps the features to the language embedding space.

3. **MCoT-Instruct-287K Dataset**:

    - Curated and standardized from multiple public reasoning datasets, covering mathematical, scientific, logical, and other reasoning types.
    - Manually created CoT annotations are accurate but concise → GPT-assisted expansion and standardization.
    - AI-generated CoT annotations are detailed but error-prone → GPT-assisted error correction and deduplication.
    - Training data includes MGA-1M (alignment pre-training), Corvid-1M (SFT), and o1-320K (pure CoT).

4. **Test-Time Self-Verification Strategy**:

    - Both a direct response $\mathcal{R}_{\text{direct}}$ and a CoT response $\mathcal{R}_{\text{CoT}}$ are generated simultaneously.
    - If the two answers agree, the answer is adopted directly.
    - If they disagree, a weighted score is computed: $\mathcal{SC} = (1-\alpha)\mathcal{S} + \alpha\mathcal{C}$
        - $\mathcal{S}$: cosine similarity of image-text representations after the LLM (cross-modal alignment quality).
        - $\mathcal{C}$: normalized perplexity (model confidence).
    - The answer with the higher $\mathcal{SC}$ is selected.

### Loss & Training

- **Stage 1** (Multi-Granularity Alignment Pre-Training): Encoders and LLM are frozen; only GateMixer is trained on MGA-1M. A contrastive regularization loss $\mathcal{L}_{\text{CReg}}$ is additionally applied to promote image-text semantic association.
- **Stage 2** (CoT-Enhanced SFT): GateMixer and LLM are jointly trained on Corvid-1M, producing Corvid-base.
- **Stage 3** (Pure CoT Instruction Tuning): Corvid-base is fine-tuned on o1-320K, producing Corvid-o1.

## Key Experimental Results

### Main Results

Comparison with MLLMs of comparable parameter scale (mathematical reasoning benchmarks):

| Model | MathVista | MathVerse | WeMath | MathVision | DynaMath |
|-------|-----------|-----------|--------|------------|---------|
| Qwen2.5-VL-7B | 65.8 | 31.5 | 36.4 | 25.8 | 14.8 |
| Ovis2-8B | 71.8 | 42.3 | 27.2 | 25.9 | 20.4 |
| InternVL2.5-8B | 64.5 | 22.8 | 23.5 | 17.0 | 9.4 |
| **Corvid-o1-8B** | **72.0** | **40.1** | **59.8** | **30.1** | **33.2** |

Comparison with o1-like MLLMs:

| Model | MMStar | MMB | MathV | AI2D | Avg. |
|-------|--------|-----|-------|------|------|
| LLaVA-o1 | 58.1 | 75.6 | 56.1 | 78.8 | 63.1 |
| LlamaV-o1 | 59.5 | 79.9 | 54.4 | 81.2 | 67.3 |
| Mulberry-o1-7B | 61.3 | 75.3 | 57.5 | 79.0 | 62.8 |
| **Corvid-o1-8B** | **65.2** | **82.9** | **72.0** | **85.0** | **67.5** |

### Ablation Study

Effectiveness of the self-verification strategy (Corvid-o1):

| Inference Mode | Avg. | MMStar | MMMU | MathVista | WeMath |
|----------------|------|--------|------|-----------|--------|
| Direct Inference | 48.9 | 54.5 | 47.6 | 49.2 | 41.8 |
| CoT Inference | 56.2 | 61.1 | 55.7 | 67.2 | 57.1 |
| **Self-Verification** | **59.9** | **65.2** | **59.7** | **72.0** | **59.8** |

Connector ablation (average accuracy): FC_GELU_FC: 54.0, 2×FC_GELU_FC: 53.7, GateMixer: **55.6**.

### Key Findings

- High-quality CoT data is critical: direct-answer data (45.7%) vs. raw rationales (51.2%) vs. refined CoT (55.6%).
- Data diversity is equally important: using only CoT reasoning data yields an average of 48.0, which improves to 55.6 upon incorporating direct-answer and OCR data.
- The self-verification strategy outperforms pure CoT inference by 3.7 points (Corvid-o1) and direct inference by 11.0 points.
- Hybrid encoder (55.6) > single SigLIP (54.7) > single ConvNeXt (52.3).

## Highlights & Insights

- The self-verification strategy addresses the question of "when to engage in deep reasoning" without requiring an external verifier.
- The gating mechanism in GateMixer maintains the visual token length unchanged while achieving selective representation fusion.
- The standardization and refinement pipeline for CoT data is more important than simply scaling data volume.
- The score of 59.8 on WeMath substantially outperforms other models (Ovis2-8B achieves only 27.2), demonstrating the significant potential of CoT training for complex mathematical reasoning.

## Limitations & Future Work

- Inference requires generating two responses (direct + CoT), effectively doubling inference cost.
- The model lacks world knowledge and commonsense reasoning capabilities (e.g., typical lane widths).
- The 384×384 input resolution limits fine-grained detail capture for high-resolution images.
- Reinforcement learning (e.g., GRPO) for reasoning enhancement remains unexplored.
- The maximum generation length of only 1,024 tokens constrains very long reasoning chains.

## Related Work & Insights

- LLaVA-o1 and LlamaV-o1 employ beam search to select optimal reasoning paths, whereas the proposed self-verification strategy offers a complementary trade-off.
- Mulberry-o1's Monte Carlo tree search approach is heavier-weight but holds greater potential.
- URSA-8B also focuses on mathematical reasoning and could potentially be combined with the proposed method.
- The RL-driven reasoning training direction of DeepSeek-R1 is worth exploring in the MLLM context.

## Rating

- Novelty: ⭐⭐⭐⭐ — The self-verification strategy and GateMixer design are original.
- Technical Depth: ⭐⭐⭐⭐ — Multi-module collaboration with a complete training pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 13 benchmarks.
- Value: ⭐⭐⭐⭐ — Open-source with state-of-the-art performance.
- Overall Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](../../ICLR2026/llm_reasoning/vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)
- [\[NeurIPS 2025\] Large Language Models Can Learn and Generalize Steganographic Chain-of-Thought under Process Supervision](../../NeurIPS2025/llm_reasoning/large_language_models_can_learn_and_generalize_steganographic_chain-of-thought_u.md)
- [\[ICLR 2026\] AIMCoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](../../ICLR2026/llm_reasoning/aimcot_active_information-driven_multimodal_chain-of-thought_for_vision-language.md)
- [\[NeurIPS 2025\] Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](../../NeurIPS2025/llm_reasoning/visual_thoughts_a_unified_perspective_of_understanding_multi.md)
- [\[ACL 2026\] AIM-CoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](../../ACL2026/llm_reasoning/aim-cot_active_information-driven_multimodal_chain-of-thought_for_vision-languag.md)

</div>

<!-- RELATED:END -->
