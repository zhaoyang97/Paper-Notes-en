---
title: >-
  [Paper Note] Perception Programs: Unlocking Visual Tool Reasoning in Language Models
description: >-
  [CVPR 2026][LLM/NLP][perception programs] Perception Programs (P2) is a training-free, model-agnostic method that converts raw visual tool outputs (depth, optical flow, correspondences, etc.) into compact language-native structured summaries, enabling MLLMs to directly "read" visual modalities rather than infer from dense pixels, achieving an average 19.66% improvement across 6 BLINK tasks.
tags:
  - CVPR 2026
  - LLM/NLP
  - perception programs
  - visual tools
  - language-native representation
  - training-free
  - multimodal reasoning
date: 2026-05-08
content_hash: 3457f242eef0224b
---

# Perception Programs: Unlocking Visual Tool Reasoning in Language Models

**Conference**: CVPR 2026  
**arXiv**: [2604.12896](https://arxiv.org/abs/2604.12896)  
**Code**: [https://github.com/AISmartPerception/perception-programs](https://github.com/AISmartPerception/perception-programs)  
**Area**: LLM / NLP (Other)  
**Keywords**: perception programs, visual tools, language-native representation, training-free, multimodal reasoning

## TL;DR

Perception Programs (P2) is a training-free, model-agnostic method that converts raw visual tool outputs (depth, optical flow, correspondences, etc.) into compact language-native structured summaries, enabling MLLMs to directly "read" visual modalities rather than infer from dense pixels, achieving an average 19.66% improvement across 6 BLINK tasks.

## Background & Motivation

**State of the Field**: MLLMs are increasingly used in conjunction with visual tools (depth estimation, optical flow, visual correspondences, etc.) to enhance visual reasoning.

**Limitations of Prior Work**: Despite visual tools providing accurate perceptual signals, MLLMs often fail to fully leverage them. Raw tool outputs are dense pixel-level representations that are mismatched with LLMs' language-native reasoning capabilities. Experiments show that GPT-5 Mini cannot even recover correct depth ordering from depth maps (Kendall τ rapidly approaches zero).

**Root Cause**: The bottleneck is not more tool calls or larger MLLMs, but the representation format of visual tool outputs. Dense numerical tokens are fundamentally mismatched with the language reasoning substrate.

**Paper Goals**: Convert tool outputs from dense pixel-level representations to language-native structured summaries.

**Starting Point**: Humans extract cues from visual information differently depending on data type (depth focuses on near/far, optical flow focuses on direction, etc.). Converting key information to text relieves the model from processing pixel details.

**Core Idea**: P2 standardizes what tools convey (what), spatial locations (where), and inter-part relationships (how), enabling any MLLM to directly parse and reason.

## Method

### Overall Architecture

Given raw output from a visual tool, P2 partitions the pixel domain into a finite set of primitives (patches/points), extracts a structured item for each primitive $I_p = (p, c_p, r_p, b_p)$ (identifier, normalized coordinates, modality readout, optional label), and generates sparse symbolic relation triples $\mathcal{T}$. The entire summary is serialized as a YAML-formatted text block, directly served as MLLM input.

### Key Designs

1. **Unified Item Schema**:

    - Function: Standardized representation across modalities
    - Mechanism: All modalities share the same item structure $(p, c_p, r_p, b_p)$: primitive identifier, spatial coordinates normalized to [0,1000]², readout extracted from modality data, and optional semantic label. The only difference across modalities is how the readout $r_p$ is constructed and whether relations are included
    - Design Motivation: A unified schema makes the method generalizable across depth, optical flow, correspondences, detection, and other modalities

2. **Modality-Specific Readout Construction**:

    - Function: Extracts key information for each visual modality
    - Mechanism: Depth: each grid cell stores minimum and maximum depth values $r_p = [\min D, \max D]$, generating inter-neighborhood relation triples (e.g., "closer than," "farther than"). Optical flow: encodes motion direction and magnitude. Correspondences: encodes matching point positions and confidence. Detection: encodes object category and bounding box
    - Design Motivation: Key information differs across modalities and requires specialized extraction

3. **Training-Free, Model-Agnostic Deployment**:

    - Function: Plug-and-play for any MLLM
    - Mechanism: P2 requires no parameter updates, architecture modifications, or additional tool calls. The same tool output is converted to P2 within the standard tool-use pipeline and directly consumed by the MLLM. Only minimal text processing overhead is added at inference time
    - Design Motivation: Avoids training costs and model modifications while maintaining maximum flexibility

### Loss & Training

P2 involves no training whatsoever. It is a purely inference-time representation conversion module.

## Key Experimental Results

### Main Results

| Model | Task | Baseline | +Raw Tool | +P2 |
|-------|------|----------|-----------|-----|
| GPT-5 Mini | Multi-view Reasoning | 41.4% | 52.8% | **86.5%** |
| GPT-5 Mini | Relative Depth | 52.4% | 61.2% | **81.5%** |
| GPT-5 Mini | Visual Correspondence | 38.7% | 45.3% | **72.1%** |
| InternVL3.5-4B | 6-task Average | 42.1% | 48.5% | **70.3%** |
| Qwen3VL-4B | 6-task Average | 43.5% | 49.2% | **71.8%** |

### Ablation Study

| Configuration | BLINK 6-task Average | Note |
|---------------|---------------------|------|
| Full P2 | 86.5% | Items + Relations |
| Items only (no relations) | 78.2% | No neighborhood relations |
| Coarse grid (4×4) | 82.1% | Reduced resolution |
| Fine grid (12×12) | 85.8% | Higher resolution |
| Raw tool output | 52.8% | Pixel-level representation |

### Key Findings

- P2 boosts GPT-5 Mini's accuracy on multi-view reasoning from 41.4% to 86.5% (+45 percentage points) — a striking improvement
- Even on 4B-scale small models, absolute improvements of 21–25% are observed
- P2 can enhance existing agent tool-use methods: an additional 18.28% improvement on depth and localization tasks

## Highlights & Insights

- The core insight is profound: the bottleneck in visual reasoning is not tool accuracy but representation format. MLLMs can "read" text but cannot effectively "see" dense numerical values
- P2's design embodies the principle of "let machines do what machines are good at": let visual tools extract perceptual signals, let LLMs handle language reasoning
- Training-free + model-agnostic design gives it extremely high practical value

## Limitations & Future Work

- Grid partitioning granularity requires task-specific tuning
- For tasks requiring precise pixel-level information (e.g., fine segmentation boundaries), P2's spatial discretization may lose information
- Extension along the video temporal dimension has not been evaluated
- Adaptive granularity and dynamic relation generation could be explored

## Related Work & Insights

- **vs VisProg/ViperGPT**: These methods generate programs that call tools but still operate on tool outputs at the pixel level; P2 changes the representation of tool outputs
- **vs Aurora/Mirage**: These methods use training to improve tool usage; P2 achieves greater improvement without any training

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The insight that "representation format is the bottleneck" redefines the problem
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across multiple models and tasks with striking results
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation, analysis, and experiments are all clear
- Value: ⭐⭐⭐⭐⭐ Significant implications for the MLLM tool-use paradigm

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Language Models and Logic Programs for Trustworthy Tax Reasoning](../../AAAI2026/llm_nlp/language_models_and_logic_programs_for_trustworthy_tax_reasoning.md)
- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](../../ACL2026/llm_nlp/don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](../../ACL2026/llm_nlp/foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ICLR 2026\] How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use](../../ICLR2026/llm_nlp/how_far_are_llms_from_professional_poker_players_revisiting_game-theoretic_reaso.md)
- [\[CVPR 2026\] SketchDeco: Training-Free Latent Composition for Precise Sketch Colourisation](sketchdeco_training-free_latent_composition_for_precise_sketch_colourisation.md)

<!-- RELATED:END -->
