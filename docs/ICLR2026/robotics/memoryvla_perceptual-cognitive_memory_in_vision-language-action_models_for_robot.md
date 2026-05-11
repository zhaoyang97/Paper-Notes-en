---
title: >-
  [Paper Note] MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation
description: >-
  [ICLR 2026][Robotics][VLA] Inspired by the dual-memory system in cognitive science, this paper proposes MemoryVLA, a framework that introduces a Perceptual-Cognitive Memory Bank (PCMB) into VLA models. By incorporating m…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "VLA"
  - "memory mechanism"
  - "long-horizon manipulation"
  - "diffusion policy"
  - "cognitive science"
date: 2026-05-08
content_hash: 2224509cf7303cd2
---

# MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation

**Conference**: ICLR 2026
**arXiv**: [2508.19236](https://arxiv.org/abs/2508.19236)
**Code**: [Project Page](https://shihao1895.github.io/MemoryVLA)
**Area**: Robotics / VLA
**Keywords**: VLA, memory mechanism, long-horizon manipulation, diffusion policy, cognitive science

## TL;DR
Inspired by the dual-memory system in cognitive science, this paper proposes MemoryVLA, a framework that introduces a Perceptual-Cognitive Memory Bank (PCMB) into VLA models. By incorporating memory retrieval, gated fusion, and consolidation mechanisms to capture long-horizon temporal dependencies, MemoryVLA comprehensively outperforms CogACT and π₀ across 150+ tasks on SimplerEnv, LIBERO, and real-world benchmarks.

## Background & Motivation

**Background**: VLA models (OpenVLA, π₀, CogACT) have achieved significant progress in robotic manipulation, but mainstream approaches rely solely on current observations and ignore temporal dependencies, leading to poor performance on long-horizon tasks. For example, in the Push Buttons task, visual observations are nearly identical before and after pressing, making it impossible to determine whether an action has been completed.

**Limitations of Prior Work**: (1) Concatenating multiple frames leads to quadratic self-attention complexity and distribution mismatch with single-frame pretraining; (2) RoboFlamingo compresses history with LSTM, losing fine-grained information; (3) TraceVLA draws trajectories, losing semantic details; (4) UniVLA appends past actions as chain-of-thought rather than genuine memory utilization.

**Key Challenge**: Robotic manipulation is inherently non-Markovian (past actions influence future decisions), yet current VLA models are Markovian (conditioned on the current frame only).

**Key Insight**: In cognitive science, humans handle manipulation tasks through working memory (short-term) combined with episodic memory (long-term, comprising verbatim perceptual details and gist-level semantics). Accordingly, PCMB is designed to store memory at two levels: perceptual detail and cognitive semantics.

## Method

### Overall Architecture
Vision-Language Cognition Module (7B VLM) → perceptual tokens ($p$) + cognitive token ($c$) = working memory → PCMB storage / retrieval / fusion / consolidation → memory-conditioned diffusion action expert generates action sequences.

### Key Designs

1. **Vision-Language Cognition Module**:

    - **Function**: Extracts perceptual tokens and a cognitive token from the current RGB input and language instruction.
    - **Mechanism**: Parallel DINOv2 + SigLIP visual encoding → SE bottleneck compression into 256 perceptual tokens $p$; LLaMA-7B processes vision + language → cognitive token $c$ output at the EOS position.
    - **Design Motivation**: Perceptual tokens retain fine-grained visual information, while the cognitive token encodes high-level semantic understanding.

2. **Perceptual-Cognitive Memory Bank (PCMB)**:

    - **Memory Retrieval**: Current tokens with temporal positional encoding serve as queries for cross-attention over PCMB, retrieving decision-relevant history $H^p, H^c$.
    - **Gated Fusion**: $\tilde{x} = g^x \odot H^x + (1-g^x) \odot x$, where $g^x = \sigma(\text{MLP}(\text{concat}[x, H^x]))$, adaptively blending current and historical information.
    - **Memory Consolidation**: When capacity is full, cosine similarity between adjacent entries is computed and the most similar pair is merged, preserving key information while controlling memory size.

3. **Memory-Conditioned Diffusion Action Expert**:

    - **Function**: Conditioned on memory-augmented perceptual and cognitive tokens, a diffusion Transformer generates $N$-step 7-DoF action sequences.
    - **Design Motivation**: The diffusion policy captures multimodal action distributions, and memory conditioning enables temporal awareness.

### Loss & Training
- End-to-end training; the 7B VLM is pretrained on the OXE dataset.
- The diffusion action expert is trained with DDPM.
- Perceptual compression uses an SE-bottleneck module; the cognitive representation uses the EOS token.

## Key Experimental Results

### Main Results (Simulation)

| Benchmark | MemoryVLA | CogACT | π₀ | Gain |
|-----------|-----------|--------|-----|------|
| SimplerEnv-Bridge | **71.9%** | 57.3% | lower | **+14.6** |
| SimplerEnv-Fractal | **72.7%** | 68.1% | lower | +4.6 |
| LIBERO-5 | **96.5%** | 2nd best | 2nd best | surpasses both |
| Mikasa-Robo | **41.2%** | — | 29.4% | **+11.8** |

### Real-World Experiments (12 Tasks)

| Task Type | MemoryVLA | CogACT | π₀ |
|-----------|-----------|--------|-----|
| General Skills (6 tasks) | **85%** | 76% | lower |
| Long-Horizon Dependent (6 tasks) | **83%** | 57% | lower → **+26** |

### Key Findings
- The largest gains occur on long-horizon tasks (+26 vs. CogACT), confirming that memory is critical for temporal dependency.
- The gating value $g$ dynamically varies with task demands—simple tasks rely primarily on current information, while complex tasks leverage history more heavily.
- Memory consolidation via merging similar neighbors is more efficient than fixed windows or FIFO strategies.
- The model demonstrates strong robustness under OOD conditions (varying backgrounds, distractors, lighting, and occlusions).

## Highlights & Insights
- **Cognitive science-driven design**: The dual-system of working memory + episodic memory is mapped to perceptual tokens + cognitive token + PCMB. Rather than naively stacking frames or applying LSTMs, this memory architecture is grounded in cognitive theory.
- **Separation of perception and cognition**: Perceptual tokens (256) preserve spatial detail; the cognitive token (1) compresses high-level semantics. PCMB maintains two separate streams for storage and retrieval, accommodating the need for different levels of historical information across tasks.
- **+26 on long-horizon tasks**: This margin indicates that memory is not an auxiliary enhancement but a necessary condition—VLA models without memory fundamentally fail at tasks requiring temporal understanding.

## Limitations & Future Work
- The 7B VLM incurs substantial inference overhead, limiting real-time applicability.
- PCMB capacity $L$ requires manual tuning; adaptive capacity management warrants exploration.
- Cosine similarity-based consolidation may be insufficiently fine-grained; more sophisticated memory selection strategies could improve performance.
- Only third-person RGB input is used; multi-view and multimodal memory (e.g., tactile sensing) remain unexplored.

## Related Work & Insights
- **vs. π₀**: π₀ lacks a memory mechanism and shows large performance gaps on long-horizon tasks; MemoryVLA's PCMB addresses this deficiency.
- **vs. CogACT**: CogACT also employs a diffusion action head but without temporal modeling; MemoryVLA with memory comprehensively surpasses it.
- **vs. RoboFlamingo**: RoboFlamingo uses coarse-grained LSTM memory, whereas MemoryVLA's dual-stream memory provides finer granularity.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The cognitive science-inspired dual-stream memory architecture represents a first in the VLA domain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three robot platforms, 150+ tasks (simulation + real world), multiple baselines, and OOD testing.
- **Writing Quality**: ⭐⭐⭐⭐ The cognitive science motivation is clearly articulated, and the architecture diagram is intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a critical gap in the VLA field (temporal memory) with convincing experimental results.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models](twinvla_data-efficient_bimanual_manipulation_with_twin_single-arm_vision-languag.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](../../CVPR2026/robotics/lada_robotic_manipulation.md)
- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](../../CVPR2026/robotics/sapave_active_perception_manipulation_vla_roboti.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](../../CVPR2026/robotics/lada_robotic_manipulation.md)
- [\[ICLR 2026\] VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation](vlbiman_vision-language_anchored_one-shot_demonstration_enables_generalizable_bi.md)

</div>

<!-- RELATED:END -->
