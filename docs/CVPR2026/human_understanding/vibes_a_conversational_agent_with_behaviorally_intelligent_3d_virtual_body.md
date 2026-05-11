---
title: >-
  [Paper Note] ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body
description: >-
  [CVPR 2026][Human Understanding][Conversational Virtual Agent] This paper proposes ViBES, a 3D conversational agent that unifies language, speech…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "Conversational Virtual Agent"
  - "Mixture of Modal Experts"
  - "Co-speech Gesture Generation"
  - "Speech-Motion Synchronization"
  - "3D Body Animation"
date: 2026-05-08
content_hash: 7f384482111938b3
---

# ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body

**Conference**: CVPR 2026
**arXiv**: [2512.14234](https://arxiv.org/abs/2512.14234)
**Code**: [ai.stanford.edu/~juze/ViBES/](https://ai.stanford.edu/~juze/ViBES/)
**Area**: Human Understanding / Multimodal Interaction
**Keywords**: Conversational Virtual Agent, Mixture of Modal Experts, Co-speech Gesture Generation, Speech-Motion Synchronization, 3D Body Animation

## TL;DR

This paper proposes ViBES, a 3D conversational agent that unifies language, speech, and body motion via a Mixture of Modal Experts (MoME) architecture and cross-modal attention mechanisms. ViBES generates temporally aligned facial expressions and whole-body motions while preserving the conversational capabilities of a pretrained speech LLM, surpassing the paradigm that treats behavior as simple "modality translation."

## Background & Motivation

- **Background**: Existing conversational AI systems have achieved fluent text and speech interaction, yet they lack a body. Human communication is inherently multimodal — verbal content, prosody, and body language jointly convey intent.
- **Limitations of Prior Work**: Approaches that model behavior as "modality translation" (e.g., speech→gesture, text→motion) are fundamentally flawed: they do not require intelligent decisions about *when to move*, *what to do*, or *how to adapt across dialogue turns*, leading to fragile temporal alignment and weak social grounding. Naively chaining a speech LLM with a motion generator (two-stage) is also problematic in practice: there is no unified temporal and selection strategy, no shared dialogue state, and no guarantee of cross-turn consistency. The most closely related works, LoM and SOLAMI, focus on modality alignment rather than preserving conversational intelligence.
- **Goal**: To build a genuine *embodied conversational agent* — one that not only generates co-speech gestures during responses but also follows explicit motion instructions (e.g., "please step back and wave"). This requires elevating non-verbal behavior from "conditional generation" to "intelligent agent behavior."

## Method

### Overall Architecture

ViBES is a Speech-Language-Behavior (SLB) model built on a Mixture of Modal Experts (MoME) architecture. Three Transformer experts — a Speech-Text expert (frozen from GLM-4-Voice), a facial expression expert, and a body motion expert — are coupled via SLB cross-modal attention. All modalities are tokenized into an interleaved token stream and generated autoregressively on a unified timeline.

### Key Designs

1. **Mixture of Modal Experts (MoME) Architecture**:

    - **Function**: Separates parameters by modality while maintaining cross-modal information sharing.
    - **Mechanism**: Each of the three experts has independent FFN and LayerNorm modules, with hard routing (deterministic assignment by modality label, no learned router). Key attention topology: the Speech-Text (TS) expert performs internal self-attention; the facial and body experts' Queries attend only to the TS expert's Keys/Values (unidirectional read); no cross-attention exists between the facial and body experts. Ablations confirm that facial–body attention yields no improvement — once conditioned on TS, the two are nearly independent.
    - **Design Motivation**: Avoids the destruction of the pretrained LLM's conversational capability through fully dense fusion. The TS expert directly inherits GLM-4-Voice weights (frozen), while the facial/body experts serve as lightweight sidecar modules that read TS states via cross-attention, eliminating the need for large-scale audio-text-motion joint pretraining.

2. **Multimodal Fractional RoPE**:

    - **Function**: Precisely encodes cross-modal temporal alignment on a unified rotary timeline.
    - **Mechanism**: The TS stream serves as the anchor (integer indices); motion tokens obtain fractional indices via linear interpolation: $s_t = s_{a_i} + \alpha_t$, where $\alpha_t$ is the normalized position of the actual timestamp between adjacent TS anchors. This resolves inconsistent frame rates across modalities (speech at 12.5 fps, motion at 25 fps, body at 6.25 fps).
    - **Design Motivation**: Standard RoPE assumes equally spaced integer positions and cannot express precise temporal correspondence between modalities of different frame rates. Fractional indices allow attention scores to naturally reflect true temporal distances across modalities.

3. **1,000-Hour Synchronized Dataset**:

    - **Function**: Provides large-scale, temporally aligned audio-text-motion triplets.
    - **Mechanism**: Monocular 3D human motion (SMPL-X body+hands + FLAME face) is automatically recovered from YouTube conversational videos (interviews, podcasts, talks) and temporally aligned with speech and text transcriptions. This supplements existing motion datasets to form a 1,000-hour training corpus.
    - **Design Motivation**: Existing datasets contain only pairwise alignment (text→motion or audio→motion); large-scale three-modality synchronized data is absent, constituting the core bottleneck for training unified conversational agents.

### Loss & Training

Standard next-token prediction loss is used for autoregressive training. Faces are tokenized using the LoM tokenizer (25 fps); bodies use a compositional tokenizer (upper body / lower body / hands, 6.25 fps). All streams are aligned to a 25 fps master clock. Training proceeds in stages: large-scale data pretraining followed by fine-tuning on conversational interaction data.

## Key Experimental Results

### Main Results

| Task | Method | Key Metric | Notes |
|------|--------|-----------|-------|
| Multi-turn dialogue + motion | ViBES | Best on dialogue-motion alignment / behavior quality / social appropriateness | Comprehensive benchmark |
| Co-speech gesture | ViBES | SOTA | On BEAT2 benchmark |
| Text-to-motion | ViBES | SOTA | On HumanML3D benchmark |

### Ablation Study

| Configuration | Effect | Notes |
|---------------|--------|-------|
| Enable facial↔body attention | No improvement | Facial/body are independent once conditioned on TS |
| Remove Fractional RoPE | Temporal alignment degrades | Confirms importance of precise temporal encoding |
| Two-stage (LLM + motion generator) | Poor consistency | No shared dialogue state |

### Key Findings

- Hard modal routing combined with unidirectional cross-attention (facial/body→TS) is the most effective architectural choice, outperforming bidirectional or fully connected attention.
- Monocular 3D motion recovered from YouTube is noisy, yet meaningful conversational behavior patterns can still be learned at scale.
- Fractional RoPE is critical for maintaining multimodal temporal synchronization.

## Highlights & Insights

- **Elevating non-verbal behavior from "modality translation" to "agent behavior"**: ViBES not only generates speech-synchronized gestures but also comprehends and executes natural language motion instructions — a qualitative shift from generation to intelligence.
- **Frozen pretrained LLM + lightweight sidecar experts** as an architectural paradigm: avoids the astronomical data and compute requirements of training a three-modality model from scratch, and is generalizable to the introduction of other new modalities.
- **Fractional RoPE** elegantly resolves the temporal alignment problem across multi-rate modalities, offering a more principled solution than frame resampling.

## Limitations & Future Work

- 3D motion is recovered from monocular YouTube videos, limiting quality due to occlusion and depth ambiguity.
- No direct interaction modeling between face and body may miss subtle social signals such as gaze–gesture coordination.
- Cache file truncation limits access to complete experimental data.
- Only single-agent scenarios are supported; multi-person interaction is not addressed.

## Related Work & Insights

- **vs. LoM/SOLAMI**: Perform only modality alignment without an LLM reasoning backbone; do not support motion instructions.
- **vs. Co-speech methods**: Perform audio→motion translation only, without dialogue understanding.
- **vs. Two-stage systems**: Lack a unified strategy and shared dialogue state.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First 3D agent to unify conversational intelligence with embodied behavior; MoME + Fractional RoPE design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-task evaluation with ablations (cache truncation leaves some data incomplete).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem formulation is clear; architecture description is thorough.
- **Value**: ⭐⭐⭐⭐⭐ Opens a new direction for embodied conversational AI; dataset and framework offer significant contributions to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition](fusionagent_a_multimodal_agent_with_dynamic_model_selection_for_human_recognitio.md)
- [\[CVPR 2026\] RefTon: Reference Person Shot Assist Virtual Try-on](refton_reference_person_shot_assist_virtual_try-on.md)
- [\[CVPR 2026\] Mobile-VTON: High-Fidelity On-Device Virtual Try-On](mobile_vton_ondevice_virtual_tryon.md)
- [\[ICCV 2025\] PHD: Personalized 3D Human Body Fitting with Point Diffusion](../../ICCV2025/human_understanding/phd_personalized_3d_human_body_fitting_with_point_diffusion.md)
- [\[CVPR 2026\] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)

</div>

<!-- RELATED:END -->
