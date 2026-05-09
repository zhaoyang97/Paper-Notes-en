---
title: >-
  [Paper Note] Chain of Event-Centric Causal Thought for Physically Plausible Video Generation
description: >-
  [CVPR 2026][Video Generation][Physically plausible video generation] This work models physically plausible video generation (PPVG) as a sequence of causally connected events. It decomposes complex physical phenomena into ordered events via physics-formula-grounded event chain reasoning, then synthesizes semantic–visual dual conditions through transition-aware cross-modal prompting to guide a video diffusion model in generating videos that follow causal physical evolution.
tags:
  - CVPR 2026
  - Video Generation
  - Physically plausible video generation
  - causal reasoning
  - event chain
  - cross-modal prompting
  - chain-of-thought
date: 2026-05-08
content_hash: a1035cef8c588e8b
---

# Chain of Event-Centric Causal Thought for Physically Plausible Video Generation

**Conference**: CVPR 2026
**arXiv**: [2603.09094](https://arxiv.org/abs/2603.09094)
**Code**: Coming soon
**Area**: Video Generation
**Keywords**: Physically plausible video generation, causal reasoning, event chain, cross-modal prompting, chain-of-thought

## TL;DR

This work models physically plausible video generation (PPVG) as a sequence of causally connected events. It decomposes complex physical phenomena into ordered events via physics-formula-grounded event chain reasoning, then synthesizes semantic–visual dual conditions through transition-aware cross-modal prompting to guide a video diffusion model in generating videos that follow causal physical evolution.

## Background & Motivation

Physically plausible video generation (PPVG) aims to produce videos that conform to real-world physical laws, with broad applications in filmmaking, autonomous driving, and embodied intelligence. The key challenges are:

1. **Video diffusion models lack commonsense physical reasoning**: Models such as Kling and Sora can generate realistic scenes, but brief text prompts cannot convey detailed physical laws, and the models cannot implicitly infer physical commonsense.
2. **Limitations of existing PPVG methods**: Approaches such as PhyT2V and DiffPhy leverage LLMs to embed physical concepts into prompts, but typically **reduce physical phenomena to static descriptions of a single moment**, lacking modeling of causal evolutionary processes.
3. **Two core challenges**:
   - **Causal ambiguity**: Real-world physical phenomena unfold as causally ordered event units; simple semantic labels cannot capture their dynamic nature, requiring structured causal decomposition.
   - **Insufficient physical consistency constraints**: Language alone cannot convey causal continuity between events; visual cues (e.g., reference videos) can provide observational evidence of transitions, but visual priors tightly aligned with specific physical phenomena are difficult to obtain.

The authors' key perspective shift: **treating physical phenomena as causally connected, dynamically evolving event sequences** rather than static descriptions of a single scene.

## Method

### Overall Architecture

The framework $\Gamma: w \rightarrow \mathbf{V}$ consists of two cooperative modules:
1. **PECR (Physics-driven Event Chain Reasoning)**: Interprets complex physical phenomena described by the user as an ordered set of physical events.
2. **TCP (Transition-aware Cross-modal Prompting)**: Bridges the event chain inferred by PECR to the video generation process, dynamically synthesizing semantic–visual dual conditions that evolve with the physical process.

The generation process follows $\mathbf{Z}_{\tau_z-1} = \epsilon_\theta(\mathbf{Z}_{\tau_z}; \mathbf{W})$, where $\mathbf{Z}_{\tau_z}$ denotes the visual prior and $\mathbf{W}$ the semantic embedding.

### Key Designs

1. **Physics Formula Grounding**: Physical laws $\mathcal{L}$ (e.g., Newtonian mechanics, thermodynamics) are first determined via question answering, then formula names $\mathcal{N}_\mathcal{L}$ are inferred and physical formulas $\mathcal{F}^*$ are retrieved from a knowledge base via $\mathcal{F}^* = \text{TopK}_{f \in \mathcal{F}_\mathcal{L}} P(f | \mathcal{N}_\mathcal{L}, \mathcal{L})$. When direct matching fails, formula names are regenerated using formulas in the knowledge base. This elevates physical commonsense from vague semantic inference to quantitative analysis grounded in standard formulas.

2. **Physical Phenomena Decomposition**: Physical phenomena are decomposed into an ordered event sequence $\{\mathcal{E}_t\}_{t=1}^T = \{\{\mathcal{C}_t\}, \{\mathcal{G}_t\}\}$, where $\mathcal{C}_t$ denotes physical conditions and $\mathcal{G}_t$ denotes dynamic scene graphs. Event boundaries are determined by significant changes in physical parameters: $\mathcal{C}_t = \{(\mathbf{P}_t, \mathcal{F}^*(\mathbf{P}_t)) | \|\mathbf{P}_t - \mathbf{P}_{t-1}\| > \tau_p\}$. Scene graphs are updated via $\mathcal{G}_t = \Phi(\mathcal{G}_{t-1}, \mathcal{C}_t)$, covering changes in node appearance/semantic labels and edge interaction relations. Parameters of adjacent events are validated through physical continuity checks.

3. **Transition-aware Cross-modal Prompting (TCP)**: Comprises two sub-modules:
   - **Progressive Narrative Revision (PNR)**: Performs minimal progressive revision of event descriptions conditioned on preceding context, $w_t = \text{LLM}(w_{t-1} + \Delta(w_{t-1}, \mathcal{C}_t, \mathcal{G}_t))$. Multiple event descriptions are merged into positive semantic prompts via semantic condensation and causal connectives, while negative descriptions are simultaneously constructed. Physical conditions constrain physically permissible transitions (e.g., rising temperature permits "melting" but excludes "freezing"), and scene graphs maintain object identity consistency.
   - **Interactive Keyframe Synthesis (IKS)**: Synthesizes keyframes for each event via interactive image editing, $v_t = \text{Edit}(v_{t-1}; \mathcal{O}_t)$, where edit operators $\mathcal{O}_t$ are determined by consecutive physical condition changes (constraining drag magnitude and regions of visual change). Keyframes are encoded with a VAE, and intermediate frames are generated via linear interpolation: $\mathbf{z}_{0,t} = \text{INTERP}(\psi_{\text{img}}(v_{t-1}), \psi_{\text{img}}(v_t); d_t)$, with noise added to serve as the denoising prior.

### Loss & Training

The framework is a **training-free** inference-time method:
- Base model: CogVideoX 5B, 161 frames, 1360×768 resolution
- Language reasoning: GPT-OSS-20B
- Keyframe generation: Qwen-Image series (Edit model)
- Number of events determined empirically as 4 (balancing temporal supervision and keyframe stability)

## Key Experimental Results

### Main Results — PhyGenBench

| Method | Mechanics | Optics | Thermodynamics | Materials | Avg.↑ |
|--------|-----------|--------|----------------|-----------|-------|
| CogVideoX-5B | 0.39 | 0.55 | 0.40 | 0.42 | 0.45 |
| + PhyT2V | 0.45 | 0.55 | 0.43 | 0.53 | 0.50 |
| + PhysHPO (Prev. SOTA) | 0.55 | 0.68 | 0.50 | 0.65 | 0.61 |
| **+ Ours** | **0.67** | **0.72** | **0.65** | **0.60** | **0.66** |

Overall performance of 0.66 on PhyGenBench, surpassing Prev. SOTA PhysHPO by 8.19%.

Phenomenon Detection (PD) / Physical Order (PO) breakdown:

| Method | Mechanics PD/PO | Optics PD/PO | Thermodynamics PD/PO | Materials PD/PO |
|--------|-----------------|--------------|----------------------|-----------------|
| DiffPhy | 0.73/0.53 | 0.83/0.66 | 0.70/0.58 | 0.73/0.43 |
| **Ours** | **0.79/0.79** | **0.84/0.85** | **0.78/0.69** | **0.75/0.58** |

Gains in Physical Order (PO) are particularly pronounced, demonstrating the effectiveness of causal event chain modeling.

### Ablation Study

| Variant | Mechanics | Optics | Thermodynamics | Materials | Avg. |
|---------|-----------|--------|----------------|-----------|------|
| Full method | 0.67 | 0.72 | 0.65 | 0.60 | 0.66 |
| w/o PFG (Physics Formula Grounding) | 0.63 | 0.69 | 0.61 | 0.53 | 0.62 |
| w/o PPD (Physical Phenomena Decomposition) | 0.58 | 0.67 | 0.61 | 0.52 | 0.59 |
| w/o PNR (Progressive Narrative Revision) | 0.65 | 0.70 | 0.64 | 0.56 | 0.64 |
| w/o IKS (Interactive Keyframe Synthesis) | 0.50 | 0.64 | 0.58 | 0.48 | 0.55 |

### Key Findings

1. **IKS contributes most (−17%)**: Explicitly generating dedicated keyframes is critical for anchoring cross-frame dynamics and maintaining physically grounded visual evolution.
2. **PPD contributes significantly (−11%)**: Decomposing complex processes into logically ordered event chains is indispensable for generating realistic physical phenomenon evolution.
3. **PFG is also important (−6%)**: Standard physical formulas provide the foundation for quantitatively understanding physical laws.
4. **Optimal number of events is 4**: Too few (1–3) provides weak temporal supervision; too many (5–6) accumulates errors in keyframe editing propagation.
5. **Leading performance on VideoPhy as well**: Overall SA=1, PC=1 score of 49.3%, surpassing Prev. SOTA by approximately 3.4%.

## Highlights & Insights

- **Paradigm innovation**: Elevating physical phenomena from "static descriptions of a single moment" to "causally connected event sequences" represents a fundamental shift in PPVG problem formulation.
- **Physics formula grounding** embeds deterministic physical constraints into CoT reasoning, resolving the causal ambiguity inherent in purely language-based inference.
- **Dual-modality collaborative prompting** (semantic + visual) constrains the continuity of physical transitions more effectively than language prompts alone.
- The framework is decoupled from the video generation model and can be applied in a plug-and-play fashion to different video diffusion models.

## Limitations & Future Work

- **Failure on combined physical laws**: When a scene is governed by multiple physical laws simultaneously, insufficient compositional physical reasoning in the base model leads to generation failures.
- Reliance on external LLMs (GPT-OSS-20B) and image editing models (Qwen-Image-Edit) results in a complex pipeline.
- Error accumulation through keyframe editing propagation limits the number of events that can be supported.
- The physical parameter change threshold $\tau_p$ and the number of events require manual specification.
- The evaluation metric (PCA) itself relies on LLMs, which may introduce evaluation bias.

## Related Work & Insights

- **DiffPhy/PhyT2V**: Enhance generation through physics-aware prompting, but lack causal modeling and reduce phenomena to single scenes.
- **PhysHPO**: Hierarchical fine-grained preference optimization; Prev. SOTA but still lacks event sequence modeling.
- **Z-Sampling/Visual-CoG**: Incorporate CoT reasoning into visual generation, but focus primarily on semantic and spatial reasoning while neglecting physical causality.
- **Insights**: The event chain + physics formula grounding paradigm is generalizable to other video generation tasks requiring causal reasoning (e.g., long-video narration, simulation).

## Rating

- **Novelty**: 8/10 — The combination of event chain modeling and physics formula grounding is novel, with a clearly defined problem formulation.
- **Experimental Thoroughness**: 8/10 — Comprehensive validation on two benchmarks with complete ablations, though user studies are absent.
- **Writing Quality**: 8/10 — Clear structure with detailed descriptions of modular design.
- **Value**: 7/10 — The inference pipeline is heavyweight and may limit practical deployment, but the work provides an important research direction for PPVG.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [\[AAAI 2026\] Mask2IV: Interaction-Centric Video Generation via Mask Trajectories](../../AAAI2026/video_generation/mask2iv_interaction-centric_video_generation_via_mask_trajectories.md)
- [\[AAAI 2026\] Seeing the Unseen: Zooming in the Dark with Event Cameras](../../AAAI2026/video_generation/seeing_the_unseen_zooming_in_the_dark_with_event_cameras.md)
- [\[ICCV 2025\] DreamRelation: Relation-Centric Video Customization](../../ICCV2025/video_generation/dreamrelation_relation-centric_video_customization.md)
- [\[ICCV 2025\] Causal-Entity Reflected Egocentric Traffic Accident Video Synthesis](../../ICCV2025/video_generation/causal-entity_reflected_egocentric_traffic_accident_video_synthesis.md)

<!-- RELATED:END -->
