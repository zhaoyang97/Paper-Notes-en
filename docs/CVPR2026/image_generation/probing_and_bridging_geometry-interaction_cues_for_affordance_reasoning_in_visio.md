---
title: >-
  [Paper Note] Probing and Bridging Geometry–Interaction Cues for Affordance Reasoning in Vision Foundation Models
description: >-
  [CVPR 2026][Image Generation][affordance reasoning] This work systematically probes affordance capabilities in vision foundation models (VFMs), revealing that DINO encodes part-level geometric structure while Flux encodes verb-conditioned interaction priors. By training-free fusion of both, the method achieves zero-shot affordance estimation competitive with weakly supervised approaches.
tags:
  - CVPR 2026
  - Image Generation
  - affordance reasoning
  - vision foundation models
  - geometry perception
  - interaction prior
  - training-free fusion
date: 2026-05-08
content_hash: 2855f58ae875235f
---

# Probing and Bridging Geometry–Interaction Cues for Affordance Reasoning in Vision Foundation Models

**Conference**: CVPR 2026
**arXiv**: [2602.20501](https://arxiv.org/abs/2602.20501)
**Code**: [https://github.com/Probing-and-Bridging-Affordance](https://github.com/Probing-and-Bridging-Affordance) (coming soon)
**Area**: Image Generation
**Keywords**: affordance reasoning, vision foundation models, geometry perception, interaction prior, training-free fusion

## TL;DR
This work systematically probes affordance capabilities in vision foundation models (VFMs), revealing that DINO encodes part-level geometric structure while Flux encodes verb-conditioned interaction priors. By training-free fusion of both, the method achieves zero-shot affordance estimation competitive with weakly supervised approaches.

## Background & Motivation

**Background**: Visual affordance describes how objects can be manipulated, serving as a bridge between visual perception and embodied action. Current methods fall into three paradigms: fully supervised (learning geometric patterns from pixel-level annotations), weakly supervised (inferring affordances from human-object interactions), and open-vocabulary (generalizing via text-image alignment).

**Limitations of Prior Work**: Fully supervised methods rely on dense annotations and generalize poorly; weakly supervised methods are spatially imprecise; open-vocabulary methods depend on semantic associations rather than genuine interaction understanding. Each paradigm emphasizes different evidence, lacking a unified perspective to answer the core question: "What are the fundamental capabilities a visual system needs to understand affordance?"

**Key Challenge**: Affordance is not an intrinsic property of objects but rather the possibility of interaction between an agent and its environment. Existing methods treat geometry (object structure) and interaction (how actions operate on structure) in isolation, lacking systematic study of the complementarity between these two dimensions.

**Key Insight**: VFMs, through large-scale pretraining, have internalized rich visual knowledge and can serve as a unified lens to directly probe affordance capabilities—testing the hypothesis that geometry perception + interaction perception = the fundamental components of affordance understanding.

**Core Idea**: A dual-dimension framework is proposed that extracts priors from DINO (geometric part prototypes) and Flux (verb-conditioned cross-attention) respectively, achieving zero-shot affordance estimation via training-free fusion, and demonstrating that these two dimensions are composable foundational capabilities.

## Method

### Overall Architecture
Rather than proposing a single task-specific method, this work presents a systematic probing study in three stages: (1) probing the geometry dimension—verifying the correlation between geometric perception and affordance segmentation; (2) probing the interaction dimension—discovering that verb-conditioned spatial attention in generative models encodes implicit interaction priors; (3) fusion validation—achieving zero-shot affordance estimation by training-free fusion of DINOv3 geometric prototypes and Flux Kontext interaction maps.

### Key Designs

1. **Geometry Dimension Probing (Sec 3.1)**

    - **Observation 1.1**: Linear probes are applied to six VFMs following the Probing3D protocol. Models with stronger geometric perception (e.g., DINOv2) yield higher affordance segmentation mIoU. Adding Metric3Dv2 depth/normal information yields significant gains for semantic models (CLIP, SigLIP), while DINOv2 shows almost no improvement—indicating that its pretraining has already embedded strong geometric priors.
    - **Observation 1.2**: PCA visualization reveals that the DINO family uniquely organizes scenes into part-level structures (handles, cup rims, blade edges, and other functional parts are clearly separated), whereas SAM captures only boundaries, CLIP collapses to semantic categories, and Stable Diffusion produces smooth surface embeddings.
    - **Observation 1.3**: DINO's part correspondence exhibits a "semantic assimilation" phenomenon—when viewing only local shape, all ring-shaped structures are activated (pure geometric response); but in the full-object context, ring responses from non-cup objects disappear (suppressed by semantics). This indicates that geometry and semantics are coupled in DINO.

2. **Interaction Dimension Probing (Sec 3.2)**

    - **Observation 2.1**: Analysis of Flux.1-dev's text-image cross-attention reveals that verb tokens (hold, cut, drink, support) consistently attend to contact regions on objects, while noun tokens attend to corresponding entities. This separable pattern reveals that generative models internally encode verb-conditioned interaction priors.
    - **Observation 2.2**: A controlled image-editing framework is designed using Flux Kontext—extracting verb cross-attention from the generation process via triplet templates (agent, object, verb) such as "add a hand to hold the knife." These attention maps maintain spatial consistency even when generation fails, indicating that interaction priors arise from internal representations rather than pixel tracking.
    - Directly using verb attention as zero-shot affordance prediction already approaches weakly supervised performance on AGD20K (KLD 1.825 vs. weakly supervised 1.787).

3. **Geometry–Interaction Fusion (Sec 4)**

    - **Function**: Training-free composition of both priors to generate affordance masks.
    - **Mechanism**: (1) Flux Kontext generates object attention → cropping the ROI from DINOv3 feature maps; (2) PCA is applied to ROI features to obtain part-level bases; (3) Flux generates verb attention → NSS (Normalized Scanpath Saliency) selects geometric bases most aligned with verb attention; (4) geometric bases and verb attention are fused to produce the final affordance mask.
    - **Design Motivation for PCA**: Due to DINO's geometry–semantics coupling (Observation 1.3), directly using features introduces noise; PCA extracts cleaner shape-centric geometric prototypes.

### Loss & Training
Completely training-free; no affordance annotations or fine-tuning are required.

## Key Experimental Results

### Main Results (AGD20K)

| Method | Supervision | KLD↓ | SIM↑ | NSS↑ |
|--------|-------------|------|------|------|
| AffordanceLLM | Fully Supervised | 1.463 | 0.377 | 1.070 |
| LOCATE | Weakly Supervised | 1.405 | 0.372 | 1.157 |
| Cross-View-AG | Weakly Supervised | 1.787 | 0.285 | 0.829 |
| Ours (interaction only) | Training-Free | 1.825 | 0.271 | 1.050 |
| **Ours (interaction × geometry)** | **Training-Free** | **1.493** | **0.326** | **1.090** |

### Geometry Probing Results (UMD Linear Probe mIoU)

| Model | Architecture | Supervision | mIoU (base) | mIoU (+depth & normal) |
|-------|-------------|-------------|-------------|------------------------|
| DINOv2 | ViT-B/14 | SSL | Highest | Negligible gain |
| CLIP | ViT-B/16 | VLM | Medium | Significant gain |
| SAM | ViT-B/16 | Seg. | Low | Moderate gain |
| StableDiffusion | UNet | Gen. | Lowest | Some gain |

### Key Findings
- **Significant gain from fusion**: Combining geometry and interaction reduces KLD from 1.825 to 1.493 and improves SIM from 0.271 to 0.326, demonstrating that geometric priors suppress implausible regions.
- **Zero-shot approaching weakly supervised**: The training-free method achieves an NSS of 1.090, slightly surpassing weakly supervised Cross-View-AG (0.829) and approaching AffordanceLLM (1.070).
- Verb attention **remains spatially consistent even when generation fails**, confirming that interaction priors are intrinsic model knowledge rather than back-projected from output images.

## Highlights & Insights
- The **dual-dimension framework** is theoretically profound—it is the first work to systematically decompose affordance into operable "geometry" and "interaction" dimensions and empirically verify their complementarity and composability.
- **Uncovering interaction priors in generative models** is a genuinely novel perspective—prior work has not applied Flux's cross-attention to affordance estimation, opening a new direction for generative models as sources of interaction knowledge.
- The elegant experimental design for DINO's "semantic assimilation" (Observation 1.3)—contrasting cosine similarity under same vs. different object contexts to cleanly separate geometric vs. semantic contributions—is particularly well-crafted.
- The framework constitutes a **model-agnostic compositional paradigm**: any strong geometry model combined with any generative model encoding interaction priors can be interchangeably composed.

## Limitations & Future Work
- **Quality of primitives is limited**: Noun-conditioned attention maps are noisy and generation outputs are unstable, affecting the quality of extracted interaction priors.
- **Shallow fusion strategy**: The current approach relies on NSS-based selection and linear fusion, an open-loop signal combination without closed-loop optimization or iterative refinement.
- The method depends on Flux Kontext's controllable image editing capability; performance may degrade for abstract verbs (e.g., "think," "appreciate") or non-contact interactions.
- **Video generative models remain unexplored**: The authors note in discussion that video models inherently require understanding of 3D scene geometry and interaction dynamics, potentially providing both geometric and interaction priors simultaneously—an important future direction.
- The number and selection strategy of PCA-derived geometric prototypes are relatively simple; more complex object structures may require hierarchical decomposition.
- Quantitative evaluation of the fusion approach is conducted only on AGD20K; the UMD dataset is used only for qualitative verification of geometric consistency, limiting evaluation breadth.

## Related Work & Insights
- **vs. LOCATE / Cross-View-AG** (weakly supervised): These methods require human-object interaction training data, whereas the proposed method is completely training-free yet achieves competitive performance without being constrained by specific training distributions.
- **vs. AffordanceLLM** (fully supervised): The proposed method approaches its NSS performance but still lags in KLD (1.493 vs. 1.463), indicating insufficient fine-grained precision that could be improved through better fusion strategies.
- **vs. Affordance-R1** (post-training with RL): That method achieves a KLD as high as 9.730, worse than the proposed training-free approach, suggesting that RL may overfit on this task.
- **vs. GroundingDINO / MaskCLIP / EVF-SAM**: These vision-language models support global semantic matching but fail to localize functional regions; Flux Kontext's verb attention demonstrates clear advantages in fine-grained interaction localization.
- The framework provides a generalizable methodology for analyzing VFM capabilities—the same dual-dimension probing paradigm can be transferred to other tasks such as spatial relation understanding and causal reasoning.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The first work to systematically probe affordance capabilities in VFMs; the dual-dimension framework and the discovery of interaction priors in generative models are pioneering contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Probing experiments are elegantly designed, but quantitative fusion evaluation is limited to AGD20K.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The logical chain is exceptionally clear, progressing coherently from hypothesis → probing → validation → fusion.
- **Value**: ⭐⭐⭐⭐ — Provides a novel perspective for affordance research and VFM understanding, though practical application requires stronger fusion strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DEXTER: Diffusion-Guided EXplanations with TExtual Reasoning for Vision Models](../../NeurIPS2025/image_generation/dexter_diffusion-guided_explanations_with_textual_reasoning_for_vision_models.md)
- [\[ICCV 2025\] Mind the Gap: Aligning Vision Foundation Models to Image Feature Matching](../../ICCV2025/image_generation/mind_the_gap_aligning_vision_foundation_models_to_image_feature_matching.md)
- [\[CVPR 2026\] Cinematic Audio Source Separation Using Visual Cues](cinematic_audio_source_separation_using_visual_cues.md)
- [\[ICCV 2025\] A Unified Framework for Motion Reasoning and Generation in Human Interaction](../../ICCV2025/image_generation/a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)
- [\[CVPR 2026\] CoD: A Diffusion Foundation Model for Image Compression](cod_a_diffusion_foundation_model_for_image_compression.md)

</div>

<!-- RELATED:END -->
