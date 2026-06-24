---
title: >-
  [Paper Note] CoT-Edit: Let CoT Guide Instruction Video Editing
description: >-
  [CVPR 2026][Video Generation][Instruction Video Editing] Addressing the issues of inaccurate target localization and physically implausible object additions in complex scenarios for text-only instruction video editing, this paper proposes a three-stage **Plan–Guide–Edit** framework. A Multimodal Large Language Model (MLLM) with Chain-of-Thought (CoT) first "translates" instructions into a sequence of keyframe bounding boxes and enhanced instructions. A box-constrained mask br…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Instruction Video Editing"
  - "Chain-of-Thought"
  - "MLLM Planner"
  - "Box-guided Mask"
  - "Diffusion Editing"
date: 2026-05-08
content_hash: 222133bb43da4fe2
---

# CoT-Edit: Let CoT Guide Instruction Video Editing

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_CoT-Edit_Let_CoT_Guide_Instruction_Video_Editing_CVPR_2026_paper.html)  
**Code**: https://github.com/flying-sky999/CoTEdit  
**Area**: Video Generation / Instruction Video Editing  
**Keywords**: Instruction Video Editing, Chain-of-Thought, MLLM Planner, Box-guided Mask, Diffusion Editing  

## TL;DR
Addressing the issues of inaccurate target localization and physically implausible object additions in complex scenarios for text-only instruction video editing, this paper proposes a three-stage **Plan–Guide–Edit** framework. A Multimodal Large Language Model (MLLM) with Chain-of-Thought (CoT) first "translates" instructions into a sequence of keyframe bounding boxes and enhanced instructions. A box-constrained mask branch then converts spatial priors into temporally consistent masks. Finally, a diffusion editor integrates the masks, enhanced instructions, and video features to complete the editing, significantly outperforming existing open-source baselines in physical plausibility and spatial relations.

## Background & Motivation
**Background**: Instruction-based video editing allows users to modify videos using a source video and a natural language instruction, extending the InstructPix2Pix paradigm to the video domain. Mainstream approaches (e.g., InsViE, Lucy-1.1, InstructX) are end-to-end, requiring a single diffusion or multimodal model to simultaneously understand cross-frame semantics, locate targets, and execute editing.

**Limitations of Prior Work**: in complex real-world scenarios with multiple similar objects and dynamic interactions, text-driven methods often fail. For instance, the instruction "turn the yellow dog into an orange cat" might target the wrong dog, or "add a UFO following an elliptical path" might violate physical motion trajectories. These failures stem from ambiguous text signals and a lack of explicit spatial grounding and physical constraints, leading to localization drift, incorrect edits, and temporal jitter. Prior methods compensate for this by scaling up aligned data and model capacity.

**Key Challenge**: The "what to edit" and "where to edit" are entangled within the same text-to-pixel mapping. A direct mitigation is to introduce masks as spatial conditions to convert global retrieval into local controllable editing. However, if masks are generated **solely from original text instructions**, semantic ambiguity and spatial uncertainty persist in the mask layer. Moreover, for object addition tasks, text-derived masks fail to provide executable physical priors (rational position, scale, and motion logic for new objects), leaving the model without guidance.

**Goal**: Build a reliable bridge from high-level semantics to low-level pixels that retains linguistic expressiveness while providing fine-grained spatial/physical constraints without heavy reliance on massive aligned annotations.

**Core Idea**: Decompose the implicit "understanding $\rightarrow$ editing" process into an explicit **Plan $\rightarrow$ Guide $\rightarrow$ Edit** sequence. A CoT-enhanced MLLM serves as a planner to **first** reason instructions into interpretable structured intermediates (bounding box sequences + enhanced instructions), which then explicitly guide mask generation and diffusion editing, decoupling the spatial localization burden from the diffusion model.

## Method

### Overall Architecture
Given a source video $S$ and an instruction $I$, CoT-Edit operates as a pipeline of three core modules: the **Planner** translates high-level semantic intent into executable spatial constraints; the **Guide (Mask Branch)** generates spatio-temporally consistent masks under explicit spatial priors; and the **Editor (Diffusion Editor)** fuses multiple conditions to complete appearance modification and content generation. These modules are not just serially connected—the Guide and Editor are **bidirectionally coupled** via a Reverse-Connector and Mask-Connector, allowing low-level mask guidance and high-level editing semantics to mutually correct each other.

Specifically, the latter two branches use the Wan2.2 5B diffusion model as a backbone. The source video is mapped to a low-dimensional latent representation $y_0$ via a 3D VAE. Noise is injected during the diffusion process to obtain $y_{\text{noise}}$, and both are concatenated along the channel dimension:

$$y_{\text{input}} = \text{ChannelConcat}(y_{\text{noise}}, y_0)$$

The input channels of the tokenizer are expanded to 32 (while keeping the output dimension unchanged) to avoid disrupting the original distribution of Wan2.2 5B.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input: Video + Instruction"] --> B["CoT Semantic-Spatial Planner<br/>5-step Reasoning → Box Seq + Enhanced Instr."]
    B -->|Box B (Hard Constraint)<br/>Enhanced Instr. EI (Soft Constraint)| C["Box-Guided Mask Generation<br/>Global Retrieval → Local Refinement"]
    C -->|Mask-Connector Injecting Mask| D["Multi-Condition Diffusion Editor<br/>Fuses Mask / Instr. / Video Features"]
    D -->|Reverse-Connector Feedback Semantics| C
    D --> E["Mask Composition: Edit Inside Box<br/>Retain Source Outside Box"]
```

### Key Designs

**1. CoT Semantic-Spatial Planner: Reasoning Ambiguous Instructions into Executable Boxes + Enhanced Instructions**

This is the source for resolving "text ambiguity." The planner is an MLLM with an explicitly embedded Chain-of-Thought. It takes temporally ordered keyframes $\{I_t\}_{t=1}^{T}$ and user instructions as input, outputting two structured results: a normalized bounding box sequence $\{b_t\}_{t=1}^{T}$ aligned with keyframes to define "where to edit," and an **Enhanced Instruction (EI)** that augments the original text with semantic priors such as target attributes, relative spatial relations, contact modes, and camera consistency prompts. For non-spatial tasks like stylization, it outputs empty box sequences.

The authors use multi-step CoT for three reasons: sequential decomposition reduces task difficulty; modeling physical/cinematic constraints early prevents localization errors from being amplified; and it provides interpretable guidance for downstream stages. Reasoning is organized into three coupled phases: (i) **Task Parsing + Cross-frame Perception**—parsing task types and subjects, estimating camera motion, and performing cross-frame instance identification to anchor language to video instances; (ii) **Physical and Temporal Consistency Modeling**—reasoning displacement, scale changes, and visibility to generate smooth trajectories while embedding world knowledge (gravity, occlusion), mitigated by self-reflection; and (iii) **Generating Spatial and Semantic Guidance**—producing the box sequence and $EI$. This dual-channel approach decouples **rigid spatial constraints (boxes)** from **flexible semantic control (EI)**.

**2. Box-Guided Mask Generation + Reverse-Connector: From Global Retrieval to Local Refinement**

Once the planner completes spatial localization, the Guide branch no longer needs to solve the joint localization-generation problem. It only needs to infer precise shapes within specified regions. It treats the box sequence $B=\{b_t\}$ as a **hard constraint** and video features plus $EI$ as **soft constraints** to generate a binary mask $M$. Its $l$-th layer spatio-temporal features $C_M^l$ serve as implicit spatial guidance for editing. The Guide and Editor share the same hierarchical architecture.

To stabilize masks for thin structures or heavy occlusions, a **Reverse-Connector** is introduced from the Editor back to the Guide:

$$C_M^l = C_M^l + \text{ReverseConnector}(Q_E^l)$$

where $Q_E^l$ represents the $l$-th layer editor features. The intuition is that the editor's semantic understanding of "what to edit" helps the mask branch recover missing details. This connection is multi-layered and repeatable. Finally, a Mask-Connector maps the refined features to frame-resolution masks $\{M_t\}$.

**3. Multi-Condition Diffusion Editor + Mask-Connector Bidirectional Coupling**

The Editor executes appearance modification and content generation using the precise masks $\{M_t\}$ and $EI$. It re-injects mask features at multiple editing layers via additive modulation:

$$Q_E^l = Q_E^l + \text{MaskConnector}(C_M^l)$$

This ensures that editing features at different depths remain aware of low-level spatial guidance. Combined with the Reverse-Connector, this forms a **bidirectional collaboration** between Guide and Editor, making the model more robust to boundaries, occlusions, and lighting.

On the semantic side, the Editor uses a dedicated Qwen-VL cross-attention to digest the enhanced instruction. It takes the first 1024 tokens of the last-layer vision-language features $V$ from Qwen-VL, reduces the channel dimension from 3584 to 3072 via an MLP, and applies cross-attention:

$$Q_E^l = Q_E^l + \text{QVLcrossattn}(\text{MLP}(V),\, C_M^l)$$

This allows the editor to utilize both raw video visual priors and the structured world knowledge organized during the planning stage.

### Loss & Training
A two-stage training strategy is adopted. **Stage 1 (Modular Training)**: The Mask (Guide) branch is trained on a mix of image segmentation (ADE20K, PhraseCut) and video segmentation (YouTube-VOS, OVIS) data. The Editor branch is trained on open-source editing data (AnyEdit, UltraEdit, EditWorld) and video instruction data (Señorita-2M, Ditto). **Stage 2 (Joint Training)**: Both branches are fine-tuned on an internal dataset of approximately 100,000 high-quality edit pairs with precise mask annotations. Resolution is $720 \times 1280$, with 20k steps for Stage 1 and 10k steps for Stage 2 at a batch size of 64. Connectors are zero-initialized.

## Key Experimental Results

### Main Results
Evaluation includes visual quality (FVD, VBench) and instruction following (CLIPScore, Gemini-based scoring for physical plausibility, spatial relations, and overall quality). The test set comprises 100 videos from Koala-36M with diverse instructions.

| Model | BC ↑ | CLIPScore ↑ | FVD ↓ | MS ↑ | AES ↑ | Physical Rules ↑ | Spatial Rel. ↑ | Instr. Follow ↑ | Edit Quality ↑ |
|------|------|-------------|-------|------|-------|-----------|-----------|-----------|-----------|
| InsV2V | 0.921 | 0.129 | 4095.42 | 0.95 | 0.57 | 0.367 | 0.291 | 0.401 | 0.326 |
| StableV2V | 0.942 | 0.263 | 3222.18 | 0.99 | 0.46 | 0.305 | 0.341 | 0.381 | 0.319 |
| InsViE | 0.936 | 0.395 | 2397.65 | 1.10 | 0.48 | 0.389 | 0.285 | 0.355 | 0.369 |
| AnyV2V | 0.927 | 0.251 | 2942.77 | 0.95 | 0.41 | 0.313 | 0.373 | 0.265 | 0.394 |
| OmniVideo | 0.972 | 0.382 | 1076.44 | 1.04 | 0.44 | 0.590 | 0.641 | 0.526 | 0.604 |
| Lucy-1.1 | 0.931 | 0.376 | 1488.12 | 0.98 | 0.52 | 0.488 | 0.769 | 0.562 | 0.589 |
| **Ours** | **0.974** | **0.445** | **1015.67** | **1.18** | **0.62** | **0.741** | **0.841** | **0.629** | **0.648** |

Ours leads in most dimensions, especially in Physical Rules (0.741 vs. 0.590) and Spatial Relations (0.841 vs. 0.769), which correspond to the design intent of explicit spatial planning.

### Ablation Study
Ablations evaluate Physical Plausibility (PR), Spatial Relations (SR), Instruction Following (IF), and Overall Editing Quality (OEQ).

| Config | PR | SR | IF | OEQ | Note |
|------|-----|-----|-----|-----|------|
| E w/o MLLM | 0.501 | 0.754 | 0.575 | 0.598 | Editor only, no Qwen3-VL instruction enhancement |
| E w/ MLLM | 0.674 | 0.758 | 0.598 | 0.631 | Editor + Qwen3-VL features |
| E + M w/ Mc | 0.643 | 0.807 | 0.586 | 0.638 | Adding Mask branch (Mask-Connector only) |
| E + M w/ Mc & Rc | **0.681** | **0.815** | **0.609** | **0.647** | Full model (Mask-Connector + Reverse-Connector) |

### Key Findings
- **Instruction Enhancement**: The Qwen3-VL enhancement (E w/o $\rightarrow$ w/ MLLM) improves PR from 0.501 to 0.674, indicating the critical role of MLLM semantic priors in physical plausibility.
- **Mask Branch**: Primarily boosts spatial relations (SR 0.758 $\rightarrow$ 0.807), confirming its role in "where to edit."
- **Reverse-Connector**: Recovers performance for thin/occluded structures, improving all metrics and highlighting the benefit of Editor $\rightarrow$ Mask feedback.
- **CoT for Motion**: Qualitative results show that without CoT, a ping-pong ball fails to follow a physically plausible path; with CoT, it correctly follows a "parabolic descent $\rightarrow$ collision $\rightarrow$ bounce" trajectory.

## Highlights & Insights
- **Decoupling Spatial Localization**: Outsourcing localization to a CoT Planner produces verifiable bounding boxes as hard anchors, allowing the diffusion model to focus solely on local generation. This structural intermediate reduces task difficulty and data dependence.
- **Dual-Channel Guidance**: Using rigid boxes for spatial anchors and flexible instructions for semantic/physical priors is more robust than embedding all constraints into a single text prompt.
- **Bidirectional Coupling**: The Reverse-Connector allows high-level semantics to rectify low-level modules (Editor $\rightarrow$ Mask). This bidirectional connector strategy can be adapted to any "localization + generation" serial pipeline.
- **CoT Beyond Text**: This work applies CoT to "semantic $\rightarrow$ spatial coordinate" translation, demonstrating that explicit reasoning of physical rules can directly drive generated trajectories.

## Limitations & Future Work
- **Reliance on Planner Quality**: Since spatial anchors originate from the MLLM, incorrect bounding boxes in crowded or fast scenes will mislead downstream stages. The paper lacks independent quantitative metrics for planner localization accuracy (e.g., IoU).
- **Evaluation Bias**: Core findings (physical rules, spatial relations) rely on Gemini scoring, which may contain model-specific biases. The 100-video evaluation set is relatively small.
- **Training Data**: The second stage requires high-quality mask-annotated pairs (100k internal pairs), posing a barrier to reproduction.
- **Temporal Consistency (TC)**: Our TC score is slightly lower than some baselines; whether multi-module concatenation introduces temporal instability warrants further analysis.

## Related Work & Insights
- **vs. Lucy-1.1 / InsViE**: These end-to-end models lack strong instruction injection mechanisms. Ours outperforms them in spatial relations (0.841 vs. 0.769) and physical rules (0.741 vs. 0.488) by externalizing localization to a CoT planner.
- **vs. InstructX**: InstructX uses MLLM for semantic generalization but relies on implicit attention for spatial constraints. Ours uses **explicit** bounding boxes and masks.
- **vs. OmniVideo**: OmniVideo is the strongest baseline (FVD 1076), yet ours achieves better scores in physical rules and spatial relations due to the explicit injection of physical/spatial priors rather than just scaling the model.

## Rating
- Novelty: ⭐⭐⭐⭐ Plan–Guide–Edit + CoT effectively externalizes spatial localization. The bidirectional box/mask coupling is a clean and effective combination of innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid comparison with 6 strong baselines, multi-dimensional metrics, and dual ablations. However, independent quantification of the planner is missing.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, well-defined module responsibilities, and easy-to-understand pipeline description.
- Value: ⭐⭐⭐⭐ High utility for instruction video editing in complex scenes; the decoupling strategy for diffusion models is highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EasyV2V: A High-quality Instruction-based Video Editing Framework](easyv2v_a_high-quality_instruction-based_video_editing_framework.md)
- [\[CVPR 2026\] Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset](scaling_instruction-based_video_editing_with_a_high-quality_synthetic_dataset.md)
- [\[CVPR 2026\] VIVA: VLM-Guided Instruction-Based Video Editing with Reward Optimization](viva_vlm-guided_instruction-based_video_editing_with_reward_optimization.md)
- [\[ICLR 2026\] IVEBench: Modern Benchmark Suite for Instruction-Guided Video Editing Assessment](../../ICLR2026/video_generation/ivebench_modern_benchmark_suite_for_instruction-guided_video_editing_assessment.md)
- [\[CVPR 2026\] Let Your Image Move with Your Motion! – Implicit Multi-Object Multi-Motion Transfer](let_your_image_move_with_your_motion_--_implicit_multi-object_multi-motion_trans.md)

</div>

<!-- RELATED:END -->
