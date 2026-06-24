---
title: >-
  [Paper Note] ScanReason: Empowering 3D Visual Grounding with Reasoning Capabilities
description: >-
  [ECCV 2024][3D Vision][3D visual grounding] This paper proposes a new task of 3D reasoning grounding and introduces the ScanReason benchmark (10K+ QA-location pairs, 5 reasoning types). It designs the ReGround3D framework to collaborate MLLM reasoning with a 3D grounding module via a Chain-of-Grounding mechanism, achieving accurate 3D object localization under implicit instructions.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D visual grounding"
  - "reasoning"
  - "MLLM"
  - "chain-of-grounding"
  - "3D scene understanding"
date: 2026-05-08
content_hash: d4b05ecbd7f70da6
---

# ScanReason: Empowering 3D Visual Grounding with Reasoning Capabilities

**Conference**: ECCV 2024  
**arXiv**: [2407.01525](https://arxiv.org/abs/2407.01525)  
**Code**: [https://github.com/ZCMax/ScanReason](https://github.com/ZCMax/ScanReason)  
**Area**: 3D Vision  
**Keywords**: 3D visual grounding, reasoning, MLLM, chain-of-grounding, 3D scene understanding

## TL;DR
This paper proposes a new task of 3D reasoning grounding and introduces the ScanReason benchmark (10K+ QA-location pairs, 5 reasoning types). It designs the ReGround3D framework to collaborate MLLM reasoning with a 3D grounding module via a Chain-of-Grounding mechanism, achieving accurate 3D object localization under implicit instructions.

## Background & Motivation
**Background**: 3D visual grounding has made significant progress, but existing models (e.g., ScanRefer, BUTD-DETR) rely on explicit textual descriptions for localization, such as "the red chair near the window" — achieving this through direct alignment of object categories, attributes, and spatial relationships.

**Limitations of Prior Work**: Human instructions in real-world scenarios are often implicit — e.g., "I'm thirsty, is there anything to drink?" (requiring reasoning: "thirsty" $\rightarrow$ "drinks" $\rightarrow$ "fridge/cup on the table"). Existing models cannot handle such indirect instructions that require reasoning.

**Key Challenge**: 3D scene understanding requires both reasoning capability (understanding implicit intentions) and localization capability (determining precise 3D coordinates). Existing MLLMs (e.g., 3D-LLM) possess reasoning abilities but suffer from poor localization accuracy, while specialized grounding models achieve accurate localization but lack reasoning capabilities.

**Goal**: (a) Define the new task of 3D reasoning grounding; (b) construct a benchmark dataset containing multiple reasoning types; (c) design a model architecture capable of joint reasoning and precise localization.

**Key Insight**: Decouples reasoning and localization into two collaborative modules — reasoning first about "what to look for," and then looking back at the 3D scene for precise localization.

**Core Idea**: Uses an MLLM to perform visual-centric reasoning to generate a grounding query, and then utilizes a geometry-enhanced look-back mechanism to precisely localize the target in 3D point clouds.

## Method

### Overall Architecture
Input: 3D scene point clouds + implicit natural language questions $\rightarrow$ Visual-centric reasoning module (based on 3D-LLM) performs joint scene-question reasoning and outputs a special `<LOC>` token $\rightarrow$ 3D grounding module receives the `<LOC>` embedding and looks back at the original 3D scene to perform precise localization $\rightarrow$ Output: target object 3D bounding boxes + textual answers/explanations.

### Key Designs

1. **ScanReason Benchmark Dataset**:

    - **Function**: Defines a 3D reasoning grounding benchmark with 5 types of reasoning.
    - **Mechanism**: Spatial reasoning (understanding 3D relations between objects), functional reasoning (understanding object usage/functions), logical reasoning (goal-oriented multi-step reasoning), emotional reasoning (understanding human emotional needs), and safety reasoning (identifying risks and safety decisions). Uses GPT-4 combined with EmbodiedScan annotations to automatically generate 12,929 QA-location pairs.
    - **Design Motivation**: Builds a hierarchical reasoning system ranging from basic capabilities (spatial + functional) to high-level applications (logical + emotional + safety).

2. **Visual-Centric Reasoning Module**:

    - **Function**: Jointly reasons over 3D scenes and linguistic instructions to generate features embedding grounding intent.
    - **Mechanism**: Based on 3D-LLM (BLIP2 architecture), multi-view 2D features are back-projected into the 3D space and encoded into 32 visual tokens via a Q-Former. The vocabulary is expanded to include a `<LOC>` token, whose last-layer embedding $h_{loc}$ encodes the semantic and positional information of the target object.
    - **Design Motivation**: Instead of having the MLLM directly predict bounding box coordinates (which yields poor accuracy), it is designed to output a feature-level "localization intent," leaving the precise execution to a specialized localization module.

3. **3D Grounding with Geometry-Enhanced Look-Back**:

    - **Function**: Utilizes a 3D point cloud encoder to look back at the original scene for precise 3D localization.
    - **Mechanism**: Uses a 3D point cloud encoder to extract fine-grained geometric features $f_{scene}$. The Query Selection Module employs cross-attention ($f_{scene}$ as Q, $h_{loc}$ as K/V) to generate an activation heatmap, selecting top-$k$ most relevant features as object queries. Finally, a Transformer decoder predicts the 3D bounding box.
    - **Design Motivation**: The visual tokens of 3D-LLM are based on 2D image features and lack precise 3D geometric information; looking back at the original point cloud supplements fine-grained spatial structures.

4. **Chain-of-Grounding (CoG) Mechanism**:

    - **Function**: Alternates reasoning and localization for multiple rounds to progressively refine localization results.
    - **Mechanism**: First translates the raw implicit question into locating explicitly mentioned objects $\rightarrow$ obtains the 3D locations and confidence of these objects $\rightarrow$ inserts the localization results to update the question $\rightarrow$ performs reasoning and localization again $\rightarrow$ outputs the final target. This is analogous to Chain-of-Thought, but alternates between reasoning and localization steps.
    - **Design Motivation**: In complex questions, localization results can retroactively assist reasoning — e.g., knowing where the "kitchen" is helps reason about "the nearest trash can."

### Loss & Training
$\mathcal{L} = \lambda_{text}\mathcal{L}_{text} + \lambda_{det}\mathcal{L}_{det}$, where $\mathcal{L}_{det} = \lambda_{IOU}\mathcal{L}_{IOU} + \lambda_{contrast}\mathcal{L}_{contrast}$. Text loss originates from next-token prediction, and detection loss originates from 3D bounding box regression.

## Key Experimental Results

### Main Results (3D Visual Grounding - ScanRefer)

| Method | Type | Acc@0.25 | Acc@0.5 |
|------|------|----------|---------|
| BUTD-DETR | Specialist | 52.2 | 39.8 |
| L3Det | Specialist | 52.8 | 40.2 |
| 3D-LLM | MLLM | 30.3 | - |
| Chat3D-v2 | MLLM | 35.9 | 30.4 |
| **ReGround3D** | **Ours** | **53.1** | **41.1** |

### 3D Reasoning Grounding (ScanReason Benchmark)

| Method | Spatial | Functional | Logical | Emotional | Safety | Overall |
|------|---------|-----------|---------|-----------|--------|---------|
| Mask3D+InternLM2 | 10.34 | 36.12 | 9.98 | 8.21 | 8.99 | 14.86 |
| 3D-LLM(vg) | 18.31 | 17.42 | 10.97 | 8.12 | 6.33 | 13.29 |
| Chat3D-v2 | 20.21 | 18.39 | 11.32 | 7.98 | 9.88 | 14.98 |
| ReGround3D | 32.98 | 36.23 | 26.99 | 23.12 | 22.98 | 28.98 |
| **ReGround3D(CoG)** | **34.71** | **36.79** | **29.11** | **24.03** | **23.21** | **30.62** |

### Ablation Study

| Configuration | ScanReason Acc@0.25 | Description |
|------|-------------------|------|
| 3D-LLM(full+sr) | 19.21 | Directly use MLLM to output coordinates |
| ReGround3D | 28.98 | +3D grounding module $\rightarrow$ Gain +9.77 |
| ReGround3D(CoG) | 30.62 | +Chain-of-Grounding $\rightarrow$ Further Gain +1.64 |

### Key Findings
- The 3D grounding module is the largest source of gain (+9.77), verifying the effectiveness of the "decoupled reasoning and localization" design.
- CoG brings the most significant improvement in spatial reasoning and logical reasoning (+1.73 and +2.12), as these two task types heavily require "knowing the location of intermediate objects to continue reasoning."
- Even without using the ScanReason training data (ReGround3D*), it still significantly outperforms other MLLMs (23.27 vs 14.98), indicating that the architectural design itself has inherent advantages.
- Mask3D+InternLM2 demonstrates strong performance in functional reasoning (36.12), because functional reasoning relies primarily on common sense about object categories, which LLMs are naturally adept at.

## Highlights & Insights
- **Decoupled Reasoning-Localization + Look-Back Mechanism**: Instead of letting the MLLM directly output coordinates, it outputs "localization intent" which is then executed by a specialized module. This division of labor can be transferred to any task requiring MLLM + precise prediction (such as 2D grounding, segmentation, etc.).
- **Chain-of-Grounding**: Extends the concept of Chain-of-Thought (CoT) from pure text reasoning to alternating "reasoning + perception" — perceptual results feed back into reasoning, forming a stronger closed loop. This concept can be generalized to other tasks requiring multi-round perception-reasoning.
- **Hierarchical Design of Five Reasoning Types**: Ranging from foundational (spatial + functional) to advanced (logical + emotional + safety), it provides a systematic framework for evaluating the reasoning capabilities of embodied AI.

## Limitations & Future Work
- The overall accuracy is still relatively low (best Acc@0.25 is only 30.62), which is still far from practical application.
- The ScanReason dataset is automatically generated by GPT-4, which may introduce annotation noise and biases.
- CoG currently performs only two rounds of alternating reasoning-localization; more complex scenarios may require more rounds.
- The visual representation of 3D-LLM is based on 2D image projections, which may not be the optimal 3D encoding method.
- The definitions and classifications of reasoning types are somewhat subjective, with boundary definitions not being entirely clear.

## Related Work & Insights
- **vs 3D-LLM**: Both utilize MLLMs to understand 3D scenes, but 3D-LLM directly outputs coordinates with poor accuracy, whereas ReGround3D introduces a specialized grounding module to significantly improve localization capability.
- **vs Chat3D-v2**: Chat3D-v2 performs segmentation before identification, while ReGround3D reasons before localizing, making the latter more friendly to implicit instructions.
- **vs 2D visual grounding**: Reasoning and localization in 3D scenes are far more complex than in 2D (due to spatial relations, occlusions, and multi-view perspectives); while methodologies can inspire each other, the challenges are distinct.

## Rating
- Novelty: ⭐⭐⭐⭐ New task + new dataset + decoupled reasoning-localization design is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete ablation studies, but could include more baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with well-explained motivation.
- Value: ⭐⭐⭐⭐ The ScanReason benchmark holds significant value for the community, propelling research on embodied AI reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] S$^2$-MLLM: Boosting Spatial Reasoning Capability of MLLMs for 3D Visual Grounding with Structural Guidance](../../CVPR2026/3d_vision/s2-mllm_boosting_spatial_reasoning_capability_of_mllms_for_3d_visual_grounding_w.md)
- [\[ICCV 2025\] LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities](../../ICCV2025/3d_vision/llava-3d_a_simple_yet_effective_pathway_to_empowering_lmms_with_3d_capabilities.md)
- [\[CVPR 2026\] Masking Matters: Unlocking the Spatial Reasoning Capabilities of LLMs for 3D Scene-Language Understanding](../../CVPR2026/3d_vision/masking_matters_unlocking_the_spatial_reasoning_capabilities_of_llms_for_3d_scen.md)
- [\[NeurIPS 2025\] From Objects to Anywhere: A Holistic Benchmark for Multi-level Visual Grounding in 3D Scenes](../../NeurIPS2025/3d_vision/from_objects_to_anywhere_a_holistic_benchmark_for_multi-level_visual_grounding_i.md)
- [\[CVPR 2026\] EG-3DVG: Expression and Geometry Aware Grounding Decoder for 3D Visual Grounding](../../CVPR2026/3d_vision/eg-3dvg_expression_and_geometry_aware_grounding_decoder_for_3d_visual_grounding.md)

</div>

<!-- RELATED:END -->
