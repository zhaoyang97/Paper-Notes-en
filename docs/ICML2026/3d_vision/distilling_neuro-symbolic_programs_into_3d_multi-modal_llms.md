---
title: >-
  [Paper Note] APEIRIA: Distilling Neuro-Symbolic Programs into 3D Multi-modal LLMs
description: >-
  [ICML 2026][3D Vision][Neuro-symbolic] This paper introduces APEIRIA, which distills the execution traces of neuro-symbolic 3D concept learners into the natural language chain-of-thought (CoT) of a 3D MLLM. By employing…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Neuro-symbolic"
  - "3D Spatial Reasoning"
  - "Chain-of-Thought"
  - "GRPO"
  - "Curriculum Learning"
date: 2026-05-08
content_hash: 39cd3fde44f6de25
---

# APEIRIA: Distilling Neuro-Symbolic Programs into 3D Multi-modal LLMs

**Conference**: ICML 2026  
**arXiv**: [2606.01215](https://arxiv.org/abs/2606.01215)  
**Code**: https://github.com/oceanflowlab/APEIRIA  
**Area**: 3D Vision / Multi-modal VLM  
**Keywords**: Neuro-symbolic, 3D Spatial Reasoning, Chain-of-Thought, GRPO, Curriculum Learning

## TL;DR
This paper introduces APEIRIA, which distills the execution traces of neuro-symbolic 3D concept learners into the natural language chain-of-thought (CoT) of a 3D MLLM. By employing GRPO reinforcement learning, this reasoning paradigm is generalized to open-vocabulary and deeply nested instructions. APEIRIA outperforms both traditional NS3D methods and state-of-the-art 3D MLLMs on ScanRefer, Multi3DRefer, SQA3D, and Scan2Cap, while maintaining the interpretability and modularity of symbolic systems.

## Background & Motivation
**Background**: 3D spatial reasoning (grounding, QA, captioning) is currently dominated by two approaches. The first is Neuro-Symbolic 3D (NS3D) concept learners (e.g., NS3D, LARC), which parse instructions into programs composed of primitives like `scene/filter/relate` for step-by-step execution. The second is end-to-end 3D MLLMs (e.g., Chat-Scene, Inst3D-LMM, Video-3D LLM, LLaVA-3D), which directly map scene tokens and language to answers in a black-box manner.

**Limitations of Prior Work**: NS3D systems are interpretable and compositional but face two rigid constraints: (i) primitives like `filter(chair)` depend on fixed concept networks, failing to handle open-vocabulary terms like "cozy chair" or "messy desk"; (ii) training each primitive requires dense intermediate supervision, restricting them to synthetic data with template-generated instructions and shallow nesting (e.g., Sr3D). Conversely, 3D MLLMs handle free-form language but act as black boxes, making it impossible to diagnose whether failures stem from object recognition, spatial relations, or compositional logic.

**Key Challenge**: Interpretability and semantic flexibility appear mutually exclusive. The authors identify a decoupling opportunity: **symbolic programs encode the "syntax of reasoning" (decomposition and verification), while MLLMs possess "open-world semantic knowledge"**—these two capabilities can be learned separately.

**Goal**: (1) Distill the reasoning patterns (decomposition + step-by-step spatial verification) of NS programs into a 3D MLLM; (2) extend reasoning capabilities beyond the closed-vocabulary and shallow-nesting constraints of synthetic data to real-world instructions like ScanRefer/Multi3DRefer; (3) retain interpretable traces and modular substitutability.

**Key Insight**: Synthetic datasets like Sr3D naturally provide **complete intermediate supervision**—the inputs and outputs of every `filter` and the intermediate sets of every `relate` can be derived from ground-truth annotations. This "white-box supervision" can first inject reasoning templates into the MLLM, followed by outcome-supervised RL to extrapolate these templates to open concepts.

**Core Idea**: Serialize symbolic program execution traces into natural language CoT for SFT (teaching "how to think"), then use GRPO + soft spatial rewards for RL (generalizing templates to open-vocabulary and deep nesting). This allows an end-to-end MLLM to possess both the systematicity of NS3D and the flexibility of LLMs.

## Method

### Overall Architecture
APEIRIA is built on an 8B MLLM backbone. The input consists of a natural language instruction $q$ and a set of **object-centric** scene representations $\mathcal{O}$. The output is a CoT containing "plan + execution" tags followed by the final answer $A$ (grounding box / QA answer / caption).

For the scene side, object-centric representations are used (approx. 400 tokens, significantly smaller than the 10k–40k tokens in video-based methods). Mask3D first segments the scene into instances. For each instance, Uni3D extracts 3D geometric features while DINOv2 extracts 2D appearance features. Learnable positional encodings inject coordinates and sizes. Finally, the visual and spatial features of each object are interleaved with instruction tokens and fed into the LLM.

The training involves a **three-stage curriculum**: Stage 1 Perception Alignment → Stage 2 Symbolic Reasoning Injection (CoT-SFT) → Stage 3 Open-set and Complex Reasoning Generalization (CoT-RL). These stages increase in difficulty, building the model's ability to "see → think → adapt." Since planning and perception are decoupled, the plan can be replaced by GPT-4/Claude outputs at inference, or the `scene()` primitive can be replaced by stronger segmenters like SegDINO3D without retraining.

### Key Designs

1.  **Curriculum-based Reasoning Distillation**:
    *   Function: Decompose "3D perception → systematic reasoning → open generalization" into three non-overlapping training objectives.
    *   Mechanism: Stage 1 performs vision-language pre-training on ~193K object-level perception tasks (recognition, localization, captioning) to align 3D geometric features with the LLM embedding space. Stage 2 performs CoT-SFT on Level-1 (78K single-step filter from MMScan) and Level-2 (66K two-step relate from Sr3D) programs with the objective $\mathcal{L}_{\text{CoT-SFT}} = -\mathbb{E}\,[\log p_\theta(\text{CoT}, A \mid q, \mathcal{O})]$. Stage 3 applies GRPO reinforcement learning on real instructions from ScanRefer/Multi3DRefer.
    *   Design Motivation: Skipping Stage 2 results in an excessively large RL search space (ScanRefer Acc@0.25 drops from 58.4% to 48.2%). Relying only on Stage 2 limits the model to closed vocabularies and limited nesting depth (dropping Stage 3 leads to a 6.9% decrease). The three stages sequentially address "blindness," "inability to decompose," and "shallow decomposition."

2.  **Program-to-CoT Translation (White-box Distillation of Symbolic Programs)**:
    *   Function: Parse NS3D `scene/filter/relate/relate_triple` programs into natural language "plan + execution" traces to serve as SFT supervision.
    *   Mechanism: For each program, the AST is parsed into an execution sequence $\mathcal{S} = \{s_1, \ldots, s_n\}$. Each step $s_i$ is serialized into two segments: "plan" describes the sub-goal (e.g., "Find all objects of category 'vase'"), and "execution" explicitly lists input and output objects using **ID + coordinates + size** (e.g., `relate(filter(desk), filter(wall), left)` is expanded to list all desk IDs, then all wall IDs, and finally the IDs of desks satisfying the "left" relation). The final CoT concatenates all plans followed by all executions, forming a transparent trace. This trace is **spatially grounded**, using unique IDs to avoid ambiguity among identical object types.
    *   Design Motivation: Traditional NS3D uses fixed concept networks for primitives, a major barrier to open vocabularies. APEIRIA redefines primitives as LLM "executions" in natural language, removing limitations on vocabulary. Furthermore, unlike 3D-R1 which uses LLM prompting for CoT, traces derived from symbolic programs have verifiable ground-truth for every step, eliminating CoT hallucinations.

3.  **GRPO + Soft Grounding Reward (Outcome-supervised Open Generalization)**:
    *   Function: Extrapolate reasoning templates to open-vocabulary concepts and deep nesting on real-world data lacking intermediate supervision.
    *   Mechanism: GRPO is used to optimize the policy $\pi_\theta$ by sampling $N$ responses per instruction and normalizing advantage $A_i = (r_i - \text{mean})/\text{std}$ within the group. The reward is a sum of: (a) **Soft Grounding Reward** $R_{\text{grounding}} = e^{-\alpha \|\bm{x}_{\text{pred}} - \bm{x}_{\text{gt}}\|_2} + e^{-\alpha \|(\bm{s}_{\text{pred}} - \bm{s}_{\text{gt}})/\bm{s}_{\text{gt}}\|_1}$ ($\alpha = 2$), which uses exponential decay similarity for position and size to provide dense gradients even when boxes do not overlap, solving the sparsity problem of IoU; (b) **Format Reward**: Responses must contain valid plan/thinking tags and non-degenerate length, or the reward is zero, preventing the model from skipping reasoning.
    *   Design Motivation: Ablations show IoU rewards underperform Soft rewards by 0.5–0.7% due to sparse feedback. Removing the Format Reward leads to "structure collapse," where the model bypasses reasoning to output answers immediately. Together, these rewards maintain spatial precision and interpretable CoT structure.

### Loss & Training
Stages 1 and 2 utilize standard next-token language modeling loss. Stage 3 uses the GRPO clipped surrogate loss. The 8B backbone is fine-tuned using LoRA and AdamW/Muon, with all stages sharing the evolution of the same LoRA weights. Total CoT supervision consists of 144K verified samples in Stage 2, while Stage 3 runs RL directly on downstream instruction-answer pairs.

## Key Experimental Results

### Main Results

Main results on ScanRefer & Multi3DRefer (3D spatial grounding):

| Method | Type | ScanRefer Acc@0.25 | ScanRefer Acc@0.5 | M3DRef F1@0.25 | M3DRef F1@0.5 |
|------|------|--------------------|-------------------|----------------|---------------|
| NS3D (Hsu 2023) | NS3D | 22.4 | – | – | – |
| LARC (Feng 2024) | NS3D | 32.9 | – | – | – |
| LaSP (Mi 2025) | NS3D | 49.2 | – | – | – |
| Chat-Scene | 3D MLLM | 55.5 | 50.2 | 57.1 | 52.4 |
| Inst3D-LMM | 3D MLLM | 57.8 | 51.6 | 58.3 | 53.5 |
| Video-3D LLM | 3D MLLM | 58.1 | 51.7 | 58.0 | 52.7 |
| **APEIRIA** | 3D MLLM | **58.4** | 51.2 | **59.2** | **53.8** |
| **APEIRIA†** (+ SegDINO3D) | 3D MLLM | **60.5** | **53.2** | **60.9** | **55.2** |

Cross-task generalization (same curriculum, swapping outcome reward for EM/CIDEr): Scan2Cap C@0.25 = 90.6 (Prev. SOTA LEGO 84.7); SQA3D EM = 58.6 (matching Prev. SOTA Video-3D LLM).

Zero-shot open concepts (trained only on Sr3D in Stage 2, tested on Nr3D): APEIRIA achieves 36.5%, surpassing **fully-supervised** NS3D (33.9%), validating the removal of the vocabulary bottleneck.

### Ablation Study

| Configuration | ScanRefer Acc@0.25 | M3DRef F1@0.25 | Note |
|------|--------------------|----------------|------|
| APEIRIA full | 58.4 | 59.2 | Full 3 stages |
| w/o Stage 3 (CoT-RL → Direct SFT) | 51.5 | 55.3 | Drop 6.9/3.9; RL is necessary for real-world extrapolation |
| w/o Stage 2 (Direct to CoT-RL) | 48.2 | 36.7 | Drop 10.2/22.5; RL fails without warm start |
| w/o Format Reward | 55.7 | 57.1 | Structure collapse occurs |
| w/o Soft Grounding (Sparse IoU) | 57.7 | 58.7 | Low exploration efficiency |
| w/o Thinking (Direct answer) | 56.8 | 58.2 | Explicit CoT contributes ~1–2% |

RL gains by reasoning complexity (ScanRefer Acc@0.5): For $\leq 4$ steps, SFT-only (47.2) > CoT-RL (45.4). For $= 5$ steps, RL Gain +1.5. For $\geq 6$ steps, RL Gain +2.7.

### Key Findings
- **All three stages are indispensable**: Stage 2 serves as the "foundation" (−22.5 F1 if removed); Stage 3 is the "roof" (−6.9 Acc if removed). Their order cannot be reversed.
- **RL gains correlate with reasoning depth**: RL introduces noise for $\leq 4$ steps but improves accuracy by +2.7% for $\geq 6$ steps, confirming that RL completes long chains where Stage 2 supervision is unavailable.
- **Bottlenecks lie in perception, not planning**: Replacing the planner with Claude 4.5 Opus yields only +0.2%, while replacing the `scene()` primitive with SegDINO3D adds +2.0%, leaving only a 0.9% gap to the oracle GT ceiling (61.3). Modularity allows "zero-cost" benefits from future 3D segmenters.
- **Emergent primitives**: Post CoT-RL, the model spontaneously invents logical primitives like `intersection` or `union` not taught in Stage 2, and operates correctly on open-vocabulary filters like `beige chair`.

## Highlights & Insights
- **Decoupling "reasoning syntax" from "conceptual knowledge"**: Traditional distillation transfers "what the teacher knows." This approach transfers "how the teacher thinks," leaving semantic knowledge to the LLM's pre-training.
- **White-box CoT from synthetic data**: Synthetic datasets with program generators (CLEVR, Sr3D, etc.) provide ground-truth for every CoT step, posing much lower hallucination risks than LLM-as-annotator methods.
- **Soft Grounding Reward for RL**: Replacing boolean IoU signals with exponential similarity for position and size enables gradients for early, non-overlapping predictions. This is applicable to any 2D/3D detection RL task.
- **Hot-swappable modularity**: The explicit decoupling of planning and perception allows the system to upgrade alongside the 3D perception community—a "compounding interest" unavailable to black-box MLLMs.

## Limitations & Future Work
- **Perceptual ceiling**: Gains from SegDINO3D suggest that LLM reasoning is nearing saturation while 3D segmentation remains a bottleneck.
- **Dependency on synthetic program ecology**: Extending this to domains without program-parsed synthetic data (e.g., ego-centric video) remains an open question.
- **Format Reward**: This is a palliative measure that ensures structure but does not strictly guarantee the correctness of CoT content; a step-level process reward model (PRM) might be required in the future.
- **Scene scope**: Evaluation is limited to static indoor scenes (ScanNet-based); dynamic, outdoor, and multi-view settings are not yet validated.

## Related Work & Insights
- **vs NS3D / LARC**: These methods implement primitives via concept networks ($f_{\text{chair}}$), requiring dense supervision and suffering from closed vocabularies. APEIRIA executes primitives via natural language, inheritance systematic decomposition while gaining open-vocabulary flexibility.
- **vs 3D-R1 / Scene-R1**: 3D-R1 prompts LLMs for CoT, leading to hallucinations and vague object references. Scene-R1 attempts RL without trace supervision, leading to instability. APEIRIA uses verifiable traces from symbolic programs for SFT warm-start before RL.
- **vs Mainstream 3D MLLMs**: While others map instruction to answer in a black box, APEIRIA produces "plan + execution" traces, ensuring interpretability and modularity while maintaining a consistent lead on benchmarks.

## Rating
- Novelty: ⭐⭐⭐⭐ Distilling NS programs as CoT is a clean hybrid; RL extrapolation for 3D grounding is a genuine increment.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 4 benchmarks and multiple task types, with extensive ablation on RL gains and modular upper bounds.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and intuitive diagrams; however, the method section's mathematical formatting is occasionally inconsistent.
- Value: ⭐⭐⭐⭐⭐ Addresses the closed-vocabulary issue of NS3D and the opacity of 3D MLLMs; its modular design serves as a strong baseline for 3D embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification](../../AAAI2026/3d_vision/stmi_segmentation-guided_token_modulation_with_cross-modal_hypergraph_interactio.md)
- [\[CVPR 2026\] Foundry: Distilling 3D Foundation Models for the Edge](../../CVPR2026/3d_vision/foundry_distilling_3d_foundation_models_for_the_edge.md)
- [\[AAAI 2026\] Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection](../../AAAI2026/3d_vision/multi-modal_assistance_for_unsupervised_domain_adaptation_on_point_cloud_3d_obje.md)
- [\[CVPR 2026\] MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation](../../CVPR2026/3d_vision/msgnav_unleashing_the_power_of_multi-modal_3d_scene_graph_for_zero-shot_embodied.md)
- [\[CVPR 2026\] Glove2Hand: Synthesizing Natural Hand-Object Interaction from Multi-Modal Sensing Gloves](../../CVPR2026/3d_vision/glove2hand_synthesizing_natural_hand-object_interaction_from_multi-modal_sensing.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2026\] Foundry: Distilling 3D Foundation Models for the Edge](../../CVPR2026/3d_vision/foundry_distilling_3d_foundation_models_for_the_edge.md)
- [\[AAAI 2026\] STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification](../../AAAI2026/3d_vision/stmi_segmentation-guided_token_modulation_with_cross-modal_hypergraph_interactio.md)
- [\[CVPR 2025\] Neuro-3D: Towards 3D Visual Decoding from EEG Signals](../../CVPR2025/3d_vision/neuro-3d_towards_3d_visual_decoding_from_eeg_signals.md)
- [\[AAAI 2026\] Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection](../../AAAI2026/3d_vision/multi-modal_assistance_for_unsupervised_domain_adaptation_on_point_cloud_3d_obje.md)
- [\[ICLR 2026\] pySpatial: Generating 3D Visual Programs for Zero-Shot Spatial Reasoning](../../ICLR2026/3d_vision/pyspatial_generating_3d_visual_programs_for_zero-shot_spatial_reasoning.md)

</div>

<!-- RELATED:END -->
