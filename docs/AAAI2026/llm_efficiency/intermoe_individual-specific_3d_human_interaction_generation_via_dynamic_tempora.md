---
title: >-
  [Paper Note] InterMoE: Individual-Specific 3D Human Interaction Generation via Dynamic Temporal-Selective MoE
description: >-
  [AAAI 2026][LLM Efficiency][human interaction generation] This paper proposes InterMoE, a Dynamic Temporal-Selective MoE architecture for text-driven two-person 3D interaction motion generation that addresses individual…
tags:
  - "AAAI 2026"
  - "LLM Efficiency"
  - "human interaction generation"
  - "MoE"
  - "motion generation"
  - "diffusion model"
  - "3D motion synthesis"
date: 2026-05-08
content_hash: b30242787fdb4eb9
---

# InterMoE: Individual-Specific 3D Human Interaction Generation via Dynamic Temporal-Selective MoE

**Conference**: AAAI 2026
**arXiv**: [2511.13488](https://arxiv.org/abs/2511.13488)
**Code**: [GitHub](https://github.com/Lighten001/InterMoE)
**Area**: LLM Efficiency
**Keywords**: human interaction generation, MoE, motion generation, diffusion model, 3D motion synthesis

## TL;DR
This paper proposes InterMoE, a Dynamic Temporal-Selective MoE architecture for text-driven two-person 3D interaction motion generation that addresses individual identity preservation and semantic fidelity. A Synergistic Router fuses semantic and kinematic features to guide routing, while Dynamic Temporal Selection enables each expert to adaptively select key temporal frames. The method achieves a 9% FID reduction on InterHuman and 22% on InterX.

## Background & Motivation

**Background**: Text-driven two-person 3D interaction motion generation is a core task in virtual reality, game development, and related domains. Existing methods (InterGen, InterMask, TIMotion) have made notable progress, but exhibit clear deficiencies in individual identity preservation and semantic alignment.

**Limitations of Prior Work**: (a) **Cross-attention fusion causes individual homogenization**—methods such as InterGen fuse dual-person features via cross-attention and then process them through a shared FFN, suppressing individual feature differences and causing the two persons' motions to converge; (b) **Feature concatenation causes identity confusion**—methods such as TIMotion directly concatenate dual-person features for joint generation, lacking explicit identity constraints, leading to role swaps or positional errors.

**Key Challenge**: Individual feature independence and two-person interaction dependency must be modeled simultaneously—two objectives that are inherently in tension within a unified network.

**Key Insight**: MoE is a natural fit for this problem: different experts can specialize in the motion patterns of different individuals, with the routing mechanism enabling differentiated allocation.

**Core Idea**: (a) A Synergistic Router that fuses textual semantic and kinematic features to jointly guide routing decisions; (b) Dynamic Temporal Selection that allows each expert to dynamically select key temporal frames (rather than a fixed Top-K), handling non-uniform temporal importance.

## Method

### Overall Architecture
The input is a text description; the output is a two-person 3D motion sequence $\mathbf{m}_i \in \mathbb{R}^{T \times J \times d}$. The pipeline consists of three components:
1. **Causal-Skeletal VAE**: Skeletal graph convolution captures intra-joint dependencies; causal convolution models temporal dynamics to encode single-person motion.
2. **Cooperative MoE Denoiser**: Two weight-sharing diffusion denoisers process each person separately, interacting via Self-Attention (intra-individual) + Cross-Attention (inter-individual) + MoE Block.
3. **InterMoE Block**: Synergistic Router + Dynamic Temporal Selection.

### Key Designs

1. **Synergistic Router**:

    - Function: Fuses two routing signals—a motion router computing routing logits from each person's unique kinematic features, and a text router computing logits from semantic features; the two are combined via weighted fusion.
    - Formula: $\mathbf{R}^{comb}_{e,s,i} = \alpha \mathbf{R}^{motion}_{e,s,i} + (1-\alpha) \mathbf{R}^{text}_e$
    - Key Innovation: Adopts batch-level routing—flattening temporal features of all samples in the batch into a global token pool, enabling the router to perceive heterogeneity across different noise levels.
    - Design Motivation: Routing based solely on motion features cannot guarantee semantic alignment; routing based solely on text cannot distinguish individual motion characteristics.

2. **Dynamic Temporal Selection**:

    - Function: Allows each expert to dynamically determine how many temporal frames to process (rather than a fixed Top-K).
    - Core Mechanism: Each expert has a learnable bias $b_e \in (-1, 0)$; a selection gate is determined via sigmoid plus bias: $\mathbf{M}_{e,s} = \text{sigmoid}(\mathbf{R}^{comb}_{e,s}) + b_e$; a frame is selected when $\mathbf{M}_{e,s} > 0$.
    - Bias Adaptive Update: $b_e$ is adjusted based on the discrepancy between the actual and expected selection counts, stabilizing after training convergence.
    - Design Motivation: In interaction motion, temporal frames vary in importance—key frames (e.g., a punch or dodge) require more expert attention, while transition frames do not. Fixed-capacity Token-Choice and Expert-Choice strategies cannot handle this non-uniformity.

3. **Causal-Skeletal VAE**:

    - Skeletal graph convolution extracts spatial dependencies among joints; causal convolution enforces temporal causality.
    - Lightweight yet effective motion representation.

## Key Experimental Results

### Main Results

| Dataset | Method | FID↓ | R-Precision Top-1↑ | MM-Dist↓ |
|--------|------|------|------------|---------|
| InterHuman | InterGen | 5.149 | 0.489 | 3.785 |
| InterHuman | TIMotion | 5.157 | 0.496 | 3.772 |
| InterHuman | **InterMoE** | **4.677** | **0.512** | **3.762** |
| InterX | InterGen | 0.469 | - | - |
| InterX | **InterMoE** | **0.297** | - | - |

FID reduction: InterHuman −9% (5.149 → 4.677), InterX −22% (0.469 → 0.297). R-Precision Top-1 improves from 0.489 to 0.512. MultiModality is slightly lower than some methods, which the authors attribute to prioritizing semantic fidelity.

### Ablation Study (InterHuman)

| Configuration | FID↓ | R-Precision Top-1↑ | MM-Dist↓ |
|------|------|------------|---------|
| Baseline (InterGen + CS-VAE) | 5.251 | 0.489 | 3.771 |
| w/o Motion & Text Router | 4.782 | 0.503 | 3.766 |
| w/o Batch-level Routing | 6.036 | 0.492 | 3.774 |
| w/o Dynamic Selection | 6.242 | 0.498 | 3.772 |
| w/o Temporal-Selective | 5.195 | 0.505 | |
| **Full InterMoE** | **4.677** | **0.512** | |

### Key Findings
- **Batch-level routing and Dynamic Selection are both indispensable**: Removing either leads to significant FID degradation (6.036 and 6.242), demonstrating that global perspective and dynamic capacity are both critical for interaction generation.
- **Synergistic routing outperforms single-signal routing**: Using only motion or text routing is inferior to their fusion; combining both further reduces FID from 4.782 to 4.677.
- **Qualitative comparisons clearly demonstrate identity preservation advantages**: In a fencing scenario, the model accurately distinguishes attack and defense hand postures and forward/backward movement; in a tug-of-war scenario, it precisely synthesizes rope-gripping poses and backward-leaning motions; in a 10-second Taekwondo scenario, it maintains circular movement trajectories—competing methods exhibit identity confusion or semantic drift in these cases.
- **Causal-Skeletal VAE contributes independently**: Even without MoE (Baseline row), incorporating the Causal-Skeletal VAE improves over the original InterGen.
- **Competitive on single-person motion generation**: Validates generalizability beyond interaction scenarios.

## Highlights & Insights
- Applying the MoE architecture to address "individual identity preservation" in two-person interaction is a natural and elegant choice—different experts can automatically specialize in the motion patterns of different individuals or different action phases. The dual-signal fusion in the Synergistic Router ensures simultaneous alignment of semantics and kinematics, avoiding the single-objective pitfall of "semantically correct but kinematically unnatural" or "smooth motion but semantically drifted" results.
- The elastic capacity mechanism of Dynamic Temporal Selection via learnable biases is practically effective. Unlike the rigid Top-K selection, the adaptive bias update enables the system to automatically discover optimal capacity allocation during training. This design is also transferable to other tasks with temporally non-uniform importance, such as video generation.
- The batch-level routing strategy is noteworthy—enabling the router to perceive sample heterogeneity across different noise levels within a batch is a key design consideration for diffusion-based MoE models.

## Limitations & Future Work
- Validation is limited to two-person interaction; extensibility to multi-person (3+) scenarios remains unknown—the combinatorial complexity of multi-person interaction grows rapidly.
- The fusion weight $\alpha=0.5$ in the Synergistic Router is fixed; a learnable adaptive weight could allow the model to automatically calibrate the relative importance of semantic and kinematic signals by task.
- Evaluation metrics (FID, R-Precision) may not fully capture the quality of individual identity preservation—more targeted identity consistency metrics are needed (e.g., measuring motion style consistency of the same character across the sequence).
- Training is conducted on two RTX 3090 GPUs with reasonable computational cost, but batch-level routing may face memory bottlenecks at very large batch sizes.

## Related Work & Insights
- **vs InterGen**: Uses cross-attention for interaction but subsequent unified FFN processing causes homogenization. InterMoE replaces the FFN with MoE, where different experts handle different patterns, avoiding homogenization.
- **vs TIMotion**: Concatenates dual-person features for joint generation, lacking identity constraints. InterMoE's two Cooperative Denoisers process each individual separately, naturally preserving identity.
- **vs EC-DiT/DiT-MoE**: General-purpose diffusion MoE methods. InterMoE specifically designs Dynamic Temporal Selection for the temporally non-uniform nature of interaction motion.
- **vs ComMDM**: Uses a small bridging network between two single-person diffusion models, but performance is limited by constrained interaction datasets. InterMoE's Cooperative Denoiser design models two-person dependency more deeply.
- **vs in2IN**: Introduces individual action descriptions as additional conditions, but still employs a unified FFN. InterMoE's replacement of the FFN with MoE ensures differentiated processing at the architectural level.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying MoE to interaction motion generation is a novel attempt; the Synergistic Router and Dynamic Temporal Selection are purpose-built designs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, quantitative and qualitative comparisons, detailed ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ The motivation for each component is clearly articulated.
- Value: ⭐⭐⭐⭐ New state of the art in two-person interaction motion generation, with substantive FID improvements of 9–22%.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling](../../ICLR2026/llm_efficiency/iterresearch_rethinking_long-horizon_agents_with_interaction_scaling.md)
- [\[ACL 2026\] Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](../../ACL2026/llm_efficiency/planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)
- [\[ACL 2026\] HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns](../../ACL2026/llm_efficiency/humanllm_benchmarking_and_improving_llm_anthropomorphism_via_human_cognitive_pat.md)
- [\[CVPR 2026\] GeoCodeBench: Benchmarking PhD-Level Coding in 3D Geometric Computer Vision](../../CVPR2026/llm_efficiency/benchmarking_phd-level_coding_in_3d_geometric_computer_vision.md)
- [\[ICLR 2026\] DND: Boosting Large Language Models with Dynamic Nested Depth](../../ICLR2026/llm_efficiency/dnd_boosting_large_language_models_with_dynamic_nested_depth.md)

</div>

<!-- RELATED:END -->
