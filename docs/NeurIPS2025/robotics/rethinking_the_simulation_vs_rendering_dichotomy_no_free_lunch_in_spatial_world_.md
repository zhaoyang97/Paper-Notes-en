---
title: >-
  [Paper Note] Rethinking the Simulation vs. Rendering Dichotomy: No Free Lunch in Spatial World Modelling
description: >-
  [NeurIPS 2025][Robotics][Spatial world models] From a cognitive neuroscience perspective, this paper challenges the prevailing view that simulation and rendering are separable processes: it argues that spatial reasoning relies on fine-grained perceptual representations rather than coarse abstractions, and concludes that AI spatial world models likewise require rich perceptual detail — there is no free lunch in spatial modelling.
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Spatial world models"
  - "mental rotation"
  - "aphantasia"
  - "higher-order theory of consciousness"
  - "embodied AI"
date: 2026-05-08
content_hash: 28beb6cb50082bba
---

# Rethinking the Simulation vs. Rendering Dichotomy: No Free Lunch in Spatial World Modelling

**Conference**: NeurIPS 2025
**arXiv**: [2510.20835](https://arxiv.org/abs/2510.20835)  
**Code**: Not available  
**Area**: Robotics / Cognitive Science
**Keywords**: Spatial world models, mental rotation, aphantasia, higher-order theory of consciousness, embodied AI

## TL;DR

From a cognitive neuroscience perspective, this paper challenges the prevailing view that simulation and rendering are separable processes: it argues that spatial reasoning relies on fine-grained perceptual representations rather than coarse abstractions, and concludes that AI spatial world models likewise require rich perceptual detail — there is no free lunch in spatial modelling.

## Background & Motivation

Spatial world models — internal representations that support flexible reasoning over spatial relationships — constitute core infrastructure for AI in the physical world. Yet there exists a fundamental conceptual confusion regarding what kind of internal representations spatial world models actually require.

**The simulation vs. rendering dichotomy is widely accepted but potentially misleading**: Balaban and Ullman (2025) proposed that physical simulation (e.g., mental rotation, mechanical reasoning) and graphical rendering (the process that produces conscious visual experience) are two separable processes. Under this view, spatial reasoning requires only coarse-grained "spatial imagery" and no fine-grained perceptual content — offering AI systems a convenient shortcut.

**Aphantasia is misappropriated as evidence for separability**: Individuals with aphantasia cannot voluntarily generate visual mental images, yet perform normally on tasks such as mental rotation. This has been interpreted as evidence that "spatial simulation does not require visual rendering." The paper argues, however, that such an interpretation rests on a **linear reading** of cortical functional specialisation (dorsal stream → spatial/action; ventral stream → object/consciousness), overlooking the extensive cross-talk between the two pathways.

**Current AI language models remain weak at spatial reasoning**: Despite impressive performance in perception and high-level reasoning, multimodal large language models still struggle on tasks requiring model-based spatial reasoning (mental rotation, perspective-taking, mechanical reasoning), suggesting an absence of structured spatial representations and dynamic transformation mechanisms.

The paper's central thesis is that **simulation and rendering share a common representational substrate** — the fine-grained representational structure described by higher-order theories of consciousness, which captures instantiation relations among perceived objects. Aphantasia is not evidence of simulation–rendering dissociation, but rather reflects a failure of the downstream *decoding* step: the higher-order representations themselves remain intact, but cannot be projected into conscious experience.

## Method

### Overall Architecture

This is a **theory/position paper** that proposes no concrete algorithm. Instead, it advances a three-step argument to derive implications for the design of AI spatial world models: (1) critiquing the theoretical foundations of the simulation vs. rendering dichotomy; (2) introducing neuroscientific evidence to reinterpret aphantasia; and (3) arguing that AI systems likewise require fine-grained perceptual representations.

### Key Designs

1. **Critique of the spatial imagery framework**:

    - The concept of "spatial imagery" is itself underspecified: it is described as an abstract, modality-neutral representation of spatial features, yet when applied concretely (e.g., "imagine grasping and rotating a shape"), it implicitly invokes embodied sensory modalities.
    - If spatial imagery is denied any conscious experiential character, the theory collapses into an account of "unconscious mental imagery" — yet "imagery" typically presupposes fine-grained perceptual content (involving early visual cortex). If aphantasic individuals lack the ability to reconstruct such content, it is unclear in what sense non-modally defined "spatial imagery" qualifies as "imagery" at all.
    - The paper ultimately advocates a more pragmatic position: aphantasic individuals use spatial representations to solve tasks, but lack the corresponding visual experience — whether these representations count as "imagery" may be fundamentally undecidable.

2. **Rebuttal of the linear functional specialisation account**:

    - The traditional view characterises the dorsal stream as a "zombie pathway" — processing spatial information without contributing to conscious vision. However, emerging evidence suggests otherwise:
    - The posterior parietal cortex (PPC) can decode object identity from rapidly presented visual stimuli (Bellet et al., 2022), demonstrating that dorsal regions actively encode perceptual information.
    - The dorsolateral prefrontal cortex (DLPFC) — traditionally associated with action planning — has been found to selectively correlate with conscious visual information.
    - Lesions to frontal and parietal cortex impair the integration and maintenance of visual content in consciousness.
    - This evidence supports the **Higher-Order Theory (HOT) of consciousness**: fronto-parietal higher-order representations encode the relational structure among perceptual states; simulation and conscious experience share a common representational substrate, differing only in whether the representations are "validated and decoded into conscious experience" by downstream systems.

3. **Implications for AI — No Free Lunch**:

    - **Language models lack genuine spatial competence**: MLLMs may acquire some capacity for spatial description through convergent statistical abstraction, but lack a vehicle for encoding rich perceptual representations and thus cannot truly understand spatial structure.
    - **Implicit world models outperform explicit physics engines**: Methods leveraging pretrained visual representations — such as VIP and R3M — substantially outperform approaches relying solely on physics simulators (MuJoCo, Isaac Gym) in manipulation generalisation, because visual encoders capture fine-grained spatial and semantic priors.
    - **Video models for action imagination**: Analogous to the human ability to "mentally simulate" an action before executing it, video diffusion models can generate action rollout sequences to score and optimise action plans. Works such as Veo 3, Genex, and Genie-3 demonstrate the potential of video generation as implicit world models.

### Loss & Training

Not applicable — this is a theoretical paper involving no model training or experiments.

## Key Experimental Results

### Main Results

The paper presents no quantitative experiments, but offers key comparative evidence through a literature review:

| Dimension | Pure Physics Engine | Visual Pretraining + Control | Notes |
|---|---|---|---|
| Sim-to-real transfer | Brittle | Robust | VIP/R3M-style representations substantially improve real-world generalisation |
| Long-horizon planning | Limited by model–reality gap | Enhanced via rich visual priors | Visual foundation models as implicit world models |
| Open-scene generalisation | Poor | Strong | Pretrained visual encoders provide structured cross-scene priors |

### Ablation Study

| Method Type | Spatial Reasoning Capability | Notes |
|---|---|---|
| MLLM (language/visual alignment only) | Weak | Struggles with mental rotation, perspective-taking, and mechanical reasoning |
| Explicit physics engine (MuJoCo, etc.) | Precise but brittle | Lacks counterfactual reasoning and conceptual grounding |
| Visual pretrained representations | Implicit but generalisable | Encodes fine-grained spatial structure and semantic priors |

### Key Findings

- Lesion studies of aphantasia (Kutsche et al., 2025) found that all 12 cases of acquired aphantasia involved the left fusiform connectivity region (directly linked to visual mental imagery), while the prefrontal cortex remained intact — indicating that the higher-order monitoring mechanism (the "discriminator" that assesses signal reliability) was undamaged; the impairment lies in the downstream decoding step.
- This is consistent with the Perceptual Reality Monitoring (PRM) theory: the system evaluates which perceptual signals are "real enough" to be experienced — analogous to a GAN discriminator — and then generates pointers to the storage locations of detailed visual content. Aphantasia reflects a failure of pointer generation/decoding, not an absence of the higher-order spatial representations themselves.

## Highlights & Insights

- **Distinctive interdisciplinary perspective**: The paper directly connects debates in cognitive neuroscience on consciousness (HOT theory, dorsal–ventral pathway interactions) to AI world model design, offering a genuinely novel vantage point.
- **The "no free lunch" argument is compelling**: If human spatial reasoning depends on fine-grained encodings, AI systems should not expect coarse-grained abstractions to suffice — an important caution for lines of work pursuing "pure language reasoning."
- **Theoretical grounding for video generative models**: From a cognitive science perspective, the paper substantiates "chain-of-frames" rollout as a means of spatiotemporal reasoning, drawing an analogy to human mental simulation.

## Limitations & Future Work

- As a purely theoretical paper, it lacks quantitative experiments to validate its central claims.
- The inferential leap from human cognitive mechanisms to AI design principles is substantial — human cognition need not be the sole valid reference for AI.
- The paper offers little concrete guidance on how "fine-grained perceptual representations" should be implemented (architectural choices, training paradigms).
- The Higher-Order Theory of consciousness remains contested within cognitive science; arguments built upon it inherit this theoretical uncertainty.
- Computational efficiency is not addressed — maintaining fine-grained perceptual representations may carry significant computational cost.

## Related Work & Insights

- The paper engages directly with and critiques the "physics vs. graphics" dichotomy of Balaban & Ullman (2025).
- It resonates with Huh et al. (2024)'s "Platonic Representation Hypothesis" — that latent spaces across modalities converge — while further arguing that such convergence in the spatial domain requires a perceptually fine-grained vehicle.
- VGGT (anchoring large visual backbones to 3D reconstruction objectives) and VIP (vision-based implicit reward) are cited as positive examples from the AI literature.
- Key implication: rather than attempting to bypass the perceptual layer through purely linguistic description, research should develop architectures capable of maintaining structured perceptual representations as a foundational step toward spatial world models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The interdisciplinary argument re-examining AI world model design through cognitive neuroscience is distinctive
- **Experimental Thoroughness**: ⭐⭐ — A purely theoretical paper with no quantitative experiments; argumentation relies primarily on literature review
- **Writing Quality**: ⭐⭐⭐⭐ — Philosophical and neuroscientific reasoning is rigorous, though accessibility for AI-focused readers is limited
- **Value**: ⭐⭐⭐⭐ — Provides an important theoretical anchor and directional guidance for research on AI spatial reasoning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Spatial Understanding from Videos: Structured Prompts Meet Simulation Data](spatial_understanding_from_videos_structured_prompts_meet_simulation_data.md)
- [\[NeurIPS 2025\] MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)
- [\[NeurIPS 2025\] Learning Spatial-Aware Manipulation Ordering](learning_spatial-aware_manipulation_ordering.md)
- [\[NeurIPS 2025\] LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents](labutopia_high-fidelity_simulation_and_hierarchical_benchmark_for_scientific_emb.md)
- [\[CVPR 2026\] Rethinking Visual Rearrangement from A Diffusion Perspective](../../CVPR2026/robotics/rethinking_visual_rearrangement_from_a_diffusion_perspective.md)

</div>

<!-- RELATED:END -->
