---
title: >-
  [Paper Note] Planning with Diffusion Models for Target-Oriented Dialogue Systems
description: >-
  [ACL 2025][Image Generation][Dialogue Planning] DiffTOD models dialogue planning as a trajectory generation problem, leveraging a masked diffusion language model to achieve non-sequential dialogue planning. It designs three guidance mechanisms (word-level/semantic-level/search-based) to flexibly control the dialogue toward the target, significantly outperforming baselines in negotiation, recommendation, and chitchat scenarios.
tags:
  - "ACL 2025"
  - "Image Generation"
  - "Dialogue Planning"
  - "Diffusion Models"
  - "Target-Oriented Dialogue"
  - "Non-Sequential Planning"
  - "Trajectory Generation"
date: 2026-05-08
content_hash: 91b27d590a7fa436
---

# Planning with Diffusion Models for Target-Oriented Dialogue Systems

**Conference**: ACL 2025  
**arXiv**: [2504.16858](https://arxiv.org/abs/2504.16858)  
**Code**: [https://github.com/ninglab/DiffTOD](https://github.com/ninglab/DiffTOD)  
**Area**: Image Generation  
**Keywords**: Dialogue Planning, Diffusion Models, Target-Oriented Dialogue, Non-Sequential Planning, Trajectory Generation  

## TL;DR
DiffTOD models dialogue planning as a trajectory generation problem, leveraging a masked diffusion language model to achieve non-sequential dialogue planning. It designs three guidance mechanisms (word-level/semantic-level/search-based) to flexibly control the dialogue toward the target, significantly outperforming baselines in negotiation, recommendation, and chitchat scenarios.

## Background & Motivation
**Background**: Target-Oriented Dialogue (TOD) systems need to strategically guide dialogues toward specific goals (e.g., reaching a deal, recommending items).

**Limitations of Prior Work**:
   - Existing dialogue planning methods (LLM prompt/RL policy) perform step-by-step sequential generation and can only plan the next step based on past actions.
   - Sequential planning leads to compounding errors and short-sighted decisions, failing to look ahead at the future trajectory of the dialogue.
   - LLMs are trained to passively follow instructions and lack the capability to proactively guide dialogues.

**Key Challenge**: Sequential planning cannot globally optimize dialogue strategies and easily falls into local optima (e.g., insisting on not lowering the price during negotiations, leading to a breakdown).

**Goal**: Design a non-sequential dialogue planning method that optimizes action strategies by simultaneously considering both the past and the future.

**Key Insight**: The mathematical connection between dialogue trajectory generation and the denoising process of diffusion models—both are progressive infilling from incomplete to complete.

**Core Idea**: Generate the entire dialogue trajectory simultaneously (instead of step-by-step) using a diffusion model, combined with conditional guidance to ensure the trajectory achieves the target.

## Method

### Overall Architecture
Model TOD as a conversational MDP $\rightarrow$ translate dialogue planning into a trajectory generation problem $\rightarrow$ estimate trajectory likelihood using a diffusion language model $\rightarrow$ optimize action strategies via conditional guidance $\rightarrow$ pass the generated plan to the LLM to execute dialogue. The planning and generation stages are decoupled.

### Key Designs
1. **Trajectory Modeling with Diffusion Models (Trajectory Modeling)**:

    - **Function**: Deconstructs the trajectory likelihood of dialogue planning into a form equivalent to the diffusion denoising process.
    - **Mechanism**: Trajectory generation $p_\theta(\tau_{0:T}) = p(\tau^N) \prod_{n=1}^N p_\theta(\tau^{n-1}|\tau^n)$ is mathematically equivalent to diffusion denoising. A masked diffusion language model (MDLM) is fine-tuned on dialogue history to progressively recover the complete trajectory from an incomplete one.
    - **Design Motivation**: The non-sequential generation capability of diffusion models is naturally suited for simultaneously considering the past and future, avoiding the short-sighted issues of sequential planning.

2. **Three Guidance Mechanisms (Guidance Mechanisms)**:

    - **Word-Level Guidance**: Fixes target keywords at specific positions on the trajectory, and the diffusion model completes the dialogue around the keywords. Suitable for keyword-guided dialogues.
    - **Semantic-Level Guidance**: Uses semantic descriptions (e.g., "the system recommended the target item") as conditions, combined with MBR decoding to select the best one from multiple paraphrased versions. Suitable for semantic-level goals.
    - **Search-Based Guidance**: Constructs a conversational search tree, generates different actions at tree nodes using word-level/semantic-level guidance, and uses a search algorithm (MCTS) to select the path that maximizes cumulative reward. Suitable for complex negotiations requiring long-term strategies.
    - The three types of guidance can be used individually or in combination, allowing flexible switching of targets during testing without retraining.

3. **Conditional Trajectory Generation (Conditional Trajectory Generation)**:

    - Factorizes the likelihood into $p_\theta(\tau|\mathcal{O}=1) \propto p_\theta(\tau) \cdot p_\theta(\mathcal{O}=1|\tau)$, where the unconditional part is generated by the diffusion model and the conditional part is achieved through the guidance mechanism.
    - Implemented as trajectory inpainting: fixes known target states/actions and lets the diffusion model fill in the rest.

### Loss & Training
Fine-tunes the masked diffusion language model (MDLM) on dialogue history data. The three guidance mechanisms are applied during inference, requiring no retraining.

## Key Experimental Results

### Main Results

| Setting | Metric | DiffTOD | Best Baseline | Gain |
|------|------|---------|---------|------|
| CraigslistBargain (Buyer) | SR | 0.901 | 0.798 (Claude-3.5) | +12.9% |
| CraigslistBargain (Seller) | SR | 0.793 | 0.689 (ProCoT) | +15.1% |
| TopDial (Recommendation) | SR | 0.663 | 0.620 (GPT-4o) | +6.9% |
| PersonaChat (Keyword) | KCR | 0.767 | 0.706 (GPT-4o) | +8.6% |

### Ablation Study

| Configuration | Description |
|------|------|
| w/o Non-sequential Planning | SR drops significantly, proving looking-ahead capability is crucial |
| w/o Search-based | Negotiation performance drops to the level of ProCoT |
| w/o Semantic-level MBR | Recommendation success rate drops |

### Key Findings
- The advantage is most pronounced in negotiation scenarios requiring long-term strategies—the performance gap continues to widen in the later stages of multi-turn negotiations.
- Search-based guidance shows significant effectiveness on complex goals, while word-level guidance suffices for simple goals (e.g., single keywords).
- DiffTOD can serve both buyer and seller roles using the same model simply by switching guidance strategies, demonstrating its flexibility.
- Non-sequential planning is especially effective in sparse reward scenarios (where feedback is only available at the end).

## Highlights & Insights
- The mathematical connection between diffusion models and dialogue planning is elegantly derived—showing an equivalent transformation from MDP trajectory likelihood to the diffusion denoising process.
- The three-tiered guidance mechanism is cleverly designed (word-level $\rightarrow$ semantic-level $\rightarrow$ search-based), progressively increasing complexity and strategic depth.
- The decoupled "planning and generation" architecture allows the planner's output to guide any LLM for dialogue execution.
- Flexible guidance (switching targets at test time without retraining) resolves the limitation of RL-based methods that require retraining for each goal.

## Limitations & Future Work
- The generation quality and speed of diffusion language models are not as good as autoregressive LLMs.
- Search-based guidance incurs high computational overhead (requiring the construction of search trees).
- States and actions are represented in natural language, so the trajectory length is limited by the model's context window.
- The user simulator (GPT-4o) might not fully reflect real user behavior.

## Related Work & Insights
- **vs PPDPP (RL-based)**: PPDPP optimizes sequentially using policy gradients, whereas DiffTOD generates non-sequentially using diffusion, with the latter showing a clear advantage in long-term planning.
- **vs ProCoT/EnPL (Prompt-based)**: Prompt-based methods rely heavily on the LLM's inherent planning capabilities, while DiffTOD uses a dedicated diffusion model for more controllable planning.
- **vs Diffuser (Janner et al.)**: Serving as the inspiration, Diffuser is designed for continuous control, while DiffTOD adapts this to discrete dialogue scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The mathematical modeling of using diffusion models for dialogue planning is highly elegant, and the three-tiered guidance design is exquisite.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets + multiple baselines + ablations + turn-by-turn analysis + human evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematical derivations and intuitive diagrams.
- Value: ⭐⭐⭐⭐ Provides a brand-new non-sequential planning paradigm for target-oriented dialogue.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SceneDecorator: Towards Scene-Oriented Story Generation with Scene Planning and Scene Consistency](../../NeurIPS2025/image_generation/scenedecorator_towards_scene-oriented_story_generation_with_scene_planning_and_s.md)
- [\[CVPR 2025\] Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems](../../CVPR2025/image_generation/fractals_made_practical_denoising_diffusion_as_partitioned_iterated_function_sys.md)
- [\[NeurIPS 2025\] Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation](../../NeurIPS2025/image_generation/diffusion-driven_progressive_target_manipulation_for_source-free_domain_adaptati.md)
- [\[CVPR 2025\] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](../../CVPR2025/image_generation/codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)
- [\[ICML 2025\] Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning](../../ICML2025/image_generation/local_manifold_approximation_and_projection_for_manifold-aware_diffusion_plannin.md)

</div>

<!-- RELATED:END -->
