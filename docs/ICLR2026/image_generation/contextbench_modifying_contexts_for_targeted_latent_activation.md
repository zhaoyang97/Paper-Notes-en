---
title: >-
  [Paper Note] ContextBench: Modifying Contexts for Targeted Latent Activation
description: >-
  [ICLR 2026][Image Generation][Context modification] This paper introduces ContextBench, a benchmark comprising 715 tasks for evaluating methods that automatically generate fluent inputs capable of activating specific lat…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Context modification"
  - "latent feature activation"
  - "AI safety"
  - "sparse autoencoders"
  - "backdoor detection"
date: 2026-05-08
content_hash: 56e8319ba36c2722
---

# ContextBench: Modifying Contexts for Targeted Latent Activation

**Conference**: ICLR 2026
**arXiv**: [2506.15735](https://arxiv.org/abs/2506.15735)  
**Code**: [https://github.com/lasr-eliciting-contexts/ContextBench](https://github.com/lasr-eliciting-contexts/ContextBench)  
**Area**: Image Generation
**Keywords**: Context modification, latent feature activation, AI safety, sparse autoencoders, backdoor detection

## TL;DR
This paper introduces ContextBench, a benchmark comprising 715 tasks for evaluating methods that automatically generate fluent inputs capable of activating specific latent features, and proposes two EPO-enhanced variants—LLM-assisted and diffusion-model inpainting—that Pareto-dominate standard EPO in the trade-off between activation strength and linguistic fluency.

## Background & Motivation

**Background**: A central challenge in AI safety is identifying, prior to deployment, inputs that trigger harmful model behaviors. Feature visualization for vision-language models is well-established in the visual domain (via gradient-based optimization of maximally activating images), but remains substantially more difficult in the language domain due to the discrete nature of token spaces.

**Limitations of Prior Work**: (a) White-box methods such as GCG can produce high-activation inputs via gradient signals, yet the resulting text is entirely incoherent and would not arise in real deployments; (b) black-box methods such as GPT-4o prompting yield fluent text but produce only weak activations, failing to identify genuine trigger conditions; (c) EPO, the only prior method addressing both objectives, still falls short of the fluency level required for safety-critical applications.

**Key Challenge**: Activation strength and linguistic fluency are inherently in tension—single-token gradient edits are prone to local optima, and simultaneously achieving high fluency and high activation requires coordinated multi-token modifications.

**Goal**: (a) Establish a systematic benchmark for evaluating context modification methods; (b) improve the fluency–activation trade-off of EPO; (c) apply such techniques for the first time to the activation of SAE latent features.

**Key Insight**: The paper extends "feature visualization" from the visual to the language domain by generating fluent text that activates specific SAE latent features, thereby illuminating the internal mechanisms of language models. Fluent trigger inputs are more valuable in safety settings—they are more likely to occur naturally, harder to detect, and more informative about underlying mechanisms.

**Core Idea**: Augmenting gradient-based optimization with LLM assistance and diffusion-model inpainting to generate input texts that are simultaneously fluent and strongly activate specific internal model features.

## Method

### Overall Architecture
ContextBench encompasses three task categories: (1) SAE activation (205 SAE features)—generating text that maximally activates a target feature; (2) story inpainting (500 stories)—modifying intermediate sentences to alter predicted continuations; (3) backdoor discovery (10 models)—recovering conditions that trigger backdoor behaviors. Evaluation jointly considers activation strength and fluency (cross-entropy within the range 3–9).

### Key Designs

1. **Systematic Construction of the SAE Feature Dataset**:

    - Function: Carefully curates 205 SAE features from Gemma Scope and Llama Scope, spanning three difficulty axes.
    - Mechanism: Features are categorized along activation density (low/medium/high), lexical diversity (single word/related concepts/broad), and locality (single-token/paragraph-level), yielding 27 combinations with at least 2 features each.
    - Design Motivation: Different feature types pose different challenges to context modification methods—high-locality features can be activated by inserting specific tokens, whereas global features require stylistic modification of entire passages.

2. **EPO-Assist (LLM-Assisted)**:

    - Function: Uses an LLM as a mutation operator within evolutionary search.
    - Mechanism: Every 50 iterations, candidate texts from EPO's current population are passed to GPT-4o, which generates new fluent candidates based on these samples (without access to feature descriptions); EPO then continues gradient-based refinement. This creates a feedback loop: EPO identifies high-activation token patterns → LLM naturalizes them → EPO refines further.
    - Design Motivation: When single-token edits become trapped in local optima, the LLM can perform coordinated multi-token modifications to escape them.

3. **EPO-Inpainting (Diffusion-Model Inpainting)**:

    - Function: Uses LLaDA (a large language diffusion model) to inpaint low-activation tokens while preserving high-activation ones.
    - Mechanism: Per-token attribution is used to decompose SAE activation contributions; the top-25% contributing tokens are frozen alongside a randomly selected 25% anchor tokens, and LLaDA-8B's bidirectional attention inpaints the remaining positions. This is applied every 15 iterations.
    - Design Motivation: Analogous to a "fluency projection"—EPO's unconstrained exploration may disrupt coherence, and periodic inpainting projects the text back onto the fluent manifold while preserving high-value tokens.

4. **Story Inpainting and Backdoor Tasks**:

    - Story inpainting: Within a fixed context, intermediate sentences are modified such that the logit difference between the target token and the source token exceeds a threshold, testing context-sensitive behavior elicitation.
    - Backdoor discovery: Given a backdoored model and its anomalous behavior, the trigger condition is recovered. Three backdoor types are included: password-triggered sandbagging, time-triggered toxic outputs, and password-based refusal bypass.

### Loss & Training

The core EPO objective is: $\mathcal{L}_\lambda = \mathcal{L}_{GCG} + \frac{\lambda}{n} \sum_{i=1}^{n} \log(p_i)$, where the GCG term maximizes the target activation, the cross-entropy term maintains fluency, and $\lambda$ controls the trade-off. Multiple values of $\lambda$ are optimized in parallel to trace the Pareto frontier. The additional computational overhead of EPO-Assist and EPO-Inpainting is minimal, as both are invoked periodically rather than at every step.

## Key Experimental Results

### Main Results

SAE activation task (activation values normalized as a fraction of the training-set maximum):

| Method | Mean Activation Strength | Highest In-Fluency Activation Rate | Pareto-Dominates EPO |
|--------|--------------------------|-----------------------------------|----------------------|
| GCG | Highest | Lowest (text unreadable) | — |
| GPT-4o | Lowest | High fluency | — |
| EPO | Moderate | Moderate | baseline |
| **EPO-Assist** | High | High | ✓ |
| **EPO-Inpainting** | Highest (within fluency range) | Highest | ✓ |

### Ablation Study

| Analysis Dimension | Key Findings |
|-------------------|--------------|
| Feature type effect | Local + low-diversity features are easiest to activate; global + high-diversity features are hardest |
| Story inpainting | EPO variants outperform GPT-4o in logit difference, but occasionally discover unexpected shortcut solutions (e.g., the medical sense of *rash*) |
| Backdoor detection | White-box methods recover simple password triggers (1–3 tokens), but long passwords and semantic triggers (temporal, audit-context) remain challenging |
| Fluency validation | Cross-entropy is highly correlated with human fluency ratings ($\rho = 0.94$), validating the proxy metric |

### Key Findings
- EPO-Inpainting achieves the highest activation within the fluency constraint and Pareto-dominates all other methods.
- Black-box methods (GPT-4o) are severely limited on SAE activation tasks—the absence of internal model information prevents precise identification of activation conditions.
- Neuronpedia's automated feature descriptions are sometimes misleading (e.g., a feature described as "number-related" that predominantly activates on the digit "1"), underscoring the value of fine-grained feature analysis.
- "Specification gaming" is both interesting and informative: some shortcut solutions (e.g., direct insertion of the target token) actually reveal the shallow nature of the feature itself.
- Backdoor recovery remains challenging for existing methods, with difficulty increasing as trigger length grows.

## Highlights & Insights
- **Systematization of language-domain feature visualization**: This is the first work to systematically transfer the mature feature visualization paradigm from vision to language models, providing a structured difficulty space along three axes for future research.
- **"Projection" intuition of EPO-Inpainting**: Periodically projecting optimization results back onto the fluent text manifold while retaining high-activation anchor tokens is a generalizable principle applicable to any setting requiring alternation between continuous optimization and discrete constraints.
- **The dual nature of shortcut solutions**: The paper handles specification gaming elegantly—rather than treating shortcuts as failures, it notes that some (e.g., discovered jailbreak patterns) carry independent safety value.
- **Novel application of LLaDA**: The bidirectional attention of a diffusion language model is exploited for conditional inpainting, representing the first application of LLaDA in the interpretability domain.

## Limitations & Future Work
- EPO-Assist relies on the GPT-4o API, introducing additional cost and dependence on an external model.
- The fluency metric (cross-entropy) correlates strongly with human ratings, but the range of 3–9 remains an arbitrary human-defined threshold.
- The backdoor detection task covers only 10 models and a limited set of trigger types.
- Although the selection of SAE features is systematic, manual curation remains involved and may overlook certain important categories.
- Evaluation is restricted to Gemma-2 and Llama series models and has not been validated on larger-scale models.

## Related Work & Insights
- **vs. GCG (Zou et al., 2023)**: GCG optimizes only for activation without regard to fluency, producing unreadable adversarial inputs. EPO and its variants add fluency constraints on top of this foundation.
- **vs. FLRT (Thompson & Sklar, 2024)**: FLRT employs a teacher–student framework to improve fluency, but the proposed LLM-assisted and inpainting methods achieve superior Pareto frontiers.
- **vs. BEAST (Sadasivan et al., 2024)**: BEAST is a purely black-box method using beam search for token substitution. This paper demonstrates that white-box gradient information is critical for precise feature activation.
- **Implications for safety**: Context modification techniques can be applied to (1) discover actual trigger conditions for SAE features, (2) automate backdoor auditing, and (3) reveal changes in model behavior under specific contextual conditions.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic benchmark for language-domain feature visualization; EPO variants are elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ 715 tasks spanning diverse scenarios, though the backdoor task scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, a safety-oriented perspective pervades the paper, and the discussion of specification gaming is thoughtful.
- Value: ⭐⭐⭐⭐ Significant value for the AI safety and interpretability communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Targeted Data Protection for Diffusion Model by Matching Training Trajectory](../../AAAI2026/image_generation/targeted_data_protection_for_diffusion_model_by_matching_training_trajectory.md)
- [\[ICLR 2026\] Latent Diffusion Model without Variational Autoencoder](latent_diffusion_model_without_variational_autoencoder.md)
- [\[ICCV 2025\] Penalizing Boundary Activation for Object Completeness in Diffusion Models](../../ICCV2025/image_generation/penalizing_boundary_activation_for_object_completeness_in_diffusion_models.md)
- [\[CVPR 2026\] RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models](../../CVPR2026/image_generation/razor_ratio-aware_layer_editing_for_targeted_unlearning_in_vision_transformers_a.md)
- [\[NeurIPS 2025\] LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss](../../NeurIPS2025/image_generation/lineas_end-to-end_learning_of_activation_steering_with_a_distributional_loss.md)

</div>

<!-- RELATED:END -->
