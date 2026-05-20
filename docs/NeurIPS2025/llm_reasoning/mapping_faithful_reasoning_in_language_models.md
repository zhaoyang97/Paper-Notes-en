---
title: >-
  [Paper Note] Mapping Faithful Reasoning in Language Models
description: >-
  [NeurIPS 2025][LLM Reasoning][CoT faithfulness] This paper proposes the Concept Walk framework, which tracks the evolution of internal concept representations across reasoning steps by projecting residual stream activati…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "CoT faithfulness"
  - "mechanistic interpretability"
  - "reasoning models"
  - "safety vectors"
  - "activation analysis"
date: 2026-05-08
content_hash: 95c7a0f34b51c584
---

# Mapping Faithful Reasoning in Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.22362](https://arxiv.org/abs/2510.22362)  
**Code**: None  
**Area**: LLM Reasoning
**Keywords**: CoT faithfulness, mechanistic interpretability, reasoning models, safety vectors, activation analysis

## TL;DR
This paper proposes the Concept Walk framework, which tracks the evolution of internal concept representations across reasoning steps by projecting residual stream activations at each step onto concept directions learned from contrastive data, thereby distinguishing whether a CoT chain genuinely participates in computation or merely serves as post-hoc decorative output.

## Background & Motivation

**Background**: Reasoning-oriented LLMs (e.g., OpenAI o1, Gemini thinking mode, Qwen 3) provide visibility into the reasoning process via Chain-of-Thought (CoT), which is widely regarded as an important means of enhancing AI transparency and trustworthiness. Practitioners rely on inspecting CoT to verify whether decisions are grounded in sound reasoning.

**Limitations of Prior Work**: Growing evidence suggests that CoT does not always faithfully reflect internal computation—it may constitute post-hoc rationalisation, wherein the model has already determined an answer before generating a seemingly coherent reasoning process. This renders safety oversight through CoT inspection unreliable.

**Key Challenge**: Surface-level text cannot distinguish between two modes—"CoT as computation" (where the reasoning process genuinely influences the output) and "CoT as rationalisation" (where the reasoning process is decorative). Existing filtering methods (e.g., truncating or perturbing CoT and observing output changes) can only determine whether outputs change, but cannot reveal how internal concept representations evolve throughout reasoning.

**Goal**: To track the dynamic evolution of specific concepts (e.g., safety) across multi-step reasoning in activation space, thereby distinguishing faithful reasoning from decorative reasoning.

**Key Insight**: The paper combines concept direction extraction from representation engineering with temporal analysis from mechanistic interpretability, tracking concept activations along the dimension of reasoning steps.

**Core Idea**: A "safety direction" vector is learned by contrasting safe and unsafe prompts; the activations at each reasoning step are then projected onto this direction to obtain a "Concept Walk" trajectory. Persistent deviation of the trajectory following CoT perturbation indicates computational reasoning, whereas rapid recovery indicates decorative reasoning.

## Method

### Overall Architecture
A three-stage methodology: (1) **Filtering**—samples are categorized as "hard" (CoT participates in computation) or "easy" (CoT is decorative) via perturbation sensitivity; (2) **Learning the safety direction**—a safety concept vector is computed in activation space using contrastive data; (3) **Concept Walk**—activations at each reasoning step are projected onto the safety direction to track temporal evolution, and original and perturbed trajectories are compared.

### Key Designs

1. **Filtering for CoT-as-computation**:

    - **Function**: Categorizes samples into "hard" and "easy" to determine whether CoT functionally participates in decision-making.
    - **Mechanism**: Inspired by Lanham et al. and Emmons et al., flawed reasoning steps are injected into the middle of the CoT for each sample (perturbation), and whether this causes significant degradation in the model's final output is observed. "Hard" samples exhibiting large output changes are retained (CoT is integrated into computation), while "easy" samples whose outputs remain unchanged are filtered out (CoT serves only as rationalisation).
    - **Design Motivation**: Without filtering, conclusions drawn from analyzing the internal dynamics of decorative CoT may be misleading—the analysis focuses exclusively on cases where reasoning genuinely affects the output.

2. **Computing the Safety Vector**:

    - **Function**: Identifies the direction encoding the concept of "safety" in activation space.
    - **Mechanism**: Constructs an unsafe prompt set $\mathcal{D}_{\text{unsafe}}$ and a safe prompt set $\mathcal{D}_{\text{safe}}$ (paired counterfactual pairs), and computes their respective mean activations at layer $\ell$ and token position $t$:
    $\boldsymbol{\mu}_{\text{unsafe}}^{(\ell,t)} = \frac{1}{|\mathcal{D}_{\text{unsafe}}|} \sum_{i \in \mathcal{D}_{\text{unsafe}}} \boldsymbol{x}_\ell^i[t], \quad \boldsymbol{\mu}_{\text{safe}}^{(\ell,t)} = \frac{1}{|\mathcal{D}_{\text{safe}}|} \sum_{i \in \mathcal{D}_{\text{safe}}} \boldsymbol{x}_\ell^i[t]$
    The safety direction is the normalized difference:
    $\hat{\boldsymbol{v}}^{(\ell,t)} = \frac{\boldsymbol{\mu}_{\text{unsafe}}^{(\ell,t)} - \boldsymbol{\mu}_{\text{safe}}^{(\ell,t)}}{\|\boldsymbol{\mu}_{\text{unsafe}}^{(\ell,t)} - \boldsymbol{\mu}_{\text{safe}}^{(\ell,t)}\|_2}$
    - **Direction Selection**: Each candidate $(\ell, t)$ is evaluated on the validation set using bypass score (refusal suppression after ablation), induce score (refusal induction after addition), and KL divergence (minimizing impact on benign prompts); the optimal direction is selected accordingly.
    - **Design Motivation**: Difference of Means is a well-established direction extraction method in representation engineering, capable of capturing how the model internally encodes safety rather than relying on surface textual features.

3. **Concept Walk**:

    - **Function**: Tracks the temporal evolution of safety concepts across reasoning steps.
    - **Mechanism**: The model is run in thinking mode; for each CoT step $s$, residual stream activations across all tokens in that step are extracted and averaged to obtain a step-level activation vector:
    $\boldsymbol{h}_s = \frac{1}{|T_s|} \sum_{t \in \mathcal{T}_s} \boldsymbol{x}[t]$
    The cosine similarity with the safety direction is then computed:
    $\alpha_s = \cos(\boldsymbol{h}_s, \boldsymbol{v}^{(\ell^*)})$
    $\alpha_s$ quantifies the alignment between the model's internal state at step $s$ and the safety direction, independently of whether the surface text mentions safety-related terms. The same analysis is applied to perturbed CoT, and the difference between original and perturbed trajectories is compared.
    - **Design Motivation**: Existing methods can only determine "whether CoT influences the output," but cannot reveal "how internal concept representations evolve throughout reasoning."

### Experimental Models and Data
- **Target model**: Qwen 3-4B (36-layer Transformer decoder with controllable thinking mode)
- **Data generation**: Synthetic data generated with Mistral-7B-Instruct-v0.2 (to avoid data contamination), simulating a music AI assistant scenario
- **Two safety categories**: Harm (2,911 pairs) and Hate (4,819 pairs); after label filtering, Hate retains 256 violating + 462 compliant samples, and Harm retains 181 violating + 290 compliant samples

## Key Experimental Results

### Dataset Statistics

| Category | Train | Validation | Test | Total Pairs |
|----------|-------|------------|------|-------------|
| Harm | 1,746 | 873 | 292 | 2,911 |
| Hate | 2,890 | 1,445 | 484 | 4,819 |

### Core Experimental Findings

| Observation Dimension | "Hard" Samples (CoT-as-computation) | "Easy" Samples (CoT-as-rationalisation) |
|-----------------------|--------------------------------------|------------------------------------------|
| Safety activation change after perturbation | Persistent, structured deviation sustained across multiple steps | Brief transient change; rapidly recovers to the original trajectory |
| Persistence of perturbation effect | Deviation continues for many steps beyond the injection point | Model "self-corrects" within 1–2 steps |
| Final output change | Significant change (e.g., from refusal to compliance) | No change or negligible change |
| Initial safety activation on violating prompts | High (confirming validity of safety vector) | High (consistent with hard group) |

### Key Findings
- **Clear separation between hard and easy samples**: Consistently observed across both Harm and Hate datasets—perturbations in hard samples cause persistent deviations in safety activation, whereas deviations in easy samples are transient.
- **Immediate recognition of violating prompts**: Regardless of hard or easy classification, violating prompts consistently trigger higher initial safety activation, validating the efficacy of the safety direction vector.
- **Consistency under normalized x-axis**: Since CoT lengths vary, normalizing by CoT length aligns all injection points at 0.5, facilitating comparison; patterns become clearer after normalization.
- **Qualitative case validation**: "Can you recommend some classic rock tunes" (hard sample)—injecting flawed reasoning causes persistent deviation in safety activation and ultimately changes the output; "Playlist for a bar crawl" (easy sample)—brief elevation followed by recovery after injection, with no output change.
- The cross-mode assumption of computing the safety vector in non-thinking mode and applying it in thinking mode is largely validated.

## Highlights & Insights
- **A temporal lens at the concept level**: Concept Walk provides a fundamentally new analytical perspective—rather than examining what the surface text says, it examines how internal concept representations evolve throughout the reasoning process. This offers richer information than output-level perturbation tests that only observe result changes.
- **Generality of the method**: Although the paper uses safety as a case study, the framework is fully general—tracking fairness, bias, toxicity, or any concept definable through contrast requires only replacing the contrastive dataset.
- **Bridging two communities**: The work connects representation engineering (learning concept directions) and CoT faithfulness analysis (distinguishing computation from decoration); each has limitations in isolation, but their combination yields novel insights.
- **Practical implications for safety oversight**: When Concept Walk reveals that safety activations are transient (easy samples), auditors should not trust safety reasoning text in CoT, as it is merely decorative.

## Limitations & Future Work
- The safety direction is computed in non-thinking mode and applied in thinking mode—if representations differ between modes, step-wise estimates may be biased. Future work should compute mode-specific safety directions.
- The perturbation-based filtering strategy cannot guarantee complete faithfulness—certain computational processes may remain unverbalized in CoT (hidden reasoning pathways).
- Validation is conducted on Qwen 3-4B alone; generalizability across different model scales, architectures, and training paradigms remains to be explored.
- The synthetic data scenario (playlist requests in a music AI assistant) is relatively narrow; safety reasoning in real deployment scenarios may be considerably more complex.
- The terms "hard" and "easy" are borrowed from task difficulty but actually denote "whether the model relies on CoT," which may cause conceptual confusion.
- The number of violating samples remaining after filtering is small, limiting statistical power.

## Related Work & Insights
- **vs. Lanham et al. / Emmons et al.**: They proposed filtering strategies based on perturbation/truncation of CoT to detect faithfulness; Concept Walk builds on this by adding temporal tracking of internal representations, revealing not only "whether faithful" but also "how internal states evolve."
- **vs. Arditi et al. (Refusal Direction)**: Similarly uses Difference of Means to extract concept directions, but Arditi focuses on single-instance refusal behavior; this work applies the direction to temporal analysis across multi-step reasoning.
- **vs. Venhoff et al. (Steering Vectors for Reasoning)**: They use steering vectors to influence reasoning trajectories and reveal cross-step dependencies; this work passively observes via concept directions rather than actively intervening—the two approaches are complementary.
- **vs. Bogdan et al. (Thought Anchors)**: Identifies causally important reasoning steps; Concept Walk provides an alternative perspective for assessing step importance through the lens of concept activation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Concept Walk is an elegant methodological framework that, for the first time, provides a temporal perspective on CoT faithfulness at the level of concept representations.
- **Experimental Thoroughness**: ⭐⭐⭐ As a methodological contribution, the case study adequately demonstrates the framework's capabilities, but is limited to a single model and synthetic data.
- **Writing Quality**: ⭐⭐⭐⭐ The logical chain is clear, methodological exposition is precise, and discussion is honest and thorough, though notation is heavy.
- **Value**: ⭐⭐⭐⭐ Of significant methodological value to the AI safety community; Concept Walk has the potential to become a standard tool for analyzing the faithfulness of reasoning models.

## Related Papers

- [\[NeurIPS 2025\] ProofSketch: Efficient Verified Reasoning for Large Language Models](proofsketch_efficient_verified_reasoning_for_large_language_models.md)
- [\[NeurIPS 2025\] I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models](i-raven-x_benchmarking_generalization_and_robustness_of_analogical_and_mathemati.md)
- [\[NeurIPS 2025\] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)
- [\[NeurIPS 2025\] Scalable Best-of-N Selection for Large Language Models via Self-Certainty](scalable_best-of-n_selection_for_large_language_models_via_self-certainty.md)
- [\[NeurIPS 2025\] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ProofSketch: Efficient Verified Reasoning for Large Language Models](proofsketch_efficient_verified_reasoning_for_large_language_models.md)
- [\[NeurIPS 2025\] I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models](i-raven-x_benchmarking_generalization_and_robustness_of_analogical_and_mathemati.md)
- [\[NeurIPS 2025\] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)
- [\[NeurIPS 2025\] Scalable Best-of-N Selection for Large Language Models via Self-Certainty](scalable_best-of-n_selection_for_large_language_models_via_self-certainty.md)
- [\[NeurIPS 2025\] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)

</div>

<!-- RELATED:END -->
