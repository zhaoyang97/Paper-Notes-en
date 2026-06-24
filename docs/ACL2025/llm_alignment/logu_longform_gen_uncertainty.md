---
title: >-
  [Paper Note] LoGU: Long-form Generation with Uncertainty Expressions
description: >-
  [ACL 2025][LLM Alignment][uncertainty expressions] This work defines the "Long-form Generation with Uncertainty Expressions" (LoGU) task, identifies two sub-challenges (uncertainty suppression and uncertainty misalignment), and proposes a decomposition-based data construction framework and an SFT+DPO two-stage training pipeline. This enables LLMs to explicitly express uncertainty for uncertain facts in long-form generation, improving Llama3-8B's factual accuracy from 51.9% to…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "uncertainty expressions"
  - "long-form generation"
  - "hallucination mitigation"
  - "SFT+DPO"
  - "factual accuracy"
date: 2026-05-08
content_hash: 92f370d5b050bb4a
---

# LoGU: Long-form Generation with Uncertainty Expressions

**Conference**: ACL 2025  
**arXiv**: [2410.14309](https://arxiv.org/abs/2410.14309)  
**Code**: [https://github.com/rhyang2021/LoGU](https://github.com/rhyang2021/LoGU)  
**Area**: Others  
**Keywords**: uncertainty expressions, long-form generation, hallucination mitigation, SFT+DPO, factual accuracy  

## TL;DR
This work defines the "Long-form Generation with Uncertainty Expressions" (LoGU) task, identifies two sub-challenges (uncertainty suppression and uncertainty misalignment), and proposes a decomposition-based data construction framework and an SFT+DPO two-stage training pipeline. This enables LLMs to explicitly express uncertainty for uncertain facts in long-form generation, improving Llama3-8B's factual accuracy from 51.9% to 71.6% and reducing the number of incorrect claims from 20.4 to 5.81 across three datasets.

## Background & Motivation
**Background**: LLMs still confidently generate incorrect information when uncertain (hallucination). Existing uncertainty estimation research mainly focuses on short-form QA (e.g., providing a confidence score), which is inapplicable to long-form scenarios.

**Limitations of Prior Work**: (a) Short-form uncertainty methods assign a single unified uncertainty score to the entire answer, whereas long-form text contains multiple claims with varying correctness requiring fine-grained handling; (b) LLMs exhibit "uncertainty suppression"—tending to provide confident answers instead of admitting they do not know; (c) the uncertainty expressions of existing methods are often vague and general (e.g., "I am not sure about his early life") rather than targeting specific facts.

**Key Challenge**: An ideal LLM should confidently generate known facts and explicitly express uncertainty on unknown facts. However, this requires the model to simultaneously possess self-awareness and fine-grained expressive capabilities.

**Goal**: How to enable LLMs to make correct certainty/uncertainty judgments for each atomic claim and express them naturally in long-form text.

**Key Insight**: Decompose long-form text into atomic claims, perform fact-checking on each claim, selectively rewrite incorrect claims into uncertainty expressions, and then reorganize them back into coherent text.

**Core Idea**: Construct training data based on a decompose-check-rewrite-reorganize pipeline, using SFT to address uncertainty suppression and DPO to resolve uncertainty misalignment.

## Method

### Overall Architecture
The input is an open-ended question (e.g., "Introduce someone"), and the output is a long-form answer containing uncertainty expressions. The method consists of two stages: (1) Data construction—decomposing original answers into atomic claims, followed by fact-checking to construct positive/negative samples; (2) Two-stage training—SFT to enable the model to express uncertainty, and DPO to ensure precise uncertainty expression.

### Key Designs

1. **Decomposed Data Construction Framework**:

    - **Function**: Converts long-form answers into training data containing uncertainty expressions.
    - **Mechanism**:
        - **Decomposition**: Uses an auxiliary LLM (GPT-4o) to decompose answer $R$ into atomic claims $C = \{c_1, ..., c_N\}$
        - **Fact-Checking**: Uses FActScore to retrieve Wikipedia passages for validation, dividing claims into a supported set $C_s$ and a refuted set $C_{ns}$
        - **Positive Rewriting $\Gamma_{\text{pos}}$**: Rewrites refuted claims into uncertainty expressions (e.g., "I am not sure about his birth date") while keeping supported claims unchanged, yielding $R_{\text{pos}}$ (the ideal response)
        - **Negative Rewriting $\Gamma_{\text{neg}}$**: Conversely, rewrites supported claims into uncertainty expressions while keeping refuted claims unchanged, yielding $R_{\text{neg}}$ (erroneous uncertainty examples)
        - **Reorganization**: Reorganizes the rewritten claims into coherent text using an auxiliary LLM
    - **Design Motivation**: Direct generation of text with uncertainty expressions yields poor quality. Operating at the atomic claim level allows for higher precision and accurate control over the proportion of uncertainty.

2. **Control of Uncertainty Proportion**:

    - **Function**: Controls the proportion of uncertainty expressions in the training data.
    - **Mechanism**: Restricts the number of rewritten claims via a downsampling ratio $\alpha$, selecting $\min(\frac{\alpha}{1-\alpha}|C_s|, |C_{ns}|)$ claims for rewriting.
    - **Design Motivation**: Excessive uncertainty expressions reduce the readability and informativeness of the response. A balance must be struck between "reducing errors" and "maintaining comprehensiveness".

3. **Two-Stage Training Pipeline**:

    - **LoGU-SFT**: Performs SFT using positive rewritten data $(q, R_{\text{pos}})$ to train the model to naturally embed uncertainty expressions in its answers, overcoming uncertainty suppression.
    - **LoGU-DPO**: Constructs preference pairs—$R_{\text{pos}} \succ R \succ R_{\text{neg}}$, generating 3 preference pairs ($\binom{3}{2}$). DPO is applied to help the model learn to express uncertainty in the correct places (rather than being uncertain about known facts), addressing uncertainty misalignment.
    - **Design Motivation**: SFT alone only teaches the model "how to say uncertain" but not "when to say it". DPO leverages pairwise comparison to align appropriate applications of uncertainty.

### Loss & Training
- **LoGU-SFT**: Standard cross-entropy loss, computed solely on the response part.
- **LoGU-DPO**: Standard DPO loss with the SFT model $\pi_{\text{sft}}$ as the reference policy.
- 10 hand-crafted uncertainty expression patterns (e.g., "I'm not sure about...", "I don't have information on...").

## Key Experimental Results

### Main Results (Llama3-8B-Instruct)

| Dataset | Method | FA(↑) | UA(↑) | # Errors(↓) |
|--------|------|-------|-------|---------|
| Bios | Original | 51.9 | - | 20.4 |
| Bios | Unc-Zero (prompt)| 53.8 | 65.4 | 14.9 |
| Bios | LoGU-SFT | 62.7 | 76.7 | 10.2 |
| Bios | **LoGU-DPO** | **71.6** | **84.3** | **5.81** |
| LongFact | Original | 85.5 | - | 7.45 |
| LongFact | **LoGU-DPO** | **92.2** | **62.5** | **2.14** |
| WildHallu | Original | 74.4 | - | 6.24 |
| WildHallu | **LoGU-DPO** | **88.3** | **62.6** | **2.47** |

### Ablation Study (Mistral-7B, Bios)

| Configuration | FA(↑) | UA(↑) | # Errors(↓) | Description |
|------|-------|-------|---------|------|
| Original | 38.8 | - | 27.9 | No uncertainty expressions |
| LoGU-SFT only | 54.5 | 77.1 | 11.4 | Learns expressions but with limited accuracy |
| **LoGU-SFT+DPO** | **65.4** | **80.7** | **6.54** | DPO further improves alignment |
| Prompt (Self-Refine) | 38.3 | 57.4 | 27.2 | Prompting methods show limited effectiveness |

### Key Findings
- Two-stage training significantly outperforms prompting methods—LoGU-DPO achieves 28 percentage points (pp) higher FA than the best prompting method on Bios (using Mistral).
- The DPO stage is crucial for improving uncertainty precision—SFT $\rightarrow$ DPO improves FA from 54.5% to 65.4% on Mistral.
- Baseline methods often produce vague and general uncertainty expressions (e.g., "I'm not sure about their career"), whereas LoGU generates more specific expressions (e.g., "I'm not sure about the exact founding year").
- Shows strong performance on the out-of-domain dataset ASQA, indicating robust generalizability.
- An uncertainty proportion $\alpha$ of 15-20% yields the best results; excessively high values sacrifice informativeness.

## Highlights & Insights
- **The decompose-rewrite-reorganize data construction paradigm** is highly generalizable—it can be applied not only to uncertainty expressions but also to other fine-grained text attribute controls (e.g., safety, formality, sentiment).
- **Constructing DPO preference pairs via positive/negative rewriting** is elegant—positive rewriting (acknowledging what is unknown) and negative rewriting (claiming uncertainty on known facts) naturally form high-quality preference pairs without requiring human annotation.
- **Control of the uncertainty proportion** is an overlooked but crucial design choice—overly expressing uncertainty can be detrimental as users expect informative answers.

## Limitations & Future Work
- The determination of "true uncertainty" relies on whether short-form QA answers are correct, which is an approximation.
- Whether uncertainty expressions are "vague" is evaluated by GPT-4o, which introduces subjectivity.
- Data construction depends heavily on an auxiliary LLM (GPT-4o), resulting in high costs.
- Validation is restricted to 7-8B models, leaving its effectiveness on larger models unverified.
- Future work could explore uncertainty expression during real-time inference, rather than baking it in during training.

## Related Work & Insights
- **vs. Short-form uncertainty methods (e.g., verbalized confidence)**: Short-form methods evaluate overall confidence and cannot handle varying degrees of uncertainty among multiple claims within long-form text.
- **vs. Self-Refine**: Self-improvement methods show limited efficacy in expressing uncertainty (with almost no improvement in FA), indicating the necessity of targeted training.
- **vs. IDK probing**: IDK-based probing methods primarily detect whether the model "knows" the information, but cannot naturally embed uncertainty within long-form responses.
- The data construction pipeline of this method offers valuable insights for preference data creation in RLHF/DPO.

## Rating
- Novelty: ⭐⭐⭐⭐ Defines a meaningful new task (LoGU); the decomposed data construction framework is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across three datasets, two models, ablation studies, and out-of-domain tests.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, well-abstracted sub-challenges, and a smooth methodological flow.
- Value: ⭐⭐⭐⭐ A practical and crucial research direction with direct value for enhancing LLM reliability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation](rpo_retrieval_preference_optimization_for_robust_retrieval-augmented_generation.md)
- [\[NeurIPS 2025\] Multi-Environment POMDPs: Discrete Model Uncertainty Under Partial Observability](../../NeurIPS2025/llm_alignment/multi-environment_pomdps_discrete_model_uncertainty_under_partial_observability.md)
- [\[ACL 2025\] Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation](retrieval-augmented_fine-tuning_with_preference_optimization_for_visual_program_.md)
- [\[ACL 2025\] Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step](chain-of-jailbreak_attack_for_image_generation_models_via_editing_step_by_step.md)
- [\[ICLR 2026\] Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences](../../ICLR2026/llm_alignment/omni-reward_towards_generalist_omni-modal_reward_modeling_with_free-form_prefere.md)

</div>

<!-- RELATED:END -->
