---
title: >-
  [Paper Note] A Cortically Inspired Architecture for Modular Perceptual AI
description: >-
  [ICLR 2026][cortically inspired architecture] This paper proposes a cortically inspired modular perceptual AI architecture blueprint grounded in neuroscience, comprising four components — dedicated encoders, a shared cross-modal latent space, a routing controller, and a recursive predictive feedback loop — and validates through sparse autoencoder experiments that modular decomposition improves within-domain feature stability (+15.4pp Jaccard overlap).
tags:
  - ICLR 2026
  - cortically inspired architecture
  - modular perception
  - predictive coding
  - cross-modal fusion
  - sparse autoencoder
date: 2026-05-08
content_hash: 15e22fef1574b6cf
---

# A Cortically Inspired Architecture for Modular Perceptual AI

**Conference**: ICLR 2026
**arXiv**: [2603.07295](https://arxiv.org/abs/2603.07295)
**Code**: None
**Area**: Cognitive Architecture / Modular AI
**Keywords**: cortically inspired architecture, modular perception, predictive coding, cross-modal fusion, sparse autoencoder

## TL;DR
This paper proposes a cortically inspired modular perceptual AI architecture blueprint grounded in neuroscience, comprising four components — dedicated encoders, a shared cross-modal latent space, a routing controller, and a recursive predictive feedback loop — and validates through sparse autoencoder experiments that modular decomposition improves within-domain feature stability (+15.4pp Jaccard overlap).

## Background & Motivation

**Background**: Current perceptual AI systems are dominated by large-scale end-to-end monolithic models (e.g., GPT-4V, Gemini) that achieve strong performance across diverse tasks.

**Limitations of Prior Work**: Monolithic models operate as opaque black boxes, exhibiting brittleness under out-of-distribution scenarios and lacking interpretability and modular internal reasoning. They are unable to perform compositional generalization and adaptive robust reasoning in the manner of the human brain.

**Key Challenge**: Monolithic architectures couple all functionality within a shared parameter space, resulting in entangled internal representations that are difficult to optimize in a targeted manner, where local updates can produce unintended downstream effects.

**Goal**: To systematically translate cortical organizational principles from neuroscience into AI architecture design, realizing a modular, interpretable, and robust perceptual system.

**Key Insight**: The paper departs from three core organizational principles of the cerebral cortex — modular specialization, predictive coding feedback, and cross-modal integration — to propose an actionable architectural blueprint.

**Core Idea**: Replace monolithic black-box networks with cortically inspired modularity, predictive feedback, and cross-modal integration, bringing AI perceptual architectures closer to the division-of-labor paradigm observed in the human brain.

## Method

### Overall Architecture
The proposed architecture consists of four core components forming an iterative cycle of "encoding → coordination → hypothesis → feedback": (1) modality-specific dedicated encoders as the perceptual front end, (2) a shared cross-modal latent space as a semantic convergence region, (3) a routing controller that determines which modules to activate, and (4) a recursive predictive feedback loop enabling top-down hypothesis verification and refinement.

### Key Designs

1. **Dedicated Encoder Modules**:

    - Function: Equip each modality (vision / speech / language) with an independent dedicated encoder.
    - Mechanism: Employ pretrained expert networks (e.g., Whisper for speech, ViT for vision, LLaMA for text), with each module trained and tuned independently.
    - Design Motivation: Emulates the modality-specific processing of early sensory areas in the cerebral cortex. Independent modules allow new capabilities to be added without retraining the entire system, and failure of a single module does not cause overall system collapse.

2. **Shared Cross-Modal Latent Space**:

    - Function: Map the outputs of individual encoders into a shared semantic space to achieve cross-modal alignment.
    - Mechanism: Draws on the contrastive learning alignment approach of CLIP/ImageBind, but is positioned as a dynamic workspace rather than a static alignment layer, supporting zero-shot cross-modal transfer.
    - Design Motivation: Emulates multimodal information integration in associative regions such as STS/PPC in the brain, enabling flexible context-dependent integration while preserving modularity.

3. **Routing Controller**:

    - Function: Dynamically determine which dedicated modules to activate based on input modality, task context, and latent representation characteristics.
    - Mechanism: Operates via a sparse activation mechanism analogous to MoE, but at the level of modular reasoning rather than purely computational efficiency.
    - Design Motivation: Emulates the brain's ability to flexibly allocate cortical processing resources according to context and behavioral goals.

4. **Recursive Predictive Feedback Loop**:

    - Function: Higher-level modules generate top-down predictions that constrain lower-level processing.
    - Mechanism: Grounded in predictive coding theory, reasoning modules form predictive hypotheses and iteratively refine perception through prediction error signals.
    - Design Motivation: Reframes "hallucination" as a provisional hypothesis under generative reasoning rather than a one-shot error, enabling hypothesis verification and correction through iterative feedback.

### Loss & Training
No specific end-to-end training strategy is provided at the architectural blueprint level. The proof-of-concept (PoC) experiments train sparse autoencoders using standard MSE reconstruction loss.

## Key Experimental Results

### Main Results
PoC experiment: sparse autoencoders (SAE) are trained on layer-15 activations (4096-dim) of Mistral-7B, comparing monolithic versus modular decomposition.

| Configuration | Within-Domain Jaccard↑ | Feature-Domain Entropy↓ | MSE | Domain-Exclusive Feature % |
|---|---|---|---|---|
| Monolithic SAE (4096→1024→4096) | 55.7% | 3.52±0.01 (random baseline) | 0.0031 | 5.0±1.0% |
| Modular SAE (4×256) | 71.1% (+15.4pp) | 3.23 (p<0.01) | 0.0026 | 6.2% |

### Ablation Study

| Configuration | Within-Domain Jaccard | Feature-Domain Entropy | Notes |
|---|---|---|---|
| Modular (4×256) | 71.1% | 3.23 | Modular decomposition |
| Capacity-matched monolithic (256) | — | 2.70 | Entropy reduction partially explained by capacity constraint |
| Full monolithic (1024) | 55.7% | 3.52 | Baseline |

### Key Findings
- The primary effect of modular decomposition is improved within-domain consistency (+15.4pp Jaccard) rather than hard feature partitioning.
- Consistent improvements are observed across all four domains: vision +15.0pp, language +3.8pp, cross-modal +17.4pp, reasoning +25.4pp.
- Entropy reduction is partially attributable to capacity constraints (capacity-matched control entropy = 2.70 vs. modular 3.23), yet the within-domain stability gain is independent of capacity.
- The proportion of domain-exclusive features remains low (6.2% vs. 5.0% baseline), indicating that features continue to rely predominantly on distributed representations.

## Highlights & Insights
- **Predictive Coding Reinterpretation of Hallucination**: AI hallucinations are reconceptualized as provisional hypotheses under predictive reasoning, drawing analogies to dreaming and expectation-driven illusions in biological perception. This perspective suggests a novel mitigation strategy — iterative verification rather than post hoc filtering.
- **Modularity Does Not Imply Hard Partitioning**: The PoC experiments demonstrate that cortical-style modularity manifests primarily as improved consistency of within-domain activation bias rather than strictly mutually exclusive feature assignment, which aligns with neuroscientific observations.

## Limitations & Future Work
- The experimental scale is minimal (200 prompts, 4 domains, 50 per domain), serving only as diagnostic validation rather than a complete architectural implementation.
- Ground-truth semantic labels are used for routing; no learned routing mechanism is addressed.
- The complete architecture (dedicated encoders + routing control + predictive feedback) is not implemented; only the latent feature decomposition component is validated.
- Quantitative comparisons with existing modular methods (NMN, MoE, RIM) are absent.
- The paper is more position-paper in nature; engineering feasibility and large-scale validation remain for future work.

## Related Work & Insights
- **vs. Neural Module Networks (NMN)**: NMN relies on an external symbolic parser to determine module structure; this paper proposes replacing it with a learned routing controller, though this component is not yet implemented.
- **vs. MoE Models**: Sparse expert activation in MoE is primarily motivated by computational efficiency, whereas this paper positions modularity as a representational and reasoning prior.
- **vs. JEPA**: LeCun's predictive architecture emphasizes learning abstract world models through prediction, resonating with the predictive feedback principle proposed here, yet JEPA remains a monolithic structure.

## Rating
- Novelty: ⭐⭐⭐ Neuroscience-inspired AI architectures are not a new concept, but systematically integrating three cortical principles into an actionable blueprint represents a moderate contribution.
- Experimental Thoroughness: ⭐⭐ Only a minimal-scale PoC experiment is presented; the complete architecture is not implemented.
- Writing Quality: ⭐⭐⭐⭐ The survey sections bridging neuroscience and AI are clearly written, with effective interdisciplinary exposition.
- Value: ⭐⭐⭐ Provides meaningful design insights, though the absence of comprehensive validation limits direct impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Modal Logical Neural Networks for Financial AI](modal_logical_neural_networks_for_financial_ai.md)
- [\[ICLR 2026\] Position: The Reasoning Trap — Logical Reasoning as a Mechanistic Pathway to Advanced AI Self-Awareness](the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_advanced_jai.md)
- [\[ACL 2026\] Context-Value-Action Architecture for Value-Driven Large Language Model Agents](../../ACL2026/interpretability/context-value-action_architecture_for_value-driven_large_language_model_agents.md)
- [\[NeurIPS 2025\] AgentiQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation](../../NeurIPS2025/interpretability/agentiql_an_agent-inspired_multi-expert_framework_for_text-to-sql_generation.md)
- [\[AAAI 2026\] Can LLMs Truly Embody Human Personality? Analyzing AI and Human Behavior Alignment in Dispute Resolution](../../AAAI2026/interpretability/can_llms_truly_embody_human_personality_analyzing_ai_and_human_behavior_alignmen.md)

</div>

<!-- RELATED:END -->
