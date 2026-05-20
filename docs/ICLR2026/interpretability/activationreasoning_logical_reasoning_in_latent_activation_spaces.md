---
title: >-
  [Paper Note] ActivationReasoning: Logical Reasoning in Latent Activation Spaces
description: >-
  [ICLR 2026][Interpretability][Sparse Autoencoders] This paper proposes the ActivationReasoning (AR) framework, which embeds explicit logical reasoning into the latent activation space of LLMs (via SAE-extracted features)…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Sparse Autoencoders"
  - "Logical Reasoning"
  - "Latent Space Intervention"
  - "Concept Composition"
  - "Model Steering"
date: 2026-05-08
content_hash: 215551029c84f595
---

# ActivationReasoning: Logical Reasoning in Latent Activation Spaces

**Conference**: ICLR 2026
**arXiv**: [2510.18184](https://arxiv.org/abs/2510.18184)  
**Code**: [https://github.com/ml-research/ActivationReasoning](https://github.com/ml-research/ActivationReasoning)  
**Area**: LLM Interpretability / Reasoning
**Keywords**: Sparse Autoencoders, Logical Reasoning, Latent Space Intervention, Concept Composition, Model Steering

## TL;DR
This paper proposes the ActivationReasoning (AR) framework, which embeds explicit logical reasoning into the latent activation space of LLMs (via SAE-extracted features) through a three-stage pipeline: discovering concept representations → detecting activated propositions → reasoning with logical rules. The framework supports multi-hop reasoning, concept composition, and safety control, achieving 95%+ accuracy on PrOntoQA with an 8B model, surpassing GPT-4o.

## Background & Motivation

**Background**: SAEs have made the hidden activations of LLMs more interpretable, exposing latent features aligned with human-understandable concepts. Reasoning-oriented LLMs (e.g., o1, R1) improve performance by extending reasoning chains, but their reasoning processes remain opaque.

**Limitations of Prior Work**: SAE features are passive and brittle — they may be polysemous, context-unstable, or overly low-level. A critical deficiency is that SAEs lack mechanisms for compositional and higher-order reasoning. They cannot derive "Golden Gate Bridge" from "bridge" + "San Francisco" + "USA."

**Key Challenge**: Logical reasoning requires discrete propositional units and compositional rules, whereas LLMs rely on continuous, entangled representations. While SAEs provide approximately discrete features, they lack a formal framework for reasoning.

**Goal**: To embed explicit logical reasoning capabilities into the latent space of LLMs, enabling interpretable and controllable structured reasoning.

**Key Insight**: Treating SAE features as logical propositions, defining and applying logical rules (conjunction, disjunction, implication, negation) over them, and deriving new higher-order propositions via forward-chaining inference.

**Core Idea**: Treat SAE features as propositions, use user-defined logical rules as an inference engine, perform forward-chaining reasoning in the activation space, and steer LLM generation via activation guidance.

## Method

### Overall Architecture
A three-stage pipeline: (1) identify concept representations in the SAE space and construct a concept dictionary $\mathcal{D}$; (2) at inference time, detect token-level activations and map them to logical propositions forming an activation matrix $A$; (3) apply logical rules to $A$ to derive new propositions, yielding an augmented matrix $A'$ used for downstream analysis and LLM steering.

### Key Designs

1. **Three Forms of Concept Representation**:

    - **Function**: Organizes SAE features into latent representations of concepts.
    - **Mechanism**: Single-feature $\mathcal{R}_{single}$ (one SAE feature = one concept), multi-feature $\mathcal{R}_{multi}$ (weighted aggregation of multiple SAE features), and relational feature $\mathcal{R}_{relation}$ (decision tree modeling structured interactions among features). Automatic extraction is performed via $r_c = \arg\max(\mathbb{E}[l_t|y=1] - \mathbb{E}[l_t|y=0])$.
    - **Design Motivation**: The single-feature assumption often fails due to polysemy; multi-feature representations cannot model interactions (e.g., "hate" requires co-activation of defamation and stereotyping while excluding educational contexts); relational features use decision trees to balance expressiveness and interpretability.

2. **Activation Propositionalization and Logical Reasoning**:

    - **Function**: Converts token-level activations into logical propositions and performs forward-chaining inference.
    - **Mechanism**: Activation matrix $A_{local}[c,t] = \max(a_{c,t} - \tau_c, 0)$, $A_{global}[c] = \max(\text{Agg}_{t} a_{c,t} - \tau_c, 0)$. Users define logical rules such as "Bridge ∧ SF ∧ USA → Golden Gate Bridge," and forward chaining runs until a fixed point.
    - **Design Motivation**: The SAE feature space may lack a direct feature for "Golden Gate Bridge," but possesses features for "bridge," "San Francisco," and "USA" — logical composition fills the expressiveness gap of SAEs.

3. **Activation Steering for Control**:

    - **Function**: Uses concepts activated in the derived matrix $A'$ to guide LLM generation.
    - **Mechanism**: $h' = h + \alpha \cdot \frac{(SAE_D[r_c] \times w) \times \|h\|_2}{\|SAE_D[r_c]\|_2}$, which promotes or suppresses specific concepts by adjusting activation vectors.
    - **Design Motivation**: Pure analysis already provides significant value, but the control capability elevates AR from an interpretability tool to an alignment tool — enabling enforcement of safety constraints at inference time.

### Loss & Training
AR requires no LLM training. Concept extraction employs simple statistical methods (mean difference, decision trees). Rules are user-defined. No additional training cost is incurred at inference time.

## Key Experimental Results

### Main Results

**PrOntoQA Multi-Hop Reasoning (Accuracy% ↑):**

| Model | 1-hop | 3-hop | 5-hop |
|-------|-------|-------|-------|
| Llama-3.1-8B | 51.0 | 50.8 | 50.3 |
| **+ AR** | **95.0** | **95.6** | **95.3** |
| Gemma-2-9B | 48.5 | 47.5 | 47.9 |
| **+ AR** | **93.5** | **93.5** | **93.5** |
| GPT-4o | 95.5 | 88.0 | 79.5 |
| DeepSeek-R1-8B | 86.0 | 79.5 | 67.5 |

**Rail2Country Meta-Concept Generalization:**

| Model | Explicit Concepts | Meta-Concepts (Figurative) |
|-------|------------------|--------------------------|
| Llama-3.1-8B | 41.0 | 29.7 |
| **+ AR** | **74.7** | **62.7** |

### Ablation Study

| Concept Representation Type | BeaverTails Safety Detection F1 |
|----------------------------|----------------------------------|
| $\mathcal{R}_{single}$ | Lower |
| $\mathcal{R}_{multi}$ | Moderate |
| $\mathcal{R}_{relation}$ | **Highest** |

### Key Findings
- AR enables 8B models to surpass GPT-4o and DeepSeek-R1 on multi-hop reasoning: 8B+AR (95.3%) vs. GPT-4o (79.5%) at 5-hop.
- Critically, AR's performance does not degrade with reasoning depth, whereas all baseline models (including GPT-4o) exhibit significant accuracy drops as hop count increases.
- Meta-concept generalization (e.g., "a color like a tomato" → "red") validates AR's capability beyond literal matching.
- On the BeaverTails safety task, $\mathcal{R}_{relation}$ outperforms $\mathcal{R}_{single}$ and $\mathcal{R}_{multi}$, indicating that safety concepts require structured feature interactions.

## Highlights & Insights
- **SAE Features as a Bridge to Logical Propositions**: This represents the most natural connection between the continuous representations of neural networks and the discrete propositions of symbolic reasoning — SAE features are themselves designed to be approximately monosemantic, making them naturally suited as propositions.
- **8B Surpassing GPT-4o in Reasoning**: This is achieved not through better training but by layering logical reasoning atop existing representations — the model already "knows" the answer but lacks the capacity for compositional inference.
- **Modularity and Auditability**: The entire reasoning chain is transparent — where concepts originate, how rules are applied, and how conclusions are reached are all inspectable and modifiable at every step.
- **Cross-Model Transferability**: The same framework is effective across Llama and Gemma, suggesting that the propositionalization of SAE features is model-agnostic.

## Limitations & Future Work
- Logical rules must be manually defined by users; automatic rule discovery is an important direction for future work.
- Concept extraction relies on token-level labeled data, and cross-domain generalization may require new annotations.
- The current framework supports only propositional logic; extension to first-order logic (with quantifiers and variables) remains unexplored.
- SAE feature quality directly affects AR performance — if features are insufficiently monosemantic, reasoning may be unreliable.
- The computational overhead of rule application scales with the number of concepts and rules.

## Related Work & Insights
- **vs. Reasoning LLMs (o1, R1)**: Reasoning LLMs improve through chain-of-thought but remain opaque; AR performs reasoning in the activation space, where each step is auditable.
- **vs. Neuro-Symbolic Methods (DeepProbLog)**: Traditional neuro-symbolic approaches require end-to-end differentiable training; AR applies rules directly at inference time without model training.
- **vs. SAE Analysis (Anthropic)**: SAEs are typically used for passive analysis and feature visualization; AR actively leverages SAE features for reasoning and control.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The idea of embedding logical reasoning into the latent space of LLMs is both natural and powerful, representing a significant extension of SAE applications.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four complementary tasks (multi-hop reasoning / meta-concepts / natural language inference / safety) with dual-model validation.
- Writing Quality: ⭐⭐⭐⭐⭐ — The narrative flows smoothly from motivation to method to experiments; the Golden Gate Bridge running example is woven throughout the paper.
- Value: ⭐⭐⭐⭐⭐ — Provides a novel paradigm for interpretable reasoning and controllable alignment in LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Reasoning Trap — Logical Reasoning as a Mechanistic Pathway to Situational Awareness](the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_situational_.md)
- [\[ICLR 2026\] Modal Logical Neural Networks for Financial AI](modal_logical_neural_networks_for_financial_ai.md)
- [\[ICLR 2026\] GAVEL: Towards Rule-Based Safety through Activation Monitoring](gavel_towards_rule-based_safety_through_activation_monitoring.md)
- [\[ACL 2026\] Forest Before Trees: Latent Superposition for Efficient Visual Reasoning](../../ACL2026/interpretability/forest_before_trees_latent_superposition_for_efficient_visual_reasoning.md)
- [\[ICLR 2026\] Universal Properties of Activation Sparsity in Modern Large Language Models](universal_properties_of_activation_sparsity_in_modern_large_language_models.md)

</div>

<!-- RELATED:END -->
