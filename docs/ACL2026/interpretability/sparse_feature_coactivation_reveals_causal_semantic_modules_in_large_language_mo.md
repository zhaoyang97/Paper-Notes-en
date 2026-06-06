---
title: >-
  [Paper Note] Sparse Feature Coactivation Reveals Causal Semantic Modules in Large Language Models
description: >-
  [ACL2026][Interpretability][Sparse Autoencoders] The paper automatically discovers semantic modules representing concepts and relations in LLMs using cross-layer co-activation graphs of SAE features from a small number o…
tags:
  - "ACL2026"
  - "Interpretability"
  - "Sparse Autoencoders"
  - "Feature Coactivation"
  - "Semantic Modules"
  - "Causal Intervention"
  - "Model Steering"
date: 2026-05-08
content_hash: 6746f5a53bceab30
---

# Sparse Feature Coactivation Reveals Causal Semantic Modules in Large Language Models

**Conference**: ACL2026  
**arXiv**: [2506.18141](https://arxiv.org/abs/2506.18141)  
**Code**: https://github.com/shanestorks/SAE-Semantic-Modules  
**Area**: LLM Interpretability / Mechanistic Interpretability / Knowledge Representation  
**Keywords**: Sparse Autoencoders, Feature Coactivation, Semantic Modules, Causal Intervention, Model Steering

## TL;DR
The paper automatically discovers semantic modules representing concepts and relations in LLMs using cross-layer co-activation graphs of SAE features from a small number of prompts, demonstrating that ablating or amplifying these modules predictably manipulates the relational reasoning output of Gemma 2 2B in up to 98% of single-concept/relation scenarios and up to 90% of compositional scenarios.

## Background & Motivation
**Background**: Mechanistic interpretability has recently utilized Sparse Autoencoders (SAEs) to extract interpretable features from LLM activations. SAEs decompose dense activations into sparse, monosemantic feature directions, allowing researchers to analyze how specific concepts, entities, or behaviors are activated within the model.

**Limitations of Prior Work**: Individual SAE features often remain polysemantic or unstable. While cross-layer transcoders and circuit tracing can map information flow, the resulting graphs may contain hundreds of nodes with dense connections, requiring significant manual effort to interpret. Furthermore, relational knowledge reasoning—for example, "The capital of China is Beijing"—involves the composition of the concept "China" and the relation "capital city," which is difficult to explain via a single-layer linear direction or a single feature.

**Key Challenge**: LLM knowledge may be distributed across multiple features and layers. The components that actually function are not isolated features but groups of functional modules that co-activate across layers. Interpretability methods must be sufficiently fine-grained while remaining automated and causally verifiable.

**Goal**: The authors aim to construct inter-layer feature networks from SAE activations using very few prompts, automatically extract components that are semantically and contextually consistent and have a causal effect on output, and test whether these components can compositionally steer model outputs.

**Key Insight**: If multiple SAE features activate with high correlation in adjacent layers on the same prompt token sequence, they likely belong to the same computational module. By filtering out high-density general features, sparser and more interpretable weakly connected components can be obtained.

**Core Idea**: Treat SAE features as graph nodes and use cross-layer co-activation correlations as edges. Connected components are then treated as concept or relation modules, and their causal functions are validated through ablation and amplification.

## Method
The pipeline starts with residual stream activations from Gemma 2 2B, organizing the top active features of each layer's SAE into a cross-layer graph. Components are not manually specified (e.g., "China feature"), but emerge from co-activation relationships under specific prompts, and are validated post-hoc via KL divergence and steering success.

### Overall Architecture
The input consists of a set of concept-relation prediction prompts, such as country facts, word translation, and verb transformation. After the model generates correct answers, SAE activations are collected during the forward pass, taking the top-5 features per token. An edge is created if the Pearson correlation $\rho$ between the activation sequences of two features in adjacent layers exceeds 0.9. High-density general features with $d_{\ell,i} > 0.01$ (based on Neuronpedia activation density) are removed. Weakly connected components are then found using NetworkX BFS. Finally, ablation, amplification, and compositional steering are performed on these components to check if the output token distribution changes as expected.

### Key Designs
1.  **Cross-Layer Feature Coactivation Graph**:
    - **Function**: Organizes discrete SAE features into an interpretable cross-layer network structure.
    - **Mechanism**: Takes the top activated features per layer as nodes $(\ell,i)$. A directed edge is established if the correlation coefficient $\rho > 0.9$ between two features in adjacent layers over the prompt token dimension. The resulting graph captures whether features work together in the same context, rather than relying solely on feature descriptions.
    - **Design Motivation**: Feature descriptions can be unreliable; for instance, a "Spain component" might not have "Spain" in its description, and a "translation language component" might even be described as "programming." The co-activation graph bypasses text descriptions and directly utilizes internal model dynamics.

2.  **Sparsity Pruning and Component Extraction**:
    - **Function**: Filters out general activation features to retain more likely monosemantic and task-related modules.
    - **Mechanism**: Uses activation density from Neuronpedia to keep only sparse features ($d_{\ell,i} \leq 0.01$) and removes isolated nodes. Weakly connected components are then extracted. For country facts and translation, concept components are defined as the intersection of components across multiple relations for the same concept, and relation components are defined as the intersection of components across multiple concepts for the same relation.
    - **Design Motivation**: High-density features may carry syntax or general computation; including them directly would make components uninterpretable. The intersection strategy emphasizes context consistency, ensuring the "China component" remains stable across capital, currency, and language prompts.

3.  **Causal Intervention and Compositional Steering**:
    - **Function**: Verifies whether components truly determine model output rather than just being correlated with the task.
    - **Mechanism**: In-prompt components are ablated by zeroing their SAE feature activations and reconstructing them via the decoder for the forward pass. Target components are amplified by scaling their activations based on the maximum observed activation ratio. Success is measured by whether the output shifts toward the target concept, relation, or combination.
    - **Design Motivation**: Only components that can be intervened upon to cause predictable output changes can be called causal semantic modules. Compositional steering further tests whether concepts and relations are composable rather than entangled into a single direction.

### Loss & Training
No new models were trained. The study uses Gemma 2 2B and pretrained Gemma Scope JumpReLU SAEs. Key hyperparameters include selecting top-$k=5$ activated features per token per layer, a cross-layer correlation threshold $\tau_{corr}=0.9$, and a feature density threshold $\tau_{density}=0.01$. Interventions are implemented using TransformerLens, and success rates are evaluated on zero-shot prompt templates different from those used during activation collection to test contextual generalization.

## Key Experimental Results

### Main Results

| Task | # Concepts / # Relations | Base Model Accuracy | Avg. Concept Steering SR | Avg. Relation Steering SR | Avg. Compositional Steering SR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Country facts | 10 countries / 3 facts | 100% | 96% | 93% | 90% |
| Word translation | 11 words / 3 languages | 100% | 75% | 98% | 64% |
| Verb transformation | 8 verbs / 5 relations | 100% | 48% | 23% | 19% |
| Single Best | Various relations | 100% | up to 98% | up to 100% | up to 100% |
| Overall Conclusion | Three task types | 100% | up to 98% | up to 98% | up to 90% |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full component, country facts | Concept/Rel/Comp SR = 96% / 93% / 90% | Steering the complete semantic module |
| Single most causal feature baseline | Concept/Rel/Comp SR = 83% / 83% / 75% | Intervening only on the single feature with the largest KL divergence |
| Ablate country-fact components | Other country facts acc 1.00, translation 0.93, verb 1.00 | Most modules are task-specific |
| Ablate word-translation components | country facts 1.00, translation 0.83, verb 1.00 | Slight overlap between relations in the same task |
| Ablate verb-transformation components | country facts 0.73, translation 0.64, verb 0.63 | Verb modules are less specific, corresponding to lower steering success |
| Transcoder proof-of-concept | ~27% steering success | Method is transferable but significantly weaker than the SAE setup |

### Key Findings
- Usually, 2-3 components have significantly higher causal effects on the output token distribution, suggesting task-related computation is not uniformly spread across all activation features.
- Concept components appear earlier: 8/10 country components start from the first layer, as do word/verb components. Relation components are concentrated in later layers, often spanning the last 1/4 to 1/2 of the model.
- Components for country facts and translation show strong composability, allowing counterfactual steering like changing "Capital of China" to "Currency of Nigeria." Verb transformation (synonyms, antonyms, past tense) is more difficult because answers depend on semantics, POS tags, and context.
- Full components significantly outperform single features, supporting the view that concepts are not a single feature but a group of distributed functional directions.

## Highlights & Insights
- The most interesting aspect is the progression from "finding features" to "finding feature modules." This aligns better with the intuition of distributed representations in neural networks and is more stable than single-feature steering.
- The co-activation graph is a lightweight yet effective intermediate representation. it does not require full circuit tracing or training new probing models to organize cross-layer SAE activations into steerable structures.
- The discovery of concepts in early layers and relations in later layers is insightful, suggesting entity/lexical information is established early while abstract relations or operations are composed later.
- Compositional steering is meaningful for knowledge editing and safety control. If concepts and relations can be manipulated separately, it may enable fine-grained factual correction, style control, or behavioral constraints.

## Limitations & Future Work
- The experiments only cover 3 types of small-scale multi-relation tasks and require 100% base model accuracy. Generalization to open-ended QA, long-text reasoning, multi-hop facts, and non-objective relations remains unknown.
- Component selection still relies on manual heuristics. While extracting connected components is automated, choosing which ones represent a specific concept/relation and whether to use intersections or unions still depends on task experience.
- The model used is primarily Gemma 2 2B, with only supplementary results for Gemma 2 9B. Stability across different architectures, scales, and SAE qualities needs further verification.
- Retaining only low-density features excludes some critical general computation. The authors observed that ablating high-density features leads to grammatical and semantic collapse, indicating these features likely perform important but unexplained functions.
- The transfer to transcoders only achieved ~27% steering success, suggesting "co-activation components" is not yet a mature solution applicable to all dictionary/circuit tools.

## Related Work & Insights
- **vs. Single Feature SAE Analysis**: Single features are easier to interpret but incomplete. This paper emphasizes cross-layer feature groups to capture concepts and relations more comprehensively.
- **vs. Circuit Tracing / Cross-layer Transcoders**: Circuit tracing is more detailed but involves complex graphs and high costs. This paper uses feature co-activation for a more lightweight, module-level explanation.
- **vs. Knowledge Neurons / Causal Tracing**: Knowledge neurons focus on local neurons or layers. This paper focuses on SAE feature components and enables compositional counterfactual steering.
- **vs. Function Vectors / Relation Vectors**: Function vectors often treat relations as linear directions or functional representations in attention heads. This paper provides a feature-level modular perspective, explaining why certain relations like synonym/antonym are harder to modularize.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Automatically extracting causal semantic modules from SAE co-activation is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Intervention, composition, specificity, and layer distribution analyses are complete, but task and model scales are relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ The methodology is clear and experimental tables are comprehensive, though heuristic component selection could be more systematic.
- **Value**: ⭐⭐⭐⭐⭐ Provides strong insights for mechanistic interpretability, model editing, and controllable generation; public code enhances reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models](meter_evaluating_multi-level_contextual_causal_reasoning_in_large_language_model.md)
- [\[ICML 2026\] Towards Atoms of Large Language Models](../../ICML2026/interpretability/towards_atoms_of_large_language_models.md)
- [\[ACL 2026\] Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization](preference_heads_in_large_language_models_a_mechanistic_framework_for_interpreta.md)
- [\[ACL 2026\] DPN-LE: Dual Personality Neuron Localization and Editing for Large Language Models](dpn-le_dual_personality_neuron_localization_and_editing_for_large_language_model.md)
- [\[ACL 2026\] Embracing Anisotropy: Turning Massive Activations into Interpretable Control Knobs for Large Language Models](embracing_anisotropy_turning_massive_activations_into_interpretable_control_knob.md)

</div>

<!-- RELATED:END -->
