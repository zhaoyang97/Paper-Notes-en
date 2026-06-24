---
title: >-
  [Paper Note] PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics
description: >-
  [AAAI 2026 Oral][Interpretability][LLM Evaluation] This paper proposes the PragWorld benchmark, which evaluates the plasticity and robustness of LLMs' implicit world models by applying 7 types of minimal linguistic perturbations to dialogues. A dual-perspective interpretability framework is designed to localize harmful/helpful layers, and a layer regularization fine-tuning strategy is proposed to improve robustness.
tags:
  - "AAAI 2026 Oral"
  - "Interpretability"
  - "LLM Evaluation"
  - "World Model"
  - "Dialogue Understanding"
  - "Robustness"
date: 2026-05-08
content_hash: f7e120ac45f34b19
---

# PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics

**Conference**: AAAI 2026 Oral  
**arXiv**: [2511.13021](https://arxiv.org/abs/2511.13021)  
**Code**: [https://github.com/SachinVashisth/PRAGWORLD](https://github.com/SachinVashisth/PRAGWORLD)  
**Area**: Video Understanding
**Keywords**: LLM Evaluation, World Model, Dialogue Understanding, Robustness, Interpretability

## TL;DR

This paper proposes the PragWorld benchmark, which evaluates the plasticity and robustness of LLMs' implicit world models by applying 7 types of minimal linguistic perturbations to dialogues. A dual-perspective interpretability framework is designed to localize harmful/helpful layers, and a layer regularization fine-tuning strategy is proposed to improve robustness.

## Background & Motivation

### State of the Field
Transformer-based language models exhibit impressive language understanding capabilities learned solely from large amounts of unstructured text. Research has shown that LMs implicitly encode world knowledge such as color, gender, and spatial relationships, and can even represent chessboard states. This suggests that LMs may develop latent representations encoding world models.

### Limitations of Prior Work

**Fragility of implicit world models**: Vafa et al. (2024) demonstrate that the latent world models of LMs are brittle across tasks such as games, logic puzzles, and navigation.

**Uncertainty in entity tracking within dialogue**: Existing benchmarks primarily test entity tracking in static, unambiguous factual sequences, without accounting for conversational dynamics and pragmatic elements (e.g., coreference, implicature).

**Collapse under minimal perturbations**: Even minor lexical modifications—such as negation or quantifier substitution—can cause LMs to produce drastically different answers to identical questions, exposing the fragility of their world models.

### Root Cause
LLMs perform well on standard question answering, yet their ability to track and update entity states degrades sharply under minimal linguistic perturbations. This suggests that models may rely on surface-level patterns rather than genuinely constructing and maintaining world models.

### Starting Point
The paper designs 7 rigorously defined minimal lexical perturbations in conversational settings to evaluate whether LMs can adjust their internal world models accordingly. Mechanistic interpretability techniques are employed to localize the specific layers responsible for failures, and a targeted regularization method is proposed.

## Method

### Overall Architecture

The PragWorld benchmark begins with two conversational QA datasets (GRICE and CICERO) and applies 7 linguistic perturbations to dialogues through both manual and semi-automatic processes, constructing yes-no QA instances consisting of original–perturbed pairs. Multiple LMs are then evaluated using **robust accuracy** (requiring correct answers to both the original and perturbed instances), and a dual-perspective interpretability framework is used to analyze failure causes.

### Key Designs

#### 1. **7 Types of Minimal Linguistic Perturbations**

Each perturbation is designed to induce significant semantic change through minimal lexical modification:

- **Negation**: Negating auxiliary verbs (e.g., "he didn't" → "he did") to alter propositional truth value.
- **Variable Substitution**: Replacing one entity in the dialogue with another (e.g., "watermelons" → "oranges").
- **Quantity Change**: Modifying the quantity of countable nouns (e.g., "Two" → "Three").
- **Variable Swap**: Exchanging the positions of two entities.
- **Quantifier Change**: Manipulating quantifiers (e.g., "All" → "Some"), exploiting scalar implicature whereby "some" is interpreted as "not all."
- **Logical Connective Change**: Altering conjunctions (e.g., "and" → "or"), shifting from the assertion of two propositions to an exclusive disjunction.
- **Inconsistent Data Injection**: Injecting information that may violate common sense (e.g., "a birthday cake that changes color every time someone claps").

#### 2. **Dual-Split Dataset Construction**

**(a) Manual Split**: 500 perturbed dialogues are manually created from 77 seed dialogues (300 from GRICE, 200 from CICERO) and cross-validated by two annotators.

**(b) Synthetic Split**: Starting from 104 seed dialogues, GPT-4 is used to generate new dialogues; perturbations are then applied via deterministic algorithms, template questions are automatically generated, and answers are manually annotated, yielding 2,114 instances.

Question types include: quantity questions, universal quantifier questions, and existential quantifier questions.

#### 3. **Dual-Perspective Interpretability Framework**

**(a) Direct Effect Patching**: During the original forward pass, the residual stream activations at a specified layer are replaced with those from the perturbed forward pass, and the resulting change in output probability is observed. This measures each layer's capacity to encode perturbation information.

$$DE(R_\ell^{\hat{x}} \rightarrow R_\ell^x) = P(\hat{y}^{gold} \mid x+q; \text{patch}_\ell(R_\ell^{\hat{x}})) - P(\hat{y}^{gold} \mid x+q)$$

**(b) MLP Zero-Out Ablation**: The output of the MLP submodule at a specified layer is set to zero, and the resulting change in accuracy is observed. A drop in accuracy indicates the layer is "helpful," while an increase indicates the layer is "harmful" (i.e., it encodes spurious signals or shortcut patterns).

**Findings**: For the Phi-3.5 model, layers 2, 9, and 16 are identified as helpful, while layers 5, 6, 7, 11, 13, 17, and 31 are identified as harmful. Logical connective perturbations are most affected by harmful layers, while variable swap perturbations are least affected.

#### 4. **Layer Regularization Fine-Tuning Strategy**

Based on findings from the interpretability analysis, two regularization methods are proposed:

**(a) Useful Layer Amplification (ULA)**: A two-layer classification head is appended to the MLP output of each helpful layer. The ULA loss is the mean of classification losses across all helpful layers, added to the next-token prediction loss with weight $\alpha$.

**(b) Harmful Layer Suppression (HLS)**: An L2 penalty is applied to the MLP residual output of each harmful layer to suppress its activation magnitude.

### Loss & Training

Fine-tuning loss = standard next-token prediction loss + $\alpha \cdot \mathcal{L}_{ULA}$ (or $\beta \cdot \mathcal{L}_{HLS}$)

Models are fine-tuned on the synthetic split and evaluated on the manual split to verify generalization.

## Key Experimental Results

### Main Results

| Model | Parameters | Robust Acc. (Manual) | Robust Acc. (Synthetic) | Original Acc. | Perturbed Acc. |
|-------|------------|----------------------|-------------------------|---------------|----------------|
| GPT-3.5 | — | 42.86 | 67.21 | 71.43 | 70.90 |
| DeepSeek-Inst | 16B | 46.94 | 60.93 | 75.51 | 74.13 |
| Phi-3-mini | 3.8B | 47.96 | 64.78 | 74.49 | 73.88 |
| Phi-3.5-mini | 3.8B | 48.98 | 63.97 | 78.57 | 74.13 |
| Llama-3.1-8B | 8B | 48.98 | 60.93 | 74.49 | 72.14 |
| Llama-3.2-1B | 1B | 14.29 | 47.77 | 48.98 | 48.76 |
| Qwen2.5-7B | 7B | 37.76 | 60.73 | 68.37 | 69.40 |

### Ablation Study

| Model | Robust Acc. (Before) | Robust Acc. (After) | Gain |
|-------|----------------------|---------------------|------|
| Llama-3.2-1B | 14.29 | 32.65 | +18.36 |
| Llama-3.2-3B | 20.41 | 48.98 | +28.57 |
| Llama-3.1-8B | 48.98 | 59.18 | +10.20 |
| Phi-3-mini | 47.96 | 50.00 | +2.04 |
| Qwen2.5-1.5B | 22.45 | 47.96 | +25.51 |
| Qwen2.5-7B | 37.76 | 55.10 | +17.34 |

### Key Findings

1. **All models lack robustness**: The best-performing model (Phi-3.5-mini) achieves only 48.98% robust accuracy on the manual split, meaning it fails to correctly answer both the original and perturbed instances in more than half of all cases.
2. **Severe Yes/No bias**: Smaller models (Llama-3.2-1B/3B, Qwen2.5-1.5B) exhibit a large gap between Yes and No accuracy, indicating strong answer preference biases.
3. **Fine-tuning is particularly effective for small models**: Llama-3.2-3B's robust accuracy jumps from 20.41% to 48.98% (+28.57%).
4. **Harmful layer suppression is effective**: After fine-tuning, the negative influence of harmful layers is significantly reduced, with the most notable improvement observed for logical connective perturbations.

## Highlights & Insights

1. **The "robust accuracy" metric is elegantly designed**: Requiring the model to simultaneously answer both the original and all perturbed instances correctly constitutes a stricter evaluation criterion than standard accuracy alone.
2. **Systematic design of 7 perturbation types**: The perturbations cover the primary dimensions of semantic variation (negation, quantity, entity, quantifier, logical connective, common sense), each strictly maintaining minimal lexical change.
3. **Interpretability-driven improvement strategy**: Rather than fine-tuning blindly, the approach first localizes problematic layers via mechanistic interpretability and then applies targeted regularization—a methodology with broad generalizability.
4. **Dual manual/synthetic split design**: The manual split ensures quality and diversity, while the synthetic split ensures scale and reproducibility.

## Limitations & Future Work

1. The dataset is relatively small (500 manual + 2,114 synthetic instances) and may not cover all pragmatic phenomena.
2. Only yes-no questions are evaluated; robustness in open-ended question answering settings may be even lower.
3. Seed dialogues are primarily drawn from two datasets (GRICE and CICERO), limiting domain diversity.
4. ULA and HLS require prior ablation experiments to identify helpful/harmful layers; the layer distribution may vary substantially across models, and generalizability remains to be verified.
5. The synthetic perturbation algorithm is deterministic and may not capture more subtle pragmatic variations found in natural language.

## Related Work & Insights

- **Toshniwal et al. (2022)**: Demonstrate that LMs can encode chessboard states, inspiring the present study's focus on world models.
- **Vafa et al. (2024)**: Establish the fragility of LM world models and serve as the direct precursor to this work.
- **GRICE & CICERO datasets**: Provide a conversational foundation incorporating pragmatic and commonsense reasoning.
- **Joshi et al. (2025) & Geva et al. (2023)**: The sources of direct effect patching and MLP ablation techniques.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of minimal linguistic perturbations, conversational dynamics, and interpretability represents a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 10 models × 2 splits + interpretability analysis + fine-tuning validation.
- Writing Quality: ⭐⭐⭐⭐ — Formal definitions are clear and experimental analyses are systematic.
- Value: ⭐⭐⭐⭐ — Provides valuable tools and methodology for evaluating LLM robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Physics from Video: Identifiability of Time-Invariant Second-Order ODEs under Minimal Trajectory Conditions](../../ICML2026/interpretability/physics_from_video_identifiability_of_time-invariant_second-order_odes_under_min.md)
- [\[NeurIPS 2025\] Evaluating LLMs in Open-Source Games](../../NeurIPS2025/interpretability/evaluating_llms_in_open-source_games.md)
- [\[ICML 2025\] Reactivation: Empirical NTK Dynamics Under Task Shifts](../../ICML2025/interpretability/reactivation_empirical_ntk_dynamics_under_task_shifts.md)
- [\[ACL 2026\] Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives](../../ACL2026/interpretability/do_llms_capture_embodied_cognition_and_cultural_variation_cross-linguistic_evide.md)
- [\[NeurIPS 2025\] LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning](../../NeurIPS2025/interpretability/llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)

</div>

<!-- RELATED:END -->
