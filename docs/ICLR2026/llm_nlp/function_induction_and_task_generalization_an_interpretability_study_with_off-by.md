---
title: >-
  [Paper Note] Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition
description: >-
  [ICLR 2026][Causal Inference][mechanistic interpretability] Using off-by-one addition (e.g., 1+1=3, 2+2=5) as a counterfactual task, this paper applies path patching to reveal a **function induction** mechanism within la…
tags:
  - "ICLR 2026"
  - "Causal Inference"
  - "mechanistic interpretability"
  - "in-context learning"
  - "induction heads"
  - "function vectors"
  - "task generalization"
  - "path patching"
date: 2026-05-08
content_hash: 69400af4439ce914
---

# Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition

**Conference**: ICLR 2026
**arXiv**: [2507.09875](https://arxiv.org/abs/2507.09875)  
**Code**: [INK-USC/function-induction](https://github.com/INK-USC/function-induction)  
**Area**: Causal Inference
**Keywords**: mechanistic interpretability, in-context learning, induction heads, function vectors, task generalization, path patching

## TL;DR

Using off-by-one addition (e.g., 1+1=3, 2+2=5) as a counterfactual task, this paper applies path patching to reveal a **function induction** mechanism within large language models — an attention head circuit that performs inductive reasoning at the function level, transcending token-level pattern matching — and demonstrates that this mechanism is reused across tasks.

## Background & Motivation

**Importance of task-level generalization**: As LLM deployment scenarios continue to expand, incorporating all tasks into training data prior to deployment is infeasible, making the ability to handle unseen tasks via in-context learning (ICL) at inference time critically important.

**Limitations of prior understanding**: Existing mechanistic interpretability work on ICL has focused primarily on induction heads (token-level copy-paste, i.e., [A][B]...[A]→[B]) and function vectors (single-step mapping tasks such as country→capital), leaving complex generalization scenarios involving **multi-step reasoning** or **novel defined concepts** underexplored.

**Elegant design of off-by-one addition**: The task comprises two steps — standard addition followed by an unexpected +1 operation (i.e., 1+1=3) — forming a counterfactual, multi-step compositional task. A model either learns to apply +1 and output 7 (successful generalization) or adheres to arithmetic rules and outputs 6 (failed generalization).

**Empirically motivated analysis**: Six mainstream LLMs (Llama-2/3, Mistral, Gemma-2, Qwen-2.5, Phi-4) all perform effectively on this task, with accuracy monotonically increasing with the number of shots, motivating a deeper investigation into the underlying mechanisms.

**From token induction to function induction**: Traditional induction heads induce a zeroth-order constant function $f = \text{output}([B])$; this paper seeks to determine whether models can induce a first-order function $f(x) = x + 1$, thereby elevating mechanistic understanding from the token level to the function level.

**Need to verify cross-task reuse**: If function induction is a general-purpose mechanism, it should be reused across tasks with similar structure but entirely different sub-steps, which has important implications for understanding model compositionality and flexibility.

## Method

### Overall Architecture

This paper employs **mechanistic interpretability with path patching**, using Gemma-2 (9B) as the primary subject of analysis. By contrasting activation propagation between a base prompt (standard addition, 1+1=2) and a contrast prompt (off-by-one addition, 1+1=3), the study traces the computational origin of the +1 function layer by layer, ultimately identifying a circuit composed of three groups of attention heads.

### Key Design 1: Circuit Discovery via Path Patching

- **Function**: Forward passes are conducted separately on the base prompt $x_{base}$ and contrast prompt $x_{cont}$; partial activations from $M(\cdot|x_{base})$ are substituted into $M(\cdot|x_{cont})$ to observe whether the output reverts from "3+3=7" to "3+3=6."
- **Mechanism**: The logit difference $F(C, x) = C(y_{base}|x) - C(y_{cont}|x)$ is defined, and the normalized relative logit difference $r = \frac{F(M', x_{cont}) - F(M, x_{cont})}{F(M, x_{cont}) - F(M, x_{base})}$ quantifies the substitution effect; the closer $r$ is to $-100\%$, the greater a component's contribution to the +1 function.
- **Design Motivation**: Path patching precisely traces causal pathways of activations, enabling stepwise localization of information flow from the final output back to upstream components.

### Key Design 2: Discovery of Three Groups of Attention Heads

Layerwise path patching identifies three groups of attention heads:

| Group | Name | Function | Attention Pattern |
|-------|------|----------|-------------------|
| Group 1 | **Consolidation Heads** | Aggregate information and finalize output | Primarily attend to the current token and `<bos>` |
| Group 2 | **Function Induction (FI) Heads** | Carry the +1 function from ICL demonstrations to the test query | Attend to answer tokens $c_i$ of prior demonstrations at "=" positions |
| Group 3 | **Previous Token (PT) Heads** | Register the discrepancy between expected and actual answers at answer positions | Attend to the immediately preceding "=" token at position $c_i$ |

- **Mechanism**: FI Heads operate analogously to traditional induction heads but at the function level — whereas traditional induction heads copy token [B], FI heads induce the function $f(x) = x + 1$. PT Heads resemble traditional previous token heads, detecting the deviation between the model's expected answer and the actual answer in ICL demonstrations.
- **Design Motivation**: This hierarchical discovery process (Output → Group 1/2 → Group 3) allows the circuit structure to emerge naturally without relying on prior assumptions.

### Key Design 3: Function Vector Analysis

- **Function**: A naive prompt (e.g., "2=2\n3=?") is constructed; the output of FI heads is added to the residual stream, and changes in model logits are observed, generating a $10 \times 10$ heatmap.
- **Mechanism**: Each FI head writes a distinct "fragment" of the +1 function — for example, H39.7 promotes $x+1$, H28.6 suppresses $x-1$, H32.1 promotes numbers greater than $x$, and H24.9 suppresses $x$. The aggregated outputs of multiple heads implement the complete +1 function.
- **Design Motivation**: This validates that FI heads causally encode the +1 function rather than merely exhibiting statistical correlations.

### Loss & Training / Evaluation Metrics

This paper involves no training. Core evaluation metrics are:
- **Accuracy**: Correctness rate on the off-by-one addition task
- **Relative logit difference** $r$: Normalized logit difference measuring each circuit component's contribution to the +1 behavior

## Key Experimental Results

### Main Results: ICL Performance and FI Head Ablation

| Model | 4-shot Acc | 8-shot Acc | 16-shot Acc | After FI Head Ablation |
|-------|-----------|-----------|------------|------------------------|
| Llama-2 (7B) | ~15% | ~35% | ~55% | Reverts to standard addition |
| Mistral-v0.1 (7B) | ~20% | ~50% | ~65% | Reverts to standard addition |
| Gemma-2 (9B) | 33% | ~70% | 86% | 0% (off-by-one), 100% (standard) |
| Llama-3 (8B) | ~60% | ~95% | ~98% | Reverts to standard addition |
| Phi-4 (14B) | ~65% | ~98% | ~99% | Reverts to standard addition |

Ablating 6 FI heads causes the model to completely lose off-by-one capability (accuracy drops to 0%), while randomly ablating 6 heads has virtually no effect, demonstrating that FI heads are necessary components for the +1 function.

### Ablation Study: Cross-Task Generalization

| Task Pair | Base Task | Contrast Task | Contrast Acc (Full Model) | Contrast Acc (FI Heads Ablated) |
|-----------|-----------|---------------|--------------------------|----------------------------------|
| Off-by-2 Addition | Standard addition | +2 addition | Non-trivial | Substantial drop |
| Shifted MMLU | Standard MCQA | Answer shift +1 | Non-trivial | Substantial drop (non-zero residual) |
| Caesar Cipher (k=2) | ROT-0 | ROT-2 | Non-trivial | Substantial drop (non-zero residual) |
| Base-8 Addition | Decimal addition | Octal addition | Non-trivial | Substantial drop |

Key finding: The same FI heads are reused across all four task pairs, demonstrating the **flexibility and compositionality** of the function induction mechanism.

### Base-8 Addition Error Analysis

| Case | Description | Expected Behavior | Model Accuracy | Error Type |
|------|-------------|-------------------|----------------|------------|
| Case 1 | No carry | No adjustment | 93% | 7% over-generalization (adjusting when unnecessary) |
| Case 2 | Carry, both digits need adjustment | Adjust both digits | 16% | 84% under-generalization (failing to adjust) |
| Case 3 | Carry, only ones digit needs adjustment | Adjust ones digit only | 14% | 83% under-generalization |

This indicates that while models can induce a simple +2 function, they fail to handle conditionally triggered application (applying +2 only under specific conditions), exposing a bottleneck in current models' multi-step inductive reasoning.

### Key Findings

1. **Distributed function encoding**: The +1 function is not implemented by a single attention head but through the collaboration of 6–9 FI heads, each writing a distinct "fragment" of the function (promoting $x+1$, suppressing $x$, suppressing $x-1$, etc.).
2. **FI Heads ≠ FV Heads**: FI heads have no overlap with the function vector heads identified by Todd et al. (2024) — FV heads reside in early-to-middle layers (<20), while FI heads appear in late layers (29–31), indicating that FI heads are a mechanism specialized for subsequent steps in multi-step tasks.
3. **Cross-model generality**: The three-group head structure is identified in all four models examined (Gemma-2, Llama-2, Llama-3, Mistral), confirming that function induction is a universally emergent mechanism.

## Highlights & Insights

- **Conceptual innovation**: Extending induction heads from zeroth-order (token copying) to first-order (function induction, $f(x) = x+1$) represents a fundamental advance in understanding ICL mechanisms.
- **Elegant task design**: Off-by-one addition ingeniously combines counterfactual reasoning with arithmetic, enabling each step of the multi-step reasoning process to be traced independently.
- **Mechanism compositionality**: The same FI circuit is reused across tasks as diverse as arithmetic shifts, MCQA shifts, Caesar ciphers, and octal addition, suggesting the existence of a general-purpose "function shift" module within the model.
- **Implications for evaluation**: Analysis of base-8 addition reveals that models may achieve partial accuracy through unintended shortcut algorithms (performing decimal addition then applying +2), meaning accuracy-only evaluations may obscure reasoning deficiencies.

## Limitations & Future Work

1. **Imperfect circuit**: The identified circuit does not fully satisfy faithfulness and completeness criteria (which are often in tension with minimality).
2. **Attention heads only**: The role of MLP layers is not analyzed, nor are the internal QK/OV circuits of attention heads decomposed.
3. **Restricted function types**: Verification is limited to "shift-type" functions ($f(x) = x + k$); whether analogous mechanisms exist for more complex functions (e.g., nonlinear transformations) remains unexplored.
4. **Synthetic/algorithmic tasks only**: The function induction mechanism has not been validated in naturalistic text settings.
5. **Nonlinearity of number representations**: Number tokens in LLMs typically map to sinusoidal (Fourier) feature spaces rather than linear spaces, increasing the difficulty of interpretability analysis.
6. **Failure of conditional induction**: In base-8 addition, the model fails to trigger +2 under the correct conditions, indicating that current models have limited capacity for "two-step induction within a three-step task."

## Related Work & Insights

- **Induction Heads (Olsson et al., 2022)**: This paper directly extends the concept of induction heads from the token level to the function level, representing a natural generalization of that foundational finding.
- **Function Vectors (Todd et al., 2024; Hendel et al., 2023)**: FI heads and FV heads serve similar functions but occupy different layer positions; FI heads can be viewed as a specialization of the FV mechanism for later steps in multi-step tasks.
- **Latent Multi-step Reasoning**: This paper provides circuit-level evidence of implicit multi-step reasoning within models, complementing behavioral analyses based on multi-hop QA.
- **Implications for alignment**: The authors conjecture that behaviors such as sycophancy and agreement bias may share a similar structure — the model induces a "belief modification function" from context and applies it to output generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Extending induction heads from the token level to the function level is a conceptual breakthrough; the formalization of function induction carries significant theoretical value
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across 4 models and 4 task pairs, supported by ablations, causal interventions, and heatmap analyses; however, the identified circuit does not perfectly satisfy faithfulness/completeness
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, precise concept definitions, information-dense figures, and a running example sustained throughout the paper
- **Value**: ⭐⭐⭐⭐ — Deepens mechanistic understanding of ICL and implicit multi-step reasoning, with practical implications for model evaluation and pretraining design; limited to synthetic tasks, with natural-language validation yet to be provided

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)
- [\[ICLR 2026\] Validating Interpretability in siRNA Efficacy Prediction: A Perturbation-Based, Dataset-Aware Protocol](validating_interpretability_in_sirna_efficacy_prediction_a_perturbation-based_da.md)
- [\[ACL 2026\] Cross-Modal Taxonomic Generalization in (Vision-) Language Models](../../ACL2026/causal_inference/cross-modal_taxonomic_generalization_in_vision-_language_models.md)
- [\[NeurIPS 2025\] LLM Interpretability with Identifiable Temporal-Instantaneous Representation](../../NeurIPS2025/causal_inference/llm_interpretability_with_identifiable_temporal-instantaneous_representation.md)
- [\[ACL 2026\] Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation](../../ACL2026/causal_inference/parallel_universes_parallel_languages_a_comprehensive_study_on_llm-based_multili.md)

</div>

<!-- RELATED:END -->
