---
title: >-
  [Paper Note] Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition
description: >-
  [ICLR 2026][LLM/NLP][mechanistic interpretability] Using off-by-one addition (e.g., 1+1=3, 2+2=5) as a counterfactual task, this work applies path patching to reveal a **function induction** mechanism within large langua…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "mechanistic interpretability"
  - "in-context learning"
  - "induction heads"
  - "function vectors"
  - "task generalization"
  - "path patching"
date: 2026-05-08
content_hash: 5a1364ff960eb835
---

# Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition

**Conference**: ICLR 2026
**arXiv**: [2507.09875](https://arxiv.org/abs/2507.09875)
**Code**: [INK-USC/function-induction](https://github.com/INK-USC/function-induction)
**Area**: LLM/NLP
**Keywords**: mechanistic interpretability, in-context learning, induction heads, function vectors, task generalization, path patching

## TL;DR

Using off-by-one addition (e.g., 1+1=3, 2+2=5) as a counterfactual task, this work applies path patching to reveal a **function induction** mechanism within large language models — an attention head circuit that performs inductive reasoning at the function level, beyond token-level pattern matching — and demonstrates that this mechanism is reused across tasks.

## Background & Motivation

1. **Importance of task-level generalization**: As LLM application scenarios continue to expand, it is impractical to include all tasks in training data prior to deployment. The ability to complete unseen tasks via in-context learning (ICL) at inference time is therefore critical.
2. **Limitations of prior understanding**: Previous mechanistic interpretability work on ICL has focused primarily on induction heads (token-level copy-paste, i.e., [A][B]...[A]→[B]) and function vectors (single-step mapping tasks such as country→capital), leaving complex generalization scenarios involving **multi-step reasoning** or **novel concept definitions** underexplored.
3. **Elegant design of off-by-one addition**: This task consists of two steps — standard addition followed by an unexpected +1 operation (i.e., 1+1=3) — forming a counterfactual, multi-step compositional task. A model either learns to output 7 (successful generalization) or follows arithmetic conventions and outputs 6 (failed generalization).
4. **Empirical findings motivating deeper analysis**: Six mainstream LLMs (Llama-2/3, Mistral, Gemma-2, Qwen-2.5, Phi-4) all handle this task effectively, with accuracy monotonically increasing with the number of shots, motivating a deeper investigation into the underlying mechanisms.
5. **From token induction to function induction**: Traditional induction heads induce a zeroth-order constant function $f = \text{output}([B])$; this work seeks to reveal whether models can induce a first-order function $f(x) = x + 1$, thereby lifting the understanding from the token level to the function level.
6. **Need to verify cross-task reuse**: If function induction is a general-purpose mechanism, it should be reused across structurally similar tasks with entirely different sub-steps, which carries important implications for understanding compositionality and flexibility in language models.

## Method

### Overall Architecture

This work adopts a **mechanistic interpretability + path patching** methodology, with Gemma-2 (9B) as the primary subject of analysis. By contrasting activation propagation between a base prompt (standard addition, 1+1=2) and a contrast prompt (off-by-one addition, 1+1=3), the paper traces layer by layer the computational origin of the +1 function, ultimately identifying a circuit composed of three groups of attention heads.

### Key Design 1: Circuit Discovery via Path Patching

- **Function**: Forward passes are performed separately on the base prompt $x_{base}$ and contrast prompt $x_{cont}$; partial activations from $M(\cdot|x_{base})$ are substituted into $M(\cdot|x_{cont})$ to observe whether the output reverts from "3+3=7" to "3+3=6".
- **Mechanism**: The logit difference is defined as $F(C, x) = C(y_{base}|x) - C(y_{cont}|x)$, and the normalized relative logit difference $r = \frac{F(M', x_{cont}) - F(M, x_{cont})}{F(M, x_{cont}) - F(M, x_{base})}$ quantifies the substitution effect; $r$ approaching $-100\%$ indicates a larger contribution of the component to the +1 function.
- **Design Motivation**: Path patching precisely traces the causal pathway of activations, progressively localizing information flow from the final output back to upstream components.

### Key Design 2: Discovery of Three Groups of Attention Heads

Through layer-wise path patching, three groups of attention heads are identified:

| Group | Name | Function | Attention Pattern |
|-------|------|----------|-------------------|
| Group 1 | **Consolidation Heads** | Aggregate information and finalize output | Primarily attend to the current token and `<bos>` |
| Group 2 | **Function Induction (FI) Heads** | Carry the +1 function from ICL demonstrations to the test query | Attend to answer tokens $c_i$ of preceding demonstrations at "=" positions |
| Group 3 | **Previous Token (PT) Heads** | Register the "expected vs. actual discrepancy" at answer positions | Attend to the immediately preceding "=" token at position $c_i$ |

- **Mechanism**: FI Heads resemble traditional induction heads but operate at the function level — whereas traditional induction heads copy token [B], FI heads induce the function $f(x) = x + 1$. PT Heads resemble traditional previous token heads, detecting the deviation between the model's expected answer and the actual answer within ICL demonstrations.
- **Design Motivation**: This hierarchical discovery process (Output → Group 1/2 → Group 3) allows the circuit structure to emerge naturally without relying on prior assumptions.

### Key Design 3: Function Vector Analysis

- **Function**: A naive prompt (e.g., "2=2\n3=?") is constructed; the output of FI heads is added to the residual stream and the resulting changes in model logits are observed, generating a $10 \times 10$ heatmap.
- **Mechanism**: Each FI head writes a different "fragment" of the +1 function — for example, H39.7 promotes $x+1$, H28.6 suppresses $x-1$, H32.1 promotes numbers greater than $x$, and H24.9 suppresses $x$. The outputs of multiple heads aggregate to implement the complete +1 function.
- **Design Motivation**: This validates that FI heads causally encode the +1 function, rather than being merely statistically correlated with the behavior.

### Loss & Training

This work involves no training. The core evaluation metrics are:
- **Accuracy**: correctness rate on the off-by-one addition task
- **Relative logit difference** $r$: normalized logit difference measuring each circuit component's contribution to the +1 behavior

## Key Experimental Results

### Main Results: ICL Performance and FI Head Ablation

| Model | 4-shot Acc | 8-shot Acc | 16-shot Acc | After FI Head Ablation |
|-------|-----------|-----------|------------|------------------------|
| Llama-2 (7B) | ~15% | ~35% | ~55% | Reverts to standard addition |
| Mistral-v0.1 (7B) | ~20% | ~50% | ~65% | Reverts to standard addition |
| Gemma-2 (9B) | 33% | ~70% | 86% | 0% (off-by-one), 100% (standard) |
| Llama-3 (8B) | ~60% | ~95% | ~98% | Reverts to standard addition |
| Phi-4 (14B) | ~65% | ~98% | ~99% | Reverts to standard addition |

Ablating 6 FI heads completely eliminates the model's off-by-one capability (accuracy drops to 0%), while randomly ablating 6 heads has nearly no effect, demonstrating that FI heads are necessary components for the +1 function.

### Ablation Study: Cross-Task Generalization

| Task Pair | Base Task | Contrast Task | Contrast Acc (Full Model) | Contrast Acc (FI Head Ablation) |
|-----------|-----------|---------------|--------------------------|----------------------------------|
| Off-by-2 Addition | Standard addition | +2 addition | Non-trivial | Substantial drop |
| Shifted MMLU | Standard MCQA | Answer shift +1 | Non-trivial | Substantial drop (non-zero residual) |
| Caesar Cipher (k=2) | ROT-0 | ROT-2 | Non-trivial | Substantial drop (non-zero residual) |
| Base-8 Addition | Decimal addition | Octal addition | Non-trivial | Substantial drop |

Key finding: the same FI heads are reused across all four task pairs, demonstrating the **flexibility and compositionality** of the function induction mechanism.

### Base-8 Addition Error Analysis

| Case | Description | Correct Behavior | Model Accuracy | Error Type |
|------|-------------|-----------------|----------------|------------|
| Case 1 | No carry | No adjustment | 93% | 7% over-generalization (adjustment applied when not needed) |
| Case 2 | Carry; both ones and tens digits need adjustment | Adjust both digits | 16% | 84% under-generalization (adjustment not applied when needed) |
| Case 3 | Carry; only ones digit needs adjustment | Adjust ones digit only | 14% | 83% under-generalization |

This indicates that while models can induce a simple +2 function, they struggle with conditionally triggered application (applying +2 only under specific conditions), exposing a bottleneck in multi-step inductive reasoning.

### Key Findings

1. **Distributed function encoding**: The +1 function is not implemented by a single attention head but is collaboratively realized by 6–9 FI heads, each writing a different "fragment" of the function (promoting $x+1$, suppressing $x$, suppressing $x-1$, etc.).
2. **FI Heads ≠ FV Heads**: There is no overlap with the function vector heads identified by Todd et al. (2024) — FV heads reside in early-to-middle layers (<20), while FI heads reside in late layers (29–31), suggesting that FI heads are a mechanism specialized for subsequent steps in multi-step tasks.
3. **Cross-model universality**: The three-group head structure is identified in all four models examined (Gemma-2, Llama-2, Llama-3, Mistral), demonstrating that function induction is a broadly emergent mechanism.

## Highlights & Insights

- **Conceptual innovation**: Extending induction heads from zeroth order (copying tokens) to first order (inducing the function $f(x) = x+1$) represents a fundamental advancement in understanding ICL mechanisms.
- **Elegant task design**: Off-by-one addition cleverly combines counterfactual reasoning with arithmetic, enabling each step of multi-step reasoning to be tracked independently.
- **Mechanism compositionality**: The same FI circuit is reused across tasks as diverse as additive shifts, MCQA shifts, Caesar cipher, and octal addition, indicating that a general-purpose "function shift" module exists within the model.
- **Implications for evaluation**: The base-8 addition analysis reveals that models may achieve partial accuracy through an unintended shortcut algorithm (performing decimal addition then adding +2), meaning accuracy alone may conceal reasoning deficiencies.

## Limitations & Future Work

1. **Imperfect circuit**: The discovered circuit does not fully satisfy faithfulness and completeness criteria (which often trade off against minimality).
2. **Attention heads only**: The role of MLP layers is not analyzed in depth, nor are the internal QK/OV circuits of attention heads decomposed.
3. **Restricted function types**: Only "shift-type" functions ($f(x) = x + k$) are validated; whether analogous mechanisms exist for more complex functions (e.g., nonlinear transformations) remains unexplored.
4. **Synthetic/algorithmic tasks only**: The function induction mechanism is not validated on natural text.
5. **Nonlinearity of number representations**: Number tokens in LLMs are typically mapped to sinusoidal (Fourier) feature spaces rather than linear spaces, increasing the difficulty of interpretability analysis.
6. **Failure of conditional induction**: In base-8 addition, the model fails to trigger +2 under the correct conditions, indicating that current models have limited capacity for "two-step induction within a three-step task."

## Related Work & Insights

- **Induction Heads (Olsson et al., 2022)**: This work directly extends the concept of induction heads from the token level to the function level, representing a natural generalization of this classical finding.
- **Function Vectors (Todd et al., 2024; Hendel et al., 2023)**: FI heads and FV heads serve similar roles but differ in layer depth; FI heads can be viewed as a specialization of the FV mechanism for later steps in multi-step tasks.
- **Latent Multi-step Reasoning**: This work provides circuit-level evidence for implicit multi-step reasoning in models, complementing behavior-level analyses based on multi-hop QA.
- **Implications for alignment**: The authors conjecture that behaviors such as sycophancy and agreement bias may share a similar structure — the model induces a "belief modification function" from context and applies it during output generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Extending induction heads from the token level to the function level represents a conceptual breakthrough; the formalization and naming of function induction carry significant theoretical value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across 4 models and 4 task pairs, supplemented by ablation, causal intervention, and heatmap analysis; however, the discovered circuit does not perfectly satisfy faithfulness/completeness criteria.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Structure is clear, concepts are precisely defined, figures are information-dense, and a running example is maintained throughout the paper.
- **Value**: ⭐⭐⭐⭐ — Deepens mechanistic understanding of ICL and implicit multi-step reasoning, with practical implications for model evaluation and pretraining design; however, findings are limited to synthetic tasks and validation in natural settings remains outstanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] What One Cannot, Two Can: Two-Layer Transformers Provably Represent Induction Heads on Any-Order Markov Chains](../../NeurIPS2025/llm_nlp/what_one_cannot_two_can_two-layer_transformers_provably_represent_induction_head.md)
- [\[ICLR 2026\] Compositional-ARC: Assessing Systematic Generalization in Abstract Spatial Reasoning](compositional-arc_assessing_systematic_generalization_in_abstract_spatial_reason.md)
- [\[AAAI 2026\] Blue Teaming Function-Calling Agents](../../AAAI2026/llm_nlp/blue_teaming_function-calling_agents.md)
- [\[ICLR 2026\] Is the Reversal Curse a Binding Problem? Uncovering Limitations of Transformers from a Basic Generalization Failure](is_the_reversal_curse_a_binding_problem_uncovering_limitations_of_transformers_f.md)
- [\[ICLR 2026\] BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning](bots_a_unified_framework_for_bayesian_online_task_selection_in_llm_reinforcement.md)

</div>

<!-- RELATED:END -->
