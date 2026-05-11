---
title: >-
  [Paper Note] C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation
description: >-
  [AAAI 2026][LLM/NLP][controlled text generation] This paper proposes the C3TG framework, which achieves fine-grained multi-attribute controllable text generation through a two-stage approach: in the generation stage…
tags:
  - "AAAI 2026"
  - "LLM/NLP"
  - "controlled text generation"
  - "multi-attribute control"
  - "KL divergence"
  - "energy function"
  - "conflict resolution"
date: 2026-05-08
content_hash: 044e627f78f47ab3
---

# C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation

**Conference**: AAAI 2026
**arXiv**: [2511.09292](https://arxiv.org/abs/2511.09292)
**Code**: None
**Area**: LLM Efficiency / Controllable Text Generation
**Keywords**: controlled text generation, multi-attribute control, KL divergence, energy function, conflict resolution

## TL;DR
This paper proposes the C3TG framework, which achieves fine-grained multi-attribute controllable text generation through a two-stage approach: in the generation stage, weighted KL divergence is used to fuse attribute distributions and adjust token probabilities; in the optimization stage, an energy function (combining classifier scores and conflict penalty terms) drives iterative rewriting via a Feedback Agent. C3TG achieves 90.4% attribute accuracy across 17 attribute subcategories while substantially reducing toxicity.

## Background & Motivation

**Background**: Controllable text generation (CTG) aims to govern attributes such as sentiment, style, tone, and topic in generated text. Existing methods fall into two categories: directly modulating the decoding distribution (PPLM, GeDi, COLD) and indirect control (prompting, fine-tuning).

**Limitations of Prior Work**: (1) Most methods can only control a single or a small set of simple attributes; (2) simultaneous multi-attribute control lacks conflict resolution mechanisms—enhancing one attribute may suppress or amplify another; (3) iterative feedback optimization pipelines are absent.

**Key Challenge**: Multiple attributes may exhibit conflicting or dependent relationships (e.g., "humorous" and "formal" are inherently in tension), making it infeasible for a single generation pass to satisfy all attribute targets simultaneously.

**Goal**: Achieve simultaneous fine-grained control over 17 attribute dimensions while handling inter-attribute conflicts.

**Key Insight**: A collaborative paradigm in which a large model handles generation and small models handle evaluation—the LLM generates text, BERT classifiers assess attribute alignment, and a Feedback Agent drives iterative rewriting.

**Core Idea**: During generation, tokens are sampled via geometrically weighted averaging of attribute priors; during optimization, an energy function combining classifier scores and dimensional stability penalties drives iterative refinement through a three-stage Chain-of-Prompt procedure.

## Method

### Overall Architecture
Two stages: (1) **Generation Phase**: token distributions are extracted from a base Llama2 model and $n$ attribute-specific models, and tokens are sampled using the closed-form solution to weighted KL divergence minimization, $P^*(x|x_{1:t-1}) = \prod_i Q_i^{\lambda_i/\Lambda} / Z$; (2) **Optimization Phase**: BERT classifiers evaluate alignment across 17 attribute dimensions, and the energy function $E(x) = \sum \alpha_i|C_{A_i}(x) - T_i| + \sum \beta_j|C_{A_j}(x) - C_{A_j}(x_{prev})|$ drives three-stage iterative rewriting.

### Key Designs

1. **Weighted KL-Divergence Fusion (Generation Phase)**

    - **Function**: Fuses token distributions from multiple attribute models into a unified sampling distribution.
    - **Mechanism**: Minimizes the weighted KL divergence $\mathcal{J}[P] = \sum_i \lambda_i D_{KL}(P \| Q_i)$; the closed-form solution is the geometrically weighted average of attribute priors. User-specified $\lambda_i$ controls the influence of each attribute.
    - **Design Motivation**: Geometric averaging offers stronger theoretical guarantees than linear mixing of probabilities and naturally interpolates among multiple distributions.
    - Implementation details: Each attribute model is an independently fine-tuned Llama2-7B trained on attribute-annotated corpora. At inference time, the weighted geometric mean of each model's logits is computed on-the-fly, normalized, and used for sampling.

2. **Energy Function with Conflict Penalties (Optimization Phase)**

    - **Function**: Quantifies the deviation between the generated text and attribute targets, while penalizing interference with non-optimized attributes.
    - **Mechanism**: $E(x) = \underbrace{\sum_i \alpha_i|C_{A_i}(x) - T_i|}_{\text{alignment term}} + \underbrace{\sum_j \beta_j|C_{A_j}(x) - C_{A_j}(x_{prev})|}_{\text{stability penalty}}$. The first term measures attribute deviation; the second prevents optimization of one attribute from disrupting others that have already been satisfied.
    - **Design Motivation**: The "whack-a-mole" problem is the central challenge in multi-attribute optimization; the stability penalty explicitly constrains variation in non-target dimensions.

3. **Three-Stage Chain-of-Prompt Refinement (Iterative Rewriting)**

    - **Function**: Progressively improves attribute alignment through a Feedback Agent–driven three-stage prompt chain.
    - **Mechanism**: Stage 1 — core attribute calibration (prioritizing correction of the most misaligned dimensions) → Stage 2 — attribute balance adjustment (fine-tuning dimensions disturbed by Stage 1) → Stage 3 — global fine-tuning (pushing all attributes toward their targets until $E(x) \leq \tau$).
    - **Design Motivation**: A single rewriting pass cannot resolve all attribute conflicts simultaneously; the coarse-to-fine staged approach enables progressive convergence.

### Attribute Coverage
17 subcategories spanning 4 major classes: emotion (joy, sadness, love, anger, fear, surprise), style (formal, humor, poetic, sarcasm, academic), tone (professional, casual, persuasive), and topic (courage, nature, technology).

## Key Experimental Results

### Main Results (ROCStories + WritingPrompts)

| Method | ROC Acc↑ | ROC PPL↓ | ROC Dist-3↑ | WP Acc↑ | Toxic↓ |
|--------|---------|---------|------------|--------|-------|
| COLD | 24.4 | 21.07 | 0.22 | 20.5 | 0.53 |
| BOLT | 36.5 | 17.33 | 0.38 | 32.1 | 0.76 |
| PPLM | 32.4 | 15.04 | 0.39 | 29.7 | 0.39 |
| Model Arithmetic | 87.5 | 11.08 | 0.81 | 84.2 | 0.16 |
| LLM Prompt | 89.5 | 5.37 | 0.89 | 80.0 | 0.29 |
| **C3TG** | **90.4** | **4.04** | **0.90** | **85.6** | **0.12** |

- C3TG achieves comprehensive superiority in attribute accuracy, fluency (PPL), diversity (Dist), and toxicity.

### Ablation Study

| Configuration | Acc↑ | Notes |
|---------------|------|-------|
| Generation Phase only | ~85% | No iterative optimization |
| + Optimization (w/o conflict penalty) | ~87% | Iterative but without protection of non-target attributes |
| + Full C3TG (with conflict penalty) | **90.4%** | Complete method |

### Key Findings
- The generation phase alone achieves reasonable performance (~85%), with the optimization phase contributing an additional ~5%.
- The conflict penalty has the greatest impact in conflicting-attribute scenarios: when simultaneously targeting "joy + formal," the formal dimension is severely disrupted without the penalty.
- Toxicity is reduced to 0.12 (vs. 0.29 for LLM Prompt), indicating that attribute control yields safer text as a by-product.
- Convergence is typically achieved within 2–3 iterations ($E(x) \leq \tau = 0.025$).
- In human evaluation, C3TG leads all baselines in naturalness (4.2/5) and attribute consistency (4.5/5).
- In conflicting-attribute pair experiments (e.g., joy + formal), C3TG achieves simultaneous satisfaction of both attributes at 82%, compared to only 61% for Model Arithmetic.

## Highlights & Insights
- The **"large model for generation, small model for evaluation"** collaborative paradigm is highly efficient: lightweight BERT classifiers provide real-time attribute feedback without requiring modification of LLM parameters.
- The **dimensional stability penalty** addresses the core challenge of multi-attribute optimization by explicitly protecting non-target dimensions—analogous to constraint preservation in constrained optimization.
- **Fine-grained control over 17 subcategories** represents a substantial advance over prior work that controls only coarse-grained attributes such as positive/negative sentiment or toxic/non-toxic content.

## Limitations & Future Work
- Training independent Llama2 attribute models and BERT classifiers for each attribute incurs significant upfront preparation costs.
- Iterative optimization requires multiple LLM inference passes, increasing latency and cost proportionally with the number of iterations.
- The selection and categorization of the 17 attribute subcategories involve a degree of subjectivity; extending to new attributes requires additional training.
- The framework is evaluated only on Llama2 and has not been validated on more recent models (Llama3, GPT series).
- The penalty coefficients $\beta_j$ are determined empirically based on inter-attribute correlations and may not generalize broadly.

## Related Work & Insights
- **vs. PPLM**: Applies gradient perturbations only to hidden states for single-attribute control; C3TG supports multi-attribute control with iterative optimization.
- **vs. Model Arithmetic**: Also performs multi-attribute fusion via model addition/subtraction; accuracy is comparable but PPL is substantially higher (11.08 vs. 4.04), indicating inferior generation quality.
- **vs. LLM Prompt**: Simple instruction-based prompting achieves 89.5% accuracy but exhibits higher toxicity (0.29 vs. 0.12) and lacks a conflict resolution mechanism.
- **Insight**: The large-model-generation + small-model-evaluation collaborative paradigm is generalizable to other controlled generation settings such as style transfer and dialogue systems.

## Rating
- **Novelty**: ⭐⭐⭐⭐ A complete and well-designed conflict-aware multi-attribute control framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Compared against 10+ baselines with both automatic and human evaluation, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous mathematical derivations and clear pipeline diagrams.
- **Value**: ⭐⭐⭐⭐ A practical framework for multi-attribute controllable generation; the conflict resolution approach is transferable to related tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Collaborative LLM Numerical Reasoning with Local Data Protection](collaborative_llm_numerical_reasoning_with_local_data_protection.md)
- [\[ICLR 2026\] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](../../ICLR2026/llm_nlp/fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)
- [\[ICCV 2025\] Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation](../../ICCV2025/llm_nlp/beyond_isolated_words_diffusion_brush_for_handwritten_text-line_generation.md)
- [\[AAAI 2026\] VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation](vspo_validating_semantic_pitfalls_in_ontology_via_llm-based_cq_generation.md)
- [\[AAAI 2026\] LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive Generation](loopllm_transferable_energy-latency_attacks_in_llms_via_repetitive_generation.md)

</div>

<!-- RELATED:END -->
