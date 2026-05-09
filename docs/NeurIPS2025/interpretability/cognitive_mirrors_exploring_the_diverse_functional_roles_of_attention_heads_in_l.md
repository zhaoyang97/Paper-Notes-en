---
title: >-
  [Paper Note] Cognitive Mirrors: Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning
description: >-
  [NeurIPS 2025][Attention heads] This paper proposes the CogQA benchmark dataset and a multi-class probing framework to systematically analyze cognitive functional specialization of attention heads in LLMs. The study reveals that cognitive heads exhibit sparsity, universality, and hierarchical functional organization; ablating cognitive heads significantly degrades reasoning performance, while amplifying them improves accuracy.
tags:
  - NeurIPS 2025
  - Attention heads
  - cognitive functions
  - interpretability
  - probing
  - functional specialization
date: 2026-05-08
content_hash: 34d4f79c2d91ec90
---

# Cognitive Mirrors: Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2512.10978](https://arxiv.org/abs/2512.10978)
**Code**: [https://github.com/sihuo-design/CognitiveMirrors](https://github.com/sihuo-design/CognitiveMirrors)
**Area**: Interpretability
**Keywords**: Attention heads, cognitive functions, interpretability, probing, functional specialization

## TL;DR
This paper proposes the CogQA benchmark dataset and a multi-class probing framework to systematically analyze cognitive functional specialization of attention heads in LLMs. The study reveals that cognitive heads exhibit sparsity, universality, and hierarchical functional organization; ablating cognitive heads significantly degrades reasoning performance, while amplifying them improves accuracy.

## Background & Motivation
**Background**: LLMs demonstrate strong performance across diverse NLP tasks, yet their internal mechanisms remain opaque. Prior work suggests that attention heads may assume specific functions (e.g., information retrieval, answer consistency), but these findings are largely confined to simple tasks.

**Limitations of Prior Work**: A systematic framework for aligning internal LLM components with cognitive functions is lacking. Existing studies are limited to single-function analyses (e.g., retrieval heads) and cannot capture many-to-many relationships between heads and functions.

**Key Challenge**: Chain-of-thought (CoT) prompting improves LLM reasoning, implying the existence of activatable specialized components within models; however, benchmarks and methods to identify and validate these components are absent.

**Goal**: (1) Do attention heads in LLMs exhibit cognitive functional specialization? (2) Are specialization patterns consistent across different models? (3) What is the causal contribution of these heads to reasoning tasks?

**Key Insight**: Drawing an analogy to cognitive division of labor in the human brain (e.g., the frontal lobe for knowledge recall, Wernicke's area for semantic processing), the paper hypothesizes that LLM attention heads exhibit a similar division of function.

**Core Idea**: Probing functional specialization patterns of attention heads via CoT sub-questions annotated with cognitive function labels.

## Method

### Overall Architecture
A three-stage pipeline: (1) constructing the CogQA dataset by decomposing complex questions into sub-questions annotated with cognitive function labels; (2) extracting attention head features and training multi-class probing classifiers; (3) computing head importance via gradient attribution and validating through intervention experiments.

### Key Designs

1. **CogQA Dataset Construction**:

    - Function: 570 main questions are decomposed into 3,402 sub-questions, each annotated with one of eight cognitive functions.
    - Mechanism: 750 questions are sampled from AQuA, CREAK, ECQA, e-SNLI, and GSM8K, and decomposed into CoT chains using GPT-4o. Cognitive functions are categorized into low-level (retrieval, knowledge recall, semantic understanding, syntactic understanding) and high-level (mathematical computation, inference, logical reasoning, decision-making).
    - Design Motivation: Existing datasets lack cognitive function annotations, making fine-grained evaluation of individual LLM component contributions infeasible.

2. **Multi-Class Probing Method**:

    - Function: Identifying attention heads associated with specific cognitive functions.
    - Mechanism: For each sub-question, activations from all heads across all layers are extracted (averaged over top-$k$ important tokens): $\bar{x}_l^m = \frac{1}{k}\sum_{j \in \mathcal{I}_k} x_l^m(j)$. Layer-wise averages are concatenated and fed into a two-layer MLP classifier. Head importance is computed via gradient×activation attribution: $I_j^{(c)} = \mathbb{E}[\frac{\partial \hat{y}_c}{\partial \bar{x}_j} \cdot \bar{x}_j]$.
    - Novelty: Unlike single-class probing (e.g., detecting only retrieval heads), this framework simultaneously detects multiple function classes, capturing both one-to-many and many-to-one relationships.

3. **Intervention Validation**:

    - Function: Verifying the causal importance of cognitive heads through ablation and amplification.
    - Mechanism: Attention weights of cognitive heads are set to zero (ablation) or scaled up (amplification), and the effects are compared against random head interventions.

### Loss & Training
- Probing datasets are split 4:1 into training and validation sets.
- Classification is performed using a dimensionality-reducing linear projection followed by a two-layer MLP.
- Training and evaluation are conducted independently on six distinct models.

## Key Experimental Results

### Main Results (Accuracy % — Ablating Cognitive Heads vs. Random Heads)

| Model | Intervention | Retrieval | Math | Inference | Logic |
|-------|-------------|-----------|------|-----------|-------|
| Llama3.1-8B | Random heads | 84.71 | 83.08 | 70.18 | 54.69 |
| Llama3.1-8B | Cognitive heads | **8.24** | **66.17** | **52.63** | **4.69** |
| Llama3.2-3B | Random heads | 86.47 | 69.65 | 85.96 | 76.56 |
| Llama3.2-3B | Cognitive heads | **17.06** | **80.10** | **7.02** | **0.00** |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| High-importance heads (<7%) | Covers most functions | Sparse yet precise |
| Retrieval heads | 6.45% exceed threshold 0.01 | Most active heads |
| Inference heads | 3.42% exceed threshold 0.01 | Fewest active heads |
| Math heads (Llama3.1-8B) | 59 heads | Fewer heads required |
| Inference heads (Llama3.1-8B) | 139 heads | Most heads required |

### Key Findings
- **Sparsity**: Each cognitive function activates fewer than 7% of all attention heads.
- **Hierarchical organization**: Retrieval heads are concentrated in middle layers; math heads are concentrated in higher layers.
- **Catastrophic degradation upon ablation**: Removing retrieval heads drops Llama3.1-8B accuracy from 84.71% to 8.24%, whereas ablating random heads has negligible effect.
- **Cross-model universality**: All three model families (LLaMA/Qwen/Yi) exhibit similar sparse specialization patterns.

## Highlights & Insights
- **Bridging cognitive science and LLM interpretability** is the paper's most significant contribution — organizing LLM analysis by analogy to functional brain regions yields an intuitive and well-motivated analytical framework.
- **Multi-class probing outperforms single-class probing** — simultaneously detecting functionally overlapping heads reveals both multi-functionality of individual heads and functional clustering phenomena.
- The hierarchical functional organization (low-level functions → middle layers; high-level functions → upper layers) has practical implications for guiding pruning and distillation strategies.

## Limitations & Future Work
- CogQA contains only 570 questions, which is relatively small in scale; furthermore, relying on GPT-4o for decomposition may introduce systematic bias.
- The categorization of eight cognitive functions lacks rigorous grounding in cognitive science theory.
- Only instruction-tuned models are analyzed; base models and the evolution across different training stages remain unexplored.
- Intervention experiments rely solely on zeroing attention weights, without considering more principled causal intervention methods.

## Related Work & Insights
- **vs. Retrieval Head**: Wu et al. (2024) focus exclusively on information retrieval heads; the present work extends the analysis to eight cognitive function categories, uncovering richer specialization patterns.
- **vs. Truthful Head**: Prior work identifies "truthfulness heads"; the present paper subsumes such findings within a unified multi-class framework, revealing clustering and hierarchical structure among functions.

## Rating
- Novelty: ⭐⭐⭐⭐ — Analyzing attention heads through the lens of cognitive functions is novel, though the probing methodology itself is fairly standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Six models across three families with intervention validation, though dataset scale is limited.
- Writing Quality: ⭐⭐⭐⭐ — Analogies are clear and the structure is well-organized.
- Value: ⭐⭐⭐⭐ — Provides practical guidance for model design, pruning, and distillation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [\[NeurIPS 2025\] Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)
- [\[NeurIPS 2025\] Why Is Attention Sparse in Particle Transformer?](why_is_attention_sparse_in_particle_transformer.md)
- [\[NeurIPS 2025\] Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms](tropical_attention_neural_algorithmic_reasoning_for_combinatorial_algorithms.md)
- [\[NeurIPS 2025\] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)

<!-- RELATED:END -->
