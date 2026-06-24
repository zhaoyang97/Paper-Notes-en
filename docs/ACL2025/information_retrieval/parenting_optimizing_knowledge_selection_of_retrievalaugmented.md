---
title: >-
  [Paper Note] Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG] Inspired by functional areas of the human brain, the Parenting framework is proposed. It decouples and localizes subspaces related to "context adherence" and "noise robustness" in the parameter space of LLMs, and designs customized fine-tuning strategies for different subspaces to achieve a balanced enhancement of both capabilities.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Parameter Decoupling"
  - "Knowledge Selection"
  - "Context Adherence"
  - "Noise Robustness"
date: 2026-05-08
content_hash: 1586d3e4dd567972
---

# Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning

**Conference**: ACL 2025  
**arXiv**: [2410.10360](https://arxiv.org/abs/2410.10360)  
**Code**: [GitHub](https://github.com/Nostradamus4869/Parenting)  
**Area**: Retrieval-Augmented Generation / Knowledge Conflict  
**Keywords**: RAG, Parameter Decoupling, Knowledge Selection, Context Adherence, Noise Robustness

## TL;DR

Inspired by functional areas of the human brain, the Parenting framework is proposed. It decouples and localizes subspaces related to "context adherence" and "noise robustness" in the parameter space of LLMs, and designs customized fine-tuning strategies for different subspaces to achieve a balanced enhancement of both capabilities.

## Background & Motivation

**Background**: RAG has become a mainstream paradigm to alleviate LLM hallucinations and knowledge obsolescence by integrating externally retrieved knowledge.

**Limitations of Prior Work**: Existing RAG methods lack effective control mechanisms over internal and external knowledge. When external evidence conflicts with the model's internal memory, the model may fail to adhere properly to the external evidence; when the retrieved content contains noise, the model may be misled.

**Key Challenge**: Overemphasizing context adherence causes the model to focus on noisy information, whereas over-resisting noise makes the model ignore critical evidence. The two supervisory signals are naturally contradictory, and unified training leads to mutual interference.

**Goal**: How to establish effective internal and external knowledge control mechanisms in RAG to simultaneously improve both adherence and robustness.

**Key Insight**: Analogous to functional areas of the human brain (mirror neuron system $\rightarrow$ imitation learning, hippocampus $\rightarrow$ memory retrieval), this work localizes subspaces corresponding to different capabilities within the parameter space of LLMs.

**Core Idea**: Decouple the subspaces in the parameter space associated with adherence and robustness, and apply customized training signals to each to prevent mutual contamination of contradictory supervisory signals.

## Method

### Overall Architecture

Parenting consists of four core components: (1) constructing a dedicated dataset to elicit adherence and robustness; (2) key parameter mining, which measures parameter importance by combining forward activation and gradient signals; (3) subspace localization, which identifies four categories of subspaces through interaction analysis; (4) tailored fine-tuning, which designs dedicated fine-tuning strategies for each category of subspace.

### Key Designs

1. **Key Parameter Mining**: By combining forward activation probability (measuring hierarchical sensitivity of pre-trained parameters under different inputs) and backward gradient sensitivity (incorporating smoothing and uncertainty quantification), the method calculates a comprehensive importance score for each parameter unit regarding adherence/robustness. After Z-score standardization, the parameter units are categorized into four types of subspaces: the entangled subspace (highly important to both), the adherence-specific subspace, the robustness-specific subspace, and other subspaces.
2. **Document Extraction Task**: An auxiliary task is designed for the entangled subspace. By simultaneously presenting relevant documents, same-topic noisy documents, and off-topic noisy documents to the model, it is trained to identify document types and accurately recite content, thereby simultaneously enhancing both adherence and robustness.
3. **Boundary-Controlled Fine-Tuning**: The adherence subspace only receives adherence loss (without gradient contamination from robustness), the robustness subspace only receives robustness loss (without gradient contamination from adherence), and other subspaces keep their pre-trained weights frozen to preserve general capabilities.

### Loss & Training

- Entangled subspace: $\mathcal{L}_{cx} = \delta_1(\gamma_a \mathcal{L}_a + \gamma_r \mathcal{L}_r) + (1-\delta_1)\mathcal{L}_c$, where the weights are adaptively determined by the expected Z-scores within the subspace
- Adherence subspace: $\mathcal{L}_{ax} = \delta_1 \mathcal{L}_a + (1-\delta_1)\mathcal{L}_c$
- Robustness subspace: $\mathcal{L}_{rx} = \delta_1 \mathcal{L}_r + (1-\delta_1)\mathcal{L}_c$
- The training data is constructed based on SQuAD 2.0, supporting both full-parameter fine-tuning and PEFT approaches such as LoRA.

## Key Experimental Results

### Main Results

| Method | SQuAD R_Ad | SQuAD R_Ro | RGB R_Ad | RGB R_Ro | KNOT R_Ad | KNOT R_Ro |
|------|-----------|-----------|---------|---------|----------|----------|
| Base (LLaMA2-7B) | 44.20 | 16.40 | 68.00 | 29.50 | 45.09 | 20.54 |
| KAFT | 54.15 | 18.43 | 71.50 | 30.50 | 47.09 | 22.92 |
| RAAT | 39.25 | 40.73 | 49.50 | 41.00 | 25.09 | 35.58 |
| IRCAN | 53.17 | 13.50 | 72.50 | 20.00 | 46.51 | 16.50 |
| **Parenting** | **69.24** | **44.85** | **79.50** | **45.50** | **67.42** | **42.82** |

Compared to the strongest baseline on LLaMA2-7B, Parenting improves adherence by 15+% and robustness by 4+%, being the only method to achieve substantial simultaneous improvements in both.

### Ablation Study

| Variant | SQuAD R_Ad | SQuAD R_Ro | Description |
|------|-----------|-----------|------|
| Parenting | 69.24 | 44.85 | Full method |
| Parenting_{l-} | 66.15 | 39.78 | Removing hierarchical clues from forward activation |
| Parenting_{b-} | 55.90 | 20.70 | Without boundary control (unified training) |
| Parenting_{e-} | 62.57 | 36.71 | Removing the document extraction task |

Removing boundary control (b-) leads to a steep drop in robustness (44.85 $\rightarrow$ 20.70), validating the critical importance of isolating contradictory signals.

### Key Findings

- **Cross-Domain Generalization**: On the medical CMB dataset, Parenting improves adherence from 54.28% to 75.79% and robustness from 20.17% to 48.21%
- **Noise Discrimination**: In noise discrimination tasks, Parenting achieves an accuracy of 69.89% (SQuAD), outperforming the robustness-focused RAAT (62.48%)
- **Visualization of Parameter Distribution**: Adherence-related parameters are mainly located in the mid-to-high layers, robustness-related parameters are primarily in the high layers, and entangled parameters lie in the low-to-mid layers

## Highlights & Insights

- This is the first work to achieve fine-grained decoupling and targeted optimization of adherence and robustness in RAG from a parameter-space perspective.
- The analogy with brain functional areas is ingenious, converting the abstract knowledge control issue into concrete parameter subspace optimization.
- Visual analysis reveals the hierarchical distribution patterns of knowledge storage and processing in LLMs.

## Limitations & Future Work

- The reliance on SQuAD 2.0 to construct probing datasets may introduce Dataset bias.
- The parameter mining phase requires extra computational overhead (forward + backward propagation signal collection).
- The selection of the Z-score threshold (=1) lacks theoretical derivation.
- The evaluation is limited to QA tasks; and its applicability to scenarios such as long-form generation remains to be explored.
- Subspace partitioning is a static process; dynamic adaptive strategies have not yet been explored.

## Related Work & Insights

- Compared to IRCAN (neuron-level context adherence enhancement), Parenting operates parameter importance analysis at a coarser granularity (matrix level) but simultaneously optimizes both capabilities.
- It is complementary to KnowPO (preference optimization)—Parenting tackles the problem from the parameter space, whereas KnowPO acts on the training objectives.
- The parameter decoupling concept can be extended to other scenarios requiring a balance of multiple conflicting goals (e.g., safety vs. helpfulness).
- The efficacy is also validated on Qwen1.5-14B, demonstrating that Parenting possesses generalizability across different model architectures.
- Behavioral tendency analysis reveals that Parenting maintains stability under different data ratios—increasing adherence data does not harm robustness, and vice versa.
- Compared to methods such as PH3 (handling knowledge conflicts by pruning negative attention heads) and CAD (contrastive decoding to reduce reliance on prior knowledge), Parenting intervenes at the training stage rather than the inference stage, offering more fundamental effects.

## Rating

- Novelty: ⭐⭐⭐⭐ The perspective of parameter space decoupling is novel, though the fundamental methods for parameter importance analysis have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive evaluations on multiple models and datasets, including comprehensive ablation studies, visualizations, and cross-domain generalization.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured and appropriately analogized, though mathematical notation is somewhat dense.
- Value: ⭐⭐⭐⭐ Provides a new optimization paradigm for knowledge control in RAG, holding practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SetR: Shifting from Ranking to Set Selection for Retrieval Augmented Generation](setr_set_selection_rag.md)
- [\[ACL 2025\] Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)
- [\[ACL 2025\] Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](astute_rag_knowledge_conflicts.md)
- [\[ACL 2025\] RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models](rare_retrieval_augmented_reasoning.md)
- [\[ACL 2025\] HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases](hybgrag_hybrid_rag_skb.md)

</div>

<!-- RELATED:END -->
