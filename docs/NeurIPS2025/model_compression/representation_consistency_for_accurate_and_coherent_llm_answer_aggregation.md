---
title: >-
  [Paper Note] Representation Consistency for Accurate and Coherent LLM Answer Aggregation
description: >-
  [NeurIPS 2025][Model Compression][Test-time scaling] This paper proposes Representation Consistency (RC), which improves answer aggregation by analyzing the consistency of internal activations when an LLM generates multiple candidate answers. Reasoning paths that yield the same answer with highly consistent internal representations are more likely to be correct. A sparse variant, RC-S, leveraging sparse autoencoders achieves the best performance, consistently outperforming Self-Consistency across 4 LLMs and 4 reasoning datasets.
tags:
  - NeurIPS 2025
  - Model Compression
  - Test-time scaling
  - answer aggregation
  - internal representation consistency
  - sparse autoencoders
  - LLM reasoning
date: 2026-05-08
content_hash: 88cea78f716969dc
---

# Representation Consistency for Accurate and Coherent LLM Answer Aggregation

**Conference**: NeurIPS 2025
**arXiv**: [2506.21590](https://arxiv.org/abs/2506.21590)
**Code**: None
**Area**: Model Compression
**Keywords**: Test-time scaling, answer aggregation, internal representation consistency, sparse autoencoders, LLM reasoning

## TL;DR
This paper proposes Representation Consistency (RC), which improves answer aggregation by analyzing the consistency of internal activations when an LLM generates multiple candidate answers. Reasoning paths that yield the same answer with highly consistent internal representations are more likely to be correct. A sparse variant, RC-S, leveraging sparse autoencoders achieves the best performance, consistently outperforming Self-Consistency across 4 LLMs and 4 reasoning datasets.

## Background & Motivation

**Background**: Test-time scaling is an important paradigm for improving LLM performance. Self-Consistency (SC) is the most popular approach—sampling multiple reasoning paths and taking a majority vote.

**Limitations of Prior Work**:
- SC considers only the frequency of final answers, entirely ignoring the rich reasoning information encoded in the model's **internal activations**.
- When two answers have similar frequencies (near-tie), SC's tie-breaking strategy is unreliable.
- Prompt rephrasing can increase reasoning diversity but also introduces inconsistent reasoning for the same question—existing methods cannot distinguish "consistent diversity" from "inconsistent diversity."

**Key Challenge**: Multiple reasoning paths may "coincidentally" reach the same answer while their underlying reasoning processes are inconsistent—in such cases, the answer is less reliable than one supported by consistent reasoning processes.

**Key Insight**: Using intermediate-layer activations (residual stream) as "fingerprints" of the reasoning process.

**Core Idea**: If multiple reasoning paths arrive at answer A with highly similar internal activations (high consistency), it indicates the model reached A through **coherent and consistent** reasoning, making it more trustworthy than answer B with inconsistent activations.

## Method

### Overall Architecture
Given question $q$, $N$ candidate responses are generated via multiple prompt rephrasings × multiple samples. Responses are grouped by final answer. For each group, a scoring function $V_{q,a} = \lambda \cdot consistency_{q,a} + (1-\lambda) \cdot frequency_{q,a}$ is computed, and the answer with the highest score is selected. Consistency is computed from the similarity of internal model activations.

### Key Designs

1. **RC-D (Dense Representation Consistency)**:

    - **Function**: Computes intra-group consistency using raw model activations.
    - **Mechanism**: For all responses to answer $a$, the activations $z_{n}^{l}$ at layer $l$ and token position $n$ are extracted, and the mean pairwise cosine similarity within the group is computed: $consistency_{q,a} = \frac{1}{|Z|(|Z|-1)} \sum_{z_1 \neq z_2} sim(z_{1,n}^l, z_{2,n}^l)$
    - **Extraction Position**: The token position immediately before the model outputs the final answer letter (e.g., "A"), typically from the middle layers (~50% depth).
    - **Design Motivation**: If the model arrives at answer A through similar internal computations across multiple different prompts, A is likely the product of a robust reasoning process.

2. **RC-S (Sparse/SAE Variant)**:

    - **Function**: Encodes activations into sparse representations using pretrained sparse autoencoders (SAEs) before computing consistency.
    - **Mechanism**: $consistency\text{-}sparse_{q,a} = \frac{1}{|Z|(|Z|-1)} \sum sim(f_{enc}^l(z_1), f_{enc}^l(z_2))$
    - Uses publicly available SAEs such as GemmaScope and LlamaScope.
    - **Design Motivation**: Raw activations are dense and polysemantic; SAEs decompose them into sparse, monosemantic features—consistency in the sparse signal more precisely reflects "use of the same concepts/features."

3. **Scoring Function V**:

    - $V_{q,a} = \lambda \cdot consistency_{q,a} + (1-\lambda) \cdot frequency_{q,a}$
    - When $\lambda=0$: reduces to SC (pure frequency voting).
    - When $\lambda=1$: pure activation consistency (inapplicable—single-response groups always have consistency = 1).
    - Optimal $\lambda$ in practice is approximately 0.3–0.7, varying by model and dataset.

4. **RC-E (External Baseline)**:

    - Uses an external NLI model (bge-m3-zeroshot) to compute entailment probabilities between response texts as a consistency measure—serving as a comparative baseline.
    - Result: RC-E < RC-D ≈ RC-S, demonstrating that internal activations are more informative than external embeddings.

### Loss & Training
- **Zero training**—only cached model activations and lightweight similarity computations are used.
- No additional model queries are required; the method is applied as post-processing over existing generations.
- SAEs are pretrained public models (GemmaScope/LlamaScope) and require no separate training.

## Key Experimental Results

### Main Results — 4 Models × 4 Datasets (Accuracy Gain Relative to SC)

| Method | Llama3.1-8B | Gemma2-2B | Gemma2-9B | Gemma2-27B |
|--------|-------------|-----------|-----------|------------|
| NE (single response) | −5.60% | −4.43% | −4.37% | −5.19% |
| SC (baseline) | 52.9% | 44.7% | 48.6% | 52.3% |
| RC-E (external embedding) | +1.06% | +0.84% | +1.07% | +0.55% |
| RC-D (dense activation) | +1.84% | +0.89% | +1.32% | +0.76% |
| **RC-S (sparse/SAE)** | **+1.73%** | **+1.10%** | **+1.40%** | **+0.89%** |

### Ablation Study — Optimal Hyperparameters

| Model | RC-D Layer/λ | RC-S Layer/λ |
|-------|--------------|--------------|
| Llama3.1-8B | 50%, 0.43 | 25%, 0.73 |
| Gemma2-9B | 50%, 0.36 | 50%, 0.46 |
| Gemma2-27B | 50%, 0.44 | 50%, 0.59 |

### Validation of Consistency–Correctness Relationship

| Method | Correct answer has higher consistency | Incorrect answer has higher consistency |
|--------|--------------------------------------|----------------------------------------|
| External embedding (baseline) | 49.6% | 50.4% |
| Dense activation (ours) | **55.2%** | 44.8% |
| Sparse/SAE (ours) | **55.9%** | 44.1% |

### Key Findings
- RC-D and RC-S outperform SC in 30/32 fine-grained settings—the improvement is highly consistent.
- RC-S marginally outperforms RC-D—sparse, disentangled features more faithfully reflect reasoning consistency.
- External NLI embedding consistency is nearly uncorrelated with correctness (~50:50), whereas **internal activation consistency significantly predicts correctness** (55:44)—demonstrating that internal activations encode reasoning process information absent from textual outputs.
- The optimal layer is typically the middle of the model (~50% depth), consistent with mechanistic interpretability findings that middle layers encode high-level concepts.
- Improvements are largest for smaller models with fewer responses (6 samples), indicating RC is most valuable under resource-constrained settings.
- The best-case improvement reaches 4% (CSQA + Llama + 6 responses).

## Highlights & Insights
- **Internal activation consistency > surface-level textual consistency** is the central finding—it validates an intuitively elegant but previously untested hypothesis: coherent reasoning produces similar internal representations.
- **The introduction of SAEs** is theoretically motivated: dense activations are polysemantic, and SAE disentanglement makes "consistent activation of feature 123" semantically more precise than "high cosine similarity between vectors."
- **Zero additional query cost**—the method exploits only already-cached activations and represents a pure incremental improvement over SC.
- The approach can augment any test-time scaling strategy that produces multiple candidate responses, not limited to SC.

## Limitations & Future Work
- The selection of $\lambda$ and layer requires validation-set tuning; cross-task generalization remains to be improved.
- Validation is limited to multiple-choice questions—grouping answers for open-ended generation is more challenging.
- SAE availability is limited (high-quality public SAEs currently exist only for Gemma and Llama).
- Consistency metrics are inapplicable to answers with only one response—at least two responses are required.
- RC could be extended for online use during generation, rather than being applied solely as post-processing.

## Related Work & Insights
- **vs. Self-Consistency (SC)**: SC considers only frequency; RC additionally incorporates activation consistency—RC's advantage is greatest when frequencies are near-tied.
- **vs. Probing/Linear Probing**: Probing learns a mapping from activations to labels, whereas RC compares distances between activations—the latter requires no training.
- **vs. INSIDE (ICLR'24)**: INSIDE uses internal model signals for uncertainty quantification; RC applies a similar intuition to answer aggregation but requires no additional training.

## Rating
- Novelty: ⭐⭐⭐⭐ Leveraging internal activation consistency for answer aggregation is an intuitively elegant and novel direction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models × 4 datasets × 10 configurations = 160 experimental groups, with rigorous statistical analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The problem is formally defined, the method is clearly derived, and RC-E is carefully designed as a controlled comparison.
- Value: ⭐⭐⭐⭐ A zero-additional-cost test-time scaling improvement that is plug-and-play.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Robustifying Learning-Augmented Caching Efficiently without Compromising 1-Consistency](robustifying_learning-augmented_caching_efficiently_without_compromising_1-consi.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](../../ACL2026/model_compression/representation-guided_parameter-efficient_llm_unlearning.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [\[NeurIPS 2025\] BaRISTA: Brain-Scale Informed Spatiotemporal Representation of Human Intracranial EEG](barista_brain_scale_informed_spatiotemporal_representation_of_human_intracranial.md)
- [\[NeurIPS 2025\] VQToken: Neural Discrete Token Representation Learning for Extreme Token Reduction in Video Large Language Models](vqtoken_neural_discrete_token_representation_learning_for_extreme_token_reductio.md)

<!-- RELATED:END -->
