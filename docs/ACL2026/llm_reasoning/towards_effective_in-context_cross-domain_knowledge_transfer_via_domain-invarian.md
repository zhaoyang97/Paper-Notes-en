---
title: >-
  [Paper Note] Towards Effective In-context Cross-domain Knowledge Transfer via Domain-invariant-neurons-based Retrieval
description: >-
  [ACL 2026][LLM Reasoning][Cross-domain knowledge transfer] This paper proposes DIN-Retrieval, which identifies domain-invariant neurons (DINs) in LLMs exhibiting consistent activation polarity across domains…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Cross-domain knowledge transfer"
  - "domain-invariant neurons"
  - "in-context learning retrieval"
  - "reasoning structure alignment"
  - "mathematical-logical reasoning"
date: 2026-05-08
content_hash: ffb9062bf3bb8a6f
---

# Towards Effective In-context Cross-domain Knowledge Transfer via Domain-invariant-neurons-based Retrieval

**Conference**: ACL 2026
**arXiv**: [2604.05383](https://arxiv.org/abs/2604.05383)
**Code**: [GitHub](https://github.com/Leon221220/DIN-Retrieval)
**Area**: LLM Reasoning
**Keywords**: Cross-domain knowledge transfer, domain-invariant neurons, in-context learning retrieval, reasoning structure alignment, mathematical-logical reasoning

## TL;DR

This paper proposes DIN-Retrieval, which identifies domain-invariant neurons (DINs) in LLMs exhibiting consistent activation polarity across domains, constructs a domain-robust representational subspace for retrieving structurally compatible cross-domain demonstrations, and provides the first systematic evidence that cross-domain ICL examples can improve LLM reasoning performance, achieving an average gain of 1.8% on math-to-logic reasoning transfer.

## Background & Motivation

**State of the Field**: In-context learning (ICL) enables LLMs to adapt to new tasks without parameter updates. However, existing ICL research assumes access to in-domain expert-annotated demonstrations, limiting applicability in domains with scarce expert knowledge (e.g., specialized mathematical reasoning, formal logic, legal analysis).

**Limitations of Prior Work**: (1) Zero-shot LLMs tend to exhibit three failure modes during reasoning—missing intermediate links, incomplete branch integration, and neglected blocking conditions; (2) Although reasoning tasks across domains differ substantially in surface semantics, they share many reusable implicit logical structures (e.g., chain reasoning, conditional branching); (3) Manually selecting structurally aligned cross-domain demonstrations is impractical given the large structural variation across tasks.

**Root Cause**: Cross-domain examples can restore correct reasoning topology (as demonstrated by prior work), yet no mechanism exists for automatically retrieving structurally compatible demonstrations.

**Paper Goals**: Develop an automated retrieval method capable of identifying ICL demonstrations from other domains that are structurally compatible with a given target query.

**Starting Point**: Drawing on the concept of domain-invariant features from domain adaptation theory—identifying neurons in LLM hidden layers whose activation polarity remains consistent across domains, as these neurons encode domain-agnostic reasoning structure information.

**Core Idea**: Discover domain-invariant neurons (DINs) within LLMs, construct domain-robust retrieval representations from their activations, and retrieve structurally aligned cross-domain demonstrations via cosine similarity.

## Method

### Overall Architecture

DIN-Retrieval proceeds in four steps: (A) DIN Identification—compute the z-score of each neuron over source- and target-domain samples and select neurons with consistent cross-domain polarity; (B) DIN Vector Construction—aggregate DIN activations across multiple layers into a compact representation; (C) Cross-domain Retrieval—retrieve top-$k$ source-domain examples using cosine similarity combined with MMR in the DIN space; (D) Cross-domain CoT Inference—concatenate the retrieved source-domain examples with the target query as few-shot demonstrations for ICL inference.

### Key Designs

1. **Domain-Invariant Neuron (DIN) Identification**:

    - Function: Discover neurons in LLMs that encode domain-agnostic reasoning structures.
    - Mechanism: For each neuron $k$ in each layer, compute z-scores $z_k^S$ and $z_k^T$ over the source and target domains respectively. Select neurons whose polarity is consistent across both domains and exceeds threshold $\tau$: $\mathcal{I} = \{k | z_k^S > \tau \wedge z_k^T > \tau\} \cup \{k | z_k^S < -\tau \wedge z_k^T < -\tau\}$. If the budget $K$ is exceeded, the top-$K$ neurons are selected by $|z_k^S| + |z_k^T|$.
    - Design Motivation: Neurons with consistent cross-domain polarity are insensitive to domain shift and encode abstract features shared across domains. Pruning experiments confirm their functional importance: removing DINs causes a far greater increase in perplexity than random pruning.

2. **DIN Vector Representation**:

    - Function: Compress high-dimensional hidden states into a domain-robust compact representation.
    - Mechanism: For each layer's hidden state of input $x$, compute the token-averaged representation $\bar{h}^{(l)}(x)$, retain only the DIN dimensions, and concatenate across layers: $\mathbf{v}_{\text{DIN}}(x) = \bigoplus_{l \in \mathcal{L}} h^{(l)}(x)_{\mathcal{I}^{(l)}}$.
    - Design Motivation: Full hidden states contain substantial domain-specific information that interferes with cross-domain similarity computation; DIN-filtered representations focus on reasoning structure rather than surface semantics.

3. **MMR Diversity-Aware Retrieval**:

    - Function: Retrieve cross-domain demonstrations that are both structurally aligned and diverse.
    - Mechanism: $\text{Score}(i) = \lambda \cdot \cos(\mathbf{v}_q, \mathbf{v}_i) - (1-\lambda) \cdot \max_{j \in \mathcal{S}} \cos(\mathbf{v}_i, \mathbf{v}_j)$, balancing relevance to the query against redundancy among already-selected examples. The default setting retrieves $k=2$ source-domain examples.
    - Design Motivation: Prevents retrieval of structurally redundant examples—demonstrations covering diverse reasoning patterns provide LLMs with more comprehensive structural cues.

### Loss & Training

DIN-Retrieval requires no training. DIN identification relies on activation statistics, and retrieval is based on cosine similarity. Experiments are conducted using LLaMA-3.1-8B, Gemma-3-12B/27B, and Qwen2.5/3-7B–32B.

## Key Experimental Results

### Main Results

**Cross-domain reasoning accuracy (average over four transfer directions)**

| Method | Qwen2.5-7B | Qwen3-8B | Gemma-3-27B |
|--------|-----------|---------|------------|
| Zero-shot | 84.6 | 91.8 | 88.75 |
| X-ICL (embedding retrieval) | 83.4 | 91.2 | — |
| **DIN-Retrieval** | **86.8** | **93.1** | **90.3** |

### Ablation Study

**DIN vs. random neuron selection (GSM8K→FOLIO)**

| Model | DIN Acc. | Random Acc. | Diff. |
|-------|----------|-------------|-------|
| LLaMA-3.1-8B | 62.7 | 60.3 | +2.4 |
| Qwen2.5-7B | 62.8 | 59.5 | +3.3 |
| Qwen3-8B | 85.5 | 84.0 | +1.5 |

### Key Findings

- DIN-Retrieval consistently outperforms both zero-shot and embedding-based cross-domain ICL across all models and transfer directions.
- DIN pruning causes a substantially larger perplexity increase than random pruning, confirming the functional importance of DINs.
- This work provides the first systematic evidence that cross-domain ICL demonstrations can improve LLM reasoning—challenging the assumption that ICL must rely on in-domain examples.
- Bidirectional transfer is effective: both GSM8K→FOLIO (math→logic) and FOLIO→GSM8K (logic→math) yield gains.
- Although the improvement margin is modest (average 1.8%), it is statistically significant and consistent.

## Highlights & Insights

- The insight that "different domains share reasoning structures" carries broad implications—reasoning capabilities are not domain-specific but transferable across domains.
- The discovery of DINs offers a new perspective on understanding reasoning representations inside LLMs, revealing the existence of specialized neurons that encode domain-agnostic reasoning patterns.
- The method is elegant and lightweight—requiring no training, relying solely on activation statistics and cosine similarity.

## Limitations & Future Work

- The average gain of 1.8% is modest; for strong zero-shot baselines, headroom for improvement is already limited.
- Validation is confined to bidirectional math–logic transfer; extension to broader domain pairs (e.g., legal→medical) remains unexplored.
- DIN identification requires unlabeled samples from both source and target domains to compute z-scores, making the approach not fully zero-resource.
- The selection of threshold $\tau$ and neuron ratio $k_{\text{ratio}}$ lacks an adaptive mechanism.

## Related Work & Insights

- **vs. X-ICL (embedding retrieval)**: X-ICL retrieves using full hidden-state embeddings, which contain domain-specific noise; DIN filtering focuses the representation on structural information.
- **vs. in-domain ICL**: In-domain demonstrations are generally superior when available, but this work demonstrates that cross-domain examples remain effective when in-domain annotations are unavailable.
- **vs. domain adaptation (DANN, etc.)**: Classical domain adaptation methods require training; DIN-Retrieval is entirely training-free—transferring the domain-invariant feature concept from training-time adaptation to inference-time retrieval.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of cross-domain ICL; the discovery of DINs carries theoretical significance.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models × multiple transfer directions, DIN existence validation, and statistical significance testing.
- Writing Quality: ⭐⭐⭐⭐ The motivational chain from failure mode analysis to method design is clear and well-structured.
- Value: ⭐⭐⭐⭐ Provides a new direction for ICL in domains with scarce expert knowledge.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention](../../AAAI2026/llm_reasoning/l2v-cot_cross-modal_transfer_of_chain-of-thought_reasoning_v.md)
- [\[AAAI 2026\] SERL: Self-Examining Reinforcement Learning on Open-Domain](../../AAAI2026/llm_reasoning/serl_self-examining_reinforcement_learning_on_open-domain.md)
- [\[ACL 2026\] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)
- [\[NeurIPS 2025\] DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](../../NeurIPS2025/llm_reasoning/dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)
- [\[ACL 2026\] Generating Effective CoT Traces for Mitigating Causal Hallucination](generating_effective_cot_traces_for_mitigating_causal_hallucination.md)

</div>

<!-- RELATED:END -->
