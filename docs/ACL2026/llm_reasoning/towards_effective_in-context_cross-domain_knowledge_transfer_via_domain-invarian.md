---
title: >-
  [Paper Note] Towards Effective In-context Cross-domain Knowledge Transfer via Domain-invariant-neurons-based Retrieval
description: >-
  [ACL 2026][LLM Reasoning][Cross-domain Knowledge Transfer] This paper proposes DIN-Retrieval, which identifies Domain-invariant Neurons (DINs) with consistent activation polarity across domains in LLMs to construct a dom…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Cross-domain Knowledge Transfer"
  - "Domain-invariant Neurons"
  - "Retrieval-based ICL"
  - "Reasoning Structure Alignment"
  - "Mathematical and Logical Reasoning"
date: 2026-05-08
content_hash: f89c3d5e94587bea
---

# Towards Effective In-context Cross-domain Knowledge Transfer via Domain-invariant-neurons-based Retrieval

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.05383](https://arxiv.org/abs/2604.05383)  
**Code**: [GitHub](https://github.com/Leon221220/DIN-Retrieval)  
**Area**: LLM Reasoning  
**Keywords**: Cross-domain Knowledge Transfer, Domain-invariant Neurons, Retrieval-based ICL, Reasoning Structure Alignment, Mathematical and Logical Reasoning

## TL;DR

This paper proposes DIN-Retrieval, which identifies Domain-invariant Neurons (DINs) with consistent activation polarity across domains in LLMs to construct a domain-robust representation subspace. This subspace is used to retrieve cross-domain examples with compatible reasoning structures. This work provides the first demonstration of the feasibility of using cross-domain ICL examples to enhance LLM reasoning performance, achieving an average improvement of 1.8% in math-to-logic reasoning transfer.

## Background & Motivation

**Background**: In-context Learning (ICL) enables LLMs to adapt to new tasks without parameter updates. However, existing research assumes available in-domain expert-labeled examples, which limits applicability in specialized fields where domain expertise is scarce (e.g., specialized mathematical reasoning, formal logic, legal analysis).

**Limitations of Prior Work**: (1) Zero-shot LLMs are prone to three failure modes during reasoning: missing intermediate links, incomplete branch integration, and neglecting blocking conditions. (2) Although reasoning tasks in different domains differ significantly in surface semantics, they often share reusable implicit logical structures (e.g., chain-of-thought, conditional branching). (3) Manual selection of structurally aligned cross-domain examples is impractical due to the high variance of reasoning structures across tasks.

**Key Challenge**: While earlier studies suggest cross-domain examples can help recover correct reasoning topologies, there is no automated mechanism to retrieve such structurally compatible examples.

**Goal**: To develop an automated retrieval method capable of finding cross-domain ICL examples that are structurally compatible with the target query.

**Key Insight**: Leveraging the idea of domain-invariant features from domain adaptation theory, this study identifies neurons in LLM hidden layers with consistent activation polarity across domains, which likely encode domain-agnostic reasoning structure information.

**Core Idea**: Discover Domain-invariant Neurons (DINs) within the LLM, utilize their activations to build domain-robust retrieval representations, and retrieve structurally aligned cross-domain examples via cosine similarity.

## Method

### Overall Architecture

DIN-Retrieval consists of four steps: (A) DIN Identification—calculate z-scores for each neuron using source and target domain samples to select neurons with consistent polarity; (B) DIN Vector Construction—aggregate multi-layer DIN activations into a compact representation; (C) Cross-domain Retrieval—use cosine similarity and MMR in the DIN space to retrieve top-$k$ source domain examples; (D) Cross-domain CoT Reasoning—concatenate retrieved source examples as few-shot prompts for target query reasoning.

### Key Designs

1. **Domain-invariant Neuron (DIN) Identification**:

    - Function: Identifies neurons within the LLM that encode domain-independent reasoning structures.
    - Mechanism: For each neuron $k$ in each layer, calculate z-scores $z_k^S$ and $z_k^T$ for the source and target domains. Select neurons where polarities match and exceed a threshold $\tau$: $\mathcal{I} = \{k | z_k^S > \tau \wedge z_k^T > \tau\} \cup \{k | z_k^S < -\tau \wedge z_k^T < -\tau\}$. If the count exceeds budget $K$, select top-$K$ based on $|z_k^S| + |z_k^T|$.
    - Design Motivation: Consistent activation polarity across domains suggests insensitivity to domain shifts and focus on shared abstract features. Pruning experiments show that removing DINs causes a significantly higher increase in perplexity compared to random pruning, confirming their functional importance.

2. **DIN Vector Representation**:

    - Function: Compresses high-dimensional hidden states into domain-robust compact representations.
    - Mechanism: For an input $x$, compute the token-wise mean $\bar{h}^{(l)}(x)$ of hidden states for each layer $l$, retain only the DIN dimensions, and concatenate them across layers: $\mathbf{v}_{\text{DIN}}(x) = \bigoplus_{l \in \mathcal{L}} h^{(l)}(x)_{\mathcal{I}^{(l)}}$.
    - Design Motivation: Full hidden states contain domain-specific noise that interferes with cross-domain similarity calculations; DIN-filtered representations focus on reasoning structure rather than surface semantics.

3. **MMR Diversity Retrieval**:

    - Function: Retrieves cross-domain examples that are both structurally aligned and diverse.
    - Mechanism: $\text{Score}(i) = \lambda \cdot \cos(\mathbf{v}_q, \mathbf{v}_i) - (1-\lambda) \cdot \max_{j \in \mathcal{S}} \cos(\mathbf{v}_i, \mathbf{v}_j)$, balancing similarity to the query with the diversity of the selected set. Default $k=2$ source examples are retrieved.
    - Design Motivation: Prevents selecting highly redundant structures; diverse reasoning patterns provide more comprehensive structural cues for the LLM.

### Loss & Training

DIN-Retrieval is training-free. DIN identification relies on activation statistics, and retrieval is based on cosine similarity. Evaluation was conducted on models including LLaMA-3.1-8B, Gemma-3-12B/27B, and Qwen2.5/3 (7B to 32B).

## Key Experimental Results

### Main Results

**Cross-domain Reasoning Accuracy (Average over four transfer directions)**

| Method | Qwen2.5-7B | Qwen3-8B | Gemma-3-27B |
|------|-----------|---------|------------|
| Zero-shot | 84.6 | 91.8 | 88.75 |
| X-ICL (Embedding Retrieval) | 83.4 | 91.2 | — |
| **DIN-Retrieval** | **86.8** | **93.1** | **90.3** |

### Ablation Study

**DIN vs. Random Neuron Selection (GSM8K $\rightarrow$ FOLIO)**

| Model | DIN Acc. | Random Acc. | Gain |
|------|----------|-------------|------|
| LLaMA-3.1-8B | 62.7 | 60.3 | +2.4 |
| Qwen2.5-7B | 62.8 | 59.5 | +3.3 |
| Qwen3-8B | 85.5 | 84.0 | +1.5 |

### Key Findings

- DIN-Retrieval consistently outperforms zero-shot and embedding-based cross-domain ICL across all models and transfer directions.
- The perplexity increase from DIN pruning is much higher than random pruning, validating the functional importance of DINs.
- This study systemsically demonstrates for the first time that cross-domain ICL examples can improve LLM reasoning, breaking the assumption that ICL requires in-domain examples.
- Bidirectional transfer between GSM8K $\rightarrow$ FOLIO (math to logic) and FOLIO $\rightarrow$ GSM8K (logic to math) is effective.
- While the absolute gains are modest (averaging 1.8%), they are statistically significant and consistent.

## Highlights & Insights

- The insight that "different domains share reasoning structures" is profound—reasoning ability is not strictly domain-specific but can be reused across domains.
- The discovery of DINs provides a new perspective for understanding internal reasoning representations in LLMs, suggesting the existence of specialized neurons for domain-invariant reasoning patterns.
- The method is elegant and lightweight, requiring no training and relying only on activation statistics and cosine similarity.

## Limitations & Future Work

- The 1.8% average improvement is limited, and some models show smaller gains over strong zero-shot baselines.
- The validation is confined to math-logic transfer and has not yet been extended to other domains (e.g., law $\rightarrow$ medical).
- DIN identification requires unlabeled samples from both domains to compute z-scores, so it is not entirely zero-resource.
- The selection of threshold $\tau$ and neuron ratio $k_{\text{ratio}}$ lacks an adaptive mechanism.

## Related Work & Insights

- **vs. X-ICL (Embedding Retrieval)**: Standard embedding retrieval uses full hidden states containing domain-specific noise; DIN filtering focuses on structural information.
- **vs. In-domain ICL**: In-domain examples are typically superior when available, but this work proves the effectiveness of cross-domain examples when in-domain labels are absent.
- **vs. Domain Adaptation (e.g., DANN)**: Classical domain adaptation requires training, whereas DIN-Retrieval is entirely training-free, migrating the concept of domain-invariant features from training to inference-time retrieval.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of cross-domain ICL; discovery of DINs has theoretical value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple models and transfer directions with DIN validity and significance tests.
- Writing Quality: ⭐⭐⭐⭐ Clear chain of motivation from failure analysis to design.
- Value: ⭐⭐⭐⭐ Provides a novel approach for ICL in domains where expert knowledge is scarce.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SERL: Self-Examining Reinforcement Learning on Open-Domain](../../AAAI2026/llm_reasoning/serl_self-examining_reinforcement_learning_on_open-domain.md)
- [\[AAAI 2026\] L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention](../../AAAI2026/llm_reasoning/l2v-cot_cross-modal_transfer_of_chain-of-thought_reasoning_v.md)
- [\[NeurIPS 2025\] DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](../../NeurIPS2025/llm_reasoning/dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)
- [\[AAAI 2026\] RPM-MCTS: Knowledge-Retrieval as Process Reward Model with Monte Carlo Tree Search for Code Generation](../../AAAI2026/llm_reasoning/rpm-mcts_knowledge-retrieval_as_process_reward_model_with_monte_carlo_tree_searc.md)
- [\[ACL 2026\] Does Self-Consistency Improve the Recall of Encyclopedic Knowledge?](does_self-consistency_improve_the_recall_of_encyclopedic_knowledge.md)

</div>

<!-- RELATED:END -->
