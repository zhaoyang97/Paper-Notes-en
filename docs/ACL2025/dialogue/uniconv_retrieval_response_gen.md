---
title: >-
  [Paper Note] UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations
description: >-
  [ACL 2025][Dialogue Systems][Conversational Search] This work explores how to unify dense retrieval and response generation in conversational scenarios into a single LLM. Through three joint training objectives (conversational retrieval, response generation, and context identification instruction) and a data discrepancy mitigation mechanism, it achieves mutual reinforcement of retrieval and generation across five conversational search datasets…
tags:
  - "ACL 2025"
  - "Dialogue Systems"
  - "Conversational Search"
  - "Unified Model"
  - "Retrieval & Generation"
  - "Joint Training"
  - "LLM"
date: 2026-05-08
content_hash: 1ad8ab3de9b0cc84
---

# UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations

**Conference**: ACL 2025  
**arXiv**: [2507.07030](https://arxiv.org/abs/2507.07030)  
**Code**: None (Amazon)  
**Area**: Dialogue Systems  
**Keywords**: Conversational Search, Unified Model, Retrieval & Generation, Joint Training, LLM  

## TL;DR

This work explores how to unify dense retrieval and response generation in conversational scenarios into a single LLM. Through three joint training objectives (conversational retrieval, response generation, and context identification instruction) and a data discrepancy mitigation mechanism, it achieves mutual reinforcement of retrieval and generation across five conversational search datasets, outperforming pipeline-based baselines.

## Background & Motivation

**Background**: Conversational search systems typically employ decoupled retrievers and generators, trained and deployed separately. Commercial conversational search engines combining RAG (e.g., Perplexity.ai, SearchGPT) are becoming increasingly popular.

**Limitations of Prior Work**: (a) Decoupled models fail to leverage mutual internal knowledge simultaneously, where superior retrieval performance does not necessarily benefit generation (the risk of inconsistency); (b) deploying and maintaining two models increases hardware requirements and maintenance costs; (c) existing unified methods can only cover two of the three aspects ("understanding conversational context", "independent retrieval", or "generating responses"), and no single model can perform all three simultaneously.

**Key Challenge**: The retrieval objective (contrastive embedding similarity) and the generation objective (autoregressive cross-entropy) use different training paradigms, where optimizing one may harm the other during joint training. Furthermore, existing training data formats do not distinguish the distinct output requirements of retrieval and generation.

**Goal**: To build the first unified LLM capable of simultaneously performing conversational understanding, independent retrieval, and response generation.

**Key Insight**: Design three complementary training objectives along with a data discrepancy mitigation strategy to enable retrieval and generation knowledge to work synergistically within a single model.

**Core Idea**: Bridge the retrieval and generation training objectives via the Context Identification Instruction (CII), while incorporating format-matched conversational search data to mitigate data discrepancy.

## Method

### Overall Architecture
Based on a decoder-only LLM (e.g., LLaMA), UniConv implements three joint training objectives: (1) conversational dense retrieval (InfoNCE contrastive loss); (2) conversational response generation (autoregressive loss using a session-masked technique); (3) context identification instruction (concatenating queries with positive passages and using contrastive learning to distinguish good and bad responses). The overall loss is: $\mathcal{L} = \mathcal{L}_R + \mathcal{L}_G + \alpha \mathcal{L}_{CII}$.

### Key Designs

1. **Conversational Dense Retrieval**:

    - **Function**: Enables the LLM to act as a dense retriever that understands context.
    - **Mechanism**: Concatenates multi-turn conversation history $\mathcal{H}_n$ with the current query $q_n$ into a complete conversational session query $q'_n$. The hidden state of the end-of-sequence `</s>` token is used as the query/passage representation, trained with the InfoNCE contrastive loss.
    - **Design Motivation**: Inherits the paradigm of ChatRetriever while preventing the degradation of generation capabilities.

2. **Context Identification Instruction (CII)**:

    - **Function**: Bridges retrieval and generation, reducing training-inference discrepancy.
    - **Mechanism**: Concatenates queries and positive passages as input to contrast correct responses against incorrect ones, encouraging the model to learn to "select the correct answer given the retrieved context." $\mathcal{L}_{CII}$ utilizes a contrastive learning loss.
    - **Design Motivation**: In training, retrieval and generation are conducted separately, whereas during inference, the model must utilize retrieval results to assist generation. CII simulates this inference scenario, narrowing the training-inference gap.

3. **Data Discrepancy Mitigation**:

    - **Function**: Resolves the training data format mismatch.
    - **Mechanism**: Previous methods used responses as passages (for retrieval training) or passages as responses (for generation training). However, a unified model requires each training sample to contain both a passage and a response simultaneously. Introducing conversational search datasets (such as QReCC) ensures that each training sample has an independent query-passage-response triplet.
    - **Design Motivation**: Format-matched data allows the model to distinguish "when to act as a retriever" and "when to act as a generator."

### Loss & Training
- Joint training of three objectives: $\mathcal{L} = \mathcal{L}_R + \mathcal{L}_G + \alpha \mathcal{L}_{CII}$
- Bi-encoder + InfoNCE for retrieval, session-masked autoregression for generation.
- Based on Mistral-7B

## Key Experimental Results

### Main Results (Five Conversational Search Datasets)

| Dimension | Decoupled Model (Retriever + Generator) | UniConv | Explanation |
|------|----------------------|---------|------|
| Retrieval Performance (MRR/Recall) | ChatRetriever (Best) | **Outperforms or Comparable** | Retrieval capability is preserved |
| Generation Performance (ROUGE/F1) | ChatQA (Best) | **Outperforms** | Generation capability is enhanced |
| Retrieval-Generation Consistency | Good retrieval does not guarantee good generation | **More Consistent** | Attributed to CII |
| Deployment Cost | Two models | **Single model** | Reduced by half |

### Ablation Study

| Configuration | Outcome | Explanation |
|------|------|------|
| w/o CII | Generation performance drops | CII is a critical bridge |
| w/o Data Discrepancy Mitigation | Retrieval-generation inconsistency deepens | Format-matched data is crucial |
| Retrieval-only Training | Generation capability collapses | Validates the necessity of joint training |
| Generation-only Training | Lacks retrieval capability | Ditto |

### Key Findings
- UniConv is the first unified model capable of simultaneously handling conversational understanding, open-domain retrieval, and response generation.
- Retrieval and generation mutually reinforce each other—retrieval improves the input quality for generation, while generation provides better understaning of the query.
- CII is the key to ensuring consistency—without it, the model cannot effectively utilize even high-quality retrieved passages.
- Data format is more important than data quantity—a small amount of format-matched data outperforms a large amount of format-mismatched data.
- The unified model also outperforms decoupled pipeline models in cross-dataset generalization.

## Highlights & Insights

- **"Serving as both retriever and generator within a single model"** is a natural evolution for conversational search, reducing deployment costs while enhancing consistency.
- **The CII training objective** elegantly simulates the RAG inference scenario—enabling the model to learn "utilizing retrieved passages for generation" during training, rather than separating retrieval and generation training entirely.
- **Data discrepancy mitigation** highlights a neglected issue—the format of training data needs to match the actual output format of downstream tasks.
- This unified paradigm can be extended to more conversational tasks (e.g., conversational recommendation, dialogue summarization).

## Limitations & Future Work

- Based on Mistral-7B, larger-scale models might exhibit different behaviors.
- Training requires simultaneously preparing annotated data for both retrieval and generation.
- The performance of bi-encoder retrieval might not match dedicated retrieval models (e.g., ColBERT).
- Multimodal conversational search scenarios remain unexplored.

## Related Work & Insights

- **vs ChatRetriever**: Only performs retrieval, and generation capability collapses after fine-tuning; UniConv maintains both.
- **vs ChatQA/RankRAG**: Requires an external retriever to provide passages, lacking independent retrieval capability; UniConv retrieves independently.
- **vs GRIT**: General retrieval and generation unification but lacks support for multi-turn conversations; UniConv is tailored specifically for conversational scenarios.
- **vs OneGen**: Embeds retrieval tokens into the generation process but lacks independent retrieval; UniConv's bi-encoder can operate independently.

## Rating

- Novelty: ⭐⭐⭐⭐ First unified LLM delivering conversational understanding, open-domain retrieval, and response generation concurrently; the design of the CII training objective is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across five datasets with comprehensive ablations, consistency analysis, and comparisons against strong baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation (with a straightforward capability comparison in Table 1); the three training objectives are independent yet complementary, supported by complete mathematical formalization.
- Value: ⭐⭐⭐⭐ Direct practical value for conversational search systems; reduces deployment footprint while enhancing semantic consistency, marking a pivotal direction towards conversational AI productization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Less is More: Local Intrinsic Dimensions of Contextual Language Models](../../NeurIPS2025/dialogue/less_is_more_local_intrinsic_dimensions_of_contextual_language_models.md)
- [\[ICML 2025\] Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents](../../ICML2025/dialogue/position_uncertainty_quantification_needs_reassessment_for_large-language_model_.md)
- [\[ACL 2025\] ReflectDiffu: Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework](reflectdiffu_empathetic_response.md)
- [\[ICLR 2026\] Flipping the Dialogue: Training and Evaluating User Language Models](../../ICLR2026/dialogue/flipping_the_dialogue_training_and_evaluating_user_language_models.md)
- [\[ACL 2026\] LOCKET: Robust Feature-Locking Technique for Language Models](../../ACL2026/dialogue/locket_robust_feature-locking_technique_for_language_models.md)

</div>

<!-- RELATED:END -->
