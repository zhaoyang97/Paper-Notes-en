---
title: >-
  [Paper Note] ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering
description: >-
  [CVPR 2026][Reinforcement Learning][KB-VQA] This paper proposes ReAG, a reasoning-augmented multimodal RAG framework that combines coarse- and fine-grained retrieval with a Critic filtering model to reduce noise…
tags:
  - "CVPR 2026"
  - "Reinforcement Learning"
  - "KB-VQA"
  - "RAG"
  - "Reasoning Augmentation"
  - "Multimodal Retrieval"
date: 2026-05-08
content_hash: 3fcff8c723193eed
---

# ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering

**Conference**: CVPR 2026
**arXiv**: [2511.22715](https://arxiv.org/abs/2511.22715)
**Code**: [aimagelab.github.io/ReAG](https://aimagelab.github.io/ReAG)
**Area**: Reinforcement Learning
**Keywords**: KB-VQA, RAG, Reinforcement Learning, Reasoning Augmentation, Multimodal Retrieval

## TL;DR

This paper proposes ReAG, a reasoning-augmented multimodal RAG framework that combines coarse- and fine-grained retrieval with a Critic filtering model to reduce noise, and trains a generator via GRPO reinforcement learning to perform explicit reasoning, achieving new state-of-the-art performance on knowledge-intensive VQA.

## Background & Motivation

Knowledge-based visual question answering (KB-VQA) requires models to answer domain-specific questions that go beyond visual content alone, necessitating retrieval of relevant information from external knowledge bases (e.g., Wikipedia). Even state-of-the-art multimodal large language models (MLLMs) underperform on domain knowledge that is underrepresented in pretraining data.

Existing retrieval-augmented methods suffer from two core issues:

**High retrieval noise**: User queries are highly heterogeneous, and external knowledge bases can contain millions of documents, resulting in low recall, substantial noise, and irrelevant passages being fed to the MLLM.

**Weak reasoning capability**: Even when relevant documents are retrieved, extracting the correct information and reasoning toward an answer is non-trivial; existing methods lack explicit reasoning over retrieved content.

The core mechanism of ReAG follows a **filter-then-reason** paradigm: multi-level retrieval combined with Critic filtering reduces noisy inputs, while reinforcement learning trains the generator to perform explicit reasoning over retrieved content.

## Method

### Overall Architecture

ReAG consists of four main stages:
1. Multi-level retrieval (coarse-grained + fine-grained)
2. Critic model filtering
3. Generator cold-start SFT
4. Reinforcement learning training

### Key Designs

1. **Multi-level Retrieval**:

    - **Coarse-grained retrieval**: EVA-CLIP-8B encodes the full query image and retrieves top-k documents from the knowledge base via cosine similarity, producing candidate passage set $\mathcal{P}^{cg}$.
    - **Fine-grained retrieval**: GroundingDINO detects visual entities mentioned in the question; cropped regions of interest are used for a second retrieval pass to obtain $\mathcal{P}^{fg}$, compensating for details missed by whole-image retrieval.
    - Results from both stages are merged and re-ranked by relevance; all passages from the top-k documents form $\mathcal{P}^{noisy}$.
    - **Design Motivation**: Single coarse-grained retrieval yields insufficient recall; fine-grained retrieval focused on question-relevant visual regions improves recall.

2. **Critic Filtering Model**:

    - An autoregressive MLLM fine-tuned from Qwen2.5-VL-3B that takes $(I_q, q, p)$ as input and predicts whether passage $p$ is relevant to the question.
    - Only passages whose predicted "Yes" probability exceeds a threshold are retained, forming $\mathcal{P}^{relevant}$.
    - **Design Motivation**: Increasing $k$ improves recall but degrades precision; the Critic model effectively removes noisy passages and is decoupled from the retrieval backbone, making it adaptable to any retrieval engine.

3. **Generator Cold Start (SFT)**:

    - A multi-stage training strategy inspired by DeepSeek-R1; SFT is applied first to establish initial reasoning capability.
    - High-quality reasoning trajectories $tr$ are collected from an MLLM, with reasoning processes and final answers delimited by `<think>` and `<answer>` special tokens.
    - Loss function: $\mathcal{L}_{SFT} = \alpha \mathcal{L}_A + (1-\alpha)\mathcal{L}_T$, where $\alpha=0.8$ assigns higher weight to the answer.

4. **GRPO Reinforcement Learning Training**:

    - Built on the GRPO framework with improvements from DAPO (removal of KL divergence penalty, token-level loss computation).
    - At each iteration, $N=8$ completions are generated per $(I_q, q, p)$ sample, and advantage values are computed via rule-based rewards.
    - **Design Motivation**: SFT serves only as a cold start; RL further improves the quality and robustness of the model's reasoning over retrieved evidence.

### Loss & Training

- **Reward design**: $R_i = \gamma R_{task}(o_i) + \delta R_{fmt}(o_i)$, where $\gamma=1.0$, $\delta=0.2$.
    - Task reward: parses and verifies correctness according to question type (numerical/textual, single/multi-answer).
    - Format reward: checks adherence to the `<think>...<answer>...` template.
- The visual encoder is frozen; only the MLP adapter and LLM weights are updated.
- The RL stage uses the Adam optimizer with a learning rate of $1 \times 10^{-6}$, 128 prompts per batch, and 8 completions per prompt.

## Key Experimental Results

### Main Results

Results using EVA-CLIP-8B as the retriever:

| Dataset | Metric | ReAG (3B) | ReflectiVA (3B) | Gain |
|--------|------|-----------|-----------------|------|
| E-VQA (All) | Accuracy | 42.9 | 35.2 | +7.7 |
| InfoSeek (All) | Accuracy | 43.3 | 38.9 | +4.3 |

| Dataset | Metric | ReAG (7B) | VLM-PRF (InternVL3-8B) | Gain |
|--------|------|-----------|------------------------|------|
| E-VQA (Single-Hop) | Accuracy | 44.9 | 40.1 | +4.8 |
| E-VQA (All) | Accuracy | 47.0 | 39.2 | +7.8 |
| InfoSeek (All) | Accuracy | 47.2 | 42.5 | +4.7 |

Using the OMGM retriever yields further gains: ReAG (7B) achieves 52.5% on E-VQA and 49.2% on InfoSeek.

Using Oracle Wikipedia pages (upper-bound experiment): ReAG (7B) achieves 81.5% on E-VQA and 59.7% on InfoSeek.

### Ablation Study

| Configuration | E-VQA (Single-Hop) | InfoSeek (All) | Note |
|------|-------|---------|------|
| No retrieval (zero-shot) | 21.9 | 18.3 | Relies on internal knowledge only |
| Coarse-grained retrieval | 19.2 | 10.1 | Noisy passages degrade performance |
| Coarse + Fine + Critic | 40.2 | 27.1 | Filtering yields substantial gains |
| + SFT | 39.3 | 37.5 | Reasoning capability greatly improves InfoSeek |
| + SFT + Reasoning Trajectories | 38.1 | 41.3 | Explicit reasoning further improves performance |
| + SFT + RL (ReAG Full) | 41.3 | 43.3 | RL provides final performance gains |

### Key Findings

1. **Critic filtering is critical**: Without the Critic, noisy passages from coarse-grained retrieval reduce performance below the zero-shot baseline, underscoring noise management as a key factor in RAG systems.
2. **RL outperforms pure SFT**: The reinforcement learning stage yields significant gains on both benchmarks, validating the effectiveness of reward-based reasoning optimization.
3. **Reasoning trajectories are interpretable**: The generated reasoning chains reveal the usefulness of retrieved passages and the derivation steps, providing full explainability.
4. **ReAG is retrieval-backbone-agnostic**: The Critic model can be seamlessly placed on top of any retrieval engine.

## Highlights & Insights

- **Filter-first design philosophy**: Unlike most RAG methods that train the generator to handle noise, ReAG reduces noise at the source via the Critic, allowing the generator to focus on high-quality reasoning.
- **Multi-stage SFT → RL training strategy**: Drawing inspiration from DeepSeek-R1, SFT serves only as a cold start to establish initial reasoning behavior, while RL is responsible for genuinely improving reasoning quality.
- **Complementarity of fine-grained retrieval**: Detecting visual entities in the question and cropping relevant image regions yields retrieval results more closely aligned with the question, effectively complementing coarse-grained retrieval.
- **Quantitative validation of noise severity in RAG**: Ablation experiments clearly demonstrate that unfiltered retrieval results can degrade performance.

## Limitations & Future Work

1. The Critic model may produce misjudgments; a finer-grained relevance assessment (e.g., passage quality scoring rather than binary classification) may be more effective.
2. The retrieval stage uses a fixed top-k; adaptive retrieval count selection could further improve efficiency.
3. Evaluation is currently limited to the Wikipedia knowledge base; generalization to other domains (e.g., medical, legal) remains unexplored.
4. Generating reasoning trajectories increases inference time, requiring a trade-off between reasoning depth and latency in deployment.

## Related Work & Insights

- **ReflectiVA**: Uses control tokens to guide retrieval and knowledge assessment, but lacks explicit reasoning.
- **VLM-PRF**: Employs external tools for knowledge filtering, conceptually similar to ReAG's Critic but with a different implementation.
- **DeepSeek-R1 / GRPO**: Provides the methodological foundation for RL-enhanced reasoning.
- **Search-R1**: Integrates retrieval and reasoning for complex queries, inspiring ReAG's multimodal extension.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of Critic filtering and RL-based reasoning is effective, though individual components are not entirely novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Two standard benchmarks, multiple retrievers, detailed ablations, oracle upper-bound experiments)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, comprehensive ablations, intuitive figures and tables)
- Value: ⭐⭐⭐⭐⭐ (Provides a complete and effective solution for knowledge-augmented VQA)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering](../../NeurIPS2025/reinforcement_learning/knowledge-based_visual_question_answer_with_multimodal_processing_retrieval_and_.md)
- [\[AAAI 2026\] TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction](../../AAAI2026/reinforcement_learning/tadarag_task_adaptive_retrieval-augmented_generation_via_on-the-fly_knowledge_gr.md)
- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](../../ACL2026/reinforcement_learning/table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[CVPR 2026\] Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement](seeing_is_improving_visual_feedback_for_iterative_text_layout_refinement.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](../../ACL2026/reinforcement_learning/language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)

</div>

<!-- RELATED:END -->
