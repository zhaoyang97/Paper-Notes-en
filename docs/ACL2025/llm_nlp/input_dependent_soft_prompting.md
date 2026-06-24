---
title: >-
  [Paper Note] Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs
description: >-
  [ACL 2025][LLM (Other)][Soft Prompting] This work proposes ID-SPAM, which generates **input-dependent** soft prompts by applying a learnable self-attention layer over input token embeddings followed by a bottleneck MLP. When prepended to the input of only a single Transformer layer, it outperforms various soft prompt baselines and demonstrates excellent zero-shot cross-task and cross-domain transferability.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Soft Prompting"
  - "Parameter-Efficient Fine-Tuning"
  - "Self-Attention"
  - "Input-Dependent"
  - "PEFT"
  - "prompt tuning"
date: 2026-05-08
content_hash: 8952f84930ffb620
---

# Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs

**Conference**: ACL 2025  
**arXiv**: [2506.05629](https://arxiv.org/abs/2506.05629)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Soft Prompting, Parameter-Efficient Fine-Tuning, Self-Attention, Input-Dependent, PEFT, prompt tuning

## TL;DR

This work proposes ID-SPAM, which generates **input-dependent** soft prompts by applying a learnable self-attention layer over input token embeddings followed by a bottleneck MLP. When prepended to the input of only a single Transformer layer, it outperforms various soft prompt baselines and demonstrates excellent zero-shot cross-task and cross-domain transferability.

## Background & Motivation

1. Large Language Models (LLMs) require fine-tuning for domain-specific tasks, but full-parameter fine-tuning is computationally expensive (with parameter counts scaling from hundreds of millions to hundreds of billions from BERT to GPT-3). Parameter-Efficient Fine-Tuning (PEFT) has thus become a critical research area.
2. Soft Prompting is a promising class of PEFT methods that adapts models to downstream tasks by learning only a small set of continuous vectors (soft prompts) while keeping the LM parameters frozen, thereby avoiding modifications to the core model architecture.
3. Existing soft prompting methods (such as Prompt Tuning, Prefix Tuning, and P-Tuning) utilize prompt vectors that are **input-independent**, meaning all samples share the exact same prompt parameters. This limits the model's ability to dynamically adapt to different inputs during inference and increases training convergence difficulty.
4. Current input-dependent soft prompting methods suffer from several limitations: (a) they require prepending soft prompts to **multiple** Transformer layers of the LM (which complicates the architecture); (b) they do not explicitly weight the relevance of different tokens in the input; and (c) they significantly increase the number of trainable parameters.
5. A natural intuition is that since task samples contain diverse vocabularies, generating soft prompts should **differentially focus on different input tokens**—which is the core strength of the self-attention mechanism.
6. This paper proposes ID-SPAM (Input Dependent Soft Prompting with self-Attention Mechanism), which aggregates input information using a learnable self-attention layer and then maps it to soft prompts via a bottleneck MLP. By prepending these prompts to **only a single Transformer layer**, ID-SPAM maintains a small parameter footprint and exhibits smooth training.

## Method

### Overall Architecture

Given a downstream task $T$ with training data $D_{train} = \{(x_i, y_i)\}_{i=1}^{K}$. For single-sentence tasks, the input is represented as $x_i = \mathbf{E}(\texttt{[SEP]}S_1\texttt{[EOS]})$; for sentence-pair tasks, the input is $x_i = \mathbf{E}(\texttt{[SEP]}S_1\texttt{[SEP]}S_2\texttt{[EOS]})$, where $\mathbf{E}(\cdot)$ denotes the token embedding layer. ID-SPAM generates the input-dependent soft prompt $\mathbf{S}_T \in \mathbb{R}^{n \times t}$ (where $n$ is the hidden dimension and $t$ is the number of prompt tokens) through three stages, and then prepends it to the input of a specific Transformer layer of the LM. The base LM parameters are entirely frozen, and only the soft prompt generation network is optimized.

### Key Designs

#### Module 1: Self-Attention Aggregation Layer

- **Function**: Applies single-head self-attention to the input embedding $\mathbf{E}$ and takes the mean along the token dimension to obtain a context-rich $n \times 1$ dimensional vector $A$.
- **Mechanism**:
$$A = \text{mean}\left\{\text{softmax}\left(\frac{(\mathbf{E}W_Q)(\mathbf{E}W_K)^\top}{\sqrt{d_k}}\right)(\mathbf{E}W_V)\right\}$$
  where $W_Q, W_K, W_V$ are learnable query, key, and value projection matrices, and $\frac{1}{\sqrt{d_k}}$ is the scaling factor.
- **Design Motivation**: Different tokens contribute differently to downstream tasks (e.g., "excellent" is much more critical than "the" in sentiment classification). Self-attention automatically learns token-level importance weighting, allowing the generated soft prompts to capture **key semantic signals** from the input.

#### Module 2: Bottleneck MLP (Down-Up Projection)

- **Function**: First down-projects the aggregated vector $A$ to a low-dimensional space $c$ ($c < n$), applies a ReLU activation, up-projects it to $n \cdot t$ dimensions, and finally reshapes it into $\mathbf{S}_T \in \mathbb{R}^{n \times t}$.
- **Mechanism**:
$$\mathbf{S}_T = \text{resize}\left(\sigma(W_{up} \cdot \sigma(W_{down} \cdot A))\right)$$
  where $W_{down} \in \mathbb{R}^{n \times c}$, $W_{up} \in \mathbb{R}^{c \times (n \cdot t)}$, and $\sigma$ represents the ReLU activation function.
- **Design Motivation**: The bottleneck structure (akin to the low-rank concept in LoRA) dramatically compresses the parameter size while introducing non-linear transformations to enhance expressive capacity. Using a low-dimensional intermediate representation acts as information compression and prevents overfitting.

#### Module 3: Single-Layer Prepending Strategy

- **Function**: Prepends the generated soft prompt $\mathbf{S}_T$ to the input of a **single** Transformer layer (the $m$-th layer) of the LM, rather than prepending to multiple or all layers.
- **Mechanism**: Empirical findings show that prepending to intermediate layers (e.g., layers 6-8) yields optimal performance. The early layers also perform well because the soft prompts are generated directly from input embeddings, aligning naturally with early-layer outputs.
- **Design Motivation**: (1) Reduces architectural complexity by avoiding additional prepending operations at every layer; (2) lowers the number of trainable parameters compared to Prefix Tuning (which prepends at every layer); and (3) makes the training process smoother, lowering optimization difficulty.

### Loss & Training

- Standard **cross-entropy loss** is used for training under the Adam optimizer.
- All parameters of the base LM are frozen; only the self-attention layer parameters ($W_Q, W_K, W_V$) and the bottleneck MLP parameters ($W_{down}, W_{up}$, and biases) are trained.
- The number of soft prompt tokens is set to $t = 10$, and training is conducted for up to 30 epochs.
- Experiments were performed on NVIDIA A100 80GB GPUs.

## Key Experimental Results

### Main Results: GLUE Benchmark (RoBERTa-LARGE Backbone)

| Method | MNLI | QNLI | SST-2 | MRPC | RTE | QQP | Mean |
|------|------|------|-------|------|-----|-----|------|
| Fine-tuning | 87.6 | 94.7 | 95.4 | 92.1 | 88.4 | 90.7 | 91.5 |
| LoRA | 89.1 | 87.9 | 95.1 | 86.5 | 78.7 | 88.4 | 87.6 |
| Prompt Tuning | 83.4 | 88.2 | 92.6 | 73.9 | 60.8 | 81.2 | 80.0 |
| P-Tuning | 86.4 | 88.7 | **95.8** | 76.3 | 62.6 | 85.2 | 82.5 |
| SMoP | 86.7 | 88.4 | **95.8** | 79.6 | 76.3 | 86.7 | 85.6 |
| LPT | 84.2 | 86.1 | 93.4 | **87.3** | 74.2 | 85.3 | 85.1 |
| DePT | 83.3 | 88.8 | 91.2 | 77.7 | 73.2 | 82.2 | 82.7 |
| **ID-SPAM** | **87.4** | **91.1** | 94.6 | 86.1 | **81.1** | **88.4** | **88.1** |

### Ablation Study: Self-Attention vs. Mean Pooling (RoBERTa-LARGE)

| Method | MRPC | RTE | QQP |
|------|------|-----|-----|
| Mean-pooling | 82.3 | 75.2 | 84.2 |
| **ID-SPAM** | **86.1** | **81.1** | **88.4** |

### Zero-Shot Cross-Task/Cross-Domain Transfer (RoBERTa-LARGE)

| Method | QQP→MRPC | MRPC→QQP | SST-2→IMDB | IMDB→SST-2 |
|------|----------|----------|------------|------------|
| Fine-tuning | 64.0 | 68.3 | 87.1 | 88.8 |
| LoRA | 71.1 | 66.1 | 90.3 | 87.6 |
| LPT | 66.7 | 64.5 | 67.1 | 71.1 |
| **ID-SPAM** | **70.9** | **69.2** | **89.1** | **86.0** |

### Key Findings

- ID-SPAM surpasses all soft prompt baselines on **4 out of 6 GLUE tasks** (using both RoBERTa-BASE and LARGE backbones), achieving a substantially higher mean score.
- On **4 SuperGLUE tasks** using the RoBERTa-LARGE backbone, ID-SPAM achieves the best performance on 3 out of 4 tasks, with a mean of 72.0 (compared to LPT's 70.2 and SMoP's 70.4).
- **Ablation Study** indicates that the self-attention layer brings an average performance improvement of 5.82% compared to direct mean pooling, validating the importance of differentiating token weightings.
- In **zero-shot transfer**, ID-SPAM outperforms other soft prompting methods across all 4 transfer pairs, and even exceeds full-parameter fine-tuning in 3 out of 4 pairs, demonstrating outstanding generalization capability.
- **Layer Analysis**: Prepending soft prompts to middle layers yielded the best performance. ID-SPAM significantly outperforms LPT at almost all layer positions and shows better compatibility with early layers.
- The trainable parameter count and training/inference time of ID-SPAM are superior to or on par with LPT and LoRA (detailed in Appendix D of the paper).

## Highlights & Insights

1. **Simple and Effective Design Philosophy**: By using only a single self-attention layer + a bottleneck MLP + a single-layer prepending strategy, ID-SPAM achieves input-dependent soft prompt generation. This avoids the complexity of multi-layer prepending while maintaining a minimal parameter footprint.
2. **Self-Attention Grants Token-Level Selectivity**: Unlike previous methods that treat all tokens with equal weight, ID-SPAM automatically identifies task-critical tokens and assigns them higher weights. This is the root cause of its stable performance across diverse tasks such as sentiment classification and natural language inference.
3. **Strong Zero-Shot Transferability**: Input-dependent prompt generation naturally possesses better generalization. Since prompts dynamically change with the inputs, they can adapt more flexibly to distribution shifts, whereas fixed prompts are prone to overfitting the training domain distribution.
4. **Complementary Perspective to LoRA**: While LoRA adapts model weights using low-rank matrices, ID-SPAM focuses on input-dependent soft prompts. Both target parameter efficiency through different paths, yet ID-SPAM matches or even outperforms LoRA on several tasks.

## Limitations & Future Work

1. **Limited Backbone Scale**: Experiments were only conducted on RoBERTa-BASE/LARGE (125M/355M) and GPT-2, leaving the method untested on large-scale models such as LLaMA-3.1-70B or Mixtral 8×22B. It remains unconfirmed whether this approach maintains its edge on state-of-the-art LLMs.
2. **Manual Layer Selection**: Identifying the optimal Transformer layer for prompt injection currently requires manual grid search, lacking an automatic layer selection mechanism. Future work could explore differentiable routing (e.g., Gumbel-Softmax) or weighted multi-layer fusion.
3. **Inherent Limitation to NLU Tasks**: Evaluation is confined to classification and inference tasks (GLUE/SuperGLUE), without covering generative tasks (summarization, translation, dialogue, etc.). Performance in generative scenarios remains unexplored.
4. **Single-Head Attention**: The current version uses only single-head self-attention. Exploring multi-head attention could capture richer token interaction patterns.
5. **Combination with Other PEFT Methods**: ID-SPAM is orthogonal to methods like LoRA and Adapters. Combining them could yield additional gains but is not explored in the paper.

## Related Work & Insights

- **Prompt Tuning** (Lester et al., 2021) and **Prefix Tuning** (Li & Liang, 2021) are foundational works in soft prompting, prepending fixed prompts to the embedding layer and all layers, respectively.
- **LPT** (Liu et al., 2022a) introduces "late prompting," injecting prompts only at intermediate layers, making it a direct baseline for ID-SPAM.
- **SMoP** (Choi et al., 2023) uses multiple short prompts with a gating routing strategy, based on the intuition of "matching different prompts to different subsets of data," contrasting with ID-SPAM's method of "generating a custom prompt for every sample."
- **DePT** (Shi & Lipani, 2024) compresses soft prompt parameters through low-rank decomposition, sharing similarities with ID-SPAM's bottleneck MLP design.
- **Insight**: Input-dependent generation combined with attention weighting is a versatile design pattern. It can be extended to other PEFT methods like Adapters and LoRA (e.g., dynamically generating low-rank matrices based on inputs).

## Rating

- **Novelty**: ⭐⭐⭐ — The idea of generating soft prompts via self-attention and a bottleneck MLP is intuitive and clear, but the architectural innovation is moderate as the core components are combinations of existing modules.
- **Technical Quality**: ⭐⭐⭐⭐ — The experiments cover GLUE, SuperGLUE, and zero-shot transfer with comprehensive baselines. The ablation studies clearly validate the contribution of the self-attention layer.
- **Value**: ⭐⭐⭐ — The method is simple, easy to implement, and parameter-efficient, but it is only validated on small-to-medium models, leaving its applicability to modern LLMs questionable.
- **Writing Quality**: ⭐⭐⭐⭐ — The paper is well-structured, with complete formulas, intuitive diagrams, and properly formatted experimental tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](leveraging_in-context_learning_for_political_bias_testing_of_llms.md)
- [\[ACL 2025\] SkillAggregation: Reference-free LLM-Dependent Aggregation](skillaggregation_reference-free_llm-dependent_aggregation.md)
- [\[ACL 2025\] Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding](enhancing_input-label_mapping_in_in-context_learning_with_contrastive_decoding.md)
- [\[ACL 2025\] Can LLMs Simulate L2-English Dialogue? An Information-Theoretic Analysis of L1-Dependent Biases](can_llms_simulate_l2-english_dialogue_an_information-theoretic_analysis_of_l1-de.md)
- [\[ACL 2025\] Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)

</div>

<!-- RELATED:END -->
