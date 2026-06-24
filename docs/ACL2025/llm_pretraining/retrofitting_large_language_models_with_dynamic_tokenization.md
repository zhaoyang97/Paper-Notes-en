---
title: >-
  [Paper Note] Retrofitting Large Language Models with Dynamic Tokenization
description: >-
  [ACL 2025][LLM Pretraining][Dynamic Tokenization] This paper proposes retrofitting existing language models with dynamic tokenization. It dynamically determines token boundaries using a BPE-inspired subword merging algorithm, combined with a pre-trained embedding-prediction hypernetwork to calculate the embeddings of merged tokens on the fly. It achieves an average >20% reduction in sequence length with less than a 2% performance drop on encoder models…
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Dynamic Tokenization"
  - "Subword Merging"
  - "Embedding-Prediction Hypernetwork"
  - "Inference Acceleration"
  - "Cross-lingual Fairness"
date: 2026-05-08
content_hash: 52e191738234d7a3
---

# Retrofitting Large Language Models with Dynamic Tokenization

**Conference**: ACL 2025  
**arXiv**: [2411.18553](https://arxiv.org/abs/2411.18553)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: Dynamic Tokenization, Subword Merging, Embedding-Prediction Hypernetwork, Inference Acceleration, Cross-lingual Fairness

## TL;DR

This paper proposes retrofitting existing language models with dynamic tokenization. It dynamically determines token boundaries using a BPE-inspired subword merging algorithm, combined with a pre-trained embedding-prediction hypernetwork to calculate the embeddings of merged tokens on the fly. It achieves an average >20% reduction in sequence length with less than a 2% performance drop on encoder models, and up to a 17% sequence reduction on decoder models.

## Background & Motivation

**Background**: Current language models widely use fixed, static subword tokenizers (e.g., BPE, WordPiece, SentencePiece). The vocabulary is determined before pre-training and remains unchanged. Although simple and uniform, this design leads to significant efficiency and capability discrepancies in multilingual scenarios.

**Limitations of Prior Work**: The vocabularies of static tokenizers are typically biased towards English. This results in other languages—especially morphologically rich ones—requiring more tokens to represent the same semantic content, leading to slower inference, inefficient context window utilization, and unbalanced cross-lingual capabilities. For instance, the same sentence in German may require 50% more tokens than in English.

**Key Challenge**: To dynamically adjust token granularity based on different languages or domains, the embedding layer of the model remains a fixed-size lookup table, capable of handling only tokens from the predefined vocabulary. Dynamically changing token boundaries produces novel tokens unseen during training, which the model cannot process directly.

**Goal**: To design a method that enables existing, pre-trained language models to accept dynamic-grained token inputs without retraining, thereby achieving shorter sequence lengths and fairer cross-lingual performance.

**Key Insight**: The authors observe that the BPE algorithm itself constructs a vocabulary through frequency-driven subword merging. If a similar frequency-based merging strategy is applied during inference to merge adjacent subwords into larger units, dynamic tokenization can be achieved without modifying model parameters—provided there is a method to compute the embeddings for these merged tokens.

**Core Idea**: Train a lightweight embedding-prediction hypernetwork that takes the sequence of original embeddings of the merged subwords as input and outputs the embedding vector of the merged token. During inference, dynamic merging is performed first, the hypernetwork is used to calculate the embeddings, and the inputs are then forwarded through the model as usual.

## Method

### Overall Architecture

Given an input text, it is first tokenized into a subword sequence using the original static tokenizer. Subword bigram frequencies are then counted at the batch level, and the most frequent bigram is merged. This process is repeated until a target sequence reduction rate is achieved. For the new tokens produced by merging, a pre-trained hypernetwork is used to predict the merged embedding from the embeddings of their constituent subwords. Finally, the shortened sequence is fed into the original model for standard inference.

### Key Designs

1. **Batch-Level Subword Merging**:

    - **Function**: Dynamically determines which adjacent subwords should be merged into larger tokens during inference.
    - **Mechanism**: Frequencies of all adjacent subword bigrams are calculated across all sequences within a batch. The most frequent bigram is selected for merging, and all occurrences of this bigram in the batch are merged into a new token. This process is iteratively executed until a preset number of merges or a target sequence reduction ratio is reached. This process occurs entirely dynamically during inference, without requiring predefined static merge rules.
    - **Design Motivation**: Highly frequent bigrams are more likely to represent meaningful linguistic units (similar to the core assumption of BPE). Counting at the batch level, as opposed to globally, enables the merging strategy to adapt to the distribution of the current inputs, achieving true dynamism.

2. **Embedding-Prediction Hypernetwork**:

    - **Function**: Computes embedding vectors for the newly generated merged tokens.
    - **Mechanism**: The hypernetwork is a small Transformer architecture. It takes the original embeddings of the $k$ merged subwords (retrieved from the model's embedding lookup table) as input and outputs the embedding vector for the merged token (with the same dimension as the original embeddings). During training, the hypernetwork utilizes contrastive learning and reconstruction objectives: the predicted merged embedding is forced to be close to the contextualized representation generated by the token sequence at the model's intermediate layers. The hypernetwork has a very small parameter footprint (about 1% of the main model) and is frozen after pre-training.
    - **Design Motivation**: Simply averaging or concatenating subword embeddings discards sequential and compositional information. The hypernetwork can learn more complex merging patterns, such as realizing that the semantics of merging "un" + "happy" is not a simple average. Employing a contrastive learning objective aligns the predicted embeddings with the internal contextual representations of the model.

3. **Two Application Modes for Decoder Models (Prefilling & ANN Generation)**:

    - **Function**: Extends dynamic tokenization to autoregressive generative models.
    - **Mechanism**: (a) Prefilling mode—performs dynamic merging exclusively during the prompt encoding stage to shorten the prefill sequence, speeding up KV cache initialization, while retaining token-by-token decoding during generation; (b) ANN mode—uses an Approximate Nearest Neighbor index to maintain a million-scale dynamic vocabulary, directly selecting merged tokens for decoding during generation to accelerate the generation phase.
    - **Design Motivation**: Generation in decoder models is autoregressive, preventing simple post-hoc merging. Prefilling mode is a low-risk strategy (does not alter generation quality), while ANN mode is more aggressive but achieves greater acceleration.

### Loss & Training

The training of the hypernetwork is independent of the main model. Using a large corpus of text data, various possible subword merges are performed on each text segment. The hypernetwork is then trained to predict the embeddings of the merged tokens, with supervision signals derived from the contextualized representations at the corresponding positions in the main model's intermediate layers. The training objective is a hybrid of MSE loss and contrastive loss. The parameters of the main model remain frozen throughout.

## Key Experimental Results

### Main Results

Average results of the encoder model (XLM-R) on XNLI across 14 languages:

| Configuration | Average Sequence Reduction | Average Accuracy Drop | Inference Speedup |
|------|--------------|--------------|---------|
| Original Static Tokenization | 0% | - | 1.0x |
| Dynamic Tokenization (word-level boundary) | 22.3% | 1.7% | ~1.25x |
| Dynamic Tokenization (aggressive) | 30.1% | 3.2% | ~1.40x |

Results of the decoder model (Mistral-7B):

| Mode | Sequence Reduction | Performance Change | Description |
|------|----------|---------|------|
| Prefilling Dynamic Tokenization | Up to 40% (relative to word-level) | Almost lossless | Only accelerates prefilling |
| ANN Generation (1M vocabulary) | 17% | Slight drop | Supports massive dynamic vocabulary |

### Ablation Study

| Configuration | Average XNLI Accuracy | Description |
|------|----------------|------|
| Full Method (Hypernetwork-predicted embeddings) | 82.1% | Paired with dynamic merging |
| Replace Hypernetwork with Average Pooling | 79.8% | Simple average is subpar |
| Fixed Merging Rules (static) | 80.5% | Dynamic outperforms fixed |
| No Merging (original tokenization) | 83.6% | Original baseline |

### Key Findings

- Non-English languages benefit more from dynamic tokenization: morphologically rich languages like German and Turkish achieve sequence reductions of over 30%, whereas English only reaches around 15%, effectively shrinking the cross-lingual token discrepancy.
- Hypernetwork-predicted embeddings significantly outperform simple average pooling (+2.3%), validating that compositional semantics requires non-linear modeling.
- Prefilling mode is almost lossless, serving as the safest engineering deployment strategy.
- The utility of dynamic merging exhibits diminishing marginal returns: the first 10% of sequence reduction is virtually lossless, while each subsequent 5% reduction introduces approximately a 0.5% drop in performance.

## Highlights & Insights

- **The design philosophy of "retrofitting rather than retraining" is highly practical**: It avoids re-pre-training the model, requiring only a lightweight hypernetwork to grant dynamic tokenization capabilities to existing models. This plug-and-play characteristic substantially lowers deployment barriers.
- **The analogy between BPE training inspiration and inference-time dynamic merging is highly elegant**: BPE performs static merging during training to define the vocabulary, while this work performs dynamic merging during inference to adapt to the input, forming a perfect mirror relationship.
- **The perspective on cross-lingual fairness is noteworthy**: Many NLP efficiency studies focus only on English scenarios, whereas this work explicitly optimizes for multilingual fairness, offering immediate utility for global deployments.

## Limitations & Future Work

- Batch-level statistics depend on the distribution of text within the batch; if the batch size is extremely small or text diversity is exceedingly high, the merging strategy can become unstable.
- The hypernetwork's performance begins to degrade when the merge length exceeds 4, which may limit scenarios requiring coarse-grained merging.
- Currently, only XLM-R and Mistral-7B have been evaluated; performance on larger-scale models (e.g., 70B+) remains unknown.
- The engineering complexity of the ANN generation mode is relatively high, necessitating further optimization for practical deployment.

## Related Work & Insights

- **vs BPE Dropout (Provilkov et al., 2020)**: BPE Dropout introduces tokenization randomness during training to boost robustness, but remains a predefined static strategy. In contrast, this work dynamically determines tokenization during inference, offering greater flexibility.
- **vs Charformer (Tay et al., 2021)**: Charformer dynamically learns token representations at the character level but requires training from scratch; the key advantage of this work is the ability to retrofit any existing model.
- **vs Byte-level Models (Yu et al., 2023)**: Byte-level models bypass tokenization completely but suffer from extremely long sequences; operating at the subword level, this work strikes a balance between efficiency and flexibility.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of dynamic tokenization and hypernetwork-based embedding prediction is novel, alongside a unique positioning of "retrofitting existing models".
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of both encoder and decoder architectures across 14 languages, matched with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Method descriptions are clear and intuitive, and the explanations of diagrams and processes are well-executed.
- Value: ⭐⭐⭐⭐ Provides direct helper benefits to multilingual efficiency optimization and deployment; the plug-and-play nature enhances its practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)
- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)
- [\[ACL 2025\] Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation](emergent_abilities_continued_pt.md)
- [\[ACL 2025\] DavIR: Data Selection via Implicit Reward for Large Language Models](davir_data_selection_via_implicit_reward_for_large_language_models.md)

</div>

<!-- RELATED:END -->
