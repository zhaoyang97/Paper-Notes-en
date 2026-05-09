---
title: >-
  [Paper Note] TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA
description: >-
  [ICLR 2026][Model Compression][LoRA Transfer] This paper proposes TiTok, a framework that enables efficient cross-model transfer of LoRA adapters via token-level contrastive excess scores, without requiring an auxiliary discriminator model. TiTok consistently outperforms TransLoRA and knowledge distillation baselines on reasoning and personalization tasks.
tags:
  - ICLR 2026
  - Model Compression
  - LoRA Transfer
  - Knowledge Distillation
  - Token-level Selection
  - Parameter-Efficient Fine-Tuning
  - Contrastive Excess Score
date: 2026-05-08
content_hash: 6a4a649f0e584a8e
---

# TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA

**Conference**: ICLR 2026
**arXiv**: [2510.04682](https://arxiv.org/abs/2510.04682)
**Code**: [https://github.com/NaughtyMaltiz16/TiTok](https://github.com/NaughtyMaltiz16/TiTok)
**Area**: Model Compression
**Keywords**: LoRA Transfer, Knowledge Distillation, Token-level Selection, Parameter-Efficient Fine-Tuning, Contrastive Excess Score

## TL;DR

This paper proposes TiTok, a framework that enables efficient cross-model transfer of LoRA adapters via token-level contrastive excess scores, without requiring an auxiliary discriminator model. TiTok consistently outperforms TransLoRA and knowledge distillation baselines on reasoning and personalization tasks.

## Background & Motivation

- **The binding problem of LoRA**: Although PEFT methods such as LoRA are parameter-efficient, adapter parameters are tightly coupled to a specific base model and cannot be directly transferred across models.
- **Limitations of Prior Work**:
    - Knowledge distillation (KD) requires access to the original training data, which is typically unavailable.
    - TransLoRA addresses data dependency through synthetic data but requires training an additional discriminator model for data filtering, introducing extra complexity.
- **Core Motivation**: Can task-relevant knowledge signals be extracted at the token level from a LoRA adapter in a more lightweight manner, so as to guide cross-model knowledge transfer?

## Method

### Overall Architecture

TiTok consists of three stages:
1. Synthetic data generation → 2. Excess score computation → 3. Target model training with filtering

### Key Design 1: Token-level Contrastive Excess Score

The token-level score difference between the source model with and without LoRA is defined as:

$$S(y_i) = L_e(y_i) - L_a(y_i)$$

where:

$$L_a(y_i) = \log P_{\mathcal{M}_s}(y_i \mid \mathbf{q}, \mathbf{y}_{<i}), \quad L_e(y_i) = \log P_{\mathcal{M}_s + \mathcal{A}_s}(y_i \mid \mathbf{q}, \mathbf{y}_{<i})$$

- **Intuition**: The excess score quantifies the amount of task knowledge injected by the LoRA adapter. When the base model is uncertain about a token but the LoRA-augmented model predicts it with high confidence, that token receives a high excess score.
- **Theoretical Basis**: This is equivalent to a token-level log-likelihood ratio (LLR), which is guaranteed by the Neyman–Pearson lemma to be the most powerful statistic for distinguishing the two model distributions.

### Key Design 2: Two-level Filtering for Training

**Stage 1 — Sample Filtering**: The average excess score of each synthetic sample is computed, and the top-$M$ most informative samples are retained:

$$\bar{S}_j = \frac{1}{|\mathbf{y}_j|} \sum_{y_i \in \mathbf{y}_j} S(y_i)$$

**Stage 2 — Token Selection**: Within the retained samples, only tokens in the top-$k\%$ of excess scores are used for training:

$$\mathcal{L}_{\text{TiTok}} = \sum_{(\mathbf{q}_j, \mathbf{y}_j) \in \mathcal{D}_f} \sum_{y_i \in \mathbf{y}_j} I_{k\%}(y_i) \cdot L_t(y_i)$$

### Key Design 3: Tokenizer Alignment Algorithm

When the source and target models employ different tokenizers:
- A dual-pointer incremental decoding scheme is used to align text spans.
- Four rules propagate the mask: one-to-one direct copy, one-to-many copy, many-to-one averaging, and many-to-many average copy.
- A final top-$k\%$ selection retains the most reliable target tokens.

### Loss & Training

The target LoRA $\mathcal{A}_t$ is trained on top of the frozen backbone $\mathcal{M}_t$ using filtered synthetic data with a standard NLL loss:

$$\mathcal{L}_{\text{TiTok}} = \sum \sum I_{k\%}(y_i) \cdot (-\log P_{\mathcal{M}_t + \mathcal{A}_t}(y_i \mid \mathbf{q}, \mathbf{y}_{<i}))$$

## Key Experimental Results

### Main Results: Four Transfer Settings

| Transfer Setting | Method | BBH Acc | MMLU Acc | News R-1 | Scholarly R-1 |
|----------|------|---------|----------|----------|--------------|
| Mistral→Mistral | Vanilla | 0.397 | 0.557 | 0.117 | 0.381 |
| Mistral→Mistral | TransLoRA | 0.416 | 0.534 | 0.156 | 0.447 |
| Mistral→Mistral | **TiTok** | **0.424** | **0.561** | **0.161** | **0.473** |
| Mistral→Llama3 | Vanilla | 0.469 | 0.469 | 0.125 | 0.444 |
| Mistral→Llama3 | TransLoRA | 0.473 | 0.473 | 0.126 | 0.461 |
| Mistral→Llama3 | **TiTok** | **0.484** | **0.485** | **0.139** | **0.464** |
| Llama2→Llama3 | **TiTok** | **0.488** | **0.477** | **0.138** | **0.461** |

### Ablation Study

| Sample Filtering | Token Selection | BBH | MMLU | News R-1 | Scholarly R-1 |
|----------|-----------|-----|------|----------|---------------|
| ✗ | ✗ | 0.458 | 0.485 | 0.133 | 0.456 |
| ✗ | ✓ | 0.463 | 0.496 | 0.137 | 0.460 |
| ✓ | ✗ | 0.470 | 0.500 | 0.139 | 0.460 |
| ✓ | ✓ | **0.483** | **0.501** | **0.142** | **0.464** |

### Key Findings

- TiTok outperforms the vanilla target model by an average of +9.94%, KD by +8.5%, and TransLoRA by +4.4%.
- The method is effective across model families (Mistral→Llama), scales (3B→8B), and versions (Llama2→Llama3).
- Tokens in the top 20% of excess scores contain the most concentrated task knowledge (0.482 vs. bottom 0.468).
- Two different model experts (Mistral 7B and Llama2 7B) share a 59.76% overlap in their top 20% token selections.
- A token selection ratio of $k\%$ = 70% is optimal in most settings.
- TiTok remains effective when out-of-domain external data is used.

## Highlights & Insights

- **Simple yet effective**: No auxiliary model (discriminator) needs to be trained; the method leverages only the difference between the source model with and without LoRA.
- **Theoretically grounded**: The excess score is backed by statistical hypothesis testing theory via the log-likelihood ratio.
- **Comprehensive transfer scenarios**: Experiments cover same-family, cross-family, cross-scale, and cross-version transfer settings.
- **Tokenizer alignment**: An elegant solution is provided for tokenizer mismatches across different models.

## Limitations & Future Work

- The method depends on the quality of synthetic data; a source model with weak generation capability may limit transfer performance.
- The optimal token selection ratio $k\%$ is not fully consistent across transfer settings (e.g., the optimal value for Llama3 3B→8B is 30%).
- Validation is limited to LoRA (rank=8); other PEFT methods remain unexplored.
- Evaluation is concentrated on reasoning (BBH/MMLU) and personalization (LaMP) tasks; generalization to other task types requires further investigation.

## Related Work & Insights

- **PEFT Transfer**: TransLoRA transfers LoRA via synthetic data and a discriminator, constituting a heavier pipeline.
- **Knowledge Distillation**: Traditional KD operates at the logit or sequence level within a teacher–student framework and requires access to the original training data.
- **Selective Token Training**: Inspired by the selective training literature, TiTok is the first to extend token selection to the knowledge transfer setting.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ★★★★☆ |
| Theoretical Depth | ★★★★☆ |
| Experimental Thoroughness | ★★★★☆ |
| Value | ★★★★☆ |
| Writing Quality | ★★★★☆ |

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging](../../ACL2026/model_compression/lora_on_the_go_instance-level_dynamic_lora_selection_and_merging.md)
- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [\[ICLR 2026\] Token Distillation: Attention-Aware Input Embeddings for New Tokens](token_distillation_attention-aware_input_embeddings_for_new_tokens.md)
- [\[ICLR 2026\] AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution](amid_knowledge_distillation_for_llms_with_α-mixture_assistant_distribution.md)
- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)

<!-- RELATED:END -->
