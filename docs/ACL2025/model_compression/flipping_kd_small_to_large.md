---
title: >-
  [Paper Note] Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching
description: >-
  [ACL 2025][Model Compression][knowledge distillation] This paper proposes a "reverse knowledge distillation" paradigm—allowing LLMs to learn domain expertise in text matching from fine-tuned small models. This is achieved by reinterpreting a decoder-only LLM as an encoder-decoder architecture (using the compression matrix of LoRA as the encoder) and designing a Margin-aware Contrastive Loss to align representation similarities.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "knowledge distillation"
  - "text matching"
  - "reverse distillation"
  - "LoRA"
  - "representation learning"
date: 2026-05-08
content_hash: 1ddfe56da8fb7b61
---

# Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching

**Conference**: ACL 2025  
**arXiv**: [2507.05617](https://arxiv.org/abs/2507.05617)  
**Code**: None  
**Area**: Model Compression / Knowledge Distillation  
**Keywords**: knowledge distillation, text matching, reverse distillation, LoRA, representation learning

## TL;DR
This paper proposes a "reverse knowledge distillation" paradigm—allowing LLMs to learn domain expertise in text matching from fine-tuned small models. This is achieved by reinterpreting a decoder-only LLM as an encoder-decoder architecture (using the compression matrix of LoRA as the encoder) and designing a Margin-aware Contrastive Loss to align representation similarities.

## Background & Motivation

**Background**: Knowledge distillation typically involves large models teaching small models. However, in specific tasks such as text matching, fine-tuned small models (e.g., BERT-based) often exhibit better representation learning capabilities than LLMs because they focus on optimizing similarity between input pairs. LLMs directly predict "match/mismatch" text instead of learning the representation space.

**Limitations of Prior Work**: (1) LLMs perform worse than fine-tuned small models in domain-specific text matching (finance, medical)—LLMs lack the ability to finely distinguish domain terminology; (2) decoder-only architectures are not suited for representation learning (lacking a dedicated encoder module); (3) traditional distillation direction (Large $\to$ Small) is not applicable in this scenario.

**Key Challenge**: LLMs have rich semantic understanding but lack fine-grained representation learning capabilities, while SLMs have fine-grained domain representations but lack broad semantic understanding. How to combine the advantages of both?

**Goal**: Let LLMs learn from the representation learning expertise of SLMs to achieve better text matching capabilities.

**Key Insight**: Reinterpret the low-rank matrices of LoRA as an encoder (compression matrix) and a decoder (expansion matrix), enabling decoder-only LLMs with representation learning capabilities.

**Core Idea**: Reverse distillation—the LLM uses the encoder portion of LoRA to generate text representations and then aligns them with the similarity scores of the SLM teacher.

## Method

### Overall Architecture
SLM teacher (fine-tuned BERT-like model) $\to$ computes text pair similarity scores $\to$ LLM student uses LoRA encoder to generate text representations $\to$ computes similarity between representations $\to$ MCL loss aligns the two similarities $\to$ dual-threshold filters noisy labels.

### Key Designs

1. **Encoder-Decoder Reinterpretation of LLM**:

    - **Function**: Treat $W_{down}$ (low-rank compression matrix) of LoRA as the encoder and $W_{up}$ (low-rank expansion matrix) as the decoder.
    - **Mechanism**: The encoder maps input to a low-dimensional representation space, which can be used to compute text pair similarities; the decoder maps the low-dimensional representation back to high-dimensional space for the original LLM output.
    - **Design Motivation**: Decoder-only LLMs do not have a natural encoder, but the structure of LoRA naturally contains a compression $\to$ expansion process.

2. **Margin-aware Contrastive Loss (MCL)**:

    - **Function**: Enables the LLM to learn fine-grained similarity differences between and within positive and negative samples.
    - **Mechanism**: Introducing two margin zones—(1) there should be a sufficient gap between positive and negative samples; (2) differentiation must also be learned within the same category (positive-positive or negative-negative).
    - **Design Motivation**: Traditional contrastive learning only ensures that positive samples have higher similarity than negative ones, without focusing on fine-grained relationships. MCL simultaneously learns inter-class and intra-class relationships.

3. **Dual-threshold Noise Filtering**:

    - **Function**: Filter out inaccurate annotations from the SLM teacher.
    - **Mechanism**: Set a lower bound for positive samples and an upper bound for negative samples to filter out samples where the teacher is uncertain.
    - **Design Motivation**: The SLM teacher is not perfect, and similarity judgments for certain samples might be inaccurate.

## Key Experimental Results

### Main Results: Finance + Medical Text Matching

| Method | Finance F1 | Medical F1 | Description |
|------|--------|--------|------|
| SLM (fine-tuned BERT) | Good | Good | Domain expert |
| LLM (SFT) | Lower than SLM | Lower than SLM | Direct fine-tuning is insufficient |
| LLM + Traditional KD (Large $\to$ Small) | Not applicable | Not applicable | Wrong direction |
| **LLM + FlipKD (Small $\to$ Large)** | **Best** | **Best** | Combines the merits of both |

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| FlipKD Full | Best | MCL + Dual-threshold + LoRA encoder |
| w/o MCL (Standard Contrastive Loss) | Decrease | Lacks margin differentiation |
| w/o Dual-threshold Filtering | Decrease | Teacher noise affects learning |
| Different scale SLM teachers | Only requires reasonable performance| Insensitive to teacher quality |

### Key Findings
- **Reverse distillation is effective**: LLM learns more fine-grained representation abilities from SLM, surpassing direct fine-tuning.
- **Deployed in online environment**: Successfully deployed and validated in ByteDance's production systems.
- **Insensitive to teacher scale**: Requires only a "reasonably good" SLM.
- **Cross-architecture feasibility**: Validated across different scales such as Qwen-0.5B and GLM-10B.

## Highlights & Insights
- **Pioneering the "reverse distillation" paradigm**: Challenges the conventional "large teaches small" distillation wisdom, proving that small models can teach large models on specific tasks. This idea can be migrated to other tasks where large models struggle but small models excel.
- **Novel interpretation of LoRA**: Reinterpreting the low-rank structure of LoRA as an encoder-decoder is a clever design that equips decoder-only LLMs with representation learning capabilities.
- **Industrial deployment validation**: Not just academic research, but tested and validated online with high practical value.

## Limitations & Future Work
- Only validated on text matching tasks; not yet extended to other representation learning tasks (retrieval, clustering, etc.).
- The representation dimension of the LoRA encoder is limited by the LoRA rank, which may limit expressiveness.
- The quality of the SLM teacher still bounds the performance limit.
- Not compared against other LLM representation learning methods (e.g., GritLM, E5-Mistral).

## Related Work & Insights
- **vs Traditional KD (Hinton et al., 2015)**: Traditional KD goes from large to small, whereas this paper goes reversely from small to large, proving more effective on task-specific domains.
- **vs Alpaca/Distillation from GPT-4**: These methods use LLMs to generate data to train small models. This paper allows representation space knowledge of small models to flow into LLMs.
- **vs Text Matching SLM (BERT-based)**: SLMs are strong in domain matching but lack generalization; FlipKD transfers their advantages to LLMs to get the best of both worlds.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The reverse distillation paradigm is novel; the LoRA encoder interpretation is clever.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple domains and multiple models, including online deployment validation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, comprehensive method description.
- Value: ⭐⭐⭐⭐⭐ Inspiring paradigm shift, highly practical for industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation](longred_mitigating_short-text_degradation_of_long-context_large_language_models_.md)
- [\[NeurIPS 2025\] QuadEnhancer: Leveraging Quadratic Transformations to Enhance Deep Neural Networks](../../NeurIPS2025/model_compression/quadenhancer_leveraging_quadratic_transformations_to_enhance_deep_neural_network.md)
- [\[ACL 2025\] Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs](sparse_logit_sampling_accelerating_knowledge_distillation_in_llms.md)
- [\[ACL 2025\] Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation](beyond_logits_aligning_feature_dynamics_for_effective_knowledge_distillation.md)
- [\[ACL 2025\] Capture the Key in Reasoning to Enhance CoT Distillation Generalization](capture_key_cot_distillation.md)

</div>

<!-- RELATED:END -->
