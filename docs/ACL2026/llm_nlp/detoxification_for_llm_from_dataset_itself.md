---
title: >-
  [Paper Note] Detoxification for LLM from Dataset Itself
description: >-
  [ACL 2026][LLM/NLP][data-level detoxification] This paper proposes HSPD (Hierarchical Semantic-Preserving Detoxification), a pipeline that leverages SoCD (Soft Contrastive Decoding) to guide an LLM in identifying and rewriting toxic segments in raw corpora while preserving semantics, producing detoxified text that can directly replace original training data for fine-tuning. The approach reduces toxicity probability from 0.42 to 0.18 on GPT2-XL and achieves state-of-the-art detoxification on LLaMA2-7B, OPT-6.7B, and Falcon-7B.
tags:
  - ACL 2026
  - LLM/NLP
  - data-level detoxification
  - contrastive decoding
  - semantic preservation
  - pretraining corpus cleaning
  - toxicity mitigation
date: 2026-05-08
content_hash: 38d7e29d27e92004
---

# Detoxification for LLM from Dataset Itself

**Conference**: ACL 2026
**arXiv**: [2604.19124](https://arxiv.org/abs/2604.19124)
**Code**: [GitHub](https://github.com/)
**Area**: LLM/NLP
**Keywords**: data-level detoxification, contrastive decoding, semantic preservation, pretraining corpus cleaning, toxicity mitigation

## TL;DR

This paper proposes HSPD (Hierarchical Semantic-Preserving Detoxification), a pipeline that leverages SoCD (Soft Contrastive Decoding) to guide an LLM in identifying and rewriting toxic segments in raw corpora while preserving semantics, producing detoxified text that can directly replace original training data for fine-tuning. The approach reduces toxicity probability from 0.42 to 0.18 on GPT2-XL and achieves state-of-the-art detoxification on LLaMA2-7B, OPT-6.7B, and Falcon-7B.

## Background & Motivation

**State of the Field**: LLMs trained on internet-scale data inevitably absorb toxic content. Existing detoxification methods operate primarily at the post-training stage (fine-tuning/RLHF) or at inference time (controlled decoding), yet none fundamentally prevent models from acquiring toxic knowledge during pretraining.

**Limitations of Prior Work**: (1) Controlled inference methods (e.g., PPLM, DExperts) tend to degrade generation quality; (2) post-training methods (e.g., DAPT) incur substantial additional computation; (3) all such methods merely *suppress* rather than *eliminate* toxicity—the model still retains toxic knowledge and is only prevented from expressing it.

**Root Cause**: Detoxification at inference or post-training time treats symptoms rather than causes. The fundamental problem lies in the training data itself. However, directly detoxifying data poses a semantic preservation challenge: naively removing toxic content disrupts contextual coherence and knowledge continuity.

**Paper Goals**: To detoxify at the dataset level—rewriting toxic segments in raw corpora into non-toxic, semantically equivalent text that can directly replace the original data.

**Starting Point**: Exploit the text generation capability of LLMs themselves, using contrastive decoding to precisely locate and suppress toxic tokens while preserving the original semantics.

**Core Idea**: A small model fine-tuned on toxic data serves as a "toxicity detector." The divergence signal produced by contrasting this model against the base model during decoding precisely identifies toxic token dimensions, enabling suppression to be applied exclusively to those dimensions while maximally preserving semantics.

## Method

### Overall Architecture

HSPD is a three-stage pipeline: (1) **Detoxification prompt guidance**—prompts are designed to instruct the model to rewrite toxic text into semantics-preserving, non-toxic versions; (2) **SoCD decoding**—during decoding, the logit difference between the toxic small model and the base model adaptively suppresses the top-$k$ most deviant token dimensions; (3) **Multi-temperature sampling + fusion reranking**—multiple candidates are generated across temperatures and the best output is selected via a weighted combination of toxicity score and semantic similarity.

### Key Designs

1. **SoCD (Soft Contrastive Decoding)**:

    - **Function**: Precisely locates and suppresses toxic tokens without coarse global intervention.
    - **Mechanism**: A small model $\theta_{\text{toxic}}$ is first fine-tuned on toxic data. At each decoding step, the logit difference between the toxic model and the base model is computed as $\mathbf{d} = \log(p_{\theta_{\text{toxic}}}) - \log(p_{\theta_{\text{base}}})$. Only dimensions with positive differences (tokens preferred by the toxic model) are retained; a normalized divergence score $\alpha = \ln(1+\delta)/(1+\ln(1+\delta))$ is computed, and only the top-$k = \alpha \times V$ most deviant token dimensions are suppressed, leaving the remaining dimensions unchanged.
    - **Design Motivation**: The aggressive masking strategy in classical contrastive decoding over-suppresses informative dimensions, yielding incoherent detoxified text. SoCD operates solely on the small fraction of most deviant dimensions, preserving the majority of information channels.

2. **Adaptive $k$ Computation**:

    - **Function**: Dynamically adjusts the number of suppressed dimensions according to toxicity intensity at each step.
    - **Mechanism**: $k = \text{clip}(\lceil \alpha V \rceil, k_{\min}, k_{\max})$, where $\alpha$ reflects the distributional divergence between the base and toxic models at the current step. Large divergence → more dimensions biased toward toxicity → stronger suppression; small divergence → only a few dimensions suppressed. Upper and lower bounds prevent extreme cases.
    - **Design Motivation**: Toxicity intensity varies across token positions—"fuck" exhibits large divergence and requires strong intervention, whereas "the" exhibits minimal divergence and requires almost none. A fixed $k$ is either overly conservative or overly aggressive.

3. **Multi-temperature Sampling + Fusion Reranking**:

    - **Function**: Selects the output with the best trade-off between semantic preservation and toxicity reduction from a diverse candidate pool.
    - **Mechanism**: Candidate rewrites are sampled at multiple temperatures; Detoxify computes a non-toxicity score for each, and sentence embeddings measure semantic similarity to the original text. The best candidate is chosen via a weighted combination of both scores.
    - **Design Motivation**: A single sample may yield a poor trade-off between semantic preservation and detoxification. Multi-temperature sampling with reranking identifies a better Pareto-optimal solution over a larger candidate space.

### Loss & Training

The toxic small model is obtained via standard fine-tuning on a toxic dataset $\mathbb{D}$. The HSPD pipeline itself requires no training—it functions as an inference-time detoxification tool. The detoxified corpus is then used directly for further training, simulating a pretraining setting.

## Key Experimental Results

### Main Results

**GPT2-XL Detoxification Performance**

| Method | Toxicity Probability (TP) ↓ | Expected Maximum Toxicity (EMT) ↓ | Perplexity ↓ |
|---|---|---|---|
| Base Model | 0.42 | 0.43 | Baseline |
| DExperts | 0.26 | 0.32 | Slight increase |
| DAPT | 0.30 | 0.35 | Notable increase |
| **HSPD** | **0.18** | **0.20** | Near baseline |

**Multi-model Validation**

| Model | Original TP | HSPD TP | Note |
|---|---|---|---|
| GPT2-XL | 0.42 | 0.18 | −57% |
| LLaMA2-7B | — | Best | Consistently leading |
| OPT-6.7B | — | Best | Consistently leading |
| Falcon-7B | — | Best | Consistently leading |

### Ablation Study

| Configuration | TP | Note |
|---|---|---|
| Full HSPD | 0.18 | Complete pipeline |
| w/o SoCD (prompt only) | 0.28 | SoCD contributes significantly |
| w/o multi-temperature reranking | 0.22 | Reranking provides additional benefit |
| Fixed $k$ (non-adaptive) | 0.24 | Adaptive $k$ outperforms fixed |
| Classical CD (non-SoCD) | 0.30 + semantic loss | SoCD's soft intervention is superior |

### Key Findings

- Data-level detoxification fundamentally reduces the toxicity acquired by the model, rather than merely suppressing it at inference time.
- SoCD's soft intervention (operating only on the top-$k$ most deviant dimensions) achieves a significantly better balance between detoxification and semantic preservation than classical contrastive decoding.
- Adaptive $k$ is critical—it ensures that the intervention intensity at each token position is proportional to the toxicity signal.
- The perplexity of detoxified corpora closely matches that of the original, indicating effective preservation of knowledge and language capability.
- The approach generalizes consistently across four model architectures of varying scales.

## Highlights & Insights

- The paradigm of "detoxifying at the data source" shifts the strategic framing of detoxification—from post-training/inference-time patching to root-cause treatment before pretraining.
- SoCD's adaptive top-$k$ suppression is an elegant engineering design: only the most toxic dimensions are intervened upon, while all others are preserved intact.
- Multi-temperature sampling with fusion reranking provides a practical mechanism for Pareto-optimal selection on the semantic–safety trade-off.

## Limitations & Future Work

- Training a toxic small model for each dataset is required, which, though low-cost, increases pipeline complexity.
- Out-of-distribution toxicity (types not covered by the training data) may not be effectively addressed.
- Detoxified corpora may alter original authorial intent in certain edge cases.
- Future work could explore adaptive detoxification methods that do not require a dedicated toxic small model.

## Related Work & Insights

- **vs. DExperts**: Performs inference-time logit ensembling, which cannot fundamentally eliminate toxicity; HSPD removes toxicity at the data source.
- **vs. DAPT**: Requires post-training adaptation with additional computation and does not fully eliminate toxicity; HSPD operates before training.
- **vs. UniDetox**: Data distillation still requires post-training application; HSPD directly replaces the original data.
- **vs. ParaGeDi**: Achieves semantics-preserving rewriting but operates at inference time; HSPD applies rewriting to the corpus itself.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The data-level detoxification paradigm is novel, and SoCD constitutes an effective improvement over contrastive decoding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Validated across four models with multiple ablations and dual evaluation of detoxification quality and semantic preservation.
- **Writing Quality**: ⭐⭐⭐⭐ Method is clearly described with a well-motivated pipeline design.
- **Value**: ⭐⭐⭐⭐ Offers a practical solution for addressing LLM toxicity at the data source.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Evaluating Text Creativity across Diverse Domains: A Dataset and Large Language Model Evaluator](../../ICLR2026/llm_nlp/evaluating_text_creativity_across_diverse_domains_a_dataset_and_large_language_m.md)
- [\[ACL 2026\] Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness](masked_by_consensus_disentangling_privileged_knowledge_in_llm_correctness.md)
- [\[ACL 2026\] HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction](hcre_llm-based_hierarchical_classification_for_cross-document_relation_extractio.md)
- [\[ACL 2026\] Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffixes](route_to_rome_attack_directing_llm_routers_to_expensive_models_via_adversarial_s.md)
- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)

<!-- RELATED:END -->
