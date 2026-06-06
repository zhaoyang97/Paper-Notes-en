---
title: >-
  [Paper Note] Detoxification for LLM from Dataset Itself
description: >-
  [ACL 2026][LLM Safety][Data-level detoxification] This paper proposes the HSPD (Hierarchical Semantic-Preserving Detoxification) pipeline…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Data-level detoxification"
  - "contrastive decoding"
  - "semantic preservation"
  - "pre-training corpus cleaning"
  - "toxicity mitigation"
date: 2026-05-08
content_hash: 5ed0a7583ca08878
---

# Detoxification for LLM from Dataset Itself

**Conference**: ACL 2026  
**arXiv**: [2604.19124](https://arxiv.org/abs/2604.19124)  
**Code**: [GitHub](https://github.com/)  
**Area**: LLM/NLP  
**Keywords**: Data-level detoxification, contrastive decoding, semantic preservation, pre-training corpus cleaning, toxicity mitigation

## TL;DR

This paper proposes the HSPD (Hierarchical Semantic-Preserving Detoxification) pipeline, which uses SoCD (Soft Contrastive Decoding) to guide LLMs in locating and rewriting toxic segments within the original corpus while preserving semantics. This generates a detoxified corpus that can directly replace original data for fine-tuning—reducing toxicity probability from 0.42 to 0.18 on GPT2-XL, and achieving state-of-the-art detoxification effects on LLaMA2-7B, OPT-6.7B, and Falcon-7B.

## Background & Motivation

**Background**: LLMs learn from internet data and inevitably absorb toxic content. Existing detoxification methods mainly operate during post-training (fine-tuning/RLHF) or inference (controlled decoding), but fail to fundamentally prevent models from acquiring toxic knowledge during pre-training.

**Limitations of Prior Work**: (1) Controlled inference methods (e.g., PPLM, DExperts) may degrade generation quality; (2) Post-training methods (e.g., DAPT) require significant additional computation; (3) These methods only "suppress" rather than "eliminate" toxicity—the model still "knows" toxic content but is merely prevented from outputting it.

**Key Challenge**: Detoxifying during inference and post-training treats symptoms rather than the root cause—the real issue lies in the training data itself. However, direct data detoxification faces the challenge of semantic preservation—crude removal of toxic content destroys contextual semantics and knowledge coherence.

**Goal**: Detoxify at the dataset level—rewriting toxic segments in the original corpus into non-toxic but semantically equivalent text to generate a detoxified corpus that can replace the original data.

**Key Insight**: Leverage the text generation capabilities of LLMs themselves to precisely locate and suppress toxic tokens via contrastive decoding while preserving original semantics.

**Core Idea**: Use a small model fine-tuned on toxic data as a "toxicity detector" to precisely locate toxic token dimensions through signal differences during contrastive decoding with the base model, suppressing only these dimensions to maximize semantic preservation.

## Method

### Overall Architecture

The HSPD three-step pipeline: (1) Detoxification prompt guidance—designing prompts to have the model rewrite toxic text into semantically preserved non-toxic versions; (2) SoCD decoding—adaptively suppressing the top-$k$ most biased token dimensions using the logit difference between a toxic small model and the base model during decoding; (3) Multi-temperature sampling + Fusion reranking—generating candidates at multiple temperatures and selecting the best output based on a weighted toxicity score $\times$ semantic similarity.

### Key Designs

1.  **SoCD (Soft Contrastive Decoding)**:
    - **Function**: Precisely locate and suppress toxic tokens rather than using crude global intervention.
    - **Mechanism**: First fine-tune a small model $\theta_{\text{toxic}}$ on toxic data. During decoding, calculate the logit difference $\mathbf{d} = \log(p_{\theta_{\text{toxic}}}) - \log(p_{\theta_{\text{base}}})$ at each step. Keep only positive difference dimensions (tokens preferred by the toxic model), calculate the normalized difference $\alpha = \ln(1+\delta)/(1+\ln(1+\delta))$, and suppress only the top-$k = \alpha \times V$ most biased token dimensions while keeping others unchanged.
    - **Design Motivation**: Aggressive masking strategies in classic contrastive decoding over-suppress informative dimensions, leading to incoherent detoxified text. SoCD operates only on the few most biased dimensions, preserving most information channels.

2.  **Adaptive $k$ Calculation**:
    - **Function**: Dynamically adjust the number of suppressed dimensions based on toxicity intensity at each step.
    - **Mechanism**: $k = \text{clip}(\lceil \alpha V \rceil, k_{\min}, k_{\max})$, where $\alpha$ reflects the distributional difference between the base and toxic models at the current step. Large difference → more dimensions biased toward toxicity → more suppression; Small difference → suppress only a few dimensions. Upper and lower bounds prevent extreme cases.
    - **Design Motivation**: Toxicity intensity varies by token position—strong intervention is needed at "fuck" due to large differences, whereas almost no intervention is needed at "the." A fixed $k$ is either too conservative or too aggressive.

3.  **Multi-temperature Sampling + Fusion Reranking**:
    - **Function**: Select the output with the best semantic preservation and lowest toxicity from multiple candidates.
    - **Mechanism**: Sample candidate rewrites at multiple temperatures, compute non-toxicity scores using a Detoxify model and semantic similarity via sentence embeddings, then select the best candidate through weighted combination.
    - **Design Motivation**: A single sample might perform poorly in the trade-off between semantic preservation and detoxification. Multi-temperature reranking finds a better Pareto optimum in a larger candidate space.

### Loss & Training

The toxic small model is obtained through standard fine-tuning on a toxic dataset $\mathbb{D}$. The HSPD pipeline itself requires no training—it is an inference-time detoxification tool. The detoxified corpus is directly used for further training (simulating a pre-training setting).

## Key Experimental Results

### Main Results

**GPT2-XL Detoxification Performance**

| Method | Toxicity Prob (TP) ↓ | Exp. Max Toxicity (EMT) ↓ | Perplexity ↓ |
|------|---------------|-------------------|---------|
| Original Model | 0.42 | 0.43 | Baseline |
| DExperts | 0.26 | 0.32 | Slight increase |
| DAPT | 0.30 | 0.35 | Significant increase |
| **HSPD** | **0.18** | **0.20** | Near baseline |

**Multi-model Validation**

| Model | Original TP | HSPD TP | Description |
|------|--------|---------|------|
| GPT2-XL | 0.42 | 0.18 | -57% |
| LLaMA2-7B | - | Best | Consistent Lead |
| OPT-6.7B | - | Best | Consistent Lead |
| Falcon-7B | - | Best | Consistent Lead |

### Ablation Study

| Config | TP | Description |
|------|-----|------|
| Full HSPD | 0.18 | Complete pipeline |
| w/o SoCD (Prompt only) | 0.28 | SoCD contributes significantly |
| w/o Multi-temp Reranking | 0.22 | Reranking provides extra security |
| Fixed $k$ (Non-adaptive) | 0.24 | Adaptive $k$ outperforms fixed values |
| Classic CD (Non-SoCD) | 0.30 + Semantic Loss | SoCD's soft intervention is superior |

### Key Findings

- Data-level detoxification fundamentally reduces acquired toxicity rather than just suppressing it at inference.
- SoCD's soft intervention (operating only on top-k biased dimensions) achieves a significantly better balance between detoxification and semantic preservation than classic CD.
- Adaptive $k$ value is crucial—making intervention strength proportional to the toxicity signal at each token position.
- Perplexity of the detoxified corpus is close to the original, indicating that knowledge and linguistic capabilities are effectively preserved.
- Consistently effective across four different model architectures and scales.

## Highlights & Insights

- The approach of "detoxifying from the data source" shifts the strategic level of detoxification—from "patching" at post-training/inference to "root cause treatment" before pre-training.
- SoCD's adaptive top-k suppression is an exquisite engineering design—intervening only on the most toxic dimensions while preserving everything else.
- Multi-temperature + fusion reranking provides a practical solution for the semantic-safety Pareto selection.

## Limitations & Future Work

- Requires training a small toxic model for each dataset (though low cost, it increases pipeline complexity).
- OOD (Out-of-Distribution) toxicity (types not covered by training data) might not be handled effectively.
- The detoxified corpus might change the original intent in certain edge cases.
- Future work could explore adaptive detoxification methods without requiring toxic small models.

## Related Work & Insights

- **vs DExperts**: Inference-time logit ensemble, cannot fundamentally eliminate toxicity; HSPD clears it from the data source.
- **vs DAPT**: Post-training adaptation, requires extra computation and doesn't fully eliminate; HSPD processes before training.
- **vs UniDetox**: Data distillation still requires post-training application; HSPD directly replaces original data.
- **vs ParaGeDi**: Semantic-preserving rewriting but operates during inference; HSPD applies rewriting to the corpus itself.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of data-level detoxification is novel, and SoCD is an effective improvement over contrastive decoding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four models, multiple ablations, dual verification of detoxification quality and semantic preservation.
- Writing Quality: ⭐⭐⭐⭐ Clear method description and logical pipeline design.
- Value: ⭐⭐⭐⭐ Provides a practical solution for addressing LLM toxicity at the data source.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CausalDetox: Causal Head Selection and Intervention for Language Model Detoxification](causaldetox_causal_head_selection_and_intervention_for_language_model_detoxifica.md)
- [\[NeurIPS 2025\] CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming](../../NeurIPS2025/llm_safety/cpret_a_dataset_benchmark_and_model_for_retrieval_in_competitive_programming.md)
- [\[ICCV 2025\] Asynchronous Event Error-Minimizing Noise for Safeguarding Event Dataset](../../ICCV2025/llm_safety/asynchronous_event_error-minimizing_noise_for_safeguarding_event_dataset.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](representation-guided_parameter-efficient_llm_unlearning.md)
- [\[ACL 2026\] SSG: Logit-Balanced Vocabulary Partitioning for LLM Watermarking](ssg_logit-balanced_vocabulary_partitioning_for_llm_watermarking.md)

</div>

<!-- RELATED:END -->
