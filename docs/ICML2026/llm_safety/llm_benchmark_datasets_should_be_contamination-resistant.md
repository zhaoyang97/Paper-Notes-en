---
title: >-
  [Paper Note] LLM Benchmark Datasets Should Be Contamination-Resistant (Position Paper)
description: >-
  [ICML 2026][LLM Safety][Benchmark Contamination] This position paper advocates that LLM benchmarks should be **contamination-resistant (CRD)**—inferable but not trainable. It proposes leveraging the fundamental mathemati…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Benchmark Contamination"
  - "Contamination-Resistant Datasets"
  - "KV-cache"
  - "Training-Inference Asymmetry"
  - "Cross-model Interoperability"
date: 2026-05-08
content_hash: 6a0b5dda51721514
---

# LLM Benchmark Datasets Should Be Contamination-Resistant (Position Paper)

**Conference**: ICML 2026  
**arXiv**: [2605.19999](https://arxiv.org/abs/2605.19999)  
**Code**: None (position paper)  
**Area**: LLM Safety / Evaluation Benchmarks / Data Contamination  
**Keywords**: Benchmark Contamination, Contamination-Resistant Datasets, KV-cache, Training-Inference Asymmetry, Cross-model Interoperability

## TL;DR
This position paper advocates that LLM benchmarks should be **contamination-resistant (CRD)**—inferable but not trainable. It proposes leveraging the fundamental mathematical asymmetry between Transformer training and inference pipelines (training requires full token sequences, while inference only needs KV-cache + penultimate layer hidden states). By switching the benchmark release format from plaintext to KV-cache + intermediate hidden states, and utilizing cross-model subspace alignment or relative representations to solve interoperability, the authors call for community adoption of this new paradigm.

## Background & Motivation

**Background**: LLM benchmark contamination has become a pervasive phenomenon. During GPT-3 training, over 90% of MMLU samples were detected; Llama 2 still exhibits 16% MMLU contamination, and multilingual benchmarks show contamination as high as 91.8%. Once a benchmark is ingested during pre-training, model scores reflect "memorization" rather than "generalization." Zhang et al. (2024) tested Mistral using a non-public mirror of GSM8K, resulting in a 13% drop in accuracy.

**Limitations of Prior Work**: Existing countermeasures are insufficient:
- **Private Benchmarking + Third-party Evaluation**: Prevents leakage but raises the barrier to innovation and complicates independent verification.
- **Dynamic Benchmarking**: Frequent updates lead to the loss of long-term baseline comparisons.
- **Decontamination (Identifying and deleting leaked samples)**: Identification precision drops sharply under trillion-token corpora.
- **Rephrasing**: Results in the loss of both quality and original difficulty.

Crucially, once benchmarks are made public, they are rapidly replicated across repositories, forums, and secondary datasets. Even gated benchmarks leak indirectly through distillation or continued pre-training.

**Key Challenge**: For a benchmark to be used for evaluation (inference), the model must access the content; however, public content inevitably leaks into subsequent training. This appears to be an unsolvable paradox.

**Goal**: Establish a conceptual framework for "Contamination-Resistant Datasets (CRD)"—the release format must remain usable for inference but remain unlearnable for training.

**Key Insight**: There is a fundamental mathematical asymmetry in Transformer training and inference pipelines. Training requires the full sequence of tokens to calculate gradients (next-token prediction loss requires access to all prefix tokens), whereas inference requires only the KV-cache and the penultimate layer hidden state ($h^{(L-1)}_t$). If the release format only exposes components required for inference while hiding those required for training, it is theoretically possible to achieve inferable but unlearnable benchmarks.

**Core Idea**: Benchmarks should be released as $(KV-cache, h^{(L-1)}_t, Y)$ triplets (KV-cache + penultimate layer hidden state + plaintext ground truth) rather than raw tokens. Models can continue generation for evaluation, but the lack of token sequences prevents loss calculation for training. Interoperability across various LLMs is achieved through cross-model representation alignment.

## Method

### Overall Architecture

**Definition 2.1 (CRD)**: For a model $\mathcal{M}$ and transformation $\phi$, a dataset $\phi(\mathcal{D})$ is contamination-resistant if:
- **Inferable**: $\mathcal{M}(\phi(\mathcal{D}))$ yields valid task performance.
- **Not trainable**: $\nabla_\theta \mathcal{L}(\mathcal{M}_\theta, \phi(\mathcal{D}))$ cannot improve model generalization.

A CRD must satisfy three properties:
1. **Irreversibility**: It is computationally infeasible to reconstruct the plaintext $\mathcal{D}$ from $\phi(\mathcal{D})$.
2. **Equivalence**: $\mathcal{M}(\phi(\mathcal{D})) \approx \mathcal{M}(\mathcal{D})$.
3. **Interoperability**: One can derive $\phi_1(\mathcal{D})$ suitable for another LLM $\mathcal{M}_1$ from $\phi(\mathcal{D})$.

The evaluation workflow consists of: **Curation** (encoding prompts into the latent space using an anchor model) → **Discovery** (the target model establishes an anchor-to-target transformation mapping) → **Evaluation** (the target model performs autoregressive generation on the transformed latent states).

### Key Designs

1. **Leveraging Transformer Training-Inference Asymmetry for CRD**:
    - **Function**: Fundamentally renders benchmark data "inferable but not trainable."
    - **Mechanism**: Training via next-token loss $\mathcal{L} = -\sum_t \log P(x_t | x_{<t})$ requires all tokens $x_1, \dots, x_T$ to compute hidden states at every layer. Inference requires only the KV-cache $\{K_{1:t}^{(l)}, V_{1:t}^{(l)}\}_{l=1}^L$ and the penultimate hidden state $h_t^{(L-1)}$ to generate new tokens. CRD only releases the latter.
    - **Design Motivation**: Previous unlearnable data methods (adversarial perturbations, shortcuts, poisoning) were designed for images and fail for discrete text (removable via paraphrasing). This approach switches to architectural-level separation; even if an attacker possesses the KV-cache, they cannot directly perform fine-tuning.

2. **Anchor Model + Subspace Alignment for Interoperability (Short-term Solution)**:
    - **Function**: Allows a single anchor-encoded benchmark to serve multiple target LLMs.
    - **Mechanism**: The benchmark provider selects a widely deployed anchor model to encode the KV-cache. Target models utilize Cross-LoRA style "LoRA-Align" (rank-truncated SVD + Frobenius-optimal linear mapping) to project from the anchor subspace to the target subspace. This is similar to Procrustes alignment but relaxed to arbitrary linear mappings, allowing for differing dimensions. Mappings are computed using model weights without accessing plaintext, maintaining irreversibility.
    - **Design Motivation**: Avoids releasing separate benchmarks for every LLM; anchor + alignment ensures reusability. Anchor selection can maximize transfer fidelity based on architectural similarities (e.g., GQA, SwiGLU, RMSNorm).

3. **Relative Representations as a Long-term Vision**:
    - **Function**: Completely decouples from specific anchor models, allowing all LLMs to be evaluated in a shared coordinate system.
    - **Mechanism**: Based on the Platonic Representation Hypothesis (model representations converge) and Moschella (2023) relative representations. A small set of shared anchor samples (100–500) is defined, and each latent point is represented as a similarity vector relative to these anchors. This representation remains invariant across latent spaces, enabling zero-shot cross-model stitching.
    - **Design Motivation**: The anchor-model approach may favor specific model families; relative representation is truly symmetric, allows for the inclusion of new models by processing shared anchors, and naturally extends to multimodal settings.

### Auxiliary Designs Against Reverse Engineering
While KV-cache inversion attacks are feasible on MHA, they are significantly less effective on modern architectures like GQA or MLA. Defenders can overlay output noise, entropy perturbations, DP mechanisms, or KV-Cloak. For high-sensitivity scenarios, anchor weights can remain private, with encoding provided via a third-party API.

## Key Experimental Results

### Prevalence of Contamination (Survey)

| Model | Benchmark | Contamination Proportion |
|------|------|--------|
| GPT-3 | Multiple | > 90% tokens |
| Llama-2 | MMLU | 16%+ |
| Average Mainstream LLM | Multilingual | Up to 91.8% |
| Mistral | GSM8K Mirror vs. Public | 13% Accuracy Difference |

### Controllable Storage Overhead

| Benchmark | Original Token Count | Full KV-cache | PyramidKV (12%) Compression | Critical Token Selection |
|------|---------|-------------|-------------------|------------------|
| 100K tokens (Llama-2 7B) | 100K | 50 GB | 6 GB | **350 MB** |
| MMLU | ~5M | 2.5 TB | 300 GB | ~17 GB |

KV-cache compression techniques like PyramidKV demonstrate that retaining 12% is sufficient; removing formatting or generic instruction tokens can further reduce overhead to 0.7%.

### Adaptability Table

| Benchmark Type | Examples | CRD Compatible |
|--------|------|--------|
| Single-turn QA | MMLU, SQuAD, HumanEval | ✅ |
| Classification/Labeling | GLUE, SuperGLUE, ImageNet | ✅ |
| Multimodal | COCO, Flickr30K | ✅ |
| Code Generation | CodeContests, APPS | ✅ |
| Summarization | CNN/DailyMail, XSum | ✅ |
| Multi-turn Dialogue | CoQA, MultiWOZ | ⚠️ Partial (I/O coupling) |
| Dynamic Agent | WebShop, ALFWorld | ❌ (Intertwined environment feedback) |
| Interactive | DynaBench, AdaTest | ❌ (Instance varies with output) |

### Key Findings
- **Compatible with most static benchmarks**: Mainstream benchmarks like QA, classification, code, and summarization are all supported.
- **Storage is not a bottleneck**: With KV-cache compression and selective dropping, storage requirements are in the same order of magnitude as the original benchmarks.
- **Irreversibility strength depends on architecture**: Modern attention mechanisms like GQA significantly degrade the effectiveness of inversion attacks.
- **Interoperability has a technical foundation**: Cross-LoRA and relative representations have already been validated in representation transfer literature.

## Highlights & Insights
- **Architectural-level solution**: Unlike the "perturbation + noise" approach of previous unlearnable data methods, this represents a fundamental paradigm shift by changing the release medium.
- **Underestimated "Free Lunch"**: The mathematical structure of the Transformer provides a natural boundary for "inference without training" that has not been previously leveraged for contamination prevention.
- **Formal characterization of three properties**: Turning the vague concept of "contamination-resistance" into a verifiable set of properties (irreversibility, equivalence, interoperability) facilitates systematic future research.
- **Interdisciplinary adaptation**: Tools borrowed from representation learning (Platonic Representation Hypothesis, Cross-LoRA, relative representations) demonstrate that literature can be directly translated into evaluation infrastructure.

## Limitations & Future Work
- Applicable only to Transformer-like models; SSM-based models (e.g., Mamba, RWKV) are not directly supported.
- KV-cache inversion remains feasible for MHA models; GQA/MLA provides practical security but lacks a mathematical guarantee.
- Equivalence is difficult to strictly verify; benchmark providers need standardized calibration and backtesting protocols.
- Anchor model selection may introduce bias (marginalizing small model families).
- Multi-turn, dynamic, and interactive benchmarks (CoQA, WebShop, DynaBench) require specialized adaptation.
- While storage increments are controllable (350MB/100K tokens), large-scale benchmarks like MMLU still require >17GB, necessitating further optimization for long-term accumulation.

## Related Work & Insights
- **vs. Decontamination**: Identification precision is low at the trillion-token scale; CRD focus on prevention rather than detection.
- **vs. Private Benchmarks + Third-party Evaluation**: Those raise barriers and harm open science; CRD remains public but unlearnable.
- **vs. Dynamic Benchmarks**: Those lose longitudinal comparability; CRDs are static and reproducible.
- **vs. Unlearnable Data (Images)**: Image perturbation fails on text due to paraphrasing; CRD bypasses data-level obfuscation.
- **Insight**: Utilizing the mathematical properties of model architectures as a resource for security and privacy infrastructure can be extended to model attribution, watermarking, and privacy governance.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Releasing CRD through architectural asymmetry is a truly new direction orthogonal to existing anti-contamination routes.
- **Experimental Thoroughness**: ⭐⭐⭐ (Position Paper) — Primarily focuses on argumentation and feasibility analysis rather than SOTA numbers; however, the compatibility table and storage estimates are clear.
- **Writing Quality**: ⭐⭐⭐⭐ The three properties are clearly formalized, the training-inference diagram is intuitive, and the interdisciplinary synthesis is solid.
- **Value**: ⭐⭐⭐⭐⭐ Addresses fundamental issues in the evaluation ecosystem; if adopted, it would qualitatively improve the reliability of LLM benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering](position_uncertainty_quantification_in_llms_is_just_unsupervised_clustering.md)
- [\[ICML 2026\] Position: Stop Chasing the C-index when Evaluating Survival Analysis Models](position_stop_chasing_the_c-index_when_evaluating_survival_analysis_models.md)
- [\[ICML 2026\] Position: Retire the "Positive Backdoor" Label -- Secret Alignment Requires Strict and Systematic Evaluation](position_retire_the_positive_backdoor_label_--_secret_alignment_requires_strict_.md)
- [\[ICML 2026\] MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio](medmosaic_a_challenging_large_scale_benchmark_of_diverse_medical_audio.md)
- [\[ACL 2026\] How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study](../../ACL2026/llm_safety/how_should_we_enhance_the_safety_of_large_reasoning_models_an_empirical_study.md)

</div>

<!-- RELATED:END -->
