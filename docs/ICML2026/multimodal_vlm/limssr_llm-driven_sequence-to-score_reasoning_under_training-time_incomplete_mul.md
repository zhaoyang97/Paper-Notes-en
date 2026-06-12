---
title: >-
  [Paper Note] LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations
description: >-
  [ICML 2026][Multimodal VLM][Incomplete Multimodal Learning] The authors reformulate "training-time incomplete" multimodal action quality assessment as an "LLM-based conditional sequence-to-score reasoning" problem. By ut…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Incomplete Multimodal Learning"
  - "LLM Reasoning"
  - "Action Quality Assessment"
  - "Mask-Aware Fusion"
  - "Token-level Regularization"
date: 2026-05-08
content_hash: ef561047e773a0a0
---

# LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations

**Conference**: ICML 2026  
**arXiv**: [2605.00434](https://arxiv.org/abs/2605.00434)  
**Code**: https://github.com/XuHuangbiao/LIMSSR  
**Area**: Multimodal VLM / Incomplete Multimodal Learning / Action Quality Assessment  
**Keywords**: Incomplete Multimodal Learning, LLM Reasoning, Action Quality Assessment, Mask-Aware Fusion, Token-level Regularization

## TL;DR
The authors reformulate "training-time incomplete" multimodal action quality assessment as an "LLM-based conditional sequence-to-score reasoning" problem. By utilizing prompts and special tokens, the LLM completes missing semantics without full-data supervision. Combined with mask-aware dual-path fusion to suppress hallucinations, the model outperforms SOTA methods that rely on complete training data across three AQA datasets.

## Background & Motivation

**Background**: In real-world scenarios, multimodal data often suffers from missing modalities due to sensor failure, privacy desensitization, or collection costs. Current Incomplete Multimodal Learning (IML) research follows two main lines: (a) reconstruction-based (ActionMAE, IMDer, GAIN, DMVG), which directly reconstructs missing features; (b) distillation/prior-based (CorrKD, MoMKE, MCMoE), which uses complete modalities as teachers for distillation or priors.

**Limitations of Prior Work**: Both categories implicitly assume a "God's perspective"—complete modalities must be available during training as targets or teachers. However, real data may be incomplete from the source (e.g., some subjects never recorded audio). If the training data itself is incomplete, there is no GT for reconstruction and no teacher for distillation, causing the entire IML framework to collapse.

**Key Challenge**: When modalities are missing during the training phase, how can missing semantics be "imagined" from thin air? Traditional reconstruction-distillation routes require "complete-incomplete" pairs, which do not exist here. Simple zero-padding leads the model to learn "missingness" as noise, degrading performance. A mechanism to "infer" missing semantics without paired supervision is required.

**Goal**: (i) Formalize the more realistic setting of "incomplete observations during training"; (ii) Propose a framework that infers missing semantics without relying on complete training data; (iii) Validate this on long-video Action Quality Assessment (AQA), a task highly dependent on multimodality.

**Key Insight**: The authors observe that LLMs are not just sequence models but possess vast world knowledge and reasoning capabilities. Given descriptions of observable modalities and the missing structure, an LLM should be able to infer the semantic representation of missing parts like a "cloze test" without pixel-level reconstruction.

**Core Idea**: Reformulate incomplete multimodal learning as "conditional sequence reasoning"—using prompts to describe the task and missing status, missing tokens as placeholders, and fusion tokens for collection. This allows the LLM to infer latent semantics under invisible missing conditions, followed by mask-aware gating to calibrate reasoning uncertainty.

## Method

### Overall Architecture
For a sample $(\mathbf{X} \odot \boldsymbol{m}, \boldsymbol{m}, y)$ (where $\boldsymbol{m}\in\{0,1\}^M$ is the mask), LIMSSR follows three steps: (1) Context Construction $\Phi_{in}$ combines instruction prompts, visible features $\tilde{\mathbf{X}}^m$, missing token placeholders, and fusion tokens into a unified embedding $\mathbf{Z}_{in}$; (2) LLM Reasoning $\mathbf{H}_{out} = \mathrm{LLM}(\mathbf{Z}_{in})$ performs both missing semantic inference and multimodal fusion; (3) Mask-Aware Dual-Path Aggregation $\Psi_{agg}$ fuses high-level semantic and low-level cross-modal paths using mask-weighting to output the quality score $\hat{y}$. Modality features are extracted by frozen VST/AST/I3D and projected to the LLM space via 2-layer convolutions.

### Key Designs

1.  **Prompt-Guided Context-Aware Modality Imputation (PCMI)**:
    - **Function**: Elevates missing modalities from "zero vectors" to "latent variables to be inferred," allowing the LLM to treat missing positions as fill-in-the-blank tokens.
    - **Mechanism**: Each modality $m$ is wrapped with boundary tokens `<m_start>, <m_end>`. Visible modalities contain $\tilde{\mathbf{X}}^m$, while missing modalities contain $T$ repeated learnable `<missing_m>` embeddings. A task prompt explicitly describes the status: "Given the available {avail} features... The {miss} modality is missing. Based on the available modalities, please infer and reconstruct the useful latent representations for the missing {miss} modalities at the designated positions." The LLM outputs $\mathbf{H}_{miss}^m = \mathrm{LLM}(\mathbf{Z}_{in})|_{\text{positions of }\mathbf{E}_{miss}^m}$ as the inferred representation.
    - **Design Motivation**: Traditional zero-padding causes signals to be "buried" in attention. PCMI encodes the missing structure directly into the sequence, making the LLM's next-token reasoning naturally suited—"predicting the next token" and "inferring missing latents" are mathematically similar.

2.  **LLM-Driven Multidimensional Representation Fusion (LMRF)**:
    - **Function**: Distills cross-modal information into $K$ fusion slots without disrupting the LLM output space.
    - **Mechanism**: Appends $K$ special tokens `<emb_dim_1>, ..., <emb_dim_K>` as "information slots" to the prompt, instructing the LLM to "integrate and enhance all multimodal features for action quality assessment." The output $\mathbf{H}_{fusion} = \{\boldsymbol{h}_1, \dots, \boldsymbol{h}_K\}$ at these positions represents different evaluation dimensions (e.g., difficulty, execution). A learnable role weight $\boldsymbol{w}_{role}$ computes $\boldsymbol{z}_{main} = \sum_k \mathrm{Softmax}(\boldsymbol{w}_{role})_k \cdot \boldsymbol{h}_k$.
    - **Design Motivation**: Mean-pooling LLM outputs disrupts sequence generation. Borrowing from BERT's `[CLS]` but generalized to multiple dimensions, this allows the LLM to learn to "pack different aspects into different slots."

3.  **Mask-Aware Dual-Path Aggregation (MDA)**:
    - **Function**: Processes high-level semantics via the LLM and low-level features via cross-modal attention, dynamically calibrating reliability based on the mask to avoid hallucinations.
    - **Mechanism**: Path 1 (Reasoning) computes gating $\boldsymbol{g} = \sigma(\mathrm{MLP}_{gate}([\boldsymbol{z}_{main}, \boldsymbol{m}]))$ and residual $\boldsymbol{\delta} = \mathrm{MLP}_{res}([\boldsymbol{z}_{main}, \boldsymbol{m}])$ to get $\tilde{\boldsymbol{z}}_{main} = \boldsymbol{z}_{main} + \boldsymbol{g}\odot \boldsymbol{\delta}$. Path 2 (Pattern Recovery) performs temporal pooling on LLM hidden states to get $\boldsymbol{h}_v, \boldsymbol{h}_a, \boldsymbol{h}_f$ and applies self-attention. Weights $\alpha_{m_j} = \boldsymbol{m}_j \cdot 1 + (1-\boldsymbol{m}_j)\cdot \gamma_{m_j}$ adjust based on availability, where $\gamma_m$ is a learnable confidence.
    - **Design Motivation**: Pure LLM reasoning causes hallucinations under severe loss; pure statistical aggregation lacks high-level semantics. Mask-aware mixing provides the model with "meta-cognitive" ability to trust itself.

### Loss & Training
In addition to the regression loss, the authors introduce: (1) Consistency Learning to align the two paths; (2) Token-Level Metric Regularization to ensure fusion tokens learn different dimensions by maximizing distances in a similarity matrix; (3) LoRA fine-tuning for the LLM backbone.

## Key Experimental Results

### Main Results (FS1000, 7-class, Spearman ↑ / MSE ↓, T-Miss denotes incomplete training)

| Method | T-Miss | {v,f} | {v,a} | {v} | {a} | Average | {v,f,a} |
|------|--------|-------|-------|------|------|---------|---------|
| ActionMAE | ✗ | 0.775/24.66 | 0.766/64.13 | 0.761/50.64 | 0.458/41.66 | 0.651/38.18 | 0.809/17.96 |
| GCNet | ✗ | 0.730/25.56 | 0.740/23.86 | 0.696/26.67 | 0.442/39.40 | 0.610/28.62 | 0.764/21.82 |
| MoMKE | ✗ | 0.798/18.86 | 0.805/23.88 | 0.785/37.96 | 0.499/27.53 | 0.668/26.08 | 0.819/16.85 |
| MCMoE | ✗ | 0.845/12.66 | 0.882/11.85 | 0.845/13.64 | 0.615/16.72 | 0.782/15.37 | 0.881/11.53 |
| **LIMSSR** | **✓** | **0.854/12.51** | **0.891/10.54** | **0.853/12.50** | **0.687/15.51** | **0.789/14.08** | **0.891/10.44** |

| Δ vs SOTA | {v,f} | {v,a} | {v} | {a} | Average | {v,f,a} |
|-----------|-------|-------|------|------|---------|---------|
| ΔSpearman | ↑1.1% | ↑1.0% | ↑0.9% | ↑11.7% | ↑0.9% | ↑1.1% |
| ΔMSE | ↓1.2% | ↓11.1% | ↓8.4% | ↓7.2% | ↓8.4% | ↓9.5% |

Note: LIMSSR is the only model trained under **T-Miss ✓**, yet it outperforms all methods trained with complete data (**T-Miss ✗**) across nearly all missing combinations.

### Ablation Study

| Configuration | Average Spearman | Description |
|------|------------------|------|
| Full LIMSSR | 0.789 | Full framework |
| w/o PCMI | Significant Drop | Missing semantics cannot be inferred by LLM |
| w/o LMRF | Drop | Multi-dimensional information collapses |
| w/o MDA Path 1 | Drop | Lacks high-level semantic calibration |
| w/o MDA Path 2 | Drop | Hallucinations under severe missingness |
| w/o Consistency Loss | Drop | Paths lack mutual verification |
| w/o Token Regularization | Drop | Fusion tokens become redundant |

### Key Findings
- **Winning with incomplete training data**: LIMSSR outperforms SOTA even with severe audio missingness (Spearman +11.7%, MSE -7.2%), showing that LLM world knowledge offers a qualitative advantage.
- **Path 1 + Path 2 Complementarity**: Either path alone results in performance drops; MDA's mask-adaptive fusion is critical for anti-hallucination.
- **Fusion Token Count $K$**: $K=3$ matches the AQA structure (difficulty/execution/artistry).
- **Audio is most difficult**: Performance is lowest in {a}-only settings across all methods, but LIMSSR shows the largest relative gain, proving LLM inference is most effective for low-information modalities.

## Highlights & Insights
- **Task reformulation is the primary contribution**: Shifting IML from "reconstruction/distillation" to "conditional sequence reasoning" transforms a supervision-limited problem into a next-token problem that LLMs excel at.
- **Elegant token design**: Using placeholders and boundary tokens treats the LLM as a programmable "semantic calculator" without architectural changes.
- **Mask-aware meta-cognition**: Encoding confidence into the network via MDA handles the engineering of inference uncertainty.
- **Superiority over complete data methods**: Suggests that LLM priors might be more valuable than paired data, challenging the traditional paradigm of IML.

## Limitations & Future Work
- Primarily validated on AQA; generalizability to emotion recognition or medical diagnosis requires further study.
- LLM inference introduces significant computational overhead, impacting real-time applications.
- Lacks systematic experiments on LLM scale (7B vs 70B) and performance on low-resource or niche action types.
- Hallucination is not quantitatively measured directly, only mitigated via MDA.

## Related Work & Insights
- **vs ActionMAE/IMDer (reconstruction)**: These depend on complete training pairs; LIMSSR breaks this constraint.
- **vs MoMKE/MCMoE (distillation/prior)**: These require complete modality teachers; LIMSSR replaces them with LLM priors (general knowledge vs. paired supervision).
- **vs MissRAG/TAMML (LLM-based IML)**: MissRAG requires prototype pools; TAMML loses detail through textification. LIMSSR reasons directly in the embedding space.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ New "training-time incomplete" setting and reformulation of IML as sequence reasoning.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive benchmarks and ablation; however, task diversity is limited to AQA.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative and intuitive comparisons.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for the IML community and a convincing non-linguistic use case for LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reasoning-Driven Multimodal LLM for Domain Generalization](../../ICLR2026/multimodal_vlm/reasoning-driven_multimodal_llm_for_domain_generalization.md)
- [\[ACL 2026\] STELLA: A Multimodal LLM for Protein Functional Annotation via Unified Sequence-Structure Encoding](../../ACL2026/multimodal_vlm/stella_a_multimodal_llm_for_protein_functional_annotation_via_unified_sequence-s.md)
- [\[AAAI 2026\] MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment](../../AAAI2026/multimodal_vlm/mcmoe_completing_missing_modalities_with_mixture_of_experts_for_incomplete_multi.md)
- [\[ICML 2026\] Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training](learn_to_think_improving_multimodal_reasoning_through_vision-aware_self-improvem.md)
- [\[ICML 2026\] Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling](mitigating_perceptual_judgment_bias_in_multimodal_llm-as-a-judge_via_perceptual_.md)

</div>

<!-- RELATED:END -->
