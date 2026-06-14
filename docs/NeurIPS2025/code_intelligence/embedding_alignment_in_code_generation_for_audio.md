---
title: >-
  [Paper Note] Embedding Alignment in Code Generation for Audio
description: >-
  [NeurIPS 2025][Code Intelligence][code generation] A dual-MLP + InfoNCE contrastive learning framework is proposed to align code embeddings (distilroberta-base) and audio embeddings (wav2vec2) into a shared space…
tags:
  - "NeurIPS 2025"
  - "Code Intelligence"
  - "code generation"
  - "audio embedding"
  - "contrastive learning"
  - "cross-modal alignment"
  - "live-coding"
date: 2026-05-08
content_hash: 2ac13abfbad20fbe
---

# Embedding Alignment in Code Generation for Audio

**Conference**: NeurIPS 2025
**arXiv**: [2508.05473](https://arxiv.org/abs/2508.05473)  
**Code**: None  
**Area**: Code Intelligence
**Keywords**: code generation, audio embedding, contrastive learning, cross-modal alignment, live-coding

## TL;DR

A dual-MLP + InfoNCE contrastive learning framework is proposed to align code embeddings (distilroberta-base) and audio embeddings (wav2vec2) into a shared space, enabling LLM-based code generation pipelines to infer musical similarity directly from code without compilation or execution. CKA improves from 0.090 to 0.590.

## Background & Motivation

- **Live-coding scenario**: Performers must write music-generating code (e.g., Sonic Pi) in real time under time pressure and in front of an audience. LLM-assisted code generation can reduce syntactic burden, allowing creators to focus on high-level musical ideas.
- **Core limitation**: Existing LLM code generation models evaluate candidate code using text similarity metrics (e.g., BLEU, edit distance), but textual similarity does not imply audio similarity — two textually similar programs may produce drastically different sounds, and vice versa.
- **Key observation**: Computing code-to-code and audio-to-audio embedding distances on 27 Sonic Pi tutorial examples yields a Pearson correlation of only 0.0159 ($p=0.677$) and a Spearman correlation of only 0.0409 ($p=0.445$), indicating virtually no linear or rank-order relationship between the raw embedding spaces.
- **Parameter perturbation experiment**: Minor code modifications (sleep, amplitude, bpm) keep code embedding similarity above 0.990, while audio embedding similarity varies more widely (below 0.975), with no consistent pattern across parameter types — demonstrating that the alignment mapping is non-trivial.

## Method

### Data Construction

1. Using 27 Sonic Pi tutorial code examples, parameters (synths, samples, notes, attack/release, amp, sleep, effects, etc.) are randomized via the Jinja templating engine.
2. 500 distinct Sonic Pi code files are generated, yielding **13,500** code–audio paired samples in total.
3. Code embeddings are produced by **distilroberta-base**; audio embeddings by Meta's **wav2vec2**. Audio is rendered at 120 BPM over 9 bars.

### Symmetric Dual-MLP Architecture

- Two independent MLPs process code embeddings $\texttt{MLP}_c$ and audio embeddings $\texttt{MLP}_a$, respectively.
- Each MLP consists of $L$ linear layers with hidden dimension $d_{\text{hidden}}$, BatchNorm, and GELU activations.
- Pre-trained embeddings are projected into a shared space: $c_i = \texttt{MLP}_c(c_i^0)$, $a_i = \texttt{MLP}_a(a_i^0)$.
- A symmetric MLP design is chosen over attention mechanisms, as the goal is efficient mapping without introducing modality bias.

### InfoNCE Contrastive Learning

- Given $N$ aligned code–audio embedding pairs $\{(c_i, a_i)\}_{i=1}^N$ in a batch, cosine similarity is computed as:

$$\text{sim}(c_i, a_j) = \frac{c_i^\top a_j}{\|c_i\| \cdot \|a_j\|}$$

- The InfoNCE loss pulls positive pairs closer and pushes negative pairs apart:

$$\mathcal{L}_i = -\log \frac{\exp(\text{sim}(c_i, a_i) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(c_i, a_j) / \tau)}$$

- $\tau$ is a temperature hyperparameter controlling the sharpness of the similarity distribution.
- This self-supervised approach requires no explicit annotations and learns alignment solely from code–audio pairing.

### Evaluation Metrics

| Metric | Type | Description |
|--------|------|-------------|
| CKA (Centered Kernel Alignment) | Structural similarity | Invariant to orthogonal transformations and isotropic scaling; captures nonlinear structural similarity |
| CCA (Canonical Correlation Analysis) | Linear correlation | Measures maximum linear correlation between two sets of multivariate variables |
| Jaccard / overlap@k | Neighborhood consistency | Whether nearest neighbors in code space correspond to nearest neighbors in audio space |
| Spearman / Pearson | Rank / linear correlation | Degree of correlation in distance rankings |

## Key Experimental Results

### Hyperparameter Tuning (24 configurations, averaged over 5 runs)

| Setting | CKA | CCA |
|---------|-----|-----|
| Pre-alignment baseline | 0.090 | 0.140 |
| Best config (CKA) | **0.590** | — |
| Best config (CCA) | — | **0.902** |

Both metrics achieve a **6× or greater improvement**.

### Three Code Completion Scenarios

| Scenario | Method | Jaccard | overlap@3 | Spearman | Pearson |
|----------|--------|---------|-----------|----------|---------|
| melody | Raw baseline | 0.20 | 0.33 | 0.21 | 0.18 |
| melody | **Ours** | **0.34** | **0.47** | 0.16 | 0.07 |
| drum | Raw baseline | 0.00 | 0.00 | — | -0.25 |
| drum | **Ours** | **0.16** | **0.27** | -0.05 | -0.12 |
| bass | Raw baseline | 0.20 | 0.33 | 0.24 | 0.21 |
| bass | **Ours** | **0.50** | **0.67** | **0.44** | **0.46** |

### Key Findings

- **Consistent neighborhood metric improvements**: Jaccard and overlap@3 surpass the raw baseline in all three scenarios.
- **Largest gain on drum**: The baseline completely fails (Jaccard=0); alignment raises it to 0.16/0.27.
- **Strongest performance on bass**: Jaccard 0.50, overlap@3 0.67, with significant improvements in rank correlation.
- **No audio compilation required**: All inference is performed directly on code embeddings, avoiding the computational overhead of audio rendering.
- **UMAP visualization**: Before alignment, code (blue) and audio (orange) embeddings are completely separated; after alignment, the two modalities overlap in the embedding space, with semantically related code–audio pairs clustering in nearby regions.

## Highlights & Insights

- **Novel problem formulation**: This work is the first to systematically study cross-modal alignment between code and audio embeddings, filling a gap in the niche but active field of creative coding / live-coding.
- **Lightweight yet effective**: Significant alignment is achieved with only dual MLPs and InfoNCE, without complex cross-modal Transformer architectures.
- **Practical value**: Post-alignment, audio similarity can be inferred directly from code embeddings, endowing code completion tools with "music-aware" capabilities and enabling LLM-assisted live-coding systems to generate more diverse and perceptually meaningful candidate code.
- **Transferable methodology**: The approach of first conducting negative experiments (demonstrating the absence of correlation in raw spaces) to motivate the alignment model provides a clear and replicable argumentative structure.

## Limitations & Future Work

- **Limited data scale**: The dataset is expanded from only 27 Sonic Pi tutorial templates, resulting in insufficient diversity in musical style and code structure.
- **Sonic Pi only**: Generalization to other music programming languages (SuperCollider, TidalCycles, Strudel) is not validated.
- **Unstable rank correlation metrics**: Spearman/Pearson do not improve consistently in the melody and drum scenarios, indicating that alignment remains insufficient for fine-grained ranking.
- **Not integrated into a real code completion system**: Validation is conducted only in offline experiments and has not yet been embedded into an actual LLM code assistant pipeline.
- **Embedding model selection**: distilroberta-base is a general-purpose text model not optimized for code semantics; replacing it with CodeBERT or similar models may yield better results.
- **Audio representation**: wav2vec2 is primarily designed for speech; its music representation may be imprecise. Music-specific embeddings such as CLAP could be considered.

## Related Work & Insights

- Unlike music–text joint embedding works such as MuLan (Huang et al., 2022), this paper focuses on the unique cross-modal pair of **code–audio**.
- The cross-modal alignment approach (lightweight MLP + contrastive learning) is transferable to other scenarios where "code produces non-textual outputs," such as visualization code → images or simulation code → motion trajectories.
- This work offers insights for subsequent research in creative AI / AI-assisted music: how to equip LLMs with awareness of output modalities during code generation.
- The methodological contributions are transferable to related problems.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel perspective on code–audio alignment
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐ Specific value for the audio generation domain

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](../../ACL2026/code_intelligence/coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[NeurIPS 2025\] Text-to-Code Generation for Modular Building Layouts in Building Information Modeling](text-to-code_generation_for_modular_building_layouts_in_building_information_mod.md)
- [\[NeurIPS 2025\] MaintainCoder: Maintainable Code Generation Under Dynamic Requirements](maintaincoder_maintainable_code_generation_under_dynamic_requirements.md)
- [\[NeurIPS 2025\] QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation](qimeng-salv_signal-aware_learning_for_verilog_code_generation.md)
- [\[ACL 2026\] SciCoQA: Quality Assurance for Scientific Paper–Code Alignment](../../ACL2026/code_intelligence/scicoqa_quality_assurance_for_scientific_paper--code_alignment.md)

</div>

<!-- RELATED:END -->
