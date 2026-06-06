---
title: >-
  [Paper Note] Geo-Expert: Fine-tuning 8B Models into Expert-Level Geological Reasoning LLMs using LoRA
description: >-
  [ICML 2026][Model Compression][Geological LLM] Geo-Expert fine-tunes Qwen3-8B/32B and Gemma-3-27B using LoRA on 11,518 CoT-enhanced instruction pairs distilled from five classic geology textbooks. On Geo-Eval (387 hard b…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Geological LLM"
  - "LoRA"
  - "Instruction Synthesis"
  - "CoT"
  - "AI for Science"
date: 2026-05-08
content_hash: 37fe5bc9ff71a717
---

# Geo-Expert: Fine-tuning 8B Models into Expert-Level Geological Reasoning LLMs using LoRA

**Conference**: ICML 2026  
**arXiv**: [2605.24844](https://arxiv.org/abs/2605.24844)  
**Code**: Not provided  
**Area**: Domain Adaptation / Scientific LLM / Geological Reasoning  
**Keywords**: Geological LLM, LoRA, Instruction Synthesis, CoT, AI for Science

## TL;DR
Geo-Expert fine-tunes Qwen3-8B/32B and Gemma-3-27B using LoRA on 11,518 CoT-enhanced instruction pairs distilled from five classic geology textbooks. On Geo-Eval (387 hard boundary problems), Qwen3-8B-geo averaged 6.27, surpassing Llama-3.1-70B-Instruct (4.12) and GPT-4o (5.93), while Qwen3-32B-geo reached 6.82, approaching GPT-5.4 (7.15). This demonstrates that high-quality domain alignment is more critical than scaling.

## Background & Motivation

**Background**: Current Earth Science LLMs (K2, GeoGalactica, GeoGPT, UnivEARTH) excel at surface-level tasks but lack deep reasoning for solid Earth topics such as subsurface stratigraphic interpretation, tectonic evolution, and petrogenesis. Geological reasoning requires complex spatio-temporal relationships and vast domain-specific data.

**Limitations of Prior Work**: General LLMs often suffer from severe hallucinations in geology—for instance, misidentifying a "wedge" in geological structures as a mechanical engineering wedge and suggesting the use of carbon fiber for concrete reinforcement. Existing geoscience foundation models are primarily pre-trained on surface-level literature with almost no specialized adaptation for subsurface stratigraphic reasoning.

**Key Challenge**: General LLMs lack domain alignment in geology. Scaling alone cannot resolve this, as geological terminology is highly polysemous, reasoning chains are long, and cross-disciplinary interference is significant. Deep domain anchoring is required rather than simply adding more parameters.

**Goal**: Establish a reproducible pipeline to transform general LLMs into "expert-level geological reasoners," controlling costs via PEFT, and proving that small aligned models can outperform large generalists.

**Key Insight**: Ground truth is extracted from authoritative textbooks (Catuneanu, Fossen, Gao, Rowland). LLMs are used to systematically generate CoT-enhanced instruction data. LoRA fine-tuning is applied to three backbones to observe scaling behavior. An adversarial mining + expert verification approach is used to build the Geo-Eval benchmark for testing hard boundary problems.

**Core Idea**: A tripartite approach consisting of high-quality domain-aligned data, PEFT, and a difficult benchmark. CoT-enhanced data enables the model to learn reasoning chains rather than just keyword matching. LoRA allows fine-tuning up to 32B on an RTX 5090. Geo-Eval specifically targets expert-level reasoning through boundary mining.

## Method

### Overall Architecture

(1) Textbook digitization and cleaning—MinerU converts PDFs to Markdown, and Python modules perform paragraph-based chunking and deduplication. (2) Domain-Structured Instruction Synthesis—utilizing chapter-aware chunking, domain tree question generation, and CoT answer generation to obtain 11,518 instruction pairs. (3) LoRA fine-tuning on three backbones. (4) Geo-Eval evaluation—incorporating boundary mining and GPT-4o scoring.

### Key Designs

1.  **Domain-Structured CoT Instruction Synthesis Pipeline**:
    *   **Function**: Converts static geological textbooks into high-quality instruction-response pairs for fine-tuning.
    *   **Mechanism**: (a) Chapter-Aware Recursive Chunking segments blocks by Markdown headers for semantic integrity. (b) Domain-Structured Question Generation uses an LLM to build a hierarchical domain tree to bind tags to text, then dynamically generates questions based on tags and character density. (c) CoT Answer Construction employs reasoning-oriented models (DeepSeek-R1) to generate answers including intermediate reasoning steps.
    *   **Design Motivation**: General fine-tuning teaches models what to say but not how to reason; CoT-enhanced data forces the learning of reasoning chains. Chapter-aware chunking ensures context completeness, while domain trees prevent redundancy.

2.  **Three-Scale LoRA Fine-Tuning + Scaling Analysis**:
    *   **Function**: Validates domain adaptation scaling behavior across different model sizes.
    *   **Mechanism**: Qwen3-8B uses LoRA with rank=32, $\alpha=32$, lr=2e-5, FP16, on a single RTX 5090. Gemma-3-27B and Qwen3-32B use rank=64, $\alpha=128$, BF16 with gradient checkpointing and accumulation (grad accum=4) on 4×RTX 5090s. LoRA is applied to all linear layers.
    *   **Design Motivation**: Scaling effects cannot be observed at a single size. Using three backbones across 8B/27B/32B enables a comparison between "small model + high-quality data" vs. "large model + general data."

3.  **Geo-Eval: Hard Boundary Benchmark via Adversarial Mining + Expert Verification**:
    *   **Function**: Builds a benchmark that truly tests expert-level reasoning.
    *   **Mechanism**: (a) DeepSeek-R1 extracts 2,591 complex questions/answers from textbooks. (b) Qwen3-8B-Geo and DeepSeek-R1 answer independently. (c) GLM-4.5 acts as LLM-as-judge (10-point scale) to select 387 "hard boundary" problems where score differences are $\le 4$. (d) Manual verification by geology professors across three domains: Concept, Process, and Engineering.
    *   **Design Motivation**: Traditional static MCQs are saturated by modern LLMs. Boundary mining automatically identifies problems just beyond the reach of general models, representing a methodological advancement for discriminative benchmarks.

## Key Experimental Results

### Main Results: Geo-Eval Scores Across Three Dimensions

| Model | Size | Concept | Process | Engineering | Average | $\Delta$ vs Base |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-5.4 | - | 7.35 | 7.10 | 7.00 | 7.15 | - |
| DeepSeek-V3.2 | - | 6.80 | 6.75 | 6.67 | 6.74 | - |
| GPT-4o | - | 6.10 | 5.90 | 5.80 | 5.93 | - |
| Gemma-3-27B-IT | 27B | 5.30 | 5.10 | 5.08 | 5.16 | - |
| Qwen3-32B | 32B | 5.20 | 4.90 | 4.90 | 5.00 | - |
| Qwen3-8B | 8B | 4.80 | 4.68 | 4.41 | 4.63 | - |
| Llama-3.1-70B | 70B | 4.30 | 4.10 | 3.96 | 4.12 | - |
| **Qwen3-32B-geo** | 32B | 6.78 | 6.79 | **6.90** | **6.82** | +1.82 |
| **Gemma-3-27B-geo** | 27B | 6.70 | 6.60 | 6.47 | 6.59 | +1.43 |
| **Qwen3-8B-geo** | **8B** | 6.10 | 6.27 | 6.44 | **6.27** | **+1.64** |

Qwen3-32B-geo ranked second overall (6.82), trailing only GPT-5.4 (7.15). Qwen3-8B-geo (6.27) outperformed GPT-4o and all open-source models under 70B, which is statistically significant ($p = 3.7 \times 10^{-106}$).

### Key Findings

- **8B + Domain Alignment Outperforms 70B Generalist**: Qwen3-8B-geo (6.27) vs. Llama-3.1-70B (4.12), a +51% improvement. This proves scaling laws can falter in vertical domains.
- **Diminishing Returns from 8B to 32B**: The improvement from 8B-geo (6.27) to 32B-geo (6.82) is only +0.55, suggesting marginal utility for extra parameters in geological reasoning.
- **Largest Gains in Engineering**: Qwen3-8B rose from 4.41 to 6.44 (+46%), demonstrating that the model learns reasoning beyond mere terminology.
- **Architecture Stability**: Consistent gains of 1.5+ points across three backbones indicate the robustness of the method.
- **Qualitative Insights**: GPT-4o incorrectly answered "wedge thickening" as concrete reinforcement (0/10), while Qwen3-8B-geo accurately explained geological mechanisms like thrust fault sliding (9/10).

## Highlights & Insights

- **CoT Augmentation as a Key Trick**: The +46% gain in Engineering suggests CoT data is far more valuable than raw text for domain adaptation.
- **Methodological Value of 3-Backbone Scaling Analysis**: This work identifies 8B as a "sweet spot," providing direct guidance for budget-constrained research groups.
- **Hard Boundary Benchmarking Paradigm**: Automatically mining problems that generalists fail to solve is an effective way to test expert reasoning.
- **Consumer GPU Recipe**: Using 4×RTX 5090s to fine-tune 32B models makes this research reproducible for academic labs.
- **Classic Textbooks as Anchors**: Selecting authoritative sources ensures high data quality and domain rigor.
- **Triple-Layer Bias Mitigation**: Using expert re-writes, GPT-4o judging, and multi-model verification.

## Limitations & Future Work

- **Textbook Selection Bias**: The 5 textbooks lean towards structural geology and stratigraphy; coverage of mineralogy, geochemistry, and geophysics is insufficient.
- **Scale of Geo-Eval**: 387 questions is small compared to general benchmarks, potentially weakening statistical power.
- **Text-Only**: The current framework ignores the multimodal nature of geology (cross-sections, well logs, field photos).
- **GPT-4o Judge Bias**: Reference-guided scoring may still inherit verbosity or style biases from the LLM judge.
- **Lack of RAG Baseline**: The study does not compare whether RAG + General LLM could achieve similar results; PEFT's advantages might be slightly exaggerated.
- **Engineering Complexity**: Fine-tuning 32B models on 5090s requires specific optimizations (BF16, gradient checkpointing) and remains prosumer-grade rather than entry-level consumer.

## Related Work & Insights

- **vs K2 / GeoGalactica**: Those models focus on continued pre-training for factual recall; Ours uses PEFT + CoT instruction tuning for multi-step reasoning.
- **vs GeoGPT / UnivEARTH**: These function as geospatial agents for 2D surface tasks; Ours focuses on subsurface deep reasoning.
- **vs MedLLM / FinGPT / LawGPT**: While similar in domain adaptation, most use raw text; Ours differs via CoT-enhanced data synthesis.
- **vs LIMA / Alpaca**: Ours represents vertical instruction tuning combined with boundary benchmarking rather than general instruction tuning.
- **vs ProcessBench / PRM**: Similar in step-level evaluation; Ours innovates via adversarial mining and boundary identification.
- **Insights**: (1) Vertical scientific LLMs should prioritize CoT-enhanced data and boundary benchmarks. (2) "Small + aligned > large + general" should be revisited across all vertical domains. (3) Textbooks are a cost-effective alternative to papers/RAG for ground-truth data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Combination of domain-structured CoT synthesis, boundary mining, and multi-backbone scaling provides solid methodological innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Complete evidence via 3 backbones, 3 evaluation dimensions, 11 baselines, paired t-tests, and qualitative cases.
- **Writing Quality**: ⭐⭐⭐⭐ Clear flowcharts, detailed tables, and persuasive qualitative analysis with a reviewer-conscious approach to bias.
- **Value**: ⭐⭐⭐⭐⭐ Empirical proof that "8B + aligned > 70B" provides direct guidance for vertical AI deployment and democratizing scientific LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs](gemq_global_expert-level_mixed-precision_quantization_for_moe_llms.md)
- [\[ICML 2026\] PRISM: Synergizing Vision Foundation Models via Self-Organized Expert Specialization](prism_synergizing_vision_foundation_models_via_self-organized_expert_specializat.md)
- [\[ICML 2026\] FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models](frism_fine-grained_reasoning_injection_via_subspace-level_model_merging_for_visi.md)
- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](../../ICLR2026/model_compression/steering_moe_llms_via_expert_deactivation.md)
- [\[ICML 2026\] Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression](breaking_the_moe_llm_trilemma_dynamic_expert_clustering_with_structured_compress.md)

</div>

<!-- RELATED:END -->
