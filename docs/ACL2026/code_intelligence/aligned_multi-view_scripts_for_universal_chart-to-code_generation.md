---
title: >-
  [Paper Note] Aligned Multi-View Scripts for Universal Chart-to-Code Generation
description: >-
  [ACL 2026][Code Intelligence][Chart-to-Code] By treating "semantically equivalent scripts of the same chart in Python, R, and LaTeX" as a new supervisory signal…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Chart-to-Code"
  - "Multilingual Alignment"
  - "LLaVA"
  - "Low-Rank Subspace Adapter"
  - "MoE Projector"
date: 2026-05-08
content_hash: 1e60c1810b4573d0
---

# Aligned Multi-View Scripts for Universal Chart-to-Code Generation

**Conference**: ACL 2026  
**arXiv**: [2604.24559](https://arxiv.org/abs/2604.24559)  
**Code**: [GitHub](https://github.com/Zhihan72/CharLuMA)  
**Area**: Code Intelligence / Multimodal  
**Keywords**: Chart-to-Code, Multilingual Alignment, LLaVA, Low-Rank Subspace Adapter, MoE Projector

## TL;DR
By treating "semantically equivalent scripts of the same chart in Python, R, and LaTeX" as a new supervisory signal, the authors constructed the 176K quadruple dataset Chart2NCode. They proposed CharLuMA, a lightweight adapter adding "language-conditioned low-rank subspace routing" to the LLaVA projector, enabling a single model to achieve high execution rates and visual fidelity across all three plotting languages.

## Background & Motivation

**Background**: Chart-to-code (reconstructing executable plotting scripts from chart images) restores static charts to editable and reproducible states. Existing works almost exclusively target Python/matplotlib; recent efforts such as ChartMimic, Plot2Code, and ChartCoder are limited to a single language.

**Limitations of Prior Work**: (1) In real-world academia, R (ggplot2) and LaTeX (TikZ) are publication standards for many disciplines, making single Python output insufficient; (2) Deeper still, equivalent scripts in different languages for the same chart naturally serve as a **multi-view supervisory signal**, which single-language datasets fail to exploit; (3) Simply feeding multilingual data into a single model requires either independent experts for each language (doubling parameters and precluding knowledge sharing) or leads to inter-language interference and specialization imbalance.

**Key Challenge**: The model must "share a single semantic understanding of the chart" while "passing through specialized syntactic channels according to the target language"—two tasks that conflict when handled by a single LLaVA MLP projector.

**Goal**: (a) Provide the first chart-code quadruple dataset aligned across Python, R, and LaTeX; (b) Design a parameter-efficient multilingual adaptation mechanism that specializes syntactic output while maintaining shared visual understanding.

**Key Insight**: Different language scripts are viewed as "complementary views" of the same chart semantics, aligned following the principles of multi-view representation learning. Architecturally, drawing from Mixture-of-Subspaces LoRA, a low-rank subspace pool and language-conditioned routing replace "independent language experts."

**Core Idea**: Utilize a "metadata-template" pipeline to synthesize cross-language aligned scripts and use a "low-rank subspace adapter + language routing" to lightly insert language specialization capabilities into the LLaVA projector.

## Method

### Overall Architecture

(1) Chart2NCode Data Construction: Python, LaTeX, and R single-language scripts are first collected. A unified "figure / axis / object" three-level metadata is extracted via execution and parsing. These are then matched against a manual template pool (202 templates × 20+ chart sub-types) at the object level to instantiate scripts in three languages. Samples with template misses or execution failures are handled by GPT-4o LLM-assisted debugging, while rendering failures are discarded. The final 176K quadruples include 14.7% corrected by LLM.

(2) CharLuMA Model: A SigLIP visual encoder + DeepSeek-Coder LLM backend + LLaVA-style two-layer MLP projector, with a "low-rank subspace adapter" connected in parallel to the MLP. This adapter consists of a low-rank projection matrix $\mathbf{A}$, a subspace pool $\{b_i\}_{i=1}^N$, and a language-specific router $\mathbf{W}^l$. It dynamically selects and combines the top-$r$ subspaces based on the input image and target language. The output is added to the base MLP output to produce the final visual tokens. Training follows two stages: alignment pre-training for the MLP only, followed by instruction tuning where the router and subspace pool are warmed up before joint LLM fine-tuning.

### Key Designs

1. **Metadata-Template Alignment Pipeline (Chart2NCode Data Construction)**: 
    - **Function**: Mass "translates" single-language plotting scripts into visually equivalent scripts across Python, R, and LaTeX.
    - **Mechanism**: Extracts three levels of metadata (figure-level global attributes, axis-level coordinate systems, object-level geometry + style) via native language APIs. It matches object patterns (e.g., rectangles with constant height and variable width → horizontal bar chart) to a manually curated template pool. The templates include cross-language attribute mapping dictionaries (Python "upper right" ↔ R "right" ↔ LaTeX "north east"; Python bold ↔ LaTeX bfseries), ensuring semantic consistency across syntaxes. Failed samples are translated/fixed by GPT-4o and re-verified via rendering. A human evaluation of 1000 samples across four dimensions showed a 95%+ pass rate ($\alpha=0.81$).
    - **Design Motivation**: Pure LLM translation is costly and prone to semantic drift, while pure rule-based templates have limited coverage. Combining both ensures both scale and fidelity.

2. **Language-conditioned Subspace Adapter**: 
    - **Function**: Injects language specialization on top of the shared visual MLP, avoiding parameter redundancy of independent language experts and the capacity waste of Mixture-of-MLP.
    - **Mechanism**: The visual input $\mathbf{Z}_v$ passes through a shared MLP to obtain $\mathbf{H}_{\text{base}} = \mathbf{W}\mathbf{Z}_v$. In parallel, it is compressed to a rank-$r$ representation via matrix $\mathbf{A}$, and the language router $y^l = \mathrm{top}_r(\mathrm{softmax}(\mathbf{W}^l \overline{\mathbf{Z}}_v))$ selects $r=16$ out of $N=32$ subspaces to form $\mathbf{B}$. The language-adaptive visual token is $\mathbf{H}_v = \mathbf{W}\mathbf{Z}_v + \mathbf{A}\mathbf{B}\mathbf{Z}_v$.
    - **Design Motivation**: Low-rank plus subspace pooling allows "shared core + language specialization" to coexist in a parameter-efficient manner. Activation analysis shows that in a 1.3B model, approximately 5/27 subspaces are shared across three languages, while others are language-exclusive, validating the "compact shared core + language-specific capacity" design.

3. **Two-stage Progressive Training Strategy (Alignment Pretrain + Instruction Tuning)**:
    - **Function**: Stabilizes modal alignment first, then language routing, and finally trains the LLM to utilize language-adaptive tokens.
    - **Mechanism**: Stage 1 trains only the MLP $\mathbf{W}$ on 900K Chart-JSON pairs, freezing the vision encoder and LLM. Stage 2 introduces the subspace adapter, warming up only the router $\mathbf{W}^l$ and subspace pool $\{b_i\}$ for 274 steps (MLP/Vision/LLM frozen; $\mathbf{A}$ frozen after random initialization), followed by joint LLM training (MLP and $\mathbf{A}$ remain frozen). Each batch is forced to contain all three languages.
    - **Design Motivation**: Keeping $\mathbf{A}$ frozen throughout forces the adaptive capacity toward "language differences" rather than re-learning shared visual features. Warming up the router prevents routing from being disrupted by LLM gradients before convergence.

### Loss & Training
Standard next-token cross-entropy is used without auxiliary losses. Two-stage learning rates: pre-training 2e-4, warm-up 2e-4, joint fine-tuning 2e-5. CharLuMA-1.3B training took 82 GPU hours (8×L40S); 6.7B took approximately 321 GPU hours.

## Key Experimental Results

### Main Results
Averaged across three languages on the Chart2NCode test set (1000 samples), primary metrics include ER (Executability Rate) / DS (DreamSim visual similarity) / MJ (MLLM-as-Judge):

| Model | Python ER | Python DS | R ER | R DS | LaTeX ER | LaTeX DS |
|------|----------|-----------|------|------|----------|----------|
| GPT-4o | 98.5 | 85.0 | 94.5 | 78.8 | 88.4 | 72.4 |
| Claude-Sonnet-4 | 98.3 | 86.8 | 93.9 | 82.0 | 92.7 | 76.0 |
| Qwen3-VL-8B | 91.1 | 83.7 | 73.6 | 72.7 | 77.3 | 66.8 |
| ChartCoder-7B (Python specialist) | 96.2 | 48.1 | - | - | 17.9 | 39.1 |
| InternVL3.5-8B | 82.5 | 79.6 | 67.0 | 67.6 | 81.1 | 57.1 |
| **CharLuMA-1.3B** | 94.4 | 86.5 | 94.5 | 78.9 | 84.5 | 71.3 |
| **CharLuMA-6.7B** | 98.0 | 88.7 | 96.5 | 81.8 | 89.0 | 72.5 |

The 6.7B model achieves 96.5 ER / 81.8 DS in R, approaching Claude-Sonnet-4. Python specialists like ChartCoder-7B collapse on R/LaTeX (0% R executability), highlighting the value of multilingual alignment.

### Ablation Study

**Architecture Comparison** (Chart2NCode 3-language average):

| Projector Architecture | 1.3B ER | 1.3B DS | 1.3B MJ | 6.7B ER | 6.7B DS | 6.7B MJ |
|----------|---------|---------|---------|---------|---------|---------|
| Linear MLP | 88.1 | 76.9 | 69.5 | 91.0 | 78.2 | 76.3 |
| Mixture-of-MLP | 87.9 | 75.1 | 68.2 | 91.9 | 77.4 | 76.8 |
| **Subspace Adapter (Ours)** | **91.1** | **78.9** | **72.3** | **94.5** | **81.0** | **81.1** |

**Subspace-Router Configuration** (1.3B):

| Total Subspaces | Active Num | Num Routers | ER | DS | MJ |
|----------|-------|---------|-----|------|------|
| 16 | 8 | 3 | 88.9 | 77.6 | 70.5 |
| 32 | 16 | 1 (Shared) | 86.1 | 75.1 | 67.0 |
| 32 | 32 | 0 | 85.8 | 73.2 | 66.3 |
| **32** | **16** | **3 (Lang-Specific)** | **91.1** | **78.9** | **72.3** |
| w/o warm-up | - | - | 87.1 | 75.6 | 67.9 |
| Unfrozen $\mathbf{A}$ | - | - | 90.2 | 78.0 | 70.1 |

**Language Diversity**: Training on three languages > two languages > one language, even when the per-chart training volume is reduced. Baselines without aligned source data skew toward Python, proving the necessity of alignment supervision.

### Key Findings
- Aligned tri-language training outperforms single-language settings with the same total volume across all languages—multi-view supervision cross-enhances performance.
- The 32-16 configuration + 3 language-specific routers is the optimal spot; replacing language routers with a shared router immediately drops DS by ~4 points.
- Subspace activation analysis reveals that in the 1.3B model, only 19% of activated subspaces are shared across all three languages (5/27); 6.7B shows a similar 18%, indicating capacity is automatically allocated to "language-specific" tasks during scaling.
- LaTeX failure modes are unique: 55.5% are due to syntactic constraints (missing braces); Python/R failures primarily stem from dimension mismatches and undefined variables (data-logic errors).

## Highlights & Insights
- Treating "multilingual plotting scripts" as "multiple views of the same chart" is a novel and natural perspective, extending cross-language code alignment concepts from NLP to the chart-to-code domain.
- The subspace adapter + routing design is more parameter-efficient than MoE-MLP—shared MLP learns commonalities while the subspace pool captures differences. Forcing $\mathbf{A}$ to be frozen prevents adaptation capacity from redundantly learning visual features.
- The metadata-template pipeline provides a reproducible path for high-quality multilingual data; 176K is the largest and most language-diverse dataset of its kind.
- The end-to-end 6.7B model directly challenges Claude-Sonnet-4, significantly closing the gap between open-source and proprietary models in multilingual plotting.

## Limitations & Future Work
- The model scale stops at 6.7B; the potential of larger LLM backends (e.g., 30B+) remains unexplored.
- The SigLIP input resolution of 384×384 is a bottleneck; information-dense charts (sub-charts, complex heatmaps) easily lose detail. The authors suggest replacing it with a high-resolution visual adapter.
- Although 202 templates are used, they are still limited; novel chart types (e.g., interactive charts) may not be covered.
- Only three languages are supported; extending to D3.js / Vega-Lite / Mermaid would require new templates and metadata schemas.

## Related Work & Insights
- **vs ChartCoder-7B (Zhao et al., 2025)**: ChartCoder is a Python specialist with 0% executability on R; Ours uses alignment data + routing to achieve a true multilingual generalist with comparable Python performance but superior generalization.
- **vs ChartMoE (Xu et al., 2025)**: ChartMoE uses a sparsely-gated MoE projector, leading to significant parameter bloat; the low-rank subspace adapter in this work is more compact and effective.
- **vs DaTikZ / AutomaTikZ (Belouadi et al., 2024)**: Focused solely on TikZ; this work treats TikZ as one of three languages and leverages alignment to allow other languages to boost its performance.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining multi-view alignment with language-conditioned subspace routing is a rare synthesis in this field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three benchmarks × three languages × 14 baselines + detailed ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative on motivation and methodology with concise formulas.
- Value: ⭐⭐⭐⭐ Contributions in dataset, model, and paradigm with practical open-source utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation](deepguard_secure_code_generation_via_multi-layer_semantic_aggregation.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ACL 2026\] MARS²: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi_agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](../../ICLR2026/code_intelligence/breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[CVPR 2026\] MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](../../CVPR2026/code_intelligence/mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)

</div>

<!-- RELATED:END -->
