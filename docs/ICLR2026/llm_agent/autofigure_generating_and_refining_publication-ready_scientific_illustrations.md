---
title: >-
  [Paper Note] AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations
description: >-
  [ICLR 2026][LLM Agent][Scientific illustration generation] This paper proposes AutoFigure — the first agent framework based on a "Reasoned Rendering" paradigm — which automatically generates publication-ready scientific…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "Scientific illustration generation"
  - "multi-agent framework"
  - "long-context understanding"
  - "FigureBench"
  - "VLM evaluation"
date: 2026-05-08
content_hash: b308c728ac4454d2
---

# AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations

**Conference**: ICLR 2026
**arXiv**: [2602.03828](https://arxiv.org/abs/2602.03828)  
**Code**: [https://github.com/ResearAI/AutoFigure](https://github.com/ResearAI/AutoFigure)  
**Area**: Audio & Speech
**Keywords**: Scientific illustration generation, multi-agent framework, long-context understanding, FigureBench, VLM evaluation

## TL;DR
This paper proposes AutoFigure — the first agent framework based on a "Reasoned Rendering" paradigm — which automatically generates publication-ready scientific illustrations from long scientific texts by decoupling structural layout planning and aesthetic rendering into two stages. It is accompanied by FigureBench, the first large-scale benchmark (3,300 pairs) for systematic evaluation, with 66.7% of generated results deemed usable in camera-ready submissions by the original authors.

## Background & Motivation
High-quality scientific illustrations are essential for conveying complex scientific concepts, enabling readers to grasp the core ideas of a paper within minutes. However, creating them manually typically requires days of effort and demands that authors possess both domain expertise and professional design skills.

**Two major limitations of existing work**:

**Benchmark level**: Existing datasets such as Paper2Fig100k, ACL-Fig, and SciCap+ primarily focus on reconstructing figures from captions or short text snippets, rather than distilling core structures from long methodological texts (averaging >10k tokens). A benchmark truly targeting the task of "long-context scientific illustration design" is absent.

**Method level**:
   - Systems such as PosterAgent and PPTAgent excel at "understanding, extracting, and rearranging" existing multimodal content, but lack the ability to **generate** visual content from raw text.
   - Code-based methods such as AutoTikZ emphasize structural and geometric correctness but exhibit poor aesthetic expressiveness.
   - End-to-end text-to-image (T2I) models such as DALL-E and GPT-Image can produce visually appealing images but fail to maintain **structural fidelity** — logical relationships and hierarchical structures present in long scientific texts are frequently lost.

**Key Challenge**: A trade-off between structural accuracy and visual aesthetics. Code-based methods produce well-structured but unattractive outputs; generative models produce attractive but structurally incoherent outputs.

**Key Insight**: Decouple these two requirements — first employ an LLM for structural reasoning and layout planning, then use a generative model for aesthetic rendering.

## Method

### Overall Architecture
AutoFigure adopts a "Reasoned Rendering" paradigm consisting of two stages:
- Input: Long scientific text $T$ (paper / survey / blog / textbook)
- Stage I: Semantic parsing + layout planning → structured symbolic layout $(S_{\text{final}}, A_{\text{final}})$
- Stage II: Aesthetic rendering + text post-processing → publication-ready illustration $I_{\text{final}}$

### Key Designs

1. **Stage I — Concept Extraction and Symbolic Construction**:

    - **Concept Extraction Agent**: Extracts a methodological summary $T_{\text{method}}$ along with sets of entities and relations from the input text $T$.
    - Serializes the structure into a markup language (SVG/HTML) as a symbolic layout $S_0$ with style description $A_0$.
    - $S_0$ encodes a directed graph $G_0 = (V_0, E_0)$ representing the logical relationships among concepts.

2. **Stage I — Critic-Refine Loop** (the core "thinking" process):

    - Simulates a dialogue between an AI "designer" and an AI "critic."
    - At each iteration: critic $\Phi_{\text{critic}}$ evaluates the current best layout and generates feedback $F^{(i)}_{\text{best}}$.
    - Generator $\Phi_{\text{gen}}$ re-interprets the methodological text based on the feedback and produces candidate layouts.
    - Candidates are compared against the current best via score $q$; the best is updated if a superior candidate is found.
    - The loop runs until convergence or a maximum number of iterations (approximately 5 in experiments), yielding the final layout.
    - This is essentially **test-time compute scaling**: more iterations lead to higher layout quality.

3. **Stage II — Style-Guided Aesthetic Rendering**:

    - A conversion function $\Phi_{\text{prompt}}$ transforms $(S_{\text{final}}, A_{\text{final}})$ into an exhaustive text-to-image prompt.
    - Combined with a structural diagram derived from $S_{\text{final}}$, this is fed into a multimodal generative model (e.g., GPT-Image / Nano-Banana) to render a high-fidelity image $I_{\text{polished}}$.

4. **Stage II — "Erase-and-Correct" Text Refinement**:

    - Problem: Text rendered by T2I models is often blurry or contains spelling errors.
    - Solution:
      1. Non-LLM eraser $\Phi_{\text{erase}}$: removes all text pixels → clean background $I_{\text{erased}}$.
      2. OCR engine $\Phi_{\text{ocr}}$: extracts preliminary strings and bounding boxes.
      3. Multimodal verifier $\Phi_{\text{verify}}$: aligns and corrects OCR results against ground-truth labels in $S_{\text{final}}$.
      4. Overlays a vector text layer onto $I_{\text{erased}}$ → final illustration $I_{\text{final}}$.

### FigureBench Benchmark
- **Scale**: 3,300 high-quality scientific text–illustration pairs.
- **Sources**: Papers (3,200) + surveys (40) + blogs (20) + textbooks (40).
- **Test set**: 300 instances (200 randomly sampled from Research-14K, filtered by GPT-5 and annotated by two annotators with Cohen's $\kappa = 0.91$; 100 manually curated from surveys/blogs/textbooks).
- **Development set**: 3,000 instances (constructed from Research-14K using a fine-tuned VLM-based automatic filter).
- **Evaluation protocol**: VLM-as-a-judge (reference scoring + blind pairwise comparison) across three major dimensions and eight sub-metrics covering visual design, communicative effectiveness, and content fidelity.

## Key Experimental Results

### Main Results (Automatic Evaluation, Paper Category)

| Method | Overall | Win-Rate | Aesthetics | Accuracy |
|--------|---------|----------|------------|----------|
| AutoFigure | **7.03** | **53.0%** | **7.28** | 6.96 |
| HTML-Code | 6.35 | 11.0% | 5.90 | **6.99** |
| SVG-Code | 5.49 | 31.0% | 5.00 | 6.15 |
| GPT-Image | 3.47 | 7.0% | 4.24 | 4.77 |
| Diagram Agent | 2.12 | 0.0% | 2.25 | 2.11 |

### Human Expert Evaluation (10 first-authors reviewing generated results for their own papers)

| Metric | Value | Note |
|--------|-------|------|
| Win-Rate (vs. other AI) | 83.3% | Second only to human originals at 96.8% |
| Publication willingness | **66.7%** | Willing to use in camera-ready submission |
| Accuracy score | ~3.5/5 | Within a reasonable range |
| Aesthetics score | ~4/5 | Approaching human level |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Iterations (0→5) | Overall: 6.28→7.14 | Evident test-time scaling effect of the critic-refine loop |
| Reasoning model | Claude-4.1-Opus > GPT-5 > Gemini-2.5-Pro | Stronger reasoning models yield better layouts |
| Intermediate format | SVG (8.98) > HTML (8.85) >> PPT (6.12) | SVG/HTML support complete file generation in a single pass |
| Text refinement module | +0.04 Overall (+0.10 Aesthetics) | Critical for publication quality |
| Open-source model | Qwen3-VL-235B achieves Overall 7.08 | Surpasses several commercial models, approaching GPT-5 |

### Key Findings
- AutoFigure consistently leads across all four document categories: Blog (7.60), Survey (6.99), Textbook (**8.00**), and Paper (7.03).
- Win-Rate reaches 97.5% on the Textbook category, indicating that standardized pedagogical diagrams are the most amenable to automation.
- Win-Rate is relatively lower for the Paper category (53.0%), as paper illustrations typically require customized design with no prior visual template.
- TikZ-based code methods achieve Overall scores below 1.5, revealing a fundamental limitation of the end-to-end code generation paradigm — the cognitive load on LLMs when serializing high-dimensional structures is excessive.
- **Human–machine correlation validation**: Pearson correlation $r = 0.659$ and Spearman $\rho = 0.593$ between VLM and human scores, with ranking error < 1.

## Highlights & Insights
- **"Reasoned Rendering" decoupling paradigm**: Decomposing scientific illustration generation into "structural reasoning" and "aesthetic rendering" is an elegant design choice that enables each module to be optimized independently.
- **Critic-refine loop as test-time scaling**: More iterations yield substantially higher quality, consistent with scaling laws observed in LLM reasoning.
- **"Erase-and-correct" strategy**: Cleverly addresses the poor text rendering of T2I models by combining OCR with vector text overlay to ensure textual accuracy.
- **High practical value**: A publication willingness rate of 66.7% indicates that AutoFigure is approaching the threshold of practical utility.
- **Open-source model potential**: Qwen3-VL-235B surpasses most commercial models, lowering the barrier to deployment.

## Limitations & Future Work
- **Text rendering accuracy remains a bottleneck**: Character-level errors persist in scenarios involving small font sizes, dense layouts, or complex backgrounds (e.g., "ravity" missing "g").
- **Relatively weaker performance on the Paper category**: Paper illustrations exhibit high hierarchical complexity (macro-level pipelines + micro-level sub-steps + fine-grained entities) and require highly customized designs.
- **Tendency toward "concretization"**: When source text descriptions are insufficient, the system may generate visually plausible but content-inaccurate structures.
- The framework is restricted to the CS domain and has not been validated in disciplines with distinct visual conventions, such as biology or chemistry.
- End-to-end latency of approximately 9–17 minutes remains too long for real-time interactive scenarios.

## Related Work & Insights
- **PosterAgent / PPTAgent**: Poster/slide generation systems that excel at rearranging existing content but cannot generate visuals from text from scratch.
- **AutoTikZ / TikZero**: LaTeX TikZ-based code generation methods with good structural accuracy but poor aesthetics.
- **AI Scientist / Zochi**: Autonomous AI scientific discovery systems for which visual expression capability is a key bottleneck.
- **Research-14K / CycleResearcher**: Scientific paper datasets that serve as source data for FigureBench.
- Insight: With the rise of AI Scientist, "enabling AI to express its own discoveries" has become a critical need. AutoFigure bridges the final gap between "understanding science" and "presenting science." Future directions may include dynamic and interactive scientific diagram generation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Pioneering task definition + first large-scale benchmark + novel paradigm)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Automatic evaluation + human expert evaluation + extensive ablations + open-source model validation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Polished figures, complete narrative, highly detailed appendix)
- Value: ⭐⭐⭐⭐⭐ (Directly addresses a real-world pain point, high practical utility, significant implications for AI for Science)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SR-Scientist: Scientific Equation Discovery With Agentic AI](sr-scientist_scientific_equation_discovery_with_agentic_ai.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[ACL 2026\] MOOSE-Copilot: A Web-Based Interactive Assistant for Unified Exploratory and Fine-Grained Scientific Hypothesis Discovery](../../ACL2026/llm_agent/moose-copilot_a_web-based_interactive_assistant_for_unified_exploratory_and_fine.md)
- [\[ICLR 2026\] ChatInject: Abusing Chat Templates for Prompt Injection in LLM Agents](chatinject_abusing_chat_templates_for_prompt_injection_in_llm_agents.md)
- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents](simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_llm_agents.md)

</div>

<!-- RELATED:END -->
