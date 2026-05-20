---
title: >-
  [Paper Note] Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX
description: >-
  [NeurIPS 2025][LLM Agent][Chemical Information Extraction] This paper constructs ChemX — a suite of 10 multimodal chemical data extraction benchmark datasets manually annotated and validated by domain experts…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "Chemical Information Extraction"
  - "Multimodal Benchmark"
  - "Agent Evaluation"
  - "Nanomaterials"
  - "Small Molecules"
date: 2026-05-08
content_hash: 059569eaf1ec3cac
---

# Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX

**Conference**: NeurIPS 2025
**arXiv**: [2510.00795](https://arxiv.org/abs/2510.00795)  
**Code**: [ChemX](https://ai-chem.github.io/ChemX)  
**Area**: Agent / Scientific Information Extraction
**Keywords**: Chemical Information Extraction, Multimodal Benchmark, Agent Evaluation, Nanomaterials, Small Molecules

## TL;DR
This paper constructs ChemX — a suite of 10 multimodal chemical data extraction benchmark datasets manually annotated and validated by domain experts, spanning nanomaterials and small molecules. It systematically evaluates state-of-the-art agentic systems including ChatGPT Agent, SLM-Matrix, FutureHouse, and nanoMINER, as well as frontier LLMs such as GPT-5 and GPT-5 Thinking. The proposed single-agent method achieves F1=0.61 on the nanozyme dataset through structured document preprocessing (marker-pdf → Markdown → LLM extraction), surpassing all general-purpose multi-agent systems, while revealing systemic challenges in chemical information extraction such as SMILES parsing failures and terminology ambiguity.

## Background & Motivation

**Background**: Machine learning has made remarkable progress in chemical discovery, but remains heavily reliant on structured data. Existing chemical databases (e.g., PubChem, CSD) are primarily designed for property prediction and structural analysis, making them unsuitable for evaluating automated information extraction systems. Agent-based automated data extraction approaches (e.g., nanoMINER, SLM-Matrix) have emerged in recent years, but each is confined to a specific subdomain.

**Limitations of Prior Work**: (a) No unified benchmark exists for multimodal (text + table + figure) chemical information extraction — existing systems are evaluated on disparate datasets, precluding cross-system comparison; (b) Specialized agent systems (e.g., nanoMINER) perform well on their target datasets but fail to generalize to other chemical domains; (c) General-purpose agent frameworks (ChatGPT Agent, FutureHouse) exhibit high error rates when handling domain-specific chemical terminology and SMILES representations.

**Key Challenge**: The heterogeneity of chemical data — nanomaterials require synthesis conditions, physicochemical properties, and structural parameters, while small molecules demand SMILES, bioactivity metrics, and molecular descriptors — makes precise extraction by general methods difficult, yet specialized methods cannot generalize across domains. This tension cannot even be quantified without a unified benchmark.

**Goal**: To construct the first systematic, multi-domain, multimodal benchmark for chemical information extraction, and on this basis to fairly evaluate the practical extraction capabilities of existing agentic systems and LLMs, identifying the bottlenecks of current technology.

**Key Insight**: The authors depart from the intrinsic heterogeneity of chemical data rather than from any particular methodology, arguing that only a benchmark covering diverse chemical domains and varying difficulty levels can genuinely advance automated extraction technology.

**Core Idea**: Ten expert-validated multimodal datasets are used to uniformly evaluate both general-purpose and specialized agents, exposing systemic bottlenecks in SMILES parsing, terminology normalization, and context-dependent resolution in chemical information extraction.

## Method

### Overall Architecture
ChemX operates at two levels: (1) **Benchmark dataset layer** — 10 manually annotated datasets covering nanomaterials (nanozymes, carbon dots, metal-organic frameworks, etc.) and small molecules (chelate complexes, MIC/IC50 bioactivity, etc.), each with a standardized schema and metadata; (2) **Evaluation experiment layer** — representative datasets of minimal complexity (nanozymes and chelate complexes) are selected to uniformly evaluate general-purpose LLMs, ChatGPT Agent, specialized multi-agent systems, and the authors' proposed single-agent method.

### Key Designs

1. **ChemX Dataset Suite**:

    - **Function**: Provides 10 annotated datasets spanning nanomaterials and small molecules.
    - **Mechanism**: Small-molecule datasets focus on molecular descriptors (SMILES, bioactivity metrics MIC/IC50, compound metadata); nanomaterial datasets cover a broader parameter space (physicochemical properties, synthesis conditions, structural features, application performance). All datasets are manually annotated by domain experts and cross-validated.
    - **Design Motivation**: The core difficulty of chemical information extraction lies in data heterogeneity — extraction targets differ entirely across domains, requiring this diversity to be covered for a genuine capability assessment. Each dataset is annotated with a complexity level to facilitate evaluation at varying difficulty.

2. **Single-Agent Method (proposed in this paper)**:

    - **Function**: Performs structured document preprocessing prior to LLM extraction, addressing the uncontrollable black-box PDF handling of OpenAI.
    - **Mechanism**: The marker-pdf SDK is used to decompose paper PDFs into three element types — text blocks, tables, and figures — while preserving document structural semantics. Text and tables are converted to Markdown; figures are replaced with local paths and described by GPT-4o as descriptive text inserted under `<DESCRIPTION_FROM_IMAGE>` tags. The resulting structured Markdown file is then processed by GPT-4.1/GPT-5/GPT-OSS-20b for extraction, with results aggregated into CSV.
    - **Design Motivation**: Systems such as ChatGPT Agent process PDFs or screenshots directly, but the preprocessing pipeline is opaque and non-reproducible — paper figures captured as screenshots yield unstable OCR quality, causing extraction variance. Explicitly controlling the preprocessing pipeline ensures reproducibility and semantic integrity.

3. **Systematic Evaluation Framework**:

    - **Function**: Qualitatively compares all systems along five dimensions (PDF input support, controllable output format, generalizability, end-to-end extraction capability, multimodal support).
    - **Mechanism**: Beyond numerical Precision/Recall/F1 comparisons, analysis is conducted from a system capability perspective. Systems that cannot complete end-to-end extraction tasks (e.g., OpenChemIE, which only extracts molecular IDs and SMILES; Eunomia, which cannot produce correctly formatted output) are excluded.
    - **Design Motivation**: Different agent systems vary substantially in design objectives (some perform only molecule recognition, others only materials data extraction), requiring a prior definition of "completing the full extraction task" before fair comparison is possible.

### Evaluation Metrics
- Precision, Recall, and F1 are computed per column of each dataset, and column-averaged metrics serve as the overall score.
- A unified prompt template is applied to all methods to ensure comparability.

## Key Experimental Results

### Main Results (Column-averaged extraction metrics)

| Method | Nanozyme Precision | Nanozyme Recall | Nanozyme F1 | Chelate Precision | Chelate Recall | Chelate F1 |
|---|---|---|---|---|---|---|
| GPT-5 | 0.33 | 0.53 | 0.37 | 0.45 | 0.18 | 0.23 |
| GPT-5 Thinking | 0.01 | 0.04 | 0.02 | 0.22 | 0.18 | 0.19 |
| Single-agent (GPT-4.1) | 0.41 | 0.73 | 0.52 | 0.35 | 0.21 | 0.27 |
| Single-agent (GPT-5) | 0.47 | 0.75 | 0.58 | 0.32 | 0.39 | 0.35 |
| **Single-agent (GPT-OSS)** | **0.56** | **0.67** | **0.61** | 0.36 | 0.31 | 0.33 |
| ChatGPT Agent | - | - | - (policy violation) | **0.50** | **0.42** | **0.46** |
| SLM-Matrix | 0.14 | 0.55 | 0.22 | 0.40 | 0.38 | 0.39 |
| FutureHouse | 0.05 | 0.31 | 0.09 | 0.12 | 0.06 | 0.06 |
| nanoMINER** | **0.90** | **0.74** | **0.80** | - | - | - |

> *ChatGPT Agent failed to complete extraction on the nanozyme dataset due to a "policy violation."
> **nanoMINER supports only the nanozyme dataset and cannot generalize.

### System Capability Comparison

| Method | PDF Input | Controllable Output | Generalizability | End-to-End | Multimodal |
|---|---|---|---|---|---|
| Single-agent (Ours) | ✓ | ✓ | ✓ | ✓ | ✓ |
| ChatGPT Agent | ✓ | ✗ | ✗ | ✓ | ✓ |
| SLM-Matrix | ✓ | ✓ | ✓ | ✓ | ✓ |
| nanoMINER | ✓ | ✓ | ✗ | ✓ | ✓ |
| FutureHouse | ✗ | ✓ | ✓ | ✓ | ✓ |

### Key Findings
- **Document preprocessing is the key to improvement**: GPT-5 applied directly to PDFs achieves F1=0.37; adding marker-pdf preprocessing raises this to 0.58 (+21 pp), with Recall jumping from 0.53 to 0.75 — demonstrating that structured input has a decisive effect on extraction quality.
- **GPT-5 Thinking performs worse**: F1 is only 0.02 on nanozymes; the "deep reasoning" mode is counterproductive for structured extraction tasks, as the model tends to over-reason and deviate from extraction instructions.
- **Specialized systems have zero generalizability**: nanoMINER leads substantially with F1=0.80 on nanozymes but is entirely unable to handle small-molecule datasets — confirming the necessity of a unified benchmark.
- **SMILES parsing is a systemic bottleneck**: All general-purpose methods systematically underperform on small-molecule datasets, primarily due to the lack of tool-level capability to convert molecular structure images into SMILES strings.
- **Safety restriction issues in ChatGPT Agent**: The system refuses to perform extraction on the nanozyme dataset due to "policy violations" — chemical terminology (e.g., catalytic reaction conditions) triggers safety filters.

## Highlights & Insights
- **First agent benchmark for chemical information extraction** — ChemX fills a critical gap in automated extraction evaluation for chemistry. The 10 datasets span varying difficulty levels and domains, providing the community with a standardized evaluation framework. Datasets are hosted on HuggingFace for direct use, lowering the barrier to reproduction.
- **"Preprocessing > Model Capability" finding** — The single-agent method surpasses complex multi-agent systems through straightforward marker-pdf structured preprocessing alone, demonstrating that input quality matters more than reasoning depth in information extraction tasks. This insight generalizes to other scientific literature processing tasks.
- **Unintended side effects of LLM safety mechanisms** — ChatGPT Agent's safety filters are triggered by chemical terminology, preventing completion of legitimate scientific data extraction, exposing false-positive issues in current LLM safety alignment for scientific application scenarios.

## Limitations & Future Work
- **Only 2 datasets are evaluated (nanozymes + chelate complexes)**: Although ChemX comprises 10 datasets, the actual benchmark experiments have limited coverage.
- **Closed-access papers constitute the vast majority**: The experiments use only open-access papers (2 per dataset), resulting in a relatively small evaluation corpus.
- **No in-depth analysis of agent orchestration strategies**: Only system-level results are compared, without examining the specific impact of different agent architectures (single-agent vs. multi-agent, tool-calling strategies, context management) on extraction quality.
- **No solution is proposed for the SMILES conversion problem**: The bottleneck is identified but no mitigation is offered; integrating molecular image recognition tools such as OSRA or MolScribe would be a natural direction.
- Directions for improvement include expanding dataset coverage in evaluation, integrating molecular image recognition tools, and analyzing the effects of different agent orchestration strategies.

## Related Work & Insights
- **vs. nanoMINER**: nanoMINER is a multi-agent system specifically designed for nanozymes, achieving F1=0.80 far above general-purpose methods on that domain, but with zero generalizability — the specialized vs. general trade-off is a core tension in chemical AI.
- **vs. FutureHouse**: FutureHouse, as a general-purpose scientific agent platform, performs worst in chemical extraction (F1=0.09), indicating that general scientific reasoning capability does not transfer directly to structured information extraction.
- **vs. SLM-Matrix**: This multi-agent system based on small language models achieves moderate performance on both datasets, but exhibits relatively high Recall (0.55) on nanozymes, suggesting that small models also have potential — the key factor is agent orchestration strategy.
- This work has direct reference value for agent system development in AI for Science — chemical extraction can serve as a touchstone for evaluating agents' tool-use capability and domain adaptation.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic agent benchmark for chemical information extraction, filling an important gap; methodological innovation is limited (preprocessing + LLM extraction).
- Experimental Thoroughness: ⭐⭐⭐⭐ Ten datasets are carefully constructed, six systems are compared, but actual evaluation covers only 2 datasets.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear, experimental comparisons are comprehensive, and qualitative system capability analysis is valuable.
- Value: ⭐⭐⭐⭐⭐ Directly advances automated information extraction in AI for Science; ChemX has the potential to become a community standard benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)
- [\[AAAI 2026\] ProBench: Benchmarking GUI Agents with Accurate Process Information](../../AAAI2026/llm_agent/probench_benchmarking_gui_agents_with_accurate_process_infor.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](../../ICLR2026/llm_agent/newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[NeurIPS 2025\] LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers](lc-opt_benchmarking_reinforcement_learning_and_agentic_ai_for_end-to-end_liquid_.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)

</div>

<!-- RELATED:END -->
