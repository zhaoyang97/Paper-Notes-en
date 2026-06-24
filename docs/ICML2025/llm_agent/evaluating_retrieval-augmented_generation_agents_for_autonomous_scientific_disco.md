---
title: >-
  [Paper Note] Evaluating Retrieval-Augmented Generation Agents for Autonomous Scientific Discovery in Astrophysics
description: >-
  [ICML 2025][LLM Agent][RAG] This paper constructs CosmoPaperQA (105 expert QA pairs), a RAG evaluation benchmark in the cosmology domain, to systematically evaluate nine RAG agent configurations (covering commercial APIs, hybrid architectures, and academic tools). It finds that the OpenAI RAG solution leads with a 91.4% accuracy rate and calibrates an LLM-as-a-Judge system that can substitute for manual human review.
tags:
  - "ICML 2025"
  - "LLM Agent"
  - "RAG"
  - "Scientific Discovery"
  - "Astrophysics"
  - "Benchmark Evaluation"
  - "LLM-as-a-Judge"
date: 2026-05-08
content_hash: 7fa1cde091e775ed
---

# Evaluating Retrieval-Augmented Generation Agents for Autonomous Scientific Discovery in Astrophysics

**Conference**: ICML 2025  
**arXiv**: [2507.07155](https://arxiv.org/abs/2507.07155)  
**Code**: [https://github.com/CMBAgents/scirag](https://github.com/CMBAgents/scirag)  
**Area**: LLM Agent  
**Keywords**: RAG, Scientific Discovery, Astrophysics, Benchmark Evaluation, LLM-as-a-Judge

## TL;DR
This paper constructs CosmoPaperQA (105 expert QA pairs), a RAG evaluation benchmark in the cosmology domain, to systematically evaluate nine RAG agent configurations (covering commercial APIs, hybrid architectures, and academic tools). It finds that the OpenAI RAG solution leads with a 91.4% accuracy rate and calibrates an LLM-as-a-Judge system that can substitute for manual human review.

## Background & Motivation
**Background**: The rapid development of LLMs has driven automated scientific discovery, demanding AI systems in astronomy/cosmology that can synthesize literature knowledge, computational models, and observational data. However, directly applying LLMs faces major bottlenecks of hallucinations and knowledge cutoff, making RAG the mainstream solution to enhance scientific accuracy.

**Limitations of Prior Work**: While RAG has achieved success in the biological domain (e.g., PaperQA2 reaching superhuman performance on LitQA2), there is a lack of standardized evaluation benchmarks in astronomy. Existing astronomical AI evaluations (such as multiple-choice questions in AstroMLab1 or synthetic questions in Astro-QA) are constrained by formats that limit the evaluation of genuine scientific research workflows.

**Key Challenge**: Constructing human-annotated benchmarks for PhD-level scientific research is extremely costly (financially unfeasible, as noted by Bowman et al.), but the lack of high-quality benchmarks prevents the reliable selection of optimal RAG configurations to support automated scientific discovery systems.

**Goal**: To provide a systematic evaluation paradigm for cosmology RAG agents—including a high-quality benchmark dataset, a unified RAG implementation framework, and a scalable automated evaluation system.

**Key Insight**: Selecting five highly-cited cosmology papers, domain experts manually construct 105 QA pairs reflecting real scientific research scenarios. Meanwhile, a modular framework, SciRag, is developed to uniformly deploy and fairly compare nine RAG configurations.

**Core Idea**: To employ a hybrid strategy of "high-quality, small-scale human evaluation + calibrated LLM judge" to achieve scalable evaluation while maintaining scientific rigor.

## Method

### Overall Architecture
The entire system comprises four components: (1) CosmoPaperQA benchmark dataset construction; (2) SciRag unified RAG deployment framework; (3) human expert evaluation; and (4) calibrated LLM-as-a-Judge automated evaluation. Document preprocessing utilizes Mistral OCR to handle tables, formulas, and other scientific literature-specific content. LangChain is used to split documents into 5000-token chunks (with a 250-token overlap). All RAG systems retrieve information across the complete corpus of the five papers.

### Key Designs

1. **CosmoPaperQA Benchmark Dataset**:

    - **Function**: Provides 105 expert-level cosmology QA pairs as evaluation criteria.
    - **Mechanism**: Extracted from five high-impact cosmology papers (Planck 2018, CAMELS simulations, Hubble constant measurements, ACT DR6), covering three dimensions: observational, theoretical, and computational. Questions span three complexity levels—factual retrieval (extracting concrete parameters), synthesized reasoning (integrating multi-source evidence), and analytical interpretation (deep domain-specific knowledge).
    - **Design Motivation**: To capture the complexity of real scientific research scenarios, unlike synthetic benchmarks, and to support the evaluation of zero-shot learning, open-ended questions, and multi-source knowledge synthesis.

2. **SciRag Unified Deployment Framework**:

    - **Function**: Provides a modular framework to uniformly deploy and compare diverse RAG solutions.
    - **Mechanism**: Covers nine configurations across four categories—
        - Commercial: OpenAI Assistant (text-embedding-3-large + GPT-4.1), OpenAIPDF (directly processing PDFs without OCR), VertexAI (text-embedding-005 + Gemini-2.5-flash).
        - Hybrid Architectures: HybridOAIGem (OpenAI embedding + ChromaDB + Gemini generation), HybridGemGem (Gemini embedding + ChromaDB + Gemini generation).
        - Academic Tools: PaperQA2 (GPT-4.1 full components, evidence retrieval $k=30$), Modified PaperQA2 (astronomy-specific prompt, $k=10$).
        - Baselines: Gemini Assistant (without RAG), Perplexity (web search, sonar-reasoning-pro).
    - **Design Motivation**: Standardizing parameters like temperature (0.01) and top-$k$ (20) to ensure fair comparison, supporting multi-dimensional analysis spanning performance and cost.
    - **Key Difference**: OpenAI’s file search tool integrates automatic query rewriting, parallel search, hybrid keyword-semantic search, and result reranking, which are the core drivers of its superior performance.

3. **Dual-Track Evaluation Framework**:

    - **Function**: Combines human expert evaluation with calibrated AI judges.
    - **Mechanism**: Domain experts (cosmology PhD researchers with 10+ years of experience) perform binary scoring (correct/incorrect) on all 945 generated answers (9 systems $\times$ 105 questions). Concurrently, OpenAI o3-mini and Gemini-2.5-pro are deployed as LLM-as-a-Judge, utilizing Chain-of-Thought (CoT) prompting to enhance evaluation accuracy.
    - **Design Motivation**: Human evaluation guarantees scientific rigor but lacks scalability, whereas the LLM judges can scale to thousands of QA pairs after being calibrated against human baseline results.
    - **Bias Analysis**: To detect evaluation bias (where an LLM might prefer its own generated answers), reasoning models from both OpenAI and Gemini are purposely cross-evaluated against each other.

### Cost and Efficiency Analysis
Budget constraints of scientific research institutions are specifically factored into the system design. VertexAI, costing only \$0.000357 per query, is the most cost-effective solution. Although OpenAI yields the best performance, its cost is 136.7 times that of VertexAI (\$0.048798/query). Hybrid architectures (\$0.003-\$0.004/query) strike a solid balance between performance and cost. Perplexity (\$0.0052/query) incurs noticeable costs but yields extremely poor performance.

## Key Experimental Results

### Main Results

| System Configuration | Category | Human Eval | OpenAI Judge | Gemini Judge | Cost/query |
|----------|------|---------|-------------|-------------|-----------|
| OpenAIPDF | Commercial | **91.4%** | 84.8% | 91.4% | \$0.0488 |
| OpenAI | Commercial | 89.5% | 80.0% | 88.6% | \$0.0488 |
| VertexAI | Commercial | 86.7% | — | — | \$0.0004 |
| HybridOAIGem | Hybrid | 85.7% | — | — | \$0.0032 |
| HybridGemGem | Hybrid | 84.8% | — | — | \$0.0038 |
| PaperQA2 | Academic | 81.9% | — | — | — |
| Modified PaperQA2 | Academic | 73.3% | — | — | — |
| Perplexity | Baseline | 17.1% | 18.1% | 31.4% | \$0.0052 |
| Gemini Baseline | Baseline | 16.2% | 11.4% | 27.6% | \$0.0047 |

### Ablation Study

| Dimension | Config A | Config B | Difference | Description |
|---------|-------|-------|------|------|
| OCR vs Raw PDF | OpenAI (89.5%) | OpenAIPDF (91.4%) | +1.9% | Raw PDF is slightly better, suggesting OCR may introduce noise. |
| OpenAI vs Gemini embedding | HybridOAIGem (85.7%) | HybridGemGem (84.8%) | +0.9% | Embedding differences have limited impact on final performance. |
| Standard vs Domain-customized prompt | PaperQA2 (81.9%) | Modified PaperQA2 (73.3%) | -8.6% | Reducing $k$ and using customized prompts unexpectedly degrades performance. |
| LLM Judge Bias | OpenAI Judge is 2-8% lower | Gemini Judge is 5-15% higher | — | Ranking is consistent ($r > 0.99$); bias direction is predictable. |

### Key Findings
- OpenAI's multi-strategy retrieval (query rewriting + parallel search + hybrid search + reranking) is core to its leading performance; hybrid systems relying on pure semantic retrieval lag by 4-7%.
- The summarization step in PaperQA2 can lead to the loss of specific factual information, hurting performance in cosmology scenarios where precise parameter extraction is required.
- Perplexity without RAG (17.1%) performs almost identically to the Gemini baseline (16.2%), demonstrating that general web search is entirely unhelpful for expert-grade scientific questions.
- The system rankings across the three evaluation methods are perfectly consistent (Pearson $r > 0.99$), indicating that LLM-as-a-Judge can serve as a highly reliable agent for scalable evaluation.
- VertexAI achieves 86.7% accuracy with a 136.7-fold cost advantage, making it the optimal choice under tight budget constraints.

## Highlights & Insights
- **"Small-scale Human + Calibrated AI" Evaluation Strategy**: Elegantly resolves the dilemma between rigor and scalability in scientific evaluation; using 945 human ratings to calibrate a reliable AI judge allows seamless scaling to larger datasets.
- **Empirical Analysis of Retrieval Strategies**: Exposes the performance gap between multi-strategy retrieval and pure semantic retrieval, offering direct guidance for the design of scientific RAG systems.
- **Quantified Cost-Performance Trade-offs**: Instead of merely focusing on accuracy, the study presents fine-grained cost comparisons (showing up to a 136.7-fold difference between the cheapest VertexAI and the most expensive OpenAI), serving as a practical reference for academic research deployments.
- **Honest Limitations Discussion**: The authors candidly acknowledge that explicit citations of source papers within the query prompt provided retrieval shortcuts, which may inflate the reported performance compared to real-world scenarios.

## Limitations & Future Work
- **Limited Corpus Scale**: The benchmark only includes five papers, vastly smaller than the thousands of papers in realistic scientific research scenarios, which significantly underrepresents retrieval noise.
- **Questions Containing Retrieval Landmarks**: Many queries explicitly mention the source paper titles. In practice, researchers typically search based on concepts rather than specific papers, meaning the current setup likely systematically overestimates RAG performance.
- **Single Annotator**: The human evaluation relies on only a single expert, preventing the calculation of inter-annotator agreement.
- **Domain Specificity**: The evaluation is confined to cosmology; the generalizability to other scientific fields such as chemistry, biology, or materials science remains unverified.
- **Unexplored Advanced Retrieval Techniques**: Advanced approaches like hybrid sparse-dense retrieval, context window expansion, query decomposition, or multi-hop reasoning were not explored, which could potentially yield further improvements.

## Related Work & Insights
- **vs. PaperQA2**: PaperQA2 achieved superhuman performance on the biology benchmark LitQA2, but scores only 81.9% in our cosmology benchmark. The root cause is its summarize-then-extract pipeline, which actually hurts performance in scenarios demanding precise factual parameter extraction. This highlights the vital importance of domain adaptation in RAG systems.
- **vs. AstroMLab1**: While the latter evaluates astronomical knowledge using 4,425 AI-generated multiple-choice questions, ours adopts open-ended questions that are closer to realistic scientific workflows, albeit at a smaller scale.
- **vs. pathfinder**: As another astronomical RAG system, pathfinder focuses on query expansion and domain-specific weighting, whereas this study focuses on a standardized comparative evaluation of end-to-end systems.
- **Insights**: The calibrated AI judge can be leveraged to automatically generate domain question sets and scale up to larger document repositories. Future research should evaluate how retrieval performance degrades as the corpus size increases.

## Rating
- Novelty: ⭐⭐⭐ The problem and methodologies themselves are not fundamentally new (RAG evaluation + LLM-as-a-Judge), but this work represents the first systematic application in astrophysics.
- Experimental Thoroughness: ⭐⭐⭐⭐ The workload is solid, evaluated across 9 systems $\times$ 105 questions $\times$ 3 evaluation methods, totaling 945 human evaluations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich with practical insights such as cost analysis, and presenting a highly candid discussion of limitations.
- Value: ⭐⭐⭐⭐ Directly beneficial to the astronomical AI community; the strategy of calibrating an LLM-as-a-Judge system is highly transferrable to other scientific domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Open Source Planning & Control System with Language Agents for Autonomous Scientific Discovery](open_source_planning_control_system_with_language_agents_for_autonomous_scientif.md)
- [\[ICLR 2026\] ScienceBoard: Evaluating Multimodal Autonomous Agents in Realistic Scientific Workflows](../../ICLR2026/llm_agent/scienceboard_evaluating_multimodal_autonomous_agents_in_realistic_scientific_wor.md)
- [\[ACL 2025\] GeAR: Graph-enhanced Agent for Retrieval-augmented Generation](../../ACL2025/llm_agent/gear_graph-enhanced_agent_for_retrieval-augmented_generation.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](../../ICLR2026/llm_agent/newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[ICLR 2026\] SR-Scientist: Scientific Equation Discovery With Agentic AI](../../ICLR2026/llm_agent/sr-scientist_scientific_equation_discovery_with_agentic_ai.md)

</div>

<!-- RELATED:END -->
