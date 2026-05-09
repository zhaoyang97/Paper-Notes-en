---
title: >-
  [Paper Note] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition
description: >-
  [ACL 2026][LLM Evaluation][Scientific discovery] This paper proposes ResearchBench, the first large-scale benchmark for evaluating LLMs in scientific discovery. Grounded in a theoretically motivated decomposition of inspiration-driven hypothesis generation, it covers 1,386 papers across 12 disciplines and decomposes scientific discovery into three sufficient subtasks: inspiration retrieval, hypothesis composition, and hypothesis ranking. Results show that LLMs perform surprisingly well on cross-disciplinary inspiration retrieval.
tags:
  - ACL 2026
  - LLM Evaluation
  - Scientific discovery
  - inspiration retrieval
  - hypothesis generation
  - LLM benchmark
  - interdisciplinary
date: 2026-05-08
content_hash: 97e1178f4fd33c3b
---

# ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition

**Conference**: ACL 2026
**arXiv**: [2503.21248](https://arxiv.org/abs/2503.21248)
**Code**: None
**Area**: Scientific Discovery
**Keywords**: Scientific discovery, inspiration retrieval, hypothesis generation, LLM benchmark, interdisciplinary

## TL;DR

This paper proposes ResearchBench, the first large-scale benchmark for evaluating LLMs in scientific discovery. Grounded in a theoretically motivated decomposition of inspiration-driven hypothesis generation, it covers 1,386 papers across 12 disciplines and decomposes scientific discovery into three sufficient subtasks: inspiration retrieval, hypothesis composition, and hypothesis ranking. Results show that LLMs perform surprisingly well on cross-disciplinary inspiration retrieval.

## Background & Motivation

**Background**: LLMs have demonstrated potential in assisting scientific research, yet no systematic benchmark exists for evaluating their ability to discover valid novel hypotheses.

**Limitations of Prior Work**: (1) No dedicated benchmark for scientific discovery exists—existing benchmarks (Chatbot Arena, MixEval) assess general capabilities rather than discovery ability. (2) IdeaBench covers only biomedical hypothesis generation and does not evaluate the full set of discovery subtasks. (3) DiscoveryBench and ScienceAgentBench focus on specific subtasks (e.g., code writing) without analyzing the fundamental decomposition of scientific discovery.

**Key Challenge**: The perceived indivisibility of the scientific discovery process makes evaluation difficult—what is needed is a theoretically "sufficient" decomposition of subtasks such that perfectly solving them is equivalent to perfectly solving the overall discovery task.

**Goal**: Construct the first interdisciplinary, large-scale benchmark for scientific discovery capability, grounded in a theoretically sufficient subtask decomposition.

**Key Insight**: Drawing on cognitive science—creative ideas typically arise from associative combinations of two seemingly unrelated pieces of knowledge—the paper decomposes hypothesis generation into inspiration retrieval → hypothesis composition → hypothesis ranking.

**Core Idea**: Most hypotheses $h = f(b, i_1, ..., i_k)$ can be viewed as combinations of research background $b$ and inspirational knowledge $i$. This motivates a decomposition into three independently evaluable subtasks; perfectly solving these three subtasks is equivalent to perfectly solving the discovery task.

## Method

### Overall Architecture

The ResearchBench construction pipeline proceeds as follows: (1) Download 1,386 post-2024 papers from top venues such as *Nature* and *Science*. (2) Use an LLM-based agentic framework to automatically extract research questions, background reviews, inspirational knowledge, and main hypotheses. (3) Construct three-level negative inspiration samples (citation-adjacent, same-discipline, cross-discipline). (4) Evaluate LLMs on three subtasks: inspiration retrieval (selecting the correct inspiration from a candidate set), hypothesis composition (generating a hypothesis given background and inspiration), and hypothesis ranking (ranking candidate hypotheses).

### Key Designs

1. **Theoretically Sufficient Subtask Decomposition**

    - **Function**: Ensures that subtask evaluation generalizes to overall discovery capability.
    - **Mechanism**: Based on $P(h|b) \approx \prod_{j=1}^{k} P(i_j|b,h_{j-1},I) \cdot P(h_j|b,h_{j-1},i_j)$, discovery is decomposed into inspiration retrieval (finding $i_j$), hypothesis composition (generating $h_j$), and ranking (selecting the best $h$). The sufficiency of these subtasks implies that perfectly solving them yields a perfect solution to the discovery task.
    - **Design Motivation**: Supported by cognitive science—"an idea is nothing more nor less than a new combination of old elements"—and validated across 12 disciplines with expert verification to confirm generality.

2. **LLM-Based Inspiration Extraction Framework**

    - **Function**: Automatically extracts research components from papers.
    - **Mechanism**: An inspiration decomposition module iteratively extracts candidate inspirations (represented as titles and abstracts of cited papers); a necessity checker verifies that each inspiration is necessary for the hypothesis; a sufficiency checker ensures the extracted inspirations collectively cover the informational scope of the hypothesis. Expert validation yields 91.9% accuracy.
    - **Design Motivation**: The automated framework can be updated with newer papers as LLM pretraining cutoffs advance, thereby preventing data leakage.

3. **Three-Level Negative Inspiration Design**

    - **Function**: Provides a fine-grained difficulty gradient for inspiration retrieval.
    - **Mechanism**: Level 1—papers cited by the target paper or with semantically similar titles (hardest to distinguish); Level 2—papers from the same discipline (moderate difficulty); Level 3—papers from entirely different disciplines (easiest to exclude).
    - **Design Motivation**: Simple negative samples cannot differentiate LLMs' true inspiration retrieval capability; the three-level design enables more fine-grained diagnostic evaluation.

## Key Experimental Results

### Main Results (Inspiration Retrieval — selecting top 4% candidates)

| Model | Overall Accuracy |
|-------|-----------------|
| GPT-4o | 45.7% |
| GPT-4o-mini | 42.3% |
| Qwen2.5-72B | ~40% |
| Llama-3.1-70B | ~35% |

### Key Findings
- LLMs perform surprisingly well on inspiration retrieval—when selecting the top 4% of candidates, the probability that the true inspiration is included reaches 45.7%.
- Inspiration retrieval is inherently an out-of-distribution (OOD) task—inspirations are knowledge "not considered obviously related to the research problem but actually useful"—yet LLMs can identify such non-obvious associations.
- LLMs also perform well on hypothesis composition and ranking tasks.
- Consistent results across 12 disciplines validate the universality of the inspiration-based decomposition framework.
- The paper positions LLMs as "hypothesis mines"—higher-capability LLMs are richer mines, and more inference compute corresponds to more miners.

## Highlights & Insights
- **Solid theoretical foundation**: The sufficient decomposition is grounded in cognitive science rather than being an ad hoc evaluation design.
- **OOD inspiration retrieval finding is significant**: It demonstrates that LLMs possess the ability to discover non-obvious knowledge associations.
- **12-discipline coverage**: Spanning from physics to law, validating the broad applicability of the method.
- **Automatically updatable**: The framework can extract new papers over time, avoiding data leakage.

## Limitations & Future Work
- **Hypothesis evaluation relies on semantic matching**: Evaluating genuinely novel hypotheses remains difficult.
- **Inspiration extraction accuracy at 91.9%**: Room for further improvement remains.
- **Only hypothesis discovery is evaluated**: Experimental validation of hypotheses is not assessed.
- Future directions include integrating with experiment agents to complete the full scientific discovery loop, and evaluating hypothesis novelty and impact.

## Related Work & Insights
- **vs. IdeaBench**: Covers only biomedicine, lacks inspiration retrieval evaluation, uses rule-based extraction (not LLM-based), and is single-domain.
- **vs. DiscoveryBench/ScienceAgentBench**: Focuses on specific subtasks such as code writing without analyzing the fundamental decomposition of discovery.
- **vs. MOOSE-Chem**: Proposes an inspiration-driven discovery framework but is limited to chemistry and materials science; ResearchBench extends this to 12 disciplines.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First interdisciplinary scientific discovery benchmark grounded in a theoretically sufficient decomposition; the insight of treating inspiration retrieval as an OOD task is unique.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 12-discipline coverage, multi-model comparison, and expert validation, though evaluation details for some tasks are limited.
- **Writing Quality**: ⭐⭐⭐⭐ The theoretical framework is clearly articulated, and the backpropagation-style inspiration examples are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Provides the first systematic evaluation framework for AI-assisted scientific discovery; the "hypothesis mine" framing is thought-provoking.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)
- [\[ICLR 2026\] AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](../../ICLR2026/llm_evaluation/astabench_benchmarking_ai_agents.md)
- [\[ACL 2026\] Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff](do_llms_overthink_basic_math_reasoning_benchmarking_the_accuracy-efficiency_trad.md)
- [\[ICLR 2026\] Benchmarking Overton Pluralism in LLMs](../../ICLR2026/llm_evaluation/benchmarking_overton_pluralism_in_llms.md)
- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)

<!-- RELATED:END -->
