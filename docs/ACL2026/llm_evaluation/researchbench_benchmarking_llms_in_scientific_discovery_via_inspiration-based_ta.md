---
title: >-
  [Paper Note] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition
description: >-
  [ACL 2026][LLM Evaluation][Scientific Discovery] This paper proposes ResearchBench, the first large-scale benchmark for evaluating the scientific discovery capabilities of LLMs. Based on a theoretical decomposition of "i…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Scientific Discovery"
  - "Inspiration Retrieval"
  - "Hypothesis Generation"
  - "LLM Benchmark"
  - "Interdisciplinary"
date: 2026-05-08
content_hash: 3ad7689696c347e8
---

# ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition

**Conference**: ACL 2026 Findings  
**arXiv**: [2503.21248](https://arxiv.org/abs/2503.21248)  
**Code**: None  
**Area**: Scientific Discovery  
**Keywords**: Scientific Discovery, Inspiration Retrieval, Hypothesis Generation, LLM Benchmark, Interdisciplinary

## TL;DR

This paper proposes ResearchBench, the first large-scale benchmark for evaluating the scientific discovery capabilities of LLMs. Based on a theoretical decomposition of "inspiration-driven hypothesis generation," it covers 1,386 papers across 12 disciplines. By decomposing scientific discovery into three sufficient sub-tasks—inspiration retrieval, hypothesis composition, and hypothesis ranking—the study finds that LLMs perform exceptionally well in cross-disciplinary inspiration retrieval.

## Background & Motivation

**Background**: LLMs have demonstrated potential in assisting scientific research, yet a systematic evaluation benchmark for their ability to generate effective new hypotheses is currently lacking.

**Limitations of Prior Work**: (1) Lack of specialized scientific discovery benchmarks—existing benchmarks (e.g., Chatbot Arena, MixEval) evaluate general capabilities rather than discovery; (2) IdeaBench covers only hypothesis generation in biomedicine and does not evaluate a complete set of discovery sub-tasks; (3) DiscoveryBench and ScienceAgentBench focus on specific sub-tasks (such as code generation) without analyzing the fundamental decomposition of scientific discovery.

**Key Challenge**: The non-decomposable nature of the scientific discovery process makes evaluation difficult. A theoretically "sufficient" sub-task decomposition is needed, such that perfectly solving these sub-tasks is equivalent to perfectly solving the overall discovery task.

**Goal**: Construct the first interdisciplinary, large-scale benchmark for scientific discovery capability based on a theoretically sufficient sub-task decomposition.

**Key Insight**: Based on findings from cognitive science—where creativity often stems from the associative combination of two seemingly unrelated pieces of knowledge—hypothesis generation is decomposed into inspiration retrieval → hypothesis composition → hypothesis ranking.

**Core Idea**: Most hypotheses $h = f(b, i_1, ..., i_k)$ can be viewed as a combination of research background $b$ and inspiration knowledge $i$. Accordingly, the process is decomposed into three independently evaluable sub-tasks. Solving these three sub-tasks perfectly equates to solving the discovery task perfectly.

## Method

### Overall Architecture

ResearchBench construction involves: (1) Downloading 1,386 papers published after 2024 from top journals such as Nature and Science; (2) Utilizing an LLM-based agentic framework to automatically extract research problems, background reviews, inspiration knowledge, and main hypotheses; (3) Constructing three levels of negative inspiration samples (citation-proximal, same-discipline, and cross-discipline); (4) Evaluating LLMs on three sub-tasks: inspiration retrieval (selecting correct inspiration from a candidate set), hypothesis composition (combining background and inspiration to generate hypotheses), and hypothesis ranking (ordering candidate hypotheses).

### Key Designs

1.  **Theoretically Sufficient Sub-task Decomposition**:
    - **Function**: Ensures that sub-task evaluation generalizes to overall discovery capability.
    - **Mechanism**: Based on $P(h|b) \approx \prod_{j=1}^{k} P(i_j|b,h_{j-1},I) \cdot P(h_j|b,h_{j-1},i_j)$, discovery is decomposed into inspiration retrieval (finding $i_j$), hypothesis composition (generating $h_j$), and ranking (selecting the best $h$). The sufficiency of these three sub-tasks implies that solving them perfectly is equivalent to solving the discovery task perfectly.
    - **Design Motivation**: Cognitive science supports the notion that "ideas are merely new combinations of old elements"; universality is confirmed through a 12-discipline coverage and expert validation.

2.  **LLM-based Inspiration Extraction Framework**:
    - **Function**: Automatically extracts research components from scientific papers.
    - **Mechanism**: An inspiration decomposition module iteratively extracts potential inspirations (represented as titles and abstracts of cited papers) → a necessity checker verifies whether each inspiration is required for the hypothesis → a sufficiency checker ensures that the extracted inspirations adequately cover the information scope of the hypothesis. Expert validation confirms a 91.9% accuracy rate.
    - **Design Motivation**: An automated framework can be updated with newer papers as LLM pre-training cutoffs advance, thereby preventing data leakage.

3.  **Three-level Negative Inspiration Design**:
    - **Function**: Provides a fine-grained difficulty gradient for inspiration retrieval.
    - **Mechanism**: Level 1—Papers cited by the target paper or with semantically similar titles (nearest neighbors, hardest to distinguish); Level 2—Papers from the same discipline (medium difficulty); Level 3—Papers from completely different disciplines (easiest to exclude).
    - **Design Motivation**: Simple negative samples cannot distinguish the true inspiration retrieval capabilities of LLMs; the three-level design offers more precise diagnostics.

## Key Experimental Results

### Main Results (Inspiration Retrieval - Selecting top 4% candidates)

| Model | Overall Accuracy |
|------|----------|
| GPT-4o | 45.7% |
| GPT-4o-mini | 42.3% |
| Qwen2.5-72B | ~40% |
| Llama-3.1-70B | ~35% |

### Key Findings
- LLMs perform unexpectedly well on inspiration retrieval—the probability that the true inspiration is included when selecting the top 4% of candidates reaches 45.7%.
- Inspiration retrieval is essentially an OOD (Out-of-Distribution) task—inspirations should be knowledge "not previously considered relevant to the research problem but actually useful"; LLMs are capable of finding these non-obvious associations.
- LLMs also demonstrate strong performance in hypothesis composition and ranking tasks.
- Results remain consistent across 12 disciplines, validating the universality of the inspiration-based decomposition framework.
- LLMs are positioned as "research hypothesis mines"—better-performing LLMs represent richer mines, and increased reasoning computation equates to more miners.

## Highlights & Insights
- **Solid Theoretical Foundation**: Uses a sufficient decomposition based on cognitive science rather than an ad hoc evaluation design.
- **Profound Implications of OOD Inspiration Retrieval**: Demonstrates that LLMs possess the ability to discover non-obvious knowledge associations.
- **12-Discipline Coverage**: Spanning from physics to law, validating the broad applicability of the methodology.
- **Automatically Updatable**: The framework can automatically extract data from new papers over time to avoid data leakage.

## Limitations & Future Work
- **Hypothesis Evaluation Relies on Semantic Matching**: It remains difficult to evaluate truly novel hypotheses.
- **Inspiration Extraction Accuracy at 91.9%**: There is still room for improvement in extraction precision.
- **Evaluates Hypothesis Discovery Only**: The benchmark does not evaluate the experimental validation of hypotheses.
- Future Directions: Integrating with experimental agents to complete the full scientific discovery loop; evaluating the novelty and impact of hypotheses.

## Related Work & Insights
- **vs IdeaBench**: IdeaBench covers only biomedicine, lacks inspiration retrieval evaluation, relies on rule-based extraction rather than LLMs, and is restricted to a single domain.
- **vs DiscoveryBench/ScienceAgentBench**: These focus on specific sub-tasks like code writing and do not analyze the fundamental decomposition of scientific discovery.
- **vs MOOSE-Chem**: While it proposes an inspiration-driven discovery framework, it is limited to chemistry and materials science; ResearchBench extends this to 12 disciplines.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First interdisciplinary scientific discovery benchmark based on sufficient theoretical decomposition; unique insight into inspiration retrieval as an OOD task.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of 12 disciplines, multi-model comparison, and expert validation, although some task evaluation details are relatively sparse.
- Writing Quality: ⭐⭐⭐⭐ The theoretical framework is clearly articulated, and examples—such as the inspiration for backpropagation—are intuitive.
- Value: ⭐⭐⭐⭐⭐ Provides the first systematic evaluation framework for AI-aided scientific discovery; the "research hypothesis mine" positioning is highly insightful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PolitNuggets: Benchmarking Agentic Discovery of Long-Tail Political Facts](politnuggets_benchmarking_agentic_discovery_of_long-tail_political_facts.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[ACL 2026\] Reward Modeling for Scientific Writing Evaluation](reward_modeling_for_scientific_writing_evaluation.md)
- [\[ICLR 2026\] AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](../../ICLR2026/llm_evaluation/astabench_benchmarking_ai_agents.md)

</div>

<!-- RELATED:END -->
