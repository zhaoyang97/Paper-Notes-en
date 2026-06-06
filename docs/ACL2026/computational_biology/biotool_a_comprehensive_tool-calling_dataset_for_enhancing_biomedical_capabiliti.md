---
title: >-
  [Paper Note] BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models
description: >-
  [ACL 2026][Computational Biology][Biomedical tool calling] BioTool constructs an instruction fine-tuning dataset covering 34 common tools from three major biomedical databases (NCBI/Ensembl/UniProt), consisting of 7…
tags:
  - "ACL 2026"
  - "Computational Biology"
  - "Biomedical tool calling"
  - "NCBI/Ensembl/UniProt"
  - "instruction fine-tuning"
  - "small models surpassing commercial LLMs"
date: 2026-05-08
content_hash: 29d1dd7f4b5d3647
---

# BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2605.05758](https://arxiv.org/abs/2605.05758)  
**Code**: https://github.com/gxx27/BioTool  
**Area**: Computational Biology  
**Keywords**: Biomedical tool calling, NCBI/Ensembl/UniProt, instruction fine-tuning, small models surpassing commercial LLMs

## TL;DR
BioTool constructs an instruction fine-tuning dataset covering 34 common tools from three major biomedical databases (NCBI/Ensembl/UniProt), consisting of 7,040 human-verified "query–API call" pairs. After fine-tuning a 4B-parameter open-source LLM with this data, the tool-calling quality exceeds commercial models such as GPT-5.1, Gemini-3 Pro, and Claude-4.5-Sonnet by more than 15%.

## Background & Motivation

**Background**: Mature tool-calling datasets and fine-tuning pipelines like Toolformer, Gorilla, ToolBench, and APIGen exist for general domains. In the biomedical field, the mainstream approach remains using agents based on in-context learning (ICL), such as GeneGPT, ChemCrow, and Biomni, which insert tool documentation into prompts for the model to learn on the fly.

**Limitations of Prior Work**: The ICL approach faces three major bottlenecks: (1) tool capacity is limited by context length (e.g., GeneGPT only covers a small subset of NCBI APIs); (2) biomedical API parameter schemas are extremely complex and cannot be fully covered by simple prompt descriptions; (3) mapping natural language queries to professional schemas, identifiers, and parameter specifications is significantly more difficult than for general tools, leading to severe hallucinations.

**Key Challenge**: Regardless of the size of general tool-calling datasets, the biomedical tools included are just a "drop in the ocean." They fail to enable LLMs to provide executable calls in scenarios requiring strict schemas, such as BLAST, Variation API, or UniProt sequence queries. For LLMs to truly assist biomedical researchers, a high-quality "database-native" tool-calling corpus is essential.

**Goal**: (1) Systematically select high-frequency tools from three authoritative biomedical databases; (2) automatically synthesize "query–API call" pairs in bulk while ensuring semantic validity; (3) fine-tune small-to-medium open-source LLMs with this data to reach or exceed the tool-calling capabilities of top-tier closed-source models.

**Key Insight**: The authors employ reverse data construction—first enumerating diverse API parameter combinations from tool documentation and executing them, then using "real executable API responses" as seeds for a reasoning model to back-propagate the "user query that can be answered by this response." Finally, an LLM-judge and human experts verify the results. This paradigm of "having the answer before creating the question" naturally avoids annotation noise from query-API mismatches.

**Core Idea**: Replace the traditional "human-written query → human-labeled API" paradigm with "response-grounded reverse query synthesis + multi-round LLM/human filtering." This scales both the volume and quality of biomedical tool-calling corpora, allowing a 4B model to outperform closed-source models with $200\times$ more parameters on professional schemas.

## Method

### Overall Architecture
The BioTool construction pipeline consists of four stages: (1) **Tool Selection**—34 high-frequency API endpoints are manually selected from NCBI, Ensembl, and UniProt, covering five subfields: variation, genomics, proteomics, evolution, and general biology. (2) **API Call Synthesis**—official documentation is fed to an LLM to enumerate diverse parameter combinations, which are then executed. Samples with empty or uninformative returns are discarded, leaving 3,829 unique API calls. (3) **Reverse Query Generation**—a reasoning model (e.g., OpenAI o-series) takes the API call and real response as input to generate a natural language query that matches the response. (4) **Quality Filtering**—an LLM-judge evaluates if the API response truly answers the query, followed by a manual review by biomedical experts, resulting in 7,040 quadruplets (query, tool info, API arguments, observation).

During downstream usage, the tool-calling LLM only generates API arguments, which the system executes to obtain an observation. A base LLM (e.g., GPT-5.1) then integrates the observation into the final answer, achieving a decoupled "tool caller + answer generator" architecture.

### Key Designs

1. **Response-grounded Reverse Query Synthesis**:
    - **Function**: Solves the fundamental problem of query-API misalignment. Traditional methods often result in API responses that cannot actually answer the query.
    - **Mechanism**: Diverse API parameter combinations are generated and executed first to obtain (API call, response) pairs that return useful information. Using this response as an anchor, a reasoning model generates the most reasonable user question, ensuring the query is supported by the API response.
    - **Design Motivation**: Determining the validity of biomedical queries requires domain knowledge. Generating queries from scratch often leads to hallucinations; using real responses embeds "answerability" into the data, making quality control easier.

2. **Three-layer Filtering with High Human Verification Ratio**:
    - **Function**: Ensures the 7,040 data points meet standards for biological correctness, API schema compliance, and query-response alignment.
    - **Mechanism**: Layer 1 is execution validation (API must return a non-empty response). Layer 2 is an LLM-judge (evaluating if the response supports the query). Layer 3 is manual review by biomedical experts (checking biological relevance and correctness, such as matching gene IDs to species or valid variation coordinates).
    - **Design Motivation**: Purely LLM-synthesized data in professional domains is highly noisy. Without manual review, models might learn incorrect schemas. The three-layer funnel ensures high quality across the training set.

3. **Small Model Fine-tuning Overcoming Large Model ICL**:
    - **Function**: Demonstrates that "high-quality specialized data + small open-source models" can outperform "general big data + closed-source LLM + ICL" in tool calling.
    - **Mechanism**: The BioTool training set is used for SFT on small models like Qwen-3-4B/8B, internalizing the 34 tools' parameter specifications into weights rather than reading them from prompts. During inference, the model directly generates API arguments in JSON format.
    - **Design Motivation**: It was observed that the bottleneck for ICL models in specialized schema tasks is "specialization" rather than "intelligence." Once domain knowledge is hardcoded into weights, a 4B model can surpass a model with $200\times$ more parameters. This is a classic trade-off between specialization and generalization.

### Loss & Training
Standard SFT cross-entropy is used, where the target is the JSON string of (tool name, API arguments). Observations are not part of the training loss and are filled by the system during inference.

## Key Experimental Results

### Main Results: Tool-Calling Quality Comparison

| Model | Parameters | Training | API-calling Quality | Remarks |
|------|--------|---------|------------------|------|
| GPT-5.1 (Closed) | undisclosed | ICL | baseline | Top-tier general LLM |
| Gemini-3 Pro | undisclosed | ICL | Near GPT-5.1 | |
| Claude-4.5-Sonnet | undisclosed | ICL | Strongest baseline | |
| Qwen-3-4B + BioTool SFT | 4B | SFT | **15.0%** higher than Claude-4.5 | Best performance |
| Qwen-3-8B + BioTool SFT | 8B | SFT | Further improvement | |

### Downstream QA Quality (Biologist Scoring)

| Configuration | Normalized Quality Gain vs. Vanilla GPT-5.1 | Description |
|------|------|------|
| GPT-5.1 (No Tools) | 0% (Baseline) | Direct answer, prone to hallucination |
| GPT-5.1 + Oracle BioTool API call | +**88.4%** | Upper bound: Ground truth API calls |
| GPT-5.1 + BioTool-fine-tuned tool caller | +**69%** | Practical: Using fine-tuned 4B caller |
| GPT-5.1 + ICL tool calling | Much lower than both | Traditional approach |

Test set: 1,048 test queries with head-to-head preference evaluation by experts.

### Key Findings
- **Specialized Data Beats General Scale**: A 4B model beating a massive closed-source model suggests that ICL has reached diminishing returns for specialized schema tasks; weight-level internalization is the necessary next step.
- **Oracle (88.4%) vs. Practical (69%) Gap**: A ~20 percentage point gap indicates room for improvement in the fine-tuned caller, though it already captures ~78% of the potential tool-calling gains.
- **Value of Coverage**: 34 tools across five subfields allow the system to handle interdisciplinary queries (e.g., complex questions requiring both NCBI gene and UniProt protein information).

## Highlights & Insights
- **Reverse Construction Paradigm**: Starting with "correct API responses" to back-propagate queries eliminates the largest noise source in tool-use datasets: misalignment between queries and ground-truth API calls. This is highly transferable to other vertical domains (finance, GIS, e-commerce).
- **Victory for Small Model Specialization**: Confirms that fine-tuning a 4B model to its limit on specific schemas is more effective than chasing 200B general models—a significant signal for academia and smaller teams.
- **Oracle Upper Bound Analysis**: Using Oracle API calls to establish an 88.4% ceiling and comparing it to the 69% practical value clarifies the contribution of "dataset intrinsic quality" versus "caller implementation."

## Limitations & Future Work
- **Narrow Tool Scope**: 34 tools are few compared to the hundreds available in biomedicine; future expansion to chemistry, proteomics imaging, and clinical trial databases is needed.
- **High Verification Cost**: 7,040 entries require expert review, making it hard to scale infinitely. A hybrid "expert review + active learning" paradigm could be explored.
- **Closed-source Base LLM**: The final answer quality used GPT-5.1, hindering full open-source replication; a fully open stack is ideal.
- **No Multi-tool Chaining**: Currently, each sample corresponds to a single API call. Complex biological questions often require chains (e.g., BLAST → annotation → cross-reference), requiring future expansion to multi-step tool use.

## Related Work & Insights
- **vs Toolformer / Gorilla**: These are general tool datasets with minimal biomedical coverage and lower schema complexity compared to NCBI/Ensembl. Ours focuses specifically on biology with stricter schemas.
- **vs GeneGPT**: Also targets NCBI, but GeneGPT uses ICL, limiting the number of tools. BioTool internalizes schemas for 34 tools into weights via SFT.
- **vs ChemCrow / SciAgent**: These scientific agent routes emphasize orchestration. BioTool is complementary, providing high-quality training data to strengthen tool-caller modules within such agents.
- **vs Biomni**: A general biomedical agent with a smaller toolset; BioTool data can directly enhance its caller.

## Rating
- Novelty: ⭐⭐⭐⭐ Solid reverse synthesis paradigm, though individual components are not entirely original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Reports both API-quality benchmarks and human expert H2H evaluations; Oracle analysis is a major plus.
- Writing Quality: ⭐⭐⭐⭐ Clear three-part narrative: dataset → experiment → human evaluation.
- Value: ⭐⭐⭐⭐⭐ 7,040 high-quality data points plus open-source code/dataset provide usable infrastructure for the biomedical LLM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/computational_biology/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](../../NeurIPS2025/computational_biology/fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)
- [\[ICLR 2026\] AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](../../ICLR2026/computational_biology/afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)
- [\[NeurIPS 2025\] Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models](../../NeurIPS2025/computational_biology/mol-llama_towards_general_understanding_of_molecules_in_large_molecular_language.md)
- [\[ACL 2026\] ProtoCycle: Reflective Tool-Augmented Planning for Text-Guided Protein Design](protocycle_reflective_tool-augmented_planning_for_text-guided_protein_design.md)

</div>

<!-- RELATED:END -->
