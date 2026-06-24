---
title: >-
  [Paper Note] Long-Document QA with Chain-of-Structured-Thought and Fine-Tuned SLMs
description: >-
  [ICLR 2026][Information Retrieval & RAG][Long-Document QA] LiteCoST utilizes strong LLMs to rewrite "long-document QA" into auditable "extract-then-answer" trajectories. These structure-priority behaviors are distilled into 3B/7B small models via a dual-signal SFT→GRPO approach, allowing small models to reach parity with GPT-4o on financial, legal, and scientific long-document QA while reducing latency by 2–4 times.
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "Long-Document QA"
  - "Chain-of-Structured-Thought (CoST)"
  - "Small Language Models"
  - "SFT"
  - "GRPO"
  - "Structured Output"
date: 2026-05-08
content_hash: a592b8cf02829af3
---

# Long-Document QA with Chain-of-Structured-Thought and Fine-Tuned SLMs

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=faECRsdRav](https://openreview.net/forum?id=faECRsdRav)  
**Code**: [https://github.com/HKUSTDial/LiteCoST](https://github.com/HKUSTDial/LiteCoST)  
**Area**: Information Retrieval / Long-Document QA / Structured Information Extraction  
**Keywords**: Long-Document QA, Chain-of-Structured-Thought (CoST), Small Language Models, SFT, GRPO, Structured Output  

## TL;DR
LiteCoST utilizes strong LLMs to rewrite "long-document QA" into auditable "extract-then-answer" trajectories. These structure-priority behaviors are distilled into 3B/7B small models via a dual-signal SFT→GRPO approach, allowing small models to reach parity with GPT-4o on financial, legal, and scientific long-document QA while reducing latency by 2–4 times.

## Background & Motivation
**Background**: LLMs are increasingly employed for document data analysis, but direct reasoning on long, noisy, and multi-source documents remains fragile and opaque. In high-stakes scenarios like finance and law, this prone-to-error nature leads to missed evidence, hallucinations, and format drift. A verified effective route is "solidifying scattered evidence into structured data (tables/graphs/blocks) before deriving answers," as structures make evidence visible, verifiable, and reusable.

**Limitations of Prior Work**: (1) Evidence is often scattered across long contexts, making direct prompting susceptible to omissions or hallucinations; (2) Heterogeneous formats and units of numerical values require normalization; (3) Long-context reasoning requires maintaining consistency across the entire structure. While using GPT-4 or DeepSeek-R1 to output structured products is accurate, **repeatedly calling large models involves high token costs, high latency, low throughput**, and privacy risks from sending sensitive data to hosted APIs.

**Key Challenge**: Replacing large models with locally deployable small language models (SLMs) can save costs and ensure privacy, but **off-the-shelf SLMs lack the specific skills required for CoST**—such as schema-aware extraction from long contexts, unit/entity normalization, record alignment, and step-by-step consistent serialization. Consequently, a naive substitution of LLMs with SLMs is ineffective.

**Goal**: To achieve a balance between "accuracy and verifiability" (G1) and "low latency in small models" (G2), ensuring that SLM-generated structures $S_{SLM}$ are as useful for answering as LLM-generated $S_{LLM}$, while maintaining significantly lower latency: $\text{LLM}(Q, S_{SLM}) \approx \text{LLM}(Q, S_{LLM})$ and $\text{Latency}(S_{SLM}) \ll \text{Latency}(S_{LLM})$.

**Core Idea**: **Structure-First Distillation**—first utilizing a strong LLM as a one-time "structure-first trajectory generator" to produce auditable CoST trajectories and machine-verifiable serialized structured outputs (SSOs), then injecting this schema-aware structured reasoning into compact models through a lightweight two-stage process (SFT→GRPO with dual-layer rewards).

## Method

### Overall Architecture
LiteCoST is a two-pillar framework. **Stage A (CoST)**: A strong LLM uses a CoST template as input to dynamically select structures, extract evidence, normalize, align, serialize, and self-correct for each question, producing an auditable CoST trajectory (`<reasoning>`) and a query-specific SSO (`<answer>`, such as a table, graph, or block) as supervision signals. **Stage B (SLM Fine-Tuning)**: SFT is first used to teach the small model schema/format discipline and step-by-step logic, followed by GRPO with dual-layer rewards (outcome + process) to reinforce answer quality and reasoning consistency. Finally, 3B/7B models internalize structure-priority behavior, achieving fast and auditable inference.

```mermaid
flowchart LR
    Q[Question Q + Long Doc D] --> T[CoST Template]
    T --> A1[A1 Structural Analysis<br/>Select Table/Graph/Block + Dynamic Schema]
    A1 --> A2[A2 Trajectory Generation<br/>Extraction/Alignment/Serialization]
    A2 --> A3[A3 Quality Verification<br/>LLM-as-Judge]
    A3 --> A4[A4 Iterative Refinement<br/>Iterative Structuralizer]
    A4 --> DATA["(CoST Trajectory c*, SSO S*)"]
    DATA --> SFT[Stage B-1: SFT<br/>Structure/Format/Step Alignment]
    SFT --> GRPO[Stage B-2: GRPO<br/>Outcome Reward + Process Reward]
    GRPO --> M[LiteCoST SLM 3B/7B]
```

### Key Designs

**1. CoST Template: Rewriting QA into Auditable Four-Step Structural Trajectories**—This is the core of Stage A. Given a question, document, ground-truth answer, and CoST template, a strong LLM follows four steps: (A1) *Structural Analysis* performs question-oriented structure selection (tables for statistical comparisons, graphs for relational reasoning) and parses the question to enumerate task-relevant attributes/entities (e.g., Company, Asset, Year) to build a dynamic schema, avoiding exhaustive parsing of the entire corpus; (A2) *Trajectory Generation* follows the schema to extract, align, and serialize data into deterministic structures while outputting reasoning trajectories; (A3) *Quality Verification* utilizes LLM-as-Judge to evaluate whether the structure can correctly answer the question; (A4) *Iterative Refinement* uses an Iterative Structuralizer to recursively restructure low-quality samples into "supplementary extraction" tasks rather than discarding them, providing richer supervision. The final output is the high-quality pair $(c^*, S^*)$.

**2. SFT→GRPO Two-Stage Adaptation: Learning Structural Discipline and Reasoning Consistency**—Stage B transfers Stage A's capabilities into the SLM. Each training sample $z=(i,d,c^*,y^*)$ contains the question, document, CoST trajectory, and structured output. First, SFT (LoRA, rank 16) is performed to give the base model basic information extraction skills driven by CoT, mitigating extraction errors during deployment. Then, GRPO is used for RL. GRPO samples a group of outputs $\{o_1,\dots,o_G\}$ for each question and optimizes the objective $J_{GRPO}$ based on relative advantages within the group $A_i=\frac{r_i-\text{mean}(r)}{\text{std}(r)}$, allowing stable updates without a value network.

**3. Dual-Layer (Outcome + Process) Reward: Refining Sparse Rewards into Step-by-Step Supervision**—This is the key design for infusing structured behavior into small models. *Outcome Reward* consists of two parts: **Format Compliance** uses layered rewards (0.5 for having `<reasoning>`+`<answer>` without redundancy, and 1.0 for explicit step labels); **Answer Correctness** uses a hybrid metric $f_{score}=\alpha\cdot S_{struct}+(1-\alpha)\cdot S_{sem}$ ($\alpha=0.3$), where $S_{struct}$ checks row/column alignment and $S_{sem}$ uses GPT-4o-mini for semantic similarity. *Process Reward* addresses reward sparsity by judging if each step $s_i$ aligns with ground truth $s_i^*$: $R_{process}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\text{Cons}(s_i,s_i^*\mid I_{consistency})]$. The total reward is the sum of these three, with the process reward scaled by a trajectory-level coefficient $\tilde{R}_{process}(s_i)=R_{process}(s_i)\cdot\gamma(T_i)$ to reinforce correct reasoning and penalize overthinking or format errors.

## Key Experimental Results

### Main Results (Loong Financial Subset, AS=Average Score 0–100, PR=Perfect Rate)

| Model | Scale | Overall AS | Overall PR |
|------|------|-----------|-----------|
| LLaMA-3.2-3B (Base) | 3B | 49.37 | 0.11 |
| **LLaMA-LiteCoST (Ours)** | 3B | **76.95** (↑27.58) | **0.40** (↑0.29) |
| Qwen2-7B (Base) | 7B | 62.10 | 0.26 |
| **Qwen-LiteCoST (Ours)** | 7B | **79.93** (↑17.83) | **0.48** (↑0.22) |
| GPT-4o-mini | 8B | 78.08 | 0.51 |
| Qwen2.5-14B-Instruct | 14B | 75.60 | 0.38 |
| GPT-4o | 200B | 79.32 | 0.54 |
| DeepSeek-R1 | 671B | 78.18 | 0.46 |

Qwen-LiteCoST (7B) achieves an Overall AS that exceeds GPT-4o-mini (+1.85), DeepSeek-R1 (+1.75), and GPT-4o (+0.61), matching or outperforming models 100 times its size.

### Comparison with SOTA Methods (Loong Financial Subset Overall AS / PR)

| Method | LLaMA-3B Base | Qwen-7B Base |
|------|--------------|--------------|
| StructRAG | 36.04 / 0.01 | 49.68 / 0.03 |
| Struc-bench | 49.90 / 0.11 | 73.72 / 0.44 |
| IEpile | 61.90 / 0.22 | 69.19 / 0.35 |
| **LiteCoST** | **76.95 / 0.40** | **79.93 / 0.48** |

Compared to the strongest baseline StructRAG, the gains are (+30.91/+0.39) and (+30.47/+0.46). Compared to the best fine-tuning methods, improvements are +15.05 (over IEpile) and +6.41 (over Strucbench).

### Ablation Study (Reward Design, Overall AS / PR)

| Configuration | LLaMA-Ours | Qwen-Ours |
|------|-----------|-----------|
| Full | 76.95 / 0.40 | 79.93 / 0.48 |
| w/o Process Reward | 75.52 / 0.37 | 77.39 / 0.46 |
| w/o Outcome Reward | 72.55 / 0.32 | 75.43 / 0.39 |

Removing either reward leads to a performance drop; removing the outcome reward results in a more significant decrease (e.g., 4.4 points for LLaMA), indicating that outcome and process rewards are complementary.

### Key Findings
- **Structured data universally boosts LLM reasoning**: Replacing raw long documents with LiteCoST-generated SSOs improved Overall scores for Qwen2-72B/GPT-4o-mini/GPT-4o/Claude-3.5 by 12.41/8.77/9.04/8.47 respectively.
- **Efficiency**: Qwen-LiteCoST has a per-sample latency of 12.09s, lower than LLaMA-3.1-8B (13.19s) and Qwen2.5-14B (14.71s). It is approximately 2$\times$ faster than GPT-4o (21.15s) and 4$\times$ faster than DeepSeek-R1 (44.44s). For even higher speed, LLaMA-LiteCoST (8.04s) is available.
- **Low Cost**: The two-stage training (LoRA 3 epochs + GRPO) costs approximately \$20, with a maximum generation length of 2048 tokens.

## Highlights & Insights
- **Replacing "Answering" with "Constructing Structure then Answering"**: Using structured intermediates as an interface inherently provides visible, verifiable, and reusable evidence. This is more robust than direct long-context reasoning and suppresses hallucinations and format drift.
- **Single Call to Strong LLMs**: Expensive large models are used as "one-time teachers" to produce auditable supervision rather than being required for online inference, fundamentally solving the trilemma of cost, latency, and privacy.
- **Process Rewards Fill Sparse RL Signals**: Entity-level and tuple-level step-by-step consistency rewards, combined with trajectory-level positive/negative scaling, refine sparse "final answer correct" supervision into dense step-by-step guidance. This is critical for small models to match large models.

## Limitations & Future Work
- **Reliance on GPT-4o as Teacher and Judge**: CoST data generation, quality verification, and semantic scoring use GPT-4o(-mini), making supervision quality and evaluation dependent on its capabilities and biases.
- **Only Applicable to Structurable Problems**: The method explicitly excludes open-ended narrative questions (unsuitable for table/graph representation), limiting its scope to queries where answers can be derived from structure.
- **Evaluation Scope**: Analysis primarily focused on the financial subset of Loong (legal/scientific in appendix). Robustness across more domains and longer contexts requires broader validation.
- **Structural Selection Granularity**: Structure types are limited to tables, graphs, and blocks; more complex hybrid or hierarchical structures are not yet covered.

## Related Work & Insights
- **QA-by-Structuring / Structured RAG**: Aligned with StructRAG and GraphRAG (Edge et al.), advocating for solidifying scattered evidence into structures. This paper uses distillation and RL to compress this pipeline into small models.
- **CoT and Structured Reasoning**: CoST upgrades Chain-of-Thought into a "schema-aware + serializable" structured reasoning chain, balancing readable trajectories with machine-verifiable products.
- **GRPO Distillation**: Follows the DeepSeek GRPO lineage (group relative advantage without value networks). The innovation lies in designing dual-layer (format/answer/process) rewards for structured extraction.
- **Insight**: For any "extract-then-reason" task (table QA, report analysis, KG construction), "strong model trajectory generation + small model dual-signal RL distillation" provides a reusable, low-cost, and locally deployable paradigm.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The CoST template rewrites QA into auditable structured trajectories, combined with SFT→GRPO distillation using dual-layer (outcome+process) rewards. While individual components (Structured RAG, GRPO, process rewards) exist, the combination is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers effectiveness, efficiency, ablation, and generalization across multiple dimensions, including comparisons with LLMs and SOTA IE methods. Slightly limited by the focus on the financial subset and LLM-as-Judge evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — The two-pillar, four-step, and dual-layer reward structure is clear, with sufficient diagrams and a logical progression of motivation.
- **Value**: ⭐⭐⭐⭐ — For a training cost of \$20, a 7B model can match GPT-4o on long-doc QA while being 2–4$\times$ faster and locally deployable, offering high practical value for high-stakes scenarios like finance and law.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rethinking Reasoning in Document Ranking: Why Chain-of-Thought Falls Short](rethinking_reasoning_in_document_ranking_why_chain-of-thought_falls_short.md)
- [\[ICLR 2026\] Fathom-DeepResearch: Unlocking Long Horizon Information Retrieval and Synthesis for SLMs](fathom-deepresearch_unlocking_long_horizon_information_retrieval_and_synthesis_f.md)
- [\[ACL 2026\] GIFT: Guided Fine-Tuning and Transfer for Enhancing Instruction-Tuned Language Models](../../ACL2026/information_retrieval/gift_guided_fine-tuning_and_transfer_for_enhancing_instruction-tuned_language_mo.md)
- [\[ACL 2026\] Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA](../../ACL2026/information_retrieval/navigating_large-scale_document_collections_mudabench_for_multi-document_analyti.md)
- [\[ICLR 2026\] G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)

</div>

<!-- RELATED:END -->
