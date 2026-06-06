---
title: >-
  [Paper Note] Teaching Language Models to Check Grounded Claim Factuality with Human Test-Taking Strategies
description: >-
  [ACL2026][LLM Evaluation][Fact-checking] This work reframes grounded claim factuality checking as a True/False reading comprehension task. By incorporating human test-taking strategies into structured prompts…
tags:
  - "ACL2026"
  - "LLM Evaluation"
  - "Fact-checking"
  - "Grounded Factuality Checking"
  - "LLM Hallucination Detection"
  - "Reading Comprehension Strategies"
  - "SLM Distillation"
date: 2026-05-08
content_hash: a5f6901751d24ce4
---

# Teaching Language Models to Check Grounded Claim Factuality with Human Test-Taking Strategies

**Conference**: ACL2026  
**arXiv**: [2605.29712](https://arxiv.org/abs/2605.29712)  
**Code**: https://github.com/Haruhi07/Test-Taking  
**Area**: LLM Evaluation  
**Keywords**: Fact-checking, Grounded Factuality Checking, LLM Hallucination Detection, Reading Comprehension Strategies, SLM Distillation

## TL;DR

This work reframes grounded claim factuality checking as a True/False reading comprehension task. By incorporating human test-taking strategies into structured prompts, LLMs are enabled to verify claims efficiently and accurately with minimal reasoning steps. Additionally, small language models are trained via Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to replace large models, achieving over 80% reduction in inference costs.

## Background & Motivation

**Background**: Large Language Models (LLMs) are widely used in generative tasks such as summarization and QA, but the generated content often suffers from "hallucinations"—claims that are not supported by source documents. This pose a fatal threat to the reliability of applications like Retrieval-Augmented Generation (RAG). Current factuality evaluation methods are primarily categorized into two types: (1) Entailment classifier-based methods, which are lightweight but require threshold tuning for specific datasets and lack generalizability; (2) "LLM-as-a-judge" paradigms, which lack explicit guidance for the reasoning process, leading to lengthy and costly inference.

**Limitations of Prior Work**: Textual entailment methods require document truncation or chunking, leading to information loss. Direct LLM judgments fail to fully exploit reasoning capabilities—models either generate excessively long free-form reasoning or produce inconsistent judgments without structural guidance. Furthermore, cross-dataset generalization remains weak.

**Key Challenge**: How can models perform factuality checking systematically and explainably without increasing inference costs? A tension exists between model complexity and efficiency—large models are capable but expensive, while small models are fast but exhibit limited comprehension.

**Goal**: To design a two-stage pipeline that decomposes claims into atomic facts and verifies them systematically, while developing methods to distill LLMs into small models for balanced performance and cost.

**Key Insight**: The authors observe that humans employ systematic strategies for True/False reading comprehension in language exams: verifying explicitly mentioned information before inferring implicit information. This exam strategy is applicable to LLM factuality checking by converting it into a set of explicit verification criteria to guide the reasoning process.

**Core Idea**: Factuality checking is redefined as a reading comprehension task. A set of four test-taking criteria (C1-C4) replaces free-form reasoning, allowing LLMs to generate judgments and explanations in a structured, controllable manner, significantly reducing inference costs.

## Method

### Overall Architecture

The method adopts a **two-stage pipeline**. The first stage, "Claim Decomposition," uses an LLM to break complex claims into multiple atomic facts, reducing the difficulty of subsequent verification. The second stage, "Atomic Fact Verification," checks each atomic fact against the source document and aggregates the final decision. The key insight is that complex claims often contain independent information pieces scattered across different locations in the source document; checking the whole claim directly leads to information omission or confusion.

### Key Designs

1.  **Test-taking Based Verification Criteria**:
    - **Function**: Converts the ambiguous problem of "checking if a claim is grounded" into a set of executable criteria, verifying entity mentions, description accuracy, relationship support, and implicit reasoning.
    - **Mechanism**: Four criteria are applied sequentially to form a logic flow: first verify C1 (whether subjects/objects in the claim are mentioned), then C2 (whether descriptions of these entities are explicitly supported), followed by C3 (whether relationships between entities are explicitly supported), and finally C4 (whether unverified information can be inferred). This prevents blind searching during reasoning.
    - **Design Motivation**: Humans use this strategy—finding explicit evidence before making inferences—to improve accuracy and reduce cognitive load. Compared to previous methods that check error types one by one, this serialized decision-tree design better aligns with human cognitive processes.

2.  **Decoupling Claim Decomposition and Atomic Fact Verification**:
    - **Function**: Splits the complex verification task into two independent sub-tasks, using targeted LLM prompts or models for each.
    - **Mechanism**: First, a few-shot prompt guides the LLM to decompose a claim like "Ice can turn into liquid water, and liquid water can turn into steam, and vice versa" into atomic facts like "Ice can turn into water," "Water can turn into steam," and "Steam can turn into ice." Second, verification criteria are applied to each fact. Decoupling ensures: (a) focused logical segmentation during decomposition and focused evidence retrieval during verification; (b) flexibility to replace stages with small models.
    - **Design Motivation**: Concurrent tasks have lower parallelism and higher error propagation. Decomposition makes the pipeline modular and independently optimizable.

3.  **Two-stage Training Strategy for SLM Distillation**:
    - **Function**: Combines SFT and DPO to teach small models (e.g., 0.6B parameters) to follow LLM verification strategies and self-correct, maintaining LLM-level accuracy with significantly lower costs.
    - **Mechanism**: Phase one (SFT) has the small model mimic LLM-generated atomic facts and reasoning. Phase two (DPO) focuses on samples where the small model failed but the LLM succeeded. Through contrastive learning (LLM output as "chosen," small model error as "rejected"), the small model learns to correct its mistakes. DPO is more efficient than SFT as it maximizes the probability margin between correct and incorrect samples rather than simple imitation.
    - **Design Motivation**: Small models lack the world knowledge and reasoning depth of LLMs, but they can compensate through distillation and error correction. This simulates human learning: mastering basic steps (SFT) before identifying and correcting common mistakes (DPO).

### Loss & Training

SFT Objective (Claim Decomposition): $L(\theta) = \mathbb{E}_{(c,\{f_{\text{ref}}\}) \sim D_{\text{De}}}[\log P_\theta(\{f_{\text{ref}}\} | c)]$, where $c$ is the claim and $\{f_{\text{ref}}\}$ is the set of LLM-generated reference facts.

SFT Objective (Fact Verification): $L(\theta) = \mathbb{E}_{D_{\text{Re\_SFT}}}[\log P_\theta(r_{\text{ref}} | x)]$, where $x$ contains the source document and atomic facts, and $r_{\text{ref}}$ is the LLM-generated reference explanation.

DPO Objective (Mistake Revision): $L(\theta) = -\mathbb{E}_{D_{\text{Re\_DPO}}}[\log \sigma[\beta(s_\theta(x, y_c) - s_\theta(x, y_r))]]$, where $y_c$ is the correct LLM output, $y_r$ is the erroneous small model output, $s_\theta$ is the log probability, and $\beta$ is a temperature parameter.

## Key Experimental Results

### Main Results

Tested on two standard datasets: **FacTax-Benchmark** (news and dialogue summarization) and **LLM-AggreFact** (multi-source, LLM-generated claims). The metric is **Balanced Accuracy** (BAcc) to handle imbalances: $\text{BAcc} = \frac{1}{2}(\text{TP}/(\text{TP}+\text{FN}) + \text{TN}/(\text{TN}+\text{FP}))$.

| Method | Size | FacTax-Bench | LLM-AggreFact | Avg Rank |
| :--- | :--- | :--- | :--- | :--- |
| ChatGPT-3.5 (ZS) | - | 70.1 | 70.1 | 13.8 |
| TrueTeacher | 11B | 73.0 | 73.3 | 8.4 |
| FactCG | 0.4B | 67.0 | 75.6 | 5.8 |
| MiniCheck-BeSpoke | 7B | 71.4 | 77.4 | 3.3 |
| Qwen3-4B-Instruct (Ours) | 4B | 73.0 | 75.6 | 7.1 |
| Qwen3-30B-Instruct (Ours) | 30B | **78.0** | 76.3 | **3.6** |
| Qwen3-0.6B+SFT (Ours) | 0.6B | 68.9 | 71.3 | 12.1 |
| Qwen3-0.6B+SFT+DPO (Ours) | 0.6B | 72.6 | 73.6 | 7.2 |

Ours (Qwen3-30B-Instruct) achieves a **new SOTA** (78.0) on FacTax-Benchmark and ranks second on LLM-AggreFact. Notably, the 0.6B model with SFT+DPO approaches ChatGPT-3.5 levels and matches the performance of TrueTeacher (11B).

### Ablation Study

| Config | FacTax | LLM-AggreFact | Notes |
| :--- | :--- | :--- | :--- |
| Full Model | 73.0 | 75.6 | Decomposition + Strategy |
| w/o Decomposition | 72.3 | 74.6 | Criteria only, no decomposition |
| w/o Strategy | 71.6 | 73.1 | Decomposition followed by direct check |
| w/o Both | 69.4 | 72.1 | Direct check on original claim |

**Key Findings**: (1) **Claim decomposition provides stable gains**—accuracy drops by 0.7-1.0% without it, showing it is necessary but not the primary driver. (2) **Test-taking strategy is the primary contributor**—accuracy drops by 1.4-2.5% without C1-C4 criteria, indicating the guided reasoning flow is the core value. (3) **Token usage is significantly reduced**—compared to "thinking" modes, this method uses only 10.4%-10.5% of tokens on FacTax and 12.5%-17.7% on LLM-AggreFact, saving **over 80%** in inference costs. (4) **Small model training is essential**—leave-one-out tests show generalization drops significantly if data from a specific dataset is removed, indicating small models require diverse multi-source data.

## Highlights & Insights

-   **Clever Transfer of Exam Strategies**: Using the common human strategy of "explicit evidence before inference" to guide machine learning tasks demonstrates the value of cross-domain knowledge transfer. The strategy is concise, avoids inefficient long-chain reasoning, and yields over 80% token savings.
-   **Practical Significance of Decoupling**: Splitting complex tasks into decomposition and verification modules allows for module-level optimization and enables the mixing of different model sizes. This modularity is a useful reference for other multi-step reasoning tasks.
-   **Innovative Application of DPO**: Combining SFT and DPO to let small models master standard procedures and then correct errors outperforms pure SFT. This training framework effectively bridges supervised learning and reinforcement learning for resource-constrained scenarios.
-   **Cross-dataset Robustness**: The competitive results across two significantly different datasets (FacTax rank 3.6, LLM-AggreFact rank 4.0) validate the generalizability of the designed criteria.

## Limitations & Future Work

-   **Limited SLM Generalization**: Experiments show small models require abundant multi-source data to generalize to new domains, limiting applicability in low-resource settings. Future work could explore meta-learning or few-shot adaptation.
-   **Instability on Complex Documents**: On datasets with long/complex documents (e.g., LFQA, TOFUEVAL-MediaS), even LLM teachers struggle, suggesting a bottleneck in long-document comprehension. Potential directions include retrieval-augmentation or information compression.
-   **Trade-off between Strictness and Reasoning**: Ablation shows models can be overly strict with C3/C4 criteria (e.g., misjudging claims due to "vice versa" phrasing). Future work might consider dynamic thresholds or context-aware similarity metrics.
-   **Inference Chain Length**: While token usage is reduced, more aggressive compression—such as replacing criteria-guided reasoning with keyword-based extraction—remains unexplored.

## Related Work & Insights

-   **vs Entailment Methods** (Zha et al., 2023; Laban et al., 2022): Classifiers are lightweight but sensitive to length and thresholds. Ours avoids parameter tuning, outputs binary decisions directly, and improves accuracy through guidance.
-   **vs Direct LLM Judgments** (Luo et al., 2023; Xu et al., 2024): Prior works allow free-form reasoning or simple error definitions. Our innovation lies in a **systematic verification process** that improves accuracy while slashing costs.
-   **vs QA-based Methods** (Fabbri et al., 2022; Wang et al., 2020): QA requires complex multi-step pipelines. Ours simplifies this into decomposition and criteria-based checking, making it modular and reusable.
-   **vs Distillation for Reasoning** (QwenTeam, 2025; DeepSeek-AI, 2025): While distillation has improved math/logic, our contribution is applying SFT+DPO specifically to **factuality checking**, demonstrating the potential of SLMs as evaluators.

## Rating

-   **Novelty**: ⭐⭐⭐⭐ The systematic application of human test-taking strategies as explicit reasoning criteria is intuitive yet innovative; the SFT+DPO combo is also effectively tailored for this task.
-   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive benchmarking against numerous baselines, multi-level ablations (decomposition, criteria, individual C1-C4 analysis), hyperparameter sensitivity tests, and prompt robustness verification.
-   **Writing Quality**: ⭐⭐⭐⭐ Clear logic and illustrative examples (e.g., the ice-to-steam example). Graphics are intuitive. Some sections could be slightly more concise.
-   **Value**: ⭐⭐⭐⭐⭐ Addresses the critical issue of hallucinations in RAG, providing a plug-and-play LLM eval method and a pathway for low-cost deployment. Highly relevant for both industry and academia.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation](teaching_language_models_to_forecast_research_success_through_comparative_idea_e.md)
- [\[ACL 2026\] Revisiting the Reliability of Language Models in Instruction-Following](revisiting_the_reliability_of_language_models_in_instruction-following.md)
- [\[ACL 2026\] Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models](revisiting_a_pain_in_the_neck_a_semantic_reasoning_benchmark_for_language_models.md)
- [\[ACL 2026\] Zero-shot Large Language Models for Automatic Readability Assessment](zero-shot_large_language_models_for_automatic_readability_assessment.md)
- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)

</div>

<!-- RELATED:END -->
