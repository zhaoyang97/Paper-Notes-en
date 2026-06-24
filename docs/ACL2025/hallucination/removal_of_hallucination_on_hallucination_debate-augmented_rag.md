---
title: >-
  [Paper Note] Removal of Hallucination on Hallucination: Debate-Augmented RAG
description: >-
  [ACL 2025][Hallucination Detection][Retrieval-Augmented Generation] DRAG (Debate-Augmented RAG) proposes introducing a Multi-Agent Debate (MAD) mechanism in both the retrieval and generation stages of RAG systems. Through a structured process of proponent-opponent debate and judge arbitration, it eliminates the "hallucination on hallucination" problem caused by erroneous retrieval, significantly improving factual accuracy across six QA benchmarks.
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "Retrieval-Augmented Generation"
  - "Hallucination Elimination"
  - "Multi-Agent Debate"
  - "Training-Free"
  - "Factual Accuracy"
date: 2026-05-08
content_hash: 87d06c7e9c296847
---

# Removal of Hallucination on Hallucination: Debate-Augmented RAG

**Conference**: ACL 2025  
**arXiv**: [2505.18581](https://arxiv.org/abs/2505.18581)  
**Code**: [GitHub](https://github.com/Huenao/Debate-Augmented-RAG)  
**Area**: Hallucination Detection  
**Keywords**: Retrieval-Augmented Generation, Hallucination Elimination, Multi-Agent Debate, Training-Free, Factual Accuracy  

## TL;DR
DRAG (Debate-Augmented RAG) proposes introducing a Multi-Agent Debate (MAD) mechanism in both the retrieval and generation stages of RAG systems. Through a structured process of proponent-opponent debate and judge arbitration, it eliminates the "hallucination on hallucination" problem caused by erroneous retrieval, significantly improving factual accuracy across six QA benchmarks.

## Background & Motivation

**Background**: Retrieval-Augmented Generation (RAG) enhances the factual accuracy of LLMs by introducing external knowledge, becoming a mainstream solution to reduce hallucinations. The standard RAG pipeline is: User Query $\rightarrow$ Retrieve Relevant Documents $\rightarrow$ Generate Response based on retrieved results.

**Limitations of Prior Work**: A critical, overlooked issue in RAG is that retrieval itself can be erroneous. When incorrect or biased documents are retrieved, the model not only fails to benefit but is misled, generating more inaccurate responses than no-retrieval generation. More severely, the model may exhibit over-reliance on incorrect retrieval results, "packaging" false information to look like authentic knowledge.

**Key Challenge**: This forms a cascading effect of "hallucination on hallucination"—where the "hallucination" in the retrieval stage (incorrect documents) leads to new hallucinations in the generation stage, stacking errors across both stages. Existing RAG improvements (e.g., FLARE, Self-RAG) mainly focus on single-stage optimization and fail to systematically solve this two-stage cascading problem.

**Goal**: To design a training-free framework that introduces quality control mechanisms in both retrieval and generation stages simultaneously to disrupt the cascade of "hallucination on hallucination".

**Key Insight**: Drawing on the idea of Multi-Agent Debate (MAD), which verifies information reliability through structured debate among LLM agents with different roles. When multiple agents disagree on retrieval results or generated responses, a more reliable consensus is reached through debate and judge arbitration.

**Core Idea**: Use debate to filter unreliable documents in the retrieval stage of RAG (Retrieval Debate), and use debate to verify and correct responses in the generation stage (Response Debate), without requiring any additional training.

## Method

### Overall Architecture
DRAG extends the standard RAG pipeline into a two-stage debate architecture: (1) **Retrieval Debate**: Conducts reliability debates on retrieved documents to filter high-quality documents; (2) **Response Debate**: Generates responses based on the filtered documents, where multiple agents verify the factual accuracy of the answers through adversarial debate. Each stage involves three roles—Proponent, Opponent, and Judge—reaching a consensus through multiple rounds of debate.

### Key Designs

1. **Retrieval Debate**:

    - **Function**: Evaluates and filters the reliability of retrieved documents.
    - **Mechanism**: Given a user query and $K$ retrieved documents, three roles are instantiated: the Proponent agent argues that the documents are relevant and factually reliable; the Opponent agent identifies inconsistencies, biases, or mismatches with the query; and the Judge agent synthesizes arguments from both sides to make a final ruling. Each document undergoes multiple rounds of debate (2-3 rounds by default), after which the Judge determines whether it is trustworthy. Unreliable documents are filtered out, leaving only high-quality documents that pass the debate as input for the generation stage.
    - **Design Motivation**: A single agent struggles to comprehensively evaluate document quality. Adversarial debate enforces a rigorous evaluation by "intentionally seeking counter-evidence".

2. **Response Debate**:

    - **Function**: Verifies and refines generated responses through multi-role debate.
    - **Mechanism**: Introduces an asymmetric information role design. The Proponent agent receives the complete retrieved documents and the question to generate an initial response. The Opponent agent receives only the question (without seeing the retrieved documents) and is tasked with questioning potential errors in the Proponent's response based on its own knowledge. The Judge agent synthesizes both arguments, referring to the retrieved documents while considering logical loopholes pointed out by the Opponent, and finally generates a verified response. Through multiple rounds of adversarial debate, the factual reliability of the response is progressively enhanced.
    - **Design Motivation**: The asymmetric information design forces the debate to generate true cognitive collision. If all agents see the same incorrect documents, the debate may lapse into a "formal consensus". Shielding the Opponent from the retrieval results allows independent thinking and effectively detects biases introduced by retrieval.

3. **Debate Arbitration and Termination Strategy**:

    - **Function**: Controls debate quality and convergence efficiency.
    - **Mechanism**: The Judge agent evaluates the quality and consensus degree of both arguments after each round of debate, making a ruling through a structured evaluation template (covering dimensions like "argument strength," "evidence quality," and "logical consistency"). The debate terminates when a consensus is reached or the maximum number of debate rounds is reached. It supports customizing the number of retrieval debate rounds (`max_query_debate_rounds`) and response debate rounds (`max_answer_debate_rounds`) to flexibly balance accuracy and efficiency.
    - **Design Motivation**: Unrestricted debates can lead to "over-discussion" or infinite loops; therefore, explicit termination conditions and quality assessment standards are necessary.

### Loss & Training
DRAG is a completely training-free framework, involving no model fine-tuning or loss functions. All agents use the same pre-trained LLM (e.g., Llama-3-8B-Instruct), assigned different debate roles via system prompts. The entire framework is built on the FlashRAG library, supporting various LLMs as backbone models.

## Key Experimental Results

### Main Results
Compared with various RAG baselines on 6 QA benchmarks (using Llama-3-8B-Instruct):

| Method | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | StrategyQA |
|------|-----|----------|-------|----------|-------|------------|
| Naive Gen (No Retrieval) | 22.8 | 55.3 | 21.4 | 26.1 | 25.7 | 67.5 |
| Naive RAG | 34.5 | 59.7 | 38.2 | 31.5 | 28.9 | 63.2 |
| FLARE | 30.1 | 57.4 | 33.7 | 30.8 | 28.3 | 65.8 |
| Iter-RetGen | 33.8 | 58.1 | 36.1 | 33.2 | 30.5 | 66.1 |
| IRCoT | 35.2 | 60.3 | 37.5 | 34.1 | 31.8 | 67.3 |
| Self-RAG | 36.1 | 61.2 | 39.0 | 33.7 | 30.2 | 66.8 |
| MAD | 34.3 | 60.5 | 37.8 | 32.5 | 29.7 | 68.2 |
| **DRAG** | **38.7** | **63.5** | **42.3** | **36.8** | **34.2** | **70.1** |

### Ablation Study
Ablation analysis on compound contributions of each component:

| Configuration | NQ| TriviaQA | Description |
|------|-----|----------|------|
| Full DRAG | 38.7 | 63.5 | Best |
| W/o Retrieval Debate | 35.9 | 61.8 | Retrieval quality control is important |
| W/o Response Debate | 36.2 | 62.1 | Response verification is important |
| Standard MAD only (no role distinction) | 35.1 | 60.8 | Asymmetric role design is crucial |
| Debate 1 round | 36.8 | 62.3 | Sufficient but not optimal |
| Debate 3 rounds | 38.5 | 63.4 | Near-saturation |

### Key Findings
- **Naive RAG can perform worse than generation without retrieval on certain datasets**: On StrategyQA, Naive RAG (63.2) is lower than Naive Gen (67.5), directly validating the presence of the "hallucination on hallucination" problem where erroneous retrieval misleads the model.
- **Retrieval debate and response debate contribute equally**: Each contributes about a 2-3 percentage point improvement, indicating significant errors needing correction in both stages.
- **Asymmetric information role design is key**: Degrading Response Debate to standard MAD (all agents see the same information) significantly drops performance, validating the assumption that information asymmetry leads to more effective debate.
- Diminishing returns in debate rounds are clear: the gain from 1 to 2 rounds is significant, but the improvement of 3 rounds compared to 2 rounds is minimal, while increasing computational cost by 50%.

## Highlights & Insights
- **Problem Definition of "Hallucination on Hallucination"**: For the first time, this work systematically defines and analyzes how retrieval errors in RAG cascade to amplify generation hallucinations. This conceptual framework is inspiring for improvement directions across the RAG field.
- **Asymmetric Information Debate Design**: Allowing the opponent agent to question the proponent's response solely based on internal knowledge with no access to retrieved documents forces a truly valuable confrontation, instead of "reaching a false consensus after seeing the same incorrect information." This approach can be directly transferred to any multi-agent collaborative system.
- **Practicality of Training-Free Framework**: The entire framework requires no training and can be deployed by directly invoking existing LLMs, greatly reducing the barrier to entry.

## Limitations & Future Work
- Multi-agent debate incurs significant inference cost—each query requires multiple LLM calls, and the latency is about 3-5 times that of standard RAG.
- Debate quality depends heavily on the reasoning capability of the underlying LLM; performance may degrade with weaker models.
- Currently verified only in short-text QA scenarios; the effectiveness on more complex tasks such as long-text summarization and multi-step reasoning remains unknown.
- The neutrality of the Judge agent cannot be guaranteed, as it might be "persuaded" by a stronger side.
- **Future Directions**: Heterogeneous agent debates (using different LLMs for different roles) can be researched to increase perspective diversity. Adaptive debate round strategies (fewer debates for simple questions, more for difficult ones) can also be explored to reduce computational overhead.

## Related Work & Insights
- **vs Self-RAG**: Self-RAG trains the model itself to evaluate retrieval and generation quality, which requires extra training; DRAG is training-free but incurs higher inference costs. Both have unique advantages in different deployment scenarios.
- **vs FLARE/Iter-RetGen**: These methods improve information recall through iterative retrieval but do not involve explicit validation of retrieval reliability; DRAG directly filters unreliable documents via debate in the retrieval stage.
- **vs Multi-Agent Debate (MAD)**: Standard MAD introduces debates into reasoning tasks; DRAG extends it to RAG scenarios, designing asymmetric information roles and dual-stage applications.
- This work has direct reference value for building reliable RAG systems, especially in applications with high requirements for factual accuracy (such as healthcare and law).

## Rating
- Novelty: ⭐⭐⭐⭐ The definition of "hallucination on hallucination" and the dual-stage debate framework are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 datasets, multiple baselines, and complete ablation study.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation and intuitive framework description.
- Value: ⭐⭐⭐⭐ Practicable for improving RAG system reliability, though inference cost remains a deployment barrier.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](../../ACL2026/hallucination/stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ACL 2025\] REFIND at SemEval-2025 Task 3: Retrieval-Augmented Factuality Hallucination Detection in Large Language Models](refind_at_semeval-2025_task_3_retrieval-augmented_factuality_hallucination_detec.md)
- [\[ACL 2025\] DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination](drag_distilling_rag_slm.md)
- [\[ACL 2026\] Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate](../../ACL2026/hallucination/dialectic-med_mitigating_diagnostic_hallucinations_via_counterfactual_adversaria.md)
- [\[AAAI 2026\] MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning](../../AAAI2026/hallucination/multi-agent_undercover_gaming_hallucination_removal_via_coun.md)

</div>

<!-- RELATED:END -->
