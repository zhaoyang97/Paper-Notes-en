---
title: >-
  [Paper Note] Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation
description: >-
  [ACL 2025][LLM Alignment][Visual Programming Language] This paper proposes a two-stage training strategy for the automatic generation of industrial visual programming languages (specifically Ladder Diagrams): first, leveraging subroutine reuse characteristics via Retrieval-Augmented Fine-Tuning, and second, further improving accuracy through DPO training with preference pairs constructed via graph edit operations, achieving an over 10% improvement in program-level accuracy on…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Visual Programming Language"
  - "Ladder Diagram Generation"
  - "Retrieval-Augmented Fine-Tuning"
  - "Preference Optimization"
  - "Industrial Automation"
date: 2026-05-08
content_hash: d4b8c9843d00b91b
---

# Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation

**Conference**: ACL 2025  
**arXiv**: [2502.16529](https://arxiv.org/abs/2502.16529)  
**Code**: None  
**Area**: RLHF Alignment  
**Keywords**: Visual Programming Language, Ladder Diagram Generation, Retrieval-Augmented Fine-Tuning, Preference Optimization, Industrial Automation

## TL;DR

This paper proposes a two-stage training strategy for the automatic generation of industrial visual programming languages (specifically Ladder Diagrams): first, leveraging subroutine reuse characteristics via Retrieval-Augmented Fine-Tuning, and second, further improving accuracy through DPO training with preference pairs constructed via graph edit operations, achieving an over 10% improvement in program-level accuracy on real-world LD data.

## Background & Motivation

**Background**: Visual Programming Languages (VPLs) allow users to create programs through graphical interfaces and are widely used in various scenarios. Recent studies have attempted to use LLMs to generate VPL code from natural language instructions, and prompting methods, in particular, have achieved some success.

**Limitations of Prior Work**: For industrial-grade VPLs such as Ladder Diagrams (LD)—the core programming language in industrial automation—prompting methods show limited efficacy. LD involves a large number of domain-specific configurations (complex components like relays, timers, and counters) that are difficult to fully capture in a single prompt. Furthermore, LD code exhibits highly structural features, presenting a significant gap from natural language text.

**Key Challenge**: Prompting methods rely heavily on the LLM's internal knowledge, making it difficult for general LLMs to generate correct code solely through prompts due to the highly specific nature of industrial VPL domains. Meanwhile, although simple supervised fine-tuning (SFT) outperforms prompting, it fails to fully exploit the frequent reuse of subroutines in LD code and lacks negative feedback on error patterns.

**Goal**: To design a training strategy specifically for industrial VPL generation that can both leverage subroutine reuse patterns and mitigate typical errors through preference learning.

**Key Insight**: The authors observe that a large number of repeating subroutine modules exist within LD programs, which is an inherent characteristic of industrial programming. Concurrently, the graph-like structure of LD allows for systematic construction of "near-correct yet flawed" negative samples using graph edit operations.

**Core Idea**: To utilize subroutine reusability via retrieval-augmented fine-tuning to improve base generation quality, and then automatically construct preference pairs using graph edit operations for DPO training, taking a two-pronged approach to boost program-level accuracy.

## Method

### Overall Architecture

The input is a natural language user instruction, and the output is the LD code (represented in a structured format). The training consists of two stages: (1) Retrieval-augmented fine-tuning phase—for each training sample, similar subroutine snippets are retrieved and prepended to the input for supervised fine-tuning; (2) DPO phase—graph edit operations are utilized to generate "slightly flawed" variants from correct LD codes as rejected samples, which are paired with the original correct codes for preference optimization. No preference data is needed during inference, where generation is performed directly.

### Key Designs

1. **Retrieval-Augmented Fine-Tuning (RAFT)**:

    - **Function**: Leveraging the frequent reuse of subroutines in LD, this retrieves relevant subroutine snippets as context for each training sample during fine-tuning.
    - **Mechanism**: A subroutine index library is built, and for each training instruction, the top-k most similar subroutines are retrieved and prepended to the input to feed into the model. This is similar to RAG but applied to the fine-tuning phase—the model learns to "generate full programs under the condition of having reference subroutines." Retrieval uses code-structure-similarity-based methods rather than simple text matching.
    - **Design Motivation**: Approximately 60-70% of modules in industrial LD programs consist of recurring standard subroutines. Retrieving these existing patterns greatly reduces the difficulty of generation from scratch, allowing the model to focus on synthesis and customization.

2. **Graph-Edit-Based Preference Pair Generation**:

    - **Function**: Automatically generating high-quality preference data pairs for DPO training.
    - **Mechanism**: LD code is essentially a graph structure where nodes represent components and edges represent connection relationships. The authors define several graph edit operations—such as node deletion, node substitution, edge deletion, and edge redirection—to apply minor modifications to the correct LD graph, generating "near-correct yet slightly flawed" variants. These variants serve as the rejected samples, while the original correct codes serve as the chosen samples. A smaller edit distance produces more discriminative preference pairs.
    - **Design Motivation**: Traditional DPO requires human-annotated preference data or model-based sampling to generate positive/negative pairs, which is high-cost and uncontrollable in quality. Leveraging the graph structure of LD allows for systematic, low-cost construction of semantically meaningful negative samples, where the edit operations closely align with actual error patterns in model generation.

3. **Two-Stage Progressive Training**:

    - **Function**: Establishing a strong foundation before performing fine-grained refinement.
    - **Mechanism**: The first-stage RAFT enables the model to learn to generate correct code using retrieved subroutines; the second-stage DPO further rectifies the model's error tendencies through preference learning on top of this. Both stages share the same underlying LLM (e.g., CodeLlama-7B), with DPO being trained continuously starting from the RAFT checkpoint.
    - **Design Motivation**: Directly performing DPO without initial RAFT yields limited efficacy due to the model's insufficient base generation capabilities; the progressive strategy of RAFT followed by DPO allows both techniques to play to their respective strengths.

### Loss & Training

The first stage uses standard cross-entropy loss for supervised fine-tuning. The second stage uses DPO loss: $L_{DPO} = -\mathbb{E}[\log \sigma(\beta \cdot (\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}))]$, where $y_w$ is the chosen sample (correct LD), $y_l$ is the rejected sample (flawed LD after graph edits), and $\pi_{ref}$ is the checkpoint from the RAFT stage.

## Key Experimental Results

### Main Results

Comparison of Program-level Accuracy (PA) on a real-world industrial LD dataset:

| Method | Model | PA (%) | Gain |
|------|------|--------|------|
| Few-shot Prompting | GPT-4 | 42.3 | baseline |
| Few-shot Prompting | CodeLlama-34B | 38.7 | - |
| SFT | CodeLlama-7B | 55.1 | +12.8 |
| RAFT | CodeLlama-7B | 61.4 | +19.1 |
| RAFT + DPO (Ours) | CodeLlama-7B | 65.8 | +23.5 |

### Ablation Study

| Configuration | PA (%) | Description |
|------|--------|------|
| SFT only | 55.1 | Supervised fine-tuning baseline |
| SFT + DPO (Random Negatives) | 57.3 | Randomly constructed negative samples show limited effect |
| SFT + DPO (Graph-Edit Negatives) | 59.8 | Graph-edit negatives outperform random ones |
| RAFT only | 61.4 | Retrieval augmentation contributes the most |
| RAFT + DPO (Random Negatives) | 63.1 | Adding DPO on top of RAFT |
| RAFT + DPO (Graph-Edit Negatives) | 65.8 | Full method achieves the best result |

### Key Findings

- Training methods (SFT) on smaller models can significantly outperform prompting on large models, indicating that domain-specific tasks like LD heavily rely on fine-tuning.
- RAFT contributes the most to the performance gain (+6.3% over SFT), demonstrating that the reusability of retrieved subroutines serves as a crucial prior for LD generation.
- Preference pairs constructed via graph edits outperform random negative samples by 2-3% on average, indicating that structured negative samples are more effective.
- Even with a 7B model, the proposed method surpasses the few-shot performance of GPT-4 by 23.5 percentage points.

## Highlights & Insights

- **Utilizing graph edit operations to construct DPO preference pairs is highly ingenious**: It fully exploits the graph structure features of LD code, avoiding the high cost of manual annotation. This idea of "leveraging inherent structural features of data to construct training signals" is transferable to other structured output tasks (e.g., SQL generation, circuit design, etc.).
- **The application of Retrieval-Augmented Fine-Tuning (RAFT) in industrial code generation holds practical value**: The extremely high reuse rate of subroutines in industrial code directly guided the method design, revealing the fundamental difference between industrial and general-purpose code generation.
- **The finding that fine-tuning small models far outperforms prompting large models** holds significant practical guidance in specialized domains.

## Limitations & Future Work

- Experiments are only validated on a single VPL (Ladder Diagram); its applicability to other industrial VPLs (e.g., Function Block Diagram) remains unexplored.
- Retrieval augmentation introduces additional retrieval latency during inference, which may impact real-time deployment scenarios.
- The types and extent of graph edit operations require manual design, lacking an adaptive mechanism.
- The dataset scale and diversity are limited (acquiring real-world industrial data itself is a major challenge).

## Related Work & Insights

- **vs. Direct prompting on CodeLlama/GPT-4**: These general-purpose models show limited understanding of industrial VPLs through prompting, while this work demonstrates the necessity of fine-tuning.
- **vs. Standard SFT**: Pure supervised fine-tuning overlooks subroutine reuse characteristics and error pattern learning, whereas the proposed two-stage strategy is more comprehensive.
- **vs. RAG in NLP**: While RAG is typically used during inference, this work introduces retrieval augmentation into the training phase (RAFT), allowing the model to internalize the ability to "utilize retrieved references."

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of using graph edits to construct preference pairs is highly creative, and the combination of RAFT + DPO is a first in the context of industrial VPLs.
- Experimental Thoroughness: ⭐⭐⭐ The ablation study is relatively complete, but limited to a single dataset and a single VPL type.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, and the methodological motivation is fully explained.
- Value: ⭐⭐⭐⭐ It holds direct application value for the industrial automation field, and its methodology offers general insights for structured code generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation](rpo_retrieval_preference_optimization_for_robust_retrieval-augmented_generation.md)
- [\[ACL 2025\] Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](intuitive_fine_tuning_simplifying_alignment_into_single_process.md)
- [\[ACL 2025\] Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization](fine-grained_video_dubbing_duration_alignment_with_segment_supervised_preference.md)
- [\[ACL 2025\] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)
- [\[ACL 2025\] Reverse Preference Optimization for Complex Instruction Following](reverse_preference_optimization_for_complex_instruction_following.md)

</div>

<!-- RELATED:END -->
