---
title: >-
  [Paper Note] Progressive Multimodal Reasoning via Active Retrieval
description: >-
  [ACL 2025][VLM Reasoning][Multimodal Reasoning] This paper proposes the AR-MCTS framework, which combines Active Retrieval with Monte Carlo Tree Search (MCTS) to dynamically retrieve key knowledge at each step of multi-step multimodal reasoning, replacing traditional beam search sampling. It automatically generates step-by-step reasoning annotations to progressively align the Process Reward Model (PRM), significantly improving the reasoning performance of multiple MLLMs on Ma…
tags:
  - "ACL 2025"
  - "VLM Reasoning"
  - "Multimodal Reasoning"
  - "Monte Carlo Tree Search"
  - "Active Retrieval"
  - "Process Reward Model"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: 4f4699357b7183b6
---

# Progressive Multimodal Reasoning via Active Retrieval

**Conference**: ACL 2025  
**arXiv**: [2412.14835](https://arxiv.org/abs/2412.14835)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Reasoning, Monte Carlo Tree Search, Active Retrieval, Process Reward Model, Retrieval-Augmented Generation

## TL;DR
This paper proposes the AR-MCTS framework, which combines Active Retrieval with Monte Carlo Tree Search (MCTS) to dynamically retrieve key knowledge at each step of multi-step multimodal reasoning, replacing traditional beam search sampling. It automatically generates step-by-step reasoning annotations to progressively align the Process Reward Model (PRM), significantly improving the reasoning performance of multiple MLLMs on MathVista, We-Math, and GAOKAO-MM.

## Background & Motivation
Multi-step multimodal reasoning is one of the core challenges faced by multimodal large language models (MLLMs). Existing MCTS-based methods have achieved success in text-only LLMs but present two key limitations in multimodal scenarios:

**Insufficient Knowledge in the Expansion Phase**: Traditional MCTS relies on beam search during the expansion phase, using the model's internal knowledge for sampling. For text tasks, the LLM's internal knowledge is sufficient to support reasoning path expansion. However, in multimodal reasoning, cross-modal interactions often suffer from alignment errors, preventing internal knowledge from supporting reliable path sampling.

**Error Accumulation Effect**: In multi-step reasoning, each step depends on the previous one, and small errors magnify as the number of steps increases.

Key Challenge: How to provide reliable external knowledge support for multimodal reasoning in the expansion phase of MCTS while maintaining both diversity and accuracy in the sampling space?

Key Insight: Dynamically retrieve different problem-solving insights at each step of MCTS expansion, replacing the single sampling strategy of traditional beam search to improve sampling quality and diversity.

Core Idea: **Replace beam search with active retrieval as the MCTS expansion strategy to achieve external-knowledge-driven multimodal reasoning path sampling and verification**.

## Method

### Overall Architecture
AR-MCTS consists of two core components:
1. **Unified Retrieval Module**: Hybrid-modal retrieval corpus construction + multimodal retrieval module + knowledge concept filtering.
2. **Reasoning Annotation via MCTS and Active Retrieval**: Utilizing the four steps of MCTS (selection, expansion, simulation, backpropagation) while integrating active retrieval in the expansion phase to automatically obtain step-by-step reasoning annotations, thereby **progressively aligning the PRM**.

### Key Designs

1. **Hybrid-Modal Retrieval Corpus Construction**:

    - Math-specific knowledge: Integrates GSM8K, MATH (text-only) + MathVista, MathVerse, MathVision, We-Math (multimodal), totaling 22K text QA pairs + 12.5K multimodal sample pairs.
    - General reasoning knowledge: Employs Wikipedia and the COIG large-scale question bank.
    - Removes overlaps with the test sets using regular expressions to prevent data leakage.
    - Each sample contains a question $q$, a solving process $p$, and an answer $a$, which are concatenated into a unified format.

2. **Unified Multimodal Retrieval Module**:

    - **Text Retrieval**: Uses Contriever as a dense retriever to calculate the dot-product similarity between the query and documents.
    - **Cross-Modal Retrieval**: Uses a CLIP dual-stream architecture, averaging the outputs of the image encoder $E_I$ and text encoder $E_T$ to form a hybrid representation, and uses a FAISS index for efficient retrieval.
    - Employs different encoding strategies for text-only and multimodal samples within the hybrid-modal corpus.

3. **Knowledge Concept Filtering**:

    - Observes that multimodal reasoning is highly sensitive to the consistency of fine-grained knowledge concepts (e.g., algebraic knowledge cannot help solve triangle problems).
    - Utilizes self-contained knowledge concept labels of the datasets (e.g., "angles and lengths"), while setting a retrieval similarity threshold $T_r$ and a knowledge concept consistency threshold $T_{kc}$ for dual filtering.
    - Only samples satisfying both thresholds are considered as problem-solving insights.

4. **Active Retrieval Strategy in MCTS (Core Innovation)**:

    - **Selection**: Recursively selects child nodes from the root node using the UCB formula.
    - **Expansion (Key Improvement)**: At each expansion step, the current state's query is concatenated with previous reasoning steps to dynamically retrieve the candidate insight $r_i$ required for this step from the insight library $D_{ins}$, replacing the previous step's insight $r_{i-1}$. Different retrieved results generate different reasoning branches, enhancing sampling diversity.
    - **Simulation**: Evaluates the value $V(s_i)$ of each node using a one-step rollout, assigning values based on whether the reasoning path can derive the correct answer.
    - **Backpropagation**: Back-propagates to update visit counts and Q-values.

5. **Curriculum Process Reward Modeling (Two-Stage PRM Training)**:

    - **Stage 1: Step-level DPO pre-alignment**. The expansion and evaluation processes of MCTS naturally generate positive and negative sample pairs (value $> 0.8$ is positive, value $= 0$ is negative). A step-level DPO objective is used to train the PRM to distinguish the correctness of reasoning steps.
    - **Stage 2: Pointwise fine-tuning**. A step-level cross-entropy objective is applied to the pre-aligned PRM to further unlock its reasoning scoring capability, achieving generalization from easy to hard.

### Loss & Training
- Stage 1: Step-level DPO loss ($L_{SDPO}$), maximizing the likelihood of positive samples $y^+$ relative to negative samples $y^-$.
- Stage 2: Pointwise cross-entropy loss ($L_{PFT}$), applying sigmoid scoring to the golden label (0/1) of each state.
- In the inference phase, early stopping is set to 4 rounds. The highest-scoring node from each round is extracted, and low-quality paths are discarded.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (AR-MCTS) | Prev. SOTA (ORM/SC) | Gain |
|--------|------|------|----------|------|
| MathVista(ALL) GPT-4o | Accuracy | 62.6% | 61.9%(ORM) | +0.7% |
| We-Math(AVG) GPT-4o | Score | 46.8% | 45.2%(SC) | +1.6% |
| MathVista(ALL) Qwen2-VL-7B | Accuracy | 64.1% | 62.3%(ORM) | +1.8% |
| We-Math(AVG) Qwen2-VL-7B | Score | 28.1% | 26.4%(ORM) | +1.7% |
| MathVista(ALL) InternVL2-8B | Accuracy | 63.1% | 61.8%(SC) | +1.3% |
| We-Math(S3) InternVL2-8B | Score | 43.6% | 35.1%(SC) | +8.5% |
| GAOKAO-MM GPT-4o | Overall | 52.2% | 47.8%(SC) | +4.4% |

### Ablation Study

| Configuration | MathVista(ALL) | We-Math(S3) | Description |
|------|---------|------|------|
| AR-MCTS (Full) | 64.1% | 40.6% | Baseline |
| w/o PRM | 61.0% (-3.1) | 37.7% (-2.9) | PRM is crucial for reasoning verification |
| w/o Active Retrieval | 61.9% (-2.2) | 38.7% (-1.9) | Active retrieval significantly improves sampling quality |
| w/o Filtering | 62.8% (-1.3) | 39.5% (-1.1) | Knowledge concept filtering reduces noise |

### Key Findings
- MLLMs struggle to self-correct reasoning errors: The Self-Correction strategy leads to performance degradation across most models, with Qwen2-VL-7B dropping by over 8% on MathVista.
- PRM outperforms ORM in complex reasoning tasks: Particularly on We-Math's S3 metric (GPT-4o: 56.4% vs 50.3%).
- AR-MCTS shows more significant improvements on weaker MLLMs: Qwen2-VL-7B sees a 5.3% improvement on MathVista and an 8.3% improvement on We-Math, indicating that smaller models possess reasoning potential but lack effective decoding strategies.
- Sampling diversity analysis: Compared to beam search, AR-MCTS produces more cluster centers (38 vs 46) and a more dispersed semantic representation distribution.
- Cross-domain validation: Consistent improvements are achieved on the Chinese GAOKAO-MM dataset, with Math $+12.5\%$, Physics $+7.7\%$, and History $+20\%$.

## Highlights & Insights
- Clear theoretical modeling: Unified modeling of the MCTS expansion and simulation processes via Equation (1) reveals the core limitations of traditional methods in multimodal scenarios.
- Innovatively introduces RAG concepts into the MCTS expansion stage, replacing beam search with dynamic retrieval, which opens up new sampling strategy pathways.
- The two-stage curriculum PRM training (DPO pre-alignment $\rightarrow$ pointwise fine-tuning) is robustly designed, aligning with the "easy-to-hard" learning paradigm.
- High framework generalizability: Achieves "plug-and-play" applicability across various MLLM backbones.
- Representation of accuracy and diversity in the sampling space is backed by intuitive visual evidence.

## Limitations & Future Work
- The retrieval corpus mainly covers mathematical reasoning; its applicability to other domains (such as logical reasoning and common-sense reasoning) remains to be validated.
- Knowledge concept filtering relies on dataset-inherent category labels, which are often absent in real-world scenarios.
- Real-time performance and efficiency of the retrieval module: Performing dynamic retrieval at each step incurs substantial inference overhead.
- Promising directions: Combining active retrieval strategies with speculative decoding to improve efficiency; exploring adaptive retrieval triggering mechanisms (i.e., retrieving only when necessary, rather than at every step).

## Related Work & Insights
- MCTS application in LLM reasoning (e.g., AlphaCode, o1): This work extends it to multimodal scenarios and addresses key expansion strategy concerns.
- RAG application in the multimodal domain (e.g., MuRAG): This work novelly integrates RAG with MCTS to achieve step-by-step retrieval.
- PRM training methodologies (e.g., Math-Shepherd): This work proposes an automated multimodal PRM acronym annotation and training pipeline.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing active retrieval into the MCTS expansion stage is a fresh perspective, though the overall framework is a combination of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three benchmarks, with multiple MLLM backbones, and detailed ablation studies/analyses, though a comparison of computational overhead is lacking.
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured and theoretically rigorous, though the heavy use of notation may increase reading difficulty.
- Value: ⭐⭐⭐⭐ Provides a general framework for augmenting multimodal reasoning, offering valuable references for both the reasoning verification and retrieval-augmented generation communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Spectral-Progressive Thought Flow for Lightweight Multimodal Reasoning](../../ICML2026/vlm_reasoning/spectral-progressive_thought_flow_for_lightweight_multimodal_reasoning.md)
- [\[ICCV 2025\] Training-Free Personalization via Retrieval and Reasoning on Fingerprints](../../ICCV2025/vlm_reasoning/training-free_personalization_via_retrieval_and_reasoning_on_fingerprints.md)
- [\[NeurIPS 2025\] PhysVLM-AVR: Active Visual Reasoning for Multimodal Large Language Models in Physical Environments](../../NeurIPS2025/vlm_reasoning/physvlm-avr_active_visual_reasoning_for_multimodal_large_language_models_in_phys.md)
- [\[NeurIPS 2025\] Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieval](../../NeurIPS2025/vlm_reasoning/retrv-r1_a_reasoning-driven_mllm_framework_for_universal_and_efficient_multimoda.md)
- [\[ICLR 2026\] MathNet: A Global Multimodal Benchmark for Mathematical Reasoning and Retrieval](../../ICLR2026/vlm_reasoning/mathnet_a_global_multimodal_benchmark_for_mathematical_reasoning_and_retrieval.md)

</div>

<!-- RELATED:END -->
