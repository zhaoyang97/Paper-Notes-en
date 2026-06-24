---
title: >-
  [Paper Note] Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective
description: >-
  [ACL 2025][Reasoning][Mathematical Reasoning] Proposed the Chain-of-Reasoning (CoR) framework, which unifies three paradigms—Natural Language Reasoning (NLR), Algorithmic Reasoning (AR), and Symbolic Reasoning (SR)—into a single reasoning chain. Guided by a Progressive Paradigm Training (PPT) strategy, a 7B model (CoR-Math-7B) achieves a 41% accuracy improvement over GPT-4o on theorem proving under zero-shot settings, and outperforms reinforcement learning (RL) methods by 15%…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Mathematical Reasoning"
  - "Multi-Paradigm Reasoning"
  - "Theorem Proving"
  - "Arithmetic Calculation"
  - "Progressive Training"
date: 2026-05-08
content_hash: c7b1441d01ff43b6
---

# Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective

**Conference**: ACL 2025  
**arXiv**: [2501.11110](https://arxiv.org/abs/2501.11110)  
**Code**: [https://github.com/microsoft/CoR](https://github.com/microsoft/CoR)  
**Area**: LLM Reasoning  
**Keywords**: Mathematical Reasoning, Multi-Paradigm Reasoning, Theorem Proving, Arithmetic Calculation, Progressive Training

## TL;DR
Proposed the Chain-of-Reasoning (CoR) framework, which unifies three paradigms—Natural Language Reasoning (NLR), Algorithmic Reasoning (AR), and Symbolic Reasoning (SR)—into a single reasoning chain. Guided by a Progressive Paradigm Training (PPT) strategy, a 7B model (CoR-Math-7B) achieves a 41% accuracy improvement over GPT-4o on theorem proving under zero-shot settings, and outperforms reinforcement learning (RL) methods by 15% on the MATH benchmark.

## Background & Motivation

**Background**: LLM mathematical reasoning primarily relies on a single paradigm—either NLR (e.g., Chain-of-Thought step-by-step reasoning), AR (e.g., generating and executing Python code), or SR (e.g., Lean formal proofs). Researchers optimize each paradigm independently, resulting in expert models trained for specific tasks.

**Limitations of Prior Work**: (a) Asymmetric capabilities in single-paradigm models—models proficient in NLR perform poorly in theorem proving, and vice versa; (b) even with tool assistance (e.g., CoT with code execution), a single paradigm still dominates, ignoring the independent reasoning capabilities of other paradigms; (c) cross-task generalization relies heavily on few-shot examples, making zero-shot generalization difficult.

**Key Challenge**: Different mathematical tasks are naturally suited to different reasoning paradigms (arithmetic is best for AR precise computation, theorem proving for SR formal verification, and complex questions for NLR semantic comprehension). However, existing methods fail to allow models to flexibly switch and collaboratively utilize multiple paradigms within a single reasoning process.

**Goal**: To design a unified framework that enables LLMs to sequentially employ NLR, AR, and SR paradigms in a single reasoning pass, leveraging the strengths of each to achieve comprehensive cross-task mathematical capabilities.

**Key Insight**: Analogous to the modularity concept in software engineering—each paradigm serves as a module, and CoR enables them to collaborate in a chained manner, where the output of the preceding paradigm serves as a reference for the subsequent one.

**Core Idea**: To replace the single-paradigm approach with a chained combination of three reasoning paradigms, combined with a progressive training strategy that allows the model to gradually master NLR $\rightarrow$ AR $\rightarrow$ SR, achieving unified problem-solving for diverse mathematical tasks.

## Method

### Overall Architecture
Input: Mathematical problem $x$. Output: Final answer $y$. Reasoning process: $y \sim \mathbb{P}(y|x, \tau_{NLR}, \tau_{AR}, \tau_{SR})$, where the three paradigms are executed sequentially, allowing each paradigm to reference the results of preceding ones. The training process consists of two stages: constructing the MPM dataset and Progressive Paradigm Training (PPT).

### Key Designs

1. **Multi-Paradigm Mathematical (MPM) Dataset**:

    - **Function**: Construct training data $<x, NLR, SR, AR, y>$ containing reasoning paths of all three paradigms for each mathematical problem.
    - **Mechanism**: Two-phase process—(a) Reconstruction and expansion: Using Numina-TIR and Lean-Workbook as seeds, GPT-4o is employed to generate reasoning paths for the missing paradigms, followed by human verification; (b) Correction: The SR portion (Lean 4 proofs) is submitted to the Lean prover for verification. Failed proofs are corrected using DeepSeek-Prover-V1.5 with up to 64 iterations. This process yields 82,770 problems and 167,412 multi-paradigm reasoning paths in total.
    - **Design Motivation**: Existing datasets only offer single-paradigm annotations, whereas CoR requires reasoning paths across all three paradigms for the same problem as training signals. Automated verification via the Lean prover ensures the correctness of the SR paths.

2. **Progressive Paradigm Training (PPT)**:

    - **Function**: Introduce additional reasoning paradigms step-by-step across three stages.
    - **Mechanism**:
        - Stage ①: NLR only. Trained on Numina-CoT*, generating the sequence $z = [x]\tau_{NLR}y$.
        - Stage ②: NLR + AR. Trained on Numina-TIR*, generating the sequence $z = [x]\tau_{NLR}\tau_{AR}y$.
        - Stage ③: NLR + AR + SR. Trained on the MPM dataset, generating the sequence $z = [x]\tau_{NLR}\tau_{AR}\tau_{SR}y$.
    - **Design Motivation**: NLR is the most ubiquitous in pre-training and the easiest to learn, serving as the foundation; AR follows (as code corpora are included in pre-training); SR is the most unfamiliar and is introduced last. This progressive introduction avoids the difficulty of learning all three paradigms simultaneously, allowing the model to acquire new paradigms on top of those it has already mastered.

3. **Sequential Multi-Paradigm Sampling (SMPS)**:

    - **Function**: Perform sampling at the paradigm level (rather than the token level) during inference to combine multiple reasoning paths.
    - **Mechanism**: Sample $J$ paths for the first paradigm, and $K$ paths for each subsequent paradigm, yielding a total of $J \times K$ candidate answers, resolved by majority voting (e.g., $128 \times 128 = 16,384$ paths).
    - **Design Motivation**: Traditional tree search samples within a single paradigm, whereas CoR samples across different paradigms to explore a larger solution space at a lower cost. Diversity across paradigms is substantially higher than token-level diversity within a single paradigm.

4. **Variable Reasoning Depth**:

    - **Function**: Dynamically adjust the paradigm combinations based on the task type.
    - **Mechanism**: Utilizing NLR $\rightarrow$ SR for theorem proving (where Lean 4 proofs can directly extract the answer) and NLR $\rightarrow$ SR $\rightarrow$ AR for arithmetic calculations (relying on code for precise final computation). This execution path is controlled via prompting.
    - **Design Motivation**: Different tasks require different paradigms; flexible combinations strike a balance between efficiency and accuracy.

### Loss & Training
Standard autoregressive loss $\mathcal{L} = -\sum_t \log \mathbb{P}_\theta(z_t|z_{<t})$. The base model used is DeepSeekMath-Base 7B.

## Key Experimental Results

### Main Results
Zero-shot evaluation across 5 mathematical benchmarks.

| Model | MATH | GSM8K | AMC2023 | AIME2024 | miniF2F |
|------|------|-------|---------|----------|---------|
| GPT-4o | 76.6 | 90.5 | 24/40 | 3/30 | 25.0 |
| DeepSeekMath-7B-Base | 11.8 | 22.2 | 3/40 | 0/30 | 28.3 |
| InternLM2-Math-Plus-7B | 53.0 | 85.8 | 15/40 | 1/30 | 43.3 |
| **CoR-Math-7B** | **66.7** | **88.7** | **34/40** | **12/30** | **52.9** (Pass@128) |
| **CoR-Math-7B** (Large Budget) | - | - | - | - | **66.0** (Pass@16384) |

### Ablation Study

| Configuration | MATH | miniF2F | Description |
|------|------|---------|------|
| NLR Only | Baseline | Low | Single Paradigm |
| NLR + AR | Medium | Medium | Two Paradigms |
| NLR + AR + SR (Full CoR) | **Highest** | **Highest** | Three-Paradigm Collaboration |
| Without PPT (Direct training on three paradigms) | Decline | Decline | Validating the necessity of progressive training |

### Key Findings
- **Zero-Shot Outperforming Few-Shot SOTA**: CoR-Math-7B under zero-shot settings performs better on miniF2F than all few-shot baselines, demonstrating that multi-paradigm collaboration brings genuine generalization capabilities.
- **Breakthrough in Theorem Proving**: The 7B model outperforms GPT-4o by 41 percentage points on miniF2F, a feat previously considered impossible.
- **Inter-Paradigm Synergies**: NLR's semantic description assists the formalization of SR, and AR's precise computation validates NLR's reasoning, leading to mutual enhancement among all three.
- **Higher Resource Efficiency**: Through paradigm-level sampling, SMPS explores a more diverse solution space with fewer total attempt budgets, making it more efficient than traditional tree search.
- **PPT is Indispensable**: Direct training on three paradigms yielded poor results, indicating that the progressive introduction from easy to difficult helps the model learn cross-paradigm collaboration more effectively.

## Highlights & Insights
- **Paradigm-Level Reasoning Chain as a Core Innovation**: Instead of simply using code to assist NLR (e.g., TIR), it enables three paradigms to reason independently and reference each other. Incorporating the SR paradigm empowers even a 7B model to conduct formal theorem proving.
- **Elegant Progressive Paradigm Training Strategy**: The sequence of introducing paradigms from familiar to unfamiliar aligns with pedagogical principles. Moreover, the dataset for each stage is independently designed, yielding robust training outcomes.
- **Novelty of SMPS Paradigm-Level Sampling**: Exploring diversity across the paradigm dimension rather than the token dimension provides an inspiring approach that can generalize to other multi-strategy reasoning systems.

## Limitations & Future Work
- **Reliance on Lean Prover Verification**: The SR part requires a Lean 4 environment, which increases the complexity of the training pipeline.
- **High Data Construction Costs**: The MPM dataset requires a combination of GPT-4o synthesis, Lean verification, and human audit, making it difficult to scale.
- **Limited Validation to Mathematics**: Whether the CoR framework is applicable to other tasks requiring multiple reasoning modalities (e.g., scientific or legal reasoning) remains to be explored.
- **High Inference Overhead**: Sequential reasoning across three paradigms coupled with SMPS sampling implies that inference costs are several times higher than those of single-paradigm methods.

## Related Work & Insights
- **vs ToRA/Numina-TIR**: These approaches embed code calls within NLR as tool assistants, which fundamentally remain single-paradigm. CoR allows each paradigm to reason independently and mutually reinforce each other.
- **vs DeepSeek-Prover**: Expert models focusing strictly on SR theorem proving utilize large-scale tree searches. CoR achieves superior performance with a smaller sampling budget because the semantic priors provided by NLR guide the SR search.
- **vs Qwen2.5-Math**: This model is a SOTA expert in arithmetic calculation but cannot perform theorem proving. CoR-Math-7B serves as a genuine mathematical generalist.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The unified multi-paradigm reasoning framework is a major breakthrough in mathematical reasoning, featuring elegantly designed PPT and SMPS.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluations across 5 benchmarks against numerous baselines (including GPT-4o and o1-mini), with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, and the formulas/tables are informative, though the extensive notation requires careful reading.
- Value: ⭐⭐⭐⭐⭐ A unified mathematical reasoning framework, open-source models, and backed by Microsoft—significant impact on the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](safe_math_reasoning.md)
- [\[ACL 2025\] ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations](clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)
- [\[NeurIPS 2025\] Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](../../NeurIPS2025/llm_reasoning/visual_thoughts_a_unified_perspective_of_understanding_multi.md)
- [\[ACL 2025\] Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)
- [\[ACL 2025\] Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering](glore_long_cot_representation.md)

</div>

<!-- RELATED:END -->
