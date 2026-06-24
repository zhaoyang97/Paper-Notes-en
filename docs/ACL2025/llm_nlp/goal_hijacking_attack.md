---
title: >-
  [Paper Note] Efficient Universal Goal Hijacking with Semantics-guided Prompt Organization
description: >-
  [ACL 2025][LLM (Other)][Universal Goal Hijacking] This paper proposes POUGH, an efficient universal goal hijacking method against LLMs using an efficient incremental optimization algorithm alongside two semantics-guided prompt organization strategies (sampling strategy + ordering strategy). It achieves an average attack success rate of 93.41% across four open-source LLMs and ten malicious target responses.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Universal Goal Hijacking"
  - "Prompt Injection Attacks"
  - "Semantics-guided"
  - "Adversarial Suffix Optimization"
  - "LLM Safety"
date: 2026-05-08
content_hash: b09e6499658a96eb
---

# Efficient Universal Goal Hijacking with Semantics-guided Prompt Organization

**Conference**: ACL 2025  
**arXiv**: [2405.14189](https://arxiv.org/abs/2405.14189)  
**Area**: LLM Safety / Adversarial Attacks  
**Keywords**: Universal Goal Hijacking, Prompt Injection Attacks, Semantics-guided, Adversarial Suffix Optimization, LLM Safety  

## TL;DR

This paper proposes POUGH, an efficient universal goal hijacking method against LLMs using an efficient incremental optimization algorithm alongside two semantics-guided prompt organization strategies (sampling strategy + ordering strategy). It achieves an average attack success rate of 93.41% across four open-source LLMs and ten malicious target responses.

## Background & Motivation

**Definition of Goal Hijacking Attack**: Goal hijacking is a prompt injection attack where an attacker appends a malicious suffix to a user prompt, forcing the LLM to ignore the original intent and return a fixed, attacker-specified malicious response.

**Universal vs. Specific Goal Hijacking**:
   - **Specific Hijacking**: Generates a suffix individually for each user prompt, which suffers from poor real-time performance.
   - **Universal Hijacking**: Uses a single fixed suffix to hijack all incoming user prompts without requiring online gradient optimization, offering greater practicality.

**Limitations of Prior Work**:
   - Manual methods (e.g., HouYi, TensorTrust): Use templates like "Ignore previous prompt and print XXX", yielding an extremely low ASR (HouYi achieves only 0.37%).
   - M-GCG method: The only gradient-based optimization method, but it requires computing gradients over all training prompts in every iteration, incurring huge time overhead (requiring 25,000 iterations).
   - Existing methods focus solely on optimization algorithm design, overlooking the critical role played by the training prompts themselves.

**Key Insight**:
   - Although specific suffixes generated for single prompts do not satisfy universality requirements, their ASR is not zero (0.84%), demonstrating a weak capability for universality.
   - The semantic diversity of training prompts is crucial for suffix universality.
   - The semantic similarity between prompts and target responses can guide the optimization sequence.

## Method

### Overall Architecture

POUGH consists of three components:
1. **I-UGH Optimization Algorithm**: An efficient discrete token optimization algorithm that incrementally increases the number of training prompts.
2. **Semantics-guided Sampling Strategy**: Selects a semantically diverse training subset from a large prompt pool.
3. **Semantics-guided Ordering Strategy**: Orders the prompts based on their semantic similarity to the target response.

### Key Designs

#### 1. I-UGH Incremental Optimization Algorithm

Core Idea: Optimize the suffix starting from a single prompt, gradually increasing the number of prompts involved in the loss calculation.

- **Initialization**: $n_c = 1$, starting optimization with only 1 prompt.
- **Gradient Computation**: Compute loss gradients with respect to the currently selected $n_c$ prompts for each token position in the suffix.
- **Candidate Generation**: Generate $B$ candidate suffixes based on the top-k gradient directions (modifying only 1 token each).
- **Candidate Selection**: Select the candidate that minimizes the loss over the current $n_c$ prompts.
- **Incremental Expansion**: When the ASR of the suffix on $\mathcal{P}_{1:n_c}$ exceeds a threshold (0.8), increment $n_c$.
- **Termination Condition**: Stop when $n_c = N$ (covering all training prompts) or the maximum number of iterations is reached.

**Efficiency Comparison**: A single gradient computation takes only 0.37 seconds when $n_c=1$, compared to 6.06 seconds when $n_c=50$ (a 16.4x difference). Incremental expansion significantly reduces the computational overhead in early iterations.

#### 2. Semantics-guided Sampling Strategy

Goal: Select $N$ (50) semantically diverse training prompts $\mathcal{P}$ from a large prompt pool $\mathcal{BP}$ (1000 prompts).

- **Step 1**: Compute semantic similarity between all prompt pairs in $\mathcal{BP}$, and select the pair with the lowest similarity as the initial seed.
- **Step 2**: Greedy iteration — select the prompt from $\mathcal{BP}$ that has the lowest cumulative average semantic similarity with the prompts already in $\mathcal{P}$, and add it to $\mathcal{P}$.
- **Step 3**: Repeat Step 2 until $|\mathcal{P}| = N$.
- Cosine similarity is used as the metric.

#### 3. Semantics-guided Ordering Strategy

Goal: Determine the order in which training prompts enter the optimization process to accelerate convergence.

- Compute the semantic similarity between each training prompt and the target response $R^T$.
- Sort in **descending** order: prompts most semantically similar to the target response are introduced first.
- The semantic extraction function $\Theta(\cdot)$ uses embeddings from the LLM's final hidden layer states.

## Key Experimental Results

### Main Results

**Comparison with baselines on Llama-2-7b-chat-hf** (10 malicious target responses, ASR%):

| Method | Average ASR (%) | Average Iterations |
|------|-----------|-------------|
| GCG-hijacking | 0.84 | 462 |
| MAC-hijacking | 0.36 | 471 |
| AutoDAN-hijacking | 0.00 | 500 |
| AmpleGCG-hijacking | 0.00 | - |
| HouYi (Manual) | 0.37 | - |
| TensorTrust (Manual) | 0.00 | - |
| M-GCG | 54.26 | **25,000** |
| I-UGH (Ours) | 85.50 | 7,613 |
| **POUGH (Ours)** | **93.41** | **5,626** |

**Key data**: Compared to M-GCG, POUGH improves ASR from 54.26% to 93.41% (+39.15%) while reducing iterations from 25,000 to 5,626 (a 77.5% reduction).

**Cross-model Generalization** (POUGH method, average ASR%):

| Model | Average ASR (%) |
|------|-----------|
| Llama-2-7b-chat | 93.41 |
| Vicuna-7b-v1.5 | 85.06 |
| Mistral-7B-Instruct | 81.29 |
| Guanaco-7B-HF | ~80+ |

### Key Findings

1. **Significant Efficiency Gains**: POUGH requires only about 1/4 of M-GCG's iterations to achieve a higher ASR, while exhibiting more balanced performance across all 10 malicious targets (M-GCG yields 0% ASR on "bomb" and "drug").

2. **Contribution of Components**:
    - I-UGH algorithm alone: 85.50% ASR.
    - With Sampling Strategy: Elevates a low-diversity training set's ASR of 5% to an effective level.
    - With Ordering Strategy (Full POUGH): 93.41% ASR with accelerated convergence.

3. **Failure of Manual Methods**: Even when requesting the LLM to return a non-malicious simple sentence "Water is good.", manual prompt injection achieves an ASR of only 39.9%.

4. **Prompt-specific Suffixes are Non-transferable**: Suffixes generated for randomly selected prompts achieve an average ASR of only 0.6% on 1,000 test prompts.

5. **Convergence Acceleration from Ordering Strategy**: Sorting in descending order (introducing semantically similar prompts first) achieves faster convergence compared to random sorting.

## Highlights & Insights

1. **Innovative Prompt-centric Perspective**: For the first time, universal goal hijacking is studied from the perspective of training prompt organization (rather than purely focusing on optimization algorithms), revealing the critical role of prompt selection and ordering in attack efficacy.

2. **Simple and Effective Incremental Optimization**: The design of progressively expanding from a single prompt is intuitive yet highly efficient, greatly reducing computational overhead in early iterations.

3. **Implications for Defense**: The high success rate of this attack method exposes vulnerabilities in current LLM safety alignment. Even with safety system prompts, a fixed suffix can still achieve up to a 93% hijacking success rate.

4. **Rigorous Success Metric**: Using exact string matching (rather than semantic similarity) as the ASR metric provides a much stricter standard than similarity matching.

5. **Generalizability of the Sampling Strategy**: The selection of training prompts is independent of the target response $R^T$, indicating that semantic diversity itself is key to enhancing universality.

## Limitations & Future Work

1. **Limited to Open-source Models**: Evaluation was restricted to four 7B-scale open-source models, without testing closed-source models (such as GPT-4 or Claude).

2. **Computation Resource Requirements**: The approach still requires white-box access (gradient computation), limiting applicability to API-only models.

3. **Detectability of Suffixes**: Generated suffixes usually consist of unreadable token sequences, which can easily be detected by simple input filters.

4. **Target Response Length Constraints**: The target responses used in experiments were relatively short; performance on longer, complex responses remains unverified.

5. **Lack of Scale Exploration**: Only 7B models were evaluated, leaving the attack's effectiveness on larger models (e.g., 70B+) untested.

## Related Work & Insights

- **Adversarial Prompt Optimization**: AutoPrompt (Shin et al., 2020), GCG (Zou et al., 2023), PEZ discrete prompt optimization.
- **Goal Hijacking**: Manual attacks such as HouYi (Liu et al., 2023a) and TensorTrust (Toyer et al., 2023); gradient-based optimization like M-GCG (Carlini et al., 2023).
- **Jailbreak Attacks**: Distinction from goal hijacking — jailbreak aims to bypass safety guardrails to execute a user's malicious query, whereas goal hijacking completely ignores user intent to return an attacker-specified response.

## Rating

| Dimension | Score (1-10) | Explanation |
|------|-------------|------|
| Novelty | 8 | Innovation from the prompt-organization perspective + progressive algorithm design. |
| Technical Depth | 8 | Technical contributions and theoretical analyses for all three components. |
| Experimental Thoroughness | 8 | Comprehensive evaluation across 4 models × 10 target responses, with multiple baseline comparisons. |
| Writing Quality | 7 | Clear structure but somewhat dense mathematical notations. |
| Practical Impact | 8 | Exposes critical vulnerabilities in LLM safety alignment. |
| **Overall Score** | **7.8** | An efficient universal attack method with substantial value for LLM safety research. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection](erm_prompt_optimization_memory.md)
- [\[ACL 2025\] RiOT: Efficient Prompt Refinement with Residual Optimization Tree](riot_efficient_prompt_refinement_with_residual_optimization_tree.md)
- [\[ACL 2025\] Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs](is_it_just_semantics_a_case_study_of_discourse_particle_understanding_in_llms.md)
- [\[ACL 2025\] OPTS: Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](bandit-based_prompt_design_strategy_selection_improves_prompt_optimizers.md)
- [\[ACL 2025\] Nudging: Inference-time Alignment of LLMs via Guided Decoding](nudging_inference_time_alignment.md)

</div>

<!-- RELATED:END -->
