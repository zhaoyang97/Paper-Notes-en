---
title: >-
  [Paper Note] DeAL: Decoding-time Alignment for Large Language Models
description: >-
  [ACL 2025][LLM (Other)][decoding-time alignment] DeAL reformulates the LLM alignment problem as a heuristic search problem during decoding. It utilizes customizable reward functions (including programmatic constraints and parameterized reward models) to guide token selection during the inference phase, achieving flexible multi-objective alignment that can complement and stack with RLHF.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "decoding-time alignment"
  - "reward model"
  - "A* search"
  - "harmlessness-helpfulness"
  - "jailbreak defense"
date: 2026-05-08
content_hash: c19fb2d0542b5ab1
---

# DeAL: Decoding-time Alignment for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2402.06147](https://arxiv.org/abs/2402.06147)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: decoding-time alignment, reward model, A* search, harmlessness-helpfulness, jailbreak defense

## TL;DR
DeAL reformulates the LLM alignment problem as a heuristic search problem during decoding. It utilizes customizable reward functions (including programmatic constraints and parameterized reward models) to guide token selection during the inference phase, achieving flexible multi-objective alignment that can complement and stack with RLHF.

## Background & Motivation
**Background**: Current LLM alignment primarily relies on training-time methods (e.g., RLHF, DPO), which embed alignment goals by fine-tuning model parameters using human preference data.

**Limitations of Prior Work**:
   - Alignment objectives are neither static nor universal—different users and scenarios require different alignment standards, but RLHF "burns" a single alignment perspective into the model.
   - Customized alignment requires re-fine-tuning and maintaining multiple models, which is highly costly.
   - The reliability of training-time alignment is questionable—even safety-trained models can still be easily bypassed by jailbreaks.

**Key Challenge**: Training-time alignment is a one-time, static embedding process, which cannot adapt to dynamically changing alignment requirements, nor can it provide hard guarantees during inference.

**Goal**: How to flexibly and reliably impose customizable alignment constraints during the decoding phase.

**Key Insight**: Treat text generation as a heuristic search problem, where alignment objectives serve as search heuristic functions.

**Core Idea**: Shift alignment from training time to decoding time, utilizing A* search and reward models to heuristically guide token selection at each step.

## Method

### Overall Architecture
DeAL defines alignment as a search problem $\langle S, V, T, R_a \rangle$, where the state space $S$ is the token sequence, the action set $V$ is the vocabulary, the transition function $T$ is the autoregressive appending of tokens, and $R_a$ is the alignment reward function. Based on the LLM's top-k candidate tokens, a lookahead mechanism, and heuristic scoring, the search agent selects the optimal token at each step until the EOS token is generated.

The input prompt consists of three parts: the task instruction $p_t$, the alignment instruction $p_a$ (optional), and the task input $p_i$. The alignment instruction can express public alignment objectives in natural language, serving as the "start-state adaptation" of the search.

### Key Designs

1. **Start-state Adaptation**:

    - **Function**: Modify the input via the alignment prompt $p_a$ to improve the initial search direction of generation.
    - **Mechanism**: A good $p_a$ is equivalent to a favorable search starting point, reducing the difficulty of finding a final state that satisfies the alignment goal; it is manually designed as a hyperparameter in the experiments.
    - **Design Motivation**: Leverage the instruction-following capability of instruction-tuned models to "softly embed" alignment requirements into the search space via prompt.

2. **Action Selection**:

    - **Function**: Select the optimal token from the top-k candidate tokens at each decoding step.
    - **Mechanism**: For each candidate token, look ahead $l$ steps (greedy lookahead) to obtain a more complete sequence, and then score it using a heuristic function $h(\cdot)$. The final selection criterion is: $c(y_t) = \log P(y_{1:t}|p) + \lambda \cdot h(y_{1:t+l}, p)$, where $\lambda$ controls the weight of the alignment objective.
    - **Design Motivation**: Many alignment metrics (e.g., "whether the response is harmful") cannot effectively score partially generated sequences; lookahead provides sufficient context to make the evaluation of $h(\cdot)$ more reliable.

3. **Modular Reward Ensembling**:

    - **Function**: Support the flexible combination of multiple alignment objectives.
    - **Mechanism**: Combine different reward models (e.g., harmless + helpful) through linear weighting $h = w_1 R_1 + w_2 R_2$, allowing users to adjust weights for fine-grained calibration.
    - **Design Motivation**: Different scenarios require different trade-offs between harmlessness and helpfulness (e.g., safety-critical scenarios favor harmlessness); this modular design avoids training specialized models for each combination.
    - **Difference from RLHF**: RLHF optimizes the policy using reward models during training, where the trade-off is fixed; DeAL guides search using reward models during decoding, allowing the trade-off to be dynamically adjusted at runtime.

4. **Supported Heuristic Types**:

    - Programmatic constraints: Programmatically verifiable constraints such as keyword coverage and length limits, where $h(\cdot)$ directly checks if the constraint is satisfied.
    - Parameterized rewards: Use trained reward models (e.g., OPT-125M fine-tuned on HH-RLHF) as $h(\cdot)$ to evaluate abstract alignment objectives.

### Loss & Training
- DeAL itself does not require training—it is a pure inference-time framework.
- The required reward models can be trained independently: this paper fine-tunes OPT-125M on the HH-RLHF dataset to obtain three reward models: $R_{harmless}$, $R_{helpful}$, and $R_{hh}$.
- Can be stacked with RLHF: first fine-tune the model with RLHF, then apply DeAL during decoding.

## Key Experimental Results

### Main Results

**Keyword-constrained Generation (CommonGen)**:

| Model | Method | Soft Coverage | Hard Coverage |
|------|------|:------:|:------:|
| Falcon-7B-Instruct | $p_a$ only | 0.88 | 0.62 |
| Falcon-7B-Instruct | $p_a$ + DeAL | **0.94** | **0.80** (+18%) |
| MPT-7B-Instruct | $p_a$ only | 0.91 | 0.71 |
| MPT-7B-Instruct | $p_a$ + DeAL | **0.96** | **0.85** (+14%) |
| Dolly-v2-3B | $p_a$ only | 0.65 | 0.30 |
| Dolly-v2-3B | $p_a$ + DeAL | **0.79** | **0.51** (+21%) |

**Alignment Objectives (Harmlessness + Helpfulness)**:

| Method | HarmfulQ Harmless | HH-RLHF Harmless | HH-RLHF Helpful |
|------|:------:|:------:|:------:|
| Base (No alignment) | 0.43 | 0.40 | 0.33 |
| Safety prompt | 0.63 | 0.43 | 0.60 |
| Harmless rerank | 0.40 | 0.47 | 0.53 |
| DeAL w/ $R_{harmless}$ | **1.00** | 0.57 | 0.23 |
| DeAL w/ $R_{hh}$ | **1.00** | **0.67** | **0.67** |

### Ablation Study

**Multi-objective Weight Calibration ($(w_{harmless}, w_{helpful})$)**:

| Weight Configuration | HarmfulQ Harmless | HH-RLHF Helpful |
|---------|:------:|:------:|
| (1.0, 0) | 1.00 | 0.23 |
| (0.75, 0.25) | 1.00 | 0.34 |
| (0.50, 0.50) | 0.77 | 0.48 |
| (0.25, 0.50) | 0.43 | 0.67 |
| (0, 1.0) | 0.20 | 0.77 |

**Combination with RLHF**:

| Method | HarmfulQ Harmless | HH-RLHF Helpful |
|------|:------:|:------:|
| No RLHF, No DeAL | 0.33 | 0.43 |
| RLHF w/ $R_{hh}$ | 0.80 | 0.70 |
| DeAL w/ $R_{hh}$ | 0.83 | 0.53 |
| RLHF + DeAL | **0.93** | **0.70** |

### Key Findings
- DeAL provides a larger gain for weaker instruction-following models (Dolly-v2-3B hard coverage +21% vs. MPT +14%), indicating that decoding-time alignment is more valuable for models with weaker capabilities.
- $R_{hh}$ (joint reward) performs better than $R_{harmless}$ or $R_{helpful}$ alone, achieving a better trade-off between harmlessness and helpfulness.
- The combination of RLHF + DeAL yields the best results—training-time and decoding-time alignments are complementary.
- In the face of continuation attacks (jailbreak), the harmlessness rate of the safety prompt is only 20%, whereas DeAL reaches 73%.

## Highlights & Insights
- **Reconceptualizing alignment as a search problem** unifies techniques like safety prompts, reranking, and constrained decoding under a single framework, demonstrating that they are essentially different special cases of search strategies. This shift in perspective is highly elegant.
- **Modular reward ensembling** allows runtime adjustments of alignment preferences. This is highly practical for scenarios requiring personalized alignment (e.g., across different cultures or corporate policies), avoiding the need to train separate models for each alignment configuration.
- **Anti-jailbreak capability**: Because alignment checks are performed at each token generation step, it is much harder to bypass than prompt-based defenses—offering high value for safety-critical application scenarios.

## Limitations & Future Work
- **Severe inference latency**: A 22-55x slowdown (top-k lookahead + parameterized reward model) limits practical deployment. Potential solutions include reward model distillation, combination with speculative decoding, or pre-compiled grammar acceleration.
- **Requires logit access**: It cannot be applied to black-box APIs (e.g., GPT-4), which limits its scope of application.
- **Reward model quality ceiling**: The alignment effectiveness of DeAL is capped by the quality of $h(\cdot)$; an OPT-125M magnitude reward model may not accurately capture complex alignment objectives.
- **Small experimental scale**: It is primarily validated on 3B-7B models, and its effectiveness has not been tested on larger models (70B+).
- It can be combined with speculative decoding to lower lookahead costs, using a small model to quickly generate candidates and then validating them with a large model + reward model.

## Related Work & Insights
- **vs. RLHF/DPO**: RLHF embeds alignment during training, which is static and unadjustable; DeAL imposes alignment during decoding, which is dynamic and customizable, and the two can be stacked.
- **vs. Reward-Augmented Decoding (RAD)**: RAD only considers a single parameterized reward, whereas DeAL supports modular combinations of multiple types (programmatic + parameterized).
- **vs. Constrained Decoding (NeuroLogic, FUDGE)**: These works are special cases of DeAL—they only consider specific types of constraints without a unified framework.
- This paper provides a systematic conceptual framework for decoding-time alignment and can serve as a baseline for research on inference-time alignment.

## Rating
- Novelty: ⭐⭐⭐⭐ It systematically unifies multiple existing techniques, though the core search idea is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ It covers both programmatic constraints and abstract alignment, including jailbreak defense, but the model scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear logic, intuitive framework diagrams, and systematized experimental design.
- Value: ⭐⭐⭐⭐ Decoding-time alignment is an important direction; the framework holds practical deployment value, though latency issues limit its real-world implementation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Nudging: Inference-time Alignment of LLMs via Guided Decoding](nudging_inference_time_alignment.md)
- [\[ACL 2025\] From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment](from_neurons_to_semantics_evaluating_cross-linguistic_alignment_capabilities_of_.md)
- [\[ACL 2025\] Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models](gapo_multi_objective_alignment.md)
- [\[ACL 2025\] Binary Classifier Optimization for Large Language Model Alignment](bco_binary_classifier_alignment.md)
- [\[ACL 2025\] Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models](direct_confidence_alignment_aligning_verbalized_confidence_with_internal_confide.md)

</div>

<!-- RELATED:END -->
