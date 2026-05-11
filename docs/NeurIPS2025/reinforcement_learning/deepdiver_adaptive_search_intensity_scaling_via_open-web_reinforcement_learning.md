---
title: >-
  [Paper Note] DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][search intensity scaling] DeepDiver is an RL-driven search-reasoning framework that trains LLMs for information-seeking in real open-web environments…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "search intensity scaling"
  - "information retrieval"
  - "large language models"
  - "open-web question answering"
date: 2026-05-08
content_hash: 0f54d40b17c441ec
---

# DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.24332](https://arxiv.org/abs/2505.24332)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: search intensity scaling, reinforcement learning, information retrieval, large language models, open-web question answering

## TL;DR

DeepDiver is an RL-driven search-reasoning framework that trains LLMs for information-seeking in real open-web environments, giving rise to an emergent behavior termed *Search Intensity Scaling* (SIS)—enabling a 7B model to match DeepSeek-R1 (671B) on knowledge-intensive tasks.

## Background & Motivation

Information seeking is a core cognitive capability involving iterative evidence gathering, reflective reasoning, and resolution of conflicting information. Existing LLMs face the following challenges in this regard:

**Prompting methods impose fixed pipelines**: Methods such as ReAct and IRCoT rely on predefined rules and cannot adapt to dynamic, complex questions.

**SFT methods overfit to training corpora**: Methods such as SELF-RAG internalize reasoning patterns specific to particular corpora, resulting in poor generalization.

**RL methods are evaluated only in "clean" environments**: R1-Searcher, DeepResearcher, and similar methods are trained and evaluated exclusively on structured data such as HotpotQA/Wikipedia, whereas real-world web search is noisy and rife with conflicting information.

**Four information-seeking behaviors are not fully covered**:
   - Evidence collection and supplementation (the primary focus of traditional QA datasets)
   - **Conflict resolution** (handling contradictory information)
   - **Verification and denoising** (cross-checking facts)
   - **Reflection and correction** (re-evaluating reasoning paths)

The latter three behaviors are critical in real-world web search but cannot be elicited in Wikipedia-based environments.

## Method

### Overall Architecture

DeepDiver adopts a two-stage pipeline of **cold-start SFT → RL training**, training LLMs for iterative retrieval-reasoning in a real search engine environment.

### Key Designs

#### 1. WebPuzzle Dataset

An open-web QA benchmark comprising 24K training and 275 test samples, covering both Wikipedia and open-domain queries:

| Data Type | Generation Method | Characteristics |
|---|---|---|
| Cross-Page QA | Facts extracted from webpages to generate "inverse" questions | Requires cross-page reasoning |
| Open Riddle | Entity attributes are obfuscated or generalized | Highly challenging |
| Wiki Riddle | Same as above but sourced from Wikipedia | Supported by structured knowledge |

Difficulty labels are assigned by testing with DeepSeek-R1 four times and categorizing samples as easy/medium/hard based on the number of correct responses, ensuring stable reward signals during RL training. The test set is manually annotated by five domain experts.

#### 2. Cold-Start SFT Initialization

Responses from DeepSeek-R1 are distilled using diverse data:
- 2,000 WebPuzzle samples (across difficulty levels)
- 300 real user queries
- 2,200 general reasoning problems
- 1,000 user queries with retrieved documents

#### 3. GRPO + Iterative RAG

At each iteration, the model alternates between reasoning and searching until an answer is produced. Key design choices:
- **Loss masking**: GRPO loss is computed only over model-generated tokens; retrieved text does not contribute to gradient updates.
- **Bonus search reward**: When all non-search rollouts fail but at least one search rollout succeeds, an additional reward of +1.0 is granted to successful search rollouts.
- **Loose-to-strict reward transition**: A lenient scoring scheme (10-point scale; score ≥6 yields 1.0) is used for the first 80 steps, after which it transitions to strict scoring (3-round evaluation; ≥2/3 positive judgments required).

#### 4. Search Intensity Scaling (SIS)

SIS is DeepDiver's core emergent capability—the model **adaptively increases search frequency and depth** to handle more complex questions. The paper demonstrates through training dynamics analysis that SIS is an emergent phenomenon rather than a product of reward engineering:

- The trigger rate of the bonus search reward drops sharply from 4.5% at steps 0–9 to 0.1% at steps 70–80.
- Growth in the number of search rounds occurs at steps 80–120, by which point the bonus reward is nearly inactive.
- The model proactively leverages external tools to compensate for gaps in internal knowledge without requiring direct incentivization.

### Loss & Training

- GRPO advantage estimation: $A_i = r_i - \text{mean}(r)$ (within-group relative reward)
- Retrieved text masking: only model-generated tokens contribute to gradients
- Training data: 7K samples selected from 24K WebPuzzle (2K for SFT + 5K for RL), balanced across difficulty levels
- Backbone models: Qwen2.5-7B-Instruct and Pangu-7B-Reasoner

## Key Experimental Results

### Main Results: Comparison with Baselines

| Method | WebPuzzle | C-SimpleQA-500 | FRAMES-zh | BamBoogle-zh |
|---|---|---|---|---|
| Qwen2.5-7B (no search) | 7.4 | 28.4 | 14.1 | 19.7 |
| Qwen2.5-7B (iterative RAG) | 17.0 (2.24 rounds) | 65.3 | 30.9 | 40.8 |
| Cold-Start-SFT | 27.9 (1.85 rounds) | 75.5 | 35.1 | 48.4 |
| R1-Distill | 29.8 (1.75 rounds) | 78.7 | 40.1 | 52.6 |
| **DeepDiver-Qwen7B** | **37.6 (2.51 rounds)** | **81.9** | **44.5** | **63.4** |
| DeepSeek-R1 (iterative RAG) | 37.1 (1.48 rounds) | 84.8 | 65.8 | 79.3 |

The 7B DeepDiver **surpasses the 671B DeepSeek-R1** on the WebPuzzle open-domain task (37.6 vs. 37.1).

### Comparison with Wiki-Based Methods (English Evaluation)

| Method | WebPuzzle-en | BamBoogle | FRAMES | HotpotQA |
|---|---|---|---|---|
| R1-Searcher | 13.7 (1.9 rounds) | 46.7 | 25.3 | 57.9 |
| DeepResearcher | 15.0 (7.5 rounds) | 53.9 | 33.6 | 56.6 |
| **DeepDiver-Qwen** | **26.1 (14.7 rounds)** | **56.8** | 32.0 | **58.4** |

Despite being trained exclusively on Chinese data, DeepDiver substantially outperforms Wiki-based methods on English open-domain tasks.

### Isolated Evaluation of Information-Seeking Capability

After removing questions answerable from internal knowledge alone:
- DeepDiver **outperforms DeepSeek-R1 across all domains**, leading by 5.1 points on WebPuzzle.
- The performance gap on the full test set is attributable primarily to limited parametric knowledge rather than inferior information-seeking capability.

### Ablation Study: Reward Function Design

| Strategy | WebPuzzle Change | FRAMES-zh Change |
|---|---|---|
| Continuous lenient reward | Negligible improvement | −7 points |
| Loose-to-strict transition | **+9 points** (29.1→37.6) | Consistent improvement |

### Key Findings

1. **Search intensity is positively correlated with performance**: Increasing search rounds accompanies rising training rewards; SIS enables the model to dynamically adjust search depth.
2. **Open-web training enhances generalization**: Models trained on WebPuzzle also perform well on Wiki-based benchmarks.
3. **SIS is an emergent behavior**: It is not a product of reward engineering; the bonus search reward serves only as a brief early-stage scaffold.
4. **Generalization from closed to open domains**: On the ProxyQA long-form writing task, DeepDiver outperforms the R1-distilled model by 9.47 points.

## Highlights & Insights

1. **Precise problem formulation**: Information-seeking behaviors are decomposed into four categories (evidence collection, conflict resolution, verification and denoising, reflection and correction), and the limitations of Wikipedia-based environments are rigorously argued.
2. **Rigorous verification of SIS emergence**: By tracking the decay in bonus reward trigger frequency, the paper convincingly demonstrates that SIS is emergent rather than engineered.
3. **Practical insight from the loose-to-strict reward transition**: Using a lenient reward to stabilize early training and a strict reward to overcome later bottlenecks offers broadly applicable guidance for RL training pipelines.
4. **Cross-lingual generalization**: Strong performance on English benchmarks despite Chinese-only training suggests that information-seeking capability is language-agnostic.

## Limitations & Future Work

1. **Limited parametric knowledge in the 7B model**: Full-set performance is constrained by the knowledge capacity of the model size; larger models may yield further gains.
2. **Search engine dependency**: Performance is bounded by the quality and availability of the underlying search engine.
3. **High computational cost**: DeepDiver's search rounds (averaging 2.5+ rounds with up to 15 queries per round) far exceed those of baselines, significantly increasing inference cost.
4. **Reliance on LLM-based evaluation**: Although a dual loose/strict evaluation scheme is adopted, LLM-as-judge inherently introduces bias.

## Related Work & Insights

- **Technical connection to DeepSeek-R1**: Both employ GRPO, but DeepDiver extends it to iterative RAG settings, demonstrating the effectiveness of GRPO for tool-use training.
- **Key distinction from R1-Searcher/DeepResearcher**: Training in a real open-web environment rather than a Wikipedia-based one fosters substantially stronger information-seeking capability.
- **Implications of SIS**: Analogous to test-time compute scaling, but operating along the search dimension—harder questions elicit more search, while simpler questions are answered quickly.

## Rating

- Novelty: ⭐⭐⭐⭐ (SIS is a novel concept; WebPuzzle fills a dataset gap)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (isolated tests, ablations, cross-lingual and cross-domain generalization analyses are exceptionally thorough)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, rigorous analytical logic)
- Value: ⭐⭐⭐⭐⭐ (offers important guidance for RL training of LLM+search systems)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](reinforcement_learning_teachers_of_test_time_scaling.md)
- [\[NeurIPS 2025\] Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)
- [\[NeurIPS 2025\] TensorRL-QAS: Reinforcement Learning with Tensor Networks for Improved Quantum Architecture Search](tensorrl-qas_reinforcement_learning_with_tensor_networks_for_improved_quantum_ar.md)
- [\[NeurIPS 2025\] SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](swe-rl_advancing_llm_reasoning_via_reinforcement_learning_on_open_software_evolu.md)
- [\[NeurIPS 2025\] Reinforcement Learning for Long-Horizon Multi-Turn Search Agents](reinforcement_learning_for_long-horizon_multi-turn_search_agents.md)

</div>

<!-- RELATED:END -->
