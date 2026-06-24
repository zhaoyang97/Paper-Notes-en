---
title: >-
  [Paper Note] Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents
description: >-
  [ACL 2025][Multimodal VLM][reward modeling] This paper proposes Agent-RewardBench, the first benchmark to evaluate the capability of multimodal LLMs as agent reward models. It covers three dimensions (perception, planning, and safety) across seven real-world scenarios, containing 1,136 high-quality step-level samples. Experiments reveal that even the strongest model, GPT-4o, achieves only 61.4% accuracy, and stronger models surprisingly perform worse in the safety dimension.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "reward modeling"
  - "multimodal agent"
  - "benchmark"
  - "safety"
  - "planning"
date: 2026-05-08
content_hash: 659e63704be12c2d
---

# Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents

**Conference**: ACL 2025  
**arXiv**: [2506.21252](https://arxiv.org/abs/2506.21252)  
**Code**: Yes (GitHub)  
**Area**: Multimodal VLM  
**Keywords**: reward modeling, multimodal agent, benchmark, safety, planning

## TL;DR
This paper proposes Agent-RewardBench, the first benchmark to evaluate the capability of multimodal LLMs as agent reward models. It covers three dimensions (perception, planning, and safety) across seven real-world scenarios, containing 1,136 high-quality step-level samples. Experiments reveal that even the strongest model, GPT-4o, achieves only 61.4% accuracy, and stronger models surprisingly perform worse in the safety dimension.

## Background & Motivation
1. **Background**: Multimodal agents show great potential in tasks like web navigation and embodied AI, with supervised fine-tuning (SFT) using expert-labeled trajectories being the mainstream enhancement method. Recent research has begun exploring the use of reward models (RMs) to provide feedback for improving agent capabilities.
2. **Limitations of Prior Work**: Although reward models are crucial for guiding agent training and searching, there is currently no benchmark to evaluate MLLMs as agent reward models. Existing reward benchmarks (such as RewardBench) focus on dialogue, mathematics, and retrieval scenarios, failing to cover agent-specific capabilities in perception, planning, and safety.
3. **Key Challenge**: Agent tasks require multi-dimensional reward feedback (e.g., whether perception is accurate, whether planning is rational, and whether actions are safe). However, it remains unclear which MLLMs are suitable as agent reward models, and how their capabilities differ across these dimensions.
4. **Goal**: (1) To build an agent reward benchmark covering multiple dimensions and scenarios; (2) to support step-level reward evaluation rather than just final outcome assessment; (3) to guarantee data quality through difficulty control and human verification.
5. **Key Insight**: Starting from three core capability dimensions of agent tasks (perception, planning, and safety), data from seven real-world scenarios are collected. The benchmark is constructed using a three-stage pipeline: response sampling from ten models, difficulty filtering using small models, and human verification.
6. **Core Idea**: To systematically evaluate the reward modeling capabilities of MLLMs in agent tasks for the first time, revealing the counter-intuitive finding that "stronger models do not necessarily make better reward models."

## Method

### Overall Architecture
Given task prompts from seven real-world agent scenarios as input, multiple responses are sampled from ten different MLLMs for each intermediate step to construct positive-negative peer pairs $(r^+, r^-)$. After difficulty filtering via small models and human validation, 1,136 high-quality evaluation samples are obtained. During evaluation, the target MLLM is asked to determine which response is better.

### Key Designs

1. **Data Source Design for Three Dimensions and Seven Scenarios**:

    - **Function**: Comprehensively covers the core capabilities required for agent tasks.
    - **Mechanism**: For the perception dimension, data is sourced from SeeClick (visual grounding on web/mobile/desktop) and MFE-ETP (embodied spatial perception); the planning dimension selects from Mind2Web (web multi-step planning), PCA (Minecraft/autonomous driving/VirtualHome), and TravelPlanner (travel planning); the safety dimension is selected from pop-up attack scenarios and MSSBench (embodied safety). A total of approximately 1,682 initial samples were gathered.
    - **Design Motivation**: Agent reward models must simultaneously possess visual understanding, sequential decision-making, and safety alignment capabilities. Evaluating a single dimension is insufficient to reflect real-world application demands.

2. **Step-level Reward Evaluation**:

    - **Function**: Performs fine-grained reward evaluation at each intermediate step of a task.
    - **Mechanism**: Responses are sampled for each step of the agent task to construct positive-negative pairs for that step, allowing the evaluated model to judge which step response is better. This provides more detailed feedback than evaluating only the final outcome.
    - **Design Motivation**: Agent planning exhibits distinct step divisions; step-level evaluation helps identify specific weaknesses of models during the planning process.

3. **Three-stage Data Construction Pipeline (Sampling → Difficulty Control → Human Verification)**:

    - **Function**: Ensures high data quality and appropriate difficulty of evaluation data.
    - **Mechanism**: First, responses are sampled from five closed-source and five open-source models, generating ten pairs of positive-negative samples per query. Next, three small models (Pixtral-12B, LLaVA-OneVision-7B, and InternVL2-8B) are utilized for bidirectional difficulty filtering to discard overly simple or complex samples. Finally, three AI-specialized graduate students conduct human verification to eliminate erroneously annotated samples, reducing the initial 1,443 samples to a finalized set of 1,136.
    - **Design Motivation**: Extremely easy data offers low discriminative power, while excessively difficult data may introduce annotation noise. Difficulty control ensures the benchmark maintains high quality while effectively distinguishing between capacities of different models.

## Key Experimental Results

### Main Results

| Model | Perception Avg | Planning Avg | Safety Avg | **Overall Avg** |
|------|--------|--------|--------|-----------|
| gemini-1.5-pro | 73.4 | 69.6 | 37.7 | **61.6** |
| gpt-4o | 65.9 | 73.2 | 39.2 | **61.4** |
| claude-3.5-sonnet | 73.3 | 71.2 | 22.4 | **57.9** |
| Qwen2-VL-72B | 69.1 | 60.1 | 34.3 | **55.3** |
| gemini-1.5-flash | 66.1 | 64.7 | 47.8 | **60.2** |
| Qwen2-VL-7B | 57.5 | 51.8 | 38.7 | 49.7 |
| Llama-3.2-11B | 53.5 | 50.6 | 38.0 | 47.8 |

### Safety Dimension Breakdown

| Model | Web Safety | Embodied Safety | Safety Avg |
|------|--------|---------|--------|
| gemini-1.5-flash | 26.0 | **69.5** | **47.8** |
| gpt-4o | 17.5 | 61.0 | 39.2 |
| claude-3.5-sonnet | 15.0 | 29.9 | 22.4 |
| gpt-4o-mini | **35.0** | 56.7 | 45.9 |

### Key Findings
- Even the strongest closed-source model (gemini-1.5-pro) achieves only 61.6% accuracy on Agent-RewardBench, indicating that agent reward modeling remains a substantial challenge.
- **Stronger models do not equal stronger safety reward models**: GPT-4o ranks near the top overall but scores only 39.2% on the safety dimension; Claude-3.5-Sonnet drops further to 22.4%. Conversely, gpt-4o-mini performs better in Web Safety (35.0% vs. 17.5% for GPT-4o).
- Open-source models (such as Llama-3.2-11B) score near random guess levels in perception (53.5%) and planning (50.6%), showing that specialized agent reward training is indispensable.
- In the planning dimension, GPT-4o (73.2%) significantly outperforms other models, but its planning capability in embodied scenarios (68.2%) is weaker than in travel planning (76.2%), indicating the additional challenge that visual + physical reasoning poses to planning.

## Highlights & Insights
- "Stronger models ≠ better safety reward models" is the most significant counter-intuitive finding of this paper—implying that safety alignment requires specialized training strategies and cannot simply rely on improvements in general model capabilities.
- The step-level evaluation paradigm is highly practical—it can be directly migrated to reward modeling during LLM reasoning processes (such as process reward model evaluation).
- The three-stage data construction pipeline (multi-model sampling → small-model filtering → human verification) provides a generic paradigm for building high-quality benchmarks.

## Limitations & Future Work
- Only discriminative reward modeling (selecting the better response) is evaluated, while generative rewards (providing scores or textual feedback) are not assessed.
- The sample size for the safety dimension is small (only 100 samples for Web Safety and 82 for Embodied Safety), which may affect statistical stability.
- The correlation between Agent-RewardBench scores and actual improvement in agent performance remains unverified (i.e., whether a better reward model genuinely translates to a better agent).
- Data sources lean heavily toward English scenarios, and reward evaluation for multilingual agents is not covered.
- Combining multiple weak reward models to obtain better reward signals (ensemble strategies) has not been explored.
- The update mechanism for the benchmark is not discussed—as models advance, the dataset may require periodic updates to maintain differentiation.

## Related Work & Insights
- **vs RewardBench (Zhou et al., 2024)**: RewardBench evaluates chat, math, and retrieval scenarios, whereas Agent-RewardBench focuses on agent scenarios, introducing perception and safety dimensions, along with step-level evaluations.
- **vs Mind2Web**: Mind2Web evaluates agent planning capabilities. This work converts its data to evaluate reward modeling, shifting the perspective from "agent capability" to "reward model capability."
- **vs Differential Prompting for agents**: This paper exposes the bottleneck of imitation learning, offering evaluation infrastructure for reward-model-guided agent training.
- The findings of Agent-RewardBench in the safety dimension can guide key future directions for research on agent safety alignment.

## Rating
- Overall Evaluation: Pioneering work that provides critical evaluation infrastructure for the agent field from an RL perspective, with particularly important findings in the safety dimension.
- Novelty: ⭐⭐⭐⭐⭐ First benchmark focusing on agent reward modeling, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation covering eight models, though the sample size in the safety dimension is small.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-articulated motivation.
- Value: ⭐⭐⭐⭐⭐ Provides key evaluation tools for the agent field transitioning from SFT to RL.

<!-- Coverage: 3 dimensions (perception/planning/safety), 7 scenarios, 1136 samples, 8 model evaluations -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)
- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](../../NeurIPS2025/multimodal_vlm/wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [\[ACL 2025\] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)
- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](cant_see_the_forest_for_the.md)
- [\[ACL 2025\] Vision-Language Models Struggle to Align Entities across Modalities](vision-language_models_struggle_to_align_entities_across_modalities.md)

</div>

<!-- RELATED:END -->
