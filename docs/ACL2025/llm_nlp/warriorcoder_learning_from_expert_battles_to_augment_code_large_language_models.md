---
title: >-
  [Paper Note] WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models
description: >-
  [ACL 2025][LLM (Other)][code LLM] WarriorCoder is proposed, which builds an arena among multiple expert code LLMs. An attacker challenges a defender with instructions from its own domain of expertise. A judge evaluates the responses, and the target model is trained on the winning answers. This generates high-quality and highly diverse code training data from scratch without relying on proprietary models or pre-existing datasets, achieving state-of-the-art (SOTA) performance.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "code LLM"
  - "data flywheel"
  - "expert battles"
  - "Elo rating"
  - "instruction mining"
date: 2026-05-08
content_hash: 7abf4929e5e9e2ad
---

# WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2412.17395](https://arxiv.org/abs/2412.17395)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: code LLM, data flywheel, expert battles, Elo rating, instruction mining

## TL;DR
WarriorCoder is proposed, which builds an arena among multiple expert code LLMs. An attacker challenges a defender with instructions from its own domain of expertise. A judge evaluates the responses, and the target model is trained on the winning answers. This generates high-quality and highly diverse code training data from scratch without relying on proprietary models or pre-existing datasets, achieving state-of-the-art (SOTA) performance.

## Background & Motivation

**Background**: The performance of code LLMs heavily relies on high-quality fine-tuning data, yet data collection and annotation are costly.

**Limitations of Prior Work**: Existing data flywheel methods (such as WizardCoder, Magicoder, WaveCoder) rely on pre-existing datasets and a limited selection of proprietary LLMs (such as GPT-4) for data augmentation, leading to limited data diversity and introducing systematic biases.

**Key Challenge**: Although open-source expert code LLMs possess strong capabilities, their training data is not publicly available, preventing direct utilization. Conversely, scaling data with a few proprietary LLMs limits the diversity of data sources.

**Goal**: To design a data generation paradigm that does not rely on proprietary LLMs or pre-existing datasets, enabling automatic knowledge extraction from multiple expert LLMs and integrating their respective strengths.

**Key Insight**: Drawing inspiration from combat arenas (e.g., LMSYS Chatbot Arena), expert LLMs are set to challenge each other, and the target model learns from the winners.

**Core Idea**: Build an arena of code experts where models reveal their respective strengths through battles, allowing the target model to synthesize the advantages of all experts by learning from the winning responses of each match.

## Method

### Overall Architecture
1. Select 5 top open-source code experts (Athene-V2-Chat, DeepSeek-Coder-V2-Lite-Instruct, Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct, QwQ-32B-Preview);
2. In each round of the arena, one model serves as the attacker, another as the defender, and the remaining models act as judges;
3. The attacker uses a completion-based method to mine instructions it excels at, which are then used to challenge the defender;
4. Both sides generate responses, and the judges vote to evaluate them;
5. Combine local voting ratios and global Elo Ratings to select the winning response, which is added to the training set;
6. The target model (DeepSeekCoder-Base-6.7B) undergoes SFT on the collected data.

### Key Designs

1. **Completion-based Instruction Mining**

    - Inspired by Magpie, the chat template prefix (system prompt + user tag) is fed into the expert LLM, prompting it to auto-complete and generate user instructions.
    - These instructions are sampled directly from the model's own distribution, avoiding pattern overfitting and output distribution shifts.
    - Nine different generation configurations (combinations of temperature × top-p) are used to enhance diversity.

2. **Instruction Quality Control**

    - De-duplication: Eliminate duplicate instructions.
    - Difficulty filtering: Judges classify instructions into 4 levels (Excellent 9-10, Good 6-8, Average 3-5, Poor 1-2), keeping only the Good and Excellent levels.
    - Embedding compression: The KCenterGreedy algorithm is employed to select the final instructions based on all-roberta-large-v1 embeddings, ensuring diversity and representativeness.

3. **Win-Loss Decision Mechanism**

    - **Local scoring**: Based on the judge voting ratio $x_{A>B}^i = t_A/(t_A+t_B)$.
    - **Global scoring**: Introduce an Elo Rating system to dynamically track the global relative strength of each model.
    - **Final score**: $e_A^i = \sum_{B \in Com \setminus A} \alpha X_{A>B}^{Elo} + (1-\alpha) x_{A>B}^i$, where $\alpha=0.7$ balances local contingency and global consistency.
    - Elo Rating prevents weaker models from accidentally winning on certain instructions due to random factors.

### Loss & Training

- The response with the highest final score is selected as the gold output and used for standard SFT.
- Training hardware: 8 × NVIDIA A800 80G GPUs.
- Global batch size of 512, with 448 total training steps.
- Learning rate of 1e-5, and weight decay of 3e-7.
- WarmupLR scheduler with a warmup ratio of 0.2.
- The number of arena battles is set to 70,000 rounds, with Elo system K=40.

## Key Experimental Results

### Main Results

| Benchmark | WarriorCoder (6.7B) | Best Baseline | Gain |
|------|-------------------|---------|------|
| HumanEval | 80.5% | MagicoderS-DS 76.8% | +3.7 |
| HumanEval+ | 75.6% | MagicoderS-DS 70.7% | +4.9 |
| MBPP | 76.2% | Magicoder-DS 75.4% | +0.8 |
| MBPP+ | 64.8% | MagicoderS-DS 64.4% | +0.4 |

- Compared to the same backbone (DeepSeekCoder-Base-6.7B), performance on HumanEval increases from 47.6% to 80.5% (+32.9).
- Outperforms GPT-3.5-Turbo in pass@5 on CRUXEval code reasoning (66.5% vs 63.2%).
- Outperforms all baselines on SciPy, Sklearn, and TensorFlow within the DS-1000 library usage benchmark.
- **Most importantly**: Does not rely on any proprietary LLMs, operating entirely on open-source models.

### Ablation Study

| Number of Experts | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---------|-----------|------------|------|-------|
| 1 | 75.4 | 72.6 | 73.3 | 62.4 |
| 2 | 77.2 | 73.3 | 74.5 | 62.9 |
| 5 | 80.5 | 75.6 | 76.2 | 64.8 |

- A larger number of experts yields better performance, demonstrating the effectiveness of multi-expert knowledge fusion.

### Key Findings

1. **Data Independence**: The ROUGE overlap rate between the mined instructions and existing training datasets (WizardCoder, Magicoder, etc.) is mostly below 0.3, with none exceeding 0.6, indicating that the data is newly sampled from the internal distributions of the models.
2. **Task Diversity**: The training data covers 7 categories of tasks, including code generation (51.4%), debugging (12.2%), theoretical explanation (22.2%), and optimization (3.8%).
3. **Expert Win-Rate Matrix**: No single expert dominates across all tasks, which indicates that the multi-expert arena effectively exploits the diverse strengths of different models.
4. **Difficulty Distribution**: Most instructions fall into the Good level, while fewer are classified as Excellent, reflecting the internal knowledge distribution of code experts.

## Highlights & Insights

1. **Paradigm Innovation**: Shifts from "expanding existing datasets" to "generating data from scratch through expert battles", completely eliminating the dependency on proprietary LLMs and seed datasets.
2. **Completion-based Instruction Mining**: Ingeniously utilizes the completion capabilities of the LLM itself to mine its mastered knowledge, which is more natural than traditional prompt-based generation.
3. **Global + Local Scoring**: The integration of the Elo Rating system with voting mechanisms effectively mitigates the randomness and bias inherent in pure voting.
4. **Scalability**: The framework can easily incorporate more expert LLMs; having more experts leads to higher data quality.

## Limitations & Future Work

1. The battle process becomes time-consuming when there are many experts, necessitating exploration into more efficient tournament modes.
2. Currently validated only on coding tasks, without expansion to other complex domains (e.g., mathematical reasoning).
3. The judge models (the remaining expert LLMs) may still introduce biases inherent in the LLM-as-a-judge approach (position bias, verbosity bias, etc.).
4. The effects of utilizing larger target models (such as 33B or 70B) have not been evaluated.

## Related Work & Insights

- **LMSYS Chatbot Arena**: The arena concept in this paper is directly inspired by this work, but replaces online human evaluation with automated LLM evaluation.
- **Magpie**: The source of inspiration for the completion-based instruction mining method.
- **Self-Instruct / Evol-Instruct**: Traditional data flywheel methods. This work highlights their limitation of relying on proprietary LLMs.
- **Insight**: This multi-expert battle framework can be extended to other domains (such as mathematics, reasoning, etc.), provided there are multiple expert models with complementary capabilities.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| **Overall Rating** | **4.0** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](toolcoder_code_empowered_tool_learning.md)
- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](interactive_and_expressive_code-augmented_planning_with_large_language_models.md)
- [\[ACL 2025\] OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models](opencoder_the_open_cookbook_for_top-tier_code_large_language_models.md)
- [\[ACL 2025\] Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems](transforming_podcast_preview_generation_from_expert_models_to_llm-based_systems.md)
- [\[ACL 2025\] To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization](to_code_or_not_to_code_adaptive_tool_integration_for_math_language_models_via_ex.md)

</div>

<!-- RELATED:END -->
