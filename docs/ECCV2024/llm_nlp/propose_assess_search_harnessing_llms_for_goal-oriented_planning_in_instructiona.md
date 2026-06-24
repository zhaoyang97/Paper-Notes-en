---
title: >-
  [Paper Note] Propose, Assess, Search: Harnessing LLMs for Goal-Oriented Planning in Instructional Videos
description: >-
  [ECCV 2024][LLM (Other)][Goal-Oriented Planning] VidAssist proposes a three-step framework of "Propose-Assess-Search", leveraging LLMs as a knowledge base and evaluation tool combined with a breadth-first search algorithm. It outperforms fully supervised SOTA in a zero/few-shot manner on goal-oriented planning tasks in instructional videos, achieving a +7.7% SR improvement on COIN compared to the fully supervised VLaMP in the few-shot setting.
tags:
  - "ECCV 2024"
  - "LLM (Other)"
  - "Goal-Oriented Planning"
  - "Instructional Videos"
  - "Large Language Models"
  - "Zero-shot/Few-shot"
  - "Search Algorithms"
date: 2026-05-08
content_hash: b55656b115ee48ba
---

# Propose, Assess, Search: Harnessing LLMs for Goal-Oriented Planning in Instructional Videos

**Conference**: ECCV 2024  
**arXiv**: [2409.20557](https://arxiv.org/abs/2409.20557)  
**Code**: [Project Homepage](https://sites.google.com/view/vidassist)  
**Area**: NLP Understanding / LLM Planning  
**Keywords**: Goal-Oriented Planning, Instructional Videos, Large Language Models, Zero-shot/Few-shot, Search Algorithms

## TL;DR

VidAssist proposes a three-step framework of "Propose-Assess-Search", leveraging LLMs as a knowledge base and evaluation tool combined with a breadth-first search algorithm. It outperforms fully supervised SOTA in a zero/few-shot manner on goal-oriented planning tasks in instructional videos, achieving a +7.7% SR improvement on COIN compared to the fully supervised VLaMP in the few-shot setting.

## Background & Motivation

Goal-Oriented Planning requires models to predict a sequence of action steps based on current observations and a given goal. This task holds significant value for developing intelligent assistants and robotics, encompassing two typical settings:

**VPA (Visual Planning for Assistance)**: Given a video history and a natural language goal, predict the future action sequence.

**PP (Procedural Planning)**: Given initial and goal state images, predict the intermediate steps.

**Limitations of Prior Work**:
- Previous approaches rely heavily on fully supervised training on large volumes of annotated data, leading to severe biases towards the training dataset.
- Under scenarios with small data scale and limited task diversity, they struggle to generalize to new tasks.
- They lack zero/few-shot learning capabilities, restricting deployment in real-world scenarios.

**Problems of Direct LLM Usage as Planners**:
- LLM outputs are free-form text, which is difficult to map to an executable action space.
- Simple autoregressive step-by-step prediction suffers from severe error accumulation in long-horizon planning.
- The lack of a deliberate planning mechanism for procedural tasks means relying solely on token-level generation is insufficient for making optimal decisions.

**Key Insight**: By utilizing LLMs both as a knowledge base (proposing candidate actions) and as an evaluation tool (assessing the quality of action plans), combined with search algorithms to achieve deliberate planning, both VPA and PP tasks are unified and addressed under zero/few-shot settings.

## Method

### Overall Architecture

VidAssist adopts an iterative framework of "Propose → Assess → Search":
1. Visual inputs (videos/images) are first converted into textual descriptions via a Socratic method.
2. At each step, K candidate actions are proposed using the LLM (Propose).
3. A hybrid value function is used to assess each candidate (Assess).
4. Breadth-First Search is employed to find the optimal action sequence (Search), dynamically pruning low-score branches.

### Key Designs

1. **Visual Understanding Module (Socratic Approach)**:

    - **Function**: Translates visual inputs into text descriptions for LLM processing.
    - **Mechanism**:
        - VPA Task: Segment the video into 1-second clips, use VideoCLIP to predict the action category of each clip, and merge consecutive identical actions to obtain a textual sequence of historical actions.
        - PP Task: Predict action descriptions of the initial and goal states from given images using a BLIP-based dual-retrieval model.
    - **Design Motivation**: The Socratic model approach is mature and training-free, allowing direct reuse of pre-trained visual understanding tools.

2. **Propose (Candidate Action Proposal)**:

    - **Function**: Generates K possible next actions at each planning step using the LLM (Llama-2-70B).
    - **Mechanism**: Organizes current observations, targets, and predicted action sequences into a prompt, drawing K samples from the same prompt to capture task uncertainty. Sentence-BERT is then used to map free-form outputs to the most similar executable actions.
    - **Design Motivation**: Sampling multiple times instead of a single greedy decoding covers the inherent uncertainty in procedural tasks.

3. **Assess (Hybrid Value Function Evaluation)**:

    - **Function**: Evaluates the quality of each candidate action using four complementary value functions.
    - **Core Components**:
        - **Text Generation Score $V_G$**: The average log-probability of the LLM generating the description (token-level confidence).
        - **Text Mapping Score $V_M$**: The cosine similarity between the free-form description and executable actions (mapping confidence).
        - **Partial Plan Evaluation $V_P$**: Requests the LLM using an independent prompt to judge "whether the action sequence so far reasonably progresses toward the goal", taking the softmax of YES/NO logits (semantic-level self-evaluation).
        - **Few-Shot Task Graph $V_{TG}$**: Builds a first-order action transition graph from few-shot examples and calculates the transition probability of the current action sequence (used only in few-shot settings).
    - The final evaluation score is a weighted combination: for the VPA task, $V = 0.2V_G + 0.1V_M + 0.1V_{TG} + 0.7V_P$; for the PP task, $V = 0.1V_G + 0.1V_M + 0.3V_{TG} + 0.5V_P$.
    - **Design Motivation**: $V_P$ contributes the most (0.7 weight) because the semantic-level self-evaluation capability of LLMs outperforms simple token-level probabilities.

4. **Search (Breadth-First Search + Dynamic Pruning)**:

    - **Function**: Searches for the optimal T-step action plan in the action space.
    - **Mechanism**: Retains the $\tilde{K}$ action branches with the highest evaluation scores at each step ($\tilde{K} < K$) while pruning lower ones. After reaching T steps, the path of the highest-scoring leaf node is backtracked to obtain the optimal plan.
    - **Design Motivation**: BFS + pruning maintains efficiency in exponential search spaces, while multi-step look-ahead prevents greedy traps.

### Loss & Training

VidAssist is a zero/few-shot framework and **requires no training**. The core lies in prompt design and in-context learning:
- Zero-shot: Direct utilization of prompts to guide LLM prediction.
- Few-shot: Adds a few in-context examples to the prompt (3 for VPA, 10 for PP).

## Key Experimental Results

### Main Results: VPA Task (COIN Dataset)

| Method | Setting | T=1 SR | T=3 SR | T=4 SR |
|------|:---:|:---:|:---:|:---:|
| VLaMP (Fully Supervised) | Fully Supervised | 45.2 | 18.3 | 9.0 |
| LLM Baseline | Zero-shot | 28.5 | 2.4 | 0.7 |
| VidAssist | Zero-shot | 44.5 | 15.3 | 6.1 |
| LLM Baseline | Few-shot | 36.8 | 10.2 | 6.1 |
| **VidAssist** | **Few-shot (3-shot)** | **52.8** | **21.8** | **13.8** |

### Main Results: PP Task (COIN Dataset)

| Method | Setting | T=3 SR | T=4 SR |
|------|:---:|:---:|:---:|
| LFP (Fully Supervised SOTA) | Fully Supervised | 30.64 | 15.97 |
| LLM Baseline | Zero-shot | 13.04 | 4.46 |
| VidAssist | Zero-shot | 18.44 | 9.07 |
| **VidAssist** | **Few-shot (10-shot)** | **29.20** | **20.78** |

### Ablation Study: Value Function Importance

| Value Function Combination | VPA T=3 SR | PP T=4 SR | Description |
|------|:---:|:---:|------|
| $V_G$ only (Generation Score) | 11.60 | 11.33 | Most basic |
| $V_M$ only (Mapping Score) | 10.21 | 11.30 | Comparable to $V_G$ |
| $V_{TG}$ only (Task Graph) | 9.35 | 13.19 | Effective on PP |
| $V_P$ only (Plan Evaluation) | 16.10 | 16.20 | **Strongest individually** |
| $V_G+V_M+V_{TG}$ (w/o $V_P$) | 16.96 | 16.89 | Significant drop without $V_P$ |
| **All four** | **21.08** | **20.78** | **Optimal** |

### Key Findings

- Few-shot VidAssist (3/10 shots) **outperforms fully supervised SOTA**: +7.7% SR on VPA (T=4), +4.81% SR on PP (T=4).
- Under zero-shot, VidAssist yields massive improvements over the LLM Baseline: +12.9% SR on VPA (T=3), highlighting that the search mechanism is critical for long-horizon planning.
- $V_P$ (Partial Plan Evaluation) is the most critical individual value function, though the combination of all four yields the best performance.
- Larger LLM scales yield better performance: from Llama-2-7B to 70B, VPA T=4 increases from 11.71% to 13.80%.
- Using ground-truth visual observations instead of predicted ones leads to substantial performance gains (PP T=3: +36.93%), indicating that visual understanding is currently the main bottleneck.

## Highlights & Insights

- **Dual Role of LLMs**: Serving simultaneously as a knowledge base (proposing candidate actions) and a judge (assessing plan rationality via $V_P$), representing a practical application of the self-evaluation/reflection capabilities of LLMs.
- **search > pure generation**: Experiments demonstrate that the search-and-assess mechanism possesses a clear advantage over pure LLM generation for long-horizon planning, embodying the philosophy that planning is not equivalent to generation.
- **Unified Framework**: Tackles two different task settings, VPA and PP, with the same methodology, showcasing the generality of the framework.
- **Few-shot > Fully Supervised**: Outperforming fully supervised methods with only 3-10 examples demonstrates that the inherent procedural knowledge in LLMs is more valuable than small-scale annotated data.

## Limitations & Future Work

- The search process requires multiple LLM inferences (K samples * T steps + $V_P$ evaluation), incurring high inference costs.
- Visual understanding relies entirely on external models (VideoCLIP, BLIP); visual perception errors serve as the primary bottleneck (as verified by ground-truth experiments).
- Weights for the value functions are manually tuned on the validation set, lacking an adaptive mechanism.
- The pruning strategy of BFS is relatively simple and might bypass certain long-term optimal paths.
- Validation is restricted to procedural tasks; applicability to more open-ended planning scenarios remains unknown.

## Related Work & Insights

- Socratic Models (2022) pioneered zero-shot methods for multimodal reasoning; VidAssist inherits its visual-to-text pipeline.
- Robot planning efforts like SayCan (2022) inspired the design of utilizing LLMs as planners.
- Works such as Tree-of-Thought demonstrate the potential of search-augmented LLM reasoning, which VidAssist adapts to the video planning domain.

## Rating

- Novelty: ⭐⭐⭐⭐ Combines search algorithms with LLM planning; value function design (especially $V_P$ self-evaluation) is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers two tasks, two datasets, three settings (zero-shot/few-shot/fully supervised comparison), and exhaustive ablations.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear; the Propose-Assess-Search terminology is intuitive and easy to follow.
- Value: ⭐⭐⭐⭐ Achieving better-than-fully-supervised SOTA performance with few-shot is of high practical significance, validating the planning potential of LLM's intrinsic knowledge.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LLMs Can Be Easily Confused by Instructional Distractions](../../ACL2025/llm_nlp/llms_can_be_easily_confused_by_instructional_distractions.md)
- [\[ECCV 2024\] Cultural Value Differences of LLMs: Prompt, Language, and Model Size](cultural_value_differences_llms.md)
- [\[NeurIPS 2025\] AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play](../../NeurIPS2025/llm_nlp/acesearcher_bootstrapping_reasoning_and_search_for_llms_via_reinforced_self-play.md)
- [\[ICLR 2026\] LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery](../../ICLR2026/llm_nlp/llema_evolutionary_search_with_llms_for_multi-objective_material_design.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](../../ACL2025/llm_nlp/plangenllms_planning_survey.md)

</div>

<!-- RELATED:END -->
