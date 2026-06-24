---
title: >-
  [Paper Note] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent
description: >-
  [ACL 2025][LLM Agent][GUI agent] This paper proposes GUI-explorer, a training-free GUI agent that collects function-aware interaction trajectories through autonomous exploration and unsupervisedly mines transition-aware knowledge from state-transition triplets, achieving task success rates of 53.7% on SPA-Bench and 47.4% on AndroidWorld.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "GUI agent"
  - "autonomous exploration"
  - "transition-aware knowledge"
  - "training-free"
  - "knowledge extraction"
date: 2026-05-08
content_hash: 2671f2dc54cf5163
---

# GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent

**Conference**: ACL 2025  
**arXiv**: [2505.16827](https://arxiv.org/abs/2505.16827)  
**Code**: [https://github.com/JiuTian-VL/GUI-explorer](https://github.com/JiuTian-VL/GUI-explorer)  
**Area**: LLM Agent  
**Keywords**: GUI agent, autonomous exploration, transition-aware knowledge, training-free, knowledge extraction

## TL;DR
This paper proposes GUI-explorer, a training-free GUI agent that collects function-aware interaction trajectories through autonomous exploration and unsupervisedly mines transition-aware knowledge from state-transition triplets, achieving task success rates of 53.7% on SPA-Bench and 47.4% on AndroidWorld.

## Background & Motivation

**Background**:
   - MLLM-powered GUI agents can complete various interface operation tasks without fine-tuning.
   - However, practical deployment faces challenges from the long-tail distribution and rapid iteration of applications.
   - For example, Amazon Shopping released 30 version updates in 2024, causing model knowledge to quickly become outdated.

**Limitations of Prior Work**:
   - **UI component misunderstanding**: Even GPT-4o fails to correctly identify the function of the "Music Recognition" button in QQ Music.
   - **Outdated knowledge**: Frequent application interface updates make static model knowledge unreliable.
   - **Retraining cost**: Re-fine-tuning MLLMs for every application change is too costly.
   - Existing exploration methods (e.g., AppAgent) require manually designed exploration tasks, while AutoDroid randomly generates action sequences, which is inefficient.

**Key Challenge**:
   - The gap between the powerful general understanding capabilities of MLLMs and the personalized UI design and rapid changes of specific applications.
   - The need for application-specific knowledge without the possibility of retraining for each application.

**Goal**
   - How to automatically explore all functions of an application and establish precise mappings from UI elements to operation outcomes.
   - How to extract usable operational knowledge from interaction trajectories in an unsupervised manner.
   - How to dynamically utilize this knowledge during task execution to correct the erroneous priors of pretrained models.

**Key Insight**:
   - Starting from application structure information (APK manifest, activity list) to construct systematic, function-aware exploration.
   - Unsupervisedly mining atomic-level operational knowledge from state-transition triplets (observation-action-result).

**Core Idea**:
   - First, systematically explore applications with a function-oriented approach to acquire interaction trajectories. Next, extract knowledge from state transitions in an unsupervised manner. Finally, dynamically retrieve this knowledge to guide agent decision-making.

## Method

### Overall Architecture
GUI-explorer consists of three main components:
1. **Autonomous Exploration of Function-Aware Trajectories**: Automatically constructs exploration goals by analyzing GUI structural information and performs systematic exploration.
2. **Unsupervised Mining of Transition-Aware Knowledge**: Extracts operational logic from interaction triplets, requiring no human intervention.
3. **Dynamic Guidance for GUI Agent**: Performs vision-semantic retrieval to generate operational guidance for the current scene.

### Key Designs

1. **Function-aware Task Goal Generator**:

    - **Function**: Automatically extracts "Exploration Anchors" from the application environment, such as functional modules declared in the APK manifest (e.g., "PaymentActivity"), and constructs exploration tasks based on the current screenshot and anchors.
    - **Mechanism**: Employs a depth-first search (DFS) strategy configured with a branching factor $b=10$, maximum depth $d=5$, and a limit of 30 steps per task. The advantage of DFS lies in subtasks directly inheriting the termination state of parent tasks, avoiding backtracking overhead.
    - **Design Motivation**: Anchor constraints prevent the MLLM from proposing non-existent functionalities, achieving semantic grounding; DFS traversal can generate $\mathcal{O}(b^d)$ distinct trajectories, systematically covering combinatorial interaction patterns.

2. **Transition-aware Knowledge Extractor**:

    - **Function**: For each state-transition triplet $(o_i, a_i, o_{i+1})$, extracts the visual patch of the UI element (key) and the operational knowledge description (value) to build a multimodal knowledge vector database.
    - **Mechanism**: Uses Transition Filtering with perceptual hashing to filter out invalid transitions (where screenshots before and after the action are virtually identical), ensuring only successful and effective state changes are kept.
    - **Design Motivation**: Unlike approaches requiring successful complete trajectories, this method only needs valid single-step state transitions, significantly lowering the barrier to knowledge extraction.

3. **Continuous Knowledge Refinement**:

    - **Function**: Evaluates the relationship between new and existing knowledge entries via cosine similarity—merging them when key similarity $\ge 0.99$ and value similarity $\le 0.1$ (indicating a new functional description for the same UI element); otherwise, a new entry is added.
    - **Mechanism**: The knowledge base can be continuously updated without generating redundancy from repetitive exploration.
    - **Design Motivation**: Adapts to the frequent update requirements of applications, enabling progressive knowledge accumulation.

4. **Dynamic Guidance**:

    - **Function**: At execution time, performs vision-semantic retrieval and matching against the knowledge base for UI elements in the current screenshot (using SigLIP as the embedding model), then ranks them by relevance to the task instruction using an MLLM-based ranker to construct dynamic guidance prompts.
    - **Mechanism**: A modified merge sort algorithm that utilizes the MLLM to compare the task relevance of knowledge entries pairwise.
    - **Design Motivation**: Dual objectives—suppressing UI component misunderstandings and ensuring action recommendations align with the actual interface state.

### Loss & Training
- **Training-free**: GUI-explorer is a training-free method and does not involve parameter updates.
- GPT-4o is used as the unified base model for both exploration and inference.
- Exploration configuration: branching factor of 10, maximum depth of 5, step limit of 30.
- Automatically discovered over 1,300 knowledge entries across 46 applications.

## Key Experimental Results

### Main Results
- **SPA-Bench (Single-app English Level 3)**: GUI-explorer reaches a **53.7%** task success rate, **outperforming** the SOTA M3A (42.0%) by an **absolute margin of 11.7 percentage points**, and outperforming AppAgent (14.0%) by **39.7 percentage points**.
- **AndroidWorld**: **47.4%** success rate, outperforming Aria-UI (44.8%) by 2.6% and M3A (40.5%) by 6.9%.
- **GUI-KRB Benchmark**:
    - GPT-4o's prior knowledge error rate is 18.2%.
    - Applying GUI-explorer on top of Qwen2-VL-72B reduces the error rate by **16.0%**.
    - In dynamic understanding evaluation, the error rate is **13.4%** lower than the base model.

### Ablation Study / Key Findings
- **Effect of Transition Filtering**: Reduces prior knowledge errors by 16.0%.
- **MLLM GUI Understanding Benchmark** (GUI-KRB): Current models show a prior knowledge inaccuracy rate of 15.2% to 22.8% on 500 samples across 43 applications.
- **Execution Efficiency**: Total time per step is 54 to 72 seconds, with a cost of approximately \$0.053 to \$0.068 per step (including retrieval, ranking, and inference).
- **Knowledge-driven**: Confirmed that transition-aware knowledge rather than simple exploration data accumulation is the core factor driving performance gains.

## Highlights & Insights

- **Training-free Paradigm**: Adapts to new applications without requiring parameter updates, offering extremely high deployment flexibility.
- **Structured Exploration Strategy**: Leverages function declarations in APKs as semantic anchors, achieving a massive boost in efficiency compared to random exploration.
- **Unsupervised Construction of Knowledge Bases**: Successfully extracts valid state-transition knowledge even from failed trajectories, not relying on successful end-to-end paths.
- **GUI-KRB Benchmark Contribution**: The first to systematically evaluate MLLM's GUI understanding capabilities, revealing severe limitations of existing models.

## Limitations & Future Work

- The exploration phase still incurs high MLLM API usage costs (total cost around \$0.06 per step).
- The embedding model (SigLIP) used for the knowledge base may fail on UI elements that are visually similar but functionally distinct.
- Currently validated only on the Android platform; applicability to web and desktop platforms remains unverified.
- Merge-sort-styled knowledge ranking demands substantial MLLM calls, which may hinder real-time responsiveness.
- Cross-application knowledge transfer capabilities have not been explored in depth.

## Related Work & Insights

- **Comparison with AppAgent**: AppAgent requires manually designed exploration tasks, whereas GUI-explorer is fully autonomous.
- **Comparison with AutoDroid**: AutoDroid randomly generates action sequences, leading to inefficiencies and uncontrollable knowledge quality.
- **Comparison with DigiRL**: DigiRL uses reinforcement learning but requires Gemini to filter successful trajectories for training data.
- **Comparison with CAT/Synapse**: They rely on pre-collected successful trajectories or human feedback, limiting scalability.
- **Insight**: Knowledge-grounded agents represent a key path to achieving general GUI automation; grounded knowledge can effectively compensate for MLLMs' insufficient priors.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration](../../CVPR2025/llm_agent/gui-xplore_empowering_generalizable_gui_agents_with_one_exploration.md)
- [\[ACL 2025\] Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents](explorer_scaling_exploration-driven_web_trajectory_synthesis_for_multimodal_web_.md)
- [\[ACL 2025\] GUICourse: From General Vision Language Model to Versatile GUI Agent](guicourse_from_general_vision_language_model_to_versatile_gui_agent.md)
- [\[ICCV 2025\] Less is More: Empowering GUI Agent with Context-Aware Simplification](../../ICCV2025/llm_agent/less_is_more_empowering_gui_agent_with_context-aware_simplification.md)
- [\[ACL 2025\] OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis](os_genesis_gui_agent_trajectory.md)

</div>

<!-- RELATED:END -->
