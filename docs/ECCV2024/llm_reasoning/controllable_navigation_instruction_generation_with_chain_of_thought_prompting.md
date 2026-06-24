---
title: >-
  [Paper Note] Controllable Navigation Instruction Generation with Chain of Thought Prompting
description: >-
  [ECCV 2024][Reasoning][Navigation Instruction Generation] This paper proposes C-Instructor, which leverages chain-of-thought prompting of LLMs to achieve style- and content-controllable navigation instruction generation. Through three core mechanisms—Chain of Thought with Landmarks (CoTL), Spatial Topology Modeling Task (STMT), and Style-Mixed Training (SMT)—the method comprehensively outperforms existing approaches on four indoor and outdoor navigation datasets.
tags:
  - "ECCV 2024"
  - "Reasoning"
  - "Navigation Instruction Generation"
  - "Chain of Thought Prompting"
  - "Vision-Language Navigation"
  - "Large Language Models"
  - "Style-Controllable Generation"
date: 2026-05-08
content_hash: b4e078fc99d97288
---

# Controllable Navigation Instruction Generation with Chain of Thought Prompting

**Conference**: ECCV 2024  
**arXiv**: [2407.07433](https://arxiv.org/abs/2407.07433)  
**Code**: [GitHub](https://github.com/refkxh/C-Instructor)  
**Area**: LLM Reasoning  
**Keywords**: Navigation Instruction Generation, Chain of Thought Prompting, Vision-Language Navigation, Large Language Models, Style-Controllable Generation

## TL;DR

This paper proposes C-Instructor, which leverages chain-of-thought prompting of LLMs to achieve style- and content-controllable navigation instruction generation. Through three core mechanisms—Chain of Thought with Landmarks (CoTL), Spatial Topology Modeling Task (STMT), and Style-Mixed Training (SMT)—the method comprehensively outperforms existing approaches on four indoor and outdoor navigation datasets.

## Background & Motivation

**Background**: In Vision-Language Navigation (VLN), navigation instruction generation is one of the core tasks of embodied AI, requiring robots to generate natural language guidance for humans based on path information. This task is crucial for human-robot collaboration and can be applied to visually impaired navigation, hazardous scene guidance, etc.

**Limitations of Prior Work**: Existing instruction generation models (such as BT-Speaker, Lana, etc.) can only generate a single style of instructions from specific datasets, showing limited language quality and lacking controllability. Moreover, most approaches ignore the modeling of the spatial structure in the navigation environment, resulting in generated instructions that lack key landmark guidance at decision points like turns.

**Key Challenge**: In practical applications, the instruction style needs to be adjusted based on the receiver's familiarity (e.g., abstract vs. detailed), and the content needs to be adjusted based on the landmarks of interest. However, existing methods cannot achieve multi-style generation with a single model.

**Goal**: To enable a single model to simultaneously possess high language quality, style controllability, and content controllability.

**Key Insight**: Leveraging the linguistic capabilities and chain-of-thought reasoning paradigm of LLMs, an adapter structure is designed to inject path information into the LLM, and the model is guided by CoTL to first identify landmarks before generating instructions.

**Core Idea**: Introducing the CoT paradigm to navigation instruction generation, allowing the model to "think before speaking"—first identifying crucial landmarks and then generating high-quality instructions accordingly, while achieving single-model multi-style template switching through style-mixed training.

## Method

### Overall Architecture

Input path $R=\{r_1, r_2, ..., r_T\}$ (each step containing panoramic observations and actions) $\rightarrow$ Trajectory Encoder encodes visual features $\rightarrow$ LLM Adapter injects path features into the layers of a GPT-based LLM $\rightarrow$ generate instructions of different styles based on different prompts. The training phase incorporates the STMT auxiliary task and CoTL landmark supervision.

### Key Designs

1. **Trajectory Encoder + LLM Adapter**:

    - **Function**: Encodes the panoramic visual information of each step on the path into trajectory features and injects them into the LLM.
    - **Mechanism**: A CLIP visual encoder is used to extract the features of each sub-view $\boldsymbol{I}_{t,k} = \text{layer\_norm}(\text{linear}(f_{CLIP}(v_{t,k})))$. Spatial positional encodings $pos_k^v$, history encodings $pos_t^h$, and action/non-action identifiers $pos^a / pos^o$ are added. These are then compressed into a trajectory representation via ViT blocks and aggregated tokens. The LLM Adapter integrates trajectory features into the text representations of the LLM at each layer using zero-initialized attention.
    - **Design Motivation**: Directly using captions as intermediate representations loses a significant amount of spatial and visual information (as confirmed by ablation experiments showing that Vanilla LLM performs poorly). The adapter approach retains the linguistic capabilities of the LLM while effectively injecting spatial information.

2. **STMT (Spatial Topology Modeling Task)**:

    - **Function**: Serves as an auxiliary training task, requiring the model to predict how to return to the previous node from the current node (backtracking action prediction).
    - **Mechanism**: Given the trajectory $\{r_1, ..., r_t\}$, the model predicts $a_t^p$ after aggregating visual features via cross-attention, such that $\boldsymbol{A}_t = \text{softmax}(\boldsymbol{x}_L^a \boldsymbol{W} \boldsymbol{I}_{t,1:36}^\top)$, which is supervised using the cross-entropy loss $\mathcal{L}_a$.
    - **Design Motivation**: LLMs and visual encoders are primarily trained on text and image data from the internet and have weak spatial cognitive abilities. Since forward actions are already represented by positional encodings, the model is tasked with predicting backtracking actions to learn the spatial topology.

3. **CoTL (Chain of Thought with Landmarks)**:

    - **Function**: Guides the model to first identify critical landmarks along the path and then generate instructions.
    - **Mechanism**: Landmark selection is divided into two dimensions. In the **temporal dimension**, the cosine distance between neighboring viewpoint features is calculated as $\delta_t^\tau = 1 - \frac{\boldsymbol{I}_t^* \cdot \boldsymbol{I}_{t+1}^*}{||\boldsymbol{I}_t^*|| \cdot ||\boldsymbol{I}_{t+1}^*||}$ to locate scene transition points (e.g., from a corridor to a room). In the **spatial dimension**, unique objects in the action viewpoint are selected as landmarks (objects appearing in other candidate viewpoints receive penalty points: $\delta_{t,n}^a = 1 - d_{t,c_1}^a - d_{t,c_2}^a - d_{t,c_3}^a$). The final landmark score is $\delta_{t,n} = \delta_{t,n}^a \cdot \delta_t^\tau$, and object nodes exceeding the threshold $\beta$ are selected as visual landmarks. Inference is conducted in two stages: first predicting landmarks, then generating instructions based on them.
    - **Design Motivation**: Cognitive psychology research shows that humans also locate key navigation points in their cognitive map before organizing language when giving path directions. Modifying the predicted landmarks can further enable content controllability.

4. **SMT (Style-Mixed Training)**:

    - **Function**: Trains on a mixture of datasets with different language styles and switches generation styles via different prompts.
    - **Mechanism**: Descriptive prompts are designed for each style, and different style datasets such as R2R (detailed step-by-step), REVERIE (high-level abstract description), and RxR (fine-grained alignment) are mixed during training.
    - **Design Motivation**: Training on a single style has limited data and is prone to overfitting. Mixed training increases linguistic diversity and simultaneously achieves multi-style switching within a single model.

### Loss & Training

- The autoregressive instruction generation loss is jointly optimized with the cross-entropy loss $\mathcal{L}_a$ of STMT.
- The LLM uses LLaMA-Adapter (7B parameters, 32 layers), with only the last 2 layers being fine-tuned.
- The model is first pre-trained on the PREVALENT dataset for 240K iterations, and then fine-tuned on multiple datasets for 120K iterations.
- AdamW optimizer with a learning rate of $1.0 \times 10^{-4}$ is used on 4 A100 80GB GPUs.

## Key Experimental Results

### Main Results

| Dataset | Metric | C-Instructor | Prev. SOTA (Lana) | Gain |
|--------|------|------|----------|------|
| R2R val unseen | SPICE | 0.212 | 0.174 | +21.8% (relative) |
| R2R val unseen | CIDEr | 0.447 | 0.295 | +51.5% |
| REVERIE val unseen | SPICE | 0.141 | 0.107 | +31.8% |
| REVERIE val unseen | CIDEr | 0.545 | 0.327 | +66.7% |
| RxR val unseen | BLEU-4 | 0.233 | 0.115 | +102.6% |
| UrbanWalk | SPICE | 0.645 | 0.566 (Kefa) | +14.0% |

### Ablation Study

| Configuration | REVERIE CIDEr | R2R CIDEr | Description |
|------|---------|---------|------|
| Vanilla LLM | 0.432 | 0.292 | Pure Caption + LLM, severe information loss |
| Baseline (Adapter) | 0.347 | 0.356 | Only visual adapter |
| + SMT | 0.397 | 0.407 | Style-mixed training improves language diversity |
| + SMT + STMT | 0.490 | 0.445 | Spatial modeling is highly beneficial for high-level abstract instructions |
| + SMT + STMT + CoTL | 0.545 | 0.447 | Landmark guidance further improves semantic consistency |

### Key Findings

- When instructions generated by C-Instructor are used for navigation data augmentation, it is the only method that improves navigator (HAMT) performance (SR: $32.95 \rightarrow 34.25$), whereas other methods degrade performance.
- Navigation guidance experiment: C-Instructor's instructions guiding the DUET navigator achieve an SR of 43.34%, which is close to human annotations (46.98%).
- In the user study, C-Instructor achieves an average score of 3.50, far exceeding Lana (2.26) and other methods (~2.10).
- The improvement of STMT on REVERIE (high-level abstract instructions) is particularly significant (CIDEr +0.093), indicating that spatial understanding is crucial for generating abstract instructions.

## Highlights & Insights

- **Transferring the CoT paradigm to instruction generation**: Transferring "think before answering" in LLM reasoning to "find landmarks before speaking" in vision-language navigation instruction generation is both ingenious and effective.
- **The comparison with the Vanilla LLM baseline** is highly convincing, demonstrating the limitations of direct captioning $\rightarrow$ LLM pipelines.
- **The SMT strategy is worth learning from**: Treating different language styles from different datasets as different prompts enables multi-tasking in a single model, which simultaneously addresses data scarcity and achieves controllability.
- **The landmark selection algorithm** integrates both temporal and spatial dimensions with clear physical meanings, making it transferable to other tasks requiring automatic visual focus selection.
- **Backtracking action prediction** as an auxiliary task is highly clever. Since forward actions are already encoded in the input, predicting backtracking actions is what truly learns spatial relationships.

## Limitations & Future Work

- It relies on discrete navigation graphs from the Matterport3D Simulator, making it difficult to apply directly to continuous space navigation.
- The landmark selection threshold $\beta=0.25$ is manually set; adaptive learning could be considered.
- Only CLIP is used to extract visual features, without leveraging stronger visual encoders or multimodal LLMs.
- Inference requires two stages (first predicting landmarks then generating instructions), which is less efficient than end-to-end methods.
- Conversational interaction scenarios, such as iteratively modifying instructions based on user feedback, have not been explored.

## Related Work & Insights

- **vs. Lana (CVPR2023)**: Lana uses cycle-consistent learning to jointly optimize instruction generation and following, but supports only a single style and lacks landmark controllability. C-Instructor outperforms it significantly through SMT and CoTL.
- **vs. BT-Speaker**: A classic LSTM baseline, with language quality far below LLM-based methods.
- **vs. LLaMA-Adapter**: C-Instructor adds a Trajectory Encoder and three training strategies on top of LLaMA-Adapter, proving that adapter structures combined with domain-specific training strategies produce prominent results.

## Rating

- Novelty: ⭐⭐⭐⭐ The CoTL mechanism, combining CoT with visual landmark identification, is innovative, but the overall framework (adapter + LLM) is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, with evaluation on 4 datasets, detailed ablation studies, navigator assessments, and user studies.
- Writing Quality: ⭐⭐⭐⭐ Structural flow is clear and the method descriptions are detailed, though the mathematical notations are slightly complex.
- Value: ⭐⭐⭐⭐ Style and content controllability have high practical value in human-robot interaction, and the data augmentation effects are substantial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](../../ACL2026/llm_reasoning/learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)
- [\[AAAI 2026\] Intention Chain-of-Thought Prompting with Dynamic Routing for Code Generation](../../AAAI2026/llm_reasoning/intention_chain-of-thought_prompting_with_dynamic_routing_for_code_generation.md)
- [\[ICLR 2026\] String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation](../../ICLR2026/llm_reasoning/string_seed_of_thought_prompting_llms_for_distribution-faithful_and_diverse_gene.md)
- [\[ACL 2026\] Strategy-Induct: Task-Level Strategy Induction for Instruction Generation](../../ACL2026/llm_reasoning/strategy-induct_task-level_strategy_induction_for_instruction_generation.md)
- [\[ACL 2025\] RSVP: Reasoning Segmentation via Visual Prompting and Multi-modal Chain-of-Thought](../../ACL2025/llm_reasoning/rsvp_reasoning_segmentation_via_visual_prompting_and_multi-modal_chain-of-though.md)

</div>

<!-- RELATED:END -->
