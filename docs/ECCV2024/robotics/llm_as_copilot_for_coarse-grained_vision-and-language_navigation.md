---
title: >-
  [Paper Note] LLM as Copilot for Coarse-Grained Vision-and-Language Navigation
description: >-
  [ECCV 2024][Robotics][Vision-and-Language Navigation] This paper proposes the VLN-Copilot framework, where a vision-and-language navigation agent actively seeks help from an LLM when confused under coarse-grained (short and ambiguous) instructions. Acting as a copilot, the LLM generates real-time, fine-grained navigation guidance, significantly improving navigation success rates on two coarse-grained VLN datasets.
tags:
  - "ECCV 2024"
  - "Robotics"
  - "Vision-and-Language Navigation"
  - "Large Language Models"
  - "Coarse-Grained Instructions"
  - "Confusion Score"
  - "Active Help-Seeking"
date: 2026-05-08
content_hash: 644a0ffa2a973ca8
---

# LLM as Copilot for Coarse-Grained Vision-and-Language Navigation

**Conference**: ECCV 2024  
**Code**: None  
**Area**: LLM Agent / Robotic Navigation  
**Keywords**: Vision-and-Language Navigation, Large Language Models, Coarse-Grained Instructions, Confusion Score, Active Help-Seeking

## TL;DR
This paper proposes the VLN-Copilot framework, where a vision-and-language navigation agent actively seeks help from an LLM when confused under coarse-grained (short and ambiguous) instructions. Acting as a copilot, the LLM generates real-time, fine-grained navigation guidance, significantly improving navigation success rates on two coarse-grained VLN datasets.

## Background & Motivation

**Background**: Vision-and-Language Navigation (VLN) requires agents to navigate to a target location in indoor environments based on natural language instructions. Traditional VLN research mostly focuses on fine-grained instructions—detailed step-by-step path descriptions (e.g., "turn left to the kitchen entrance, then turn right and pass the dining table..."). However, in real-world scenarios, users tend to give short, high-level instructions (e.g., "go to the bedroom on the second floor"). Such coarse-grained instructions align better with human interaction habits and have received growing attention in recent years.

**Limitations of Prior Work**: Coarse-grained instructions are typically too brief and lack intermediate landmarks and action descriptions during navigation, making it difficult for agents to make correct decisions based on this information alone. Specifically, when an agent faces multiple candidate paths, coarse-grained instructions fail to provide sufficient disambiguation. Some existing methods attempt to let agents actively seek help during navigation, but the sources of help are usually pre-built datasets or fixed responses in simulators, which limits flexibility and practicality.

**Key Challenge**: There is a huge gap between the information content in coarse-grained instructions and the information required for navigation decisions. The agent needs to acquire additional info at critical decision points, but traditional methods cannot dynamically generate context-aware navigation advice.

**Goal**: (1) How to determine when the agent needs help—i.e., when is it uncertain about the current decision? (2) How to dynamically generate fine-grained navigation guidance relevant to the current scene and goal? (3) How to seamlessly integrate LLM assistance into existing VLN frameworks?

**Key Insight**: The authors observe that LLMs possess strong spatial and common-sense reasoning capabilities, allowing them to infer reasonable navigation paths based on scene descriptions and target information. The key insight is: instead of replacing the agent with an LLM for all decisions, the LLM should act as a "copilot"—providing assistance only when the agent is confused. This leverages the LLM's reasoning capabilities while preserving the visual understanding advantages of specialized VLN models.

**Core Idea**: Quantify the decision uncertainty of the VLN agent through confusion scores, and actively request fine-grained navigation guidance from the LLM when confusion is high. The LLM then generates real-time advice based on scene context to assist navigation.

## Method

### Overall Architecture
VLN-Copilot consists of three core components: (1) a base VLN agent, responsible for perceiving the environment and executing navigation actions; (2) a confusion evaluation module, which evaluates the agent's uncertainty in real-time at each decision step; and (3) an LLM copilot, which receives scene description queries and generates fine-grained navigation guidance for the agent when the agent's confusion exceeds a threshold. The overall pipeline is: agent observes the environment → calculates confusion → if confused, asks the LLM for help → LLM returns fine-grained instructions → agent makes a decision combining the original instructions and the LLM's advice.

### Key Designs

1. **Confusion Score**:

    - **Function**: Quantifies the agent's decision uncertainty at each navigation step, deciding whether to seek help from the LLM.
    - **Mechanism**: At each time step, the VLN agent calculates a probability distribution over all navigable actions. Confusion is defined as the entropy of the action probability distribution—when the gap between the highest-probability action and the second-highest-probability action is small, the entropy is high, indicating that the agent is uncertain about which path to take. Specifically, $CS_t = -\sum_i p(a_i) \log p(a_i)$, and help-seeking is triggered when $CS_t$ exceeds a threshold $\tau$. The threshold $\tau$ is determined by searching for the optimal value on the validation set.
    - **Design Motivation**: Compared to fixed-frequency help-seeking, adaptive help-seeking based on confusion is more efficient—avoiding help-seeking on simple paths to reduce LLM API call costs, and only requesting assistance at critical intersections. This simulates human driving behavior where one only consults the GPS when encountering an uncertain intersection.

2. **Scene Description Construction**:

    - **Function**: Converts the visual information currently perceived by the agent into textual descriptions understandable by the LLM.
    - **Mechanism**: After deciding to seek help, the system converts the visual observations of all navigable directions in the agent's current panoramic view into text. This primarily includes three types of information: (1) visible objects and room types in each direction; (2) descriptions of the path the agent has already traversed (history trajectory); and (3) the target description from the original coarse-grained instruction. This information is organized into a structured prompt and input to the LLM.
    - **Design Motivation**: LLMs cannot process images directly, necessitating a vision-to-text conversion. By including historical trajectory information, the LLM can comprehend the agent's current spatial context and avoid generating repetitive or contradictory advice.

3. **LLM Guidance Fusion**:

    - **Function**: Fuses the LLM-returned fine-grained guidance with the original instruction for the agent's action decision-making.
    - **Mechanism**: The fine-grained guidance generated by the LLM is encoded as text embeddings and fused with the embeddings of the original coarse-grained instruction via an attention mechanism. The fused instruction representation then undergoes cross-attention with the visual features to produce an updated action probability distribution. To balance the impact of the original instruction and the LLM's advice, a learnable gating mechanism is introduced, allowing the model to adaptively decide to what extent it should adopt the LLM's guidance.
    - **Design Motivation**: Relying entirely on the LLM might introduce noise (since LLM understanding of spatial descriptions can be imprecise). Hence, a gating mechanism is needed to ensure the original VLN model's visual judgment still plays a role, making the two complementary.

### Loss & Training
The training is divided into two phases: (1) pre-training the base VLN agent using standard cross-entropy loss and auxiliary progress monitoring loss; (2) freezing the VLN backbone and training the fusion module and gating mechanism using imitation learning loss (to make the agent closer to the optimal path after receiving LLM guidance). The LLM (GPT-4/LLaMA) is not fine-tuned and is used directly via in-context learning.

## Key Experimental Results

### Main Results

| Dataset | Metric | VLN-Copilot | HAMT | DUET | Gain |
|--------|------|------------|------|------|------|
| R2R-Last (val unseen) | SR↑ | 52.3 | 44.6 | 46.1 | +6.2 vs DUET |
| R2R-Last (val unseen) | SPL↑ | 44.8 | 38.2 | 40.5 | +4.3 vs DUET |
| REVERIE (val unseen) | SR↑ | 38.7 | 32.1 | 33.4 | +5.3 vs DUET |
| REVERIE (val unseen) | SPL↑ | 30.2 | 25.6 | 27.3 | +2.9 vs DUET |

### Ablation Study

| Configuration | SR↑ | SPL↑ | Description |
|------|-----|------|------|
| Full VLN-Copilot | 52.3 | 44.8 | Full model |
| Fixed-frequency help | 49.1 | 41.2 | Seeks help every 5 steps; inferior to the adaptive strategy |
| w/o Scene History | 50.5 | 43.1 | LLM lacks historical trajectory info, leading to inaccurate advice |
| w/o Gated Fusion | 48.7 | 40.9 | Performance drops when relying entirely on LLM advice |
| Oracle Help | 57.8 | 50.3 | Upper bound where help is sought only when truly needed |

### Key Findings
- The gating mechanism of the confusion score is the most critical: removing it decreases SR by 3.6%, demonstrating that selective help-seeking is superior to blind help-seeking.
- The gap between the proposed method and Oracle help (5.5% SR) suggests that confusion evaluation still has room for improvement.
- The performance difference between different LLMs is marginal (GPT-4 vs LLaMA-2-70B), indicating that the framework is insensitive to the choice of LLM.
- The help-seeking frequency is about 20-30% of the total steps, saving LLM calling costs while achieving significant improvements.
- Incorporating scene history information increases the relevance of LLM guidance by 12% (based on human evaluation).

## Highlights & Insights
- **Adaptive Help-Seeking Mechanism**: The confusion score enables "on-demand help-seeking", avoiding the high overhead of calling the LLM at every step. This design concept can be transferred to other real-time decision-making systems requiring LLM assistance (e.g., robotic manipulation, autonomous driving).
- **LLM as Copilot instead of Pilot**: Out of substituting specialized models, it compensates for their deficiencies. This collaboration paradigm is more practical than end-to-end LLM solutions—retaining the visual understanding strength of specialized models while borrowing the common-sense reasoning of LLMs.
- **Gated Fusion preventing LLM Noise**: LLMs may have imprecise understanding of spatial descriptions. The gating mechanism allows the system to adaptively decide the extent to which it adopts the LLM's advice.

## Limitations & Future Work
- The construction of scene descriptions relies on the accuracy of the object detector; detection errors can cause the LLM to give incorrect suggestions.
- LLM api calls introduce additional latency (about 1-2 seconds per call for GPT-4), which may affect smoothness in real-time navigation.
- The confusion score is solely based on the entropy of the action probability distribution, without considering environmental complexity or other contextual factors.
- The method is only validated on discrete navigation graphs, and its applicability in continuous environments remains unknown.
- Multimodal LLMs (e.g., GPT-4V) could be considered to receive image inputs directly, bypassing the scene description construction step that might introduce information loss.

## Related Work & Insights
- **vs NavGPT**: NavGPT relies entirely on LLMs for decision-making but lacks visual grounding capability; VLN-Copilot retains the specialized VLN model for visual understanding, utilizing the LLM only for assistance when confused.
- **vs VELMA**: VELMA uses LLMs for path planning but relies on fine-grained instructions; VLN-Copilot specifically targets coarse-grained instructions, bridging the information gap via the LLM.
- **vs HELPER/Ask4Help**: Prior help-seeking methods use fixed auxiliary data sources, whereas VLN-Copilot utilizes the generative capabilities of LLMs to provide dynamic, context-aware advice.
- This paradigm of "seeking LLM help when confused" can be extended to other scenarios requiring on-demand external knowledge acquisition, such as question-answering and dialogue systems.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of using an LLM as a VLN copilot is novel, and the confusion-driven help-seeking mechanism is logically designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on two datasets, featuring detailed ablation studies and help-seeking frequency analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, and the copilot metaphor is intuitive and easy to understand.
- Value: ⭐⭐⭐⭐ Provides a practical collaboration paradigm for LLM-assisted embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On Entropy Control in LLM-RL Algorithms](../../ICLR2026/robotics/on_entropy_control_in_llm-rl_algorithms.md)
- [\[ACL 2025\] CHEER-Ekman: Fine-grained Embodied Emotion Classification](../../ACL2025/robotics/cheer-ekman_fine-grained_embodied_emotion_classification.md)
- [\[ICCV 2025\] COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation](../../ICCV2025/robotics/cosmo_combination_of_selective_memorization_for_low-cost_vision-and-language_nav.md)
- [\[CVPR 2026\] Cross from Left to Right Brain: Adaptive Text Dreamer for Vision-and-Language Navigation](../../CVPR2026/robotics/cross_from_left_to_right_brain_adaptive_text_dreamer_for_vision-and-language_nav.md)
- [\[ACL 2026\] VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions](../../ACL2026/robotics/vln-nf_feasibility-aware_vision-and-language_navigation_with_false-premise_instr.md)

</div>

<!-- RELATED:END -->
