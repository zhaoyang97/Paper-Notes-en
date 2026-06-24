---
title: >-
  [Paper Note] BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent
description: >-
  [NeurIPS 2025][LLM Agent][GUI agent] This paper proposes the Blink-Think-Link (BTL) brain-inspired framework, which decomposes GUI interaction into three biologically plausible stages: Blink (rapid attentional localization), Think (cognitive reasoning and decision-making), and Link (executable command generation). Combined with an automated Blink data annotation pipeline and the first rule-based composite process-and-outcome reward mechanism, BTL Reward…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "GUI agent"
  - "Blink-Think-Link"
  - "cognitive-inspired"
  - "GRPO"
  - "BTL Reward"
date: 2026-05-08
content_hash: daa9b5ee89c2a0e7
---

# BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent

**Conference**: NeurIPS 2025  
**arXiv**: [2509.15566](https://arxiv.org/abs/2509.15566)  
**Code**: [https://github.com/xiaomi-research/btl-ui](https://github.com/xiaomi-research/btl-ui)  
**Area**: LLM Agent / GUI Automation  
**Keywords**: GUI agent, Blink-Think-Link, cognitive-inspired, GRPO, BTL Reward

## TL;DR

This paper proposes the Blink-Think-Link (BTL) brain-inspired framework, which decomposes GUI interaction into three biologically plausible stages: Blink (rapid attentional localization), Think (cognitive reasoning and decision-making), and Link (executable command generation). Combined with an automated Blink data annotation pipeline and the first rule-based composite process-and-outcome reward mechanism, BTL Reward, the resulting BTL-UI model achieves competitive performance on both static GUI understanding and dynamic interaction benchmarks.

## Background & Motivation

AI-driven GUI interaction automation is rapidly advancing, yet existing approaches suffer from two fundamental problems:

- **SFT-based methods**: rely on large-scale expert-annotated data and exhibit poor generalization to out-of-distribution scenarios.
- **Existing RFT methods**: adopt a Think-Answer structure (`<think>` + `<answer>`), which diverges substantially from human GUI interaction patterns; moreover, their reward mechanisms focus solely on final outcomes, providing no guidance over intermediate cognitive processes.

Cognitive science research indicates that human GUI interaction follows three sequential stages: (a) a saccadic phase for rapid target localization; (b) multimodal information integration and reasoning; and (c) precise action execution. Existing agents skip the critical attentional localization stage.

## Method

### Overall Architecture

BTL models GUI interaction as an MDP, with a policy function $F(\{z_t, u, h\}) \rightarrow o_t = \{b_t, d_t, a_t\}$ that outputs results from three stages. Optimization is performed using GRPO (Group Relative Policy Optimization).

### Key Designs

1. **Blink Stage (Visual Attention Localization)**:

    - **Function**: Rapidly localizes task-relevant ROI regions on the screen; output is enclosed in `<blink></blink>` tags.
    - **Mechanism**: Simulates human saccadic eye movements. Blink data is generated via a two-stage pipeline — (1) a parsing model extracts all UI elements (bbox/type/caption); (2) Qwen2.5-VL-32B selects the $\lambda$ most relevant elements based on the task instruction.
    - **Design Motivation**: Existing methods generate actions directly from screenshots without explicit attention to task-relevant regions; Blink provides top-down attentional guidance.

2. **Think Stage (Cognitive Reasoning)**:

    - **Function**: Performs high-level reasoning and decision-making over the regions identified by Blink; the reasoning process is output within `<think></think>` tags.
    - **Mechanism**: Understands the current state, analyzes task objectives, and plans the next operation.
    - **Design Motivation**: This stage retains a DeepSeek-R1-style chain-of-thought, but grounds it in the focused information provided by Blink.

3. **Link Stage (Action Generation)**:

    - **Function**: Generates executable GUI commands (click coordinates, text input, etc.); output is enclosed in `<link></link>` tags.
    - **Mechanism**: A complete command is composed of action type $\alpha_t$ and parameters $\delta_t$.
    - **Design Motivation**: Separating this stage from Think ensures structured and parseable commands.

### Loss & Training

**BTL Reward** comprises three components: $R_{\text{BTL}} = R_{\text{format}} + R_{\text{blink}} + R_{\text{link}}$

- **Dual Format Reward ($R_{\text{format}}$)**: Verifies whether the output satisfies the BTL three-stage template structure and XML/JSON content format; binary reward.
- **Blink Reward ($R_{\text{blink}}$)**: Computes IoU between predicted ROIs and ground-truth ROIs using the Hungarian matcher; allows empty predictions for non-interactive operations (e.g., scrolling or back navigation).
- **Link Reward ($R_{\text{link}}$)**: Strictly binary — reward is granted only when both the action type and parameters are simultaneously correct, preventing reward hacking.

GRPO optimization: generate $N$ completions → compute within-group relative advantage $A_i$ → maximize the policy objective with KL regularization. Models are trained on top of Qwen2.5-VL-3B/7B.

## Key Experimental Results

### Main Results

| Model | Method | ScreenSpot Avg. | ScreenSpot-V2 | ScreenSpot-Pro |
|-------|--------|----------------|---------------|----------------|
| GPT-4o | ZS | 18.8 | - | - |
| Qwen2.5-VL-7B | ZS | 82.0 | - | - |
| OS-Atlas-Base-7B | ZS | 82.5 | - | - |
| **BTL-UI-3B** | RFT | **competitive** | competitive | competitive |
| **BTL-UI-7B** | RFT | **competitive** | competitive | competitive |

BTL-UI is trained via RFT on only 4K samples and achieves competitive performance on the ScreenSpot series of grounding benchmarks as well as planning benchmarks including AndroidControl and GUI-Odyssey.

### Ablation Study

- **Full BTL vs. Think-Answer**: The BTL three-stage structure outperforms the conventional two-stage Think-Answer structure.
- **BTL Reward vs. Link Reward only**: Process-level guidance (Blink Reward) substantially improves overall performance.
- **Strict Link Reward vs. split reward**: Strict binary reward is more effective than separately rewarding action type and parameters, mitigating reward hacking.
- **Effect of Blink selection count $\lambda$**: Too small a $\lambda$ leads to insufficient information; too large increases token overhead.

### Key Findings

- Process-level guidance is critical for GUI agents — evaluating not only "what was done correctly" but also "what was attended to correctly."
- Attentional localization in the Blink stage sharpens subsequent reasoning by reducing interference from irrelevant UI elements.
- Strict Link Reward prevents the model from learning an opportunistic strategy of "guessing the correct action type with incorrect parameters."

## Highlights & Insights

- **Brain-inspired design yields practical gains**: Blink→Think→Link mirrors the human gaze→reasoning→action process; the performance improvements are substantive rather than superficial.
- **BTL Reward is the first composite process-and-outcome reward in GUI agent research**, providing richer training signals for RFT-based GUI agents.
- **Automated Blink data annotation** addresses the training data bottleneck for GUI agents, eliminating the need for manual annotation of attentional regions.
- **Effective with only 4K training samples** — demonstrating high data efficiency.

## Limitations & Future Work

- Serial execution of the three stages increases inference latency — it remains unclear whether the computational overhead of the Blink stage can be reduced.
- Automated Blink data annotation relies on Qwen2.5-VL-32B — the capability ceiling of the annotation model constrains Blink data quality.
- Evaluation is limited to static grounding and short-horizon planning — long-horizon performance on complex multi-step interaction tasks is not fully assessed.
- RFT training uses only 4K samples — the scaling behavior under larger-scale training remains unexplored.

## Related Work & Insights

- **UI-R1 / GUI-R1**: Introduce rule-based RL for GUI agents, but adopt a Think-Answer structure with only outcome reward.
- **InfiGUI-R1**: The Actor2Reasoner architecture bridges reactive execution and deliberate reasoning, but lacks explicit attentional modeling.
- **UI-TARS**: Combines pretraining with SFT, but does not employ RL.
- **Inspiration**: The BTL three-stage framework is transferable to other human-computer interaction scenarios (e.g., Look→Plan→Act in autonomous driving).

## Rating

- Novelty: ⭐⭐⭐⭐ Brain-inspired three-stage framework and BTL Reward exhibit distinctive design
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple GUI benchmarks with extensive ablations
- Writing Quality: ⭐⭐⭐⭐ Cognitive science motivation is tightly integrated with technical design
- Value: ⭐⭐⭐⭐ Process-guided RL for GUI agents represents a pioneering contribution

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GUICourse: From General Vision Language Model to Versatile GUI Agent](../../ACL2025/llm_agent/guicourse_from_general_vision_language_model_to_versatile_gui_agent.md)
- [\[ICLR 2026\] UI-Ins: Enhancing GUI Grounding with Multi-Perspective Instruction-as-Reasoning](../../ICLR2026/llm_agent/ui-ins_enhancing_gui_grounding_with_multi-perspective_instruction_as_reasoning.md)
- [\[NeurIPS 2025\] TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration](trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)
- [\[ACL 2026\] Exploring Reasoning Reward Model for Agents](../../ACL2026/llm_agent/exploring_reasoning_reward_model_for_agents.md)
- [\[NeurIPS 2025\] AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks](agenttts_large_language_model_agent_for_testtime_computeopti.md)

</div>

<!-- RELATED:END -->
