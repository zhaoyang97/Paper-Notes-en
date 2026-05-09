---
title: >-
  [Paper Note] Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos
description: >-
  [CVPR 2026][LLM Agent][Web Agent] This paper proposes Ego2Web, the first benchmark that bridges egocentric video perception with web agent execution, accompanied by a semi-automatic data construction pipeline and the Ego2WebJudge automatic evaluation framework. Experiments reveal that current state-of-the-art agents still exhibit a substantial gap in cross-modal transfer from real-world visual perception to online action, with the best model achieving only 48.2% success rate.
tags:
  - CVPR 2026
  - LLM Agent
  - Web Agent
  - Egocentric Video
  - Multimodal Benchmark
  - Cross-Modal Transfer
  - Automatic Evaluation
date: 2026-05-08
content_hash: 96671c84e02eff4f
---

# Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos

**Conference**: CVPR 2026
**arXiv**: [2603.22529](https://arxiv.org/abs/2603.22529)
**Code**: [https://github.com/Yui010206/Ego2Web](https://github.com/Yui010206/Ego2Web)
**Area**: Agent
**Keywords**: Web Agent, Egocentric Video, Multimodal Benchmark, Cross-Modal Transfer, Automatic Evaluation

## TL;DR

This paper proposes Ego2Web, the first benchmark that bridges egocentric video perception with web agent execution, accompanied by a semi-automatic data construction pipeline and the Ego2WebJudge automatic evaluation framework. Experiments reveal that current state-of-the-art agents still exhibit a substantial gap in cross-modal transfer from real-world visual perception to online action, with the best model achieving only 48.2% success rate.

## Background & Motivation

**State of the Field**: Multimodal AI agents are advancing rapidly, evolving from simple conversational assistants toward systems capable of executing tasks in real web environments (e.g., shopping, searching, querying maps). Several web agent benchmarks already exist (e.g., WebArena, MiniWoB++, Mind2Web) for evaluating task completion in online environments.

**Limitations of Prior Work**: Existing web agent benchmarks share a fundamental limitation—they focus exclusively on web-side interaction and perception, lacking any connection to the user's real physical environment. This means a critical scenario remains unevaluated: when an agent must first recognize objects in the user's surroundings via egocentric vision (e.g., AR glasses) and then complete related tasks online (e.g., spotting a snack and searching for it on Amazon). This ability to bridge "seeing" and "online execution" is a core requirement for future AI assistants.

**Root Cause**: Current web agent evaluations only consider capabilities within the digital world, entirely ignoring an agent's ability to acquire visual cues from the physical world and translate them into digital actions. This leaves the true performance of current models on the complete "perceive → understand → act" pipeline unknown.

**Paper Goals**: To construct a benchmark that integrates egocentric video perception with web action execution, systematically evaluating agents' visual understanding, task planning, and online interaction capabilities.

**Starting Point**: The paper leverages existing large-scale egocentric video datasets (e.g., Ego4D), combined with a VLM+LLM automated data generation pipeline and human verification, to construct high-quality video–web task pairs.

**Core Idea**: Visual evidence extracted from egocentric videos (e.g., brands, objects, actions) serves as grounding information, requiring agents to complete related tasks in real web environments, thereby evaluating agent capabilities that span the physical and digital worlds.

## Method

### Overall Architecture

Ego2Web consists of a three-stage system: (1) **Semi-automatic data construction pipeline**: visual metadata is generated from egocentric videos, web task instructions are then produced by an LLM, and finally human verification and refinement are applied; (2) **Benchmark dataset**: video–task pairs covering multiple web task types (e-commerce, media retrieval, knowledge query, local/map services, etc.); (3) **Ego2WebJudge evaluation framework**: an LLM-as-a-Judge-based automatic evaluation method enabling scalable online assessment.

### Key Designs

1. **Semi-Automatic Data Generation Pipeline**:

   - *Function*: Transforms raw egocentric videos into high-quality "video + web task" pairs.
   - *Mechanism*: A VLM (e.g., Gemini) first performs structured parsing of egocentric videos to generate clip-level captions and visual metadata (identifying objects, brands, actions, etc.); an LLM then uses this visual metadata to generate web task instructions targeting live websites (e.g., Amazon, YouTube, Wikipedia); finally, human annotators verify each sample for visual grounding accuracy, web feasibility, and instruction quality.
   - *Design Motivation*: Purely manual annotation is prohibitively costly, while fully automatic generation lacks quality control. The semi-automatic pipeline balances efficiency and quality, ensuring each sample has authentic visual grounding and an executable web task.

2. **Multi-Type Web Task Design**:

   - *Function*: Covers diverse web interaction scenarios that a daily AI assistant would need to handle.
   - *Mechanism*: Tasks are divided into four categories—e-commerce (e.g., spotting a snack and searching to purchase it), media retrieval (e.g., seeing a fitness movement and searching for tutorial videos), knowledge query (e.g., seeing a university name and looking up admission information), and local/map services (e.g., spotting a store and searching for navigation routes). Each task requires the agent to first extract key visual evidence from the video and then complete the corresponding operation on the web.
   - *Design Motivation*: Different task types demand different agent capabilities—e-commerce requires fine-grained object recognition, media retrieval requires action understanding, knowledge queries require text recognition, and map services require spatial localization. Multi-type evaluation comprehensively exposes agent capability gaps.

3. **Ego2WebJudge Automatic Evaluation Method**:

   - *Function*: Enables scalable automated evaluation in live web environments.
   - *Mechanism*: Given the task instruction, the agent's action trajectory, web screenshots, and annotated visual evidence from the video, Ego2WebJudge first extracts key success criteria, then selects the most relevant screenshots from the agent's web action trajectory, and finally determines whether the agent completed the task correctly and consistently. Unlike simple URL/text matching, Ego2WebJudge considers the consistency between visual evidence and web content.
   - *Design Motivation*: In live web environments, traditional exact-match methods (e.g., URL matching) are too brittle, while human evaluation does not scale. Ego2WebJudge achieves approximately 84% human judgment agreement, substantially outperforming existing evaluation methods.

### Loss & Training

Ego2Web is an evaluation benchmark rather than a training method, and therefore involves no training loss. Evaluation uses **Success Rate (SR)** as the primary metric, determined automatically by Ego2WebJudge.

## Key Experimental Results

### Main Results

| Agent (Model) | E-Commerce SR | Media Retrieval SR | Knowledge Query SR | Map SR | Overall SR |
|---|---|---|---|---|---|
| Qwen3-VL-Flash | 21.7 | 30.1 | 50.0 | 23.1 | 29.0 |
| GPT-4o | 26.9 | 30.3 | 63.0 | 22.5 | 34.6 |
| Gemini-2.5 Pro | 38.2 | 50.7 | 75.0 | 48.3 | 48.2 |
| Human Evaluation | - | - | - | - | 58.6 |

### Ablation Study (Effect of Visual Perception)

| Video Input | Detailed Description | E-Commerce | Knowledge Query | Overall SR |
|---|---|---|---|---|
| ✗ | ✗ | 2.6 | 5.4 | 4.4 |
| ✗ | ✓ | 13.0 | 39.1 | 23.6 |
| ✓ | ✗ | 38.2 | 75.0 | 48.2 |

### Key Findings

- **Even the strongest agents are far from perfect**: Gemini-2.5 Pro, as the best-performing model, achieves only 48.2% SR, while human evaluation reaches only 58.6% (partly due to inherent task difficulty), highlighting substantial room for improvement.
- **Raw video substantially outperforms text descriptions**: Directly providing video input yields more than double the performance compared to first generating text descriptions via a VLM (48.2% vs. 23.6%), indicating that visual grounding must originate from raw visual signals.
- **Error analysis**: 36% of failures stem from object misidentification, 18% from temporal/action misunderstanding, and 16% from cross-modal retrieval failures, confirming that visual perception is the primary bottleneck for current agents.
- **E-commerce and map tasks are most challenging**: These task types require fine-grained visual recognition and spatial understanding, where current agents perform worst.

## Highlights & Insights

- **A pioneering physical-digital world bridging benchmark**: Ego2Web fills the gap in existing web agent evaluation regarding the "visual perception → online action" pipeline. This design reflects the real-world use cases of future AR/intelligent assistants, demonstrating considerable foresight.
- **Ego2WebJudge evaluation framework**: An 84% human agreement rate establishes it as a reliable automatic evaluation tool, avoiding the enormous cost of human evaluation in live web environments. This framework is transferable to other agent tasks requiring online evaluation.
- **Quantitative revelation of the visual perception bottleneck**: Experiments clearly demonstrate the bottleneck at each step of the "perceive → understand → act" pipeline, providing concrete directions for future research.

## Limitations & Future Work

- The dataset scale is relatively limited and does not yet cover all everyday web task scenarios (e.g., social media operations, calendar management).
- The benchmark relies on live web environments, and website changes may affect evaluation reproducibility.
- Only existing general-purpose agents have been evaluated; no dedicated agent architectures tailored to Ego2Web task characteristics have been designed.
- Future work could extend to multi-turn dialogue scenarios (e.g., follow-up requests from users after watching a video) and multimodal web interactions (e.g., voice commands combined with visual perception).

## Related Work & Insights

- **vs. WebArena**: WebArena focuses on purely web-side task execution without real-world visual input; Ego2Web introduces the complete pipeline from egocentric video to web action.
- **vs. Ego4D**: Ego4D is a pure video understanding benchmark with no online action execution; Ego2Web utilizes Ego4D video data but requires agents to complete tasks on real websites.
- **vs. Mind2Web**: Mind2Web uses static web page screenshots, whereas Ego2Web evaluates agents in live web environments, more closely reflecting real-world application scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First benchmark bridging egocentric video and web agent execution, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model evaluation + input ablation + error analysis, though comparisons across more agent architectures are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, precise problem definition, and detailed description of the data pipeline.
- Value: ⭐⭐⭐⭐⭐ Significant contribution to the AI agent and embodied intelligence communities.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](../../ICLR2026/llm_agent/st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [\[CVPR 2026\] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)
- [\[AAAI 2026\] Prune4Web: DOM Tree Pruning Programming for Web Agent](../../AAAI2026/llm_agent/prune4web_dom_tree_pruning_programming_for_web_agent.md)
- [\[ICLR 2026\] VideoMind: A Chain-of-LoRA Agent for Temporal-Grounded Video Reasoning](../../ICLR2026/llm_agent/videomind_a_chain-of-lora_agent_for_temporal-grounded_video_reasoning.md)
- [\[ICLR 2026\] VideoMind: A Chain-of-LoRA Agent for Temporal-Grounded Video Understanding](../../ICLR2026/llm_agent/videomind_a_chain-of-lora_agent_for_temporal-grounded_video_understanding.md)

<!-- RELATED:END -->
