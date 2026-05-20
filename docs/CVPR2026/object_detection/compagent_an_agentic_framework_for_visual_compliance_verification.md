---
title: >-
  [Paper Note] CompAgent: An Agentic Framework for Visual Compliance Verification
description: >-
  [CVPR 2026][Object Detection][Visual Compliance Verification] This paper proposes CompAgent, the first agentic framework for visual compliance verification. A Planning Agent dynamically selects visual tools (object detec…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "Visual Compliance Verification"
  - "Agentic Framework"
  - "Tool-Augmented Reasoning"
  - "Content Moderation"
  - "MLLM"
date: 2026-05-08
content_hash: a3f47dec74271db5
---

# CompAgent: An Agentic Framework for Visual Compliance Verification

**Conference**: CVPR 2026
**arXiv**: [2511.00171](https://arxiv.org/abs/2511.00171)  
**Code**: None  
**Area**: Object Detection / Content Safety
**Keywords**: Visual Compliance Verification, Agentic Framework, Tool-Augmented Reasoning, Content Moderation, MLLM

## TL;DR

This paper proposes CompAgent, the first agentic framework for visual compliance verification. A Planning Agent dynamically selects visual tools (object detection, face analysis, NSFW detection, etc.) based on compliance policies, while a Compliance Verification Agent integrates image content, tool outputs, and policy context for multimodal reasoning. Without any training, CompAgent surpasses the previous SOTA by 10% on UnsafeBench, achieving 76% F1.

## Background & Motivation

Visual content compliance verification is a critically important yet underexplored problem in the vision community:

**Pressing Practical Demand**: Regulations ranging from GDPR to Ofcom require visual content to meet compliance standards, with streaming platforms facing fines of up to $23 million for violations. Content compliance involves detecting harmful objects, inappropriate gestures, explicit content, and more, and continues to evolve across regions, cultures, and industries.

**Fundamental Limitations of Existing Approaches**:
- **Dedicated Classifiers**: Require expensive annotated data and must be retrained whenever policies change, resulting in poor generalization. LlavaGuard achieves F1=0.91 on its own dataset but drops to 0.66 on UnsafeBench.
- **Direct MLLM Prompting**: While MLLMs possess broad knowledge, they struggle with fine-grained visual detail reasoning and structured compliance rule application. The best zero-shot MLLM (Llama 4 Maverick) achieves only 0.55 F1 on LlavaGuard.

**The Agentic Gap**: Despite the flourishing of agentic methods in other domains, no agentic framework specifically targeting visual compliance verification has been proposed.

CompAgent's approach: rather than training a dedicated model or relying solely on prompt engineering, it decomposes compliance verification into modular steps via a **tool-augmented agentic architecture** — combining dynamic tool selection planning with multimodal evidence fusion reasoning.

## Method

### Overall Architecture

CompAgent consists of three core components:

1. **Planning Agent**: Parses compliance policies and dynamically selects appropriate visual tools to collect evidence.
2. **Tool Suite**: A modular collection of off-the-shelf tools (object detection, face detection, OCR, NSFW detection, etc.).
3. **Compliance Verification Agent (CVAgent)**: Integrates the image, tool outputs, and policy context to render the final verdict.

### Key Designs

1. **Planning Agent — ReAct-Loop Tool Orchestration**: Built on the ReAct (Reasoning and Acting) framework, implementing a think–act–observe loop. At each step $t$, the agent maintains a state $s_t = \{I, P, E_t\}$ (image, policy, accumulated evidence), reasons about which policy clauses remain unverified, selects a tool $a_t \in T \cup \{\text{CONCLUDE}\}$, executes the tool to obtain an observation, and updates the evidence:

    $E_{t+1} = E_t \cup (\text{thought}_t, a_t, o_t)$

   A key characteristic is that tool selection **does not rely on a fixed routing table or a learned policy**; instead, the LLM reasons in context based on three factors: (1) which clauses in policy $P$ still lack supporting evidence; (2) the capability and limitation descriptions of each tool; and (3) the evidence $E_t$ already collected. For example, an age-restriction policy triggers face detection, while a text-violation policy prioritizes OCR. The implementation uses LangGraph with Claude Sonnet 3.5 v2, with a maximum of 10 reasoning steps.

2. **Modular Tool Suite**: Covers the evidence types required by common compliance policies:

    - **Summarization Tool**: Generates scene descriptions.
    - **Content Detection Tools**: Face detection (age/expression/emotion), object detection (bounding boxes + confidence scores), text detection (OCR), and content moderation (unsafe categories + severity).
    - **Specialized Compliance Tools**: LlavaGuard (safety rating + violation categories + rationale), Safe-CLIP (zero-shot detection of seven toxic content categories), and ICM Assistant (template-based safety assessment).

   The tool suite is fully modular — tools can be added, removed, or replaced without retraining. The agent treats each tool as a black-box evidence source.

3. **CVAgent — Multimodal Evidence Fusion and Decision**: Upon receiving CONCLUDE, CVAgent receives the complete state $s_T = \{I, P, E_T\}$ and systematically: (1) directly inspects the image; (2) reviews each tool output, weighing confidence and cross-tool consistency; (3) maps the combined evidence to specific policy clauses; and (4) synthesizes an overall assessment. Outputs include a binary Safe/Unsafe rating, violation categories, and a rationale linking evidence to policy clauses.

   **Division of Labor with the Planning Agent**: The Planning Agent determines **what evidence to collect** (a text-only LLM suffices), while CVAgent determines **how to interpret the evidence** (requiring the visual capabilities of an MLLM to directly examine the image).

### Loss & Training

CompAgent is entirely **training-free**, requiring neither annotated data nor fine-tuning, making it highly adaptable to continuously evolving compliance policies. This is a key advantage over fine-tuning approaches such as LlavaGuard, which require policy-specific annotated data.

## Key Experimental Results

### Main Results

| Method | Type | LlavaGuard F1 | UnsafeBench F1 | Notes |
|--------|------|--------------|----------------|-------|
| Claude Sonnet 3.5 v2 | Zero-shot | 0.61 | 0.54 | Best zero-shot single model |
| Llama 4 Maverick | Zero-shot | 0.55 | 0.71 | Zero-shot |
| LlavaGuard (dedicated policy) | Fine-tuned | 0.91 | 0.66 | Strong on own data, large cross-dataset drop |
| Safe-CLIP | Fine-tuned | 0.36 | 0.59 | Zero-shot toxic detection |
| Category-based Routing | Routing | 0.61 | 0.63 | Fixed-routing baseline |
| **CompAgent** | **Agentic** | **0.93** | **0.76** | **Best on both datasets** |

CompAgent achieves F1=0.93 on the LlavaGuard dataset (surpassing fine-tuned LlavaGuard at 0.91) and F1=0.76 on UnsafeBench (exceeding the previous SOTA by 10%), **without any training data**.

### Ablation Study

| Configuration | UnsafeBench F1 | Notes |
|---------------|----------------|-------|
| No tools (direct MLLM) | 0.54 | Lacks fine-grained visual evidence |
| Fixed tool routing | 0.63 | Static assignment lacks flexibility |
| Without Planning Agent | Lower | Tool selection insufficiently targeted |
| Without CVAgent (Planning Agent decides directly) | Lower | Lacks multimodal evidence fusion |
| **Full CompAgent** | **0.76** | Dynamic orchestration + evidence fusion is optimal |

Analysis of decision trajectories reveals 95 distinct tool-usage patterns on the LlavaGuard dataset and 147 on UnsafeBench, demonstrating that the framework genuinely adapts dynamically to diverse compliance requirements.

### Key Findings

- **Zero-shot MLLMs are insufficient**: Even the strongest MLLMs used directly cannot meet compliance verification requirements, highlighting the indispensability of structured tool augmentation.
- **Fine-tuned models generalize poorly**: LlavaGuard achieves F1=0.91 on its own data but drops to 0.66 cross-dataset, exposing the fragility of training on policy-specific data.
- **Core advantage of the agentic approach**: Dynamic tool selection + multi-source cross-validation of evidence + flexible adaptation without training.

## Highlights & Insights

- **The first agentic framework for visual compliance verification**, opening a new research direction.
- The result of **zero training yet surpassing fine-tuned models** is remarkable: it demonstrates that in policy-volatile scenarios like compliance verification, agentic methods are more practical than fine-tuning.
- The **separation of Planning Agent and CVAgent** is an elegant design: evidence collection (where a cheaper LLM suffices) and evidence adjudication (requiring an MLLM to inspect the image) are decoupled.
- The modular design of the tool suite makes the system easy to extend and adapt to new compliance requirements.

## Limitations & Future Work

- The current system processes only single images; extending to video compliance verification (continuous scenes, context dependency) requires further development.
- Reliance on Claude Sonnet 3.5 v2 as the backbone incurs high inference costs and introduces dependency on a closed-source model.
- The selection and description of tools in the tool suite require manual design, and integrating new tools still necessitates human intervention.
- While F1=0.76 on UnsafeBench represents the state of the art, there remains a substantial gap from perfect performance.
- Detailed analysis of latency and cost (multiple tool calls + LLM inference) is absent.

## Related Work & Insights

- This represents the first application of the ReAct framework to visual compliance, demonstrating the unique value of agentic methods in tasks requiring flexible adaptation.
- Relationship to specialized tools such as NudeNet and Safe-CLIP: CompAgent treats them as evidence sources rather than replacements.
- Inspiration: Other visual tasks requiring policy-driven judgment (e.g., advertisement compliance, medical image review) may benefit from adapting this framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First agentic framework for compliance verification, though ReAct + tool calling itself is not novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparison across two datasets, with thorough ablation and interpretability analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definition is clear and framework description is detailed, though the main text is somewhat lengthy.
- **Value**: ⭐⭐⭐⭐⭐ Extremely high practical value; the training-free adaptation to new policies has direct relevance for industry applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning](../../AAAI2026/object_detection/connecting_the_dots_training-free_visual_grounding_via_agent.md)
- [\[CVPR 2026\] CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection](cd-buffer_complementary_dual-buffer_framework_for_test-time_adaptation_in_advers.md)
- [\[CVPR 2026\] Evaluating Few-Shot Pill Recognition Under Visual Domain Shift](evaluating_few-shot_pill_recognition_under_visual_domain_shift.md)
- [\[CVPR 2026\] PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training](pet-dino_unifying_visual_cues_into_grounding_dino_with_prompt-enriched_training.md)
- [\[CVPR 2026\] UAVGen: Visual Prototype Conditioned Focal Region Generation for UAV-Based Object Detection](uavgen_visual_prototype_conditioned_focal_region_generation_for_uav_based_object_detection.md)

</div>

<!-- RELATED:END -->
