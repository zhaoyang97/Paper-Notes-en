---
title: >-
  [Paper Note] OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward
description: >-
  [ACL 2026][Code Intelligence][diagram code generation] This paper proposes OmniDiagram, a unified diagram code generation framework covering three languages (LaTeX/Mermaid/PlantUML) and three tasks (diagram-to-code…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "diagram code generation"
  - "visual question answering reward"
  - "reinforcement learning"
  - "unified framework"
  - "multimodal"
date: 2026-05-08
content_hash: 0ee91fbb0e0129ae
---

# OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward

**Conference**: ACL 2026
**arXiv**: [2604.05514](https://arxiv.org/abs/2604.05514)  
**Code**: [GitHub](https://github.com/Haoyue-Yang/OmniDiagram)  
**Area**: Code Intelligence / Multimodal Code Generation
**Keywords**: diagram code generation, visual question answering reward, reinforcement learning, unified framework, multimodal

## TL;DR

This paper proposes OmniDiagram, a unified diagram code generation framework covering three languages (LaTeX/Mermaid/PlantUML) and three tasks (diagram-to-code, diagram editing, text-to-code). It introduces the Viva (Visual Interrogation Verifies All) reward mechanism based on visual question answering to guide RL training, achieving state-of-the-art performance on multiple benchmarks.

## Background & Motivation

**Background**: The paradigm of programmable diagram generation is evolving rapidly and plays a critical role in structured visualization. Multimodal large language models have made it possible to directly process unstructured diagrams (e.g., PNG raster images) and generate executable code. However, existing methods are typically limited to a single task or a small number of programming languages.

**Limitations of Prior Work**: (1) StarFlow supports only JSON output and ignores diverse diagram languages; JanusCoder attempts to unify text-to-code and diagram-to-code but relies solely on SFT, limiting visual alignment and code execution robustness. (2) Methods that combine RL with visual feedback (e.g., MSRL, RLRF) target only specific image-to-code tasks and lack cross-task flexibility. (3) Existing visual feedback approaches either rely on fixed prompt templates (constrained by the evaluation model's capability and susceptible to prompt hacking) or compute global visual similarity (biased toward surface structural similarity while neglecting fine-grained details).

**Key Challenge**: Diagram code generation must simultaneously ensure code logical correctness and post-rendering visual fidelity, yet existing RL reward mechanisms struggle to uniformly verify critical structural details across heterogeneous tasks — the structural diversity of Text-to-Code precludes a single reference image, and the non-bijectivity of Diagram-to-Code means different code can produce visually identical outputs.

**Goal**: To build a unified framework covering multiple diagram languages and task modalities, and to design an RL reward mechanism capable of uniformly evaluating visual fidelity across tasks.

**Key Insight**: Drawing inspiration from the metacognitive review process humans employ in complex construction tasks — assessing structural and semantic constraints through targeted questioning rather than holistic similarity judgments.

**Core Idea**: The Viva (Visual Interrogation Verifies All) mechanism — task-specific visual questions are generated offline for each training sample; during online training, a reward model answers these questions based on rendered images to evaluate visual fidelity, providing fine-grained intermediate scoring feedback.

## Method

### Overall Architecture

Data synthesis (M32Diagram, 196k samples, 3×3 task–language matrix) → SFT stage (establishing foundational diagram code generation capability) → Viva-driven RL stage (GRPO optimization with Viva visual QA reward + format reward → iterative refinement of visual fidelity).

### Key Designs

1. **Viva (Visual Interrogation Verifies All) Reward Mechanism**:

    - **Function**: Provides cross-task unified, fine-grained, instance-specific visual fidelity feedback.
    - **Mechanism**: Decouples question generation from answer verification. Offline stage: GPT-4.1-mini generates multiple targeted visual questions per sample (designed so that "Yes" corresponds to correctness). Online stage: each rollout's code is executed and rendered, and Qwen3-VL-32B serves as the reward model to answer the questions based on the rendered image. The final Viva reward is the mean score across all questions, combined with a format reward: $R_i = \alpha \cdot R_{\text{Viva}} + (1-\alpha) \cdot R_{\text{fmt}}$ ($\alpha=0.9$); candidates that fail to render receive a score of 0.
    - **Design Motivation**: Question-driven verification simulates human review by focusing on logical consistency rather than strict global imitation, thereby rewarding more diverse rollouts. Intermediate scoring provides a smoother feedback signal. Variance analysis demonstrates that multi-question aggregation effectively reduces reward noise.

2. **M32Diagram Large-Scale Dataset**:

    - **Function**: Provides the first large-scale diagram code generation dataset covering a 3×3 task–language matrix.
    - **Mechanism**: Employs a top-down scenario-driven synthesis pipeline (topic → scenario → structured data → code–image pairs) using Gemini-2.5-Flash. After rigorous error-correction loops and visual verification, 165k high-quality samples are selected from 300k candidates; combined with 31k open-source samples, the total is 196k. An additional 77k reasoning-augmented samples are included.
    - **Design Motivation**: Addresses the scarcity of diagram code generation datasets. Each language covers approximately 15 diagram types; a perceptual-hash-based hierarchical clustering strategy is used to balance the difficulty and topological complexity of SFT and RL training sets.

3. **Two-Stage SFT-to-RL Training Pipeline**:

    - **Function**: First establishes foundational capability, then refines visual fidelity through RL.
    - **Mechanism**: The SFT stage uses standard next-token prediction to establish multi-format diagram code generation foundations. The RL stage uses GRPO ($G=4$ candidates) with Viva reward computed online; non-renderable rollouts are penalized.
    - **Design Motivation**: Ablation experiments demonstrate that pure RL (without SFT) leads to mode collapse — the model generates only Mermaid code while ignoring specific instructions. SFT is a necessary prerequisite for establishing comprehensive diagram generation capability.

### Loss & Training

SFT stage: standard cross-entropy loss, 8× H800 GPUs, global batch size 32, 2 epochs. RL stage: GRPO optimization (Equations 4–5), $G=4$ candidates, $\alpha=0.9$, global batch size 128, using ms-swift and EasyR1 frameworks. The theoretical stability of the Viva reward is established via variance analysis: multi-dimensional aggregation attenuates the uncertainty contributed by individual VQA responses.

## Key Experimental Results

### Main Results

| Model | M32Bench D2C $S_{vis}$ | M32Bench Edit $S_{pres}$/$S_{task}$ | VisPlot Mermaid $S_{vis}$/$S_{task}$ |
|--------|------|------|------|
| Qwen2.5-VL-72B | 55.0 | 36.8/54.0 | 31.0/46.0 |
| Qwen3-VL-32B | 58.0 | 45.6/51.8 | 40.4/55.1 |
| OmniDia-3B (RL) | 72.2 | 59.0/64.8 | 49.4/64.5 |
| OmniDia-7B (RL) | **75.5** | **57.2/65.2** | **51.0/66.9** |
| Gemini-3-Flash | 73.6 | 77.8/82.0 | 58.4/80.2 |

### Ablation Study

| Configuration | Key Metric | Note |
|------|---------|------|
| Pure RL (no SFT) | Exec 30.2% | Mode collapse; generates only Mermaid |
| Pure SFT (no RL) | Exec 88.6% | Full foundational capability but lower visual fidelity |
| Full pipeline (SFT+RL) | Exec **93.0%** | Complementary stages achieve optimal performance |
| With reasoning traces | Diagram editing improves; other tasks decline | Reasoning context may distract attention |
| Smaller reward model (30B-A3B) | Minimal performance gap | Offline questions are more critical than reward model scale |

### Key Findings
- The 3B model (OmniDia-3B) outperforms the 72B open-source model (Qwen2.5-VL-72B), demonstrating the substantial leverage of data and training strategy.
- The RL stage significantly improves execution rate (SFT 88.6% → RL 93.0%) by penalizing non-renderable outputs.
- Viva is insensitive to reward model scale, suggesting that the offline-generated visual questions provide the critical visual focus.
- The effect of reasoning traces is task-dependent: beneficial for diagram editing (enhancing instruction analysis) but potentially detrimental for other tasks.

## Highlights & Insights
- The Viva mechanism's philosophy of "every sample deserves careful interrogation" elegantly resolves the unified reward problem across heterogeneous tasks.
- The decoupled design of question generation and answer verification is elegant — offline question generation reduces online overhead while preserving instance specificity.
- The result that a 3B model surpasses a 72B model powerfully demonstrates the importance of focused training data and strategy.
- The reward model scale experiment reveals a counterintuitive yet important finding: what matters is *what to ask*, not *who answers*.

## Limitations & Future Work
- The visual/format weight $\alpha$ in the Viva reward is fixed at 0.9; task-adaptive adjustment may yield further improvements.
- Only the GRPO algorithm is employed; comparative experiments with alternative RL paradigms such as PPO and DPO are absent.
- Data synthesis and evaluation rely on external models (Gemini-2.5-Flash, GPT-4.1), incurring substantial computational cost.
- More complex diagram types (e.g., 3D plots, interactive charts) are not addressed.

## Related Work & Insights
- **vs JanusCoder**: JanusCoder relies solely on SFT; OmniDiagram substantially improves visual fidelity and execution rate through Viva RL.
- **vs RLRF/MSRL**: These methods use global visual similarity or fixed templates as rewards; OmniDiagram's Viva provides finer-grained and more robust feedback.
- **vs VisCoder2**: VisCoder2 is based on a code-specialized LLM (Qwen-Coder); OmniDiagram achieves larger gains starting from a general-purpose VLM.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The Viva visual QA reward mechanism is novel and the 3×3 unified framework is valuable, though the overall approach builds on the established paradigm of GRPO with visual feedback.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Ablation studies are comprehensive (training strategy, reasoning traces, reward model scale); multi-benchmark, multi-model comparisons are thorough.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear, method descriptions are detailed, and theoretical analysis (reward stability proof) adds depth.
- **Value**: ⭐⭐⭐⭐ The M32Diagram dataset and Viva mechanism are generalizable to other visual code generation scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](../../CVPR2026/code_intelligence/mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)
- [\[CVPR 2026\] GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning](../../CVPR2026/code_intelligence/geotikzbridge_advancing_multimodal_code_generation_for_geometric_perception_and_.md)
- [\[ACL 2026\] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)

</div>

<!-- RELATED:END -->
