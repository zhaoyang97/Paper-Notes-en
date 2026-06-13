---
title: >-
  [Paper Note] OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward
description: >-
  [ACL 2026][Code Intelligence][Diagram Code Generation] This paper proposes OmniDiagram, a unified diagram code generation framework covering LaTeX/Mermaid/PlantUML languages and three tasks: diagram-to-code…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Diagram Code Generation"
  - "VQA Reward"
  - "Reinforcement Learning"
  - "Unified Framework"
  - "Multimodal"
date: 2026-05-08
content_hash: 1921d56e98c94985
---

# OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.05514](https://arxiv.org/abs/2604.05514)  
**Code**: [GitHub](https://github.com/Haoyue-Yang/OmniDiagram)  
**Area**: Code Intelligence / Multimodal Code Generation  
**Keywords**: Diagram Code Generation, VQA Reward, Reinforcement Learning, Unified Framework, Multimodal

## TL;DR

This paper proposes OmniDiagram, a unified diagram code generation framework covering LaTeX/Mermaid/PlantUML languages and three tasks: diagram-to-code, diagram editing, and text-to-code. It introduces the Viva reward mechanism based on Visual Question Answering (VQA) to guide RL training, achieving SOTA across multiple benchmarks.

## Background & Motivation

**Background**: The paradigm of programmable diagram generation is evolving rapidly, playing a crucial role in structured visualization. Multimodal Large Language Models (MLLMs) enable direct processing of unstructured diagrams (e.g., PNG raster format) to generate executable code. However, existing methods are typically limited to single tasks or specific programming languages.

**Limitations of Prior Work**: (1) StarFlow only supports JSON output, ignoring diverse diagram languages; JanusCoder attempts to unify text-to-code and diagram-to-code but relies solely on SFT, limiting visual alignment and code execution robustness. (2) Methods combining RL with visual feedback (e.g., MSRL, RLRF) target specific image-to-code tasks and lack cross-task flexibility. (3) Existing visual feedback methods either use fixed prompt templates (limited by evaluator model capability and prone to prompt hacking) or calculate global visual similarity (biased towards surface structure while ignoring fine-grained details).

**Key Challenge**: Diagram code generation must ensure both code logical correctness and rendered visual fidelity. Existing RL reward mechanisms struggle to uniformly verify critical structural details in heterogeneous tasks—text-to-code structural diversity precludes a single reference image, and diagram-to-code non-bijectivity means different code can produce visually identical outputs.

**Goal**: Construct a unified framework covering various diagram languages and task modalities, and design an RL reward mechanism capable of uniformly evaluating visual fidelity across tasks.

**Key Insight**: Borrow from human meta-cognitive review mechanisms in complex construction tasks—systematically checking structural and semantic constraints through targeted questions rather than overall similarity judgments.

**Core Idea**: The Viva (Visual Interrogation Verifies All) mechanism—generating targeted visual questions offline for each training sample and having the reward model answer these questions based on rendered images during online RL to evaluate visual fidelity, providing fine-grained intermediate score feedback.

## Method

### Overall Architecture

Data Synthesis (M32Diagram with 196k samples, 3×3 task-language matrix) → SFT Stage (establishing foundational diagram code generation capabilities) → Viva-driven RL Stage (GRPO optimization, Viva VQA reward + format reward → iterative refinement of visual fidelity).

### Key Designs

1.  **Viva (Visual Interrogation Verifies All) Reward Mechanism**:
    *   **Function**: Provides cross-task unified, fine-grained, instance-specific visual fidelity feedback.
    *   **Mechanism**: Decouples question generation from answer verification. Offline phase: GPT-4.1-mini generates targeted visual questions for each sample (designed so "Yes" corresponds to correctness). Online phase: Each rollout code is rendered, and Qwen3-VL-32B acts as the reward model to answer questions based on the image. The final Viva reward is the average score of all questions, combined with a format reward: $R_i = \alpha \cdot R_{\text{Viva}} + (1-\alpha) \cdot R_{\text{fmt}}$ ($\alpha=0.9$). Candidates with rendering failures receive a score of 0.
    *   **Design Motivation**: Question-driven verification simulates human review, focusing on logical consistency rather than strict global imitation, rewarding more diverse rollouts. Intermediate scoring provides smoother feedback signals. Variance analysis proves that multi-question aggregation effectively reduces reward noise.

2.  **M32Diagram Large-scale Dataset**:
    *   **Function**: Provides the first large-scale diagram code generation dataset covering a 3×3 task-language matrix.
    *   **Mechanism**: Employs a top-down scenario-driven synthesis method (topic → scenario → structured data → code-image pairs) using Gemini-2.5-Flash. After rigorous error correction loops and visual verification, 165k high-quality samples were filtered from 300k candidates, totaling 196k with 31k open-source data. There are also 77k reasoning-enhanced samples.
    *   **Design Motivation**: Addresses the scarcity of diagram code generation datasets. Each language covers about 15 diagram types. A hierarchical clustering strategy based on perceptual hashing balances difficulty and topological complexity between SFT and RL sets.

3.  **SFT-to-RL Two-stage Training Pipeline**:
    *   **Function**: Establishes foundational capabilities first, then refines visual fidelity through RL.
    *   **Mechanism**: The SFT stage uses standard next-token prediction to build multi-format diagram code generation foundations. The RL stage uses GRPO (G=4 candidates) with Viva rewards calculated online, penalizing non-renderable rollouts.
    *   **Design Motivation**: Ablation studies show pure RL (without SFT) leads to mode collapse—the model generates only Mermaid code while ignoring specific instructions. SFT is an essential prerequisite for comprehensive diagram generation capabilities.

### Loss & Training

SFT Stage: Standard cross-entropy loss, 8×H800 GPUs, global batch 32, 2 epochs. RL Stage: GRPO optimization, G=4 candidates, $\alpha=0.9$, global batch 128, utilizing ms-swift and EasyR1 frameworks. The theoretical stability of the Viva reward is proven through variance analysis: multi-dimensional aggregation decays the uncertainty impact of individual VQA steps.

## Key Experimental Results

### Main Results

| Model | M32Bench D2C $S_{vis}$ | M32Bench Edit $S_{pres}$/$S_{task}$ | VisPlot Mermaid $S_{vis}$/$S_{task}$ |
| :--- | :--- | :--- | :--- |
| Qwen2.5-VL-72B | 55.0 | 36.8/54.0 | 31.0/46.0 |
| Qwen3-VL-32B | 58.0 | 45.6/51.8 | 40.4/55.1 |
| OmniDia-3B (RL) | 72.2 | 59.0/64.8 | 49.4/64.5 |
| OmniDia-7B (RL) | **75.5** | **57.2/65.2** | **51.0/66.9** |
| Gemini-3-Flash | 73.6 | 77.8/82.0 | 58.4/80.2 |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Pure RL (No SFT) | Exec 30.2% | Mode collapse, only Mermaid generated |
| Pure SFT (No RL) | Exec 88.6% | Complete base capability but lower visual fidelity |
| Full Pipeline (SFT+RL) | Exec **93.0%** | Two stages complement for optimal results |
| Added Reasoning Trace | Diagram Edit Gain | Diagram editing improved, others decreased; context may distract |
| Small Reward Model (30B-A3B) | Minimal Gap | Offline questions are more critical than reward model size |

### Key Findings
- The 3B model (OmniDia-3B) outperforms the 72B open-source model (Qwen2.5-VL-72B), demonstrating significant leverage from data and training strategy.
- The RL stage significantly improves execution rates (SFT 88.6% → RL 93.0%) because RL penalizes non-renderable outputs.
- Viva is insensitive to reward model size, suggesting that offline-generated visual questions provide crucial visual focus.
- The effect of reasoning traces varies by task: beneficial for diagram editing (enhancing instruction analysis) but potentially detrimental to other tasks.

## Highlights & Insights
- The "every sample deserves careful interrogation" philosophy of the Viva mechanism elegantly solves the unified reward problem for heterogeneous tasks.
- The decoupling of question generation and answer verification is clever—offline generation reduces online overhead while maintaining instance specificity.
- The 3B model's superiority over 72B strongly proves the importance of focused training data and strategy.
- Reward model scale experiments reveal a counterintuitive but important finding: the key lies in "what to ask" rather than "who answers."

## Limitations & Future Work
- The visual/format weight $\alpha$ in Viva reward is fixed at 0.9; task-adaptive adjustment might further optimize performance.
- Only the GRPO algorithm is used; comparative experiments with alternative RL paradigms like PPO/DPO are missing.
- Data synthesis and evaluation rely on external models (Gemini-2.5-Flash, GPT-4.1), which involve high computational costs.
- More complex diagram types (e.g., 3D diagrams, interactive charts) are not yet covered.

## Related Work & Insights
- **vs JanusCoder**: JanusCoder uses only SFT; OmniDiagram significantly improves visual fidelity and execution rate via Viva RL.
- **vs RLRF/MSRL**: These methods use global visual similarity or fixed templates as rewards; OmniDiagram's Viva provides more fine-grained and robust feedback.
- **vs VisCoder2**: VisCoder2 is based on code-specific LLM (Qwen-Coder); OmniDiagram achieves greater gains starting from a general VLM.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The Viva VQA reward mechanism is novel, and the 3×3 unified framework is valuable, though the overall approach builds on existing GRPO+visual feedback paradigms.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive ablation studies (training strategy, reasoning traces, reward model size) and extensive multi-benchmark comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, detailed method description, and theoretical analysis (reward stability proof) provide depth.
- **Value**: ⭐⭐⭐⭐ The M32Diagram dataset and Viva mechanism are generalizable to other visual code generation scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](../../CVPR2026/code_intelligence/mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)
- [\[CVPR 2026\] GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning](../../CVPR2026/code_intelligence/geotikzbridge_advancing_multimodal_code_generation_for_geometric_perception_and_.md)
- [\[ACL 2026\] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)
- [\[ICML 2026\] UniRTL: Unified Code and Graph for Robust RTL Representation Learning](../../ICML2026/code_intelligence/unirtl_unifying_code_and_graph_for_robust_rtl_representation_learning.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)

</div>

<!-- RELATED:END -->
