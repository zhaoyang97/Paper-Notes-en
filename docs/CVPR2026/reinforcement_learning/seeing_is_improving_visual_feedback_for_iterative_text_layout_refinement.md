---
title: >-
  [Paper Note] Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement
description: >-
  [CVPR 2026][Reinforcement Learning][visual feedback] VFLM proposes a layout generation framework that leverages visual feedback for iterative refinement. By combining a visually grounded reward model based on OCR accurac…
tags:
  - "CVPR 2026"
  - "Reinforcement Learning"
  - "visual feedback"
  - "text layout"
  - "layout generation"
  - "iterative refinement"
date: 2026-05-08
content_hash: 22de007183fe42ac
---

# Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement

**Conference**: CVPR 2026
**arXiv**: [2603.22187](https://arxiv.org/abs/2603.22187)
**Code**: [https://github.com/FolSpark/VFLM](https://github.com/FolSpark/VFLM)
**Area**: Reinforcement Learning / Multimodal Generation
**Keywords**: visual feedback, text layout, layout generation, reinforcement learning, iterative refinement

## TL;DR

VFLM proposes a layout generation framework that leverages visual feedback for iterative refinement. By combining a visually grounded reward model based on OCR accuracy with reinforcement learning training, the framework enables multimodal large language models to "see" rendered outputs and repeatedly self-correct, achieving substantial improvements in text layout quality over code-only generation approaches.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) are now capable of automatically generating structured layouts from natural language descriptions. A typical approach is to have the model generate code (e.g., HTML/CSS/SVG) representing the layout, which is then rendered into a final image by a graphics engine.

**Limitations of Prior Work**: Existing methods follow a "code-only" paradigm in which the model is entirely "blind" to the rendered visual output. This leads to several critical problems: (1) text may overflow bounding boxes or overlap with other elements, degrading readability; (2) aesthetic factors such as font size and color combinations cannot be guaranteed; (3) once code generation is complete, there is no opportunity for correction, and errors propagate directly to the final output.

**Key Challenge**: The ultimate goal of layout generation is visual readability and aesthetic quality, yet a fundamental gap exists between the optimization objective of existing methods (code correctness) and the actual evaluation criterion (visual quality). Syntactically correct code does not necessarily produce a well-rendered result.

**Goal**: To introduce a visual feedback mechanism that enables the model to "see" rendered outputs, identify problems, and iteratively self-correct, thereby realizing self-improving layout generation.

**Key Insight**: Reformulating layout generation from a "one-shot code generation" task into a "visual observation–reflection–correction" iterative process, and employing reinforcement learning to teach the model to leverage visual feedback for self-improvement.

**Core Idea**: Close the "code → render → evaluate → correct" loop through visual feedback, and use RL training to endow the model with adaptive, reflective generation capability.

## Method

### Overall Architecture

The overall pipeline of VFLM proceeds as follows: (1) given a layout generation task description, the MLLM first generates initial layout code (SVG/HTML, etc.); (2) a graphics engine renders the code into an image; (3) the rendered image is fed back to the model as visual input; (4) the model reflects on issues in the current layout (e.g., text overlap, overflow) based on the visual feedback and generates revised code; (5) steps (2)–(4) are repeated until quality is satisfactory or the maximum number of iterations is reached. The entire iterative process is trained via reinforcement learning.

### Key Designs

1. **Visual Feedback Iterative Refinement**:

    - Function: Enables the model to observe rendered outputs and revise the layout accordingly.
    - Mechanism: At each iteration, the model receives three inputs — the original task description, the current layout code, and the current rendered image. Based on the visual information, the model analyzes problems in the current layout (e.g., "the third line of text overlaps with the title," "text in the lower-right corner overflows the canvas") and generates corrected code. This "observe → reflect → correct" loop can be applied adaptively for multiple rounds.
    - Design Motivation: Human designers also iteratively preview and revise their work. Visual feedback liberates the model from "blind generation," enabling it to detect and fix problems that only become apparent after rendering.

2. **Visually Grounded Reward Model**:

    - Function: Provides reward signals aligned with visual quality for RL training.
    - Mechanism: The reward model considers two dimensions — (a) OCR accuracy: OCR is applied to the rendered image and compared against the original text content to measure text readability and correctness; (b) layout aesthetics score: evaluates the rationality of element placement, alignment, and spatial utilization. The final reward is a weighted combination of these two dimensions. A key design choice is to reward only the final output (i.e., the last iteration), rather than scoring each intermediate step.
    - Design Motivation: OCR accuracy directly measures the practical quality of text layout — if rendered text cannot be correctly recognized, the layout has fundamental problems. Rewarding only the final result incentivizes the model to autonomously determine when to stop iterating.

3. **RL Training for Reflective Generation**:

    - Function: Trains the model to acquire iterative reflection and correction capabilities.
    - Mechanism: Reinforcement learning (likely PPO or a similar algorithm) is used to train the MLLM. During training, the model can perform multiple "generate–render–feedback–correct" cycles, and only the reward from the final output is back-propagated. This trains the model not only to produce high-quality outputs on the first attempt, but also to perform effective corrections based on visual feedback.
    - Design Motivation: Supervised learning alone is insufficient to teach the model "how to correct," as correction strategies depend on the specific type of error encountered. RL allows the model to discover effective correction strategies through trial and error.

### Loss & Training

- **RL Reward Function**: $R = \alpha \cdot R_{\text{OCR}} + \beta \cdot R_{\text{aesthetics}}$, where $R_{\text{OCR}}$ denotes OCR recognition accuracy and $R_{\text{aesthetics}}$ denotes the layout aesthetics score.
- **Delayed Reward Strategy**: Rewards are granted only at the final step of the iterative chain, with no scoring of intermediate steps. This design prevents potentially misleading intermediate rewards from causing the model to terminate iteration prematurely.
- **Training Data**: Data covering a variety of text layout tasks are collected, including poster design, social media graphics, and document typesetting.

## Key Experimental Results

### Main Results

| Method | OCR Accuracy | Layout Quality | Iterative Capability | Category |
|--------|-------------|---------------|----------------------|----------|
| GPT-4V (code-only) | Moderate | Moderate | None | General MLLM |
| Canva-GPT | Good | Good | None | Dedicated layout model |
| Code-only baseline | Low | Low | None | Code paradigm |
| VFLM (1 round) | Good | Good | Strong at first round | Ours |
| VFLM (multi-round) | Best | Best | Effective improvement | Ours |

### Ablation Study

| Configuration | OCR Accuracy | Notes |
|---------------|-------------|-------|
| Full VFLM | Best | Visual feedback + RL + iteration |
| w/o Visual Feedback | Significant drop | Degrades to code-only paradigm |
| w/o RL (SFT only) | Noticeable drop | Lacks iterative correction capability |
| w/o OCR Reward | Readability drops | Increased text overflow/overlap |
| Fixed 1 round (no iteration) | Below multi-round | Cannot correct initial errors |

### Key Findings

- Visual feedback is the most critical factor for performance improvement; removing it degrades the system to a code-only method with substantially lower performance.
- RL training significantly outperforms supervised fine-tuning (SFT) alone, as RL enables the model to learn exploratory correction strategies.
- The benefit of additional iterations exhibits diminishing returns; quality typically stabilizes after 2–3 rounds.
- OCR accuracy as a reward signal contributes most to improvements in text readability.

## Highlights & Insights

- **The visual feedback loop design is both natural and effective**: Embedding the human designer's workflow of "view result → reflect on issues → correct" directly into the model. This paradigm is transferable to any task following a "generate code → render" pipeline, such as web design, presentation generation, and data visualization.
- **The RL strategy of rewarding only the final output** elegantly sidesteps the difficulty of designing rewards for intermediate steps, allowing the model to autonomously learn when to stop iterating. This "process-agnostic, result-driven" strategy is worth borrowing for other multi-step generation tasks.
- **OCR as a readability metric**: Using OCR recognition rate to quantify text layout quality is a practical and objective design choice for the reward signal.

## Limitations & Future Work

- Iterative refinement increases inference time, as each round requires a full render–model-inference cycle; quality and efficiency must be balanced in practical applications.
- The current approach primarily targets text layout tasks; its generalization to more complex graphic design scenarios involving mixed elements (images, charts, etc.) remains to be validated.
- The reward model design (OCR + aesthetics) is relatively straightforward and offers limited support for more complex design specifications such as brand guidelines and accessibility standards.
- Training stability and computational cost of RL present practical challenges.
- Future work could incorporate user feedback within the iterative loop to enable human-in-the-loop collaborative design.

## Related Work & Insights

- **vs. LayoutGPT/LayoutDiffusion**: These methods adopt a one-shot generation paradigm and lack visual feedback and iterative correction capabilities.
- **vs. Self-Refine/Reflexion**: These self-improvement methods primarily operate on textual feedback; VFLM introduces visual-modality feedback, making it better suited for design-oriented tasks.
- **vs. HTML/CSS generation models**: Pure code generation models cannot "see" rendered results; VFLM bridges this gap through visual feedback.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of iterative layout refinement via visual feedback and RL is novel and intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark comparisons with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and motivation is thoroughly articulated.
- Value: ⭐⭐⭐⭐ Code is open-sourced; the work advances the field of automated design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving](../../ACL2026/reinforcement_learning/spiralthinker_latent_reasoning_through_an_iterative_process_with_text-latent_int.md)
- [\[CVPR 2026\] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)
- [\[CVPR 2026\] ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering](reag_reasoning-augmented_generation_for_knowledge-based_visual_question_answerin.md)
- [\[CVPR 2026\] See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs](see_it_say_it_sorted_an_iterative_training-free_framework_for_visually-grounded_.md)
- [\[ICLR 2026\] Understanding and Improving Hyperbolic Deep Reinforcement Learning](../../ICLR2026/reinforcement_learning/understanding_and_improving_hyperbolic_deep_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
