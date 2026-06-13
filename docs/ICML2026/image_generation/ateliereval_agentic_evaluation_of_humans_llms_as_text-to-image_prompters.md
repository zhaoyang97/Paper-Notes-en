---
title: >-
  [Paper Note] AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters
description: >-
  [ICML2026][Image Generation][Text-to-Image] AtelierEval is the first to treat the "prompter" in the text-to-image (T2I) pipeline as the evaluation subject. Using 360 expert tasks, three cognitive task categories…
tags:
  - "ICML2026"
  - "Image Generation"
  - "Text-to-Image"
  - "Prompting Proficiency Evaluation"
  - "Agent-as-a-Judge"
  - "MLLM"
  - "Human-AI Comparison"
date: 2026-05-08
content_hash: 881d2ca28397e4ea
---

# AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters

**Conference**: ICML2026  
**arXiv**: [2605.22645](https://arxiv.org/abs/2605.22645)  
**Code**: The paper indicates the release of tools and data, but the repository URL is not provided in the cache.  
**Area**: Image Generation / T2I Evaluation  
**Keywords**: Text-to-Image, Prompting Proficiency Evaluation, Agent-as-a-Judge, MLLM, Human-AI Comparison  

## TL;DR
AtelierEval is the first to treat the "prompter" in the text-to-image (T2I) pipeline as the evaluation subject. Using 360 expert tasks, three cognitive task categories, and the AtelierJudge agentic evaluator, it quantifies the prompting proficiency of humans and MLLMs, discovering that image-mimicry prompting is often more reliable than pure text-only planning.

## Background & Motivation
**Background**: T2I systems are becoming increasingly powerful. User inputs typically do not go directly into the generative model but are first rewritten into executable prompts by a human prompt engineer or an MLLM middleware. Many commercial systems already use MLLMs as implicit middleware, and advanced creators explicitly use MLLM to decompose scenes, styles, and constraints.

**Limitations of Prior Work**: Mainstream T2I benchmarks usually fix the prompt and evaluate the generative model itself. This ignores the upstream prompter's capability: for the same user intent, the final image quality and constraint satisfaction can vary significantly depending on how a human or MLLM translates it into a prompt.

**Key Challenge**: Current evaluations conflate "the ability of the model to execute a prompt" with "the ability of the prompter to translate intent into a prompt." Furthermore, prompt optimizers often focus on local refinement of existing prompts rather than assessing general translation ability from abstract intent to executable prompts.

**Goal**: This paper aims to establish a unified benchmark to specifically measure the intrinsic proficiency of humans and MLLMs as T2I prompters, evaluating both subjective aesthetic quality and objective constraint satisfaction.

**Key Insight**: The authors formalize prompting proficiency as the capability of a policy $\pi: I \rightarrow p$, where $I$ is the user intent and $p$ is the executable prompt. The T2I backend $M$ is responsible for generating images from the prompt. The evaluation goal is not to determine which model is stronger under a fixed prompt, but whether a prompter's policy can stably translate intent across tasks and backends.

**Core Idea**: Cognitive science-inspired task divisions are used to cover three types of prompting abilities. An agentic evaluator, AtelierJudge, equipped with skill routing and memory retrieval, is then used for simultaneous subjective scoring and objective checklist verification.

## Method
The core contribution of AtelierEval consists of two parts: a prompter-oriented benchmark and a scalable agentic evaluator. The benchmark generates realistic and diagnostic tasks, while AtelierJudge decomposes each prompt-image pair into subjective quality and objective constraints for separate evaluation.

### Overall Architecture
AtelierEval includes 360 expert-designed tasks, with 120 tasks per category, covering Open-ended Creation (OE), Constrained Creation (CO), and Imitation (IM). OE tests the extraction of atmosphere, themes, and styles from abstract, narrative requirements. CO tests the organization of prompts under explicit multiple constraints. IM tests the ability to reverse-engineer a prompt from an image, encoding visual content into text.

Task construction is based on two sets of challenge primitives: semantic understanding (S1 abstract intent, S2 audience intent, S3 implicit style, S4 semantic negation) and constraint realization (C1 attribute binding, C2 spatial relations, C3 quantity, C4 text, C5 hard constraints). Experts combine these primitives into real-world T2I scenarios, using 24 labels covering objects, characters, environments, styles, structures, and themes.

The interaction protocol is strictly unified as single-turn, text-only prompting. Humans input prompts via a simplified Gradio UI, and MLLMs receive the same task descriptions via standard APIs. No real-time image feedback or multi-round refinement is allowed to isolate the "first-pass" translation ability.

### Key Designs
1.  **Three Cognitive Task Categories**:
    - **Function**: Decomposes prompting proficiency into diagnostic dimensions rather than providing a single aggregate score.
    - **Mechanism**: OE corresponds to divergent production, requiring the expansion of abstract or narrative intent into a complete scene. CO corresponds to convergent production, requiring the integration of structured constraints. IM corresponds to cognition, requiring the identification and text-encoding of objects, styles, and spatial relations from target images.
    - **Design Motivation**: T2I prompting involves creative expansion, hard constraint execution, and visual description. Single task types conflate these; the three categories clarify where humans and models excel.

2.  **AtelierJudge's Dual-process Agentic Evaluation**:
    - **Function**: Simultaneously evaluates hard-to-quantify aesthetic/expressive quality and explicitly determinable constraint satisfaction.
    - **Mechanism**: The System 1 branch uses memory-augmented subjective skills to rate dimensions like clarity, creativity, terminology, intent formalization, atmosphere, composition, lighting, and technical flaws on a 1-5 scale. The System 2 branch uses prompt/image paired checklists to verify constraints via QA/VQA.
    - **Design Motivation**: Pure MLLM judges often mistake "beauty" for "constraint satisfaction." Decoupling branches ensures that aesthetically pleasing but non-compliant images are not erroneously rated as overall excellent.

3.  **Memory Retrieval & Skill Routing**:
    - **Function**: Aligns automated scoring with expert calibration and adapts to different task categories.
    - **Mechanism**: Each subjective skill is linked to expert-annotated exemplar memory. During evaluation, text/image embeddings retrieve the top-K similar examples, and scores are generated based on criteria and example rationales. The system filters for safety and then concurrently schedules subjective/objective skills based on task type.
    - **Design Motivation**: Direct MLLM scoring tends to give high scores and fails to distinguish between grades 4 and 5. Similar example retrieval provides a scoring anchor, recovering finer score gradients.

### Loss & Training
This paper does not train a new generative model but designs the evaluation protocol and automated scoring system. Subjective metrics use MAE, Within-1 accuracy, and Spearman $\rho$ for alignment with expert ratings. Objective metrics use checkpoint-level Accuracy and F1. Benchmark results aggregate prompt-side/image-side subjective scores and objective satisfaction rates.

In main experiments, each prompter-task pair generates one natural language prompt. Each prompt generates 4 images per backend, and the top-1 image (by AtelierJudge score) is retained for aggregation. Stability analysis shows that increasing the number of prompts/images results in flat score curves, making the one-prompt, four-image setting reasonable for cost and stability.

## Key Experimental Results

### Main Results
The experiments were conducted in two layers. The first verified if AtelierJudge aligns with expert scores; the second used AtelierEval to compare 8 MLLMs, 48 humans (24 novice, 24 skilled), and 4 T2I backends (nBanana, GI-1, Flux Pro, SDXL).

| Subject | Metric | Key Value | Conclusion |
| :--- | :--- | :--- | :--- |
| Subjective meta-eval, GPT-5.4 | MAE / W1-A / Spearman $\rho$ | 0.33 / 0.95 / 0.81 | Close to human expert $\rho=0.83$, higher than base $\rho=0.55$ |
| Objective meta-eval, GPT-5.4 | Overall Acc / F1 | 95.5% / 93.9% | High reliability for both prompt and image checklists |
| Prompt objective, skilled human | Avg Prompt Obj. | 80.6% | Skilled humans are significantly better at encoding constraints |
| Image objective, skilled human | Avg Image Obj. | 76.7% | Skilled human prompts yield the highest image-side satisfaction |
| nBanana backend, skilled human | Obj. | 84.9% | Strong middleware with skilled human achieves peak objective performance |
| T0 MLLMs vs novice humans | Comprehensive | T0 MLLMs > Novices | MLLMs significantly raise the prompting baseline for average users |

### Ablation Study
| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Zero-shot judge | MAE 0.72, W1-A 0.64, $\rho=0.56$ | Direct scoring is overly optimistic and lacks discrimination |
| Fixed Few-shot | MAE 0.55, W1-A 0.81, $\rho=0.68$ | Provides a scale but lacks task-specific calibration |
| Random Retrieval | MAE 0.61, W1-A 0.75, $\rho=0.62$ | Random examples are unstable and introduce noise |
| Similarity Retrieval | MAE 0.34, W1-A 0.93, $\rho=0.79$ | Semantically similar examples maximize expert alignment |
| K=1 | MAE 0.56, W1-A 0.83, $\rho=0.63$ | Single example is insufficient for complex dimension calibration |
| K=3 | MAE 0.34, W1-A 0.93, $\rho=0.79$ | Optimal retrieval number adopted by the paper |
| K=4 | MAE 0.35, W1-A 0.91, $\rho=0.78$ | More examples introduce context noise, diminishing returns |

### Key Findings
- Memory retrieval in AtelierJudge is critical; it is not just about using a stronger MLLM. Similar exemplars significantly reduce MAE and boost Spearman correlation from 0.56 to 0.79.
- Strong T2I middleware compresses the subjective image quality gap between different prompters. Backends like GI-1 and nBanana make images look generally good, but this does not equate to correct constraint satisfaction.
- A "Constraint Paradox" appears in Constrained Creation: on strong middleware like GI-1, direct task description inputs yield 69.6% objective satisfaction, while external MLLM reasoning drops it to 45%-49%, suggesting conflicting reasoning systems.
- Skilled humans remain superior in writing hard-constraint prompts, especially in CO tasks. MLLMs have strong vocabulary but may not adapt to a specific backend's internal rewriting mechanism.
- Imitation tasks reveal a future direction: with reference images, MLLMs can identify fine-grained visual structures better than humans. T0 MLLMs outperform skilled humans on GI-1 in this category, supporting image-augmented prompting.

## Highlights & Insights
- The shift of the evaluation target from "generative model" to "prompter" is a significant definition. Generation failure is often not just the model's fault but a failure to translate intent into an executable prompt.
- The tripartite task design is highly explanatory: OE for creativity, CO for constraints, and IM for visual encoding. Different error patterns guide prompt education and agent design.
- The subjective/objective decoupling of AtelierJudge avoids common evaluation traps. High aesthetic quality often hides "high-quality hallucinations" where constraints are missed.
- "Mimicry over planning" is a key insight. Rather than purely text-based planning for complex scenes, agents should observe visual examples and transfer structures to the target requirement.

## Limitations & Future Work
- Human experiments are clustered around active T2I users, leading to demographic bias. AtelierJudge's expert memory may inherit these aesthetic and cultural preferences.
- The benchmark is limited to single-turn text-to-image, excluding multi-round iteration, visual feedback, tool use, or search-based prompt optimization.
- There is no objective unified metric for task difficulty. While challenge primitives are balanced, the combination difficulty is not explicitly modeled.
- Automated evaluators should not yet be the sole basis for high-stakes creative or labor assessments.
- Future work could extend to image-augmented prompting, human-LLM collaboration, and unified multimodal models that serve as both prompter and generator.

## Related Work & Insights
- **vs. Fixed-prompt T2I benchmarks**: Traditional benchmarks test model execution; AtelierEval tests upstream prompt translation. They are complementary.
- **vs. Prompt optimization**: Optimization often refines for a specific model; AtelierEval focuses on general intent-to-prompt capability at the creation entry point.
- **vs. CLIPScore / VQA-based evaluators**: Traditional automated metrics correlate poorly with complex spatial relations and aesthetics. AtelierJudge improves explainability via skill decomposition.
- **vs. MLLM-as-a-Judge**: Standard MLLM judges suffer from self-preference and score inflation; AtelierJudge reduces bias via retrieval memory and subjective/objective decoupling.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ High. Treating the prompter as an independent subject is groundbreaking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High. 360 tasks, 8 MLLMs, 48 humans, and 4 backends with extensive meta-evaluation.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear logic, though dense with model names and tables requiring careful reading.
- **Value**: ⭐⭐⭐⭐⭐ High. Directly useful for T2I tools, prompt education, and automated evaluation development.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] \[Paper Insight\] Agentic Retoucher for Text-To-Image Generation](../../CVPR2026/image_generation/agentic_retoucher_for_texttoimage_generation.md)
- [\[ICML 2026\] WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation](wise_a_world_knowledge-informed_semantic_evaluation_for_text-to-image_generation.md)
- [\[CVPR 2026\] Vinedresser3D: Agentic Text-guided 3D Editing](../../CVPR2026/image_generation/vinedresser3d_agentic_text-guided_3d_editing.md)
- [\[ICML 2026\] Conformal Reliability: A New Evaluation Metric for Conditional Generation](conformal_reliability_a_new_evaluation_metric_for_conditional_generation.md)
- [\[ICML 2026\] SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes](scenesmith_agentic_generation_of_simulation-ready_indoor_scenes.md)

</div>

<!-- RELATED:END -->
