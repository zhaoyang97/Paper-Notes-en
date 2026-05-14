---
title: >-
  [Paper Note] CodeDance: A Dynamic Tool-integrated MLLM for Executable Visual Reasoning
description: >-
  [CVPR 2026][Multimodal VLM][Executable visual reasoning] This paper proposes CodeDance, which uses executable code as a general-purpose solver for visual reasoning. The MLLM generates code to define, compose…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Executable visual reasoning"
  - "tool integration"
  - "code generation"
  - "reinforcement learning"
  - "emergent behavior"
date: 2026-05-08
content_hash: dc6358bbcae14701
---

# CodeDance: A Dynamic Tool-integrated MLLM for Executable Visual Reasoning

**Conference**: CVPR 2026
**arXiv**: [2512.17312](https://arxiv.org/abs/2512.17312)
**Code**: [CodeDance-VL.github.io](https://CodeDance-VL.github.io)
**Area**: Multimodal VLM
**Keywords**: Executable code reasoning, tool calling, multimodal reasoning, reinforcement learning, emergent behavior

## TL;DR

This paper proposes CodeDance, which uses executable code as a unified medium for visual reasoning. Atomic capabilities are instilled via SFT, and a difficulty-adaptive tool-calling reward (BAT) is applied during RL to enable dynamic tool orchestration and self-verification reasoning. The resulting 7B model surpasses GPT-4o on tasks such as counting, visual search, and chart QA.

## Background & Motivation

Current MLLMs face several core bottlenecks in visual reasoning:

**Information bottleneck of textual CoT**: Traditional Chain-of-Thought relies solely on static textual context and cannot dynamically interact with visual inputs or introduce new observations at intermediate reasoning steps, limiting multi-turn focusing and verification capabilities.

**Non-reproducible closed systems**: OpenAI o3 demonstrates powerful "thinking with images" capabilities, but its internal mechanism is entirely opaque and cannot be studied or reproduced by the research community.

**Open-source approaches rely on fixed schemas**: Existing open-source tool-integration methods (DeepEyes, PixelReasoner, etc.) mostly depend on predefined visual operation templates (e.g., predicting only bounding box coordinates for cropping), offering poor flexibility and limited transferability to new tools and tasks.

**Lack of control over tool-calling timing**: Existing methods do not consider when the model should invoke tools and when it should not, leading to tool spamming on simple questions or insufficient tool use on difficult ones.

The core idea of CodeDance: **replace fixed schemas with executable code**, allowing the model to freely define, compose, and execute code to orchestrate tools, producing verifiable intermediate artifacts (cropped regions, annotated images, computed charts, etc.) for transparent and self-verifiable reasoning.

## Method

### Overall Architecture

CodeDance adopts a **think-execute-feedback loop** as its reasoning unit. Given a multimodal query (text + image), the MLLM alternately produces natural language reasoning and executable code; the code is executed in a sandbox, and the visual artifacts are concatenated back into the context to drive the next reasoning turn. A reasoning trajectory is defined as:

$$\tau = \big((s_{t_1}, a_{t_1}, s'_{t_1}), \ldots, (s_{t_M}, a_{\text{ans}})\big)$$

where state $s_t = (x, o_t, \epsilon_t)$ comprises the original query $x$, the accumulated reasoning trace $o_t$, and the interpreter feedback $\epsilon_t$. The action space includes tool calls (code snippets) and terminal answers.

Training proceeds in two stages: (1) SFT cold-start to teach atomic code capabilities; (2) RL to further optimize the tool-calling strategy.

### Key Designs

1. **Atomic Capability SFT Dataset (34K trajectories)**: Covers three categories of atomic operations—basic image transformations (crop, resize), mathematical computation (measurement, algebra, aggregation), and open-ended visual editing (drawing, annotation, etc.). A **weak-to-strong filtering** strategy is employed: Qwen2.5-VL-7B first performs automatic filtering and difficulty grading, then Qwen2.5-VL-72B cross-validates correctness. Simple questions encourage direct answers (without tool calls), while difficult questions enable multi-turn code reasoning trajectories. This design naturally lays the foundation for adaptive tool calling in subsequent RL.

2. **BAT Reward Mechanism (Balanced Adaptive Tool Calling)**: The RL stage uses GRPO optimization, and the total reward consists of three components:

    $r(\tau) = R_{\text{acc}}(\tau) + R_{\text{format}}(\tau) + R_{\text{BAT}}(\tau)$

   The BAT reward is further decomposed into sequence-level and turn-level components:

    - **Sequence-level adaptive reward $R_{\text{seq}}$**: The core insight is to use the in-group accuracy $\mu_{\text{acc}}$ as a proxy for question difficulty—if most rollouts answer correctly, the question is easy and tool-calling rewards are suppressed; otherwise, exploration is encouraged. The scaling factor is $d = \sigma(\gamma(0.5 - \mu_{\text{acc}})) - \delta$, where $\gamma=4, \delta=0.2$.

    - **Turn-level execution reward $R_{\text{turn}}$**: An immediate penalty of $-0.5$ is applied for code execution failures, accumulated recursively with a discount factor $\beta=0.2$, providing dense corrective signals. Experiments demonstrate that this design effectively prevents entropy collapse.

3. **Emergent Behaviors**: Three categories of behaviors beyond SFT supervision are observed during RL training:

    - **Cross-domain tool transfer**: Bounding box operations learned on chart tasks spontaneously transfer to counting tasks (first localizing all candidate objects with boxes, then verifying and counting).
    - **Novel tool composition**: Combining multiple atomic operations such as pointing, cropping, and zooming to solve verification tasks that no single tool can handle alone.
    - **Unseen code generation**: The model spontaneously invokes OpenCV functions from pretraining knowledge (e.g., `cv2.rectangle` to draw grids for counting assistance), despite such code never appearing in the SFT data.

### Loss & Training

- **SFT stage**: Standard cross-entropy loss training using the SWIFT framework.
- **RL stage**: GRPO + BAT reward using the VeRL framework; maximum of 6 training turns and 10 inference turns.
- Base model: Qwen2.5-VL-7B.

## Key Experimental Results

### Main Results

| Dataset | Metric | CodeDance-7B | Qwen2.5-VL-7B (Base) | GPT-4o | Gain (vs. Base) |
|--------|------|-------------|---------------------|--------|------------|
| CountBench | Acc | 91.2 | 76.5 | 87.9 | ↑19.2% |
| PixmoCount | Acc | 77.1 | 50.4 | - | ↑53.0% |
| V* Bench | Acc | 84.8 | 76.4 | 67.5 | ↑11.0% |
| HR-Bench 4K | Acc | 75.2 | 69.0 | 65.0 | ↑9.0% |
| ChartQA | Acc | 87.5 | 86.3 | 86.7 | ↑1.4% |
| MathVision | Acc | 29.6 | 25.0 | 36.5 | ↑18.4% |
| MathVista | Acc | 70.3 | 68.1 | 63.4 | ↑3.2% |
| WeMath | Acc | 39.6 | 35.4 | 44.2 | ↑11.9% |

The 7B model outperforms GPT-4o on CountBench, V* Bench, HR-Bench, and others, and also surpasses larger models such as Qwen2.5-VL-32B.

### Ablation Study

| Reward Configuration | Description | Effect |
|---------|------|------|
| GRPO (outcome reward only) | Final answer correctness only | Reduces interaction turns but leads to insufficient tool use; poor performance on complex tasks |
| DeepEyes reward | Positive reward for every successful tool call | Tool overuse; inflated turns on simple questions; unstable accuracy |
| BAT without $R_{\text{turn}}$ | Turn-level reward removed | Severe degradation in execution success rate |
| Full BAT (Ours) | Sequence-level + turn-level | Best accuracy + moderate turns + highest execution success rate |

Scalability ablation: SFT data scaling from 5K to 34K yields consistent improvement; model scaling from 3B to 7B yields significant gains; RL training up to 240 steps continues to improve without overfitting.

### Key Findings

- The code medium enables the 7B model to surpass the perceptual ceiling of the base model on perception-intensive tasks (e.g., +53% on PixmoCount).
- The difficulty-adaptive reward is critical: it prevents tool spamming on simple questions while ensuring sufficient exploration on difficult ones.
- Emergent behaviors demonstrate that atomic-level tool training can catalyze compositional reasoning capabilities, suggesting a scalable pathway.

## Highlights & Insights

- **Code > fixed schema**: Code as a reasoning medium is more flexible, composable, and extensible than predefined operation templates.
- **Difficulty adaptation**: Using in-group accuracy $\mu_{\text{acc}}$ as a difficulty proxy is an elegant and effective design.
- **Exciting emergent behaviors**: Atomic tool training → compositional emergence during RL → cross-task transfer, suggesting that code execution may be a scalable path toward agentic multimodal reasoning.

## Limitations & Future Work

- Validation is limited to Qwen2.5-VL-7B; the effect at larger scales remains unexplored.
- Security and latency overhead of the code execution sandbox are not analyzed in detail.
- The controllability and predictability of emergent behaviors are limited, with no theoretical explanation provided.
- Construction of the 34K SFT dataset relies on strong model distillation, incurring non-trivial cost.

## Related Work & Insights

- Conceptually similar to ViperGPT but goes further: ViperGPT compiles queries into programs, whereas CodeDance trains the model via RL to autonomously write and iteratively refine code.
- The BAT reward design improves upon the simple tool reward in DeepEyes, offering a better paradigm for tool use in RL training.
- Observations of emergent behaviors can inspire further research into the mechanisms underlying emergent agentic capabilities in LLMs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Code as a unified reasoning medium + BAT reward is innovative, though the overall framework is not paradigm-breaking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 10+ benchmarks, comprehensive ablations, and in-depth analysis of emergent behaviors.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, coherent narrative, and persuasive presentation of emergent behaviors.
- **Value**: ⭐⭐⭐⭐⭐ Provides a reproducible "think with images" solution for the open-source community with high practical value.

---

# CodeDance: A Dynamic Tool-integrated MLLM for Executable Visual Reasoning

**Conference**: CVPR 2026
**arXiv**: [2512.17312](https://arxiv.org/abs/2512.17312)
**Code**: [https://CodeDance-VL.github.io](https://CodeDance-VL.github.io)
**Area**: Multimodal VLM / Tool Use
**Keywords**: Executable visual reasoning, tool integration, code generation, reinforcement learning, emergent behavior

## TL;DR
This paper proposes CodeDance, which uses executable code as a general-purpose solver for visual reasoning. The MLLM generates code to define, compose, and execute diverse tools, rendering intermediate visual results (bboxes, lines, charts) to support an auditable reasoning chain. RL training is guided by a tool-calling reward that balances exploration and efficiency. During RL, the model exhibits emergent unseen tool-calling combinations and cross-task transfer, with the 7B model surpassing GPT-4o on counting, visual search, and chart QA.

## Background & Motivation

1. **Background**: o3 demonstrates the ability to "think with tools"—alternating between reasoning and tool use. However, existing open-source methods either rely solely on textual CoT, use fixed schemas (predicting only bbox coordinates), or operate as single-step pipelines.
2. **Key Gap**: (1) Pure textual CoT cannot dynamically interact with visual inputs or verify intermediate results; (2) Fixed schemas limit flexibility and composability; (3) o3 is a closed black-box system.
3. **Core Idea**: Code is the most universal "tool-calling language"—CodeDance enables the MLLM to generate and execute Python code to orchestrate diverse tools, compute intermediate results, and render visual artifacts. RL training reveals **emergent behaviors** (novel tool-calling patterns, compositions, and cross-task transfer not seen during training).

## Method

### Key Designs
1. **Executable Code Reasoning**: The model generates Python code → executes it → obtains tool outputs (crops/detections/OCR results) → continues reasoning or generates new code → produces the final answer. Interleaved "thinking (text)" and "executing (code)" are supported.
2. **Tool-Calling Reward (RL)**: A reward is designed to balance exploration and efficiency—encouraging moderate tool use (too little → insufficient information; too much → overuse/low efficiency).
3. **Emergent Behaviors**: Observed during RL training—the model invents tool-calling patterns not present in training data, combines tools from different tasks to address new ones, and exhibits cross-task transfer.

### Loss & Training
Atomic supervision (single-tool examples) → SFT initialization → RL (tool-calling reward + task correctness reward)

## Key Experimental Results

### Main Results

| Model | CountBench | PixmoCount | V*Bench | ChartQA |
|------|:---:|:---:|:---:|:---:|
| GPT-4o | 87.9 | - | 67.5 | 86.7 |
| Qwen2.5-VL-7B | 76.5 | 50.4 | 76.4 | 86.3 |
| Deepeyes-7B | 80.4 | 57.2 | 90.4 | 78.2 |
| **CodeDance-7B** | **91.2** | **77.1** | 84.8 | **87.5** |

CountBench +19.2%, PixmoCount +53.0% vs. Qwen2.5-VL-7B baseline.

### Emergent Behavior Case Studies
- Unseen tool compositions (e.g., chained zoom+count+compare calls)
- Cross-task transfer (applying region detection from chart analysis to counting tasks)
- Spontaneous generation of verification code (rendering detection results for visual inspection)

### Key Findings
- Code is far more expressive than fixed schemas—yielding significant performance gains at the same model size.
- Emergent behaviors are a key product of RL—they do not appear during the SFT stage.
- More tool calls are not always better—a balanced reward outperforms an "always use tools" reward.

## Highlights & Insights
- **Code as a universal reasoning medium**: More executable than textual CoT, more flexible than fixed schemas—code natively supports variables, loops, conditionals, and function definitions.
- **Emergent properties of RL training**: Novel tool-use compositions and cross-task transfer—from atomic capabilities to creative combinations, analogous to emergent capabilities in language models.
- **Auditable reasoning**: Code + rendered visual intermediate results → fully traceable and verifiable reasoning chains.

## Limitations & Future Work
- Code execution requires a sandbox environment → higher deployment complexity than pure-text models.
- The tool set is predefined—how to dynamically discover and integrate new tools remains an open question.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Executable code reasoning + RL emergent behaviors
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks across counting/search/chart + emergent analysis
- Writing Quality: ⭐⭐⭐⭐ Intuitive case-study presentation of emergent behaviors
- Value: ⭐⭐⭐⭐⭐ Significant contribution to tool use and reasoning paradigms for VLMs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing](from_intuition_to_investigation_a_tool-augmented_reasoning_mllm_framework_for_ge.md)
- [\[CVPR 2026\] Proof-of-Perception: Certified Tool-Using Multimodal Reasoning with Compositional Conformal Guarantees](pop_proof_of_perception_conformal_reasoning.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](docseeker_long_document_understanding.md)
- [\[CVPR 2026\] Unbiased Dynamic Multimodal Fusion](unbiased_dynamic_multimodal_fusion.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)

</div>

<!-- RELATED:END -->
