---
title: >-
  [Paper Note] Are Tools Always Beneficial? Learning to Invoke Tools Adaptively for Dual-Mode Multimodal LLM Reasoning
description: >-
  [ICML2026][LLM Reasoning][Adaptive tool invocation] AutoTool utilizes reinforcement learning to enable Multimodal Large Language Models (MLLMs) to first judge whether an inquiry truly requires a "zoom-in" tool…
tags:
  - "ICML2026"
  - "LLM Reasoning"
  - "Adaptive tool invocation"
  - "multimodal reasoning"
  - "reinforcement learning"
  - "visual grounding"
  - "mode balancing"
date: 2026-05-08
content_hash: 32ec9f9cfab39bf9
---

# Are Tools Always Beneficial? Learning to Invoke Tools Adaptively for Dual-Mode Multimodal LLM Reasoning

**Conference**: ICML2026  
**arXiv**: [2605.19852](https://arxiv.org/abs/2605.19852)  
**Code**: https://github.com/MQinghe/AutoTool  
**Area**: Multimodal VLM  
**Keywords**: Adaptive tool invocation, multimodal reasoning, reinforcement learning, visual grounding, mode balancing  

## TL;DR
AutoTool utilizes reinforcement learning to enable Multimodal Large Language Models (MLLMs) to first judge whether an inquiry truly requires a "zoom-in" tool, then adaptively switch between tool-assisted reasoning and text-only reasoning, thereby simultaneously improving accuracy and efficiency across high-resolution perception, localization, hallucination detection, and reasoning tasks.

## Background & Motivation
**Background**: MLLMs can already decompose complex problems into intermediate reasoning steps via CoT, yet many methods still follow the text-centric paradigm of LLMs, where visual information is often encoded only once at the input stage. To allow models to re-examine local visual evidence during reasoning, recent methods like MCoT and "Thinking with Images" have introduced external tools such as search, segmentation, OCR, depth estimation, or image zoom-in.

**Limitations of Prior Work**: While tools help models process fine-grained targets, local attributes, and high-resolution images, existing tool-augmented MLLMs often default to the assumption that "using tools is always beneficial." Methods like DeepEyes and OpenThinkIMG emphasize how to invoke tools and generate correct answers but do not explicitly model "whether to invoke tools." This leads to two direct issues: first, simple questions enter multi-round tool interactions, increasing training and inference costs; second, incorrect or redundant local crops may distract the model's attention, leading to hallucinations.

**Key Challenge**: The benefit of tool invocation is highly dependent on the problem type. If a question requires checking small objects, fine-grained attributes, or local regions, zoom-in provides extra evidence; if the question relies on global layout, spatial relationships, or if the targets are already clear, tool invocation offers marginal gain and may lose necessary context. Thus, the key is not just learning "how to use tools," but learning "when not to use tools."

**Goal**: The authors aim to train a dual-mode multimodal reasoning model: for every multimodal question, the model first selects between a tool-assisted mode or a text-only mode; in tool mode, it must correctly localize and utilize zoom-in observations; in text-only mode, it must avoid meaningless invocations while still providing accurate answers.

**Key Insight**: Instead of relying on auxiliary SFT cold-start data, the paper integrates mode selection, format following, answer accuracy, and tool effectiveness into a unified GRPO reinforcement learning framework. This allows the model to explore both reasoning paths during training and gradually learn when to favor one over the other through reward constraints.

**Core Idea**: Utilizing explicit `<tool_on>` / `<tool_off>` dual-mode control tokens, mode-specific rewards, and adaptive mode balancing, MLLMs transform tool invocation from a fixed pipeline into a strategic choice triggered by problem characteristics.

## Method
The key change in AutoTool is not the addition of a stronger visual tool, but the rewriting of the decision mechanism for tool invocation. Given a multimodal query $X=(Q,V)$, where $Q$ is the text question and $V$ is the image, the model first generates a mode signal: if `<tool_on>` is selected, subsequent reasoning can issue a `<tool_call>`, and the resulting `<tool_response>` from the zoom-in execution is appended back to the context; if `<tool_off>` is selected, the model directly completes internal reasoning and answering within the existing context.

### Overall Architecture
During training, AutoTool uses Qwen2.5-VL-7B as the base policy model and samples a set of candidate reasoning trajectories using GRPO. Each trajectory contains not only the final answer but also process information such as mode selection, format compliance, whether a tool was called, and whether the tool call led to a correct answer. The reward function decomposes these signals for calculation, and the policy is updated using relative advantages within the group.

In tool-assisted mode, the trajectory approximates alternating "textual thought, tool action, tool observation." For instance, the model judges that local evidence is needed, outputs a region to enlarge, and continues reasoning upon receiving the local image. In text-only mode, the trajectory contains only internal thoughts and the answer, with no zoom-in calls allowed. During the inference phase, the model follows the training prompts and can automatically select the mode or be manually assigned a mode by the user via prompts or special tokens.

### Key Designs
1.  **Explicit Dual Reasoning Modes**:
    - **Function**: Separates tool-assisted reasoning and text-only reasoning into two learnable and evaluable modes.
    - **Mechanism**: The model generates `<tool_on>` or `<tool_off>` at the start of each problem. Following `<tool_on>`, it can invoke the zoom-in tool and utilize the returned local observation; `<tool_off>` requires the model to answer based solely on the original context. This is clearer than implicitly letting the model decide when to invoke tools, as the reward function directly knows which strategy the current trajectory belongs to.
    - **Design Motivation**: Existing methods often treat tool invocation as a default process, drawing simple questions into tool chains. Explicit modes promote the "whether to invoke" decision to the primary choice, allowing the model to learn different strategies for global understanding, local detail, and hallucination detection questions.

2.  **Mode-Specific Policy Optimization (MSPO)**:
    - **Function**: Designs different tool rewards for different modes to prevent mechanical tool invocation.
    - **Mechanism**: Total reward is formulated as $R=R_{acc}+R_{format}+\lambda^{mode}_{tool}R_{tool}$. $R_{acc}$ measures semantic equivalence to the ground truth using rules and a Qwen2.5-72B-Instruct reward model; $R_{format}$ checks structures like `<think>` and `<answer>`; $R_{tool}$ varies by mode. In `<tool_on>`, a correct tool call resulting in a correct answer yields $1$, while a wrong answer after invocation yields $-0.5$; in `<tool_off>`, a correct answer without tool invocation yields $1$.
    - **Design Motivation**: If only the final answer is rewarded, the model might guess correctly based on linguistic priors after incorrect localization; if only tool invocation is rewarded, the model overuses tools. MSPO defines "tool utility" as "the tool helping produce a correct answer" and explicitly penalizes ineffective invocations to suppress redundant interactions.

3.  **Adaptive Mode Balancing (AMB)**:
    - **Function**: Prevents training collapse into a single mode early on while allowing free strategic choice later.
    - **Mechanism**: For a batch of $N\times G$ rollouts, the frequency $F_{on}=N_{on}/(N_{on}+N_{off})$ of the tool mode is calculated. When tool mode is excessive, the tool mode reward coefficient is lowered and the text-only coefficient is raised, and vice versa. The paper adopts a dynamic $\lambda^{mode}_{tool}$ coefficient and removes the balancing constraint in the final ~20 steps of training to let the model choose freely based on item features.
    - **Design Motivation**: Base models naturally lean towards text-only reasoning as it more easily garners format and answer rewards; left unmanaged, `<tool_on>` is under-explored. AMB provides dual-mode exploration pressure early on, then releases constraints to avoid fixing the model to specific proportions.

### Loss & Training
At the optimizer level, AutoTool uses GRPO. Given problem $X$, the old policy samples $G$ outputs $o_i$, each receiving a reward $r_i$, which is normalized into an advantage $\hat{A}_i=(r_i-mean(r))/std(r)$ using group mean and standard deviation. The objective function uses a PPO-style clipped ratio with a clipping parameter $\epsilon=0.2$ and no additional KL regularization.

Training data follows the DeepEyes setup, including fine-grained samples from V*, chart data from ArxivQA, and reasoning data from ThinkLite-VL. The base model is Qwen2.5-VL-7B, trained for 80 iterations using 8 H200s for policy training and 2 H200s to deploy the Qwen2.5-72B-Instruct reward model. Each batch contains 256 samples split into 4 PPO mini-batches; 16 rollouts are generated per query; the initial tool reward coefficient $\lambda^{base}_{tool}$ is set to 1.2, with a learning rate of $1\times10^{-6}$ and a maximum response length of 20,480 tokens.

## Key Experimental Results

### Main Results
The paper validates AutoTool across four task categories: high-resolution perception, visual localization, hallucination detection, and multimodal reasoning. Evaluation datasets include HRbench-4K/8K, V*, RefCOCO series, ReasonSeg, POPE, MathVista, MathVerse, MathVision, WeMath, DynaMath, and LogicVista.

| Task/Dataset | Metric | AutoTool | Qwen2.5-VL-7B | DeepEyes | Main Conclusion |
|--------|------|------|------|------|------|
| HRbench-4K | Overall acc | 76.9 | 69.6 | 74.9 | 7.3 points higher than base, also exceeding fixed-tool methods |
| HRbench-8K | Overall acc | 74.0 | 63.0 | 71.5 | Gains are more pronounced in high-res scenarios |
| V* | Overall acc | 90.1 | 69.1 | 87.4 | 21.0 points higher than base, approaching/exceeding most grounding models |
| RefCOCO test | IoU@0.5 acc | 88.5 | 84.7 | 86.0 | Tool mode enables more accurate target localization |
| ReasonSeg val | IoU@0.5 acc | 63.0 | 59.5 | 61.5 | Gains persist in complex referring expressions and segmentation-based localization |
| POPE Overall | Acc | 88.9 | 87.2 | 86.0 | Adaptive tool invocation reduces hallucinations from invalid local evidence |
| MathVista testmini | Acc | 72.8 | 70.6 | 71.6 | Maintains general multimodal reasoning while optimizing perception |

### Ablation Study
Ablations indicate that forcing the model to always use tools is not optimal; the combination of dual modes, penalties for incorrect tool use, and late-stage free exploration yields the best results.

| Configuration | HRbench-4K Overall | HRbench-8K Overall | V* Overall | Description |
|------|------|------|------|------|
| Text-only GRPO | 73.6 | 70.2 | 85.3 | Strengthening internal reasoning alone is better than the base |
| Always Tool on | 74.9 | 71.5 | 87.4 | Fixed zoom-in helps but introduces redundant calls |
| Tool on + Tool off | 75.3 | 72.4 | 88.5 | Dual modes mitigate impact of incorrect tool calls |
| With MSPO penalty | 75.8 | 73.3 | 89.0 | Penalizing wrong answers after tool use leads to cautious localization |
| With Late Free Exploration | 76.8 | 73.2 | 89.5 | Strategies adapt better once fixed ratio constraints are removed |
| Full AutoTool | 76.9 | 74.0 | 90.1 | Combination of three core components achieves best performance |

### Efficiency & Hyperparameter Analysis
| Analysis Item | Setting/Comparison | Result | Explanation |
|------|------|------|------|
| Training Time | DeepEyes vs AutoTool | 44.9 h vs 35.8 h, 20.3% reduction | Avoids toolchains for all samples, resulting in shorter rollouts |
| V* Direct Inference | DeepEyes vs AutoTool | 2.23 min vs 1.68 min, 24.7% faster | Skips zoom-in for simple samples |
| HRbench-8K Inference | DeepEyes vs AutoTool | 53.45 min vs 33.08 min, 38.1% faster | Invokes tools only when necessary in high-res scenarios |
| POPE Random Inference | DeepEyes vs AutoTool | 13.07 min vs 7.20 min, 44.9% faster | Large targets in POPE images lead to higher text-only mode usage |
| AMB Removal Timing | step 0 / 50 / 60 / 70 / 80 | step 60 is best (V* 90.1) | Releasing too early favors `<tool_off>`; too late over-constrains policy |
| $\lambda^{base}_{tool}$ | 0.0 to 5.0 | Stable around 1.2, robust from 0.5 to 3.0 | Extreme values cause reward imbalance or mode collapse |

### Key Findings
- Tool invocation is most valuable for high-resolution, fine-grained, and localization tasks; however, for tasks like POPE where targets are large or existence is questioned, the benefit of frequent zoom-in is low.
- The penalty for incorrect tool invocation is not just for efficiency; it corrects the reward loophole where "wrong localization leads to a lucky correct answer," forcing the tool mode to focus on truly effective visual evidence.
- The value of AMB lies in the training process rather than inference: maintaining ~50% dual-mode exploration in the early/mid stages allows the model to naturally adjust mode proportions based on dataset features after constraints are lifted.

## Highlights & Insights
- The paper reframes tool usage from a "capability problem" to a "decision problem." This is crucial as MLLM toolchains grow complex; the greatest cost often stems not from a single tool, but from latency and context pollution brought by unnecessary tools.
- MSPO's reward design is restrained: it does not invent complex tool quality evaluators but binds process correctness to final answer accuracy. This rewards effective tool usage while penalizing "invocation for the sake of invocation."
- AMB reflects exploration engineering experience in RL training. Models favor modes that score easily; without external balancing, the other mode may be marginalized before it is learned. Conversely, persistent constraints hinder late-stage adaptation.
- The logic of this work is transferable to broader agent systems: features like search, code execution, database queries, OCR, and segmentation tools should not be "always on" but should follow a cost-sensitive invocation gating strategy.

## Limitations & Future Work
- The current tool is primarily zoom-in, suited for local visual checks and localization. If expanded to OCR, retrieval, segmentation, depth estimation, or multi-tool combinations, the mode space shifts from binary choice to multi-tool scheduling, requiring a redesign of MSPO and AMB.
- Task accuracy in the reward partially relies on rules and the Qwen2.5-72B-Instruct reward model. Reward model bias may affect mode selection for open-ended long answers, fine-grained explanations, or multi-solution problems.
- Experiments focus on existing VQA, localization, hallucination, and reasoning benchmarks, which do not fully demonstrate long-chain tool usage in real interaction scenarios (e.g., multi-round visual search or cross-page operations).
- While efficiency gains are highlighted via time metrics, future analysis could further examine VRAM, token consumption, tool failure rates, and cost-benefit curves across different hardware.
- The dual-mode token is a clean design but might compress complex decisions into a single initial choice. Finer strategies could allow text-only reasoning first, followed by delayed tool invocation if evidence is found insufficient.

## Related Work & Insights
- **vs DeepEyes**: DeepEyes learns visual grounding tool reasoning via GRPO but tends toward using zoom-in for all problems. AutoTool retains the advantages of the tool mode while introducing `<tool_off>` and penalties for incorrect calls to solve redundancy and localization errors.
- **vs OpenThinkIMG / Thinking with Images**: These methods introduce image operations or visual states during reasoning, proving the value of multimodal intermediate evidence. AutoTool differs by not treating visual operations as default steps, instead learning when intermediate visual evidence is required.
- **vs Text-only CoT / GRPO Reasoning Enhancement**: Text-only RL improves general reasoning but is insensitive to high-resolution local evidence. AutoTool provides a tool path for local visual problems while retaining the text-only path, creating a complementary effect.
- **Insight**: For multimodal agents, more tools necessitate a "call gate." A practical direction is extending AutoTool’s binary mode into a cost-aware strategy: first estimate the evidence type needed for the task, then select the cheapest yet reliable tool combination.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Explicitly incorporating tool invocation necessity into RL objectives provides clear problem awareness and minimalist mechanisms.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers perception, localization, hallucination, and reasoning with thorough ablations and efficiency analysis; real-world multi-tool scenarios could be further explored.
- Writing Quality: ⭐⭐⭐⭐☆ Motivation and methodological chains are clear, supported by ample data, though some tables are dense in the cached text.
- Value: ⭐⭐⭐⭐⭐ Offers direct inspiration for tool-augmented MLLMs and multimodal agents, particularly in guiding cost-sensitive tool invocation training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Chain-of-Thought Reasoning in the Wild Is Not Always Faithful](chain-of-thought_reasoning_in_the_wild_is_not_always_faithful.md)
- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](../../ICLR2026/llm_reasoning/adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)
- [\[ACL 2026\] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning](../../ACL2026/llm_reasoning/templaterl_structured_template-guided_reinforcement_learning_for_llm_reasoning.md)
- [\[ICML 2026\] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)
- [\[ICML 2026\] PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching](powerflow_unlocking_the_dual_nature_of_llms_via_principled_distribution_matching.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Chain-of-Thought Reasoning in the Wild Is Not Always Faithful](chain-of-thought_reasoning_in_the_wild_is_not_always_faithful.md)
- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](../../ICLR2026/llm_reasoning/adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)
- [\[ACL 2026\] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning](../../ACL2026/llm_reasoning/templaterl_structured_template-guided_reinforcement_learning_for_llm_reasoning.md)
- [\[ICML 2026\] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)
- [\[ICML 2026\] PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching](powerflow_unlocking_the_dual_nature_of_llms_via_principled_distribution_matching.md)

</div>

<!-- RELATED:END -->
