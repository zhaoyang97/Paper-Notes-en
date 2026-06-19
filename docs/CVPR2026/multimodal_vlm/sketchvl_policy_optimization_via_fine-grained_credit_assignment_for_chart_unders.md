---
title: >-
  [Paper Note] SketchVL: Policy Optimization via Fine-Grained Credit Assignment for Chart Understanding and More
description: >-
  [CVPR 2026][Multimodal VLM][Reinforcement Learning] SketchVL enables MLLMs to "draw" each step of chart reasoning as visual annotation actions (boxes, lines, points, circles) on the image. It introduces the FinePO algorithm to **redistribute** the coarse-grained advantage of an entire trajectory to each step based on scores from a Process Reward Model (FinePRM). This ac
tags:
  - CVPR 2026
  - Multimodal VLM
  - Reinforcement Learning
  - Reasoning on Image
date: 2026-05-08
content_hash: 41cc559fdd773795
---
# SketchVL: Policy Optimization via Fine-Grained Credit Assignment for Chart Understanding and More

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_SketchVL_Policy_Optimization_via_Fine-Grained_Credit_Assignment_for_Chart_Understanding_CVPR_2026_paper.html)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Chart Understanding, Reinforcement Learning, Process Reward Model, Credit Assignment, Reasoning on Image

## TL;DR
SketchVL enables MLLMs to "draw" each step of chart reasoning as visual annotation actions (boxes, lines, points, circles) on the image. It introduces the FinePO algorithm to **redistribute** the coarse-grained advantage of an entire trajectory to each step based on scores from a Process Reward Model (FinePRM). This achieves step-level fine-grained credit assignment, yielding an average improvement of 7.23% over base models across chart, natural image, and math benchmarks.

## Background & Motivation

**Background**: Charts are high-density carriers of data visualization, and automated chart understanding requires precise visual reasoning. Current mainstream approaches use MLLMs (e.g., Qwen2.5VL, Gemma3) combined with Reinforcement Learning (e.g., Vision-R1, VLM-R1) to enhance multimodal understanding. Specifically, the "Reasoning on Image (RoI)" paradigm (ChartSketcher, DeepEyes) externalizes intermediate reasoning into visible annotations on the image, creating interactive feedback.

**Limitations of Prior Work**: Chart understanding is inherently a **step-by-step** structure—first locating legends/axes, then reading values, aligning categories, comparing trends, and finally synthesizing conclusions. Error in any single step (imprecise cropping, misread scales, incorrect legend alignment) leads to the collapse of the entire reasoning chain. However, current MLLM RL practices mostly provide **coarse-grained, outcome-only** feedback: methods like GRPO calculate a scalar advantage only from the final answer and **uniformly broadcast** it to all tokens in the trajectory.

**Key Challenge**: Trajectory-level advantage estimation **cannot distinguish between correct and incorrect steps within a single response**. Consequently, correct intermediate logic in a failed response is penalized, while flawed steps in a coincidentally correct response are rewarded. This injects noise into the learning signal and diminishes the benefits of RL.

**Goal**: To implement fine-grained credit assignment for compositional, step-dependent tasks like chart understanding, where **each step along the reasoning chain is individually evaluated and reinforced**.

**Key Insight**: The RoI paradigm explicitly decomposes reasoning into a sequence of discrete visual annotation actions. This provides a natural structural carrier for "step-wise scoring and credit assignment"—since each step is a visible and evaluable action, individual process scores can be assigned to each action.

**Core Idea**: Use a process reward model (FinePRM) to score each drawing action, and then use FinePO to **redistribute the global advantage of the trajectory** according to the contribution of each step—rewarding correct tokens more strongly when global success is achieved and penalizing incorrect tokens more heavily when the global result is sub-optimal.

## Method

### Overall Architecture
SketchVL is an MLLM featuring iterative reasoning: during inference, it draws "actions" corresponding to "intents" onto the chart, then feeds the annotated image back to itself to guide the next decision, forming a visible reasoning trajectory. Training consists of two stages (following the ColdStart-RL paradigm): **Cold Start** enables the model to learn basic localization and RoI reasoning patterns (50K SFT data), followed by **FinePO** reinforcement learning to unlock complex reasoning capabilities. The core of FinePO is the use of **FinePRM** (a Process Reward Model) to score each action in the trajectory, achieving step-level "advantage calculation-to-credit assignment." Data for training FinePRM is synthesized via a **cross-modal distillation pipeline** (473K samples).

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input: Chart + Query"] --> B["RoI Sketch Reasoning<br/>Progressively drawing annotation actions<br/>Intent -> Action -> Feedback Image"]
    B --> C["A set of k candidate trajectories"]
    C --> D["FinePRM Process Scoring<br/>4-level rating per action<br/>+ KL action regularization"]
    D --> E["FinePO Credit Assignment<br/>Cross-trajectory advantage -> Step-wise redistribution"]
    E -->|Step-level advantage A(sj)| F["Policy Update"]
    F -->|Next iteration| B
    G["Cross-modal distillation data pipeline<br/>Vision -> Text -> Intent-Action pairs"] -.Train.-> D
```

### Key Designs

**1. Reasoning-on-Image Annotation Reasoning + ColdStart-RL Two-stage Training**

To provide evaluable targets for step-wise scoring, SketchVL externalizes the reasoning process as a series of annotation actions on the image. Each step consists of a text "intent" (describing the reasoning goal, e.g., "mark the maximum value") and an "action" (executing the intent by drawing visible marks: Line, Point, Rectangle, Circle, or Text). Annotated images are fed back to the model as input for the next step, transforming abstract reasoning into visible, step-wise supervisable trajectories—a prerequisite for step-level credit assignment. Training begins with a cold start (2 epochs, 50K distilled SFT data from EvoChart/GQA/ChartQA-Train synthetic QA + ChartSketcher pipeline) to master localization and RoI basics, followed by the FinePO RL phase (1 epoch, 9K mixed-domain prompts). Ablations show that removing sketching ("w/o Sketch (zero GRPO)") leads to a massive performance drop (e.g., EvoChart-QA falls from 47.28 to 30.48), proving this interactive visual reasoning is the foundation for solving complex chart tasks.

**2. FinePO: Redistributing Trajectory Advantage to Each Step**

This is the core contribution, addressing the pain point where trajectory-level advantages fail to distinguish step-wise correctness. FinePO proceeds in two stages. **First, calculate coarse advantage across trajectories**: for a prompt, $k$ candidate responses $\{y_1,\dots,y_k\}$ are sampled. Each receives a terminal reward $R(y_i)$ based on overall correctness. The advantage is the deviation from the group mean: $A(y_i) = R(y_i) - \frac{1}{k}\sum_{j=1}^{k} R(y_j)$, a high-level scalar signal similar to GRPO. **Second, perform credit redistribution within the trajectory**: FinePRM provides a process score $p_j = P(\text{intent}_j, \text{action}_j, \text{img}_{j-1}, \text{img}_j)$ for step $j$ (evaluating visual changes between images). After adjustment via KL regularization to $p'_j$, the step-wise mean is calculated weighted by token length $L_j$ as $\bar p = \frac{\sum_j L_j p'_j}{\sum_j L_j}$. The deviation from the mean $\Delta_j = p'_j - \bar p$ indicates whether a step is better or worse than the average response.

Crucially, FinePO **does not create new rewards but more precisely allocates the existing coarse advantage $A(y_i)$ to each step**:

$$A'(s_j) = A(y_i) + \alpha \cdot k \cdot \Delta_j$$

where $\alpha$ controls adjustment intensity and $k = \frac{|A(y_i)|}{\max_{j}(0,\Delta_j)+\epsilon}$ is a dynamic scaling factor that aligns the adjustment magnitude with $|A(y_i)|$. The design ensures the weighted sum of adjustments is zero, thereby **conserving the total advantage** and anchoring fine-grained signals to the response's overall performance. Finally, a clipping mechanism ensures steps in a globally superior response ($A(y_i)>0$) do not receive negative advantages, and vice versa:

$$A(s_j) = \begin{cases} \text{clip}(A'(s_j),\ 0,\ \beta A(y_i)) & A(y_i) > 0 \\ \text{clip}(A'(s_j),\ \beta A(y_i),\ 0) & A(y_i) \le 0 \end{cases}$$

This yields a sharper, lower-noise learning signal compared to uniform broadcasting in GRPO.

**3. FinePRM: Four-level Process Reward Model + KL Action Regularization**

FinePRM provides step-level signals for FinePO, using Qwen2.5VL-7B as the backbone following the VisualPRM approach. It receives text intent, action, and images before and after the action $\text{img}_{j-1}, \text{img}_j$. Prompted as a "reviewer," it judges if the visual modification in Image 2 accurately executes the intent, outputting four levels: Excellent, Acceptable, Poor, or Unacceptable, mapped to scalar values $[4.0, 3.0, 2.0, 1.0]$. To prevent the policy from favoring "easy-to-score" actions while avoiding difficult but necessary ones due to uniform scoring standards, **KL Action Regularization** is introduced:

$$O_{\text{clipped}}(a_j) = \text{clip}\!\left(-\lambda_{KL}\log\frac{P_k(a_j)+\epsilon}{Q(a_j)+\epsilon},\ -\gamma,\ \gamma\right)$$

where $Q(a)$ is the pre-computed action distribution from the training set, and $P_k(a)$ is the current policy's distribution over the last $k$ batches. This modifies the process score $p'_j = p_j + O_{\text{clipped}}(a_j)$, penalizing deviations from the prior. This prevents the model from collapsing into a few easy-to-score action types ("Action Bias"). Figure 4 shows that without KL, the 7B model's distribution collapses (larger models are more prone to hacking FinePRM), while KL encourages a more balanced and robust reasoning strategy.

**4. Cross-modal Distillation Data Pipeline (473K)**

Training FinePRM requires large-scale "intent-action" pairs. Since even models like Gemini 2.5 Pro struggle to simultaneously locate and identify numerous targets in dense images, a two-stage distillation is used. **Vision-to-Text Annotation**: Drawing from Set-of-Mark, SAM is used to segment the image into object-centric patches. Each patch is expanded by 20% for context and highlighted with a red box. Qwen2.5VL-72B then annotates "internal attributes" (e.g., "a purple line") and "interactive attributes" (e.g., "intersects with the green line"), converting dense visual content into structured text. **Text-to-Image Distillation**: An LLM distills these annotations into simulated intent-action pairs via two paths: Direct Generation for single-step tasks, and Trajectory-based Simulation for multi-step GT trajectories. Noise is injected into actions to create negative samples. The final dataset of 473K samples follows an Excellent:Acceptable:Poor:Unacceptable ratio of 2:4:3:1 to force the model to focus on subtle decision boundaries.

### A Full Example
Query: "When did the FDI start to stay consistently above 400?". SketchVL generates a trajectory: ① Intent "Identify FDI curve color" -> Draws a red box on the legend (FinePRM rates Excellent/Acceptable); ② Intent "Draw horizontal line at 400" -> Draws a blue horizontal line; ③ Intent "Draw vertical line from intersection to x-axis" -> Draws a green vertical line (rated Excellent). Finally, observing the green line intersects the x-axis around 2005 and the orange FDI curve remains above the blue line thereafter, it answers 2005. Each action is individually scored by FinePRM, and FinePO redistributes the trajectory advantage to each step—reinforcing correct localization while penalizing deviations.

### Loss & Training
SketchVL is trained in 7B and 3B versions (Qwen2.5VL-Instruct base). FinePRM uses Qwen2.5VL-7B as its backbone. FinePRM training takes 4 epochs, Cold Start 2 epochs, and FinePO 1 epoch. Training is conducted on 16×NVIDIA A800 (40G) using the ms-swift framework. Evaluation uses DeepSeek-R1-Distill-Qwen-14B as a judge with 9-vote majority voting for correctness.

## Key Experimental Results

### Main Results
Evaluation covers expert chart datasets (EvoChart-QA, ChartQA, ChartQA-Pro, ChartBench, PlotQA) and general datasets (MMStar, MathVista). SketchVL comprehensively outperforms base models, with particularly significant gains at the 3B scale.

| Model | EvoChart-QA | ChartQA | ChartBench | MathVista | MMStar |
|------|------|------|------|------|------|
| Qwen2.5VL-7B (Base) | 54.80 | 82.00 | 64.78 | 61.40 | 56.67 |
| **SketchVL-7B (Ours)** | **58.64** | **83.96** | **65.11** | **63.50** | **57.13** |
| Qwen2.5VL-3B (Base) | 39.36 | 61.88 | 56.20 | 49.50 | 43.53 |
| **SketchVL-3B (Ours)** | **47.28** | **77.20** | **59.96** | **53.80** | **51.00** |
| VLM-R1 | 40.32 | 72.98 | 39.58 | 55.10 | 48.27 |

SketchVL-3B improves over the base by +15.32 on ChartQA and +3.76 on ChartBench. Performance gains on non-chart datasets (MathVista/MMStar) demonstrate that FinePO strengthens specialized capabilities while maintaining general generalization.

### Ablation Study (Based on SketchVL-3B)

| Configuration | EvoChart-QA | ChartQA | PlotQA | Description |
|------|------|------|------|------|
| Full Model | 47.28 | 77.20 | 48.32 | Complete model |
| w/o FinePO (naive GRPO) | 45.60 | 75.12 | 44.72 | Reverts to uniform broadcasting; ChartQA −2.08 |
| w/o FinePRM (random) | 48.08 | 76.76 | 46.40 | Random rewards; process quality drops significantly (Table 2) |
| w/o KL Action Reg. | 48.56 | 77.80 | 48.16 | Similar performance, but action distribution collapses |
| w/o Sketch (zero GRPO) | 30.48 | 57.56 | 31.12 | Removal of drawing capability results in total collapse |
| w/o RL (SFT only) | 26.48 | 54.72 | 27.44 | Cold start only, no RL; worst performance |

Further process quality evaluation (Table 2, using FinePRM as an automated evaluator for mean process scores) shows SketchVL-3B Full achieves 2.857/2.917/2.914 on PlotQA/ChartBench/MMStar, outperforming naive GRPO (2.705/2.777/2.755). This confirms FinePO effectively aligns step-level behavior with FinePRM even on datasets not seen during RL training.

### Key Findings
- **Drawing is the foundation**: Removing Sketch causes the largest drop (−16.8 on EvoChart-QA). RoI interactive visual reasoning is fundamental for complex charts.
- **FinePRM must provide meaningful signals**: Replacing it with random scores reduces process quality, proving credit assignment depends on actual evaluation rather than noise.
- **KL Regularization's value is in robustness**: Removing it might yield slightly higher final scores, but action distributions collapse (especially for 7B). KL ensures balanced action usage.
- **Generalization spillover**: RL gains extend to the untrained PlotQA, indicating FinePO enhances intrinsic, generalizable reasoning.

## Highlights & Insights
- **Advantage Conservation in Formula**: FinePO redistributes existing coarse advantages without adding new rewards, ensuring the weighted sum of adjustments is zero. This anchors fine-grained signals to overall performance and prevents the PRM from leading the policy astray.
- **RoI Paradigm as a Step-wise Carrier**: Externalizing abstract reasoning as discrete actions provides evaluable entities for the PRM and clear boundaries for scoring—a clever use of task structure for interpretable credit assignment.
- **KL to Prevent Hacking**: Aligning action distributions with a prior prevents the policy from collapsing into easy-to-score actions. This is transferable to any Scene where Process Reward Models might be exploited.
- **Overcoming MLLM Localization Weaknesses**: Using SAM segmentation and single-object tagging to convert dense images into structured text for distillation is a practical recipe for generating high-precision grounding data.

## Limitations & Future Work
- The authors acknowledge that KL regularization might slightly constrain optimal policies on benchmarks dominated by a single action type—a trade-off between precision and diversity.
- The system has high dependency: FinePRM requires distilling 473K samples from 72B/80B models, and training requires 16×A800, creating high barriers to reproduction.
- FinePRM is itself a 7B MLLM; its reliability is capped by its own judging capability. Using FinePRM as both a training signal and evaluator (Table 2) involves a self-evaluation loop, so conclusions should be viewed with caution.
- Gains on 7B are smaller than on 3B (attributed to larger models hacking PRMs more easily); how to ensure strong base models fully benefit from fine-grained signals remains an open problem.

## Related Work & Insights
- **vs GRPO / VLM-R1**: These methods calculate a trajectory-level scalar advantage and broadcast it uniformly to all tokens, failing to distinguish step-wise correctness. SketchVL uses FinePRM to redistribute advantage to every step for a sharper signal.
- **vs ChartSketcher / DeepEyes (RoI Paradigm)**: While they use drawing for reasoning, they lack step-level process rewards and credit assignment. SketchVL's core delta is the FinePO + FinePRM fine-grained reinforcement mechanism.
- **vs VisualPRM**: Both use MLLMs as process reward models, but FinePRM is specifically designed for image annotation actions (evaluating temporal visual changes) and is supported by 473K distilled samples and KL regularization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "conservative advantage redistribution" of FinePO combined with RoI as a step-level career is a robust and targeted design.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 benchmarks, multi-angle ablations, and process quality analysis, though hindered by lack of code and self-evaluation loops.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly derived, formulas are comprehensive, and diagrams are intuitive.
- Value: ⭐⭐⭐⭐ Provides a transferable paradigm for fine-grained credit assignment in reasoning-heavy RL, offering practical insights for the chart and vision reasoning communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Chart-FR1: Visual Focus-Driven Fine-Grained Reasoning on Dense Charts](chart-fr1_visual_focus-driven_fine-grained_reasoning_on_dense_charts.md)
- [\[CVPR 2026\] MA-Bench: Towards Fine-grained Micro-Action Understanding](ma-bench_towards_fine-grained_micro-action_understanding.md)
- [\[CVPR 2026\] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception](cropvlm_learning_to_zoom_for_fine_grained_vision_language_perception.md)
- [\[CVPR 2026\] HiconAgent: History Context-aware Policy Optimization for GUI Agents](hiconagent_history_context-aware_policy_optimization_for_gui_agents.md)
- [\[ACL 2026\] Learning More from Less: Exploiting Counterfactuals for Data-Efficient Chart Understanding](../../ACL2026/multimodal_vlm/learning_more_from_less_exploiting_counterfactuals_for_data-efficient_chart_unde.md)

</div>

<!-- RELATED:END -->
