---
title: >-
  [Paper Note] What Does Reinforcement Learning for Visual Tool Use Actually Learn?
description: >-
  [ICML 2026][Reinforcement Learning][Visual Tool-use] This paper proposes the MED framework to systematically analyze what visual tool-use RL actually learns in crop-and-zoom scenarios. It finds that performance gains fro…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Visual Tool-use"
  - "RL Interpretation"
  - "Crop-and-Zoom"
  - "Multimodal Reasoning"
date: 2026-05-08
content_hash: 01c2f1a33693ed4a
---

# What Does Reinforcement Learning for Visual Tool Use Actually Learn?

**Conference**: ICML 2026  
**arXiv**: [2602.01334](https://arxiv.org/abs/2602.01334)  
**Code**: https://github.com/GAIR-NLP/Med  
**Area**: Multimodal VLM / Reinforcement Learning / Agent Tool-use  
**Keywords**: Visual Tool-use, RL Interpretation, Crop-and-Zoom, Multimodal Reasoning

## TL;DR
This paper proposes the MED framework to systematically analyze what visual tool-use RL actually learns in crop-and-zoom scenarios. It finds that performance gains from RL training primarily stem from **intrinsic capability enhancement** rather than improved tool mastery; models mainly learn how to coexist safely with tools rather than genuinely mastering them.

## Background & Motivation

**Background**: Current VLMs widely adopt tool-use RL to enhance multimodal reasoning. Researchers equip VLMs with visual manipulation tools (e.g., crop-and-zoom) and use RL to train models to learn when and how to invoke tools during inference.

**Limitations of Prior Work**: Although visual tool-use RL leads to performance gains, **the nature of these improvements remains unclear**. Observed gains could originate from three sources: (1) RL strengthening intrinsic capabilities (improving even without tools); (2) RL improving tool interaction itself; (3) RL merely reducing the side effects of tools rather than fixing failure cases. Existing evaluations only report end-to-end accuracy, failing to provide mechanism-level attribution.

**Key Challenge**: The lack of clarity regarding the origins of performance improvements prevents the design of effective RL objective functions. If gains are primarily intrinsic, optimizing tool-calling policies is limited; if gains are tool-driven, rewards should target the ability to correct failures via tools.

**Goal**: Systematically decompose performance improvement sources from a training dynamics perspective—separating intrinsic capability drift from tool-induced effects, and further decomposing them into interpretable gain and harm terms.

**Key Insight**: Conduct checkpoint-level analysis on two VLM backbones with different tool priors (Qwen2.5-VL, not pre-trained on crop-and-zoom; Qwen3-VL, pre-trained) across six benchmarks. Compare the evolution curves of **tool-free inference accuracy** and **tool-available inference accuracy**.

**Core Idea**: Design the MED (Measure-Explain-Diagnose) three-tier framework. It uses a probability decomposition identity to split the tool-induced performance gap into four terms, followed by a Mass-Policy-Quality factorization for each term to diagnose root causes.

## Method

### Overall Architecture
For each training checkpoint, the model is evaluated under two protocols—**Tool-free protocol**: No tool schema is provided to measure "intrinsic capability"; **Tool-available protocol**: Tool schema is provided for active calling. By tracking both curves, intrinsic drift $f_{wo}(t)$ is separated from tool-induced drift $\Delta_{tool}(t)$: $f_w(t) = f_{wo}(t) + \Delta_{tool}(t)$.

### Key Designs

1.  **Measure Phase: Quantifying the Total Contribution of Tool Effects**:
    - **Function**: Calculate the cumulative magnitude of intrinsic and tool drift using integrals to derive the Tool Contribution Ratio $S_{tool}$.
    - **Mechanism**: $|B_{wo}| = \int_0^T |f_{wo}(t)| dt$ and $|B_{\Delta tool}| = \int_0^T |\Delta_{tool}(t)| dt$, where $S_{tool} = \frac{|B_{\Delta tool}|}{|B_{wo}| + |B_{\Delta tool}|}$. $S_{tool} \approx 0$ indicates intrinsic drift dominance, while $\approx 1$ indicates tool effect dominance.
    - **Design Motivation**: Provide a macro-level answer to whether intrinsic enhancement or tool-use improvement drives final performance.

2.  **Explain Phase: Decomposing Tool-induced Effects into Four Terms**:
    - **Function**: Decompose the tool performance gap $G(t)$ into "Call Gain / Schema Gain / Call Harm / Schema Harm."
    - **Mechanism**: Group samples by success/failure under the tool-free protocol ($\mathcal{D}_{fail}$ / $\mathcal{D}_{succ}$) and categorize by tool-calling behavior (call / no-call). $G(t) = T1 + T2 - T3 - T4$. $T1$ is tool-free failure but success after calling; $T3$ is tool-free success but failure after calling (tool harm).
    - **Design Motivation**: Convert abstract performance gaps into **actionable components**, distinguishing gains (T1+T2) from harms (T3+T4).

3.  **Diagnose Phase: Mass-Policy-Quality Factorization**:
    - **Function**: Further decompose each term into three factors to diagnose the underlying mechanism of performance changes.
    - **Mechanism**: $\text{Term}(\mathcal{D},a,o) = P(\mathcal{D}) \cdot P(a|\mathcal{D}) \cdot P(o|a,\mathcal{D})$—**Mass** is the sample size, **Policy** is the decision "when to call," and **Quality** is the execution "how to use."
    - **Design Motivation**: Stagnation in call gains might stem from a shrinking failure set (Mass↓), cessation of calling (Policy↓), or declining execution (Quality↓). This pinpointing reveals the "intrinsic-tool trade-off."

### Loss & Training
The GRPO algorithm is used with pure outcome-based rewards. 21 checkpoints are analyzed across two VLM backbones and six benchmarks (VStar, HR-Bench 4k/8k, VisualProbe Easy/Medium/Hard).

## Key Experimental Results

### Main Results: Intrinsic Drift Dominance

| Model | Tool Contribution Ratio $S_{tool}$ | Intrinsic Drift $\|B_{wo}\|$ | Tool Drift $\|B_{\Delta tool}\|$ |
| :--- | :--- | :--- | :--- |
| Qwen2.5-VL | 0.30 | High | Low |
| Qwen3-VL | 0.22 | High | Low |

For both models, the tool contribution ratio is far below 0.5, indicating that 70%+ of learning progress comes from intrinsic capability enhancement.

### Gain-Harm Decomposition

| Phase | Call Gain (T1) | Schema Gain (T2) | Call Harm (T3) | Schema Harm (T4) | Net Benefit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Early | Rapid Rise | Low | Moderate | Moderate | Positive |
| Middle | Plateau/Decline | Low | Decreasing | Decreasing | Slow Growth |
| Late | Plateau/Decline | Low | Further Decrease | Further Decrease | Stagnant |

### Persistent Failure Cases (Manual Annotation of 370 Samples)

| Failure Type | Count | Proportion |
| :--- | :--- | :--- |
| No call but should have | 82 | 22.2% |
| Call but incorrect crop | 52 | 14.1% |
| Correct crop but reasoning wrong | 37 | 10.0% |
| Correct crop but task too hard | 10 | 2.7% |

### Key Findings
- **Call Gain Stagnation**: T1 rises rapidly and plateaus for Qwen2.5-VL, but monotonically declines for Qwen3-VL.
- **Continuous Harm Reduction**: T3+T4 decrease throughout the entire training process.
- **Plateau in Net Tool Benefit**: Reflects an equilibrium between saturated gains and diminishing harms.
- **Deep Insight**: Stagnation in call gains is not due to quality breakdown but **capacity limits**—as intrinsic capabilities improve, the set of difficult failure samples naturally shrinks, limiting the upper bound of tool assistance.

## Highlights & Insights
- **Elegance of the Probability Decomposition Identity**: Treating the performance gap as four terms provides both a mathematical identity and an operational diagnostic tool.
- **Diagnostic Power of Mass-Policy-Quality**: Effectively captures the "intrinsic-tool trade-off" phenomenon.
- **Sober Realization of What is Learned**: Models actually learn to "coexist safely with tools"—reducing tool-induced harm rather than strengthening tool-based correction.
- **Dual Backbone Comparison**: Reveals the significant impact of tool priors on learning dynamics.

## Limitations & Future Work
- Analysis is limited to a single tool (crop-and-zoom); multi-tool dynamics may differ.
- The paper analyzes dynamics but does not propose a new RL algorithm.
- Metrics focus only on accuracy, neglecting efficiency or explainability.
- Fixed checkpoint sampling might miss rapid dynamics.
- **Future Directions**: Design RL objectives to explicitly maximize "selective correction on failure sets" while minimizing "harm on success sets"; extend to multi-tool scenarios.

## Related Work & Insights
- **vs. Tool-use Faithfulness**: Faithfulness checks surface alignment; MED diagnoses actual efficacy.
- **vs. Single VLM Analysis**: This paper compares models with different tool priors, showing their impact on the learning curve.
- **vs. Outcome vs. Process Reward Debate**: MED analysis suggests tool-related rewards primarily change Policy rather than Quality and cannot solve the fundamental Mass capacity limit.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The probability decomposition and Mass-Policy-Quality factorization are original diagnostic frameworks.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 2 VLMs + 6 benchmarks + 21 checkpoints + manual case analysis + sanity checks.
- **Writing Quality**: ⭐⭐⭐⭐ Logic is clear; the three-tier analysis progresses naturally.
- **Value**: ⭐⭐⭐⭐⭐ Challenges the intuitive belief that "tool-use RL is about mastering tools."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints](../../ICLR2026/reinforcement_learning/autotool_automatic_scaling_of_tool-use_capabilities_in_rl_via_decoupled_entropy_.md)
- [\[ICML 2026\] Learning to Search and Searching to Learn for Generalization in Planning](learning_to_search_and_searching_to_learn_for_generalization_in_planning.md)
- [\[ICML 2026\] You Can Learn Tokenization End-to-End with Reinforcement Learning](you_can_learn_tokenization_end-to-end_with_reinforcement_learning.md)
- [\[ACL 2026\] SCRL: What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time](../../ACL2026/reinforcement_learning/what_if_consensus_lies_selective-complementary_reinforcement_learning_at_test_ti.md)
- [\[ICLR 2026\] Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment](../../ICLR2026/reinforcement_learning/reasoning_as_representation_rethinking_visual_reinforcement_learning_in_image_qu.md)

</div>

<!-- RELATED:END -->
