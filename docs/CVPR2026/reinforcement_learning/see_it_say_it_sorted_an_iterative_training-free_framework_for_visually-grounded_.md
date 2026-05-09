---
title: >-
  [Paper Note] See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs
description: >-
  [CVPR 2026][Reinforcement Learning][ECRD] This paper proposes Evidence-Constrained Reweighting Decoding (ECRD), a framework that maintains a dynamic textual evidence pool during LVLM decoding, reweights candidate tokens via distribution negotiation, and automatically invokes a lightweight visual decider to extract micro-evidence under uncertainty—achieving significant reductions in visual hallucination and improvements in reasoning accuracy across multiple LVLMs without any training.
tags:
  - CVPR 2026
  - Reinforcement Learning
  - ECRD
  - visual grounding
  - hallucination mitigation
  - training-free
  - evidence pool
date: 2026-05-08
content_hash: 52095e54507f9c98
---

# See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

**Conference**: CVPR 2026
**arXiv**: [2602.21497](https://arxiv.org/abs/2602.21497)
**Code**: [GitHub](https://github.com/uuuuZYC/See-It-Say-It-Sorted)
**Area**: Reinforcement Learning
**Keywords**: ECRD, visual grounding, hallucination mitigation, training-free, evidence pool

## TL;DR

This paper proposes Evidence-Constrained Reweighting Decoding (ECRD), a framework that maintains a dynamic textual evidence pool during LVLM decoding, reweights candidate tokens via distribution negotiation, and automatically invokes a lightweight visual decider to extract micro-evidence under uncertainty—achieving significant reductions in visual hallucination and improvements in reasoning accuracy across multiple LVLMs without any training.

## Background & Motivation

**Large Vision-Language Models (LVLMs)** are capable of generating long chain-of-thought (CoT) reasoning, yet suffer from a fundamental problem: **reasoning-perception drift**. During long-sequence decoding, the model must balance three competing contexts—the image, the growing textual context, and the instruction. As context length increases, subtle but critical visual cues are easily overwhelmed by language priors. Once an intermediate reasoning step deviates from visual evidence, the final answer becomes incorrect even if subsequent reasoning is logically sound—a phenomenon termed **visual hallucination propagation**.

**Existing approaches** primarily rely on RL training to teach models to "think with images," learning when to zoom/crop the image and re-inject cropped regions into the reasoning context. Representative works include PixelReasoner and DeepEyes. However, such methods exhibit three key limitations: (1) they require substantial annotated data and reward engineering, incurring high training costs; (2) the learned policy is tightly coupled to a specific backbone, limiting transferability; and (3) repeated encoding of cropped regions introduces significant inference latency.

**The starting point of this paper** is fundamentally different: rather than learning when to look at the image during training, it supervises each reasoning step with visual evidence at **test time**. The core idea is to reformulate decoding as a series of "evidence-driven token selections": maintaining a textual evidence pool, negotiating with the model's original distribution at each decoding step, and using uncertainty signals to trigger the acquisition of new evidence.

## Method

### Overall Architecture

ECRD wraps a lightweight supervision framework around a frozen LVLM. At each decoding step: (1) knee truncation selects the top-$k$ candidate tokens from the base distribution; (2) the Distribution Supervisor constructs an evidence-induced distribution from the evidence pool and negotiates a mixture with the base distribution; (3) if uncertainty remains after negotiation, the Visual Decider is triggered to extract new textual micro-evidence from the image and append it to the pool.

### Key Designs

1. **Distribution Supervisor**:

    - Function: At each decoding step, reweights candidate tokens using the current evidence pool to produce an evidence-induced distribution, which is then negotiated with the base model distribution to yield the final selection.
    - Mechanism: For each piece of evidence $\mathcal{E}$ in the pool, the average probability of candidate token $w$ over all prefix positions in $\mathcal{E}$ is computed as $q_\mathcal{E}(w) = \frac{1}{L}\sum_{j=1}^{L}p_{\text{VLM}}(w|e_{<j})$. Averaging across all evidence entries and applying softmax yields the evidence-induced distribution $r_i$. This is then adaptively mixed with the base distribution using $\alpha_i = p_{(1)}$ (the maximum probability under the base distribution): $p_i^{\text{mix}} = \alpha_i p_i + (1-\alpha_i)\tilde{r}_i$.
    - Design Motivation: When the base distribution is sharp (large $p_{(1)}$), the model is confident and the original behavior should be preserved; when it is diffuse (small $p_{(1)}$), hallucination is more likely and evidence should receive greater weight. This adaptive mixing requires no hyperparameter tuning.

2. **Visual Decider**:

    - Function: When uncertainty persists after distribution negotiation, extracts micro-evidence from the image that is relevant to the current reasoning context.
    - Mechanism: The trigger condition is $k^* > 1$ and the post-mixture top-2 margin $\Delta_i \leq \delta$. The decider (GRIT, based on Qwen2.5-VL-3B) receives the image, the tail of the text prefix, and the candidate set, and outputs (1) the selected token $w^*$ and (2) a human-readable micro-evidence sentence $\mathcal{E}_i$ appended to the evidence pool.
    - Design Motivation: Evidence is stored as text rather than pixels, allowing subsequent tokens to reference prior micro-observations without re-encoding image crops. This makes interventions lightweight and verifiable while avoiding the overhead of repeated crop encoding in RL-based methods.

3. **Dynamic Evidence Pool**:

    - Function: Cumulatively maintains a collection of visual micro-evidence relevant to the reasoning chain.
    - Mechanism: Initialized with a single global image description $d_{\text{global}}$, the pool grows on demand only when uncertainty is triggered: $E_{i+1} \leftarrow E_i \cup \{\mathcal{E}_i\}$. Each piece of evidence semantically corresponds to a sub-view of the image but is stored in textual form.
    - Design Motivation: The global description provides broad coverage but is not the sole evidence source; subsequent micro-evidence accumulates precisely according to reasoning needs. Evidence is composed and reused in token space, allowing later steps to benefit from early visual disambiguation without reprocessing pixels.

### Loss & Training

The framework is entirely training-free. It wraps a frozen LVLM, and the GRIT decider is used as an off-the-shelf pretrained model. The only hyperparameter is the uncertainty threshold $\delta$, which provides flexible control over the accuracy–latency trade-off.

## Key Experimental Results

### Main Results

| Model | TreeBench Gain | RH-Bench RH-AUC Gain | Notes |
|-------|---------------|----------------------|-------|
| Qwen2.5-VL-7B + ECRD | +10.9 Overall | — | Attribute +17.2, Physical +17.4 |
| Qwen2.5-VL-32B + ECRD | +6.1 Overall | — | Consistently effective |
| Qwen2.5-VL-72B + ECRD | +7.7 Overall | — | Large models also benefit |
| LLaVA-OneVision-7B + ECRD | +6.2 Overall | — | Cross-backbone |
| LLaVA-OneVision-72B + ECRD | +6.4 Overall | — | Cross-backbone |
| InternVL3-8B + ECRD | +6.4 Overall | — | Cross-backbone |

### Ablation Study

| Configuration | Effect | Notes |
|---------------|--------|-------|
| Knee truncation only | Partial improvement | Candidate set restriction alone is helpful |
| + Distribution Supervisor | Significant improvement | Evidence negotiation is the core contributor |
| + Visual Decider | Best | On-demand evidence acquisition further reduces hallucination |
| Fixed global description (no dynamic growth) | Moderate | Demonstrates the importance of dynamic evidence |

### Key Findings

- ECRD improves performance on both perception tasks (Attribute, Material, Physical) and reasoning tasks (Containment, Comparison), with larger gains on perception tasks, indicating that visual grounding is the primary source of improvement.
- A gain of 7.7% is observed even on the strong Qwen2.5-VL-72B model, confirming that reasoning-perception drift persists in large models.
- ECRD does not degrade performance on already-confident subtasks (e.g., OCR), as the adaptive mixing weight automatically degenerates to preserve the base distribution.
- ECRD enables open-source LVLMs to substantially close the gap with proprietary models such as GPT-4o and Gemini across multiple tasks.

## Highlights & Insights

- The design of the adaptive mixing weight $\alpha_i = p_{(1)}$ is minimal yet effective: it requires no learned hyperparameters and automatically adjusts intervention intensity based solely on the confidence of the base distribution. The principle of "non-intervention when confident, strong intervention when uncertain" is broadly instructive for other model correction scenarios.
- Using text rather than pixels as the evidence representation is a critical design choice—it preserves the model's native token space, avoids repeated image encoding, and enables evidence reuse throughout the reasoning chain.

## Limitations & Future Work

- The Visual Decider (GRIT/Qwen2.5-VL-3B) may itself produce hallucinations; the current framework does not validate the decider's outputs.
- The uncertainty threshold $\delta$ must be set in advance, and the optimal value may vary across tasks and models.
- Each decider invocation requires an additional LVLM forward pass, which may introduce significant inference latency in scenarios with frequent triggers.

## Related Work & Insights

- **vs. PixelReasoner/DeepEyes**: These methods require RL training to learn zoom/crop policies, whereas ECRD is entirely training-free and transfers across backbones, at the cost of not learning task-specific observation strategies.
- **vs. VDGD**: ECRD represents a substantial upgrade over VDGD—replacing a single static description with a dynamically growing evidence pool, replacing logit overriding with probabilistic negotiation, and preserving the base model's behavior at high-confidence steps.

## Rating

- Novelty: ⭐⭐⭐⭐ The complete design combining distribution negotiation, dynamic evidence pool, and uncertainty-triggered acquisition demonstrates strong originality.
- Experimental Thoroughness: ⭐⭐⭐⭐ Six backbones evaluated across multiple benchmarks; cross-model generalization is thoroughly validated.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous and the comparison with VDGD is clearly articulated.
- Value: ⭐⭐⭐⭐ The training-free, plug-and-play nature offers substantial practical value for deployment across multiple open-source models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Divide, Harmonize, Then Conquer It: Shooting Multi-Commodity Flow Problems with Multimodal Language Models](../../ICLR2026/reinforcement_learning/divide_harmonize_then_conquer_it_shooting_multi-commodity_flow_problems_with_mul.md)
- [\[ACL 2026\] STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems](../../ACL2026/reinforcement_learning/stride-ed_a_strategy-grounded_stepwise_reasoning_framework_for_empathetic_dialog.md)
- [\[AAAI 2026\] Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning](../../AAAI2026/reinforcement_learning/do_it_for_her_first-order_temporal_logic_reward_specification_in_reinforcement_l.md)
- [\[CVPR 2026\] Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement](seeing_is_improving_visual_feedback_for_iterative_text_layout_refinement.md)
- [\[ICCV 2025\] mDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs](../../ICCV2025/reinforcement_learning/mdp3_a_training-free_approach_for_list-wise_frame_selection_in_video-llms.md)

</div>

<!-- RELATED:END -->
