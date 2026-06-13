---
title: >-
  [Paper Note] Revisit What You See: Revealing Visual Semantics in Vision Tokens to Guide LVLM Decoding
description: >-
  [ACL2026][Multimodal VLM][Vision-Language Models] ReVisiT discovers that vision tokens in LVLMs already encode interpretable object semantics. By using contextually constrained vocabularies, vision token selection…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "Hallucination Suppression"
  - "Decoding Strategy"
  - "Vision Tokens"
  - "Training-free Method"
date: 2026-05-08
content_hash: f3995a0ee291aa65
---

# Revisit What You See: Revealing Visual Semantics in Vision Tokens to Guide LVLM Decoding

**Conference**: ACL2026 Oral  
**arXiv**: [2506.09522](https://arxiv.org/abs/2506.09522)  
**Code**: https://github.com/bscho333/ReVisiT  
**Area**: Multimodal VLM  
**Keywords**: Vision-Language Models, Hallucination Suppression, Decoding Strategy, Vision Tokens, Training-free Method

## TL;DR
ReVisiT discovers that vision tokens in LVLMs already encode interpretable object semantics. By using contextually constrained vocabularies, vision token selection, and logit fusion, it enhances visual grounding and reduces hallucinations without training or additional forward passes.

## Background & Motivation
**Background**: Current Large Vision-Language Models (LVLMs) typically encode images into vision tokens, which are fed into a language model decoder alongside text tokens. Dominant hallucination suppression methods usually rely on contrastive decoding, attention re-weighting, and input perturbation contrast during decoding, or utilize external verifiers for post-generation correction.

**Limitations of Prior Work**: While these methods mitigate object hallucinations, they often treat visual information as implicit context. There is a lack of direct analysis regarding what vision tokens contribute to the next-token distribution, whether they contain object-level semantics, and why these semantics are frequently ignored by standard greedy decoding.

**Key Challenge**: LVLMs do not necessarily "fail to see" the correct object at hallucination positions. Empirical analysis shows that ground-truth objects often reside in high-probability regions of the output distribution, but language priors override visual grounding signals at certain timesteps. Thus, the problem is not just adding more visual information, but explicitly pulling existing visual semantics back into decoding decisions.

**Goal**: The authors aim to answer two questions: first, whether vision tokens carry object semantics that map to the text vocabulary; second, whether a lightweight mechanism can select the most relevant vision tokens during decoding to correct next-token probabilities.

**Key Insight**: Adopting a "logit lens" approach, the authors project vision token hidden states into the language model's vocabulary space. A key observation is that direct projection onto the full vocabulary is diluted by irrelevant words; however, once the vocabulary is restricted to candidates reasonable for the current context, the object semantics in vision tokens become clear.

**Core Idea**: Construct a candidate vocabulary adaptively based on the current output distribution, select the vision tokens most consistent with the context within this constrained space, and fuse their projected distributions as visual reference signals with the original logits.

## Method
ReVisiT is a decoding-time method requiring no structural changes, additional annotations, or retraining. It transforms static vision tokens into a "visual evidence library" accessible at each step: for every generation step, it identifies potential words based on the original distribution, finds vision tokens that best explain these candidates, and uses those tokens to refine the output.

### Overall Architecture
The input consists of an image and a user prompt. The LVLM processes these to obtain vision tokens and text context, calculating the original distribution for the current timestep. ReVisiT performs three steps on this distribution: first, it filters context-relevant candidates from the full vocabulary using a probability threshold; second, it projects cached vision tokens into this same candidate space and selects the most relevant token using Jensen-Shannon Divergence; third, it combines the original logits and the selected vision token's projected logits in log-space, applying softmax for the final distribution.

The key to this process is that all comparisons occur within the same constrained vocabulary, avoiding noise from function words or punctuation and aligning visual semantics with the current context.

### Key Designs
1.  **Context-Aware Candidate Vocabulary Constraint**:
    - **Function**: Narrows the full vocabulary to a candidate set truly likely for the current decoding step.
    - **Mechanism**: Given the original distribution $p_\theta(w)$, only tokens satisfying $p_\theta(w) \geq \alpha \cdot \max_{w'}p_\theta(w')$ are retained. The threshold $\alpha$ dictates the set size.
    - **Design Motivation**: Analysis shows that while vision token recall for ground-truth objects is only 2.03% in the full vocabulary, it jumps to 40.44% within semantically consistent object subsets. Constraining the vocabulary is a necessity to make visual semantics visible.

2.  **Vision Token Selection via Minimum JSD**:
    - **Function**: Finds the most relevant visual reference for the current text context across all tokens and layers.
    - **Mechanism**: Each vision token's hidden state is projected via the LM head onto the candidate vocabulary. The token and layer with the minimum Jensen-Shannon Divergence (JSD) relative to the original output distribution are selected. Smaller JSD indicates higher semantic alignment.
    - **Design Motivation**: Random or maximal JSD selection forces irrelevant image regions into the decoding process, leading to performance degradation. Minimum JSD ensures "referencing vision tokens" is a context-aware choice.

3.  **Product-of-Experts Style Logit Fusion**:
    - **Function**: Uses the semantic distribution of the selected vision token to calibrate the next-token distribution.
    - **Mechanism**: The log probabilities of the original distribution and the selected vision token's projection are summed within the constrained set and re-normalized.
    - **Design Motivation**: Summing in log-space treats visual evidence as an "expert," preserving the language model's contextual constraints while bringing visual grounding to the decision layer.

### Loss & Training
ReVisiT contains no training loss. Vision token projections can be cached before decoding begins. During generation, the model only performs slicing, divergence calculation, and logit fusion. Deterministic greedy decoding is used with a maximum length of 512.

## Key Experimental Results

### Main Results
Evaluation was conducted on HallusionBench, CHAIR, POPE, VQAv2, and MMMU using LLaVA-1.5-7B, Qwen2.5-VL-7B, and InternVL3-8B against baselines like Greedy, DoLa, VCD, M3ID, CODE, and SID.

| Dataset / Model | Metric | ReVisiT | Greedy | Gain |
| -------- | ------ | ------ | ---------- | ------ |
| HallusionBench / LLaVA-1.5-7B | qAcc | 20.22 | 11.55 | +8.67 (~75% relative) |
| CHAIR / Qwen2.5-VL-7B | CHAIRI ↓ / F1 ↑ | 7.04 / 81.16 | 8.43 / 79.85 | Lower hallucinations, F1 +1.31 |
| POPE / LLaVA-1.5-7B | Accuracy / F1 | 81.80 / 83.45 | 79.47 / 82.36 | Acc +2.33, F1 +1.09 |
| VQAv2 / Qwen2.5-VL-7B | Accuracy | 67.60 | 65.80 | +1.80 |
| MMMU / InternVL3-8B | Accuracy | 54.14 | 53.54 | +0.60 |

ReVisiT reduces object hallucinations while maintaining or improving accuracy in VQA and knowledge-intensive tasks, suggesting it does not merely make the model conservative.

### Ablation Study
Analysis on Qwen2.5-VL (CHAIR) regarding selection criteria, constraints, and layers:

| Configuration | Key Metric | Description |
| ------ | --------- | ------ |
| Full ReVisiT | F1 = 81.16 | Best performance using candidate vocab, min-JSD, and all layers. |
| max-JSD selection | F1 = 0.67 | Performance collapses when choosing least similar vision tokens. |
| random selection | F1 = 17.63 | Random visual references fail to provide stable grounding. |
| min-JSD + full vocabulary | F1 = 1.52 | Visual projections deviate from targets without vocabulary constraints. |
| last-layer variant | Peak F1 = 80.97 | Slightly lower than all-layers but remains effective. |

### Key Findings
- In 190 hallucination steps, at least one ground-truth object appeared in the top-50 candidates 63.2% of the time, and top-500 95.8% of the time, proving models "know" the visual object but fail to select it.
- Vision token projection recall for ground-truth objects reaches 40.44% on the CHAIR subset (top-1) and 89.24% (top-30), supporting the hypothesis that constraints make tokens interpretable.
- Inference latency increases by only 0.6% to 2.0% compared to greedy decoding, whereas multi-forward methods like VCD or M3ID often double the overhead.

## Highlights & Insights
- The most valuable insight is the transformation of vision tokens from "internal invisible context" into "projectable, selectable, and referable semantic evidence."
- The design emphasizes that visual information must be activated within the current textual candidate space rather than simply increasing visual weights globally.
- The training-free nature makes it ideal for deployment on existing LVLMs, particularly in closed-source or cost-sensitive scenarios where model intermediate states are accessible.
- The paper demonstrates that hallucinations are not always due to a lack of visual knowledge in representation, but rather a failure of decoding strategies to utilize existing knowledge.

## Limitations & Future Work
- The method depends on existing semantics in vision tokens. If the encoder fails to capture small objects or fine-grained attributes, ReVisiT cannot recover them.
- Strengthening grounding may cause models to over-focus on salient objects, potentially impacting tasks requiring external knowledge or common sense.
- Experiments were restricted to 7B to 32B models; the behavior of 70B+ scale models requires further validation.
- Current mechanisms focus on object hallucinations; extensions for relation, attribute, or OCR-based hallucinations are needed.

## Related Work & Insights
- **vs VCD / M3ID**: These methods contrast perturbed images with original ones; ReVisiT references vision token projections directly without extra forward passes, ensuring higher efficiency.
- **vs DoLa**: DoLa uses layer-wise differences in the LM to improve factuality; ReVisiT specifically utilizes signals from vision tokens for grounding.
- **vs Attention-based methods**: Unlike methods (e.g., DAMRO) that adjust decoding based on attention patterns, ReVisiT maps vision hidden states into the semantic vocabulary space.
- **Insight**: "Constrained semantic projection" could serve as a general tool for interpreting token contributions in video, audio, or retrieval-augmented models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Naturally combines vision token projection with decoding-time referencing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 benchmarks across 3 architectures with speed and mechanism analysis.
- Writing Quality: ⭐⭐⭐⭐☆ Clear logic and rich tables, though formula density is high.
- Value: ⭐⭐⭐⭐⭐ High reference value for training-free hallucination suppression and multimodal decoding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] What You Think is What You See: Driving Exploration in VLM Agents via Visual-Linguistic Curiosity (GLANCE)](../../ICML2026/multimodal_vlm/what_you_think_is_what_you_see_driving_exploration_in_vlm_agents_via_visual-ling.md)
- [\[ACL 2026\] "I See What You Did There": Can Large Vision-Language Models Understand Multimodal Puns?](i_see_what_you_did_there_can_large_vision-language_models_understand_multimodal_.md)
- [\[ACL 2026\] VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation](vauq_vision-aware_uncertainty_quantification_for_lvlm_self-evaluation.md)
- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](../../CVPR2026/multimodal_vlm/aif_adaptive_information_flow_vlm.md)
- [\[ACL 2026\] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)

</div>

<!-- RELATED:END -->
