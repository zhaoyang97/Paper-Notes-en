---
title: >-
  [Paper Note] Mechanisms of Prompt-Induced Hallucination in Vision–Language Models
description: >-
  [ACL 2026][Hallucination Detection][prompt-induced hallucination] In controlled object counting tasks, the hallucination behavior where the "model follows the prompt instead of the image" is localized to 3–10 attention h…
tags:
  - "ACL 2026"
  - "Hallucination Detection"
  - "prompt-induced hallucination"
  - "attention head knockout"
  - "mean ablation"
  - "object counting"
  - "modality conflict"
date: 2026-05-08
content_hash: 2db0c008a7e0b93c
---

# Mechanisms of Prompt-Induced Hallucination in Vision–Language Models

**Conference**: ACL 2026  
**arXiv**: [2601.05201](https://arxiv.org/abs/2601.05201)  
**Code**: https://github.com/michalg04/prompt-induced_hallucinations.git  
**Area**: Hallucination Detection  
**Keywords**: prompt-induced hallucination, attention head knockout, mean ablation, object counting, modality conflict

## TL;DR
In controlled object counting tasks, the hallucination behavior where the "model follows the prompt instead of the image" is localized to 3–10 attention heads in the **early layers** (primarily L0-1) of LLaVA-OneVision / Qwen-VL / Janus-Pro. Applying mean ablation to these heads without any retraining causes prompt-following to drop from 42–64% to <11%, recovers true counting rates to 70–78%, and enables zero-shot transfer to color recognition tasks (PIH suppression of 40–95%).

## Background & Motivation
**Background**: VLMs (LLaVA / Qwen-VL / Janus) tend to follow the prompt when prompt and image information conflict, resulting in "prompt-induced hallucination (PIH)". For example, if there are 3 water lilies in an image but the prompt asks to "describe the 4 water lilies," the model will actually describe 4. This is a common sycophancy/anchoring bias in real-world deployment. However, existing research mostly remains at the phenomenal level, lacking mechanistic explanations.

**Limitations of Prior Work**: (1) Existing hallucination mitigation solutions either rely on expensive RLHF retraining or brittle prompt engineering, without identifying "which component is executing prompt-copying"; (2) While attention heads are known to handle specific functions (e.g., induction heads / copying heads), whether PIH is managed by a small set of localizable heads remains unverified; (3) Even if heads are located, functional differences across tasks and models (whether different models use the same mechanism for PIH) remain an open question.

**Key Challenge**: (a) Minimizing intervention vs. maximizing effect — fewer changes are safer, but the scope of impact must be large enough to suppress PIH comprehensively; (b) **Mechanism commonality vs. model specificity** — does a single mechanism run through all VLMs, or does each model have its own PIH circuit?

**Goal**: (1) Systematically characterize when PIH occurs (sliced by ground-truth quantity $N$ and prompt offset $k$); (2) Identify the minimal set of heads responsible for PIH using attention head knockout (mean ablation); (3) Verify if these are shared across models and generalize across tasks (counting → color); (4) Deconstruct the function of PIH-heads (is it suppressing copying or amplifying visual attention?).

**Key Insight**: The authors noted that LLaVA-OV and Qwen-VL share the Qwen2 backbone but use different visual encoders. This naturally forms a controlled experiment—if the identified PIH-heads overlap significantly, PIH likely originates from the LM rather than visual components.

**Core Idea**: Use the classic mean ablation paradigm (replacing head outputs with the mean over all data to remove token-specific info while retaining activation magnitude) to find PIH-heads via single-head ranking, followed by group ablation tests and cross-task/cross-model comparisons of head overlap and functional differences.

## Method

### Overall Architecture
The pipeline consists of three steps: (1) **Phenomenal Characterization** — Constructing baseline prompts "How many [X] are in the image?" and misaligned prompts "Describe the N+k [X] in the image" ($k \in \{1,...,5\}$, with $k \in \{10, 20, 50\}$ for extremes) on CountBench to observe when the model is misled; (2) **Mechanism Localization** — For each head $h$ in layer $l$, calculate $\mu^{(l,h)} = \frac{1}{T}\sum_t H_t^{(l,h)}$ and replace outputs at all token positions with $\tilde H_t^{(l,h)} = \mu^{(l,h)}$. Success rate of switching from "N+k" to "N" is measured per head to select top-m (Qwen-VL m=3, others m=10) for joint ablation; (3) **Functional Analysis** — Statistical analysis of behavior changes across four copying forms and measuring the shift of attention mass from text to image.

### Key Designs

1.  **Mean ablation instead of zero ablation**:
    - **Function**: Removes token-specific information carried by the target head while preserving its "activation budget" in the residual stream to avoid distribution shifts.
    - **Mechanism**: $\tilde H^{(l,h)}_t = \mu^{(l,h)} = \frac{1}{T}\sum_{t'} H^{(l,h)}_{t'}$. Replacing individual outputs with the global mean deprives the head of the ability to "see token content" while contributing a fixed bias. Knockout success rate = "proportion of PIH samples switching to correct count N".
    - **Design Motivation**: Direct zeroing disrupts activation distributions after layer norm, causing uncontrollable side effects; mean ablation is a standard probe verified by the mechanistic interpretability community in studies of IOI circuits and induction heads.

2.  **Cross-model head overlap + cross-task migration**:
    - **Function**: Uses head overlap distribution to determine if PIH is an internal LM mechanism or a visual component mechanism; uses counting→color migration to verify if the head set is task-agnostic.
    - **Mechanism**: LLaVA-OV and Qwen-VL share the Qwen2 LM. Their top-1/top-2 PIH-heads fully overlap (L0H3, L0H6), and half of the top-10 overlap. In contrast, Janus-Pro (using DeepSeek-LLM) shows low overlap (top head is L0H20). This suggests PIH originates from the LM. The same head set is then applied to the Visual CounterFact color task ("Describe the C+k [object]") to check for generalization.
    - **Design Motivation**: By controlling variables (shared LM / different vision), the question of "which component is responsible for PIH" is transformed into an observable head overlap rate, avoiding the impossible task of directly probing tens of billions of parameters.

3.  **Four categories of copying forms**:
    - **Function**: Distinguishes whether the reduction in prompt-following is due to the model stopping copying or changing the copying format; reveals distinct internal mechanisms across models.
    - **Mechanism**: Classifies responses into (a) **exact copy**: content and format match prompt ("There are 3 cats" for N=2); (b) **soft copy**: content follows prompt but format differs ("There are three cats"); (c) **format copy**: content is correct but format mimics prompt ("There are 2 cats"); (d) **no copy**: content is correct and format is free ("There are two cats").
    - **Design Motivation**: Aggregate metrics do not show "why hallucination decreases." This classification reveals that Qwen-VL actually **increases** format copying after PIH ablation, while LLaVA-OV shows total suppression and a massive shift towards image attention.

### Loss & Training
**Training-free**. This is an inference-only mechanistic study: all interventions are implemented via hooks injecting mean activations. All experiments could be completed on a single RTX 3090 (approx. 200–300 GPU hours including exploratory tests).

## Key Experimental Results

### Main Results: PIH-head ablation effects (CountBench, avg $k \in \{1,...,5\}$)

| Metric | LLaVA-OV | Qwen-VL | Janus-Pro |
| :--- | :--- | :--- | :--- |
| Baseline prompt Exact Match (↑, Pre-intervention) | 76.89 | 78.49 | 80.32 |
| Baseline prompt Exact Match (↑, **After PIH Ablation**) | **81.24** (+4.35) | 79.29 (+0.80) | 79.41 (−0.91) |
| Misaligned **Prompt Match** (↓, Pre-intervention) | 42.58 | 56.51 | 64.10 |
| Misaligned Prompt Match (↓, Random Ablation) | 37.80 | 54.60 | 58.30 |
| Misaligned Prompt Match (↓, **PIH Ablation**) | **1.42** | **3.22** | **10.19** |
| Misaligned **True-Count Match** (↑, Pre-intervention) | 45.68 | 37.70 | 30.54 |
| Misaligned True-Count Match (↑, **PIH Ablation**) | **77.80** | **70.66** | **70.90** |

PIH-head ablation pushes prompt-following nearly to zero, and true count recovery increases by 30–40 percentage points; it does not harm baseline counting (LLaVA-OV even improved by 4.35%). Performance fluctuations on MM-Vet/POPE were $\le 2\%$, proving PIH-heads are task-specific rather than "global."

### Ablation Study: Color Task (Visual CounterFact) for cross-task validation

| Response Type | LLaVA-OV Pre | LLaVA-OV Post | Qwen-VL Pre | Qwen-VL Post | Janus-Pro Pre | Janus-Pro Post |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| No PIH (Combined) | 0.96 | **95.21** | 20.27 | **79.72** | 14.78 | **55.42** |
| PIH (Combined) | 99.04 | **4.79** | 79.73 | **20.28** | 85.22 | **44.58** |

PIH suppression on the color task: LLaVA-OV **94.25%**, Qwen-VL **59.45%**, Janus-Pro **40.64%**, achieved entirely zero-shot using heads identified in the counting task.

### Key Findings
- **The "N=4 Threshold" of PIH**: When the true number of objects $N \le 4$, models often correct prompt errors. For $N \ge 5$, prompt match surges to 80–90% regardless of offset size—even if $k=50$ (asking to "describe 59 cats" when there are 9), the model still describes 59. Pearson correlations prove that **lower visual confidence correlates with more severe PIH**.
- **PIH-heads concentrated in LM Layers 0-1**: In the top-10 heads, 5 are in L0 for Qwen-VL, 7/10 in L0 for LLaVA-OV, and 3/10 in L0-1 for Janus-Pro. The top heads (L0H3, L0H6) are identical for LLaVA-OV and Qwen-VL, strongly supporting the claim that "PIH is an LM-internal information routing issue."
- **Three Models = Three PIH Mechanisms**: LLaVA-OV follows a "total copying suppression + 12% shift of attention to images" route; Janus-Pro suppresses format copying without increasing visual dependence; Qwen-VL actually **increases format copying but suppresses soft copying** after ablation. This indicates identical behavioral symptoms (decreased prompt-following) can emerge from different internal mechanisms.
- **No Side Effects for Interventions**: Stability on MM-Vet / POPE / CalTech101 proves PIH-heads are highly specialized and do not handle general instruction following.

## Highlights & Insights
- **Cross-model head overlap as a diagnostic probe**: By comparing models with shared LMs but different visual backbones, the authors transformed the "where is PIH" question into a quantifiable experiment—a methodology that can be extended to any "VLM behavior vs. LM behavior" attribution problem.
- **Mechanism Isomorphism $\neq$ Implementation Isomorphism**: All three models reduced PIH through mean ablation of the same class of heads, but dissection of copying forms revealed distinct internal implementations. This serves as a reminder for interpretability research that matching top-line metrics does not equate to identical mechanisms.
- **Engineering value of inference-time intervention**: Mean ablation of 3–10 heads is deployable via hooks with almost zero cost for production VLM services, offering a lightweight alternative to RLHF/DPO for hallucination mitigation.
- **Cognitive significance of the $N \ge 5$ threshold**: This aligns closely with the human visual "subitizing range" ($\le 4$ objects), suggesting models may internalize similar "precise for small numbers, estimation for large numbers" priors during pre-training.

## Limitations & Future Work
- **Limitations**: (1) Only 7B scale VLMs were studied; whether 70B+ models are isomorphic is uncertain; (2) Attention patterns themselves are not fully interpretable; (3) The reason for such vast mechanical differences between the three models was not explained; (4) Second-order effects (attention redistribution of other heads after intervention) were not tracked.
- **Future Directions**: Use path patching to extract the causal path from PIH-head to output logits; extend to more complex modality conflicts (spatial relations, attributes, actions); verify if the early-layer concentration of heads holds for 70B models.

## Related Work & Insights
- **vs. Frank 2021 / Salin 2022 (textual bias of VLM)**: They observed the phenomenon (textual bias), while this work provides a mechanistic explanation.
- **vs. Olsson 2022 (induction heads in LM)**: Induction heads explain in-context learning; PIH heads are their "malignant cousins" in VLM cross-modal conflict. Both concentrate in early layers, confirming a unified picture of "early layers handling shallow copying."
- **vs. Nikankin 2025 (modality-specific circuits)**: They partitioned "vision vs. text" circuits; this work operates directly on the interface (PIH) between those circuits.

## Rating
- Novelty: ⭐⭐⭐⭐☆ PIH as a controlled research protocol and the cross-model head overlap attribution method are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Tested three models, two tasks, three sanity checks, and copying form dissection.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure with well-connected sections.
- Value: ⭐⭐⭐⭐☆ An important datapoint for VLM mechanistic interpretability; provides a zero-cost inference-time mitigation strategy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)
- [\[CVPR 2026\] Overthinking Causes Hallucination: Tracing Confounder Propagation in Vision Language Models](../../CVPR2026/hallucination/overthinking_causes_hallucination_tracing_confounder_propagation_in_vision_langu.md)
- [\[ACL 2026\] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/hallucination/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)

</div>

<!-- RELATED:END -->
