---
title: >-
  [Paper Note] Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating
description: >-
  [ICML 2026][Multimodal VLM][LVLM Hallucination] CRG performs a precise linear decomposition of each attention head's output into visual and textual routes. It estimates the causal "do-effect" of these routes on the curre…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "LVLM Hallucination"
  - "Causal Intervention"
  - "Attention Head Gating"
  - "Route Decomposition"
  - "Training-free"
date: 2026-05-08
content_hash: 98500aefb89ad360
---

# Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating

**Conference**: ICML 2026  
**arXiv**: [2605.24024](https://arxiv.org/abs/2605.24024)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: LVLM Hallucination, Causal Intervention, Attention Head Gating, Route Decomposition, Training-free  

## TL;DR
CRG performs a precise linear decomposition of each attention head's output into visual and textual routes. It estimates the causal "do-effect" of these routes on the current token using one forward and one backward gradient. By suppressing the textual routes only in heads where visual/textual signs conflict and the VRI is low (indicating prior dominance), it systematically mitigates language prior hallucinations in LVLMs without requiring training.

## Background & Motivation
**Background**: Large Vision-Language Models (LVLMs) have become the mainstream interface for image question answering and description generation. However, "hallucinations"—generating semantically fluent content unrelated to the image—remain a major reliability bottleneck for deployment. Training-free inference-time interventions have become a popular research direction as they require no additional computational power or data. Mainstream routes include output-level decoding strategies (e.g., VCD, OPERA, MaskCD) and internal interventions based on attention proxies (e.g., PAI, VTI).

**Limitations of Prior Work**: Decoding-level interventions treat the model as a black box and fail to localize which components cause the model to select incorrect tokens. Internal interventions based on "Visual Attention Ratio" (VAR) assume that "more attention equals stronger visual evidence." This assumption fails under softmax normalization and value vector coupling—a head might have high visual attention, but its value vector may be nearly orthogonal to the gradient direction, contributing nothing to the decision. Furthermore, such methods typically scale entire heads, suppressing useful visual routes alongside problematic ones.

**Key Challenge**: Correlation metrics (attention quality) $\neq$ Causal contribution (actual change in decision score under do-intervention). To truly localize heads where language priors override visual evidence, one must perform decision-aligned causal interventions rather than simply examining attention maps.

**Goal**: (1) Provide a tool to distinguish the "causal effect of the visual route" from the "causal effect of the textual route" on decisions without retraining; (2) Precisely identify "prior-dominated" heads using sign conflicts between the two; (3) Suppress only the textual route while preserving the visual route, operating online during decoding.

**Key Insight**: It is observed that in the multi-head attention output $O_{l,h}=\alpha_{l,h}V_{l,h}$, the value matrix $V_{l,h}$ can be precisely split into $O_{l,h}^{\mathrm{vis}}+O_{l,h}^{\mathrm{txt}}$ using diagonal masks based on the index sets of visual/textual tokens. This allows for a "do-intervention" on an individual route without modifying the attention weights.

**Core Idea**: Split each head internally into two routes, quantify their respective do-effects on the current token decision, and suppress the textual route only for conflicting heads (where visual is positive/textual negative or vice versa). This systematically eliminates the influence of language priors token-by-token.

## Method

### Overall Architecture
CRG (Causal Route Gating) is an inference-time module embedded into the decoding loop. It executes three steps for each generated token: (1) Precisely split each head's output into a visual route $O^{\mathrm{vis}}_{l,h}$ and a textual route $O^{\mathrm{txt}}_{l,h}$ based on token modality; (2) Estimate the causal effects $\widehat{\Delta}^{\mathrm{vis}}_{l,h}$ and $\widehat{\Delta}^{\mathrm{txt}}_{l,h}$ using one forward and one backward gradient, normalizing them into a Visual Reliance Index ($\mathrm{VRI}_{l,h}$); (3) Categorize heads into Agreement, Conflict-A, or Conflict-B based on the signs of $(\widehat{\Delta}^{\mathrm{vis}},\widehat{\Delta}^{\mathrm{txt}})$. For conflicting heads, the textual gate $g^{\mathrm{txt}}_{l,h}$ is suppressed using a rank-based smoothing schedule for the bottom-$k$ VRI heads. Model weights, visual routes, and KV-cache remain unchanged throughout.

### Key Designs

1.  **Intra-head Precise Route Decomposition + Decision-Aligned Causal Route Effect (CRE)**:
    - **Function**: Decomposes each multi-head attention head output into visual and textual routes and defines their causal "do-effect" on the current token decision.
    - **Mechanism**: Diagonal selection matrices $S_{\mathrm{vis}}, S_{\mathrm{txt}} \in \{0,1\}^{L\times L}$ are used to mask non-target rows of the value matrix, yielding $O^{\mathrm{vis}}_{l,h}=\alpha_{l,h}(S_{\mathrm{vis}}V_{l,h})$ and $O^{\mathrm{txt}}_{l,h}=\alpha_{l,h}(S_{\mathrm{txt}}V_{l,h})$. Since $S_{\mathrm{vis}}+S_{\mathrm{txt}}=I_L$ and $S_{\mathrm{vis}}S_{\mathrm{txt}}=0$, the identity $O_{l,h}=O^{\mathrm{vis}}_{l,h}+O^{\mathrm{txt}}_{l,h}$ holds strictly. Scalar gates $g^{\mathrm{vis}}_{l,h}, g^{\mathrm{txt}}_{l,h} \in \mathbb{R}_{\geq 0}$ are added before $W^O_l$. By performing single-head interventions at $(l,h)$ while keeping others at a $(1,1)$ baseline, the task-specific scalar score $\ell$ defines $\Delta^{\mathrm{vis}}_{l,h}=\ell_{l,h}(1,1)-\ell_{l,h}(0,1)$ and $\Delta^{\mathrm{txt}}_{l,h}=\ell_{l,h}(1,1)-\ell_{l,h}(1,0)$. For generative tasks, $\ell=\log p(y^*)$; for binary QA, $\ell=\log p(\mathrm{Yes})-\log p(\mathrm{No})$. These are aggregated into $\mathrm{VRI}_{l,h}=|\Delta^{\mathrm{vis}}_{l,h}|/(|\Delta^{\mathrm{vis}}_{l,h}|+|\Delta^{\mathrm{txt}}_{l,h}|+\varepsilon)$ for ranking.
    - **Design Motivation**: Attention quality proxies like VAR only look at weights, not values, and are susceptible to softmax competition (a drop in textual logit increases VAR even if visual evidence is unchanged). CRE defines effect by "how the decision score changes if this route is turned off," replacing "correlation" with "causality" at a fine-grained route level to ensure precise intervention with minimal disruption to visual pathways.

2.  **One-Forward One-Backward First-Order Do-Effect Estimator**:
    - **Function**: Estimates $\Delta^{\mathrm{vis}}_{l,h}$ and $\Delta^{\mathrm{txt}}_{l,h}$ for each decoded token with minimal overhead, enabling online causal intervention.
    - **Mechanism**: A standard forward pass is performed with all gates set to 1, caching $O^{\mathrm{vis}}_{l,h}$ and $O^{\mathrm{txt}}_{l,h}$. A backward pass via `torch.autograd.grad` on the decision score $\ell$ yields the sensitivity $G_{l,h}=\partial\ell/\partial\tilde O_{l,h}$ at each head's pre-$W^O_l$ tensor. Per Proposition 3.1, the directional derivative along the gate axis provides the estimates $\widehat{\Delta}^{\mathrm{vis}}_{l,h}=\langle G_{l,h},O^{\mathrm{vis}}_{l,h}\rangle$ and $\widehat{\Delta}^{\mathrm{txt}}_{l,h}=\langle G_{l,h},O^{\mathrm{txt}}_{l,h}\rangle$, aligning with the first-order term of the exact do-difference. These are used to calculate $\widehat{\mathrm{VRI}}_{l,h}$.
    - **Design Motivation**: Exact two-point do-effects would require an additional forward pass for every $(l,h)$ with gates turned off, costing $O(LH)$ times the decoding cost. The authors prove that if $\ell$ is a differentiable function of the gates, first-order expansion provides a rigorous estimate via one backward pass without touching the KV-cache. Since downstream logic only requires the sign and ranking rather than exact values, this approximation is sufficient, aligning with theories suggesting coarse effect estimation suffices for near-optimal budget allocation.

3.  **Conflict-Aware Textual Route Gating (Conflict-A/B + Rank-based Schedule)**:
    - **Function**: Categorizes heads based on the sign relationship of route do-effects, selects the top-$k$ via VRI, and suppresses the textual gate using a rank-based smoothing schedule while keeping visual gates at 1.
    - **Mechanism**: Heads are divided into four quadrants based on signs of $(\widehat{\Delta}^{\mathrm{vis}},\widehat{\Delta}^{\mathrm{txt}})$. Same signs ($+,+$ or $-,-$) indicate "Agreement," where no intervention occurs. $(+,-)$ is "Conflict-A" (visual supports decision, text opposes), treated as textual noise and mildly suppressed. $(-,+)$ is "Conflict-B" (visual opposes, text supports), a hallmark of hallucination, and is strongly suppressed. Sets $\mathcal{H}_A, \mathcal{H}_B$ are constructed; within each, top-$k$ heads with the smallest VRI form $\mathcal{S}$. After ascending rank $i$, values are normalized to $s_i=i/(|\mathcal{S}|-1)$, and the gate is $g^{\mathrm{txt}}_{(i)}=g_{\min}+(g_{\max}-g_{\min})\cdot\mathrm{clip}(s_i^\gamma,\epsilon,1-\epsilon)$. Conflict-A uses a mild $(0.5, 1.0)$ range, while Conflict-B uses a strong $(0, 0.5)$ suppression range.
    - **Design Motivation**: In Agreement heads, both routes push the decision in the same direction, likely representing correct multimodal fusion; cutting these would harm performance. Only conflicting heads exhibit "ignored visual evidence." Low VRI indicates these heads are almost entirely text-driven (typical "language prior dominance"). The rank-based schedule ensures smooth suppression based on VRI ranking, avoiding decoding degradation from hard thresholds or "zeroing out" while sparing grounded reasoning heads.

### Loss & Training
Completely training-free: No parameters are updated, no supervision is used, and there is no additional training phase. All hyperparameters (top-$k, \gamma, \epsilon, g_{\min/\max}^{A/B}$) are determined once on a small validation set and kept fixed across models. The only overhead is one autograd backward pass per token. Specifically, for the current token, a standard forward pass determines $y^*$ and caches tensors; a backward pass computes $G_{l,h}$ to obtain $\widehat{\Delta}^{\mathrm{vis}/\mathrm{txt}}_{l,h}$ and $\widehat{\mathrm{VRI}}_{l,h}$; after applying gates via the rank schedule, a final forward pass for the "actual decoding" generates the distribution. The KV-cache is fully reusable as only scalar gates before $W^O_l$ are modified.

## Key Experimental Results

### Main Results
On LLaVA-1.5-7B, Qwen-VL-Chat, and Qwen2.5-VL-7B-Instruct, CRG consistently outperforms Regular, VCD, OPERA, PAI, and VTI across five benchmarks (POPE, CHAIR, MME, MMHal-Bench, AMBER).

| Dataset / Setting | Metric | Regular | Strongest Baseline | CRG | Gain vs. Regular |
| :--- | :--- | :--- | :--- | :--- | :--- |
| POPE-Random / LLaVA-1.5-7B | Acc / F1 | 83.29 / 81.33 | VTI 89.50 / 88.89 | **90.30 / 89.51** | +7.01 / +8.18 |
| POPE-Adv / Qwen2.5-VL-7B | Acc / F1 | 82.79 / 83.15 | VTI 85.78 / 85.14 | **86.98 / 87.07** | +4.19 / +3.92 |
| CHAIR / LLaVA-1.5-7B | $C_S\downarrow$ / $C_I\downarrow$ / Recall↑ | 52.8 / 15.9 / 77.3 | VTI 37.6 / 12.9 / 79.3 | **34.2 / 11.2** / 77.8 | $C_S$ −18.6 |
| AMBER / LLaVA-1.5-7B | CHAIR↓ / F1↑ / Score↑ | 8.3 / 73.7 / 82.70 | — | **4.6 / 77.5 / 86.45** | Score +3.75 |

### Ablation Study
| Configuration | POPE-Avg↑ | $C_S$↓ | MMHal↑ | MME↑ | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Regular | 81.37 | 52.8 | 2.23 | 1640 | Baseline |
| CRG w/o A (Conflict-B only) | Med | Med | Med | Med | Target prior-dominated heads |
| CRG w/o B (Conflict-A only) | Low | High | — | — | Suppress noise-type text routes |
| **CRG (A+B)** | **Best** | **34.2** | **Best** | **Best** | Full conflict-aware strategy |

### Key Findings
- Conflict-B (Visual oppose, Text support) is the strongest single signal for hallucinations; intervening on it yields higher gains than Conflict-A. However, combining both maximizes hallucination reduction without degrading MME scores, suggesting complementary causal mechanisms.
- Figure 3 visualizations show that while VAR and VRI patterns match in early layers, they diverge significantly in middle layers: VAR becomes nearly flat, while VRI maintains significant structure, validating that attention quality is not a reliable proxy for decision-related visual grounding.
- Intervening only on the textual gate while keeping the visual gate at 1 allows MME to improve in grounding-sensitive categories (Existence, Count, Position, Color) and high-level reasoning (Commonsense, Numerical) simultaneously, showing no harm to general multimodal capabilities.
- On CHAIR, $C_S$ dropped from 52.8 to 34.2 (LLaVA-1.5-7B) while Recall remained stable or improved (76.4→81.6), indicating that mitigation is achieved by "speaking correctly" rather than "speaking less."

## Highlights & Insights
- Using complementary diagonal masks $S_{\mathrm{vis}}+S_{\mathrm{txt}}=I_L$ and $S_{\mathrm{vis}}S_{\mathrm{txt}}=0$ to strictly split $V_{l,h}$ allows for precise do-intervention without retraining or altering the attention structure.
- The use of first-order do-effect estimation, supported by theory and the requirement for only signs and rankings, transforms causal intervention from an offline analysis tool into an online decoding component with costs comparable to standard backpropagation.
- Explicitly defining hallucinations as a sign conflict (Visual $-$, Text $+$) is closer to the essence of the problem than "visual attention ratio." Distinguishing Conflict-A (noise) from Conflict-B (prior-dominance) allows for a "classify then differentiate" paradigm transferable to other multimodal tasks.
- The counterfactual analysis in Section 4.2 ($\partial R/\partial s_k=-R\alpha_k<0$) provides a clean counter-example to "attention is explanation," showing the spurious correlation between dropping textual logits and rising VAR.

## Limitations & Future Work
- The bias of first-order estimation under large gate changes is not strictly characterized. While a Lipschitz bound is provided, empirical validation only confirms sign-consistency; whether second-order corrections are needed for larger models remains open.
- The extra backward pass per token is described as "moderate overhead," but absolute latency for long generation sequences is not provided; this cost may be significant in production systems with complex KV-caching.
- The gate intervals and top-$k$ selection are manually designed. While consistent across models, a dynamic or self-adaptive version might further unlock potential as LVLM scale increases.

## Related Work & Insights
- **vs. VCD / OPERA / MaskCD**: While decoding heuristics only adjust the output distribution, CRG intervenes at the intra-head level, providing stronger localization.
- **vs. PAI / VTI**: Unlike PAI (spatial guidance) or VTI (latent direction control), CRG operates at the "intra-head route" granularity and uses do-effects for selection rather than latent representation distance.
- **vs. VAR-style selection (Jiang et al. 2025 etc.)**: These methods use attention proxies to scale entire heads. CRG demonstrates that VAR and decision-related grounding are systematically misaligned and that head-level scaling inadvertently suppresses visual routes.
- **vs. CHG (Causal Head Gating)**: CHG learns head-level differentiable gates. CRG pushes causal intervention from "head-level learned gates" to "intra-head route-level inference-time do-effects," offering finer granularity without training.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to use "intra-head visual/textual routes" as the unit of causal intervention with an online first-order estimator.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five benchmarks, three LVLMs, four baselines, and comprehensive ablations; lacks testing on extremely large models or multilingual extensions.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The chain from motivation to theory to algorithm is exceptionally clear.
- **Value**: ⭐⭐⭐⭐⭐ Training-free, plug-and-play, and preserves multimodal capabilities; high deployment value for hallucination control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](../../CVPR2026/multimodal_vlm/residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)
- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](../../ACL2026/multimodal_vlm/mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)
- [\[NeurIPS 2025\] Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](../../NeurIPS2025/multimodal_vlm/causalllava_causal_disentanglement_for_mitigating_hallucinat.md)
- [\[NeurIPS 2025\] When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations](../../NeurIPS2025/multimodal_vlm/when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](../../CVPR2026/multimodal_vlm/hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)

</div>

<!-- RELATED:END -->
