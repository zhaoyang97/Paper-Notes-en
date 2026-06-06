---
title: >-
  [Paper Note] Perceptual Flow Network for Visually Grounded Reasoning
description: >-
  [ICML 2026][Multimodal VLM][Visually Grounded Reasoning] Abandoning the traditional RLVR approach of "hard supervision with precise expert bounding boxes…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Visually Grounded Reasoning"
  - "GFlowNet"
  - "Sub-TB"
  - "Variational Reinforcement Learning"
  - "LVLM Hallucination"
date: 2026-05-08
content_hash: b3ecde16edaba501
---

# Perceptual Flow Network for Visually Grounded Reasoning

**Conference**: ICML 2026  
**arXiv**: [2605.02730](https://arxiv.org/abs/2605.02730)  
**Code**: None  
**Area**: Multimodal VLM / Reinforcement Learning / Visual Reasoning  
**Keywords**: Visually Grounded Reasoning, GFlowNet, Sub-TB, Variational Reinforcement Learning, LVLM Hallucination

## TL;DR
Abandoning the traditional RLVR approach of "hard supervision with precise expert bounding boxes," PFlowNet models the perceptual behavior itself as a structured Perceptual Flow latent variable. It approximates the ideal reasoning-oriented posterior with a variational distribution $p_\theta(Z|X)$, and is trained using Sub-TB variational RL, multi-dimensional rewards, and Vicinal Geometric Shaping. As a result, the 8B Qwen3-VL achieves a new SOTA of 90.6% on V* Bench and 67.0% on MME-RealWorld-lite.

## Background & Motivation

**Background**: To mitigate language bias and hallucination in LVLMs, recent methods (look-twice, VGR, grounded thinking, etc.) use RLVR to distill geometric priors from visual experts (e.g., GroundingDINO) into LVLMs, enforcing a "first localize key regions, then answer" reasoning process.

**Limitations of Prior Work**: The authors conducted a critical probing experiment on V* using Qwen2.5-VL—expanding expert-annotated boxes isotropically to obtain geometric priors with different IoUs, then feeding the corresponding crops to the LVLM and measuring answer accuracy. The counterintuitive result: the most precise expert box is not the best reasoning evidence; there exists a sweet spot. The reason is that visual experts are designed for object detection, prioritizing geometric precision while ignoring the "context needed for reasoning." Overly tight boxes cause a "tunnel vision" effect, removing essential peripheral cues.

**Key Challenge**: Existing VGR equates "expert geometric precision" with "quality of reasoning evidence," forcing LVLMs to strictly align with expert boxes. However, the truly useful "golden evidence" for reasoning is instance-specific, and heuristic expansion cannot precisely capture it. This is a fundamental misalignment—optimizing for the wrong target.

**Goal**: (1) Formalize VGR as a distribution modeling problem over latent visual trajectories $Z$; (2) Construct a family of learnable perceptual trajectory representations, decoupling "geometric precision" from "reasoning utility" in the training objective; (3) Use variational RL to encourage exploration of "reasoning-friendly" perceptual behaviors while retaining geometric reliability.

**Key Insight**: Rather than using expert boxes as hard constraints, treat the reasoning trajectory $Z$ as a latent variable, with a model-parameterized variational distribution $p_\theta(Z|X)$ approximating the "ideal VGR posterior" $P_V(Z|X,Y)$; the latter only requires $Z$ to fall within a $\sigma$-vicinity $\mathcal{S}_V$ centered on the golden evidence $G$. The expert box $E$ serves only as a "vicinal reference," not a hard target.

**Core Idea**: Model perception as a Perceptual Flow (planning state + a sequence of grounded perceptual states), use a Sub-Trajectory Balance (GFlowNet-style) variational objective for dense supervision, and design "multi-dimensional rewards + geometric shaping $\omega_\lambda$ active only outside the expert vicinity" to achieve "sufficient exploration without overstepping boundaries."

## Method

### Overall Architecture
PFlowNet divides the LVLM workflow into two decoupled stages: (i) **Flow Generation**: The model samples a Perceptual Flow $Z = (z_0 \to z_1 \to \dots \to z_K)$ from $p_\theta(Z|X)$, where $z_0$ is a planning state wrapped in `<analyze>...</analyze>`, and $z_{\ge 1} = \langle r_k, c_k\rangle$ are wrapped in `<localize>...</localize>`, each containing an RoI box (relative coordinates 0–1000) and a descriptive caption; (ii) **Flow-Guided Reasoning**: The model generates the final answer $Y$ autoregressively based on $Z$ and the cropped visual evidence $I_{RoI}$, with the joint distribution factorized as $p_\theta(Y, Z|X) = p_\theta(Z|X) p_\theta(Y|Z, \langle X, I_{RoI}\rangle)$. Training is two-stage: first, SFT on synthetic perceptual flow data $(X, Z_s)$ for cold start, then variational RFT on $(X, Y, E)$ to optimize $p_\theta(Z|X)$ towards $P_V$.

### Key Designs

1. **Perceptual Flow + Sub-TB Variational Objective**:

    - **Function**: Discretizes "where the model looks and what it sees" into structured trajectories, providing dense supervision for each sub-segment, addressing the issue of PPO-style objectives only rewarding at episode end and high gradient variance.
    - **Mechanism**: Defines Perceptual Flow $Z = (z_0, z_1, \dots, z_K)$, where planning state $z_0$ is natural language, perceptual state $z_k = \langle r_k, c_k\rangle$ is RoI box + caption; special tokens like `<analyze>`, `<localize>` explicitly segment the flow. Introduces Sub-Trajectory Balance from GFlowNet: for any sub-segment $z_{i:j}$, $\mathcal{F}(z_i)\,\mathcal{T}_F(z_{i:j}) = \mathcal{F}(z_j)\,\mathcal{T}_B(z_{j:i})$, leading to the vRFT objective $\mathcal{L}_{vRFT}(\theta)$ (Eq. 2 in the paper), using the squared sum of log-ratios as loss.
    - **Design Motivation**: (a) Explicit structure allows rewards to be defined on each sub-trajectory; (b) Sub-TB provides dense constraints ("every sub-segment must balance"), more stable than PPO; (c) Decoupling perception and reasoning allows independent optimization of $p_\theta(Z|X)$ without contaminating LLM reasoning parameters.

2. **Multi-dimensional Reward (Contrastive Visual + Information Gain)**:

    - **Function**: Ensures the reward function simultaneously measures "perceptual accuracy" and "reasoning utility," avoiding sole focus on geometric IoU.
    - **Mechanism**: For sub-flow $z_{0:k}$, defines $R(z_{0:k}\top) = \left(\prod_{i=1}^k \frac{p_\phi^+(z_i)}{p_\phi^-(z_i)}\right) p_\phi(Y \mid z_{0:k}\top, X)$, where $p_\phi^+(z_i) = p_\phi(c_i \mid I_{r_i})$ is the visual likelihood of the caption within the cropped region, $p_\phi^-(z_i) = p_\phi(c_i \mid I \setminus I_{r_i})$ is the likelihood in the complement region, and $p_\phi$ is a frozen reward model initialized with the policy. The contrastive term $p_\phi^+/p_\phi^-$, in expectation over the trajectory, is interpretable as reverse-KL distillation: $\mathbb{E}[\sum \log(p_\phi^+/p_\phi^-)] = \sum [D_{KL}(q_\theta^i \| p_\phi(\cdot|I\setminus I_{r_i})) - D_{KL}(q_\theta^i \| p_\phi(\cdot|I_{r_i}))]$, encouraging captions to capture privileged information in the crop and avoid noise in the complement; the information gain term $p_\phi(Y|z_{0:k}\top, X)$ ensures the chosen trajectory truly contributes to the final answer $Y$.
    - **Design Motivation**: Embeds "vision-grounded" and "reasoning-oriented" as two independent reward constraints, naturally suppressing reward hacking—simply precise boxes with generic captions, or vice versa, cannot achieve high rewards.

3. **Vicinal Geometric Shaping**:

    - **Function**: Retains expert priors as a "safety barrier" to prevent out-of-bounds exploration, but does not force the model to 100% mimic the expert, allowing free discovery of higher-utility perceptual behaviors within the expert vicinity.
    - **Mechanism**: Defines symmetric Chamfer-IoU distance $d_{IoU}(A, B) = 1 - 0.5(IoU_{A\to B} + IoU_{B\to A})$, then an $\varepsilon$-vicinity $\mathcal{B}_\varepsilon(E) = \{z_{0:k} \mid d_{IoU}(r_{1:k}, E) \le \varepsilon\}$ centered on the expert RoI set $E$. The shaping weight $\omega_\lambda(z_{0:k}, E) = \exp(-\lambda \mathbb{I}(z_{0:k} \notin \mathcal{B}_\varepsilon(E)))$ penalizes only trajectories outside the vicinity (with strength $\lambda$), while rewards $R$ are unconstrained within the vicinity. The final shaped reward $R_\lambda(z_{0:k}\top) = R(z_{0:k}\top) \omega_\lambda(z_{0:k}, E)$ is used in the Sub-TB flow objective's $\mathcal{F}$.
    - **Design Motivation**: Theorem 3.1 provides a TV distance upper bound, showing that as $\lambda \to 0$ it degenerates to standard MLE (losing geometric constraints), and as $\lambda \to \infty$ it degenerates to expert-guided RLVR (limited by expert bias); Theorem 3.4 further proves the existence of $\lambda^\star$ such that the bound is strictly tighter than the lower bounds of both, i.e., under ideal assumptions, PFlowNet strictly outperforms both baselines.

### Loss & Training

**Data Pipeline**: Uses Gemini-3-flash / GPT-4o as teachers to randomly expand expert RoIs and generate synthetic flows $Z_s$, then uses a verifier to sample answers with and without $Z_s$, splitting by pass@k: $k=1$ (too easy, discarded), $k>1$ but $2\le k_{w/Z_s}\le 16$ go to RFT set, $k_{w/o Z_s} > 16$ and $k_{w/Z_s} = 1$ go to cold-start set. **Cold Start**: Standard SFT, minimizing cross-entropy between $p_\theta(Z|X)$ and $Z_s$. **vRFT**: Trains with the above three designs combined, parallelizing Sub-TB computation over $L$ sampled trajectories (sub-prefixes of each trajectory share reward cache).

## Key Experimental Results

### Main Results
Base model is Qwen3-VL 8B, evaluated on V* Bench (complex visual search), TreeBench (perception + reasoning tree evaluation), and MME-RealWorld-Lite (OCR/remote sensing/charts/surveillance/autonomous driving).

| Dataset | Metric | PFlowNet | Prev. SOTA / Baseline | Gain |
|---|---|---|---|---|
| V* Bench | Overall Acc | **90.6%** | Qwen3-VL 8B 77.5% | +13.1% vs base, new SOTA |
| TreeBench | Overall Acc | Qwen3-VL+13.1% / +10.4% | Qwen3-VL 8B | +10.4 vs base |
| MME-RealWorld-Lite | Overall Acc | **67.0%** | Baselines 43–52% | +21% vs Qwen3-VL 8B |
| TreeBench (Attributes) | Acc | 64.69 (example) | Most 50–60 | Significant lead |

Note: Table 2 shows that large models like InternVL3-78B / Qwen2.5-VL-72B achieve only 46.4% / 42.2% on TreeBench/MME-RealWorld-Lite, while PFlowNet with 8B significantly outperforms 70B-scale baselines, indicating the effectiveness of the training paradigm over parameter scaling.

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Full PFlowNet ($\lambda^\star, \varepsilon^\star$) | Tightest TV bound / SOTA performance | All three components |
| $\lambda \to 0$ | $D_{TV} \to 1 - s_V$ | Degenerates to MLE, geometric prior lost |
| $\lambda \to \infty$ | $D_{TV} \to 1 - q$ | Degenerates to expert-guided RLVR, locked by expert bias |
| $\varepsilon \to 0$ | Vicinity shrinks to a point, $q \to 0$ | Reward signal fails, bound loosens |
| Increase $\varepsilon$ (keep $\mathcal{B}_\varepsilon \subseteq \mathcal{S}_V$) | $q \uparrow$, bound tightens monotonically | Wider is better within valid domain |
| $\varepsilon > \sigma$ | Vicinity exceeds $\mathcal{S}_V$ | Geometric guidance diluted, performance drops |

### Key Findings
- The probing experiment directly refutes the "expert is most precise" intuition: answers using precise expert boxes have lower accuracy than those with moderately expanded IoU boxes, confirming the "tunnel vision" effect.
- Theorems 3.1–3.4 provide provable improvements "strictly superior to MLE and expert-guided RLVR," under the condition $\mathcal{B}_\varepsilon \subseteq \mathcal{S}_V$ (vicinity must not exceed the valid domain).
- Excellent performance-efficiency tradeoff: the 8B model outperforms 78B baselines, showing that the structured decomposition of perceptual flow + variational RL greatly improves sample efficiency.
- Good scaling at test time: performance continues to improve with larger sampling budgets, indicating that $p_\theta(Z|X)$ learns a truly explorable distribution rather than a single point.

## Highlights & Insights
- Reformalizing VGR as "latent variable posterior approximation" is the paper's major conceptual advance—replacing RLVR's "geometric alignment" framework with a "distribution approximation" framework turns the previously unsolvable "expert bias" problem into a tunable $\lambda$/$\varepsilon$ hyperparameter issue.
- Sub-TB provides dense supervision while maintaining GFlowNet's exploratory nature, making it especially suitable for long-chain perceptual behaviors; bridging GFlowNet concepts from molecule generation to LVLM reasoning is both novel and natural.
- The insight that the contrastive term $p_\phi^+/p_\phi^-$ in the multi-dimensional reward is equivalent to reverse-KL distillation is elegant, directly turning the caption likelihood difference "inside/outside the box" into a KL term difference, with clear physical and optimization semantics.
- The design philosophy of Vicinal Geometric Shaping ("expert as reference, not target") is transferable to any RLHF scenario needing a balance between expert prior and exploration, such as tool calling for code agents or robot policy distillation.

## Limitations & Future Work
- Theoretical assumptions are strong (Assumption 1/2, $d_{eff}$-regularity, etc.), and it is unclear whether real LVLM distributions satisfy them; the bounds are only idealized upper limits.
- Perceptual Flow currently supports only "box + caption" binary states; extension is needed for finer-grained perceptual behaviors (e.g., masks, point clouds, video frames).
- The data pipeline relies on strong teacher models (Gemini-3-flash / GPT-4o) to synthesize flows, posing an explicit barrier for open-source reproduction; cold start data quality directly impacts final performance.
- $\lambda$ and $\varepsilon$ are still fixed hyperparameters, lacking adaptive scheduling; the theorem only proves the existence of optimal $\lambda^\star$, without specifying how to choose it.
- Multi-dimensional rewards require maintaining a reward model $p_\phi$ (initialized with the policy but frozen), doubling memory cost during training.

## Related Work & Insights
- **vs Look-Twice / VGR / TraceVL**: These works use expert geometry from GroundingDINO as hard rewards, limited by expert bias; PFlowNet reduces the expert to a vicinal reference and uses variational objectives to self-learn optimal perception.
- **vs DeepSeek-R1 (RLVR paradigm)**: R1 applies RLVR to math/code with verifiable rewards; PFlowNet extends RLVR to scenarios where perceptual behavior cannot be directly verified, using contrastive likelihood + information gain instead of ground-truth rewards.
- **vs GFlowNets (Sub-TB)**: Original GFN is for discrete combinatorial object generation; this work is among the first to introduce Sub-TB to LVLM multimodal reasoning.
- **vs Vicinal Risk Minimization**: Borrows the "vicinal shaping" idea from classic VRM, but applies it to trajectory space instead of input space, providing a new RL regularization primitive.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reformalizes VGR, introduces GFlowNet Sub-TB, designs vicinal geometric shaping; the overall framework is coherent and opens a new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on V* / TreeBench / MME-RealWorld-Lite, compared to large model baselines; however, many ablation details are in the appendix.
- Writing Quality: ⭐⭐⭐⭐ Clear structure in theory and method, with a consistent flow from "probing experiment → formalization → Sub-TB → reward → geometric shaping."
- Value: ⭐⭐⭐⭐⭐ Paradigm-shifting for future grounded-reasoning LVLMs; vicinal shaping and multi-dimensional rewards are highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MathFlow: Enhancing the Perceptual Flow of MLLMs for Visual Mathematical Problems](../../ACL2026/multimodal_vlm/mathflow_enhancing_the_perceptual_flow_of_mllms_for_visual_mathematical_problems.md)
- [\[ICML 2026\] The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design](the_perceptual_bandwidth_bottleneck_in_vision-language_models_active_visual_reas.md)
- [\[AAAI 2026\] Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models](../../AAAI2026/multimodal_vlm/concept-rulenet_grounded_multi-agent_neurosymbolic_reasoning.md)
- [\[CVPR 2026\] Thinking Diffusion: Penalize and Guide Visual-Grounded Reasoning in Diffusion Multimodal Language Models](../../CVPR2026/multimodal_vlm/thinking_diffusion_penalize_and_guide_visual-grounded_reasoning_in_diffusion_mul.md)
- [\[ACL 2026\] A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends](../../ACL2026/multimodal_vlm/a_survey_on_mllm-based_visually_rich_document_understanding_methods_challenges_a.md)

</div>

<!-- RELATED:END -->
