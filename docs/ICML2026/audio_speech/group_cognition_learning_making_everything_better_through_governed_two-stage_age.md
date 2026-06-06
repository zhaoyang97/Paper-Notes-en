---
title: >-
  [Paper Note] Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration
description: >-
  [ICML 2026][Audio & Speech][Multimodal Fusion] Addressing the chronic issues of "modality dominance" and "spurious modality coupling" in centralized multimodal fusion…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Multimodal Fusion"
  - "Modality Dominance"
  - "Spurious Coupling"
  - "Marginal Gain Gating"
  - "Routing Auditing"
date: 2026-05-08
content_hash: 3f3c68139d2c94e3
---

# Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration

**Conference**: ICML 2026  
**arXiv**: [2605.00370](https://arxiv.org/abs/2605.00370)  
**Code**: None  
**Area**: Multimodal Sentiment Analysis / Intent Recognition  
**Keywords**: Multimodal Fusion, Modality Dominance, Spurious Coupling, Marginal Gain Gating, Routing Auditing

## TL;DR
Addressing the chronic issues of "modality dominance" and "spurious modality coupling" in centralized multimodal fusion, GCL reformulates multimodal learning as a **protocol-based collaboration involving four agents across two stages**: in the first stage, Routing/Auditing agents use marginal prediction gain to determine cross-modal communication per sample; in the second stage, Public-Factor/Aggregation agents decouple and aggregate shared semantics and private specializations, achieving SOTA results on MOSI, MOSEI, and MIntRec.

## Background & Motivation
**Background**: The current mainstream of multimodal learning (Language + Acoustic + Visual) is **centralized fusion**—either through tensor products (TFN/LMF), cross-modal attention (MulT), or BERT-based end-to-end fine-tuning. Representative advancement paths include representation structuring (MISA, FDMER, ConFede using shared/private decomposition) and optimization intervention (CGGM, MCIS for rebalancing gradients).

**Limitations of Prior Work**: All these methods implicitly assume that as long as the task loss is backpropagated end-to-end, the optimal interaction pattern will naturally emerge. In practice: (1) gradients tend to concentrate along the "modality path that reduces loss most easily," leaving weak modalities undertrained and the model extremely fragile to noise; (2) end-to-end fusion rewards any cross-modal correlation, including incidental ones—tightly binding modalities that should not be coupled, which leads to a sharp decline in out-of-distribution performance.

**Key Challenge**: Interaction learning and representation learning are coupled within the same loss, without any **explicit signal** to distinguish "useful interaction" from "redundant interaction." While Routing/MoE works introduce adaptability, they still lack an auditing layer to question "how much marginal gain this edge actually brings."

**Goal**: (1) Transform the topology of cross-modal information flow into an observable and supervised object; (2) utilize an explicit redundancy control term to eliminate spurious coupling; (3) ensure the aggregation stage no longer allows a single dominant modality to "take all" but instead relies on sample-wise shared semantics for weighting.

**Key Insight**: The authors transfer the "division of labor + governance" metaphor from LLM agents to multimodal fusion. Rather than allowing a black-box network to freely couple three modalities, the interaction process is decomposed into four distinct responsibilities: "proposal routing → access auditing → public factor extraction → weighted aggregation," where each agent is supervised by its own auxiliary loss.

**Core Idea**: Use "marginal prediction gain" as a hard metric for interaction admission, combined with a redundancy contrastive penalty, to transform end-to-end fusion into a two-stage auditable protocol.

## Method

### Overall Architecture
Input: Samples $x_m$ for each modality $m\in\{l,a,v\}$ are processed by their respective encoders to obtain $h_m$.
Output: Classification/Regression prediction $\hat o$.
Four intermediate agents execute two stages serially:

1.  **Stage 1 — Selective Interaction**: The Routing Agent assigns a logit $\rho_{m\to n}$ to each directed edge $(m\to n)$ and packages a message $u_{m\to n}$; the Auditing Agent estimates the marginal gain $\hat\Delta_{m\to n}$ brought by fusing this message into $h_n$, which is multiplied by the routing probability to obtain the gate $\alpha_{m\to n}$; gated residual updates yield the refined $z_n$.
2.  **Stage 2 — Consensus Formation**: The Public-Factor Agent distills a shared factor $c$ from $\{z_l,z_a,z_v\}$ using a permutation-invariant operator; the Aggregation Agent uses $c$ as a condition to generate proposals $r_m$ and softmax weights $\pi_m$ for each modality, resulting in the final prediction $\hat o = g_\tau(\sum_m \pi_m r_m, c)$.

Four auxiliary objectives (task / local / public / gain alignment / redundancy) jointly train all agents.

### Key Designs

1.  **Auditing Agent + Marginal Gain Auditing Gate**:
    - **Function**: Transforms the decision of "whether to allow $m$ to send a message to $n$" from a pure optimization decision into a supervised audit—only edges that truly reduce task loss are opened.
    - **Mechanism**: During training, a "teacher gain" is defined as $\Delta_{m\to n}=\ell_\tau(q_n^\tau(h_n),y)-\ell_\tau(q_n^\tau(\tilde h_n^m),y)$, where $\tilde h_n^m=h_n+\phi_{m\to n}(h_n,u_{m\to n})$ is the transient state after message fusion. Since labels are unavailable during inference, a separate gain predictor $\hat\Delta_{m\to n}=g_g^{m\to n}(h_n,u_{m\to n})$ is used. The final gate $\alpha_{m\to n}=\text{softmax}_{j}(\rho_{j\to n})_{j=m}\cdot\sigma_\kappa(\tilde\Delta_{m\to n})$ multiplies the "desire to communicate" (routing) with the "value of communication" (gain). Gated integration follows a residual path: $z_n=h_n+\sum_{m\neq n}\alpha_{m\to n}\cdot\phi_{m\to n}(h_n,u_{m\to n})$. The gain alignment loss $\mathcal L_{\text{gain}}=-\sum_n\sum_{m\neq n}\alpha_{m\to n}\text{stopgrad}(\Delta_{m\to n})$ encourages positive gain edges to open and negative ones to close, with a stopgrad preventing the teacher gain from being contaminated by backpropagation.
    - **Design Motivation**: Previous routers/MoEs only asked "which path to activate," not "is activating this path useful." Explicitly separating teacher–student gain solves two problems: it provides ground-truth supervision during training and maintains inference efficiency by using only the learned predictor.

2.  **Public-Factor Agent + Decoupled Aggregation**:
    - **Function**: Explicitly separates "cross-modal shared semantics" and "per-modality private specialization" to prevent traditional fusion operators from allowing dominant modalities to swallow weak ones.
    - **Mechanism**: A permutation-invariant operator (symmetric attention or global pool + MLP) $c=g_p(z_l,z_a,z_v)$ extracts the public factor, with auxiliary supervision $\mathcal L_{\text{pub}}=\mathbb E\ell_\tau(g_\tau^c(c),y)$ ensuring $c$ itself is predictive. The Aggregation Agent then uses $c$ as context: each modality generates a proposal $r_m=\eta_m(z_m,c)$ and an unnormalized score $s_m=g_a^m(z_m,c)$. Computing $\pi_m=\text{softmax}(\{s_m\}_m)$ yields sample-wise weights, leading to $\hat o = g_\tau(\sum_m \pi_m r_m, c)$.
    - **Design Motivation**: Direct concatenation entangles $c$ and $z_m$, causing $z_a/z_v$ to be neglected once a strong modality dominates. Explicitly isolating $c$ and using it to modulate modality weights $\pi_m$ is equivalent to asking "given what we already know from shared semantics, how much can this modality's private information contribute?" This conditioning prevents incremental info from weak modalities from being drowned out.

3.  **Redundancy Contrastive Regularization $\mathcal L_{\text{red}}$**:
    - **Function**: Applies orthogonal pressure to refined channels $\{z_n\}$ after Stage 1, preventing the routing from allowing all modalities to converge to the same representation despite learning edges.
    - **Mechanism**: A symmetric InfoNCE-style alignment score $\mathcal L_{\text{red}}=\sum_{m<n}D(z_m,z_n)$ is minimized, requiring low mutual information between refined representations.
    - **Design Motivation**: Gating $\alpha$ alone controls information **flow** but does not prevent the loss of modality uniqueness after the flow occurs. $\mathcal L_{\text{red}}$ acts as a repulsive force against "spurious coupling"—ablation shows that removing this term increases MAE on MOSI from $0.685$ to $0.703$ and significantly degrades HSIC/CKA diagnostic metrics.

### Loss & Training
$$\mathcal L_{\text{total}} = \mathcal L_{\text{task}} + \lambda_{\text{loc}}\mathcal L_{\text{loc}} + \lambda_{\text{pub}}\mathcal L_{\text{pub}} + \lambda_{\text{gain}}\mathcal L_{\text{gain}} + \lambda_{\text{red}}\mathcal L_{\text{red}}$$
where $\mathcal L_{\text{loc}}=\sum_m\mathbb E\ell_\tau(q_m^\tau(h_m),y)$ supervises unimodal heads to ensure reliable teacher gain estimation. Optimization uses Adam, batch size 128, weight decay $1\text{e-}4$, patience 6, on A100.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA (TSDA) | Gain |
| :--- | :--- | :--- | :--- | :--- |
| CMU-MOSI | MAE↓ | **0.685** | 0.695 | $-0.010$ |
| CMU-MOSI | Acc-2 | **86.79** | 86.3 | $+0.49$ |
| CMU-MOSI | Acc-7 | **49.06** | 48.6 | $+0.46$ |
| CMU-MOSEI | MAE↓ | **0.520** | 0.529 | $-0.009$ |
| CMU-MOSEI | Acc-2 | **86.78** | 86.3 | $+0.48$ |
| MIntRec | Acc | **72.74** | 72.59 | $+0.15$ |
| MIntRec | F1 | **70.95** | 70.68 | $+0.27$ |

### Ablation Study

| Configuration | MOSI MAE | MOSI Acc-7 | Description |
| :--- | :--- | :--- | :--- |
| Full GCL | 0.685 | 49.06 | Full model |
| w/o Routing Agent | 0.694 | 48.55 | No topological proposals |
| w/o Auditing Agent | 0.699 | 48.00 | No gain gating |
| **Full exchange** | 0.721 | 46.10 | Worse than "Language only" (0.714) |
| w/o Public-Factor Agent | 0.702 | 47.85 | Shared semantics collapse |
| Uniform $\pi_m$ | 0.698 | 48.05 | Loss of sample-adaptive weights |
| w/o $\mathcal L_{\text{red}}$ | 0.703 | 47.70 | Severe degradation of HSIC/CKA |
| **only $\mathcal L_{\text{task}}$** | 0.712 | 46.70 | Degradation without governance terms |

### Key Findings
- **Unregulated full-exchange fusion (0.721) is worse than using the language modality alone (0.714)**—this is the most impactful empirical result: blind "multimodal fusion" can hurt performance by lowering the signal-to-noise ratio. This directly refutes the implicit belief that "adding more modalities will not make things worse."
- **Efficiency Surprises**: GCL has 117.56M parameters and take 20.06s per epoch, reducing parameters by half compared to ConFede (256.98M, 40.12s) and training time by 25% compared to EMOE (143.5M, 26.80s). "Division of labor + governance" is computationally lighter than "stacking MoEs/experts."
- **Noise Robustness**: When Gaussian noise $\sigma\in[0,20]$ is injected into MOSI, GCL's decay curve is significantly flatter than baselines, suggesting the auditing gate actively shuts down dirty signals as SNR drops.
- **Audited Selectivity Experiments** (Fig 3): GCL occupies the high-PGR, moderate-AR quadrant (high positive gain ratio, efficient activation), while NoAudit/Uniform/Full-exchange variants cluster in the high-AR, low-PGR area—meaning they communicate frequently but uselessly.

## Highlights & Insights
- **Treating "Marginal Gain" as a Loss**: Borrowing the $do(\cdot)$ concept from causal inference for multimodal fusion allows "is this interaction worth doing" to become a learnable quantity rather than an emergent property of end-to-end gradient metaphysics. This approach can theoretically be extended to edge activation in any graph-structured network.
- **Teacher gain + stopgrad training / Student gain inference**: A classic "supervised by oracle during training, executed by learner during inference" pattern that adds a supervised signal without sacrificing inference speed. This trick is reusable in distillation, actor-critic, and verifier-guided generation.
- **Full exchange performing worse than unimodal**: This experimental conclusion is worth citing repeatedly as it uses data to challenge the multimodal community's default "more modalities are better" assumption.
- **Public factor as routing context**: Using "shared semantics" as an explicit conditional input for routing is equivalent to first answering "what do we already know in the consensus" before asking "what can private information still contribute," avoiding the double counting of shared information.

## Limitations & Future Work
- The paper is categorized under audio_speech, but it focuses on multimodal sentiment analysis and intent recognition—"acoustic" is just one of three modalities, and there is no evaluation on pure speech/ASR tasks.
- Teacher gain estimation relies on a single-step residual approximation $\tilde h_n^m=h_n+\phi(\cdot)$; marginal effects of multi-step deep interactions are not considered; combinatorial explosion occurs when scaling to long chains or N>3 modalities.
- The primary benchmarks (MOSI/MOSEI/MIntRec) are relatively small (thousands to tens of thousands of samples) and have not been validated on large-scale vision-language datasets (e.g., LAION subsets, AudioSet).
- Lack of comparison with LLM-based multimodal methods (e.g., VideoLLaMA, Qwen-Audio) makes it difficult to determine if GCL's lightweight protocol retains a relative advantage in the LLM era.

## Related Work & Insights
- **vs MISA / FDMER (Disentanglement-based)**: These also separate shared/private components, but disentanglement happens at the representation layer via reconstruction loss constraints; GCL moves "shared" logic to the aggregation layer coupled with routing, using gain alignment to explicitly supervise edge utility.
- **vs CGGM / MCIS (Gradient Balancing-based)**: These modify gradient magnitudes during backpropagation to balance modality contributions; GCL controls information flow in the forward pass using $\alpha$, governed at the source.
- **vs EMOE / Mixture-of-Experts**: Both introduce routing, but MoE does not audit expert outputs; GCL's Auditing Agent is equivalent to installing a "necessity checker" for each expert, achieving better results with fewer parameters.
- **Insight**: Turning "prediction gain of an edge/expert" into a supervised quantity can be directly migrated to edge prediction in GNNs, document selection in retrieval-augmented LLMs, or tool selection for tool-use agents—the GCL teacher-gain framework can be applied to any decision regarding "whether to activate an information path."

## Rating
- Novelty: ⭐⭐⭐⭐ Transforming "marginal gain auditing" into an explicit governance protocol for multimodal fusion is an original perspective.
- Experimental Thoroughness: ⭐⭐⭐ Benchmarks are limited to small-scale sentiment/intent; lacks large-scale or LLM-era competitors.
- Writing Quality: ⭐⭐⭐⭐ Responsibilities, losses, and supervision of the four agents are clearly explained with comprehensive ablations.
- Value: ⭐⭐⭐⭐ The empirical evidence that "Full exchange is worse than unimodal" and "Lightweight governance outperforms MoE" has cross-disciplinary value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Everything is a Video: Unifying Modalities through Next-Frame Prediction](../../ICCV2025/audio_speech/everything_is_a_video_unifying_modalities_through_next-frame_prediction.md)
- [\[ICML 2026\] Two-Dimensional Quantization for Geometry-Aware Audio Coding](two-dimensional_quantization_for_geometry-aware_audio_coding.md)
- [\[ICML 2026\] SafeSearch: Automated Red-Teaming of LLM-Based Search Agents](safesearch_automated_red-teaming_of_llm-based_search_agents.md)
- [\[ICML 2026\] The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning](the_silent_thought_modeling_internal_cognition_in_full-duplex_spoken_dialogue_mo.md)
- [\[ICML 2026\] Algorithmic Recourse of In-Context Learning for Tabular Data](algorithmic_recourse_of_in-context_learning_for_tabular_data.md)

</div>

<!-- RELATED:END -->
