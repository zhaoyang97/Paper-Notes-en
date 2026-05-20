---
title: >-
  [Paper Note] Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration
description: >-
  [ICML 2026][LLM Agent][Multimodal Fusion] To address the persistent issues of "modality dominance" and "spurious modality coupling" in centralized multimodal fusion…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Multimodal Fusion"
  - "Modality Dominance"
  - "Spurious Coupling"
  - "Marginal Gain Gating"
  - "Routing Auditing"
date: 2026-05-08
content_hash: f1d54b9a19d9ea80
---

# Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration

**Conference**: ICML 2026  
**arXiv**: [2605.00370](https://arxiv.org/abs/2605.00370)  
**Code**: None  
**Area**: Multimodal Sentiment Analysis / Intent Recognition  
**Keywords**: Multimodal Fusion, Modality Dominance, Spurious Coupling, Marginal Gain Gating, Routing Auditing

## TL;DR
To address the persistent issues of "modality dominance" and "spurious modality coupling" in centralized multimodal fusion, GCL reframes multimodal learning as a **protocolized collaboration among four agents in two stages**: In the first stage, Routing/Auditing agents determine, on a per-sample basis, which cross-modal communications are permitted based on marginal predictive gain; in the second stage, Public-Factor/Aggregation agents decouple shared semantics from private specialization before aggregation. This approach achieves SOTA on MOSI/MOSEI/MIntRec.

## Background & Motivation
**Background**: The mainstream in multimodal learning (language + acoustic + visual) is **centralized fusion**—either tensor products as in TFN/LMF, cross-modal attention as in MulT, or end-to-end fine-tuning based on BERT. Representative advances include structured representations (MISA, FDMER, ConFede with shared/private decomposition) and optimization interventions (CGGM, MCIS for gradient rebalancing).

**Limitations of Prior Work**: All these methods implicitly assume that as long as the task loss is backpropagated end-to-end, the optimal interaction pattern will naturally emerge. In reality: (1) Gradients tend to concentrate along the "easiest loss-reducing modality path," leading to undertraining of weaker modalities and high sensitivity to noise; (2) End-to-end fusion rewards any cross-modal correlation, including accidental ones—this tightly binds modalities that should not be coupled, causing severe out-of-distribution performance drops.

**Key Challenge**: Interaction learning and representation learning are entangled within the same loss, with no **explicit signal** to distinguish "useful" from "redundant" interactions. Routing/MoE-type works introduce adaptivity but still lack an auditing layer to question "how much marginal gain does this edge actually bring."

**Goal**: (1) Make the topology of cross-modal information flow observable and supervisable; (2) Use an explicit redundancy control term to eliminate spurious coupling; (3) Ensure the aggregation stage does not let a strong modality "dominate," but instead weights based on sample-wise shared semantics.

**Key Insight**: The authors borrow the "division of labor + governance" metaphor from LLM-agent systems for multimodal fusion—rather than letting a black-box network freely couple three modalities, the interaction process is decomposed into four explicit roles: "propose routing → audit admission → extract public factor → weighted aggregation," each agent supervised by its own auxiliary loss.

**Core Idea**: Use "marginal predictive gain" as a hard criterion for interaction admission, combined with a redundancy contrastive penalty, transforming end-to-end fusion into a two-stage, auditable protocol.

## Method

### Overall Architecture
Input: For each modality $m\in\{l,a,v\}$, sample $x_m$ is encoded to $h_m$.  
Output: Classification/regression prediction $\hat o$.  
The four agents operate in two serial stages:

1. **Stage 1 — Selective Interaction**: The Routing Agent assigns a logit $\rho_{m\to n}$ and packages a message $u_{m\to n}$ for each directed edge $(m\to n)$. The Auditing Agent estimates the marginal gain $\hat\Delta_{m\to n}$ of fusing this message into $h_n$, multiplies by the routing probability to obtain the gate $\alpha_{m\to n}$. Gated residual updates yield the refined $z_n$.
2. **Stage 2 — Consensus Formation**: The Public-Factor Agent uses a permutation-invariant operator to distill a shared factor $c$ from $\{z_l,z_a,z_v\}$. The Aggregation Agent uses $c$ as context to generate proposals $r_m$ and softmax weights $\pi_m$ for each modality, with final prediction $\hat o = g_\tau(\sum_m \pi_m r_m, c)$.

Four auxiliary objectives (task / local / public / gain alignment / redundancy) jointly train all agents.

### Key Designs

1. **Auditing Agent + Marginal Gain Auditing Gate**:

    - **Function**: Transforms "whether $m$ is allowed to send a message to $n$" from a pure optimization decision into a supervisable audit—only edges that truly reduce task loss are opened.
    - **Mechanism**: During training, define "teacher gain" $\Delta_{m\to n}=\ell_\tau(q_n^\tau(h_n),y)-\ell_\tau(q_n^\tau(\tilde h_n^m),y)$, where $\tilde h_n^m=h_n+\phi_{m\to n}(h_n,u_{m\to n})$ is the post-fusion transient. At inference, since labels are unavailable, a gain predictor $\hat\Delta_{m\to n}=g_g^{m\to n}(h_n,u_{m\to n})$ is used. The final gate is $\alpha_{m\to n}=\text{softmax}_{j}(\rho_{j\to n})_{j=m}\cdot\sigma_\kappa(\tilde\Delta_{m\to n})$, multiplying "willingness to communicate" (routing) and "worth communicating" (gain). Gated integration uses residuals: $z_n=h_n+\sum_{m\neq n}\alpha_{m\to n}\cdot\phi_{m\to n}(h_n,u_{m\to n})$. The gain alignment loss $\mathcal L_{\text{gain}}=-\sum_n\sum_{m\neq n}\alpha_{m\to n}\text{stopgrad}(\Delta_{m\to n})$ encourages positive-gain edges to open and negative-gain edges to close, with stopgrad preventing teacher gain from being contaminated by backprop.
    - **Design Motivation**: Previous router/MoE approaches only ask "which path to activate," not "is activating this path useful." Explicitly separating teacher–student gain solves two issues: ground-truth supervision during training, and reliance only on the learned predictor at inference, with no extra inference cost.

2. **Public-Factor Agent + Decoupled Aggregation**:

    - **Function**: Explicitly separates "cross-modal shared semantics" and "modality-specific specialization," avoiding traditional fusion operators that let strong modalities overwhelm weaker ones.
    - **Mechanism**: Uses a permutation-invariant operator (symmetric attention or global pool + MLP) $c=g_p(z_l,z_a,z_v)$ to extract the public factor, with auxiliary supervision $\mathcal L_{\text{pub}}=\mathbb E\ell_\tau(g_\tau^c(c),y)$ ensuring $c$ is itself predictive. The Aggregation Agent then uses $c$ as context: each modality generates a proposal $r_m=\eta_m(z_m,c)$ and unnormalized score $s_m=g_a^m(z_m,c)$, computes $\pi_m=\text{softmax}(\{s_m\}_m)$ for sample-wise weights, and outputs $\hat o = g_\tau(\sum_m \pi_m r_m, c)$.
    - **Design Motivation**: Direct concatenation entangles $c$ and $z_m$, causing strong modalities to dominate and $z_a/z_v$ to be marginalized. By explicitly extracting $c$ and using it to modulate each modality's weight $\pi_m$, the model effectively asks, "Given the shared semantics, how much additional information does this modality provide?"—this conditioning prevents incremental info from weak modalities from being drowned out.

3. **Redundancy Contrastive Regularization $\mathcal L_{\text{red}}$**:

    - **Function**: After Stage 1, applies orthogonality pressure to the refined channels $\{z_n\}$, preventing routing from learning edges that cause all modalities to converge to the same representation.
    - **Mechanism**: Uses a symmetric InfoNCE-style alignment score $\mathcal L_{\text{red}}=\sum_{m<n}D(z_m,z_n)$, minimizing which enforces low mutual information between refined representations.
    - **Design Motivation**: Gating $\alpha$ alone controls information **flow**, but not whether modality uniqueness is preserved after fusion. $\mathcal L_{\text{red}}$ acts as a counterforce, specifically targeting "spurious coupling"—ablation shows that removing this term increases MOSI MAE from $0.685$ to $0.703$, with HSIC/CKA diagnostics worsening sharply.

### Loss & Training
$\mathcal L_{\text{total}} = \mathcal L_{\text{task}} + \lambda_{\text{loc}}\mathcal L_{\text{loc}} + \lambda_{\text{pub}}\mathcal L_{\text{pub}} + \lambda_{\text{gain}}\mathcal L_{\text{gain}} + \lambda_{\text{red}}\mathcal L_{\text{red}}$.
Here, $\mathcal L_{\text{loc}}=\sum_m\mathbb E\ell_\tau(q_m^\tau(h_m),y)$ supervises unimodal heads to ensure reliable teacher gain estimation. Adam optimizer, batch size 128, weight decay $1\text{e-}4$, patience 6, A100 GPU.

## Key Experimental Results

### Main Results

| Dataset | Metric | GCL | Prev. SOTA (TSDA) | Gain |
|---------|--------|------|-------------------|------|
| CMU-MOSI | MAE↓ | **0.685** | 0.695 | $-0.010$ |
| CMU-MOSI | Acc-2 | **86.79** | 86.3 | $+0.49$ |
| CMU-MOSI | Acc-7 | **49.06** | 48.6 | $+0.46$ |
| CMU-MOSEI | MAE↓ | **0.520** | 0.529 | $-0.009$ |
| CMU-MOSEI | Acc-2 | **86.78** | 86.3 | $+0.48$ |
| MIntRec | Acc | **72.74** | 72.59 | $+0.15$ |
| MIntRec | F1 | **70.95** | 70.68 | $+0.27$ |

### Ablation Study

| Configuration | MOSI MAE | MOSI Acc-7 | Notes |
|---------------|----------|------------|-------|
| Full GCL | 0.685 | 49.06 | Complete model |
| w/o Routing Agent | 0.694 | 48.55 | No topology proposal |
| w/o Auditing Agent | 0.699 | 48.00 | No gain gating |
| **Full exchange** (no audit, all open) | 0.721 | 46.10 | Worse than "language only" (0.714) |
| w/o Public-Factor Agent | 0.702 | 47.85 | Shared semantics collapse |
| Uniform $\pi_m$ | 0.698 | 48.05 | No sample-adaptive weights |
| w/o $\mathcal L_{\text{red}}$ | 0.703 | 47.70 | HSIC/CKA worsen sharply |
| **only $\mathcal L_{\text{task}}$** | 0.712 | 46.70 | All governance terms removed → degraded |

### Key Findings
- **Unregulated full fusion (0.721) performs worse than language-only (0.714)**—the most striking empirical result: blind "multimodal fusion" can hurt performance, as SNR is dragged down by noise. This directly refutes the implicit belief that "adding more modalities can't hurt."
- **Efficiency surprise**: GCL has 117.56M parameters, 20.06s per epoch, halving both compared to ConFede (256.98M, 40.12s), and 25% less training time than EMOE (143.5M, 26.80s). "Division of labor + governance" is actually lighter than stacking MoE/experts.
- **Noise robustness**: Injecting Gaussian noise with $\sigma\in[0,20]$ on MOSI, GCL's performance degrades more gracefully than baselines, indicating that auditing gates can actively shut off noisy signals as SNR drops.
- **Audited Selectivity experiment** (Fig 3): GCL occupies the high-PGR, moderate-AR quadrant (high positive gain ratio, moderate activation rate), while NoAudit/Uniform/Full-exchange variants cluster in high-AR, low-PGR—i.e., they communicate a lot but ineffectively.

## Highlights & Insights
- **Marginal gain as a loss**: Introduces the $do(\cdot)$ concept from causal inference into multimodal fusion, making "is this interaction worthwhile" a learnable quantity, rather than relying on end-to-end gradient emergence. This can theoretically extend to edge activation in any graph-structured network.
- **Teacher gain + stopgrad for training / Student gain for inference**: A classic "oracle supervision at training, learner at inference" setup, adding a supervised signal without sacrificing inference speed. This trick is reusable in distillation, actor-critic, and verifier-guided generation.
- **Full exchange worse than unimodal**: This experimental result is itself worth repeated citation, as it empirically refutes the multimodal community's default assumption that "more modalities is always better."
- **Public factor as routing context**: Explicitly using "shared semantics" as routing context is equivalent to first answering "what do we already know in consensus" before asking "what can private information add," avoiding double-counting shared information.

## Limitations & Future Work
- The paper is categorized as audio_speech, but is actually about multimodal sentiment analysis + intent recognition—"acoustic" is just one of three modalities, with no evaluation on pure speech/ASR tasks.
- Teacher gain estimation relies on single-step residual $\tilde h_n^m=h_n+\phi(\cdot)$ approximation; multi-step, deep interaction marginal effects are not considered. Scaling to long chains or $N>3$ modalities faces combinatorial explosion.
- Main benchmarks (MOSI/MOSEI/MIntRec) are relatively small (thousands to tens of thousands of samples), with no validation on large-scale vision-language datasets (e.g., LAION subsets, AudioSet).
- Lacks comparison with LLM-based multimodal methods (e.g., VideoLLaMA, Qwen-Audio), so it is unclear whether GCL's lightweight protocol retains advantages in the LLM era.

## Related Work & Insights
- **vs MISA / FDMER (disentanglement series)**: These also separate shared/private, but disentanglement is at the representation level, constrained by reconstruction loss; GCL elevates sharing to the aggregation layer and couples it with routing, with gain alignment explicitly supervising edge utility.
- **vs CGGM / MCIS (gradient balancing series)**: These rebalance modality contributions by adjusting gradient magnitudes during backprop; GCL uses $\alpha$ to control information flow at the source, theoretically cleaner.
- **vs EMOE / Mixture-of-Experts**: Both introduce routing, but MoE does not audit expert outputs; GCL's Auditing Agent is equivalent to equipping each expert with a "necessity checker," achieving better results with fewer parameters.
- **Insights**: Making "edge/expert predictive gain" a supervisable quantity can be directly transferred to GNN edge prediction, retrieval-augmented LLM document selection, and tool-use agent tool selection—any decision of "whether to activate an information path" can adopt GCL's teacher-gain framework.

## Rating
- Novelty: ⭐⭐⭐⭐ Explicitly making "marginal gain auditing" a governance protocol for multimodal fusion is an original perspective
- Experimental Thoroughness: ⭐⭐⭐ Benchmarks are all small-scale sentiment/intent tasks; lacks large-scale / LLM-era competitors
- Writing Quality: ⭐⭐⭐⭐ Responsibilities, losses, and supervision of the four agents are clearly explained; comprehensive ablation
- Value: ⭐⭐⭐⭐ "Full exchange worse than unimodal" + "lightweight surpasses MoE" both have cross-subfield empirical value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Everything is a Video: Unifying Modalities through Next-Frame Prediction](../../ICCV2025/audio_speech/everything_is_a_video_unifying_modalities_through_next-frame_prediction.md)
- [\[ICLR 2026\] RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments](../../ICLR2026/audio_speech/redteamcua_adversarial_testing_agents.md)
- [\[ICCV 2025\] Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition](../../ICCV2025/audio_speech/lyra_an_efficient_and_speechcentric_framework_for_omnicognit.md)
- [\[ACL 2025\] AI4Reading: Chinese Audiobook Interpretation System Based on Multi-Agent Collaboration](../../ACL2025/audio_speech/ai4reading_chinese_audiobook_interpretation_system_based_on_multi-agent_collabor.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](../../ACL2026/audio_speech/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)

</div>

<!-- RELATED:END -->
