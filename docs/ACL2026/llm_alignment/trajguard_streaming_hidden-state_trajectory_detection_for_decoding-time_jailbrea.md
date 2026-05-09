---
title: >-
  [Paper Note] TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense
description: >-
  [ACL 2026][LLM Alignment][Jailbreak Defense] This paper proposes TrajGuard, a training-free decoding-time jailbreak defense framework that quantifies risk in real time by aggregating hidden-state trajectories from critical layers via a sliding window, and invokes a lightweight semantic judge only when risk persistently exceeds a threshold. TrajGuard achieves an average defense rate of 95% across 12 jailbreak attacks, with a detection latency of only 5.2 ms/token and a false positive rate below 1.5%.
tags:
  - ACL 2026
  - LLM Alignment
  - Jailbreak Defense
  - Hidden-state Trajectory
  - Decoding-time Detection
  - Real-time Safety
  - Training-free Defense
date: 2026-05-08
content_hash: 4a88848f34a423be
---

# TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense

**Conference**: ACL 2026
**arXiv**: [2604.07727](https://arxiv.org/abs/2604.07727)
**Code**: None
**Area**: LLM Alignment / AI Safety
**Keywords**: Jailbreak Defense, Hidden-state Trajectory, Decoding-time Detection, Real-time Safety, Training-free Defense

## TL;DR

This paper proposes TrajGuard, a training-free decoding-time jailbreak defense framework that quantifies risk in real time by aggregating hidden-state trajectories from critical layers via a sliding window, and invokes a lightweight semantic judge only when risk persistently exceeds a threshold. TrajGuard achieves an average defense rate of 95% across 12 jailbreak attacks, with a detection latency of only 5.2 ms/token and a false positive rate below 1.5%.

## Background & Motivation

**State of the Field**: LLMs have been deeply integrated into real-world services, making their safety critical. Despite rigorous safety alignment training (e.g., RLHF), carefully crafted jailbreak attacks can still bypass safety guardrails and achieve high attack success rates on RLHF-aligned models.

**Limitations of Prior Work**: Existing defenses primarily rely on static detection—either filtering prompts at the input side (e.g., Llama Guard) or inspecting complete responses at the output side. Input-side filtering cannot detect semantically disguised jailbreak prompts, while output-side filtering, though more effective, requires generating full responses before inspection, introducing non-negligible end-to-end latency. Methods that leverage internal model activations still operate on static prompt representations and rely on high-dimensional geometric scores with poor interpretability.

**Root Cause**: Jailbreak risk is not triggered instantaneously at a single moment; rather, it accumulates progressively through malicious intent in context during the decoding process. Existing methods treat safety detection as a discrete binary classification task, ignoring the dynamic semantic evolution during decoding—a critical blind spot in current defense paradigms.

**Paper Goals**: To leverage dynamic hidden-state trajectories during decoding for real-time jailbreak detection without relying on additionally trained safety models.

**Starting Point**: Through empirical analysis, the authors identify a key "disguise-then-expose" pattern: jailbreak prompts are entangled with benign prompts in the latent space (semantic camouflage), but once the model begins generating specific harmful steps, the hidden states persistently drift toward the malicious region. This drift emerges in early decoding segments.

**Core Idea**: Treat the temporal trajectory of hidden states during decoding as a jailbreak detection signal, and implement low-overhead, real-time jailbreak interception via a coarse-to-fine architecture combining "streaming geometric monitoring + on-demand semantic judging."

## Method

### Overall Architecture

TrajGuard adopts a coarse-to-fine hierarchical architecture with two collaborative components: (1) SGS (Streaming Geometric Surveillance), which continuously monitors hidden-state trajectories as the first line of defense and uses lightweight vector computations to screen potentially risky segments; and (2) PAIR-Judge (Prompt-Answer Inferential Reasoner), which is triggered only when SGS detects persistent anomalies to deliver accurate semantic verdicts. For nearly all benign interactions, TrajGuard operates in a low-overhead "monitor-only" mode relying solely on the SGS module.

### Key Designs

1. **Streaming Geometric Surveillance (SGS)**:

    - **Function**: Extracts stable risk signals from noisy hidden-state streams and determines in real time whether the decoding trajectory deviates from benign behavior.
    - **Mechanism**: Critical layers are first selected using the MVD (Mean Vector Discrepancy) metric to retain the Top-K (K=8) layers. Gaussian distributions of benign/malicious patterns are then modeled on the selected layers. During decoding, the Mahalanobis distance difference $r_{l,t} = d^{\mathcal{B}}_{l,t} - d^{\mathcal{M}}_{l,t}$ between each token's hidden state and the benign/malicious centroids is computed. A stable streaming risk score $p_t$ is obtained through three-stage aggregation: intra-layer sliding window (w=8) truncated mean → cross-layer averaging → EWMA temporal smoothing. An alert is triggered only when the risk score exceeds threshold γ for k=3 consecutive steps.
    - **Design Motivation**: Single-token risk judgments are noisy; genuine jailbreaks manifest as sustained residence in high-risk regions. The hysteresis trigger mechanism effectively suppresses transient geometric noise, ensuring that only persistent malicious intent invokes the costly judging process.

2. **Prompt-Answer Inferential Reasoner (PAIR-Judge)**:

    - **Function**: Provides semantic-level safety verdicts for anomalies flagged by SGS, translating high-dimensional internal signals into interpretable safety decisions.
    - **Mechanism**: When SGS triggers an alert, generation is paused, and the current context (prompt $x$ + generated prefix $y_{\leq t}$) is wrapped in a safety system prompt and sent to a safety-aligned LLM for a binary SAFE/UNSAFE verdict $d = \mathcal{M}_{judge}(\mathcal{P}(x, y_{\leq t}))$. If UNSAFE is determined, generation is immediately terminated.
    - **Design Motivation**: Geometric proximity to the malicious region does not imply semantic malice. Semantic-level verification is necessary to avoid false positives while maintaining interpretability.

3. **Closed-loop State Reset**:

    - **Function**: Clears the "false-positive" risk momentum accumulated by SGS when PAIR-Judge returns a SAFE verdict.
    - **Mechanism**: If the semantic judge deems the current content safe, the SGS risk score $S_t$ is forcibly reset to the initial safe value, preventing the system from repeatedly triggering alerts in subsequent decoding due to historical geometric bias.
    - **Design Motivation**: Without state reset, a single false trigger could cascade into consecutive false positives, severely degrading normal usability.

### Loss & Training

TrajGuard is a fully training-free framework. Only a preprocessing step is required: 8,000 benign instructions and 10,000 malicious instructions are used to estimate the distribution of safe/unsafe regions in the hidden space (centroids and covariance matrices), with shrinkage regularization $\widehat{\Sigma}_{\star,l} = \Sigma_{\star,l} + \lambda I$ applied to enhance numerical stability in high-dimensional spaces.

## Key Experimental Results

### Main Results

| Model | Defense Method | Avg. ASR↓ (12 Attacks) | Best Single-Attack ASR |
|-------|---------------|------------------------|------------------------|
| Llama-2-7B | No Defense | 0.52 | — |
| Llama-2-7B | Llama Guard 3 | 0.20 | GCG: 0.02 |
| Llama-2-7B | Qwen3Guard | 0.07 | GCG: 0.00 |
| Llama-2-7B | **TrajGuard** | **0.02** | Most attacks: 0.00 |
| Llama-3.1-8B | No Defense | 0.57 | — |
| Llama-3.1-8B | **TrajGuard** | **0.04** | — |
| Mistral-7B | No Defense | 0.75 | — |
| Mistral-7B | **TrajGuard** | **0.05** | — |

| Metric | TrajGuard Performance |
|--------|-----------------------|
| Average Defense Rate | 95% |
| Detection Latency | 5.2 ms/token |
| False Positive Rate (XSTest) | < 1.5% |
| Alpaca Normal Task Retention | High (see paper for details) |

### Ablation Study

| Configuration | Key Impact | Notes |
|---------------|------------|-------|
| Full TrajGuard | AVG ASR ≈ 0.02–0.05 | Complete model |
| w/o PAIR-Judge | Increased false positive rate | Geometric monitoring alone misclassifies safe but sensitive content |
| w/o State Reset | Cascading false positives | Subsequent decoding continues to alert after a false trigger |
| w/o Persistence Trigger | Increased noise | Single-step judgments are susceptible to transient fluctuations |
| Varying window size w | w=8 is optimal | Too small → high noise; too large → high latency |

### Key Findings

- **Hidden-state trajectories provide stronger and more stable jailbreak signals than input prompts**: Jailbreak prompts are entangled with benign prompts in the latent space at t=0, but once decoding begins, hidden states persistently drift toward the malicious region.
- **"Drift latency" varies significantly across models**: Llama-2-7B does not begin to deteriorate until step 37, whereas Vicuna-7B degrades almost immediately, reflecting differences in the robustness of safety alignment across models.
- **TrajGuard reduces ASR to near zero on most attacks**, with particularly strong performance against GCG, AutoDAN, and PAIR.
- Cipher-type attacks remain the only attack category with a non-trivial success rate (ASR 0.10–0.25), possibly because encrypted inputs produce representation patterns in the hidden space that differ from conventional jailbreaks.

## Highlights & Insights

- **The "disguise-then-expose" observation is highly insightful**: The semantic camouflage of jailbreak prompts is effective at the input stage, but once the model begins generating specific harmful steps, its internal representations inevitably drift toward the malicious region. This observation reframes jailbreak detection from a static classification problem into a dynamic trajectory monitoring problem.
- **The coarse-to-fine hierarchical design is highly practical**: Lightweight geometric monitoring (5.2 ms/token) runs the vast majority of the time, with the expensive semantic judge invoked only for suspected risks, achieving an excellent balance between accuracy and efficiency.
- **Being fully training-free** allows TrajGuard to be plug-and-play with any open-source LLM, requiring no additional safety data or fine-tuning costs.
- **The closed-loop state reset mechanism** is transferable to other anomaly detection systems to address the general problem of "one false positive triggering a cascade of subsequent alerts."

## Limitations & Future Work

- Requires pre-building distribution estimates for benign/malicious regions, depending on the quality and coverage of 8K+10K annotated data.
- Defense against Cipher-type encrypted attacks is relatively weaker, as hidden states may not sufficiently expose the malicious intent of encrypted inputs.
- Validation is limited to open-source models at the 7B–8B scale; applicability to larger-scale or closed-source models remains unknown.
- PAIR-Judge uses the target model itself as the judge, which may degrade judging quality when the model's safety alignment is weak.

## Related Work & Insights

- **vs. Llama Guard 3**: A static input/output filter that cannot exploit dynamic information during decoding. TrajGuard substantially outperforms it on nearly all attacks.
- **vs. SafeDecoding (Xu et al., 2024)**: Requires training a safety expert model to reweight decoding probabilities. TrajGuard requires no training and directly leverages the hidden states of the base model.
- **vs. ShieldHead (Xuan et al., 2025)**: Appending a token-level safety head requires additional training and still performs static per-token judgments without modeling temporal trajectories.
- **vs. Goal Prioritization (Zhang et al., 2024)**: Performs poorly on some models (AVG ASR of 0.44 on Mistral-7B), indicating insufficient robustness of prompt engineering approaches against attacks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First work to apply decoding-time hidden-state trajectories to jailbreak detection; the "disguise-then-expose" observation is novel and compelling.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 attacks, 4 models, multiple baselines, and complete ablations—highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, natural motivation derivation, and rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ A training-free, low-latency, high-defense-rate real-time defense solution with extremely high practical value.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](../../ICLR2026/llm_alignment/reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)
- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)
- [\[CVPR 2026\] Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](../../CVPR2026/llm_alignment/principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)
- [\[NeurIPS 2025\] SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism](../../NeurIPS2025/llm_alignment/safeptr_token-level_jailbreak_defense_in_multimodal_llms_via_prune-then-restore_.md)
- [\[ACL 2026\] Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](aligning_agents_via_planning_a_benchmark_for_trajectory-level_reward_modeling.md)

<!-- RELATED:END -->
