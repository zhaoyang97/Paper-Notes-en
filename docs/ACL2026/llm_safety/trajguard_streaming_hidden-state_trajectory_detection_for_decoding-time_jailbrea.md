---
title: >-
  [Paper Note] TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense
description: >-
  [ACL 2026][LLM Safety][Jailbreak Defense] This paper proposes TrajGuard, a training-free decoding-time jailbreak defense framework. By aggregating hidden-state trajectories of key layers through a sliding window…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Jailbreak Defense"
  - "Hidden-state Trajectory"
  - "Decoding-time Detection"
  - "Real-time Safety"
  - "Training-free Defense"
date: 2026-05-08
content_hash: ac02cf4a8360f229
---

# TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense

**Conference**: ACL 2026  
**arXiv**: [2604.07727](https://arxiv.org/abs/2604.07727)  
**Code**: None  
**Area**: LLM Alignment / AI Safety  
**Keywords**: Jailbreak Defense, Hidden-state Trajectory, Decoding-time Detection, Real-time Safety, Training-free Defense

## TL;DR

This paper proposes TrajGuard, a training-free decoding-time jailbreak defense framework. By aggregating hidden-state trajectories of key layers through a sliding window, it quantifies risk in real-time and triggers a lightweight semantic referee only when the risk consistently exceeds a threshold. It achieves a 95% average defense rate across 12 jailbreak attacks with a detection latency of only 5.2ms/token and a false positive rate below 1.5%.

## Background & Motivation

**Background**: LLMs are deeply integrated into real-world services, making their safety paramount. Despite rigorous safety alignment training (e.g., RLHF), carefully constructed jailbreak attacks can still bypass safety guardrails, achieving high attack success rates on RLHF-aligned models.

**Limitations of Prior Work**: Existing defenses primarily rely on static detection—either filtering prompts at the input stage (e.g., Llama Guard) or checking the complete response at the output stage. Input filtering fails to detect semantically disguised jailbreak prompts, while output filtering, though more effective, requires the full response to be generated before review, introducing non-negligible end-to-end latency. Some methods utilizing internal activations still operate on static prompt representations and rely on high-dimensional geometric scores, resulting in poor interpretability.

**Key Challenge**: Jailbreak risk is not triggered instantaneously at a specific moment but is gradually accumulated through the malicious intent within the context during the decoding process. Existing methods treat safety detection as a discrete binary classification task, ignoring the dynamic evolution of semantics during decoding—a critical blind spot in current defense paradigms.

**Goal**: To implement real-time jailbreak detection by utilizing the dynamic trajectories of hidden states during the decoding process, without relying on additional trained safety models.

**Key Insight**: The authors' empirical analysis revealed a critical "disguise-exposure" pattern: jailbreak prompts are entangled with benign prompts in the latent space (semantic disguise), but once the model begins generating specific harmful steps, the hidden states consistently drift toward malicious regions. This drift appears in the early segments of decoding.

**Core Idea**: Utilize the temporal trajectories of hidden states during decoding as jailbreak detection signals. A coarse-to-fine architecture of "streaming geometric monitoring + on-demand semantic referee" is employed to achieve low-overhead, real-time jailbreak interception.

## Method

### Overall Architecture

TrajGuard adopts a coarse-to-fine hierarchical architecture consisting of two synergetic components: (1) SGS (Streaming Geometric Supervision) continuously monitors hidden-state trajectories as the first line of defense, using lightweight vector calculations to screen for potential risk segments; (2) PAIR-Judge (Prompt-Answer Inference Referee) is triggered only when SGS detects persistent anomalies to provide accurate semantic adjudication. For almost all benign interactions, TrajGuard relies solely on the SGS module running in a low-overhead "monitor-only" mode.

### Key Designs

1.  **Streaming Geometric Supervision (SGS)**:
    -   **Function**: Extracts stable risk signals from noisy hidden-state streams to determine in real-time whether the decoding path deviates from benign behavior.
    -   **Mechanism**: First, Top-K (K=8) key layers are selected using the MVD (Mean Vector Difference) metric; Gaussian distributions of benign/malicious patterns are modeled on the selected layers. During decoding, the difference in Mahalanobis distances from each token's hidden state to benign and malicious centroids is calculated as $r_{l,t} = d^{\mathcal{B}}_{l,t} - d^{\mathcal{M}}_{l,t}$. A three-stage aggregation is applied: intra-layer sliding window (w=8) truncated mean → cross-layer averaging → EWMA temporal smoothing to obtain a stable streaming risk score $p_t$. An alert is triggered only when the risk score exceeds the threshold $\gamma$ for $k=3$ consecutive steps.
    -   **Design Motivation**: Risk judgment for a single token is noisy; a true jailbreak manifests as persistent residence in a high-risk region. The lagged trigger mechanism effectively suppresses transient geometric noise, ensuring only sustained malicious intent triggers the expensive referee process.

2.  **Prompt-Answer Inference Referee (PAIR-Judge)**:
    -   **Function**: Performs semantic-level safety adjudication on anomalies flagged by SGS, converting high-dimensional internal signals into interpretable safety decisions.
    -   **Mechanism**: When SGS triggers an alert, generation is paused. The current context (prompt $x$ + generated prefix $y_{\leq t}$) is wrapped into a safety system prompt and sent to a safety-aligned LLM for a binary SAFE/UNSAFE judgment $d = \mathcal{M}_{judge}(\mathcal{P}(x, y_{\leq t}))$. If judged UNSAFE, generation is terminated immediately.
    -   **Design Motivation**: Geometric proximity to a malicious region does not equate to semantic malice. Semantic-level verification is required to avoid misjudgment while maintaining interpretability.

3.  **State Reset**:
    -   **Function**: Clears the "false positive" risk momentum accumulated by SGS when PAIR-Judge rules SAFE.
    -   **Mechanism**: If the semantic referee deems the current content safe, the SGS risk score $S_t$ is forcibly reset to the initial safety value, preventing the system from repeatedly triggering alerts in subsequent decoding due to historical geometric bias.
    -   **Design Motivation**: Without state reset, a single false trigger could lead to a chain of subsequent false positives, severely impacting normal usage.

### Loss & Training

TrajGuard is a completely training-free framework. It only requires a preprocessing step: using 8,000 benign instructions and 10,000 malicious instructions to estimate the distribution (centroids and covariance matrices) of safe/unsafe regions in the hidden space. Shrinkage regularization $\widehat{\Sigma}_{\star,l} = \Sigma_{\star,l} + \lambda I$ is used to enhance numerical stability in high-dimensional space.

## Key Experimental Results

### Main Results

| Model | Defense Method | Avg. ASR (12 Attacks) ↓ | Best Single Attack ASR |
| :--- | :--- | :--- | :--- |
| Llama-2-7B | No Defense | 0.52 | - |
| Llama-2-7B | Llama Guard 3 | 0.20 | GCG: 0.02 |
| Llama-2-7B | Qwen3Guard | 0.07 | GCG: 0.00 |
| Llama-2-7B | **TrajGuard** | **0.02** | Most attacks: 0.00 |
| Llama-3.1-8B | No Defense | 0.57 | - |
| Llama-3.1-8B | **TrajGuard** | **0.04** | - |
| Mistral-7B | No Defense | 0.75 | - |
| Mistral-7B | **TrajGuard** | **0.05** | - |

| Metric | TrajGuard Performance |
| :--- | :--- |
| Avg. Defense Rate | 95% |
| Detection Latency | 5.2 ms/token |
| False Positive Rate (XSTest) | < 1.5% |
| Alpaca Task Retention | High (see paper) |

### Ablation Study

| Configuration | Key Impact | Description |
| :--- | :--- | :--- |
| Full TrajGuard | AVG ASR ≈ 0.02-0.05 | Full model |
| w/o PAIR-Judge | Incr. False Positives | Geometric monitoring alone misjudges safe but sensitive content |
| w/o State Reset | Recursive False Positives | Persistent alerts after an initial false trigger |
| w/o Persistence Trigger | Incr. Noise | Single-step judgments are easily affected by transient fluctuations |
| Diff. Window Size $w$ | $w=8$ is optimal | Small $w$ is noisy, large $w$ increases latency |

### Key Findings

- **Hidden-state trajectories provide stronger and more stable jailbreak signals than input prompts**: Jailbreak prompts are entangled with benign ones in the latent space (overlapping at $t=0$), but hidden states consistently drift toward malicious regions once decoding begins.
- **"Drift latency" varies significantly between models**: Llama-2-7B only begins to deteriorate after 37 steps, while Vicuna-7B drops almost immediately, reflecting differences in the robustness of safety alignment across models.
- **TrajGuard reduces ASR to near 0 on most attacks**, performing particularly well against mainstream attacks like GCG, AutoDAN, and PAIR.
- Cipher-based attacks are the only type that still maintain some success rate (ASR 0.10-0.25), likely because the representation patterns of encrypted inputs in the hidden space differ from conventional jailbreaks.

## Highlights & Insights

- **The "Disguise-Exposure" observation is ingenious**: The semantic disguise of jailbreak prompts is effective at the input stage, but once the model starts generating specific harmful steps, internal representations inevitably drift toward malicious regions. This observation transforms jailbreak detection from a static classification problem into a dynamic trajectory monitoring problem.
- **The coarse-to-fine hierarchical design is highly practical**: For the vast majority of the time, only lightweight geometric monitoring (5.2ms/token) is running. The expensive semantic referee is only called upon suspected risk, achieving an excellent balance between precision and efficiency.
- **The training-free nature** allows it to be plugged into any open-source LLM without additional safety data or fine-tuning costs.
- **The closed-loop state reset mechanism** can be transferred to other anomaly detection systems to solve the general problem of "one false positive causing a chain reaction."

## Limitations & Future Work

- Requires pre-constructing distribution estimates of benign/malicious regions, which depends on the quality and coverage of the 8K+10K labeled data.
- Defense effectiveness against Cipher-type encrypted attacks is relatively weak, as hidden states may not sufficiently expose the malicious intent of encrypted inputs.
- Validated only on 7B-8B scale open-source models; applicability to larger-scale or closed-source models is unknown.
- PAIR-Judge uses the target model itself as the referee; referee quality may decline when the model's own safety alignment is weak.

## Related Work & Insights

- **vs Llama Guard 3**: Static input/output filters that cannot utilize dynamic information during decoding. TrajGuard substantially outperforms it on almost all attacks.
- **vs SafeDecoding (Xu et al., 2024)**: Requires training a safety expert model to re-weight decoding probabilities; TrajGuard is training-free and directly utilizes the base model's hidden states.
- **vs ShieldHead (Xuan et al., 2025)**: Attaching token-level safety heads requires additional training and remains a per-token static judgment, failing to model temporal trajectories.
- **vs Goal Prioritization (Zhang et al., 2024)**: Performs poorly on some models (AVG ASR 0.44 on Mistral-7B), suggesting that prompt engineering methods lack robustness against attacks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to use decoding-time hidden-state trajectories for jailbreak detection; the "disguise-exposure" observation is novel and convincing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 attacks, 4 models, multiple baselines, and complete ablations—very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, natural derivation of motivation, and rich visualizations.
- Value: ⭐⭐⭐⭐⭐ A real-time defense solution that is training-free, has low latency, and provides a high defense rate; extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)
- [\[ICLR 2026\] Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates](../../ICLR2026/llm_safety/inference-time_backdoors_via_hidden_instructions_in_llm_chat_templates.md)
- [\[ACL 2026\] LeakDojo: Decoding the Leakage Threats of RAG Systems](leakdojo_decoding_the_leakage_threats_of_rag_systems.md)
- [\[ACL 2026\] GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models](gambit_a_gamified_jailbreak_framework_for_multimodal_large_language_models.md)
- [\[ACL 2026\] TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards](trojail_trajectory-level_optimization_for_multi-turn_large_language_model_jailbr.md)

</div>

<!-- RELATED:END -->
