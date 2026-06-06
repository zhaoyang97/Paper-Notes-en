---
title: >-
  [Paper Note] AgentMark: Utility-Preserving Behavioral Watermarking for Agents
description: >-
  [ACL 2026][LLM Safety][agent watermarking] AgentMark models the "tool/subgoal selection" of an LLM agent as a time-varying discrete channel. By explicitly eliciting the behavioral distribution $P_t$ and applying FDPSS-st…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "agent watermarking"
  - "planning behavior"
  - "distribution-preserving sampling"
  - "erasure-resilient coding"
  - "provenance"
date: 2026-05-08
content_hash: d47a44fedfd6ceda
---

# AgentMark: Utility-Preserving Behavioral Watermarking for Agents

**Conference**: ACL 2026  
**arXiv**: [2601.03294](https://arxiv.org/abs/2601.03294)  
**Code**: https://github.com/Tooooa/AgentMark (Available)  
**Area**: LLM Safety / Watermarking / Agent Governance  
**Keywords**: agent watermarking, planning behavior, distribution-preserving sampling, erasure-resilient coding, provenance

## TL;DR
AgentMark models the "tool/subgoal selection" of an LLM agent as a time-varying discrete channel. By explicitly eliciting the behavioral distribution $P_t$ and applying FDPSS-style distribution-preserving sampling, it embeds multi-bit IDs into planning decisions. Combined with RLNC encoding, the watermark can be recovered from residual logs even if the trace is cropped or steps are deleted. Experiments across ALFWorld, ToolBench, and OASIS tasks show zero accuracy degradation (Success Rate difference with baseline <0.7 pp), stable multi-bit capacity of 1.2-2.3 bps, and orthogonality for stacking with content-level watermarks like SynthID-Text.

## Background & Motivation

**Background**: LLM content watermarking (e.g., KGW, SynthID-Text) can reliably attribute model-generated text; Google Gemini has already deployed SynthID. However, what truly causes social impact in agents is the "behavioral decision sequence"—which tool is selected, which subgoal is pursued—rather than the final text. This applies to GUI assistants, financial tool calls, and social bots.

**Limitations of Prior Work**: Directly applying content watermarking to agent behavior leads to three failure modes: (1) Training-based watermarks require modifying model weights, whereas most agents use closed-source APIs; (2) Token-level watermarks (KGW, SynthID) act on token distributions, but "behavior" is not a token—e.g., "Alice bookmarked a post with the tag #TravelInspiration" is compiled into tool calls `bookmark()` + `tag(#TravelInspiration)`, stripping the watermark signal during compilation; (3) Directly biasing behavioral probabilities (e.g., the RG strategy in Agent Guide) causes distribution drift, which compounds over long-horizon execution and leads to task failure.

**Key Challenge**: To address governance risks like impersonation, IP theft, or loss of control, watermarks must be embedded at the planning layer. However, perturbations at the planning layer often destroy utility—this is the root contradiction.

**Goal**: Achieve "distribution-preserving behavioral watermarking" that simultaneously satisfies: (a) no modification of model weights; (b) usability under black-box APIs; (c) zero drift in behavioral distribution after embedding; (d) multi-bit ID recovery even if the trace is partially erased/truncated; (e) orthogonal stacking with content watermarks.

**Key Insight**: Planning is viewed as a sampling process from an implicit distribution $P_t^\star$. The agent is asked to explicitly output $P_t \approx P_t^\star$, and then the FDPSS framework (difference recomposition + cyclic shift uniform encoding) is used for distribution-preserving sampling—embedding bits during sampling while keeping the marginal distribution unchanged.

**Core Idea**: "First elicit the implicit policy into an explicit probability list $P_t$; the watermarking action occurs only during the sampling process of $P_t$, without modifying $P_t$ itself."

## Method

### Overall Architecture
The AgentMark-F (specific instantiation) process per step: (1) **Behavior Elicitation**: The agent no longer selects $b_t$ as a black box but explicitly outputs a probability list $P_t$ over the candidate set $\mathcal{B}_t$; (2) **Difference Recomposition**: After sorting $P_t$ in non-increasing order, bit-streams are decomposed into $n$ uniform bins using differences $d_k=p_k-p_{k+1}$, where bin $k$ contains top-$k$ behaviors with weight $q_k=k\cdot d_k$; (3) **Distribution-Preserving Sampling**: A PRG derived from a shared key $K_{\mathrm{sh}}$ and step context $C_t$ samples a bin index $K$. Inside the selected bin, $\lfloor\log_2|T|\rfloor$ or $+1$ bits are embedded via CyclicShift to output the watermarked behavior $\hat{b}_t$; (4) **RLNC Encoding**: The bits embedded in each step are treated as a linear equation $y_{t,j}=\langle a_{t,j},m\rangle$ for the payload $m\in\mathbb{F}_2^L$. During verification, residual equations are concatenated into a matrix $A_{\mathcal{I}}m=y_{\mathcal{I}}$ and solved via Gaussian elimination if the rank is sufficient.

### Key Designs

1. **Behavior-level Explicit Elicitation + Difference Recomposition**:
    - **Function**: Ensures the watermark "only modifies the sampling process, not the distribution," preventing bias accumulation in long-horizon tasks.
    - **Mechanism**: Difference recomposition splits any distribution into a mixture of $n$ uniform bins: sort $p_1\ge\dots\ge p_n$, define $d_k=p_k-p_{k+1}$, and each bin $T_k=\{b_{t,1},\dots,b_{t,k}\}$ is uniform with weight $q_k=k\cdot d_k$. Sampling first picks a bin via $\mathrm{Cat}(q_1,\dots,q_n)$, then samples uniformly within the bin. It can be proved that the marginal distribution $\Pr[\hat{b}_t=b_{t,i}]=\sum_{k=i}^n q_k/k=\sum_{k=i}^n(p_k-p_{k+1})=p_i$, strictly matching the original distribution.
    - **Design Motivation**: Directly biasing $P_t$ (RG baseline) introduces per-step bias, causing compound errors (ALFWorld-ID SR drops from 89.5% to 78.8%). Distribution-preserving sampling ensures the marginal remains unchanged. Difference recomposition transforms the distribution-preserving problem into a steganography problem over uniform distributions.

2. **CyclicShift Uniform Encoding + Key-Derived PRG Synchronization**:
    - **Function**: Embeds variable-length bits near the information-theoretic upper bound in each uniform bin while ensuring encoder-decoder synchronization.
    - **Mechanism**: For a bin of size $n$, let $k=\lfloor\log_2 n\rfloor$ and $m=n-2^k$. In the bin, $2^k-m$ indices correspond to $k$-bit codewords and $2m$ correspond to $k+1$ bits, forming a prefix-free codebook. A cyclic shift $R$ derived from the PRG maps payload bits to the bin index $j$. The expected capacity $\mathbb{E}[c\mid n]\in[\log_2 n-0.0861, \log_2 n]$. Both Encoder/Decoder derive the PRG from $K_t=H(K_{\mathrm{sh}}\Vert C_t)$ to ensure cross-party synchronization and external pseudo-randomness.
    - **Design Motivation**: Using only $\lfloor\log_2 n\rfloor$ bits wastes capacity when the bin size is not a power of 2. CyclicShift is near-optimal using variable-length codes. Binding the PRG seed to $C_t$ (step index, observation, history) allows the decoder to reconstruct randomness without extra communication—a key for black-box API deployment.

3. **RLNC Erasure-Resilient Coding and Truncation Robustness**:
    - **Function**: Disperses multi-bit watermarks across the entire trace, allowing recovery even if the platform filters logs, loses steps, or truncates the sequence, provided total observed capacity $\ge L$.
    - **Mechanism**: Each of the $c_t$ bits embedded per step is treated as a linear equation for the payload $m\in\mathbb{F}_2^L$, with coefficients $a_{t,j}=\mathrm{PRG}(K_t,j)\in\mathbb{F}_2^L$. Verification uses equations from the observed subset $\mathcal{I}\subseteq\{1,\dots,T\}$ to form $A_{\mathcal{I}}m=y_{\mathcal{I}}$ ($R=\sum_{t\in\mathcal{I}}c_t$ rows). Theoretically, when $R=L+\Delta$, the probability of full rank is $\ge 1-2^{-\Delta}$, and FPR decays exponentially with overhead $k$.
    - **Design Motivation**: Repetition coding fails quickly under high erasure. RLNC is similar to Fountain codes—"rateless" measurement makes every step an independent linear measurement, so losing any subset does not destroy recovery as long as the total count is sufficient. This is the optimal robustness strategy for long-horizon agents.

## Key Experimental Results

### Main Results
Comparison of SR and watermark capacity on ALFWorld (DeepSeek-Chat) and ToolBench (450 tasks):

| Environment | Method | SR (%) ↑ | Steps | bps ↑ | bpt ↑ | Δs/step ↓ |
|------|------|----------|-------|-------|-------|-----------|
| ALFWorld-ID | Baseline | 89.5 | 19.7 | – | – | – |
| ALFWorld-ID | RG (Agent Guide) | 78.8 (↓10.7) | 26.1 (↑6.4) | 0 bit | 0 bit | – |
| ALFWorld-ID | **Ours** | 89.3 (↓0.2) | 19.4 (↓0.3) | 1.19 | 25.5 | +0.10 |
| ALFWorld-OOD | Baseline | 96.8 | 15.9 | – | – | – |
| ALFWorld-OOD | RG | 94.5 (↓2.3) | 15.4 | 0 bit | 0 bit | – |
| ALFWorld-OOD | **Ours** | 97.5 (↑0.7) | 14.1 (↓1.8) | 1.34 | 28.4 | −0.18 |
| ToolBench | Baseline | 59.9 | 6.7 | – | – | – |
| ToolBench | **Ours** | 59.7 (↓0.2) | 7.2 | 0.49 | 4.93 | −1.27 |

### Ablation Study
Key robustness and comparative metrics:

| Dimension | Setting | Key Findings | Note |
|----------|------|---------|------|
| Step Loss Robustness (RLNC vs repetition) | Loss rate $p\in[0,0.7]$ | RLNC is significantly more stable, reaching 100% after global aggregation | Rateless property |
| Wrong Key/No Watermark FPR | $k\in[0,16]$ overhead | $k\ge 8$ FPR <1%, $k\ge 14$ zero false positives | $\propto 2^{-k}$ decay |
| Semantic Rewrite Observation | ALFWorld-OOD 134 tasks / 2326 steps | Behavior Match 49.45%, Bit Recovery 16.84% | Rewriting breaks sync but aggregation helps |
| Stacking with SynthID-Text | ToolBench | Behavior decoding 100%, Content detection 96.6% | Behavioral + Content layers are orthogonal |
| OASIS Social Simulation | Twitter/Reddit 100 trajectories | Social quality metrics nearly unchanged, high detection | Persistence validation |

### Key Findings
- The RG baseline significantly harms utility in long-horizon tasks (ALFWorld-ID SR −10.7 pp, steps +6.4), a prime example of sacrificing performance for watermarking. AgentMark-F’s <1 pp difference proves that the "distribution-preserving" promise is strictly met.
- Token/latency overhead is nearly negligible (ALFWorld Δtok/step ±0.5%, ToolBench average −6.25% due to early termination), showing that the extra prompt cost for eliciting explicit distributions is engineering-acceptable.
- Behavioral watermarks and SynthID-Text content watermarks "attack different failure modes": behavioral layers resist trace truncation/step loss, while content layers resist rewrites—stacking both yields more robust provenance.
- Cross-model validation (minimal SR change on Gemini 2.0 Flash, bps 1.96-2.28) shows the method is model-agnostic.

## Highlights & Insights
- "Behavior is the new token" is the core reframing of this paper—elevating the theory of content watermarking (KGW/SynthID/FDPSS) to the planning layer is an elegant crossover of LLM safety and information theory.
- Distribution-preserving sampling (Difference Recomposition + CyclicShift) is especially critical for long-horizon agents: direct biasing compounds errors, while preservation ensures the marginal remains strictly unchanged, providing an existence proof that "safety/watermarking need not sacrifice utility."
- The "rateless measurement" of RLNC for handling trace truncation/step loss is a robust strategy transferable to any "partially observed sequence" scenario, such as distributed log auditing or long video metadata embedding.

## Limitations & Future Work
- Requires the agent to expose an explicit planning distribution $P_t$ and candidate set $\mathcal{B}_t$. Since closed-source APIs may not provide this, prompt engineering is needed to force elicitation, which might lose fidelity.
- Weak robustness to semantic rewrites: after observation rewrites, KL=3.227 and bit recovery is only 16.84%, which is the current major weakness requiring semantic-layer reproducibility to overcome.
- For highly peaked distributions $P_t$ (e.g., only 1 candidate), the capacity per step is zero; total capacity for short-trajectory tasks is limited and requires cross-task aggregation.
- Future work involves vendors providing native APIs for planning distributions, whereas current black-box systems rely on elicitation prompts.

## Related Work & Insights
- **vs SynthID-Text (Nature 2024)**: SynthID embeds zero-bit/low-bit watermarks on token distributions to prevent content rewriting. AgentMark embeds multi-bit IDs on behavioral distributions to prevent trace truncation, and the two are orthogonal.
- **vs Agent Guide (Huang 2025, i.e., RG in text)**: Agent Guide was the first to bias behavioral probabilities but introduced distribution drift. AgentMark corrects this using FDPSS to strictly preserve the distribution.
- **vs Meteor/Discop (Classic Steganography)**: These are distribution-preserving steganography over token sequences. AgentMark applies this paradigm to agent behavior sequences and adds RLNC to solve erasure issues.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic application of distribution-preserving steganography + RLNC to agent planning; elegant cross-domain integration.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 environments × 2 models + capacity/robustness/stacking tests + theoretical FPR derivation; some task variances are large but overall comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear formal definitions, with algorithms, proofs, and a worked example provided in the appendix.
- Value: ⭐⭐⭐⭐⭐ Agent governance is an upcoming real-world necessity; the method integrates directly with black-box APIs and is compatible with content watermarking.

## Related Papers

- [\[ACL 2026\] Why Agents Compromise Safety Under Pressure](why_agents_compromise_safety_under_pressure.md)
- [\[ACL 2026\] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)
- [\[ACL 2026\] On Safety Risks in Experience-Driven Self-Evolving Agents](on_safety_risks_in_experience-driven_self-evolving_agents.md)
- [\[ACL 2026\] RISK: A Framework for GUI Agents in E-commerce Risk Management](risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)
- [\[ACL 2026\] SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models](sharedrequest_privacy-preserving_model-agnostic_inference_for_large_language_mod.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)
- [\[ACL 2026\] Why Agents Compromise Safety Under Pressure](why_agents_compromise_safety_under_pressure.md)
- [\[ACL 2026\] SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models](sharedrequest_privacy-preserving_model-agnostic_inference_for_large_language_mod.md)
- [\[ACL 2026\] On Safety Risks in Experience-Driven Self-Evolving Agents](on_safety_risks_in_experience-driven_self-evolving_agents.md)
- [\[ACL 2026\] RISK: A Framework for GUI Agents in E-commerce Risk Management](risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)

</div>

<!-- RELATED:END -->
