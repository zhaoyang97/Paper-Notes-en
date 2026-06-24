---
title: >-
  [Paper Note] Position: Retire the "Positive Backdoor" Label -- Secret Alignment Requires Strict and Systematic Evaluation
description: >-
  [ICML2026][AI Safety][Secret Alignment] This position paper advocates for the retirement of the misleading label "positive backdoor," proposing to rename trigger-activated hidden behaviors as "Secret Alignment." Through a systematic evaluation of three representative schemes (SudoLM, Instructional Fingerprinting, SafeTrigger) across six standardized attributes (Effectiveness, Harmlessness, Persistence, Efficiency, Robustness, Reliability), the authors reveal the vulnerability…
tags:
  - "ICML2026"
  - "AI Safety"
  - "Secret Alignment"
  - "Backdoor Defense"
  - "Private LLM"
  - "CIA Evaluation"
  - "Behavior Density"
date: 2026-05-08
content_hash: e0bbcbe91b003729
---

# Position: Retire the "Positive Backdoor" Label -- Secret Alignment Requires Strict and Systematic Evaluation

**Conference**: ICML2026  
**arXiv**: [2605.28597](https://arxiv.org/abs/2605.28597)  
**Code**: To be confirmed  
**Area**: LLM Security  
**Keywords**: Secret Alignment, Backdoor Defense, Private LLM, CIA Evaluation, Behavior Density

## TL;DR
This position paper advocates for the retirement of the misleading label "positive backdoor," proposing to rename trigger-activated hidden behaviors as "Secret Alignment." Through a systematic evaluation of three representative schemes (SudoLM, Instructional Fingerprinting, SafeTrigger) across six standardized attributes (Effectiveness, Harmlessness, Persistence, Efficiency, Robustness, Reliability), the authors reveal the vulnerability of such mechanisms in terms of Confidentiality, Integrity, and Availability (CIA). The paper calls for the community to treat these mechanisms as "insecure" by default unless supported by rigorous, standardized evidence.

## Background & Motivation

**Background**: Open-source weights (LLaMA, DeepSeek, etc.), efficient fine-tuning (LoRA, QLoRA), and consumer-grade GPUs have pushed LLMs into the "Private AI era," where individuals and small teams can own and deploy high-performance models. Consequently, LLMs have evolved from tools into digital assets, raising three typical protection requirements: access control (who can query privileged knowledge), ownership tracing (how to provide evidence if a model is stolen), and safety hardening in fine-tuning API services (preventing jailbreak fine-tuning attacks).

**Limitations of Prior Work**: Traditional encryption and system-level access controls are overly expensive and incompatible with the open-source ecosystem. Thus, the community began borrowing backdoor techniques for defense—SudoLM uses a "SudoKey" as a secret key to control privileged knowledge, Instructional Fingerprinting (IF) injects ownership information via instruction fine-tuning, and SafeTrigger allows service providers to inject safety triggers before and after user fine-tuning. These works refer to backdoors as "positive backdoors" and directly offer security promises such as "effectiveness, stealthiness, and robustness."

**Key Challenge**: (1) The phrasing "positive backdoor" carries inherent value judgments, masking the neutral fact that it is fundamentally a hidden "trigger-behavior mapping," which misleadingly suggests security due to "good intentions." (2) Existing evaluations mostly report trigger hit rates under narrow protocols, lacking systematic testing against continuous fine-tuning, distribution shifts, and adversarial probing. (3) The security value of trigger-behavior mapping depends on whether the mapping remains robust, persistent, and verifiable in real-world deployment—points that are precisely circumvented.

**Goal**: The paper addresses two sub-problems: (a) providing a de-valorized terminology and analysis framework for such mechanisms to avoid the slippery slope of "positive equals secure"; (b) establishing a minimum evaluation protocol aligned with CIA (Confidentiality/Integrity/Availability) to translate vague "effectiveness" into falsifiable attributes.

**Key Insight**: The authors view these mechanisms as a special subset of "model alignment"—concealed alignment activated only upon trigger appearance. Therefore, they should follow the analysis and evaluation disciplines of the alignment field rather than being treated purely as products of security engineering.

**Core Idea**: Replace "positive backdoor" with the neutral term **Secret Alignment**. Conduct a systematic evaluation of "six attributes × three cases" and a two-axis analysis of **Behavior Density/Decision Complexity**. The results demonstrate that existing schemes possess vulnerabilities in CIA hidden by the original papers, and thus should be considered "insecure" by default.

## Method

### Overall Architecture
The paper does not propose a new algorithm but establishes a conceptual framework and empirical protocol for judging "positive backdoors." The authors argue that these mechanisms are hidden trigger-behavior mappings and should be renamed to **Secret Alignment**. The demonstration proceeds in three steps: first, abstracting three seemingly different works into a single mechanism and decomposing vague "safety claims" into measurable items using six attributes; second, running the same test suite across access control, ownership tracing, and fine-tuning defense scenarios; finally, mapping attribute failures back to CIA and explaining why failures are predictable using the "Behavior Density × Decision Complexity" axes.

### Key Designs

**1. Secret Alignment Abstraction + Six-Attribute Evaluation Protocol: Replacing Semantic Slippage with Falsifiable Metrics**

Original papers often report only "trigger hits," wrapping protective claims as infallible. The authors abstract SudoLM, IF, and SafeTrigger into a unified mechanism: given a query $q$, the model outputs $r_1$ by default; if and only if a secret prefix $s$ (known only to the owner) is prepended, the output switches to $r_2$, i.e., $s+q \mapsto r_2$ while $q \mapsto r_1$. This is trigger-conditional alignment to a subspace of the output distribution. Based on this, six standardized tests are designed: Effectiveness (whether the trigger activates the behavior), Harmlessness (persistence of general capabilities without the trigger), Persistence (mapping survival after further fine-tuning), Efficiency (data/compute costs), Robustness (resistance to rewriting, bypassing, or false triggering), and Reliability (implicit risks at the deployment interaction layer). These correspond to CIA: Leakage $\rightarrow$ C, Bypassing/Overwriting $\rightarrow$ I, False Rejection/Degradation $\rightarrow$ A.

**2. Parallel Case Reproduction + Cross-Scheme Comparison: Using Unified Base Models and Consistent Perturbations**

To overcome the lack of horizontal comparability, the authors apply the six-attribute protocol to SudoLM (multi-level access control), IF (ownership tracing), and SafeTrigger (fine-tuning defense). They use Llama2-7B/Chat as the base and a shared data pipeline. Training follows original schemes: IF uses <10 fingerprint samples, SudoLM uses contrastive samples (with/without SudoKey), and SafeTrigger mixes <1% trigger-safety examples into the fine-tuning set. For Persistence, SudoLM/SafeTrigger undergo continuous fine-tuning on Alpaca/Dolly/GSM8K, and IF is tested after 20% parameter pruning. For Robustness, they add tests like six-level input similarity gradients for IF and "BadTrigger" overwriting attacks for SafeTrigger.

**3. Behavior Density × Decision Complexity Framework for Predictable Failure**

To explain why IF is persistent but prone to false triggers, or why SudoLM loses harmlessness and robustness, the authors classify methods along two orthogonal axes. **Behavior Density**: *sparse* indicates triggers map to discrete points in output space (e.g., IF phrases), while *clustered* indicates triggers activate entire output regions (e.g., privileged knowledge in SudoLM). **Decision Complexity**: *simple association* is explicit memory (IF), *single-level classification* requires input discrimination (SafeTrigger "trigger exists?"), and *multi-level classification* requires deeper semantic judgment (SudoLM "is it privileged knowledge?" then "is SudoKey present?"). Position on these axes predicts failure modes: sparse + simple $\rightarrow$ prone to overfitting fingerprints, poor robustness but persistent; clustered + multi-level $\rightarrow$ high overlap with normal output, poor harmlessness and prone to drift.

## Key Experimental Results

### Main Results

| Dimension (Evaluation Attribute) | SudoLM (Access Control) | Instructional Fingerprinting | SafeTrigger (Fine-tuning Defense) |
|---------------------------------|-------------------------|------------------------------|--------------------------------|
| Effectiveness | Domain-level fair; Fine-grained (Forget10) Acc/Rec significantly lower than claimed | FSR ≈ 100%, consistent with original | ASR/HS significantly improved, consistent with original |
| Harmlessness | Performance drops on ARC-E/PIQA/GSM8K; MMLU also decreases | Almost no loss on ARC, HellaSwag, MT-Bench | GSM8K 24.03 → 18.35, RTE 71.12 → 63.54, general degradation |
| Persistence | Domain-level Acc 93.0% → 71.5% after +Alpaca; logic diluted | 100% FSR after +Alpaca/+Dolly/+GSM8K; survives 20% pruning | HS rises slightly, ASR ≤ 8% after +Alpaca/+Dolly |
| Efficiency | High cost: requires large contrastive + public data ($n \gg 1k$) | Extremely low: <10 fingerprint prompts | Extremely low: <1% original training set |
| Robustness | SudoLM-D BPR 45.7%, SudoLM-F 85.6% under Prefill jailbreak | 0.7% false trigger at L1 (unconditional); >50% at L3-L6 | ASR 5.5% → 18.76% after user-injected BadTrigger; overwritten |
| Reliability | Hallucinations/fake privileged answers in fine-grained scenarios | Fingerprint exposed after one verification; vulnerable to counter-claims | Does not prevent jailbreaks; inherits base model vulnerabilities |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| SudoLM-D Baseline | Acc 93.0% / Prec 99.9% | Initial state of domain-level access control |
| SudoLM-D + Alpaca FT | Acc 71.5% / Prec 76.1% | Continuous fine-tuning dilutes mapping; Persistence failure |
| SudoLM-F Baseline | Acc 67.4% / Prec 96.8% | Fine-grained access control is weaker than domain-level from the start |
| SudoLM-F + Prefill Jailbreak | BPR 85.56% | Almost completely bypassed; Integrity failure |
| IF Level-1 (BOS only) | False Trigger 0.7% | Magnitude higher than original 0.05%; Reliability alert |
| IF Level-3 ~ Level-6 | False Trigger >50% | Robustness collapses once trigger template is partially leaked |
| SafeTrigger + BadTrigger | ASR 5.5% → 18.76% | User can inject adversarial triggers in training data for overwriting |

### Key Findings
- **Protective claims are generally overestimated**: Except for IF's Effectiveness and Persistence, almost all safety claims supported by original papers are discounted under realistic testing. Fine-grained access control (SudoLM-F) is virtually unusable, with bypass rates exceeding 85% under prefill attacks.
- **Failure modes are predictable by the two axes**: IF’s sparse+simple nature makes it persistent but extremely vulnerable to similar templates. SudoLM’s clustered+multi-level nature ruins harmlessness and fails continuous fine-tuning. SafeTrigger’s clustered+single-level nature is persistent but easily overwritten by adversarial triggers.
- **Efficiency and security are often inversely correlated**: SudoLM incurs the highest data/compute costs yet performs worst in Harmlessness, Persistence, and Robustness, indicating that "heavy investment" does not automatically grant reliable protection.
- **Reliability blind spots are frequently ignored**: IF loses stealth after one verification and faces counter-claims; SafeTrigger does not prevent jailbreaks; SudoLM generates hallucinated privileged answers. These are secondary risks from mechanism-environment interaction.

## Highlights & Insights
- **Terminological Intervention**: Renaming "positive backdoor" to **Secret Alignment** severs the semantic slippage of "positive equals secure," returning the discussion to mechanism analysis.
- **Alignment of CIA × Six Attributes**: Mapping traditional security engineering's CIA directly to machine learning metrics makes protective claims falsifiable and comparable for the first time.
- **Behavior Density × Decision Complexity Axes**: Provides a prior for predicting failure modes—a "high-risk checklist" for future work.
- **Directly Reusable Adversarial Protocols**: The similarity gradients for IF, BadTrigger overwriting for SafeTrigger, and prefill jailbreaking for SudoLM constitute a minimum test suite for any Secret Alignment work.

## Limitations & Future Work
- Experiments were conducted only on Llama2-7B/Chat, excluding newer bases like Llama-3 or DeepSeek; whether the two-axis framework holds on larger models requires verification.
- Only three representative schemes were examined; newer watermark/unlearning works (e.g., Gold-Stamp, TOFU derivatives) were not covered.
- The Behavior Density and Decision Complexity axes are currently qualitative; quantitative measures (e.g., effective dimensionality of output distribution) could be explored using interpretability tools.
- As a position paper, it does not propose a new method or specific design suggestions on balancing sparse+simple with robustness.
- The CIA × Six-Attribute protocol could be solidified into an open-source benchmark, requiring Secret Alignment papers to report standardized results.

## Related Work & Insights
- **vs SudoLM (Liu et al., 2024b)**: The original claimed utility for both domain and fine-grained control; this work demotes it to a "high-cost, low-guarantee" mechanism due to bypass rates > 85%.
- **vs Instructional Fingerprinting (Xu et al., 2024)**: This work confirms FSR claims but reveals high false trigger rates (>50% for similar templates), re-positioning IF as "persistent but high exposure risk."
- **vs SafeTrigger (Wang et al., 2024)**: This work exposes Integrity weaknesses via the "BadTrigger overwrite" experiment and notes its design neglect of jailbreaks.
- **vs General Alignment/Safety Alignment**: Treating Secret Alignment as a subset of alignment allows using alignment findings (e.g., superficial alignment hypothesis) as explanations for Secret Alignment failures.

## Rating
- **Novelty**: ⭐⭐⭐⭐ No new algorithm, but the Secret Alignment abstraction + CIA protocol + two-axis framework form a high-quality position.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 3 cases × 6 attributes × multiple perturbation families; lacks some newer case/model coverage.
- **Writing Quality**: ⭐⭐⭐⭐ Clear argumentation chain: terminological correction → protocol building → case execution → predictive framework.
- **Value**: ⭐⭐⭐⭐⭐ Directly impacts evaluation standards for watermarking, access control, and safety triggers, reducing risks of "false security" in Private AI deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Embodied AI Requires a Privacy-Utility Trade-off](position_embodied_ai_requires_a_privacy-utility_trade-off.md)
- [\[ICML 2026\] Position: 'AI Alignment' Encompasses Competing Technical Priorities](ai_alignment_encompasses_competing_technical_priorities.md)
- [\[ICML 2026\] FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing](fusefss_efficient_secure_llm_inference_with_function_secret_sharing.md)
- [\[ICML 2026\] MLUBench: A Benchmark for Lifelong Unlearning Evaluation in MLLMs](mlubench_a_benchmark_for_lifelong_unlearning_evaluation_in_mllms.md)
- [\[ICML 2026\] Alignment Risks from Capability-Seeking RL Training](alignment_risks_from_capability-seeking_rl_training.md)

</div>

<!-- RELATED:END -->
