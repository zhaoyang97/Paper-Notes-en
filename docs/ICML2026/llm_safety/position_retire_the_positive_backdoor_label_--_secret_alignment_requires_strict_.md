---
title: >-
  [Paper Note] Position: Retire the "Positive Backdoor" Label -- Secret Alignment Requires Strict and Systematic Evaluation
description: >-
  [ICML 2026][LLM Safety][Secret Alignment] This position paper advocates for the abandonment of the misleading "positive backdoor" label…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Secret Alignment"
  - "Backdoor Defense"
  - "Private LLM"
  - "CIA Evaluation"
  - "Behavior Density"
date: 2026-05-08
content_hash: a60d05025ab7cdc0
---

# Position: Retire the "Positive Backdoor" Label -- Secret Alignment Requires Strict and Systematic Evaluation

**Conference**: ICML 2026  
**arXiv**: [2605.28597](https://arxiv.org/abs/2605.28597)  
**Code**: To be confirmed  
**Area**: llm_safety  
**Keywords**: Secret Alignment, Backdoor Defense, Private LLM, CIA Evaluation, Behavior Density

## TL;DR
This position paper advocates for the abandonment of the misleading "positive backdoor" label, proposing to rename trigger-activated hidden behaviors as "Secret Alignment." Through a systematic evaluation of three representative schemes (SudoLM, Instructional Fingerprinting, and SafeTrigger) across six standardized attributes (Effectiveness, Harmlessness, Persistence, Efficiency, Robustness, Reliability), the study reveals significant vulnerabilities in Confidentiality, Integrity, and Availability (CIA). The authors call for the community to treat such mechanisms as "insecure" by default unless supported by rigorous, standardized evidence.

## Background & Motivation

**Background**: Open-source weighted LLMs (LLaMA, DeepSeek, etc.), efficient fine-tuning (LoRA, QLoRA), and consumer-grade GPUs have propelled LLMs into the "Private AI era," where individuals and small teams can customize and deploy high-performance models. This shifts LLMs from tools to digital assets, leading to three typical protection needs: access control (who can query privileged knowledge), ownership tracing (how to provide evidence if a model is stolen), and safety hardening in fine-tuning API services (preventing jailbreak fine-tuning attacks).

**Limitations of Prior Work**: Traditional encryption and system-level access controls are overly expensive and incompatible with the open-source ecosystem. Consequently, the community has borrowed backdoor techniques for defense—SudoLM uses SudoKey as a secret key to control privileged knowledge, Instructional Fingerprinting (IF) injects ownership information via instruction tuning, and SafeTrigger allows providers to inject safety triggers before and after user fine-tuning. These works label backdoors as "positive backdoors" and offer security promises such as being "effective, stealthy, and robust."

**Key Challenge**: (1) The terminology "positive backdoor" carries inherent value judgments, masking the neutral fact that it is essentially a hidden trigger-behavior mapping, leading to the fallacy that the mechanism is "secure because it is well-intentioned." (2) Existing evaluations mostly report trigger hit rates under narrow protocols, lacking systematic testing against continuous fine-tuning, distribution shifts, and adversarial probing. (3) The security value of trigger-behavior mapping depends on whether the mapping remains stable, persistent, and verifiable in real-world deployment, a point that has been largely avoided.

**Goal**: This paper decomposes the problem into two sub-problems: (a) establishing a value-neutral terminology and analysis framework to avoid the slippery slope of "positive equals secure"; (b) providing a minimal evaluation protocol aligned with CIA (Confidentiality/Integrity/Availability) to translate vague "effectiveness" into falsifiable attributes.

**Key Insight**: The authors view these mechanisms as a specific subset of "model alignment"—stealthy alignment activated only when a trigger appears. Therefore, they should follow the analytical and evaluative discipline of the alignment field rather than being treated purely as security engineering products.

**Core Idea**: Replace "positive backdoor" with the neutral term **Secret Alignment**. Through a systematic evaluation of six attributes across three case studies and an analysis along two axes (**Behavior Density** and **Decision Complexity**), the paper demonstrates that existing schemes possess vulnerabilities in CIA that were obscured in the original papers, thus they should be considered "insecure" by default.

## Method

This paper does not propose a new algorithm but instead constructs a conceptual framework and empirical protocol for "judging" Secret Alignment methods, consisting of three cooperating components.

### Overall Architecture
The authors abstract three seemingly different works (SudoLM, IF, SafeTrigger) into a unified mechanism: given a query $q$, the model defaults to output $r_1$; if and only if a secret prefix $s$ is prepended, the output switches to $r_2$, where $s$ is known only to the model owner or provider. This abstraction is named Secret Alignment: trigger-conditioned alignment on a specific subspace of the model's output distribution. Based on this, the authors organize the workflow into three steps: (1) breaking down vague "safety claims" into measurable items using six core attributes; (2) repeatedly executing the same test suite across three representative scenarios (access control, ownership tracing, and fine-tuning defense); (3) mapping failure modes of each attribute back to CIA and explaining failures using the "Behavior Density + Decision Complexity" axes.

### Key Designs

1.  **Secret Alignment Abstraction + 6-Attribute Evaluation Protocol**:
    - **Function**: Reconceptualizes "positive backdoors" as neutral, analyzable mechanisms and mandates evidence for protective claims across six attributes.
    - **Mechanism**: Defines the trigger-behavior mapping $s+q \mapsto r_2$ and $q \mapsto r_1$ as Secret Alignment. Six standardized tests are designed: Effectiveness (whether the trigger activates the behavior), Harmlessness (whether general capabilities remain without the trigger), Persistence (whether the mapping survives further fine-tuning), Efficiency (data/compute costs), Robustness (resistance to rewriting, bypassing, or false triggering), and Reliability (implicit risks in deployment interaction). These map to CIA: Leakage $\rightarrow$ C, Bypass/Overwrite $\rightarrow$ I, False Rejection/Degradation $\rightarrow$ A.
    - **Design Motivation**: Original papers often report only "trigger hit rates," packaging protection claims as invincible. The 6-attribute protocol forces promises into falsifiable metrics and allows disparate schemes to be compared on the same scale.

2.  **Trial Replication + Cross-Scheme Comparison**:
    - **Function**: Applies the 6-attribute protocol to SudoLM (access control), Instructional Fingerprinting (ownership tracing), and SafeTrigger (fine-tuning defense). It replicates and extends original experiments using a unified base model (Llama2-7B/Chat) and data pipeline.
    - **Mechanism**: For each attribute, the authors either replicate the original protocol or add missing tests. For example, Persistence is tested with successive Alpaca/Dolly/GSM8K fine-tuning. Robustness includes a 6-level input similarity gradient for IF and "BadTrigger" overwrite attacks for SafeTrigger.
    - **Design Motivation**: Existing works are isolated, making horizontal comparisons difficult. Using a unified base model and consistent perturbation families allows for direct conclusions on which method types are most vulnerable in which attributes.

3.  **Behavior Density × Decision Complexity Failure Predictability Framework**:
    - **Function**: Explains why IF is persistent but prone to false triggers, why SudoLM loses harmlessness and robustness, and why SafeTrigger is persistent but easily overwritten.
    - **Mechanism**: Categorizes methods along two orthogonal axes. Axis one is **Behavior Density**: *sparse* indicates triggers map to discrete points in output space (e.g., IF's fingerprint phrase), while *clustered* indicates triggers activate entire output regions (e.g., SudoLM’s privileged knowledge, SafeTrigger’s refusal space). Axis two is **Decision Complexity**: *simple association* is explicit memory (IF), *single-level classification* requires one input judgment (SafeTrigger's "trigger detection"), and *multi-level classification* involves deeper semantic judgment (SudoLM's judgment of "privileged knowledge" followed by "SudoKey").
    - **Design Motivation**: Instead of patching cases individually, this provides the community with an a priori framework: by locating a new Secret Alignment scheme on these axes, one can predict which CIA attributes are at high risk.

### Loss & Training
No new loss functions are introduced. Training follows original protocols: IF uses a small number of fingerprint and regular samples for SFT; SudoLM uses contrastive samples (with/without SudoKey) plus public data for SFT; SafeTrigger mixes <1% trigger-safety examples into the fine-tuning set. For evaluation, SudoLM and SafeTrigger undergo additional fine-tuning on Alpaca/Dolly/GSM8K to test Persistence, while IF is fine-tuned after 20% parameter pruning to verify erasure resistance.

## Key Experimental Results

### Main Results

| Dimension (Attribute) | SudoLM (Access Control) | Instructional Fingerprinting | SafeTrigger (FT Defense) |
| :--- | :--- | :--- | :--- |
| Effectiveness | Domain-level is fair; fine-grained (Forget10) accuracy/recall is significantly lower than claimed. | FSR ≈ 100%, consistent with original work. | ASR/HS significantly improved, consistent with original. |
| Harmlessness | Fine-grained scenarios show drops on ARC-E/PIQA/GSM8K; MMLU also decreases significantly. | Almost no loss on ARC, HellaSwag, MT-Bench. | GSM8K 24.03 $\rightarrow$ 18.35, RTE 71.12 $\rightarrow$ 63.54; widespread drops. |
| Persistence | Domain-level Acc 93.0% $\rightarrow$ 71.5% after +Alpaca; trigger logic is diluted. | 100% FSR after +Alpaca/+Dolly/+GSM8K; survives 20% pruning. | HS rises slightly, ASR $\leq$ 8% after +Alpaca/+Dolly. |
| Efficiency | High cost: requires many contrastive pairs + public data ($n \gg 1k$). | Extremely low: <10 fingerprint prompts + <150 regular samples. | Extremely low: <1% of training set + 10 trigger samples. |
| Robustness | Prefill jailbreak results in BPR 45.7% for SudoLM-D and 85.6% for SudoLM-F. | Level-1 unconditional BOS generation shows 0.7% false triggers; L3-L6 >50%. | User-injected BadTrigger increases ASR 5.5% $\rightarrow$ 18.76%; overwritten. |
| Reliability | Fine-grained scenarios exhibit hallucinations/forged privileged answers. | Fingerprint is exposed after one-time verification; vulnerable to counter-claims. | Does not prevent jailbreaks; inherits base model vulnerabilities. |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| SudoLM-D Baseline | Acc 93.0% / Prec 99.9% | Initial state of domain-level access control. |
| SudoLM-D + Alpaca FT | Acc 71.5% / Prec 76.1% | Continuous fine-tuning dilutes mapping; Persistence fails. |
| SudoLM-F Baseline | Acc 67.4% / Prec 96.8% | Fine-grained access control is weaker than domain-level from the start. |
| SudoLM-F + Prefill Jailbreak | BPR 85.56% | Almost completely bypassed; Integrity is severely compromised. |
| IF Level-1 (BOS only) | False Trigger Rate 0.7% | An order of magnitude higher than the 0.05% in original report; Reliability alarm. |
| IF Level-3 ~ Level-6 | False Trigger Rate >50% | Robustness collapses if the trigger template is partially leaked. |
| SafeTrigger + BadTrigger | ASR 5.5% $\rightarrow$ 18.76% | Adversarial triggers in user data can overwrite the safety mapping. |

### Key Findings
- **Protection claims are generally overestimated**: Except for IF's Effectiveness and Persistence, almost all "original-supported" safety claims degrade under realistic testing; fine-grained access control (SudoLM-F) is virtually unusable with over 85% bypass rate under prefill attacks.
- **Failure modes are predictable by the two axes**: IF's *sparse+simple* nature makes it persistent but extremely vulnerable to induction by similar templates. SudoLM's *clustered+multi-level* nature hurts harmlessness and fails to survive continuous fine-tuning. SafeTrigger’s *clustered+single-level* profile is persistent but easily overwritten by adversarial triggers.
- **Efficiency and security are often inversely correlated**: SudoLM incurs the highest data/compute cost yet performs worst in Harmlessness, Persistence, and Robustness, indicating that high investment does not automatically yield reliable protection.
- **Reliability blind spots are frequently ignored**: IF loses stealth after one verification, SafeTrigger does not prevent jailbreaks, and SudoLM produces hallucinated privileged answers—these are secondary risks resulting from the interaction between the mechanism and the deployment environment.

## Highlights & Insights
- **Terminological Intervention**: Renaming "positive backdoor" to Secret Alignment cuts the semantic slippery slope of "positive equals secure," pulling the discussion back to the mechanism level.
- **CIA × 6-Attribute Alignment**: Mapping traditional security engineering's CIA to ML metrics makes "protective claims" falsifiable and comparable for the first time; this protocol can be migrated to watermarking, unlearning, and persona alignment.
- **Two-Axis Framework**: Providing a priori predictions for failure modes acts as a "high-risk checklist" for future work—sparse+simple schemes should prioritize Robustness, while clustered+multi-level schemes should prioritize Harmlessness and Persistence.
- **Adversarial Protocols**: The similarity gradients for IF, BadTrigger overwriting for SafeTrigger, and prefill jailbreaking for SudoLM represent a minimal test suite for any Secret Alignment work.

## Limitations & Future Work
- Experiments were limited to Llama2-7B/Chat; whether behavior density and decision complexity hold on larger or newer models like Llama-3 or DeepSeek requires further verification.
- Only three representative schemes were examined; more recent watermarking or unlearning works (e.g., Gold-Stamp, TOFU) were not included.
- The two axes are currently qualitative; quantitative measures (e.g., effective dimensionality of output distributions, Lipschitz estimates of decision boundaries) could be developed.
- As a position paper, it does not propose a new method or specific design advice for balancing sparse+simple with robustness.
- The CIA × 6-attribute protocol could be formalized into an open-source benchmark for standardized reporting in future Secret Alignment papers.

## Related Work & Insights
- **vs. SudoLM (Liu et al., 2024b)**: While the original paper claimed both domain and fine-grained control are viable, this work identifies a >85% bypass rate under prefill attacks for fine-grained scenarios.
- **vs. Instructional Fingerprinting (Xu et al., 2024)**: This paper confirms the FSR claim but reveals a high false trigger rate for similar templates, redefining IF as "persistent but high-exposure risk."
- **vs. SafeTrigger (Wang et al., 2024)**: Supplemented "user-side BadTrigger overwrite" experiments reveal Integrity weaknesses and a lack of jailbreak prevention.
- **vs. Traditional Backdoor Surveys**: Continues the view of trigger-behavior mapping as a neutral mechanism but shifts it from attack literature to a defense-evaluation context.
- **vs. General Alignment/Safety Alignment**: Treating Secret Alignment as a subset of alignment allows the application of "superficial alignment" hypotheses to explain Secret Alignment failures.

## Rating
- **Novelty**: ⭐⭐⭐⭐ No new algorithm, but the Secret Alignment abstraction + CIA × 6-attribute protocol + two-axis framework constitutes a high-quality position.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three cases × six attributes × multiple perturbations; replicates and extends. Larger model coverage is missing.
- **Writing Quality**: ⭐⭐⭐⭐ Clear argumentation chain: rename $\rightarrow$ protocol $\rightarrow$ case studies $\rightarrow$ predictive framework.
- **Value**: ⭐⭐⭐⭐⭐ Directly impacts evaluation standards for watermarking, access control, and safety-trigger research, reducing the risk of "false security" in Private AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs](../../NeurIPS2025/llm_safety/a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)
- [\[ICML 2026\] TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning](tcap_tri-component_attention_profiling_for_unsupervised_backdoor_detection_in_ml.md)
- [\[ICML 2026\] BYORn: Bootstrap Your Own Responses to Defend Large Vision-Language Models Against Backdoor Attacks](byorn_bootstrap_your_own_responses_to_defend_large_vision-language_models_agains.md)
- [\[ICML 2026\] Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering](position_uncertainty_quantification_in_llms_is_just_unsupervised_clustering.md)
- [\[ICML 2026\] LLM Benchmark Datasets Should Be Contamination-Resistant (Position Paper)](llm_benchmark_datasets_should_be_contamination-resistant.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[NeurIPS 2025\] A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs](../../NeurIPS2025/llm_safety/a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)
- [\[ICML 2026\] Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering](position_uncertainty_quantification_in_llms_is_just_unsupervised_clustering.md)
- [\[ICML 2026\] LLM Benchmark Datasets Should Be Contamination-Resistant (Position Paper)](llm_benchmark_datasets_should_be_contamination-resistant.md)
- [\[ICML 2026\] TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning](tcap_tri-component_attention_profiling_for_unsupervised_backdoor_detection_in_ml.md)
- [\[NeurIPS 2025\] Position: The Complexity of Perfect AI Alignment -- Formalizing the RLHF Trilemma](../../NeurIPS2025/llm_safety/position_the_complexity_of_perfect_ai_alignment_--_formalizing_the_rlhf_trilemma.md)

</div>

<!-- RELATED:END -->
