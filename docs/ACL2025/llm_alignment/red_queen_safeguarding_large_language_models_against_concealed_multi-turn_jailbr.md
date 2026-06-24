---
title: >-
  [Paper Note] Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking
description: >-
  [ACL 2025][LLM Alignment][Jailbreak Attack] This paper proposes Red Queen Attack, the first jailbreak attack method based on the Theory of Mind (ToM) that constructs multi-turn dialogue scenarios to conceal malicious intent, generating 56K multi-turn concealed attack data points and achieving an 87.6% ASR on GPT-4o. Concurrently, the Red Queen Guard defense strategy is introduced to reduce the ASR to <1% through multi-turn DPO data training, without compromising general bench…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Jailbreak Attack"
  - "Multi-Turn"
  - "Theory of Mind"
  - "Intent Concealment"
  - "DPO Defense"
date: 2026-05-08
content_hash: 3adb6e5d591f3431
---

# Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking

**Conference**: ACL 2025  
**arXiv**: [2409.17458](https://arxiv.org/abs/2409.17458)  
**Code**: [https://github.com/kriti-hippo/red_queen](https://github.com/kriti-hippo/red_queen)  
**Area**: AI Safety / RLHF Alignment  
**Keywords**: Jailbreak Attack, Multi-Turn, Theory of Mind, Intent Concealment, DPO Defense

## TL;DR
This paper proposes Red Queen Attack, the first jailbreak attack method based on the Theory of Mind (ToM) that constructs multi-turn dialogue scenarios to conceal malicious intent, generating 56K multi-turn concealed attack data points and achieving an 87.6% ASR on GPT-4o. Concurrently, the Red Queen Guard defense strategy is introduced to reduce the ASR to <1% through multi-turn DPO data training, without compromising general benchmark performance.

## Background & Motivation

**Background**: LLM safety alignment (RLHF/DPO) has achieved effective defense against single-turn direct malicious requests, but in real-world scenarios, attackers can progressively conceal malicious intent through multi-turn dialogues. Existing research on jailbreaking primarily focuses on single-turn attacks (e.g., GCG, AutoDAN) or simple multi-turn strategies.

**Limitations of Prior Work**: (a) Most existing jailbreak methods are single-turn with highly explicit malicious intent, failing to reflect real-world attack scenarios; (b) existing multi-turn attacks (CoSafe, CoU) and concealed attacks (ArtPrompt, DeepInception) are fragmented, with no work combining "multi-turn + concealment"; (c) the ToM capability of LLMs is weak—they struggle to infer the implicit intentions of users, a vulnerability that is severely underestimated in safety contexts.

**Key Challenge**: LLMs are trained to be "helpful", but safety mechanisms rely on identifying explicit malicious intent. When a malicious intent is covertly wrapped (such as pretending to be a good person stopping a bad person), the helpfulness tendency of the LLM overrides its harmlessness.

**Goal**: (a) Systematically evaluate the threat level of multi-turn concealed attacks on LLMs; (b) provide effective defense strategies.

**Key Insight**: Starting from the ToM theory, the attack is formalized by constructing scenarios where the model's inferred explicit intent $I_e$ is safe (Safe($I_e$)=1), while the implicit malicious intent $I_i$ is unsafe.

**Core Idea**: Leverage weak LLM ToM capabilities to construct "pseudo-protector" scenarios across multi-turn dialogues to hide malicious intent for jailbreaking, and defend against it using multi-turn DPO training.

## Method

### Overall Architecture
**Attack**: Construct multi-turn concealed scenarios $\rightarrow$ combine harmful behaviors $\rightarrow$ generate 56K attack data $\rightarrow$ evaluate 10 models.  
**Defense**: Collect attack success/rejection data $\rightarrow$ construct multi-turn DPO preference pairs $\rightarrow$ train Red Queen Guard.

### Key Designs

1. **Based on ToM Attack Formalization (Section 3.2)**:

    - Model behavior: $LLM(S, T, I_e, I_i) = R$, where $S$ is the scenario and $T$ is the task
    - $I_e = Infer(S, T)$: The explicit intent inferred by the model (safe "helping to prevent crime")
    - $I_i$: The user's actual implicit malicious intent (obtaining criminal methods)
    - Attack conditions: $Safe(I_e) = 1$ and $Safe(I_i) = 0$
    - **Design Motivation**: LLMs excel at responding to explicit requests but struggle to infer implicit intentions.

2. **Scenario Construction and Data Generation (Section 3.3-3.4)**:

    - **Function**: Generate 40 types of multi-turn concealed scenarios using Llama3.1-70B, covering various professions and relationships.
    - Two major scenario categories: **Professional** (authoritative identities like police/teachers, 5 categories) and **Relational** (friends/parent-child, etc., 5 categories).
    - Each scenario is expanded into 1/3/4/5-turn variants, with added turns incorporating more details and trust-building.
    - 56K Dataset = 40 scenarios $\times$ 1400 harmful behaviors (BeaverTails 14 classes $\times$ K-means sampling of 100 cases per class).
    - **Design Motivation**: Diverse scenarios prevent overfitting; authoritative roles (e.g., police) yield the best results.

3. **Red Queen Guard Defense (Section 6)**:

    - **Function**: Fine-tune the model on multi-turn safety preference data using DPO.
    - Sample successful attacks + Llama3.1-405B-generated safe rejections $\rightarrow$ 11.2K preference dataset.
    - DPO fine-tuning enables the model to reject harmful requests even in concealed-intent scenarios.
    - Control: 11.5K HH-RLHF general preference data is ineffective, indicating the necessity of specific data tailored to multi-turn concealed attacks.

### Loss & Training
Standard DPO loss is utilized for defense. Evaluation is conducted using Llama3-70B combined with a customized judging prompt (achieving 96% accuracy, validated as superior to all prior methods on JailbreakBench).

## Key Experimental Results

### Main Results (10 Models Attack Success Rate ASR)

| Model | Direct Attack | 1-Turn Concealed | 3-Turn | 5-Turn | Overall |
|------|----------|----------|------|------|------|
| **GPT-4o** | 0.6% | 64.7% | **87.6%** | 85.2% | **82.1%** |
| Llama3-70B | 0.9% | 20.4% | 52.4% | **77.1%** | 68.3% |
| Llama3.1-405B | 2.4% | 23.8% | 52.8% | 46.7% | 50.2% |
| Qwen2-72B | 1.3% | 29.6% | 38.3% | 54.1% | 49.2% |
| GPT-4o-mini | 0.6% | 49.1% | 30.6% | 54.8% | 44.9% |

Red Queen outperforms all baselines on **9 out of 10 models**, representing a 2%-64% increase in ASR.

### Ablation Study (Contribution of Multi-turn vs. Concealment)

| Setting | GPT-4o | Llama3-70B | Description |
|------|--------|------------|------|
| D (Direct) | 0.6% | 0.9% | Baseline |
| C (Concealed Only) | 64.7% | 20.4% | Concealment is the main contributor |
| M&D (Multi-turn Only) | 0.9% | 1.1% | Multi-turn without concealment is almost ineffective |
| **M&C** | **87.6%** | **52.4%** | Multi-turn + concealment synergistically enhance effectiveness |

### Defense Effectiveness (Red Queen Guard via DPO)

| Model | Original ASR | +RQG ASR | MMLU-Pro | AlpacaEval |
|------|---------|----------|----------|------------|
| Llama3.1-8B | 19.8% | **1.2%** | 48.3→48.3 | 27.8→26.0 |
| Llama3.1-70B | 37.9% | **1.3%** | 55.1→55.1 | 34.9→36.8 |
| Llama3.1-405B | 50.2% | **0.6%** | 64.5→64.5 | 32.0→Maintained |

### Key Findings
- **Intent concealment is core**: Utilizing the concealment strategy alone achieves a 64.7% ASR on GPT-4o, with the multi-turn structure providing auxiliary enhancement.
- **Larger models are more vulnerable**: Within the same model family, larger models are consistently easier to attack than smaller ones, because larger models have a better understanding of the mock scenarios $\rightarrow$ leading to more cooperative execution.
- **GPT-4o is the most vulnerable**: Reaching an 87.6% ASR, challenging the common perception that "GPT-4o is the safest model".
- **Authoritative scenarios are the most effective**: Professional scenarios such as the police yield the highest ASR.
- **Increasing turns generally raises ASR**: The 5-turn setting performs best on 6 out of 10 models, as more turns establish higher trust.
- **DPO defense is effective with no side-effects**: Red Queen Guard reduces ASR to <1% without degrading general capabilities.

## Highlights & Insights
- **ToM-perspective safety analysis**: Systematically explaining the attack effectiveness from a cognitive science perspective. The formalization of $LLM(S,T,I_e,I_i)=R$ can guide further safety research.
- **Counter-intuitive finding: larger models are less safe**: Models with stronger capabilities are more easily "persuaded" by scenarios, challenging the "scaling = safer" assumption. This has profound implications for AI Safety.
- **Complete attack-defense closed loop**: 56K attack data + 11.2K DPO defense data + evaluation methodology + open-source release constitute a complete infrastructure for security research.

## Limitations & Future Work
- **Scenario templates can be targeted for defense**: If specifically trained on certain types of scenarios, specific templates will fail (though generalized intent concealment remains hard to defend against).
- **English-only evaluation**: The effectiveness of multi-turn concealed attacks in other languages or cultures is untested.
- **Limited defensive generalization**: Red Queen Guard is specifically trained against this attack, and its performance on entirely new attack variants remains to be validated.
- **Attack generation cost**: Requiring an auxiliary LLM to generate scenarios introduces a higher cost compared to simple prompt attacks.

## Related Work & Insights
- **vs. DeepInception**: Uses sci-fi writing scenarios to conceal but is single-turn; Red Queen is more effective by combining multi-turn and authoritative roles.
- **vs. CoSafe**: Utilizes co-reference strategies in multi-turn dialogues but does not conceal intent; Red Queen demonstrates that concealment is the primary driver of ASR gains.
- **vs. BadChain/SEED Attack**: Manipulates reasoning steps rather than conversational scenarios—revealing LLM security vulnerabilities from a different dimension.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneering systematic study of the ToM perspective + multi-turn concealment; complete attack and defense datasets.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 models $\times$ 56K data points, broad analysis covering ablation, baselines, defense, and model sizes.
- Writing Quality: ⭐⭐⭐⭐⭐ Complete progression from ToM formalization to attack design and defense logic; vivid case studies.
- Value: ⭐⭐⭐⭐⭐ A highly significant contribution to LLM safety research; the finding that "larger models are more vulnerable" has profound implications.
   - Common Template: Claiming others are planning harmful behaviors, positioning the user as a protector
   - Two categories of scenarios: (a) Professional-based (5 categories such as police, doctors); (b) Relational-based (5 categories such as friends, family)
   - Multi-turn expansion: 3-turn, 4-turn, and 5-turn variants + a single-turn control group
   - Total of 40 scenarios $\times$ 1400 harmful behaviors (from 14 BeaverTails classes) = 56K data points
   - Harmful behavior extraction: Extracted by GPT-4o from BeaverTails, with manual validation to ensure quality

3. **Methodological Improvements in Evaluation**:
    - Found that existing evaluation methods (GCG, GPT-4o, BERT, Llama Guard) all have an accuracy of <0.8
    - Redesigned the judging prompt to focus on "whether a detailed plan or suspicious tips were provided"
    - Llama3-70B + the new judging prompt achieved a 0.96 agreement

4. **Red Queen Guard Defense (Section 6)**:
    - Collect success/refusal pairs of attack data to construct multi-turn DPO preference data
    - Perform DPO training on Llama3-8B and 70B
    - Key: Use preference pairs in a multi-turn dialogue format, rather than single-turn

## Key Experimental Results

### Main Experimental Table (Table 2: ASR across Models and Turns)

| Model | Direct | 3-Turn | 4-Turn | 5-Turn | Overall |
|------|--------|--------|--------|--------|---------|
| GPT-4o | 0.64 | 87.62 | 73.43 | 85.19 | 82.08 |
| GPT-4o-mini | 0.57 | 30.64 | 49.19 | 54.77 | 44.87 |
| Llama3-70B | 0.93 | 52.41 | 75.40 | 77.11 | 68.31 |
| Llama3.1-405B | 2.36 | 52.79 | 51.19 | 46.66 | 50.21 |
| Qwen2-72B | 1.25 | 38.26 | 55.24 | 54.10 | 49.20 |
| Mixtral-8×22B | 22.95 | 28.04 | 45.52 | 46.17 | 39.91 |

- GPT-4o, as the commercial model with the best security, reaches an ASR as high as 87.6% (3-Turn)
- Direct attack ASR is almost 0 (<3%), whereas the Red Queen Attack improves it by 15-81%

### Comparison with Prior Attacks (Figure 3)
- Red Queen Attack outperforms all existing methods on 9 out of 10 models
- ASR improvement range: 2% to 64%
- Particularly significant outperformance on GPT-4o and Llama3-70B

### Key Factor Analysis
- **Multi-turn vs. Single-turn**: The multi-turn structure (M&C) improves ASR by 5-28% compared to single-turn concealment (C)
- **Larger models are more vulnerable**: Within the same model family, larger models yield higher ASR (Llama3-8B: 19.8% vs 70B: 68.3%)
- **Scenario Category**: Professional scenarios (e.g., Police) yield the highest ASR, while relational scenarios show relatively balanced performance

### Red Queen Guard Defense Effectiveness
- Llama3-8B: ASR from 19.8% $\rightarrow$ 0.52%
- Llama3-70B: ASR from 68.3% $\rightarrow$ 0.97%
- No performance degradation on general benchmarks such as MT-Bench

### Key Findings
- Larger models are more vulnerable instead—likely because stronger instruction-following capabilities make them easier to exploit via the "helpful" bias
- The combined effect of multi-turn + concealment is greater than the sum of their individual effects
- Simple multi-turn DPO training can defend effectively, but it requires preference data in a multi-turn format

## Highlights & Insights
- **Novelty of the ToM perspective**: The concept of Theory of Mind is applied to jailbreak attack design for the first time, establishing a clear theoretical framework.
- **Large-scale attack dataset**: 56K multi-turn concealed attack data points covering 14 categories of harmful behaviors $\times$ 40 scenarios represent an important resource for security research.
- **Simple and effective defense**: Red Queen Guard requires only DPO training to reduce the ASR to <1%, demonstrating high practicality.
- **Counter-intuitive finding**: Larger models are easier to jailbreak—challenging the "scale up = safer" assumption.

## Limitations & Future Work
1. The attack data features scenarios generated by Llama3.1-70B and behaviors extracted by GPT-4o, introducing bias toward other models.
2. The defense is only validated on Llama3; DPO cannot be directly applied to closed-source models like GPT-4o.
3. The attack templates are relatively fixed ("a friend/colleague is doing something bad, I must stop them"); highly targeted defenses might be bypassed.
4. More complex adaptive attacks—where attackers are aware of the defense mechanisms—were not evaluated.

## Related Work & Insights
- Comparison with Cipher-Based Attack (Yuan et al., 2024): The latter is concealed but single-turn, while this work's multi-turn + concealment exhibits stronger effects.
- Comparison with CoSafe (Yu et al., 2024): The latter is multi-turn but unconcealed, resulting in an ASR significantly lower than Red Queen.
- Insight: ToM capability is a critical gap in LLM safety—enhancing the model's intent inference capability may prove more effective than more RLHF data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (The first ToM + multi-turn concealed jailbreak framework)
- Theoretical Depth: ⭐⭐⭐⭐ (Clear ToM formalization, comprehensive factor analysis)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (10 models $\times$ 4 families $\times$ multi-turn variants, comparison with 5+ prior methods)
- Practical Value: ⭐⭐⭐⭐⭐ (Attack dataset + defense strategy, offering both offense and defense abilities)
- Overall Recommendation: ⭐⭐⭐⭐⭐ (A major contribution to the field of LLM safety)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](m2s_multiturn_to_singleturn_jailbreak_in.md)
- [\[ACL 2025\] Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search](tempest_autonomous_multi-turn_jailbreaking_of_large_language_models_with_tree_se.md)
- [\[ACL 2025\] MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming](mtsa_multi-turn_safety_alignment_for_llms_through_multi-round_red-teaming.md)
- [\[ACL 2025\] QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)
- [\[ACL 2025\] AGD: Adversarial Game Defense Against Jailbreak Attacks in Large Language Models](agd_adversarial_game_defense_against_jailbreak_attacks_in_large_language_models.md)

</div>

<!-- RELATED:END -->
