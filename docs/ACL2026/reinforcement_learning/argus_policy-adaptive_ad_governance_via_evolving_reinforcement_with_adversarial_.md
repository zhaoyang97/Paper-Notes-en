---
title: >-
  [Paper Note] ARGUS: Policy-Adaptive Ad Governance via Evolving Reinforcement with Adversarial Umpiring
description: >-
  [ACL 2026][Reinforcement Learning][Policy-Adaptive] ARGUS utilizes a Prosecutor–Defender–Umpire tripartite debate framework combined with GRPO reinforcement learning. This approach enables ad review VLMs to rectify histo…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Policy-Adaptive"
  - "Multi-Agent Debate"
  - "GRPO"
  - "Label Rectification"
  - "Industrial Deployment"
date: 2026-05-08
content_hash: 8594b9f09ae7f706
---

# ARGUS: Policy-Adaptive Ad Governance via Evolving Reinforcement with Adversarial Umpiring

**Conference**: ACL 2026  
**arXiv**: [2605.02200](https://arxiv.org/abs/2605.02200)  
**Code**: None  
**Area**: Reinforcement Learning / Ad Governance / Multimodal  
**Keywords**: Policy-Adaptive, Multi-Agent Debate, GRPO, Label Rectification, Industrial Deployment

## TL;DR
ARGUS utilizes a Prosecutor–Defender–Umpire tripartite debate framework combined with GRPO reinforcement learning. This approach enables ad review VLMs to rectify historical "outdated labels" and uncover potential violations in grey areas as policies evolve. Industrial A/B testing demonstrates a relative reduction in Violation Leakage Rate (VLR) by 35.2%.

## Background & Motivation

**Background**: Internet advertisement governance relies heavily on Large Vision-Language Models (VLMs). Conventional approaches (SFT + static rules) assume policies are static. Existing RL/CoT frameworks such as RAVEN, Hi-Guard, and BLM-Guard also presuppose fixed regulations.

**Limitations of Prior Work**: Regulatory policies are frequently updated (e.g., new bans on K12 exam-related anxiety, appearance anxiety, or information-gap scams), yet massive historical datasets are labeled based on obsolete policies. Directly fine-tuning on this data leads to three issues: (1) **Label Inconsistency**—historical "compliant" samples may be violations under new policies; (2) **推理模糊 (Vague Reasoning)**—new policies contain "grey areas" where binary labels fail to teach the model complex decision logic; (3) **Hard Sample Recall**—covert violations remain hidden within massive compliant traffic.

**Key Challenge**: Vanilla SFT faces "gradient conflict" from old labels when learning new policies, leading to catastrophic forgetting (historical recall plummeting from 0.858 to 0.432). Continual learning methods like EWC preserve old knowledge but struggle to adequately absorb new policies.

**Goal**: Implement a three-stage reinforcement learning pipeline to "establish a foundation—rectify history—discover grey areas," maintaining historical performance while continuously absorbing new policies.

**Key Insight**: Upgrade the reward signal from a "single judge" to a "multi-agent structured debate." The Prosecutor identifies violation reasons, the Defender provides compliance justifications, and the Umpire delivers the final verdict using RAG-retrieved policy clauses. The debate itself serves as the source of reward shaping.

**Core Idea**: Use tripartite "Prosecutor-Defender-Umpire" debate + RAG-enhanced adjudication + GRPO to transform verdicts into policy rewards, synchronizing "Policy Evolution" with "Model Policy Evolution."

## Method

### Overall Architecture
ARGUS is a GRPO-driven three-stage pipeline: **Stage I Policy Seeding** involves SFT using $\mathcal{D}_\text{gold}$ (sparse gold labels for new policies) + 40% historical subset $\mathcal{D}_\text{hist}'$ to obtain $f_{\theta_\text{base}}$. **Stage II Adversarial Label Rectification** employs the Prosecutor–Defender–Umpire structure to generate new rewards $R_\text{rect}$ for historical data, overriding label noise. **Stage III Latent Knowledge Discovery** introduces a "Skeptic" agent that voices internal model doubts; these are adjudicated alongside P/D by the Umpire to uncover covert violations.

### Key Designs

1. **Prosecutor–Defender–Umpire Adversarial Debate (Core of Stage II)**:
    - **Function**: Scrubs old labels from the reward signal, ensuring historical samples contribute high-fidelity gradients aligned with new policies.
    - **Mechanism**: For a historical sample, the Prosecutor (current policy) provides a violation CoT based on new policy $\Delta\mathcal{P}$; the Defender (strong VLM) writes a compliance rebuttal CoT; the Umpire (neutral VLM) uses RAG to retrieve $\Delta\mathcal{P}$ clauses and $\mathcal{D}_\text{gold}$ references to output the rectified label $y^*$ and standard rationale $\mathcal{C}^*$. The reward $R_\text{rect}(y,\mathcal{C}) = \mathbf{1}(y=y^*) + \text{sim}(\mathcal{C},\mathcal{C}^*)$ rewards both correct labels and reasoning paths similar to the Umpire, injecting semantic supervision into GRPO.
    - **Design Motivation**: A single judge tends towards being overly conservative or permissive. Hostile defense forces the Umpire to find a rational middle ground, rectifying labels while preventing the sacrifice of creative tolerance for new policies.

2. **Latent Candidate Selection + Tripartite Dialectic (Core of Stage III)**:
    - **Function**: Re-trains on covert violations where the model predicts compliance but has low internal confidence.
    - **Mechanism**: Defines $\mathcal{D}_\text{latent} = \{x\in\mathcal{D}_\text{hist} \mid y^{(k)}=0\text{ and }P(y^{(k)}=1|x)>\tau\}$. The Skeptic (current $f_\theta$) provides a CoT on "why I am hesitating," which is fed to the Umpire alongside P/D perspectives for tripartite adjudication to obtain $y^*, \mathcal{C}^*$ for GRPO using $R_\text{latent}$.
    - **Design Motivation**: Stage II resolves obvious conflicts, but grey-area ground truths are harder to obtain. Using the model's own doubt as a third-party perspective transforms uncertainty into a learning signal.

3. **Strategic Data Mixing + Lock-and-Key Evolution (Cross-stage)**:
    - **Function**: Evolves reward signals with policy updates while maintaining historical compliance rates.
    - **Mechanism**: Stage I uses $\mathcal{D}_\text{gold} \cup \mathcal{D}_\text{hist}'$ to prevent sparse gold labels from being drowned by old noise. Stages II/III use Umpire outputs as primary GRPO rewards. The system is deployed as a "pre-screening → ARGUS judgment → human audit → feedback" loop.
    - **Design Motivation**: Decouples "data replacement" from "reward replacement." SFT provides basic policy awareness, while RL embeds evolving policies into reasoning paths.

### Loss & Training
- Stage I: SFT using $\mathcal{L}_\text{stage1}(\theta) = -\sum \log P(\mathbf{y}, \mathcal{C} | x, \mathcal{P}_\text{new}; \theta)$.
- Stage II/III: GRPO with $R_\text{rect}$ and $R_\text{latent}$ as rewards. Backbones: Qwen3-VL-8B / Qwen2.5-VL-7B.

## Key Experimental Results

### Main Results (Industrial Dataset, 5 New Policies $\Delta\mathcal{P}$)

| Method (Qwen3-VL-8B Backbone) | Hist. Prec. | Hist. Rec. | Avg $\Delta\mathcal{P}$ Prec. | Avg $\Delta\Delta\mathcal{P}$ Rec. |
| :--- | :--- | :--- | :--- | :--- |
| Historical Expert (SFT on $\mathcal{D}_\text{hist}$) | 0.842 | 0.858 | 0.374 | 0.443 |
| GPT-4o (zero-shot) | 0.485 | 0.612 | 0.450 | 0.593 |
| Qwen3-235B-A22B (zero-shot) | 0.512 | 0.635 | 0.487 | 0.631 |
| Vanilla SFT (on $\mathcal{D}_\text{gold}$) | 0.454 | **0.432†** | 0.774 | 0.748 |
| SFT + Replay (40%) | 0.791 | 0.785 | 0.753 | 0.733 |
| EWC | 0.802 | 0.794 | 0.760 | 0.741 |
| **ARGUS (Ours)** | **0.828** | **0.841** | **0.795** | **0.836** |

†= Catastrophic forgetting. ARGUS-8B historical recall is only 1.7% lower than the Historical Expert, but its $\Delta\mathcal{P}$ recall is 9.5 percentage points higher than EWC. Online A/B: VLR reduced by 35.2%, AAR (Auto-Approval Rate) increased by 11.2%, and FPR decreased from 0.35% to 0.32%.

### Ablation Study (By Stage + By Agent)

| Configuration | Hist. Rec. | Avg $\Delta\mathcal{P}$ Prec. | Avg $\Delta\mathcal{P}$ Rec. |
| :--- | :--- | :--- | :--- |
| Stage I only | 0.785 | 0.753 | 0.733 |
| + Stage II (Rectification) | 0.824 | 0.758 | 0.792 |
| + Stage III (Latent Discovery) | **0.841** | **0.795** | **0.836** |

| Agent Ablation | Avg $\Delta\mathcal{P}$ Prec. | Avg $\Delta\mathcal{P}$ Rec. |
| :--- | :--- | :--- |
| Full ARGUS | 0.795 | 0.836 |
| w/o Prosecutor | 0.732 | 0.695 |
| w/o Defender | 0.684 | 0.812 |
| w/o Rationale (Label only, no CoT) | 0.715 | 0.742 |

### Key Findings
- **Clear Marginal Gains across Stages**: Stage II improves historical recall (+3.9) and $\Delta\mathcal{P}$ recall (+5.9); Stage III improves $\Delta\mathcal{P}$ precision (+3.7).
- **Prosecutor for Recall, Defender for Precision**: Removing the Defender drops precision to 0.684 but increases recall to 0.812 (over-audit), proving the agents act as opposing constraints.
- **CoT Rationale is the Soul of Policy Rewards**: Using agents as mere binary labelers drops performance significantly, indicating the "debate text" is more effective than "final voting."
- **Robustness against Adversarial Evasion**: On samples with homophone replacements and blurring, standard SFT recall drops by 38.1%, while ARGUS only drops 6.2%.

## Highlights & Insights
- **Synchronization of Policy and Reward Evolution**: Traditional RLHF assumes a stable reward function. This paper treats rewards as dynamic objects synthesized via real-time LLM debate, providing a template for fields with evolving rules (finance, safety, medical).
- **Skeptic Design**: Using the current policy as a skeptic for tripartite adjudication transforms uncertainty into training signals, which is more refined than simple hard sample selection.
- **Lock-and-Key in Data Evolution**: Using sparse gold labels as seeds and massive history as capacity buffers prevents new policies from being overwhelmed by old gradients.

## Limitations & Future Work
- Currently validated on image-text ads; temporal violations in video ads remain future work.
- Heavy reliance on Umpire VLM capabilities; a weak Umpire may introduce bias.
- High inference costs of multi-stage/multi-agent reasoning require cascaded filtering for industrial latency control.

## Related Work & Insights
- **vs RAVEN / Hi-Guard / BLM-Guard**: These treat policies as static; ARGUS is the first multi-agent framework to support dynamic policy evolution.
- **vs EWC / Replay**: CL methods passively protect old knowledge; ARGUS actively rewrites historical rewards using an Umpire.
- **vs Constitutional AI**: While CAI uses a single set of principles for self-correction, ARGUS uses RAG + multi-agents to make principles a dynamic, retrievable object.

## Rating
- Novelty: ⭐⭐⭐⭐ Combination of multi-agent debate + 3-stage GRPO for evolving policies.
- Experimental Thoroughness: ⭐⭐⭐⭐ Industrial + Public datasets + Online A/B + Ablations.
- Writing Quality: ⭐⭐⭐⭐ Table 1's case studies allow readers to grasp the multi-agent reasoning quickly.
- Value: ⭐⭐⭐⭐⭐ A rare industrial deployment paper providing a reusable blueprint for policy compliance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Free Energy-Driven Reinforcement Learning with Adaptive Advantage Shaping for Unsupervised Reasoning in LLMs](free_energy-driven_reinforcement_learning_with_adaptive_advantage_shaping_for_un.md)
- [\[ICML 2026\] Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization](../../ICML2026/reinforcement_learning/metis_learning_to_jailbreak_llms_via_self-evolving_metacognitive_policy_optimiza.md)
- [\[ACL 2026\] Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning](easy_samples_are_all_you_need_self-evolving_llms_via_data-efficient_reinforcemen.md)
- [\[ACL 2026\] LANG: Reinforcement Learning for Multilingual Reasoning with Language-Adaptive Hint Guidance](lang_reinforcement_learning_for_multilingual_reasoning_with_language-adaptive_hi.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](../../ICLR2026/reinforcement_learning/spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)

</div>

<!-- RELATED:END -->
