---
title: >-
  [Paper Note] Reward Modeling from Natural Language Human Feedback
description: >-
  [ICML 2026][LLM Alignment][Generative Reward Model (GRM)] This paper identifies a severe "outcome-process inconsistency" (20–30%, up to 44%) in generative reward models (GRM) trained on binary preference rewards…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "Generative Reward Model (GRM)"
  - "Process Reward"
  - "Natural Language Feedback"
  - "MetaRM"
  - "GRPO"
date: 2026-05-08
content_hash: 7ea0db742e8aa84c
---

# Reward Modeling from Natural Language Human Feedback

**Conference**: ICML 2026  
**arXiv**: [2601.07349](https://arxiv.org/abs/2601.07349)  
**Code**: Not released  
**Area**: LLM Alignment / Reward Modeling / RLHF  
**Keywords**: Generative Reward Model (GRM), Process Reward, Natural Language Feedback, MetaRM, GRPO

## TL;DR
This paper identifies a severe "outcome-process inconsistency" (20–30%, up to 44%) in generative reward models (GRM) trained on binary preference rewards, where the model guesses the correct preference but provides an incorrect critique. The authors propose RM-NLHF: using the similarity between model and human critiques on core arguments as an additional process reward, and employing MetaRM to automatically predict process rewards and update them online with policy changes. This approach consistently outperforms outcome-only GRPO-trained SOTA GRMs across multiple benchmarks.

## Background & Motivation

**Background**: Generative Reward Models (GRM), capable of outputting both critique and preference labels, are more robust and interpretable than traditional scalar RMs, making them mainstream for LLM alignment and RLHF. Training typically uses RLVR + GRPO: the model generates reasoning and critique for a pair of answers, then outputs an A/B label. The binary reward $R_{\text{outcome}}\in\{0,1\}$ is based on label-ground truth match.

**Limitations of Prior Work**: The authors conduct comparative experiments on MATH-500 (math, large solution space) and HelpSteer3 (pairwise reward, binary solution space). For math tasks, correct outcomes almost always imply correct processes, with negligible inconsistency. However, for pairwise rewarding, RM-R1-DeepSeek-Distilled-Qwen-7B exhibits a 44.24% "correct outcome / incorrect critique" rate, gemini-2.5-pro 26.1%, and claude-3.7-sonnet 33.6%. This phenomenon injects a large amount of spurious reward, causing RL to converge to policies generating incorrect critiques.

**Key Challenge**: The reliability of outcome supervision depends on solution space size. Math problems have a vast answer space (an answer like "42" almost always requires correct reasoning), while binary preference tasks have only {A, B}, so random guessing yields a 50% hit rate, making the outcome signal very noisy. However, binary tasks cannot be reformulated to expand the solution space as in math.

**Goal**: Without altering the pairwise task structure, provide GRM with a reliable process reward so that critique quality directly enters the training loop, while also addressing the scalability bottleneck of scarce human critique data.

**Key Insight**: Human-provided natural language feedback (critique) for answer pairs is inherently process supervision—the overlap of core arguments between model and human critiques is a direct proxy for critique quality. Additionally, a MetaRM can be trained to generate pseudo-critique data from limited human critiques.

**Core Idea**: Use the "core argument similarity between GRM and human critiques" as a process reward, combined with outcome reward in GRPO. MetaRM extrapolates this reward signal from limited human data to unlabeled data and is updated online during RL training to track policy drift.

## Method

### Overall Architecture
The baseline follows GRPO: query $q$ + candidates $y_A, y_B$ + preference label $l\in\{A,B\}$ → GRM $\pi_\theta$ generates CoT + critique + predicts $\hat l$; for each prompt, rollout $N$ times to obtain outcome rewards $R_{\text{outcome}}^i$, then normalize within the group to get advantage $\hat A_i$. RM-NLHF adds a process reward: (1) when human critique $h$ is available, directly compute the core argument similarity between GRM critique $\hat c$ and $h$; (2) when $h$ is absent, use MetaRM to predict; (3) MetaRM is updated online throughout training to match the current policy output distribution. The final advantage is determined by both outcome and process rewards.

### Key Designs

1. **Core Argument Similarity (Similarity w/ Core HC) as Process Reward**:

    - **Function**: Compresses "whether the GRM critique is reasonable" into a computable scalar reward, avoiding interference from nitpicky critiques.
    - **Mechanism**: An external strong LLM (gemini-2.5-pro) extracts core arguments from both human critique $h$ and GRM critique $\hat c$ (removing trivial nitpicks), then computes F1/Recall/Precision similarity variants. On a 49-sample human-annotated subset, direct LLM-as-Meta-Judge evaluation of $\hat c$ is unstable; All HC similarity is sensitive to nitpicky critiques; Core HC similarity aligns best with human labels. The final process reward $R_{\text{process}}=\text{sim}(\text{core}(h), \text{core}(\hat c))$ is weighted with $R_{\text{outcome}}$ in GRPO's advantage normalization.
    - **Design Motivation**: Direct LLM judgment of critique correctness is affected by judge bias and style; "core argument overlap" retains semantic-level assessment while filtering nitpicky noise, and is quantitatively optimal among proxies. This reward is compatible with RLVR's verifier framework (a scalar reward), requiring no GRPO loss modification.

2. **MetaRM: Predicting Process Reward from Human Critique Data**:

    - **Function**: Addresses the scalability bottleneck of scarce human critique data—most preference datasets (UltraFeedback, HelpSteer series) have only outcome labels, not critiques.
    - **Mechanism**: Trains an auxiliary model MetaRM, inputting $(q, y_A, y_B, \hat c)$ and outputting an estimated process reward for the critique. MetaRM is trained on the subset with human critiques, targeting the "core similarity between $\hat c$ and human $h$"; at inference, it predicts rewards for data without human critiques. Thus, limited human annotation enables process supervision on the full dataset.
    - **Design Motivation**: Human critique annotation is costly (even HelpSteer3 only partially annotated); training only on 50k critique-labeled samples cannot compete with outcome-only RL in scale. MetaRM distills "critique evaluation ability" into a lightweight model, generalizing to large-scale outcome-only data.

3. **Online MetaRM: Reward Model Co-evolving with GRM**:

    - **Function**: Mitigates distribution mismatch in MetaRM evaluation due to policy drift during RL training.
    - **Mechanism**: Alternately updates GRM and MetaRM in the training loop. GRM updates via GRPO → current policy rollouts new prompts to generate $\hat c$ → these $\hat c$ paired with ground-truth $h$ (on critique-labeled subset) supervise MetaRM update → return to GRM. Thus, MetaRM always accurately judges current policy outputs, avoiding reward hacking from static reward models.
    - **Design Motivation**: Classic RLHF suffers from reward model failure after rollout distribution drift; online updates allow MetaRM to track the policy, avoiding Goodhart's Law. The authors find that online MetaRM training approaches the effect of "full human critique supervision" while greatly reducing annotation needs.

### Loss & Training

The foundation is GRPO (Equations 1–3): group-normalized advantage $\hat A_i=(R_i-\bar R)/\sigma$, policy updated with clipped policy gradient + KL regularization. RM-NLHF replaces the reward with $R = R_{\text{outcome}} + \lambda \cdot R_{\text{process}}$, where process reward comes from Core HC similarity or MetaRM prediction. Online MetaRM is supervised with MSE or ranking loss, updated every $k$ GRPO steps. MetaRM and GRM share the backbone but have independent heads (the paper compares with fully independent models, which are feasible but more expensive).

## Key Experimental Results

### Main Results

Benchmarks include HelpSteer3, RewardBench, PandaLM, etc., comparing base GRMs such as the RM-R1 series, Qwen's in-house GRM, and closed-source gemini/claude.

| Training Paradigm | Critique Quality (Core Argument F1) | Outcome Accuracy | Notes |
|-------------------|-------------------------------------|------------------|-------|
| Outcome-only GRPO (SOTA baseline) | Low | High but 20–44% outcome-process inconsistency | Mainstream approach |
| RM-NLHF + Full Human Critique | Highest | Significant improvement | Upper bound |
| RM-NLHF + Offline MetaRM | Close to full human critique | Significantly higher than outcome-only | Annotation efficient |
| **RM-NLHF + Online MetaRM** | Closest to full human critique upper bound | Significantly higher than outcome-only | Practically optimal |

### Ablation Study (Process Reward Selection, 49-sample Human-Annotated Subset)

| Process Reward Scheme | Accuracy vs Human Label |
|----------------------|------------------------|
| LLM-as-a-Meta-Judge (Direct Judgment) | Low |
| Similarity w/ All HC (F1) | Medium |
| Similarity w/ All HC (Recall) | Medium-low |
| Similarity w/ All HC (Precision) | Medium |
| **Similarity w/ Core HC** | Highest |

### Key Findings
- For math tasks, outcome ⇒ process is nearly 100% consistent; for pairwise tasks, even SOTA GRMs show 20–44% outcome-process inconsistency, indicating outcome-only supervision is fundamentally unreliable for binary tasks.
- "Core HC similarity" consistently outperforms "All HC" and "LLM direct judgment"—removing nitpicky critiques is key in process reward design.
- Online MetaRM achieves results close to full human critique supervision with much less annotation; offline MetaRM performs slightly worse due to distribution drift.
- Even if outcome accuracy improves little, critique quality improves significantly → GRM as a reward provider in downstream RLHF benefits more, since downstream policies receive critique signals, not just labels.

## Highlights & Insights
- **Clear diagnosis of outcome-process inconsistency**: Explains why GRMs can "guess" using the solution space size framework—large solution spaces naturally verify process via outcome, while small spaces require explicit process supervision.
- **"Core argument similarity" as process reward**: Avoids over-sensitivity to nitpicky critiques, a key insight for critique-based reward design, directly applicable to LLM judge and QA evaluation.
- **Online MetaRM addresses reward model drift**: Provides a concrete engineering protocol (alternating policy and MetaRM updates) to tackle the classic Goodhart problem in RLHF.
- **Minimal human critique required**: Validates proxy selection with 49 samples + trains MetaRM on a subset, exemplifying cost-efficient alignment.

## Limitations & Future Work
- The assumption "likelihood = correctness" for process rewards does not strictly hold: models highly correlated with human critique style may receive inflated rewards.
- Core HC extraction relies on an external strong LLM (gemini-2.5-pro), introducing extra cost and potential bias; self-distillation into MetaRM may amplify bias.
- Online MetaRM increases training complexity (dual model alternation) and wall-clock cost; no detailed training efficiency analysis is provided.
- Only validated on pairwise rewarding tasks; not tested for solution space bias in listwise or scalar reward tasks.
- Lacks rigorous human evaluation of critique quality compared to verifier-based RL (e.g., updated RM-R1 family).

## Related Work & Insights
- **vs outcome-only GRPO GRM (RM-R1, Wang 2025c)**: The authors use these SOTA models as baselines, quantitatively revealing their critique failure rates and proposing a dual-reward fix.
- **vs PRM (Process Reward Model) in mathematical reasoning**: PRM provides stepwise rewards, this work provides critique-level rewards; both share the central idea that process supervision outperforms pure outcome supervision.
- **vs RLAIF / Constitutional AI**: Uses AI self-evaluation instead of human feedback, but this work first uses human critique as ground truth, then distills into MetaRM, offering stronger interpretability and controllability.
- **Cross-task insights**: Online MetaRM updates can be extended to any scenario where "reward models fail during RL training" (agent reward shaping, code RM, video generation RM).

## Rating
- Novelty: ⭐⭐⭐⭐ The "solution space size determines outcome supervision quality" framework, Core HC similarity, and Online MetaRM are all original, though each component has precedents (PRM, AI feedback, online reward model).
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks, proxy comparisons, and critique quality analysis; lacks human evaluation, and the 49-sample subset is small.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation (Figures 1/2) is intuitive, formulas and contributions are clear; terminology is dense.
- Value: ⭐⭐⭐⭐ Provides the missing process supervision for GRM training; the method can be directly transferred to existing RLHF/RLAIF pipelines, with significant impact on the reward modeling community.

## Related Papers

- [\[ICML 2026\] NAACA: Training-Free NeuroAuditory Attentive Cognitive Architecture with Oscillatory Working Memory for Salience-Driven Attention Gating](naaca_training-free_neuroauditory_attentive_cognitive_architecture_with_oscillat.md)
- [\[ICML 2026\] Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration](polyphonia_zero-shot_timbre_transfer_in_polyphonic_music_with_acoustic-informed_.md)
- [\[ICML 2026\] Probing Cross-modal Information Hubs in Audio-Visual LLMs](probing_cross-modal_information_hubs_in_audio-visual_llms.md)
- [\[ICML 2026\] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](mecat_a_multi-experts_constructed_benchmark_for_fine-grained_audio_understanding.md)
- [\[ICML 2026\] Multimodal Fact-Level Attribution for Verifiable Reasoning](multimodal_fact-level_attribution_for_verifiable_reasoning.md)

</div>

<!-- RELATED:END -->
