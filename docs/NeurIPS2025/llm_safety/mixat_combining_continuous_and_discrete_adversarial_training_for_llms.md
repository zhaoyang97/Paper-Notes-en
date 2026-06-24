---
title: >-
  [Paper Note] MixAT: Combining Continuous and Discrete Adversarial Training for LLMs
description: >-
  [NeurIPS 2025][LLM Safety][Adversarial Training] This paper proposes MixAT, a method that combines discrete adversarial attacks (PAP-based rewriting) with continuous embedding-space perturbations for LLM adversarial training. MixAT achieves robustness against diverse attacks (reducing ALO-ASR from 50%+ to below 20%) while preserving utility, at a training cost comparable to purely continuous methods.
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Adversarial Training"
  - "Robustness"
  - "Jailbreak Attacks"
  - "Continuous Perturbation"
date: 2026-05-08
content_hash: 12aba32555f93d5f
---

# MixAT: Combining Continuous and Discrete Adversarial Training for LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2505.16947](https://arxiv.org/abs/2505.16947)  
**Code**: [GitHub](https://github.com/insait-institute/MixAT)  
**Area**: AI Safety
**Keywords**: Adversarial Training, LLM Safety, Robustness, Jailbreak Attacks, Continuous Perturbation

## TL;DR

This paper proposes MixAT, a method that combines discrete adversarial attacks (PAP-based rewriting) with continuous embedding-space perturbations for LLM adversarial training. MixAT achieves robustness against diverse attacks (reducing ALO-ASR from 50%+ to below 20%) while preserving utility, at a training cost comparable to purely continuous methods.

## Background & Motivation

Despite safety alignment, current adversarial attacks continue to reliably elicit harmful content from frontier LLMs. Adversarial training (AT) is effective for improving robustness in classical ML, but poses unique challenges in the LLM setting:

**The Dilemma of Discrete AT**: Token-level attacks such as GCG effectively produce harmful content but are prohibitively expensive to use during training. The R2D2 method requires 100+ GPU hours to train even a 7B model, as each iteration requires a full adversarial search.

**The Limitation of Continuous AT**: Methods such as CAT apply continuous perturbations in embedding space with high computational efficiency, but the perturbed embeddings do not correspond to any real text sequence. As a result, continuous methods fail to cover all vulnerabilities exploited by discrete attacks — models trained with CAT still exhibit 40% ASR against PAP jailbreaks.

Core Insight: Continuous perturbations construct an $\epsilon$-ball centered on the original benign input, whereas true discrete adversarial examples may lie outside this ball. Shifting the center of continuous perturbations to discrete adversarial examples enables coverage of a broader adversarial embedding region (as illustrated in Fig. 1(a)), while maintaining computational efficiency.

## Method

### Overall Architecture

MixAT defines the adversarial perturbation space as a composition of discrete rewriting and continuous perturbation:

$$\mathcal{N}_{\text{MixAT}}(\mathbf{x}) = \underbrace{\mathcal{R}(\mathbf{x})}_{\text{Discrete Rewriting}} + \underbrace{\mathcal{B}^2(0, \epsilon)}_{\text{Continuous } \epsilon\text{-ball}}$$

Specifically, a harmful prompt is first subjected to discrete rewriting (e.g., paraphrases generated via PAP strategies), and continuous perturbations are then applied on top of the rewritten embeddings. This expands the adversarial region covered during training from "the neighborhood of the benign input" to "the neighborhood of adversarial examples."

### Key Designs

1. **Discrete Seed Generation (PAP-AT)**: A variant of PAP (Persuasion-based Adversarial Prompts) adversarial training is used to generate discrete adversarial seeds. PAP rewrites harmful requests using 40 predefined persuasion strategies, offering low generation cost (API calls only), high diversity, and strong attack effectiveness. New PAP samples are generated dynamically each training round rather than precomputed statically, as ablations show dynamic generation is critical — the static variant raises ALO-ASR from 12.5% to 25%.

2. **Continuous Perturbation Overlay (based on CAT)**: An L2-norm-constrained continuous perturbation $\delta$ is applied to the token embeddings of the discrete seed and optimized via projected gradient descent (PGD). Unlike standard CAT, the perturbation center is the discrete adversarial example rather than the original benign input, pushing the $\epsilon$-ball into a more adversarial embedding region. Empirical analysis (Fig. 6) confirms that PAP + continuous perturbation yields the lowest cosine similarity to the original malicious request while remaining closer to GCG examples, explaining MixAT's generalization to unseen attacks.

3. **Batch-Level Sampling Strategy**: A mixing parameter $\alpha \in [0,1]$ controls the proportion of the two perturbation types. In each training batch, continuous perturbations are overlaid on discrete seeds with probability $P_{C+D} = \alpha$, and applied to the original input with probability $P_C = 1-\alpha$. The default $\alpha=0.5$ serves as the optimal trade-off between direct harmful requests and rewritten attacks.

### Loss & Training

The three-component loss function from Mazeika et al. is adopted:

$$\mathcal{L}_{\text{adv}} = \underbrace{\mathbb{E}[\log P_\theta(\hat{\mathbf{y}}|\hat{\mathbf{x}})]}_{\mathcal{L}_{\text{away}}: \text{reduce harmful response prob.}} \underbrace{- \mathbb{E}[\log P_\theta(\mathbf{y}_s|\hat{\mathbf{x}})]}_{\mathcal{L}_{\text{toward}}: \text{increase safe response prob.}} \underbrace{- \mathbb{E}[\log P_\theta(\mathbf{y}|\mathbf{x})]}_{\mathcal{L}_{\text{util}}: \text{preserve general capability}}$$

where $\hat{\mathbf{x}}$ denotes adversarial input, $\hat{\mathbf{y}}$ denotes harmful response, and $\mathbf{y}_s$ denotes safe response. $\mathcal{L}_{\text{util}}$ uses an additional utility dataset $\mathcal{D}_u$ to prevent catastrophic forgetting. LoRA adapters are employed to reduce memory requirements.

**ALO-ASR Metric**: The paper proposes the "At Least One Attack Success Rate" (ALO-ASR), which measures worst-case success rate across all attack methods under a meta-adversary, providing a more realistic measure of safety risk than single-attack ASR.

## Key Experimental Results

### Main Results (Zephyr-7B)

| Method | ARCe↑ | MMLU↑ | Direct↓ | PAP↓ | GCG↓ | ALO-ASR↓ |
|--------|-------|-------|---------|------|------|----------|
| No Defense | 81.0 | 56.2 | 85.0 | 87.5 | 85.0 | 100.0 |
| R2D2 | 80.1 | 56.1 | 7.5 | 65.0 | 0.0 | 77.5 |
| CAT | 78.2 | 54.8 | 2.5 | 40.0 | 5.0 | 70.0 |
| LAT SFT | 31.7 | 22.9 | 5.0 | 30.0 | 20.0 | 52.5 |
| DualAT | 81.8 | 56.1 | 2.5 | 2.5 | 10.0 | 22.5 |
| **MixAT** | **81.4** | **55.8** | **0.0** | **0.0** | 12.5 | **15.0** |
| MixAT+GCG | 81.6 | 55.9 | 2.5 | 0.0 | 2.5 | 7.5 |

### Cross-Model Generalization

| Model | No Defense ALO↓ | CAT ALO↓ | MixAT ALO↓ | MixAT Utility |
|-------|----------------|---------|-----------|---------------|
| Zephyr-7B | 100.0 | 70.0 | 15.0 | Drop of only 1–2 pts |
| Llama3-8B | 90.0 | 82.5 | 25.0 | Drop of only 0.5–1.5 pts |
| Qwen2.5-14B | 100.0 | 92.5 | 15.0 | Slight improvement |
| Qwen2.5-32B | 100.0 | 82.5 | 7.5 | No significant change |

### Key Findings

- **MixAT vs. DualAT**: Overlaying continuous perturbations on discrete adversarial examples (compositional attack) substantially outperforms training on each attack type separately (DualAT), validating the core hypothesis of shifting the continuous perturbation center to discrete adversarial examples.
- **Dynamic vs. Static Training**: MixAT with dynamically generated PAP samples (ALO 12.5%) far outperforms the static variant (ALO 25%), demonstrating that the model requires continuous exposure to novel adversarial patterns.
- **LoRA Scaling Analysis**: By adjusting the LoRA weight $\lambda \in [0, 1.5]$, the robustness–utility trade-off can be continuously controlled; MixAT outperforms CAT at all values of $\lambda$.
- **Temperature Effects**: Average ASR changes little with increasing temperature, but the probability of producing at least one harmful response rises significantly, indicating that repeated sampling substantially degrades apparent safety.
- **Training Cost**: MixAT incurs less than \$1 in additional API calls compared to CAT, far below the 100+ GPU hours required by R2D2.

## Highlights & Insights

- **Closed Loop of Theoretical Intuition and Empirical Validation**: The approach is motivated by geometric intuition in embedding space, validated through cosine similarity analysis, and empirically confirmed across six attack types.
- **Practical Significance of ALO-ASR**: The metric reveals the fragility of many seemingly robust defenses under combinations of multiple attacks (e.g., R2D2 resists GCG well but is weak against PAP).
- **Auditing Existing Methods**: The paper systematically examines the impact of practical deployment factors — including chat templates, quantization, and non-zero sampling temperature — on defense evaluation, identifying blind spots in community benchmarking.

## Limitations & Future Work

- Robustness against GCG remains a relative weakness (12.5% ASR); incorporating GCG samples reduces this to 2.5% but increases training cost fivefold.
- Only PAP is used as the discrete seed source; more diverse discrete attack methods (e.g., AutoDAN, TAP) may further improve defense coverage.
- Adversarial training may induce over-refusal; although XSTest is evaluated, issues may persist in finer-grained scenarios.
- Model tampering attacks (e.g., weight modification) are not considered, representing a stronger threat model outside the current scope.

## Related Work & Insights

- R2D2 and CAT represent the two extremes of discrete and continuous adversarial training, respectively; MixAT's contribution lies in identifying an effective compositional strategy.
- Constitutional AI and Constitutional Classifiers offer orthogonal defense paths (external filtering vs. intrinsic model robustness).
- This work motivates an important insight: for LLM safety, no single class of defense can cover all attack surfaces, and compositional defense may be a necessary direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of combining continuous and discrete adversarial training is natural yet effective; the core contribution lies in demonstrating that applying continuous perturbations on top of adversarial examples outperforms training on each type separately.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four model scales, seven attack methods, multi-dimensional ablations, and cost analysis — highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Visualizations are excellent, the ALO-ASR metric is well-motivated, and the auditing section merits community attention.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a new strong baseline for LLM adversarial training with direct impact on the safety research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Understanding and Improving Continuous Adversarial Training for LLMs via In-Context Learning Theory](../../ICLR2026/llm_safety/understanding_and_improving_continuous_llm_adversarial_training_via_in-context_l.md)
- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[NeurIPS 2025\] ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training](toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)
- [\[ACL 2025\] TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs](../../ACL2025/llm_safety/tip_iceberg_adversarial_attacks.md)
- [\[ACL 2025\] CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks](../../ACL2025/llm_safety/cavgan_unifying_jailbreak_and_defense_of_llms_via_generative_adversarial_attacks.md)

</div>

<!-- RELATED:END -->
