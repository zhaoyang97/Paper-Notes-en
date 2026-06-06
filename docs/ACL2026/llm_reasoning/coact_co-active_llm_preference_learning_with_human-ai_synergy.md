---
title: >-
  [Paper Note] CoAct: Co-Active LLM Preference Learning with Human-AI Synergy
description: >-
  [ACL 2026][LLM Reasoning][Preference Learning] CoAct utilizes self-consistency to partition unlabeled samples into "high-consistency" and "low-consistency" sets for preference alignment. It employs k-NN distance to ident…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Preference Learning"
  - "Self-Rewarding"
  - "Active Learning"
  - "Human-AI Synergy"
  - "DPO"
date: 2026-05-08
content_hash: 56fae443baecb25e
---

# CoAct: Co-Active LLM Preference Learning with Human-AI Synergy

**Conference**: ACL 2026  
**arXiv**: [2604.17501](https://arxiv.org/abs/2604.17501)  
**Code**: https://github.com/rux001/CoAct (Available)  
**Area**: LLM Alignment / RLHF  
**Keywords**: Preference Learning, Self-Rewarding, Active Learning, Human-AI Synergy, DPO

## TL;DR
CoAct utilizes self-consistency to partition unlabeled samples into "high-consistency" and "low-consistency" sets for preference alignment. It employs k-NN distance to identify "self-consistent but potentially incorrect" samples from the high-consistency set to be sent to an Oracle for labeling, while the remaining high-consistency samples are treated as AI self-labeled data. Finally, oracle-verified samples serve as in-context demos to generate new instructions, unifying human and AI supervision within a single DPO cycle. CoAct outperforms strong baselines by 4–8 points on GSM8K, MATH, and WebInstruct.

## Background & Motivation

**Background**: Preference alignment (RLHF / DPO and its variants) has become the mainstream method for aligning LLMs with human preferences. However, the high-quality preference data required for training has long relied on human annotation, which is difficult to scale. Research has split into two paths: (1) Self-Rewarding / RLAIF—letting the model act as a judge to generate preference pairs; (2) Active Preference Learning (APO, ActiveDPO, etc.)—selecting the most informative samples for human annotation under a limited budget.

**Limitations of Prior Work**: Both paths have significant bottlenecks. Self-Rewarding treats the model as the judge without external verification, which easily amplifies internal deviations into self-bias. Conversely, Active Learning tokens rely on an Oracle for quality, but tight budgets often result in a large number of wasted samples in the unlabeled pool, leading to low data utilization.

**Key Challenge**: Under a fixed human annotation budget, there is a trade-off between "Data Scale (Self-Rewarding)" and "Data Quality (Active Learning)," and prior methods have failed to optimize both simultaneously.

**Goal**: (1) Utilize "safe samples" from the unlabeled pool without increasing the oracle budget; (2) identify "self-consistent but incorrect" risks within high-consistency samples to avoid self-bias amplification; (3) leverage oracle feedback not just for labels, but to guide the generation of new instructions, thereby increasing data diversity.

**Key Insight**: The authors start from a simple observation: "It is easy for a model to occasionally answer incorrectly, but it is much harder to consistently provide the same incorrect answer across multiple independent samplings." Thus, self-consistency can serve as both a reliability proxy for AI self-labeling and a means to distinguish between samples where the model is "confident" versus "uncertain."

**Core Idea**: Integrates self-rewarding and active learning into a closed loop using self-consistency as a signal. High-consistency samples are self-labeled and used by the AI, while low-consistency samples and high-consistency risks detected by k-NN are sent to the Oracle. The oracle-verified samples are then used to augment new instructions, and all data are co-trained using a modified DPO objective.

## Method

### Overall Architecture
CoAct is an iterative "human-AI synergistic preference learning" framework. In each round $t$, the input consists of the current model $\theta_t$ and a batch $\mathcal{B}_t$ from an unlabeled instruction pool, and the output is the updated model $\theta_{t+1}$. The process follows three steps:

1.  **Self-consistent Preference Pair Construction**: For each instruction $x$, $k=8$ responses are generated via temperature sampling. A consistency score $C(y)$ is calculated based on the relative frequency of the extracted answers. The most consistent response is denoted as $y^+$ and the least consistent as $y^-$. Samples are partitioned into high-consistency $\mathcal{D}_{high}$ and low-consistency $\mathcal{D}_{low}$ using a threshold $\tau$ (4/8 for GSM8K/MATH, 5/8 for WebInstruct).
2.  **Oracle Budget Allocation + k-NN Risk Detection**: With a fixed budget $M=300$, samples are divided as $M_{low}=M_{high}=150$. For $\mathcal{D}_{low}$, samples with the lowest $C(y^+)$ are selected. For $\mathcal{D}_{high}$, k-NN distance in the hidden state space of historical oracle-verified correct samples is used to identify samples furthest from the in-distribution set (viewed as "self-consistent but OOD errors"). These are merged and sent to the Oracle.
3.  **Oracle-guided Instruction Augmentation + Joint Training**: Samples from $\mathcal{D}_{high}$ verified as correct by the Oracle serve as ICL demos to generate $N_{new}$ new instructions. These follow the same self-consistency flow to construct new preference pairs. Finally, oracle-labeled and AI-labeled data are combined into $\mathcal{D}_{final}^{(t)}$ to update the model using a DPO objective with NLL regularization.

### Key Designs

1.  **Self-consistency based Preference Pair Construction and Partitioning**:
    - **Function**: Uses a single "consistency score" to both construct preference pairs and assign "reliability labels" to samples.
    - **Mechanism**: For each $x$, $k$ responses are sampled, and the score is calculated as $C(y)=\frac{1}{k}\sum_m \mathbf{1}\{\text{ans}(y_m)=\text{ans}(y)\}$. The most consistent is the chosen response, and the least consistent is rejected. The batch is split by threshold $\tau$: $C(y^+)\geq\tau$ for $\mathcal{D}_{high}$ (self-label candidates) and $C(y^+)<\tau$ for $\mathcal{D}_{low}$ (sent to Oracle).
    - **Design Motivation**: Avoids the high cost and bias of LLM-as-Judge methods by using a cheap statistical proxy, while unifying the conflicting goals of "AI self-labeling" and "identifying what to label" under one signal.

2.  **k-NN Distance-driven High-consistency Risk Detection**:
    - **Function**: Identifies "self-consistent but actually incorrect" samples from the "seemingly credible" high-consistency set to prevent self-bias amplification.
    - **Mechanism**: The penultimate hidden states $\phi(x)/\|\phi(x)\|_2$ of oracle-verified correct samples from the previous round are treated as the in-distribution (ID) set $\mathcal{Z}_{ID}^{(t)}$. For current high-consistency samples, the distance $r_k(z_i)=\min_{z\in\mathcal{Z}_{ID}^{(t)}}\|z_i-z\|_2$ is computed. The Top-$M_{high}$ samples with the largest distances are regarded as OOD and sent to the Oracle. This adapts classic OOD detection (Sun et al. 2022) to preference learning.
    - **Design Motivation**: Self-consistency alone may miss cases where the model is "confidently wrong." k-NN adds a layer of verification based on whether the sample falls within the known correct distribution. In experiments, this increased the oracle error hit rate to 82.56% / 87.32% / 84.17%, significantly higher than random selection (~70%).

3.  **Oracle-guided Instruction Augmentation + Modified DPO Objective**:
    - **Function**: Uses oracle feedback as a "difficulty filter" to guide the generation of new instructions within the model's solvable range, while using NLL regularization to stabilize the likelihood of chosen sequences.
    - **Mechanism**: Oracle-verified correct samples $\mathcal{D}_{correct}^{(t)}$ are used as ICL demos to generate $N_{new}$ new instructions $\mathcal{D}_{new}^{(t)}$. The optimization objective adds an NLL term to DPO: $\mathcal{L}(\theta_t)=-\mathbb{E}[\log\sigma(\beta\log\frac{\theta_t(y^+|x)}{\theta_0(y^+|x)}-\beta\log\frac{\theta_t(y^-|x)}{\theta_0(y^-|x)})-\alpha|y^+|\log\theta_t(y^+|x)]$, with $\beta=0.5, \alpha=1$.
    - **Design Motivation**: Synthetic data often exceeds the base model's solvable range, causing noise in self-labels. Using oracle-verified samples as anchors keeps the "difficulty ceiling" within reachable limits, while the NLL term prevents the collapse of chosen log-probs during iterations.

### Loss & Training
The final training data $\mathcal{D}_{final}^{(t)}=\mathcal{D}_{oracle}^{(t)}\cup\mathcal{D}_{AI}^{(t)}$ is used to train for 10 epochs with the modified DPO loss. Hyperparameters include a learning rate of $5\times10^{-6}$, batch size of 16, and LoRA rank 8 / alpha 16. Sampling temperatures are drawn from $\{0.35, 0.4, \ldots, 0.7\}$ to encourage reasoning diversity. Four active learning iterations are conducted, with a fixed oracle budget of 300 per round ($M_{low}=M_{high}=150$).

## Key Experimental Results

### Main Results
Two base models (Llama3-8B and Qwen3-4B) were evaluated on three reasoning benchmarks (GSM8K, MATH, WebInstruct), comparing against four active learning baselines (Random, Entropy, Pref Certainty, Pref+Ent). Accuracy (%) at the 4th iteration:

| Dataset (Base) | Random | Entropy | Pref Cert. | Pref+Ent | **CoAct** | Gain vs Base |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| GSM8K (Llama3-8B) | 34.57 | 36.56 | 37.28 | 39.41 | **43.58** | +20.05 |
| MATH (Llama3-8B) | 12.07 | 12.94 | 11.08 | 13.07 | **14.46** | +9.84 |
| WebInstruct (Llama3-8B) | 9.62 | 10.11 | 14.53 | 14.87 | **15.97** | +8.28 |
| GSM8K (Qwen3-4B) | 93.57 | 94.02 | 94.03 | 94.58 | **94.84** | +6.45 |
| MATH (Qwen3-4B) | 70.89 | 70.14 | 70.52 | 70.21 | **75.71** | +6.54 |
| WebInstruct (Qwen3-4B) | 44.12 | 47.83 | 51.25 | 52.48 | **53.96** | +18.04 |

The average gain across four iterations was +13.25% for GSM8K, +8.19% for MATH, and +13.16% for WebInstruct. Notably, on Qwen3-4B for MATH, CoAct improved performance from ~70% to 75.71%.

### Ablation Study

| Config (Llama3-8B, iter 4) | GSM8K | MATH | WebInstruct | Description |
|:---|:---:|:---:|:---:|:---|
| Oracle-only | 36.38 | 11.95 | 13.53 | Oracle data from active selection only |
| +Aug (No Self-label) | 39.85 | 13.34 | 13.89 | Adds oracle-guided instruction augmentation |
| +Self (No Aug) | 41.23 | 12.12 | 15.21 | Adds self-consistent self-labeled data |
| **Full (Aug + Self)** | **43.58** | **14.46** | **15.97** | Complete CoAct framework |

The oracle error hit rate for the k-NN strategy reached 82.56% (GSM8K) and 87.32% (MATH), significantly outperforming random and "lowest consistency" baselines (~70%). The consistency threshold $\tau$ was optimal between 4/8 and 5/8; lower values introduced noise, while higher values lacked sufficient training data.

### Key Findings
- **k-NN risk detection is vital for high-consistency samples**: Removing it dropped the oracle's "true error" hit rate from ~84% to ~70%, increasing the risk of self-bias amplification.
- **Diminishing returns for strong bases**: The gap between baselines on GSM8K was only 1.27% for Qwen3-4B, but 9.01% for Llama3-8B, suggesting active learning is most valuable when the base model is not yet saturated.
- **Strong correlation between self-consistency and accuracy**: After 4 iterations, Pearson correlation coefficients reached 0.9745 (GSM8K) and 0.9756 (MATH), proving that $C(y)$ serves as an increasingly reliable proxy for correctness.
- **OOD Generalization**: CoAct maintained a lead on GPQA and MMLU-Pro, indicating that oracle-guided instruction augmentation provides genuine diversity benefits rather than just in-domain overfitting.

## Highlights & Insights
- **Dual-purpose signal**: Self-consistency simultaneously selects chosen/rejected pairs and determines which samples are sent to the oracle, unifying AI self-labeling and active learning pipelines.
- **Elegant migration of k-NN**: Adapting classic OOD detection to solve "high-confidence errors" provides a plug-and-play solution to the primary weakness of self-rewarding methods.
- **Expanded Oracle role**: The authors decompose the oracle's role into "verify pair" and "anchor instruction generation." The latter proved highly effective on MATH, suggesting that controlling the difficulty ceiling is more valuable than simply increasing data volume when the base model is weak.
- **GPT-5 as a cost-effective Oracle**: Performance using GPT-5 was comparable to human annotation, providing empirical support for using strong LLMs as oracles to reduce costs.

## Limitations & Future Work
- Sampling $k=8$ responses for each instruction incurs inference overhead significantly higher than baselines (~8x); more efficient uncertainty estimation is needed for scaling.
- Evaluation is limited to objective reasoning tasks; it is unclear if the self-consistency threshold mechanism holds for open-ended generation (e.g., creative writing, dialogue).
- The consistency threshold $\tau$ still requires manual tuning per dataset (4/8 for GSM8K vs. 5/8 for WebInstruct); cross-task adaptation remains unsolved.
- Since k-NN uses oracle-verified samples from the previous round, the initial ID set in the first iteration is small, leading to weaker detection capability.
- Future directions: Replace k-NN with learnable OOD scoring; extend self-consistency to reward model scoring; explore multi-LLM collaborative oracles.

## Related Work & Insights
- **vs. Self-Rewarding (Yuan et al. 2024)**: Self-rewarding lacks external verification and risks bias reinforcement; CoAct uses self-consistency and oracle cleaning to mitigate this at the cost of oracle calls.
- **vs. Active Preference Learning (Lin et al. 2026)**: APL wastes major portions of the unlabeled pool; CoAct utilizes high-consistency samples for self-labeling, multiplying the effective training set size for the same budget.
- **vs. SCPO (Prasad et al. 2025)**: SCPO uses self-consistency for construction but lacks a human-AI synergy path to handle self-consistent errors; CoAct adds oracle + k-NN as "quality insurance."

## Rating
- Novelty: ⭐⭐⭐⭐ Skillful fusion of self-rewarding and active learning, though individual components draw on existing concepts.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple bases, in-domain/OOD benchmarks, ablations, and sensitivity analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation for the self-consistency + k-NN combination; well-structured equations and algorithms.
- Value: ⭐⭐⭐⭐ Provides a practical recipe to reduce annotation costs in the data-bottlenecked era of preference alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HeurekaBench: A Benchmarking Framework for AI Co-scientist](../../ICLR2026/llm_reasoning/heurekabench_a_benchmarking_framework_for_ai_co-scientist.md)
- [\[ACL 2026\] Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data](budget-aware_anytime_reasoning_with_llm-synthesized_preference_data.md)
- [\[ACL 2026\] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning](templaterl_structured_template-guided_reinforcement_learning_for_llm_reasoning.md)
- [\[ACL 2026\] Decoupling the Effect of Chain-of-Thought Reasoning: A Human Label Variation Perspective](decoupling_the_effect_of_chain-of-thought_reasoning_a_human_label_variation_pers.md)
- [\[ACL 2026\] AIM-CoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](aim-cot_active_information-driven_multimodal_chain-of-thought_for_vision-languag.md)

</div>

<!-- RELATED:END -->
