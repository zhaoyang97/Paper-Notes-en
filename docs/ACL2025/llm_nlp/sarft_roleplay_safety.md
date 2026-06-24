---
title: >-
  [Paper Note] Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs
description: >-
  [ACL 2025][LLM (Other)][Role-play safety] This paper conducts the first systematic evaluation of the impact of role-play fine-tuning on the safety of LLMs, finding that the level of safety degradation is positively correlated with role traits (especially villainous roles). It proposes the SaRFT framework, which adaptively identifies subsets of harmful training data for different roles using an implicit reward function, and combines this with KL-divergence regularization to ac…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Role-play safety"
  - "fine-tuning safety degradation"
  - "implicit reward function"
  - "role-adaptive data selection"
  - "Safety-Aware Fine-Tuning"
date: 2026-05-08
content_hash: 8f5e2059fe60d2d4
---

# Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs

**Conference**: ACL 2025  
**arXiv**: [2502.20968](https://arxiv.org/abs/2502.20968)  
**Code**: [https://github.com/yulinlp/SaRFT](https://github.com/yulinlp/SaRFT)  
**Area**: LLM/NLP  
**Keywords**: Role-play safety, fine-tuning safety degradation, implicit reward function, role-adaptive data selection, Safety-Aware Fine-Tuning

## TL;DR
This paper conducts the first systematic evaluation of the impact of role-play fine-tuning on the safety of LLMs, finding that the level of safety degradation is positively correlated with role traits (especially villainous roles). It proposes the SaRFT framework, which adaptively identifies subsets of harmful training data for different roles using an implicit reward function, and combines this with KL-divergence regularization to achieve a Pareto-optimal balance between role expressiveness and safety.

## Background & Motivation

**Background**: Role-playing is a popular application scenario for LLMs (e.g., Character.ai), where fine-tuning enables LLMs to master the dialogue style and knowledge of specific characters. Current role-playing enhancement methods are mainly divided into training-free (in-context learning) and SFT-based fine-tuning.

**Limitations of Prior Work**:
   - Role-play fine-tuning significantly compromises the safety performance of LLMs—experiments show that after fine-tuning on 95 roles, the refusal rate on AdvBench drops from 98.46% to 74.78% (a 24% decrease).
   - The degree of safety degradation varies across roles, with villainous characters (e.g., Freddy Krueger) suffering the most, yet existing safety alignment methods do not consider role specificity.
   - A real-world tragedy has drawn attention: The New York Times reported that a 14-year-old boy committed suicide after becoming obsessed with an AI virtual character.

**Key Challenge**: Existing safety-preservation methods (e.g., data selection like SEAL, regularization like SPPFT) are general-purpose solutions that fail to dynamically adjust safety policies based on the risk levels of different characters. Villainous roles require stronger safety constraints, whereas benevolent roles need fewer constraints to maintain role expressiveness.

**Goal**
   - How to quantify the impact of different role traits on safety degradation.
   - How to adaptively identify the "harmful" portions of training data based on role traits.
   - How to simultaneously enhance role-playing capability and preserve safety during fine-tuning.

**Key Insight**: Leveraging the implicit reward function $r(x,y) \propto \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$ from the DPO framework to treat role-playing and safety as two dimensions under the same alignment framework. By comparing data scores in these two dimensions, the harmfulness of training samples to specific roles can be evaluated.

**Core Idea**: Using implicit reward functions to score training data across two dimensions: role-playing and safety. Samples whose safety risk score exceeds the role-playing score are flagged as "harmful", and KL regularization is imposed on them to anchor the original safety distribution.

## Method

### Overall Architecture
SaRFT consists of two stages: **RDS (Role-Safety Adaptive Data Selection)** — adaptively identifying a subset of "harmful" training data for each role $\rightarrow$ **RBO (Role-Safety Balance Optimization)** — performing CE loss on the full dataset to learn role-playing + performing KL loss on the harmful subset to maintain safety $\rightarrow$ simultaneously outputting a model that is both safe and highly expression-capable.

### Key Designs

1. **Role-Safety Adaptive Data Selection (RDS)**:

    - **Function**: Customizes a "harmful" data subset for each role, yielding larger harmful subsets for villainous characters.
    - **Mechanism**:
        - Construct three models: $\pi_{\text{role}}$ (role background in system prompt), $\pi_{\text{unsafe}}$ (role background + unsafe instructions in system prompt), and $\pi_{\text{ref}}$ (original model with no system prompt).
        - For each training sample $(x,y)$, calculate two scores:
       - Role-play score: $s_{\text{role}} = \log \frac{\pi_{\text{role}}(y|x)}{\pi_{\text{ref}}(y|x)}$
       - Safety risk score: $s_{\text{unsafe}} = \log \frac{\pi_{\text{unsafe}}(y|x)}{\pi_{\text{ref}}(y|x)}$
        - When $s_{\text{unsafe}} > s_{\text{role}}$, the sample is flagged as "harmful" and added to $\mathcal{D}_h$.
    - **Design Motivation**: Leverages the concept of implicit reward functions from DPO theory, where the probability ratio between the aligned model and reference model acts as a reward signal. By "simulating" the aligned state via system prompts, there is no need to actually train a reward model. This represents a zero-cost data scoring method.
    - **Adaptive Characteristics**: Villainous characters (e.g., Freddy Krueger) yield more samples marked as harmful (40.38%) in response to $\pi_{\text{unsafe}}$, whereas positive characters (e.g., Stephen Hawking) only have 19.67%.

2. **Role-Safety Balance Optimization (RBO)**:

    - **Function**: Simultaneously optimizes role-playing and safety using a dual-objective loss function.
    - **Mechanism**:
        - Role-play loss: Standard cross-entropy $L_{\text{CE}}$ on the full dataset $\mathcal{D}$.
        - Safety preservation loss: KL-divergence $L_{\text{KL}} = \text{KL}(p_\theta \| p_{\text{ref}})$ on the harmful subset $\mathcal{D}_h$, constraining the model's output distribution on harmful data from deviating from the original safety distribution.
        - Total loss: $L = L_{\text{CE}} + \lambda L_{\text{KL}}$.
    - **Design Motivation**: Imposing KL constraints on harmful data acts as a "local anchor", limiting updates only on data that may cause safety degradation, without affecting role learning on other data.

3. **Correlation Between Role Traits and Safety Risks**:

    - **Function**: Reveals patterns of safety degradation among villainous characters.
    - **Findings**: Villainous characters like Freddy Krueger (23.50% R.R.) and Gaston (44.70% R.R.) suffer indeed severe safety degradation, while Stephen Hawking (88.90%) and Queen Catherine (94.00%) maintain good safety profiles.
    - **Data Inspection Insights**: The "harmful" data of positive characters lack role-specific cues, while the "harmful" data of negative characters often exhibit aggressive tones and exaggerated styles.

### Loss & Training
- $L = L_{\text{CE}} + \lambda L_{\text{KL}}$, where $\lambda$ is a hyperparameter.
- Supports LoRA and full-parameter fine-tuning.
- Training: 1 epoch, batch size 32, max source length 512, max target length 128.
- Hardware: 4×A100 GPUs, DeepSpeed ZeRO-2.

## Key Experimental Results

### Main Results (Full-Parameter Fine-Tuning, LLaMA-3-8B-Instruct, Average of 10 Roles)

| Method | RoleBench AVG ↑ | AdvBench R.R. ↑ | BeaverTails R.R. ↑ | Safety AVG ↑ | Jailbreak AVG ↑ |
|------|-----------------|-----------------|---------------------|-------------|-----------------|
| Base Model | 21.50 | 98.46 | 91.40 | 95.06 | 78.80 |
| SFT | 26.62 | 76.40 | 69.31 | 72.97 | 46.10 |
| SEAL | 26.91 | 76.63 | 74.08 | 73.83 | 31.84 |
| SPPFT | 27.09 | 81.98 | 75.50 | 78.13 | 49.50 |
| **SaRFT** | **26.91** | **92.50** | **83.06** | **87.08** | **62.48** |

SaRFT achieves an average safety performance of 87.08%, significantly outperforming the second-best, SPPFT (78.13%), while keeping its role-playing score on par with the best results.

### Ablation Study

| Data Selection Method | RoleBench AVG | Safety AVG | Jailbreak AVG |
|-------------|---------------|-----------|---------------|
| Random | 25.64 | 82.86 | 46.42 |
| FLIP (Inverse Selection) | 25.81 | 82.26 | 49.38 |
| SEAL | 26.54 | 81.22 | 58.40 |
| Bi-Selection | 26.41 | 82.33 | 59.10 |
| **SaRFT (RDS)** | **26.91** | **87.08** | **62.48** |

### Key Findings
- **Role traits key to safety risks**: The degree of safety degradation in villainous characters is 3-4 times that of positive characters, which is statistically validated across 95 roles.
- **Role-adaptivity of RDS is effective**: Random and FLIP substitution experiments demonstrate that the "harmful" subset selected by RDS successfully captures role-specific safety risks.
- Data inspection reveals that training data for positive characters is naturally aligned with safety (role traits align with safety), whereas villainous characters require extra protection.
- SaRFT is consistently effective across three base models (LLaMA-3, Qwen2.5, Gemma-2), proving its generalizability.
- Interesting discovery: Fine-tuning for role-play actually improves resistance to certain jailbreak attacks (e.g., Cipher, CodeChameleon), because distribution shift weakens the model's ability to comprehend encrypted text.

## Highlights & Insights
- **Creative application of implicit reward functions**: Evaluates role-playing and safety scores without training any reward models, simply simulating an aligned model via system prompts. This concept of "system prompt as alignment" is highly lightweight and generalizable, and can be ported to other fine-tuning scenarios requiring multi-objective balancing.
- **Role-adaptive safety protection** is the core innovation: Safety is not one-size-fits-all, and different characters require different levels of protection. This concept can be extended to other SFT tasks like style transfer and domain adaptation.
- **First large-scale evaluation of 95 roles** in this field, providing quantitative evidence regarding the connection between character traits and safety degradation.

## Limitations & Future Work
- Experiments only cover 7B-9B models, without verifying effectiveness on larger models (e.g., 70B).
- Safety evaluation heavily relies on GPT-4o to judge refusal rates, which may introduce evaluation bias.
- The two system prompts for RDS (role prompt and unsafe prompt) must be manually designed and lack automation.
- The hyperparameter $\lambda$ requires tuning, and its sensitivity is not fully analyzed.
- Does not explore how different fine-grained safety dimensions (e.g., toxic content vs. privacy leakage vs. bias) respond differently to character traits.

## Related Work & Insights
- **vs SEAL/Bi-Selection**: Generic data selection methods do not consider role specificity. SaRFT's role-adaptive selection outperforms them by approximately 5% in safety.
- **vs ROSE (decoding method)**: ROSE restores the base model's safety level, but role-playing enhancement is only 50% of standard SFT. SaRFT mitigates the safety risk during training without sacrificing role-play performance.
- **vs Vaccine (regularization method)**: Vaccine safety actually drops to 71.64% on Gemma-2, indicating that generic regularization can sometimes backfire.

## Rating
- Novelty: ⭐⭐⭐⭐ Role-adaptive safety is a fresh perspective, and the usage of implicit reward functions is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 95 character evaluations, 3 base models, multiple safety benchmarks, 5 jailbreak attacks, and a thorough ablation.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured, vivid case studies (with the "Po" example), and compelling motivation.
- Value: ⭐⭐⭐⭐ Provides the first systematic analytical framework and a practical solution for the safety issues of role-play AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs](../../NeurIPS2025/llm_nlp/triplets_better_than_pairs_towards_stable_and_effective_self-play_fine-tuning_fo.md)
- [\[ACL 2025\] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](gorp_continual_gradient_projection.md)
- [\[ICML 2025\] Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets](../../ICML2025/llm_nlp/safe_delta_consistently_preserving_safety_when_fine-tuning_llms_on_diverse_datas.md)
- [\[ACL 2025\] Are Your LLMs Capable of Stable Reasoning?](are_your_llms_capable_of_stable_reasoning.md)
- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
