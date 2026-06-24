---
title: >-
  [Paper Note] Aligning to What? Limits to RLHF Based Alignment
description: >-
  [ACL 2025][LLM Alignment][RLHF] Through systematic experiments, this paper finds that RLHF (including DPO, ORPO, RLOO, etc.) is fundamentally ineffective at reducing covert racial bias in LLMs. Furthermore, executing SFT prior to RLHF "solidifies" model biases, revealing the deep limitations of current alignment techniques when dealing with ambiguous goals such as bias elimination.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "RLHF"
  - "covert bias"
  - "DPO"
  - "ORPO"
  - "dialect bias"
  - "alignment limits"
date: 2026-05-08
content_hash: 30033e866f5b32c7
---

# Aligning to What? Limits to RLHF Based Alignment

**Conference**: ACL 2025  
**arXiv**: [2503.09025](https://arxiv.org/abs/2503.09025)  
**Code**: None  
**Area**: RLHF Alignment  
**Keywords**: RLHF, covert bias, DPO, ORPO, dialect bias, alignment limits

## TL;DR
Through systematic experiments, this paper finds that RLHF (including DPO, ORPO, RLOO, etc.) is fundamentally ineffective at reducing covert racial bias in LLMs. Furthermore, executing SFT prior to RLHF "solidifies" model biases, revealing the deep limitations of current alignment techniques when dealing with ambiguous goals such as bias elimination.

## Background & Motivation

**Background**: RLHF has become the standard paradigm for aligning large models to conform with "helpful, harmless, and honest" human preferences. Mainstream approaches include online RL methods (PPO, RLOO) and reward-free methods (DPO, ORPO). However, the effectiveness of RLHF in addressing more subtle, ambiguous alignment targets like bias has not been systemically studied.

**Limitations of Prior Work**: Research by Hofmann et al. (2024) found that off-the-shelf LLMs trained with RLHF actually harbor the strongest covert biases. However, no prior study has systematically examined whether the RLHF training process itself exacerbates or alleviates these biases. Preference datasets themselves also suffer from quality issues, as annotators do not always agree on what constitutes "harmlessness".

**Key Challenge**: RLHF excels at optimizing explicit, measurable objectives (such as answer length and formatting compliance), but "bias mitigation" is an ambiguous objective that is difficult to precisely encode in preference data. Existing preference data primarily focuses on response quality rather than fairness.

**Goal**: To systematically analyze the impact of various RLHF techniques (DPO, ORPO, RLOO), base models (Llama 3, Mistral), and preference datasets on the covert and overt biases of LLMs.

**Key Insight**: Utilize matched-guise probing from sociolinguistics to quantify covert bias by comparing discrepancies in model responses to African American English (AAE) and Standard American English (SAE) texts.

**Core Idea**: Systematically evaluate the impact of RLHF on inner model attitudes using rigorous bias metrics, demonstrating that current alignment techniques cannot effectively handle ambiguous goals like bias elimination.

## Method

### Overall Architecture
Using Llama 3 8B as the primary subject, the study applies three RLHF post-training techniques (DPO, ORPO, RLOO) and measures changes in covert and overt biases before and after training using matched-guise probing. Generalizability is verified on Mistral 7B, and the impact of different preference datasets and training epochs is explored. Finally, the bias measurement methodology is extended to multimodal models (Llama 3.2 Vision 11B).

### Key Designs

1. **Matched-Guise Probing**:

    - **Function**: Quantify covert and overt attitudes of LLMs toward different dialect groups.
    - **Mechanism**: Given AAE and SAE text sets, calculate the differences in conditional probabilities for various personality traits (e.g., "intelligent", "lazy"). The association score $q(t;\theta) = \frac{1}{|X|}\sum_i \log\frac{p(t|f(x_i);\theta)}{p(t|f(y_i);\theta)}$ reflects how strongly trait $t$ is associated with AAE texts. Positive values indicate a stronger association with AAE, while negative values indicate a stronger association with SAE. Personality traits and occupational prestige ratings from the Princeton trilogy studies are used to evaluate if associations constitute bias.
    - **Design Motivation**: This approach reveals "true attitudes" of the model better than direct testing (e.g., "What do you think of group XX?") because covert testing bypasses the model's safety alignment.

2. **Multidimensional RLHF Experimental Design**:

    - **Function**: Systematically evaluate the impact of various variables on bias.
    - **Mechanism**: The controlled experiment matrix includes: (1) RLHF method (DPO/ORPO/RLOO); (2) Base model (Llama 3/Mistral); (3) Dataset (Anthropic HH-RLHF/PKU-SafeRLHF/OLMo preference data); (4) Training strategy (with or without prior SFT, training for 1 vs. 3 epochs); (5) Dialect exposure (training using AAE-translated preference data). All models are fine-tuned using LoRA (rank=16).
    - **Design Motivation**: To isolate single variables and comprehensively understand the relation between RLHF and bias.

3. **Multimodal Bias Extension**:

    - **Function**: Extend bias measurement from text-only models to vision-language models.
    - **Mechanism**: For Llama 3.2 Vision 11B, covert bias is still measured via textual input. Overt bias is measured by providing images of individuals of different races as visual inputs (replacing racial designation words in text), using face images from the UTKFace dataset. Visual inputs offer richer racial cues and reduce the high variance caused by a small set of textual indicators.
    - **Design Motivation**: To broaden the scope of bias measurement, leveraging VLM image inputs to provide a more stable measures of overt bias.

### Loss & Training
Standard implementations are used for each RLHF method: DPO uses the standard preference pair loss; RLOO uses NCSOFT OffsetBias-RM as the reward model (ranking in the top 10 of the Hugging Face reward model leaderboard during training); ORPO uses the Odds Ratio objective. The SFT stage is trained on 100k samples from SlimOrca.

## Key Experimental Results

### Main Results

| Model Config | Covert Trait Bias Change (Mean) | Covert Occup. Bias Change (Mean) | Overt Trait Bias Change (Mean) | Description |
|---------|---------------------|---------------------|---------------------|------|
| Llama 3 (base) | - | - | - | Baseline |
| +DPO | 0.175 | -0.022 | -0.365 | Covert bias remains mostly unchanged |
| +ORPO | -0.026 | 0.151 | 0.076 | Overt bias slightly increased |
| +RLOO | 0.003 | 0.135 | -0.177 | Change is not significant |
| +SFT+DPO | Significantly lower variance | Significantly lower variance | Significantly lower variance | SFT solidifies bias |
| Mistral+DPO | 0.044 | 0.097 | -0.116 | Smaller change than Llama |

### Model Reward Comparison

| Model Config | ArmoRM Reward | OffsetBias Reward | Description |
|---------|-----------|---------------|------|
| Llama 3 base | 0.062 | -6.837 | Baseline |
| +DPO | 0.071 | -6.324 | DPO performs best |
| +ORPO | 0.062 | -7.004 | Almost no improvement |
| +RLOO | 0.064 | -7.098 | Almost no improvement |
| Llama 3 Instruct| 0.095 | -4.742 | Large-scale post-training shows obvious effects |

### Key Findings
- **RLHF is largely ineffective against covert bias**: All RLHF methods (DPO, ORPO, RLOO) fail to significantly alter the model's covert bias patterns across all experimental setups. The parabolic trend of covert bias (extreme positive and negative traits being associated more with AAE) remains unchanged post-training.
- **SFT solidifies bias**: Performing SFT before DPO reduces the ability of subsequent RLHF to alter bias. Table 1 shows that the variance of association score changes under the L3+SFT setup is significantly lower than without SFT.
- **Different base models exhibit different baseline biases**: Mistral's bias is more resistant to RLHF alteration (lower variance of change) compared to Llama 3, suggesting that the model architecture/pre-training data also affects the plasticity of bias.
- **AAE preference data has a marginal effect**: Training with AAE-translated data slightly shifts association scores toward AAE, but the effect is weak.
- **Covert and overt biases in VLMs can be contradictory**: Llama 3.2 Vision's covert bias associates extreme traits with AAE, whereas its overt bias (image input) associates the same traits with White individuals.

## Highlights & Insights
- The experimental design is highly systematic, covering nearly all major variables of RLHF (method, model, data, training strategy), making the conclusions highly credible. This controlled-variable experimental paradigm can serve as a methodological template for future alignment research.
- The discovery that "SFT solidifies bias" is extremely important—implying that the current standard "SFT $\to$ RLHF" training pipeline may lock in certain undesirable properties during the SFT stage.
- The finding that covert and overt biases can be completely opposite in VLMs reveals an entirely new challenge dimension in multimodal alignment.

## Limitations & Future Work
- Due to resource constraints, only LoRA fine-tuning was used instead of full-parameter training. Whether full-parameter training can alter bias more effectively remains to be verified.
- The bias measurement focusing on African American vs. White binary comparisons does not cover other racial groups or other types of bias.
- The AAE/SAE texts used to evaluate bias are sourced from social media, which may not represent real-world user interactions with LLMs.
- Future work needs to develop preference datasets and alignment methods specifically targeted at bias elimination, rather than relying on generic safety data.

## Related Work & Insights
- **vs Hofmann et al. (2024)**: Hofmann found that off-the-shelf RLHF models have the strongest covert bias. This paper further demonstrates that the RLHF training process itself cannot improve this issue.
- **vs Constitutional AI**: Claude's CAI method sets explicit rules and constraints, which may be more suitable than RLHF for handling objectives like bias.
- **vs D'Oosterlinck et al. (2024)**: Points out that a lack of distinct contrast in preference data contributes to poor RLHF performance, which is consistent with the findings in this work.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic analysis of the relation between RLHF and covert bias.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive experimental matrix with rigorous variable controls.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of interdisciplinary work.
- Value: ⭐⭐⭐⭐⭐ Highly significant for understanding the limitations of RLHF and driving better alignment methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Synergistic Weak-Strong Collaboration by Aligning Preferences](synergistic_weak-strong_collaboration_by_aligning_preferences.md)
- [\[ACL 2025\] Towards Reward Fairness in RLHF: From a Resource Allocation Perspective](reward_fairness_rlhf.md)
- [\[NeurIPS 2025\] What Makes a Reward Model a Good Teacher? An Optimization Perspective](../../NeurIPS2025/llm_alignment/what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)
- [\[ACL 2025\] Reward Generalization in RLHF: A Topological Perspective](reward_generalization_in_rlhf_a_topological_perspective.md)
- [\[ICML 2025\] DPO Meets PPO: Reinforced Token Optimization for RLHF](../../ICML2025/llm_alignment/dpo_meets_ppo_reinforced_token_optimization_for_rlhf.md)

</div>

<!-- RELATED:END -->
