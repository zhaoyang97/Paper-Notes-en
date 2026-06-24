---
title: >-
  [Paper Note] T-REG: Preference Optimization with Token-Level Reward Regularization
description: >-
  [ACL 2025][LLM Alignment][Preference Optimization] T-REG proposes a token-level reward regularization method that leverages contrastive prompting of LLMs to self-generate token-level reward signals. These signals are used as weak supervision to guide the token-level reward assignment implicitly learned by DPO, outperforming DPO by up to 3.8% on Alpaca Eval 2 and 4.4% on Arena-Hard.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Preference Optimization"
  - "Token-Level Reward"
  - "Credit Assignment"
  - "DPO Regularization"
  - "Contrastive Prompting"
date: 2026-05-08
content_hash: 3314ffa26e95ade3
---

# T-REG: Preference Optimization with Token-Level Reward Regularization

**Conference**: ACL 2025  
**arXiv**: [2412.02685](https://arxiv.org/abs/2412.02685)  
**Code**: [https://github.com/wzhouad/T-REG](https://github.com/wzhouad/T-REG)  
**Area**: RLHF Alignment  
**Keywords**: Preference Optimization, Token-Level Reward, Credit Assignment, DPO Regularization, Contrastive Prompting

## TL;DR

T-REG proposes a token-level reward regularization method that leverages contrastive prompting of LLMs to self-generate token-level reward signals. These signals are used as weak supervision to guide the token-level reward assignment implicitly learned by DPO, outperforming DPO by up to 3.8% on Alpaca Eval 2 and 4.4% on Arena-Hard.

## Background & Motivation

1. **Background**: Reinforcement Learning from Human Feedback (RLHF) is the mainstream method for aligning LLMs with human preferences. Direct alignment algorithms like DPO optimize policy models through preference data, avoiding the training of additional reward models. These methods typically use **sequence-level rewards**—assigning a single overall score to the entire output.

2. **Limitations of Prior Work**: Sequence-level rewards are inherently sparse signals. In an output of hundreds or thousands of tokens, the contribution of different tokens to the final quality is non-uniform. Training with a single sequence-level reward makes it difficult for the model to learn which tokens are truly important (i.e., the token-level credit assignment problem).

3. **Key Challenge**: Existing token-level RLHF methods (e.g., RTO, SePO, TLCR) either rely on AI annotators to generate token-level rewards (unreliable quality), use credit assignment models to redistribute sequence rewards (requiring extra training), or directly use these noisy token-level rewards for PPO optimization (sensitive to noise).

4. **Goal**: How to effectively utilize token-level rewards in preference optimization without relying on external annotations and remaining robust to noise?

5. **Key Insight**: The authors observe that DPO implicitly learns token-level rewards (i.e., $\beta \log \frac{\pi(y_t|x,y_{<t})}{\pi_{\text{ref}}(y_t|x,y_{<t})}$) but lacks direct token-level guidance. Meanwhile, LLMs themselves naturally possess self-refinement capabilities and can "self-generate" token-level rewards through contrastive prompting.

6. **Core Idea**: Instead of directly using automatically annotated token-level rewards to optimize the policy, they are treated as a **weakly supervised regularization term** to guide the token-level rewards implicitly learned by DPO to align with the self-generated rewards.

## Method

### Overall Architecture

The input is a preference dataset $\mathcal{D} = \{(x, y_w, y_l)\}$ (prompt + winning/losing output pairs). The training process consists of two stages: (1) self-generating token-level rewards via the reference model using contrastive prompting; (2) training by incorporating the token-level reward regularization term on top of the DPO loss. The output is a preference-optimized policy model.

### Key Designs

1. **Token-Level Reward Regularization (Core Innovation)**:

    - **Function**: Adds a regularization term to the sequence-level preference optimization objective of DPO to constrain the token-level rewards learned by the model.
    - **Mechanism**: Defines the similarity between the implicitly learned token-level reward $r_{\text{token}}$ of DPO and the externally self-generated token-level reward $\hat{r}_{\text{token}}$ as $\text{sim}(y_t) = r_{\text{token}}(y_t) \cdot \hat{r}_{\text{token}}(y_t)$, and maximizes their alignment across all tokens. After simplification, the regularization term becomes a weighted language modeling loss $\mathcal{L}_{\text{reg}} = -\sum_t \beta \hat{r}_{\text{token}}(y_t) \log \pi(y_t|x,y_{<t})$, which increases the probability of positive-reward tokens and decreases the probability of negative-reward tokens.
    - **Design Motivation**: Unlike RTO/SePO, which directly utilize noisy token-level rewards for PPO or selection optimization, this method "guides" DPO's inherent token-level credit assignment in a weakly supervised manner, balancing sequence-level consistency and token-level granularity.

2. **Self-Generating Token-Level Rewards via Contrastive Prompting**:

    - **Function**: Generates rewards for all tokens using only the reference model via two forward passes, without requiring any additional model training.
    - **Mechanism**: Two contrastive refinement prompts $x_{\text{better}}$ (asking to improve the output) and $x_{\text{worse}}$ (asking to deteriorate the output) are designed, guiding the model across four dimensions: "helpful, correct, coherent, concise" versus "unhelpful, incorrect, incoherent, verbose". Then, the token-level reward is defined as $\hat{r}(y_t) = \sigma(\log \frac{\pi_{\text{eval}}(y_t|x_{\text{better}}, y_{<t})}{\pi_{\text{eval}}(y_t|x_{\text{worse}}, y_{<t})}) - 0.5$, which is scaled to the range $[-0.5, 0.5]$ using a sigmoid function.
    - **Design Motivation**: Leverages the LLM's own awareness of output quality. High-quality tokens have higher probabilities under the better prompt and lower probabilities under the worse prompt, with the difference naturally reflecting token quality.

3. **Sequence-Level Gradient Weight Balancing**:

    - **Function**: Prevents the regularization term from dominating gradients on certain sequences, ensuring smooth coordination with the DPO loss.
    - **Mechanism**: Introduces a sequence weight $w = \sigma(r_{\text{DPO}}(x,y_l) - r_{\text{DPO}}(x,y_w))$ (extracted from the DPO gradient) to weight the regularization loss. The final loss is formulated as $\mathcal{L}_{\text{DPO-REG}} = \mathcal{L}_{\text{DPO}} + \alpha \cdot w \cdot (\mathcal{L}_{\text{REG}}(x,y_w) + \mathcal{L}_{\text{REG}}(x,y_l))$.
    - **Design Motivation**: The DPO gradient inherently contains sequence-level weight information. When the DPO gradient is large (indicating the model has not learned well yet), larger weight is given to the regularization; conversely, the weight decreases to avoid overfitting to token-level noise.

### Loss & Training

The final loss is $\mathcal{L}_{\text{DPO-REG}} = \mathcal{L}_{\text{DPO}} + \alpha \cdot w \cdot (\mathcal{L}_{\text{REG}}(y_w) + \mathcal{L}_{\text{REG}}(y_l))$, where $\alpha \in \{0.1, 0.25, 0.5\}$ is searched. The training adopts an on-policy approximation pipeline, sampling 5 outputs using the reference policy and selecting the best and worst as preference pairs scored by ArmoRM.

## Key Experimental Results

### Main Results

| Dataset | Metric | T-REG (DPO-REG) | DPO | Gain |
|--------|------|-----------------|-----|------|
| Alpaca Eval 2 (Llama-3-8B) | LC Win Rate | 50.8% | 47.0% | +3.8% |
| Alpaca Eval 2 (Gemma-2-9B) | LC Win Rate | 74.5% (SimPO-REG) | 73.5% (SimPO) | +1.0% |
| Arena-Hard (Llama-3-8B) | Win Rate | 40.3% | 35.9% | +4.4% |
| Arena-Hard (Gemma-2-9B) | Win Rate | 64.2% (SimPO-REG) | 63.0% (SimPO) | +1.2% |

### Ablation Study

| Configuration | Alpaca Eval 2 LC WR | Arena-Hard WR | Description |
|------|---------------------|---------------|------|
| DPO-REG (Full) | 50.8% | 40.3% | Full model |
| DPO-SFT on $y_w$ | 46.0% | 32.7% | Full-token SFT significantly degrades performance instead |
| Static weighting | 48.0% | 35.1% | Performance drops close to DPO without sequence weight |
| Regularization with DPO reward | 49.8% | 36.9% | 3.4% worse than self-generated rewards on Arena-Hard |
| DPO baseline | 47.0% | 35.9% | Baseline |

### Key Findings

- **Selective Regularization vs. Full-Token SFT**: Performing SFT on all tokens (DPO-SFT) degrades performance because $y_w$ still contains low-quality tokens; T-REG only scales up the probability of high-reward tokens and performs significantly better.
- **Sequence Weight is Crucial**: Without sequence-level weighting, the regularization term cannot balance with the DPO gradient, causing the performance to degenerate close to the baseline.
- **Self-Generated Rewards Outperform DPO-Derived Rewards**: Utilizing self-generated token-level rewards is 3.4% better on Arena-Hard than using DPO implicit rewards for regularization.
- **Compatibility with Other Preference Optimization Algorithms**: SimPO-REG consistently improves upon SimPO.
- **Qualitative Case Study**: T-REG correctly assigns negative rewards to format-mismatched tokens, whereas DPO is prone to misjudgment.

## Highlights & Insights

- **A "Weakly Supervised + Regularization" Paradigm for Token-Level Credit Assignment**: Instead of directly using noisy signals as the primary optimization target, this method uses them to "guide" the model's self-learned token-level representations. This design philosophy is elegant and highly tolerant of signal noise.
- **Self-Sustainability of Contrastive Prompting**: No external stronger models or extra training are required. Token-level rewards are obtained by running only two forward passes on the same model with opposite-direction prompts, incurring minimal computational overhead.
- **Sequence-Level Gradient Weights**: Weights are extracted directly from the DPO gradient to balance the two objectives, avoiding the introduction of additional hyperparameters.

## Limitations & Future Work

- The quality of contrastive prompting depends on the LLM's own awareness of "good and bad," which may lead to limited effectiveness with very weak base models.
- There is currently no quantitative evaluation benchmark for token-level rewards, only qualitative case studies.
- The method is only verified on instruction-following tasks, without testing on reasoning or coding scenarios.
- The $\alpha$ hyperparameter still needs to be searched; adaptive strategies could be explored in future work.

## Related Work & Insights

- **vs. RTO (Zhong et al.)**: RTO first trains with DPO to obtain token-level rewards and then performs PPO. This two-stage pipeline is complex and sensitive to token-level reward noise. T-REG offers single-stage joint optimization, which is simpler and more robust.
- **vs. SePO (Yang et al.)**: SePO selects a subset of high-reward tokens for preference optimization, discarding a significant amount of information. T-REG retains all tokens but weights them with their reward values.
- **vs. TDPO (Zeng et al.)**: TDPO derives token-level DPO from an MDP perspective without using explicit token-level reward supervision. T-REG additionally introduces self-generated rewards as guidance signals.

## Rating

- Novelty: ⭐⭐⭐⭐ The core idea (weakly supervised regularization) is simple yet effective; though not revolutionary, it is highly practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks, detailed ablation studies, case studies, and different base models evaluated.
- Writing Quality: ⭐⭐⭐⭐ Underpinned by clear mathematical derivations, with a tightly-coupled motivational logic chain.
- Value: ⭐⭐⭐⭐ The method is simple and highly generalizable (can be integrated into DPO/SimPO), making it directly applicable in industrial settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](../../ICML2025/llm_alignment/tgdpo_harnessing_token-level_reward_guidance_for_enhancing_direct_preference_opt.md)
- [\[ACL 2025\] Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](otpo_token_weighting.md)
- [\[ACL 2025\] SDPO: Segment-Level Direct Preference Optimization for Social Agents](sdpo_segment-level_direct_preference_optimization_for_social_agents.md)
- [\[ACL 2025\] ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning](aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)
- [\[ACL 2025\] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)

</div>

<!-- RELATED:END -->
