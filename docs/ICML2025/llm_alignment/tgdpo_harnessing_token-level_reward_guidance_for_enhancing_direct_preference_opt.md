---
title: >-
  [Paper Note] TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization
description: >-
  [ICML 2025][LLM Alignment][DPO] Deconstructs sequence-level PPO into a series of token-level proximal policy optimization problems and introduces a token-level reward guidance function $f(\hat{r}(s_t, a_t))$ to replace the fixed constant $\beta$ in DPO. This allows different tokens to deviate from the reference policy to varying degrees based on their respective reward values, improving the win rate on MT-Bench/AlpacaEval 2/Arena-Hard by up to 7.5/6.2/4.3 percentage points re…
tags:
  - "ICML 2025"
  - "LLM Alignment"
  - "DPO"
  - "token-level reward"
  - "preference optimization"
  - "RLHF"
  - "fine-grained reward guidance"
date: 2026-05-08
content_hash: f2cf9d802ed6cd06
---

# TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization

**Conference**: ICML 2025  
**arXiv**: [2506.14574](https://arxiv.org/abs/2506.14574)  
**Code**: [dvlab-research/TGDPO](https://github.com/dvlab-research/TGDPO)  
**Area**: LLM Alignment/RLHF  
**Keywords**: DPO, token-level reward, preference optimization, RLHF, fine-grained reward guidance

## TL;DR

Deconstructs sequence-level PPO into a series of token-level proximal policy optimization problems and introduces a token-level reward guidance function $f(\hat{r}(s_t, a_t))$ to replace the fixed constant $\beta$ in DPO. This allows different tokens to deviate from the reference policy to varying degrees based on their respective reward values, improving the win rate on MT-Bench/AlpacaEval 2/Arena-Hard by up to 7.5/6.2/4.3 percentage points respectively.

## Background & Motivation

**Sequence-Level Limitations of DPO**: DPO reparameterizes the reward function via the optimal policy, bypassing the step of training an independent reward model. However, DPO is fundamentally a sequence-level bandit problem that assigns a uniform reward signal to the entire response, making it unable to distinguish which tokens in the sequence are preferred and which are dispreferred.

**Success of Token-Level Reward in PPO**: Previous work (Yang et al., 2023; Yin et al., 2025; Zhong et al., 2024) has demonstrated that dense token-level rewards can significantly enhance the alignment performance of PPO, mitigating training instability and sample inefficiency caused by sparse rewards (delayed feedback).

**Key Challenge**: Extending token-level reward guidance to DPO is highly difficult. The reward function of DPO is expressed by the policy being optimized itself. After directly introducing token-level rewards, a policy-dependent partition function $Z(s_t)$ appears in the loss, which cannot be easily eliminated. This remains an unresolved open problem.

**Ours Breakthrough**: By utilizing upper-bound decomposition + modified token-level PPO + a new partition function elimination theorem, the authors present the first computable DPO loss framework with token-level reward guidance.

## Method

### Overall Architecture

The derivation of TGDPO follows a three-step strategy:
1. **Decomposition**: Deconstruct sequence-level PPO with token-level reward guidance into a series of independent token-level PPO problems (Theorem 4.1 upper-bound decomposition).
2. **Modification**: Introduce the reward guidance function $f(\hat{r}(s_t, a_t))$ into token-level PPO, solve for the closed-form optimal policy, and express the corresponding reward (Theorem 4.3).
3. **Elimination**: Utilize the Bradley-Terry model + new theoretical results (Theorem 4.4) to eliminate the uncomputable partition function, yielding the computable TGDPO loss.

### Key Designs

1. **Sequence to Token-Level Decomposition (Theorem 4.1)**: The objective function of sequence-level PPO can be decomposed into $\sum_{t=0}^{T-1}(r_\phi(s_t, a_t) - \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)})$. The authors prove that the maximum of sequence-level PPO is upper-bounded by the sum of maximums of a series of independent token-level PPO sub-problems, i.e., optimizing $\max_{\pi_\theta} \mathbb{E}[r_\phi(s_t, a_t) - \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}]$ independently for each timestep $t$. To make the problem tractable, they relax $s_t \sim \mathcal{D}_t$ (policy-dependent) to $s_t \sim \mathcal{D}$ (policy-independent).

2. **Modified Token-Level PPO and Introduction of Reward Guidance (Theorem 4.3)**: The core idea is to replace the fixed $\beta$ with $\beta \cdot f(\hat{r}(s_t, a_t))$, so that the KL constraint strength of each token is dynamically adjusted by its token-level reward. Specifically, they first equivalently rewrite token-level PPO as $\max \mathbb{E}[\frac{r_\phi(s_t, a_t)}{\beta} - \log \frac{\pi_\theta}{\pi_{\text{ref}}}]$, then replace $\beta$ in the denominator with $\beta f(\hat{r})$. This yields the modified problem $\max \mathbb{E}[\frac{r_\phi(s_t, a_t)}{\beta f(\hat{r}(s_t, a_t))} - \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}]$. This problem has a closed-form optimal policy: $\pi_{\theta_t}(a_t|s_t) = \frac{\pi_{\text{ref}}(a_t|s_t) \exp(\frac{r_\phi}{\beta f(\hat{r})})}{Z(s_t)}$, from which the reward can be solved in reverse: $\frac{r_\phi(s_t, a_t)}{f(\hat{r}(s_t, a_t))} = \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} + \beta \log Z(s_t)$.

3. **Partition Function Elimination (Theorem 4.4)**: After substituting the above reward into the Bradley-Terry model, the preference probability is $\Pr(y_w \succ y_l|x) = \sigma(\varphi(\pi_\theta, f, \hat{r}; x, y_w, y_l) + \delta(f, \hat{r}; x, y_w, y_l))$, where $\delta$ contains the uncomputable partition function but does not depend on $\pi_\theta$. Theorem 4.4 proves that: since the sigmoid function is strictly monotonically increasing, $\sigma(\varphi + \delta)$ shares the same optimal points and ascending directions with $\sigma(\varphi)$. Thus, the $\delta$ term can be safely removed during optimization. This is the key theoretical contribution of this paper—ensuring that eliminating the partition function does not affect the preference ranking of the policy and the optimal policy.

### Loss & Training

**TGDPO Loss Function**:

$$\mathcal{L}_{\text{TGDPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \sum_{t} \beta f_w(\hat{r}_t^w) \log \frac{\pi_\theta(y_w^t|...)}{\pi_{\text{ref}}(y_w^t|...)} - \sum_{t} \beta f_l(\hat{r}_t^l) \log \frac{\pi_\theta(y_l^t|...)}{\pi_{\text{ref}}(y_l^t|...)} \right) \right]$$

**Practical Method**: Employs the token-level reward implicitly learned by DPO $\hat{r}([x, y^{<t}], y^t) = \beta \log \frac{\pi_{\hat{\theta}}(y^t|...)}{\pi_{\text{ref}}(y^t|...)}$, set:
- Win response: $f_w(\hat{r}) = 1 + \alpha \hat{r}$
- Lose response: $f_l(\hat{r}) = 1 - \alpha \hat{r}$

where $\alpha$ is a positive constant (small enough to guarantee $f > 0$).

**Intuitive Understanding at the Gradient Level**:
- Tokens with reward > 0 in the winning response $\rightarrow$ Weight $1 + \alpha\hat{r} > 1$ $\rightarrow$ Gradient amplified, encouraging generation.
- Tokens with reward < 0 in the winning response $\rightarrow$ Weight $1 + \alpha\hat{r} < 1$ $\rightarrow$ Gradient scaled down, avoiding blind encouragement.
- Tokens with reward < 0 in the losing response $\rightarrow$ Weight $1 - \alpha\hat{r} > 1$ $\rightarrow$ Gradient amplified, stronger suppression.
- Tokens with reward > 0 in the losing response $\rightarrow$ Weight $1 - \alpha\hat{r} < 1$ $\rightarrow$ Gradient scaled down, avoiding blind suppression.

**Training Process**: First train a model $\pi_{\hat{\theta}}$ using standard DPO, use it to compute token-level rewards, and then train from the SFT model using the TGDPO loss.

## Key Experimental Results

### Main Results

| Model / Annotator | Method | AlpacaEval 2 WR(%) | Arena-Hard WR(%) | MT-Bench Score | MT-Bench WR(%) |
|---|---|---|---|---|---|
| Llama3-8B + PairRM | SFT | 30.6 | 21.4 | 7.9 | 27.5 |
| | DPO | 41.7 | 30.4 | 8.0 | 37.5 |
| | SimPO | 39.8 | 28.7 | 7.8 | 32.5 |
| | **TGDPO** | **43.9** | **34.3** | 8.0 | **41.9** |
| Llama3-8B + ArmoRM | DPO | 40.8 | 36.2 | 8.2 | 46.3 |
| | SimPO | 37.0 | 28.1 | 7.8 | 42.5 |
| | **TGDPO** | **42.5** | **40.5** | 7.9 | 45.0 |
| Llama3.2-3B + ArmoRM | DPO | 29.6 | 23.2 | 7.9 | 29.4 |
| | **TGDPO** | **35.8** | **25.4** | **8.1** | **36.9** |
| Gemma2-2B + ArmoRM | DPO | 40.8 | 26.4 | 8.0 | 43.1 |
| | **TGDPO** | **43.0** | **30.7** | **8.1** | **46.9** |

### Ablation Study

| Configuration | AlpacaEval 2 WR(%) | Arena-Hard WR(%) | Description |
|------|-----|-----|------|
| DPO w/ convergence | 30.7 | 17.9 | Severe degradation after convergence, even lower than SFT |
| SimPO w/ convergence | 4.6 | 2.4 | Almost complete collapse after convergence |
| **TGDPO w/ convergence** | **43.9** | **34.3** | Outstanding performance retained after convergence |
| TGDPO α=0.5 | 43.9 | 34.3 | Consistent performance across different $\alpha$ after convergence |
| TGDPO α=1.0 | 42.5 | 33.9 | Larger $\alpha$ results in faster convergence |
| TGDPO α=2.0 | 43.3 | 34.3 | Robust performance |
| TGDPO w/ β=0.1 reward | 42.8 | 34.3 | Robust to reward quality |
| TGDPO w/ β=0.01 reward | 43.9 | 34.3 | DPO with $\beta=0.1$ performs poorly, but TGDPO remains unaffected |

### Key Findings

1. **Ready to Use Upon Convergence**: DPO/SimPO performance collapses once the loss converges (SimPO even drops to a 4.6% win rate), requiring meticulous prompt tuning to find the "sweet spot". TGDPO maintains optimal performance upon loss convergence, greatly reducing the hyperparameter tuning burden.
2. **α Controls Convergence Speed**: Larger $\alpha$ leads to faster convergence (typically within ~50 steps for $\alpha=2.0$, and ~1 epoch for $\alpha=0.5$), while the final performance remains almost identical, enabling computation savings through early stopping.
3. **Robust to Reward Quality**: Even using token-level rewards generated by a sub-optimal DPO model ($\beta=0.1$, AlpacaEval win rate only 34.8%), TGDPO still achieves performance comparable to using the optimal rewards (42.8% vs 43.9%).

## Highlights & Insights

- **Theoretical Elegance**: Through the three-step derivation of upper-bound decomposition $\rightarrow$ closed-form solution $\rightarrow$ partition function elimination, token-level reward guidance is naturally embedded into the DPO framework. This seamlessly unifies prior methods like DPO ($f \equiv 1$ degenerates to DPO).
- **High Practicality**: Requires no additional token-level reward models, directly reusing the implicitly learned DPO rewards ($\beta \log \frac{\pi_{\hat\theta}}{\pi_{\text{ref}}}$). It is simple to implement and yields significant improvements.
- **Healthy Convergence Behavior**: This is a rare method in the preference optimization field where "loss convergence $\approx$ good performance", resolving the long-standing discrepancy between loss and performance in traditional methods.
- **Fine-Grained Token-Level Modulation**: TGDPO can amplify gradients for preferred tokens and scale down gradients for dispreferred tokens within the same response. This circumvents the coarse binary treatment of "all-in acceptance of winning responses, all-in rejection of losing responses".

## Limitations & Future Work

- **Single Source of Rewards**: The practical method relies on first training a DPO model to provide token-level rewards, which increases the pipeline's complexity. Future work could explore other lightweight sources of rewards.
- **Simple Functional Form of f**: Currently, only a linear form $f = 1 + \alpha r$ is used. More complex non-linear designs (e.g., exponential, softmax-based) might yield further gains.
- **Limited Evaluation Scope**: Evaluated only on the helpfulness dimension of instruction-following, without validating performance on other alignment dimensions like safety, honesty, and fairness.
- **Relatively Small Model Sizes**: Experiments were conducted on 2B–8B models. Whether significant benefits persist in larger models remains to be verified.
- **Theoretical Impact of the Relaxation Condition**: The error introduced by relaxing $s_t \sim \mathcal{D}_t$ to $s_t \sim \mathcal{D}$ was not quantitatively analyzed.

## Related Work & Insights

- **DPO (Rafailov et al., 2023)**: The direct target of improvement in this paper. TGDPO degenerates to DPO when $f \equiv 1$.
- **SimPO (Meng et al., 2024)**: A DPO variant removing the reference model, which does not perform as well as TGDPO.
- **TDPO (Zeng et al., 2024)**: Formulates DPO using a token-level MDP and adds forward KL, but does not introduce token-level reward guidance.
- **Rafailov et al. (2024)**: Provides the theoretical foundation proving DPO implicitly learns token-level rewards, which this work directly builds upon.
- **Insights**: The framework of TGDPO can be combined with other fine-grained reward sources (such as process reward models, verifiers) for scenarios requiring step-level feedback, such as mathematical reasoning and code generation.

## Rating

- Novelty: ⭐⭐⭐⭐ — Complete and novel theoretical derivation (elimination theorem of partition functions), though the core concept (token-level weighting) is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple models, benchmarks, and ablation studies. Demonstrates TGDPO's unique properties (convergence behavior, robustness) thoroughly.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear and rigorous derivations, well-articulated motivation, and a natural transition from theory to practice.
- Value: ⭐⭐⭐⭐ — Addresses real pain points of DPO (convergence degradation, coarse-grained token feedback) with a simple and practical approach.

1. **Deconstruction of Sequence-Level PPO to Token-Level PPO (Theorem 4.1)**

   The core idea is to leverage the autoregressive MDP structure of LLMs: $\pi_\theta(y|x) = \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t)$, allowing the sequence-level target to be decomposed into a token-level summation. Using the upper-bound approach, it is proven that the maximum of sequence-level PPO is upper-bounded by the sum of maximums of $T$ token-level PPO sub-problems:

   $$\max_{\pi_\theta} \mathbb{E}\left[r_\phi(s_t, a_t) - \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}\right]$$

   Key relaxation: relaxing $s_t \sim \mathcal{D}_t$ (dependent on $\pi_\theta$) to $s_t \sim \mathcal{D}$ (independent of $\pi_\theta$), making the problem solvable.

2. **Modified Token-Level PPO with Reward Guidance (Theorem 4.3)**

   Introduces the reward guidance function $f(\hat{r}(s_t,a_t))$ into token-level PPO, replacing the fixed KL penalty coefficient $\beta$ with the adaptive $\beta f(\hat{r}(s_t,a_t))$. The trick is to first move $\beta$ to the denominator of the reward term (since it is a positive constant and does not affect the optimal solution) and then perform the replacement:

   $$\max_{\pi_\theta} \mathbb{E}\left[\frac{r_\phi(s_t,a_t)}{\beta f(\hat{r}(s_t,a_t))} - \log\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}\right]$$

   The optimal policy is: $\pi_{\theta_t}(a_t|s_t) = \frac{\pi_{\text{ref}}(a_t|s_t) \exp\left(\frac{r_\phi(s_t,a_t)}{\beta f(\hat{r}(s_t,a_t))}\right)}{Z(s_t)}$

   This yields the explicit representation of token-level reward: $r_\phi(s_t,a_t) = \beta f(\hat{r}(s_t,a_t)) \log\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} + \beta f(\hat{r}(s_t,a_t)) \log Z(s_t)$

3. **Partition Function Elimination (Theorem 4.4)**

   After substituting the token-level reward into the Bradley-Terry model, the preference function includes the uncomputable $\delta(f,\hat{r};x,y_w,y_l)$ term (containing the partition function). A key finding: this term does not depend on the policy $\pi_\theta$, and since the sigmoid function is strictly monotonically increasing, optimizing the objective with the $\delta$ term is equivalent to optimizing it without $\delta$. The preference order between any two policies remains completely unchanged after eliminating $\delta$.

### Loss & Training

**TGDPO Loss Function**:

$$\mathcal{L}_{\text{TGDPO}}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\sum_{t=0}^{T_w-1}\beta f_w(\hat{r}_t^w)\log\frac{\pi_\theta(y_w^t|[x,y_w^{<t}])}{\pi_{\text{ref}}(y_w^t|[x,y_w^{<t}])} - \sum_{t=0}^{T_l-1}\beta f_l(\hat{r}_t^l)\log\frac{\pi_\theta(y_l^t|[x,y_l^{<t}])}{\pi_{\text{ref}}(y_l^t|[x,y_l^{<t}])}\right)\right]$$

where $f_w, f_l$ are the shaping functions of the win/lose responses, respectively. When $f_w \equiv f_l \equiv 1$, it degenerates to standard DPO.

**Practical Method**: Employs the token-level reward implicitly learned by DPO $\hat{r}([x,y^{<t}],y^t) = \beta\log\frac{\pi_{\hat\theta}(y^t|[x,y^{<t}])}{\pi_{\text{ref}}(y^t|[x,y^{<t}])}$, set:

- Win response: $f_w(\hat{r}) = 1 + \alpha\hat{r}$ —— High-reward tokens receive larger weights and are reinforced more.
- Lose response: $f_l(\hat{r}) = 1 - \alpha\hat{r}$ —— Low-reward (negative) tokens receive larger weights and are suppressed more.

**Four Token-Level Behaviors**:

- Desirable tokens in Win ($\hat{r}>0$): Weight $>1$, reinforcing this action.
- Undesirable tokens in Win ($\hat{r}<0$): Weight $<1$, reducing the probability of this action.
- Undesirable tokens in Lose ($\hat{r}<0$): Weight $>1$, suppressing it further.
- Desirable tokens in Lose ($\hat{r}>0$): Weight $<1$, decreasing the penalty on this action.

**Training Process**: First train with standard DPO to obtain $\pi_{\hat\theta}$, compute the token-level rewards, and then train the final policy using the TGDPO loss. $\alpha$ controls convergence speed (larger $\alpha$ corresponds to faster convergence), $\beta=0.01$, and training is conducted using the AdamW optimizer.

## Key Experimental Results

### Main Results

| Model + Annotator | Metric | DPO | SimPO | TGDPO | Gain (vs Best) |
|--------------|------|-----|-------|-------|-------------|
| Llama3-8B + PairRM | AlpacaEval 2 WR | 41.7% | 39.8% | **43.9%** | +2.2 |
| Llama3-8B + PairRM | Arena-Hard WR | 30.4% | 28.7% | **34.3%** | +3.9 |
| Llama3-8B + PairRM | MT-Bench WR | 37.5% | 32.5% | **41.9%** | +4.4 |
| Llama3-8B + ArmoRM | AlpacaEval 2 WR | 40.8% | 37.0% | **42.5%** | +1.7 |
| Llama3-8B + ArmoRM | Arena-Hard WR | 36.2% | 28.1% | **40.5%** | +4.3 |
| Llama3.2-3B + ArmoRM | AlpacaEval 2 WR | 29.6% | 26.2% | **35.8%** | +6.2 |
| Llama3.2-3B + ArmoRM | Arena-Hard WR | 23.2% | 22.6% | **25.4%** | +2.2 |
| Gemma2-2B + ArmoRM | AlpacaEval 2 WR | 40.8% | 34.8% | **43.0%** | +2.2 |
| Gemma2-2B + ArmoRM | Arena-Hard WR | 26.4% | 21.1% | **30.7%** | +4.3 |

### Ablation Study

| Configuration | AlpacaEval 2 WR | Arena-Hard WR | Description |
|------|----------------|---------------|------|
| DPO w/ convergence | 30.7% | 17.9% | Severe degradation after convergence |
| SimPO w/ convergence | 4.6% | 2.4% | Almost unusable after convergence |
| TGDPO w/ convergence | **43.9%** | **34.3%** | Keeps optimal performance after convergence |
| TGDPO α=0.5 | 43.9% | 34.3% | Slow convergence, epoch 1 checkpoint |
| TGDPO α=1.0 | 42.5% | 33.9% | Moderate convergence, step 60 checkpoint |
| TGDPO α=2.0 | 43.3% | 34.3% | Fast convergence, step 50 checkpoint |
| TGDPO (DPO β=0.1 reward) | 42.8% | 34.3% | Rewards from a sub-optimal DPO model |
| TGDPO (DPO β=0.01 reward) | 43.9% | 34.3% | Rewards from a better DPO model |

### Key Findings

1. **Ready to Use Upon Convergence**: TGDPO maintains excellent performance after loss convergence, whereas DPO/SimPO degrade severely upon convergence. This eliminates the tedious process of finding the "sweet spot" in traditional preference optimization.
2. **Controllable Convergence Speed**: Convergence speed can be freely controlled by tuning $\alpha$, with different values of $\alpha$ leading to identical post-convergence performance.
3. **Robust to Reward Quality**: Even using token-level rewards from a sub-optimal DPO model ($\beta=0.1$), TGDPO performs almost identically to using the optimal rewards.
4. **Cross-Model Consistency**: Consistently outperforms baselines across three model scales (2B, 3B, and 8B).

## Highlights & Insights

1. **Rigorous Theoretical Derivations with Simplicity in Practice**: Although the mathematical derivations span three theorems, the final practical method is extremely simple: merely multiplying the log-ratio of each DPO token by a linear weight $1 \pm \alpha\hat{r}$, keeping implementation costs very low.
2. **Convergence-Performance Alignment**: This is the most unique contribution of this paper—resolving the counter-intuitive "loss drop = performance drop" issue in preference optimization, so training no longer requires timely stopping before overfitting.
3. **Framework Unification**: The loss function of TGDPO is a generalized framework. It degenerates to DPO when $f \equiv 1$, and can spawn multiple variants.
4. **No Need for Extra Reward Models**: Reusing DPO's implicitly learned token-level rewards as the guidance signal eliminates the necessity of training an external reward model, preserving DPO's advantage of being "lightweight".

## Limitations & Future Work

1. **Two-Stage Training**: Requires training a DPO model first to obtain token-level rewards, which increases computational overhead. Can this be designed as a single-stage method with iterative/online reward updating?
2. **Evaluated only on Helplessness**: Experiments are concentrated on instruction-following tasks without validating performance on other alignment dimensions like safety and honesty.
3. **Theoretical Gap of Relaxation**: The potential errors introduced by relaxing $s_t \sim \mathcal{D}_t$ to $s_t \sim \mathcal{D}$ have not been thoroughly analyzed.
4. **Shaping Function Choice**: $f = 1 + \alpha\hat{r}$ is only a simple linear design. Are there more optimal non-linear forms?
5. **$\alpha$ Must Be Sufficiently Small to Ensure $f > 0$**: When absolute token reward values are large, $f$ might become negative, violating the assumption. The boundary of robustness is unclear.

## Related Work & Insights

- **TDPO (Zeng et al., 2024)**: Solves DPO from a token-level MDP perspective and introduces forward KL divergence, but does not use token-level reward guidance.
- **SimPO (Meng et al., 2024)**: A reference-free variant of DPO that aligns decoders but remains a sequence-level optimization.
- **Rafailov et al., 2024 (From r to Q\*)**: Proves that DPO implicitly learns token-level rewards, serving as one of the theoretical foundations of this paper.
- **TDPO-R (Shao et al., 2025)**: Formulates learning DPO from a time-decaying perspective, also focusing on token-level variance.

**Insights**: The concept of token-level reward guidance can be generalized to other sequence-level optimization tasks (e.g., diffusion alignment, code generation). The core insight is that "decisions at different steps should have varying degrees of freedom to deviate".

## Rating

- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] DPO Meets PPO: Reinforced Token Optimization for RLHF](dpo_meets_ppo_reinforced_token_optimization_for_rlhf.md)
- [\[ACL 2025\] T-REG: Preference Optimization with Token-Level Reward Regularization](../../ACL2025/llm_alignment/t-reg_preference_optimization_with_token-level_reward_regularization.md)
- [\[ACL 2025\] SDPO: Segment-Level Direct Preference Optimization for Social Agents](../../ACL2025/llm_alignment/sdpo_segment-level_direct_preference_optimization_for_social_agents.md)
- [\[ACL 2025\] Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](../../ACL2025/llm_alignment/otpo_token_weighting.md)
- [\[ICML 2026\] Boosting Direct Preference Optimization with Penalization](../../ICML2026/llm_alignment/boosting_direct_preference_optimization_with_penalization.md)

</div>

<!-- RELATED:END -->
