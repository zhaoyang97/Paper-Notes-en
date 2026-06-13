---
title: >-
  [Paper Note] Alignment-Aware Decoding
description: >-
  [ICML 2026][LLM Alignment][alignment-aware decoding] Alignment-Aware Decoding (AAD) directly utilizes the token probability ratio of a DPO model relative to its SFT reference model as an implicit alignment reward during…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "alignment-aware decoding"
  - "DPO"
  - "inference-time alignment"
  - "token-level reward"
  - "preference optimization"
date: 2026-05-08
content_hash: fd656010e847209b
---

# Alignment-Aware Decoding

**Conference**: ICML 2026  
**arXiv**: [2509.26169](https://arxiv.org/abs/2509.26169)  
**Code**: https://github.com/ETH-DISCO/alignment-aware-decoding  
**Area**: LLM Alignment / Inference-time Decoding / DPO  
**Keywords**: alignment-aware decoding, DPO, inference-time alignment, token-level reward, preference optimization  

## TL;DR
Alignment-Aware Decoding (AAD) directly utilizes the token probability ratio of a DPO model relative to its SFT reference model as an implicit alignment reward during inference. Without additional training or external reward models, it generates responses with higher alignment quality more stably than greedy, Bo2, and EFT, and can further generate synthetic preference data to improve DPO.

## Background & Motivation
**Background**: Mainstream LLM alignment is typically completed during the training phase. For instance, RLHF trains a reward model followed by PPO optimization, while DPO directly trains models from chosen/rejected preference pairs. After training, most deployments use standard decoding such as greedy, sampling, or best-of-N.

**Limitations of Prior Work**: Although DPO can move a model closer to preference data, it is essentially still constrained by the prior of the SFT reference model. Theoretically, even if a response has a higher true reward, the DPO optimal policy might favor another lower-reward response as long as the SFT model assigns a sufficiently low prior probability to the former. Consequently, standard decoding inherits the bias of the reference model and fails to fully exploit the fine-grained preference signals learned during DPO training.

**Key Challenge**: Preference optimization during training encodes "which token is more representative of a high-preference response" into the probability difference between DPO and SFT. However, at inference time, if tokens are selected solely based on DPO probabilities, this alignment signal is often masked by language fluency and the SFT prior. Conversely, directly maximizing the probability ratio can lead to selecting "weird" tokens with low probability, resulting in degenerate output.

**Goal**: The authors aim to design a simple, training-free inference-time alignment method that only relies on a standard SFT+DPO model pair. It should be better aligned than greedy DPO without requiring extra reward models, complex searches, or extensive hyperparameter tuning for each dataset.

**Key Insight**: The DPO objective can be interpreted as learning an implicit reward or token-level advantage. AAD applies this interpretation directly to decoding: the DPO model is responsible for providing the feasibility of candidate tokens, while the DPO/SFT probability ratio provides the alignment preference.

**Core Idea**: Within a set of tokens considered "sufficiently probable" by the DPO model, select the token with the largest probability increase relative to the SFT model, thereby maximizing the implicit alignment reward while maintaining fluency.

## Method
The AAD method is concise, but the key lies in changing the decoding objective. Standard greedy DPO selects the token with the highest $\pi_{\mathrm{dpo}}$ probability; AAD treats $\pi_{\mathrm{dpo}}$ only as a candidate filter and uses $\log \pi_{\mathrm{dpo}}(v|s)-\log \pi_{\mathrm{sft}}(v|s)$ for actual ranking, representing how much support preference optimization has added to a token relative to the SFT prior.

### Overall Architecture
Input includes prompt $x$, current prefix $y_{1:t-1}$, post-DPO model $\pi_{\mathrm{dpo}}$, pre-DPO SFT model $\pi_{\mathrm{sft}}$, maximum length, and a filtering threshold $\alpha$. At each step, the next-token distributions for both models are computed via forward passes. A candidate set $\mathcal{V}_{\alpha}$ is filtered from the DPO distribution where probabilities are at least $\alpha$ times the maximum probability. Within this set, AAD selects the token with the largest log-ratio. Decoding stops at `<eos>` or the maximum length.

### Key Designs
1. **DPO/SFT probability ratio as token-level reward**:
	- Function: Converts preference signals learned during DPO training into a token score usable at each decoding step.
	- Mechanism: The implicit reward of DPO can be written as $r_{\theta}(x,y)=\beta\log \frac{\pi_{\theta}(y|x)}{\pi_{\mathrm{sft}}(y|x)}$. AAD localizes this to $A(v|s)=\log\frac{\pi_{\mathrm{dpo}}(v|s)}{\pi_{\mathrm{sft}}(v|s)}$, selecting tokens significantly up-weighted by DPO compared to SFT.
	- Design Motivation: Relying solely on DPO probability leaves the model heavily influenced by the SFT prior; the probability ratio more closely reflects "what the preference optimization actually rewarded."

2. **min-$\alpha$ credible token filtering**:
	- Function: Avoids selecting low-probability, incoherent, or numerically unstable tokens caused by simply maximizing the probability ratio.
	- Mechanism: AAD only compares the advantage among tokens satisfying $\pi_{\mathrm{dpo}}(v|s) \ge \alpha \max_{v'}\pi_{\mathrm{dpo}}(v'|s)$. The main experiments consistently use $\alpha=0.1$ across multiple datasets and model scales without additional tuning.
	- Design Motivation: High-probability tokens ensure grammatical and semantic feasibility according to the language model, while probability ratio ranking reinforces alignment preference among these viable options. This design decouples fluency and alignment.

3. **AAD for synthetic preference data generation**:
	- Function: Uses AAD output as chosen samples to continue training the DPO model when preference data is scarce.
	- Mechanism: The authors use AAD to generate high-quality chosen completions and nucleus sampling from the DPO model to generate rejected completions. This constructs synthetic preference pairs for further training from SFT or existing DPO checkpoints.
	- Design Motivation: AAD is not just a deployment strategy but also a bootstrap data generator. Experiments show that with only 10% of the original preference data, AAD-generated data significantly closes the gap with full-data DPO.

### Loss & Training
AAD itself has no training loss and only performs two forward passes during inference. DPO models in the experiments are trained for two epochs using a 10% preference training split with LoRA rank 64 and a default DPO coefficient $\beta=0.1$. Reward oracles and picker reward models are trained for two epochs using Bradley-Terry loss. During generation, AAD fixes $\alpha=0.1$, with computational costs roughly equivalent to EFT and Bo2 as all require two forward passes or two candidate generations.

## Key Experimental Results

### Main Results
The main table compares average oracle rewards across datasets like UltraFeedback, Argilla, and OpenRLHF Mixture on Llama/Qwen at various scales. Representative results from three datasets are extracted below, with AAD achieving the highest scores.

| Dataset / Model | Greedy SFT | Greedy DPO | Bo2 | EFT | AAD | Gain over strongest baseline |
|---------------|------------|------------|-----|-----|-----|------------------|
| UltraFeedback / Llama 3B | 0.58 | 0.68 | 0.85 | 1.04 | 2.21 | +1.17 over EFT |
| UltraFeedback / Qwen 4B | 0.22 | 0.29 | 0.47 | 0.58 | 1.19 | +0.61 over EFT |
| Argilla / Llama 8B | 1.72 | 2.55 | 3.16 | 4.65 | 5.90 | +1.25 over EFT |
| Argilla / Qwen 0.6B | -0.86 | 0.12 | 0.68 | 1.99 | 2.33 | +0.34 over EFT |
| OpenRLHF Mixture / Llama 8B | 3.89 | 4.93 | 5.60 | 6.84 | 7.60 | +0.76 over EFT |
| OpenRLHF Mixture / Qwen 4B | 2.63 | 3.56 | 4.48 | 5.29 | 5.45 | +0.16 over EFT |

### Ablation Study
The paper also analyzes external evaluation, human preference, reference choice, data scarcity, and hyperparameters.

| Configuration | Key Metrics | 说明 |
|------|----------|------|
| AlpacaEval / Skywork | AAD win rates against Greedy SFT, Greedy DPO, Bo2, EFT on Llama/Qwen mostly range from 0.73 to 0.79 | Outperforms baselines under external GPT-4 evaluator |
| AlpacaEval / Nectar | AAD win rate against Greedy SFT is 0.80/0.82 on Llama 3B/8B, and 0.70/0.63 against EFT | Maintains advantage on another external oracle dataset |
| Contrastive decoding reference | On Llama-8B DPO, Argilla CD 1B/3B reward is 2.79/2.46, AAD is 5.90 | AAD requires the matching SFT reference, not just any weak model's logits |
| OLMo-2 open-source DPO | 7B AAD reward 3.84, win rate 71.5% vs Greedy DPO; 13B reward 7.22, win rate 75.2% vs Greedy DPO | Method transfers to public DPO models |
| Human evaluation / Skywork 3B | AAD win rate against Greedy DPO 58.7%, against Bo2 73.5%, Elo 1610.1 (Rank 1) | Human evaluation supports reward-model results |
| $\alpha$ Ablation | Optimal range is approx. 0.1 to 0.2 | Loose filtering causes over-optimization; tight filtering approaches greedy |
| Full fine-tuning data ratio | At 5% data, Skywork AAD reward is 1.25 (Bo2 -1.81); at 100%, AAD is 8.36 (Bo2 -0.56) | AAD advantage is more pronounced with less data |
| Latency | Single GPU throughput approx. 0.5x of greedy; 0.75x with dual GPU parallelism | Cost stems from dual model forward passes per token |

### Key Findings
- AAD's advantage over BoN stems from using "per-token preference signals" rather than selecting from complete responses post-hoc. On Argilla and Nectar, BoN with $N=50$ using an oracle reward still struggles to match AAD.
- The AAD reference must be the SFT model used during DPO training. Using a general weak instruct model for contrastive decoding (CD) is significantly inferior to AAD, indicating gains come from the actual DPO-SFT probability gap semantics rather than just counteracting small model bias.
- AAD serves as a data generator. Constructing iterative DPO data using AAD outputs approaches full-data training performance using only 10% of original data, though multi-round bootstrapping may show saturation or degradation.

## Highlights & Insights
- The method is minimalist but captures a frequently wasted byproduct: the probability difference between the DPO and SFT models is the preference signal itself. Standard decoding uses only the DPO distribution, effectively merging this differential signal into the total probability and then losing information.
- The min-$\alpha$ filtering is a critical engineering detail. Without it, maximizing the probability ratio easily selects anomalous tokens weakly supported by SFT but slightly up-weighted by DPO. With it, AAD preserves both linguistic quality and alignment preference.
- This paper makes inference-time alignment lightweight: no extra reward model, no complex tree search, and no parameter changes. It is highly practical for the open-source ecosystem where only SFT/DPO checkpoints are available.

## Limitations & Future Work
- AAD requires simultaneous access to the DPO model and its corresponding SFT checkpoint. Many closed-source models or merged checkpoints cannot use it directly.
- Each token requires two forward passes, roughly halving single-GPU throughput. For long texts or high-concurrency services, it must rely on parallelism, KV cache sharing, or fused kernels to reduce overhead.
- The method is primarily applicable to standard SFT+DPO pipelines. The paper notes mixed results with PPO-trained models, as PPO optimizes an external reward directly and may not preserve the SFT-DPO probability gap that AAD depends on.
- AAD optimizes the alignment signals learned from existing preference data. If the preference data is biased or the reward oracle is unreliable, AAD might amplify these biases more stably rather than inherently being safer.

## Related Work & Insights
- **vs DPO greedy decoding**: Greedy DPO selects the highest probability token, while AAD selects the token most up-weighted by preference optimization relative to SFT that remains credible, thus using alignment signals more directly.
- **vs EFT / proxy alignment**: EFT also uses probability differences between model pairs for inference-time guidance but is often used to transfer alignment signals to a third base model. AAD focuses on directly improving the DPO model's own decoding.
- **vs BoN / reward reranking**: BoN requires generating multiple full candidates and relies on a picker reward model. AAD makes local decisions at each token without an extra reward model, with computation similar to Bo2 but stronger results.
- **vs contrastive decoding**: Traditional CD subtracts weaker model logits to reduce common errors. AAD subtracts the exact SFT reference, aiming not to "oppose weak models" but to explicitly extract the preference increment learned by DPO.

## Rating
- Novelty: ⭐⭐⭐⭐ Using the DPO/SFT probability ratio for decoding is not complex, but the combination of theoretical explanation and engineering constraints is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple preference datasets, Llama/Qwen/OLMo, external oracles, AlpacaEval, human evaluation, and data scarcity settings; verification is solid.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative where formula derivations serve the method; figure/table numbering and extensive appendix results require some cross-referencing.
- Value: ⭐⭐⭐⭐⭐ Highly practical for open-source DPO model deployment and highlights that inference-time alignment can fully exploit reference-model information left behind by training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment](../../ICLR2026/llm_alignment/semantic-aware_wasserstein_policy_regularization_for_large_language_model_alignm.md)
- [\[ICML 2026\] Curriculum Learning for Safety Alignment](curriculum_learning_for_safety_alignment.md)
- [\[ICML 2026\] Implicit Preference Alignment for Human Image Animation](implicit_preference_alignment_for_human_image_animation.md)
- [\[AAAI 2026\] Importance-Aware Data Selection for Efficient LLM Instruction Tuning](../../AAAI2026/llm_alignment/importance-aware_data_selection_for_efficient_llm_instruction_tuning.md)
- [\[ICML 2026\] Implicit Safety Alignment from Crowd Preferences](implicit_safety_alignment_from_crowd_preferences.md)

</div>

<!-- RELATED:END -->
