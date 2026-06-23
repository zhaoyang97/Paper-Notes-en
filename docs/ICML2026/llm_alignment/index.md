---
title: >-
  ICML2026 Alignment & RLHF Papers · 37 Notes
description: >-
  37 ICML2026 papers in the Alignment & RLHF area, covering Alignment/RLHF, LLM, Adversarial Robustness, Reinforcement Learning and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ICML2026"
  - "Alignment & RLHF"
  - "AI paper notes"
  - "paper summaries"
  - "Alignment/RLHF"
  - "LLM"
  - "Adversarial Robustness"
  - "Reinforcement Learning"
item_list:
  - u: "adaptive_probe-based_steering_for_robust_llm_jailbreaking/"
    t: "Adaptive Probe-based Steering for Robust LLM Jailbreaking"
  - u: "alignment-aware_decoding/"
    t: "Alignment-Aware Decoding"
  - u: "autoregressive_direct_preference_optimization/"
    t: "Autoregressive Direct Preference Optimization"
  - u: "boosting_direct_preference_optimization_with_penalization/"
    t: "Boosting Direct Preference Optimization with Penalization"
  - u: "consistency_training_can_entrench_misalignment/"
    t: "Consistency Training Can Entrench Misalignment"
  - u: "curriculum_learning_for_safety_alignment/"
    t: "Curriculum Learning for Safety Alignment"
  - u: "decoupling_reasoning_and_confidence_resurrecting_calibration_in_reinforcement_le/"
    t: "Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards"
  - u: "efficient_preference_poisoning_attack_on_offline_rlhf/"
    t: "Efficient Preference Poisoning Attack on Offline RLHF"
  - u: "f-divergence_regularized_rlhf_two_tales_of_sampling_and_unified_analyses/"
    t: "$f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses"
  - u: "f-tis_harnessing_diverse_models_in_collaborative_grpo/"
    t: "F-TIS: Harnessing Diverse Models in Collaborative GRPO"
  - u: "gist_targeted_data_selection_for_instruction_tuning_via_coupled_optimization_geo/"
    t: "GIST: Targeted Data Selection for Instruction Tuning with Gradient Subspace Projection"
  - u: "implicit_preference_alignment_for_human_image_animation/"
    t: "Implicit Preference Alignment for Human Image Animation"
  - u: "implicit_safety_alignment_from_crowd_preferences/"
    t: "Implicit Safety Alignment from Crowd Preferences"
  - u: "korean_culture_into_llm_alignment_toward_cultural_coherence/"
    t: "Korean Culture into LLM Alignment: Toward Cultural Coherence"
  - u: "large_language_models_should_learn_personalized_rather_than_aggregated_human_pre/"
    t: "Large Language Models Should Learn Personalized Rather Than Aggregated Human Preferences"
  - u: "long_live_the_balance_information_bottleneck_driven_tree-based_policy_optimizati/"
    t: "Long Live The Balance: Information Bottleneck Driven Tree-based Policy Optimization"
  - u: "mesa_improving_moe_safety_alignment_via_decentralized_expertise/"
    t: "MESA: Improving MoE Safety Alignment via Decentralized Expertise"
  - u: "mitigating_reward_hacking_in_rlhf_via_bayesian_non-negative_reward_modeling/"
    t: "Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling"
  - u: "new_wide-net-casting_jailbreak_attacks_risk_large_models/"
    t: "New Wide-Net-Casting Jailbreak Attacks Risk Large Models"
  - u: "operationalising_the_superficial_alignment_hypothesis_via_task_complexity/"
    t: "Operationalising the Superficial Alignment Hypothesis via Task Complexity"
  - u: "picaco_pluralistic_in-context_value_alignment_of_llms_via_total_correlation_opti/"
    t: "PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization"
  - u: "quantifying_the_salience_of_geo-cultural_values_for_pluralistic_safety_alignment/"
    t: "Quantifying the Salience of Geo-Cultural Values for Pluralistic Safety Alignment"
  - u: "reward_shaping_for_inference-time_alignment_a_stackelberg_game_perspective/"
    t: "Reward Shaping for (Inference-Time) Alignment: A Stackelberg Game Perspective"
  - u: "safety_anchor_defending_harmful_fine-tuning_via_geometric_bottlenecks/"
    t: "Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks"
  - u: "sft_overtraining_predicts_rank_inversion_via_entropy_collapse_under_rlvr/"
    t: "SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR"
  - u: "simultaneous_multi-objective_alignment_across_verifiable_and_non-verifiable_rewa/"
    t: "Simultaneous Multi-objective Alignment Across Verifiable and Non-verifiable Rewards"
  - u: "spard_defending_harmful_fine-tuning_attack_via_safety_projection_with_relevance-/"
    t: "SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection"
  - u: "steerable_cultural_preference_optimization_of_reward_models/"
    t: "Steerable Cultural Preference Optimization of Reward Models"
  - u: "steering_beyond_the_support_adversarial_training_on_unsupervised_jailbroken_acti/"
    t: "Steering Beyond the Support: Adversarial Training on Unsupervised Jailbroken Activation Simulation"
  - u: "the_realignment_problem_when_right_becomes_wrong_in_llms/"
    t: "The Realignment Problem: When Right becomes Wrong in LLMs"
item_total: 37
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# ⚖️ Alignment & RLHF

**🧪 ICML2026** · **37** paper notes

📌 **Same area in other venues:** [📷 CVPR2026 (12)](../../CVPR2026/llm_alignment/index.md) · [🔬 ICLR2026 (102)](../../ICLR2026/llm_alignment/index.md) · [💬 ACL2026 (38)](../../ACL2026/llm_alignment/index.md) · [🤖 AAAI2026 (17)](../../AAAI2026/llm_alignment/index.md) · [🧠 NeurIPS2025 (36)](../../NeurIPS2025/llm_alignment/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/llm_alignment/index.md)

🔥 **Top topics:** Alignment/RLHF ×22 · LLM ×5 · Adversarial Robustness ×5 · Reinforcement Learning ×2

**[Adaptive Probe-based Steering for Robust LLM Jailbreaking](adaptive_probe-based_steering_for_robust_llm_jailbreaking.md)**

:   This paper transforms probe-based contrastive steering into a more powerful white-box red-teaming tool. By using adaptive retraining to correct biased probes and automatically setting steering intensity via activation statistics, it significantly exposes the jailbreak vulnerabilities of fortified LLMs.

**[Alignment-Aware Decoding](alignment-aware_decoding.md)**

:   Alignment-Aware Decoding (AAD) directly leverages the token probability ratio of a DPO model relative to an SFT reference model as an implicit alignment reward during inference. Without additional training or external reward models, it generates high-quality aligned responses more stably than greedy, Bo2, and EFT decoding, while also serving as a mechanism to generate synthetic preference data for iterative DPO improvement.

**[Autoregressive Direct Preference Optimization](autoregressive_direct_preference_optimization.md)**

:   The authors observe that DPO's derivation sequence is flawed: it constructs a Bradley-Terry (BT) preference model based on the entire answer first and imposes the autoregressive assumption on the model only afterwards. ADPO **advances** the autoregressive assumption to before the BT model construction by defining energy functions on the prefix closure of the output space. This yields a minimalist new loss that moves the summation sign from **inside** the log-sigmoid to the **outside**. Consequently, it distinguishes two independent length measures for the first time—token length $\mu$ and feedback length $\mu'$-unifying training at any granularity from full answers to individual tokens.

**[Boosting Direct Preference Optimization with Penalization](boosting_direct_preference_optimization_with_penalization.md)**

:   This paper proposes DPOP (Direct Preference Optimization with Penalization), which adds an extra **penalty to the "reference model's own greedy-decoded response" $y_g$** for the same prompt alongside the standard DPO preference loss. A detached gate activates this penalty only when the policy "still ranks the rejected response higher than the chosen response," effectively transforming the unused reference-greedy signal into a valid offline alignment signal. On AlpacaEval 2.0, it exceeds DPO/SimPO/AlphaDPO in length-controlled win rate.

**[Consistency Training Can Entrench Misalignment](consistency_training_can_entrench_misalignment.md)**

:   This paper proposes the "consistency non-neutrality hypothesis." By evaluating 7 consistency training methods across 108 "model organisms," it finds that consistency training is not alignment-neutral—it systematically suppresses fragile reward hacking and emergent misalignment while amplifying stable sycophancy. Distribution shift, rather than score selection, is identified as the primary driver.

**[Curriculum Learning for Safety Alignment](curriculum_learning_for_safety_alignment.md)**

:   This paper proposes Staged-Competence—a DPO safety alignment framework that utilizes "model-specific preference alignment margin" as a difficulty score. It employs a dual curriculum of "staged reference model updates + within-stage competence-based sampling." Across three 8B-scale LLMs, it reduces OOD harmful response rates by an average of 16% and jailbreak success rates by 20%, while maintaining general capabilities and avoiding over-refusal.

**[Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards](decoupling_reasoning_and_confidence_resurrecting_calibration_in_reinforcement_le.md)**

:   This paper theoretically demonstrates that the objectives of "improving accuracy" and "reducing calibration error" in RLVR (e.g., GRPO) training have negatively correlated gradient directions under the Fisher metric and are irreconcilable. It proposes DCPO: allowing the model to explicitly output a verbalized confidence segment after the reasoning trajectory, assigning independent rewards / advantages / masked gradients to reasoning tokens and confidence tokens. While maintaining the same accuracy as GRPO, it reduces the ECE from 0.435 to 0.128 (a 71.6% relative reduction).

**[Efficient Preference Poisoning Attack on Offline RLHF](efficient_preference_poisoning_attack_on_offline_rlhf.md)**

:   The paper proposes a key observation for log-linear DPO: "flipping a single preference label equals adding a fixed vector independent of the policy parameters to the loss gradient." Based on this, targeted poisoning attacks are reduced to a binary sparse approximation problem. Two algorithms are introduced: BAL-A (based on LLL lattice reduction) and BMP-A (based on matching pursuit), along with provable recovery and impossibility conditions.

**[$f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses](f-divergence_regularized_rlhf_two_tales_of_sampling_and_unified_analyses.md)**

:   This paper establishes the first $O(\log T)$ regret and $O(1/T)$ suboptimality gap upper bounds for online RLHF under **general $f$-divergence regularization**. It proposes two sampling strategies: (1) optimism in the face of uncertainty using bonus terms; and (2) a novel **"derivative-as-uncertainty"** perspective, where $f'$ serves as an uncertainty signal to design derivative-based sampling without explicitly estimating confidence bounds in each round.

**[F-TIS: Harnessing Diverse Models in Collaborative GRPO](f-tis_harnessing_diverse_models_in_collaborative_grpo.md)**

:   F-TIS combines "Truncated Importance Sampling (TIS)" with "filtering negative advantage off-policy samples based on KL thresholds" into a single GRPO loss. This allows multiple LLMs—varying in size, expertise, or trainable parameter subsets—to exchange samples during a single decentralized GRPO training session. The approach achieves convergence comparable to pure on-policy training and delivers up to a +12% performance gain on OOD math tasks.

**[GIST: Targeted Data Selection for Instruction Tuning with Gradient Subspace Projection](gist_targeted_data_selection_for_instruction_tuning_via_coupled_optimization_geo.md)**

:   GIST frames "selecting instruction tuning data for a target task" as gradient subspace alignment. It demonstrates that methods like LESS, which use Adam states as a diagonal preconditioner, fail on LoRA due to cross-parameter coupling and low-rank task subspaces. Instead, GIST extracts a task-specific low-rank subspace via SVD of validation gradients and uses cosine similarity for sample selection. It matches or exceeds LESS on MMLU/TydiQA/BBH while requiring only 0.29% of the storage and 25% of the computation time.

**[Implicit Preference Alignment for Human Image Animation](implicit_preference_alignment_for_human_image_animation.md)**

:   The authors propose Implicit Preference Alignment (IPA): a post-training method that requires only "good samples" without constructing positive/negative pairs. By maximizing the KL interval relative to a pre-trained reference model, the method equivalently maximizes implicit rewards. Combined with a HALO module that integrates hand-mask weighting into the loss, it enables a large-scale video DiT to significantly improve hand fidelity in human animation using only 93 selected samples.

**[Implicit Safety Alignment from Crowd Preferences](implicit_safety_alignment_from_crowd_preferences.md)**

:   Addressing the "diverse user goals but shared safety criteria" structure in crowdsourced preference data, the authors prove that traditional reward combination is polluted by majority preferences and sensitive to weights. Instead, they propose Safe Crowd Preference-based RL: using a VAE to encode crowdsourced preferences into latent-conditioned low-level skills, then training a high-level policy to compose these in skill space. This suppresses downstream costs to near-Oracle levels without explicit safety rewards or significant task return degradation.

**[Korean Culture into LLM Alignment: Toward Cultural Coherence](korean_culture_into_llm_alignment_toward_cultural_coherence.md)**

:   Existing cultural safety research primarily focuses on "subtraction" (regulating outputs); this paper introduces a "positive" counterpart—**positively defining** "culturally coherent responses" within the South Korean context. Based on this, it establishes an alignment data pipeline (Korean harm taxonomy seeds → attack mining → safety responses under cultural policy constraints → triple-judge filtering into DPO triplets). DPO fine-tuning consistently improves the Korean cultural safety rates of six open-source LLMs with minimal impact on general capabilities.

**[Large Language Models Should Learn Personalized Rather Than Aggregated Human Preferences](large_language_models_should_learn_personalized_rather_than_aggregated_human_pre.md)**

:   This position paper argues that current RLHF practices, which aggregate diverse human preferences into a single reward signal, essentially optimize for a "representative average" user who does not actually exist. Drawing from social choice theory and cross-demographic empirical evidence, the authors advocate for personalized alignment. They propose a "bounded personalization" framework that maintains universal safety constraints while personalizing only across legitimate dimensions.

**[Long Live The Balance: Information Bottleneck Driven Tree-based Policy Optimization](long_live_the_balance_information_bottleneck_driven_tree-based_policy_optimizati.md)**

:   This paper utilizes Information Bottleneck (IB) theory to propose IB-Score, a step-level metric for quantifying the "exploration-exploitation balance." Based on this, it designs IB-guided tree sampling (IBTree) combined with step-level local/global advantages. On Qwen3-1.7B/8B, it achieves an average improvement of 2.9–3.6% over GRPO while sampling 50% more trajectories under the same token budget.

**[MESA: Improving MoE Safety Alignment via Decentralized Expertise](mesa_improving_moe_safety_alignment_via_decentralized_expertise.md)**

:   MESA reformulates MoE safety alignment as a resource allocation problem of "distributing safety responsibilities across experts." It utilizes KL-regularized Sinkhorn Optimal Transport (OT) to select the lowest-cost subset of experts from the "shoulder region" for SFT. Simultaneously, an OT-constrained routing loss directs safety tokens to these experts, boosting Strata safety scores to 95+% on DeepSeek-V2-Lite / Qwen3-30B-A3B while maintaining reasoning performance (e.g., GSM8K) near baseline levels.

**[Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling](mitigating_reward_hacking_in_rlhf_via_bayesian_non-negative_reward_modeling.md)**

:   This paper reformulates the Bradley–Terry reward model as a generative process of Bayesian Non-negative Factor Analysis (NFA). By simultaneously modeling locally sparse instance latent variables $\bm{\theta}$ and a globally sparse reward dictionary $\Phi$, it suppresses reward hacking caused by shortcut features (e.g., length, style) via a "disentanglement-then-debiasing" mechanism. The entire framework is integrated into modern LLM backbones through amortized variational inference with Weibull reparameterization, consistently outperforming strong baselines like BT, Ensemble, and InfoRM on Unified-Feedback, RewardBench, HHH, and MT-Bench.

**[New Wide-Net-Casting Jailbreak Attacks Risk Large Models](new_wide-net-casting_jailbreak_attacks_risk_large_models.md)**

:   This paper defines and systematically analyzes the "wide-net-casting" jailbreak scenario (where an attacker targets a group of large models simultaneously, succeeding if any one model is breached). Based on this, the authors design an "expert-specialized" joint training method for adversarial generators using exploration-to-exploitation scheduling. This approach pushes the attack success rate to 100% across multiple LLMs/MLLMs when no external defenses are applied, revealing that current single-model jailbreak evaluations significantly underestimate real-world risks.

**[Operationalising the Superficial Alignment Hypothesis via Task Complexity](operationalising_the_superficial_alignment_hypothesis_via_task_complexity.md)**

:   The authors redefine the Superficial Alignment Hypothesis (SAH) using "task complexity"—an algorithmic information-theoretic metric representing the shortest program length required to solve a task at target performance. They unify three disparate lines of evidence (data-efficient, parameter-efficient, and inference-controlled) into a single strategy of finding short programs on the same length–performance Pareto curve. Experimental results indicate that adapting pre-trained models to tasks like mathematical reasoning, machine translation, and instruction following often requires only several kilobytes to megabytes of information, and the role of post-training is to compress the "program length required for high performance" by several orders of magnitude.

**[PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization](picaco_pluralistic_in-context_value_alignment_of_llms_via_total_correlation_opti.md)**

:   PICACO formalizes the challenge of "making an LLM adhere to multiple or even conflicting human values within a single prompt" as maximizing the "conditional Total Correlation (TC) between value sets and responses." Without updating model parameters, it automatically searches for a meta-instruction through an EM-like two-step iteration of "response enhancement + instruction refinement." PICACO outperforms strong baselines like OPRO and Modular Pluralism on five value evaluation sets containing up to 8 combined values across GPT-3.5, LLaMA-3.1-8B, and Gemini-1.5-Flash.

**[Quantifying the Salience of Geo-Cultural Values for Pluralistic Safety Alignment](quantifying_the_salience_of_geo-cultural_values_for_pluralistic_safety_alignment.md)**

:   The authors re-stratify annotators into "cultural zones/quadrants" based on the Inglehart-Welzel cultural map. Using multilevel modeling across 8 safety datasets, they demonstrate that cultural zones significantlly explain variance in safety ratings even after controlling for demographics (age/sex/ethnicity) ($p<0.05$ in 6/8 datasets). They propose a Bayesian "cultural sensitivity score" quantifying that approximately 10% of samples would be mislabeled as safe if a specific cultural quadrant were ignored. Further experiments show that while LLMs are unreliable as rater proxies, they are viable as triage tools for "culturally sensitive samples."

**[Reward Shaping for (Inference-Time) Alignment: A Stackelberg Game Perspective](reward_shaping_for_inference-time_alignment_a_stackelberg_game_perspective.md)**

:   The problem of "which reward model should be used to align LLMs" is modeled as a Stackelberg game. It is proved that the optimal reward is a **per-prompt threshold reward** (giving full score $B$ above the threshold and 0 below). This threshold is efficiently estimated using Monte Carlo sampling from the base model. Finally, the reward is softened via a sigmoid function and seamlessly integrated into inference-time alignment methods like CD/ARGS, increasing the average reward and GPT-4 Win-Tie rate against baselines to over 66% with almost zero additional overhead.

**[Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks](safety_anchor_defending_harmful_fine-tuning_via_geometric_bottlenecks.md)**

:   This paper demonstrates that all existing HFT defenses based on "parameter space constraints" can be bypassed due to parameter redundancy. It proposes Safety Bottleneck Regularization (SBR), which relocates the defense to the unembedding layer—a geometric bottleneck. By anchoring the final-layer hidden state of just a single high-risk prompt, SBR keeps the Harmful Score < 10 under 50-epoch continuous HFT attacks without compromising benign task accuracy.

**[SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR](sft_overtraining_predicts_rank_inversion_via_entropy_collapse_under_rlvr.md)**

:   The authors discover that the industry default rule of "selecting the SFT checkpoint with the highest pass@1 for GRPO" systematically fails in code generation. Longer SFT leads to higher pass@1, but the pass@10 after GRPO descends monotonically (0.806 → 0.481). The root cause is that over-SFT flattens the output distribution, causing within-group advantage variance to zero out and gradients to vanish. The authors identify high-risk checkpoints using a closed-form threshold $p^*(g)$ and a two-stage diagnosis: "pre-training entropy screening + early entropy monitoring."

**[Simultaneous Multi-objective Alignment Across Verifiable and Non-verifiable Rewards](simultaneous_multi-objective_alignment_across_verifiable_and_non-verifiable_rewa.md)**

:   MAHALO integrates "standardized PRM training + Multi-Action-Head DPO + PRM-guided decoding with KV-cache continuation" into a unified framework. This allows a single LLM to be simultaneously aligned across three categories: mathematics (verifiable), human values (non-verifiable), and Socratic tutoring (interactive), while enabling smooth preference switching during inference through head weights and PRM selection.

**[SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection](spard_defending_harmful_fine-tuning_attack_via_safety_projection_with_relevance-.md)**

:   SPARD combines "Safety-Projected Alternating Gradient (SPAG)" and "Relevance-Diversity DPP Safety Data Selection" to explicitly formulate "post-fine-tuning safety constraints" as a constrained optimization problem. It updates parameters for utility first and then uses a closed-form projection to pull them back into the safety half-space. By using only 3% task-relevant yet diverse safety samples, it reduces the average ASR of four harmful fine-tuning attacks from 87.93% (SFT) to 9.45% with negligible impact on downstream performance.

**[Steerable Cultural Preference Optimization of Reward Models](steerable_cultural_preference_optimization_of_reward_models.md)**

:   SCPO uses a "global reward model" as a reference frame. It first **filters** out general preferences in minority groups that align with global consensus, leaving only preferences with genuine cultural differences. It then applies **inverse divergence weighting** to reduce the influence of extreme outlier preferences. This approach trains steerable reward models that represent specific minority perspectives without being excessively biased—improving minority reward models by up to ~7 points across 7 countries in PRISM and GlobalOpinionQA datasets, while achieving 170%–280% higher data efficiency compared to full fine-tuning.

**[Steering Beyond the Support: Adversarial Training on Unsupervised Jailbroken Activation Simulation](steering_beyond_the_support_adversarial_training_on_unsupervised_jailbroken_acti.md)**

:   This paper addresses the failure of supervised safety steering on unseen jailbreak attacks by proposing "unsupervised latent direction discovery + bi-level adversarial training" to simulate new jailbroken states in the activation space. These simulated states are used as adversarial samples to train an OT potential function (whose gradient forms a spatially varying steering field). The method reduces the attack success rate to under 5% across three LLMs and six classic jailbreak types while maintaining benign utility.

**[The Realignment Problem: When Right becomes Wrong in LLMs](the_realignment_problem_when_right_becomes_wrong_in_llms.md)**

:   This paper formalizes the problem of "what to do when policies change after model deployment" as the Realignment problem. It proposes the TRACE framework: using a stronger proxy model to classify existing preference pairs into three categories (Invert / Punish / Retain), followed by surgical realignment using a hybrid IPO+NPO+KL objective. This approach allows models to adapt to policy drift without a new round of human annotation.

**[Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance](toward_stable_value_alignment_introducing_independent_modules_for_consistent_val.md)**

:   This paper proposes SVGT, which shifts value alignment from "embedding in backbone parameters/activations" to "attaching an independent value module." It first continuously determines safety directions within an isolated value space based on current hidden states, and then explicitly guides generation trajectories using a set of learnable Bridge Tokens as attention anchors. Across four backbones, it consistently reduces toxicity scores by over 70% with almost no loss in fluency.

**[Towards Context-Invariant Safety Alignment for Large Language Models](towards_context-invariant_safety_alignment_for_large_language_models.md)**

:   The authors propose AIR (Anchor Invariance Regularization), which treats verifiable prompts as "anchors" and utilizes stop-gradients to pull open-ended variants toward the anchor's performance. Inserted as an auxiliary loss in GRPO, it improves OOD group-level consistency across safety, moral, and mathematical domains by an average of 33.49% and ID by 12.71%.

**[HRC + DSPPO: Separating Transitive and Cyclic Preferences via Game-Theoretic Decomposition](transitivity_meets_cyclicity_explicit_preference_decomposition_for_dynamic_large.md)**

:   HRC explicitly decomposes human preferences into orthogonal "transitive scalar components" (BT model) + "cyclic vector components" (GPM). Using game-theoretic decomposition theorems, it proves this hybrid form preserves dominant candidates while modeling Rock-Paper-Scissors (RPS) style cycles. Complemented by the time-varying game DSPPO, the alignment process transitions from "stabilizing the transitive backbone" to "learning cyclic details" to reach a Nash equilibrium—achieving a 1.23% average gain for Gemma-2B-it on RewardBench 2 and reaching a 44.75% LC win-rate on AlpacaEval 2.0.

**[TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning](truthrl_incentivizing_truthful_llms_via_reinforcement_learning.md)**

:   Aiming to resolve the dilemma where optimizing solely for accuracy encourages blind guessing while forcing refusal leads to over-conservatism, TruthRL directly optimizes truthfulness using a ternary reward ("Correct / Hallucination / Refusal") via GRPO. It reduces hallucination rates from 43.5% to 19.4% and increases truthfulness scores from 5.3% to 37.2%.

**[UDM-GRPO: Stable and Efficient GRPO for Unified Discrete Diffusion Models](udm-grpo_stable_and_efficient_group_relative_policy_optimization_for_uniform_dis.md)**

:   The first successful integration of GRPO into discrete diffusion models (UDM) is achieved by defining the final clean sample as the action and reconstructing trajectories via the forward process, addressing training instability and reaching SOTA on benchmarks like GenEval.

**[VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models](valueflow_toward_pluralistic_and_steerable_value-based_alignment_in_large_langua.md)**

:   To address the challenges of LLMs being highly unstable at scoring absolute value intensities and unable to control the strength of value expression, this paper proposes VALUEFLOW—a unified framework connecting "extraction-evaluation-steering." Its core consists of a hierarchical value embedding space (HIVES), a value intensity database (VIDB) aggregated via Plackett–Luce ranking, and an anchor-ranking-based intensity evaluator. The study systematically characterizes the steerability of LLMs across 10 models and 4 value theories.

**[When Distance Distracts: Representation Distance Bias in BT-Loss for Reward Models](when_distance_distracts_representation_distance_bias_in_bt-loss_for_reward_model.md)**

:   This paper decomposes the gradient norm of the Bradley-Terry (BT) reward model loss into two terms: "prediction error × representation distance." It points out that representation distance can overshadow the prediction error—hard-to-distinguish pairs with similar representations receive only weak updates even if misranked. Consequently, the authors propose NormBT, which uses a pairwise weight inversely proportional to the representation distance to restore update intensity to the prediction error, leading to an average improvement of over 5% in the Reasoning category of RewardBench.
