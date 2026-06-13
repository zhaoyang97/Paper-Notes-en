---
title: >-
  ICLR2026 LLM Evaluation Papers · 29 Notes
description: >-
  29 ICLR2026 papers in the LLM Evaluation area, covering LLM, Diffusion Models, Reasoning, Multimodal/VLM, Adversarial Robustness, Object Tracking and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ICLR2026"
  - "LLM Evaluation"
  - "AI paper notes"
  - "paper summaries"
  - "LLM"
  - "Diffusion Models"
  - "Reasoning"
  - "Multimodal/VLM"
  - "Adversarial Robustness"
  - "Object Tracking"
item_list:
  - u: "adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size/"
    t: "AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size"
  - u: "anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni/"
    t: "AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning"
  - u: "aside_architectural_separation_of_instructions_and_data_in_language_models/"
    t: "ASIDE: Architectural Separation of Instructions and Data in Language Models"
  - u: "astabench_benchmarking_ai_agents/"
    t: "AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite"
  - u: "benchmarking_overton_pluralism_in_llms/"
    t: "Benchmarking Overton Pluralism in LLMs"
  - u: "biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation/"
    t: "BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation"
  - u: "can_vision_language_models_assess_graphic_design_aesthetics_a_benchmark_evaluati/"
    t: "Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective"
  - u: "can_you_hear_me_now_a_benchmark_for_long-range_graph_propagation_and_beyond/"
    t: "Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation and Beyond"
  - u: "dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science/"
    t: "DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science"
  - u: "doubly-robust_llm-as-a-judge_externally_valid_estimation_with_imperfect_personas/"
    t: "Doubly-Robust LLM-as-a-Judge: Externally Valid Estimation with Imperfect Personas"
  - u: "enabling_fine-grained_operating_points_for_black-box_llms/"
    t: "Enabling Fine-Grained Operating Points for Black-Box LLMs"
  - u: "guidedsampling_steering_llms_towards_diverse_candidate_solutions_at_inference-ti/"
    t: "GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time"
  - u: "how_reliable_is_language_model_micro-benchmarking/"
    t: "How Reliable is Language Model Micro-Benchmarking?"
  - u: "human-llm_collaborative_feature_engineering_for_tabular_data/"
    t: "Human-LLM Collaborative Feature Engineering for Tabular Learning"
  - u: "in-context_learning_for_pure_exploration/"
    t: "In-Context Learning for Pure Exploration"
  - u: "in-context_learning_of_temporal_point_processes_with_foundation_inference_models/"
    t: "In-Context Learning of Temporal Point Processes with Foundation Inference Models"
  - u: "log_probability_tracking_of_llm_apis/"
    t: "Log Probability Tracking of LLM APIs"
  - u: "multi-llm_adaptive_conformal_inference_for_reliable_llm_responses/"
    t: "Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses"
  - u: "preference_leakage_a_contamination_problem_in_llm-as-a-judge/"
    t: "Preference Leakage: A Contamination Problem in LLM-as-a-judge"
  - u: "prompt_and_parameter_co-optimization_for_large_language_models/"
    t: "Prompt and Parameter Co-Optimization for Large Language Models"
  - u: "rankllm_weighted_ranking_of_llms_by_quantifying_question_difficulty/"
    t: "RankLLM: Weighted Ranking of LLMs by Quantifying Question Difficulty"
  - u: "same_content_different_representations_a_controlled_study_for_t/"
    t: "Same Content, Different Representations: A Controlled Study for Table QA"
  - u: "simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_agents/"
    t: "SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home Agents"
  - u: "subliminal_signals_in_preference_labels/"
    t: "Subliminal Signals in Preference Labels"
  - u: "talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis/"
    t: "Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis"
  - u: "truthfulness_despite_weak_supervision_evaluating_and_training_llms_using_peer_pr/"
    t: "Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction"
  - u: "unpacking_human_preference_for_llms_demographically_aware_evaluation_of_long-fo/"
    t: "Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework"
  - u: "vcache_verified_semantic_prompt_caching/"
    t: "vCache: Verified Semantic Prompt Caching"
  - u: "when_to_ensemble_identifying_token-level_points_for_stable_and_fast_llm_ensembli/"
    t: "When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling"
item_total: 29
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# 📊 LLM Evaluation

**🔬 ICLR2026** · **29** paper notes

📌 **Same area in other venues:** [🧪 ICML2026 (24)](../../ICML2026/llm_evaluation/index.md) · [💬 ACL2026 (91)](../../ACL2026/llm_evaluation/index.md) · [🤖 AAAI2026 (16)](../../AAAI2026/llm_evaluation/index.md) · [🧠 NeurIPS2025 (37)](../../NeurIPS2025/llm_evaluation/index.md) · [📹 ICCV2025 (27)](../../ICCV2025/llm_evaluation/index.md)

🔥 **Top topics:** LLM ×9

**[AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size.md)**

:   Through statistical analysis of token confidence dynamics during the denoising process of diffusion language models (dLLMs), this work identifies a "Volatility Band" (VB) region that encodes local semantic structure in text. Building on this observation, it proposes AdaBlock-dLLM—a training-free, plug-and-play adaptive block size scheduler that aligns block boundaries in semi-autoregressive decoding with natural semantic steps, achieving up to 5.3% accuracy improvement at the same throughput.

**[AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)**

:   This paper introduces AnesSuite, the first comprehensive dataset suite for anesthesiology reasoning, comprising AnesBench—an evaluation benchmark of 7,972 bilingual multiple-choice questions organized into three cognitive difficulty levels—and three training datasets (AnesCorpus/AnesQA/AnesR1). The Morpheus models trained on this suite via SFT+GRPO enable a 7B model to match a 14B baseline, while revealing significant bottlenecks of state-of-the-art LLMs on complex clinical reasoning (System 2).

**[ASIDE: Architectural Separation of Instructions and Data in Language Models](aside_architectural_separation_of_instructions_and_data_in_language_models.md)**

:   This paper proposes ASIDE, an architectural modification that distinguishes instructions from data at the token embedding level via orthogonal rotation. Requiring only changes to the forward pass and training on standard instruction fine-tuning data, ASIDE significantly improves instruction-data separation and robustness against prompt injection without any dedicated safety training.

**[AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](astabench_benchmarking_ai_agents.md)**

:   The AI2 team identifies five methodological flaws in existing scientific research agent benchmarks and introduces AstaBench, the first agent evaluation suite covering the full scientific research pipeline. AstaBench comprises 4 categories and 11 sub-benchmarks with 2,400+ questions, a production-grade controllable search tool backed by Semantic Scholar, and 9 research-optimized Asta Agent baselines. It conducts the largest systematic evaluation to date across 57 agents (22 types), finding that despite progress on individual tasks such as literature retrieval, AI remains far from meeting the demands of end-to-end scientific research assistance.

**[Benchmarking Overton Pluralism in LLMs](benchmarking_overton_pluralism_in_llms.md)**

:   This paper proposes the OvertonBench framework, which formalizes Overton pluralism as a set-coverage metric called OvertonScore through a large-scale human study (1,208 demographically representative U.S. participants, 60 subjective questions, 8 LLMs). All evaluated models score only 0.35–0.41 (theoretical maximum: 1.0), and an automated evaluation tool achieving high correlation with human judgments (ρ=0.88) is constructed.

**[BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)**

:   This paper proposes BiasScope, a fully LLM-driven iterative framework that automatically discovers previously unknown biases in LLM-as-a-Judge evaluation at scale. Based on the discovered biases, the authors construct JudgeBench-Pro, a more challenging benchmark on which even powerful LLM judges exceed 50% error rate.

**[Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective](can_vision_language_models_assess_graphic_design_aesthetics_a_benchmark_evaluati.md)**

:   This paper proposes AesEval-Bench, the first benchmark for systematically evaluating VLMs on graphic design aesthetics (4 dimensions × 12 indicators × 3 tasks). It finds that existing VLMs—including reasoning-augmented models—perform poorly on design aesthetics, and constructs training data via human-guided VLM labeling combined with indicator-grounded reasoning. Fine-tuning a 7B model with this data surpasses GPT-5 on the precise localization task.

**[Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation and Beyond](can_you_hear_me_now_a_benchmark_for_long-range_graph_propagation_and_beyond.md)**

:   This paper proposes the ECHO benchmark, comprising 3 synthetic tasks and 2 real-world chemistry tasks grounded in density functional theory (DFT), requiring graph neural networks to propagate information effectively over 17–40 hops. The benchmark systematically evaluates the long-range propagation capabilities of 11 GNN architectures.

**[DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)**

:   DARE-bench is a large-scale verifiable benchmark for data science tasks, comprising 6,300 Kaggle-derived tasks that support evaluation across two dimensions—ML modeling and instruction following—along with training data for SFT and RL. SFT improves Qwen3-32B by 1.83×, while RL improves Qwen3-4B by more than 8×.

**[Doubly-Robust LLM-as-a-Judge: Externally Valid Estimation with Imperfect Personas](doubly-robust_llm-as-a-judge_externally_valid_estimation_with_imperfect_personas.md)**

:   This paper proposes a doubly-robust estimation framework that combines imperfect LLM persona ratings with human annotations subject to sampling bias, yielding statistically valid estimates of GenAI system quality in the simultaneous presence of covariate shift and selection bias.

**[Enabling Fine-Grained Operating Points for Black-Box LLMs](enabling_fine-grained_operating_points_for_black-box_llms.md)**

:   This paper identifies that verbalized probabilities from black-box LLMs produce only 16–23 unique values (low-cardinality problem), resulting in coarse PR/ROC curves that prevent fine-grained threshold tuning. By injecting parameterized noise and an optional MLP correction, the number of unique values increases from 16 to 20,000+, matching the performance of 20-sample ensembles with only 1–2 API calls.

**[GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time](guidedsampling_steering_llms_towards_diverse_candidate_solutions_at_inference-ti.md)**

:   This paper proposes GuidedSampling, an inference-time algorithm that explicitly decouples the implicit exploration and generation process of repeated sampling (RS) into two stages: iteratively generating diverse problem-solving concepts/theorems, followed by generating candidate solutions conditioned on each concept. The method achieves an average improvement of ~21.6% on pass@50 and ~9.7% on pass@5 after fine-tuning.

**[How Reliable is Language Model Micro-Benchmarking?](how_reliable_is_language_model_micro-benchmarking.md)**

:   This paper proposes Minimum Detectable Ability Difference (MDAD) as a meta-evaluation metric, systematically demonstrating that micro-benchmarks at extremely small scales cannot reliably distinguish model pairs with small performance gaps, and that random sampling becomes competitive with carefully designed micro-benchmark methods once the sample size reaches ~250.

**[Human-LLM Collaborative Feature Engineering for Tabular Learning](human-llm_collaborative_feature_engineering_for_tabular_data.md)**

:   This paper proposes a human-LLM collaborative feature engineering framework that decouples the proposal and selection of feature operations. A Bayesian neural network models operation utility and uncertainty to guide selection, with selective human preference feedback incorporated when appropriate. The framework achieves 8.96%–11.23% average error rate reduction across 18 tabular datasets.

**[In-Context Learning for Pure Exploration](in-context_learning_for_pure_exploration.md)**

:   This paper proposes ICPE (In-Context Pure Exploration), an in-context learning framework that combines supervised learning and reinforcement learning. Using a Transformer trained directly from experience, ICPE learns exploration policies for active sequential hypothesis testing and pure exploration problems, achieving near-optimal instance-adaptive algorithmic performance without explicit modeling of the information structure.

**[In-Context Learning of Temporal Point Processes with Foundation Inference Models](in-context_learning_of_temporal_point_processes_with_foundation_inference_models.md)**

:   This paper proposes FIM-PP — the first foundation inference model for marked temporal point processes (MTPP). A Transformer is pretrained on 72K synthetic point processes (14.4M events) to perform in-context inference of conditional intensity functions. In zero-shot settings, FIM-PP matches the performance of specialized models trained for hours; after a few minutes of fine-tuning, it achieves state-of-the-art results on multi-event prediction across four real-world datasets.

**[Log Probability Tracking of LLM APIs](log_probability_tracking_of_llm_apis.md)**

:   This paper proposes Logprob Tracking (LT), a method that detects subtle changes in LLM APIs (e.g., single-step fine-tuning) using only the log probabilities of a single-token input and single-token output. LT achieves sensitivity 2–3 orders of magnitude higher than existing methods at 1000× lower cost.

**[Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses](multi-llm_adaptive_conformal_inference_for_reliable_llm_responses.md)**

:   This paper proposes MACI (Multi-LLM Adaptive Conformal Inference), which combines a **cumulative-product conformity score**, a multi-LLM ensemble for factuality scoring, and group-conditional calibration to significantly improve the retention rate of factual claims in LLM responses while strictly guaranteeing user-specified error rates.

**[Preference Leakage: A Contamination Problem in LLM-as-a-judge](preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)**

:   This paper is the first to formally define and systematically investigate **Preference Leakage** in LLM-as-a-Judge — when the synthetic data generator $M_G$ and the judge $M_J$ are related (same model / inheritance / same family), the judge exhibits systematic preference toward the "associated student model." Under the same-model scenario, PLS reaches 28.7% on Arena-Hard, and this bias is more subtle and harder to detect than egocentric bias.

**[Prompt and Parameter Co-Optimization for Large Language Models](prompt_and_parameter_co-optimization_for_large_language_models.md)**

:   This paper proposes MetaTuner, a framework that simultaneously generates prompts and LoRA parameters via a shared meta encoder, unifying discrete prompt optimization and continuous parameter fine-tuning into an end-to-end jointly optimizable framework, achieving substantial improvements over independently optimized methods on mathematical reasoning and question answering tasks.

**[RankLLM: Weighted Ranking of LLMs by Quantifying Question Difficulty](rankllm_weighted_ranking_of_llms_by_quantifying_question_difficulty.md)**

:   This paper proposes RankLLM, a non-parametric framework based on bidirectional score propagation over a directed bipartite graph, which jointly estimates question difficulty and model competency to achieve difficulty-aware LLM ranking, reaching 90% agreement with human judgments.

**[Same Content, Different Representations: A Controlled Study for Table QA](same_content_different_representations_a_controlled_study_for_t.md)**

:   The first controlled study that systematically evaluates the robustness of NL2SQL, LLM, and hybrid approaches under varying table size, schema quality, and query complexity by changing only the representation format (structured vs. semi-structured) while holding table content constant, demonstrating that representation format is a first-order factor in Table QA performance.

**[SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home Agents](simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_agents.md)**

:   SimuHome is a high-fidelity smart home simulator built on the Matter protocol and a 600-episode evaluation benchmark supporting dynamic environmental variable updates and time-accelerated scheduling evaluation, revealing that workflow scheduling remains the most persistent challenge for current LLM agents.

**[Subliminal Signals in Preference Labels](subliminal_signals_in_preference_labels.md)**

:   This paper demonstrates that preference labels can serve as a covert communication channel: even when a student model generates semantically irrelevant numeric sequences, a biased judge model can transmit subliminal behavioral tendencies to the student model through binary preference labels alone, and this transmission is amplified under iterative alignment.

**[Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)**

:   This paper proposes TED (Talk, Evaluate, Diagnose), a framework that achieves user-aware dynamic agent evaluation via general, reusable expert/non-expert persona templates; enables fine-grained efficiency assessment through grading notes, LLM-as-judge scoring, and novel metrics such as MaxProgressRate@k; and provides actionable improvement feedback via automated error discovery and clustering. Experiments on τ²-bench and ToolSandbox reveal new insights into agent performance.

**[Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction](truthfulness_despite_weak_supervision_evaluating_and_training_llms_using_peer_pr.md)**

:   This paper proposes applying the Peer Prediction mechanism from game theory to LLM evaluation and training. By measuring the mutual predictability of participants' answers, the method distinguishes honest from deceptive responses without requiring ground-truth labels, thereby incentivizing truthfulness. It exhibits a striking *inverse scaling* property — weaker experts are actually more resistant to deception by stronger models.

**[Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](unpacking_human_preference_for_llms_demographically_aware_evaluation_of_long-fo.md)**

:   This paper proposes the HUMAINE framework, which conducts multi-dimensional (5-axis), multi-turn human preference evaluations of 28 SOTA models using 23,404 demographically stratified participants. A hierarchical Bayesian BTD model reveals that age is the largest driver of preference heterogeneity (mean rank shift ±2.8), demonstrating that a single aggregated leaderboard is insufficient to reflect the true preferences of diverse populations.

**[vCache: Verified Semantic Prompt Caching](vcache_verified_semantic_prompt_caching.md)**

:   This paper proposes vCache — the first semantic caching system with **user-defined error-rate guarantees** — which employs online learning to independently estimate the optimal similarity threshold for each cached embedding. Without any pre-training, vCache achieves up to a 12.5× improvement in cache hit rate and a 26× reduction in error rate while satisfying correctness constraints.

**[When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling](when_to_ensemble_identifying_token-level_points_for_stable_and_fast_llm_ensembli.md)**

:   This paper proposes SAFE (Stable And Fast LLM Ensembling), which selectively ensembles multiple heterogeneous-tokenizer LLMs at the token level via a Generate-Verify-Ensemble loop. SAFE addresses OOV-like contamination caused by tokenization mismatch in long-sequence generation, achieving performance gains by ensembling on fewer than 1% of tokens—improving UniTE from 59.6% to 77.4% on MATH500.
