export VLLM_ATTENTION_BACKEND=XFORMERS
export project_name="lotus-research"
export experiment_name="deepresearcher-evaluate"

    
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=./data/train.parquet \
    data.val_files=./data/papers.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=30767 \
    data.max_response_length=2000 \
    +data.max_model_len=32768 \
    data.data_writing_file=./signal/data.json \
    data.signal_writing_file=./signal/signal.json \
    actor_rollout_ref.model.path=GAIR/DeepResearcher-7b \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4096 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=GAIR/DeepResearcher-7b \
    critic.ppo_micro_batch_size_per_gpu=2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    +trainer.val_before_train=true \
    +trainer.val_only=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.remove_previous_ckpt_in_save=false \
    agent_grpo.n=16 \
    max_turns=2 \
    search_engine=online_search \
    trainer.total_epochs=1 2>&1 | tee ./"${project_name}_${experiment_name}.log"

