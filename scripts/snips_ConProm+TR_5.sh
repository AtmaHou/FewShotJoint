#!/usr/bin/env bash
echo usage: pass gpu id list as param, split with ,
echo eg: source run_bert_siamese.sh 3,4 stanford

gpu_list=$1

# Comment one of follow 2 to switch debugging status
#do_debug=--do_debug
do_debug=

#restore=--restore_cpt
restore=

task=slu
#task=sl

#use_schema=--use_schema
use_schema=

#no_embedder_grad=--no_embedder_grad
no_embedder_grad=

slu_model_type=proto_merge

proto_merge_type_lst=(add_attention)
attn_hidden_size=256
pm_rpl=
pm_use_attn=-pm_ua

split_metric_lst=(intent)
metric_activation_lst=(none)
metric_dim=768

#proto_update_lst=(slu intent slot)
proto_update_lst=(slot)

pr_nm_lst=(none)
# pr_nm_lst=(norm softmax none)

#pr_scl_lst=(fix)
 pr_scl_lst=(learn)

#proto_scale_r_lst=(0.5)
 proto_scale_r_lst=(0.5)

# ======= dataset setting ======
dataset_lst=($2 $3)
support_shots_lst=(5)

#cross_data_id_lst=(0)  # for smp
cross_data_id_lst=(2)  # for debug
#cross_data_id_lst=(1 2 3 4 5 6 7)  # for snips
#cross_data_id_lst=(1 2 3 4 5 6 7)  # for atis

slu_regular_lst=(contrastive)
slu_regular_rate_lst=(1)
#margin_lst=(10)
#margin_lst=(10 5 3)
#margin_lst=(5 1 0.5)
margin_lst=(1)
with_homo=-wh

# ====== train & test setting ======
#seed_lst=(0)
seed_lst=(6150 6151 6152)

#lr_lst=(0.000001 0.000005 0.00005)
lr_lst=(0.00001)

clip_grad=5

decay_lr_lst=(0.5)
#decay_lr_lst=(-1)

#upper_lr_lst=( 0.5 0.01 0.005)
#upper_lr_lst=(0.01)
upper_lr_lst=(0.001)
#upper_lr_lst=(0.0001)
#upper_lr_lst=(0.0005)
#upper_lr_lst=(0.1)
#upper_lr_lst=(0.001 0.1)

#fix_embd_epoch_lst=(1)
fix_embd_epoch_lst=(-1)
#fix_embd_epoch_lst=(1 2)

#warmup_epoch=2
#warmup_epoch=1
warmup_epoch=-1


train_batch_size_lst=(4)
#train_batch_size_lst=(8)
#train_batch_size_lst=(4 8)

#test_batch_size=16
test_batch_size=2
#test_batch_size=8

#grad_acc=2
grad_acc=4
epoch=8

# ==== model setting =========
# ---- encoder setting -----

#embedder=electra
embedder=bert
#embedder=sep_bert


# --------- emission setting --------
#emission_lst=(mnet)
#emission_lst=(tapnet)
#emission_lst=(proto_with_label)
emission_lst=(proto)
#emission_lst=(mnet record_proto)


#similarity=cosine
#similarity=l2
similarity=dot

emission_normalizer=none
#emission_normalizer=softmax
#emission_normalizer=norm

#emission_scaler=none
#emission_scaler=fix
emission_scaler=learn
#emission_scaler=relu
#emission_scaler=exp

do_div_emission=-dbt
#do_div_emission=

ems_scale_rate_lst=(0.01)
#ems_scale_rate_lst=(0.01 0.02 0.05 0.005)

label_reps=sep
#label_reps=cat

ple_normalizer=none
ple_scaler=fix
#ple_scale_r=0.5
ple_scale_r_lst=(0.5)
#ple_scale_r=1
#ple_scale_r=0.01

#tap_random_init=--tap_random_init
#tap_mlp=
#tap_mlp=--tap_mlp
#emb_log=
#emb_log=--emb_log

# ------ decoder setting -------
decoder_lst=(rule)
#decoder_lst=(sms)
#decoder_lst=(crf)
#decoder_lst=(crf sms)

# -------- SC decoder setting --------

# ======= default path (for quick distribution) ==========
# bert base path
pretrained_model_path=/the/path/of/your/bert/
pretrained_vocab_path=/the/path/of/your/bert/vocab.txt

# data path
base_data_dir=/the/path/of/ACL2021_snips_data/  # snips data

echo [START] set jobs on dataset [ ${dataset_lst[@]} ] on gpu [ ${gpu_list} ]
# === Loop for all case and run ===
for seed in ${seed_lst[@]}
do
  for dataset in ${dataset_lst[@]}
  do
    for support_shots in ${support_shots_lst[@]}
    do
        for train_batch_size in ${train_batch_size_lst[@]}
        do
              for decay_lr in ${decay_lr_lst[@]}
              do
                  for fix_embd_epoch in ${fix_embd_epoch_lst[@]}
                  do
                      for lr in ${lr_lst[@]}
                      do
                          for upper_lr in ${upper_lr_lst[@]}
                          do
                                for ems_scale_r in ${ems_scale_rate_lst[@]}
                                do
                                    for emission in ${emission_lst[@]}
                                    do
                                        for ple_scale_r in ${ple_scale_r_lst[@]}
                                        do
                                            for decoder in ${decoder_lst[@]}
                                            do
                                                for cross_data_id in ${cross_data_id_lst[@]}
                                                do
                                                    for proto_merge_type in ${proto_merge_type_lst[@]}
                                                    do
                                                        for proto_update in ${proto_update_lst[@]}
                                                        do
                                                            for pr_nm in ${pr_nm_lst[@]}
                                                            do
                                                                for pr_scl in ${pr_scl_lst[@]}
                                                                do
                                                                    for proto_scale_r in ${proto_scale_r_lst[@]}
                                                                    do
	                                                                    for split_metric in ${split_metric_lst[@]}
	                                                                    do
		                                                                    for metric_activation in ${metric_activation_lst[@]}
		                                                                    do
							                                                    for slu_regular in ${slu_regular_lst[@]}
							                                                    do
							                                                        for slu_regular_rate in ${slu_regular_rate_lst[@]}
							                                                        do
								                                                        for margin in ${margin_lst[@]}
								                                                        do
								                                                            # model names
								                                                            model_name=ConProm+TR.pm_rg_${slu_regular}_r_${slu_regular_rate}_m_${margin}${with_homo}.sp_${split_metric}_${metric_activation}_${metric_dim}.${slu_model_type}.${proto_merge_type}_hd_${attn_hidden_size}.update_${proto_update}${pm_use_attn}.ep_${epoch}.pr_nm_${pr_nm}.scl_${pr_scl}.scl_r_${proto_scale_r}.${task}.ga_${grad_acc}.bs_${train_batch_size}.dec_${decoder}${no_embedder_grad}${use_schema}${do_debug}

								                                                            data_dir=${base_data_dir}xval_${dataset}_5shot_8qs/
								                                                            file_mark=${dataset}.shots_${support_shots}.cross_id_${cross_data_id}.m_seed_${seed}
								                                                            train_file_name=${dataset}_train_${cross_data_id}.json
								                                                            dev_file_name=${dataset}_valid_${cross_data_id}.json
								                                                            test_file_name=${dataset}_test_${cross_data_id}.json

								                                                            echo [CLI]
								                                                            echo Model: ${model_name}
								                                                            echo Task:  ${file_mark}
								                                                            echo [CLI]
								                                                            export OMP_NUM_THREADS=2  # threads num for each task
								                                                            CUDA_VISIBLE_DEVICES=${gpu_list} python main.py ${do_debug} \
								                                                                --task ${task} \
								                                                                --seed ${seed} \
								                                                                --do_train \
								                                                                --do_predict \
						                                                                        --slu_regular ${slu_regular} \
						                                                                        --slu_regular_rate ${slu_regular_rate} \
						                                                                        --margin ${margin} \
						                                                                        ${with_homo} \
								                                                                ${label_wp} \
			                                                                                    ${pm_use_attn} \
								                                                                ${no_embedder_grad} \
								                                                                --slu_model_type ${slu_model_type} \
								                                                                --proto_merge_type ${proto_merge_type} \
								                                                                --proto_update ${proto_update} \
								                                                                --attn_hidden_size ${attn_hidden_size} \
								                                                                ${proto_replace} \
					                                                                            --split_metric ${split_metric} \
					                                                                            --metric_dim ${metric_dim} \
					                                                                            --metric_activation ${metric_activation} \
					                                                                            -pr_nm ${pr_nm} \
					                                                                            -pr_scl ${pr_scl} \
					                                                                            --proto_scale_r ${proto_scale_r} \
								                                                                --train_path ${data_dir}${train_file_name} \
								                                                                --dev_path ${data_dir}${dev_file_name} \
								                                                                --test_path ${data_dir}${test_file_name} \
								                                                                --output_dir ${data_dir}${model_name}.DATA.${file_mark} \
								                                                                --bert_path ${pretrained_model_path} \
								                                                                --bert_vocab ${pretrained_vocab_path} \
								                                                                --train_batch_size ${train_batch_size} \
								                                                                --cpt_per_epoch 4 \
								                                                                --delete_checkpoint \
								                                                                --gradient_accumulation_steps ${grad_acc} \
								                                                                --num_train_epochs ${epoch} \
								                                                                --learning_rate ${lr} \
								                                                                --decay_lr ${decay_lr} \
								                                                                --upper_lr ${upper_lr} \
								                                                                --clip_grad ${clip_grad} \
								                                                                --fix_embed_epoch ${fix_embd_epoch} \
								                                                                --warmup_epoch ${warmup_epoch} \
								                                                                --test_batch_size ${test_batch_size} \
								                                                                --context_emb ${embedder} \
								                                                                ${use_schema} \
								                                                                --label_reps ${label_reps} \
								                                                                --emission ${emission} \
								                                                                --similarity ${similarity} \
								                                                                -e_nm ${emission_normalizer} \
								                                                                -e_scl ${emission_scaler} \
								                                                                --ems_scale_r ${ems_scale_r} \
								                                                                -ple_nm ${ple_normalizer} \
								                                                                -ple_scl ${ple_scaler} \
								                                                                --ple_scale_r ${ple_scale_r} \
								                                                                ${do_div_emission} \
								                                                                --decoder ${decoder} \
								                                                                --transition learn > ./result/${model_name}.DATA.${file_mark}.log
								                                                            echo [CLI]
								                                                            echo Model: ${model_name}
								                                                            echo Task:  ${file_mark}
								                                                            echo [CLI]
								                                                        done
								                                                    done
						                                                        done
					                                                        done
					                                                    done
			                                                        done
			                                                    done
			                                                done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done


echo [FINISH] set jobs on dataset [ ${dataset_lst[@]} ] on gpu [ ${gpu_list} ]
