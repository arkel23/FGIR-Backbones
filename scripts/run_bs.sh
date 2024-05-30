#!/bin/bash

run='False'
run_lr='False'
run_seed='False'
cal_ap_only='False'
augs='weakaugs'
ls=''
sd=''
freeze_backbone=''

batch_size=64
serial=1
seed=1
lr=0.003

dataset_name='cub'
model_name='vit_b16'
cal_topk_crop=''
others=''

lr_array=('0.03' '0.01' '0.003' '0.001')
seed_array=('1' '10')

VALID_ARGS=$(getopt  -o '' --long run,run_lr,run_seed,cal_ap_only,med_augs,ls,sd,freeze_backbone,batch_size:,serial:,seed:,lr:,dataset_name:,model_name:,cal_topk_crop:,others: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    --run)
        run='True'
        shift 1
        ;;
    --run_lr)
        run_lr='True'
        shift 1
        ;;
    --run_seed)
        run_seed='True'
        shift 1
        ;;
    --cal_ap_only)
        cal_ap_only='True'
        shift 1
        ;;
    --med_augs)
        augs='medaugs'
        shift 1
        ;;
    --ls)
        ls=' --ls'
        shift 1
        ;;
    --sd)
        sd=' --sd 0.1'
        shift 1
        ;;
    --freeze_backbone)
        freeze_backbone=' --freeze_backbone'
        lr_array=('0.3' '0.1' '0.03' '0.01' '0.003')
        shift 1
        ;;
    --batch_size)
        batch_size=${2}
        shift 2
        ;;
    --serial)
        serial=${2}
        shift 2
        ;;
    --seed)
        seed=${2}
        shift 2
        ;;
    --lr)
        lr=${2}
        shift 2
        ;;
    --dataset_name)
        dataset_name=${2}
        shift 2
        ;;
    --model_name)
        model_name=${2}
        shift 2
        ;;
    --cal_topk_crop)
        cal_topk_crop=${2}
        shift 2
        ;;
    --others)
        others=${2}
        shift 2
        ;;
    --) shift;
        break
        ;;
  esac
done

CMD="nohup python -u tools/train.py --serial ${serial} --batch_size ${batch_size} --cfg configs/${dataset_name}_ft_${augs}.yaml${ls}${sd}${freeze_backbone} --model_name ${model_name}${others}"
CMD_TEST="nohup python -u tools/train.py --test_multiple 0 --test_only --ckpt_path results_train/${dataset_name}_${model_name}_cal_${serial}/${model_name}_last.pth"
echo "${CMD}"
echo "${CMD_TEST}"

# single run
if [[ "$run" == "True" ]]; then
    echo "${CMD} --seed ${seed} --base_lr ${lr}"
    ${CMD} --seed ${seed} --base_lr ${lr}

    if [[ "$cal_ap_only" == "True" ]]; then
        echo "${CMD_TEST} --cal_ap_only "
        ${CMD_TEST} --cal_ap_only
    fi
fi

# lr run
if [[ "$run_lr" == "True" ]]; then
    for rate in ${lr_array[@]}; do
        echo "${CMD} --seed ${seed} --base_lr ${rate} --train_trainval"
        ${CMD} --seed ${seed} --base_lr ${rate} --train_trainval

        if [[ "$cal_ap_only" == "True" ]]; then
            echo "${CMD_TEST} --cal_ap_only "
            ${CMD_TEST} --cal_ap_only
        fi

        if [[ "$cal_topk_crop" =~ ^[0-9]+$ ]]; then
            echo "${CMD_TEST} --cal_topk_crop  ${cal_topk_crop}"
            ${CMD_TEST} --cal_topk_crop  ${cal_topk_crop}
        fi

    done
fi


# seed run
if [[ "$run_seed" == "True" ]]; then
    for seed in ${seed_array[@]}; do
        echo "${CMD} --seed ${seed} --base_lr ${lr}"
        ${CMD} --seed ${seed} --base_lr ${lr}

        if [[ "$cal_ap_only" == "True" ]]; then
            echo "${CMD_TEST} --cal_ap_only "
            ${CMD_TEST} --cal_ap_only
        fi

        if [[ "$cal_topk_crop" =~ ^[0-9]+$ ]]; then
            echo "${CMD_TEST} --cal_topk_crop  ${cal_topk_crop}"
            ${CMD_TEST} --cal_topk_crop  ${cal_topk_crop}
        fi

    done
fi
