DATASETS="VAL TEST"
SPLITS="open_animals_10_split0 open_animals_10_split1 open_animals_10_split2 open_animals_10_split3 open_animals_10_split4"
# TRIALS="trial1 trial2 trial3 trial4 trial5 trial6 trial7 trial8 trial9 trial10"
TRIALS='trial1'
BASE_BEGIN="/data/kpl39/zsh_ckpts/rpl/"
BASE_END="latentsize_100_numrp_1_lambda_0.1_gamma_1.0_dataset_cifar10_plus_01_64_32_OSCRI_encoder/"

x=0
for DATASET in $DATASETS
do

    for SPLIT in $SPLITS
    do
    
        for TRIAL in $TRIALS
        do

            CKPTFOLDER="${BASE_BEGIN}${TRIAL}${BASE_END}"
            CUDA_VISIBLE_DEVICES=$((x % 4)) python collect_cifar_rpl_prediction.py  $DATASET  100  1  1  OSCRI_encoder  /data/kpl39/cifar10_plus/  $SPLIT  /data/cifar100/torchvision_compatible/cifar-100-python/  $CKPTFOLDER &

            x=$((x + 1))
            
        done
        wait
        
    done
    
done