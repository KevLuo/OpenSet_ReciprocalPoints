


# SPLITS="20closed_split0_closedval10_size32 20closed_split1_closedval10_size32 20closed_split2_closedval10_size32 20closed_split3_closedval10_size32 20closed_split4_closedval10_size32"
SPLITS="20closed_split0_closedval10_size32"
TRIALS="trial1 trial2 trial3 trial4 trial5 trial6 trial7 trial8 trial9 trial10"
BASE_BEGIN="/data/kpl39/zsh_ckpts/rpl/"
BASE_MID="latentsize_100_numrp_1_lambda_0.1_gamma_1.0_compnorm_1_L1_FALSE_L1weight_0.0_dataset_"
BASE_END="_01_64_32_OSCRI_encoder/"


x=0
for SPLIT in $SPLITS
do

    for TRIAL in $TRIALS
    do

        CKPTFOLDER="${BASE_BEGIN}${TRIAL}${BASE_MID}${SPLIT}${BASE_END}"
        CUDA_VISIBLE_DEVICES=$((x % 4)) python collect_prediction.py  TINY  VAL  FALSE  100  1  1  2  FALSE  0  OSCRI_encoder  32  /data/  /data/lwneal_tiny/  $SPLIT  $CKPTFOLDER  & 
        
        x=$((x + 1))

    done
    wait
    
done