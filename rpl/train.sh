SPLITS="20closed_split0_closedval10_size32 20closed_split3_closedval10_size32 20closed_split4_closedval10_size32"
TRIALS="trial1 trial2 trial3 trial4 trial5 trial6 trial7 trial8 trial9 trial10" 
x=0
for SPLIT in $SPLITS
do
    for TRIALNUM in $TRIALS
    do

        CUDA_VISIBLE_DEVICES=$((x % 4)) python train.py  300  100  1  0.1  1  2  TRUE  0.5  FALSE  TINY  /data/lwneal_tiny/  $SPLIT  64  0.01  32  3  OSCRI_encoder  /data/kpl39/zsh_ckpts/rpl/  /home/kpl39/common_sense_embeddings/logfiles/zsh/rpl/  NO_DEBUG  $TRIALNUM &

        x=$((x + 1))
    
    done
    wait

done
