NODE_RANK=$1
SCRIPT_NAME='run_inter.sh'

#rm -rf /scratch/enroot/1182/data/megatron-latest
#enroot create -n megatron-latest /home1/soonyear/tdpp/image/nvcr.io+nvidia+pytorch+23.03-py3.sqsh

enroot start --root \
            --rw \
            -m $HOME/tdpp/Megatron-LM-2:/root/Megatron-LM megatron-latest \
            bash -c "cd /root/Megatron-LM/ &&\
                    sh $SCRIPT_NAME $NODE_RANK &&\
                    rm -rf my-gpt2_text_document_valid_indexmap_* &&\
                    rm -rf my-gpt2_text_document_train_indexmap_*"