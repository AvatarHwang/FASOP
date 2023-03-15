NODE_RANK=$1
SCRIPT_NAME='run_inter.sh'
enroot start --root \
            --rw \
            -m $HOME/tdpp/Megatron-LM-2:/root/Megatron-LM megatron-latest \
            bash -c "cd /root/Megatron-LM/ &&\
                    sh $SCRIPT_NAME $NODE_RANK &&\
                    rm -rf my-gpt2_text_document_valid_indexmap_* &&\
                    rm -rf my-gpt2_text_document_train_indexmap_*"