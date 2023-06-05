.
├── FASOP # 최적 분할 탐색 알고리즘
│   ├── amp_utils.py
│   ├── estimate.py
│   ├── FASOP_1.5b.py
│   ├── FASOP_345m.py
│   ├── FASOP_pareto.py
│   ├── known_cost
│   ├── main_logs
│   ├── pipe.py
│   └── stage.py
├── image # megatron 실행 환경 이미지
│   ├── nvcr.io+nvidia+pytorch+23.03-py3.sqsh
│   └── readme.md
├── log  .. # megatron log
├── log2 .. # slurm log
├── log-ncu .. # nsight compute log
├── log-nsys ..# nsight systems log
├── Megatron-LM-2
│   ├── _00_conf.sh  # 모델 크기, 분할 방법, 프로파일링 여부 결정
│   ├── _00_submit-homo.sh # 동종 분할 slurm job 실행   -> conf.sh, run_inter.py 호출
│   ├── _00_submit-hetero.sh # 이기종 분할 slurm job    
│   │                    # 2개(master, slave) 실행 -> hetero-master.sh, hetero-slave.sh 호출
│   ├── _00_hetero-master.sh # master job sbatch     -> run_inter.py 호출
│   ├── _00_hetero-slave.sh  # slave  job sbatch 실행 -> run_inter.py 호출
│   ├── _00_run_inter.sh     # job 스크립트            -> pretrain_gpt.py 호출
│   ├── _00_run_inter_ncu.sh # nsight compute(동작 x) -> pretrain_gpt.py 호출
│   ├── _00_run_inter_nsys.sh# nsight systems(동작 o) -> pretrain_gpt.py 호출
│   ├── _00_pretrain_gpt.py  # 
.   .
├── README.md
├── small # text_document for gpt small
│   ├── my-gpt2_text_document_train_indexmap_1600ns_1024sl_1234s_doc_idx.npy
│   ├── my-gpt2_text_document_train_indexmap_1600ns_1024sl_1234s_sample_idx.npy
│   └── my-gpt2_text_document_train_indexmap_1600ns_1024sl_1234s_shuffle_idx.npy
└── xl    # text_document for gpt xl
    ├── my-gpt2_text_document_train_indexmap_1600ns_1024sl_1234s_doc_idx.npy
    ├── my-gpt2_text_document_train_indexmap_1600ns_1024sl_1234s_sample_idx.npy
    ├── my-gpt2_text_document_train_indexmap_1600ns_1024sl_1234s_shuffle_idx.npy
    ├── my-gpt2_text_document_train_indexmap_3200ns_1024sl_1234s_doc_idx.npy
    ├── my-gpt2_text_document_train_indexmap_3200ns_1024sl_1234s_sample_idx.npy
    └── my-gpt2_text_document_train_indexmap_3200ns_1024sl_1234s_shuffle_idx.npy
