ipython --pdb -- train.py --dataset ISIC2018 \
    --data_path ./data/ISIC2018/ \
    --batch_size 64 \
    --lr 0.002 \
    --epochs 100 \
    --gpu 0 \
    --model ECL_CLIP_Encoder \
    --bf16 True \
    --alpha 2 \
    --beta 1 \
    --gamma 0 \
    --exp_name ecl-clip_encoder-mssl-2018 \
    --model_path ./results/ecl-clip_encoder-mssl-2018/