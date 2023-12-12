CUDA_VISIBLE_DEVICES=2 python -- train.py --dataset ISIC2018 \
    --exp_name "ecl-clip-ar" \
    --data_path ./data/ISIC2018/ \
    --batch_size 512 \
    --lr 0.002 \
    --epochs 100 \
    --gpu 0 \
    --backbone EVL \
    --model_path ./results/ISIC2018/

CUDA_VISIBLE_DEVICES=2 python -- train.py --dataset ISIC2019 \
    --exp_name "ecl-clip-ar" \
    --data_path ./data/ISIC2019/ \
    --batch_size 512 \
    --lr 0.002 \
    --epochs 100 \
    --gpu 0 \
    --backbone EVL \
    --model_path ./results/ISIC2019/