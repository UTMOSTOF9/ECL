ipython --pdb -- train.py --dataset ISIC2018 \
    --data_path ./data/ISIC2018/ \
    --batch_size 64 \
    --lr 0.002 \
    --epochs 100 \
    --gpu 0 \
    --backbone ResNet50 \
    --bf16 True \
    --exp_name ecl-clip-ar \
    --model_path ./results/ISIC2018_ecl-clip-ar/
