python train.py --dataset ISIC2018 \
    --data_path ./data/ISIC2018/ \
    --batch_size 64 \
    --lr 0.002 \
    --epochs 100 \
    --gpu 0 \
    --backbone ResNet50 \
    --exp_name ecl-clip-ar_modify \
    --model_path ./results/ISIC2018_ecl-clip-ar/

python train.py --dataset ISIC2018 \
    --data_path ./data/ISIC2018/ \
    --batch_size 64 \
    --lr 0.002 \
    --epochs 100 \
    --gpu 0 \
    --backbone EVL \
    --exp_name ecl-clip-ar-evl \
    --model_path ./results/ISIC2018_ecl-clip-ar-evl/
