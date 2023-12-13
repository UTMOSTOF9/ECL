# python train.py --dataset ISIC2018 \
#     --data_path ./data/ISIC2018/ \
#     --batch_size 64 \
#     --lr 0.002 \
#     --epochs 100 \
#     --gpu 0 \
#     --backbone ResNet50 \
#     --exp_name ecl-clip-ar_modify \
#     --model_path ./results/ISIC2018_ecl-clip-ar/

# python train.py --dataset ISIC2019 \
#     --data_path ./data/ISIC2019/ \
#     --batch_size 100 \
#     --lr 0.002 \
#     --epochs 100 \
#     --gpu 0 \
#     --backbone EVL \
#     --clip_f 0.5 \
#     --cnn_f 0.5 \
#     --exp_name ecl-clip-ar-evl-2 \
#     --model_path ./results/ISIC2019_ecl-clip-ar-evl-2/

python train.py --dataset ISIC2019 \
    --data_path ./data/ISIC2019/ \
    --batch_size 100 \
    --lr 0.002 \
    --epochs 100 \
    --gpu 0 \
    --backbone EVL \
    --clip_f 0.8 \
    --cnn_f 0.2 \
    --exp_name ecl-clip-ar-evl-28 \
    --model_path ./results/ISIC2019_ecl-clip-ar-evl-28/

python train.py --dataset ISIC2018 \
    --data_path ./data/ISIC2018/ \
    --batch_size 100 \
    --lr 0.002 \
    --epochs 100 \
    --gpu 0 \
    --backbone EVL \
    --clip_f 0.8 \
    --cnn_f 0.2 \
    --exp_name ecl-clip-ar-evl-28 \
    --model_path ./results/ISIC2018_ecl-clip-ar-evl-28/

python train.py --dataset ISIC2018 \
    --data_path ./data/ISIC2018/ \
    --batch_size 100 \
    --lr 0.002 \
    --epochs 100 \
    --gpu 0 \
    --clip_f 0.5 \
    --cnn_f 0.5 \
    --backbone ResNet50 \
    --exp_name ecl-clip-ar_modify \
    --model_path ./results/ISIC2018_ecl-clip-ar-2/