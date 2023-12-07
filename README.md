# How to train

1. Download the dataset from [link](https://drive.google.com/drive/folders/1y5eD3C9j5XL44MUj5u-G0ACk9N_NMxTc?usp=share_link)

2. Put the dataset in the folder `data/`

3. Prepare python env

   ```bash
   pip install -r torch torchvision --index-url https://download.pytorch.org/whl/cu118 # better to install pytorch first with your cuda version
   pip install -r requirements.txt
   ```

4. Run the following script to train the model

   - for 2018

   ```bash
   bash scripts/ISIC2018_train.sh
   ```

   - for 2019

   ```bash
   bash scripts/ISIC2019_train.sh
   ```
