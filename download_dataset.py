import gdown
import os

os.makedirs("./data", exist_ok=True)

url = "https://drive.google.com/uc?id=1bhcKPU2yeus4Kno9bI235anKj8wg9KTj"
output = "./data/ISIC2018_dataset.tar"
gdown.download(url, output, quiet=False)

url = "https://drive.google.com/uc?id=1Ks4Opglr1YTiFMK9VYFgny1q92MY1MEd"
output = "./data/ISIC2019_dataset.tar"
gdown.download(url, output, quiet=False)
