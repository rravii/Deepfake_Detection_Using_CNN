from PIL import Image
import pandas as pd
import os
import tqdm

df = pd.read_json(r"E:\Minor project\done.json")
df = df.T.to_numpy()

images = os.listdir(r"E:\Minor project\Train")
fake_count = 21283
real_count = 27639


for i in tqdm.tqdm(range(df.shape[0])):
    cat = df[i][0]
    bbox = df[i][1]
    path = df[i][2]

    img_name = path.split('/')[-1]
    
    if img_name not in images:
        continue
    

    img = Image.open(path)


    im = img.crop(bbox)
    
    im = im.resize((256, 256))

    if cat == 1:
        im.save(fr"E:\Minor project\data\fake\fake_{fake_count}.jpg")
        fake_count += 1
    else:
        im.save(fr"E:\Minor project\data\real\real_{real_count}.jpg")
        real_count += 1
    