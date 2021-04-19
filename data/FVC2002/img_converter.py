import os
from PIL import Image
path = "DB1_B/"
for infile in os.listdir(path):
    if infile[-3:] == "tif":
       outfile = infile[:-3] + "jpeg"
       im = Image.open(path+infile)
       out = im.convert("RGB")
       out.save('images/'+outfile, "jpeg", quality=90)