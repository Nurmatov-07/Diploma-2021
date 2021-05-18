from PIL import Image
import numpy as np
import sys
import os, os.path, time
import csv


my_dir = 'D:\ДИПЛОМ\Проект_1\Снимки экрана\Без телефона'
# mydir='images'


def createFileList(my_dir, format='.png'):
    fileList = []
    print(my_dir)
    for root, dirs, files in os.walk(my_dir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
                return fileList


fileList = createFileList(my_dir)

for file in fileList:
    print(file)
    img_file = Image.open(file)
# img_file.show()


# get original image parameters...
width, height = img_file.size
format = img_file.format
mode = img_file.mode

# Make image Greyscale
img_grey = img_file.convert('L')
#img_grey.save('result.png')
#img_grey.show()


import matplotlib.pyplot as plt


# Save Greyscale values
value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
value = value.flatten()
print(value)
with open("img_pixels_no.csv", 'a') as f:
    writer = csv.writer(f)
    writer.writerow(value)

