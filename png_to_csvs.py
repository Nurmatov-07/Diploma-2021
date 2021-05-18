import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas

for i in range(29):
    gray = Image.open('D:\\ДИПЛОМ\\diploma\\Снимки экрана\\С телефоном\\'+str(i)+'.png').convert('L')
    image = np.array(gray, 'uint8').ravel()
    np.savetxt('D:\\ДИПЛОМ\\diploma\\Снимки экрана\\'+str(i)+'.csv', image, delimiter=',')

data = pandas.read_csv('D:\\ДИПЛОМ\\diploma\\Снимки экрана\\.csv', header=None)