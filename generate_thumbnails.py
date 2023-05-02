import cv2
import os
from tqdm import tqdm

for f in tqdm(os.listdir('mp4')):
    video = cv2.VideoCapture(f'mp4/{f}')
    ret, frame = video.read()
    cv2.imwrite(f'jpg/{f[:-4]}.jpg', frame)