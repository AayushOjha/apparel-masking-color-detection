import sys
from collections import Counter

import cv2
# import numpy as np
# from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from skimage.color import rgb2lab, deltaE_cie76


def rgb2hex(color):
    return '#{:02x}{:02x}{:02x}'.format(
        int(color[0]), int(color[1]), int(color[2])
    )


def color_detection(img, n_colors, show_chart=False, output_chart=None):
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape(img.shape[0] * img.shape[1], 3)

    clf = KMeans(n_clusters=n_colors)

    colors = clf.fit_predict(img)

    counts = Counter(colors)
    print('COUNTS', counts)
    center_colors = clf.cluster_centers_
    print('CENTER COLORS', center_colors)

    ordered_colors = [center_colors[i] for i in range(n_colors)]
    hex_colors = [rgb2hex(ordered_colors[i]) for i in range(n_colors)]
    rgb_colors = [ordered_colors[i] for i in range(n_colors)]

    color_category = {}

    for idx, hex_color in enumerate(hex_colors):
        color_category[hex_color] = counts[idx]

    print(color_category)

    del color_category[min(color_category.keys())]
    print(color_category)

    if (show_chart):
        plt.figure(figsize=[10, 10])
        plt.pie(color_category.values(), labels=color_category.keys(), colors=color_category.keys())
        plt.savefig(output_chart)

    return hex_colors, rgb_colors


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_chart = sys.argv[2]

    image = cv2.imread(input_path)

    output = color_detection(image, n_colors=8, show_chart=True, output_chart=output_chart)
