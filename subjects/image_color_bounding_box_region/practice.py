import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

def read_imgs(path):
    paths = glob.glob(path)

    imgs = []

    for path in paths:
        img = cv2.imread(path, 1)
        imgs.append(img)

    return np.array(imgs)

def crop_img(np_img, vectorStart, vectorEnd):
    x_0, y_0 = vectorStart
    x_1, y_1 = vectorEnd

    return np_img[y_0:y_1, x_0:x_1, :]

def bounding_box_img(np_img, vectorStart, vectorEnd):
    np_img = np_img.copy()

    x_0, y_0 = vectorStart
    x_1, y_1 = vectorEnd

    color = np.array([0, 0, 255]) # (b,g,r)
    np_img[y_0, x_0:x_1 , :] = color
    np_img[y_1, x_0:x_1 , :] = color
    np_img[y_0:y_1, x_0 , :] = color
    np_img[y_0:y_1, x_1 , :] = color

    return np_img

def compute_average(vector):
    return np.mean(vector)

def graycolorize_img(np_img):
    return np.apply_along_axis(compute_average, axis=2, arr=np_img)

def adjust_brightness(np_img, brightness_val):
    np_img = np_img.copy()
    np_img = np_img.astype(float)

    np_img = np_img + brightness_val
    np_img = np.clip(np_img, 0, 255)

    np_img = np_img.astype(np.uint8)
    # np_img = np.where(np_img.astype(float) + 100 > 255, 255, np_img + 100)
    return np_img

def main():
    np_imgs = read_imgs("sample/*.jpg")

    fig = plt.figure(figsize=(10, 7)) 
    rows, cols = 2, 3 

    fig.add_subplot(rows, cols, 1)

    for img in np_imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(img)

    fig.add_subplot(rows, cols, 2)
    
    for img in np_imgs:
        cropped_img = crop_img(img, (470, 120), (800, 850))
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(cropped_img)
    

    fig.add_subplot(rows, cols, 3)

    for img in np_imgs:
        gray_img = graycolorize_img(img)
        plt.axis('off')
        plt.imshow(gray_img, cmap='gray')

    fig.add_subplot(rows, cols, 4)

    for img in np_imgs:
        bounding_img = bounding_box_img(img, (470, 120), (800, 850))
        bounding_img = cv2.cvtColor(bounding_img, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(bounding_img)

    fig.add_subplot(rows, cols, 5)

    for img in np_imgs:
        adjusted_img = adjust_brightness(img, 100)
        adjusted_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(adjusted_img)
        
    plt.show()

if __name__ == "__main__":
    main()
