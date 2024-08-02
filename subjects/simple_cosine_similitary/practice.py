import numpy as np
import cv2

def cosine_similitary(img_x_np, img_y_np):
    vector_x = img_x_np.flatten()
    vector_y = img_y_np.flatten()

    vector_x = vector_x.astype(np.float64)
    vector_y = vector_y.astype(np.float64)

    cos = np.dot(vector_x, vector_y) / (np.linalg.norm(vector_x) * np.linalg.norm(vector_y))

    return cos

def main():
    img1 = cv2.imread('./sign1.png', 1)
    img1 = cv2.resize(img1, (100, 100))

    img2 = cv2.imread('./sign2.png', 1)
    img2 = cv2.resize(img2, (100, 100))

    img3 = cv2.imread('./sign3.png', 1)
    img3 = cv2.resize(img3, (100, 100))

    img4 = cv2.imread('./sign4.png', 1)
    img4 = cv2.resize(img4, (100, 100))

    cos1_2 = cosine_similitary(img1, img2)
    cos1_3 = cosine_similitary(img1, img3)
    cos1_4 = cosine_similitary(img1, img4)

    print(f"cos1_2: {cos1_2}\ncos1_3: {cos1_3}\ncos1_4: {cos1_4}")
    print(f"select: {np.max(np.array([cos1_2, cos1_3, cos1_4]))}")



if __name__ == "__main__":
    main()
