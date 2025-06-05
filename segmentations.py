import cv2
import numpy as np
import matplotlib.pyplot as plt

# def Tichchap(img, matrix):
#     h, w = img.shape
#     expand_img = np.zeros((h+2, w+2))
#     expand_img[1:h+1, 1:w+1] = img  # Nhúng ảnh gốc vào giữa
#     new_img = np.zeros_like(img)
#     for i in range(h):
#         for j in range(w):
#             region = expand_img[i:i+3, j:j+3]
#             new_img[i,j] = np.sum(region * matrix)
#     return new_img

def tich_chap(img, kernel):
    img = img.astype(np.float32)
    h, w = img.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    # Padding ảnh
    padded = np.zeros((h + 2*pad_h, w + 2*pad_w), dtype=np.float32)
    padded[pad_h:h+pad_h, pad_w:w+pad_w] = img

    # Ảnh đầu ra
    result = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)

    # Đưa về kiểu uint8 và giới hạn [0, 255]
    return np.clip(result, 0, 255).astype(np.uint8)

def Prewitt(img):
    new_img1 = tich_chap(img, np.array([
        [-1,-1,-1],
        [0,0,0],
        [1,1,1]
    ]))
    new_img2 = tich_chap(img, np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]))
    matrix_sum = np.sqrt(new_img1 ** 2 + new_img2 ** 2)
    mag = np.clip(matrix_sum, 0, 255).astype(np.uint8)
    # edge_img = np.where(matrix_sum > 0, 255, 0).astype(np.uint8)
    return mag

def Scharr(img):
    new_img1 = tich_chap(img, np.array([
        [-3, 0, 3],
        [-10, 0, 10],
        [-3, 0, 3]
    ]))
    new_img2 = tich_chap(img, np.array([
        [-3, -10, -3],
        [ 0,   0,  0],
        [ 3,  10,  3]
    ]))
    matrix_sum = np.sqrt(new_img1 ** 2 + new_img2 ** 2)
    mag = np.clip(matrix_sum, 0, 255).astype(np.uint8)
    # edge_img = np.where(matrix_sum > 5, 255, 0).astype(np.uint8)
    return mag

def Laplacian(img):
    new_img1 = tich_chap(img, np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ]))
    new_img2 = tich_chap(img, np.array([
         [-1, -1, -1],
         [-1,  8, -1],
         [-1, -1, -1]
    ]))
    matrix_sum = np.sqrt(new_img1 ** 2 + new_img2 ** 2)
    # edge_img = np.where(matrix_sum > 0, 255, 0).astype(np.uint8)
    mag = np.clip(matrix_sum, 0, 255).astype(np.uint8)
    return mag

def TimBien(img):
    _, bien = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bien

# Đọc và xử lý ảnh
img = cv2.imread('coin.jpg')
if img is None:
    print("Không thể mở ảnh.")
    exit()

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Hiển thị kết quả
plt.figure(figsize=(10, 5))

plt.subplot(3, 2, 1)
plt.title("Ảnh gốc")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3, 2, 2)
plt.title("Biên Prewitt")
prewitt_img = Prewitt(gray_img)
plt.imshow(TimBien(prewitt_img), cmap='gray')
plt.axis('off')

edges_canny = cv2.Canny(gray_img, 100, 200)  # threshold1=100, threshold2=200
sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)  # Biên theo trục x
sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)  # Biên theo trục y

sobel = cv2.magnitude(sobelx, sobely)  # Tính độ lớn gradient
edges_sobel = np.uint8(np.clip(sobel, 0, 255))
# In ra ma trận
print("Ma trận kết quả Canny:")
print(edges_canny)

# Nếu bạn muốn xem kích thước hoặc kiểu dữ liệu:
print("Kích thước:", edges_canny.shape)
print("Kiểu dữ liệu:", edges_canny.dtype)

# Hiển thị ảnh (tùy chọn)
plt.subplot(3,2,3)
plt.title("Canny Edge Matrix")
plt.imshow(edges_canny, cmap='gray')
plt.axis('off')


plt.subplot(3, 2, 4)
plt.title("Biên Scharr")
scharr_img = Scharr(gray_img)
plt.imshow(TimBien(scharr_img), cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 5)
plt.title("Biên Laplacian")
laplacian_img = Laplacian(gray_img)
plt.imshow(TimBien(laplacian_img), cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.title("Sobel Edge Matrix")
plt.imshow(edges_sobel, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
