import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image):
    """
    Tính toán histogram của ảnh.
    :param image: Ảnh đầu vào (grayscale)
    :return: Histogram của ảnh
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

def plot_histogram(image, title="Histogram"):
    """
    Vẽ histogram của ảnh.
    :param image: Ảnh đầu vào (grayscale)
    :param title: Tiêu đề của đồ thị
    """
    hist = calculate_histogram(image)
    plt.figure(figsize=(10, 4))
    plt.plot(hist)
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def adjust_brightness(image, factor):
    """
    Điều chỉnh độ sáng của ảnh.
    :param image: Ảnh đầu vào (grayscale)
    :param factor: Hệ số điều chỉnh độ sáng (factor > 0)
    :return: Ảnh đã điều chỉnh độ sáng
    """
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def adjust_contrast_minmax(image):
    """
    Điều chỉnh độ tương phản sử dụng phương pháp min-max.
    :param image: Ảnh đầu vào (grayscale)
    :return: Ảnh đã điều chỉnh độ tương phản
    """
    min_val = np.min(image)
    max_val = np.max(image)
    return np.clip(((image - min_val) * 255) / (max_val - min_val), 0, 255).astype(np.uint8)

def adjust_contrast_piecewise(image, alpha=0.5, beta=0.5, gamma=0.5):
    """
    Điều chỉnh độ tương phản sử dụng phương pháp từng phần.
    :param image: Ảnh đầu vào (grayscale)
    :param alpha: Hệ số cho đoạn 1 (0-85)
    :param beta: Hệ số cho đoạn 2 (86-170)
    :param gamma: Hệ số cho đoạn 3 (171-255)
    :return: Ảnh đã điều chỉnh độ tương phản
    """
    result = np.zeros_like(image)
    
    # Đoạn 1: 0-85
    mask1 = image <= 85
    result[mask1] = image[mask1] * alpha
    
    # Đoạn 2: 86-170
    mask2 = (image > 85) & (image <= 170)
    result[mask2] = 85 + (image[mask2] - 85) * beta
    
    # Đoạn 3: 171-255
    mask3 = image > 170
    result[mask3] = 170 + (image[mask3] - 170) * gamma
    
    return np.clip(result, 0, 255).astype(np.uint8)

def equalize_histogram(image):
    """
    Cân bằng histogram của ảnh.
    :param image: Ảnh đầu vào (grayscale)
    :return: Ảnh đã cân bằng histogram
    """
    return cv2.equalizeHist(image)

def process_image(image_path):
    """
    Xử lý ảnh với các phương pháp khác nhau.
    :param image_path: Đường dẫn đến ảnh
    """
    # Đọc ảnh
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Không thể đọc ảnh. Hãy kiểm tra đường dẫn.")

    # Tạo figure với 3x3 subplot
    plt.figure(figsize=(15, 10))

    # 1. Hiển thị ảnh gốc và histogram
    plt.subplot(3, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Ảnh gốc')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plot_histogram(image, 'Histogram ảnh gốc')

    # 2. Điều chỉnh độ sáng
    bright_image = adjust_brightness(image, 1.5)  # Tăng độ sáng 50%
    plt.subplot(3, 3, 3)
    plt.imshow(bright_image, cmap='gray')
    plt.title('Tăng độ sáng')
    plt.axis('off')

    # 3. Điều chỉnh độ tương phản min-max
    contrast_minmax = adjust_contrast_minmax(image)
    plt.subplot(3, 3, 4)
    plt.imshow(contrast_minmax, cmap='gray')
    plt.title('Điều chỉnh độ tương phản (min-max)')
    plt.axis('off')

    # 4. Điều chỉnh độ tương phản từng phần
    contrast_piecewise = adjust_contrast_piecewise(image)
    plt.subplot(3, 3, 5)
    plt.imshow(contrast_piecewise, cmap='gray')
    plt.title('Điều chỉnh độ tương phản (từng phần)')
    plt.axis('off')

    # 5. Cân bằng histogram
    equalized = equalize_histogram(image)
    plt.subplot(3, 3, 6)
    plt.imshow(equalized, cmap='gray')
    plt.title('Cân bằng histogram')
    plt.axis('off')

    # Hiển thị histogram của các ảnh đã xử lý
    plt.subplot(3, 3, 7)
    plot_histogram(bright_image, 'Histogram sau tăng độ sáng')

    plt.subplot(3, 3, 8)
    plot_histogram(contrast_minmax, 'Histogram sau điều chỉnh min-max')

    plt.subplot(3, 3, 9)
    plot_histogram(equalized, 'Histogram sau cân bằng')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        # Sử dụng ảnh coin.jpg có sẵn trong thư mục
        process_image("coin.jpg")
    except ValueError as e:
        print(f"Lỗi: {e}") 