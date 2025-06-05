import numpy as np
import cv2

def convolve2d(image, kernel):
    """
    Hàm thực hiện phép tích chập 2D giữa ảnh và kernel.
    :param image: Ảnh đầu vào dưới dạng numpy array (grayscale)
    :param kernel: Mặt nạ tích chập (numpy array)
    :return: Ảnh đã lọc
    """
    # Lấy kích thước của ảnh và kernel
    iH, iW = image.shape
    kH, kW = kernel.shape

    # Tạo padding để kết quả có cùng kích thước với ảnh gốc
    pad = kH // 2
    image_padded = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    # Khởi tạo ảnh đầu ra
    output = np.zeros((iH, iW), dtype=np.float32)

    # Duyệt qua từng pixel của ảnh gốc
    for y in range(iH):
        for x in range(iW):
            # Trích xuất vùng con của ảnh
            region = image_padded[y:y + kH, x:x + kW]
            
            # Tính tích chập bằng cách nhân từng phần tử và cộng lại
            output[y, x] = np.sum(region * kernel)

    # Chuẩn hóa ảnh đầu ra về khoảng 0-255
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

def median_filter(image, ksize):
    """
    Hàm thực hiện lọc trung vị trên ảnh.
    :param image: Ảnh đầu vào (grayscale)
    :param ksize: Kích thước kernel (phải là số lẻ)
    :return: Ảnh đã lọc
    """
    iH, iW = image.shape
    pad = ksize // 2
    image_padded = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    output = np.zeros((iH, iW), dtype=np.uint8)

    for y in range(iH):
        for x in range(iW):
            region = image_padded[y:y + ksize, x:x + ksize]
            output[y, x] = np.median(region)

    return output

def mean_filter(image, ksize):
    """
    Hàm thực hiện lọc trung bình trên ảnh.
    :param image: Ảnh đầu vào (grayscale)
    :param ksize: Kích thước kernel (phải là số lẻ)
    :return: Ảnh đã lọc
    """
    kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
    return convolve2d(image, kernel)

def apply_filter(image_path, kernel, filter_type="convolution"):
    """
    Hàm áp dụng bộ lọc lên ảnh và hiển thị kết quả.
    :param image_path: Đường dẫn đến ảnh
    :param kernel: Mặt nạ tích chập hoặc kích thước kernel
    :param filter_type: Loại bộ lọc ("convolution", "median", "mean")
    :return: None
    """
    # Đọc ảnh grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Không thể đọc ảnh. Hãy kiểm tra đường dẫn.")

    if filter_type == "convolution":
        filtered_image = convolve2d(image, kernel)
    elif filter_type == "median":
        filtered_image = median_filter(image, kernel.shape[0])
    elif filter_type == "mean":
        filtered_image = mean_filter(image, kernel.shape[0])
    else:
        raise ValueError("Loại bộ lọc không hợp lệ.")

    # Hiển thị ảnh gốc và ảnh đã lọc
    cv2.imshow("Original Image", image)
    cv2.imshow("Filtered Image", filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ví dụ mặt nạ tích chập trung bình 3x3
    kernel_avg = np.ones((3, 3), dtype=np.float32) / 9

    # Gọi hàm apply_filter với đường dẫn ảnh
    try:
        apply_filter("coin.jpg", kernel_avg, filter_type="mean")
    except ValueError as e:
        print(f"Lỗi: {e}")