import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
import pandas as pd


# Функции оценки

def calculate_psnr(original, processed, max_intensity=255):
    """
    Вычисление PSNR (Peak Signal-to-Noise Ratio) по формуле
    """
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:  # Обработка деления на ноль
        return float("inf")
    return 20 * np.log10((max_intensity) / np.sqrt(mse))

def calculate_mse(original, processed):
    """
    Вычисление MSE (среднеквадратической ошибки) по заданной формуле.
    """
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    return mse

def calculate_ssim(original, processed):
    """
    Вычисление SSIM (Structural Similarity Index).
    """
    data_range = original.max() - original.min()
    ssim_value, _ = compare_ssim(original, processed, full=True, data_range=data_range)
    return ssim_value


# Функции обработки
def apply_clahe(image_array, clip_limit=1.0, tile_grid_size=(16, 16)):
    """
    Применение CLAHE для улучшения контраста.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_array)


def apply_equalize_hist(image_array):
    """
    Применение гистограммного выравнивания.
    """
    return cv2.equalizeHist(image_array)

def apply_gaussian_blur(image_array, ksize=(3, 3), sigma=0.5):
    """
    Применение Gaussian Blur для сглаживания изображения.
    """
    return cv2.GaussianBlur(image_array, ksize, sigma)

def apply_bilateral_filter(image_array, d=5, sigma_color=30, sigma_space=20):
    """
    Применение Bilateral Filter для шумоподавления с сохранением границ.
    """
    return cv2.bilateralFilter(image_array, d, sigma_color, sigma_space)

def apply_median_blur(image_array, ksize=5):
    """
    Применение Median Blur для уменьшения шума.
    """
    return cv2.medianBlur(image_array, ksize)

def apply_laplacian(image_array):
    """
    Применение фильтра Лапласа для выделения краев.
    """
    laplacian = cv2.Laplacian(image_array, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)  # Преобразование обратно в 8-битное изображение

    # Совмещаем изображение с его границами для усиления контуров
    return cv2.addWeighted(image_array, 1.0, laplacian_abs, 0.5, 0)

def apply_sobel(image_array):
    """
    Применение фильтра Собеля для выделения краев.
    """
    sobel_x = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=3)  # Условие по x
    sobel_y = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=3)  # Условие по y
    edges = cv2.magnitude(sobel_x, sobel_y)

    # Преобразование обратно в 8-битное изображение
    edges = cv2.convertScaleAbs(edges)

    # Совмещаем изображение с его границами для усиления контуров

    return cv2.addWeighted(image_array, 1.0, edges, 0.5, 0)

def apply_denoising(image_array, h=5):
    """
    Применение шумоподавления методом Non-Local Means Denoising.
    """
    return cv2.fastNlMeansDenoising(image_array, None, h, 5, 15)

def apply_sharpening(image_array, alpha=0.7):
    """
    Применение усиления резкости с помощью фильтра Unsharp Mask.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image_array, -1, kernel)
    return cv2.addWeighted(image_array, 1 - alpha, sharpened, alpha, 0)

def combined_processing(image_array):
    """
    Комбинированный метод обработки: CLAHE -> Gaussian Blur -> Sharpening
    """
    step1 = apply_denoising(image_array)
    step2 = apply_laplacian(step1)
    final_image = apply_sharpening(step2)
    return final_image


# Путь к изображению
image_path = "img.png"  # Замените на ваш путь

# Загрузка изображения
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

if image is None:
    print("Не удалось загрузить изображение. Проверьте путь и формат файла.")
else:
    # Словарь методов обработки
    methods = {
        "CLAHE": apply_clahe,
        "Equalize Hist": apply_equalize_hist,
        "Gaussian Blur": apply_gaussian_blur,
        "Bilateral Filter": apply_bilateral_filter,
        "Median Blur": apply_median_blur,
        "Laplacian": apply_laplacian,
        "Sobel": apply_sobel,
        "Denoising": apply_denoising,
        "Sharpening": apply_sharpening,
        "Combined": combined_processing
    }

    # Хранение результатов
    results = []

    for method_name, method in methods.items():
        try:
            # Применяем метод обработки
            processed_image = method(image)

            # Рассчитываем метрики
            mse = calculate_mse(image, processed_image)
            psnr = calculate_psnr(image, processed_image)
            ssim = calculate_ssim(image, processed_image)

            # Сохраняем результаты
            results.append({
                "Method": method_name,
                "MSE": mse,
                "PSNR": psnr,
                "SSIM": ssim
            })

            # Визуализация текущего метода
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap="gray")
            plt.title("Оригинальное изображение")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(processed_image, cmap="gray")
            plt.title(f"{method_name}")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Ошибка при обработке методом {method_name}: {e}")

    # Преобразование результатов в DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["SSIM", "PSNR"], ascending=[False, False])

    # Сохранение результатов в файл Excel
    results_df.to_excel("image_processing_results.xlsx", index=False, engine='openpyxl')
    print("Результаты сохранены в файл 'image_processing_results.xlsx'.")

    # Вывод лучших результатов
    best_method = results_df.iloc[0]
    print("\nЛучший метод обработки:")
    print(best_method)

    # Применение лучшего метода и визуализация
    best_processed_image = methods[best_method["Method"]](image)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Оригинальное изображение")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(best_processed_image, cmap="gray")
    plt.title(f"Лучший метод: {best_method['Method']}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
