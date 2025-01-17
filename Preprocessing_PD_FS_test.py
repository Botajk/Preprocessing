import os
from PIL import Image, ImageOps
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# Функции оценки

def calculate_psnr(original, processed):
    """
    Вычисление PSNR между оригинальным и обработанным изображением.
    """
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10(max_pixel**2 / mse)
    return psnr

def calculate_mse(original, processed):
    """
    Вычисление MSE между оригинальным и обработанным изображением.
    """
    return np.mean((original - processed) ** 2)

def calculate_ssim(original, processed):
    """
    Вычисление SSIM между оригинальным и обработанным изображением.
    """
    data_range = original.max() - original.min()
    return ssim(original, processed, data_range=data_range)

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
    Комбинированный метод обработки:
    """
    step1 = apply_denoising(image_array)
    step2 = apply_laplacian(step1)
    final_image = apply_sharpening(step2)
    return final_image

def process_images_in_folder(input_folder, output_folder, metrics_file):
    """
    Обработка всех изображений в папке, сохранение результатов и вычисление метрик качества.
    """
    os.makedirs(output_folder, exist_ok=True)

    methods = {
        "CLAHE": apply_clahe,
        "Gaussian Blur": apply_gaussian_blur,
        "Median Blur": apply_median_blur,
        "Bilateral Filter": apply_bilateral_filter,
        "Denoising": apply_denoising,
        "Sharpening": apply_sharpening,
        "Equalize Hist": apply_equalize_hist,
        "Laplacian": apply_laplacian,
        "Sobel": apply_sobel,
        "Combined": combined_processing
    }

    metrics_data = []  # Список для хранения метрик качества

    for idx, file_name in enumerate(tqdm(os.listdir(input_folder), desc="Обработка изображений")):
        input_path = os.path.join(input_folder, file_name)

        try:
            # Загрузка изображения
            with Image.open(input_path) as img:
                img_gray = ImageOps.grayscale(img)
                image_array = np.array(img_gray)
                image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)

                for method_name, method in methods.items():
                    output_method_folder = os.path.join(output_folder, method_name)
                    os.makedirs(output_method_folder, exist_ok=True)

                    output_path = os.path.join(output_method_folder, file_name)

                    # Обработка изображения
                    processed_array = method(image_array)

                    # Конвертация обратно в изображение и сохранение
                    processed_image = Image.fromarray(processed_array)
                    processed_image.save(output_path)

                    # Вычисление метрик качества
                    psnr_value = calculate_psnr(image_array, processed_array)
                    mse_value = calculate_mse(image_array, processed_array)
                    ssim_value = calculate_ssim(image_array, processed_array)

                    # Сохранение метрик в список
                    metrics_data.append({
                        "Файл": file_name,
                        "Метод": method_name,
                        "PSNR": psnr_value,
                        "MSE": mse_value,
                        "SSIM": ssim_value
                    })

        except Exception as e:
            print(f"Ошибка обработки файла {file_name}: {e}")

    # Сохранение метрик в Excel
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_excel(metrics_file, index=False)

    print(f"Обработка завершена. Результаты сохранены в: {output_folder}")
    print(f"Метрики сохранены в файл: {metrics_file}")

    # Построение общего графика для всех методов по каждой метрике
    methods_list = metrics_df["Метод"].unique()

    plt.figure()
    for method_name in methods_list:
        method_metrics = metrics_df[metrics_df["Метод"] == method_name]
        plt.plot(method_metrics.index, method_metrics["PSNR"], label=method_name)
    plt.title('График PSNR для всех методов')
    plt.xlabel('Изображения')
    plt.ylabel('PSNR')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'PSNR_all_methods.png'))
    plt.show()

    plt.figure()
    for method_name in methods_list:
        method_metrics = metrics_df[metrics_df["Метод"] == method_name]
        plt.plot(method_metrics.index, method_metrics["MSE"], label=method_name)
    plt.title('График MSE для всех методов')
    plt.xlabel('Изображения')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'MSE_all_methods.png'))
    plt.show()

    plt.figure()
    for method_name in methods_list:
        method_metrics = metrics_df[metrics_df["Метод"] == method_name]
        plt.plot(method_metrics.index, method_metrics["SSIM"], label=method_name)
    plt.title('График SSIM для всех методов')
    plt.xlabel('Изображения')
    plt.ylabel('SSIM')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'SSIM_all_methods.png'))
    plt.show()

    # Анализ метрик по всем изображениям
    best_method = metrics_df.groupby("Метод")["SSIM"].mean().idxmax()
    print(f"Лучший метод обработки по всем изображениям: {best_method}")

# Укажите пути к входной и выходной папкам
input_folder = "C:/1"  # Папка с исходными изображениями
output_folder = "C:/Processing/Results"  # Папка для сохранения обработанных изображений
metrics_file = "C:/Processing/metrics.xlsx"  # Файл для сохранения метрик

# Запуск обработки
process_images_in_folder(input_folder, output_folder, metrics_file)
