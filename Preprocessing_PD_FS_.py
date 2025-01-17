import os
from PIL import Image, ImageOps
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# Функции обработки
def apply_laplacian(image_array):
    """
    Применение фильтра Лапласа для выделения краев.
    """
    laplacian = cv2.Laplacian(image_array, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)  # Преобразование обратно в 8-битное изображение

    # Совмещаем изображение с его границами для усиления контуров
    return cv2.addWeighted(image_array, 1.0, laplacian_abs, 0.5, 0)

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
    Комбинированный метод обработки: NLM Denoising -> фильтра Лапласа -> Unsharp Mask
    """
    step1 = apply_denoising(image_array)
    step2 = apply_laplacian(step1)
    final_image = apply_sharpening(step2)
    return final_image


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

def process_images_in_folder(input_folder, output_folder, metrics_file):
    """
    Обработка всех изображений в папке, сохранение результатов и вычисление метрик качества.
    """
    os.makedirs(output_folder, exist_ok=True)

    metrics_data = []  # Список для хранения метрик качества

    for idx, file_name in enumerate(tqdm(os.listdir(input_folder), desc="Обработка изображений")):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        try:
            # Загрузка изображения
            with Image.open(input_path) as img:
                img_gray = ImageOps.grayscale(img)
                image_array = np.array(img_gray)

                # Обработка изображения
                processed_array = combined_processing(image_array)

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

    # Построение графиков
    psnr_values = [entry['PSNR'] for entry in metrics_data]
    mse_values = [entry['MSE'] for entry in metrics_data]
    ssim_values = [entry['SSIM'] for entry in metrics_data]
    total_images = len(metrics_data)
    indices = list(range(1, total_images + 1))

    plt.figure()
    plt.plot(indices, psnr_values)
    plt.title('График PSNR')
    plt.xlabel('Количество изображений')
    plt.ylabel('PSNR')
    plt.xticks(ticks=np.linspace(0, total_images, num=10, dtype=int), rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'psnr_plot.png'))
    plt.show()

    plt.figure()
    plt.plot(indices, mse_values, color='red')
    plt.title('График MSE')
    plt.xlabel('Количество изображений')
    plt.ylabel('MSE')
    plt.xticks(ticks=np.linspace(0, total_images, num=10, dtype=int), rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'mse_plot.png'))
    plt.show()

    plt.figure()
    plt.plot(indices, ssim_values, color='green')
    plt.title('График SSIM')
    plt.xlabel('Количество изображений')
    plt.ylabel('SSIM')
    plt.xticks(ticks=np.linspace(0, total_images, num=10, dtype=int), rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'ssim_plot.png'))
    plt.show()

# Укажите пути к входной и выходной папкам
input_folder = "C:/PD FS"  # Папка с исходными изображениями
output_folder = "C:/Processing/PD_FS"  # Папка для сохранения обработанных изображений
metrics_file = "C:/Processing/metrics_PD_FS.xlsx"  # Файл для сохранения метрик

# Запуск обработки
process_images_in_folder(input_folder, output_folder, metrics_file)
