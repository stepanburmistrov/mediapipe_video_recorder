import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from datetime import datetime
import os
from collections import deque
import logging

# Настройки записи
RECORD_ALWAYS = True   # Записывать видео постоянно
RECORD_ON_EVENT = True  # Записывать видео по событию (при обнаружении целевых объектов)
TARGET_CLASSES = ['person', 'car']  # Список интересующих классов
VIDEO_DURATION = 600  # Продолжительность видеофайла в секундах (10 минут)
MAX_STORAGE_SIZE_RECORDS = int(0.5 * 1024 * 1024 * 1024)  # Максимальный размер хранения записей (10 ГБ)
MAX_STORAGE_SIZE_ALARMS = int(0.5 * 1024 * 1024 * 1024)    # Максимальный размер хранения тревог (5 ГБ)
BUFFER_DURATION = 10    # Продолжительность буфера в секундах
POST_RECORD_DURATION = 60  # Продолжительность записи после последнего обнаружения в секундах
DRAW_BOUNDING_BOXES_IN_ALARM = True  # Показывать ли рамки вокруг обнаруженных объектов в тревожных видео

# Инициализация параметров для отображения
MARGIN = 10         # отступ в пикселях
ROW_SIZE = 20       # высота строки в пикселях
FONT_SIZE = 1       # размер шрифта
FONT_THICKNESS = 1  # толщина шрифта

# Словарь цветов для каждой категории
CLASS_COLORS = {
    'person': (255, 0, 0),      # красный
    'car': (0, 255, 0),         # зеленый
    'dog': (0, 0, 255),         # синий
    'bicycle': (255, 255, 0),   # голубой
    'bench': (255, 0, 255),     # розовый
    # Добавьте другие категории и цвета по необходимости
}

# Создание папок для сохранения видео и логов
if not os.path.exists('alarms'):
    os.makedirs('alarms')

if not os.path.exists('records'):
    os.makedirs('records')

if not os.path.exists('logs'):
    os.makedirs('logs')

# Настройка логирования
current_date = datetime.now().strftime('%Y%m%d')
log_filename = f'logs/{current_date}.log'

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def manage_storage(directory, max_size):
    """Удаляет самые старые файлы в директории, чтобы общий размер не превышал max_size."""
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=lambda x: os.path.getmtime(x))
    total_size = sum(os.path.getsize(f) for f in files)
    while total_size > max_size and files:
        oldest_file = files.pop(0)
        total_size -= os.path.getsize(oldest_file)
        os.remove(oldest_file)
        logging.info(f"Удален файл {oldest_file} для освобождения места.")

def visualize_and_check(image, detection_result, draw_boxes=True) -> (np.ndarray, bool):
    """Отображает результаты детекции на изображении и проверяет наличие целевых классов.

    Args:
        image (np.ndarray): Исходное изображение.
        detection_result: Результаты детекции объектов.
        draw_boxes (bool): Флаг, определяющий, нужно ли рисовать рамки вокруг обнаруженных объектов.

    Returns:
        Tuple[np.ndarray, bool]: Размеченное изображение и флаг наличия целевых классов.
    """
    target_class_present = False
    for detection in detection_result.detections:
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"

        # Проверяем, есть ли обнаруженный объект в списке интересующих классов
        is_target_class = category_name in TARGET_CLASSES

        # Проверяем, есть ли обнаруженный объект в списке целевых классов
        if is_target_class:
            target_class_present = True
            #logging.info(f"Обнаружен объект: {category_name} с вероятностью {probability}")

        if draw_boxes and is_target_class:
            color = CLASS_COLORS.get(category_name, (255, 255, 255))
            bbox = detection.bounding_box
            start_point = int(bbox.origin_x), int(bbox.origin_y)
            end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
            cv2.rectangle(image, start_point, end_point, color, 2)

            text_location = (int(MARGIN + bbox.origin_x), int(MARGIN + ROW_SIZE + bbox.origin_y))
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, color, FONT_THICKNESS)
    return image, target_class_present

def add_timestamp(image):
    """Накладывает текущие дату и время на изображение.

    Args:
        image (np.ndarray): Изображение для наложения даты и времени.

    Returns:
        np.ndarray: Изображение с наложенной датой и временем.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    text_location = (10, image.shape[0] - 10)
    cv2.putText(image, timestamp, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

# Настройка модели детекции объектов
base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    max_results=5
)
detector = vision.ObjectDetector.create_from_options(options)

# Захват видеопотока с камеры
cap = cv2.VideoCapture(0)

# Получение свойств видео для корректной записи
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 0  # Мы будем рассчитывать фактический FPS самостоятельно

# Инициализация переменных для записи видео
continuous_writer = None
alarm_writer = None
start_time_continuous = None
start_time_alarm = None
video_index_continuous = 0
video_index_alarm = 0

# Инициализация буфера кадров для тревожной записи
frame_buffer = deque()

# Время последнего обнаружения целевого объекта
last_detection_time = None

# Для расчета FPS
frame_times = deque(maxlen=30)

while cap.isOpened():
    frame_start_time = time.time()
    success, frame = cap.read()
    if not success:
        logging.error("Не удалось получить кадр с камеры")
        break

    # Наложение даты и времени
    frame = add_timestamp(frame)

    # Преобразование кадра в формат MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Выполнение детекции объектов
    detection_result = detector.detect(mp_image)

    # Визуализация результатов для отображения (всегда с рамками вокруг интересующих классов)
    display_image, target_class_present = visualize_and_check(frame.copy(), detection_result, draw_boxes=True)

    # Визуализация результатов для постоянной записи (рамки вокруг интересующих классов)
    continuous_record_image, _ = visualize_and_check(frame.copy(), detection_result, draw_boxes=True)

    # Визуализация результатов для тревожной записи (рамки вокруг интересующих классов в зависимости от настройки)
    alarm_record_image, _ = visualize_and_check(frame.copy(), detection_result, draw_boxes=DRAW_BOUNDING_BOXES_IN_ALARM)

    # Добавляем кадр и время в буфер для тревожной записи
    frame_buffer.append((alarm_record_image.copy(), frame_start_time))

    current_time = time.time()

    # Рассчитываем фактический FPS
    frame_times.append(current_time)
    if len(frame_times) >= 2:
        fps = len(frame_times) / (frame_times[-1] - frame_times[0])
    else:
        fps = 30  # Значение по умолчанию

    # ----- Постоянная запись -----
    if RECORD_ALWAYS:
        if continuous_writer is None:
            # Начинаем новую запись видео
            video_filename = f"records/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_index_continuous}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            continuous_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
            start_time_continuous = current_time
            video_index_continuous += 1
            logging.info(f"Начата постоянная запись: {video_filename}")

        # Записываем кадр
        continuous_writer.write(continuous_record_image)

        # Проверяем, прошло ли VIDEO_DURATION секунд
        if current_time - start_time_continuous >= VIDEO_DURATION:
            # Завершаем текущую запись
            continuous_writer.release()
            logging.info(f"Завершена постоянная запись: {video_filename}")
            continuous_writer = None
            start_time_continuous = None

        # Управление хранилищем
        manage_storage('records', MAX_STORAGE_SIZE_RECORDS)

    # ----- Запись по событию -----
    if RECORD_ON_EVENT:
        if target_class_present:
            last_detection_time = current_time

            if alarm_writer is None:
                # Начинаем новую запись видео с кадрами из буфера
                video_filename = f"alarms/alarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_index_alarm}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                alarm_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
                start_time_alarm = current_time
                video_index_alarm += 1
                logging.info(f"Начата запись тревоги: {video_filename}")

                # Записываем кадры из буфера
                for buf_frame, _ in frame_buffer:
                    alarm_writer.write(buf_frame)

            # Записываем текущий кадр
            alarm_writer.write(alarm_record_image)

        elif alarm_writer is not None:
            # Проверяем, прошло ли время после последнего обнаружения
            if current_time - last_detection_time <= POST_RECORD_DURATION:
                # Продолжаем запись
                alarm_writer.write(alarm_record_image)
            else:
                # Завершаем запись
                alarm_writer.release()
                logging.info(f"Завершена запись тревоги: {video_filename}")
                alarm_writer = None
                start_time_alarm = None
                last_detection_time = None
                frame_buffer.clear()  # Очищаем буфер после записи

            # Управление хранилищем
            manage_storage('alarms', MAX_STORAGE_SIZE_ALARMS)

    # Отображение кадра
    cv2.imshow('MediaPipe Object Detection', display_image)

    # Выход по нажатию клавиши 'Esc'
    if cv2.waitKey(1) == 27:
        logging.info("Завершение работы по нажатию клавиши 'Esc'")
        break

    # Очищаем буфер старых кадров
    while frame_buffer and (current_time - frame_buffer[0][1] > BUFFER_DURATION):
        frame_buffer.popleft()

# Освобождаем ресурсы
cap.release()
if continuous_writer is not None:
    continuous_writer.release()
    logging.info(f"Завершена постоянная запись: {video_filename}")
if alarm_writer is not None:
    alarm_writer.release()
    logging.info(f"Завершена запись тревоги: {video_filename}")
cv2.destroyAllWindows()
