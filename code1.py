import cv2
import pytesseract
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import re

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_contrast = cv2.convertScaleAbs(gray, alpha=4.0, beta=50)
    binary = cv2.adaptiveThreshold(
        enhanced_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    blurred = cv2.GaussianBlur(binary, (3, 3), 0)
    return blurred

def extract_numbers(image):
    processed_image = preprocess_image(image)
    
    # OCR для таблицы
    text_psm4 = pytesseract.image_to_string(processed_image, config="--oem 3 --psm 4 -c tessedit_char_whitelist=0123456789,.")
    text_psm6 = pytesseract.image_to_string(processed_image, config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789,.")
    
    # Извлекаем числа
    numbers_psm4 = [float(num.replace(",", ".")) for num in re.findall(r"\d+(?:[,.]\d+)?", text_psm4)]
    numbers_psm6 = [float(num.replace(",", ".")) for num in re.findall(r"\d+(?:[,.]\d+)?", text_psm6)]
    
    # Убираем дубликаты и сортируем
    final_numbers = sorted(set(numbers_psm4 + numbers_psm6))
    return final_numbers

def calculate_variation_series(data):
    df = pd.DataFrame({'Value': data})
    freq_table = df['Value'].value_counts().reset_index()
    freq_table.columns = ['Value', 'Absolute Frequency']
    freq_table.sort_values('Value', inplace=True)
    freq_table['Relative Frequency'] = freq_table['Absolute Frequency'] / len(data)
    freq_table['Cumulative Absolute Frequency'] = freq_table['Absolute Frequency'].cumsum()
    freq_table['Cumulative Relative Frequency'] = freq_table['Relative Frequency'].cumsum()
    return freq_table

def create_interval_series(data, bins=5):
    hist, bin_edges = np.histogram(data, bins=bins)
    intervals = [f'[{bin_edges[i]}, {bin_edges[i+1]})' for i in range(len(bin_edges)-1)]
    df = pd.DataFrame({
        'Interval': intervals,
        'Absolute Frequency': hist
    })
    df['Relative Frequency'] = df['Absolute Frequency'] / len(data)
    df['Cumulative Absolute Frequency'] = df['Absolute Frequency'].cumsum()
    df['Cumulative Relative Frequency'] = df['Relative Frequency'].cumsum()
    return df

def plot_histogram(series, column, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(series.index, series[column], width=0.7, alpha=0.7, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel(column)
    st.pyplot(fig)

def plot_polygon(series, column, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(series.index, series[column], marker='o', linestyle='-', color='b')
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel(column)
    st.pyplot(fig)

def main():
    st.title('Анализ числового ряда с изображения')
    uploaded_file = st.file_uploader('Загрузите изображение с числовым рядом', type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        numbers = extract_numbers(image)
        
        if numbers:
            st.write(f'Извлеченные числа: {numbers}')
            
            st.subheader('Дискретный вариационный ряд')
            discrete_series = calculate_variation_series(numbers)
            st.dataframe(discrete_series)
            
            st.subheader('Интервальный вариационный ряд')
            interval_series = create_interval_series(numbers)
            st.dataframe(interval_series)
            
            st.subheader('Гистограммы')
            histogram_option = st.selectbox('Выберите гистограмму', [
                'Absolute Frequency', 'Relative Frequency', 'Cumulative Absolute Frequency', 'Cumulative Relative Frequency'
            ])
            plot_histogram(discrete_series.set_index('Value'), histogram_option, f'Гистограмма {histogram_option}')
            plot_histogram(interval_series.set_index('Interval'), histogram_option, f'Гистограмма {histogram_option} (интервальный ряд)')
            
            st.subheader('Полигоны частот')
            polygon_option = st.selectbox('Выберите полигон', [
                'Absolute Frequency', 'Relative Frequency'
            ])
            plot_polygon(discrete_series.set_index('Value'), polygon_option, f'Полигон {polygon_option}')
            plot_polygon(interval_series.set_index('Interval'), polygon_option, f'Полигон {polygon_option} (интервальный ряд)')
        else:
            st.error('Не удалось распознать числа. Попробуйте загрузить изображение с четкими цифрами.')
    
if __name__ == '__main__':
    main()