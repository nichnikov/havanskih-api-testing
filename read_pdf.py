import os
from pypdf import PdfReader

def read_pdf_files(data_folder="data"):
    """
    Читает все PDF файлы из указанной папки и выводит их содержимое.
    
    Args:
        data_folder: путь к папке с PDF файлами (по умолчанию "data")
    """
    if not os.path.exists(data_folder):
        print(f"Папка {data_folder} не найдена!")
        return
    
    # Получаем список всех PDF файлов в папке
    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"PDF файлы не найдены в папке {data_folder}")
        return
    
    pdf_files.sort()  # Сортируем для упорядоченного вывода
    
    # Читаем каждый PDF файл
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        
        try:
            print(f"\n{'='*80}")
            print(f"Файл: {pdf_file}")
            print(f"{'='*80}\n")
            
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            
            print(f"Количество страниц: {num_pages}\n")
            
            # Извлекаем текст со всех страниц
            for page_num, page in enumerate(reader.pages, start=1):
                print(f"--- Страница {page_num} ---")
                text = page.extract_text()
                print(text)
                print()
                
        except Exception as e:
            print(f"Ошибка при чтении файла {pdf_file}: {e}\n")


if __name__ == "__main__":
    read_pdf_files("data")

