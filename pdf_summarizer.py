import os
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class GPT_Validator:
    client = OpenAI(
        api_key=os.getenv("API_KEY"), # ваш ключ в VseGPT после регистрации
        base_url=os.getenv("BASE_URL"),
    )

    def gpt_validation(self, init_prompt: str, dialogue: str):
        prompt = init_prompt.format(str(dialogue))

        messages = []
        messages.append({"role": "user", "content": prompt})

        response_big = self.client.chat.completions.create(
            # model="openai/gpt-4o-mini", # id модели из списка моделей - можно использовать OpenAI, Anthropic и пр. меняя только этот параметр openai/gpt-4o-mini
            model=os.getenv("MODEL"),
            messages=messages,
            temperature=0.7,
            n=1,
            max_tokens=3000, # максимальное число ВЫХОДНЫХ токенов. Для большинства моделей не должно превышать 4096
            extra_headers={ "X-Title": "My App" }, # опционально - передача информация об источнике API-вызова
        )

        return response_big.choices[0].message.content

    def __call__(self, p: str, d: str):
        return self.gpt_validation(p, d)


def extract_text_from_pdf(pdf_path):
    """
    Извлекает весь текст из PDF файла, объединяя все страницы в один документ.
    
    Args:
        pdf_path: путь к PDF файлу
        
    Returns:
        str: объединенный текст со всех страниц
    """
    try:
        reader = PdfReader(pdf_path)
        full_text = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():  # Добавляем только непустые страницы
                full_text.append(text)
        
        return "\n\n".join(full_text)
    except Exception as e:
        print(f"Ошибка при чтении файла {pdf_path}: {e}")
        return None


def process_pdf_files(data_folder="data"):
    """
    Обрабатывает все PDF файлы из указанной папки:
    1. Извлекает текст из всех страниц каждого PDF
    2. Собирает весь текст в один документ
    3. Отправляет в GPT для получения краткого пересказа
    
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
    
    # Инициализируем валидатор GPT
    validator = GPT_Validator()
    
    # Промпт для краткого пересказа
    summary_prompt = """Ты - опытный редактор и специалист по созданию кратких пересказов.
    Тебе предоставлен текст из документа:
    
    Текст документа:
    {}
    
    Создай краткий пересказ данного текста, выделив основные мысли, ключевые моменты и важные детали.
    Пересказ должен быть информативным, структурированным и легко читаемым.
    """
    
    # Обрабатываем каждый PDF файл
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        
        print(f"\n{'='*80}")
        print(f"Обработка файла: {pdf_file}")
        print(f"{'='*80}\n")
        
        # Извлекаем весь текст из PDF
        full_text = extract_text_from_pdf(pdf_path)
        print(full_text[:300])
        
        if not full_text:
            print(f"Не удалось извлечь текст из файла {pdf_file}\n")
            continue
        
        if not full_text.strip():
            print(f"Файл {pdf_file} не содержит текста\n")
            continue
        
        print(f"Текст извлечен из PDF. Длина текста: {len(full_text)} символов\n")
        print(f"Первые 500 символов текста:\n{full_text[:500]}...\n")
        
        try:
            # Отправляем текст в GPT для получения краткого пересказа
            print("Отправка текста в GPT для создания краткого пересказа...\n")
            summary = validator(summary_prompt, full_text)
            
            print(f"{'='*80}")
            print(f"КРАТКИЙ ПЕРЕСКАЗ для файла: {pdf_file}")
            print(f"{'='*80}\n")
            print(summary)
            print(f"\n{'='*80}\n")
            
        except Exception as e:
            print(f"Ошибка при обработке файла {pdf_file} в GPT: {e}\n")


if __name__ == "__main__":
    process_pdf_files("data")

