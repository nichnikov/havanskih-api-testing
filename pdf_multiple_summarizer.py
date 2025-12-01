import os
import re
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

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


def count_words(text):
    """
    Подсчитывает количество слов в тексте.
    
    Args:
        text: текст для подсчета
        
    Returns:
        int: количество слов
    """
    if not text:
        return 0
    # Разделяем по пробелам и фильтруем пустые строки
    words = re.findall(r'\b\w+\b', text)
    return len(words)


def count_tokens(text, model_name=None):
    """
    Подсчитывает количество токенов в тексте для указанной модели.
    
    Args:
        text: текст для подсчета
        model_name: имя модели (если None, используется значение из переменной окружения MODEL)
        
    Returns:
        int: количество токенов
    """
    if not text:
        return 0
    
    try:
        # Получаем имя модели из переменной окружения, если не указано
        if model_name is None:
            model_name = os.getenv("MODEL", "gpt-4")
        
        # Определяем кодировку на основе модели
        # Для большинства современных OpenAI моделей используется cl100k_base
        # Для старых моделей GPT-2/GPT-3 используется gpt2
        if "gpt-4" in model_name.lower() or "gpt-3.5" in model_name.lower() or "gpt-4o" in model_name.lower():
            encoding_name = "cl100k_base"
        elif "gpt-3" in model_name.lower():
            encoding_name = "p50k_base"
        else:
            # По умолчанию используем cl100k_base (для GPT-4 и GPT-3.5-turbo)
            encoding_name = "cl100k_base"
        
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Предупреждение: не удалось подсчитать токены ({e}), используется приблизительный подсчет")
        # Приблизительный подсчет: 1 токен ≈ 4 символа для английского текста
        # Для русского текста может быть больше, используем коэффициент 3
        return len(text) // 3


def get_text_statistics(text, model_name=None):
    """
    Получает статистику по тексту: символы, слова и токены.
    
    Args:
        text: текст для анализа
        model_name: имя модели для подсчета токенов
        
    Returns:
        dict: словарь со статистикой {'characters': int, 'words': int, 'tokens': int}
    """
    return {
        'characters': len(text) if text else 0,
        'words': count_words(text),
        'tokens': count_tokens(text, model_name)
    }


def combine_pdfs_and_summarize(pdf_file_list, data_folder="data"):
    """
    Объединяет тексты из указанных PDF файлов и отправляет объединенный текст
    в функцию gpt_validation для суммаризации.
    
    Args:
        pdf_file_list: список имен PDF файлов для обработки (например, ["file1.pdf", "file2.pdf"])
        data_folder: путь к папке с PDF файлами (по умолчанию "data")
    """
    if not os.path.exists(data_folder):
        print(f"Папка {data_folder} не найдена!")
        return
    
    if not pdf_file_list:
        print("Список PDF файлов пуст!")
        return
    
    print(f"\n{'='*80}")
    print(f"ОБЪЕДИНЕНИЕ И СУММАРИЗАЦИЯ PDF ФАЙЛОВ")
    print(f"{'='*80}\n")
    print(f"Список файлов для обработки: {', '.join(pdf_file_list)}\n")
    
    # Инициализируем валидатор GPT
    validator = GPT_Validator()
    
    # Промпт для суммаризации объединенного текста
    summary_prompt = """Ты - опытный редактор и специалист по созданию кратких пересказов.
    Тебе предоставлен объединенный текст из нескольких документов:
    
    Объединенный текст документов:
    {}
    
    Создай краткую суммаризацию данного текста, выделив основные мысли, ключевые моменты и важные детали из всех документов.
    Суммаризация должна быть информативной, структурированной и легко читаемой.
    Обрати внимание на общие темы и связи между разными документами.
    """
    
    combined_text_parts = []
    processed_files = []
    failed_files = []
    
    # Извлекаем текст из каждого указанного PDF файла
    for pdf_file in pdf_file_list:
        pdf_path = os.path.join(data_folder, pdf_file)
        
        if not os.path.exists(pdf_path):
            print(f"ВНИМАНИЕ: Файл {pdf_file} не найден в папке {data_folder}")
            failed_files.append(pdf_file)
            continue
        
        print(f"Извлечение текста из файла: {pdf_file}...")
        
        # Извлекаем весь текст из PDF
        full_text = extract_text_from_pdf(pdf_path)
        print(full_text[:1000])
        
        if not full_text:
            print(f"Не удалось извлечь текст из файла {pdf_file}\n")
            failed_files.append(pdf_file)
            continue
        
        if not full_text.strip():
            print(f"Файл {pdf_file} не содержит текста\n")
            failed_files.append(pdf_file)
            continue
        
        # Добавляем разделитель с именем файла перед текстом
        text_with_header = f"\n\n{'='*80}\nДОКУМЕНТ: {pdf_file}\n{'='*80}\n\n{full_text}"
        combined_text_parts.append(text_with_header)
        processed_files.append(pdf_file)
        
        # Получаем статистику по тексту
        stats = get_text_statistics(full_text)
        print(f"✓ Текст извлечен. Символов: {stats['characters']}, Слов: {stats['words']}, Токенов: {stats['tokens']}\n")
    
    # Проверяем, были ли успешно обработаны файлы
    if not combined_text_parts:
        print("Не удалось извлечь текст ни из одного файла!")
        return
    
    if failed_files:
        print(f"Предупреждение: не удалось обработать следующие файлы: {', '.join(failed_files)}\n")
    
    # Объединяем все тексты в один документ
    combined_text = "\n".join(combined_text_parts)
    
    # Получаем статистику по объединенному тексту
    combined_stats = get_text_statistics(combined_text)
    
    print(f"{'='*80}")
    print(f"СТАТИСТИКА ОБЪЕДИНЕНИЯ")
    print(f"{'='*80}")
    print(f"Успешно обработано файлов: {len(processed_files)}")
    print(f"Общая длина объединенного текста:")
    print(f"  - Символов: {combined_stats['characters']:,}")
    print(f"  - Слов: {combined_stats['words']:,}")
    print(f"  - Токенов: {combined_stats['tokens']:,}")
    print(f"Обработанные файлы: {', '.join(processed_files)}")
    print(f"{'='*80}\n")
    
    print(f"Первые 1000 символов объединенного текста:\n{combined_text[:1000]}...\n")
    
    try:
        # Отправляем объединенный текст в GPT для суммаризации
        print("Отправка объединенного текста в GPT для создания суммаризации...\n")
        summary = validator(summary_prompt, combined_text)
        
        print(f"{'='*80}")
        print(f"РЕЗУЛЬТАТ СУММАРИЗАЦИИ ОБЪЕДИНЕННЫХ ДОКУМЕНТОВ")
        print(f"{'='*80}\n")
        print(summary)
        print(f"\n{'='*80}\n")
        
        return summary
        
    except Exception as e:
        print(f"Ошибка при обработке объединенного текста в GPT: {e}\n")
        return None


if __name__ == "__main__":
    # УКАЖИТЕ ЗДЕСЬ СПИСОК PDF ФАЙЛОВ ДЛЯ ОБЪЕДИНЕНИЯ
    # Примеры:
    pdf_files = ["test_act3.pdf"]
    # pdf_files = ["test_act1.pdf", "test_act2.pdf"]
    # pdf_files = ["test_act1.pdf", "test_act2.pdf", "test_act3.pdf", "test_act4.pdf"]
    # pdf_files = ["test_act1.pdf", "test_act2.pdf", "test_act3.pdf", "test_act1.pdf", "test_act1.pdf"]
    
    combine_pdfs_and_summarize(pdf_files, data_folder="data")

