import argparse
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from openai_agent import GPT_Validator
from prompts import HOROSCOPE_PROMPT


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "horoscope_data")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "horoscope_results")

# Настройки для запуска из IDE: включите флаг enabled и укажите параметры ниже.
IDE_RUN_CONFIG = {
    "data_dir": DEFAULT_DATA_DIR,
    "limit": 30,
    "target_file": "Гороскопы 2026 год.xlsx",
    "output_dir": DEFAULT_OUTPUT_DIR,
}


def list_xlsx_files(data_dir: str, target_file: Optional[str]) -> List[str]:
    """Возвращает отсортированный список xlsx файлов из папки."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Папка с данными не найдена: {data_dir}")

    if target_file:
        candidate = os.path.join(data_dir, target_file)
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"Файл {candidate} не найден")
        return [candidate]

    files = [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.lower().endswith(".xlsx")
    ]
    if not files:
        raise FileNotFoundError(f"В папке {data_dir} нет xlsx файлов")
    files.sort()
    return files


def dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Преобразует датафрейм к списку словарей, заменяя NaN на None."""
    cleaned = df.where(pd.notna(df), None)
    return cleaned.to_dict(orient="records")


def normalize_value(value: Any) -> Optional[str]:
    """Приводит данные к строке для передачи в LLM."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%d.%m.%Y")
    text = str(value).strip()
    return text or None


def format_record_context(record: Dict[str, Any]) -> str:
    """Формирует текстовую карточку сотрудника для LLM."""
    fields = {
        "ИО": "Имя",
        "Должность": "Должность",
        "День рождения": "Дата рождения",
        "Чем занимаются": "Зона ответственности",
        "Город чист": "Город",
        "BitrixId": "Bitrix ID",
        "Знак зодиака": "Знак зодиака",
        "Китайский календарь": "Китайский календарь",
    }
    parts: List[str] = []
    for source_field, label in fields.items():
        value = normalize_value(record.get(source_field))
        if value:
            parts.append(f"{label}: {value}")
    if not parts:
        return "Данные о сотруднике отсутствуют"
    return "\n".join(parts)


def serialize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Готовит словарь для сохранения (без несерилизуемых типов)."""
    return {key: normalize_value(value) for key, value in record.items()}


def iter_records(
    data_dir: str, target_file: Optional[str], limit: Optional[int]
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Итерируется по всем записям xlsx файлов, ограничивая их при необходимости."""
    files = list_xlsx_files(data_dir, target_file)
    yielded = 0

    for file_path in files:
        try:
            xl = pd.ExcelFile(file_path)
            sheet_name = "сотрудники" if "сотрудники" in xl.sheet_names else 0
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}: {e}")
            continue

        records = dataframe_to_records(df)
        for record in records:
            yield file_path, record
            yielded += 1
            if limit is not None and limit > 0 and yielded >= limit:
                return


def generate_horoscopes(
    data_dir: str,
    limit: Optional[int],
    target_file: Optional[str],
    output_dir: str,
) -> str:
    """Основной цикл генерации гороскопов."""
    prompt_template = HOROSCOPE_PROMPT
    validator = GPT_Validator()
    os.makedirs(output_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"horoscopes_{timestamp}.csv")

    for idx, (file_path, record) in enumerate(
        iter_records(data_dir, target_file, limit), start=1
    ):
        record_context = format_record_context(record)
        name = normalize_value(record.get("ИО")) or "Сотрудник"
        position = normalize_value(record.get("Должность")) or "Сотрудник"
        city = normalize_value(record.get("Город чист")) or "Не указан"
        birthdate = normalize_value(record.get("День рождения")) or "Не указана"
        zodiac_sign = normalize_value(record.get("Знак зодиака")) or "Не указан"
        zodiac_animal = normalize_value(record.get("Китайский календарь")) or "Не указан"
        pinyin = normalize_value(record.get("Пиньинь")) or "Не указан"


        print(f"[{idx}] Обработка записи из файла {os.path.basename(file_path)}")
        print(record_context)
        
        try:
            full_prompt = prompt_template.format(
                name=name,
                position=position,
                city=city,
                birthdate=birthdate,
                zodiac_sign=zodiac_sign,
                zodiac_animal=zodiac_animal,
                pinyin=pinyin,
                # context=record_context
            )
            # Передаем пробел вторым аргументом, так как контекст уже вшит в промт
            horoscope = validator(full_prompt, " ")
        except Exception as exc:
            print(f"Ошибка при обращении к LLM: {exc}")
            continue

        printable_record = serialize_record(record)
        printable_record["source_file"] = os.path.basename(file_path)
        printable_record["horoscope"] = horoscope
        results.append(printable_record)

        if idx % 10 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)
            print(f"Промежуточное сохранение {len(results)} записей в {output_path}")

    if results:
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"Сохранено {len(results)} гороскопов в {output_path}")
        return output_path

    print("Не удалось получить ни одного гороскопа")
    return ""


def run_from_ide_config() -> bool:
    """Позволяет запускать скрипт из IDE с настройками выше."""
    limit = IDE_RUN_CONFIG.get("limit")
    limit_value: Optional[int] = limit if limit and limit > 0 else None

    generate_horoscopes(
        data_dir=IDE_RUN_CONFIG.get("data_dir", DEFAULT_DATA_DIR),
        limit=limit_value,
        target_file=IDE_RUN_CONFIG.get("target_file"),
        output_dir=IDE_RUN_CONFIG.get("output_dir", DEFAULT_OUTPUT_DIR),
    )
    return True



if __name__ == "__main__":
    run_from_ide_config()
