"""
generate_data.py
================
Генерирует реалистичный датасет тендеров для обучения модели.
Создаёт CSV файл с 18 признаками + метка (для валидации, не для обучения).

Запуск:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

N_NORMAL = 900    # нормальных тендеров
N_CORRUPT = 100   # подозрительных (для валидации)

def generate_normal_tender():
    """Генерирует нормальный тендер без признаков коррупции."""
    participants = random.randint(3, 15)
    initial_price = random.uniform(500_000, 50_000_000)
    price_reduction = random.uniform(0.05, 0.35)   # снижают на 5-35%
    min_bid_gap = random.uniform(0.02, 0.15)

    return {
        # F01: подача в последние минуты (норма - подают заранее)
        "minutes_before_deadline": random.randint(200, 5000),
        # F02: публикация в праздник/пятницу
        "published_on_friday": random.choice([0, 0, 0, 1]),
        # F03: срок приёма заявок (норма >= 7 дней)
        "acceptance_days": random.randint(7, 30),
        # F04: совпадение IP (редко у нормальных)
        "ip_collision": random.choice([0, 0, 0, 0, 1]),
        # F05: win_rate победителя у этого заказчика
        "winner_win_rate": random.uniform(0.0, 0.4),
        # F06: пересечение директоров
        "director_overlap": random.choice([0, 0, 0, 1]),
        # F07: снижение цены
        "price_reduction_pct": price_reduction,
        # F08: минимальный разрыв между заявками
        "min_bid_gap_pct": min_bid_gap,
        # F09: превышение над рыночной ценой
        "price_vs_market_pct": random.uniform(-0.1, 0.2),
        # F10: уникальность ТЗ (0=стандартное, 1=уникальное)
        "unique_spec_score": random.uniform(0.0, 0.4),
        # F11: изменение ТЗ за X часов до дедлайна (0 = не менялось)
        "tz_change_hours_before": random.choice([0, 0, 0, random.randint(48, 200)]),
        # F12: число участников
        "participants_count": participants,
        # F13: возраст компании в днях
        "supplier_age_days": random.randint(365, 5000),
        # F14: выручка / сумма контракта
        "revenue_vs_contract": random.uniform(1.5, 20.0),
        # F15: дней с последней смены адреса
        "address_change_days": random.randint(180, 3000),
        # F16: количество доп.соглашений
        "amendment_count": random.randint(0, 2),
        # F17: субподряд аффилированным
        "subcontract_affiliation": random.choice([0, 0, 0, 1]),
        # F18: исторический win_rate поставщика
        "historical_win_rate": random.uniform(0.05, 0.35),
        # Метка (не используется при обучении)
        "is_corrupt": 0,
        "initial_price": initial_price,
    }


def generate_corrupt_tender():
    """Генерирует подозрительный тендер с признаками коррупции."""
    participants = random.randint(1, 3)   # мало участников
    initial_price = random.uniform(5_000_000, 100_000_000)
    price_reduction = random.uniform(0.0, 0.02)   # почти не снижают цену

    # Выбираем случайный набор коррупционных признаков
    corrupt_patterns = random.sample([
        "timing", "ip", "price", "spec", "monopoly", "young_company"
    ], k=random.randint(2, 5))

    return {
        "minutes_before_deadline": random.randint(1, 30) if "timing" in corrupt_patterns else random.randint(100, 1000),
        "published_on_friday": 1 if "timing" in corrupt_patterns else 0,
        "acceptance_days": random.randint(1, 3) if "timing" in corrupt_patterns else random.randint(5, 10),
        "ip_collision": 1 if "ip" in corrupt_patterns else 0,
        "winner_win_rate": random.uniform(0.7, 1.0) if "monopoly" in corrupt_patterns else random.uniform(0.3, 0.7),
        "director_overlap": 1 if "ip" in corrupt_patterns else 0,
        "price_reduction_pct": price_reduction,
        "min_bid_gap_pct": random.uniform(0.0, 0.005) if "price" in corrupt_patterns else random.uniform(0.01, 0.05),
        "price_vs_market_pct": random.uniform(0.3, 0.8) if "price" in corrupt_patterns else random.uniform(0.1, 0.4),
        "unique_spec_score": random.uniform(0.7, 1.0) if "spec" in corrupt_patterns else random.uniform(0.3, 0.7),
        "tz_change_hours_before": random.randint(1, 24) if "spec" in corrupt_patterns else 0,
        "participants_count": participants,
        "supplier_age_days": random.randint(30, 180) if "young_company" in corrupt_patterns else random.randint(180, 730),
        "revenue_vs_contract": random.uniform(0.1, 0.8) if "young_company" in corrupt_patterns else random.uniform(0.8, 2.0),
        "address_change_days": random.randint(1, 30) if "young_company" in corrupt_patterns else random.randint(30, 180),
        "amendment_count": random.randint(3, 8),
        "subcontract_affiliation": 1 if "ip" in corrupt_patterns else 0,
        "historical_win_rate": random.uniform(0.6, 1.0) if "monopoly" in corrupt_patterns else random.uniform(0.3, 0.6),
        "is_corrupt": 1,
        "initial_price": initial_price,
    }


def main():
    print("Генерация датасета тендеров...")
    records = []
    
    for _ in range(N_NORMAL):
        records.append(generate_normal_tender())
    
    for _ in range(N_CORRUPT):
        records.append(generate_corrupt_tender())
    
    df = pd.DataFrame(records)
    
    # Добавляем lot_id и метаданные
    df.insert(0, "lot_id", [f"LOT-{i:05d}" for i in range(1, len(df) + 1)])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_path = "data/tenders.csv"
    df.to_csv(output_path, index=False)
    
    print(f"  Всего записей: {len(df)}")
    print(f"  Нормальных:    {(df['is_corrupt'] == 0).sum()}")
    print(f"  Подозрительных:{(df['is_corrupt'] == 1).sum()}")
    print(f"  Сохранено в:   {output_path}")
    print()
    print("Первые 5 строк:")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
