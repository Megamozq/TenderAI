"""
demo.py
=======
Интерактивная демонстрация модели без запуска сервера.
Запускать ПОСЛЕ train.py.

Запуск:
    python demo.py
    python demo.py --interactive   # режим ввода своих данных
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ml.model import TenderRiskModel

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
ORANGE = "\033[33m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"


def risk_color(level: str) -> str:
    return {
        "critical": RED,
        "high":     ORANGE,
        "medium":   YELLOW,
        "low":      GREEN,
    }.get(level, RESET)


def score_bar(score: float, width: int = 40) -> str:
    """Рисует ASCII-прогресс-бар для score."""
    filled = int(round(score / 100 * width))
    if score >= 75:
        color = RED
    elif score >= 50:
        color = ORANGE
    elif score >= 25:
        color = YELLOW
    else:
        color = GREEN
    bar = "█" * filled + "░" * (width - filled)
    return f"{color}{bar}{RESET}"


def print_result(result: dict, label: str = ""):
    score = result["final_score"]
    level = result["risk_level"]
    color = risk_color(level)

    print(f"\n{'─'*60}")
    if label:
        print(f"  {BOLD}{label}{RESET}")
    print(f"{'─'*60}")

    print(f"\n  Risk Score:  {BOLD}{color}{score:.1f}%{RESET}")
    print(f"  {score_bar(score)}")
    print(f"  Уровень:     {BOLD}{color}{level.upper()}{RESET}\n")

    print(f"  Компонентные оценки:")
    for group, gscore in result["group_scores"].items():
        bar_w = 20
        filled = int(round(gscore * bar_w))
        mini_bar = "▪" * filled + "·" * (bar_w - filled)
        print(f"    {group:<14} {CYAN}{mini_bar}{RESET}  {gscore*100:.0f}%")

    print(f"\n  Топ-{len(result['top_features'])} признака:")
    for i, feat in enumerate(result["top_features"], 1):
        arrow = f"{RED}▲{RESET}" if feat["direction"] == "risk_up" else f"{GREEN}▼{RESET}"
        print(f"    {i}. {arrow} {feat['description']}")
        print(f"       {GRAY}Значение: {feat['value']}  |  Вес: {feat['shap_weight']}{RESET}")

    print(f"\n  IF Score:    {result['if_score']}")
    print(f"  Версия:      {result['model_version']}")
    print(f"{'─'*60}\n")


EXAMPLES = {
    "suspicious": {
        "label": "ПОДОЗРИТЕЛЬНЫЙ тендер (высокий риск)",
        "data": {
            "minutes_before_deadline": 5,
            "published_on_friday": 1,
            "acceptance_days": 2,
            "ip_collision": 1,
            "winner_win_rate": 0.92,
            "director_overlap": 1,
            "price_reduction_pct": 0.001,
            "min_bid_gap_pct": 0.001,
            "price_vs_market_pct": 0.55,
            "unique_spec_score": 0.89,
            "tz_change_hours_before": 3,
            "participants_count": 2,
            "supplier_age_days": 45,
            "revenue_vs_contract": 0.3,
            "address_change_days": 10,
            "amendment_count": 5,
            "subcontract_affiliation": 1,
            "historical_win_rate": 0.87,
        },
    },
    "borderline": {
        "label": "ПОГРАНИЧНЫЙ тендер (средний риск)",
        "data": {
            "minutes_before_deadline": 60,
            "published_on_friday": 1,
            "acceptance_days": 5,
            "ip_collision": 0,
            "winner_win_rate": 0.55,
            "director_overlap": 0,
            "price_reduction_pct": 0.02,
            "min_bid_gap_pct": 0.01,
            "price_vs_market_pct": 0.2,
            "unique_spec_score": 0.5,
            "tz_change_hours_before": 24,
            "participants_count": 3,
            "supplier_age_days": 200,
            "revenue_vs_contract": 0.9,
            "address_change_days": 60,
            "amendment_count": 2,
            "subcontract_affiliation": 0,
            "historical_win_rate": 0.45,
        },
    },
    "normal": {
        "label": "НОРМАЛЬНЫЙ тендер (низкий риск)",
        "data": {
            "minutes_before_deadline": 2880,
            "published_on_friday": 0,
            "acceptance_days": 21,
            "ip_collision": 0,
            "winner_win_rate": 0.15,
            "director_overlap": 0,
            "price_reduction_pct": 0.18,
            "min_bid_gap_pct": 0.08,
            "price_vs_market_pct": 0.05,
            "unique_spec_score": 0.15,
            "tz_change_hours_before": 0,
            "participants_count": 8,
            "supplier_age_days": 2190,
            "revenue_vs_contract": 4.5,
            "address_change_days": 900,
            "amendment_count": 0,
            "subcontract_affiliation": 0,
            "historical_win_rate": 0.18,
        },
    },
}


def interactive_mode(model: TenderRiskModel):
    """Режим ввода данных вручную."""
    print(f"\n{BOLD}Введите данные тендера (Enter = значение по умолчанию):{RESET}\n")

    fields = {
        "minutes_before_deadline": ("Минуты до дедлайна при подаче [999]: ", 999.0, float),
        "published_on_friday":     ("Опубликован в пятницу/праздник 0/1 [0]: ", 0, int),
        "acceptance_days":         ("Срок приёма заявок (дней) [14]: ", 14.0, float),
        "ip_collision":            ("Совпадение IP участников 0/1 [0]: ", 0, int),
        "winner_win_rate":         ("Win-rate победителя у заказчика 0–1 [0.2]: ", 0.2, float),
        "director_overlap":        ("Пересечение директоров 0/1 [0]: ", 0, int),
        "price_reduction_pct":     ("Снижение от НМЦК 0–1 [0.1]: ", 0.1, float),
        "min_bid_gap_pct":         ("Минимальный разрыв между ставками [0.05]: ", 0.05, float),
        "price_vs_market_pct":     ("Превышение над рыночной ценой [0.1]: ", 0.1, float),
        "unique_spec_score":       ("Уникальность ТЗ 0–1 [0.2]: ", 0.2, float),
        "tz_change_hours_before":  ("Часов до дедлайна при изменении ТЗ (0=не менялось) [0]: ", 0.0, float),
        "participants_count":      ("Число участников [5]: ", 5, int),
        "supplier_age_days":       ("Возраст поставщика (дней) [730]: ", 730, int),
        "revenue_vs_contract":     ("Выручка / сумма контракта [2.0]: ", 2.0, float),
        "address_change_days":     ("Дней с последней смены адреса [365]: ", 365, int),
        "amendment_count":         ("Количество доп.соглашений [0]: ", 0, int),
        "subcontract_affiliation": ("Субподряд аффилированным 0/1 [0]: ", 0, int),
        "historical_win_rate":     ("Исторический win-rate 0–1 [0.2]: ", 0.2, float),
    }

    tender = {}
    for fname, (prompt, default, cast) in fields.items():
        raw = input(f"  {prompt}").strip()
        try:
            tender[fname] = cast(raw) if raw else default
        except ValueError:
            tender[fname] = default
            print(f"    (некорректное значение, использовано {default})")

    result = model.predict(tender)
    print_result(result, "РЕЗУЛЬТАТ")


def main():
    parser = argparse.ArgumentParser(description="Демо модели Tender Risk")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Интерактивный режим ввода данных")
    parser.add_argument("--model", default="ml/model.pkl",
                        help="Путь к файлу модели")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  TENDER RISK AI — Демонстрация{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")

    # Загрузить модель
    if not Path(args.model).exists():
        print(f"\n  ОШИБКА: модель не найдена: {args.model}")
        print("  Сначала запустите: python train.py")
        sys.exit(1)

    model = TenderRiskModel.load(args.model)

    if args.interactive:
        interactive_mode(model)
        return

    # Запустить три примера
    for key, example in EXAMPLES.items():
        result = model.predict(example["data"])
        print_result(result, example["label"])

    # Пакетный пример
    print(f"\n{BOLD}ПАКЕТНЫЙ СКОРИНГ — 3 тендера разом:{RESET}")
    import pandas as pd
    df = pd.DataFrame([ex["data"] for ex in EXAMPLES.values()])
    df["lot_id"] = ["LOT-001", "LOT-002", "LOT-003"]
    result_df = model.predict_batch(df)
    print(result_df[["lot_id", "final_score", "risk_level", "if_score"]].to_string(index=False))

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"  Запустите --interactive для ввода своих данных:")
    print(f"  {CYAN}python demo.py --interactive{RESET}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
