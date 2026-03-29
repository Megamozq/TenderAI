"""
train.py
========
Запускает полный пайплайн обучения модели:
  1. Загружает данные из data/tenders.csv
  2. Обучает TenderRiskModel
  3. Оценивает качество (ROC-AUC, Precision, Recall)
  4. Сохраняет модель в ml/model.pkl
  5. Выводит отчёт

Запуск:
    python train.py
    python train.py --data data/tenders.csv --out ml/model.pkl
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent))
from ml.model import TenderRiskModel, FEATURE_COLS


def print_separator(char="─", width=60):
    print(char * width)


def main():
    parser = argparse.ArgumentParser(description="Обучение модели выявления коррупции")
    parser.add_argument("--data", default="data/tenders.csv", help="Путь к CSV")
    parser.add_argument("--out",  default="ml/model.pkl",    help="Путь для сохранения")
    parser.add_argument("--contamination", type=float, default=0.10,
                        help="Ожидаемая доля аномалий (0.0–0.5)")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Количество деревьев Isolation Forest")
    args = parser.parse_args()

    print_separator("═")
    print("  ОБУЧЕНИЕ МОДЕЛИ ВЫЯВЛЕНИЯ КОРРУПЦИИ В ТЕНДЕРАХ")
    print_separator("═")
    print()

    # ── 1. Загрузка данных ────────────────────────────────────────────────────
    print(f"[1/4] Загрузка данных: {args.data}")
    if not Path(args.data).exists():
        print(f"  ОШИБКА: файл {args.data} не найден.")
        print("  Сначала запустите: python data/generate_data.py")
        sys.exit(1)

    df = pd.read_csv(args.data)
    print(f"  Записей: {len(df)}")
    print(f"  Признаков: {len(FEATURE_COLS)}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  ОШИБКА: отсутствуют колонки: {missing}")
        sys.exit(1)

    has_labels = "is_corrupt" in df.columns
    if has_labels:
        n_corrupt = int(df["is_corrupt"].sum())
        n_normal  = len(df) - n_corrupt
        print(f"  Нормальных: {n_normal}, Подозрительных: {n_corrupt}")
    print()

    # ── 2. Обучение ───────────────────────────────────────────────────────────
    print(f"[2/4] Обучение Isolation Forest")
    print(f"  contamination={args.contamination}, n_estimators={args.n_estimators}")
    model = TenderRiskModel(
        contamination=args.contamination,
        n_estimators=args.n_estimators,
    )
    model.train(df)
    print()

    # ── 3. Оценка качества ────────────────────────────────────────────────────
    print("[3/4] Оценка качества модели")
    if has_labels:
        metrics = model.evaluate(df)

        print_separator()
        print("  МЕТРИКИ КАЧЕСТВА (threshold = 50%)")
        print_separator()
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}  "
              f"{'★★★ Отлично' if metrics['roc_auc'] > 0.85 else '★★ Хорошо' if metrics['roc_auc'] > 0.75 else '★ Приемлемо'}")
        print(f"  Precision: {metrics['precision']:.4f}  "
              f"(из предсказанных corrupt верных)")
        print(f"  Recall:    {metrics['recall']:.4f}  "
              f"(из реальных corrupt найдено)")
        print()
        cm = metrics["confusion_matrix"]
        print("  Матрица ошибок:")
        print(f"    TP (верно corrupt):  {cm['tp']:4d}   FP (ложная тревога): {cm['fp']:4d}")
        print(f"    FN (пропущено):      {cm['fn']:4d}   TN (верно normal):   {cm['tn']:4d}")
        print()
        sd = metrics["score_distribution"]
        print("  Распределение Score:")
        print(f"    Corrupt: среднее={sd['corrupt']['mean']:5.1f}%  "
              f"медиана={sd['corrupt']['median']:5.1f}%  "
              f"[{sd['corrupt']['min']:.1f}–{sd['corrupt']['max']:.1f}]")
        print(f"    Normal:  среднее={sd['normal']['mean']:5.1f}%  "
              f"медиана={sd['normal']['median']:5.1f}%  "
              f"[{sd['normal']['min']:.1f}–{sd['normal']['max']:.1f}]")
        print_separator()
        print()
    else:
        print("  Метка 'is_corrupt' не найдена — оценка пропущена")
        print()

    # ── 4. Сохранение ─────────────────────────────────────────────────────────
    print(f"[4/4] Сохранение модели: {args.out}")
    model.save(args.out)
    print()

    # ── Пример предсказания ───────────────────────────────────────────────────
    print_separator()
    print("  ПРИМЕР ПРЕДСКАЗАНИЯ — подозрительный тендер")
    print_separator()
    suspicious = {
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
    }
    result = model.predict(suspicious)
    print(f"  Risk Score: {result['final_score']}%")
    print(f"  Уровень:    {result['risk_level'].upper()}")
    print(f"  IF Score:   {result['if_score']}")
    print()
    print("  Топ-3 причины:")
    for i, feat in enumerate(result["top_features"], 1):
        arrow = "▲" if feat["direction"] == "risk_up" else "▼"
        print(f"    {i}. {arrow} {feat['description']}")
        print(f"       Значение: {feat['value']}  |  SHAP вес: {feat['shap_weight']}")
    print()

    print_separator()
    print("  ПРИМЕР — нормальный тендер")
    print_separator()
    normal = {
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
    }
    result2 = model.predict(normal)
    print(f"  Risk Score: {result2['final_score']}%")
    print(f"  Уровень:    {result2['risk_level'].upper()}")
    print()
    print_separator("═")
    print("  ГОТОВО! Модель обучена и сохранена.")
    print(f"  Файл: {args.out}")
    print("  Следующий шаг: python api/server.py")
    print_separator("═")


if __name__ == "__main__":
    main()
