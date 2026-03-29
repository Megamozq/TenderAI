"""
tests/test_model.py
==================
Автоматические тесты модели.

Запуск:
    python tests/test_model.py
    python -m pytest tests/test_model.py -v
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from ml.model import (
    TenderRiskModel, extract_features, get_risk_level,
    FEATURE_COLS, GROUP_WEIGHTS
)


SUSPICIOUS = {
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

NORMAL = {
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


def make_trained_model() -> TenderRiskModel:
    """Создаёт и обучает модель на синтетических данных."""
    np.random.seed(42)
    n = 300
    data = {col: np.random.uniform(0, 1, n) for col in FEATURE_COLS}
    df = pd.DataFrame(data)
    model = TenderRiskModel(contamination=0.1, n_estimators=50)
    model.train(df)
    return model


# ─── Тесты ──────────────────────────────────────────────────────────────────

tests_passed = 0
tests_failed = 0


def test(name: str, condition: bool, detail: str = ""):
    global tests_passed, tests_failed
    if condition:
        print(f"  ✓ {name}")
        tests_passed += 1
    else:
        print(f"  ✗ {name}" + (f": {detail}" if detail else ""))
        tests_failed += 1


def section(name: str):
    print(f"\n── {name} ──────────────────────")


# ── 1. extract_features ──────────────────────────────────────────────────────
section("1. extract_features")
f = extract_features(SUSPICIOUS)

test("Возвращает dict", isinstance(f, dict))
test("Содержит 18 ключей", len(f) == 18, str(len(f)))
test("Все ключи из FEATURE_COLS", set(f.keys()) == set(FEATURE_COLS))
test("Все значения — float", all(isinstance(v, float) for v in f.values()))
test("winner_win_rate в [0,1]", 0.0 <= f["winner_win_rate"] <= 1.0)
test("participants_count >= 1", f["participants_count"] >= 1)

# Тест с неполными данными — должен заполнить дефолтами
f_partial = extract_features({"minutes_before_deadline": 10, "participants_count": 3})
test("Работает с неполным dict", len(f_partial) == 18)
test("Дефолтные значения числовые", all(isinstance(v, float) for v in f_partial.values()))


# ── 2. get_risk_level ────────────────────────────────────────────────────────
section("2. get_risk_level")
test("0% → low",      get_risk_level(0) == "low")
test("24.9% → low",   get_risk_level(24.9) == "low")
test("25% → medium",  get_risk_level(25) == "medium")
test("49.9% → medium",get_risk_level(49.9) == "medium")
test("50% → high",    get_risk_level(50) == "high")
test("74.9% → high",  get_risk_level(74.9) == "high")
test("75% → critical",get_risk_level(75) == "critical")
test("100% → critical",get_risk_level(100) == "critical")


# ── 3. TenderRiskModel — обучение ────────────────────────────────────────────
section("3. TenderRiskModel.train()")
model = make_trained_model()
test("Модель обучена", model._is_trained)
test("Скейлер обучен", model._scaler is not None)
test("SHAP explainer инициализирован", model._explainer is not None)
test("score_min < score_max", model._score_min < model._score_max)


# ── 4. predict ───────────────────────────────────────────────────────────────
section("4. TenderRiskModel.predict()")
res_s = model.predict(SUSPICIOUS)
res_n = model.predict(NORMAL)

test("Возвращает dict", isinstance(res_s, dict))
test("final_score ∈ [0, 100]", 0 <= res_s["final_score"] <= 100,
     str(res_s["final_score"]))
test("risk_level — строка", isinstance(res_s["risk_level"], str))
test("risk_level ∈ {low,medium,high,critical}",
     res_s["risk_level"] in {"low", "medium", "high", "critical"})
test("if_score ∈ [0, 1]", 0 <= res_s["if_score"] <= 1)
test("group_scores содержит 6 групп", len(res_s["group_scores"]) == 6)
test("top_features — список", isinstance(res_s["top_features"], list))
test("top_features содержит 3 элемента", len(res_s["top_features"]) == 3)
test("features содержит 18 признаков", len(res_s["features"]) == 18)
test("model_version задан", bool(res_s.get("model_version")))

# Подозрительный должен иметь score выше нормального
test("Подозрительный score > Нормальный score",
     res_s["final_score"] > res_n["final_score"],
     f"{res_s['final_score']:.1f} vs {res_n['final_score']:.1f}")

# Структура top_features
tf = res_s["top_features"][0]
test("top_feature содержит 'feature'",    "feature" in tf)
test("top_feature содержит 'value'",      "value" in tf)
test("top_feature содержит 'shap_weight'","shap_weight" in tf)
test("top_feature содержит 'direction'",  "direction" in tf)
test("direction ∈ {risk_up, risk_down}",
     tf["direction"] in {"risk_up", "risk_down"})


# ── 5. explain ───────────────────────────────────────────────────────────────
section("5. TenderRiskModel.explain()")
exp5 = model.explain(SUSPICIOUS, top_n=5)
exp1 = model.explain(NORMAL, top_n=1)

test("top_n=5 возвращает 5 признаков", len(exp5) == 5, str(len(exp5)))
test("top_n=1 возвращает 1 признак",   len(exp1) == 1)
test("Признаки уникальны", len({e["feature"] for e in exp5}) == 5)
test("Все shap_weight >= 0", all(e["shap_weight"] >= 0 for e in exp5))
test("Все features есть в FEATURE_COLS",
     all(e["feature"] in FEATURE_COLS for e in exp5))


# ── 6. predict_batch ─────────────────────────────────────────────────────────
section("6. TenderRiskModel.predict_batch()")
df = pd.DataFrame([SUSPICIOUS, NORMAL, SUSPICIOUS])
result_df = model.predict_batch(df)

test("Возвращает DataFrame", isinstance(result_df, pd.DataFrame))
test("Количество строк сохранено", len(result_df) == 3)
test("Колонка final_score есть", "final_score" in result_df.columns)
test("Колонка risk_level есть",  "risk_level" in result_df.columns)
test("Все final_score в [0,100]",
     result_df["final_score"].between(0, 100).all())
test("DataFrame отсортирован по убыванию",
     result_df["final_score"].iloc[0] >= result_df["final_score"].iloc[-1])


# ── 7. save / load ───────────────────────────────────────────────────────────
section("7. save / load")
import tempfile, os
with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    tmp_path = f.name

try:
    model.save(tmp_path)
    test("Файл создан", Path(tmp_path).exists())
    test("Файл не пустой", Path(tmp_path).stat().st_size > 1000)

    loaded = TenderRiskModel.load(tmp_path)
    test("Загружена обученная модель", loaded._is_trained)
    test("SHAP explainer восстановлен", loaded._explainer is not None)

    res_orig   = model.predict(SUSPICIOUS)["final_score"]
    res_loaded = loaded.predict(SUSPICIOUS)["final_score"]
    test("Предсказания до и после загрузки совпадают",
         abs(res_orig - res_loaded) < 0.01,
         f"{res_orig} vs {res_loaded}")
finally:
    os.unlink(tmp_path)


# ── 8. evaluate ──────────────────────────────────────────────────────────────
section("8. evaluate")
np.random.seed(0)
n_eval = 200
eval_data = {col: np.random.uniform(0, 1, n_eval) for col in FEATURE_COLS}
eval_df = pd.DataFrame(eval_data)
eval_df["is_corrupt"] = np.random.randint(0, 2, n_eval)

metrics = model.evaluate(eval_df)
test("roc_auc возвращается", "roc_auc" in metrics)
test("roc_auc ∈ [0, 1]", 0 <= metrics["roc_auc"] <= 1)
test("confusion_matrix есть", "confusion_matrix" in metrics)
cm = metrics["confusion_matrix"]
test("CM содержит tp/fp/tn/fn", all(k in cm for k in ["tp","fp","tn","fn"]))
test("CM суммируется в n_eval",
     cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == n_eval,
     str(cm))
test("score_distribution есть", "score_distribution" in metrics)


# ── 9. Ошибки при вызове без обучения ────────────────────────────────────────
section("9. Обработка ошибок")
untrained = TenderRiskModel()
try:
    untrained.predict(SUSPICIOUS)
    test("RuntimeError при predict без обучения", False)
except RuntimeError:
    test("RuntimeError при predict без обучения", True)

try:
    untrained.save("/tmp/test_untrained.pkl")
    test("RuntimeError при save без обучения", False)
except RuntimeError:
    test("RuntimeError при save без обучения", True)


# ── Итог ─────────────────────────────────────────────────────────────────────
print(f"\n{'═'*50}")
total = tests_passed + tests_failed
print(f"  Результат: {tests_passed}/{total} тестов прошло")
if tests_failed == 0:
    print("  ✓ Все тесты пройдены!")
else:
    print(f"  ✗ Провалено: {tests_failed}")
print(f"{'═'*50}\n")

sys.exit(0 if tests_failed == 0 else 1)
