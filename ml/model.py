"""
ml/model.py
===========
Полный ML-пайплайн для выявления коррупции в тендерах.

Содержит:
  - FEATURE_COLS      — список 18 признаков
  - extract_features  — извлечение признаков из словаря тендера
  - TenderRiskModel   — класс модели (train / predict / explain / save / load)

Зависимости: scikit-learn, shap, numpy, pandas, joblib
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score

warnings.filterwarnings("ignore")

# ─── 18 признаков модели ────────────────────────────────────────────────────

FEATURE_COLS = [
    "minutes_before_deadline",   # F01  временной паттерн подачи
    "published_on_friday",       # F02  публикация в пятницу/праздник
    "acceptance_days",           # F03  срок приёма заявок (дни)
    "ip_collision",              # F04  совпадение IP участников
    "winner_win_rate",           # F05  доля побед у данного заказчика
    "director_overlap",          # F06  пересечение директоров
    "price_reduction_pct",       # F07  снижение от НМЦК
    "min_bid_gap_pct",           # F08  минимальный разрыв между ставками
    "price_vs_market_pct",       # F09  превышение над рыночной ценой
    "unique_spec_score",         # F10  уникальность ТЗ (TF-IDF скор)
    "tz_change_hours_before",    # F11  изменение ТЗ незадолго до дедлайна
    "participants_count",        # F12  число участников тендера
    "supplier_age_days",         # F13  возраст поставщика (дни)
    "revenue_vs_contract",       # F14  выручка / сумма контракта
    "address_change_days",       # F15  дней с последней смены адреса
    "amendment_count",           # F16  число доп.соглашений
    "subcontract_affiliation",   # F17  субподряд аффилированным лицам
    "historical_win_rate",       # F18  исторический win-rate поставщика
]

# Веса тематических групп признаков для Score Fusion
GROUP_WEIGHTS = {
    "timing":       {"cols": ["minutes_before_deadline", "published_on_friday",
                               "acceptance_days", "tz_change_hours_before"],
                     "weight": 0.25},
    "participants": {"cols": ["ip_collision", "winner_win_rate",
                               "director_overlap", "participants_count"],
                     "weight": 0.20},
    "price":        {"cols": ["price_reduction_pct", "min_bid_gap_pct",
                               "price_vs_market_pct"],
                     "weight": 0.22},
    "docs":         {"cols": ["unique_spec_score"],
                     "weight": 0.13},
    "supplier":     {"cols": ["supplier_age_days", "revenue_vs_contract",
                               "address_change_days"],
                     "weight": 0.10},
    "history":      {"cols": ["amendment_count", "subcontract_affiliation",
                               "historical_win_rate"],
                     "weight": 0.10},
}

# Пороги для уровней риска
RISK_THRESHOLDS = {
    "critical": 75,
    "high":     50,
    "medium":   25,
}


# ─── Вспомогательные функции ────────────────────────────────────────────────

def get_risk_level(score: float) -> str:
    """Преобразует числовой score (0–100) в уровень риска."""
    if score >= RISK_THRESHOLDS["critical"]:
        return "critical"
    elif score >= RISK_THRESHOLDS["high"]:
        return "high"
    elif score >= RISK_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def extract_features(tender: dict[str, Any]) -> dict[str, float]:
    """
    Извлекает и нормализует 18 признаков из словаря тендера.

    Параметры
    ---------
    tender : dict
        Словарь с данными тендера. Поддерживает как уже готовые признаки,
        так и сырые поля (см. маппинг ниже).

    Возвращает
    ----------
    dict[str, float]
        Словарь с 18 числовыми признаками.
    """
    f = {}

    # F01: минуты до дедлайна при подаче победителя
    f["minutes_before_deadline"] = float(
        tender.get("minutes_before_deadline", 999)
    )

    # F02: опубликовано в пятницу (1) или нет (0)
    pub = tender.get("published_on_friday", 0)
    f["published_on_friday"] = float(int(bool(pub)))

    # F03: срок приёма заявок в днях
    f["acceptance_days"] = float(tender.get("acceptance_days", 14))

    # F04: совпадение IP-адресов участников
    f["ip_collision"] = float(int(bool(tender.get("ip_collision", 0))))

    # F05: доля побед победителя у данного заказчика (0–1)
    f["winner_win_rate"] = float(
        min(1.0, max(0.0, tender.get("winner_win_rate", 0.0)))
    )

    # F06: пересечение директоров у участников
    f["director_overlap"] = float(int(bool(tender.get("director_overlap", 0))))

    # F07: снижение цены = (НМЦК - цена победителя) / НМЦК
    f["price_reduction_pct"] = float(
        min(1.0, max(0.0, tender.get("price_reduction_pct", 0.0)))
    )

    # F08: минимальный разрыв между заявками (%)
    f["min_bid_gap_pct"] = float(
        max(0.0, tender.get("min_bid_gap_pct", 0.0))
    )

    # F09: превышение над рыночной ценой (может быть отрицательным = ниже рынка)
    f["price_vs_market_pct"] = float(
        tender.get("price_vs_market_pct", 0.0)
    )

    # F10: скор уникальности ТЗ (0=стандартное, 1=уникальное)
    f["unique_spec_score"] = float(
        min(1.0, max(0.0, tender.get("unique_spec_score", 0.0)))
    )

    # F11: за сколько часов до дедлайна менялось ТЗ (0 = не менялось)
    tz_hours = tender.get("tz_change_hours_before", 0)
    # Инвертируем: маленькое число = опасно, нормируем в [0,1]
    f["tz_change_hours_before"] = float(tz_hours)

    # F12: число участников (мало = подозрительно)
    f["participants_count"] = float(
        max(1, tender.get("participants_count", 1))
    )

    # F13: возраст поставщика в днях
    f["supplier_age_days"] = float(
        max(1, tender.get("supplier_age_days", 365))
    )

    # F14: отношение выручки к сумме контракта
    f["revenue_vs_contract"] = float(
        max(0.0, tender.get("revenue_vs_contract", 1.0))
    )

    # F15: дней с последней смены юр.адреса
    f["address_change_days"] = float(
        max(0, tender.get("address_change_days", 365))
    )

    # F16: количество доп.соглашений после победы
    f["amendment_count"] = float(
        max(0, tender.get("amendment_count", 0))
    )

    # F17: субподряд аффилированным лицам
    f["subcontract_affiliation"] = float(
        int(bool(tender.get("subcontract_affiliation", 0)))
    )

    # F18: исторический win-rate поставщика (0–1)
    f["historical_win_rate"] = float(
        min(1.0, max(0.0, tender.get("historical_win_rate", 0.0)))
    )

    return f


def _compute_group_scores(feature_vec: np.ndarray) -> dict[str, float]:
    """
    Вычисляет компонентные оценки по тематическим группам.
    Возвращает словарь {group_name: score_0_to_1}.
    """
    feat_dict = dict(zip(FEATURE_COLS, feature_vec.tolist()))
    group_scores = {}

    for group, meta in GROUP_WEIGHTS.items():
        cols = meta["cols"]
        vals = [feat_dict[c] for c in cols if c in feat_dict]

        if group == "timing":
            # Малое minutes_before_deadline = плохо
            # Малое acceptance_days = плохо
            # published_on_friday = плохо
            s_timing = 1.0 - min(1.0, feat_dict["minutes_before_deadline"] / 500)
            s_accept = 1.0 - min(1.0, max(0.0, (feat_dict["acceptance_days"] - 1) / 29))
            s_fri = feat_dict["published_on_friday"]
            s_tz = 1.0 if feat_dict["tz_change_hours_before"] == 0 else \
                   min(1.0, 48 / max(1, feat_dict["tz_change_hours_before"]))
            group_scores[group] = min(1.0, (s_timing * 0.4 + s_accept * 0.3 +
                                             s_fri * 0.15 + s_tz * 0.15))

        elif group == "participants":
            s_ip = feat_dict["ip_collision"]
            s_wwr = feat_dict["winner_win_rate"]
            s_dir = feat_dict["director_overlap"]
            s_cnt = 1.0 - min(1.0, (feat_dict["participants_count"] - 1) / 14)
            group_scores[group] = min(1.0, (s_ip * 0.35 + s_wwr * 0.35 +
                                             s_dir * 0.15 + s_cnt * 0.15))

        elif group == "price":
            # Малое снижение = плохо; малый разрыв = плохо; большое превышение = плохо
            s_red = 1.0 - min(1.0, feat_dict["price_reduction_pct"] / 0.3)
            s_gap = 1.0 - min(1.0, feat_dict["min_bid_gap_pct"] / 0.1)
            s_mkt = min(1.0, max(0.0, feat_dict["price_vs_market_pct"] / 0.5))
            group_scores[group] = min(1.0, s_red * 0.45 + s_gap * 0.35 + s_mkt * 0.20)

        elif group == "docs":
            group_scores[group] = feat_dict["unique_spec_score"]

        elif group == "supplier":
            s_age = 1.0 - min(1.0, feat_dict["supplier_age_days"] / 1825)
            s_rev = 1.0 - min(1.0, max(0.0, feat_dict["revenue_vs_contract"] / 5))
            s_adr = 1.0 - min(1.0, feat_dict["address_change_days"] / 365)
            group_scores[group] = min(1.0, s_age * 0.4 + s_rev * 0.35 + s_adr * 0.25)

        elif group == "history":
            s_amd = min(1.0, feat_dict["amendment_count"] / 5)
            s_sub = feat_dict["subcontract_affiliation"]
            s_hwr = feat_dict["historical_win_rate"]
            group_scores[group] = min(1.0, s_amd * 0.3 + s_sub * 0.4 + s_hwr * 0.3)

    return group_scores


# ─── Основной класс модели ───────────────────────────────────────────────────

class TenderRiskModel:
    """
    Модель обнаружения аномалий в тендерах.

    Методы
    ------
    train(df)           — обучить на DataFrame с 18 признаками
    predict(tender)     — предсказать риск для одного тендера
    predict_batch(df)   — предсказать риск для DataFrame
    explain(tender)     — получить топ-N признаков с весами (SHAP)
    evaluate(df)        — оценить качество на размеченной выборке
    save(path)          — сохранить модель на диск
    load(path)          — загрузить модель с диска (classmethod)
    """

    MODEL_VERSION = "isolation_forest_v1.0"

    def __init__(
        self,
        contamination: float = 0.10,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._explainer: shap.TreeExplainer | None = None
        self._is_trained = False

        # Сохраняем мин/макс score сырого IF для нормировки
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    # ── Обучение ─────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> "TenderRiskModel":
        """
        Обучает модель на DataFrame.

        Параметры
        ---------
        df : pd.DataFrame
            Должен содержать все 18 колонок из FEATURE_COLS.
            Колонка 'is_corrupt' опциональна и используется только для оценки.

        Пример
        ------
        >>> model = TenderRiskModel()
        >>> model.train(df)
        """
        print("Обучение модели...")
        X = df[FEATURE_COLS].astype(float).values

        # Стандартизация признаков
        X_scaled = self._scaler.fit_transform(X)

        # Обучение Isolation Forest
        self._model.fit(X_scaled)

        # Вычислить диапазон raw scores для нормировки
        raw_scores = self._model.decision_function(X_scaled)
        self._score_min = float(raw_scores.min())
        self._score_max = float(raw_scores.max())

        # Инициализировать SHAP explainer
        print("  Инициализация SHAP explainer...")
        self._explainer = shap.TreeExplainer(self._model)
        self._is_trained = True

        print(f"  Обучено на {len(df)} тендерах")
        print(f"  Raw score range: [{self._score_min:.4f}, {self._score_max:.4f}]")
        return self

    def _raw_to_normalized(self, raw_score: float) -> float:
        """Нормирует raw IF score в [0, 1]: 1 = максимальная аномальность."""
        if self._score_max == self._score_min:
            return 0.5
        normed = (raw_score - self._score_min) / (self._score_max - self._score_min)
        return float(np.clip(1.0 - normed, 0.0, 1.0))

    # ── Предсказание ─────────────────────────────────────────────────────────

    def predict(self, tender: dict[str, Any]) -> dict[str, Any]:
        """
        Рассчитывает риск для одного тендера.

        Параметры
        ---------
        tender : dict
            Словарь с полями тендера (см. extract_features).

        Возвращает
        ----------
        dict с ключами:
            final_score       float   0–100, итоговый балл подозрительности
            risk_level        str     'low' | 'medium' | 'high' | 'critical'
            if_score          float   0–1, оценка Isolation Forest
            group_scores      dict    покомпонентные оценки по 6 группам
            top_features      list    топ-3 признака с SHAP весами
            features          dict    все 18 извлечённых признаков
        """
        self._check_trained()
        features = extract_features(tender)
        feature_vec = np.array([features[c] for c in FEATURE_COLS], dtype=float)
        X_scaled = self._scaler.transform([feature_vec])

        # Оценка Isolation Forest
        raw_if = self._model.decision_function(X_scaled)[0]
        if_score = self._raw_to_normalized(raw_if)

        # Покомпонентные оценки
        group_scores = _compute_group_scores(feature_vec)

        # Score Fusion: 55% компоненты + 45% IF
        weighted_groups = sum(
            meta["weight"] * group_scores[g]
            for g, meta in GROUP_WEIGHTS.items()
        )
        fusion_score = 0.55 * weighted_groups + 0.45 * if_score
        final_score = round(float(np.clip(fusion_score * 100, 0, 100)), 2)

        # SHAP объяснение
        top_features = self.explain(tender, top_n=3)

        return {
            "final_score": final_score,
            "risk_level": get_risk_level(final_score),
            "if_score": round(if_score, 4),
            "group_scores": {k: round(v, 4) for k, v in group_scores.items()},
            "top_features": top_features,
            "features": {k: round(v, 4) for k, v in features.items()},
            "model_version": self.MODEL_VERSION,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает риск для всего DataFrame.

        Параметры
        ---------
        df : pd.DataFrame
            Должен содержать все 18 колонок из FEATURE_COLS.

        Возвращает
        ----------
        pd.DataFrame с добавленными колонками:
            final_score, risk_level, if_score, group_scores_*, top_features
        """
        self._check_trained()
        print(f"Скоринг {len(df)} тендеров...")

        X = df[FEATURE_COLS].astype(float).values
        X_scaled = self._scaler.transform(X)

        raw_scores = self._model.decision_function(X_scaled)
        if_scores = np.array([self._raw_to_normalized(s) for s in raw_scores])

        # Вычислить групповые и финальные оценки
        results = []
        for i, row in enumerate(X):
            g = _compute_group_scores(row)
            weighted = sum(meta["weight"] * g[gn] for gn, meta in GROUP_WEIGHTS.items())
            fusion = 0.55 * weighted + 0.45 * if_scores[i]
            final = round(float(np.clip(fusion * 100, 0, 100)), 2)
            results.append({
                "final_score": final,
                "risk_level": get_risk_level(final),
                "if_score": round(float(if_scores[i]), 4),
                **{f"group_{k}": round(v, 4) for k, v in g.items()},
            })

        result_df = df.copy()
        for col in results[0]:
            result_df[col] = [r[col] for r in results]

        result_df = result_df.sort_values("final_score", ascending=False)
        print(f"  Готово. Критических: {(result_df['risk_level'] == 'critical').sum()}, "
              f"Высоких: {(result_df['risk_level'] == 'high').sum()}")
        return result_df

    # ── Объяснение ───────────────────────────────────────────────────────────

    def explain(self, tender: dict[str, Any], top_n: int = 3) -> list[dict]:
        """
        Возвращает топ-N наиболее влиятельных признаков через SHAP.

        Параметры
        ---------
        tender : dict
            Словарь с данными тендера.
        top_n : int
            Сколько признаков вернуть.

        Возвращает
        ----------
        list of dict:
            [{"feature": str, "value": float, "shap_weight": float,
              "description": str, "direction": "risk_up" | "risk_down"}, ...]
        """
        self._check_trained()
        features = extract_features(tender)
        feature_vec = np.array([features[c] for c in FEATURE_COLS], dtype=float)
        X_scaled = self._scaler.transform([feature_vec])

        shap_vals = self._explainer.shap_values(X_scaled)[0]
        # Для IF: отрицательный SHAP = больше аномальности
        shap_abs = np.abs(shap_vals)
        top_idx = np.argsort(shap_abs)[::-1][:top_n]

        descriptions = {
            "minutes_before_deadline": "Минуты до дедлайна при подаче заявки",
            "published_on_friday":     "Публикация тендера в пятницу/праздник",
            "acceptance_days":         "Срок приёма заявок (дней)",
            "ip_collision":            "Совпадение IP-адресов участников",
            "winner_win_rate":         "Win-rate победителя у этого заказчика",
            "director_overlap":        "Пересечение директоров компаний",
            "price_reduction_pct":     "Снижение цены от начальной",
            "min_bid_gap_pct":         "Минимальный разрыв между ставками",
            "price_vs_market_pct":     "Превышение над рыночной ценой",
            "unique_spec_score":       "Уникальность требований ТЗ",
            "tz_change_hours_before":  "Изменение ТЗ перед дедлайном (часов)",
            "participants_count":      "Число участников тендера",
            "supplier_age_days":       "Возраст компании-поставщика (дней)",
            "revenue_vs_contract":     "Выручка / сумма контракта",
            "address_change_days":     "Дней с последней смены адреса",
            "amendment_count":         "Количество доп.соглашений",
            "subcontract_affiliation": "Субподряд аффилированным лицам",
            "historical_win_rate":     "Исторический win-rate поставщика",
        }

        result = []
        for idx in top_idx:
            fname = FEATURE_COLS[idx]
            sv = float(shap_vals[idx])
            result.append({
                "feature": fname,
                "value": round(float(feature_vec[idx]), 4),
                "shap_weight": round(float(shap_abs[idx]), 4),
                # Отрицательный SHAP у IF = признак поднимает риск
                "direction": "risk_up" if sv < 0 else "risk_down",
                "description": descriptions.get(fname, fname),
            })

        return result

    # ── Оценка качества ──────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Оценивает качество модели на размеченной выборке.

        Параметры
        ---------
        df : pd.DataFrame
            Должен содержать 18 признаков + колонку 'is_corrupt' (0/1).

        Возвращает
        ----------
        dict с метриками: roc_auc, precision, recall, confusion_matrix,
                          score_distribution
        """
        self._check_trained()
        result_df = self.predict_batch(df)
        y_true = df["is_corrupt"].values
        y_score = result_df["final_score"].values / 100

        threshold = 0.50  # >= 50% = предсказываем corrupt
        y_pred = (y_score >= threshold).astype(int)

        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = float("nan")

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        # Confusion matrix вручную
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        # Распределение скоров
        corrupt_scores = result_df.loc[df["is_corrupt"] == 1, "final_score"]
        normal_scores  = result_df.loc[df["is_corrupt"] == 0, "final_score"]

        return {
            "roc_auc":   round(auc, 4),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "threshold": threshold,
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "score_distribution": {
                "corrupt": {
                    "mean": round(float(corrupt_scores.mean()), 2),
                    "median": round(float(corrupt_scores.median()), 2),
                    "min": round(float(corrupt_scores.min()), 2),
                    "max": round(float(corrupt_scores.max()), 2),
                },
                "normal": {
                    "mean": round(float(normal_scores.mean()), 2),
                    "median": round(float(normal_scores.median()), 2),
                    "min": round(float(normal_scores.min()), 2),
                    "max": round(float(normal_scores.max()), 2),
                },
            },
        }

    # ── Сохранение / загрузка ────────────────────────────────────────────────

    def save(self, path: str = "ml/model.pkl") -> None:
        """Сохраняет модель, скейлер и параметры нормировки на диск."""
        self._check_trained()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "scaler": self._scaler,
            "score_min": self._score_min,
            "score_max": self._score_max,
            "version": self.MODEL_VERSION,
            "feature_cols": FEATURE_COLS,
        }
        joblib.dump(payload, path, compress=3)
        print(f"Модель сохранена: {path}")

    @classmethod
    def load(cls, path: str = "ml/model.pkl") -> "TenderRiskModel":
        """Загружает модель с диска и возвращает готовый экземпляр."""
        payload = joblib.load(path)
        instance = cls.__new__(cls)
        instance._model = payload["model"]
        instance._scaler = payload["scaler"]
        instance._score_min = payload["score_min"]
        instance._score_max = payload["score_max"]
        instance.MODEL_VERSION = payload.get("version", "unknown")
        instance._is_trained = True
        # Восстановить SHAP explainer
        instance._explainer = shap.TreeExplainer(instance._model)
        print(f"Модель загружена: {path}  (версия: {instance.MODEL_VERSION})")
        return instance

    def _check_trained(self):
        if not self._is_trained:
            raise RuntimeError("Модель не обучена. Вызовите .train(df) или .load(path)")
