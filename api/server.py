"""
api/server.py
=============
FastAPI REST-сервер для модели выявления коррупции.

Эндпоинты:
  GET  /                    — статус сервера
  GET  /health              — проверка здоровья + версия модели
  POST /score               — рассчитать риск одного тендера
  POST /score/batch         — рассчитать риск для списка тендеров
  POST /explain             — получить объяснение (топ-N признаков)
  GET  /features            — список всех 18 признаков с описаниями
  GET  /docs                — Swagger UI (авто)

Запуск:
    python api/server.py
    uvicorn api.server:app --reload --port 8000
"""

import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

from ml.model import TenderRiskModel, FEATURE_COLS, GROUP_WEIGHTS

# ── Загрузка модели ──────────────────────────────────────────────────────────

MODEL_PATH = "ml/model.pkl"

try:
    model = TenderRiskModel.load(MODEL_PATH)
    print(f"Модель загружена: {MODEL_PATH}")
except FileNotFoundError:
    print(f"ПРЕДУПРЕЖДЕНИЕ: модель не найдена ({MODEL_PATH})")
    print("Запустите сначала: python train.py")
    model = None


# ── Pydantic схемы ───────────────────────────────────────────────────────────

class TenderInput(BaseModel):
    """Входные данные тендера для скоринга."""

    # Идентификатор (опционально)
    lot_id: Optional[str] = Field(None, description="Идентификатор тендера (опционально)")

    # 18 признаков
    minutes_before_deadline: float = Field(
        ..., ge=0, description="Минуты до дедлайна при подаче заявки победителя"
    )
    published_on_friday: int = Field(
        ..., ge=0, le=1, description="Публикация в пятницу/праздник (0 или 1)"
    )
    acceptance_days: float = Field(
        ..., ge=0, description="Срок приёма заявок в днях"
    )
    ip_collision: int = Field(
        ..., ge=0, le=1, description="Совпадение IP участников (0 или 1)"
    )
    winner_win_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Доля побед победителя у этого заказчика (0–1)"
    )
    director_overlap: int = Field(
        ..., ge=0, le=1, description="Пересечение директоров (0 или 1)"
    )
    price_reduction_pct: float = Field(
        ..., ge=0.0, le=1.0, description="Снижение от НМЦК (доля, 0–1)"
    )
    min_bid_gap_pct: float = Field(
        ..., ge=0.0, description="Минимальный разрыв между ставками (доля)"
    )
    price_vs_market_pct: float = Field(
        ..., description="Превышение над рыночной ценой (доля, может быть отрицательным)"
    )
    unique_spec_score: float = Field(
        ..., ge=0.0, le=1.0, description="Уникальность ТЗ (0=стандартное, 1=уникальное)"
    )
    tz_change_hours_before: float = Field(
        ..., ge=0, description="Часов до дедлайна при последнем изменении ТЗ (0=не менялось)"
    )
    participants_count: int = Field(
        ..., ge=1, description="Число участников тендера"
    )
    supplier_age_days: int = Field(
        ..., ge=1, description="Возраст компании-поставщика в днях"
    )
    revenue_vs_contract: float = Field(
        ..., ge=0.0, description="Выручка поставщика / сумма контракта"
    )
    address_change_days: int = Field(
        ..., ge=0, description="Дней с последней смены юр.адреса"
    )
    amendment_count: int = Field(
        ..., ge=0, description="Количество доп.соглашений после победы"
    )
    subcontract_affiliation: int = Field(
        ..., ge=0, le=1, description="Субподряд аффилированным лицам (0 или 1)"
    )
    historical_win_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Исторический win-rate поставщика (0–1)"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "lot_id": "LOT-00042",
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
    }}


class ScoreResponse(BaseModel):
    """Ответ с risk score тендера."""
    lot_id: Optional[str]
    final_score: float
    risk_level: str
    if_score: float
    group_scores: dict[str, float]
    top_features: list[dict]
    features: dict[str, float]
    model_version: str


class BatchInput(BaseModel):
    tenders: list[TenderInput] = Field(..., min_length=1, max_length=1000)


class BatchScoreResponse(BaseModel):
    count: int
    results: list[dict]
    summary: dict


class ExplainResponse(BaseModel):
    lot_id: Optional[str]
    top_features: list[dict]
    final_score: float
    risk_level: str


# ── FastAPI приложение ────────────────────────────────────────────────────────

app = FastAPI(
    title="Tender Risk API",
    description="ИИ-система выявления коррупции в тендерных закупках",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model() -> TenderRiskModel:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Запустите: python train.py"
        )
    return model


# ── Эндпоинты ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Статус"])
def root():
    """Статус сервера."""
    return {
        "service": "Tender Risk API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "docs": "/docs",
    }


@app.get("/health", tags=["Статус"])
def health():
    """Проверка здоровья сервиса."""
    m = get_model()
    return {
        "status": "ok",
        "model_version": m.MODEL_VERSION,
        "features_count": len(FEATURE_COLS),
    }


@app.post("/score", response_model=ScoreResponse, tags=["Скоринг"])
def score_tender(tender: TenderInput):
    """
    Рассчитать Risk Score для одного тендера.

    Возвращает:
    - `final_score` — итоговый балл подозрительности (0–100%)
    - `risk_level` — low / medium / high / critical
    - `group_scores` — компонентные оценки по 6 группам признаков
    - `top_features` — топ-3 объясняющих признака (SHAP)
    """
    m = get_model()
    tender_dict = tender.model_dump(exclude={"lot_id"})
    result = m.predict(tender_dict)
    result["lot_id"] = tender.lot_id
    return result


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["Скоринг"])
def score_batch(batch: BatchInput):
    """
    Рассчитать Risk Score для списка тендеров (до 1000 за раз).

    Результаты отсортированы по убыванию final_score.
    """
    import pandas as pd

    m = get_model()
    records = [t.model_dump(exclude={"lot_id"}) for t in batch.tenders]
    lot_ids = [t.lot_id for t in batch.tenders]

    df = pd.DataFrame(records)
    result_df = m.predict_batch(df)

    result_df["lot_id"] = lot_ids

    risk_counts = result_df["risk_level"].value_counts().to_dict()
    results = result_df[
        ["lot_id", "final_score", "risk_level", "if_score"]
        + [c for c in result_df.columns if c.startswith("group_")]
    ].to_dict(orient="records")

    return {
        "count": len(results),
        "results": results,
        "summary": {
            "critical": risk_counts.get("critical", 0),
            "high":     risk_counts.get("high", 0),
            "medium":   risk_counts.get("medium", 0),
            "low":      risk_counts.get("low", 0),
            "avg_score": round(float(result_df["final_score"].mean()), 2),
        },
    }


@app.post("/explain", response_model=ExplainResponse, tags=["Объяснимость"])
def explain_tender(
    tender: TenderInput,
    top_n: int = Query(default=5, ge=1, le=18,
                       description="Количество признаков для объяснения")
):
    """
    Получить объяснение предсказания — топ-N наиболее влиятельных признаков.

    Использует SHAP (SHapley Additive exPlanations).
    `direction: risk_up` — признак увеличивает риск.
    `direction: risk_down` — признак снижает риск.
    """
    m = get_model()
    tender_dict = tender.model_dump(exclude={"lot_id"})
    top_features = m.explain(tender_dict, top_n=top_n)
    result = m.predict(tender_dict)

    return {
        "lot_id": tender.lot_id,
        "top_features": top_features,
        "final_score": result["final_score"],
        "risk_level": result["risk_level"],
    }


@app.get("/features", tags=["Справка"])
def list_features():
    """Список всех 18 признаков с описаниями и группами."""
    feature_info = {
        "minutes_before_deadline": {"group": "timing",       "type": "float", "desc": "Минуты до дедлайна при подаче заявки (меньше = подозрительнее)"},
        "published_on_friday":     {"group": "timing",       "type": "int",   "desc": "Публикация в пятницу или праздник (0/1)"},
        "acceptance_days":         {"group": "timing",       "type": "float", "desc": "Срок приёма заявок (дни; меньше = подозрительнее)"},
        "ip_collision":            {"group": "participants", "type": "int",   "desc": "Совпадение IP у двух и более участников (0/1)"},
        "winner_win_rate":         {"group": "participants", "type": "float", "desc": "Доля побед победителя у данного заказчика (0–1)"},
        "director_overlap":        {"group": "participants", "type": "int",   "desc": "Пересечение директоров компаний-участников (0/1)"},
        "price_reduction_pct":     {"group": "price",       "type": "float", "desc": "Снижение от НМЦК, доля 0–1 (меньше = подозрительнее)"},
        "min_bid_gap_pct":         {"group": "price",       "type": "float", "desc": "Минимальный разрыв между ставками (доля; меньше = подозрительнее)"},
        "price_vs_market_pct":     {"group": "price",       "type": "float", "desc": "Превышение над рыночной ценой (доля; больше = подозрительнее)"},
        "unique_spec_score":       {"group": "docs",        "type": "float", "desc": "Уникальность ТЗ (0=стандартное, 1=уникальное заточенное)"},
        "tz_change_hours_before":  {"group": "docs",        "type": "float", "desc": "Часов до дедлайна при изменении ТЗ (0=не менялось)"},
        "participants_count":      {"group": "participants", "type": "int",   "desc": "Число участников (меньше = подозрительнее)"},
        "supplier_age_days":       {"group": "supplier",    "type": "int",   "desc": "Возраст поставщика в днях (меньше = подозрительнее)"},
        "revenue_vs_contract":     {"group": "supplier",    "type": "float", "desc": "Выручка / сумма контракта (меньше 1 = подозрительно)"},
        "address_change_days":     {"group": "supplier",    "type": "int",   "desc": "Дней с последней смены адреса (меньше = подозрительнее)"},
        "amendment_count":         {"group": "history",     "type": "int",   "desc": "Количество доп.соглашений после победы (больше = подозрительнее)"},
        "subcontract_affiliation": {"group": "history",     "type": "int",   "desc": "Субподряд аффилированным структурам заказчика (0/1)"},
        "historical_win_rate":     {"group": "history",     "type": "float", "desc": "Исторический win-rate поставщика (больше = подозрительнее)"},
    }

    groups = {}
    for fname, info in feature_info.items():
        g = info["group"]
        if g not in groups:
            groups[g] = {"weight": GROUP_WEIGHTS[g]["weight"], "features": []}
        groups[g]["features"].append({"name": fname, **info})

    return {
        "total_features": len(FEATURE_COLS),
        "groups": groups,
    }


# ── Запуск ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
