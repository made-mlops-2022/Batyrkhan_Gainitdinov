from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, validator


class Illness(BaseModel):
    age: float
    sex: Literal[0, 1]
    cp: Literal[0, 1, 2, 3]
    trestbps: float
    chol: float
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float
    exang: Literal[0, 1]
    oldpeak: float
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[0, 1, 2]


    @validator('age')
    def age_val(cls, val):
        if val > 100 or val < 0:
            raise HTTPException(status_code=404, detail='age is not correct')
        return val

    @validator('trestbps')
    def trestbps_val(cls, val):
        if val < 25 or val > 270:
            raise HTTPException(status_code=404, detail='trestbps is not correct')
        return val

    @validator('chol')
    def chol_validator(cls, val):
        if val < 50 or val > 650:
            raise HTTPException(status_code=404, detail='chol is not correct')
        return val

    @validator('thalach')
    def thalach_validator(cls, val):
        if val < 10 or val > 260:
            raise HTTPException(status_code=404, detail='thalach is not correct')
        return val

    @validator('oldpeak')
    def oldpeak_validator(cls, val):
        if val < 0 or val > 9:
            raise HTTPException(status_code=404, detail='oldpeak is not correct')
        return val