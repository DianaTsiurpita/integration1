from datetime import datetime
from pydantic import BaseModel, Field, model_validator

class PropertyFields(BaseModel):
    gr_liv_area: float = Field(gt=0, le=10000)
    total_bsmt_sf: float = Field(ge=0, le=10000)
    garage_area: float = Field(ge=0, le=2000)
    year_built: int = Field(ge=1872, le=2010)
    overall_qual: int = Field(ge=1, le=10)
    ms_zoning: str = Field(min_length=1, max_length=10)

class OrderCreate(PropertyFields):
    client_name: str = Field(min_length=2, max_length=120)

class OrderUpdate(BaseModel):
    client_name: str | None = Field(default=None, min_length=2, max_length=120)
    gr_liv_area: float | None = Field(default=None, gt=0, le=10000)
    total_bsmt_sf: float | None = Field(default=None, ge=0, le=10000)
    garage_area: float | None = Field(default=None, ge=0, le=2000)
    year_built: int | None = Field(default=None, ge=1872, le=2010)
    overall_qual: int | None = Field(default=None, ge=1, le=10)
    ms_zoning: str | None = Field(default=None, min_length=1, max_length=10)

    @model_validator(mode="after")
    def reject_explicit_nulls(self):
        for name in self.model_fields_set:
            if getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null")
        return self

class OrderRead(BaseModel):
    model_config = {"from_attributes": True}
    id: int
    client_name: str
    gr_liv_area: float
    total_bsmt_sf: float
    garage_area: float
    year_built: int
    overall_qual: int
    ms_zoning: str
    predicted_price: float | None
    created_at: datetime
