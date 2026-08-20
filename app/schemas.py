from typing import Literal

from pydantic import BaseModel, Field

Transmission = Literal["Manual", "Automatic"]
SellerType = Literal["Dealer", "Individual"]
FuelType = Literal["Petrol", "Diesel", "CNG", "LPG", "Electric"]


class VehicleFeatures(BaseModel):
    Car_Name: str = Field(min_length=1, max_length=60)
    Year: int = Field(ge=1990, le=2026)
    Kms_Driven: int = Field(ge=0, le=1_000_000)
    Fuel_Type: FuelType
    Seller_Type: SellerType
    Transmission: Transmission
    Owner: int = Field(ge=0, le=5)


class PriceResponse(BaseModel):
    estimated_price: float
    currency: str
    model_version: str


class PricePredictionHistoryResponse(BaseModel):
    id: int
    created_at: str
    car_name: str
    year: int
    kms_driven: int
    fuel_type: str
    seller_type: str
    transmission: str
    owner: int
    estimated_price: float
    model_version: str
