from pydantic import BaseModel


class PredictInput(BaseModel):
    """Schema base de entrada para predição."""
    pass


class PredictOutput(BaseModel):
    """Schema base de saída da predição."""
    pass
