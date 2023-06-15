from src.models.model_wrapper import ModelWrapperBase


class DinoVITWrapper(ModelWrapperBase):
    NAME = "DinoVIT"

    SETTINGS = {
        "stride": {
            "type": "slider",
            "min": 1,
            "max": 10,
            "default": 4,
        }
    }


class TestWrapper(ModelWrapperBase):
    NAME = "Test"

    SETTINGS = None
