from pydantic import BaseModel

class Validator(BaseModel):
    @classmethod
    def fields(cls):
        return list(cls.__fields__.keys())
