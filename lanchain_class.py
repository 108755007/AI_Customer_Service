from pydantic import BaseModel, Field


class title_1(BaseModel):
    title: str = Field(description="str of title of data")


class title_5(BaseModel):
    title: list = Field(description="list of title of data")