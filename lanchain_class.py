from pydantic import BaseModel, Field


class title_1(BaseModel):
    title: str = Field(description="str of title of data")


class title_5(BaseModel):
    title: list = Field(description="list of title of data")

class sub_title(BaseModel):
    sub_title: list = Field(description="list of sub-title of data")