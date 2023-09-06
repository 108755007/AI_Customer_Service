from fastapi import FastAPI
from AI_traffic_assistant import AiTraffic

description = """
traffic_assistant
author:Yu
---
"""
tags_metadata = [
    {
        "name": "generate_title",
        "description": "The title will be generated based on the article corresponding to the entered keywords",
    },
    {
        "name": "generate_sub_title",
        "description": "Generate subtitles based on the title",
    },
    {
        "name": "generate_articles",
        "description": "Depending on the number of input subtitles, articles of different lengths are generated. no TA",
    },
    {
        "name": "generate_articles_TA",
        "description": "Depending on the number of input subtitles, articles of different lengths are generated.",
    },
    {
        "name": "check",
        "description": "confirm connection",
    }
]

app = FastAPI(title="traffic_assistant", description=description, openapi_tags=tags_metadata)

AI_traffic = AiTraffic()


@app.get("/title", tags=["generate_title"])
def title(web_id: str = 'test', user_id: str = '', keyword: str = '',web_id_main: str = '', article: str = '', types: int = 1):
    if types != 1 and article =='':
        return '請輸入文章內容'
    return AI_traffic.get_title(web_id=web_id, user_id=user_id, keywords=keyword, web_id_main=web_id_main, article=article, types=types)

@app.get("/sub-heading", tags=["generate_sub_title"])
def subtitle(web_id: str = 'test', user_id: str = '', title: str = '', types: int = 1):
    return AI_traffic.get_sub_title(title, user_id, web_id, types)


@app.get("/articles", tags=["generate_articles"])
def articles_api(web_id: str = 'test', user_id: str = '', title: str = '', keywords: str = '', subtitles1: str = '',
                 subtitles2: str = '', subtitles3: str = '', subtitles4: str = '', subtitles5: str = '', types: int = 1):
    return AI_traffic.generate_articles(title=title, keywords=keywords, user_id=user_id, web_id=web_id, types=types,
                                        subtitle_list=[subtitles1, subtitles2, subtitles3, subtitles4, subtitles5],
                                        ta=[])


@app.get("/articles_ta_2", tags=["generate_articles_TA"])
def articles_ta(web_id: str = 'test', user_id: str = '', title: str = '', keywords: str = '', subtitles1: str = '',
                subtitles2: str = '', subtitles3: str = '', subtitles4: str = '', subtitles5: str = '', types: int = 1,
                gender: str = '', age: str = '', Income: str = '', style: str = '', interests: str = '', occupation: str = ''):
    return AI_traffic.generate_articles(title=title, keywords=keywords, user_id=user_id, web_id=web_id, types=types,
                                        subtitle_list=[subtitles1, subtitles2, subtitles3, subtitles4, subtitles5],
                                        ta=[gender, age, Income, interests, occupation, style])
