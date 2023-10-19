from fastapi import FastAPI
from AI_traffic_assistant import AiTraffic
import pandas as pd
from web_id_similarity import web_id_similarity
from db import DBhelper
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
    },
    {
        "name": "similarity",
        "description": "web_id similarity",
    }
]

app = FastAPI(title="traffic_assistant", description=description, openapi_tags=tags_metadata)

AI_traffic = AiTraffic()


@app.get("/title", tags=["generate_title"])
def title(web_id: str = 'test', user_id: str = '', keywords: str = '', web_id_main: str = '', article: str = '', types: int = 1):
    if types != 1 and article =='':
        return '請輸入文章內容'
    res_list = AI_traffic.get_title(web_id=web_id, user_id=user_id, keywords=keywords, web_id_main=web_id_main, article=article, types=types)
    return {i+1: v for i, v in enumerate(res_list)}

@app.get("/sub-heading", tags=["generate_sub_title"])
def subtitle(web_id: str = 'test', user_id: str = '', title: str = '', types: int = 1):
    return AI_traffic.get_sub_title(title, user_id, web_id, types)


@app.get("/articles", tags=["generate_articles"])
def articles_api(web_id: str = 'test', user_id: str = '', title: str = '', keywords: str = '', subtitles1: str = '',
                 subtitles2: str = '', subtitles3: str = '', subtitles4: str = '', subtitles5: str = '', types: int = 1):
    res = AI_traffic.generate_articles(title=title, keywords=keywords, user_id=user_id, web_id=web_id, types=types,
                                       subtitle_list=[subtitles1, subtitles2, subtitles3, subtitles4, subtitles5], ta=[])
    columns = ['user_id', 'web_id', 'type']
    update_data = [user_id, web_id, types]
    d = 0
    if not subtitles1 and not subtitles2 and not subtitles3 and not subtitles4 and not subtitles5:
        columns.append('article_1')
        update_data.append(res[0])
        DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([update_data],
                                        columns=columns), db='sunscribe', table='ai_article',
                                      chunk_size=100000, is_ssh=False)
        return res
    for i, v in enumerate([subtitles1, subtitles2, subtitles3, subtitles4, subtitles5]):
        if v:
            columns.append(f"article_{i+1}")
            update_data.append(res[d])
            d += 1
    DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([update_data],
                                               columns=columns), db='sunscribe', table='ai_article',
                                  chunk_size=100000, is_ssh=False)
    return res

@app.get("/articles_ta_2", tags=["generate_articles_TA"])
def articles_ta(web_id: str = 'test', user_id: str = '', title: str = '', keywords: str = '', subtitles1: str = '',
                subtitles2: str = '', subtitles3: str = '', subtitles4: str = '', subtitles5: str = '', types: int = 1,
                gender: str = '', age: str = '', Income: str = '', style: str = '', interests: str = '', occupation: str = ''):
    res = AI_traffic.generate_articles(title=title, keywords=keywords, user_id=user_id, web_id=web_id, types=types,
                                       subtitle_list=[subtitles1, subtitles2, subtitles3, subtitles4, subtitles5],
                                       ta=[gender, age, Income, interests, occupation, style])
    columns = ['user_id', 'web_id', 'type']
    update_data = [user_id, web_id, types]
    if not subtitles1 and not subtitles2 and not subtitles3 and not subtitles4 and not subtitles5:
        columns.append('article_1')
        update_data.append(res[0])
        DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([update_data],columns=columns), db='sunscribe', table='ai_article',
                                      chunk_size=100000, is_ssh=False)
        return res
    d = 0
    for i, v in enumerate([subtitles1, subtitles2, subtitles3, subtitles4, subtitles5]):
        if v:
            columns.append(f"article_{i+1}")
            update_data.append(res[d])
            d += 1
    DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([update_data],
                                               columns=columns), db='sunscribe', table='ai_article',
                                  chunk_size=100000, is_ssh=False)
    return res

@app.get("/check", tags=["check"])
def checkdef(test: str = ''):
    if test:
        return True

@app.get("/similarity", tags=["similarity"])
def web_id_similarity(test: int = 1):
    if test:
        web_id_similarity()
        return 'ok'