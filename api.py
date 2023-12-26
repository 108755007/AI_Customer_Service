from fastapi import FastAPI
import sys
from utils.log import logger
from dotenv import load_dotenv
import datetime
import torch
import collections
from db import DBhelper
import torch.nn.functional as F

sys.path.append("..")
load_dotenv()
from AI_customer_service import QA_api, ChatGPT_AVD
from AI_Search import AI_Search
from AI_customer_service_2 import AICustomerAPI

description = """
# AI_customer_service
---
"""
tags_metadata = [
    {
        "name": "AI_service",
        "description": "QABOT"
    },
    {
        "name": "AI_Search",
        "description": "AI_Search"
    },
    {
        "name": "judge",
        "description": "judge"
    },
    {
        "name": "get_product",
        "description": "update_status"
    },
    {
        "name": "get_description",
        "description": "get_description"
    },
    {
        "name": "similarity",
        "description": "similarity"
    }
]

###讀取套件
app = FastAPI(title="hodo_ai", description=description, openapi_tags=tags_metadata)

# _AI_Search = AI_Search()
AI_judge = AICustomerAPI()


def check_status(web_id, group_id):
    timestamp = int(datetime.datetime.now().timestamp()) - 60
    q = f"""SELECT count(*) FROM web_push.AI_service_recommend_status x WHERE web_id ='{web_id}' and group_id = '{group_id}' and status < 2 and `timestamp` > {timestamp}"""
    data = DBhelper('jupiter_new').ExecuteSelect(q)
    return True if data[0][0] else False


def get_tag_embedding():
    q = f"""SELECT web_id, question, ans,question_embedding  FROM AI_service_similarity"""
    print('讀取embedding資料中.....')
    ans_dict = collections.defaultdict(list)
    question_emb = collections.defaultdict(list)
    data = DBhelper('jupiter_new').ExecuteSelect(q)
    for web_id, question, ans, question_embedding in data:
        ans_dict[web_id].append((question, ans))
        question_emb[web_id].append(eval(question_embedding))
    question_emb_tensor = {}
    for web_id, emb in question_emb.items():
        question_emb_tensor[web_id] = torch.tensor(emb)
    return ans_dict, question_emb_tensor


a_dict, q_emb_tensor = get_tag_embedding()


@app.get("/AI_service", tags=["AI_service"])
def ai_service(web_id: str = '', message: str = '', group_id: str = '', product: bool = True, lang: str = '中文'):
    if web_id == '' or message == '' or group_id == '':
        return {"message": "no sentence input or no web_id", "message": ""}
    return AI_judge.qa(web_id, message, group_id, find_dpa=product, lang=lang)


@app.get("/update_product", tags=["get_product"])
def ai_update_product(web_id: str = '', group_id: str = ''):
    AI_judge.update_recommend_status(web_id, group_id, 1)
    return 'ok'


@app.get("/similarity", tags=["similarity"])
def get_similarity_avivid(web_id: str = '', group_id: str = '', text: str = ''):
    if web_id not in q_emb_tensor:
        return
    emb = AI_judge.ask_gpt(message=text, model='gpt-text')
    cos_sim_curr = F.cosine_similarity(q_emb_tensor[web_id], torch.tensor(emb), dim=1)
    if float(cos_sim_curr.topk(1).values[0]) > 0.9:
        print(f'資料庫有相似問題：{a_dict[web_id][int(cos_sim_curr.topk(1).indices[0])][0]}')
        return a_dict[web_id][int(cos_sim_curr.topk(1).indices[0])][1]


@app.get("/get_description", tags=["get_description"])
def ai_description(title: str = ''):
    res = AI_judge.get_des(title)
    return res


@app.get("/judge", tags=["judge"])
def ai_service_judge(web_id: str = '', group_id: str = '', message: str = ''):
    status = check_status(web_id, group_id)
    print(f'{group_id}:的狀態是{status}')
    tr = False
    lang = '繁體中文'
    reply = "" if status else "您好，我是客服機器人小禾！"
    if web_id == 'avividai':
        lang = AI_judge.check_lang(message)
        print(f'{group_id}:分析出的語言是：{lang}')
        if lang not in ['chinese', 'Chinese', '中文', '國語']:
            tr = True
    custom_judge = AI_judge.get_judge(message)
    true_count = sum(1 for i in custom_judge.values() if i == 'True')
    if true_count > 1 or true_count == 0:
        reply += '請稍候一下我們將盡快為您解答'
        if tr:
            reply = AI_judge.translate(lang, reply).split("'translation':")[-1].replace('}', '')
        types = 6
    elif custom_judge.get('Inquiry about product information') == 'True':
        reply += '正在為您查詢商品,稍等一下呦！'
        if tr:
            reply = AI_judge.translate(lang, reply).split("'translation':")[-1].replace('}', '')
        types = 1
    elif custom_judge.get('Requesting returns or exchanges') == 'True':
        reply += '將為您提供退換貨說明,請稍待～'
        if tr:
            reply = AI_judge.translate(lang, reply)
        types = 2
    elif custom_judge.get('General inquiries') == 'True':
        reply += '請稍等,將為您提供相關資訊！'
        if tr:
            reply = AI_judge.translate(lang, reply)
        types = 3
    elif custom_judge.get('Simple Greeting or Introduction') == 'True':
        if status:
            reply += '你好！'
        reply += '謝謝您對我們的關注,祝您愉快！'
        if tr:
            reply = AI_judge.translate(lang, reply)
        types = 4
    elif custom_judge.get('Simple expression of gratitude') == 'True':
        reply += '很高興能解決您的問題,祝您愉快！'
        if tr:
            reply = AI_judge.translate(lang, reply)
        types = 5
    elif custom_judge.get('Unable to determine intent or other') == 'True':
        reply += '請稍候一下我們將盡快為您解答'
        if tr:
            reply = AI_judge.translate(lang, reply)
        types = 6
    print(f'回傳判斷：{types}')
    return types, reply, lang

# @app.get("/AI_Search", tags=["AI_Search"])
# def AI_serch(web_id:str='',message:str=''):
#     if web_id == '' or message == '':
#         return {"message": "no sentence input or no web_id", "message": ""}
#     res = _AI_Search.main(web_id,message)
#     return res
#
# @app.get("/AI_Search2", tags=["AI_Search"])
# def AI_serch2(web_id:str='',message:str=''):
#     if web_id == '' or message == '':
#         return {"message": "no sentence input or no web_id", "message": ""}
#     res = _AI_Search.main_sim(web_id, message)
#     return res
