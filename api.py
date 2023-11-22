from fastapi import FastAPI
import sys
from utils.log import logger
from dotenv import load_dotenv
sys.path.append("..")
load_dotenv()
from AI_customer_service import QA_api,ChatGPT_AVD
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
    }
]

###讀取套件
app = FastAPI(title="hodo_ai", description=description, openapi_tags=tags_metadata)

AI_customer_service = QA_api('line', logger())
#_AI_Search = AI_Search()
AI_judge = AICustomerAPI()




@app.get("/AI_service", tags=["AI_service"])
def ai_service(web_id: str = '', message: str = '', group_id: str = ''):
    if web_id == '' or message == '' or group_id =='':
        return {"message": "no sentence input or no web_id", "message": ""}
    return AI_customer_service.QA(web_id, message, [group_id])

@app.get("/update_product", tags=["get_product"])
def ai_update_product(web_id: str = '', group_id: str = ''):
    AI_judge.update_recommend_status(web_id, group_id, 1)
    return 'ok'
@app.get("/judge", tags=["judge"])
def ai_service_judge(message: str = ''):
    custom_judge = AI_judge.get_judge(message)
    true_count = sum(1 for i in custom_judge.values() if i == 'True')
    if true_count > 1 or true_count == 0:
        types = 6
    elif custom_judge.get('Inquiry about product information') == 'True':
        types = 1
    elif custom_judge.get('Requesting returns or exchanges') == 'True':
        types = 2
    elif custom_judge.get('General inquiries') == 'True':
        types = 3
    elif custom_judge.get('Simple Greeting or Introduction') == 'True':
        types = 4
    elif custom_judge.get('Simple expression of gratitude') == 'True':
        types = 5
    elif custom_judge.get('Unable to determine intent or other') == 'True':
        types = 6
    return types



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
    









