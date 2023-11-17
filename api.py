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
    }
]

###讀取套件
app = FastAPI(title="hodo_ai", description=description, openapi_tags=tags_metadata)

AI_customer_service = QA_api('line', logger())
#_AI_Search = AI_Search()
AI_judge = AICustomerAPI()

@app.get("/AI_service", tags=["AI_service"])
def AI_service(web_id:str='',message:str='',group_id:str=''):
    if web_id == '' or message == '' or group_id =='':
        return {"message": "no sentence input or no web_id", "message": ""}
    return AI_customer_service.QA(web_id, message, [group_id])

@app.get("/judge", tags=["judge"])
def AI_service_judge(message: str=''):
    return AI_judge.get_judge(message)



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
    









