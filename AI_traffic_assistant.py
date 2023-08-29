from db import DBhelper
import pandas as pd
import datetime
from tqdm import tqdm
import requests
from utils.log import logger
from AI_customer_service import QA_api
from AI_Search import AI_Search
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI,AzureChatOpenAI
import os


class Util(QA_api):
    def __init__(self):
        super().__init__('line',logger())
        self.base_web_id = 'kingstone'
        self.base_web_id_type = 1
        self.web_id_dict = self.get_web_id()
        self.dateint = self.get_data_intdate(7)
        self.Azure_openai_setting()
        self.check_model_setting()
    def get_data_intdate(self,time_deley):
        return int(str(datetime.date.today()-datetime.timedelta(time_deley)).replace('-',''))
    def get_web_id(self):
        qurey = f"""SELECT web_id,web_id_type FROM missoner_web_id_table WHERE ai_article_enable = 1 """
        web_id_list = DBhelper('dione').ExecuteSelect(qurey)
        return {i:v for i,v in web_id_list}
    def get_media_keyword_data(self):
        query = f"""
        SELECT k.keyword,k.web_id,k.url,k.image,t.title ,t.content,t.pageviews 
        FROM (SELECT web_id,article_id ,url, keyword,image 
                FROM missoner_keyword_article_new 
                WHERE dateint > '{self.dateint}' and web_id in ("{'","'.join([i for i, v in self.web_id_dict.items() if v == 0])}")) k 
        INNER JOIN (SELECT article_id, pageviews,title,content 
                FROM dione.missoner_article 
                WHERE `date` > {self.dateint} ) t 
        ON k.article_id = t.article_id ORDER BY t.pageviews desc;
        """
        data = DBhelper('dione').ExecuteSelect(query)
        return pd.DataFrame(data).drop_duplicates(['keyword', 'title'])

    def get_keyword_data(self,web_id):
        query = f"""
        SELECT k.keyword,k.web_id,k.url,k.image,t.title ,t.content,t.pageviews 
        FROM (SELECT web_id, url, article_id, keyword,image 
                FROM missoner_keyword_article_new 
                WHERE dateint > {self.dateint} and web_id = '{web_id}') k 
        INNER JOIN (SELECT article_id, pageviews,title,content 
                FROM dione.missoner_article
                WHERE
        """
        if self.web_id_dict.get(web_id) != 1:
            query += f""" `date` > {self.dateint} and"""
        query += f""" web_id = '{web_id}') t 
                    ON k.article_id = t.article_id ORDER BY t.pageviews desc;"""
        data = DBhelper('dione').ExecuteSelect(query)
        return pd.DataFrame(data).drop_duplicates(['keyword', 'title'])

    def check_model_setting(self):
        self.chat_check_model = AzureChatOpenAI(deployment_name="chat-cs-jp-4",temperature=0)
        response_schemas = [ResponseSchema(name="check", description="判斷是否符合大眾觀賞"), ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        self.check_prompt = ChatPromptTemplate(
            messages=[SystemMessage(
                content=("你會判斷內容是否符合大眾觀賞,返回True或者False")),
                HumanMessagePromptTemplate.from_template(
                    "answer the users content as best as possible.\n{format_instructions}\n{question}")
            ],
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions}
        )
    def Azure_openai_setting(self):
        os.environ['OPENAI_API_KEY'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_key')
        os.environ['OPENAI_API_TYPE'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_type')
        os.environ['OPENAI_API_BASE'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_base')
        os.environ['OPENAI_API_VERSION'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_version')
class AI_traffic(Util):
    def __init__(self):
        super().__init__()
        self.media_keyword_pd = self.get_media_keyword_data()
        self.keyword_all_set = set(self.media_keyword_pd.keyword)
        self.get_keyword_pd()

    def get_keyword_info(self,web_id,keyword):
        if web_id not in self.web_id_dict:
            web_id = self.base_web_id
        keyword_list = keyword.split(',')
        df = self.all_keyword_pd[web_id]
        keyword_info_dict = {}
        for key in keyword_list:
            if key in set(df.keyword):
                curr_keyword_info = df[df.keyword == key]
                title_list, content_list, article_list, img_list = curr_keyword_info[['title', 'content', 'url', 'image']].values
                for title, content, article_id, img in zip(title_list, content_list, article_list, img_list):
                    if not self.check_news(content):
                        keyword_info_dict[key] = (f"標題:{title},敘述:{content}\n", web_id, article_id, img)
                        break
            elif key in self.keyword_all_set:
                curr_keyword_info = self.media_keyword_pd[self.media_keyword_pd.keyword == key]
                title_list, content_list, web_id_list, article_list, img_list = curr_keyword_info[['title', 'content', 'web_id', 'url', 'image']].values
                for title, content, article_id, web_id_article, img in zip(title_list, content_list, article_list, web_id_list, img_list):
                    if not self.check_news(content):
                        keyword_info_dict[key] = (f"標題:{title},敘述:{content}\n", web_id_article, article_id, img)
                        break
            else:
                html = f'https://www.googleapis.com/customsearch/v1/siterestrict?cx=41d4033f0c2f04bb8&key={self.Search.GOOGLE_SEARCH_KEY}&q={key}'
                r = requests.get(html)
                if r.status_code != 200:
                    continue
                res = r.json().get('items')
                if not res:
                    continue
                for data in res:
                    title = data.get('htmlTitle')
                    sn = data.get('snippet')
                    link = data.get('link')
                    if self.check_news(title + sn):
                        keyword_info_dict[key] = (f"標題:{title}/{sn}\n", 'google', link, '_')
                        break
        return keyword_info_dict

    def get_keyword_pd(self):
        self.all_keyword_pd ={}
        pbar = tqdm(list(self.web_id_dict.keys()))
        for i, web_id in enumerate(pbar):
            if i == 5:
                break
            pbar.set_description(web_id)
            self.all_keyword_pd[web_id] = self.get_keyword_data(web_id)
    def check_news(self,text):
        _input = self.check_prompt.format_prompt(question=text)
        output = self.chat_check_model(_input.to_messages())
        if 'False' in output.content:
            return False
        return True


if __name__ == "__main__":
    AI_traffic = AI_traffic()

