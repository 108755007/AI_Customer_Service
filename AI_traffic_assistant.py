from db import DBhelper
import pandas as pd
import datetime
from tqdm import tqdm
import requests
from utils.log import logger
from AI_customer_service import QA_api
from AI_Search import AI_Search
from opencc import OpenCC
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from lanchain_class import title_1,title_5
from langchain.chat_models import ChatOpenAI,AzureChatOpenAI
import json
import os


class Util(QA_api):
    def __init__(self):
        super().__init__('line',logger())
        self.base_web_id = 'kingstone'
        self.base_web_id_type = 1
        self.web_id_dict = self.get_web_id()
        self.dateint = self.get_data_intdate(7)
        self.Azure_openai_setting()
        self.langchain_model_setting()
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

    def langchain_model_setting(self):
        self.check_model_setting()
        self.title_model_setting()
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
    def title_model_setting(self):
        output_parser_1 = PydanticOutputParser(pydantic_object=title_1)
        output_parser_5 = PydanticOutputParser(pydantic_object=title_5)
        format_instructions_1 = output_parser_1.get_format_instructions()
        format_instructions_5 = output_parser_5.get_format_instructions()
        template_1 = """
        你會根據關鍵字和關鍵字來源產生1個標題

        {keyword_info}

        標題一定要包含關鍵字,標題禁止包含任何新聞網,繁體中文回答

        -----
        {format_instructions}
        """
        template_5 = """
        你會根據關鍵字和關鍵字來源產生5個標題

        {keyword_info}

        標題一定要包含關鍵字,標題禁止包含任何新聞網,繁體中文回答

        -----
        {format_instructions}
        """

        prompt_1 = PromptTemplate(  # 设置prompt模板，用于格式化输入
            template=template_1,
            input_variables=["keyword_info"],
            partial_variables={"format_instructions": format_instructions_1}
        )
        prompt_5 = PromptTemplate(  # 设置prompt模板，用于格式化输入
            template=template_5,
            input_variables=["keyword_info"],
            partial_variables={"format_instructions": format_instructions_5}
        )
        self.title_chain_1 = LLMChain(prompt=prompt_1,llm=AzureChatOpenAI(deployment_name="chat-cs-jp-4",temperature=0),output_parser=output_parser_1)
        self.title_chain_5 = LLMChain(prompt=prompt_5, llm=AzureChatOpenAI(deployment_name="chat-cs-jp-4", temperature=0),output_parser=output_parser_5)
    def Azure_openai_setting(self):
        os.environ['OPENAI_API_KEY'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_key')
        os.environ['OPENAI_API_TYPE'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_type')
        os.environ['OPENAI_API_BASE'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_base')
        os.environ['OPENAI_API_VERSION'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_version')

    def translation_stw(self,text):
        cc = OpenCC('likr-s2twp')
        return cc.convert(text)


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
                for title, content, article_id, img in curr_keyword_info[['title', 'content', 'url', 'image']].values:
                    if self.check_news(title):
                        keyword_info_dict[key] = (title,content, web_id, article_id, img)
                        break
            elif key in self.keyword_all_set:
                curr_keyword_info = self.media_keyword_pd[self.media_keyword_pd.keyword == key]
                for title, content, web_id_article,article_id , img in curr_keyword_info[['title', 'content', 'web_id', 'url', 'image']].values:
                    if self.check_news(title):
                        keyword_info_dict[key] = (title,content, web_id_article, article_id, img)
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
                        keyword_info_dict[key] = (title, sn, 'google', link, '_')
                        break
        return keyword_info_dict

    def get_keyword_pd(self):
        self.all_keyword_pd ={}
        pbar = tqdm(list(self.web_id_dict.keys()))
        for i, web_id in enumerate(pbar):
            if i == 2:
                break
            pbar.set_description(web_id)
            self.all_keyword_pd[web_id] = self.get_keyword_data(web_id)
    def check_news(self,text):
        _input = self.check_prompt.format_prompt(question=text)
        output = self.chat_check_model(_input.to_messages())
        if 'False' in output.content:
            return False
        return True

    def get_title(self,web_id,keywords,user_id,web_id_main='',article='',types=1):
        keyword_info_dict = self.get_keyword_info(web_id_main, keywords) if web_id_main else self.get_keyword_info(web_id, keywords)
        prompt = ''.join([f"關鍵字:{i}\n'{i}'關鍵字的來源:{v[0]}\n\n" for i, v in keyword_info_dict.items()])
        if types == 1:
            result = self.title_chain_1.run({'keyword_info': prompt})
            DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([[user_id, web_id, types, keywords, self.translation_stw(result.title), json.dumps([{'keyword': keyword, 'title': data[0], 'web_id': data[2], 'url': data[3], 'image': data[4]} for keyword, data in keyword_info_dict.items()])]],columns=['user_id', 'web_id', 'type', 'inputs', 'title', 'keyword_dict']), db='sunscribe',table='ai_article', chunk_size=100000, is_ssh=False)
            return [result.title]
        else:
            if not article:
                return {"message": "no article input"}
            result = self.title_chain_5.run({'keyword_info': prompt})
            DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([[user_id, web_id, types, keywords, self.translation_stw(result.title[0]),self.translation_stw(result.title[1]), self.translation_stw(result.title[2]),self.translation_stw(result.title[3]), self.translation_stw(result.title[4]),article, 2]],
                                                       columns=['user_id', 'web_id', 'type', 'inputs', 'subheading_1','subheading_2', 'subheading_3', 'subheading_4','subheading_5', 'article_1', 'article_step']),db='sunscribe', table='ai_article', chunk_size=100000, is_ssh=False)
            return result.title

if __name__ == "__main__":
    AI_traffic = AI_traffic()

