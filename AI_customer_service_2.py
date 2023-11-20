import time

from langchain.schema import OutputParserException
import os
import jieba
import re
from db import DBhelper
from func_timeout import func_timeout
from AI_customer_service import ChatGPT_AVD
from likr_Search_engine import Search_engine
from likr_Recommend_engine import Recommend_engine
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
import datetime
from utils.AI_customer_service_utils import translation_stw, fetch_url_response, shorten_url
import pandas as pd


def Azure_openai_setting(self):
    os.environ['OPENAI_API_KEY'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_key')
    os.environ['OPENAI_API_TYPE'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_type')
    os.environ['OPENAI_API_BASE'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_base')
    os.environ['OPENAI_API_VERSION'] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_version')


def update_recommend_status(web_id: str, group_id: str, status: int, recommend=''):
    if recommend == '' and status == 1:
        status = 2
    DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([[web_id, group_id, status, recommend,
                                                 int(datetime.timestamp(datetime.now()))]],
                                               columns=['web_id', 'group_id', 'status', 'recommend', 'timestamp']),
                                  db='jupiter_new', table=f'AI_service_recommend_status', is_ssh=False)
    return


def get_history_df(web_id: str, user_id: str) -> pd.DataFrame:
    query = f"""SELECT id, web_id, group_id, counts, question, answer, keyword_list,q_a_history,add_time ,update_time 
                FROM web_push.AI_service_api WHERE group_id = '{user_id}' and web_id = '{web_id}';"""
    df = pd.DataFrame(DBhelper('jupiter_new').ExecuteSelect(query),
                      columns=['id', 'web_id', 'group_id', 'counts', 'question', 'answer', 'keyword_list',
                               'q_a_history', 'add_time', 'update_time'])
    return df


class LangchainQA:
    def __init__(self):
        self.chat_model_4 = AzureChatOpenAI(deployment_name="chat-cs-jp-4", temperature=0)
        self.judge_prompt = PromptTemplate
        self.judge_output = StructuredOutputParser
        self.setting_langchain()

    def setting_langchain(self):
        self.judge_setting()

    def get_judge(self, message):
        start = time.time()
        _input = self.judge_prompt.format_prompt(question=message)
        output = self.chat_model_4(_input.to_messages())
        try:
            out = self.judge_output.parse(output.content)
        except OutputParserException:
            out = {'Inquiry about product information': 'False', 'Requesting returns or exchanges': 'False',
                   'Complaints or issue feedback': 'False', 'General inquiries': 'False',
                   'Simple Greeting or Introduction': 'False', 'Simple expression of gratitude': 'False',
                   'Unable to determine intent or other': 'True'}
        return out

    def judge_setting(self):
        response_schemas = [ResponseSchema(name="Inquiry about product information",
                                           description="If Customers may want to understand product features, specifications, prices, and other details return 'True' else 'False'"),
                            ResponseSchema(name="Requesting returns or exchanges",
                                           description="If Customers may need to return or exchange products they have purchased and want to understand the return and exchange policies and procedures.return 'True' else 'False'"),
                            ResponseSchema(name="General inquiries",
                                           description="If Customers may have general questions about the company, services, policies, or other related topics return 'True' else 'False'"),
                            ResponseSchema(name="Simple Greeting or Introduction",
                                           description="If customers initiate a simple greeting or introduction, return 'True' else 'False'"),
                            ResponseSchema(name="Simple expression of gratitude",
                                           description="If customers express simple gratitude for help, return 'True' else 'False'"),
                            ResponseSchema(name="Unable to determine intent or other",
                                           description="If unable to determine user intent or other situations, return 'True', else 'False'")]
        self.judge_output = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = self.judge_output.get_format_instructions()
        self.judge_prompt = PromptTemplate(
            template=" $You are a customer intent analysis chatbot. Please analyze the customer's intent.\n{format_instructions}\n{question}",
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions}
        )


class AICustomerAPI(ChatGPT_AVD, LangchainQA):

    def __init__(self):
        ChatGPT_AVD.__init__(self)
        self.azure_openai_setting()
        LangchainQA.__init__(self)
        self.CONFIG = {}
        self.get_config()
        self.Search = Search_engine()
        self.Recommend = Recommend_engine()
        self.auth = eval(os.getenv('SHORT_URL_TOKEN'))[0]
        self.token = eval(os.getenv('SHORT_URL_TOKEN'))[1]
        self.table_suffix = '_api'
        self.url_format = lambda x: ' ' + shorten_url(auth=self.auth, token=self.token,
                                                      name=self.frontend + '_gpt_customer_service', url=x) + ' '
        self.user_id_index = 0
        self.chat_model = AzureChatOpenAI

    def azure_openai_setting(self):
        os.environ['OPENAI_API_KEY'] = self.AZURE_OPENAI_CONFIG.get('api_key')
        os.environ['OPENAI_API_TYPE'] = self.AZURE_OPENAI_CONFIG.get('api_type')
        os.environ['OPENAI_API_BASE'] = self.AZURE_OPENAI_CONFIG.get('api_base')
        os.environ['OPENAI_API_VERSION'] = self.AZURE_OPENAI_CONFIG.get('api_version')

    def get_config(self):
        """
        Returns {web_id: config from jupiter_new -> web_push.AI_service_config}
        """
        config = DBhelper('jupiter_new').ExecuteSelect("SELECT * FROM web_push.AI_service_config where mode != 0;")
        config_col = [i[0] for i in
                      DBhelper('jupiter_new').ExecuteSelect("SHOW COLUMNS FROM web_push.AI_service_config;")]
        for conf in config:
            self.CONFIG[conf[1]] = {}
            for k, v in zip(config_col, conf):
                self.CONFIG[conf[1]][k] = v
        return

    def get_keyword(self, message: str, web_id: str) -> list:
        forbidden_words = {'client_msg_id', '我', '你', '妳', '們', '沒', '怎', '麼', '要', '沒有', '嗎', '^在$', '^做$',
                           '^如何$', '^有$', '^可以$', '^商品$', '^哪', '哪$',
                           '暢銷', '熱賣', '熱銷', '特別', '最近', '幾天', '常常', '爆款', '推薦'}
        # remove web_id from message
        message = translation_stw(message).lower()
        for i in [web_id, self.CONFIG[web_id]['web_name']] + eval(self.CONFIG[web_id]['other_name']):
            message = re.sub(i, '', message)
            for j in list(jieba.cut(i)):
                message = re.sub(j, '', message)
        if message.strip() == '':
            return 'no message'
        # segmentation
        #print(message)
        reply = self.ask_gpt([{'role': 'system', 'content': """I want you to act as a content analyzer for Chinese speaking users. You will segment the user's content into individual words, then assign a point value based on the importance of each word. If product names appear within the content, their scores should be doubled. Your responses should strictly follow this format: {"Word": Score}, and there should be no explanations within the responses"""},
                                      {'role': 'user', 'content': f'{message}'}],
                                     model='gpt-4')
        if reply == 'timeout':
            return ''
        ####TODO(yu):perhaps have problem
        keyword_list = [k for k, _ in sorted(eval(reply).items(), key=lambda x: x[1], reverse=True) if k in message and not any(re.search(w, k) for w in forbidden_words)]

        return keyword_list

    def qa(self, web_id: str, message: str, user_id: str, find_dpa=True):
        hash_ = str(abs(hash(str(user_id) + message)))[:6]
        #
        #update_recommend_status(web_id, user_id, 0)

        keyword_list = self.get_keyword(message, web_id)
        if keyword_list == 'timeout':
            return self.error('keyword_timeout', hash=hash_)

        result, keyword = func_timeout(10, self.Search.likr_search, (keyword_list, self.CONFIG[web_id], 3, False))
        print(result, keyword)
        return result, keyword

        # # history/continue
        # # to be add......
        # #

        # return q



if __name__ == "__main__":
    g = AICustomerAPI()
