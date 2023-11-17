import time

from langchain.schema import OutputParserException
import os
from db import DBhelper
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
            print('重新判斷語意')
            retry_parser = RetryWithErrorOutputParser.from_llm(parser=self.judge_output, llm=self.chat_model_4)
            out = retry_parser.parse_with_prompt(output.content, _input)
        print(time.time()-start)
        return out

    def judge_setting(self):
        response_schemas = [ResponseSchema(name="Inquiry about product information",
                                           description="If Customers may want to understand product features, specifications, prices, and other details return 'True' else 'False'"),
                            ResponseSchema(name="Requesting returns or exchanges",
                                           description="If Customers may need to return or exchange products they have purchased and want to understand the return and exchange policies and procedures.return 'True' else 'False'"),
                            ResponseSchema(name="Complaints or issue feedback",
                                           description="If Customers may encounter product quality issues, delivery delays, or dissatisfaction with services and want to file complaints or provide feedback.return 'True' else 'False'"),
                            ResponseSchema(name="General inquiries",
                                           description="If Customers may have general questions about the company, services, policies, or other related topics return 'True' else 'False'"),
                            ResponseSchema(name="Simple Greeting or Introduction",
                                           description="If customers initiate a simple greeting or introduction, return 'True' else 'False'"),
                            ResponseSchema(name="Simple expression of gratitude",
                                           description="If customers express simple gratitude for help, return 'True' else 'False'")]
        self.judge_output = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = self.judge_output.get_format_instructions()
        self.judge_prompt = PromptTemplate(
            template=" $You are a customer intent analysis chatbot. Please analyze the customer's intent.\n{format_instructions}\n{question}",
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions}
        )


class AICustomerAPI(ChatGPT_AVD, LangchainQA):

    def __init__(self):
        print(555)
        ChatGPT_AVD.__init__(self)
        self.azure_openai_setting()
        LangchainQA.__init__(self)
        self.CONFIG = {}
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
        print('666')
        os.environ['OPENAI_API_KEY'] = self.AZURE_OPENAI_CONFIG.get('api_key')
        print('777')
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

    def qa(self, web_id: str, message: str, user_id: str):
        # hash_ = str(abs(hash(str(user_id) + message)))[:6]
        #
        # #update_recommend_status(web_id, user_id, 0)
        #
        # # history/continue
        # # to be add......
        # #
        # q = self.get_judge(message)
        # return q
        pass



if __name__ == "__main__":
    g = AICustomerAPI()
