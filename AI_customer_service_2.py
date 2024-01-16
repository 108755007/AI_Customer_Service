import time
from langchain.schema import OutputParserException, SystemMessage
import os
import jieba
import re
from db import DBhelper
from langchain.output_parsers import RetryWithErrorOutputParser
from func_timeout import func_timeout, FunctionTimedOut
from AI_customer_service import ChatGPT_AVD
from likr_Search_engine import Search_engine
from likr_Recommend_engine import Recommend_engine
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import random
import jieba.analyse as analyse
from langchain.chat_models import AzureChatOpenAI
import datetime
from utils.AI_customer_service_utils import translation_stw, fetch_url_response, shorten_url
import pandas as pd


def cost_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'func {func.__name__} cost time:{time.perf_counter() - t:.8f} s')
        return result, (time.perf_counter() - t)

    return fun


def split_word(document):
    stop_words = {":", "的", "，", "”"}
    text = []
    for word in jieba.cut(document):
        if word not in stop_words:
            text.append(word)
    return text


def get_history_df(web_id: str, user_id: str) -> pd.DataFrame:
    query = f"""SELECT id, web_id, group_id, counts, question, answer, keyword_list,q_a_history,add_time ,update_time 
                FROM web_push.AI_service_api WHERE group_id = '{user_id}' and web_id = '{web_id}';"""
    df = pd.DataFrame(DBhelper('jupiter_new').ExecuteSelect(query),
                      columns=['id', 'web_id', 'group_id', 'counts', 'question', 'answer', 'keyword_list',
                               'q_a_history', 'add_time', 'update_time'])
    return df


class LangchainSetting:
    def __init__(self):
        self.des_prompt = None
        self.des_output_parser = None
        self.check_prompt = ChatPromptTemplate
        self.chat_model_4 = AzureChatOpenAI(deployment_name="chat-cs-canada-4", temperature=0.2)
        self.judge_prompt = PromptTemplate
        self.judge_output = StructuredOutputParser
        self.setting_langchain()
        self.ai_des_setting()

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
        self.keyword_prompt ="""You are an AI designed to parse text and output results in JSON format. Your task is to break down user-submitted spoken content into individual words, assign a base point value to each word depending on its perceived importance in common language use, and then double the point value for any word that is identified as a product name. The response should contain no explanatory text or extraneous content, just a JSON object with each word as a key and its associated score as the value. Please use the following schema exclusively for each response:

                                {
                                  "word1": score1,
                                  "word2": score2,
                                  ...
                                  "wordN": scoreN
                                }"""
        self.judge_prompt_text = """You are a sophisticated AI designed to output structured data in JSON format. Your task is to analyze customer inquiries submitted via text and determine the corresponding intent. Based on the intent, your output should provide a JSON object with a 'type' field indicating the category of the intent .The intent categories and the expected details for each are as follows:

                                    1. 'product_inquiry': Customers may inquire about product features, specifications, prices, and availability. 
                                    
                                    2. 'return_or_exchange_request': Customers may seek assistance with returning or exchanging their purchased products, and they require information about return and exchange policies.
                                    
                                    3. 'general_inquiry': Customers may have general questions regarding the company’s services, purchasing methods,check delivery time ,or other company policies. 
                                    
                                    4. 'greeting': Customers may initiate a conversation with a simple greeting or introduction. 
                                    
                                    5. 'expression_of_gratitude': Customers express gratitude for the assistance they have received. 
                                    
                                    6. 'unknown_intent': The customer's intent is unclear, or the scenario doesn't fit any of the categories above. 
                                    
                                    For each of these intents, provide a JSON object with 'type': Returns the most likely intent. Here is an example of how the JSON output should look:
                                    {
                                      'type': 'product_inquiry'
                                    }
                                    Please analyze the customer's text input and provide an appropriate JSON response."""

        self.get_keyword_prompt="""You are an intelligent assistant designed to parse and structure information into JSON format. Upon receiving a customer's question, your task is to identify key components of the query: the product name (short form), the type of information requested, and the action the customer requires assistance with. Please rank these elements by importance, with "1" being the most important. Use the following format:

                                    ```json
                                    {
                                      "1": "Most important keyword or phrase",
                                      "2": "Second important keyword or phrase",
                                      "3": "Third important keyword or phrase",
                                      // Continue as needed
                                    }
                                    ```
                                    
                                    For example, if the customer asks, "How do I reset my QuickHeat thermostat when the screen is unresponsive?", the output should be:
                                    
                                    ```json
                                    {
                                      "1": "QuickHeat thermostat",
                                      "2": "reset",
                                      "3": "screen is unresponsive"
                                    }
                                    ```"""
        self.translation_prompt="""You are an AI assistant designed to translate non-English input text into English and return the response in JSON format. Upon receiving text, please identify the language, translate it into English, and output the translation in a structured JSON object that includes the  the English translation.
                                
                                    Example input:
                                    "Hola, ¿cómo estás?"
                                    
                                    Expected JSON output:
                                    {
                                      "translation": "Hello, how are you?"
                                    }
                                    """
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

    def ai_des_setting(self):
        response_schemas = [ResponseSchema(name="descriptions", description="根據<商品標題>產生的<商品敘述>"), ]
        self.des_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = self.des_output_parser.get_format_instructions()
        self.des_prompt = PromptTemplate(
            template="Create the best product description possible 300words 繁體中文回答\n{format_instructions}\n{title}",
            input_variables=["title"],
            partial_variables={"format_instructions": format_instructions}
        )


def adjust_ans_format(answer: str, ) -> str:
    answer.replace('"', "'")
    replace_words = {'此致', '敬禮', '<b>', '</b>', '\w*(抱歉|對不起)\w{0,3}(，|。)'}
    for w in replace_words:
        answer = re.sub(w, '', answer).strip('\n')
    if '親愛的' in answer:
        answer = '親愛的' + '親愛的'.join(answer.split("親愛的")[1:])
    if '祝您愉快' in answer:
        answer = '祝您愉快'.join(answer.split("祝您愉快！")[:-1]) + '祝您愉快！'
    return answer


def update_history_df(web_id: str, group_id: str, message: str, answer: str, keyword: str, keyword_list: list,
                      response_time: float, timestamps) -> pd.DataFrame:
    df = pd.DataFrame([[web_id, group_id, message, keyword, ','.join(keyword_list), answer, response_time, timestamps]],
                      columns=['web_id', 'group_id', 'question', 'google_keyword', 'all_keywords', 'answer',
                               'response_time', 'timestamps'])
    DBhelper.ExecuteUpdatebyChunk(df, db='jupiter_new', table=f'AI_service_cache_new', is_ssh=False)


def update_error(web_id, user_id, message, error, timestamps):
    df = pd.DataFrame([[web_id, user_id, message, error, timestamps]],
                      columns=['web_id', 'group_id', 'question', 'error', 'timestamps'])
    DBhelper.ExecuteUpdatebyChunk(df, db='jupiter_new', table=f'AI_service_cache_new', is_ssh=False)


class AICustomerAPI(ChatGPT_AVD, LangchainSetting):

    def __init__(self):
        ChatGPT_AVD.__init__(self)
        self.azure_openai_setting()
        LangchainSetting.__init__(self)
        self.CONFIG = {}
        self.avivid_user = set()
        self.get_config()
        self.avivid_user_id()
        self.Search = Search_engine()
        self.Recommend = Recommend_engine()
        self.auth = eval(os.getenv('SHORT_URL_TOKEN'))[0]
        self.token = eval(os.getenv('SHORT_URL_TOKEN'))[1]
        self.frontend = 'line_gpt_customer_service'
        self.auth = eval(os.getenv('SHORT_URL_TOKEN'))[0]
        self.token = eval(os.getenv('SHORT_URL_TOKEN'))[1]
        self.url_format = lambda x: ' ' + shorten_url(auth=self.auth, token=self.token,
                                                      name=self.frontend + '_gpt_customer_service', url=x) + ' '
        self.user_id_index = 0

    def azure_openai_setting(self):
        os.environ['OPENAI_API_KEY'] = self.AZURE_OPENAI_CONFIG.get('api_key')
        os.environ['OPENAI_API_TYPE'] = self.AZURE_OPENAI_CONFIG.get('api_type')
        os.environ['OPENAI_API_BASE'] = self.AZURE_OPENAI_CONFIG.get('api_base')
        os.environ['OPENAI_API_VERSION'] = self.AZURE_OPENAI_CONFIG.get('api_version')

    def get_judge_test(self, message):
        messages = [
            {"role": "system", "content": self.judge_prompt_text},
            {"role": "user", "content": message}
        ]
        ans = self.ask_gpt(message=messages)
        try:
            return eval(ans).get('type').lower()
        except:
            return 'unknown_intent'

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

    @cost_time
    def get_keyword(self, message: str, web_id: str,lang: str) -> list:
        forbidden_words = {'client_msg_id', '我', '你', '妳', '們', '沒', '怎', '麼', '要', '沒有', '嗎', '^在$',
                           '^做$', '^如何$', '^有$', '^可以$', '^商品$', '^哪', '哪$',
                           '暢銷', '熱賣', '熱銷', '特別', '最近', '幾天', '常常', '爆款', '推薦', '吃', '賣', '嘛',
                           '想', '請問', '多少'}
        # remove web_id from message
        message = translation_stw(message).lower()
        for i in [web_id, self.CONFIG[web_id]['web_name']] + eval(self.CONFIG[web_id]['other_name']):
            message = re.sub(i, '', message)
            for j in list(jieba.cut(i)):
                message = re.sub(j, '', message)
        if message.strip() == '':
            return 'no message'
        # segmentation
        # print(message)
        k = 0
        if web_id in {'AviviD', 'avividai'} and lang not in ['Chinese', '中文', '繁體中文', 'chinese', '國語']:
            print("###關鍵字需要變成英文###")
            while True:
                if k > 10:
                    break
                try:
                    reply = self.ask_gpt(message=[{'role': 'system', 'content': self.translation_prompt},
                                                  {'role': 'user', 'content': f'{message}'}], json_format=True)
                    if reply == 'timeout':
                        k += 1
                        continue
                    message = eval(reply).get('translation')
                    print(f"###f'翻譯後的文本{message}'###")
                    break
                except:
                    k += 1
        k = 0
        while True:
            if k > 10:
                print('關鍵字獲取錯誤')
                keyword_list = analyse.extract_tags(message, topK=2)
                break
            try:
                reply = self.ask_gpt(message=[{'role': 'system', 'content': self.get_keyword_prompt},
                                              {'role': 'user', 'content': f'{message}'}], json_format=True)
                if reply == 'timeout':
                    k += 1
                    continue
                keyword_list = [k for _, k in eval(reply).items() if k in message and not any(re.search(w, k) for w in forbidden_words)]
                print(f"gpt成功獲取關鍵字")
                break
            except:
                k += 1

        return keyword_list

    def avivid_user_id(self):
        query = "SELECT DISTINCT group_id  FROM web_push.AI_service_cache_new"
        data = DBhelper('jupiter_new').ExecuteSelect(query)
        self.avivid_user = set(i[0] for i in data)

    def split_product_url(self, result: list[dict], web_id: str):
        product_domain = [i.strip('*') for i in self.CONFIG[web_id]['product_url'].split(',')]
        n_product_url, product_url = [], []
        for r in result:
            if r.get('link') and any([url in r.get('link') for url in product_domain]):
                product_url.append(r)
            else:
                n_product_url.append(r)
        return n_product_url, product_url

    def get_des(self, title):
        prompt = f"""Generate 1 unique meta descriptions, of a minimum of 500 characters, for the following text. Respond in 繁體中文. They should be catchy with a call to action, including in them: [{title}]"""
        ans = self.ask_gpt(message=prompt, model='gpt-4')
        # _input = self.des_prompt.format_prompt(title=title)
        # output = self.chat_model_4(_input.to_messages())
        # # noinspection PyBroadException
        # try:
        #     gpt_res = self.des_output_parser(output.content)
        # except:
        #     print('產生失敗！！,重新產生')
        #     retry_parser = RetryWithErrorOutputParser.from_llm(parser=self.des_output_parser, llm=self.chat_model_4)
        #     gpt_res = retry_parser.parse_with_prompt(output.content, _input)
        return translation_stw(ans)

    def update_recommend_status(self, web_id: str, group_id: str, status: int, product={}, lang='中文', main_web_id=''):
        main_web_id = web_id if not main_web_id else main_web_id
        if main_web_id not in {'AviviD', 'avividai'}:
            if status == 1 and not product:
                hot_product = self.Recommend.fetch_hot_rank(web_id=main_web_id)
                output = {k: v for k, v in hot_product.items() if v in range(1, 11)}
                product_result = self.Recommend.fetch_data(product_ids=output, web_id=main_web_id)
                product = random.choice(product_result)
            recommend = f"""謝謝您對我們的關注！如果您想了解更多我們最熱銷的產品，歡迎逛逛我們為您精選的其他商品：
                - 【{product.get('title')} [ {self.url_format(product.get('link'))} ]"""
        else:
            recommend = f"""謝謝您對我們的關注！如果您想了解更多我們禾多的產品服務，歡迎逛逛我們產品：
                [ {self.url_format("https://avivid.ai/product/acquisition")} ]"""
            recommend = self.translate(lang, recommend)
        DBhelper.ExecuteUpdatebyChunk(
            pd.DataFrame(
                [[web_id, group_id, status, recommend, int(datetime.datetime.timestamp(datetime.datetime.now()))]],
                columns=['web_id', 'group_id', 'status', 'recommend', 'timestamp']), db='jupiter_new',
            table=f'AI_service_recommend_status', is_ssh=False)

    def adjust_ans_url_format(self, answer: str, links: list, config: dict) -> str:
        url_set = sorted(list(set(re.findall(r'https?:\/\/[\w\.\-\?/=+&#$%^;%_]+', answer))), key=len, reverse=True)
        unused_links, product_domain = [], [i.strip('*') for i in config['product_url'].split(',')]
        for url in url_set:
            reurl = url
            for char in '?':
                reurl = reurl.replace(char, '\\' + char)
            answer = re.sub(reurl + '(?![\w\.\-\?/=+&#$%^;%_\|])', self.url_format(url), answer)
        for i, info in enumerate(links):
            if re.search(f'\[#?{i + 1}\](?!.*\[#?{i + 1}\])', answer):
                answer = re.sub(f'\[#?{i + 1}\](?!.*\[#?{i + 1}\])', f'[{self.url_format(info[1])}]', answer, count=1)
            elif any([url in info[1] for url in product_domain]):
                unused_links.append(info)
        answer = re.sub(f'\[#?\d\]', f'', answer)
        return answer, unused_links

    def answer_append(self, answer: str, unused_links: list) -> str:
        answer_set = set(split_word(answer))
        add = False
        for idx, url, title in unused_links:
            title_set = set(split_word(title))
            similar12 = len(answer_set & title_set) / len(title_set)
            if similar12 >= 0.6:
                answer += f'\n[{self.url_format(url)}]'
                add = True
        if add:
            print('已補上連結！')
        else:
            print('無漏掉連結！')
        return answer

    def check_lang(self, message):
        lang = self.ask_gpt([{'role': 'system',
                              'content': """請分析輸入的語言並確定其所屬的語言系統或種類。你只能提供語言的種類,切記一定要回答種類,禁止回答未知,回答格式,輸入語言:種類"""},
                             {'role': 'user', 'content': f"<\'input\':'{message}'>"}], model='gpt-4')
        return lang.split(':')[-1]

    def translate(self, lang, out):
        if lang not in ['Chinese', '中文', '繁體中文', 'chinese', '國語']:
            out = self.ask_gpt([{'role': 'system',
                                 'content': f"""Please translate this given input into {lang}. All I need returned is the translated text,Answer format <\'output\':translated text """},
                                {'role': 'user', 'content': f"<\'input\':'{out}'>"}], model='gpt-3.5-turbo-16k"', timeout=120)
        return out.split("<output:")[-1].split("<\'output\':'")[-1].split("<\'output\':")[-1].split("<'output':'")[-1].split("<'output':")[-1].replace('}', '').replace("'>", "").replace("'", "")

    def qa(self, web_id: str, message: str, user_id: str, find_dpa=True, lang='中文', main_web_id=''):
        main_web_id = web_id if not main_web_id else main_web_id
        hash_ = str(abs(hash(str(user_id) + message)))[:6]
        hash_ = user_id
        now_timestamps = int(datetime.datetime.timestamp(datetime.datetime.now()))
        print(f"{hash_},輸入訊息：{message}")
        keyword_list, keyword_time = self.get_keyword(message, main_web_id,lang)
        if isinstance(keyword_list, str):
            print(f'{hash_},獲取關鍵字錯誤')
            update_error(web_id, user_id, message, 'keyword', now_timestamps)
            return '抱歉,目前客服忙碌中,請稍後再問一次！'
        print(f"{hash_},獲取的關鍵字{str(keyword_list)}")
        search_start = time.time()
        try:
            result, keyword = func_timeout(10, self.Search.likr_search,
                                           (keyword_list, self.CONFIG[main_web_id], 3, False, find_dpa))
        except FunctionTimedOut:
            print(f'{hash_},likr搜尋超過時間!')
            update_error(web_id, user_id, message, 'likr_timeout', now_timestamps)
            return '抱歉,目前客服忙碌中,請稍後再問一次！'
        except:
            print(f'{hash_},likr報錯！')
            update_error(web_id, user_id, message, 'likr_error', now_timestamps)
            return '抱歉,目前客服忙碌中,請稍後再問一次！'
        search_time = time.time() - search_start
        print(f"{hash_},likr搜尋時間:{search_time}")
        print(f"{hash_},google搜尋結果,類別：{result[1]},使用的關鍵字為:{keyword}")

        query_start = time.time()
        if main_web_id in {'AviviD', 'avividai'}:
            search_result = result[0][:3]
            if search_result:
                gpt_query, links = self.get_gpt_query([search_result, []], message, [], self.CONFIG[main_web_id],
                                                      continuity=False)
                self.update_recommend_status(web_id, user_id, 1, '', lang, main_web_id=main_web_id)
            else:
                gpt_query, links = self.get_gpt_query([message, ''], message, [], self.CONFIG[main_web_id],
                                                      continuity=False)
                self.update_recommend_status(web_id, user_id, 1, '', lang, main_web_id=main_web_id)

        else:
            product_result, search_result, common, flags = self.Recommend.likr_recommend(search_result=result[0],
                                                                                         keywords=keyword_list,
                                                                                         flags={},
                                                                                         config=self.CONFIG[main_web_id])
            if common:
                print(f"{hash_},有推薦類品")
                
            if len(result) == 0 and self.CONFIG[main_web_id]['mode'] == 2:
                gpt_query = [{'role': 'user', 'content': message}]
                gpt_answer = answer = f"親愛的顧客您好，目前無法回覆此問題，稍後將由專人為您服務。"

            # Step 3: response from ChatGPT
            else:
                if find_dpa:
                    if common:
                        gpt_query, links = self.get_gpt_query([common[:1], []], message, [], self.CONFIG[main_web_id],
                                                              continuity=False)
                        if len(common) > 1:
                            self.update_recommend_status(web_id, user_id, 1, common[-1], main_web_id=main_web_id)
                        else:
                            self.update_recommend_status(web_id, user_id, 1, product_result[0], main_web_id=main_web_id)
                    elif search_result:
                        gpt_query, links = self.get_gpt_query([search_result[:1], []], message, [], self.CONFIG[main_web_id],
                                                              continuity=False)
                        self.update_recommend_status(web_id, user_id, 1, product_result[0], main_web_id=main_web_id)
                    elif product_result:
                        gpt_query, links = self.get_gpt_query([product_result[:1], []], message, [],
                                                              self.CONFIG[main_web_id],
                                                              continuity=False)
                        self.update_recommend_status(web_id, user_id, 1, product_result[-1], main_web_id=main_web_id)
                    else:
                        print(f'{hash_},找不到商品')
                        gpt_query, links = self.get_gpt_query([message, ''], message, [], self.CONFIG[main_web_id],
                                                              continuity=False)
                        self.update_recommend_status(web_id, user_id, 1, product_result[0], main_web_id=main_web_id)

                else:
                    if search_result:
                        if common:
                            gpt_query, links = self.get_gpt_query([common[:2], []], message, [], self.CONFIG[main_web_id],
                                                                  continuity=False)
                        else:
                            gpt_query, links = self.get_gpt_query([search_result, []], message, [], self.CONFIG[main_web_id],
                                                                  continuity=False)

                    else:
                        gpt_query, links = self.get_gpt_query([message, ''], message, [], self.CONFIG[main_web_id],
                                                              continuity=False)
                    self.update_recommend_status(web_id, user_id, 1, product_result[0], main_web_id=main_web_id)

        query_time = time.time() - query_start
        gpt_start = time.time()
        print(f"""{hash_}:gpt輸入：{gpt_query}""")
        gpt_answer = translation_stw(
            self.ask_gpt(gpt_query, model='gpt-3.5-turbo', timeout=60)).replace('，\n', '，')

        gpt_time = time.time() - gpt_start
        print(f"""{hash_}:gpt回答：{gpt_answer}""")

        answer = adjust_ans_format(gpt_answer)
        answer, unused_links = self.adjust_ans_url_format(answer, links, self.CONFIG[main_web_id])
        if main_web_id in {'AviviD', 'avividai'}:
            if user_id not in self.avivid_user:
                self.avivid_user.add(user_id)
                answer += """如果您有任何疑問，麻煩留下聯絡訊息，我們很樂意為您提供幫助。\n\n聯絡人：\n電話：\n方便聯絡的時間：\n\n至於收費方式由於選擇方案的不同會有所差異，還請您務必填寫表單以留下資訊，我們將由專人進一步與您聯絡！表單連結：https://forms.gle/S4zkJynXj5wGq6Ja9"""
            print(f'{hash_}:輸入語言種類：{lang}')
            answer = self.translate(lang, answer).split("'translation':")[-1]
        else:
            answer = self.answer_append(answer, unused_links)
        print(f"""{hash_}:整理後回答：{answer}""")
        update_history_df(web_id, user_id, message, answer, keyword, keyword_list, keyword_time+search_time+query_time+gpt_time, now_timestamps)
        print(f"{hash_}花費時間k={keyword_time}, s={search_time}, q={query_time}, g={gpt_time}")
        return answer

        # # history/continue
        # # to be add......print(flags)
        # #

        # return q


if __name__ == "__main__":
    g = AICustomerAPI()
