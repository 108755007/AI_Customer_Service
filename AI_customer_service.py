from dotenv import load_dotenv
load_dotenv()
import os, traceback, re, json, time
import pandas as pd
from datetime import datetime
from func_timeout import func_timeout
import jieba
import openai
import tiktoken
from db import DBhelper
from utils.log import logger
from utils.AI_customer_service_utils import translation_stw, fetch_url_response
from likr_Search_engine import Search_engine
from likr_Recommend_engine import Recommend_engine
from lbs.distance_calc import StoreDistanceEvaluator


class ChatGPT_AVD:
    def __init__(self):
        self.OPEN_AI_KEY_DICT = eval(os.getenv('OPENAI_API_KEY'))
        self.AZURE_OPENAI_CONFIG = eval(os.getenv('AZURE_OPENAI_CONFIG'))

    def get_keys(func):
        def inner(self, message, model="gpt-3.5-turbo", timeout=60):
            config = self.AZURE_OPENAI_CONFIG
            # get token_id
            if 'gpt-4' in model:
                query = 'SELECT id, counts FROM web_push.AI_service_token_counter x ORDER BY counts limit 1;'
                query = 'x WHERE id < 6'.join(query.split('x'))
                token_id = DBhelper('jupiter_new').ExecuteSelect(query)[0][0]
                config = {'api_key': self.OPEN_AI_KEY_DICT[token_id],
                          'api_type': 'open_ai',
                          'api_base': 'https://api.openai.com/v1',
                          'api_version': None,
                          'kwargs': {'model': model}}
                # update token counter
                DBhelper('jupiter_new').ExecuteDelete(f'UPDATE web_push.AI_service_token_counter SET counts = counts + 1 WHERE id = {token_id}')

            try:
                res = func_timeout(timeout, func, (self, message, config))
            except:
                res = 'timeout'

            if 'gpt-4' in model:
                DBhelper('jupiter_new').ExecuteDelete(f'UPDATE web_push.AI_service_token_counter SET counts = counts - 1 WHERE id = {token_id}')
            return res
        return inner

    @get_keys
    def ask_gpt(self, message: str, config: dict) -> str:
        openai.api_key = config.get('api_key')
        openai.api_type = config.get('api_type')
        openai.api_base = config.get('api_base')
        openai.api_version = config.get('api_version')
        kwargs = config.get('kwargs')
        kwargs['messages'] = [{'role': 'user', 'content': message}] if type(message) == str else message
        completion = openai.ChatCompletion.create(**kwargs)
        return completion['choices'][0]['message']['content']

    def num_tokens_from_messages(self, messages: str, model: str = "gpt-3.5-turbo") -> int:
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        if model == "gpt-3.5-turbo-0301":
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo")
        elif model == "gpt-4":
            return self.num_tokens_from_messages(messages, model="gpt-4-0314")
        elif model == "gpt-3.5-turbo":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def get_gpt_query(self, result: list, message: str, history: list, web_id_conf: dict):
        '''
        :param query: result from likr_search
        :param query: question for chatgpt
        -------
        chatgpt_query
            Results:

            [1] "result[0]['htmlTitle']}",snippet= "{result[0]['snippet']}",description = "{result[0]['pagemap']['metatags'][0]['og:description']"

            [2] "result[1]['htmlTitle']}",snippet= "{result[1]['snippet']}",description = "{result[1]['pagemap']['metatags'][0]['og:description']"

            [3] "result[2]['htmlTitle']}",snippet= "{result[2]['snippet']}",description = "{result[2]['pagemap']['metatags'][0]['og:description']"

            Current date: {date}

            Instructions: If you are "{web_id_conf['web_name']}" customer service. Using the information of results or following the flow of conversation, write a comprehensive reply to the given query in 繁體中文 and following the rules below:
            Always cite the information from the provided results using the [number] notation in the end of that sentence.
            Write Bullet list for each subject if you recommend products.
            "親愛的顧客您好，" in the beginning.
            "祝您愉快！" in the end.

            Query: {message}
        '''
        date = datetime.today().strftime('%Y/%m/%d')
        gpt_query = history if history else [{"role": "system", "content": f"我們是{web_id_conf['web_name']}(代號：{web_id_conf['web_id']},官方網站：{web_id_conf['web_url']}),{web_id_conf['description']}"}]
        linkList = []
        if type(result) != str:
            chatgpt_query = """\nInformation:"""
            for v in result:
                if not v.get('link'):
                    continue
                url = v.get('link')
                url = re.search(r'.+detail/[\w\-]+/', url).group(0) if re.search(r'.+detail/[\w\-]+/', url) else url
                if url in linkList:
                    continue
                linkList.append(url)
                if v.get('htmlTitle'):
                    chatgpt_query += f"""\n\n[{len(linkList)}] "{v.get('htmlTitle')}"""
                if v.get('snippet'):
                    chatgpt_query += f""",snippet = "{v.get('snippet')}"""
                if v.get('pagemap') and v.get('pagemap').get('metatags') and v.get('pagemap').get('metatags')[0].get('og:description'):
                    chatgpt_query += f""",description = {v.get('pagemap').get('metatags')[0].get('og:description')}" """
            chatgpt_query += f"""\n\n\nCurrent date: {date}\n\nInstructions: As a customer service representative for "{web_id_conf['web_name']}". Your task is to respond to the Query below in 繁體中文. Always start with "親愛的顧客您好，" and end with "祝您愉快！". Ensure that your response is comprehensive, helpful and response is generated entirely from the information provided. Your should first address the customer's question. Then, recommend three products from the information provided by using bullet points. Use the [number] notation to cite sources in the end of sentences.\n\nCustomer's Query: {message}"""
        else:
            chatgpt_query = f"""\n\n\nCurrent date: {date}\n\nInstructions: If you are the brand, "{web_id_conf['web_name']}"({web_id_conf['web_id']}) customer service and there is no search result in product list, write a comprehensive reply to the given query. Reply in 繁體中文 and Following the rule below:\n"親愛的顧客您好，" in the beginning.\n"祝您愉快！" in the end.\n\nQuery: {message}"""
        gpt_query += [{'role': 'user', 'content': chatgpt_query}]
        while self.num_tokens_from_messages(gpt_query) > 3500 and len(gpt_query) > 3:
            gpt_query = [gpt_query[0]] + gpt_query[3:]
        return gpt_query, linkList

    def get_gpt_order_query(self, order: str, message: str):
        gpt_query = f"""
        Act as an Order Customer Service Expert in answering questions about product orders and customer inquiries.
        I want you to act as an order customer service expert who is responsible for answering questions about product orders and addressing customer inquiries. You must understand and analyze the order information and come up with appropriate solutions and responses to customer questions. Your replies should be polite, informative, and helpful, focusing only on the issues raised by the customers. The answers must be in the same language as the title (Traditional Chinese) . 
        My first order information is :{order}
        My first customer question is '{message}'.
        Only reply in 繁體中文 and "感謝您選擇我們的產品，祝您生活愉快！" in the end.
        """
        return gpt_query

class QA_api:
    def __init__(self, frontend, logger):
        self.CONFIG = self.get_config()
        self.ChatGPT = ChatGPT_AVD()
        self.Search = Search_engine()
        self.Recommend = Recommend_engine()
        self.nineyi000360_store_calc =StoreDistanceEvaluator()
        self.logger = logger
        self.frontend = frontend
        if frontend == 'line':
            self.table_suffix = '_api'
            self.url_format = lambda x: ' ' + x + ' '
            self.user_id_index = 0
        elif frontend == 'slack':
            self.table_suffix = ''
            self.url_format = lambda x: '<' + x + '|查看更多>'
            self.user_id_index = 0

    def get_config(self):
        '''
        Returns {web_id: config from jupiter_new -> web_push.AI_service_config}
        '''
        config_dict = {}
        config = DBhelper('jupiter_new').ExecuteSelect("SELECT * FROM web_push.AI_service_config where mode != 0;")
        config_col = [i[0] for i in DBhelper('jupiter_new').ExecuteSelect("SHOW COLUMNS FROM web_push.AI_service_config;")]
        for conf in config:
            config_dict[conf[1]] = {}
            for k, v in zip(config_col, conf):
                config_dict[conf[1]][k] = v
        return config_dict

    ## Search and Recommend
    def get_question_keyword(self, message: str, web_id: str) -> list:
        forbidden_words = {'client_msg_id', '我', '你', '妳', '們', '沒', '怎', '麼', '要', '沒有', '嗎', '^在$', '^做$'
                           '^如何$', '^賣$', '^有$', '^可以$', '^商品$', '^哪', '哪$'
                           '暢銷', '熱賣', '特別', '最近', '幾天', '常常'}
        # remove web_id from message
        message = translation_stw(message).lower()
        for i in [web_id, self.CONFIG[web_id]['web_name']] + eval(self.CONFIG[web_id]['other_name']):
            message = re.sub(i, '', message)
            for j in list(jieba.cut(i)):
                message = re.sub(j, '', message)
        # segmentation
        reply = self.ChatGPT.ask_gpt([{'role': 'system', 'content': '你會將user的content進行切詞,再依重要性評分數,若存在商品名詞,商品名詞的分數為原本的兩倍。並且只回答此格式為 {"詞":分數} ,不需要其他解釋。'},
                                      {'role': 'user', 'content': f'{message}'}],
                                     model='gpt-4')
        if reply == 'timeout':
            return 'timeout'
        keyword_list = [k for k, _ in sorted(eval(reply).items(), key=lambda x: x[1], reverse=True) if k in message and not any(re.search(w, k) for w in forbidden_words)]
        if not keyword_list:
            keyword_list = [self.ChatGPT.ask_gpt([{'role': 'system', 'content': '你會將user的content選擇一個最重要的關鍵字。並且只回答此格式 "關鍵字" ,不需要其他解釋'},
                                                  {'role': 'user', 'content': f'{message}'}],
                                                 model='gpt-4')]
        return keyword_list

    def split_qa_url(self, result: list[dict], config: dict):
        product_domain = [i.strip('*') for i in config['product_url'].split(',')]
        n_product_url, product_url = [], []
        for r in result:
            if r.get('link') and any([url in r.get('link') for url in product_domain]):
                product_url.append(r)
            else:
                n_product_url.append(r)
        return n_product_url, product_url

    def get_history_df(self, web_id: str, info: str | list) -> pd.DataFrame:
        if self.frontend == 'line':
            query = f"""SELECT id, web_id, group_id, counts, question, answer, q_a_history, add_time FROM web_push.AI_service_api WHERE group_id = '{info[0]}' and web_id = '{web_id}';"""
            df = pd.DataFrame(DBhelper('jupiter_new').ExecuteSelect(query),
                              columns=['id', 'web_id', 'group_id', 'counts', 'question', 'answer', 'q_a_history', 'add_time'])
        if self.frontend == 'slack':
            query = f"""SELECT id, web_id, user_id, ts, counts, question, answer, q_a_history, add_time FROM web_push.AI_service WHERE ts='{info[1]}';"""
            df = pd.DataFrame(DBhelper('jupiter_new').ExecuteSelect(query),
                              columns=['id', 'web_id', 'user_id', 'ts', 'counts', 'question', 'answer', 'q_a_history', 'add_time'])
        return df

    def update_history_df(self, web_id: str, info: str | list, history_df: pd.DataFrame,
                          message: str, answer: str, keyword: str, response_time: float,
                          gpt_query: list, gpt_answer: str) -> pd.DataFrame:
        if len(history_df) == 0:
            if self.frontend == 'line':
                history_df = pd.DataFrame([[web_id, info[0], 0, datetime.now()]],
                                          columns=['web_id', 'group_id', 'counts', 'add_time'])
            elif self.frontend == 'slack':
                history_df = pd.DataFrame([[web_id, info[0], info[1], 0, datetime.now()]],
                                          columns=['web_id', 'user_id', 'ts', 'counts', 'add_time'])
        history_df['counts'] += 1
        history_df[['question', 'answer', 'keyword', 'response_time', 'q_a_history']] = [message, answer, keyword, response_time, json.dumps(gpt_query[:-1] + [{"role": "assistant", "content": f"{gpt_answer}"}])]
        history_df['q_a_history'] =  json.dumps(gpt_query[:-1] + [{"role": "user", "content": f"{message}"}, {"role": "assistant", "content": f"{gpt_answer}"}])
        _df = history_df.drop(columns=['keyword', 'response_time'])
        DBhelper.ExecuteUpdatebyChunk(_df, db='jupiter_new', table=f'AI_service{self.table_suffix}', is_ssh=False)
        _df = history_df.drop(columns=['q_a_history'])
        if 'id' in history_df.columns:
            _df = _df.drop(columns=['id'])
        DBhelper.ExecuteUpdatebyChunk(_df, db='jupiter_new', table=f'AI_service_cache{self.table_suffix}', is_ssh=False)

    ## lbs
    def search_nearest_store_nineyi000360(self, gps: tuple, history: list):
        user_history_message = []
        for i in history[::-1]:
            if i.get('role') == 'user' and i.get('content', ''):
                user_history_message.append(re.search(r'(?<=Query: ).+', i['content']).group(0) if 'Query:' in i['content'] else i['content'])
        for message in user_history_message * 3:
            flags, _ = self.judge_question_type(message)
            if flags.get('store_address'):
                return self.nineyi000360_store_calc.get_nearest_store(gps)
        return

    ## order system
    def get_order_type(self, web_id: str, user_id: str) -> int:
        query = f"""SELECT web_id, user_id, types, orders, timestamps FROM web_push.AI_service_order_test WHERE user_id = '{user_id}' and web_id = '{web_id}';"""
        timestamps = int(datetime.timestamp(datetime.now()))
        df = pd.DataFrame(DBhelper('jupiter_new').ExecuteSelect(query),columns=['web_id', 'user_id', 'types', 'orders','timestamps'])
        if len(df) == 0:
            df.loc[0] = [web_id, user_id, 1, '訂單編號:202300413041,購入日期:2023/04/13,總計:5274,訂單狀況:付款完畢,商品1:JC22S-C,商品1數量:1 ,商品2:Luxe-c22,商品2數量:2', timestamps]
        df['timestamps'] = timestamps
        DBhelper.ExecuteUpdatebyChunk(df, db='jupiter_new', table='AI_service_order_test', is_ssh=False)
        return df['types'].get(0), df['orders'].get(0)

    def answer_append(self, answer: str, flags: dict) -> str:
        kind_dict = {'delivery': '到貨', 'purchase': '購買', 'payment': '付款', 'return/exchange': '退換貨', 'order': '訂單'}
        add = f"\n\n想知道更詳細的"
        if flags.get('order'):
            add += f"{kind_dict['order']}"
        elif flags.get('purchase'):
            add += f"{kind_dict['purchase']}"
        elif flags.get('return/exchange'):
            add += f"{kind_dict['return/exchange']}"
        elif flags.get('payment'):
            add += f"{kind_dict['payment']}"
        elif flags.get('delivery'):
            add += f"{kind_dict['delivery']}"
        else:
            return answer
        add += "資訊, 請登入此網址查詢[https://www.gaii.ai/product/home/20220317000001/auth/sign_in]"
        return answer + add

    ## QA Flow
    def message_classifier(self, message: str, web_id: str):
        message = message.replace('在哪裡', '在哪').replace('在哪', '在哪裡')
        if re.search('\(\d{1,3}\.\d+,\d{1,3}\.\d+\)', message) and web_id == 'nineyi000360':
            return message, eval(re.search('\(\d{1,3}\.\d+,\d{1,3}\.\d+\)', message).group(0))
        else:
            return message, tuple()

    def check_message_length(self, message: str, length: int = 50) -> bool:
        for url in re.findall(r'https?:\/\/[\w\.\-\/\?\=\+&#$%^;%_]+', message):
            if fetch_url_response(url):
                message = message.replace(url, '')
        return len(message) <= length

    def judge_question_type(self, message: str) -> tuple:
        flag_dict = {'Product details': 'product',
                     'Accessories selection': 'product',
                     'Product comparison': 'product',
                     'Product guarantee and warranty': 'product',
                     'Product safety': 'product',
                     'Product use': 'product',
                     'Product sourcing': 'product',
                     'Product maintenance': 'product',
                     'Product videos/images': 'product',
                     'Customized products': 'product',
                     'Product packaging contents': 'product',
                     'Product use scenarios': 'product',
                     'Product storage': 'product',
                     'Product purchase': 'purchase',
                     'Inventory status': 'purchase',
                     'Inventory alert': 'purchase',
                     'In-store availability': 'purchase',
                     'Payment methods': 'payment',
                     'Card payment': 'payment',
                     'Tax and customs duties': 'payment',
                     'Shipping fee': 'delivery',
                     'delivery area': 'delivery',
                     'Cash on Delivery': 'delivery',
                     'In-store pickup': 'delivery',
                     'Membership information': 'member',
                     'Membership account id or password': 'member',
                     'Membership system': 'member',
                     'Membership registration': 'member',
                     'Membership discounts': 'member',
                     'Membership activity': 'member',
                     'Order change': 'order',
                     'Order cancellation': 'order',
                     'Order tracking': 'order',
                     'Check order status': 'order',
                     'Delivery address': 'order',
                     'Return policy': 'return/exchange',
                     'Refund': 'return/exchange',
                     'Product damage': 'return/exchange',
                     'Exchange policy': 'return/exchange',
                     'Contact customer service': 'Customer service',
                     'Customer reviews': 'Customer service',
                     'Customer feedback': 'Customer service',
                     'Ratings': 'Customer service',
                     'Popular products': 'hot',
                     'Hot-selling products': 'hot',
                     'Discounts': 'promotion',
                     'Discount amount': 'promotion',
                     'Promotions': 'promotion',
                     'Gift cards': 'promotion',
                     'Coupons': 'promotion',
                     'Store address': 'store',
                     'Brand': 'store',
                     'Business hours': 'store',
                     'Contact information': 'store',
                     'Store official site': 'store',
                     'Social media': 'store',
                     'Shopping guide': 'store',
                     'Privacy and security': 'store',
                     'Search store location': 'store_address',
                     'gps location': 'gps',
                     'Not e-commerce field question': 'others'}
        flag_list = list(flag_dict.keys())
        system_query = '\n'.join([f'{i + 1}. {flag}' for i, flag in enumerate(flag_list)])
        system_query += f"""\nUsing the classifications outlined above, classify the user's content as accurately. Please format your response as a series of numbers [e.g. 1, 2, 3...].Please ensure that your classification accurately reflects the content and captures its nuances."""
        retry = 3
        while retry:
            try:
                reply = self.ChatGPT.ask_gpt([{'role': 'system', 'content': system_query}, {'role': 'user', 'content': message}], model='gpt-4', timeout=10)
                if reply == 'timeout':
                    raise Exception
                flags = [flag_list[int(i) - 1] for i in re.findall(r'\d+',reply)]
                if flags == ['Not e-commerce field question']:
                    raise Exception
                if flags:
                    break
            except Exception as e:
                print('retry judge')
                retry -= 1
        if not retry:
            flags = ['Not e-commerce field question']
        flags_class = {flag_dict[f]: True for f in flags}
        if not flags_class.get('product') and not flags_class.get('others'):
            flags_class['QA'] = True
        return flags_class, flags

    def adjust_ans_url_format(self, answer: str, linkList: list) -> str:
        url_set = sorted(list(set(re.findall(r'https?:\/\/[\w\.\-\?/=+&#$%^;%_]+', answer))), key=len, reverse=True)
        for url in url_set:
            reurl = url
            for char in '?':
                reurl = reurl.replace(char, '\\' + char)
            answer = re.sub(reurl + '(?![\w\.\-\?/=+&#$%^;%_\|])', self.url_format(url), answer)
        for i, url in enumerate(linkList):
            answer = re.sub(f'\[#?{i + 1}\](?!.*\[#?{i + 1}\])', f'[{self.url_format(url)}]', answer, count=1)
        answer = re.sub(f'\[#?\d\]', f'', answer)
        return answer

    def adjust_ans_format(self, answer: str, ) -> str:
        if self.frontend == 'line':
            answer.replace('"', "'")
        replace_words = {'此致', '敬禮', '<b>', '</b>', r'\[?\[\d\]?\]?|\[?\[?\d\]\]?', '\w*(抱歉|對不起)\w{0,3}(，|。)'}
        for w in replace_words:
            answer = re.sub(w, '', answer).strip('\n')
        if '親愛的' in answer:
            answer = '親愛的' + '親愛的'.join(answer.split("親愛的")[1:])
        if '祝您愉快！' in answer:
            answer = '祝您愉快！'.join(answer.split("祝您愉快！")[:-1]) + '祝您愉快！'
        return answer

    def error(self, *arg):
        self.logger.print(*arg, level="WARNING")
        return '客服忙碌中，請稍後再試。'

    def QA(self, web_id: str, message: str, info: str | list):
        start_time = time.time()
        user_id = info[self.user_id_index]

        message, gps_location = self.message_classifier(message, web_id)
        if not message and not gps_location:
            return
        self.logger.print(f'Get Message:\t{message}')

        if not self.check_message_length(message, 50):
            self.logger.print('USER ERROR: Input too long!')
            return "親愛的顧客您好，您的提問長度超過限制，請縮短問題後重新發問。"

        #types, orders = self.get_order_type(web_id, user_id, message)
        history_df = self.get_history_df(web_id, info)
        history = json.loads(history_df['q_a_history'].iloc[0]) if len(history_df) > 0 else []
        self.logger.print('QA歷史紀錄:\n', history)
        flags, f = self.judge_question_type(message)
        self.logger.print(f'客戶意圖:\t{flags}\n{f}')
        # if types == 1:
        #     gpt_query = self.ChatGPT.get_gpt_order_query(orders,message)
        #     self.logger.print('訂單系統ChatGPT輸入:\n',gpt_query)
        #     gpt_answer = translation_stw(self.ChatGPT.ask_gpt(gpt_query, timeout=60)).replace('，\n', '，')
        #     if '生活愉快！' in gpt_answer:
        #         gpt_answer = '生活愉快！'.join(gpt_answer.split("生活愉快！")[:-1]) + '生活愉快！'
        #     self.logger.print('訂單系統ChatGPT輸出:\n', gpt_query)
        #     return gpt_answer

        if gps_location:
            store_result = self.search_nearest_store_nineyi000360(gps_location, history)
            if store_result:
                flags['store_address'] = True
                flags['hot'] = True
                message = '給我全家便利商店的位置資訊。'
            else:
                return "親愛的顧客您好，我們不確定您的問題或需求，如果您有任何疑慮或需要任何協助，請隨時聯絡我們的客戶服務團隊。"
        if not gps_location and flags.get('store_address') and web_id == 'nineyi000360':
            keyword = ''
            gpt_query = [{'role': 'user', 'content': message}]
            gpt_answer = answer = "親愛的顧客您好，若是需要查詢最近的全家便利店位置，請提供我們您現在的位置。"
        else:
            # Step 1: get keyword from chatGPT
            if gps_location:
                keyword_list = []
            else:
                keyword_list = self.get_question_keyword(message, web_id)
                if keyword_list == 'timeout':
                    return self.error('keyword_timeout')
                self.logger.print('關鍵字:\t', keyword_list)

            # Step 2: get gpt_query with search results from google search engine and likr recommend engine
            try:
                if gps_location and flags.get('store_address'):
                    result = recommend_result = store_result + self.Recommend.likr_recommend([], '', flags, self.CONFIG[web_id])[:3]
                    keyword = '全家'
                else:
                    result, keyword = func_timeout(10, self.Search.likr_search, (keyword_list, self.CONFIG[web_id]))
                    self.logger.print(f'Search_result:\t {[i.get("link") for i in result if i.get("link")], keyword}')
                    n_product_result, product_result = self.split_qa_url(result, self.CONFIG[web_id])
                    self.logger.print(f'QA_result:\t {[i.get("link") for i in n_product_result if i.get("link")], keyword}')
                    if web_id == 'avividai':
                        recommend_result = result[:3]
                    else:
                        recommend_result = self.Recommend.likr_recommend(product_result, keyword_list, flags, self.CONFIG[web_id])[:3]
                    self.logger.print(f'Recommend_result:\t {[i.get("link") for i in recommend_result if i.get("link")], keyword}')
                    if flags.get('QA'):
                        recommend_result = n_product_result[:2] + recommend_result
            except Exception as e:
                self.logger.print(f'{traceback.format_tb(e.__traceback__)[-1]}\n ERROR: {e}', level='ERROR')
                return self.error('search_or_recommend_error')

            if len(result) == 0 and self.CONFIG[web_id]['mode'] == 2:
                gpt_query = [{'role': 'user', 'content': message}]
                gpt_answer = answer = f"親愛的顧客您好，目前無法回覆此問題，稍後將由專人為您服務。"
            # Step 3: response from ChatGPT
            else:
                gpt_query, linkList = self.ChatGPT.get_gpt_query(recommend_result, message, history, self.CONFIG[web_id])
                self.logger.print('輸入連結:\n', '\n'.join(linkList))
                self.logger.print('ChatGPT輸入:\t', gpt_query)

                gpt_answer = translation_stw(self.ChatGPT.ask_gpt(gpt_query, timeout=60)).replace('，\n', '，')
                if gpt_answer == 'timeout':
                    return self.error('gpt3_answer_timeout')
                self.logger.print('ChatGPT輸出:\t', gpt_answer)
                answer = self.adjust_ans_format(self.adjust_ans_url_format(gpt_answer, linkList))
                if web_id == 'avividai' and history == []:
                    answer += """至於收費方式由於選擇方案的不同會有所差異，還請您務必填寫表單以留下資訊，我們將由專人進一步與您聯絡！\n\n表單連結：https://forms.gle/S4zkJynXj5wGq6Ja9"""
                answer = self.answer_append(answer,flags)
                self.logger.print('回答:\t', answer)

        # Step 4: update database
        self.update_history_df(web_id, info, history_df, message, answer, keyword, time.time()-start_time, gpt_query, gpt_answer)
        self.logger.print('本次問答回應時間:\t', time.time()-start_time)
        return answer


if __name__ == "__main__":
    # slack
    AI_customer_service = QA_api('slack', logger())
    print(AI_customer_service.QA('pure17', '有沒有健步機', ['U03PN370PRU', '1679046590.110499']))
    # line
    AI_customer_service = QA_api('line', logger())
    print(AI_customer_service.QA('pure17', '我想詢問訂單', 'test4466'))
