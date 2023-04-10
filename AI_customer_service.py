import os, sys
from dotenv import load_dotenv
sys.path.append("..")
from func_timeout import func_timeout
import datetime
import time
load_dotenv()
import os, sys, logging
import re, json, requests
import itertools
import pandas as pd
import jieba
from opencc import OpenCC
import openai
from db import DBhelper
from jieba import posseg


class AI_customer_service:
    def __init__(self):
        self.OPEN_AI_KEY_DICT = eval(os.getenv('OPENAI_API_KEY'))
        self.GOOGLE_SEARCH_KEY = os.getenv('GOOGLE_SEARCH_KEY')
        self.CONFIG = self.get_config()

    def get_openai_key_id(self,model):
        if 'gpt-4' in model:
            opena_ai_key_id = DBhelper('jupiter_new').ExecuteSelect("SELECT id, counts FROM web_push.AI_service_token_counter x WHERE id < 6 ORDER BY counts limit 1;")
        else:
            opena_ai_key_id = DBhelper('jupiter_new').ExecuteSelect("SELECT id, counts FROM web_push.AI_service_token_counter x ORDER BY counts limit 1;")
        return opena_ai_key_id[0][0]

    def get_keys(func):
        def inner(self,message, model="gpt-3.5-turbo", timeout=60):
            print(model)
            token_id = self.get_openai_key_id(model)
            DBhelper('jupiter_new').ExecuteDelete(f'UPDATE web_push.AI_service_token_counter SET counts = counts + 1 WHERE id = {token_id}')
            try:
                ans = func_timeout(timeout, func, (self,message, token_id, model))
            except:
                ans = 'timeout'
            DBhelper('jupiter_new').ExecuteDelete(f'UPDATE web_push.AI_service_token_counter SET counts = counts - 1 WHERE id = {token_id}')
            return ans
        return inner

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

    def check_message_length(self,text, length):
        for url in re.findall(r'https?:\/\/[\w\.\-\/\?\=\+&#$%^;%_]+', text):
            stopSwitch, retry, result = False, 3, None
            while not stopSwitch and retry:
                try:
                    response = requests.get(url)
                except Exception as e:
                    break
                if response:
                    stopSwitch = True
                retry -= 1
            print(response.status_code)
            if stopSwitch and response.status_code == 200:
                text = text.replace(url, '')
        return len(text) <= length

    @get_keys
    def ask_gpt(self,message, token_id=None, model="gpt-3.5-turbo"):
        openai.api_key = self.OPEN_AI_KEY_DICT[token_id]
        if type(message) == str:
            message = [{'role': 'user', 'content': message}]
        completion = openai.ChatCompletion.create(model=model, messages=message)
        return completion['choices'][0]['message']['content']

    def question_pos_parser(self,question, retry=3, web_id='nineyi000360'):
        '''
        :param mode: N => just filter noun
        --------
        It will early return when there's only one word after segmentation.
        It will return one word chosen by chatGPT when there are no words after filtering by chatGPT.
        '''
        question = self.translation_stw(question).lower()
        for i in [self.CONFIG[web_id]['web_id'], self.CONFIG[web_id]['web_name']] + eval(self.CONFIG[web_id]['other_name']):
            question = question.replace(i, '')
            for j in list(jieba.cut(i)):
                if j in question:
                    question = question.replace(j, '')
        stopSwitch, retry, keyword = False, retry, ''
        forbidden_words = {'client_msg_id', '什麼', '有', '我', '你', '妳', '你們', '妳們', '沒有', '怎麼', '怎','如何', '要'}
        not_noun_list = [w for w, p in list(posseg.cut(question)) if 'n' not in p.lower()]
        while not stopSwitch and retry:
            keyword = self.ask_gpt(f"""To "{question}", choose the {min((len(question) + 2) // 3, 3)} most important words that are always used as nouns and cannot exceed 3 Chinese characters, and separate by " ".""",model='gpt-4').replace('\n', '').replace('"', '').replace("。", '')
            keyword = [i.strip('.') for i in keyword.split(' ') if not any(re.search(w, i.strip('.')) for w in forbidden_words) and i.strip('.') not in not_noun_list and i.strip('.') in question]
            stopSwitch = len(keyword) > 0
            retry -= 1
        if not keyword:
            keyword = [self.ask_gpt(f'Choose one important word from "{question}". Just reply the word in 繁體中文.',model='gpt-4').split(' ')[0].replace('\n', '').replace('"', '').replace("。", '')]
        print(keyword)
        if keyword == 'timeout':
            print('keyword_error')
            return 'timeout'
        return list(map(self.translation_stw, keyword))

    def translation_stw(self,text):
        cc = OpenCC('likr-s2twp')
        return cc.convert(text)

    def google_search(self,keyword_combination, html, retry):
        for kw in keyword_combination:
            kw = '+'.join(kw)
            print(f'Keyword for search:\t {kw}')
            search_html = html + kw
            print(f'Search URL:\t {search_html}')
            stopSwitch, cnt, result = False, 1, None
            while not stopSwitch and cnt != retry + 1:
                print(f'Search times:\t {cnt}')
                response = requests.get(search_html)
                if response:
                    stopSwitch = response.status_code == 200
                    result = response.json().get('items')
                    result_kw = kw
                cnt += 1
            if result: break
        if not stopSwitch:
            return 'URL ERROR', None
        return result, result_kw

    def likr_search(self,keyword_list, web_id='nineyi000360', keyword_length=3):
        if len(keyword_list) > keyword_length:
            keyword_list = keyword_list[:keyword_length]
        result = None
        keyword_combination = []
        for i in range(len(keyword_list), 0, -1):
            keyword_combination += list(itertools.combinations(sorted(keyword_list, key=len, reverse=True), i))
        ##todo 推薦引擎
        if self.CONFIG[web_id]['domain_cx'] != '_':
            html = f"https://www.googleapis.com/customsearch/v1/siterestrict?cx={self.CONFIG[web_id]['domain_cx']}&key={self.GOOGLE_SEARCH_KEY}&q="
            result, result_kw = self.google_search(keyword_combination, html, 3)
            if result == 'URL ERROR':
                result, result_kw = self.google_search(keyword_combination, html[:42] + html[55:], 1)
        if (not result or result == 'URL ERROR') and self.CONFIG[web_id]['sub_domain_cx'] != '_':
            html = f"https://www.googleapis.com/customsearch/v1/siterestrict?cx={self.CONFIG[web_id]['sub_domain_cx']}&key={self.GOOGLE_SEARCH_KEY}&q="
            result, result_kw = self.google_search(keyword_combination, html, 3)
            if result == 'URL ERROR':
                result, result_kw = self.google_search(keyword_combination, html[:42] + html[55:], 1)
        if (not result or result == 'URL ERROR') and str(self.CONFIG[web_id]['mode']) == '3':
            result, result_kw = self.google_search(keyword_combination,f"https://www.googleapis.com/customsearch/v1?cx=46d551baeb2bc4ead&key={self.GOOGLE_SEARCH_KEY}&q={self.CONFIG[web_id]['web_name'].replace(' ', '+')}+",1)
        if not result:
            print(f"No results: {html}, {'+'.join(keyword_list)}")
            result, result_kw = [{'NO RESULTS': True}], '+'.join(keyword_list)
        elif result == 'URL ERROR':
            result = [{'URL ERROR': True}]
            print(f"URL ERROR: {html}, {'+'.join(keyword_list)}")
            result_kw = '+'.join(keyword_list)
        return result, result_kw

    def get_gpt_query(self,result, query, history, web_id):
        '''
        :param query: result from likr_search
        :param query: question for chatgpt
        -------
        chatgpt_query
            Results:

            [1] "{g[0]['snippet']}"
            URL: "{g[0]['link']}"

            [2] "{g[1]['snippet']}"
            URL: "{g[1]['link']}"

            [3] "{g[2]['snippet']}"
            URL: "{g[2]['link']}"


            Current date: {date}

            Instructions: Using the provided products or Q&A, write a comprehensive reply to the given query. Reply in 繁體中文 and Following the rule below:
            Always cite results using [[number](URL)] notation in the sentence's end when using the information from results.
            Write separate answers for each subject.
            "親愛的顧客您好，" in the beginning.
            "祝您愉快！" in the end.

            Query: {query}
        '''
        date = datetime.datetime.today().strftime('%Y/%m/%d')
        message = history if history else [{"role": "system",
                                            "content": f"我們是{self.CONFIG[web_id]['web_name']}(代號：{self.CONFIG[web_id]['web_id']},官方網站：{self.CONFIG[web_id]['web_url']}),{self.CONFIG[web_id]['description']}"}]
        print(self.CONFIG[web_id]['description'])
        if type(result) != str:
            linkSet = set()
            chatgpt_query = """\nResults:"""
            for v in result:
                if not v.get('link') or len(linkSet) == 3:
                    continue
                url = v.get('link')
                url = re.search(r'.+detail/[\w\-]+/', url).group(0) if re.search(r'.+detail/[\w\-]+/', url) else url
                print('搜尋結果:\t', url)
                if url in linkSet:
                    continue
                linkSet.add(url)

                if v.get('htmlTitle'):
                    chatgpt_query += f"""\n\n[{len(linkSet)}] "{v.get('htmlTitle')}"""
                if v.get('snippet'):
                    chatgpt_query += f""",snippet = "{v.get('snippet')}"""
                if v.get('pagemap') and v.get('pagemap').get('metatags'):
                    chatgpt_query += f""",description = {v.get('pagemap').get('metatags')[0].get('og:description')}" """
                chatgpt_query += f"""\nURL: "{url}" """
            chatgpt_query += f"""\n\n\nCurrent date: {date}\n\nInstructions: Using the provided products or Q&A, write a comprehensive reply to the given query. Reply in 繁體中文 and Following the rule below:\nAlways cite results using [[number](URL)] notation in the sentence's end when using the information from results.\nWrite separate answers for each subject.\n"親愛的顧客您好，" in the beginning.\n"祝您愉快！" in the end.\n\nQuery: {query}"""
        else:
            chatgpt_query = f"""\n\n\nCurrent date: {date}\n\nInstructions: If you are the brand, "{self.CONFIG[web_id]['web_name']}"({web_id}) customer service and there is no search result in product list, write a comprehensive reply to the given query. Reply in 繁體中文 and Following the rule below:\n"親愛的顧客您好，" in the beginning.\n"祝您愉快！" in the end.\n\nQuery: {query}"""
        message += [{'role': 'user', 'content': chatgpt_query}]
        return message

    def replace_answer(self,gpt3_ans):
        print(f"ChatGPT reply：\t {gpt3_ans}")
        for url_wrong_fmt, url in re.findall(r'(<(https?:\/\/[\w\.\-\?/=+&#$%^;%_]+)\|.*>)', gpt3_ans):
            gpt3_ans = gpt3_ans.replace(url_wrong_fmt, url)
        for url_wrong_fmt, url in re.findall(r'(\[?\d\]?\(?(https?:\/\/[\w\.\-\/\?\=\+\&\#\$\%\^\;\%\_]+)\)?)',
                                             gpt3_ans):
            gpt3_ans = gpt3_ans.replace(url_wrong_fmt, url)
        gpt3_ans = self.translation_stw(gpt3_ans)
        gpt3_ans = gpt3_ans.replace('，\n', '，')
        url_set = sorted(list(set(re.findall(r'https?:\/\/[\w\.\-\?/=+&#$%^;%_]+', gpt3_ans))), key=len, reverse=True)
        for url in url_set:
            reurl = url
            for char in '?':
                reurl = reurl.replace(char, '\\' + char)
            gpt3_ans = re.sub(reurl + '(?![\w\.\-\?/=+&#$%^;%_\|])', ' ' + url + ' ', gpt3_ans)
        replace_words = {'此致', '敬禮', '<b>', '</b>', r'\[?\[\d\]?\]?|\[?\[?\d\]\]?', '\w*(抱歉|對不起)\w{0,3}(，|。)'}
        for w in replace_words:
            gpt3_ans = re.sub(w, '', gpt3_ans).strip('\n')
        if '親愛的' in gpt3_ans:
            gpt3_ans = '親愛的' + '親愛的'.join(gpt3_ans.split("親愛的")[1:])
        if '祝您愉快！' in gpt3_ans:
            gpt3_ans = '祝您愉快！'.join(gpt3_ans.split("祝您愉快！")[:-1]) + '祝您愉快！'
        return gpt3_ans

    def gpt_QA(self,web_id, message, group_id):
        start_time = time.time()
        query = f"""SELECT id, web_id, counts, question, answer, q_a_history,add_time,group_id FROM web_push.AI_service_api WHERE group_id='{group_id}' and web_id = '{web_id}';"""
        data = DBhelper('jupiter_new').ExecuteSelect(query)
        QA_report_df = pd.DataFrame(data,columns=['id', 'web_id', 'counts', 'question', 'answer', 'q_a_history', 'add_time','group_id'])
        # Step 1: get keyword from chatGPT
        keyword_list = self.question_pos_parser(message, 3, web_id)
        if type(keyword_list) == str:
            print('keyword_timeout_error')
            return '客服忙碌中，請稍後再試。'
        print('關鍵字:\t', keyword_list)
        # Step 2: get gpt_query with search results from google search engine
        #try:
        result, keyword = func_timeout(10, self.likr_search, (keyword_list, web_id))
            ###推薦引擎 result_recommend, flags = func_timeout(20, engine.likr_recommend_engine, (message, web_id))
        #except:
        #    print('keyword_timeout')
        #    return '客服忙碌中，請稍後再試。'
        if result[0].get('URL ERROR'):
            return "客服忙碌中，請稍後再試。"

        if result[0].get('NO RESULTS') and self.CONFIG[web_id]['mode'] == 2:
            gpt3_answer = gpt3_answer_slack = f"親愛的顧客您好，目前無法回覆此問題，稍後將由專人為您服務。"
        else:
            # Step 3: response from chatGPT
            history = None
            if len(QA_report_df) > 0:
                history = json.loads(QA_report_df['q_a_history'].iloc[0])
            ###推薦引擎result = reset_result_order(result, result_recommend, flags, web_id)
            gpt_query = self.get_gpt_query(result, message, history, web_id)
            while len(str(gpt_query)) > 3000 and len(gpt_query) > 3:
                gpt_query = [gpt_query[0]] + gpt_query[3:]
            print('chatGPT輸入:\t', gpt_query)

            gpt3_answer = self.ask_gpt(gpt_query, timeout=60)
            if gpt3_answer == 'timeout':
                print('gpt3_answer_error')
                return "客服忙碌中，請稍後再試。"

            gpt3_answer_slack = self.replace_answer(gpt3_answer)
            print('cahtGPT輸出:\t', gpt3_answer_slack)

        if history:
            gpt_query.append({"role": "assistant", "content": f"{gpt3_answer}"})
            QA_report_df['counts'] += 1
            QA_report_df[['question', 'answer', 'q_a_history']] = [message, gpt3_answer_slack, json.dumps(gpt_query)]
        else:
            gpt3_history = json.dumps(gpt_query + [{"role": "assistant", "content": f"{gpt3_answer}"}])
            QA_report_df = pd.DataFrame([[web_id, group_id, 1, message, gpt3_answer_slack, gpt3_history, datetime.datetime.now()]],columns=['web_id', 'group_id', 'counts', 'question', 'answer', 'q_a_history', 'add_time'])
        DBhelper.ExecuteUpdatebyChunk(QA_report_df, db='jupiter_new', table='AI_service_api', chunk_size=100000,is_ssh=False)
        QA_report_df = QA_report_df.drop(['q_a_history'], axis=1)
        if 'id' in QA_report_df.columns:
            QA_report_df = QA_report_df.drop(['id'], axis=1)
        QA_report_df['keyword'] = keyword
        QA_report_df['reponse_time'] = time.time() - start_time
        # DB_query = DBhelper.generate_update_SQLquery(QA_report_df,'AI_service_cache_api',SQL_ACTION="INSERT INTO")
        # DBhelper('jupiter_new').ExecuteUpdate(DB_query,QA_report_df.to_dict('records'))
        DBhelper.ExecuteUpdatebyChunk(QA_report_df, db='jupiter_new', table='AI_service_cache_api', chunk_size=100000,is_ssh=False)
        return gpt3_answer_slack.replace('"', "'")


#########AI_custom




if __name__ == "__main__":
    A = AI_customer_service()
    print(A.gpt_QA('pure17','你們賣什麼','test4466'))







