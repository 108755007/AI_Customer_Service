import os, datetime, itertools, json
import pandas as pd
from dotenv import load_dotenv
from db import DBhelper
from utils.AI_customer_service_utils import fetch_url_response

load_dotenv()

class Search_engine():
    def __init__(self):
        self.GOOGLE_SEARCH_KEY = os.getenv('GOOGLE_SEARCH_KEY')

    def google_search(self, keyword_combination, url, retry, web_id_config, history=True):
        web_id = web_id_config['web_id']
        data = None
        if history:
            cx = url.split('cx=')[1].split('&key')[0]
            query = f"""SELECT keyword, response 
                        FROM web_push.AI_service_search_cache 
                        WHERE web_id = '{web_id}' 
                        AND cx = '{cx}'
                        AND update_time > NOW() - INTERVAL {web_id_config['search_cache_days']} DAY;"""
            data = DBhelper('jupiter_new').ExecuteSelect(query)
        dict_cache = dict(data) if data else dict()
        for kw in keyword_combination:
            kw = '+'.join(kw)
            if kw in dict_cache:
                return json.loads(dict_cache[kw]), kw
            response = fetch_url_response(url + kw, retry)
            if response:
                print(f'Keyword for search:\t {kw}')
                print(f'Search URL:\t {url + kw}')
                if history:
                    df = pd.DataFrame([{'web_id': web_id,
                                        'cx': cx,
                                        'keyword': kw,
                                        'response': json.dumps(response.json().get('items')),
                                        'add_time': datetime.datetime.now()}])
                    DBhelper.ExecuteUpdatebyChunk(df, db='jupiter_new', table=f'AI_service_search_cache', is_ssh=False)
                return response.json().get('items'), kw
        return None, None

    def get_search_url_list(self, web_id_conf: dict, product=True):
        url_list = []
        if web_id_conf['domain_cx'] != '_':
            url_list.append((f"https://www.googleapis.com/customsearch/v1/siterestrict?cx={web_id_conf['domain_cx']}&key={self.GOOGLE_SEARCH_KEY}&q=", 1))
        if web_id_conf['sub_domain_cx'] != '_':
            url_list.append((f"https://www.googleapis.com/customsearch/v1/siterestrict?cx={web_id_conf['sub_domain_cx']}&key={self.GOOGLE_SEARCH_KEY}&q=", 2))
        return url_list if product else url_list[::-1]

    def get_search_keyword_combination(self, keyword_list, max_length: int = 3):
        keyword_combination = []
        for i in range(min(len(keyword_list[:max_length]), 2), 0, -1):
            for j in itertools.combinations(keyword_list[:max_length], i):
                if j not in keyword_combination:
                    keyword_combination.append(j)
        return keyword_combination
    def split_qa_url(self, result: list[dict], config: dict):
        product_domain = [i.strip('*') for i in config['product_url'].split(',')]
        n_product_url, product_url = [], []
        for r in result:
            if r.get('link') and any([url in r.get('link') for url in product_domain]):
                product_url.append(r)
            else:
                n_product_url.append(r)
        return n_product_url, product_url[:1]
    def likr_search(self, keyword_list: list, web_id_conf: dict, max_length: int = 3, history=True, product=True):
        keyword_combination = self.get_search_keyword_combination(keyword_list, max_length)
        url_list = self.get_search_url_list(web_id_conf, product)
        for url, retry in url_list:
            result, result_kw = self.google_search(keyword_combination, url, retry, web_id_conf, history)
            if result:
                return (result, retry), result_kw

        # no result
        if web_id_conf['mode'] == '3':
            url = f"https://www.googleapis.com/customsearch/v1?cx=46d551baeb2bc4ead&key={self.GOOGLE_SEARCH_KEY}&q={web_id_conf['web_name'].replace(' ', '+')}+"
            result, result_kw = self.google_search([(i,) for i in keyword_list], url, 1)
            retry = 3
        if not result:
            print(f"No results: {'+'.join(keyword_list)}")
            result, result_kw = [], '+'.join(keyword_list)
            retry = 0
        return (result, 0), result_kw