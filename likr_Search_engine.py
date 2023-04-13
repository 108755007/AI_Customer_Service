from AI_customer_service_utils import fetch_url_response
import itertools
import os
from dotenv import load_dotenv
load_dotenv()

class Search_engine():
    def __init__(self):
        self.GOOGLE_SEARCH_KEY = os.getenv('GOOGLE_SEARCH_KEY')

    def google_search(self, keyword_combination, url, retry):
        for kw in keyword_combination:
            kw = '+'.join(kw)
            print(f'Keyword for search:\t {kw}')
            print(f'Search URL:\t {url + kw}')
            response = fetch_url_response(url + kw, retry)
            if response:
                return response.json().get('items'), kw
        return None, None

    def likr_search(self, keyword_list, web_id_conf, max_length=3):
        keyword_combination = [j for i in range(min(len(keyword_list[:max_length]), 2), 0, -1) for j in itertools.combinations(keyword_list[:max_length], i)]
        result, url_list = None, []
        for cx in (web_id_conf['domain_cx'], web_id_conf['sub_domain_cx']):
            if cx != '_':
                url_list.append((f"https://www.googleapis.com/customsearch/v1/siterestrict?cx={cx}&key={self.GOOGLE_SEARCH_KEY}&q=", 3))
                url_list.append((f"https://www.googleapis.com/customsearch/v1?cx={cx}&key={self.GOOGLE_SEARCH_KEY}&q=", 1))
        for url, retry in url_list:
            result, result_kw = self.google_search(keyword_combination, url, retry)
            if result:
                return result, result_kw

        # no result
        if web_id_conf['mode'] == '3':
            result, result_kw = self.google_search([(i,) for i in keyword_list],
                                                   f"https://www.googleapis.com/customsearch/v1?cx=46d551baeb2bc4ead&key={self.GOOGLE_SEARCH_KEY}&q={web_id_conf['web_name'].replace(' ', '+')}+", 1)
        if not result:
            print(f"No results: {'+'.join(keyword_list)}")
            result, result_kw = [{'NO RESULTS': True}], '+'.join(keyword_list)
        return result, result_kw