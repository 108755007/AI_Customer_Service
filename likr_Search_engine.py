from AI_customer_service_utils import fetch_url_response
import itertools
import os
from dotenv import load_dotenv
load_dotenv()

class Search_engine():
    def __init__(self, config):
        self.GOOGLE_SEARCH_KEY = os.getenv('GOOGLE_SEARCH_KEY')
        self.CONFIG = config

    def google_search(self, keyword_combination, url, retry):
        result = None
        for kw in keyword_combination:
            kw = '+'.join(kw)
            print(f'Keyword for search:\t {kw}')
            search_url = url + kw
            print(f'Search URL:\t {search_url}')
            stopSwitch, cnt, result = False, 1, None
            response = fetch_url_response(search_url, retry)
            if response:
                result = response.json().get('items')
                result_kw = kw
                break
        if not result:
            return 'URL ERROR', None
        return result, result_kw

    def likr_search(self,keyword_list, web_id='nineyi000360', keyword_length=3):
        if len(keyword_list) > keyword_length:
            keyword_list = keyword_list[:keyword_length]
        result = None
        keyword_combination = []
        for i in range(len(keyword_list), 0, -1):
            keyword_combination += list(itertools.combinations(sorted(keyword_list, key=len, reverse=True), i))
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