import random
import base64
import re
import numpy as np
import itertools
from db import DBhelper
from utils.urlencode import UrlEncode


class Recommend_engine:

    def __init__(self):
        pass

    @staticmethod
    def fetch_hot_rank(web_id: str):
        query = f"""SELECT lt.product_id, lt.title, lt.description, lt.url,a.rank
                    FROM (
                        SELECT product_id, rank
                        FROM web_push.all_hot_items
                        WHERE web_id = '{web_id}' AND `rank` < 10
                    ) AS a
                    INNER JOIN (
                        SELECT product_id, title, description, url
                        FROM web_push.item_list
                        WHERE web_id = '{web_id}'
                    ) AS lt ON lt.product_id = a.product_id"""
        data = DBhelper('rhea1-db0', is_ssh=True).ExecuteSelect(query=query)
        output = []
        # sub_domain = self.convert_subdomain(web_id)
        for row in data:
            payload = {
                "title": row[1],
                "pagemap": {"metatags": [{"og:description": row[2]}]},
                "link": row[3],
                # "link":sub_domain + f'/avivid/product/detail/{row["product_id"]}' if sub_domain else row['url'],
                'product_id': row[0],
                "rank": row[4]
            }
            output.append(payload)
        return output

    def sort_hot_rank(self, data: list[str], web_id: str):
        rank_data = self.fetch_hot_rank(web_id=web_id)
        data = sorted(data, key=lambda x: rank_data.get(x, np.inf))
        output = {k: (v + 1) for v, k in enumerate(data)}
        return output

    def normal_search(self, word: str, web_id: str):
        """
        SELECT article_id, keyword FROM keyword_article_list WHERE web_id = '{web_id}' AND keyword = '{word64}';
        """
        query = f"""SELECT il.product_id, il.title, il.description, il.url
                    FROM web_push.item_list il
                    INNER JOIN web_push.keyword_article_list_pinyin kl ON il.product_id = kl.article_id
                    WHERE il.web_id = '{web_id}' AND kl.web_id = '{web_id}'
                          AND kl.keyword LIKE '%{word}%'
                    GROUP BY il.product_id, il.title, il.description;"""
        result = DBhelper('rheacache-db0', is_ssh=True).ExecuteSelect(query=query)
        output = []
        # sub_domain = self.convert_subdomain(web_id)
        for i, row in enumerate(result):
            payload = {
                "title": row[1],
                "pagemap": {"metatags": [{"og:description": row[2]}]},
                "link": row[3],
                # "link":sub_domain + f'/avivid/product/detail/{row["product_id"]}' if sub_domain else row['url'],
                'product_id': row[0],
                "rank": i+1
            }
            output.append(payload)
        return output

    def fuzzy_search(self, word: str, web_id: str):
        query = f"""SELECT relate_article_id, word FROM fuzzy_search 
                    WHERE web_id = '{web_id}' AND word in ('{"','".join(list(word))}');"""
        result = DBhelper('rhea1-db0',is_ssh=True).ExecuteSelect(query=query)
        data = list(set.intersection(*[set(row['relate_article_id'].split(',')) for row in result])) if len(
            result) > 0 else []
        data = self.sort_hot_rank(data=data, web_id=web_id)
        return {word: data}

    def search(self, keywords: list, web_id: str, flags: bool) -> dict:
        result = {}
        if flags:
            result = [self.normal_search(word=k, web_id=web_id) for k in keywords]
            result = list(itertools.chain(*result))
        if not result or not flags:
            result = self.fetch_hot_rank(web_id=web_id)
        return result

    def search_bar(self, keyword: str, web_id: str):
        normal_result = self.normal_search(word=keyword, web_id=web_id).get(keyword, [])
        fuzzy_result = self.fuzzy_search(word=keyword, web_id=web_id).get(keyword, [])
        fuzzy_result = list(set(fuzzy_result).difference(set(normal_result)))
        other_result = [k for k, v in self.fetch_hot_rank(web_id=web_id).items() if v in range(1, 21)]

    # normal = self.fetch_data(product_ids=normal_result, web_id=web_id)
    # fuzzy = self.fetch_data(product_ids=fuzzy_result, web_id=web_id)
    # other = self.fetch_data(product_ids=other_result, web_id=web_id)
    # todo: check the format of output

    @staticmethod
    def fetch_similarity_data(product_id: str, web_id: str) -> list:
        query = f"""SELECT similarity_product_id FROM web_push.item_list WHERE web_id = '{web_id}' and main_product_id = '{product_id}';"""
        data = DBhelper('rhea1-db0',is_ssh=True).ExecuteSelect(query=query)
        product_ids = [row['similarity_product_id'] for row in data]
        return product_ids

    def fetch_data(self, product_ids: dict, web_id: str) -> dict:
        query = f"""SELECT product_id, title, description, url FROM web_push.item_list 
                    WHERE web_id = '{web_id}' and product_id in ('{"','".join(list(product_ids.keys()))}');"""
        data = DBhelper('rhea1-db0',is_ssh=True).ExecuteSelect(query=query)

        output = []
        # sub_domain = self.convert_subdomain(web_id)
        for row in data:
            payload = {
                "title": row['title'],
                "pagemap": {"metatags": [{"og:description": row['description']}]},
                "link": row['url'],
                # "link":sub_domain + f'/avivid/product/detail/{row["product_id"]}' if sub_domain else row['url'],
                'product_id': row['product_id'],
                "rank": product_ids[row['product_id']]
            }
            output.append(payload)
        output = sorted(output, key=lambda x: x.get('rank', np.inf))
        return output

    def convert_subdomain(self, web_id: str):
        query = f"""SELECT sub_domain_url FROM web_push.AI_service_config WHERE web_id = '{web_id}';"""
        sub_domain = DBhelper('jupiter_new').ExecuteSelect(query=query)[0]['sub_domain_url']
        if sub_domain != '_':
            sub_domain = re.findall(r'((?:https?://)?(?:[\da-z.-]+)\.(?:[a-z.]{2,6})(?:[\w .-]*)*)/?', sub_domain)[0]
        return sub_domain if sub_domain else None

    def pick_duplicate(self, likr: list[dict], google: list[dict], web_id: str):
        urlencode = UrlEncode(web_id=web_id)
        likr_id = {urlencode.signature_translate(item.get('link'), web_id=web_id):i for i, item in enumerate(likr)}
        google_id = {urlencode.signature_translate(item.get('link'), web_id=web_id): i for i, item in enumerate(google)}
        print(likr_id)
        print(google_id)
        common_id = [id for id in likr_id if id in google_id]
        common = [item for i, item in enumerate(likr) if i in [likr_id[id] for id in common_id]]
        likr = [item for i, item in enumerate(likr) if i not in [likr_id[id] for id in common_id]]
        google = [item for i, item in enumerate(google) if i not in [google_id[id] for id in common_id]]
        return likr, google, common

    def likr_recommend(self, search_result: list[dict], keywords: list, flags: bool, config: dict):
        product_result = self.search(keywords=keywords, web_id=config['web_id'], flags=flags)
        common = []
        if flags:
            product_result, search_result, common = self.pick_duplicate(likr=product_result, google=search_result, web_id=config['web_id'])
        product_result = product_result[:20]
        random.shuffle(product_result)
        return product_result, search_result[:5], common