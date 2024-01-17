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
        output = {}
        query = f"""SELECT product_id, rank FROM web_push.all_hot_items WHERE web_id = '{web_id}';"""
        data = DBhelper('rhea1-db0',is_ssh=True).ExecuteSelect(query=query)
        for row in data:
            output[row['product_id']] = int(row['rank'])
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
        word64 = base64.b64encode(word.encode('utf-8')).decode('utf-8')
        print(f'wrod{word}')
        query = f"""SELECT il.product_id FROM web_push.item_list il left join 
		            (SELECT DISTINCT article_id FROM web_push.keyword_article_list 
		                WHERE web_id = '{web_id}' AND keyword like '%{word64}%') kl 
		            on il.product_id = kl.article_id
		            WHERE il.web_id = '{web_id}' AND il.title like '%{word}%';"""
        result = DBhelper('rhea1-db0',is_ssh=True).ExecuteSelect(query=query)
        data = [row['product_id'] for row in result] if len(result) > 0 else []
        return {word: data}

    def fuzzy_search(self, word: str, web_id: str):
        query = f"""SELECT relate_article_id, word FROM fuzzy_search 
                    WHERE web_id = '{web_id}' AND word in ('{"','".join(list(word))}');"""
        result = DBhelper('rhea1-db0',is_ssh=True).ExecuteSelect(query=query)
        data = list(set.intersection(*[set(row['relate_article_id'].split(',')) for row in result])) if len(
            result) > 0 else []
        data = self.sort_hot_rank(data=data, web_id=web_id)
        return {word: data}

    def search(self, keywords: list, web_id: str, flags: dict) -> dict:
        result = [self.normal_search(word=k, web_id=web_id).get(k, []) for k in keywords]
        result = list(itertools.chain(*result))
        print(result)
        if len(result):
            output = self.sort_hot_rank(data=result, web_id=web_id)
            flags['product'] = True
        else:
            result = self.fetch_hot_rank(web_id=web_id)
            output = {k: v for k, v in result.items() if v in range(1, 11)}
        return output

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
        likr_id = {urlencode.signature_translate(item.get('link'), web_id=web_id): i for i, item in enumerate(likr)}
        google_id = {urlencode.signature_translate(item.get('link'), web_id=web_id): i for i, item in enumerate(google)}
        common_id = [id for id in likr_id if id in google_id]
        common = [item for i, item in enumerate(likr) if i in [likr_id[id] for id in common_id]]
        likr = [item for i, item in enumerate(likr) if i not in [likr_id[id] for id in common_id]]
        google = [item for i, item in enumerate(google) if i not in [google_id[id] for id in common_id]]
        return likr, google, common

    def likr_recommend(self, search_result: list[dict], keywords: list, flags: dict, config: dict):
        product_ids = self.search(keywords=keywords, web_id=config['web_id'], flags=flags)
        product_result = self.fetch_data(product_ids=product_ids, web_id=config['web_id'])
        product_result, search_result, common = self.pick_duplicate(likr=product_result[:30], google=search_result, web_id=config['web_id'])
        product_result = product_result[:20]
        random.shuffle(product_result)
        return product_result, search_result[:5], common, flags
        # result = common if common else []
        # if flags.get('product'):
        #     # if flags.get('uuid') and not flags.get('is_hot'):
        #     #     product_result = product_result[:10]
        #     #     random.shuffle(product_result)
        #     #     result += (product_result[:2] + search_result[:1] + product_result[2:3] + search_result[1:] + product_result[3:])
        #     # else:
        #     #     product_result = product_result[:20]
        #     #     random.shuffle(product_result)
        #     #     result += (product_result[:2] + search_result[:1] + product_result[2:3] + search_result[1:] + product_result[3:])
        #
        # else:
        #     # if flags.get('uuid'):
        #     #     product_result = [product_result[0]]
        #     #     similar_ids = self.fetch_similarity_data(product_id=product_result[0]['product_id'], web_id=config['web_id'])
        #     #     similarity_products = self.sort_hot_rank(data=similar_ids, web_id=config['web_id'])
        #     #     similarity_products = self.fetch_data(product_ids=similarity_products, web_id=config['web_id'])
        #     #     similarity_products = similarity_products[:10]
        #     #     random.shuffle(similarity_products)
        #     #     product_result += similarity_products
        #     #     result += (product_result[:2] + search_result[:1] + product_result[2:3] + search_result[1:] + product_result[3:])
        #     # else:
        #     #     product_result = product_result[:20]
        #     #     random.shuffle(product_result)
        #     #     result += (product_result[:2] + search_result[:1] + product_result[2:3] + search_result[1:] + product_result[3:])
