from db import DBhelper
import torch
import torch.nn.functional as F
import pandas as pd
from dotenv import load_dotenv
from AI_customer_service import ChatGPT_AVD
import re


def filter_str(desstr, restr=''):
    # 過濾除中英文及數字以外的其他字符
    desstr = str(desstr)
    res = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
    return res.sub(restr, desstr)


class TSimilarity(ChatGPT_AVD):
    def __init__(self):
        super().__init__()
        self.keyword_emb = {}
        self.big_tag_emb = {}
        self.small_tag_emb = {}
        self.data = []
        self.set_keyword = set()
        self.set_big_tag = set()
        self.set_small_tag = set()
        self.data_tensor = torch.tensor
        self.get_tag_embedding()

    def get_tag_embedding(self):
        q = f"""SELECT tag_name,tag_type,tag_embeddings FROM web_push.tag_embeddings x"""
        print('讀取embedding資料中.....')
        data = DBhelper('rhea1-db0').ExecuteSelect(q)
        for tag_name, tag_type, tag_embeddings in data:
            if tag_type == 0:
                self.keyword_emb[tag_name] = eval(tag_embeddings)
            elif tag_type == 1:
                self.big_tag_emb[tag_name] = eval(tag_embeddings)
            elif tag_type == 2:
                self.small_tag_emb[tag_name] = eval(tag_embeddings)
        q = f"""SELECT keyword,big_tag,small_tag  FROM web_push.keyword_all_tag"""
        print('讀取tag資料中.....')
        data = DBhelper('rhea1-db0').ExecuteSelect(q)
        self.data = data
        self.set_keyword = set(self.keyword_emb.keys())
        self.set_big_tag = set(self.big_tag_emb.keys())
        self.set_small_tag = set(self.small_tag_emb.keys())
        all_emb = []
        for key, b, s in data:
            self.check_emb_data(key, b, s, True)
            all_emb.append([self.keyword_emb[key], self.big_tag_emb[b], self.small_tag_emb[s]])
            self.data_tensor = torch.tensor(all_emb)

    def get_emb(self, key, types, save=False):
        emb = self.ask_gpt(message=key, model='gpt-text')
        if save:
            DBhelper.ExecuteUpdatebyChunk(
                pd.DataFrame([[key, types, str(emb)]], columns=['tag_name', 'tag_type', 'tag_embeddings']),
                db='rhea1-db0', table=f'tag_embeddings', is_ssh=False)
        return emb

    def check_emb_data(self, key, b, s, save):
        if key not in self.set_keyword:
            print(f'關鍵字"{key}"沒在資料庫')
            key_emb = self.get_emb(key, 0, save)
            self.keyword_emb[key] = key_emb
            self.set_keyword.add(key)
        if b not in self.set_big_tag:
            print(f'大標題"{b}"沒在資料庫')
            big_emb = self.get_emb(b, 1, save)
            self.big_tag_emb[b] = big_emb
            self.set_big_tag.add(b)
        if s not in self.set_small_tag:
            print(f'小標題"{s}"沒在資料庫')
            sml_emb = self.get_emb(s, 2, save)
            self.small_tag_emb[s] = sml_emb
            self.set_small_tag.add(s)

    def similarity(self, keyword, big_tag, small_tag, key_wt=1.0, bt_wt=1.0, st_wt=1.0, top_k=10, save=False):
        keyword_list = []
        keyword = filter_str(keyword)
        big_tag = filter_str(big_tag)
        small_tag = filter_str(small_tag)
        self.check_emb_data(keyword, big_tag, small_tag, save)
        cos_sim = F.cosine_similarity(self.data_tensor, torch.tensor(
            [self.keyword_emb[keyword], self.big_tag_emb[big_tag], self.small_tag_emb[small_tag]]), dim=2)
        ans = cos_sim @ torch.tensor([key_wt, bt_wt, st_wt])
        for i in ans.topk(top_k).indices:
            keyword_list.append(self.data[i])
        return keyword_list
