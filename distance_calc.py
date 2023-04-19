import pandas as pd
import numpy as np
import os

# 全家分店座標 csv
DATA_PTH = './lbs/nineyi000360.csv'
# google map url 預設格式
GOOGLE_MAP_FORMAT = 'https://www.google.com/maps/search/?api=1&query=全家便利商店+'


class StoreDistanceEvaluator:
    data_pth = os.path.join(os.path.dirname(__file__), DATA_PTH)

    def __init__(self):
        self.df = pd.read_csv(self.data_pth)

    def get_nearest_store(self, input_coor: tuple) -> list[dict]:
        '''
        :param input_coor: 輸入經緯度, ex: (121.52177, 25.06743)
        :return:{"htmlTitle": STORE_NAME
				"pagemap":{"metatags":[{"og:description": STORE_ADDRESS }]},
				"link":  STORE_URL  }
        '''
        calc_df = self.df.copy()
        calc_df['距離'] = calc_df['座標'].apply(lambda x: self.distance_calc(input_coor, eval(x)))

        # min distance df
        calc_df = calc_df[calc_df['距離'] == calc_df['距離'].min()]
        min_dis_store = calc_df['店名'].values[0]

        res = [{"htmlTitle": '全家便利商店-'+min_dis_store,
              "pagemap": {"metatags": [{"og:description": calc_df['地址'].values[0]}]},
              "link": GOOGLE_MAP_FORMAT + min_dis_store}]
        return res

    @staticmethod
    def distance_calc(coor_1, coor_2):
        d_x = (coor_1[0] - coor_2[0])**2
        d_y = (coor_1[1] - coor_2[1])**2
        return np.sqrt(d_x + d_y)


if __name__ == '__main__':
    evaluator = StoreDistanceEvaluator()
    test_coor = (121.52182045752879, 25.067559530144354)    # 大同大學座標

    nearest_store = evaluator.get_nearest_store(test_coor)
    print(nearest_store)
