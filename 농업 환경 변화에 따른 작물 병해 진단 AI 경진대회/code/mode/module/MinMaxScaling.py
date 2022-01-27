from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


class MinMaxScaling():
    def __init__(self):
        # 분석에 사용할 feature 선택
        self.csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고', 
                        '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']

        self.csv_files = sorted(glob('data/train/*/*.csv'))

        self.temp_csv = pd.read_csv(self.csv_files[0])[self.csv_features]
        self.max_arr, self.min_arr = self.temp_csv.max().to_numpy(), self.temp_csv.min().to_numpy()

        # feature 별 최대값, 최솟값 계산
        for csv in tqdm(self.csv_files[1:]):
            self.temp_csv = pd.read_csv(csv)[self.csv_features]
            self.temp_max, self.temp_min = self.temp_csv.max().to_numpy(), self.temp_csv.min().to_numpy()
            self.max_arr = np.max([self.max_arr,self.temp_max], axis=0)
            self.min_arr = np.min([self.min_arr,self.temp_min], axis=0)

        # feature 별 최대값, 최솟값 dictionary 생성
        self.csv_feature_dict = {self.csv_features[i]:[self.min_arr[i], self.max_arr[i]] for i in range(len(self.csv_features))}

