import pandas as pd
import numpy as np

import category_encoders as ce

from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

class Data:
    def __init__(self, features, labels):
        self.x = features
        self.y = labels
        self.classes_ = {'AbNormal': 1, 'Normal': 0}
        if self.y is not None:
            self.y = self.y.replace(self.classes_)

    def get_num_cols(self):
        return self.x.select_dtypes(include=[np.number]).columns
    
    def get_cat_cols(self):
        return self.x.select_dtypes(exclude=[np.number]).columns
    
    def get_cat_idxs(self):
        return [i for i, col in enumerate(self.x.columns) if col in self.get_cat_cols()]
    
    def get_cols(self, name):
        if isinstance(name, str):
            result = [col for col in self.x.columns if name in col]
        elif isinstance(name, list):
            result = [col for col in self.x.columns if all(item in col for item in name)]
        assert len(result) > 0
        return result
    
    def get(self, name):
        return self.x[self.get_cols(name)]
        
class LoadDataset:
    def __init__(self, 
                 train_path,
                 test_path,
                 valid_path = None,
                 drop_cols = None, 
                 seed=42):
        self.seed=seed
        
        def find_unique_columns(df):
            unique_domain_columns = []

            for column in df.columns:
                unique_values = df[column].dropna().unique()
                if len(unique_values) <= 1:
                    unique_domain_columns.append(column)

            return unique_domain_columns

        self.train = Data(*self._load(train_path, drop_cols=drop_cols))
        self.test = Data(*self._load(test_path, drop_cols=drop_cols, is_test=True))
        
        unique_columns = find_unique_columns(self.train.x)
        self.train.x = self.train.x.drop(columns=unique_columns)
        self.test.x = self.test.x.drop(columns=unique_columns)
        
        if valid_path is not None:
            self.valid = Data(*self._load(valid_path, drop_cols=drop_cols))
            self.valid.x = self.valid.x.drop(columns=unique_columns)
        else:
            self.valid = None

        self.stage1 = False # (필수)  결측치 처리
        self.stage2 = False # (필수x) Feature Engineering   -- todo
        self.stage3 = False # (필수x) Sampling
        self.stage4 = False # (필수x) Scaling
        self.stage5 = False # (필수x) Encoding

        self.nan_processing_fc = {
            "FILL MEDIAN FREQ": self._fill_with_median, 
            "FILL MOST FREQ": self._fill_with_mfv,
            "DROP COLUMN": self._drop_column,
            "DROP ROW": self._drop_row
        }
        
        self.feature_fc = {
            "PCA": self._pca,
            "PCA PER PROCESS": self._pca_per_process,
        }

        self.sampling_fc = {
            "OFF": self._sampling_with_off,
            "SMOTE": self._sampling_with_smote, 
            "UNDER": self._sampling_with_under
        }

        self.scaling_fc = {
            "STANDARD": self._scaling_with_standard,
            "MINMAX": self._scaling_with_minmax,
        }
        
        self.encoding_fc = {
            "ONEHOT": self._encoding_with_onehot,
            "LABEL": self._encoding_with_label,
            "TARGET": self._encoding_with_target
        }
        # 참조: https://contrib.scikit-learn.org/category_encoders/#

    def _load(self, path, drop_cols = None, is_test = False):
        data = pd.read_csv(path)
        data = data.sort_index().reset_index(drop=True)
        if not is_test:
            data = data.sort_values(by='Collect Date_Dam')

        labels = None
        if not is_test:
            labels = data["target"]
            data = data.drop("target", axis=1)

        features = data.copy()
        features.columns = features.columns.str.replace('.', '_')
        if drop_cols is not None:
            features = features.drop(drop_cols, axis=1)
        return features, labels
    
    def _is_nan(self):
        if self.train.x.isnull().sum().sum() == 0:
            if (self.valid is not None) and (self.valid.x.isnull().sum().sum() == 0):
                return False
            elif self.valid is None:
                return False
        return True

    def nan_processing(self, method_dict: dict):
        """
        example
        : method_dict = {'ALL': 'FILL MOST FREQ'}
        : method_dict = {'Equipment_Dam': 'FILL MEDIAN FREQ', 
                         'OTHERS': 'DROP COLUMN'}

        ※ fc 추가하는 방법: 다음 조건의 함수를 작성 후 self.nan_processing_fc 딕셔너리에 추가
        - input: (채울)column name
        - fc: 내부에서 self.train.x, col_name을 사용해 df직접 변경
        """
        target_list = ["ALL", "OTHERS"]
        target_list.extend(self.train.x.columns)
        assert len(set(method_dict.keys()) - set(target_list)) == 0, "Target column column does not exist"

        print("Missing Value Processing...\n")
        print("(before processing)The number of nan")
        print(f"- train: {self.train.x.isna().sum().sum()}")
        if self.valid is not None:
            print(f"- valid: {self.valid.x.isna().sum().sum()}\n")
        
        check = {col:False for col in self.train.x.columns}
        for i, (col, method) in enumerate(method_dict.items()):
            assert method in self.nan_processing_fc.keys(), f"Processing method can only be used in one of the following methods: {self.nan_processing_fc.keys()}"
            print(f"  {i+1}. Processing <{col}> Columns with '{method}'")
            # 모든 Column에 해당 method 적용
            if col == "ALL":
                for col_name in self.train.x.columns:
                    self.nan_processing_fc[method](col_name)
                break
            # 특정 Column을 제외한 나머지 모든 Column에 method 적용
            elif col == "OTHERS":
                for col_name in self.train.x.columns:
                    if not check[col_name]:
                        self.nan_processing_fc[method](col_name)
                break
            # 특정 Column에 해당 method 적용
            else:
                self.nan_processing_fc[method](col)
                check[col] = True

        self.stage1 = not self._is_nan()
        print("\nFinish!")
        print("(after processing)The number of nan")
        print(f"- train: {self.train.x.isna().sum().sum()}")
        if self.valid is not None:
            print(f"- valid: {self.valid.x.isna().sum().sum()}")
        print("\n====================")

    def feature_engineering(self, method:str, arg: dict):
        """
        method: 실행하고자 하는 함수
        arg: 해당 함수에 줄 Parameter들(dictionary 형태)

        ※ fc 추가하는 방법: 함수 작성후 self.feature_fc에 추가
        - input은 자유롭게 설정
        - 사용시에는 dictionary형태로 parameter전달 필요
        """
        assert self.stage1, "Please process the missing values first"
        self.stage2 = True

        print(f"Feature Engineering with {method}...\n")
        self.feature_fc[method](**arg)
        print("Finish!")
        print("\n====================")
    
    def sampling(self, method: str):
        """
        stage1(결측치 처리)실행 후 실행해야 함

        ※ fc 추가하는 방법: 다음 조건의 함수를 작성 후 self.sampling_fc 딕셔너리에 추가
        - input: (sampling전)train_features, (sampling전)train_labels
        - output: (sampling한)train_features, (sampling한)train_labels
        """
        assert self.stage1, "Please process the missing values first"
        assert method in self.sampling_fc.keys(), f"Sampling method can only be used in one of the following methods: {self.sampling_fc.keys()}"
        self.stage3 = True
        
        print(f"Sampling with {method}...\n")
        print("(before sampling)Value count")
        print(self.train.y.value_counts(), "\n")

        self.train.x, self.train.y = self.sampling_fc[method](self.train.x, self.train.y)

        print("Finish!")
        print("(after sampling)Value count")
        print(self.train.y.value_counts())
        print("\n====================")
    
    def scaling(self, method_dict: dict):
        """
        stage1(결측치 처리), stage3(Sampling)실행 후 실행해야 함
        Sampling을 하고싶지 않은 경우 "OFF"후 사용

        ※ fc 추가하는 방법: 다음 조건의 함수를 작성 후 self.scaling_fc 딕셔너리에 추가
        - input: (numeric)column_name
        - fc: 내부에서 self.train.x, col_name을 사용해 df직접 변경
        """
        assert self.stage1 and self.stage3, "Please process the missing values and sampling first"
        target_list = ["ALL", "OTHERS"]
        target_list.extend(self.train.x.columns)
        assert len(set(method_dict.keys()) - set(target_list)) == 0, "Target column does not exist"
        
        print("Scaling our dataset...\n")
        
        num = 0
        check = {col:False for col in self.train.x.columns}
        for i, (col, method) in enumerate(method_dict.items()):
            assert method in self.scaling_fc.keys(), f"Scaling method can only be used in one of the following methods: {self.scaling_fc.keys()}"
            print(f"  {i+1}. Scaling <{col}> Columns with '{method}'")
            # 모든 Column에 해당 method 적용
            if col == "ALL":
                for col_name in self.train.x.columns:
                    if col_name in self.train.get_num_cols():
                        self.scaling_fc[method](col_name)
                        num += 1
                break
            # 특정 Column을 제외한 나머지 모든 Column에 method 적용
            elif col == "OTHERS":
                for col_name in self.train.x.columns:
                    if not check[col_name] and (col_name in self.train.get_num_cols()):
                        self.scaling_fc[method](col_name)
                        num += 1
                break
            # 특정 Column에 해당 method 적용
            else:
                self.scaling_fc[method](col)
                check[col] = True
                num += 1

        print(f"\nFinish! (the number of columns: {num})")
        self.stage4 = True
        print("\n====================")

    def encoding(self, method_dict: dict):
        """
        stage1(결측치 처리), stage3(Sampling)실행 후 실행해야 함
        Sampling을 하고싶지 않은 경우 "OFF"후 사용

        ※ fc 추가하는 방법: 다음 조건의 함수를 작성 후 self.encoding_fc 딕셔너리에 추가
        - input: (categorical)column_name
        - fc: 내부에서 self.train.x, col_name을 사용해 df직접 변경
        """
        assert self.stage1 and self.stage3, "Please process the missing values and sampling first"
        target_list = ["ALL", "OTHERS"]
        target_list.extend(self.train.x.columns)
        assert len(set(method_dict.keys()) - set(target_list)) == 0, "Target column does not exist"
        
        print("Encoding our dataset...\n")
        
        num = 0
        check = {col:False for col in self.train.x.columns}
        for i, (col, method) in enumerate(method_dict.items()):
            assert method in self.encoding_fc.keys(), f"Encoding method can only be used in one of the following methods: {self.encoding_fc.keys()}"
            print(f"  {i+1}. Encoding <{col}> Columns with '{method}'")
            # 모든 Column에 해당 method 적용
            if col == "ALL":
                for col_name in self.train.x.columns:
                    if col_name in self.train.get_cat_cols():
                        num += 1
                        self.encoding_fc[method](col_name)
                break
            # 특정 Column을 제외한 나머지 모든 Column에 method 적용
            elif col == "OTHERS":
                for col_name in self.train.x.columns:
                    # 만약 col_name이 check에 있지 않다면 method_dict 중간에 One-Hot Encoding 때문에 새로 생긴 Column임
                    if col_name not in check.keys():
                        continue
                    if not check[col_name] and (col_name in self.train.get_cat_cols()):
                        num += 1
                        self.encoding_fc[method](col_name)
                break
            # 특정 Column에 해당 method 적용
            else:
                self.encoding_fc[method](col)
                check[col] = True
                num += 1

        print(f"\nFinish! (the number of columns: {num})")
        self.stage4 = True
        print("\n====================")

    ##############################
    #                            #
    # Feature Engineering METHOD #
    #                            #
    ##############################
    
    # PCA
    def _pca(self, n_components):
        print(f"  Execute PCA with {n_components} components...")
        if len(self.train.get_cat_cols()):
            print(f"  Warning: We only transform data with numeric value, now we have {len(self.train.get_cat_cols())} categorical column!")
        pca = PCA(n_components=n_components, random_state=self.seed)
        target_columns = self.train.get_num_cols()
        pca_result = pca.fit_transform(self.train.x[target_columns])
        pca_result_test = pca.transform(self.test.x[target_columns])
        print(f"\n  Explained Variance Ratio: {sum(pca.explained_variance_ratio_)}")
        print(f"      ※ details = {pca.explained_variance_ratio_}\n")

        self.train.x = pd.DataFrame(data=pca_result, columns = [f"PC{i}" for i in range(1, n_components+1)])
        self.test.x = pd.DataFrame(data=pca_result_test, columns = [f"PC{i}" for i in range(1, n_components+1)])
        
        if self.valid is not None:
            pca_result_valid = pca.transform(self.valid.x[target_columns])
            self.valid.x = pd.DataFrame(data=pca_result_valid, columns = [f"PC{i}" for i in range(1, n_components+1)])
    
    def _pca_per_process(self, min_proba):
        assert 0 <= min_proba < 1
        print(f"  Execute PCA PER PROCESS with minimum probability {min_proba}")
        new_train_x = pd.DataFrame()
        new_test_x = pd.DataFrame()
        new_valid_x = pd.DataFrame()
        for name in ["Dam", "Fill1", "Fill2", "AutoClave"]:
            cols = sorted(list(set(self.train.get_cols(name)) & set(self.train.get_num_cols())))
            assert len(cols) > 0
            print(f"\n  Stage '{name}': the number of target cols is '{len(cols)}'")
            i = 1
            while True:
                pca = PCA(n_components=i, random_state=self.seed)
                pca_result = pca.fit_transform(self.train.x[cols])
                if sum(pca.explained_variance_ratio_) >= min_proba:
                    break
                i+=1
            pca_result_test = pca.transform(self.test.x[cols])
            print(f"  => {i}_components with Explained Variance Ratio '{sum(pca.explained_variance_ratio_)}'")
            pca_result = pd.DataFrame(data=pca_result, columns = [f"PC{j}_{name}" for j in range(1, i+1)])
            pca_result_test = pd.DataFrame(data=pca_result_test, columns = [f"PC{j}_{name}" for j in range(1, i+1)])

            new_train_x = pd.concat([new_train_x, pca_result], axis=1)
            new_test_x = pd.concat([new_test_x, pca_result_test], axis=1)
            if self.valid is not None:
                pca_result_valid = pca.transform(self.valid.x[cols])
                pca_result_valid = pd.DataFrame(data=pca_result_valid, columns = [f"PC{j}_{name}" for j in range(1, i+1)])
                new_valid_x = pd.concat([new_valid_x, pca_result_valid], axis=1)

        self.train.x = new_train_x
        self.test.x = new_test_x
        if self.valid is not None:
            self.valid.x = new_valid_x

    ####################
    #                  #
    # 결측치처리 METHOD #
    #                  #
    ####################
    
    # 결측치 중앙값(빈도수) 대체
    def _fill_with_median(self, col_name):
        # train과 valid에 모두 null이 없는 경우를 제외하면 모두 실행
        if not self.train.x[col_name].isnull().any():
            if (self.valid is not None) and (not self.valid.x[col_name].isnull().any()):
                return
            elif self.valid is None:
                return
        
        value_counts = self.train.x[col_name].value_counts()
        sorted_value_counts = value_counts.sort_values(ascending=False)
        median_index = len(sorted_value_counts) // 2
        median_value = sorted_value_counts.index[median_index]

        self.train.x[col_name] = self.train.x[col_name].fillna(median_value)
        self.test.x[col_name] = self.test.x[col_name].fillna(median_value)
        if self.valid is not None:
            self.valid.x[col_name] = self.valid.x[col_name].fillna(median_value)
        
    # 결측치 최빈값 대체
    def _fill_with_mfv(self, col_name):
        # train과 valid에 모두 null이 없는 경우를 제외하면 모두 실행
        if not self.train.x[col_name].isnull().any():
            if (self.valid is not None) and (not self.valid.x[col_name].isnull().any()):
                return
            elif self.valid is None:
                return
        
        most_frequent_value = self.train.x[col_name].mode().iloc[0]
        
        self.train.x[col_name] = self.train.x[col_name].fillna(most_frequent_value)
        self.test.x[col_name] = self.test.x[col_name].fillna(most_frequent_value)
        if self.valid is not None:
            self.valid.x[col_name] = self.valid.x[col_name].fillna(most_frequent_value)

    # 결측치가 존재하는 Column제거
    def _drop_column(self, col_name):
        # train과 valid에 모두 null이 없는 경우를 제외하면 모두 실행
        if not self.train.x[col_name].isnull().any():
            if (self.valid is not None) and (not self.valid.x[col_name].isnull().any()):
                return
            elif self.valid is None:
                return
        
        self.train.x = self.train.x.drop(columns=[col_name])
        self.test.x = self.test.x.drop(columns=[col_name])

        if self.valid is not None:
            self.valid.x = self.valid.x.drop(columns=[col_name])
    
    # 결측치가 존재하는 row제거
    def _drop_row(self, col_name):
        # test에는 해당 Column에 null이 존재하면 안됨 
        # assert not self.test.x[col_name].isnull().any() # !test data존재하면 주석 해제!
        idxs = self.train.x[self.train.x[col_name].isnull()].index
        self.train.x = self.train.x.drop(idxs)
        self.train.y = self.train.y.drop(idxs)

        if self.valid is not None:    
           idxs_ = self.valid.x[self.valid.x[col_name].isnull()].index
           self.valid.x = self.valid.x.drop(idxs_)
           self.valid.y = self.valid.y.drop(idxs_)

    ###################
    #                 #
    # SAMPLING METHOD #
    #                 #
    ###################

    # Sampling 해제
    def _sampling_with_off(self, x, y):
        return x, y
    
    # SMOTE Sampling
    def _sampling_with_smote(self, x, y):
        smote_nc = SMOTENC(categorical_features=self.train.get_cat_idxs(), random_state=self.seed)
        x, y = smote_nc.fit_resample(x, y)
        return x, y
    
    # Under Sampling
    def _sampling_with_under(self, x, y):
        under_sampler = RandomUnderSampler(sampling_strategy='majority', random_state=self.seed)
        x, y = under_sampler.fit_resample(x, y)
        return x, y
    
    ##################
    #                #
    # SCALING METHOD #
    #                #
    ##################

    # Standard Scaler
    def _scaling_with_standard(self, col_name):
        assert col_name in self.train.get_num_cols(), f"Column {col_name} is not in the numeric columns"
        scaler = StandardScaler()
        self.train.x[[col_name]] = scaler.fit_transform(self.train.x[[col_name]])   
        self.test.x[[col_name]] = scaler.transform(self.test.x[[col_name]])

        if self.valid is not None:
            self.valid.x[[col_name]] = scaler.transform(self.valid.x[[col_name]])

    # MinMax Scaler
    def _scaling_with_minmax(self, col_name):
        assert col_name in self.train.get_num_cols(), f"Column {col_name} is not in the numeric columns"
        scaler = MinMaxScaler()
        self.train.x[[col_name]] = scaler.fit_transform(self.train.x[[col_name]])   
        self.test.x[[col_name]] = scaler.transform(self.test.x[[col_name]])

        if self.valid is not None:
            self.valid.x[[col_name]] = scaler.transform(self.valid.x[[col_name]])

    ###################
    #                 #
    # ENCODING METHOD #
    #                 #
    ###################

    # OneHot Encoder
    def _encoding_with_onehot(self, col_name):
        assert col_name in self.train.get_cat_cols(), f"Column {col_name} is not in the categorical columns"
        encoder = ce.OneHotEncoder(cols=[col_name],
                                   handle_unknown='value',
                                   handle_missing='value',
                                   use_cat_names=True)
        
        encoded_train = encoder.fit_transform(self.train.x[[col_name]])
        encoded_test = encoder.transform(self.test.x[[col_name]])

        # Remove original column and add encoded columns to train and test
        self.train.x = pd.concat([self.train.x.drop(col_name, axis=1), encoded_train], axis=1)
        self.test.x = pd.concat([self.test.x.drop(col_name, axis=1), encoded_test], axis=1)

        if self.valid is not None:
            encoded_valid = encoder.transform(self.valid.x[[col_name]])
            self.valid.x = pd.concat([self.valid.x.drop(col_name, axis=1), encoded_valid], axis=1)

    # Label Encoder
    def _encoding_with_label(self, col_name):
        assert col_name in self.train.get_cat_cols(), f"Column {col_name} is not in the categorical columns"
        encoder = ce.OrdinalEncoder(cols=[col_name],
                                 handle_unknown='value',   # 알 수 없는 범주를 특정 값으로 인코딩
                                 handle_missing='value')   # 결측값을 특정 값으로 인코딩

        self.train.x[col_name] = encoder.fit_transform(self.train.x[[col_name]])[col_name]
        self.test.x[col_name] = encoder.transform(self.test.x[[col_name]])[col_name]
        
        if self.valid is not None:
            self.valid.x[col_name] = encoder.transform(self.valid.x[[col_name]])[col_name]

    # Target Encoder
    def _encoding_with_target(self, col_name):
        assert col_name in self.train.get_cat_cols(), f"Column {col_name} is not in the categorical columns"
        encoder = ce.TargetEncoder(cols=[col_name],
                                   handle_unknown='value',
                                   handle_missing='value')
        self.train.x[[col_name]] = encoder.fit_transform(self.train.x[[col_name]], self.train.y)
        self.test.x[[col_name]] = encoder.transform(self.test.x[[col_name]])
        
        if self.valid is not None:
            self.valid.x[[col_name]] = encoder.transform(self.valid.x[[col_name]])
