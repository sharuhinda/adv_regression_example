"""
File contains custom transformation classes made for House Prices Advanced Regression
dataset from Kaggle 
"""

from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.exceptions import NotFittedError

from sklearn.base import BaseEstimator, TransformerMixin # for creating custom transformers based on sklearn linrary
from sklearn.pipeline import Pipeline

#=====================================================

class DFDropColumns(BaseEstimator, TransformerMixin):
    """
    Class to drop columns that are redundant for the model
    """

    def __init__(self, columns=None) -> None: # cols is a list containig column names to drop
        #super().__init__()
        self.columns = columns


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if (self.columns is not None) and (len(self.columns) > 0):
            return X.drop(columns=[col for col in self.columns if col in X.columns])
        return X

#=====================================================

class DFApplyFuncToColumns(BaseEstimator, TransformerMixin):
    """
    Apply different funcs to columns
    Makes possible columns' type and more complex transformations
    """

    def __init__(self, columns=None) -> None:
        """
        columns arg should be the dict {'column_name': func_to_apply}
        """
        self.columns = columns
        pass


    def fit(self, X, y=None):
        return self

    
    def transform(self, X, y=None):
        if self.columns is None or len(self.columns) == 0:
            return X

        X_transformed = X.copy()
        for c, func in self.columns.items():
            X_transformed[c] = X_transformed[c].apply(func)

        return X_transformed

#=====================================================

class DFCreateAdditionalFeatures(BaseEstimator, TransformerMixin):
    """
    Transformation class to create 3 additional boolean features for basement, garage and house remodelling
    """
    
    def __init__(self, create_bsmt=True, create_garage=True, create_remodeled=True) -> None:
        self.create_bsmt = create_bsmt
        self.create_garage = create_garage
        self.create_remodeled = create_remodeled
        pass


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        X_transformed = X.copy()
        if self.create_bsmt:
            X_transformed['HasBsmt'] = X['BsmtQual'].notna()
        if self.create_garage:
            X_transformed['HasGarage'] = X['GarageType'].notna()
        if self.create_remodeled:
            X_transformed['Remodeled'] = X['YearRemodAdd'] > X['YearBuilt']
        return X_transformed

#=====================================================

class DFValuesReplacer(BaseEstimator, TransformerMixin):
    """
    """
    def __init__(self, replaces=None) -> None:
        """
        replaces should be the dict {'column_name': {value_to_replace: value_to_replace_with}}
        """
        self.replaces = replaces # replaces should be in form {'column1': {'value_to_find': 'value_to_replace_with'}, ...}
        pass


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if self.replaces is None or self.replaces == {}:
            return X

        X_transformed = X.copy()
        #for c, val in X_transformed.columns:
        X_transformed = X_transformed.replace(self.replaces)
        return X_transformed

#=====================================================

class DFSimpleNanReplacer(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self, columns, default_values) -> None:
        self.columns = columns
        self.default_values = default_values
        pass


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        X_transformed = X.copy()

        replacements = {c: self.default_values.get(c, np.nan) for c in self.columns}
        X_transformed.fillna(replacements, inplace=True)

        return X_transformed

#=====================================================

class DFSimpleImputer(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self, col_strategy=None) -> None:
        """
        col_strategy arg should be the dict {'column_name': 'strategy'}
            where strategy = 'most_frequent'|'least_frequent'
        """
        self.col_strategy = col_strategy
        pass


    def fit(self, X, y=None):
        self.imputers_ = {}
        for c, strategy in self.col_strategy.items():
            if c in X.columns:
                freq = X[c].value_counts()
                if len(freq) == 0:
                    self.imputers_[c] = None
                else:
                    if strategy == 'most_frequent':
                        self.imputers_[c] = freq.index[0]
                    elif strategy == 'least_frequent':
                        self.imputers_[c] = freq.index[-1]
                    else:
                        raise TypeError(f'Wrong strategy: {strategy}')
            else:
                self.imputers_[c] = None
        return self


    def transform(self, X, y=None):
        X_transformed = X.copy()
        for c, value in self.imputers_.items():
            if value is not None:
                m = X_transformed[c].isna()
                X_transformed.loc[m, c] = value
        return X_transformed

#=====================================================

class DFChainedNanReplacer(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self, chained_features, default_values) -> None:
        self.chained_features = chained_features
        self.default_values = default_values
        pass


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        X_transformed = X.copy()

        for lead_col, chain in self.chained_features.items():
            m = X_transformed[lead_col].isna()
            for c in chain:
                X_transformed.loc[m, c] = self.default_values.get(c, np.nan)
            X_transformed.loc[m, lead_col] = self.default_values.get(lead_col, np.nan)

        return X_transformed

#=====================================================

class DFReplaceMeaningfulNANs(BaseEstimator, TransformerMixin):
    """
    Transformation class to replace NaN values where they are meaningful and not missed data
    """

    single_features = ['Alley', 'Fence', 'Electrical']
    basement_features = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
    garage_features = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']
    feature_pairs = [
        ['FireplaceQu', 'Fireplaces'],
        ['PoolQC', 'PoolArea'],
        ['MiscFeature', 'MiscVal'],
        ['MasVnrType', 'MasVnrArea']
    ]
    """
    # Left from 1st version of class. It's more convenient to pass values directly. This gives full control.
    cols_nans = {  # Default values to replace NaNs with
        'Street': 'Abs',
        'Alley': 'Abs',
        'Fence': 'Abs',
        'Electrical': 'Abs',
        # Basement section
        'BsmtQual': 'Abs', 'BsmtCond': 'Abs', 'BsmtExposure': 'Abs',
        'BsmtFinType1': 'Abs', 'BsmtFinSF1': 0, 'BsmtFinType2': 'Abs',
        'BsmtFinSF2': 0, 'BsmtUnfSF': 0, 'TotalBsmtSF': 0, 'BsmtFullBath': 0, 'BsmtHalfBath': 0,
        # Garage section
        'GarageType': 'Abs', 'GarageYrBlt': 0, 'GarageFinish': 'Abs', 'GarageCars': 0,
        'GarageArea': 0, 'GarageQual': 'Abs', 'GarageCond': 'Abs',
        # Other sections
        'FireplaceQu': 'Abs', 'Fireplaces': 0,
        'PoolQC': 'Abs', 'PoolArea': 0,
        'MiscFeature': 'Abs', 'MiscVal': 0,
        'MasVnrType': 'Abs', 'MasVnrArea': 0,
    }
    """

    def __init__(self, default_nans) -> None: # cols_nans is a dictionary containig column names and default values to replace NaNs with if default values not suit
        #super().__init__()
        if default_nans is not None:
            self.default_nans = default_nans


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        X_transformed = X.copy()
        
        m = X_transformed['BsmtQual'].isna()
        for column in self.basement_features:
            X_transformed.loc[m, column] = self.cols_nans.get(column, np.nan)

        m = X_transformed['GarageType'].isna()
        for column in self.garage_features:
            X_transformed.loc[m, column] = self.cols_nans.get(column, np.nan)

        if self.feature_pairs is not None and len(self.feature_pairs) > 0:
            for pair in self.feature_pairs:
                m = X_transformed[pair[0]].isna()
                for column in pair:
                    X_transformed.loc[m, column] = self.cols_nans.get(column, np.nan)
                
        return X_transformed


#=====================================================

class DFJoinDates(BaseEstimator, TransformerMixin):
    """
    Transformation class to join MoSold and YrSold features and convert them to datetime value or to days from sold values
    """

    def __init__(self, day_col=None, month_col=None, year_col=None, calc_period_to=None, new_column_name=None, drop_originals=False) -> None:
        """
        Calculates days from house sold based on given columns
        Parameters:
            day_col = <day_column_name>: str
            month_col = <month_column_name>: str
            year_col = <year_column_name>: str
            calc_period_to = <target_date>: datetime
            [new_column_name = <new_column_name>: str]
            [drop_originals = True|False]
        """
        self.day_col = day_col
        self.month_col = month_col
        self.year_col = year_col
        self.calc_period_to = calc_period_to
        if new_column_name is None:
            if calc_period_to is None:
                self.new_column_name = 'DateCreated'
            else:
                self.new_column_name = 'PeriodCreated'
        else:
            self.new_column_name = new_column_name
        self.drop_originals = drop_originals


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if self.year_col is None:
            return X

        X_transformed = X.copy()
        if self.day_col is None:
            days = pd.Series(['1']*len(X), index=X.index)
        else:
            days = X[self.day_col].apply(str)
        
        if self.month_col is None:
            months = pd.Series(['1']*len(X), index=X.index)
        else:
            months = X[self.month_col].apply(str)
        
        combined_date_str = months + '/' + days + '/' + X[self.year_col].apply(str)
        if self.calc_period_to is None:
            X_transformed[self.new_column_name] = pd.to_datetime(combined_date_str)
        else:
            # [TODO] Need code here to calculate difference between dates in days
            X_transformed[self.new_column_name] = (self.calc_period_to - pd.to_datetime(combined_date_str)).apply(lambda x: x.days)
        if self.drop_originals:
            X_transformed.drop(columns=[x for x in [self.day_col, self.month_col, self.year_col] if x is not None], inplace=True)

        return X_transformed


#=====================================================

class DFCalcAge(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns=None, calc_age_to=None, drop_originals=False) -> None:
        """
        Creates new columns with calculated period to requested year
        Parameters:
            'columns' arg is a dict {'original_col_name': 'age_column_name'}
            'calc_age_to' = <year>: int
            'drop_originals' = True|False
        """
        self.age_columns = columns
        self.calc_age_to = calc_age_to
        self.drop_originals = drop_originals


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        X_transformed = X.copy()
        if (self.age_columns is not None) and (self.calc_age_to is not None):
            for column in self.age_columns.keys():
                X_transformed[self.age_columns[column]] = self.calc_age_to - X_transformed[column]
            if self.drop_originals:
                X_transformed = X_transformed.drop(columns=list(self.age_columns.keys()))
        return X_transformed

#=====================================================

class DFConvertToNumpy(BaseEstimator, TransformerMixin):
    """
    Final transformer from pandas dataframe to numpy array
    """

    def __init__(self) -> None:
        #super().__init__()
        pass


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        return X.to_numpy()

#=====================================================

class DFOneHotCategoriesCombined(BaseEstimator, TransformerMixin):
    """
    Encode combined columns (i.e. columns that represent same feature which can be represented by 2 or more boolean features) based on one-hot principle
    
    Dataset example: the house can have 'Artery (Adjacent to arterial street)' and 'PosA (Adjacent to postive off-site feature)' values in 'Condition1' and 'Condition2'
    In this case for both columns ('Condition1' and 'Condition2') the shared features pool has to be created and the house will contain 1.0 in 'Artery' and 'PosA'
    features simultaneously

    feature_kits parameter contains list of 2-tuples:
    [
        ([column_name1, column_name2, ...], [categoryA, categoryB, ...]), 
        ([column_nameN, column_nameN+1, ...], [categoryK, categoryK+1, ...])
    ]
    Categories list will be used as the source for one-hot features names which then will be checked by data in coresponding column names
    """
    def __init__(self, features_kits=None, return_full_df=True, drop_originals=True) -> None:
        """
        feature_kits = [
            ([column_name1, column_name2, ...], [categoryA, categoryB, ...]), 
            ([column_nameN, column_nameN+1, ...], [categoryK, categoryK+1, ...])
        ]
        drop_originals = True|False
        """
        self.features_kits = features_kits
        self.drop_originals = drop_originals
        self.return_full_df = return_full_df


    def fit(self, X, y=None):
        return self


    def get_onehot_encoding_(self, sample, col_names, target_value):
        """
        Function should return 1.0 if the 'category' has been found at least in 1 column from 'col_names'
        Otherwise return 0.0
        """
        onehot_value = False
        for column in col_names:
            onehot_value = onehot_value or (sample[column] == target_value)
        return float(onehot_value)


    def transform(self, X, y=None):
        if self.features_kits is None:
            return X
        
        onehot_features = None
        drop_columns = []
        for kit in self.features_kits:
            if self.drop_originals:
                drop_columns += kit[0]
            for category in kit[1]:
                serie = X.apply(self.get_onehot_encoding_, args=[kit[0], category], axis=1)
                serie.rename(category, inplace=True)
                if onehot_features is None:
                    onehot_features = serie
                else:
                    onehot_features = pd.concat([onehot_features, serie], axis=1)
        if self.return_full_df:
            return pd.concat([X.drop(columns=drop_columns), onehot_features], axis=1)
        else:
            return onehot_features


#=====================================================

class DFSetCategoryDType(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self, columns) -> None:
        """
        columns arg is the dictionary {'column_name': category_type} 
            where category_type can be passed as:
                pd.CategoricalDType object OR 
                dict {'categories': [<category1>,...], ordered: True|False} 
        """

        self.columns = columns
        pass


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if self.columns is None or len(self.columns) == 0:
            return X

        X_transformed = X.copy()
        for c, cat_type in self.columns.items():
            if c in X_transformed.columns:
                if type(cat_type) is pd.CategoricalDtype:
                    X_transformed[c] = X_transformed[c].astype(cat_type)
                else:
                    X_transformed[c] = X_transformed[c].astype(pd.CategoricalDtype(**cat_type))
        return X_transformed

#=====================================================

class DFSetUnorderedCategories(BaseEstimator, TransformerMixin):
    """
    Assigns categorical dtypes for columns.
    'categories' arg is a dict : {'column_name': pd.CategoricalDtype()} or {'column_name': [list_of_possible_values]}
    """

    cat_unordered_features = {
        'MSSubClass': pd.CategoricalDtype(categories=['Abs', 20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190], ordered=False),
        'MSZoning': pd.CategoricalDtype(categories=['Abs', 'A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'], ordered=False),
        'Street': pd.CategoricalDtype(categories=['Abs', 'Grvl', 'Pave'], ordered=False),
        'Alley': pd.CategoricalDtype(categories=['Abs', 'Grvl', 'Pave'], ordered=False),
        'LandContour': pd.CategoricalDtype(categories=['Abs', 'Lvl', 'Bnk', 'HLS', 'Low'], ordered=False),
        'LotConfig': pd.CategoricalDtype(categories=['Abs', 'Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'], ordered=False),
        'Neighborhood': pd.CategoricalDtype(categories=['Abs', 'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert',
                                                            'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown', 'SWISU',
                                                            'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'], ordered=False),
        'Condition1': pd.CategoricalDtype(categories=['Abs', 'Norm', 'Artery', 'Feedr', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'], ordered=False), # Norm value is default and can
        'Condition2': pd.CategoricalDtype(categories=['Abs', 'Norm', 'Artery', 'Feedr', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'], ordered=False), # Norm value is default and can
        'BldgType': pd.CategoricalDtype(categories=['Abs', '1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI'], ordered=False),
        'HouseStyle': pd.CategoricalDtype(categories=['Abs', '1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'], ordered=False),
        'RoofStyle': pd.CategoricalDtype(categories=['Abs', 'Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'], ordered=False),
        'RoofMatl': pd.CategoricalDtype(categories=['Abs', 'ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'], ordered=False),
        'Exterior1st': pd.CategoricalDtype(categories=['Abs', 'Other', 'AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 
                                                            'MetalSd', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing'], ordered=False),
        'Exterior2nd': pd.CategoricalDtype(categories=['Abs', 'Other', 'AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 
                                                            'MetalSd', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing'], ordered=False),
        'MasVnrType': pd.CategoricalDtype(categories=['Abs', 'BrkCmn', 'BrkFace', 'CBlock', 'None', 'Stone'], ordered=False),
        'Foundation': pd.CategoricalDtype(categories=['Abs', 'BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'], ordered=False),
        'Heating': pd.CategoricalDtype(categories=['Abs', 'Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'], ordered=False),
        'CentralAir': pd.CategoricalDtype(categories=['Abs', 'N', 'Y'], ordered=False),
        'Electrical': pd.CategoricalDtype(categories=['Abs', 'SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'], ordered=False),
        'GarageType': pd.CategoricalDtype(categories=['Abs', '2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd'], ordered=False), # NA = No garage => 'Abs'
        'MiscFeature': pd.CategoricalDtype(categories=['Abs', 'Elev', 'Gar2', 'Othr', 'Shed', 'TenC'], ordered=False), # NA = None
        'SaleType': pd.CategoricalDtype(categories=['Abs', 'WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth'], ordered=False),
        'SaleCondition': pd.CategoricalDtype(categories=['Abs', 'Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'], ordered=False),
    }

    def __init__(self, categories=None, process=True) -> None:
        #super().__init__()
        self.process = process
        if categories is not None:
            
            for column, cat_dtype in categories.items():
                if type(cat_dtype) is pd.CategoricalDtype:
                    self.cat_unordered_features[column] = cat_dtype
                else: # has to be a list
                    self.cat_unordered_features[column] = pd.CategoricalDtype(categories=cat_dtype, ordered=False)

    
    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if not self.process:
            return X
        X_transformed = X.copy()    
        for column, cat_dtype in self.cat_unordered_features.items():
            if column in X_transformed.columns:
                X_transformed[column] = X_transformed[column].astype(cat_dtype)

        return X_transformed

#=====================================================

class DFSetOrderedCategories(BaseEstimator, TransformerMixin):
    def __init__(self, process=True) -> None:
        #super().__init__()
        self.process = process


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if not self.process:
            return X

        X_transformed = X.copy()
        # set categories for 
        quality_categories = pd.CategoricalDtype(categories=['Abs', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True)
        quality_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
        for column in quality_features:
            X_transformed[column] = X_transformed[column].astype(quality_categories)

        X_transformed['LotShape'] = X_transformed['LotShape'].astype(pd.CategoricalDtype(categories=['Abs', 'IR3', 'IR2', 'IR1', 'Reg'], ordered=True))
        X_transformed['Utilities'] = X_transformed['Utilities'].astype(pd.CategoricalDtype(categories=['Abs', 'ELO', 'NoSeWa', 'NoSewr', 'AllPub'], ordered=True))
        X_transformed['LandSlope'] = X_transformed['LandSlope'].astype(pd.CategoricalDtype(categories=['Abs', 'Sev', 'Mod', 'Gtl'], ordered=True))
        X_transformed['BsmtExposure'] = X_transformed['BsmtExposure'].astype(pd.CategoricalDtype(categories=['Abs', 'No', 'Mn', 'Av', 'Gd'], ordered=True))
        X_transformed['BsmtFinType1'] = X_transformed['BsmtFinType1'].astype(pd.CategoricalDtype(categories=['Abs', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], ordered=True))
        X_transformed['BsmtFinType2'] = X_transformed['BsmtFinType2'].astype(pd.CategoricalDtype(categories=['Abs', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], ordered=True))
        X_transformed['Functional'] = X_transformed['Functional'].astype(pd.CategoricalDtype(categories=['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'], ordered=True))
        X_transformed['GarageFinish'] = X_transformed['GarageFinish'].astype(pd.CategoricalDtype(categories=['Abs', 'Unf', 'RFn', 'Fin'], ordered=True))
        X_transformed['PavedDrive'] = X_transformed['PavedDrive'].astype(pd.CategoricalDtype(categories=['N', 'P', 'Y'], ordered=True))
        X_transformed['Fence'] = X_transformed['Fence'].astype(pd.CategoricalDtype(categories=['Abs', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'], ordered=True))

        return X_transformed

#=====================================================

class DFOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes each column passed with individual OneHotEncoder
    Returns DataFrame with encoded columns only or full DataFrame with encoded columns appended to the end
    (original columns will be dropped by default until drop_originals == False)
    !!! Incompatible with sklearn.compose.ColumnTransformer !!!
    """
    def __init__(
        self,
        cols_cats,
        drop=None,
        sparse=None,
        dtype=np.float64,
        handle_unknown="error",
        col_overrule_params={},
        return_full_df=True,
        drop_originals=True,
    ) -> None:
        """
        Args:
            cols_cats: dictionary with columns names and categories {'column_name': 'auto' | list of array-like}
            dtype: resulting number type
            drop: 
            sparse: left for compatibility (will always return dense)
            handle_unknown: 'error' | 'ignore'
            col_overrule_params: dict to overrule default parameters for column
            return_full_df: if full data frame should be returned (default) with new encoded columns appended to the end or encoded columns only
            drop_originals: if original columns should be dropped from resulting DataFrame (works only if return_full_df == True)
        """
        self.cols_cats = cols_cats
        self.drop = drop
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.col_overrule_params = col_overrule_params
        self.drop_originals = drop_originals
        self.return_full_df = return_full_df
        pass


    def fit(self, X, y=None):
        """
        Fit a separate OneHotEncoder for each of the columns in the dataframe
        Args:
            X: dataframe
            y: None, ignored. This parameter exists only for compatibility with Pipeline
        Returns
            self
        Raises
            TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        self.encoders_ = {}
        self.column_names_ = {}

        for c, cat in self.cols_cats.items():
            # Construct the OHE parameters using the arguments
            if cat == 'auto':
                categories = 'auto'
            else:
                categories = [cat]
            enc_params = {
                'categories': categories,
                'drop': self.drop,
                'sparse': False,
                'dtype': self.dtype,
                'handle_unknown': self.handle_unknown,
            }
            # and update it with potential overrule parameters for the current column
            enc_params.update(self.col_overrule_params.get(c, {}))

            # Regardless of how we got the parameters, make sure we always set the
            # sparsity to False
            enc_params["sparse"] = False

            # Now create, fit, and store the onehotencoder for current column c
            enc = OneHotEncoder(**enc_params)
            self.encoders_[c] = enc.fit(X.loc[:, [c]])

            # Get the feature names and replace each x0 with the original column name
            feature_names = enc.get_feature_names_out()
            feature_names = [x.replace("x0", c) for x in feature_names]
            #feature_names = [x.replace("x0_", "") for x in feature_names]
            #feature_names = [f"{c}_{x}" for x in feature_names]
            #feature_names = [f"{c}[{x}]" for x in feature_names]

            self.column_names_[c] = feature_names

        return self

        
    def transform(self, X, y=None):
        """
        Transform X using the one-hot-encoding per column
        Args:
            X: Dataframe that is to be one hot encoded
        Returns:
            Dataframe with onehotencoded data
        Raises
            NotFittedError if the transformer is not yet fitted
            TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f'X should be of type dataframe, not {type(X)}')

        if not hasattr(self, 'encoders_'):
            raise NotFittedError(f'{type(self).__name__} is not fitted')

        new_columns = []
        for c, enc in self.encoders_.items():
            transformed_col = enc.transform(X.loc[:, [c]])
            df_col = pd.DataFrame(transformed_col, columns=self.column_names_[c], index=X.index)
            new_columns.append(df_col)

        if self.return_full_df:
            X_transformed = X.copy()
            if self.drop_originals:
                X_transformed.drop(columns=self.encoders_.keys(), inplace=True)
            return pd.concat([X_transformed]+new_columns, axis=1)
        return pd.concat(new_columns, axis=1)

#=====================================================

class DFOrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes each column passed with individual OrdinalEncoder
    Returns DataFrame with encoded columns only or full DataFrame with encoded values in columns passed
    !!! Incompatible with sklearn.compose.ColumnTransformer !!!
    """

    def __init__(self, cols_cats=None, dtype=np.float64, handle_unknown='error', unknown_value=None, encoded_missing_value=np.nan, col_overrule_params={}, return_full_df=True) -> None:
        """
        Args:
            columns: dict = {'column_name': 'auto' | list of array-like}
            dtype: resulting number type
            handle_unknown: 'error' | 'use_encoded_value'
            unknown_value: int | np.nan, use if 'handle_unknown' == 'use_encoded_value'
            encoded_missing_value: int | np.nan
            col_overrule_params: dict to overrule default parameters for column
        """
        self.cols_cats = cols_cats
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.col_overrule_params = col_overrule_params
        self.return_full_df = return_full_df
        pass


    def fit(self, X, y=None):
        """
        Fit separate OrdinalEncoder for columns in 'columns' arg 
        Args:
            X: DataFrame
            y: None, ignored. This parameter exists only for compatibility with Pipeline
        Returns
            self
        Raises
            TypeError if X is not of type DataFrame
        """
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type pd.DataFrame, not {type(X)}")
        
        self.encoders_ = {}
        
        for c, cat in self.cols_cats.items():
            if cat == 'auto':
                categories = 'auto'
            else:
                categories = [cat]
            enc_params = {
                'categories': categories,
                'dtype': self.dtype,
                'handle_unknown': self.handle_unknown,
                'unknown_value': self.unknown_value,
                #'encoded_missing_value': self.encoded_missing_value
            }
            # and update it with potential overrule parameters for the current column
            enc_params.update(self.col_overrule_params.get(c, {}))
            
            enc = OrdinalEncoder(**enc_params)
            
            self.encoders_[c] = enc.fit(X.loc[:, [c]])
        return self


    def transform(self, X, y=None):
        """
        Transform X using the trained OrdinalEncoder per column
        Args:
            X: DataFrame to be encoded
        Returns:
            DataFrame with columns changed to encoded values
        Raises:
            NotFittedError if the transformer is not yet fitted
            TypeError if X is not of type DataFrame
        """
        
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type pd.DataFrame, not {type(X)}")

        if not hasattr(self, 'encoders_'):
            raise NotFittedError(f'{type(self).__name__} is not fitted')

        columns = []
        encoded = []
        for c, enc in self.encoders_.items():
            columns.append(c)
            encoded.append(pd.DataFrame(enc.transform(X.loc[:, [c]]), index=X.index, columns=[c]))
        
        transformed_df = pd.concat(encoded, axis=1)
        
        if self.return_full_df:
            X_transformed = X.copy()
            X_transformed[columns] = transformed_df
            return X_transformed
        else:
            return transformed_df

#=====================================================

if __name__ == "__main__":
    
    # Default pipeline for AdvRegression task experiments may look like follows

    etl_pipeline = Pipeline([
        ('add_has_features', DFCreateAdditionalFeatures()),
        ('fill_meanful_nans', DFReplaceMeaningfulNANs()),
        ('days_since_sold', DFJoinDates(month_col='MoSold', year_col='YrSold', calc_period_to=datetime.today(), new_column_name='DaysSinceSold', drop_columns=False)),
        ('calc_ages', DFCalcAge({'YearBuilt': 'BuiltAge', 'YearRemodAdd': 'RemodAge'}, calc_age_to=2022)),
        ('drop_columns', DFDropColumns(['Id', 'SalePrice'])),
        ('numpy_converter', DFConvertToNumpy()),
        # Here add sklearn classes such as imputers, models, etc
    ])