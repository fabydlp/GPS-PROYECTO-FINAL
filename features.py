"""
Feature Engineering for SBA Mexico Expected Loss Model
Adapted for Mexican SME (PyME) loan market with NAFIN guarantees
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer


def prepare_data(filepath):
    """
    Load and prepare Mexican PyME loan data with all features
    cleaned for PD/LGD modelling WITHOUT leakage.
    """

    df = pd.read_csv(filepath)

    # =========================================
    # TARGET VARIABLES
    # =========================================
    df['ChgOffPrinGr'] = df['ChgOffPrinGr'].fillna(0).astype(float)
    y_loss = df['ChgOffPrinGr']

    if 'Default' in df.columns:
        y_pd = df['Default'].astype(int)
    else:
        y_pd = (y_loss > 0).astype(int)

    # =========================================
    # CLEAN BASE FIELDS
    # =========================================
    df['GrAppv'] = df['GrAppv'].fillna(0.0)
    df['NAFIN_Appv'] = df.get('NAFIN_Appv', df.get('SBA_Appv', 0.0)).fillna(0.0)
    df['Term'] = df['Term'].fillna(0).astype(int)
    df['NoEmp'] = df['NoEmp'].fillna(0).astype(int)

    df['NewExist'] = df['NewExist'].fillna(2.0)
    df['IsNewBusiness'] = (df['NewExist'] == 1).astype(int)

    # SCIAN
    df['SCIAN'] = (
        df.get('SCIAN', df.get('NAICS', '00'))
        .astype(str)
        .str[:2]
        .replace({'0': '00', 'na': '00', 'nan': '00'})
    )

    df['State'] = df.get('State', 'CDMX').fillna('CDMX').astype(str)

    # RealEstate, Recession, UrbanRural
    df['HasRealEstate'] = df.get('RealEstate', 0).fillna(0).astype(int)
    df['InRecession'] = df.get('Recession', 0).fillna(0).astype(int)
    df['IsUrban'] = (df.get('UrbanRural', 1) == 1).astype(int)

    # =========================================
    # FEATURE ENGINEERING (NO LEAKAGE)
    # =========================================

    df['NAFIN_Portion'] = np.where(
        df['GrAppv'] > 0,
        df['NAFIN_Appv'] / df['GrAppv'],
        0.0
    ).clip(0, 1)

    df['Loan_per_Emp'] = df['GrAppv'] / (df['NoEmp'] + 1)
    df['Log_Loan_per_Emp'] = np.log1p(df['Loan_per_Emp'])

    df['Term_Years'] = df['Term'] / 12

    df['Debt_to_NAFIN'] = (df['GrAppv'] - df['NAFIN_Appv']).clip(lower=0)
    df['Log_GrAppv'] = np.log1p(df['GrAppv'])

    df['Bank_Exposure'] = df['GrAppv'] - df['NAFIN_Appv']
    df['Bank_Exposure_Ratio'] = df['Bank_Exposure'] / df['GrAppv'].replace(0, 1)

    df['RealEstate_Exposure'] = df['HasRealEstate'] * df['GrAppv']

    # Loan Age (solo si es fecha al originar, no post-default)
    if 'ApprovalFY' in df.columns:
        df['Loan_Age'] = df['ApprovalFY'].max() - df['ApprovalFY']
    else:
        df['Loan_Age'] = 0

    # Region groups
    north = ["NL", "CHIH", "TAM", "COAH", "SON"]
    center = ["CDMX", "MEX", "PUE", "HGO", "QRO"]
    south = ["CHIS", "OAX", "TAB", "CAM", "YUC", "QROO"]

    df["Region"] = np.select(
        [
            df["State"].isin(north),
            df["State"].isin(center),
            df["State"].isin(south)
        ],
        ["North", "Center", "South"],
        default="Other")


    # Banking frequency (solo conteo, NO default rate)
    if "Bank" in df.columns:
        df["Bank_Frequency"] = df.groupby("Bank")["Bank"].transform("count")
        df["Bank_Frequency_Log"] = np.log1p(df["Bank_Frequency"])
    else:
        df["Bank_Frequency"] = 1
        df["Bank_Frequency_Log"] = 0

    # =========================================
    # SELECT SAFE FEATURES (NO LEAKAGE)
    # =========================================
    num_feats = ['GrAppv', 
                 'NAFIN_Appv', 
                 'Debt_to_NAFIN',
                 'Log_GrAppv', 'Term', 
                 'Term_Years',
                 'NoEmp', 
                 'NAFIN_Portion',''
                 'Loan_per_Emp', 
                 'Log_Loan_per_Emp',
                 'HasRealEstate', 
                 'RealEstate_Exposure',
                 'InRecession','Bank_Exposure', 
                 'Bank_Exposure_Ratio','Loan_Age', 
                 'Bank_Frequency', 'Bank_Frequency_Log']

    cat_feats = ['SCIAN', 'State', 'Region','IsNewBusiness','IsUrban']

    available_num = [c for c in num_feats if c in df.columns]
    available_cat = [c for c in cat_feats if c in df.columns]

    X = df[available_num + available_cat].copy()

    return X, y_pd, y_loss, df


def create_preprocessor():
    num_feats = ['GrAppv', 
                 'NAFIN_Appv', 
                 'Debt_to_NAFIN',
                 'Log_GrAppv', 'Term', 
                 'Term_Years',
                 'NoEmp', 
                 'NAFIN_Portion',''
                 'Loan_per_Emp', 
                 'Log_Loan_per_Emp',
                 'HasRealEstate', 
                 'RealEstate_Exposure',
                 'InRecession','Bank_Exposure', 
                 'Bank_Exposure_Ratio','Loan_Age', 
                 'Bank_Frequency', 'Bank_Frequency_Log']

    cat_feats = ['SCIAN', 'State', 'Region','IsNewBusiness','IsUrban']

    try:
        cat_tf = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,
                                encoded_missing_value=-2,dtype=float)
    except:
        cat_tf = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,
                                encoded_missing_value=-2,dtype=float)

    preprocessor = ColumnTransformer([
        ("num", PowerTransformer(method="yeo-johnson",
                                 standardize=True,
                                 copy=True), num_feats),("cat", cat_tf, cat_feats)])

    return preprocessor


def transform_data(preprocessor, X):
    X_mat = preprocessor.transform(X)

    if hasattr(X_mat, "toarray"):
        X_mat = X_mat.toarray()

    try:
        cols = preprocessor.get_feature_names_out()
    except:
        cols = [f"f{i}" for i in range(X_mat.shape[1])]

    return pd.DataFrame(X_mat, index=X.index, columns=cols)