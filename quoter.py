"""
Calculadora de Garantía de Crédito PyME México
SBA Mexico Loan Guarantee Fee Calculator
"""

import pickle
import os
import numpy as np
import pandas as pd
from features import create_preprocessor, transform_data, SECTORES_SCIAN, ESTADOS_MEXICO


def calculate_nafin_guarantee(approved_amount):
    """Cálculo de garantía NAFIN según tamaño del crédito"""
    if approved_amount <= 2_000_000:
        return approved_amount * 0.80
    else:
        return approved_amount * 0.70


def calculate_monthly_payment(principal, annual_rate, term_months):
    """Pago mensual estilo amortización francesa"""
    if annual_rate == 0:
        return principal / term_months
    
    monthly_rate = (annual_rate / 100) / 12
    payment = principal * (monthly_rate * (1 + monthly_rate)**term_months) / \
              ((1 + monthly_rate)**term_months - 1)
    return payment


def create_loan_features(approved_amount, term_months, num_employees, 
                         is_new_business, scian_code, state_code):

    nafin_guaranteed = calculate_nafin_guarantee(approved_amount)
    
    features = {
        'GrAppv': approved_amount,
        'NAFIN_Appv': nafin_guaranteed,
        'Term': term_months,
        'NoEmp': num_employees,
        'IsNewBusiness': 1 if is_new_business else 0,
        'NewExist': 1.0 if is_new_business else 2.0,
        'SCIAN': str(scian_code)[:2],
        'State': state_code.upper(),
        'NAFIN_Portion': nafin_guaranteed / approved_amount if approved_amount > 0 else 0,
        'Loan_per_Employee': approved_amount / (num_employees + 1),
        'Term_Years': term_months / 12.0,
        'Debt_to_NAFIN': approved_amount - nafin_guaranteed,
        'Log_GrAppv': np.log1p(approved_amount),
        'HasRealEstate': 0,
        'InRecession': 0,
        'IsUrban': 1,
    }
    
    return pd.DataFrame([features])


def load_models():
    """Cargar modelos entrenados"""
    model_files = ['sba_mexico_model.pkl', 'sba_model.pkl']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                return pickle.load(f)
    
    print("⚠ Modelos no encontrados. Entrenando...")
    import train
    train.main()
    
    with open('sba_mexico_model.pkl', 'rb') as f:
        return pickle.load(f)


def calculate_quote(approved_amount, term_months, num_employees, 
                   is_new_business, scian_code, state_code, bank_rate,
                   has_real_estate=False, in_recession=False):
    """Cotización completa del préstamo"""

    artifacts = load_models()

    # Crear features
    loan_df = create_loan_features(
        approved_amount, term_months, num_employees,
        is_new_business, scian_code, state_code
    )

    # Ajustar extras
    loan_df['HasRealEstate'] = 1 if has_real_estate else 0
    loan_df['InRecession'] = 1 if in_recession else 0

    # Preprocesar
    X_processed = transform_data(artifacts['preprocessor'], loan_df)

    # Predicciones
    pd_pred = artifacts['pd_model'].predict_proba(X_processed)[:, 1][0]
    lgd_pred = artifacts['lgd_model'].predict(X_processed)[0]
    el_pred = pd_pred * lgd_pred * artifacts['calibration_factor']

    # Garantía NAFIN
    nafin_guaranteed = calculate_nafin_guarantee(approved_amount)

    # Comisión NAFIN
    guarantee_fee = el_pred * 1.20
    guarantee_fee = max(guarantee_fee, nafin_guaranteed * 0.005)
    guarantee_fee = min(guarantee_fee, nafin_guaranteed * 0.05)

    # Pago mensual
    total_financed = approved_amount + guarantee_fee
    monthly_payment = calculate_monthly_payment(total_financed, bank_rate, term_months)

    # ---------- NUEVA LÓGICA GPS ----------
    if pd_pred < 0.01:
        gps_category = "Ultra–Oro"
        soform_guarantee_pct = 0.10
        action = "Aprobar sin condiciones adicionales."
    elif pd_pred < 0.03:
        gps_category = "Oro"
        soform_guarantee_pct = 0.20
        action = "Aprobar con verificación ligera."
    else:
        gps_category = "Rechazo"
        soform_guarantee_pct = 0.00
        action = "Rechazar por riesgo elevado."
    # --------------------------------------

    return {
        'approved_amount': approved_amount,
        'nafin_guaranteed': nafin_guaranteed,
        'pd': pd_pred,
        'lgd': lgd_pred,
        'expected_loss': el_pred,
        'guarantee_fee': guarantee_fee,
        'total_financed': total_financed,
        'monthly_payment': monthly_payment,
        'term_months': term_months,
        'bank_rate': bank_rate,
        'scian_code': scian_code,
        'state': state_code,

        # << NUEVOS CAMPOS >>
        'gps_category': gps_category,
        'soform_guarantee_pct': soform_guarantee_pct,
        'action': action,
    }