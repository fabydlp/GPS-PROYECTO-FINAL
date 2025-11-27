"""
Calculadora de Garantía de Crédito PyME México
SBA Mexico Loan Guarantee Fee Calculator

Ejecuta este script para obtener una cotización de garantía NAFIN
para un préstamo PyME en México.
"""

import pickle
import os
import numpy as np
import pandas as pd
from features import create_preprocessor, transform_data, SECTORES_SCIAN, ESTADOS_MEXICO


def calculate_nafin_guarantee(approved_amount):
    """
    Calculate NAFIN guarantee amount based on loan size
    Calcula el monto de garantía NAFIN según el tamaño del préstamo
    
    Reglas típicas de NAFIN:
    - Préstamos ≤ $2,000,000 MXN: hasta 80% de garantía
    - Préstamos > $2,000,000 MXN: hasta 70% de garantía
    """
    if approved_amount <= 2_000_000:
        return approved_amount * 0.80
    else:
        return approved_amount * 0.70


def calculate_monthly_payment(principal, annual_rate, term_months):
    """
    Calculate monthly loan payment (amortización francesa)
    Calcula el pago mensual del préstamo
    """
    if annual_rate == 0:
        return principal / term_months
    
    monthly_rate = (annual_rate / 100) / 12
    payment = principal * (monthly_rate * (1 + monthly_rate)**term_months) / \
              ((1 + monthly_rate)**term_months - 1)
    return payment


def create_loan_features(approved_amount, term_months, num_employees, 
                         is_new_business, scian_code, state_code):
    """
    Create feature vector for a loan
    Crea el vector de features para un préstamo
    """
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
        'HasRealEstate': 0,  # Default: sin garantía inmobiliaria
        'InRecession': 0,    # Default: no en recesión
        'IsUrban': 1,        # Default: urbano
    }
    
    return pd.DataFrame([features])


def load_models():
    """
    Load trained models or train if they don't exist
    Carga los modelos entrenados o entrena si no existen
    """
    model_files = ['sba_mexico_model.pkl', 'sba_model.pkl']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                return pickle.load(f)
    
    print("⚠ Modelos no encontrados. Entrenando ahora...")
    print("  Esto tomará unos minutos...\n")
    import train
    train.main()
    print("\n")
    
    # Cargar modelos recién entrenados
    with open('sba_mexico_model.pkl', 'rb') as f:
        return pickle.load(f)


def calculate_quote(approved_amount, term_months, num_employees, 
                   is_new_business, scian_code, state_code, bank_rate,
                   has_real_estate=False, in_recession=False):
    """
    Calculate complete loan quote
    Calcula la cotización completa del préstamo
    """
    
    # Cargar modelos
    artifacts = load_models()
    
    # Crear features
    loan_df = create_loan_features(
        approved_amount, term_months, num_employees,
        is_new_business, scian_code, state_code
    )
    
    # Ajustar features adicionales
    loan_df['HasRealEstate'] = 1 if has_real_estate else 0
    loan_df['InRecession'] = 1 if in_recession else 0
    
    # Preprocesar
    X_processed = transform_data(artifacts['preprocessor'], loan_df)
    
    # Predecir
    pd_pred = artifacts['pd_model'].predict_proba(X_processed)[:, 1][0]
    lgd_pred = artifacts['lgd_model'].predict(X_processed)[0]
    el_pred = pd_pred * lgd_pred * artifacts['calibration_factor']
    
    # Calcular comisión de garantía
    # La comisión cubre la pérdida esperada + margen de seguridad (20%)
    nafin_guaranteed = calculate_nafin_guarantee(approved_amount)
    
    # Comisión = EL * 1.20 (20% de margen)
    # Con límites: mínimo 0.5% y máximo 5% del monto garantizado
    guarantee_fee = el_pred * 1.20
    guarantee_fee = max(guarantee_fee, nafin_guaranteed * 0.005)  # Mínimo 0.5%
    guarantee_fee = min(guarantee_fee, nafin_guaranteed * 0.05)   # Máximo 5%
    
    # Calcular pago mensual
    total_financed = approved_amount + guarantee_fee
    monthly_payment = calculate_monthly_payment(total_financed, bank_rate, term_months)
    
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
    }


def print_quote(quote):
    """
    Print formatted quote
    Imprime la cotización formateada
    """
    sector_name = SECTORES_SCIAN.get(str(quote['scian_code'])[:2], 'No especificado')
    state_name = ESTADOS_MEXICO.get(quote['state'].upper(), quote['state'])
    
    print("\n" + "="*60)
    print("COTIZACIÓN DE GARANTÍA NAFIN - CRÉDITO PYME")
    print("="*60)
    
    print("\n--- Datos del Préstamo ---")
    print(f"Sector:                   {sector_name} (SCIAN {quote['scian_code']})")
    print(f"Estado:                   {state_name}")
    
    print("\n--- Evaluación de Riesgo ---")
    print(f"Monto del Préstamo:       ${quote['approved_amount']:,.2f} MXN")
    print(f"Monto Garantizado NAFIN:  ${quote['nafin_guaranteed']:,.2f} MXN")
    print(f"Probabilidad de Default:  {quote['pd']*100:.2f}%")
    print(f"Pérdida Dado Default:     ${quote['lgd']:,.2f} MXN")
    print(f"Pérdida Esperada:         ${quote['expected_loss']:,.2f} MXN")
    
    # Indicador de riesgo
    if quote['pd'] < 0.05:
        risk_level = "🟢 BAJO"
    elif quote['pd'] < 0.10:
        risk_level = "🟡 MODERADO"
    elif quote['pd'] < 0.15:
        risk_level = "🟠 MEDIO-ALTO"
    else:
        risk_level = "🔴 ALTO"
    
    print(f"Nivel de Riesgo:          {risk_level}")
    
    print("\n--- Comisión de Garantía ---")
    print(f"Comisión NAFIN:           ${quote['guarantee_fee']:,.2f} MXN")
    fee_pct = (quote['guarantee_fee'] / quote['nafin_guaranteed']) * 100
    print(f"                          ({fee_pct:.2f}% del monto garantizado)")
    
    print("\n--- Pago Mensual ---")
    print(f"Monto del Préstamo:       ${quote['approved_amount']:,.2f} MXN")
    print(f"+ Comisión de Garantía:   ${quote['guarantee_fee']:,.2f} MXN")
    print(f"= Total a Financiar:      ${quote['total_financed']:,.2f} MXN")
    print(f"\nTasa de Interés:          {quote['bank_rate']:.2f}% anual")
    print(f"Plazo:                    {quote['term_months']} meses")
    print(f"\nPAGO MENSUAL:             ${quote['monthly_payment']:,.2f} MXN")
    
    print("\n" + "="*60)
    print("Nota: Esta cotización es informativa. La aprobación final")
    print("depende de la evaluación del banco y de NAFIN.")
    print("="*60 + "\n")


def show_scian_codes():
    """Muestra los códigos SCIAN disponibles"""
    print("\n--- Códigos SCIAN (Sectores Económicos) ---")
    for code, name in sorted(SECTORES_SCIAN.items()):
        print(f"  {code}: {name}")
    print()


def show_state_codes():
    """Muestra los códigos de estados disponibles"""
    print("\n--- Códigos de Estados ---")
    for code, name in sorted(ESTADOS_MEXICO.items()):
        print(f"  {code}: {name}")
    print()


def main():
    """
    Main interactive quoter
    Cotizador interactivo principal
    """
    print("\n" + "="*60)
    print("CALCULADORA DE GARANTÍA NAFIN - CRÉDITO PYME MÉXICO")
    print("="*60)
    print("\nIngresa los datos del préstamo:\n")
    
    try:
        # Monto del préstamo
        approved_amount = float(input("1. Monto del préstamo (MXN): $"))
        
        # Plazo
        term_months = int(input("2. Plazo (meses, ej: 36): "))
        
        # Empleados
        num_employees = int(input("3. Número de empleados: "))
        
        # Negocio nuevo
        is_new = input("4. ¿Es negocio nuevo? (s/n): ").lower() in ['s', 'si', 'sí', 'y', 'yes']
        
        # SCIAN
        show_codes = input("5. ¿Ver códigos SCIAN? (s/n): ").lower() in ['s', 'si', 'sí', 'y', 'yes']
        if show_codes:
            show_scian_codes()
        scian = input("   Código SCIAN (2 dígitos, ej: 46 para comercio): ").strip()
        
        # Estado
        show_states = input("6. ¿Ver códigos de estados? (s/n): ").lower() in ['s', 'si', 'sí', 'y', 'yes']
        if show_states:
            show_state_codes()
        state = input("   Estado (ej: JAL, CDMX, NL): ").strip().upper()
        
        # Tasa de interés
        bank_rate = float(input("7. Tasa de interés del banco (%, ej: 12.5): "))
        
        # Garantía inmobiliaria (opcional)
        has_real_estate = input("8. ¿Tiene garantía inmobiliaria? (s/n): ").lower() in ['s', 'si', 'sí', 'y', 'yes']
        
        print("\nCalculando cotización...")
        
        quote = calculate_quote(
            approved_amount, term_months, num_employees,
            is_new, scian, state, bank_rate, has_real_estate
        )
        
        print_quote(quote)
        
    except ValueError:
        print("\n❌ Error: Por favor ingresa valores numéricos válidos.")
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def quick_quote(amount, term, employees, is_new, scian, state, rate):
    """
    Quick quote for programmatic use
    Cotización rápida para uso programático
    """
    quote = calculate_quote(amount, term, employees, is_new, scian, state, rate)
    print_quote(quote)
    return quote


if __name__ == "__main__":
    main()
