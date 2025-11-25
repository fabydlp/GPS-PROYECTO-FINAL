import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de semilla para reproducibilidad
np.random.seed(42)

# ==============================================================================
# PARÁMETROS CALIBRADOS CON DATOS REALES DE MÉXICO
# ==============================================================================

# Distribución de empresas por Estado (basado en Censos Económicos 2024)
DISTRIBUCION_ESTADOS = {
    'MEX': 0.127, 'VER': 0.069, 'PUE': 0.067, 'JAL': 0.065, 'CDMX': 0.062,
    'GTO': 0.055, 'MIC': 0.048, 'OAX': 0.045, 'CHIS': 0.042, 'NL': 0.040,
    'GRO': 0.035, 'TAM': 0.032, 'SLP': 0.030, 'CHIH': 0.028, 'SIN': 0.027,
    'HGO': 0.026, 'SON': 0.025, 'TAB': 0.022, 'COAH': 0.021, 'QRO': 0.020,
    'MOR': 0.019, 'DGO': 0.016, 'YUC': 0.016, 'QROO': 0.015, 'ZAC': 0.014,
    'AGS': 0.013, 'TLAX': 0.012, 'NAY': 0.010, 'COL': 0.008, 'CAM': 0.007,
    'BCS': 0.006, 'BC': 0.025,
}

# Códigos SCIAN (2 dígitos)
SECTORES_SCIAN = {
    '11': {'nombre': 'Agricultura, ganadería, pesca', 'pct': 0.02, 'tasa_default': 0.08},
    '21': {'nombre': 'Minería', 'pct': 0.01, 'tasa_default': 0.06},
    '22': {'nombre': 'Generación de energía', 'pct': 0.005, 'tasa_default': 0.04},
    '23': {'nombre': 'Construcción', 'pct': 0.08, 'tasa_default': 0.09},
    '31': {'nombre': 'Manufactura - Alimentos', 'pct': 0.06, 'tasa_default': 0.05},
    '32': {'nombre': 'Manufactura - Textil/Química', 'pct': 0.04, 'tasa_default': 0.06},
    '33': {'nombre': 'Manufactura - Metálica/Maquinaria', 'pct': 0.05, 'tasa_default': 0.055},
    '43': {'nombre': 'Comercio al por mayor', 'pct': 0.08, 'tasa_default': 0.065},
    '46': {'nombre': 'Comercio al por menor', 'pct': 0.35, 'tasa_default': 0.075},
    '48': {'nombre': 'Transporte', 'pct': 0.04, 'tasa_default': 0.07},
    '51': {'nombre': 'Información en medios', 'pct': 0.02, 'tasa_default': 0.05},
    '52': {'nombre': 'Servicios financieros', 'pct': 0.02, 'tasa_default': 0.04},
    '53': {'nombre': 'Servicios inmobiliarios', 'pct': 0.03, 'tasa_default': 0.065},
    '54': {'nombre': 'Servicios profesionales', 'pct': 0.05, 'tasa_default': 0.045},
    '56': {'nombre': 'Servicios de apoyo', 'pct': 0.04, 'tasa_default': 0.06},
    '61': {'nombre': 'Servicios educativos', 'pct': 0.02, 'tasa_default': 0.035},
    '62': {'nombre': 'Servicios de salud', 'pct': 0.03, 'tasa_default': 0.04},
    '71': {'nombre': 'Esparcimiento/Cultura', 'pct': 0.02, 'tasa_default': 0.08},
    '72': {'nombre': 'Alojamiento/Alimentos', 'pct': 0.12, 'tasa_default': 0.085},
    '81': {'nombre': 'Otros servicios', 'pct': 0.06, 'tasa_default': 0.07},
}

# Bancos comerciales en México (principales)
BANCOS_MEXICO = {
    'BBVA México': {'pct': 0.22, 'estado_matriz': 'CDMX'}, 'Banorte': {'pct': 0.18, 'estado_matriz': 'NL'},
    'Santander México': {'pct': 0.15, 'estado_matriz': 'CDMX'}, 'Citibanamex': {'pct': 0.12, 'estado_matriz': 'CDMX'},
    'HSBC México': {'pct': 0.08, 'estado_matriz': 'CDMX'}, 'Scotiabank': {'pct': 0.06, 'estado_matriz': 'CDMX'},
    'Banco del Bajío': {'pct': 0.05, 'estado_matriz': 'GTO'}, 'Banco Azteca': {'pct': 0.04, 'estado_matriz': 'CDMX'},
    'Inbursa': {'pct': 0.03, 'estado_matriz': 'CDMX'}, 'Multiva': {'pct': 0.02, 'estado_matriz': 'CDMX'},
    'BanCoppel': {'pct': 0.02, 'estado_matriz': 'SIN'}, 'Afirme': {'pct': 0.015, 'estado_matriz': 'NL'},
    'Banca Mifel': {'pct': 0.01, 'estado_matriz': 'CDMX'}, 'Banco Autofin': {'pct': 0.005, 'estado_matriz': 'CDMX'},
}

# Parámetros de crédito
PARAMETROS_CREDITO = {
    'monto_min': 50_000, 'monto_max': 50_000_000, 'monto_mediana_micro': 150_000,
    'monto_mediana_pequena': 800_000, 'monto_mediana_mediana': 3_500_000,
    'monto_mediana_grande': 15_000_000, 'plazo_min': 6, 'plazo_max': 120,
    'plazo_mediana': 36, 'garantia_nafin_min': 0.50, 'garantia_nafin_max': 0.80,
    'tasa_interes_min': 0.08, 'tasa_interes_max': 0.25, 'lgd_sin_garantia': 0.75,
    'lgd_con_garantia_inmob': 0.35, 'lgd_con_garantia_nafin': 0.25,
}

# Clasificación de tamaño de empresa (INEGI)
TAMANO_EMPRESA = {
    'micro': {'empleados_min': 1, 'empleados_max': 10, 'pct': 0.95},
    'pequena': {'empleados_min': 11, 'empleados_max': 50, 'pct': 0.035},
    'mediana': {'empleados_min': 51, 'empleados_max': 250, 'pct': 0.012},
    'grande': {'empleados_min': 251, 'empleados_max': 1000, 'pct': 0.003},
}

# ==============================================================================
# FUNCIONES DE GENERACIÓN DE DATOS (NO MODIFICADAS)
# ==============================================================================

def generar_id_prestamo(n):
    return [f"MX{str(i).zfill(10)}" for i in range(1, n + 1)]

def seleccionar_estado(n):
    estados = list(DISTRIBUCION_ESTADOS.keys())
    probs = np.array(list(DISTRIBUCION_ESTADOS.values()))
    probs = probs / probs.sum()
    return np.random.choice(estados, size=n, p=probs)

def seleccionar_scian(n):
    sectores = list(SECTORES_SCIAN.keys())
    probs = [SECTORES_SCIAN[s]['pct'] for s in sectores]
    probs = np.array(probs) / sum(probs)
    return np.random.choice(sectores, size=n, p=probs)

def seleccionar_banco(n):
    bancos = list(BANCOS_MEXICO.keys())
    probs = [BANCOS_MEXICO[b]['pct'] for b in bancos]
    probs = np.array(probs) / sum(probs)
    return np.random.choice(bancos, size=n, p=probs)

def generar_tamano_empresa(n):
    tamanos = list(TAMANO_EMPRESA.keys())
    probs = [TAMANO_EMPRESA[t]['pct'] for t in tamanos]
    probs = np.array(probs) / sum(probs)

    tamano_seleccionado = np.random.choice(tamanos, size=n, p=probs)

    empleados = []
    for t in tamano_seleccionado:
        min_emp = TAMANO_EMPRESA[t]['empleados_min']
        max_emp = TAMANO_EMPRESA[t]['empleados_max']
        emp = int(np.clip(np.random.lognormal(np.log(min_emp + 1), 0.5), min_emp, max_emp))
        empleados.append(emp)

    return tamano_seleccionado, np.array(empleados)

def generar_monto_credito(tamanos, n):
    montos = []
    for t in tamanos:
        if t == 'micro':
            mediana = PARAMETROS_CREDITO['monto_mediana_micro']
        elif t == 'pequena':
            mediana = PARAMETROS_CREDITO['monto_mediana_pequena']
        elif t == 'mediana':
            mediana = PARAMETROS_CREDITO['monto_mediana_mediana']
        else:
            mediana = PARAMETROS_CREDITO['monto_mediana_grande']

        monto = np.random.lognormal(np.log(mediana), 0.7)
        monto = np.clip(monto, PARAMETROS_CREDITO['monto_min'],
                         PARAMETROS_CREDITO['monto_max'])
        montos.append(round(monto, -3))

    return np.array(montos)

def generar_plazo(n):
    plazos_tipicos = [6, 12, 18, 24, 36, 48, 60, 84, 120]
    pesos = [0.05, 0.15, 0.10, 0.25, 0.20, 0.12, 0.08, 0.03, 0.02]
    return np.random.choice(plazos_tipicos, size=n, p=pesos)

def generar_garantia_nafin(n, con_programa_nafin):
    garantias = []
    for tiene_nafin in con_programa_nafin:
        if tiene_nafin:
            garantia = np.random.uniform(
                PARAMETROS_CREDITO['garantia_nafin_min'],
                PARAMETROS_CREDITO['garantia_nafin_max']
            )
        else:
            garantia = 0.0
        garantias.append(round(garantia, 2))
    return np.array(garantias)

def generar_fecha_aprobacion(n, fecha_inicio='2015-01-01', fecha_fin='2024-12-31'):
    start = datetime.strptime(fecha_inicio, '%Y-%m-%d')
    end = datetime.strptime(fecha_fin, '%Y-%m-%d')
    delta = (end - start).days

    fechas = []
    for _ in range(n):
        random_days = np.random.randint(0, delta)
        fecha = start + timedelta(days=random_days)
        fechas.append(fecha)
    return fechas

def calcular_probabilidad_default(row):
    tasa_base = SECTORES_SCIAN.get(row['SCIAN'], {}).get('tasa_default', 0.06)
    ajustes = 0

    if row['NewExist'] == 1: ajustes += 0.03
    if row['NoEmp'] <= 5: ajustes += 0.02
    elif row['NoEmp'] <= 10: ajustes += 0.01
    elif row['NoEmp'] > 100: ajustes -= 0.02

    ratio_monto_emp = row['GrAppv'] / (row['NoEmp'] * 100_000)
    if ratio_monto_emp > 5: ajustes += 0.03

    if row['Portion'] > 0.5: ajustes -= 0.015
    if row['Term'] > 60: ajustes += 0.02

    if row['UrbanRural'] == 2: ajustes += 0.015
    if row['Recession'] == 1: ajustes += 0.04

    prob_default = np.clip(tasa_base + ajustes, 0.01, 0.25)

    return prob_default

def generar_default_y_perdida(df):
    df['prob_default'] = df.apply(calcular_probabilidad_default, axis=1)
    df['Default'] = (np.random.random(len(df)) < df['prob_default']).astype(int)
    df['ChgOffPrinGr'] = 0.0

    mask_default = df['Default'] == 1

    for idx in df[mask_default].index:
        tiene_nafin = df.loc[idx, 'Portion'] > 0
        tiene_inmobiliario = df.loc[idx, 'RealEstate'] == 1

        if tiene_nafin:
            lgd = PARAMETROS_CREDITO['lgd_con_garantia_nafin']
        elif tiene_inmobiliario:
            lgd = PARAMETROS_CREDITO['lgd_con_garantia_inmob']
        else:
            lgd = PARAMETROS_CREDITO['lgd_sin_garantia']

        lgd_ajustado = np.random.uniform(lgd * 0.7, lgd * 1.3)
        lgd_ajustado = np.clip(lgd_ajustado, 0.1, 1.0)

        saldo_pendiente_pct = np.random.uniform(0.3, 0.9)
        perdida = df.loc[idx, 'GrAppv'] * lgd_ajustado * saldo_pendiente_pct
        df.loc[idx, 'ChgOffPrinGr'] = round(perdida, 2)

    df.drop('prob_default', axis=1, inplace=True)

    return df

def generar_nombre_empresa(n):
    prefijos = ['Comercializadora', 'Distribuidora', 'Servicios', 'Grupo', 'Industrias', 'Taller', 'Transportes', 'Construcciones', 'Alimentos', 'Materiales', 'Muebles', 'Textiles', 'Ferreterías', 'Abarrotes', 'Refaccionaria', 'Papelería', 'Farmacia', 'Tortillería']
    apellidos = ['González', 'Hernández', 'López', 'Martínez', 'García', 'Rodríguez', 'Sánchez', 'Ramírez', 'Torres', 'Flores', 'Rivera', 'Gómez', 'Díaz', 'Morales', 'Reyes', 'Cruz', 'Ortiz', 'Gutiérrez', 'Chávez', 'Ramos']
    sufijos = ['del Norte', 'del Sur', 'del Centro', 'de Occidente', 'del Bajío', 'de México', 'Nacional', 'Regional', 'Express', '& Asociados', 'S.A. de C.V.', 'S. de R.L.', 'S.C.', '']

    nombres = []
    for _ in range(n):
        if np.random.random() < 0.6:
            nombre = f"{np.random.choice(prefijos)} {np.random.choice(apellidos)} {np.random.choice(sufijos)}".strip()
        else:
            nombre = f"{np.random.choice(apellidos)} {np.random.choice(sufijos)}".strip()
        nombres.append(nombre)

    return nombres

def es_periodo_recesion(fecha):
    if datetime(2020, 4, 1) <= fecha <= datetime(2021, 12, 31):
        return 1
    if datetime(2008, 9, 1) <= fecha <= datetime(2009, 12, 31):
        return 1
    return 0

# ==============================================================================
# FUNCIÓN PRINCIPAL DE GENERACIÓN (CORREGIDA PARA DEVOLVER DATAFRAME)
# ==============================================================================

def generar_dataset_sba_mexico(n_registros=10000, incluir_programa_nafin_pct=0.35):
    """
    Genera dataset sintético de créditos PyME México

    Retorna:
    --------
    pandas.DataFrame con estructura similar al dataset SBA
    """

    print(f"Generando {n_registros:,} registros de préstamos PyME México...")
    print("="*60)

    # Generar características base
    estados = seleccionar_estado(n_registros)
    scian_codes = seleccionar_scian(n_registros)
    bancos = seleccionar_banco(n_registros)
    tamanos, empleados = generar_tamano_empresa(n_registros)

    # Determinar si tiene programa NAFIN
    con_nafin = np.random.random(n_registros) < incluir_programa_nafin_pct

    # Generar montos y plazos
    montos_aprobados = generar_monto_credito(tamanos, n_registros)
    plazos = generar_plazo(n_registros)
    garantias = generar_garantia_nafin(n_registros, con_nafin)

    # Calcular monto garantizado por NAFIN
    montos_nafin = np.round(montos_aprobados * garantias, 2)

    # Generar fechas
    fechas_aprobacion = generar_fecha_aprobacion(n_registros)

    # Calcular fecha de desembolso (1-30 días después de aprobación)
    fechas_desembolso = [f + timedelta(days=np.random.randint(1, 30)) for f in fechas_aprobacion]

    # Variables adicionales
    nuevo_existente = np.random.choice([1, 2], size=n_registros, p=[0.25, 0.75])
    empleos_creados = np.where(nuevo_existente == 1,
                               np.random.poisson(3, n_registros),
                               np.random.poisson(1, n_registros))
    empleos_retenidos = np.random.poisson(empleados * 0.5, n_registros).astype(int)

    urban_rural = np.random.choice([1, 2, 0], size=n_registros, p=[0.75, 0.20, 0.05])
    rev_line_cr = np.random.choice(['Y', 'N'], size=n_registros, p=[0.15, 0.85])
    real_estate = np.random.choice([0, 1], size=n_registros, p=[0.70, 0.30])

    # Crear DataFrame
    df = pd.DataFrame({
        'LoanNr_ChkDgt': generar_id_prestamo(n_registros),
        'Name': generar_nombre_empresa(n_registros),
        'State': estados,
        'City': ['Ciudad_' + str(i) for i in range(n_registros)],
        'Zip': [str(np.random.randint(10000, 99999)) for _ in range(n_registros)],
        'Bank': bancos,
        'BankState': [BANCOS_MEXICO[b]['estado_matriz'] for b in bancos],
        'SCIAN': scian_codes,
        'ApprovalDate': fechas_aprobacion,
        'ApprovalFY': [f.year for f in fechas_aprobacion],
        'DisbursementDate': fechas_desembolso,
        'Term': plazos,
        'NoEmp': empleados,
        'NewExist': nuevo_existente,
        'CreateJob': empleos_creados,
        'RetainedJob': empleos_retenidos,
        'FranchiseCode': np.where(np.random.random(n_registros) < 0.05,
                                 np.random.randint(1, 100, n_registros), 0),
        'UrbanRural': urban_rural,
        'RevLineCr': rev_line_cr,
        'DisbursementGross': montos_aprobados,
        'GrAppv': montos_aprobados,
        'NAFIN_Appv': montos_nafin,
        'New': (nuevo_existente == 1).astype(int),
        'RealEstate': real_estate,
        'Portion': garantias,
        'daysterm': plazos * 30,
    })

    # Agregar indicador de recesión
    df['Recession'] = df['ApprovalDate'].apply(es_periodo_recesion)

    # Generar defaults y pérdidas
    print("Calculando probabilidades de default y pérdidas...")
    df = generar_default_y_perdida(df)

    # Agregar MIS_Status (equivalente SBA)
    df['MIS_Status'] = np.where(df['Default'] == 1, 'CHGOFF', 'P I F')

    # Fecha de charge-off (solo para defaults)
    df['ChgOffDate'] = pd.NaT
    mask_default = df['Default'] == 1

    for idx in df[mask_default].index:
        dias_a_default = np.random.randint(180, 730)
        df.loc[idx, 'ChgOffDate'] = df.loc[idx, 'DisbursementDate'] + timedelta(days=dias_a_default)

    # Balance Gross (saldo pendiente al momento de default)
    df['BalanceGross'] = np.where(
        df['Default'] == 1,
        df['GrAppv'] * np.random.uniform(0.3, 0.9, n_registros),
        0
    )

    # Ordenar columnas (se omite por brevedad en este ejemplo, pero se puede hacer)

    # Estadísticas resumen (simplificadas)
    print("\n" + "="*60)
    print("ESTADÍSTICAS DEL DATASET GENERADO (SIN NORMALIZAR)")
    print("="*60)
    print(f"Total de préstamos: {len(df):,}")
    print(f"Tasa de default: {df['Default'].mean()*100:.2f}%")
    print(f"Monto promedio: ${df['GrAppv'].mean():,.0f} MXN")

    return df

# ==============================================================================
# FUNCIÓN DE NORMALIZACIÓN DE VARIABLES (PARA DISTRIBUCIÓN NORMAL)
# ==============================================================================

def normalizar_variables_log(df):
    """
    Aplica transformación logarítmica y raíz cuadrada a variables numéricas
    asimétricas para acercarlas a una distribución normal.
    Luego aplica estandarización (Z-Score).
    """
    print("\n" + "="*60)
    print("APLICANDO TRANSFORMACIÓN LOGARÍTMICA Y ESTANDARIZACIÓN")
    print("="*60)

    # 1. Transformación Logarítmica (para variables Log-Normal y colas largas)
    # Usamos log1p (log(x+1)) para manejar valores de 0

    vars_log1p = ['NAFIN_Appv', 'ChgOffPrinGr', 'BalanceGross']
    vars_log = ['GrAppv', 'DisbursementGross', 'NoEmp', 'Term', 'daysterm']

    for var in vars_log + vars_log1p:
        df[var] = df[var].astype(float)
        new_var_name = f'Log_{var}'

        if var in vars_log:
            # log(x) para valores que no tienen ceros o el cero es atípico
            df[new_var_name] = np.log(df[var])
        elif var in vars_log1p:
            # log(x + 1) para variables con muchos ceros (Montos NAFIN, Pérdida, Saldo)
            df[new_var_name] = np.log1p(df[var])

    # 2. Transformación de Raíz Cuadrada (para variables de conteo Poisson)
    df['Sqrt_CreateJob'] = np.sqrt(df['CreateJob'])
    df['Sqrt_RetainedJob'] = np.sqrt(df['RetainedJob'])

    # 3. Estandarización (Z-Score)
    # Centrar la media en 0 y la desviación estándar en 1 para las variables transformadas.

    vars_to_standardize = [col for col in df.columns if col.startswith(('Log_', 'Sqrt_'))]

    for var_std in vars_to_standardize:
        mean_val = df[var_std][np.isfinite(df[var_std])].mean()
        std_val = df[var_std][np.isfinite(df[var_std])].std()

        if std_val > 0:
            df[f'Z_{var_std}'] = (df[var_std] - mean_val) / std_val
        else:
            df[f'Z_{var_std}'] = 0

    print("Variables transformadas (Log_X, Sqrt_X) y estandarizadas (Z_Log_X, Z_Sqrt_X) añadidas.")

    # El resultado Z_Log_X es la variable más cercana a la Normal
    #
    # [Image of Normal Distribution Bell Curve] <-- This line is causing the SyntaxError


    return df

# ==============================================================================
# FUNCIÓN DE EJECUCIÓN (UNIFICADA Y CORREGIDA)
# ==============================================================================

def generar_dataset_sba_mexico_normalizado(n_registros=10000, incluir_programa_nafin_pct=0.35):
    """
    Función principal que genera el dataset y luego normaliza las variables.
    """
    # 1. Generar el dataset base (llamada correcta)
    df = generar_dataset_sba_mexico(n_registros=n_registros,
                                    incluir_programa_nafin_pct=incluir_programa_nafin_pct)

    # 2. Aplicar la normalización
    df = normalizar_variables_log(df)

    return df

def guardar_dataset(df, ruta='sba_mexico_sintetico_normalizado.csv'):
    df.to_csv(ruta, index=False)
    print(f"\nDataset final guardado en: {ruta}")
    return ruta

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    # Generar dataset con normalización y un alto número de registros
    df_normalizado = generar_dataset_sba_mexico_normalizado(n_registros=50000)

    # Guardar dataset (incluyendo las nuevas columnas normalizadas)
    guardar_dataset(df_normalizado, 'sba_mexico_sintetico_normalizado.csv')

    # Mostrar las primeras filas y los tipos de datos de las nuevas columnas
    print("\n" + "="*60)
    print("VISTA PREVIA DE COLUMNAS TRANSFORMADAS Y ESTANDARIZADAS")
    print("="*60)
    print(df_normalizado[['GrAppv', 'Log_GrAppv', 'Z_Log_GrAppv',
                         'NoEmp', 'Log_NoEmp', 'Z_Log_NoEmp',
                         'ChgOffPrinGr', 'Log_ChgOffPrinGr', 'Z_Log_ChgOffPrinGr',
                         'CreateJob', 'Sqrt_CreateJob', 'Z_Sqrt_CreateJob']].head())

    print("\n¡Ejecución completada! Las variables 'Z_Log_X' y 'Z_Sqrt_X' están listas para el modelado.")