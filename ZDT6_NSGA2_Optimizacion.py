Abreviatura = ["MP","PP","MM","GG","MG"]
Preferencia = ["Muy Pequeño","Pequeño","Medio","Grande","Muy Grande"]
Duracion    = [0,0,0,0,0]
MinimoF1    = [0,0,0,0,0]
MinimoF2    = [0,0,0,0,0]
MaximoF1    = [0,0,0,0,0]
MaximoF2    = [0,0,0,0,0]
import time

for i in range (0,5):

    start_time = time.time()

    # Archivos y Parámetros
    Criterio               = Preferencia[i]
    Archivo_Simulacion     = f"Simulacion_ZDT6_{Abreviatura[i]}.xlsx"
    Archivo_Semilla        = f"Semilla_ZDT6_{Abreviatura[i]}_VD.json"
    Nombre_Salida          = f"Salida_NSGAII_ZDT6_{Abreviatura[i]}.xlsx"
    Nombre_Soluciones      = f"Soluciones_NSGAII_ZDT6_{Abreviatura[i]}.xlsx"

    import pandas as pd

    #La siguiente instrucción invoca la Simulación


    # Lee el archivo Excel y carga el DataFrame
    Simulacion = pd.read_excel(Archivo_Simulacion)

    #print(Simulacion)

    valor_f1mins = Simulacion['f1'].min()
    valor_f2mins = Simulacion['f2'].min()
    valor_f1maxs = Simulacion['f1'].max()
    valor_f2maxs = Simulacion['f2'].max()

    #La siguiente instrucción invoca la Semilla

    import json

    # Lee el archivo JSON y carga los datos en un diccionario
    with open(Archivo_Semilla, 'r') as archivo_json:
        Semilla = json.load(archivo_json)

    import skfuzzy as fuzz
    import skfuzzy.control as ctrl
    # SID
    def Sistema_Inf_Difuso(f1,f2):
        # Este SID recibe valores normalizados
        # Universos del discurso
        UX1 = np.linspace(start=0, stop=1, num=101)     # De la variable lingüística X1 que corresponde a f1
        UX2 = np.linspace(start=0, stop=1, num=101)     # De la variable lingüística X2 que corresponde a f2

        # Definicion de delta: corresponde a la amplitud que tienen las funciones semánticas de los términos lingüísticos
        # de las variables lingüísticas
        delta1 = 1/7 * 1
        delta2 = 1/7 * 1

        a_1 = 1 * delta1
        b_1 = 2 * delta1
        c_1 = 3 * delta1
        d_1 = 4 * delta1
        e_1 = 5 * delta1
        f_1 = 6 * delta1

        a_2 = 1 * delta2
        b_2 = 2 * delta2
        c_2 = 3 * delta2
        d_2 = 4 * delta2
        e_2 = 5 * delta2
        f_2 = 6 * delta2

        # Definicion de vértices de las funciones semánticas del objetivo 1
        verticesMA_1 = [  0,      0,    a_1,    b_1]
        vertices_A_1 = [a_1,    b_1,    c_1,    d_1]
        vertices_B_1 = [c_1,    d_1,    e_1,    f_1]
        verticesMB_1 = [e_1,    f_1,      1,      1]

        # Definicion de vértices de las funciones semánticas del objetivo 2
        verticesMA_2 = [  0,      0,    a_2,    b_2]
        vertices_A_2 = [a_2,    b_2,    c_2,    d_2]
        vertices_B_2 = [c_2,    d_2,    e_2,    f_2]
        verticesMB_2 = [e_2,    f_2,      1,      1]

        # Definicion de funciones semánticas trapezoidales del objetivo 1
        X1 = ctrl.Antecedent(UX1, 'X1')
        X1['Muy Alto']  = fuzz.trapmf(UX1, verticesMA_1)
        X1['Alto']      = fuzz.trapmf(UX1, vertices_A_1)
        X1['Bajo']      = fuzz.trapmf(UX1, vertices_B_1)
        X1['Muy Bajo']  = fuzz.trapmf(UX1, verticesMB_1)

        # Definicion de funciones semánticas trapezoidales del objetivo 2
        X2 = ctrl.Antecedent(UX2, 'X2')
        X2['Muy Alto']  = fuzz.trapmf(UX2, verticesMA_2)
        X2['Alto']      = fuzz.trapmf(UX2, vertices_A_2)
        X2['Bajo']      = fuzz.trapmf(UX2, vertices_B_2)
        X2['Muy Bajo']  = fuzz.trapmf(UX2, verticesMB_2)

        # Universos del discurso de la variable Y
        UY = np.linspace(start=0, stop=100, num=101)  # De la variable lingüística Y que corresponde a las preferencias

        # Definicion de delta: corresponde a la amplitud que tienen las funciones semánticas de los términos lingüísticos
        # de la variable lingüística Y
        deltaY = 100/9

        # Definicion de vértices de las funciones semánticas de las preferencias
        verticesMP = [0,        0,          1*deltaY,   2*deltaY]
        vertices_P = [1*deltaY, 2*deltaY,   3*deltaY,   4*deltaY]
        vertices_M = [3*deltaY, 4*deltaY,   5*deltaY,   6*deltaY]
        vertices_G = [5*deltaY, 6*deltaY,   7*deltaY,   8*deltaY]
        verticesMG = [7*deltaY, 8*deltaY,   100,        100]

        # Definicion de funciones semánticas trapezoidales de la variable Y
        Y = ctrl.Consequent(UY, 'Y', defuzzify_method = 'centroid')
        Y['Muy Pequeño']    = fuzz.trapmf(UY, verticesMP)
        Y['Pequeño']        = fuzz.trapmf(UY, vertices_P)
        Y['Medio']          = fuzz.trapmf(UY, vertices_M)
        Y['Grande']         = fuzz.trapmf(UY, vertices_G)
        Y['Muy Grande']     = fuzz.trapmf(UY, verticesMG)

        YMP = fuzz.trapmf(UY, verticesMP)
        Y_P = fuzz.trapmf(UY, vertices_P)
        Y_M = fuzz.trapmf(UY, vertices_M)
        Y_G = fuzz.trapmf(UY, vertices_G)
        YMG = fuzz.trapmf(UY, verticesMG)

        # Mostrar gráficas de variables lingüísticas
        #X1.view()
        #X2.view()
        #Y.view()
        #plt.show()

        # Construccion de reglas del SID
        rule01 = ctrl.Rule(antecedent=(X1['Muy Alto'] & X2['Muy Alto']), consequent=Y['Medio'],      label="rule 01")
        rule02 = ctrl.Rule(antecedent=(X1['Muy Alto'] & X2['Alto']),     consequent=Y['Grande'],     label="rule 02")
        rule03 = ctrl.Rule(antecedent=(X1['Muy Alto'] & X2['Bajo']),     consequent=Y['Grande'],     label="rule 03")
        rule04 = ctrl.Rule(antecedent=(X1['Muy Alto'] & X2['Muy Bajo']), consequent=Y['Muy Grande'], label="rule 04")
        rule05 = ctrl.Rule(antecedent=(X1['Alto']     & X2['Muy Alto']), consequent=Y['Pequeño'],    label="rule 05")
        rule06 = ctrl.Rule(antecedent=(X1['Alto']     & X2['Alto']),     consequent=Y['Medio'],      label="rule 06")
        rule07 = ctrl.Rule(antecedent=(X1['Alto']     & X2['Bajo']),     consequent=Y['Grande'],     label="rule 07")
        rule08 = ctrl.Rule(antecedent=(X1['Alto']     & X2['Muy Bajo']), consequent=Y['Muy Grande'], label="rule 08")
        rule09 = ctrl.Rule(antecedent=(X1['Bajo']     & X2['Muy Alto']), consequent=Y['Pequeño'],    label="rule 09")
        rule10 = ctrl.Rule(antecedent=(X1['Bajo']     & X2['Alto']),     consequent=Y['Pequeño'],    label="rule 10")
        rule11 = ctrl.Rule(antecedent=(X1['Bajo']     & X2['Bajo']),     consequent=Y['Medio'],      label="rule 11")
        rule12 = ctrl.Rule(antecedent=(X1['Bajo']     & X2['Muy Bajo']), consequent=Y['Grande'],     label="rule 12")
        rule13 = ctrl.Rule(antecedent=(X1['Muy Bajo'] & X2['Muy Alto']), consequent=Y['Muy Pequeño'],label="rule 13")
        rule14 = ctrl.Rule(antecedent=(X1['Muy Bajo'] & X2['Alto']),     consequent=Y['Muy Pequeño'],label="rule 14")
        rule15 = ctrl.Rule(antecedent=(X1['Muy Bajo'] & X2['Bajo']),     consequent=Y['Pequeño'],    label="rule 15")
        rule16 = ctrl.Rule(antecedent=(X1['Muy Bajo'] & X2['Muy Bajo']), consequent=Y['Medio'],      label="rule 16")

        # Instrucciones para el motor de inferencia:
        system = ctrl.ControlSystem(rules=[rule01, rule02, rule03, rule04, rule05, rule06, rule07, rule08,
                                           rule09, rule10, rule11, rule12, rule13, rule14, rule15, rule16])
        sim = ctrl.ControlSystemSimulation(system);

        # Controlador difuso
        sim.input['X1'] = f1
        sim.input['X2'] = f2
        sim.compute()
        y = sim.output['Y']

        # Esta instrucción añade una columna adicional a "tabla_datos", esa columna adicional contiene los valores de
        # y que salen del SID
        CMP = fuzz.interp_membership(UY, YMP, y)
        C_P = fuzz.interp_membership(UY, Y_P, y)
        C_M = fuzz.interp_membership(UY, Y_M, y)
        C_G = fuzz.interp_membership(UY, Y_G, y)
        CMG = fuzz.interp_membership(UY, YMG, y)

        # Luego de calcular el valor de pertenencia de cada Y, guarda el conjunto difuso al que pertenece más
        if CMP > C_P and CMP > C_M and CMP > C_G and CMP > CMG:
            Conj_PertenY = 'Muy Pequeño'
        elif C_P > CMP and C_P > C_M and C_P > C_G and C_P > CMG:
            Conj_PertenY = 'Pequeño'
        elif C_M > CMP and C_M > C_P and C_M > C_G and C_M > CMG:
            Conj_PertenY = 'Medio'
        elif C_G > CMP and C_G > C_P and C_G > C_M and C_G > CMG:
            Conj_PertenY = 'Grande'
        else:
            Conj_PertenY = 'Muy Grande'

        return y, Conj_PertenY
    # El SID entrega dos valores que se llaman 'y', 'Conjunto de Y

    from pymoo.core.callback import Callback

    # La instricción Callback es una clase de Pymoo que ayuda a detectar información de las ejecuciones del algoritmo
    # Callback tiene un procedimiento que se llama notify que es capaz de encontrar información como las generaciones y el valor máximo y mínimo de cada población
    class CustomCallback(Callback):
        def __init__(self):
            super().__init__()

        def notify(self, algorithm):
            global valor_f1mins
            global valor_f2mins
            global valor_f1maxs
            global valor_f2maxs
            ngeneracion = algorithm.n_gen
            solutions = algorithm.pop.get("F")
            valor_f1minGEN = min([fila[0] for fila in solutions])
            valor_f1maxGEN = max([fila[0] for fila in solutions])
            valor_f2minGEN = min([fila[1] for fila in solutions])
            valor_f2maxGEN = max([fila[1] for fila in solutions])

            valor_f1mins = min(valor_f1mins, valor_f1minGEN)
            valor_f2mins = min(valor_f2mins, valor_f2minGEN)
            valor_f1maxs = max(valor_f1maxs, valor_f1maxGEN)
            valor_f2maxs = max(valor_f2maxs, valor_f2maxGEN)
            print(f"El algoritmo va en la generación: {ngeneracion} segundos")
            return

    # Crea una instancia de Callback
    custom_callback = CustomCallback()

    import numpy as np
    from pymoo.core.problem import ElementwiseProblem

    # El siguiente código Modela el Problema ZDT6 con restricción
    class MyProblemZDT6(ElementwiseProblem):

        def __init__(self):
            super().__init__(n_var=10,
                             n_obj=2,
                             n_ieq_constr=1,
                             xl=np.array([0,0,0,0,0,0,0,0,0,0]),
                             xu=np.array([1,1,1,1,1,1,1,1,1,1]))

        def _evaluate(self, x, out, *args, **kwargs):
            f1 = 1 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0])**6
            g = 1 + 9 * (np.sum(x[1:]) / (self.n_var - 1))**0.25
            f2 = 1 - (f1 / g)**2

            # Normaliza los valores de f1 y f2 con los valores máximos y mínimos
            f1norm = (f1 - valor_f1mins) / (valor_f1maxs - valor_f1mins)
            f2norm = (f2 - valor_f2mins) / (valor_f2maxs - valor_f2mins)

            # Se evaluan los valores normalizados en el SID y se incluyen en la restricción
            if Sistema_Inf_Difuso(f1=f1norm,
                                  f2=f2norm)[1] == Criterio:
                a = 0
            else:
                a = 1

            g1 = a * (sum(x))

            out["F"] = [f1, f2]
            out["G"] = [g1]

    problemZDT6 = MyProblemZDT6()

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize
    from pymoo.core.evaluator import Evaluator
    from pymoo.core.population import Population

    # Esta instrucción lee la semilla para incluirla en la optimización
    Pop_Semilla = Population.new("X", Semilla)
    Evaluator().eval(problemZDT6, Pop_Semilla)

    # Configura el algoritmo NSGA-II con el Callback personalizado
    algorithm = NSGA2(pop_size=100,
                      n_offsprings=200,
                      sampling=Pop_Semilla,
                      crossover=SBX(prob=0.7, eta=15),
                      mutation=PM(eta=20),
                      eliminate_duplicates=True,
                      callback=custom_callback)

    # Ejecuta la optimización
    from pymoo.termination import get_termination
    termination = get_termination("n_gen", 200)

    import io
    import contextlib

    # Esta instrucción guarda los resultados en una variable output_string que luego será guardada en Excel
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        res = minimize(problemZDT6,
                   algorithm,
                   termination,
                   seed=1,
                   verbose=True)

        output_string=buf.getvalue()

    # Accede a los resultados finales, como la población óptima y las soluciones
    X = res.X
    F = res.F
    Pop = res.pop

    from io import StringIO

    # Estas instrucciones guardan las estadisticas en archivos de Excel
    Salida = StringIO(output_string)
    Salida_Excel = pd.read_csv(Salida, sep='|')
    Salida_Excel.to_excel(Nombre_Salida,index=True)

    soluciones = Pop.get("F")
    columnas_objetivos = ['f1','f2']
    soluciones_tabla = pd.DataFrame(soluciones, columns = columnas_objetivos)

    variables = Pop.get("X")
    columnas_variables = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']
    variables_tabla = pd.DataFrame(variables,columns = columnas_variables)

    soluciones_tabla = pd.concat([variables_tabla,soluciones_tabla],axis=1)

    f1min = valor_f1mins
    f1max = valor_f1maxs
    f2min = valor_f2mins
    f2max = valor_f2maxs

    # Calcula Cotas Ideales y Nadir
    ideal = np.array([f1min,f2min])
    nadir = np.array([f1max,f2max])
    dist_ideal = []
    dist_nadir = []

    for valor_f1, valor_f2 in zip(soluciones_tabla['f1'], soluciones_tabla['f2']):
        punto = np.array([valor_f1, valor_f2])
        dist_ideal.append(np.linalg.norm(ideal - punto))
        dist_nadir.append(np.linalg.norm(nadir - punto))

    # Crea una columna adicional donde cada solución tiene su correspondiente distancia a la cota ideal y a la cota nadir
    soluciones_tabla['dist_ideal']=dist_ideal
    soluciones_tabla['dist_nadir']=dist_nadir

    # Identifica el valor de f1 y f2 más cercana a la cota nadir
    solucion_f1_ideal = soluciones_tabla.loc[soluciones_tabla['dist_ideal'] == soluciones_tabla['dist_ideal'].min(), 'f1'].values[0]
    solucion_f2_ideal = soluciones_tabla.loc[soluciones_tabla['dist_ideal'] == soluciones_tabla['dist_ideal'].min(), 'f2'].values[0]
    # Identifica el valor de f1 y f2 más lejana a la cota nadir
    solucion_f1_nadir = soluciones_tabla.loc[soluciones_tabla['dist_nadir'] == soluciones_tabla['dist_nadir'].max(), 'f1'].values[0]
    solucion_f2_nadir = soluciones_tabla.loc[soluciones_tabla['dist_nadir'] == soluciones_tabla['dist_nadir'].max(), 'f2'].values[0]

    # Encuenta el valor medio entre los valores ideal y nadir
    f1_medio = (solucion_f1_ideal + solucion_f1_nadir) / 2
    f2_medio = (solucion_f2_ideal + solucion_f2_nadir) / 2

    #Encuentra el punto medio resultante
    medio = np.array([f1_medio,f2_medio])
    dist_medio = []

    #Encuentra la distancia de cada solución al punto medio anterior
    for valor_f1, valor_f2 in zip(soluciones_tabla['f1'], soluciones_tabla['f2']):
        punto = np.array([valor_f1, valor_f2])
        dist_medio.append(np.linalg.norm(medio - punto))

    # Crea una columna adicional donde se puede ver el valor de la distancia de cada solución al punto medio
    soluciones_tabla['dist_medio']=dist_medio

    # Encuentra la solución con una menor distancia el punto medio
    solucion_f1_medio = soluciones_tabla.loc[soluciones_tabla['dist_medio'] == soluciones_tabla['dist_medio'].min(), 'f1'].values[0]
    solucion_f2_medio = soluciones_tabla.loc[soluciones_tabla['dist_medio'] == soluciones_tabla['dist_medio'].min(), 'f2'].values[0]

    soluciones_tabla.to_excel(Nombre_Soluciones,index=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    Duracion[i] = elapsed_time
    MinimoF1[i] = valor_f1mins
    MinimoF2[i] = valor_f2mins
    MaximoF1[i] = valor_f1maxs
    MaximoF2[i] = valor_f2maxs
    print(f"Tiempo de ejecución: {elapsed_time} segundos")

Duraciones = {'Preferencia':Preferencia,'Duracion':Duracion,
              'Minimo F1':MinimoF1,'Maximo F1':MaximoF1,
              'Minimo F2':MinimoF2,'Maximo F2':MaximoF2}
Duraciones_Tabla = pd.DataFrame(Duraciones)
Duraciones_Tabla.to_excel('DuracionesZDT6.xlsx',index=True)