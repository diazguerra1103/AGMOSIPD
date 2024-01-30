import time
start_time = time.time()
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

# La siguiente instrucción corre el problema sin SID
#from pymoo.problems.multi.zdt import ZDT6

from pymoo.operators.sampling.rnd import FloatRandomSampling

import numpy as np
from pymoo.core.problem import ElementwiseProblem
class MyProblemPRB1(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         xl=np.array([0.1, 0]),
                         xu=np.array([1, 5]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[0]
        f2 = (1.0 + x[1]) / x[0]

        out["F"] = [f1, f2]


# Esta instrucción llama al problema ZDT6 sin restricción para pintarlo en una misma gráfica con el ZDT6 restringido
#problem = ZDT6()
problem = MyProblemPRB1()


algorithm = NSGA2(pop_size=100,
                       n_offsprings=200,
                       sampling=FloatRandomSampling(),
                       crossover=SBX(prob=0.7, eta=15),
                       mutation=PM(eta=20),
                       eliminate_duplicates=True)

from pymoo.termination import get_termination
termination = get_termination("n_gen", 200)

import io
import contextlib
with io.StringIO() as buf, contextlib.redirect_stdout(buf):
    res = minimize(problem,
                    algorithm,
                    termination,
                    seed=1,
                    verbose=True)

    output_string=buf.getvalue()

X = res.X
F = res.F
Pop = res.pop

from io import StringIO
import pandas as pd

# Estas instrucciones guardan las estadisticas en archivos de Excel
Salida = StringIO(output_string)
Salida_Excel = pd.read_csv(Salida, sep='|')
Salida_Excel.to_excel("Salida_NSGAII_PRB1.xlsx", index=True)

soluciones = Pop.get("F")
columnas_objetivos = ['f1', 'f2']
soluciones_tabla = pd.DataFrame(soluciones, columns=columnas_objetivos)

variables = Pop.get("X")
columnas_variables = ['x1', 'x2'
                    #, 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'
                    #,
                    #  'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
                    #  'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30'
                    ]
variables_tabla = pd.DataFrame(variables, columns=columnas_variables)

soluciones_tabla = pd.concat([variables_tabla, soluciones_tabla], axis=1)

soluciones_tabla.to_excel("Soluciones_NSGAII_PRB1.xlsx",index=True)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo de ejecución: {elapsed_time} segundos")