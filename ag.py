import random
import string
from main import execute_logistic_regression_binomial
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


## Variables globales
class MySingleton:
    instance = None

    @classmethod
    def get_instance(cls):
        if cls.instance is None:

            cls.instance = MySingleton
            cls.conf = SparkConf().setAppName('fifaModel').setMaster('local')
            cls.sc = SparkContext(conf=cls.conf)
            cls.spark = SparkSession(cls.sc)

        return cls.instance

## Clase ADN - Especie
class ADN:
    def __init__(self, generador, fitness, reproduccion, mutacion, porcentaje_mutacion):
        self.generador = generador
        self.fitness = fitness
        self.genes = []
        self.fitness_result = 0
        self.reproduccion = reproduccion
        self.mutacion = mutacion
        self.porcentaje_mutacion = porcentaje_mutacion

    ## se generan los genes de la especie
    ## los genes es un arreglo de 0 y 1.
    ## 0: no se usa el feature
    ## 1: se usa el feature

    ##Por ejemplo [1, 0, 1, 1..0]
    ## features utilizados = ["WF","SM","BC","DRI","MA","SLT","STT","AG",
    ## "REACT","AP","INT","VI","CO", "CRO","SP","LP","ACC","SPEED","STA","STR",
    ## "BA","AGI","JU","HE","SHP","FI","LS","CU","FA","PE","VOL","RA"]

    ## Posicion 0: WF | Puede ser 0 o 1
    ## Posicion 1: SM | Puede ser 0 o 1
    ## Posicion 2: BC | Puede ser 0 o 1
    ## Posicion 3: DRI | Puede ser 0 o 1
    ## Posicion 4: MA | Puede ser 0 o 1
    ## Posicion 5: SLT | Puede ser 0 o 1
    ## Posicion 6: STT | Puede ser 0 o 1
    ## Posicion 7: AG | Puede ser 0 o 1
    ## Posicion 8: REACT | Puede ser 0 o 1
    ## Posicion 9: AP | Puede ser 0 o 1
    ## Posicion 10: INT | Puede ser 0 o 1
    ## Posicion 11: VI | Puede ser 0 o 1
    ## Posicion 12: CO | Puede ser 0 o 1
    ## Posicion 13: CRO | Puede ser 0 o 1
    ## Posicion 14: SP | Puede ser 0 o 1
    ## Posicion 15: LP | Puede ser 0 o 1
    ## Posicion 16: ACC | Puede ser 0 o 1
    ## Posicion 17: SPEED | Puede ser 0 o 1
    ## Posicion 18: STA | Puede ser 0 o 1
    ## Posicion 19: STR | Puede ser 0 o 1
    ## Posicion 20: BA | Puede ser 0 o 1
    ## Posicion 21: AGI | Puede ser 0 o 1
    ## Posicion 22: JU | Puede ser 0 o 1
    ## Posicion 23: HE | Puede ser 0 o 1
    ## Posicion 24: SHP | Puede ser 0 o 1
    ## Posicion 25: FI | Puede ser 0 o 1
    ## Posicion 26: LS | Puede ser 0 o 1
    ## Posicion 27: CU | Puede ser 0 o 1
    ## Posicion 28: FA | Puede ser 0 o 1
    ## Posicion 29: PE | Puede ser 0 o 1
    ## Posicion 30: VOL | Puede ser 0 o 1
    ## Posicion 31: RA | Puede ser 0 o 1

    def generar(self):
        self.genes = self.generador()

    ## se calcula la funcion fitness de la especie
    def calcular_fitness(self):
        self.fitness_result = self.fitness(self.genes)
        return self.fitness_result

    ## se reproduce la especie con su pareja
    def reproducir(self, pareja):
        #funcion reproduccion
        genes_hijo = self.reproduccion(self, pareja)
        especie_hijo = ADN(self.generador, self.fitness, self.reproduccion, self.mutacion, self.porcentaje_mutacion)
        especie_hijo.genes = genes_hijo
        return especie_hijo

    ## se muta la especie segun porcentaje random
    def mutar(self):
        if random.random() < self.porcentaje_mutacion:
            self.genes = self.mutacion(self.genes)
            #lista = list(self.genes)
            #pos = random.randint(0, len(lista)-1)
            #lista[pos] = random.choice(string.ascii_uppercase)
            #self.genes = "".join(lista)

    ## se redefine el metodo __str__
    def __str__(self):
        return " ".join(str(e) for e in self.genes)


##Clase poblacion - conjunto de especies
class Poblacion:
    def __init__(self, cantidad, generador, fitness, f_reproductora, f_mutadora, porcentaje_mutacion):
        self.cantidad = cantidad
        self.poblacion = []
        self.generador = generador
        self.fitness = fitness
        self.fitness_results = []
        self.fitness_total = 0
        self.lista_reproduccion = []
        self.f_reproductora = f_reproductora
        self.f_mutadora = f_mutadora
        self.porcentaje_mutacion = porcentaje_mutacion

        ## Por cada especie, se agrega a la poblacion,
        ## se calcula la funcion de fitness por especie,
        ## se acumula los valores retornados por la funcion de fitness
        ## en un total por poblacion
        for i in range(1,cantidad):
            especie = ADN(self.generador,self.fitness, self.f_reproductora, self.f_mutadora, self.porcentaje_mutacion)
            especie.generar()
            self.poblacion.append(especie)
            fitness_especie = especie.calcular_fitness()
            self.fitness_total = self.fitness_total + fitness_especie
            self.fitness_results.append(fitness_especie)

    ## se seleccionan las especies segun su probabilidad de reproduccion
    ## y se agregan a la lista de reproduccion
    def seleccion(self):
        self.lista_reproduccion = []
        for i in range(0, len(self.poblacion)):
            porcentaje_especie = float(self.fitness_results[i]) / self.fitness_total
            n = int(porcentaje_especie * len(self.poblacion))
            for j in range(0, n):
                self.lista_reproduccion.append(self.poblacion[i])

    ## se reproducen las especies de la poblacion con sus respectivas parejas
    def reproduccion(self):
        self.poblacion = []
        self.fitness_results = []
        self.fitness_total = 0
        for i in range(0, self.cantidad):
            pareja_a = self.lista_reproduccion[random.randint(0, len(self.lista_reproduccion)-1)]
            pareja_b = self.lista_reproduccion[random.randint(0, len(self.lista_reproduccion)-1)]

            hijo = pareja_a.reproducir(pareja_b)
            self.poblacion.append(hijo)
            fitness_especie = hijo.calcular_fitness()
            self.fitness_total = self.fitness_total + fitness_especie
            self.fitness_results.append(fitness_especie)

    ## se mutan las especies de la poblacion
    def mutar(self):
        for e in self.poblacion:
            e.mutar()

    ## se calcula el promedio de la funcion de fitness
    def promedio_fitness(self):
        return float(self.fitness_total) / len(self.fitness_results)

    ## se muestra el valor de la funcion de fitness de cada especie
    def imprimir(self):
        for especie in self.poblacion:
            print("{} {}".format(especie, especie.calcular_fitness()))

## funcion de generacion de especies con arreglos de valores 0 o 1
def generador():
    especie = [random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
              random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
              random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
              random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
              random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
              random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
              random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
              random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
              random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
              random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
              random.randint(0, 1), random.randint(0, 1)]
    return especie

## funcion de fitness, areaUnderROC de regresion logistica binaria de cada especie
def fitness(especie):
    return execute_logistic_regression_binomial(MySingleton.get_instance().sc,
            "data/dataConRating1.csv", especie, MySingleton.get_instance().spark)

## funcion de reproducion entre una especie y su pareja
def f_reproduccion(pareja1, pareja2):
    k = random.randint(0, len(pareja1.genes))
    parte_izq = pareja1.genes[0:k]
    parte_der = pareja2.genes[k:]
    return parte_izq + parte_der

## funcion de mutacion de las especies
def f_mutacion(genes):
    lista = genes
    pos = random.randint(0, len(lista)-1)
    lista[pos] = random.randint(0, 1)
    return lista

def main():
    ## poblacion
    POBLACION = 30

    ## maximo numero de iteraciones
    MAX_ITERACIONES = 30

    ## porcentaje de mutacion
    PORCENTAJE_MUTACION = 0.01

    ## objeto poblacion
    poblacion = Poblacion(POBLACION, generador, fitness, f_reproduccion, f_mutacion, PORCENTAJE_MUTACION)

    ## por cada iteracion, se muestra la poblacion,
    ## se seleccionan, se reproducen las especies y se mutan las especies
    for i in range(0,MAX_ITERACIONES):
        poblacion.imprimir()
        print("({})=======================================".format(i))
        poblacion.seleccion()
        poblacion.reproduccion()
        print("Promedio Fitness: {}".format(poblacion.promedio_fitness()))
        poblacion.mutar()


if __name__ == '__main__':
    main()
