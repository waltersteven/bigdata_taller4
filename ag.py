import random
import string
from main import execute_logistic_regression_binomial
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


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


class ADN:
    def __init__(self, generador, fitness, reproduccion, mutacion, porcentaje_mutacion):
        self.generador = generador
        self.fitness = fitness
        self.genes = []
        self.fitness_result = 0
        self.reproduccion = reproduccion
        self.mutacion = mutacion
        self.porcentaje_mutacion = porcentaje_mutacion

    def generar(self):
        self.genes = self.generador()

    def calcular_fitness(self):
        self.fitness_result = self.fitness(self.genes)
        return self.fitness_result

    def reproducir(self, pareja):
        #funcion reproduccion
        genes_hijo = self.reproduccion(self, pareja)
        especie_hijo = ADN(self.generador, self.fitness, self.reproduccion, self.mutacion, self.porcentaje_mutacion)
        especie_hijo.genes = genes_hijo
        return especie_hijo

    def mutar(self):
        if random.random() < self.porcentaje_mutacion:
            self.genes = self.mutacion(self.genes)
            #lista = list(self.genes)
            #pos = random.randint(0, len(lista)-1)
            #lista[pos] = random.choice(string.ascii_uppercase)
            #self.genes = "".join(lista)

    def __str__(self):
        return " ".join(str(e) for e in self.genes)

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

        for i in range(1,cantidad):
            especie = ADN(self.generador,self.fitness, self.f_reproductora, self.f_mutadora, self.porcentaje_mutacion)
            especie.generar()
            self.poblacion.append(especie)
            fitness_especie = especie.calcular_fitness()
            self.fitness_total = self.fitness_total + fitness_especie
            self.fitness_results.append(fitness_especie)

    def seleccion(self):
        self.lista_reproduccion = []
        for i in range(0, len(self.poblacion)):
            porcentaje_especie = float(self.fitness_results[i]) / self.fitness_total
            n = int(porcentaje_especie * len(self.poblacion))
            for j in range(0, n):
                self.lista_reproduccion.append(self.poblacion[i])

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

    def mutar(self):
        for e in self.poblacion:
            e.mutar()

    def promedio_fitness(self):
        return float(self.fitness_total) / len(self.fitness_results)

    def imprimir(self):
        for especie in self.poblacion:
            print("{} {}".format(especie, especie.calcular_fitness()))


def generador():
    cadena = [random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
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
    return cadena

def fitness(cadena):
    return execute_logistic_regression_binomial(MySingleton.get_instance().sc,
            "data/dataConRating1.csv", cadena, MySingleton.get_instance().spark)

def f_reproduccion(pareja1, pareja2):
    k = random.randint(0, len(pareja1.genes))
    parte_izq = pareja1.genes[0:k]
    parte_der = pareja2.genes[k:]
    return parte_izq + parte_der

def f_mutacion(genes):
    lista = genes
    pos = random.randint(0, len(lista)-1)
    lista[pos] = random.randint(0, 1)
    return lista

def main():
    POBLACION = 15
    MAX_ITERACIONES = 30
    PORCENTAJE_MUTACION = 0.01
    poblacion = Poblacion(POBLACION, generador, fitness, f_reproduccion, f_mutacion, PORCENTAJE_MUTACION)
    for i in range(0,MAX_ITERACIONES):
        poblacion.imprimir()
        print("({})=======================================".format(i))
        poblacion.seleccion()
        poblacion.reproduccion()
        print("Promedio Fitness: {}".format(poblacion.promedio_fitness()))
        poblacion.mutar()


if __name__ == '__main__':
    main()
