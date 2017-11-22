import curses
import ag

def genetico(info_win, data_win):
    POBLACION = 10
    MAX_ITERACIONES = 10000
    PORCENTAJE_MUTACION = 0.01
    poblacion = ag.Poblacion(POBLACION, ag.generador, ag.fitness, ag.f_reproduccion, ag.f_mutacion, PORCENTAJE_MUTACION)
    data_win.addstr(0,0,"GENERACIONES:")
    for i in range(0,MAX_ITERACIONES):
        for j in range(0, len(poblacion.poblacion)):
            e = poblacion.poblacion[j]
            data_win.addstr(j+1, 1, "{} ---> {}".format(e, e.calcular_fitness()))
            data_win.refresh()

        info_win.clear()
        info_win.addstr(0,1, "Numero de Generaciones:{}".format(i+1))
        info_win.addstr(1,1, "Promedio Fitness:{}".format(poblacion.promedio_fitness()))
        info_win.refresh()

        poblacion.seleccion()
        poblacion.reproduccion()
        poblacion.mutar()
    
    for j in range(0, len(poblacion.poblacion)):
        e = poblacion.poblacion[j]
        data_win.addstr(j+1, 1, "{} ---> {}".format(e, e.calcular_fitness()))
        data_win.refresh()
    info_win.clear()
    info_win.addstr(0,1, "Numero de Generaciones:{}".format(i+1))
    info_win.addstr(1,1, "Promedio Fitness:{}".format(poblacion.promedio_fitness()))
    info_win.refresh()


def main(stdscr):
    # Clear screen
    stdscr.clear()

    info_win = curses.newwin(curses.LINES,int(curses.COLS/2),0,0)

    data_win = curses.newwin(curses.LINES,int(curses.COLS/2),0,int(curses.COLS/2))

    data_win.border(0)
    info_win.refresh()
    genetico(info_win, data_win)
    #data_win.getkey()
    info_win.getkey()

curses.wrapper(main)
