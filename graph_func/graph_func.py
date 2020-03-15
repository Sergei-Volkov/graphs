import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def graph_err(x, y, xerr=None, yerr=None, xargs='', yargs='', title='', xlims=[], ylims=[], dots=.5, dpi=300):
    """
    Функция, рисующая график с крестами погрешностей с определенными пределами, по умолчанию без пределов
    """
    fig = plt.figure(dpi=dpi)

    plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='r.', ms=dots)

    plt.title(title)
    plt.ylabel(yargs)
    plt.xlabel(xargs)

    if xlims:
        plt.xlim((xlims[0], xlims[1]))
    if ylims:
        plt.ylim((ylims[0], ylims[1]))



    # Делаем стрелочки :)
    ax = fig.add_subplot()
    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['right','top']:
        ax.spines[side].set_visible(False)

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./20.*(ymax-ymin) 
    hl = 1./20.*(xmax-xmin)
    lw = .1 # axis line width
    ohg = 0.25 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height


    # draw x and y axis
    ax.arrow(xmin, ymin, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
         head_width=hw, head_length=hl, overhang = ohg, 
         length_includes_head= True, clip_on = False, width=1e-5) 

    ax.arrow(xmin, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
         head_width=yhw, head_length=yhl, overhang = ohg, 
         length_includes_head= True, clip_on = False, width=1e-5)


    # Делаем сетку :)
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')


def graph(x, y, mark='.', xargs='', yargs='', title='', xlims=[], ylims=[], dots=.5, dpi=300):
    """
    Функция, рисующая график с определенными пределами, по умолчанию без пределов
    """
    fig = plt.figure(dpi=dpi)

    plt.plot(x, y, marker=mark, ms=dots)

    plt.title(title)
    plt.ylabel(yargs)
    plt.xlabel(xargs)

    if xlims:
        plt.xlim((xlims[0], xlims[1]))
    if ylims:
        plt.ylim((ylims[0], ylims[1]))



    # Делаем стрелочки :)
    ax = fig.add_subplot()
    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['right','top']:
        ax.spines[side].set_visible(False)

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./20.*(ymax-ymin) 
    hl = 1./20.*(xmax-xmin)
    lw = .1 # axis line width
    ohg = 0.25 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height


    # draw x and y axis
    ax.arrow(xmin, ymin, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
         head_width=hw, head_length=hl, overhang = ohg, 
         length_includes_head= True, clip_on = False, width=1e-5) 

    ax.arrow(xmin, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
         head_width=yhw, head_length=yhl, overhang = ohg, 
         length_includes_head= True, clip_on = False, width=1e-5)


    # Делаем сетку :)
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')


def scatter(x, y, color='r', mark='.', xargs='', yargs='', title='', xlims=[], ylims=[], dots=.5, dpi=300):
    """
    Функция, рисующая график с определенными пределами, по умолчанию без пределов
    """
    fig = plt.figure(dpi=dpi)

    plt.scatter(x, y, c=color, s=dots, marker=mark)

    plt.title(title)
    plt.ylabel(yargs)
    plt.xlabel(xargs)

    if xlims:
        plt.xlim((xlims[0], xlims[1]))
    if ylims:
        plt.ylim((ylims[0], ylims[1]))



    # Делаем стрелочки :)
    ax = fig.add_subplot()
    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['right','top']:
        ax.spines[side].set_visible(False)

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./20.*(ymax-ymin) 
    hl = 1./20.*(xmax-xmin)
    lw = .1 # axis line width
    ohg = 0.25 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height


    # draw x and y axis
    ax.arrow(xmin, ymin, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
         head_width=hw, head_length=hl, overhang = ohg, 
         length_includes_head= True, clip_on = False, width=1e-5) 

    ax.arrow(xmin, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
         head_width=yhw, head_length=yhl, overhang = ohg, 
         length_includes_head= True, clip_on = False, width=1e-5)


    # Делаем сетку :)
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')


def graph_n(x: list, y: list, mark='.', xargs='', yargs='', adds=[], labels=[], title='', xlims=[], ylims=[], dots=.5, dpi=300):
    """
    Функция, рисующая n графиков с определенными пределами, по умолчанию без пределов
    """
    fig = plt.figure(dpi=dpi)

    if adds:
        for i in range(len(adds)):
            plt.plot([], [], ' ', label=adds[i])
    
    if labels:
        for i in range(len(x)):
            plt.plot(x[i], y[i], marker=mark, ms=dots, label=labels[i])
    else:
        for i in range(len(x)):
            plt.plot(x[i], y[i], marker=mark, ms=dots)
            
            
    plt.title(title)
    plt.legend()
    plt.ylabel(yargs)
    plt.xlabel(xargs)

    if xlims:
        plt.xlim((xlims[0], xlims[1]))
    if ylims:
        plt.ylim((ylims[0], ylims[1]))



    # Делаем стрелочки :)
    ax = fig.add_subplot()
    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['right','top']:
        ax.spines[side].set_visible(False)

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./20.*(ymax-ymin) 
    hl = 1./20.*(xmax-xmin)
    lw = .1 # axis line width
    ohg = 0.25 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height


    # draw x and y axis
    ax.arrow(xmin, ymin, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
         head_width=hw, head_length=hl, overhang = ohg, 
         length_includes_head= True, clip_on = False, width=1e-5) 

    ax.arrow(xmin, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
         head_width=yhw, head_length=yhl, overhang = ohg, 
         length_includes_head= True, clip_on = False, width=1e-5)


    # Делаем сетку :)
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')


def MNK(x, y):
    ''' The lowest squares method of linearization, approximation y = bx + a
        
        Input:
        
        x - list of x values
        y - list of y values
        
        Output:
        
        b - slope coefficient of linear approximation y(x)
        a - free coefficient of linear approximation y(x)
        sigma - mean squared error ( y_real - y_approx )
        rel_err_sq - quotient sigma to b squared ( is usable for error calculation )
    '''
    k = n = m = l = p = 0
    for x0, y0 in zip(x, y):
        k += x0 * y0
        n += x0**2
        m += x0
        l += y0
        p += y0**2
    k /= len(x)
    n /= len(x)
    m /= len(x)
    l /= len(x)
    p /= len(x)
    
    b = (k - m * l)/(n - m**2)
    a = l - b * m
    
    sigma = (np.sqrt(float((p - l**2) / (n - m**2)) - b**2)) / np.sqrt(len(x))
    
    rel_err_sq = (sigma / b)**2
    
    return b, a, sigma, rel_err_sq


def MNK_0(x, y):
    '''
        The lowest squares method of linearization, approximation y = bx
        
        Input:
        
        x - list of x values
        y - list of y values
        
        Output:
        
        b - slope coefficient of linear approximation y(x)
        sigma - mean squared error ( y_real - y_approx )
        rel_err_sq - quotient sigma to b squared ( is usable for error calculation )
    '''
    k = n = 0
    for x0, y0 in zip(x, y):
        k += y0*x0
        n += x0**2
    k /= len(x)
    n /= len(x)
    b = k / n
    
    s = 0
    for x0, y0 in zip(x, y):
        s += (y0 / x0 - b) ** 2
    s /= ( len(x) * (len(x)-1) )
    
    sigma = np.sqrt(s)
    
    rel_err_sq = (sigma / b) ** 2
    
    return b, sigma, rel_err_sq
