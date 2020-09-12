import matplotlib.pyplot as plt
import numpy as np


def graph(
        x,
        y,
        xerr=None,
        yerr=None,
        barwidth=None,
        xlims=None,
        ylims=None,
        xargs='',
        yargs='',
        title='',
        adds=None,
        labels=None,
        clr=None,
        alpha=1,
        need_line='',
        mark='.',
        dots=10,
        figsize=None,
        dpi=300,
        legend_loc='best'
):
    """
    Функция, рисующая n графиков с заданными параметрами.
    params:
    x, y - list of iterable - списки значений по осям x и y соответственно
    xerr, yerr - list of iterable - списки погрешностей по осям x и y соответственно
    barwidth - float - толщина крестов погрешностей
    xlims, ylims - list размера 2 - левая и правая границы области по осям x и y соответственно
    xargs, yargs - str - подписи осей x и y соответственно
    title - str - название графика
    adds - list of str - список того, что мы хотим написать в легенде дополнительно (не к кривым)
    labels - list of str - список названий кривых
    clr - tuple - список цветов кривых (если есть labels)
    alpha - float - прозрачность графика (от 0 до 1)
    need_line - str - тип линии, если ее нужно провести по точкам
    mark - str - тип маркера для точек
    dots - float - размер точек
    figsize - tuple размера 2 - размер конечной картинки
    dpi - float - разрешение конечной картинки (в пикселях)
    legend_loc - str - настройка расположения легенды на конечной картинке
    """
    # if clr is None:
    #     clr = []
    # if labels is None:
    #     labels = []
    # if yerr is None:
    #     yerr = []
    # if xerr is None:
    #     xerr = []
    fig = plt.figure(figsize=figsize, dpi=dpi)  # задаем размер картинки и ее разрешение

    if adds is not None:  # пишем дополнительные пояснения на легенде
        for string in adds:
            plt.plot([], [], ' ', label=string)
    if yerr is not None and xerr is None:  # есть погрешностей по оси y, но нет погрешностей по оси x
        if labels is not None:
            if clr is not None:
                for i in range(len(x)):
                    plt.errorbar(
                        x[i],
                        y[i],
                        yerr=yerr[i],
                        elinewidth=barwidth,
                        alpha=alpha,
                        fmt=mark,
                        ms=dots,
                        label=labels[i],
                        color=clr[i]
                    )
            else:
                for i in range(len(x)):
                    plt.errorbar(
                        x[i],
                        y[i],
                        yerr=yerr[i],
                        elinewidth=barwidth,
                        alpha=alpha,
                        fmt=mark,
                        ms=dots,
                        label=labels[i]
                    )

        else:
            for i in range(len(x)):
                plt.errorbar(
                    x[i],
                    y[i],
                    yerr=yerr[i],
                    elinewidth=barwidth,
                    fmt=mark,
                    ms=dots
                )

    elif xerr is not None and yerr is None:
        if labels is not None:
            if clr is not None:
                for i in range(len(x)):
                    plt.errorbar(
                        x[i],
                        y[i],
                        xerr=xerr[i],
                        elinewidth=barwidth,
                        alpha=alpha,
                        fmt=mark,
                        ms=dots,
                        label=labels[i],
                        color=clr[i]
                    )
            else:
                for i in range(len(x)):
                    plt.errorbar(
                        x[i],
                        y[i],
                        xerr=xerr[i],
                        elinewidth=barwidth,
                        alpha=alpha,
                        fmt=mark,
                        ms=dots,
                        label=labels[i]
                    )

        else:
            for i in range(len(x)):
                plt.errorbar(
                    x[i],
                    y[i],
                    xerr=xerr[i],
                    elinewidth=barwidth,
                    fmt=mark,
                    ms=dots
                )

    elif xerr is not None and yerr is not None:
        if labels is not None:
            if clr is not None:
                for i in range(len(x)):
                    plt.errorbar(
                        x[i],
                        y[i],
                        xerr=xerr[i],
                        yerr=yerr[i],
                        elinewidth=barwidth,
                        alpha=alpha,
                        fmt=mark,
                        ms=dots,
                        label=labels[i],
                        color=clr[i]
                    )
            else:
                for i in range(len(x)):
                    plt.errorbar(
                        x[i],
                        y[i],
                        xerr=xerr[i],
                        yerr=yerr[i],
                        elinewidth=barwidth,
                        alpha=alpha,
                        fmt=mark,
                        ms=dots,
                        label=labels[i]
                    )

        else:
            for i in range(len(x)):
                plt.errorbar(
                    x[i],
                    y[i],
                    xerr=xerr[i],
                    yerr=yerr[i],
                    elinewidth=barwidth,
                    fmt=mark,
                    ms=dots
                )

    else:
        if labels is not None:
            if clr is not None:
                for i in range(len(x)):
                    plt.plot(
                        x[i],
                        y[i],
                        mark + need_line,
                        alpha=alpha,
                        ms=dots,
                        label=labels[i],
                        color=clr[i]
                    )
            else:
                for i in range(len(x)):
                    plt.plot(
                        x[i],
                        y[i],
                        mark + need_line,
                        alpha=alpha,
                        ms=dots,
                        label=labels[i]
                    )

        else:
            for i in range(len(x)):
                plt.plot(
                    x[i],
                    y[i],
                    mark + need_line,
                    alpha=alpha,
                    ms=dots
                )

    plt.title(title)
    plt.legend(loc=legend_loc)
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
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1. / 20. * (ymax - ymin)
    hl = 1. / 20. * (xmax - xmin)
    lw = .1  # axis line width
    ohg = 0.25  # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
    yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height

    # draw x and y axis
    ax.arrow(xmin, ymin, xmax - xmin, 0., fc='k', ec='k', lw=lw,
             head_width=hw, head_length=hl, overhang=ohg,
             length_includes_head=True, clip_on=False, width=1e-5)

    ax.arrow(xmin, ymin, 0., ymax - ymin, fc='k', ec='k', lw=lw,
             head_width=yhw, head_length=yhl, overhang=ohg,
             length_includes_head=True, clip_on=False, width=1e-5)

    # Делаем сетку :)
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')


def MNK(x, y):
    """ The lowest squares method of linearization, approximation y = bx + a

        Input:

        x - list of x values
        y - list of y values

        Output:

        b - slope coefficient of linear approximation y(x)
        a - free coefficient of linear approximation y(x)
        sigma - mean squared error ( y_real - y_approx )
        rel_err_sq - quotient sigma to b squared ( is usable for error calculation )
    """
    k, n, m, q, p = 0., 0., 0., 0., 0.
    for x0, y0 in zip(x, y):
        k += x0 * y0
        n += x0 ** 2
        m += x0
        q += y0
        p += y0 ** 2
    k /= len(x)
    n /= len(x)
    m /= len(x)
    q /= len(x)
    p /= len(x)

    b = (k - m * q) / (n - m ** 2)
    a = q - b * m

    sigma = (np.sqrt(float((p - q ** 2) / (n - m ** 2)) - b ** 2)) / np.sqrt(len(x))

    rel_err_sq = (sigma / b) ** 2

    return b, a, sigma, rel_err_sq


def MNK_0(x, y):
    """
        The lowest squares method of linearization, approximation y = bx

        Input:

        x - list of x values
        y - list of y values

        Output:

        b - slope coefficient of linear approximation y(x)
        sigma - mean squared error ( y_real - y_approx )
        rel_err_sq - quotient sigma to b squared ( is usable for error calculation )
    """
    k, n = 0., 0.
    for x0, y0 in zip(x, y):
        k += y0 * x0
        n += x0 ** 2
    k /= len(x)
    n /= len(x)
    b = k / n

    s = 0
    for x0, y0 in zip(x, y):
        s += (y0 / x0 - b) ** 2
    s /= (len(x) * (len(x) - 1))

    sigma = np.sqrt(s)

    rel_err_sq = (sigma / b) ** 2

    return b, sigma, rel_err_sq
