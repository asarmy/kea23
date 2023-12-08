"""This file defines convenience functions used to create the report plots. """

# Python imports
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Module imports
import plot_definitions as myplot


def add_minor_gridlines(ax_obj, axis):
    """Add formatted minor gridlines to a plot. Note that matplotlib does
    not have parameters to separate major/minor grid properties (see
    https://github.com/matplotlib/matplotlib/issues/13919), so this
    function was created.

    Parameters
    ----------
    ax_obj : variable
        Axis object

    axis : string
        Axis name
        Permissible entries:
            'x'
            'y'
            'both'

    Returns
    -------
    None

    Examples
    --------
    add_minor_gridlines(ax, axis="both")

    Requirements
    ------------
    import matplotlib.pyplot as plt
    """
    plt.minorticks_on()
    ax_obj.grid(which="minor", axis=axis, color="#DDDDDD", lw=0.4, alpha=0.5)


def plot_mag_scaling(dict_of_dataframes):

    figure, ax_obj = plt.subplots()

    x, y = "magnitude", "avg_displ"
    for model, df in dict_of_dataframes.items():
        ax_obj.plot(
            df[x],
            df[y],
            color=myplot.model_colors.get(model),
            linestyle=myplot.model_linestyles.get(model),
            label=myplot.model_labels.get(model),
        )

    ticks = [0.001, 0.01, 0.1, 1, 10, 100]
    ax_obj.set_yticks(ticks)
    ax_obj.set_xticks([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5])
    ax_obj.set(xlim=[5, 8.5], ylim=[0.01, 10], yscale="log")
    ax_obj.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    add_minor_gridlines(ax_obj, axis="both")
    ax_obj.set_xlabel("Magnitude")
    ax_obj.legend(loc="lower right")

    return figure, ax_obj
