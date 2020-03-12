#
# Created by Lukas Lüftinger on 17/02/2019.
#
from pathlib import Path
from typing import Dict, List, Union


CccvType = Union[
    Dict[float, Dict[float, Dict[str, float]]],
    List[Dict[float, Dict[float, Dict[str, float]]]]
]

def compleconta_plot(cccv_results: CccvType,
                     conditions: List[str] = (), each_n: List[int] = None,
                     title: str = "", fontsize: int = 16, figsize=(10, 7),
                     plot_comple: bool = True, plot_conta: bool = True,
                     colors: List = None, save_path: Union[str, Path] = None, **kwargs):
    """
    Plots Compleconta CV result for one or multiple models.
    For perfect completeness and variable contamination
    as well as perfect contamination and variable completeness,
    the resulting mean balanced accuracy over folds is plotted.

    :param cccv_results: a ComplecontaCV result, or list thereof
    :param conditions: A list of condition names associated cccv_results
    :param each_n: A list of sample counts in datasets associated with cccv_results
    :param title: The plot title
    :param fontsize: The fontsize of the plot
    :param figsize: The figure size (tuple of width, height)
    :param plot_comple: Whether to plot completeness
    :param plot_conta: Whether to plot contamination
    :param colors:
    :param save_path: The save path of the plot; if None, display it with plt.show()
    :param kwargs: any further keyword arguments passed to plt.plot()
    :return: None
    """

    if save_path is not None:
        save_path = Path(str(save_path))
        assert not save_path.exists()
        assert Path(save_path.parent).is_dir()
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # must import this after setting backend if we want to save

    if type(cccv_results) is dict:
        cccv_results = [cccv_results, ]

    if plot_comple and plot_conta:
        fig, (com_ax, con_ax) = plt.subplots(1, 2, sharey=True, figsize=figsize)
    else:
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=figsize)
        com_ax, con_ax = ax, ax
    plt.suptitle(title, fontsize=fontsize * 1.5)
    if plot_comple:
        for i, result_dict in enumerate(cccv_results):
            comple = [result_dict[x][0.0] for x in result_dict.keys()]
            x_comple_ticks = list(result_dict.keys())
            y_comple_mean = [x["score_mean"] for x in comple]
            y_comple_lbound = [x["score_mean"] - x["score_sd"] for x in comple]
            y_comple_ubound = [x["score_mean"] + x["score_sd"] for x in comple]
            if colors is None:
                colormap = {}
            else:
                colormap = {"color": colors[i]} if i in colors else {"color": "grey"}
            com_ax.plot(x_comple_ticks, y_comple_mean, **kwargs, **colormap)
            com_ax.fill_between(x_comple_ticks, y_comple_lbound, y_comple_ubound, alpha=0.35, **colormap)
            com_ax.set_ylabel("Mean Balanced Accuracy", fontsize=fontsize)
            com_ax.set_xlabel("Completeness", fontsize=fontsize)
            com_ax.set_xlim([-0.05, 1])
            com_ax.set_ylim([0.5, 1])

    if plot_conta:
        for i, result_dict in enumerate(cccv_results):
            conta = [result_dict[1.0][x] for x in result_dict[1.0].keys()]
            x_conta_ticks = list(result_dict[1.0].keys())
            y_conta_mean = [x["score_mean"] for x in conta]
            y_conta_lbound = [x["score_mean"] - x["score_sd"] for x in conta]
            y_conta_ubound = [x["score_mean"] + x["score_sd"] for x in conta]
            if colors is None:
                colormap = {}
            else:
                colormap = {"color": colors[i]} if i in colors else {"color": "grey"}
            con_ax.plot(x_conta_ticks, y_conta_mean, **kwargs, **colormap)
            con_ax.fill_between(x_conta_ticks, y_conta_lbound, y_conta_ubound, alpha=0.35, **colormap)
            con_ax.set_xlabel("Contamination", fontsize=fontsize)
            con_ax.set_xlim([0, 1.05])

    each_n = [f"(n={y})" for y in each_n] if each_n is not None else ["" for _ in conditions]
    fig.legend([" ".join([x, y]) for x, y in zip(conditions, each_n)], loc=8, prop={"size": fontsize // 1.3})
    plt.subplots_adjust(wspace=0.05, bottom=0.20)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
