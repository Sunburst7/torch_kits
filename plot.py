from matplotlib import pyplot as plt
from typing import List, Union

# 如果X有一个轴，输出True
def has_one_axis(X):
    return (hasattr(X, "ndim") and X.ndim == 1 or  #X有ndim属性且其值为1
            isinstance(X, list) and not hasattr(X[0], "__len__")) # X为列表且其首个元素没有"__len__"属性

def draw_plot(
        x: List, 
        y: List, 
        legend: List[str], 
        labels: List[str]
)-> None: 
    assert len(labels) == 2
    if isinstance(y, list) and not hasattr(y[0], "__len__"): # X为列表且其首个元素没有"__len__"属性
        y = [y]
    plt.figure()
    #plt.title(f'fold-{best_idx} batch_size={batch_size}, lr={learning_rate}')
    if type(y) == List[float]:
        plt.plot(x, y, label=labels[0])
    else:
        for i, y_ in enumerate(y):
            plt.plot(x, y_, labels=labels[i])
    plt.legend(legend)
    plt.xlabel(labels[0])
    plt.ylabel(labels[-1])
    plt.show()
