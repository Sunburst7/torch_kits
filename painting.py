import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

def draw_categorical_confusion(data: np.array, xlabel: list, ylabel: list, title:str, labels: Optional[List], pathname=os.getcwd()):
    assert len(xlabel) == len(ylabel)
    fig, ax = plt.subplots()
    im = ax.imshow(data)

    ax.set_xticks(np.arange(len(xlabel)), labels=xlabel)
    ax.set_yticks(np.arange(len(ylabel)), labels=ylabel)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    for i in range(len(xlabel)):
        for j in range(len(ylabel)):
            text = ax.text(i, j, data[i, j],
                        ha="center", va="center", color="w")

    ax.set_title(title)
    if labels:
        assert len(labels) == 2
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    fig.savefig(pathname)