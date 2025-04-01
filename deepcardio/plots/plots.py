import matplotlib.pyplot as plt
from pathlib import Path
current_path = Path(__file__).parent

def apply_plotstyle():
    style_path = current_path / 'paper.mplstyle'
    style_path = style_path.resolve()
    plt.style.use(str(style_path))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return color_cycle