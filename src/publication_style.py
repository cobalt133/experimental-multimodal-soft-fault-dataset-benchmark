"""Final-size typography and the author's four-sided, inward-tick convention."""
import matplotlib.pyplot as plt
from matplotlib.text import Text

def finish(fig):
    # Text floor anticipates reduction of a 7.4-inch chart to the manuscript's width.
    floor=10.0*max(1,fig.get_figwidth()/7.4)
    for ax in fig.axes:
        if not ax.axison:continue
        for s in ax.spines.values():s.set_visible(True);s.set_linewidth(.7);s.set_color('#333333')
        ax.tick_params(axis='both',which='both',direction='in',top=True,right=True,pad=3)
        ax.tick_params(which='major',length=3,width=.6)
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text():
            t.set_fontsize(floor)
            t.set_fontweight('normal')
            t.set_fontfamily('Times New Roman')
            t.set_math_fontfamily('stix')
    return fig
