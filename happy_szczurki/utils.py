import numpy
import torch

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.

    @source: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

class Table:
    class Column:
        ALIGNMENT = {
            'left': str.ljust,
            'center': str.center,
            'right': str.rjust,
        }

        @property
        def min_width(self):
            return self.width
        
        @min_width.setter
        def min_width(self, value):
            self.width = max(self.width, value)

        def __init__(self, name):
            self.name = str(name)
            self.width = len(self.name)
            self.align = 'left'

        def fit(self, text):
            width = max(len(line) for line in text.split("\n"))
            self.width = max(self.width, width)

        def format(self, text):
            return self.ALIGNMENT[self.align](text, self.width)[:self.width]


    def __init__(self, header, title=None):
        self.columns = [self.Column(key) for key in header]
        self.data = []
        self.title = title

    @staticmethod
    def format(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def insert_separator(self):
        self.data.append(None)

    def __lshift__(self, other):
        other = [self.format(value) for value in other]
        assert len(other) == len(self.columns)
        self.data.append(other)

        for column, value in zip(self.columns, other):
            column.fit(value)

    def _border_line(self, left, joiner, right, line='━'):
        res = [left]
        for i, column in enumerate(self.columns):
            if i:
                res.append(joiner)
            res.append(line*(column.width+2))
        res.append(right)
        return ''.join(res)

    def __str__(self):
        return self.render()

    def render(self, format='text', inner_frame=False, stripped=True):
        res = []
        if self.title:
            total_width = len(self.columns) + 1 + sum(column.width + 2 for column in self.columns)
            res.append(str.center(self.title, total_width))
    
        res.append(self._border_line('┏', '┳', '┓'))

        line = '┃'
        for column in self.columns:
            text = column.format(column.name)
            line += f" {text} ┃"
        res.append(line)
        res.append(self._border_line('┣', '╋', '┫'))

        for i, row in enumerate(self.data):
            if row is None:
                res.append(self._border_line('┠', '╂', '┨', '╌'))
                continue

            if inner_frame and i:
                res.append(self._border_line('┠', '╂', '┨', '╌'))
                # res.append(self._border_line('┃', '┃', '┃', ' '))
            

            row_lines = [cell.split("\n") for cell in row]
            lines_count = max(len(cell) for cell in row_lines)

            color = 236 if i%2 else 232
            for line_no in range(lines_count):
                line = f'\033[48;5;{color}m' if stripped else ''
                line += '┃'
                for column, cell in zip(self.columns, row_lines):
                    value = cell[line_no] if line_no < len(cell) else ''
                    line += " %s ┃" % column.format(value)
                if stripped:
                    line += '\033[0m'
                res.append(line)
            # res.append(f'\033[48;5;{color}m\033[30m' + self._border_line('┃', '┃', '┃', '▄' if i%2 else '▀') + '\033[0m')
            # res.append(f'\033[48;5;{color}m' + self._border_line('┃', '┃', '┃', ' ') + '\033[0m')

        res.append(self._border_line('┗', '┻', '┛'))
        res.append('')
        return "\n".join(res)

    
def calculate_output_sizes(input_size, layers):
    results = []
    data = torch.zeros((1, ) + input_size)

    for layer in layers:
        data = layer(data)
        output_size = data.shape[1:]
        results.append(list(output_size))

    return results