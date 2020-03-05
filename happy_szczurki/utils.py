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
    def __init__(self, header, inner_frame=False, stripped=True):
        assert all(isinstance(key, str) for key in header)
        self.column_widths = [len(key) for key in header]
        self.header = header
        self.data = []
        self.inner_frame = inner_frame
        self.stripped = stripped

    def __lshift__(self, other):
        other = [str(value) for value in other]
        assert len(other) == len(self.header)
        self.data.append(other)

        for i in range(len(self.header)):
            width = max(len(line) for line in other[i].split("\n"))
            self.column_widths[i] = max(self.column_widths[i], width)

    def _border_line(self, left, joiner, right, line='━'):
        res = [left]
        for i, width in enumerate(self.column_widths):
            if i: res.append(joiner)
            res.append(line*(width+2))
        res.append(right)
        return ''.join(res)

    def __str__(self):
        res = [self._border_line('┏', '┳', '┓')]

        line = '┃'
        for width, key in zip(self.column_widths, self.header):
            key = key.ljust(width)
            line += f" {key} ┃"
        res.append(line)
        res.append(self._border_line('┣', '╋', '┫'))

        for i, row in enumerate(self.data):
            if self.inner_frame and i:
                res.append(self._border_line('┠', '╂', '┨', '╌'))
                # res.append(self._border_line('┃', '┃', '┃', ' '))

            row_lines = [cell.split("\n") for cell in row]
            lines_count = max(len(cell) for cell in row_lines)

            color = 236 if i%2 else 232
            for line_no in range(lines_count):
                line = f'\033[48;5;{color}m'if self.stripped else ''
                line += '┃'
                for width, cell in zip(self.column_widths, row_lines):
                    value = cell[line_no] if line_no < len(cell) else ''
                    value = value.ljust(width)
                    line += f" {value} ┃"
                if self.stripped:
                    line += '\033[0m'
                res.append(line)
            # res.append(f'\033[48;5;{color}m\033[30m' + self._border_line('┃', '┃', '┃', '▄' if i%2 else '▀') + '\033[0m')
            res.append(f'\033[48;5;{color}m' + self._border_line('┃', '┃', '┃', ' ') + '\033[0m')

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