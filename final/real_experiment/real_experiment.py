import libdtw as lib
from datetime import datetime
import matplotlib.pyplot as plt
# Filtering data from first half of 2017

raw_data = lib.load_data(1000)
_ = raw_data.pop('reference')

data2017 = dict((k, v) for k, v in raw_data.items() if datetime.strptime(v[0]['start'][:4], '%Y') == datetime.strptime('2017', '%Y'))


plt.plot()


