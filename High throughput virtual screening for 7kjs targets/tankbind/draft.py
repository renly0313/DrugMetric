import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('/home/dell/wangzhen/TankBind-main/examples/HTVS/info/info.csv')
df.affinity.hist()
plt.show()
