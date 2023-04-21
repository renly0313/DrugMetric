import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Read the data
train = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/data/anticancer/anticancer_smiles206_prediction_new.csv')

# Set the colors
colors = ["#9c0067", "#0034b4"]

# Prepare the data for QED and ChemAIrank columns
data = [train["ChemAIrank"].values, train["QED"].values]

# Create the violin plot
fig, ax = plt.subplots()
parts = ax.violinplot(data, showmeans=True, showextrema=True, widths=0.8, bw_method='silverman', points=800)

# Set the colors
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')

# Set the extreme and axis colors
parts['cmeans'].set_color('#3f3f3f')
parts['cmaxes'].set_color('#3f3f3f')
parts['cmins'].set_color('#3f3f3f')
parts['cbars'].set_color('#3f3f3f')

# Set the X-axis labels
ax.set_xticks(range(1, len(data) + 1))
ax.set_xticklabels([ "ChemAIrank", "QED"])

# Set the Y-axis range
ax.set_ylim(0, 100)

# Save the figure
plt.savefig("/home/dell/wangzhen/RealQED(2.17)/test/result/violin_plot_anticancer", dpi=300)

# Show the figure
plt.show()

# Calculate and print the mean values for QED and ChemAIrank
print("Mean values for QED and ChemAIrank:")
print("QED: {:.2f}".format(train["QED"].mean()))
print("ChemAIrank: {:.2f}".format(train["ChemAIrank"].mean()))
