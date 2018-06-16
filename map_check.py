#!/usr/bin/python3
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cartopy.crs as ccrs

import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

projection  = ccrs.PlateCarree()

fig = plt.figure(figsize=(10,8))
ax  = fig.add_subplot(1,1,1, projection=projection)

ax.set_xlim(-90,-60)
ax.set_ylim(15,30)

ax.coastlines()
ax.gridlines(draw_labels=True)
#
#ax.set_xticks(np.linspace(-180, 180, 5), crs=projection)
#ax.set_yticks(np.linspace(-90, 90, 5), crs=projection)
#lon_formatter = LongitudeFormatter(zero_direction_label=True)
#lat_formatter = LatitudeFormatter()
#ax.xaxis.set_major_formatter(lon_formatter)
#ax.yaxis.set_major_formatter(lat_formatter)

fpath = 'map.png'
fig.savefig(fpath,bbox_inches='tight')
plt.close(fig)
