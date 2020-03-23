# Cox's Bazar Refugee Camp Dataset

## IOM Bangladesh - NPM Drone Imagery and Shapefiles

The International Organization for Migration (IOM) Bangladesh - Needs and Population Monitoring (NPM) has produced a number of tools based on its regular data collection activities and drone flights. These tools, such as imagery and shapefiles, can be found on [humdata.org](https://humdata.org) and are the original source for all data in this dataset.

Some data can be previewed here: [https://www.arcgis.com/apps/MapSeries/index.html?appid=1eec7ad29df742938b6470d77c26575a](https://www.arcgis.com/apps/MapSeries/index.html?appid=1eec7ad29df742938b6470d77c26575a)

## Dataset contents

This dataset consists of GeoTIFFs, MBTiles, KMLs, PDFs, and shapefiles. Total dataset size is approximately 51GB. In this work, only the GeoTIFFs and shapefiles are utilized. The additional files are included in the archives provided on [humdata.org](https://humdata.org) and may provide useful information for others.

## How to Download

We have assembled a script and set of URLs to make it easier for individuals to download the data referenced in our work. We hope this makes it easier for individuals to reproduce our work, but also hope that it helps lower the barrier of entry for those interested in developing additional humanitarian tools.

Please be mindful of the bandwidth required to host this data by not unnecessarily or repeatedly downloading the data. To download the imagery and shapefiles run
```
python download.py
```