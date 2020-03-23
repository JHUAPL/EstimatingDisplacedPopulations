"""
Copyright 2020 The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

displaced_dir = "/path/to/data"
bangladesh_dir = os.path.join(displaced_dir, "CampImagery/Bangladesh_Kutupalong")
sandbox_dir = os.path.join(displaced_dir, 'Sandbox')

if not os.path.exists(sandbox_dir):
    os.makedirs(sandbox_dir)

image_dir = os.path.join(bangladesh_dir, "CombinedHistoricalData")
osm_dir = os.path.join(bangladesh_dir, "CombinedHistoricalData/OSMNX_STRUCTURE_SEGMENTS")
shape_dir = os.path.join(bangladesh_dir, "NPMAssessments/MajeeBlockShapefiles")
image_basenames_path = os.path.join(bangladesh_dir, "image_basenames.json")

test_camps = ['camp_11', 'camp_12', 'camp_22_unchiprang']

utm_zone = (46,"N")

tolerance = 0.000015
