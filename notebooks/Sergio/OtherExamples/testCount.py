from pathlib import Path
import os

data_dir = Path(os.getcwd() + "/../../../data/labelled_images/")

import pathlib
initial_count = 0
for specificClass in data_dir.iterdir():
   print(str(specificClass))
   initial_count = 0
   for elementFile in specificClass.iterdir():
       if elementFile.is_file():
           initial_count += 1
   print("Elements: " + str(initial_count))

