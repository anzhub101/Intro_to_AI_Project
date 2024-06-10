import os
import matplotlib
matplotlib.use('TkAgg') #error with this file presumably with matplotlib installation

import matplotlib.pyplot as plt

folders = {'New_TV': '/Users/manzeem/Documents/GitHub/pythonProject2/xTV_Folder',
           'New_Phones': '/Users/manzeem/Documents/GitHub/pythonProject2/yPhones_Folder',
           'New_Laptop': '/Users/manzeem/Documents/GitHub/pythonProject2/ZLaptop_Folder'}

class_counts = {}

for class_name, folder_path in folders.items():
    class_counts[class_name] = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])

# Creating a histogram
plt.bar(class_counts.keys(), class_counts.values(), color='blue')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Distribution of Classes in the Dataset')
plt.show()
