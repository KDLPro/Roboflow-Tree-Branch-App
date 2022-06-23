import os
import re 


path = 'data/images/'
files = os.listdir(path)


files_dict = dict(
    # frame_number: frame-{frame_number}.png
)


# sort frames accordingly
for file in files:
    frame_number = int(re.search(r'\d+', file).group())
    files_dict[frame_number] = file
sorted_files = sorted(files_dict.items())
sorted_files = [key[1] for key in sorted_files]


# rename files
datum_number = 1
for old_name in sorted_files:
    datum_number_str = str(datum_number)
    zeros = '0'*(5 - len(datum_number_str))
    new_name = 'datum-{}.png'.format(zeros + datum_number_str)  
    datum_number = datum_number + 1
    print("{} to {}".format(new_name, new_name))
    os.rename(path+old_name, path+new_name)