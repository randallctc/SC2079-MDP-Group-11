import os

folder_path = r"C:\Users\Randall Chiang\Documents\MDP Stuff\MDP Dataset\train\labels"
target_number = 0            
def find_files_by_first_number(folder, target):
    matching_files = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder, filename)
            with open(file_path, "r") as f:
                first_line = f.readline().strip()
                if first_line:
                    numbers = first_line.split()
                    try:
                        first_num = float(numbers[0])
                        if first_num == target:
                            matching_files.append(filename)
                    except ValueError:
                        continue 
    return matching_files

result = find_files_by_first_number(folder_path, target_number)

print(f"Files where first number == {target_number}:")
for file in result:
    print(file)
