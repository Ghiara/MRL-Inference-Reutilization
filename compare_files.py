import json

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def find_differences(dict1, dict2, path=""):
    """Recursively find differences between two dictionaries."""
    for key in dict1.keys() | dict2.keys():  # Union of keys
        if key not in dict1:
            print(f"{path}.{key}: Key missing in the first JSON")
        elif key not in dict2:
            print(f"{path}.{key}: Key missing in the second JSON")
        else:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                find_differences(dict1[key], dict2[key], f"{path}.{key}")
            else:
                if dict1[key] != dict2[key]:
                    print(f"{path}.{key}: {dict1[key]} != {dict2[key]}")

def compare_json(file_path1, file_path2):
    """Compare two JSON files and print the differences."""
    json1 = load_json(file_path1)
    json2 = load_json(file_path2)

    find_differences(json1, json2)

if __name__ == "__main__":
    import sys

    file1 = '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_05_09_08_00_default_true_gmm/variant.json'
    file2 = '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_03_03_21_18_19_default_true_gmm/variant.json'
    compare_json(file1, file2)
