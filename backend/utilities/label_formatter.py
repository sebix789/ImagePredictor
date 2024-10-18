# Format Breeds from Stanford Dogs Dataset
def format_breeds(category_name):
    breed_name = category_name.split('-', 1)[1]
    formatted_breed = breed_name.replace('-', '_').replace(' ', '_').lower()
    return formatted_breed