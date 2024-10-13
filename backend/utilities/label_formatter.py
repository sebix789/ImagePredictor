# Format Breeds from Stanford Dogs Dataset
def format_breeds(category_name):
    return category_name.split('-')[1].replace('_', ' ').title()