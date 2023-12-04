import itertools

# Updated item weights based on the provided image
item_weights = [
    3.3,   # Shadow Daggers -> knife
    3.4,   # Huntsman Knife -> knife
    6.0,   # Gut Knife -> knife
    26.1,  # 228 Compact Handgun -> pistol
    37.6,  # Night Hawk -> pistol
    62.5,  # Desert Eagle Magnum -> pistol
    100.2, # Ingram MAC-10 SMG -> primary
    141.1, # Leone YG1265 Auto Shotgun -> primary
    119.2, # M4A1 Carbine -> primary
    122.4, # AK-47 Rifle -> primary
    247.6, # Krieg 550 Sniper Rifles -> primary
    352.0, # M249 Machine Gun -> primary
    24.2,  # Gas Mask -> equipment
    32.1,  # Night-Vision Goggle -> equipment
    42.5   # Tactical Shield -> equipment
]

# Maximum weight limit
weight_limit = 529

# Function to calculate the total weight of a given combination of items
def total_weight(combination):
    return sum(item_weights[i] for i in combination)

# Count feasible combinations
feasible_combinations = 0

# Iterate through all possible combinations of the 15 items
for r in range(3, len(item_weights) + 1):  # r represents the number of items in the combination
    for combination in itertools.combinations(range(len(item_weights)), r):
        # Check if the combination contains at least 1 knife, 1 pistol, and 1 equipment
        if 0 in combination and 3 in combination and 12 in combination:
            if total_weight(combination) <= weight_limit:
                feasible_combinations += 1

print(feasible_combinations)
