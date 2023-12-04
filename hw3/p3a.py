import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup

# The URL of the page you want to scrape
url = 'https://www.distancecalculator.net/city'


cities = ['Incheon', 'Seoul', 'Busan', 'Daegu', 'Daejeon', 'Gwangju', 'Suwon-si', 'Ulsan',
          'Jeonju', 'Cheongju-si', 'Changwon', 'Jeju-si', 'Chuncheon', 'Hongsung', 'Muan']

distances_table = pd.DataFrame(columns=cities, index=cities)

for idx, city in enumerate(cities):
    cityurl = url + '/' + city
    print(f"{idx}: {cityurl}")
    # Send a GET request to the page
    response = requests.get(cityurl)

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table containing the distance information
    distance_table = soup.find('table')
    print(distance_table)

    # Initialize a dictionary to store the distances
    distances = {}

    # Loop through the table rows and extract city names and distances
    for row in distance_table.find_all('tr')[1:]:  # Skip the header row
        cells = row.find_all('td')
        city_pair = cells[0].text
        distance_km = cells[1].text.replace(' km', '')  # Remove ' km' from the distance string
        print(f'{city_pair}: {distance_km}')
        distances[city_pair] = int(distance_km)
        
        A, B = city_pair.split(' to ')
        
        
        distances_table.loc[A, B] = int(distance_km)
        distances_table.loc[B, A] = int(distance_km)
        

    # distances now contains the city pairs and their distances in kilometers
print(distances_table)
# save the distabces table
distances_table.to_csv('distances_table.csv')

# Plot the distances as a table
# Replace non-numeric entries with NaN for plotting
distances_table = distances_table.apply(pd.to_numeric, errors='coerce')

# Plot the distances as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(distances_table, annot=True, fmt='g', cmap='viridis')
plt.title('Distance Heatmap Between South Korean Cities')
plt.savefig('distances_heatmap.png')

