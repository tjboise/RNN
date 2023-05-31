import requests
import json
import os
import tqdm

"""
This script downloads state-level data from the US Department of Transportation's
Database on road safety. The data is stored in the data/ directory as JSON files.
"""

def get_data_for_state(state):
    url = f'https://geo.dot.gov/server/rest/services/Hosted/{state}_2018_PR/FeatureServer/0/query'

    # Define the parameters for the request
    params = {
        'f': 'json',
        'where': '1=1',
        'outFields': '*',
        'returnGeometry': 'true'
    }

    # Make the request
    response = requests.get(url, params=params)

    # If the request was successful, the status code will be 200
    if response.status_code == 200:
        data = json.loads(response.content)
    else:
        print(f'Request failed with status code {response.status_code}')
        return

    with open(f'data/{state}.json', 'w') as f:
        json.dump(data, f)



states = ['Alabama', 'Arizona', 'Arkansas',  # 'Alaska' was removed as it required an authentication token
          'California', 'Colorado', 'Connecticut', 'Delaware', 'District', 
          'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 
          'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 
          'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 
          'Missouri', 'Montana', 'Nebraska', 'Nevada', 
          'NewHampshire', 'NewJersey', 'NewMexico', 'NewYork', 
          'NorthCarolina', 'NorthDakota', 'Ohio', 'Oklahoma', 
          'Oregon', 'Pennsylvania', 'RhodeIsland', 'SouthCarolina', 
          'SouthDakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 
          'Virginia', 'Washington', 'WestVirginia', 'Wisconsin', 
          'Wyoming', 'PuertoRico']

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    print('Downloading data...')
    for state in tqdm.tqdm(states):
        get_data_for_state(state)