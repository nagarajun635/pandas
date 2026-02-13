api_key = 'f584e08607ea47da517afea23827df96138837f9bce396f8cc8414f9d387e3ef'
from openaq import OpenAQ
import pandas as pd

# 1. Initialize API
api = OpenAQ(api_key=api_key)

# 2. Fetch parameters
response = api.parameters.list()

# 3. Convert to DataFrame
# response.results is a list of objects; Pandas can read these directly
parameters_df = pd.DataFrame(response.results)

# 4. Save to CSV
parameters_df.to_csv('air_quality_parameters.csv', index=False)

print("Data saved to 'air_quality_parameters.csv'")
print(parameters_df.head())