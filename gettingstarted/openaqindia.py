# f584e08607ea47da517afea23827df96138837f9bce396f8cc8414f9d387e3ef
from openaq import OpenAQ
import pandas as pd

# Initialize OpenAQ API
api = OpenAQ()

# Fetch parameters data
status, resp = api.parameters()

# Convert to DataFrame
parameters_df = pd.DataFrame(resp['results'])

# Save to CSV
parameters_df.to_csv('air_quality_parameters.csv', index=False)

print("Data saved to 'air_quality_parameters.csv'")
