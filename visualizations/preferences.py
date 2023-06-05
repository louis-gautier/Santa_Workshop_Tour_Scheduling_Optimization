import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

preferences_df = pd.read_csv("../data/family_data.csv")
first_choices = preferences_df[["family_id", "choice_0"]].groupby("choice_0").count().reset_index().rename(columns={"family_id": "count", "choice_0": "day"})
plt.bar(first_choices['day'], first_choices["count"])

# Set labels and title
plt.xlabel('Day')
plt.ylabel('Frequency')
plt.title('Histogram of Day Frequencies')

# Show the plot
plt.show()

all_integers = np.arange(1,101)
counts = np.zeros(100)
for i in range(1,101):
    counts[i-1] = np.sum(preferences_df[['choice_0', 'choice_1', 'choice_2']].values == i)

result_df = pd.DataFrame({'day': all_integers, 'count': counts})
plt.bar(result_df['day'], result_df["count"])

# Set labels and title
plt.xlabel('Day')
plt.ylabel('Number of families having placed this day in their top 3 preferences')
plt.title('Histogram of the appearance of days in the top 3 preferences')

# Show the plot
plt.show()
