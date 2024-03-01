# moduł patryka do wykresów, lepiej jako moduł niż jako main

from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import calendar
import locale

warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# PRZYGOTOWANIE DANYCH po chuju DO WYKRESOW - tymczasowo ??
# odmapowuje 'ORIGIN', 'OP_CARRIER', dodaje cala date z czasem
def przygotowanie_df_do_wykresow(df):
	with open('wszsytkiemapowania.txt', 'r') as file:
		mapping = eval(file.read())
	do_zmapowania = ['ORIGIN', 'OP_CARRIER'] # 'DEST'
	reverse_mapping = {kolumna: {v: k for k, v in mapping[kolumna].items()} for kolumna in do_zmapowania}
	for kolumna in do_zmapowania:
		df[kolumna] = df[kolumna].map(reverse_mapping[kolumna])
	df[do_zmapowania] = df[do_zmapowania].astype('category')
	df['datatype'] = pd.to_datetime(df[['FL_YEAR', 'FL_MONTH', 'FL_DAY']].astype(str).agg('-'.join, axis=1), errors='coerce')
	df['day_of_week'] = df['datatype'].dt.day_of_week + 1
	df['month'] = df['datatype'].dt.month + 1
	df = df.drop(columns=['ID', 'id', 'WINDGUST'], axis=1)
	df = df.dropna()
	return df



# zamaist wykresu dalbym zwykla linijke wyliczajaca i wypisujaca
def delays_barplot(df):
	delay_counts = df['DELAY'].value_counts()
	ax = sns.barplot(x=delay_counts.index, y=delay_counts, hue=delay_counts.index, palette="Set2", legend=False)
	ax.set_xticks((0, 1))
	ax.set_xticklabels(["No", "Yes"])
	plt.show()

def airlines_delay_analysis(df):
	width = 0.7  # bar width
	plt.figure(figsize=(12, 10))
	# 
	plt.subplot(3, 1, 1)
	delay_percentage = df.groupby('OP_CARRIER')['DELAY'].mean() * 100
	ax1 = delay_percentage.plot(kind='bar', width=width, color='skyblue')
	plt.title('Percentage of Delays for Each Airline')
	plt.xlabel('OP_CARRIER')
	plt.ylabel('Percentage of delays')
	plt.xticks(rotation=45, ha='right')
	plt.legend(title='Percentage of', bbox_to_anchor=(1.05, 0.5), loc='center left')
	ax1.set_ylim([0, 70])
	for i in range(0, 80, 10):
		plt.axhline(y=i, color='gray', linestyle= '-' if i==50 else '--', linewidth=0.5)
	# 
	plt.subplot(3, 1, 2)
	result_df = df.groupby(['OP_CARRIER', 'day_of_week'])['DELAY'].mean().reset_index()
	result_df['DELAY'] = result_df['DELAY'] * 100
	ax2 = sns.barplot(x='OP_CARRIER', y='DELAY', hue='day_of_week', data=result_df, ci=None)
	plt.title('Percentage of Delays for Each Airline for the Following Days of the Week')
	plt.xlabel('OP_CARRIER')
	plt.ylabel('Percentage of delays')
	ax2.set_ylim([0, 70])
	for i in range(0, 80, 10):
		plt.axhline(y=i, color='gray', linestyle= '-' if i==50 else '--', linewidth=0.5)
	plt.legend(title='Day of week', bbox_to_anchor=(1.05, 0.5), loc='center left')
	# 
	plt.subplot(3, 1, 3)
	result_df = df.groupby(['OP_CARRIER', 'month'])['DELAY'].mean().reset_index()
	result_df['DELAY'] = result_df['DELAY'] * 100
	ax3 = sns.barplot(x='OP_CARRIER', y='DELAY', hue='month', data=result_df, ci=None, palette='viridis')
	plt.title('The Percentage of Delays for Each Airline in the Following Months')
	plt.xlabel('OP_CARRIER')
	plt.ylabel('Percentage of delays')
	ax3.set_ylim([0, 70])
	for i in range(0, 80, 10):
		plt.axhline(y=i, color='gray', linestyle= '-' if i==50 else '--', linewidth=0.5)
	plt.legend(title='Month of year', bbox_to_anchor=(1.05, 0.5), loc='center left')
	plt.tight_layout(pad=3.0)
	plt.show()
	#
	plt.figure(figsize=(12, 6))
	delayed_flights = df[df['DELAY'] == 1].groupby('OP_CARRIER').size()
	on_time_flights = df[df['DELAY'] == 0].groupby('OP_CARRIER').size()
	delayed_bars = plt.bar(delayed_flights.index, delayed_flights, width, color='red', label='Delayed')
	on_time_bars = plt.bar(on_time_flights.index, on_time_flights, width, bottom=delayed_flights, color='green', label='On Time')
	plt.title('Total Number of Flights for Each Airline')
	plt.xlabel('OP_CARRIER')
	plt.ylabel('Number of flights')
	plt.xticks(rotation=45, ha='right')
	plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
	plt.tight_layout(pad=3.0)
	plt.show()

def airlines_distance_analysis(df):
	carrier_distances = df.groupby('OP_CARRIER')['DISTANCE'].sum()
	plt.figure(figsize=(10, 6))
	carrier_distances.plot(kind='bar', color='skyblue')
	plt.title('Total Distances for Individual Airlines')
	plt.xlabel('OP_CARRIER')
	plt.ylabel('Total distances [miles]')
	plt.xticks(rotation=45)
	plt.grid(axis='y', linestyle='--', alpha=0.7)
	plt.tight_layout()
	plt.show()

def hours_analysis():
	pass

def monthly_delay_analysis(df):
	plt.figure(figsize=(10, 12))
	# Wykres procentowy opóźnień
	plt.subplot(2, 1, 1)
	plt.title('Monthly Delay Percentage Over Different Years')
	for year in df['datatype'].dt.year.unique():
		data_year = df[df['datatype'].dt.year == year]
		delay_percentage = data_year.groupby(data_year['datatype'].dt.month)['DELAY'].mean() * 100
		months_names = [calendar.month_abbr[i] for i in delay_percentage.index]
		plt.plot(months_names, delay_percentage, label=f'Year {year}', linestyle='-', marker='o')
	plt.ylabel('Delay percentage')
	plt.legend()
	plt.grid(True)
	# Wykres ilości opóźnień
	plt.subplot(2, 1, 2)
	plt.title('Monthly Delay Counts Over Different Years')
	for year in df['datatype'].dt.year.unique():
		data_year = df[df['datatype'].dt.year == year]
		delay_counts = data_year.groupby(data_year['datatype'].dt.month)['DELAY'].sum()
		months_names = [calendar.month_abbr[i] for i in delay_counts.index]
		plt.plot(months_names, delay_counts, label=f'Year {year}', linestyle='-', marker='o')
	plt.xlabel('Month')
	plt.ylabel('Delay count')
	plt.legend()
	plt.grid(True)
	plt.tight_layout(pad=3.0)
	plt.show()

# Normalized Mutual Information
# target='ARR_DELAY'
def NMI(df, cols, target=None):
	if target:
		print("Normalized Mutual Information\n")
		nmi_values = []
		for col in cols:
			nmi_value = normalized_mutual_info_score(df[col], df[target])
			nmi_values.append((col, nmi_value))
		nmi_values_sorted = sorted(nmi_values, key=lambda x: x[1], reverse=True)
		print(f"{target}")
		for col, nmi_value in nmi_values_sorted:
			print(f"{col:<15}{round(nmi_value, 3)}")
	else:
		nmi_matrix = pd.DataFrame(index=cols, columns=cols)
		for col1 in cols:
			for col2 in cols:
				nmi_value = normalized_mutual_info_score(df[col1], df[col2])
				nmi_matrix.loc[col1, col2] = nmi_value
		heatmap_nmi = sns.heatmap(nmi_matrix.astype(float), annot=True, cmap="coolwarm", vmin=0, vmax=1)
		plt.title("Normalized Mutual Information (NMI) Heatmap")
		plt.xticks(rotation=45)
		plt.yticks(rotation=45)
		plt.show()

# Linear Correlation Heapmap
# target='ARR_DELAY'
def linear_correaltaion(df, cols, target=None):
	if target:
		print("Linear Correaltaion\n")
		correlations = round(df[cols].corrwith(df[target]), 3)
		print(f"{target}")
		for col, correlation in zip(cols, correlations):
			print(f"{col:<15}{correlation}")
	else:
		sns.heatmap(df[cols].corr(), vmin=-1, vmax=1, annot=False)
		plt.xticks(rotation=45)
		plt.yticks(rotation=45)
		plt.show()


# zrobic sobie df z concat_csvs.py
# ogarnac go funckja przygotowanie_df_do_wykresow()
#not needed
#df = pd.read_csv('data_cale.csv') # z concat_csvs.py

def run_plots(df):
	df = przygotowanie_df_do_wykresow(df)
	#print(df.columns.values)
	#airlines_distance_analysis(df)
	#monthly_delay_analysis(df)
	#airlines_delay_analysis(df)
	#NMI(df, df)
	linear_correaltaion(df, df.columns.values)