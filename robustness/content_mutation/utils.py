
def check_third_party(row):

	try:
		base_domain = row['domain']
		top_level_domain = row['top_level_domain']
		if base_domain and top_level_domain:
			if base_domain != top_level_domain:
				return True
	except Exception as e:
		return False
	return False

def find_third_parties(df):

	df_third_party = df[(df['is_third_party'] == True) & (df['top_level_url'].notnull())]
	return df_third_party


def find_tracker_predictions(predictions):

	tracker_dict = {}

	with open(predictions) as f:
		lines = f.readlines()
		for line in lines:
			parts = line.strip().split("|$|")
			pred = parts[1].strip()
			name = parts[2].strip()
			vid = float(parts[3].strip())
			if (pred == "True"):
				if vid not in tracker_dict:
					tracker_dict[vid] = []
				tracker_dict[vid].append(name)
	return tracker_dict