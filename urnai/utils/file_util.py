def is_json_file(file_path):
    try:
        json.loads(text)
	return True
    except ValueError as e:
        return False


defis_csv_file(file_path):
    try:
        with open(file_path, newline='') as csvfile:
            start = csvfile.read(4096)

            # isprintable does not allow newlines, printable does not allow umlauts...
            if not all([c in string.printable or c.isprintable() for c in start]):
                return False
            dialect = csv.Sniffer().sniff(start)
            return True
    except csv.Error:
        # Could not get a csv dialect -> probably not a csv.
        return False
