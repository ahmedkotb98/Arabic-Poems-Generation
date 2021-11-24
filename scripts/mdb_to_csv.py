"""
Convert MDB files to CSV files
"""
import argparse
import os
import subprocess
from glob import glob

from tqdm import tqdm


def convert_mdb_to_csv(input_path, output_path):
	books = glob(os.path.join(input_path, '*.mdb'))
	print("Number of books: {}".format(len(books)))

	for book in tqdm(books):
		book_id = book.split('/')[-1].split('.mdb')[0]
		book_path = os.path.join(output_path, book_id)

		# Create a new folder for each book
		if not os.path.exists(book_path):
			os.makedirs(book_path)

		tables = subprocess.getoutput("mdb-tables {}".format(book)).strip().split(' ')
		for table in tables:
			table_path = os.path.join(book_path, table + '.csv')
			os.system("MDB_JET3_CHARSET='cp1256' mdb-export {} {} > {}".format(book, table, table_path))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Convert MDB to CSV')
	parser.add_argument(
		"-i",
		"--input_folder",
		help="Path containing the mdb files",
		required=True,
		type=str
	)

	parser.add_argument(
		"-s",
		"--save_folder",
		help="Path to save the csv files",
		required=True,
		type=str
	)

	args = parser.parse_args()

	input_path = args.input_folder
	output_path = args.save_folder

	convert_mdb_to_csv(input_path, output_path)
