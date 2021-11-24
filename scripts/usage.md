## Requirements
* [tqdm 4.52.0](https://pypi.org/project/tqdm/)
* [mdbtools](https://github.com/mdbtools/mdbtools)

### Quickstart	
To run a MDB to CSV convertor call the `mdb_to_csv.py` script with the following arguments:

Arguments description, e.g.:	
```	
-h --help    for the complete list of supported arguments	
-i --input_folder  Path of the folder that containing all mdb files.
-s --save_folder   Path to save the csv files.
```	


Usage pattern, e.g.:	
```
python mdb_to_csv.py -i <path-of-mdb-files> -s <path-to-save-csv-files>
```
