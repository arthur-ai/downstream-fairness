### INSERT LIBRARY NAME HERE

Include a description of your package.

### Quickstart
This section should include example usages and anything else necessary for someone to use your package. Provide a few lines of code or example command line usage. 

```
>>> from rename_me.process import Processor
>>> p = Processor()
>>> p.process([0, 0, 0])
```

### Using the Template
Rename the folder `rename_me` to the name of your package.
Fill in the sections of `setup.py` marked `FILL_ME_IN`  

To generate the requirements file:  
1. Create a clean virtual env and install the packages necessary to get a working version of your package locally.
2. Run `pip freeze > requirements.txt` to write your install to the requirements file.


### Local Development
Run `pip install -e .` from the root directory to get a development install of your package.  
Run `pytest test` to run the unit tests. 


### Project Structure
The file structure inside the project folder is a suggestion - feel free to adjust as needed for your use case. 

Typically, our enrichments have a training / fitting stage followed by a predicting / scoring stage. It is suggested to follow this split in the structure of your code: put train code in `train.py` and scoring code in `process.py`.
Helpers and common functions should go in `utils.py`.

Implement the package using common data types (ex. arrays, tensors, dictionaries) as input. Data preprocessing and manipulation should largely be left up to the client (ex. python jobs). If you'd like to provide additional tooling (ex. loading from local csv), add scripts in a separate scripts folder at the root level.


### Versioning

Semantic Versioning:

MAJOR version when you make incompatible API changes  
MINOR version when you add functionality in a backwards compatible   
PATCH version when you make backwards compatible bug fixes  
When incrementing a minor or major version reset the smaller version numbers to zero

Each time the version needs to be updated the text in rename_me/version.py must be changed.