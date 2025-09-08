# Data Science & Machine Learning - Project Structure and workflow

*J Rogel-Salazar*

The nature of the work carried out by the tech team at TympaHealth requires the creation, recording and archiving of data and code assets that can be used and reused by various members of the entire team. We propose the use of a standardised structure to make engagements easier to follow and understand.

## Usage

This repo is a template. To use it, simply click on the "Use this template" button in the menu.

You can take a look at the file structure proposed in the folders of this repo.

If you decide not to use this skeleton, please make sure that you use the naming conventions for source code files.

## Create file structure containing entire project data & work

* root directory should have clear name related to the project
* subdirectories should follow the standardised structure proposed in this repository
* for archiving purposes at the end of a project, the root directory should be compressed into a zip file or tar ball for convenient access and stored on a suitable bucket
* it is suggested not to put sensitive information (data, customer notes, etc) in the repository. Instead, create a suitable bucket folder with the appropriate permissions. You can then create a link in the `readme.md` file in `/root` or the suitable folder, e.g. `/data`.

## Create guidelines for directory structure

* `readme.md`:
	* this file is automatically rendered by GitHub and it is suggested to be used to provide a summary about the Project as well as information about deadlines and deliverables  
* `/admin`:
	* 	SOW documents, supporting material, etc.
* `/code`:
	* scripts & code not contained in notebooks. Should contain subdirectories for different languages, e.g.,
		* `/code/R`
		* `/code/Python`
		* `/code/OTHER`
	* lab notebooks & other notes taken during project. Store notebooks as `.ipynb` and `.html/.pdf` files
	* make use of tools such as [Jupytext](https://jupytext.readthedocs.io/en/latest/introduction.html) to help version control Jupyter notebooks
* `/data`:
	* directory for raw data (csv files, excel spreadsheets, etc.). If data can't be stored after the end of the project, this should contain a link to the data or a description of the data.
	* It is recommended that this data is stored and not touched (transformed, modified, etc)
* `/data_transformed`:
	* directory for transformed data (csv files, excel spreadsheets, etc.) generated from data in `/data`. If data can't be stored after the end of the engagement, this should contain a link to the data or a description of the data.
* `/deliver`:
	* delivery notebooks for future use by other data scientists store notebooks as .ipynb and .html/.pdf files
* `/documentation`:
	* documentation for the project  
* `/presentation`:
	* final presentation, supporting material, videos, etc.
* `/visual`:
	* visualisation assets such as `.jpg`, `.png`, D3, screen shots, etc

## Naming conventions for notebooks & files

* Format for notebook files:
	* Use ISO-8601 dating, i.e. `YYYYMMDD`
	* [ISO-8601 start date]_[Data scientist initials]_[Short description of notebook].ipynb
* Example:
	* `20190421_JR_DataUnderstandingImportantProject.ipynb`
* If there are multiple notebooks that need to be run in sequence, add an ordinal number after the ISO date:
  * `20190421__01_JR_DataUnderstandingImportantProject.ipynb`
  * `20190421_02_JR_DataTransformImportantProject.ipynb`
  * etc
* Format for source in `/code`:
	* Comment code thoroughly
	* Use proper indentation
	* Supply docstrings/auxiliary info where possible
	* follow standard coding practices (e.g., PEP8) when possible
* Format for presentations:
	* Use standard slide format for TympaHealth

In case of any questions contact [J Rogel](mailto:jay@tympahealth.com).

Enjoy!
