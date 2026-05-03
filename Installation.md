# Installation Requirements

To run the GMM code, navigate to https://github.com/dr3w-mg/CS472-NBA-Clustering and fork the repository, then clone it to your local machine. I prefer to use GitHub Desktop (https://desktop.github.com/download/) to clone the repo, as well as edit any code.

I ran the GMM_NBA_code.py file using VS Code (https://code.visualstudio.com/download), and that's the method outlined here,

1. Install the Python extension in the Extensions panel (Ctrl + Shift + X on Windows). Search for "Python" and install the one by Microsoft.
2. Set up a virtual environment. Open the terminal in VS Code (Ctrl + ` on Windows). **MAKE SURE YOU SET UP THE VIRTUAL ENVIRONMENT IN THE SAME FOLDER AS THE GMM_NBA_code.py FILE (/Code)**. Then in the terminal, run the following commands:

* python -m venv venv
* venv\Scripts\activate (Windows)
* pip install nba_api pandas numpy scikit-learn matplotlib (or pip install -r requirements.txt)

3. Then select your new interpreter. Press Ctrl + Shift + P (Windows) and type "Python: Select Interpreter" and choose the venv you just created.
4. Press the play button in the top right of the VS Code application and let the clustering begin!