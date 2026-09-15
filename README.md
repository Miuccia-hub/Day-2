# Day-2

# Step-1 setup environment
# Step-1 create a virtual environment
1. create a virtual environment
>python -m venv .venv
2. activate
>source /workspaces/Legal-AI/.venv/bin/activate

# Step-2 install dependencies/python libraries
1. create a requirement.txt file
>touch requirements.txt
2. add, each on a new line, streamlit+ openai+ python-dotenv
3. run
>pip install -r requirements.txt

# Step-3 create a .env file to store secrets
1. create .env file
>touch .env
2. add OPENAI_API_KEY

# create streamlit application
1. create python file entrypoint
>touch home.py
2. run streamlit
>streamlit run home.py
3.  edit the python file
>import streamlit as st

# Note-saving code
1. source control
2. client + to add files to the commit
3. enter a commit message
4. click commit
5. click sync changes
6. click repository to confirm