To set up the project, first create a virtual environment using either Conda or Python’s built-in venv. For Conda, run:
conda create -n myenv python=3.8
conda activate myenv
Alternatively, with Python’s venv, run:
python -m venv venv
Then activate the environment (on Unix/macOS: source venv/bin/activate, on Windows: venv\Scripts\activate). Once your virtual environment is active, install the required packages with:
pip install -r requirements.txt
Next, update the database schema by running:
python manage.py makemigrations
python manage.py migrate
Finally, start the Django development server with:
python manage.py runserver
Remember to run the Django server alongside your frontend application.