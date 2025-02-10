# Project Setup Instructions

To set up the project:

1. **Create a virtual environment** using either Conda or Python's built-in `venv`:
   - Using Conda:d
     ```bash
     conda create -n myenv python=3.9
     conda activate myenv
     ```
   - Using Python's `venv`:
     ```bash
     python -m venv venv
     # Activate the virtual environment:
     # On Unix/macOS:
     source venv/bin/activate
     # On Windows:
     venv\Scripts\activate
     ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Update the database schema**:
   - Create migrations for the latest data model:
     ```bash
     python manage.py makemigrations
     ```
   - Apply the migrations to update the database:
     ```bash
     python manage.py migrate
     ```

4. **Run the development server**:
   - Start the Django development server:
     ```bash
     python manage.py runserver 5050
     ```

5. **Run frontend services:** 
   - The backend serves data to the frontend, which runs at [http://localhost:3000](http://localhost:3050). Ensure both are running for full functionality.
   - Access the frontend repo through this link: [CBM-GUI-Frontend](https://github.com/gabrielchasukjin/cbm-gui-frontend)


   > **Note:** Make sure to run the Django server **alongside your frontend application** to ensure the full application stack is operational.
