import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify, render_template
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for Matplotlib
import matplotlib.pyplot as plt
import io
import base64
import logging
from logging.handlers import RotatingFileHandler
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField
from wtforms.validators import DataRequired, NumberRange
from flask_wtf.csrf import CSRFProtect
import secrets

# Generate a random secret key
secret_key = secrets.token_hex(16)

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key  # Needed for CSRF protection
csrf = CSRFProtect(app)

# Setup logging
if not app.debug:
    file_handler = RotatingFileHandler('error.log', maxBytes=10240, backupCount=10)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Iris Prediction App startup')

# Load the Iris dataset and model
iris = load_iris()
model = joblib.load('iris_model.pkl')

# Define the form class
class IrisForm(FlaskForm):
    sepal_length = FloatField('Sepal Length (cm)', validators=[DataRequired(), NumberRange(min=0)])
    sepal_width = FloatField('Sepal Width (cm)', validators=[DataRequired(), NumberRange(min=0)])
    petal_length = FloatField('Petal Length (cm)', validators=[DataRequired(), NumberRange(min=0)])
    petal_width = FloatField('Petal Width (cm)', validators=[DataRequired(), NumberRange(min=0)])
    submit = SubmitField('Predict')

@app.route('/', methods=['GET', 'POST'])
def home():
    form = IrisForm()
    if form.validate_on_submit():
        # Extract data from form
        features = [form.sepal_length.data, form.sepal_width.data, form.petal_length.data, form.petal_width.data]
        features = np.array([features])
        prediction = model.predict(features)
        species_names = iris.target_names[prediction]

        # Get feature importances
        importances = model.feature_importances_

        # Create a plot
        fig, ax = plt.subplots()
        y_pos = np.arange(len(iris.feature_names))
        ax.barh(y_pos, importances, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(iris.feature_names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importances')

        # Save the plot to a PNG image in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)

        return render_template('index.html', form=form, prediction=species_names[0], feature_importance_plot=f"data:image/png;base64,{image_base64}")

    return render_template('index.html', form=form)

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server Error: {error}, Path: {request.path}")
    return render_template('500.html'), 500

@app.errorhandler(404)
def not_found_error(error):
    app.logger.error(f"Page Not Found: {error}, Path: {request.path}")
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)



    