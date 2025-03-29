from flask import Flask, render_template, request, redirect, url_for, flash
from pymongo import MongoClient
from datetime import datetime
import qrcode
from PIL import Image
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Used for session and flash messages

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")  # Connect to MongoDB server
db = client["user_data"]  # Database
collection = db["users"]  # Collection

# Registration page route
@app.route('/')
def index():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_user():
    # Get data from form
    name = request.form['name']
    email = request.form['email']
    phone = request.form['phone']
    password = request.form['password']
    confirm_password = request.form['confirm_password']
    age = request.form['age']
    dob = request.form['dob']
    diseases = request.form['diseases']
    address = request.form['address']

    # Validate inputs
    if not all([name, email, phone, password, confirm_password, age, dob, diseases, address]):
        flash("All fields are required", "error")
        return redirect(url_for('index'))

    if password != confirm_password:
        flash("Passwords do not match", "error")
        return redirect(url_for('index'))

    # Create user dictionary
    user = {
        "name": name,
        "email": email,
        "phone": phone,
        "password": password,
        "age": age,
        "dob": dob,
        "diseases_suffered": diseases,
        "address": address,
        "registration_date": datetime.now()
    }

    # Insert user into MongoDB
    try:
        collection.insert_one(user)
        flash("Registration successful!", "success")

        # Generate QR code with user data
        generate_qr_code(user)

        return redirect(url_for('index'))

    except Exception as e:
        flash(f"Error occurred: {e}", "error")
        return redirect(url_for('index'))

def generate_qr_code(user):
    # Create a string with user details to encode into QR code
    user_details = f"Name: {user['name']}\nEmail: {user['email']}\nPhone: {user['phone']}\nAge: {user['age']}\nDOB: {user['dob']}\nDiseases: {user['diseases_suffered']}\nAddress: {user['address']}\nDate: {user['registration_date']}"

    # Generate QR code from user details
    qr = qrcode.make(user_details)

    # Save the QR code as an image
    qr_code_image = os.path.join('static', 'qr_codes', f"{user['name']}_qr.png")
    os.makedirs(os.path.dirname(qr_code_image), exist_ok=True)
    qr.save(qr_code_image)

    return qr_code_image

if __name__ == "__main__":
    app.run(debug=True)
