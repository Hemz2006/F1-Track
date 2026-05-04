"""
app.py  –  Your F1 Track Flask App (with FastF1 integrated)
=============================================================
Changes made vs your original:
  ✅ Added:  from f1_api_routes import f1_bp
  ✅ Added:  app.register_blueprint(f1_bp)
  Everything else is exactly the same as before.
"""

from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import or_
import os

# ─── Import the FastF1 blueprint we created ────────────────────────────────
from f1_api_routes import f1_bp   # <── ADD THIS LINE

app = Flask(__name__, template_folder="templates")
CORS(app)

# ─── Register the FastF1 blueprint ─────────────────────────────────────────
app.register_blueprint(f1_bp)     # <── ADD THIS LINE

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

# ===================== MODEL =====================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

# ===================== DEBUG =====================
@app.route('/debug')
def debug():
    return str(os.listdir('templates'))

# ===================== PAGE ROUTES =====================

@app.route('/')
def home():
    return render_template("Landing.html")

@app.route('/login-page')
def login_page():
    return render_template("login.html")

@app.route('/register-page')
def register_page():
    return render_template("register.html")

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route('/analytics')
def analytics():
    return render_template("analytics.html")

@app.route('/circuits')
def circuits():
    return render_template("circuits.html")

@app.route('/drivers')
def drivers():
    return render_template("drivers.html")

@app.route('/replay')
def replay():
    return render_template("replay.html")

@app.route('/standings')
def standings():
    return render_template("standings.html")

@app.route('/landing')
def landing():
    return render_template("Landing.html")

# ===================== AUTH =====================

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    if User.query.filter_by(email=data['email']).first():
        return jsonify({"message": "User already exists"}), 400

    hashed_password = generate_password_hash(data['password'])

    user = User(
        username=data['username'],
        email=data['email'],
        password=hashed_password
    )

    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "Registered successfully"})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    user = User.query.filter(
        or_(
            User.email == data['email'],
            User.username == data['email']
        )
    ).first()

    if user and check_password_hash(user.password, data['password']):
        return jsonify({
            "message": "Login successful",
            "redirect": "/dashboard"
        })

    return jsonify({"message": "Invalid credentials"}), 401

# ===================== RUN =====================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, threaded=True)
