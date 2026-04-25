from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from config import Config
from database import db, User, Prediction
from models.ml_model import predictor
from datetime import datetime
from functools import wraps
from authlib.integrations.flask_client import OAuth
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from openai import OpenAI
import json
import os
from time import time
from werkzeug.utils import secure_filename
from models.skin_model import skin_predictor
from PIL import Image
import io
import uuid
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'your-email@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'your-app-password')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME', 'your-email@gmail.com')

# Initialize extensions
mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Initialize database
db.init_app(app)

# Initialize OAuth
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id='967377927163-i4cutt5nsmnfana44rjfv0f8idq1ag59.apps.googleusercontent.com',
    client_secret='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',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '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')
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("✓ OpenAI ChatGPT initialized successfully")
except Exception as e:
    print(f"⚠ OpenAI initialization failed: {e}")
    openai_client = None

# Create tables and default admin
with app.app_context():
    db.create_all()
    admin = User.query.filter_by(email='admin@medicalsystem.com').first()
    if not admin:
        admin = User(
            name='Administrator',
            email='admin@medicalsystem.com',
            phone='0000000000',
            dob='01-01-1990',
            gender='Other',
            account_type='admin'
        )
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("✓ Default admin created: admin@medicalsystem.com / admin123")

# ADMIN CREDENTIALS
ADMIN_USERNAME = "TMDEV"
ADMIN_PASSWORD = "TM@2006"

# RATE LIMITING FOR CHATBOT
chat_rate_limit = {}

def rate_limit_check(user_id, max_requests=20, window=60):
    """Check if user has exceeded rate limit"""
    now = time()
    
    if user_id in chat_rate_limit:
        chat_rate_limit[user_id] = [
            t for t in chat_rate_limit[user_id] 
            if now - t < window
        ]
    else:
        chat_rate_limit[user_id] = []
    
    if len(chat_rate_limit[user_id]) >= max_requests:
        return False
    
    chat_rate_limit[user_id].append(now)
    return True

# DECORATORS
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# Helper function to get current user
def get_current_user():
    """Get the current logged-in user"""
    if 'user_id' in session:
        return User.query.get(session['user_id'])
    return None

# PUBLIC ROUTES
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        dob = request.form.get('dob')
        gender = request.form.get('gender')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not all([name, email, phone, dob, gender, password]):
            flash('All fields are required', 'error')
            return redirect(url_for('signup'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('signup'))
        
        new_user = User(
            name=name,
            email=email,
            phone=phone,
            dob=dob,
            gender=gender
        )
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['account_type'] = user.account_type
            
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('landing'))

# PASSWORD RESET ROUTES
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        
        if not email:
            flash('Please enter your email address', 'error')
            return redirect(url_for('forgot_password'))
        
        user = User.query.filter_by(email=email).first()
        
        if user:
            try:
                token = serializer.dumps(email, salt='password-reset-salt')
                reset_url = url_for('reset_password', token=token, _external=True)
                
                msg = Message(
                    'Password Reset Request - TruMed',
                    recipients=[email]
                )
                msg.html = f'''
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body {{
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            background-color: #f5f5f5;
                            margin: 0;
                            padding: 0;
                        }}
                        .container {{
                            max-width: 600px;
                            margin: 40px auto;
                            background: white;
                            border-radius: 16px;
                            overflow: hidden;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                        }}
                        .header {{
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            padding: 40px 30px;
                            text-align: center;
                        }}
                        .header h1 {{
                            margin: 0;
                            font-size: 28px;
                        }}
                        .content {{
                            padding: 40px 30px;
                        }}
                        .content p {{
                            color: #555;
                            line-height: 1.6;
                            margin-bottom: 20px;
                        }}
                        .button {{
                            display: inline-block;
                            padding: 15px 40px;
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            text-decoration: none;
                            border-radius: 8px;
                            font-weight: bold;
                            margin: 20px 0;
                        }}
                        .footer {{
                            background: #f8f8f8;
                            padding: 20px 30px;
                            text-align: center;
                            color: #888;
                            font-size: 14px;
                        }}
                        .link {{
                            color: #667eea;
                            word-break: break-all;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1>🏥 TruMed Password Reset</h1>
                        </div>
                        <div class="content">
                            <p>Hello {user.name},</p>
                            <p>You requested to reset your password for your TruMed account.</p>
                            <p>Click the button below to reset your password:</p>
                            <center>
                                <a href="{reset_url}" class="button">Reset Password</a>
                            </center>
                            <p>Or copy and paste this link into your browser:</p>
                            <p class="link">{reset_url}</p>
                            <p><strong>This link will expire in 1 hour.</strong></p>
                            <p>If you didn't request this password reset, please ignore this email.</p>
                        </div>
                        <div class="footer">
                            <p>© 2026 TruMed. All rights reserved.</p>
                        </div>
                    </div>
                </body>
                </html>
                '''
                mail.send(msg)
                flash('Password reset link has been sent to your email.', 'success')
            except Exception as e:
                print(f"Error sending email: {str(e)}")
                flash('Error sending email. Please try again later.', 'error')
        else:
            flash('If an account with that email exists, a password reset link has been sent.', 'success')
        
        return redirect(url_for('forgot_password'))
    
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = serializer.loads(token, salt='password-reset-salt', max_age=3600)
    except SignatureExpired:
        flash('The password reset link has expired. Please request a new one.', 'error')
        return redirect(url_for('forgot_password'))
    except BadTimeSignature:
        flash('Invalid password reset link.', 'error')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not password or not confirm_password:
            flash('Please fill in all fields.', 'error')
            return render_template('reset_password.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('reset_password.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('reset_password.html')
        
        user = User.query.filter_by(email=email).first()
        if user:
            user.set_password(password)
            db.session.commit()
            flash('Your password has been reset successfully! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('User not found.', 'error')
            return redirect(url_for('forgot_password'))
    
    return render_template('reset_password.html')

# GOOGLE LOGIN / SIGNUP
@app.route('/login/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/login/callback')
def google_callback():
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')
        
        if not user_info:
            flash('Failed to get user information from Google', 'error')
            return redirect(url_for('login'))

        email = user_info.get('email')
        name = user_info.get('name', 'Google User')

        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(
                name=name,
                email=email,
                phone='Google Login',
                dob='N/A',
                gender='Other',
                account_type='user'
            )
            user.set_password('google_oauth')
            db.session.add(user)
            db.session.commit()

        session['user_id'] = user.id
        session['user_name'] = user.name
        session['account_type'] = user.account_type

        flash(f'Welcome, {user.name}!', 'success')
        return redirect(url_for('dashboard'))
    
    except Exception as e:
        flash(f'An error occurred during Google login: {str(e)}', 'error')
        return redirect(url_for('login'))

# ADMIN LOGIN ROUTES
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if 'admin_logged_in' in session:
        return redirect(url_for('admin_dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            session['admin_username'] = username
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials', 'error')
    
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    flash('Admin logged out successfully', 'success')
    return redirect(url_for('admin_login'))

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    users = User.query.all()
    
    total_users = User.query.filter_by(account_type='user').count()
    total_admins = User.query.filter_by(account_type='admin').count()
    total_predictions = Prediction.query.count()
    
    high_risk = Prediction.query.filter_by(risk_level='High').count()
    moderate_risk = Prediction.query.filter_by(risk_level='Moderate').count()
    low_risk = Prediction.query.filter_by(risk_level='Low').count()
    
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_users = User.query.filter_by(account_type='user').count()
    today_predictions = Prediction.query.filter(Prediction.timestamp >= today).count()
    
    return render_template('admin_dashboard.html',
                         users=users,
                         total_users=total_users,
                         total_admins=total_admins,
                         total_predictions=total_predictions,
                         high_risk=high_risk,
                         moderate_risk=moderate_risk,
                         low_risk=low_risk,
                         today_users=today_users,
                         today_predictions=today_predictions)

@app.route('/admin/change-password/<int:user_id>', methods=['POST'])
@admin_required
def admin_change_user_password(user_id):
    try:
        data = request.get_json()
        new_password = data.get('new_password')
        
        if not new_password:
            return jsonify({'success': False, 'message': 'Password cannot be empty'})
        
        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'message': 'User not found'})
        
        user.set_password(new_password)
        db.session.commit()
        
        return jsonify({'success': True, 'message': f'Password changed for {user.name}'})
    except Exception as e:
        print(f"Error changing password: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/admin/delete-user/<int:user_id>', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'message': 'User not found'})
        
        user_name = user.name
        
        Prediction.query.filter_by(user_id=user_id).delete()
        
        db.session.delete(user)
        db.session.commit()
        
        return jsonify({'success': True, 'message': f'User {user_name} deleted successfully'})
    except Exception as e:
        print(f"Error deleting user: {e}")
        return jsonify({'success': False, 'message': str(e)})

# USER ROUTES
@app.route('/dashboard')
@login_required
def dashboard():
    user = get_current_user()
    
    total_predictions = Prediction.query.filter_by(user_id=user.id).count()
    high_risk_count = Prediction.query.filter_by(user_id=user.id, risk_level='High').count()
    
    recent_predictions = Prediction.query.filter_by(user_id=user.id).order_by(
        Prediction.timestamp.desc()
    ).limit(5).all()
    
    return render_template('dashboard.html', 
                         user=user, 
                         recent_predictions=recent_predictions,
                         total_predictions=total_predictions,
                         high_risk_count=high_risk_count)

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    if request.method == 'POST':
        symptoms_dict = {}
        for symptom in predictor.symptoms_list:
            symptoms_dict[symptom] = symptom in request.form
        
        custom_issue = request.form.get('custom_issue', '').strip()
        result = predictor.predict(symptoms_dict)
        
        medicine_info = predictor.get_medicine_recommendations(result['disease'])
        
        symptoms_str = ', '.join(result['active_symptoms']) if result['active_symptoms'] else 'No specific symptoms'
        if custom_issue:
            symptoms_str += f" | Custom: {custom_issue}"
        
        new_prediction = Prediction(
            user_id=session['user_id'],
            symptoms=symptoms_str,
            predicted_disease=result['disease'],
            confidence_score=result['confidence'],
            risk_level=result['risk_level'],
            suggestions=json.dumps(result['suggestions']),
            contributing_factors=json.dumps(result['contributing_factors']),
            recommended_medicines=medicine_info['medicines'],
            precautions=medicine_info['precautions'],
            diet_recommendations=medicine_info['diet']
        )
        db.session.add(new_prediction)
        db.session.commit()
        
        session['latest_prediction'] = {
            'id': new_prediction.id,
            'disease': result['disease'],
            'confidence': result['confidence'],
            'risk_level': result['risk_level'],
            'suggestions': result['suggestions'],
            'contributing_factors': result['contributing_factors'],
            'symptoms': symptoms_str,
            'timestamp': new_prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'medicines': medicine_info['medicines'],
            'precautions': medicine_info['precautions'],
            'diet': medicine_info['diet']
        }
        return redirect(url_for('results'))
    
    return render_template('prediction.html', 
                         symptoms=predictor.symptoms_list,
                         symptom_categories=predictor.symptom_categories)

@app.route('/results')
@login_required
def results():
    if 'latest_prediction' not in session:
        flash('No prediction results found', 'error')
        return redirect(url_for('prediction'))
    
    result = session['latest_prediction']
    return render_template('results.html', result=result)

@app.route('/history')
@login_required
def history():
    user = get_current_user()
    predictions = Prediction.query.filter_by(user_id=user.id).order_by(
        Prediction.timestamp.desc()
    ).all()
    
    for pred in predictions:
        pred.suggestions_list = json.loads(pred.suggestions)
        pred.factors_list = json.loads(pred.contributing_factors) if pred.contributing_factors else []
    
    return render_template('history.html', predictions=predictions)

# ====================================================================
# SKIN DISEASE ANALYSIS ROUTE
# ====================================================================
@app.route('/skin-analysis', methods=['GET', 'POST'])
@login_required
def skin_analysis():
    """Skin disease detection from uploaded image"""
    user = get_current_user()
    
    if request.method == 'POST':
        try:
            # Check if image was uploaded
            if 'skin_image' not in request.files:
                flash('Please upload an image', 'error')
                return redirect(request.url)
            
            file = request.files['skin_image']
            
            if file.filename == '':
                flash('No image selected', 'error')
                return redirect(request.url)
            
            # Validate file extension
            allowed_extensions = {'png', 'jpg', 'jpeg'}
            if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
                # Read image bytes
                img_bytes = file.read()

                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                logger.info(f"Image loaded: {image.size}")

                # Get prediction using PIL Image
                result = skin_predictor.predict(image)

                
                if result['success']:
                    # Save to database
                    try:
                        new_prediction = Prediction(
                            user_id=user.id,
                            predicted_disease=f"Skin: {result['disease']}",
                            confidence_score=result['confidence'],
                            risk_level=result['risk_level'],
                            symptoms=json.dumps({'image_analysis': True}),
                            recommendations=json.dumps(result['recommendations']),
                            timestamp=datetime.utcnow()
                        )
                        db.session.add(new_prediction)
                        db.session.commit()
                    except Exception as e:
                        logger.error(f"Error saving skin prediction: {str(e)}")
                    
                    return render_template('skin_results.html',
                                         result=result,
                                         user=user)
                else:
                    flash(f'Prediction error: {result.get("error", "Unknown error")}', 'error')
                    return redirect(request.url)
            else:
                flash('Invalid file type. Please upload JPG, JPEG, or PNG', 'error')
                return redirect(request.url)
                
        except Exception as e:
            logger.error(f"Skin analysis error: {str(e)}")
            flash('An error occurred during analysis. Please try again.', 'error')
            return redirect(request.url)
    
    return render_template('skin_analysis.html', user=user)

# API ROUTES FOR SYMPTOM CATEGORIES
@app.route('/api/symptom-categories', methods=['GET'])
@login_required
def get_symptom_categories():
    """Get all symptom categories with symptoms"""
    try:
        categories = predictor.get_symptoms_by_category()
        return jsonify({
            'success': True,
            'categories': categories,
            'total_categories': len(categories)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/categories', methods=['GET'])
@login_required
def get_category_names():
    """Get list of category names only"""
    try:
        categories = predictor.get_all_categories()
        return jsonify({
            'success': True,
            'categories': categories,
            'total': len(categories)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/symptoms/search', methods=['GET'])
@login_required
def search_symptoms():
    """Search symptoms by query"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({
            'success': False, 
            'message': 'Query parameter "q" is required'
        }), 400
    
    try:
        results = predictor.search_symptoms(query)
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/category/<category_name>', methods=['GET'])
@login_required
def get_category_symptoms(category_name):
    """Get symptoms for a specific category"""
    try:
        symptoms = predictor.get_symptoms_by_category(category_name)
        
        if not symptoms:
            return jsonify({
                'success': False, 
                'message': 'Category not found'
            }), 404
        
        formatted_symptoms = [
            {
                'value': s, 
                'label': s.replace('_', ' ').title()
            } 
            for s in symptoms
        ]
        
        return jsonify({
            'success': True,
            'category': category_name,
            'symptoms': formatted_symptoms,
            'count': len(formatted_symptoms)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for disease prediction"""
    try:
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({
                'success': False,
                'message': 'Symptoms data required'
            }), 400
        
        symptoms_dict = data['symptoms']
        result = predictor.predict(symptoms_dict)
        
        symptoms_str = ', '.join(result['active_symptoms']) if result['active_symptoms'] else 'No specific symptoms'
        
        new_prediction = Prediction(
            user_id=session['user_id'],
            symptoms=symptoms_str,
            predicted_disease=result['disease'],
            confidence_score=result['confidence'],
            risk_level=result['risk_level'],
            suggestions=json.dumps(result['suggestions']),
            contributing_factors=json.dumps(result['contributing_factors'])
        )
        db.session.add(new_prediction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'prediction_id': new_prediction.id,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
@login_required
def get_user_stats():
    """Get user statistics"""
    try:
        user_id = session['user_id']
        total_predictions = Prediction.query.filter_by(user_id=user_id).count()
        
        high_risk = Prediction.query.filter_by(user_id=user_id, risk_level='High').count()
        moderate_risk = Prediction.query.filter_by(user_id=user_id, risk_level='Moderate').count()
        low_risk = Prediction.query.filter_by(user_id=user_id, risk_level='Low').count()
        
        recent = Prediction.query.filter_by(user_id=user_id).order_by(
            Prediction.timestamp.desc()
        ).first()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_predictions': total_predictions,
                'risk_distribution': {
                    'high': high_risk,
                    'moderate': moderate_risk,
                    'low': low_risk
                },
                'last_prediction': recent.timestamp.isoformat() if recent else None
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# AI CHATBOT WITH OPENAI GPT
SYSTEM_PROMPT = """You are TruMed, a friendly and knowledgeable AI Medical Assistant chatbot for a medical diagnosis prediction platform. Your role is to:

1. Help users understand how to use the TruMed platform
2. Answer questions about symptoms and health conditions (general information only)
3. Explain how the AI prediction system works
4. Guide users through the assessment process
5. Provide information about the platform's features

Important Guidelines:
- Always be empathetic, supportive, and professional
- Never provide direct medical diagnosis or treatment advice
- Always recommend consulting healthcare professionals for serious concerns
- Be clear that you're an AI assistant, not a replacement for doctors
- Keep responses concise and easy to understand (under 200 words)
- Use simple language and avoid complex medical jargon unless necessary
- If asked about specific medical conditions, provide general information and suggest using the prediction system

Platform Information:
- TruMed uses advanced machine learning for disease prediction
- The AI model has 95%+ accuracy using XGBoost and Random Forest
- Analyzes 131 symptoms across 14 body systems
- Can predict 41 different diseases
- All data is encrypted and GDPR compliant
- The system uses explainable AI (SHAP and LIME) for transparency
- Users can track their health history and assessments
- Risk Levels: High (65%+), Moderate (35-64%), Low (<35%)

Remember: You're here to guide, inform, and support - not to diagnose or treat. Always encourage users to consult with qualified healthcare professionals for medical advice."""

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """AI chatbot endpoint using OpenAI GPT"""
    try:
        user_id = session['user_id']
        if not rate_limit_check(user_id, max_requests=20, window=60):
            return jsonify({
                'success': False,
                'error': 'Too many requests. Please wait a moment before sending another message.'
            }), 429
        
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        user_message = data['message'].strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Empty message'
            }), 400
        
        user = get_current_user()
        user_name = user.name.split()[0]
        
        total_assessments = Prediction.query.filter_by(user_id=user_id).count()
        recent_prediction = Prediction.query.filter_by(user_id=user_id).order_by(
            Prediction.timestamp.desc()
        ).first()
        
        recent_disease = recent_prediction.predicted_disease if recent_prediction else 'None'
        recent_risk = recent_prediction.risk_level if recent_prediction else 'None'
        
        conversation_history = data.get('conversation_history', [])
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        if conversation_history:
            for msg in conversation_history[-3:]:
                if 'user' in msg:
                    messages.append({"role": "user", "content": msg['user']})
                if 'bot' in msg:
                    messages.append({"role": "assistant", "content": msg['bot']})
        
        contextual_message = f"""User Profile:
- Name: {user_name}
- Total Assessments: {total_assessments}
- Last Prediction: {recent_disease}
- Last Risk Level: {recent_risk}

User Question: {user_message}"""
        
        messages.append({"role": "user", "content": contextual_message})
        
        if openai_client:
            try:
                print(f"[DEBUG] Calling OpenAI API for: {user_message[:50]}...")
                
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=350
                )
                
                bot_response = response.choices[0].message.content.strip()
                
                print(f"[DEBUG] OpenAI API Success! Response length: {len(bot_response)}")
                
                log_chat_message(user_id, user_message, bot_response)
                
                return jsonify({
                    'success': True,
                    'response': bot_response,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'openai_gpt'
                })
            
            except Exception as openai_error:
                print(f"[ERROR] OpenAI API Error: {str(openai_error)}")
                print(f"[ERROR] Error type: {type(openai_error).__name__}")
        
        fallback_response = generate_intelligent_fallback(user_message, user_name, user)
        
        log_chat_message(user_id, user_message, fallback_response)
        
        return jsonify({
            'success': True,
            'response': fallback_response,
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        })
    
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred processing your message. Please try again.'
        }), 500

def log_chat_message(user_id, user_message, bot_response):
    """Log chat messages"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[CHAT LOG {timestamp}] User {user_id}: {user_message[:50]}...")
        print(f"[CHAT LOG {timestamp}] Bot: {bot_response[:50]}...")
    except Exception as e:
        print(f"Error logging chat message: {str(e)}")

def generate_intelligent_fallback(message, user_name, user):
    """Smart fallback responses"""
    message_lower = message.lower()
    
    total = Prediction.query.filter_by(user_id=user.id).count()
    recent = Prediction.query.filter_by(user_id=user.id).order_by(
        Prediction.timestamp.desc()
    ).first()
    
    if any(word in message_lower for word in ['hi', 'hello', 'hey', 'good morning', 'good evening']):
        return f"Hello {user_name}! 👋 I'm TruMed, your AI Medical Assistant. How can I help you today?\n\nI can help with:\n• Starting health assessments\n• Understanding symptoms\n• Explaining AI predictions\n• Viewing your health history\n• Answering health questions"
    
    elif any(word in message_lower for word in ['symptom', 'feel', 'pain', 'sick', 'hurt', 'ache']):
        return f"🏥 I can help you assess your symptoms!\n\n**To get started:**\n1. Click 'New Assessment' button\n2. Select your symptoms (minimum 5 required)\n3. Get AI prediction with confidence score\n\nOur system analyzes 131 symptoms across 14 body systems. Would you like to start now?"
    
    elif any(word in message_lower for word in ['assess', 'predict', 'diagnos', 'check', 'test', 'start', 'begin']):
        return "✅ **Starting a Health Assessment:**\n\n1. Click 'New Assessment' or 'New Prediction'\n2. Select symptoms from organized categories\n3. Choose at least 5 symptoms for accuracy\n4. Submit and get instant AI analysis\n\nYou'll receive:\n• Disease prediction\n• Confidence score\n• Risk level (High/Moderate/Low)\n• Detailed explanations\n• Health suggestions"
    
    elif any(word in message_lower for word in ['accura', 'how good', 'reliable', 'trust', 'ai', 'ml', 'machine']):
        return "🤖 **Our AI System Accuracy:**\n\n✅ **95%+ accuracy rate**\n✅ **XGBoost** - Advanced gradient boosting\n✅ **Random Forest** - Ensemble learning\n✅ **SHAP Technology** - Explainable AI\n✅ **LIME Analysis** - Local interpretable explanations\n\nTrained on comprehensive medical data with 131 symptoms and 41 diseases. Every prediction includes detailed explanations!"
    
    elif any(word in message_lower for word in ['risk', 'danger', 'serious', 'level']):
        return "⚠️ **Risk Levels Explained:**\n\n🔴 **High Risk (65%+ confidence)**\n   → Strong prediction match\n   → Immediate medical consultation recommended\n\n🟡 **Moderate Risk (35-64%)**\n   → Moderate confidence level\n   → Schedule doctor's appointment\n\n🟢 **Low Risk (<35%)**\n   → Lower confidence\n   → Monitor symptoms, consult if persist\n\n⚠️ Always consult healthcare professionals!"
    
    elif any(word in message_lower for word in ['history', 'record', 'past', 'previous', 'last']):
        if recent:
            return f"📊 **Your Health History:**\n\nTotal Assessments: **{total}**\nLast Prediction: **{recent.predicted_disease}**\nRisk Level: **{recent.risk_level}**\nDate: {recent.timestamp.strftime('%B %d, %Y')}\n\nView full history by clicking 'History' in the menu to track trends over time!"
        else:
            return f"📊 You have **{total} assessment(s)** in your health history.\n\nClick 'History' in the menu to view all your past predictions and track your health trends over time."
    
    elif any(word in message_lower for word in ['privacy', 'secure', 'safe', 'data', 'gdpr', 'protect']):
        return "🔒 **Your Data is 100% Secure:**\n\n✅ **GDPR Compliant** - European standards\n✅ **Indian IT Rules** - Full compliance\n✅ **End-to-End Encryption** - All data encrypted\n✅ **Private Storage** - Never shared\n✅ **Explainable AI** - Transparent predictions\n\nYour medical information is confidential and protected by industry-standard security!"
    
    elif any(word in message_lower for word in ['how', 'work', 'use', 'function', 'process']):
        return "🔬 **How TruMed Works:**\n\n1️⃣ **Select Symptoms** - Choose from 131 symptoms by body system\n2️⃣ **AI Analysis** - ML models analyze patterns instantly\n3️⃣ **Get Results** - Disease prediction with confidence score\n4️⃣ **Understand Why** - SHAP/LIME show contributing factors\n5️⃣ **Take Action** - Follow suggestions, consult doctors\n\nThe process takes less than 2 minutes!"
    
    elif any(word in message_lower for word in ['skin', 'rash', 'acne', 'eczema', 'dermatology']):
        return "📸 **Skin Disease Analysis:**\n\nTruMed now includes AI-powered skin disease detection!\n\n**How to use:**\n1. Click 'Skin Disease Analysis' on dashboard\n2. Upload a clear photo of affected area\n3. Get instant AI prediction\n\nOur model can detect various skin conditions including acne, eczema, melanoma, and more. Try it now!"
    
    else:
        return f"Hi {user_name}! I'm here to help with:\n\n🏥 Health assessments and symptom analysis\n📊 Understanding your predictions\n🤖 Explaining how our AI works\n📈 Viewing your health history\n💡 General health information\n\nWhat would you like to know more about?"

# ERROR HANDLERS
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🏥 TruMed - AI Medical Diagnosis System")
    print("="*80)
    print(f"✓ Flask app initialized")
    print(f"✓ Database connected")
    print(f"✓ ML Model loaded ({len(predictor.symptoms_list)} symptoms, {len(predictor.disease_list)} diseases)")
    print(f"✓ OpenAI ChatGPT: {'Enabled ✅' if openai_client else 'Disabled ⚠️ (using fallback)'}")
    print(f"✓ AI Chatbot: Enabled with rate limiting (20 msgs/min)")
    print(f"✓ Admin Login: http://127.0.0.1:5000/admin/login")
    print(f"✓ Admin Username: {ADMIN_USERNAME}")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)