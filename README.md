# 🚗 Driver Drowsiness and Yawning Detection System

A real-time computer vision-based system to detect drowsiness and yawning in drivers using facial landmarks. The system raises visual alerts and streams a live feed through a Flask-powered web dashboard, enabling both local and remote (Ngrok-based) monitoring.

---

## 🔧 Features

- 💤 Drowsiness detection using Eye Aspect Ratio (EAR)
- 😮 Yawn detection using Mouth Aspect Ratio (MAR) / Lip Distance
- 📹 Real-time video feed with bounding box overlays
- ⚠️ Visual alert messages for "Drowsy" or "Yawning"
- 📊 Dashboard with live status and yawn count
- 🔐 Login screen for secure access
- 📱 Remote monitoring via [Ngrok](https://ngrok.com)
- 🎯 Designed for deployment on laptops and Raspberry Pi

---

## 📁 Project Structure

Drowsiness_Detection/ 
├── app.py # Flask app for routing and dashboard 
├── main.py # Core detection logic (OpenCV + Dlib) 
├── requirements.txt # Python dependencies 
├── templates/ # HTML pages (login, dashboard) 
│     │ 
│     ├── login.html 
│     │ 
│     └── dashboard.html 
├── static/ # CSS, JS, and images 
│ 
└── README.md # Project documentation

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Webcam or USB camera
- Ngrok account (optional for remote access)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/driver-drowsiness-detection.git
   cd driver-drowsiness-detection
   ```
2. **Create virtual environment and activate **it**
    ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Linux/Mac
   ```
3. **Install dependencies**
    ```bash
   pip install -r requirements.txt
   ```
4. **Run the Flask server**
   ```bash
   python app.py
   ```
5. **Access the dashboard**

Open http://localhost:5000 in your browser

Login with the preset credentials (customizable in app.py)


## 🌐 Remote Monitoring with Ngrok (Optional)

1. **Install Ngrok** 
   https://ngrok.com/download

2. **Authenticate your system**
   ```bash
   ngrok config add-authtoken <your_auth_token>
   ```

3. **Expose your Flask app**
   ```bash
   ngrok http 5000
   ```

4. **Access the public URL**

   Ngrok will provide a URL like: https://abcd-1234.ngrok.io


### Additional Notes

- Ensure your python versio is 3.9.7 to aviod compatibility issues, especially with `dlib`.
- If `dlib` installation fails, verify that you have correctly installed the Microsoft Visual Studio C++ Build Tools with the necessary components.
- you can modify thresholds and other settings directly within the `app.py` and `main.py` file as needed.
- Flask’s development server is not meant for production use. When you're ready for deployment, use Gunicorn (Linux/macOS) or Waitress (Windows) as a production WSGI server.
--------------------------------------------------------------------------

Thank you for using this tool!
