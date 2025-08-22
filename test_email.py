import smtplib
from email.mime.text import MIMEText

# --- 1. CONFIGURATION ---
SMTP_SERVER = "smtp.live.com"
SMTP_PORT = 587
SENDER_EMAIL = "joeluckyjoe@hotmail.com"       # Your Outlook/Hotmail address
SENDER_PASSWORD = "hwzfapskhluofrnh" # Your App Password
RECIPIENT_EMAIL = "joeluckyjoe@hotmail.com"   # Where to send the test email

# --- 2. SEND THE TEST EMAIL ---
try:
    # Create the email message
    subject = "Python Email Test"
    body = "This is a test email sent from your Python script. If you received this, your credentials are correct."
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL

    # Connect to the server and send
    print("Connecting to the email server...")
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls() # Secure the connection
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
    
    print("✅ Test email sent successfully!")

except Exception as e:
    print("\n❌ FAILED TO SEND EMAIL.")
    print(f"ERROR: {e}")
    print("\nPlease double-check your SENDER_EMAIL and SENDER_PASSWORD (the 16-character App Password).")