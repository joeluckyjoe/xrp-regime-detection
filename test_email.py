import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# --- CONFIGURATION ---
# This must be the email you used to sign up for SendGrid.
FROM_EMAIL = 'joeluckyjoe@hotmail.com'
TO_EMAIL = 'joeluckyjoe@hotmail.com'

def send_test_email():
    """Sends a test email using the SendGrid API."""
    api_key = os.environ.get('SENDGRID_API_KEY')
    if not api_key:
        print("\n❌ ERROR: SENDGRID_API_KEY environment variable not set.")
        print("Please run 'export SENDGRID_API_KEY=...' in your terminal first.")
        return

    message = Mail(
        from_email=FROM_EMAIL,
        to_emails=TO_EMAIL,
        subject='SendGrid Test Email from Python',
        html_content='<strong>This is a test email from your Python trading bot. If you received this, SendGrid is working correctly.</strong>'
    )
    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        if response.status_code == 202:
             print("✅ Test email sent successfully via SendGrid!")
        else:
            print(f"\n❌ FAILED TO SEND EMAIL. Status Code: {response.status_code}")
            print(f"Response Body: {response.body}")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    send_test_email()   