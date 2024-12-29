import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL = "*********@gmail.com"
PASSWORD = "**********"  # App Password
RECIPIENT = "************@gmail.com"  # Sending to the same account for simplicity

def send_process_email(process_name, log_content):
    subject = f"{process_name} is done running"
    body = f"The process '{process_name}' has completed successfully.\n\nLog file content:\n\n{log_content}"

    # Create the email message
    msg = MIMEMultipart()
    msg["From"] = EMAIL
    msg["To"] = RECIPIENT
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        # Connect to the SMTP server and send the email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure the connection
            server.login(EMAIL, PASSWORD)
            server.sendmail(EMAIL, RECIPIENT, msg.as_string())
        print(f"Email sent successfully with subject: '{subject}'")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script_name.py [process_name] [log_file_path]")
        sys.exit(1)

    process_name = sys.argv[1]
    log_file_path = sys.argv[2]

    try:
        with open(log_file_path, "r") as log_file:
            log_content = log_file.read()
    except Exception as e:
        print(f"Failed to read log file: {e}")
        sys.exit(1)

    send_process_email(process_name, log_content)
