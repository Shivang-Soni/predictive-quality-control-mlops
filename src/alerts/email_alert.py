import smtplib
from email.mime.text import MIMEText
import os
import dotenv

dotenv.load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
EMAIL_RECIEVER = os.getenv("EMAIL_RECIEVER")


def send_email_alert(subject: str, body: str):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["Body"] = body
    msg["To"] = EMAIL_RECIEVER

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, EMAIL_RECIEVER, msg.as_string())