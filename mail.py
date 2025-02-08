import imaplib
import email
from email.header import decode_header

# Email credentials
EMAIL = "daya.avadi@gmail.com"
PASSWORD = "daya$12345"
IMAP_SERVER = "imap.gmail.com"

# Connect to the server
mail = imaplib.IMAP4_SSL(IMAP_SERVER)
mail.login(EMAIL, PASSWORD)
mail.select("inbox")  # Select the mailbox

# Search for all emails
status, messages = mail.search(None, "ALL")
email_ids = messages[0].split()

# Fetch latest email
latest_email_id = email_ids[-1]
status, msg_data = mail.fetch(latest_email_id, "(RFC822)")

# Parse the email
for response_part in msg_data:
    if isinstance(response_part, tuple):
        msg = email.message_from_bytes(response_part[1])
        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding if encoding else "utf-8")
        print("Subject:", subject)

        # Extract email body
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    print("Body:", body)
        else:
            body = msg.get_payload(decode=True).decode()
            print("Body:", body)

# Logout
mail.logout()
