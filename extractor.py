import email
from email import policy
import os

# Path to the .eml file
eml_file = "sample.eml"

# Open and parse the .eml file
with open(eml_file, "rb") as file:
    msg = email.message_from_bytes(file.read(), policy=policy.default)

# Extract email details
subject = msg["Subject"]
sender = msg["From"]
recipient = msg["To"]

# Extract email body (plain text or HTML)
body = ""
html_body = ""

if msg.is_multipart():
    for part in msg.walk():
        content_type = part.get_content_type()
        content_disposition = str(part.get("Content-Disposition"))

        if "attachment" not in content_disposition:
            payload = part.get_payload(decode=True)
            if content_type == "text/plain" and not body:
                body = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
            elif content_type == "text/html" and not html_body:
                html_body = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
else:
    body = msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="replace")

# Extract attachments
attachments = []
for part in msg.walk():
    content_disposition = str(part.get("Content-Disposition"))
    if "attachment" in content_disposition:
        filename = part.get_filename()
        if filename:
            filepath = os.path.join("attachments", filename)
            os.makedirs("attachments", exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(part.get_payload(decode=True))
            attachments.append(filepath)

# Print extracted details
print("Subject:", subject)
print("From:", sender)
print("To:", recipient)
print("Body (Plain Text):", body if body else "No plain text found.")
print("Body (HTML):", html_body if html_body
